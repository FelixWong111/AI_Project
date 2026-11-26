"""P07 adapter for P06's frozen detection observations and shared count rule."""
from __future__ import annotations

import copy
from pathlib import Path

from src.models.baseline3.inference import create_backend as load_detector
from src.work_orders import judge_observed, validate_counts


class FrozenYOLOBackend:
    is_mock = False

    def __init__(self, config, *, detector_factory=None):
        self.detector = (detector_factory or load_detector)(model_id='baseline3', config={'config_path': config})
        self.metadata = copy.deepcopy(self.detector.metadata)
        self.metadata.update(model='baseline3', deployment_stage='real_detector_partial_rejection',
                             baseline3_mode='yolo_frozen', business_decision_owner='baseline3_adapter',
                             decision_stage='shared observed-vs-expected count rule; learned ID0 forces FAIL',
                             size_verification='NOT_PROVIDED: calibration/size tolerances absent',
                             rejection_validation='PARTIAL: learned class0 only; not arbitrary unknown/incorrect-size rejection')
        files = (Path(config), Path(self.metadata['weights']))
        self.files = files
        self.signature = self._signature()

    def _signature(self):
        return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in self.files)

    def synchronize(self):
        self.detector.synchronize()

    def infer(self, frame_bgr, expected_counts):
        from src.p07_bridge import validate_result
        expected = validate_counts(expected_counts)
        if self._signature() != self.signature:
            raise RuntimeError('Baseline3 files changed; reload the model')
        observation = self.detector.infer(frame_bgr, expected)
        validate_result(observation, 'baseline3', frame_bgr.shape)
        # P06 uses REVIEW only as a detection-only transport placeholder. It is
        # not an image judgeability classifier. Preserve genuine truncation review.
        if len(observation['objects']) >= self.detector._params['max_det']:
            if observation['non_catalogue_count'] > 0:
                return dict(self._decision(observation, expected),
                            message='WARNING: non-catalogue detected: FAIL. Detection cap reached; counts may be truncated.')
            return dict(observation, message='Detection cap reached: counts may be truncated; REVIEW')
        return self._decision(observation, expected)

    @staticmethod
    def _decision(observation, expected):
        result = judge_observed(dict(observation, status='OBSERVED'), expected)
        id0 = result['non_catalogue_count']
        result['message'] = (f'WARNING: {id0} non-catalogue object(s). ' if id0 else '') + (
            'Count-rule result; learned ID0 only. Calibrated size verification is not provided.')
        result['baseline3_mode'] = 'yolo_frozen'
        return result

    def rejudge(self, result, expected_counts):
        expected = validate_counts(expected_counts)
        if result.get('status') in ('ERROR', 'REVIEW'):
            return copy.deepcopy(result)
        return self._decision(result, expected)


class SizeVerifiedBackend:
    """Business decisions over P06 evidence; never change detections on rejudge."""
    is_mock = False

    def __init__(self, pipeline_path, approval_path=None, *, provider_factory=None):
        import hashlib
        import json
        from src.p06_pipeline import create_backend as load_pipeline
        from src.p07_bridge import repository_path
        self.provider = (provider_factory or load_pipeline)(model_id='baseline3', config={'pipeline_config_path': str(pipeline_path)})
        self.settings = json.loads(Path(pipeline_path).read_text(encoding='utf-8-sig'))
        paths = [Path(pipeline_path), Path(self.provider.metadata['weights'])]
        for key in ('detector_config_path', 'specs_path', 'calibration_path',
                    'foreground_config_path', 'measurement_policy_path'):
            if self.settings.get(key):
                paths.append(repository_path(self.settings[key]))
        registry = self.settings.get('measurement_policy_path')
        if registry:
            refs = json.loads(repository_path(registry).read_text(encoding='utf-8-sig'))['records']
            paths.extend(repository_path(r['path']) for r in refs)
        if self.settings.get('background') and self.settings.get('foreground_enabled', True):
            paths.append(repository_path(self.settings['background']['path']))
        for filename in ('src/adapters/baseline3.py', 'src/p06_pipeline.py', 'src/p06_geometry.py',
                         'src/p06_screw_edges.py', 'src/p06_size_evidence.py',
                         'src/p06_measurement_policy.py', 'src/p06_foreground.py',
                         'src/models/baseline3/inference.py', 'src/p07_bridge.py'):
            paths.append(repository_path(filename))
        self.approval_reason = 'ACCEPTANCE_NOT_CONFIRMED'
        self.approval = None
        if approval_path:
            approval_path = repository_path(approval_path)
            approval = json.loads(approval_path.read_text(encoding='utf-8-sig'))
            required = {'status', 'approved_by', 'confirmed_at', 'file_hashes', 'evidence_path', 'evidence_sha256'}
            if (set(approval) != required or approval['status'] != 'CONFIRMED'
                    or not all(isinstance(approval[k], str) and approval[k].strip()
                               for k in ('approved_by', 'confirmed_at', 'evidence_path'))):
                raise ValueError('Invalid acceptance confirmation record')
            evidence = repository_path(approval['evidence_path'])
            if hashlib.sha256(evidence.read_bytes()).hexdigest() != approval['evidence_sha256']:
                raise ValueError('Acceptance evidence SHA256 mismatch')
            # Confirmation must bind all physical conditions, specifications and validation scopes.
            required_keys = ('specs_path', 'calibration_path', 'measurement_policy_path', 'foreground_config_path')
            targets = {k: repository_path(self.settings[k]) for k in required_keys}
            targets['pipeline_config_path'] = Path(pipeline_path)
            if set(approval['file_hashes']) != set(targets):
                raise ValueError('Acceptance confirmation must bind pipeline, specs, scale, policy and foreground settings')
            for key, path in targets.items():
                if hashlib.sha256(path.read_bytes()).hexdigest() != approval['file_hashes'][key]:
                    raise ValueError('Acceptance scope changed: ' + key)
            self.approval, self.approval_reason = approval, None
            paths.extend((approval_path, evidence))
        self.files = tuple(dict.fromkeys(paths))
        self.signature = self._signature()
        self.metadata = copy.deepcopy(self.provider.metadata)
        self.metadata.update(model='baseline3', baseline3_mode='size_verified',
            business_decision_owner='baseline3_adapter', deployment_stage='size_evidence_integration',
            decision_stage='confirmed rejection/count mismatch -> FAIL; incomplete evidence -> REVIEW; all gates -> PASS',
            acceptance_confirmation=self.approval, acceptance_reason=self.approval_reason,
            size_verification='P06 measurements; provisional unless calibration, scope and acceptance are confirmed',
            rejection_validation='Foreground/physical validation required; no automatic claim of complete unknown rejection')
        self.metadata['rule_rejection_score_convention'] = (
            'Size-confirmed non-catalogue objects retain the original detector score, '
            'not a fabricated class0 probability; detector_class_id/class_assignment are in diagnostics.json')
        self.metadata['evidence_files_sha256'] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.files}

    def _signature(self):
        return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in self.files)

    def _check(self):
        if self._signature() != self.signature:
            raise RuntimeError('Baseline3 evidence/config files changed; reload the model')

    def synchronize(self):
        self.provider.synchronize()

    def infer(self, frame_bgr, expected_counts):
        self._check()
        expected = validate_counts(expected_counts)
        observation = self.provider.infer(frame_bgr, expected)
        return self._decision(observation, expected)

    def rejudge(self, result, expected_counts):
        self._check()
        expected = validate_counts(expected_counts)
        if result.get('status') == 'ERROR':
            return copy.deepcopy(result)
        observation = self.provider.rejudge(result['_baseline3_observation'], expected)
        return self._decision(observation, expected)

    def _decision(self, observation, expected):
        from src.p07_bridge import validate_result
        size = observation['frame_size']
        validate_result(observation, 'baseline3', (size['height'], size['width'], 3))
        result = copy.deepcopy(observation)
        reasons, failures = [], []
        p06 = observation['p06_evidence']
        cal = self.metadata.get('calibration')
        if not cal:
            reasons.append('CALIBRATION_MISSING')
        else:
            if not cal['validated']: reasons.append('CALIBRATION_NOT_VALIDATED')
            if (cal['frame_width'], cal['frame_height']) != (size['width'], size['height']):
                reasons.append('CALIBRATION_RESOLUTION_MISMATCH')
            if cal['capture_geometry_id'] != self.settings['capture_geometry_id']:
                reasons.append('CAPTURE_GEOMETRY_MISMATCH')
        if self.approval_reason: reasons.append(self.approval_reason)
        if not p06['foreground_enabled']:
            reasons.append('FOREGROUND_DISABLED: unknown-object coverage not verified')
        elif p06['background_reason']:
            reasons.append(p06['background_reason'])
        if not p06['counts_complete']:
            reasons.append('FOREGROUND_COUNTS_NOT_COMPLETE')
        if p06['unresolved_candidates']:
            reasons.append('UNRESOLVED_FOREGROUND_CANDIDATES')
        raw = p06['raw_observations']
        cap = self.provider.detector._params['max_det']
        truncated = len(raw['objects']) >= cap
        if truncated: reasons.append('DETECTION_CAP_REACHED')
        if raw['non_catalogue_count'] > 0 or observation['non_catalogue_count'] > 0:
            failures.append('NON_CATALOGUE_DETECTED')
        ambiguous = bool(p06['unresolved_candidates']) or (
            p06['foreground_enabled'] and bool(p06['background_reason']))
        if observation['counts'] != dict({'0': 0}, **expected):
            if not truncated and not ambiguous:
                failures.append('COUNT_MISMATCH')
            else:
                reasons.append('COUNT_MISMATCH_WITH_INCOMPLETE_OBSERVATIONS')
        for obj in result['objects']:
            if obj['class_id'] == 0: continue
            evidence = obj['size_evidence']
            confirmed = (not self.approval_reason and cal and cal['validated']
                         and evidence.get('family_confirmed') is True
                         and evidence.get('measurement_policy_reason') == 'VALIDATED_SCOPE_MATCHED')
            conclusion = evidence.get('conclusion')
            if confirmed and conclusion == 'OUTSIDE_FAMILY_CANDIDATE':
                obj['inspection_rejection'] = 'CONFIRMED_WRONG_SIZE_OR_TYPE'
                obj['detector_class_id'] = obj['class_id']
                obj['class_id'] = 0
                obj['class_assignment'] = 'size_rule_non_catalogue'
                failures.append(f"object {obj['detection_id']}: CONFIRMED_WRONG_SIZE_OR_TYPE")
            elif confirmed and conclusion == 'SPEC_COMPATIBLE_CANDIDATE':
                if obj['class_id'] not in evidence['compatible_class_ids']:
                    obj['inspection_rejection'] = 'CONFIRMED_CLASS_SPEC_CONFLICT'
                    failures.append(f"object {obj['detection_id']}: CONFIRMED_CLASS_SPEC_CONFLICT")
            else:
                reason = evidence.get('reason') or evidence.get('measurement_policy_reason') or conclusion
                reasons.append(f"object {obj['detection_id']}: {reason or 'SIZE_EVIDENCE_INCOMPLETE'}")
        # Canonical semantic labels are independent of the work order. Preserve
        # untouched YOLO labels/counts in p06_evidence.raw_observations and cache.
        counts = dict.fromkeys('012345', 0)
        for obj in result['objects']:
            counts[str(obj['class_id'])] += 1
        result.update(counts=counts, non_catalogue_count=counts['0'])
        status = 'FAIL' if failures else ('REVIEW' if reasons else 'PASS')
        result.update(status=status, predicted_label=status if status != 'REVIEW' else None,
            prediction_score=None, score_type='none', expected_counts=dict(expected),
            inspection=dict(mode='size_verified', failure_reasons=failures, review_reasons=reasons,
                            calibration=copy.deepcopy(cal),
                            specifications=copy.deepcopy(self.metadata['size_tolerance']['items']),
                            acceptance_confirmed=self.approval_reason is None,
                            calibration_validated=bool(cal and cal['validated']),
                            counts_complete=p06['counts_complete']),
            _baseline3_observation=copy.deepcopy(observation), baseline3_mode='size_verified',
            message=('WARNING: ' if failures else '') + '; '.join(failures + reasons) if failures or reasons
                    else 'PASS: count, size, calibration and foreground evidence complete')
        return result


def create_backend(*, model_id, config):
    if model_id != 'baseline3':
        raise ValueError('This adapter supports baseline3 only; final is not configured')
    from src.p07_bridge import repository_path
    mode = config.get('baseline3_mode', 'size_verified')
    if mode == 'yolo_frozen':
        return FrozenYOLOBackend(repository_path(config.get('baseline3_config', 'config/baseline3.json')))
    if mode == 'size_verified':
        return SizeVerifiedBackend(repository_path(config.get('baseline3_pipeline_config', 'config/p06_pipeline_candidate.json')),
                                   config.get('baseline3_acceptance_approval'))
    raise ValueError('baseline3_mode must explicitly be size_verified or yolo_frozen; no fallback')
