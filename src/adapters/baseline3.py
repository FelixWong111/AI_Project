"""P07 adapter for P06's frozen detection observations and shared count rule."""
from __future__ import annotations

import copy
from pathlib import Path

from src.models.baseline3.inference import create_backend as load_detector
from src.work_orders import judge_observed, validate_counts


class Baseline3Backend:
    is_mock = False

    def __init__(self, config):
        self.detector = load_detector(model_id='baseline3', config={'config_path': config})
        self.metadata = copy.deepcopy(self.detector.metadata)
        self.metadata.update(model='baseline3', deployment_stage='real_detector_partial_rejection',
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
        return result

    def rejudge(self, result, expected_counts):
        expected = validate_counts(expected_counts)
        if result.get('status') in ('ERROR', 'REVIEW'):
            return copy.deepcopy(result)
        return self._decision(result, expected)


def create_backend(*, model_id, config):
    if model_id != 'baseline3':
        raise ValueError('This adapter supports baseline3 only; final is not configured')
    from src.p07_bridge import repository_path
    return Baseline3Backend(repository_path(config.get('baseline3_config', 'config/baseline3.json')))
