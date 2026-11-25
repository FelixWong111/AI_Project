"""P06 detection, foreground and provisional size evidence behind a P07 factory."""
import copy
import hashlib
import json
from pathlib import Path

import cv2

from .p06_inference import create_backend as create_detector, _observations, _expected, _frame
from .p06_foreground import ForegroundAssociator
from .p06_size_evidence import SizeEvidenceProcessor, load_specs
from .p06_measurement_policy import load_policy

ROOT=Path(__file__).resolve().parents[1]


class EvidencePipeline:
    def __init__(self, size_processor, *, capture_geometry_id, foreground=None, foreground_enabled=True):
        if not isinstance(capture_geometry_id,str) or not capture_geometry_id.strip():
            raise ValueError('Explicit capture geometry ID required')
        self.size_processor=size_processor
        self.geometry_id=capture_geometry_id
        self.foreground=foreground
        if type(foreground_enabled) is not bool:raise ValueError('foreground_enabled must be boolean')
        self.foreground_enabled=foreground_enabled

    def process(self, frame, observations, *, quality=None, confirmed_families=()):
        _frame(frame)
        h,w=frame.shape[:2]
        if observations.get('status')!='REVIEW' or observations.get('frame_size')!=dict(width=w,height=h):
            raise ValueError('Expected original-frame detector result')
        raw=observations.get('objects')
        if not isinstance(raw,list):raise ValueError('Missing detector objects')
        for i,obj in enumerate(raw):
            if type(obj.get('detection_id')) is not int or obj['detection_id']!=i or type(obj.get('class_id')) is not int:
                raise ValueError('Invalid raw detection identity')
        checked=_observations([[o[k] for k in ('xmin','ymin','xmax','ymax','score','class_id')] for o in raw],w,h)
        counts=observations.get('counts')
        if (not isinstance(counts,dict) or set(counts)!=set('012345')
                or any(type(v) is not int or v<0 for v in counts.values())
                or type(observations.get('non_catalogue_count')) is not int
                or checked['counts']!=counts or checked['non_catalogue_count']!=observations['non_catalogue_count']):
            raise ValueError('Invalid raw detector counts')
        # Validate observations even if no background is configured.
        quality={} if quality is None else quality
        ids={o['detection_id'] for o in observations.get('objects',[])}
        if not isinstance(quality,dict) or any(type(k) is not int or k not in ids for k in quality):
            raise ValueError('Unknown source detection quality ID')
        if any(type(k) is not int or k not in ids for k in confirmed_families):
            raise ValueError('Unknown source family ID')
        foreground=None
        background_reason='FOREGROUND_DISABLED' if not self.foreground_enabled else 'BACKGROUND_REFERENCE_MISSING'
        if self.foreground_enabled and self.foreground is not None:
            foreground=self.foreground.process(frame,observations,capture_geometry_id=self.geometry_id)
            background_reason='' if foreground['background_usable'] else 'BACKGROUND_OR_SCENE_CHANGED'
        suppressed=set(foreground['suppressed_detection_ids']) if foreground else set()
        kept=[obj for obj in observations['objects'] if obj['detection_id'] not in suppressed]
        h,w=frame.shape[:2]
        canonical=_observations([[o[k] for k in ('xmin','ymin','xmax','ymax','score','class_id')] for o in kept],w,h)
        source_ids=[obj['detection_id'] for obj in kept]
        translated_quality={i:quality[sid] for i,sid in enumerate(source_ids) if sid in quality}
        translated_family=[i for i,sid in enumerate(source_ids) if sid in confirmed_families]
        result=self.size_processor.process(frame,canonical,capture_geometry_id=self.geometry_id,
                                           quality=translated_quality,confirmed_families=translated_family)
        candidate_by_detection={}
        if foreground:
            for group in foreground['candidates']:
                for did in group['detection_ids']:candidate_by_detection[did]=group
        for obj,sid in zip(result['objects'],source_ids):
            obj['source_detection_id']=sid
            group=candidate_by_detection.get(sid)
            size=obj['size_evidence']
            conclusion=size.get('conclusion','INSUFFICIENT_EVIDENCE')
            association_ok=(not self.foreground_enabled) or (group is not None and group['class_id'] is not None and not background_reason)
            rejection='INSUFFICIENT_EVIDENCE'
            if association_ok and obj['class_id']==0:
                rejection='DETECTOR_NON_CATALOGUE_CANDIDATE'
            elif association_ok and conclusion=='OUTSIDE_FAMILY_CANDIDATE':
                rejection='SIZE_OUTSIDE_CONFIRMED_FAMILY_CANDIDATE'
            elif association_ok and conclusion=='SPEC_COMPATIBLE_CANDIDATE':
                rejection='LEGAL_SPEC_COMPATIBLE_CANDIDATE'
            obj['rejection_evidence']=dict(status='REVIEW',conclusion=rejection,
                                           reason=background_reason or (group['reason'] if group else 'NO_ASSOCIATION'),
                                           association_candidate_id=group['candidate_id'] if group else None)
        result['p06_evidence']=dict(schema_version=1,raw_observations=copy.deepcopy(observations),
                                   foreground=foreground,background_reason=background_reason,
                                   foreground_enabled=self.foreground_enabled,
                                   proposed_counts=foreground['proposed_counts'] if foreground else None,
                                   counts_complete=foreground['counts_complete'] if foreground else False,
                                   unresolved_candidates=[copy.deepcopy(g) for g in foreground['candidates'] if g['class_id'] is None] if foreground else [],
                                   coordinate_convention='object boxes original-frame pixels; measurement lines crop-local with crop_xyxy origin',
                                   decision_owner='P08 shared inspection',formal_decision_enabled=False)
        result['message']='P06 integrated candidate evidence; REVIEW pending shared decision and physical validation'
        return result


class IntegratedBackend:
    is_mock=False

    def __init__(self, detector, pipeline, provenance):
        self.detector=detector
        self.pipeline=pipeline
        self.metadata=copy.deepcopy(detector.metadata)
        self.metadata.update(p06_pipeline_version='candidate-evidence-v1',pipeline_config=provenance,
                             foreground_enabled=pipeline.foreground_enabled,
                             measurement_validation_records=copy.deepcopy(pipeline.size_processor.policy.records)
                                 if pipeline.size_processor.policy is not None else [],
                             calibration=copy.deepcopy(pipeline.size_processor.calibration),
                             size_tolerance=copy.deepcopy(pipeline.size_processor.specs),
                             decision_stage='integrated_provisional_evidence_REVIEW')

    def synchronize(self):
        self.detector.synchronize()

    def infer(self, frame_bgr, expected_counts):
        _expected(expected_counts)
        observations=self.detector.infer(frame_bgr,expected_counts)
        result=self.pipeline.process(frame_bgr,observations)
        result['p06_cache_sha256']=self._cache_digest(result)
        return result

    @staticmethod
    def _cache_digest(result):
        value={k:v for k,v in result.items() if k not in ('p06_cache_sha256','expected_counts')}
        return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def rejudge(self, result, expected_counts):
        """Validate cached canonical observations. Never redetect or remeasure."""
        _expected(expected_counts)
        if result.get('p06_cache_sha256')!=self._cache_digest(result):
            raise ValueError('Cached evidence changed')
        size=result.get('frame_size',{})
        rows=[[o[k] for k in ('xmin','ymin','xmax','ymax','score','class_id')] for o in result.get('objects',[])]
        checked=_observations(rows,size['width'],size['height'])
        if (result.get('status')!='REVIEW' or checked['counts']!=result.get('counts')
                or checked['non_catalogue_count']!=result.get('non_catalogue_count')
                or 'p06_evidence' not in result):
            raise ValueError('Invalid cached integrated result')
        output=copy.deepcopy(result)
        output['expected_counts']=copy.deepcopy(expected_counts)
        return output


def create_backend(*,model_id,config):
    if model_id!='baseline3' or not isinstance(config,dict) or set(config)!={'pipeline_config_path'}:
        raise ValueError('Expected baseline3 with pipeline_config_path')
    def path(value):
        candidate=Path(value)
        return candidate if candidate.is_absolute() else ROOT/candidate
    config_path=path(config['pipeline_config_path'])
    settings=json.loads(config_path.read_text(encoding='utf-8-sig'))
    required={'schema_version','detector_config_path','specs_path','calibration_path',
              'capture_geometry_id','background','foreground_config_path'}
    if not required<=set(settings) or set(settings)-required-{'foreground_enabled','measurement_policy_path'} or settings['schema_version']!=1:
        raise ValueError('Invalid pipeline configuration')
    enabled=settings.get('foreground_enabled',True)
    if type(enabled) is not bool:raise ValueError('foreground_enabled must be boolean')
    policy=load_policy(path(settings['measurement_policy_path']),ROOT) if settings.get('measurement_policy_path') else None
    specs=load_specs(path(settings['specs_path']))
    calibration=json.loads(path(settings['calibration_path']).read_text(encoding='utf-8-sig')) if settings['calibration_path'] else None
    foreground=None
    if enabled and settings['background'] is not None:
        background=settings['background']
        if not isinstance(background,dict) or set(background)!={'path','file_sha256','reference_id','roi'}:
            raise ValueError('Background requires path, SHA256, reference ID and ROI')
        image_path=path(background['path'])
        if hashlib.sha256(image_path.read_bytes()).hexdigest()!=background['file_sha256']:
            raise ValueError('Background file SHA256 mismatch')
        image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Background could not be decoded')
        params=json.loads(path(settings['foreground_config_path']).read_text(encoding='utf-8-sig'))
        foreground=ForegroundAssociator(image,reference_id=background['reference_id'],
                                        capture_geometry_id=settings['capture_geometry_id'],roi=background['roi'],config=params)
    pipeline=EvidencePipeline(SizeEvidenceProcessor(specs,calibration,policy=policy),
                              capture_geometry_id=settings['capture_geometry_id'],foreground=foreground,foreground_enabled=enabled)
    detector=create_detector(model_id=model_id,config={'config_path':str(path(settings['detector_config_path']))})
    return IntegratedBackend(detector,pipeline,dict(path=str(config_path),file_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()))
