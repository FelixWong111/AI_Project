"""Explicit Stage 2 development size bridge; frozen production algorithms stay intact.

Empirical visible-size bands are not instructor-approved manufacturing tolerances.
No filenames, labels, annotations, expected counts or other models enter measurement.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

from .inference import ROOT, _frame, _observations
from src.work_orders import validate_counts


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def size_conclusion(measurement, class_id, bands):
    required=bands[str(class_id)]
    dimensions=measurement.get('dimensions',{})
    outside=[k for k,v in dimensions.items() if k in required and
             not required[k]['min_mm']<=v['mm_estimate']<=required[k]['max_mm']]
    if outside:
        return 'OUTSIDE_DEVELOPMENT_BAND',outside
    missing=[k for k in required if k not in dimensions]
    if missing or measurement['status']!='MEASURED_CANDIDATE':
        return 'UNMEASURABLE',missing
    return 'WITHIN_DEVELOPMENT_BAND',[]


class DevelopmentForeground:
    """P06 association after ordinary robust quadratic illumination correction.

    A real empty Train frame is the reference. Large-scale exposure/vignetting
    variation is fitted outside detections with robust residual trimming. This
    does not reconstruct specular metal, use Highlight-Mask, or identify unknown
    foreground as ID0 without evidence. Foreground computation is at 1/4 scale;
    original YOLO boxes and full-resolution metrology are untouched.
    """
    def __init__(self,background,config):
        from src.p06_foreground import ForegroundAssociator
        self.config=copy.deepcopy(config)
        self.width=756;self.height=1008
        self.background=cv2.resize(background,(self.width,self.height),interpolation=cv2.INTER_AREA)
        self.gray=cv2.cvtColor(self.background,cv2.COLOR_BGR2GRAY).astype(np.float32)
        y,x=np.mgrid[-1:1:complex(self.height),-1:1:complex(self.width)]
        self.basis=np.stack([np.ones_like(x),x,y,x*x,x*y,y*y],axis=2).astype(np.float32)
        self.associator=ForegroundAssociator(cv2.cvtColor(self.gray.astype(np.uint8),cv2.COLOR_GRAY2BGR),
            reference_id=config['reference_id'],capture_geometry_id=config['capture_geometry_id'],config=config['parameters'])

    def process(self,frame,observation):
        reduced=cv2.resize(frame,(self.width,self.height),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(reduced,cv2.COLOR_BGR2GRAY).astype(np.float32)
        sx=self.width/frame.shape[1];sy=self.height/frame.shape[0]
        boxes=[];usable=np.ones(gray.shape,np.uint8)
        for o in observation['objects']:
            x1,y1,x2,y2=[o[k] for k in ('xmin','ymin','xmax','ymax')]
            boxes.append([x1*sx,y1*sy,x2*sx,y2*sy,o['score'],o['class_id']])
            l,t=max(0,int(x1*sx)-5),max(0,int(y1*sy)-5)
            r,b=min(self.width,int(np.ceil(x2*sx))+5),min(self.height,int(np.ceil(y2*sy))+5)
            usable[t:b,l:r]=0
        sample=self.basis[::8,::8].reshape(-1,6)
        response=(gray-self.gray)[::8,::8].ravel()
        keep=usable[::8,::8].ravel().astype(bool)
        if keep.sum()<100:
            return dict(counts_complete=False,reason='INSUFFICIENT_VISIBLE_BACKGROUND',candidates=[])
        for _ in range(4):
            coefficients=np.linalg.lstsq(sample[keep],response[keep],rcond=None)[0]
            residual=response-sample@coefficients
            median=float(np.median(residual[keep]));mad=float(np.median(np.abs(residual[keep]-median)))
            keep=usable[::8,::8].ravel().astype(bool)&(np.abs(residual-median)<max(6.,4*1.4826*mad))
            if keep.sum()<100:
                return dict(counts_complete=False,reason='ILLUMINATION_FIT_UNSTABLE',candidates=[])
        illumination=np.einsum('ijk,k->ij',self.basis,coefficients)
        corrected=np.clip(gray-illumination,0,255).astype(np.uint8)
        scaled=_observations(boxes,self.width,self.height)
        result=self.associator.process(cv2.cvtColor(corrected,cv2.COLOR_GRAY2BGR),scaled,
                        capture_geometry_id=self.config['capture_geometry_id'])
        result.pop('detector_observations',None)
        result.update(coordinate_space='756x1008 foreground diagnostic only; main objects remain original pixels',
            illumination_coefficients=coefficients.tolist(),background_fit_rms=float(np.sqrt(np.mean(residual[keep]**2))),
            reason='' if result['counts_complete'] else 'UNRESOLVED_FOREGROUND',
            physical_unknown_validation=False)
        return result


class SizeDevelopmentBackend:
    is_mock=False

    def __init__(self,path,*,detector_factory=None):
        from .candidate import detector_factory as frozen_factory
        from src.models.baseline1.features import load_image_bgr
        self.path=Path(path);self.config=read_json(self.path)
        if self.config.get('mode')!='size_development_candidate':
            raise ValueError('Explicit size_development_candidate configuration required')
        self.selection_path=ROOT/self.config['selection_path'];self.calibration_path=ROOT/self.config['calibration_path']
        self.selection=read_json(self.selection_path);self.calibration=read_json(self.calibration_path)
        self.specifications=read_json(ROOT/self.selection['nominal_specs_path'])['items']
        if hashlib.sha256(self.calibration_path.read_bytes()).hexdigest()!=self.selection['calibration_sha256']:
            raise ValueError('Selected calibration SHA changed')
        if self.selection.get('validated') is not False or self.calibration.get('instructor_approved') is not False:
            raise ValueError('Development configuration cannot impersonate physical approval')
        self.detector=(detector_factory or frozen_factory)(model_id='baseline3',config={'config_path':ROOT/self.config['detector_config']})
        self.foreground=DevelopmentForeground(load_image_bgr(str(ROOT/self.config['foreground']['background_path'])),self.config['foreground'])
        self.files=[self.path,self.selection_path,self.calibration_path,ROOT/self.config['detector_config'],
                    Path(self.detector.metadata['weights']),ROOT/self.config['foreground']['background_path'],
                    Path(__file__),ROOT/'src/p06_geometry.py',ROOT/'src/p06_foreground.py']
        self.signature=self._signature()
        self.metadata=copy.deepcopy(self.detector.metadata)
        self.metadata.update(model='baseline3',baseline3_mode='size_development_candidate',
            business_decision_owner='baseline3_adapter',deployment_stage='DEVELOPMENT_ONLY_NOT_PHYSICALLY_ACCEPTED',
            calibration=copy.deepcopy(self.calibration),size_tolerance=self.selection['bands'],
            size_verification='Measured original pixels + empirical same-specimen development bands; provisional millimetres',
            acceptance_confirmed=False,independent_caliper_validation=False,instructor_approved=False,
            measurement_strategy='P06 body_profile / parallel hexagon flats / annular ellipse; selected segmentation parameters per family',
            foreground='P06 association + robust polynomial illumination correction; unknown types remain REVIEW',
            score_convention=self.metadata['score_convention']+'; reassigned ID0 retains original detector score, not a calibrated ID0 probability')
        self.metadata['model_version']+='-size-dev-v1-'+hashlib.sha256(self.selection_path.read_bytes()).hexdigest()[:10]

    def _signature(self):
        return tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in self.files)

    def synchronize(self):
        self.detector.synchronize()

    def infer(self,frame_bgr,expected_counts):
        _frame(frame_bgr);expected=validate_counts(expected_counts)
        if self._signature()!=self.signature:raise RuntimeError('Size candidate files changed; reload')
        observation=self.detector.infer(frame_bgr,expected)
        h,w=frame_bgr.shape[:2]
        compatible=[w,h]==self.calibration['frame_size']
        evidence=dict(mode='size_development_candidate',acceptance_confirmed=False,
            calibration=dict(self.calibration,calibration_id='ruler10-fixed-collection-size-dev-v1',validated=False),
            specifications=copy.deepcopy(self.specifications),
            calibration_compatible=compatible,calibration_status='PROVISIONAL_RULER_IMAGE_FIT',
            measurement_calls=0,measurement_complete=0,review_reasons=[],failure_reasons=[],
            geometry_id=self.calibration['capture_geometry_id'],caliper_mae=None)
        if not compatible:
            evidence['review_reasons'].append('CALIBRATION_RESOLUTION_OR_CROP_UNSUPPORTED: Camo field of view needs its own calibration')
        from src.p06_geometry import measure_object,DEFAULT_PARAMETERS
        for obj in observation['objects']:
            obj['detector_class_id']=obj['class_id']
            if obj['class_id']==0:
                obj['size_evidence']=dict(conclusion='DETECTOR_NON_CATALOGUE',reason='Already ID0; no catalogue geometry assumed')
                continue
            if not compatible:
                obj['size_evidence']=dict(conclusion='UNMEASURABLE',reason='CALIBRATION_FRAME_MISMATCH',provisional=True)
                continue
            cid=obj['class_id'];family='1' if cid in (1,2,3) else str(cid)
            chosen=self.selection['selected'][family]['parameters'];params={k:v for k,v in chosen.items() if k!='targeted'}
            x1,y1,x2,y2=[obj[k] for k in ('xmin','ymin','xmax','ymax')]
            pad=max(16,round(max(x2-x1,y2-y1)*.18))
            l,t=max(0,int(x1)-pad),max(0,int(y1)-pad);r,b=min(w,int(np.ceil(x2))+pad),min(h,int(np.ceil(y2))+pad)
            measurement=measure_object(frame_bgr[t:b,l:r],cid,self.calibration['px_per_mm'],
                dict(DEFAULT_PARAMETERS,**params),target_box=[x1-l,y1-t,x2-l,y2-t] if chosen['targeted'] else None)
            # Full-resolution lines are traceable in the unletterboxed frame.
            for dim in measurement['dimensions'].values():
                dim['line_xy_original']=[[float(x+l),float(y+t)] for x,y in dim['line_xy']]
            conclusion,details=size_conclusion(measurement,cid,self.selection['bands'])
            evidence['measurement_calls']+=1
            evidence['measurement_complete']+=int(measurement['status']=='MEASURED_CANDIDATE')
            measurement.pop('contours_xy',None)
            measurement['diagnostics']={k:v for k,v in measurement['diagnostics'].items() if not isinstance(v,list)}
            obj['size_evidence']=dict(conclusion=conclusion,reason=measurement['reason'],measurement=measurement,
                provisional=True,calibration_status='PROVISIONAL_RULER_IMAGE_FIT',
                standard_specification_source=self.selection['nominal_specs_path'],
                development_acceptance_bands=self.selection['bands'][str(cid)],
                instructor_approved=False,details=details,
                measurement_policy_reason='DEVELOPMENT_EMPIRICAL_BANDS / NOT_INSTRUCTOR_APPROVED',
                quality=dict(complete=measurement['status']=='MEASURED_CANDIDATE',
                             ruler_scale_envelope=self.calibration['relative_conservative_scale_envelope'],
                             independent_error_bound_mm=None))
            if conclusion=='OUTSIDE_DEVELOPMENT_BAND':
                obj['class_id']=0;obj['class_assignment']='provisional_size_rule_non_catalogue'
                evidence['failure_reasons'].append(f"object {obj['detection_id']}: OUTSIDE_DEVELOPMENT_SIZE_BAND {details}")
            elif conclusion=='UNMEASURABLE':
                evidence['review_reasons'].append(f"object {obj['detection_id']}: {measurement['reason'] or details}")
        if compatible:
            # Association uses untouched detector semantics, independent of order and size relabelling.
            raw=_observations([[o[k] for k in ('xmin','ymin','xmax','ymax','score')]+[o['detector_class_id']]
                               for o in observation['objects']],w,h)
            foreground=self.foreground.process(frame_bgr,raw)
            evidence['foreground']=foreground
            if not foreground['counts_complete']:evidence['review_reasons'].append(foreground['reason'])
        else:evidence['foreground']=dict(counts_complete=False,reason='UNSUPPORTED_CAPTURE_GEOMETRY')
        observation['inspection']=evidence
        return self.rejudge(observation,expected)

    def rejudge(self,result,expected_counts):
        expected=validate_counts(expected_counts);fresh=copy.deepcopy(result)
        inspection=fresh['inspection'];failures=list(inspection['failure_reasons']);reviews=list(inspection['review_reasons'])
        counts=dict.fromkeys('012345',0)
        for o in fresh['objects']:counts[str(o['class_id'])]+=1
        if counts['0']:failures.append('NON_CATALOGUE_DETECTED')
        mismatch=any(counts[k]!=expected[k] for k in expected)
        if mismatch:failures.append('COUNT_MISMATCH')
        if len(fresh['objects'])>=self.detector._params['max_det']:reviews.append('DETECTION_CAP_REACHED')
        status='FAIL' if failures else 'REVIEW' if reviews else 'PASS'
        fresh.update(status=status,predicted_label=status if status in ('PASS','FAIL') else None,
            prediction_score=None,score_type='none',counts=counts,non_catalogue_count=counts['0'],
            expected_counts=expected,difference={k:counts[k]-expected[k] for k in expected},
            baseline3_mode='size_development_candidate',
            message='DEVELOPMENT CANDIDATE / provisional dimensions / physical acceptance pending. '+
                    ('; '.join(dict.fromkeys(failures+reviews)) if failures or reviews else 'Count, size-band and foreground checks passed.'))
        inspection['decision_failure_reasons']=failures;inspection['decision_review_reasons']=reviews
        return fresh


def create_backend(*,model_id,config):
    if model_id!='baseline3':raise ValueError('Size candidate is Baseline3 only')
    if config.get('baseline3_mode')!='size_development_candidate':raise ValueError('No implicit count-only fallback')
    path=Path(config['size_candidate_config']);path=path if path.is_absolute() else ROOT/path
    return SizeDevelopmentBackend(path)
