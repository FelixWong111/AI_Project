"""Independent provisional size evidence. Final inspection decisions belong to P08."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from .p06_data import CLASSES
from .p06_geometry import measure_object
from .p06_inference import _frame, _observations

DIMENSIONS = {
    1: ('underhead_length', 'thread_major_diameter'),
    2: ('underhead_length', 'thread_major_diameter'),
    3: ('underhead_length', 'thread_major_diameter'),
    4: ('across_flats',),
    5: ('outer_diameter', 'inner_diameter'),
}


def number(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def validate_specs(config):
    if not isinstance(config, dict):
        raise ValueError('Catalogue must be a dictionary')
    if config.get('units') != 'mm' or config.get('formal_decision_enabled') is not False:
        raise ValueError('Expected provisional mm catalogue with formal decisions disabled')
    items = config.get('items', [])
    if (not isinstance(items, list) or len(items) != 5
            or any(type(item.get('class_id')) is not int for item in items)
            or sorted(item['class_id'] for item in items) != [1, 2, 3, 4, 5]):
        raise ValueError('Expected unique catalogue IDs 1..5')
    for item in items:
        cid = item['class_id']
        if item.get('class_name') != CLASSES[cid] or set(item.get('dimensions', {})) != set(DIMENSIONS[cid]):
            raise ValueError('Catalogue class or dimension mapping differs')
        for dim in item['dimensions'].values():
            reference, allowance = dim.get('reference_mm'), dim.get('provisional_allowance_mm')
            if not number(reference, True) or not number(allowance, True) or allowance >= reference:
                raise ValueError('Invalid reference dimension or allowance')
    return copy.deepcopy(config)


def load_specs(path):
    return validate_specs(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def validate_calibration(calibration):
    if calibration is None:
        return None
    required = {'calibration_id', 'capture_geometry_id', 'frame_width', 'frame_height',
                'px_per_mm', 'validated'}
    if not isinstance(calibration, dict) or set(calibration) != required:
        raise ValueError('Invalid calibration fields')
    if any(not isinstance(calibration[k], str) or not calibration[k].strip()
           for k in ('calibration_id', 'capture_geometry_id')):
        raise ValueError('Calibration needs traceable calibration/capture IDs')
    if any(type(calibration[k]) is not int or calibration[k] <= 0
           for k in ('frame_width', 'frame_height')):
        raise ValueError('Invalid calibrated resolution')
    if (not isinstance(calibration['px_per_mm'], (list, tuple))
            or len(calibration['px_per_mm']) != 2
            or not all(number(v, True) for v in calibration['px_per_mm'])
            or type(calibration['validated']) is not bool):
        raise ValueError('Invalid x/y scales or validation flag')
    return copy.deepcopy(calibration)


def compare_dimensions(measurement, specs, quality=None, *, calibration_validated=False,
                       family_confirmed=False):
    """Compare a measured screw to all three screw specifications before exclusion.

    quality contains externally verified per-dimension error bounds and evidence
    IDs. Geometry CANDIDATE status alone does not supply those bounds.
    """
    specs = validate_specs(specs)
    cid = measurement.get('class_id')
    if type(cid) is not int or cid not in range(6):
        raise ValueError('Invalid measurement class')
    if type(calibration_validated) is not bool or type(family_confirmed) is not bool:
        raise ValueError('Validation flags must be boolean')
    quality = {} if quality is None else quality
    if not isinstance(quality, dict) or set(quality) - set(DIMENSIONS.get(cid, ())):
        raise ValueError('Unknown measurement quality dimensions')
    dims = measurement.get('dimensions', {})
    if not isinstance(dims, dict) or set(dims) - set(DIMENSIONS.get(cid, ())):
        raise ValueError('Unknown measured dimensions')
    if cid == 0:
        return dict(status='REVIEW', conclusion='DETECTOR_NON_CATALOGUE_OBSERVATION',
                    compatible_class_ids=[], comparisons=[], reason='No fixed ID0 size specification')
    checked = {}
    for key in DIMENSIONS[cid]:
        dim = dims.get(key)
        assessment = quality.get(key)
        if assessment is not None:
            if (not isinstance(assessment, dict)
                    or set(assessment) != {'verified', 'max_error_mm', 'evidence_id'}
                    or type(assessment['verified']) is not bool
                    or not number(assessment['max_error_mm'])
                    or not isinstance(assessment['evidence_id'], str) or not assessment['evidence_id'].strip()):
                raise ValueError('Invalid per-dimension error evidence')
        if dim is None:
            checked[key] = dict(state='MISSING', reason=measurement.get('dimension_status', {}).get(key, {}).get('reason', 'NO_MEASUREMENT'))
            continue
        if not isinstance(dim, dict) or not number(dim.get('mm_estimate'), True):
            raise ValueError('Invalid measured millimetres')
        usable = (measurement.get('status') in ('MEASURED_CANDIDATE', 'PARTIAL_CANDIDATE')
                  and calibration_validated and assessment is not None and assessment['verified'])
        checked[key] = dict(state='VERIFIED' if usable else 'UNVERIFIED', value_mm=dim['mm_estimate'],
                            max_error_mm=assessment['max_error_mm'] if usable else None,
                            evidence_id=assessment['evidence_id'] if assessment else None)
    family = (1, 2, 3) if cid in (1, 2, 3) else (cid,)
    comparisons, compatible = [], []
    complete = True
    for item in specs['items']:
        if item['class_id'] not in family:
            continue
        relations = {}
        for key, spec in item['dimensions'].items():
            check = checked[key]
            lower = spec['reference_mm'] - spec['provisional_allowance_mm']
            upper = spec['reference_mm'] + spec['provisional_allowance_mm']
            relation = dict(allowed_mm=[lower, upper], **check)
            if 'value_mm' in check:
                value = check['value_mm']
                relation['point_within_provisional_range'] = lower <= value <= upper
            if check['state'] == 'VERIFIED':
                value, error = check['value_mm'], check['max_error_mm']
                lo, hi = value-error, value+error
                relation['measurement_interval_mm'] = [lo, hi]
                if hi < lower or lo > upper:
                    relation['state'] = 'OUTSIDE'
                elif lo >= lower and hi <= upper:
                    relation['state'] = 'WITHIN'
                else:
                    relation['state'] = 'BORDERLINE'
            relations[key] = relation
        excluded = any(v['state'] == 'OUTSIDE' for v in relations.values())
        all_within = all(v['state'] == 'WITHIN' for v in relations.values())
        if not excluded:
            compatible.append(item['class_id'])
            complete = complete and all_within
        comparisons.append(dict(class_id=item['class_id'], excluded=excluded, dimensions=relations))
    conclusion = 'INSUFFICIENT_EVIDENCE'
    if family_confirmed and not compatible:
        conclusion = 'OUTSIDE_FAMILY_CANDIDATE'
    elif family_confirmed and len(compatible) == 1 and complete:
        conclusion = 'SPEC_COMPATIBLE_CANDIDATE'
    return dict(status='REVIEW', conclusion=conclusion, compatible_class_ids=compatible,
                family_confirmed=family_confirmed, comparisons=comparisons,
                catalogue_version=specs.get('catalogue_version'),
                tolerance_basis=specs.get('status'), formal_decision_enabled=False)


class SizeEvidenceProcessor:
    """Enrich frozen detector observations without relabelling or recounting."""
    def __init__(self, specs, calibration=None, policy=None):
        self.specs = validate_specs(specs)
        self.calibration = validate_calibration(calibration)
        self.policy = policy

    def process(self, frame_bgr, observations, *, capture_geometry_id, quality=None,
                confirmed_families=()):
        _frame(frame_bgr)
        h, w = frame_bgr.shape[:2]
        objects = observations.get('objects')
        if (not isinstance(objects, list) or observations.get('status') != 'REVIEW'
                or observations.get('frame_size') != dict(width=w, height=h)):
            raise ValueError('Expected detection-only observations matching original frame')
        rows = []
        for index, obj in enumerate(objects):
            if type(obj.get('detection_id')) is not int or obj['detection_id'] != index or type(obj.get('class_id')) is not int:
                raise ValueError('Invalid detector IDs')
            rows.append([obj[k] for k in ('xmin','ymin','xmax','ymax','score','class_id')])
        fresh = _observations(rows,w,h)
        cached_counts = observations.get('counts')
        if (not isinstance(cached_counts,dict) or set(cached_counts)!=set('012345')
                or any(type(v) is not int or v<0 for v in cached_counts.values())
                or type(observations.get('non_catalogue_count')) is not int
                or fresh['counts'] != cached_counts or fresh['non_catalogue_count'] != observations.get('non_catalogue_count')):
            raise ValueError('Detection counts disagree')
        quality = {} if quality is None else quality
        if not isinstance(quality,dict) or any(type(k) is not int or k not in range(len(objects)) for k in quality):
            raise ValueError('Unknown quality detection ID')
        if any(type(i) is not int or i not in range(len(objects)) for i in confirmed_families):
            raise ValueError('Unknown confirmed-family detection ID')
        calibration = self.calibration
        reason = ''
        if calibration is None:
            reason = 'CALIBRATION_MISSING'
        elif (calibration['frame_width'],calibration['frame_height']) != (w,h):
            reason = 'CALIBRATION_RESOLUTION_MISMATCH'
        elif calibration['capture_geometry_id'] != capture_geometry_id:
            reason = 'CAPTURE_GEOMETRY_MISMATCH'
        enriched = copy.deepcopy(observations)
        if self.policy is not None and (quality or confirmed_families):
            raise ValueError('Runtime policy cannot be overridden by caller quality flags')
        for obj in enriched['objects']:
            cid = obj['class_id']
            if reason or cid == 0:
                obj['size_evidence'] = dict(status='REVIEW', reason=reason or 'ID0_HAS_NO_FIXED_SIZE')
                continue
            x1,y1,x2,y2 = (obj[k] for k in ('xmin','ymin','xmax','ymax'))
            pad = max(16,round(max(x2-x1,y2-y1)*.18))
            left,top = max(0,math.floor(x1)-pad),max(0,math.floor(y1)-pad)
            right,bottom = min(w,math.ceil(x2)+pad),min(h,math.ceil(y2)+pad)
            crop = frame_bgr[top:bottom,left:right].copy()
            if min(crop.shape[:2]) < 10:
                obj['size_evidence'] = dict(status='REVIEW',reason='CROP_TOO_SMALL')
                continue
            measured = measure_object(crop,cid,calibration['px_per_mm'])
            object_quality=quality.get(obj['detection_id'])
            family_confirmed=obj['detection_id'] in confirmed_families
            policy_reason='EXTERNAL_QUALITY_API'
            if self.policy is not None:
                object_quality,family_confirmed,policy_reason=self.policy.assess(
                    measured,calibration,capture_geometry_id,(w,h))
            # Keep diagnostic and dimension coordinates together in crop-local
            # space, with an explicit translation to the original frame.
            evidence = compare_dimensions(measured,self.specs,object_quality,
                                          calibration_validated=calibration['validated'],
                                          family_confirmed=family_confirmed)
            evidence['measurement_policy_reason']=policy_reason
            evidence.update(measurement=measured,crop_xyxy=[left,top,right,bottom],
                            measurement_coordinate_system='crop-local pixels; add crop origin for original frame',
                            calibration=copy.deepcopy(calibration))
            obj['size_evidence'] = evidence
        enriched.update(status='REVIEW',predicted_label=None,prediction_score=None,score_type=None,
                        message='P06 provisional size evidence; final decision requires shared inspection rules')
        return enriched
