"""Shared inspection requirements; deliberately independent of ground truth/models."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / 'config/inspection.json'
COUNT_KEYS = tuple('12345')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def validate_counts(value):
    if not isinstance(value, Mapping) or set(value) != set(COUNT_KEYS):
        raise ValueError('expected_counts must contain exactly string keys "1".."5"')
    if any(type(v) is not int or v < 0 for v in value.values()):
        raise ValueError('counts must be non-negative integers (not bools)')
    return {key: value[key] for key in COUNT_KEYS}


def reference_counts():
    config = json.loads(REFERENCE_CONFIG.read_text(encoding='utf-8-sig'))
    if type(config['non_catalogue_expected']) is not int or config['non_catalogue_expected'] != 0:
        raise ValueError('ID 0 expected count must remain zero')
    return validate_counts(config['reference_expected_counts'])


def image_filename(value):
    if not isinstance(value, str) or not value or value in {'.', '..'}:
        raise ValueError('image_id must be a filename')
    if any(token in value for token in ('/', '\\', ':', '\x00')):
        raise ValueError(f'Invalid image_id: {value!r}')
    return value


def validate_cases(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError('expected_counts.json must be a nonempty list')
    seen, cases = set(), []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'image_id', 'expected_counts'}:
            raise ValueError('Each case must contain only image_id and expected_counts; no ground truth')
        iid = image_filename(entry['image_id'])
        if Path(iid).suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'image_id needs a real image extension, not a case ID: {iid}')
        if iid in seen:
            raise ValueError(f'Duplicate image_id: {iid}')
        seen.add(iid)
        cases.append(dict(image_id=iid, expected_counts=validate_counts(entry['expected_counts'])))
    return cases


def load_cases(path):
    return validate_cases(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def unsupported_counts():
    fixed = ','.join(str(v) for v in reference_counts().values())
    return dict(status='UNSUPPORTED_COUNTS', predicted_label=None, prediction_score=None, score_type='none',
                objects=[], counts=None, non_catalogue_count=None,
                message=f'Only fixed counts [{fixed}] supported. Restore defaults or switch to Baseline3.',
                _p07_inference_executed=False)


def judge_observed(result, expected):
    """Rejudge one existing detector observation; never call a model or alter boxes."""
    expected = validate_counts(expected)
    updated = copy.deepcopy(result)
    if updated['status'] in ('ERROR', 'REVIEW'):
        return updated
    counts = updated.get('counts')
    if not isinstance(counts, dict) or set(counts) != set('012345') or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError('Detected counts must contain non-negative integers for IDs 0..5')
    status = 'PASS' if counts['0'] == 0 and all(counts[c] == expected[c] for c in COUNT_KEYS) else 'FAIL'
    updated.update(status=status, predicted_label=status, prediction_score=None, score_type='none')
    return updated
