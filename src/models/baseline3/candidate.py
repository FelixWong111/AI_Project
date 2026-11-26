"""Independent YOLO11n candidate factory; reuse P06 observations, size evidence and P07 decisions."""
from __future__ import annotations
import copy
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

from .inference import ROOT, PARAMETERS, P06Backend, _cuda_runtime, _load_model, _require
from src.data.p06_labelme import CLASSES


def detector_factory(*, model_id, config):
    _require(model_id == 'baseline3' and set(config) == {'config_path'}, 'Candidate requires an explicit detector config')
    path=Path(config['config_path']);path=path if path.is_absolute() else ROOT/path
    frozen=json.loads(path.read_text(encoding='utf-8-sig'))
    _require(frozen.get('candidate_schema') == 'yolo11n-candidate/v1', 'Not an independent candidate configuration')
    _require(frozen.get('architecture') == 'YOLO11n' and frozen.get('model_id') == model_id, 'Candidate architecture differs')
    _require(frozen.get('class_names') == {str(i):n for i,n in enumerate(CLASSES)}, 'Candidate class mapping differs')
    params=frozen['inference']
    _require(set(params) == set(PARAMETERS), 'Missing/unknown inference parameter')
    for key, value in PARAMETERS.items():
        _require(type(params[key]) is type(value), 'Inference parameter type differs: '+key)
        if key in ('conf','iou'):
            _require(math.isfinite(params[key]) and 0 < params[key] < 1, 'Invalid threshold: '+key)
        else:
            _require(params[key] == value, 'Only explicitly selected confidence/NMS may differ: '+key)
    _require(version('ultralytics') == frozen['ultralytics_version'] == '8.3.236', 'Ultralytics version changed')
    weights=ROOT/frozen['weights']
    with weights.open('rb') as f:
        _require(hashlib.file_digest(f,'sha256').hexdigest() == frozen['weights_sha256'], 'Candidate weight SHA mismatch')
    runtime=_cuda_runtime();model=_load_model(weights)
    _require(model.task == 'detect' and model.names == dict(enumerate(CLASSES)), 'Candidate checkpoint classes/task differ')
    model.to('cuda:0')
    backend=P06Backend(frozen,weights,runtime,model)
    backend.metadata.update(model_version=frozen['candidate_version']+'-'+frozen['weights_sha256'][:12],
                            candidate_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            initialization=frozen['initialization'], selection_scope='Train/Validation only')
    return backend


def create_backend(*, model_id, config):
    from src.adapters.baseline3 import FrozenYOLOBackend, SizeVerifiedBackend
    from src.p06_pipeline import create_backend as pipeline_factory
    from src.p07_bridge import repository_path
    _require(model_id == 'baseline3', 'Candidate is not Final or an image classifier')
    mode=config.get('baseline3_mode')
    if mode == 'yolo_frozen':
        return FrozenYOLOBackend(repository_path(config['baseline3_config']), detector_factory=detector_factory)
    if mode == 'size_verified':
        def provider_factory(**kwargs):
            return pipeline_factory(**kwargs,detector_factory=detector_factory)
        return SizeVerifiedBackend(repository_path(config['baseline3_pipeline_config']),
                                   config.get('baseline3_acceptance_approval'),provider_factory=provider_factory)
    raise ValueError('Candidate mode must be explicit; no fallback to count-only')
