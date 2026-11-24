"""Frozen YOLO11n observations and P07-compatible detection-only factory.

No training, dataset scanning, PASS/FAIL decision, size calibration or P08 dependency.
"""
from __future__ import annotations

import copy
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import threading

import numpy as np
from src.data.p06_labelme import CLASSES

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_CONFIG=ROOT/'config/baseline3.json'
PARAMETERS=dict(imgsz=640,conf=.25,iou=.7,device=0,batch=1,half=False,
                agnostic_nms=False,augment=False,max_det=300)

def _require(value,message):
    if not value:raise ValueError(message)

def _expected(value):
    _require(isinstance(value,dict) and set(value)==set('12345'),'expected_counts must contain exactly string keys 1..5')
    _require(all(type(n) is int and n>=0 for n in value.values()),'expected_counts values must be non-negative integers, not bool')

def _frame(value):
    _require(isinstance(value,np.ndarray) and value.dtype==np.uint8 and value.ndim==3
             and value.shape[2]==3 and min(value.shape[:2])>0,'frame must be nonempty uint8 HxWx3 BGR')

def _observations(rows,width,height):
    """Count exactly the retained rows; never clip, relabel or silently discard boxes."""
    counts=dict.fromkeys('012345',0);objects=[]
    for row in rows:
        _require(len(row)==6 and all(type(v) in (int,float) and math.isfinite(v) for v in row),'Invalid/nonfinite detection row')
        x1,y1,x2,y2,score,cid=row
        _require(float(cid).is_integer() and 0<=cid<6,'Unknown or fractional class ID')
        _require(0<=x1<x2<=width and 0<=y1<y2<=height,'Box outside original image or zero area')
        _require(0<=score<=1,'Detection confidence outside [0,1]')
        cid=int(cid);counts[str(cid)]+=1
        objects.append(dict(detection_id=len(objects),class_id=cid,xmin=float(x1),ymin=float(y1),
                            xmax=float(x2),ymax=float(y2),score=float(score)))
    return dict(status='REVIEW',predicted_label=None,prediction_score=None,score_type=None,
        non_catalogue_count=counts['0'],counts=counts,objects=objects,frame_size=dict(width=width,height=height),
        message='P06 detection/counts only; PASS/FAIL, size checking and complete rejection are not evaluated.'
                +(' max_det reached: counts may be truncated.' if len(objects)>=PARAMETERS['max_det'] else ''))

def _cuda_runtime():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()<1:
        raise RuntimeError('CUDA device 0 unavailable; P06 will not fall back to CPU or Mock')
    return torch

def _load_model(weights):
    # Process-local settings only; never download a replacement checkpoint.
    cache=ROOT/'outputs/baseline3/cache';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR',str(cache))
    os.environ.setdefault('YOLO_OFFLINE','true')
    from ultralytics import YOLO
    return YOLO(str(weights),task='detect')

class P06Backend:
    """One loaded model, serialized inference, stateless count-only rejudgement."""
    is_mock=False

    def __init__(self,frozen,weights,runtime,model):
        self._runtime=runtime;self._model=model;self._lock=threading.Lock()
        self._params=copy.deepcopy(frozen['inference'])
        self.metadata=dict(model_id='baseline3',model_version='p06-yolo11n-640-epoch58-'+frozen['weights_sha256'][:12],
            model_file_size_bytes=weights.stat().st_size,weights=str(weights),weights_sha256=frozen['weights_sha256'],
            output_capability='object_level',class_catalogue=dict(enumerate(CLASSES)),
            thresholds=copy.deepcopy(self._params),score_convention=frozen['score_convention'],
            preprocessing='uint8 BGR -> Ultralytics aspect-preserving letterbox imgsz=640; original-pixel xyxy output; no EXIF transform on incoming frames',
            coordinate_convention='continuous original-image pixels, top-left xyxy; 0<=xmin<xmax<=width, 0<=ymin<ymax<=height',
            hardware=dict(device='cuda:0',gpu=runtime.cuda.get_device_name(0)),
            rejection=frozen['rejection_scope'],size_tolerance=None,calibration=None,
            decision_stage='detection_only_REVIEW; no PASS/FAIL decision',ultralytics_version=frozen['ultralytics_version'])

    def synchronize(self):
        self._runtime.cuda.synchronize(0)

    def infer(self,frame_bgr,expected_counts):
        _frame(frame_bgr);_expected(expected_counts)
        height,width=map(int,frame_bgr.shape[:2])
        with self._lock:
            results=self._model.predict(source=frame_bgr.copy(order='C'),**self._params,
                                        save=False,verbose=False,stream=False)
            _require(len(results)==1,'Expected exactly one image result')
            result=results[0]
            _require(tuple(result.orig_shape)==(height,width),'Result does not match original input dimensions')
            _require(result.boxes is not None and result.boxes.data.is_cuda,'Detection results were not on CUDA')
            rows=result.boxes.data.cpu().tolist()
        return _observations(rows,width,height)

    def rejudge(self,result,expected_counts):
        """Validate/copy cached observations; intentionally never call the detector."""
        _expected(expected_counts)
        _require(isinstance(result,dict) and result.get('status')=='REVIEW','Expected a P06 detection-only cached result')
        size=result.get('frame_size',{})
        _require(set(size)=={'width','height'} and all(type(v) is int and v>0 for v in size.values()),'Missing original frame size')
        objects=result.get('objects')
        _require(isinstance(objects,list),'Missing cached objects')
        rows=[]
        for index,obj in enumerate(objects):
            _require(type(obj.get('detection_id')) is int and obj['detection_id']==index,'Cached detection IDs invalid')
            _require(type(obj.get('class_id')) is int,'Cached class ID invalid')
            rows.append([obj[k] for k in ['xmin','ymin','xmax','ymax','score','class_id']])
        fresh=_observations(rows,size['width'],size['height'])
        cached_counts=result.get('counts')
        _require(isinstance(cached_counts,dict) and set(cached_counts)==set('012345')
                 and all(type(v) is int and v>=0 for v in cached_counts.values())
                 and type(result.get('non_catalogue_count')) is int,'Invalid cached count types')
        _require(fresh['counts']==cached_counts and fresh['non_catalogue_count']==result.get('non_catalogue_count'),
                 'Cached objects and counts disagree')
        return fresh

def create_backend(*,model_id,config):
    """P07 module:factory entry. config is model_config, not the whole P07 config."""
    _require(model_id=='baseline3','P06 currently provides baseline3 only, not final/baseline1/baseline2')
    _require(isinstance(config,dict) and set(config)<={'config_path'},'model_config accepts only config_path')
    path=Path(config.get('config_path',DEFAULT_CONFIG))
    path=path if path.is_absolute() else ROOT/path
    frozen=json.loads(path.read_text(encoding='utf-8-sig'))
    _require(frozen.get('schema_version')==1 and frozen.get('model_id')==model_id and frozen.get('architecture')=='YOLO11n','Invalid frozen model configuration')
    _require(frozen.get('class_names')=={str(i):name for i,name in enumerate(CLASSES)},'Frozen class mapping differs')
    params=frozen.get('inference',{})
    _require(set(params)==set(PARAMETERS) and all(type(params[k]) is type(v) and params[k]==v for k,v in PARAMETERS.items()),
             'Frozen inference settings differ; do not silently change resolution/threshold/device')
    _require(version('ultralytics')==frozen.get('ultralytics_version')=='8.3.236','Unexpected Ultralytics version')
    weights=Path(frozen['weights']);weights=weights if weights.is_absolute() else ROOT/weights
    with weights.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    _require(actual==frozen.get('weights_sha256'),'Checkpoint SHA-256 mismatch')
    runtime=_cuda_runtime();model=_load_model(weights)
    _require(model.task=='detect' and model.names==dict(enumerate(CLASSES)),'Checkpoint class/task mapping differs')
    model.to('cuda:0')
    device=next(model.model.parameters()).device
    _require(device.type=='cuda' and device.index==0,'Model was not loaded on CUDA device 0')
    return P06Backend(frozen,weights,runtime,model)
