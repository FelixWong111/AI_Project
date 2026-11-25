"""Real frozen GPU inference on a synthetic frame; no dataset access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.p06_pipeline import create_backend
from src.p07_bridge import validate_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/p06_pipeline_stage3_gpu')
    dest=parser.parse_args().output.resolve()
    if dest.exists() or not dest.is_relative_to(ROOT/'output'):parser.error('Choose a new output directory')
    dest.mkdir(parents=True)
    bg=np.full((480,640,3),240,np.uint8)
    frame=bg.copy()
    cv2.rectangle(frame,(220,160),(270,330),(25,)*3,-1)
    cv2.rectangle(frame,(270,210),(430,270),(25,)*3,-1)
    cv2.imwrite(str(dest/'synthetic_background.png'),bg)
    cv2.imwrite(str(dest/'synthetic_input.png'),frame)
    cfg=json.loads((ROOT/'config/p06_pipeline_candidate.json').read_text(encoding='utf-8'))
    cfg.update(calibration_path=None,capture_geometry_id='SYNTHETIC_GPU_PIPELINE_FIXTURE',
               background=dict(path=str(dest/'synthetic_background.png'),
                               file_sha256=hashlib.sha256((dest/'synthetic_background.png').read_bytes()).hexdigest(),
                               reference_id='SYNTHETIC_WHITE_BACKGROUND',roi=None))
    path=dest/'runtime_config.json';path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    backend=create_backend(model_id='baseline3',config={'pipeline_config_path':str(path)})
    expected=dict.fromkeys('12345',1)
    backend.synchronize();start=time.perf_counter()
    result=backend.infer(frame,expected)
    backend.synchronize();elapsed=time.perf_counter()-start
    validate_result(result,'baseline3',frame.shape)
    rejudged=backend.rejudge(result,dict.fromkeys('12345',0))
    assert rejudged['objects']==result['objects']
    for name,value in [('result.json',result),('metadata.json',backend.metadata),
                       ('verification.json',dict(evidence='REAL_FROZEN_GPU_INFERENCE_ON_SYNTHETIC_FRAME',
                         elapsed_seconds=elapsed,p07_result_valid=True,cached_rejudge=True,
                         real_part_accuracy='NOT_RUN',physical_size_accuracy='NOT_RUN'))]:
        (dest/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('GPU pipeline completed',len(result['objects']),'detections',
          len(result['p06_evidence']['unresolved_candidates']),'unresolved candidates')


if __name__=='__main__':main()
