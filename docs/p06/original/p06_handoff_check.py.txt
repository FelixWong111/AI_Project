"""Dataset-independent, real-GPU P07 factory/worker check. Never trains."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.camera import Frame
from src.p07_bridge import load_backend,validate_result
from src.p07_ui import InferenceWorker

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,help='Optional non-Test local photo; default is a synthetic white frame')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/p06_handoff_check')
    parser.add_argument('--config',type=Path,default=ROOT/'config/p06_p07_detection_only.json',
                        help='P07 config; use config/p06_p07_pipeline_candidate.json for integrated evidence')
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'output'):raise ValueError('Choose a NEW directory under repository output/')
    if args.image:
        path=args.image.resolve()
        if any(part.lower() in {'test','test_v2','test_set'} for part in path.parts):raise ValueError('Do not use the independent Test set for handoff checks')
        image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Cannot decode the supplied image')
        evidence='USER_SUPPLIED_NON_TEST_PHOTO'
    else:
        image=np.full((480,640,3),255,np.uint8);evidence='SYNTHETIC_FRAME_PIPELINE_ONLY'
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    backend=load_backend('real','baseline3',config)
    import torch
    tensor=torch.tensor([1.,2.,3.],device='cuda:0').square().sum()
    assert tensor.is_cuda and tensor.item()==14.
    expected=config['expected_counts'];worker=InferenceWorker()
    try:
        packet=Frame(1,image,perf_counter(),0.,evidence)
        assert worker.submit(backend,'baseline3',packet,expected,'unused',1)
        worker.future.result(timeout=90);first=worker.poll()
        validate_result(first['result'],'baseline3',image.shape)
        if first['result']['status']!='REVIEW':raise RuntimeError(first['result'].get('message','Inference failed'))
        assert worker.submit(backend,'baseline3',packet,dict.fromkeys('12345',0),'unused',1,first['result'])
        worker.future.result(timeout=90);second=worker.poll()
        assert second['result']['objects']==first['result']['objects'] and second['result']['counts']==first['result']['counts']
        assert second['inference_ms'] is None
        result=dict(status='PASS',meaning='Factory, CUDA, worker and cached observation compatibility; NOT model accuracy or tray PASS',
            evidence=evidence,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cuda_tensor_result=tensor.item(),
            metadata=backend.metadata,result=first['result'],rejudge_observations_unchanged=True,
            training_started=False,test_accessed=False)
        out.mkdir(parents=True,exist_ok=False)
        (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        print(json.dumps(result,indent=2,ensure_ascii=False));return 0
    finally:worker.close()

if __name__=='__main__':raise SystemExit(main())
