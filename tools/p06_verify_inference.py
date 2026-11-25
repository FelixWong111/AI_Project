"""Real GPU P06/P07 detection-only verification; no training or Test access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.camera import Frame
from src.p07_bridge import load_backend,validate_result
from src.p07_ui import InferenceWorker
from tools.p06_accepted_data import accepted_plan,check_sources
from tools.p06_train_baseline import iou
from tools.p06_preview_annotations import digest

def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def read_image(path):
    image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f'Cannot decode {path}')
    return image

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/p06_inference_checks')
    args=parser.parse_args();out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'output') or out.exists():raise ValueError('Need a new output directory below repository output/')
    out.mkdir(parents=True)
    try:
        paths=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
        protected={p:digest(ROOT/p) for p in paths}
        save(out/'protected_git_files_sha256.json',protected)
        source=ROOT/'data/p03/development_v2'
        plan=accepted_plan(source,ROOT/'output/p06_gt_correction_01/acceptance')
        frozen=json.loads((ROOT/'config/baseline3.json').read_text(encoding='utf-8'))
        p07_config=json.loads((ROOT/'config/p07.json').read_text(encoding='utf-8'))
        p07_config.update(bridge='src.models.baseline3.inference:create_backend',model_config={'config_path':'config/baseline3.json'})
        save(out/'p07_detection_only.json',p07_config)
        backend=load_backend('real','baseline3',p07_config)
        save(out/'metadata.json',backend.metadata)
        expected=dict.fromkeys('12345',1);results=[]
        references=json.loads((ROOT/'output/p06_runs/p03_baseline3_gtfix01_revalidation/val_predictions/prediction_summary.json').read_text(encoding='utf-8'))
        ref={r['image_id']:r for r in references}
        max_coordinate_error=0.;max_score_error=0.;min_iou=1.
        for row in plan['records']:
            if row['split']!='val':continue
            image=read_image(source/'images/validation'/row['image_id']);before=hashlib.sha256(image.tobytes()).hexdigest()
            backend.synchronize();start=perf_counter();result=backend.infer(image,expected);backend.synchronize()
            seconds=perf_counter()-start
            validate_result(result,'baseline3',image.shape)
            assert result['status']=='REVIEW' and result['predicted_label'] is None
            assert hashlib.sha256(image.tobytes()).hexdigest()==before
            reference=ref[row['image_id']];assert list(result['counts'].values())==reference['class_counts']
            unmatched=list(reference['predictions'])
            for obj in result['objects']:
                box=[obj[k] for k in ['xmin','ymin','xmax','ymax']]
                candidates=[p for p in unmatched if p['class_id']==obj['class_id']];assert candidates
                match=max(candidates,key=lambda p:iou(box,p['xyxy']));overlap=iou(box,match['xyxy'])
                error=max(abs(a-b) for a,b in zip(box,match['xyxy']));score_error=abs(obj['score']-match['confidence'])
                assert overlap>=.999 and score_error<=.001,'Reference inference behavior changed'
                max_coordinate_error=max(max_coordinate_error,error);max_score_error=max(max_score_error,score_error);min_iou=min(min_iou,overlap)
                unmatched.remove(match)
            assert not unmatched
            results.append(dict(image_id=row['image_id'],seconds=seconds,**result))
        assert len(results)==46
        save(out/'validation_observations.json',results)
        # Blank image and a human-labelled empty training background are engineering cases, not accuracy estimates.
        empty_cases=[]
        for name,image in [('synthetic_white',np.full((480,640,3),255,np.uint8)),('train_fail123',read_image(source/'images/train/fail123.jpg'))]:
            result=backend.infer(image,expected);validate_result(result,'baseline3',image.shape)
            empty_cases.append(dict(case=name,**result))
        save(out/'empty_cases.json',empty_cases)
        worker=InferenceWorker();frame=Frame(1,read_image(source/'images/validation/pass59.jpg'),perf_counter(),0.,'development_validation')
        try:
            # Spy wraps the real GPU implementation; it never supplies fabricated predictions.
            with patch.object(backend._model,'predict',wraps=backend._model.predict) as spy:
                assert worker.submit(backend,'baseline3',frame,expected,'unused',1)
                worker.future.result(timeout=60);first=worker.poll();assert first['result']['status']=='REVIEW'
                assert spy.call_count==1 and first['inference_ms'] is not None
                assert worker.submit(backend,'baseline3',frame,dict.fromkeys('12345',0),'unused',1,first['result'])
                worker.future.result(timeout=60);second=worker.poll()
                assert spy.call_count==1 and second['inference_ms'] is None
                assert first['result']['objects']==second['result']['objects'] and first['result']['counts']==second['result']['counts']
                save(out/'p07_worker.json',dict(status='PASS',real_gpu_predict_calls=spy.call_count,first_result=first['result'],
                    first_inference_ms=first['inference_ms'],rejudge_inference_ms=second['inference_ms'],rejudge_observations_unchanged=True))
        finally:worker.close()
        # One small real development image through the unmodified application, headless, no camera opened.
        command=[sys.executable,'-B',str(ROOT/'app.py'),'--backend','real','--model','baseline3','--config',str(out/'p07_detection_only.json'),
            '--image',str(source/'images/train/fail02.jpg'),'--headless','--max-frames','1','--timeout','60','--output-dir',str(out/'p07_headless')]
        with (out/'p07_app.log').open('x',encoding='utf-8') as log:
            process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        assert process.returncode==0,'P07 headless application failed; inspect p07_app.log'
        check_sources(plan)
        assert all(digest(ROOT/p)==h for p,h in protected.items()),'Tracked team files changed'
        assert digest(ROOT/frozen['weights'])==frozen['weights_sha256']
        completion=dict(status='PASS',training_started=False,test_accessed=False,validation_images=46,
            total_objects=sum(len(r['objects']) for r in results),counts_and_p07_validation='PASS',
            previous_640_reference=dict(min_box_iou=min_iou,max_coordinate_error=max_coordinate_error,max_score_error=max_score_error),
            empty_case_counts={r['case']:r['counts'] for r in empty_cases},
            p07_factory_and_worker='PASS',rejudge_real_predict_calls=0,p07_app=dict(command=command,exit_code=process.returncode,evidence='OFFLINE_HEADLESS_ONLY'),
            protected_tracked_files_unchanged=len(protected),source_files_unchanged=len(plan['source_sha256']),weights_unchanged=True,
            limitations=['All status REVIEW: detection/counting only, no PASS/FAIL/size/open-set decision',
                'Existing fine/thick confusion and cross-class overlapping boxes are retained, not silently repaired',
                'No live camera/GUI/FPS validation; no Test evaluation'])
        save(out/'completion.json',completion);print(json.dumps(completion,indent=2));return 0
    except BaseException:
        import traceback
        (out/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':raise SystemExit(main())
