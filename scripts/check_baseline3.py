"""Real CUDA detector integration checks, using validation data only."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.camera import Frame
from src.data.development import ROOT, digest, write_json
from src.models.baseline1.features import load_image_bgr
from src.p07_bridge import load_backend, validate_result
from src.p07_ui import InferenceWorker
from src.work_orders import reference_counts


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    before=dict(models={p.relative_to(ROOT).as_posix():digest(p) for d in ('baseline1','baseline2')
                        for p in (ROOT/'models'/d).rglob('*') if p.is_file()},
                baseline1_config=digest(ROOT/'config/baseline1.json'),
                p03={p.relative_to(ROOT).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
                     for d in ROOT.glob('P03_*') for p in d.rglob('*') if p.is_file()})
    out=ROOT/'outputs/p06_merge'/str(time.time_ns());out.mkdir(parents=True)
    data=json.loads((ROOT/'outputs/training/data/manifest.json').read_text(encoding='utf-8'))
    validation=[r for r in data['records'] if r['split']=='validation']
    assert len(validation)==46
    chosen=next(r for r in validation if r['label']=='PASS')
    path=ROOT/chosen['path'];assert digest(path)==chosen['sha256']
    config=json.loads((ROOT/'config/p07.json').read_text(encoding='utf-8'))
    fixed=reference_counts();backend=load_backend('real','baseline3',config)
    frame=load_image_bgr(str(path));original=frame.copy()
    engine=backend.native.detector._model
    with patch.object(engine,'predict',wraps=engine.predict) as predict:
        worker=InferenceWorker()
        try:
            worker.submit(backend,'baseline3',Frame(1,frame,time.perf_counter(),0,'validation'),fixed,'unused',1)
            worker.future.result(timeout=90);first=worker.poll();result=first['result']
            assert result['status'] in ('PASS','FAIL') and predict.call_count==1
            worker.submit(backend,'baseline3',first['packet'],dict.fromkeys('12345',0),'unused',1,result)
            worker.future.result(timeout=30);changed=worker.poll()
            assert predict.call_count==1 and changed['inference_ms'] is None
            assert result['objects']==changed['result']['objects'] and result['counts']==changed['result']['counts']
        finally:worker.close()
    assert np.array_equal(frame,original)
    evidence=dict(official_held_out_test='NOT_RUN',input_image=chosen['image_id'],metadata=backend.metadata,
                  real_cuda_worker='PASS',input_pixels_unchanged=True,count_only_rejudge_without_detection=True,
                  commands=[],gui_desktop_manual_click='NOT_RUN',camera_latency='NOT_RUN')
    write_json(out/'worker.json',dict(first={k:v for k,v in result.items() if not k.startswith('_')},
                                    rejudged=changed['result']))
    checkpoint=engine.ckpt
    record=dict(source='Embedded in P06 frozen best.pt; historical training evidence, not rerun here',
                weights_sha256=backend.metadata['weights_sha256'],date=checkpoint.get('date'),version=checkpoint.get('version'),
                license=checkpoint.get('license'),serialized_epoch=checkpoint.get('epoch'),
                reported_selected_epoch=58,reported_epoch_source='P06 model_version; stripped checkpoint epoch is not proof',
                train_args=checkpoint.get('train_args'),stored_metrics=checkpoint.get('train_metrics'),
                train_results=checkpoint.get('train_results'),official_held_out_test='NOT_RUN')
    write_json(ROOT/'models/baseline3/training_record.json',record)
    def command(args,log):
        run=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           encoding='utf-8',timeout=240)
        (out/log).write_text(run.stdout,encoding='utf-8')
        evidence['commands'].append(dict(command=subprocess.list2cmdline([sys.executable,'-B',*args]),exit_code=run.returncode))
        assert run.returncode==0,run.stdout
    # The baseline3 batch includes all approved validation images. No test input is enumerated.
    batch=out/'batch-baseline3'
    command(['predict.py','--input_dir',str(path.parent),'--model','baseline3','--counts_file',
             'outputs/work_orders/validation.json','--output_dir',str(batch)],'batch.log')
    images=rows(batch/'images.csv');objects=rows(batch/'objects.csv');counts=rows(batch/'counts.csv')
    assert len(images)==46 and len(counts)==46*6
    object_counts=Counter((r['image_id'],r['class_id']) for r in objects)
    for count in counts:
        assert int(count['observed_count'])==object_counts[(count['image_id'],count['class_id'])]
        assert int(count['difference'])==int(count['observed_count'])-int(count['expected_count'])
        assert int(count['expected_count'])==(0 if count['class_id']=='0' else fixed[count['class_id']])
    for image in images:
        assert image['prediction_score']==''
        if int(image['non_catalogue_count'])>0:assert image['status']=='FAIL'
    id0=[r['image_id'] for r in images if int(r['non_catalogue_count'])>0]
    evidence.update(validation_batch_rows=46,boxes_count_difference_consistency='PASS',
                    real_predicted_id0_fail_examples=id0,real_id0_rule='PASS' if id0 else 'NOT_OBSERVED')
    # UI processing on genuine results, via the actual app controller and real models.
    import app
    from src.p07_dashboard import regions
    backends={name:load_backend('real',name,dict(config,_live_scene_guard=True)) for name in ('baseline1','baseline2')}
    backends['baseline3']=backend
    callback=None;latest={};step=0;ticks=0;actions=[]
    seen=[];real_present=app.present
    def register(title,fn):
        nonlocal callback
        callback=fn
    def present(packet,result,expected,model,mode,state):
        latest.update(model=model,result=result,loading=state.get('loading'))
        if result and result['status'] in ('PASS','FAIL'):seen.append((model,result['status']))
        return real_present(packet,result,expected,model,mode,state)
    sequence=[('baseline1','snapshot'),('baseline1','model:baseline2'),('baseline2','snapshot'),
              ('baseline2','model:baseline3'),('baseline3','snapshot'),('baseline3','quit')]
    def wait(delay):
        nonlocal step,ticks
        ticks+=1
        if step<len(sequence):
            model,action=sequence[step]
            ready=latest.get('result') and latest['result']['status'] in ('PASS','FAIL')
            if callback and latest.get('model')==model and ready and not latest.get('loading'):
                x1,y1,x2,y2=dict(regions())[action]
                callback(cv2.EVENT_LBUTTONUP,(x1+x2)//2,(y1+y2)//2,0,None)
                actions.append(action);step+=1
        if ticks>6000:raise AssertionError('Controller timed out waiting for genuine model results')
        time.sleep(.005)
        return -1
    with patch.object(app,'load_backend',side_effect=lambda mode,model,cfg:backends[model]), \
         patch.object(app,'present',side_effect=present),patch.object(app,'list_cameras',return_value=[]), \
         patch.object(cv2,'namedWindow'),patch.object(cv2,'resizeWindow'),patch.object(cv2,'setMouseCallback',side_effect=register), \
         patch.object(cv2,'imshow'),patch.object(cv2,'waitKey',side_effect=wait), \
         patch.object(cv2,'getWindowProperty',return_value=1),patch.object(cv2,'destroyAllWindows'):
        assert app.main(['--model','baseline1','--image',str(path),'--output-dir',str(out/'controller')])==0
    exports=list((out/'controller').rglob('images.csv'));assert len(exports)==3
    assert {rows(p)[0]['model_id'] for p in exports}=={'baseline1','baseline2','baseline3'}
    evidence['gui_controller']=dict(status='PASS',scope='Real weights/inference; only native window calls stubbed',actions=actions,seen=seen)
    # Genuine headless app paths: normal validation and a detected ID0 example, if present.
    names=[chosen['image_id']]+id0[:1]
    for i,name in enumerate(names):
        record_path=next(ROOT/r['path'] for r in validation if r['image_id']==name)
        command(['app.py','--headless','--model','baseline3','--image',str(record_path),
                 '--max-frames','1','--timeout','90','--output-dir',str(out/f'app-{i}')],f'app-{i}.log')
    # Protect all pre-existing baseline1/2 artifacts and configuration, plus P03 metadata.
    assert all(digest(ROOT/file)==sha for file,sha in before['models'].items())
    assert digest(ROOT/'config/baseline1.json')==before['baseline1_config']
    after={p.relative_to(ROOT).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
           for d in ROOT.glob('P03_*') for p in d.rglob('*') if p.is_file()}
    assert after==before['p03']
    evidence.update(existing_model_hashes_unchanged=True,p03_metadata_unchanged=True,
                    size_calibration='NOT_PROVIDED',size_tolerances='NOT_PROVIDED',open_set_rejection='NOT_VERIFIED')
    write_json(out/'acceptance.json',evidence)
    write_json(ROOT/'outputs/p06_merge/latest.json',dict(directory=out.relative_to(ROOT).as_posix()))
    print('BASELINE3_ACCEPTANCE',out,flush=True)


if __name__=='__main__':main()
