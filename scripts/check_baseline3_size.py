"""Real integration diagnostics on approved Validation only, never formal Test.

No training, changed thresholds or synthetic physical calibration. An optional
pipeline config is a debug entry for a separately recorded real workstation.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
from src.camera import Frame
from src.models.baseline1.features import load_image_bgr
from src.p07_bridge import load_backend
from src.p07_dashboard import render
from src.p07_ui import InferenceWorker
from src.work_orders import reference_counts

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-config',type=Path,help='Existing separately recorded config, never a generated calibration')
    args=parser.parse_args()
    out=ROOT/'outputs/p07_size_integration_20261125'/str(time.time_ns());out.mkdir(parents=True)
    def write(name,value):
        (out/name).write_bytes((json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode())
    data=json.loads((ROOT/'results/data_manifests/development.json').read_text(encoding='utf-8'))
    records={r['image_id']:r for r in data['records'] if r['split']=='validation'}
    names=['pass102.jpg','fail27.jpg','fail140.jpg','fail130.jpg']
    assert all(n in records for n in names)
    config=json.loads((ROOT/'config/p07.json').read_text(encoding='utf-8'))
    if args.pipeline_config:
        config['model_config']['baseline3_pipeline_config']=str(args.pipeline_config.resolve())
    write('runtime.json',config)
    fixed=reference_counts()
    backend=load_backend('real','baseline3',config)
    frozen=load_backend('real','baseline3',json.loads((ROOT/'config/p07_yolo_frozen.json').read_text(encoding='utf-8')))
    evidence=dict(official_test='NOT_RUN',physical_camera='NOT_RUN',physical_calibration='NOT_VERIFIED',
                  live_performance='NOT_RUN',config=str(out/'runtime.json'),commands=[],cases=[],metadata=backend.metadata)
    images={}
    for name in names:
        r=records[name];p=ROOT/r['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        image=load_image_bgr(str(p));images[name]=image
        before=image.copy()
        with patch.object(backend.native.provider.detector,'infer',wraps=backend.native.provider.detector.infer) as infer:
            result=backend.infer(image,fixed)
            changed=backend.rejudge(result,dict.fromkeys('12345',0))
            restored=backend.rejudge(changed,fixed)
            assert infer.call_count==1
        raw=frozen.infer(image,fixed)
        assert raw['counts']==result['p06_evidence']['raw_observations']['counts']
        if not result['p06_evidence']['foreground_enabled']:
            assert raw['counts']==result['counts']
        assert changed['counts']==result['counts'] and changed['objects']==result['objects']
        assert restored['status']==result['status'] and np.array_equal(image,before)
        if not args.pipeline_config:
            assert result['status']==('REVIEW' if name=='pass102.jpg' else 'FAIL')
        evidence['cases'].append(dict(image_id=name,status=result['status'],frozen_status=raw['status'],
            counts=result['counts'],inspection=result['inspection'],
            measurements=[o.get('size_evidence') for o in result['objects']],
            detector_calls_with_two_rejudgements=1,observations_preserved=True))
        write(name+'.json',{k:v for k,v in result.items() if not k.startswith('_')})
        if name=='pass102.jpg':
            canvas=render(Frame(1,image,0,0,'validation'),result,fixed,'baseline3','real',
                dict(show_diagnostics=True,diagnostic_page=0,source='image',device='cuda:0'))
            cv2.imencode('.png',canvas)[1].tofile(out/'size-details.png')
    worker=InferenceWorker()
    try:
        worker.submit(backend,'baseline3',Frame(1,images[names[0]],0,0,'validation'),fixed,None,0)
        deadline=time.monotonic()+90
        answer=None
        while answer is None and time.monotonic()<deadline:
            answer=worker.poll();time.sleep(.01)
        assert answer and answer['result']['status']!='ERROR'
        evidence['real_worker']='PASS'
    finally:worker.close()
    def command(arguments,log,expected_exit=0):
        run=subprocess.run([sys.executable,'-B',*arguments],cwd=ROOT,capture_output=True,
                           encoding='utf-8',errors='replace',timeout=240,
                           env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        (out/log).write_bytes((run.stdout+'\n'+run.stderr).encode())
        evidence['commands'].append(dict(args=arguments,exit_code=run.returncode))
        assert run.returncode==expected_exit,run.stdout+run.stderr
    write('counts.json',[dict(image_id=n,expected_counts=fixed) for n in names])
    source=(ROOT/records[names[0]]['path']).parent
    for mode,cfg in [('size',str(out/'runtime.json')),('frozen','config/p07_yolo_frozen.json')]:
        command(['predict.py','--config',cfg,'--input_dir',str(source),'--model','baseline3',
                 '--counts_file',str(out/'counts.json'),'--output_dir',str(out/(mode+'-batch'))],mode+'-batch.log')
        with (out/(mode+'-batch')/'images.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        assert len(rows)==4 and all(r['status']!='ERROR' for r in rows)
        assert (out/(mode+'-batch')/'diagnostics.json').is_file()
    # Real app image path: same size mode and full diagnostic sidecar.
    command(['app.py','--config',str(out/'runtime.json'),'--model','baseline3','--image',str(source/names[0]),
             '--headless','--max-frames','1','--timeout','90','--output-dir',str(out/'image-app')],'image-app.log')
    snapshot=next((out/'image-app').rglob('diagnostics.json'))
    diag=json.loads(snapshot.read_text(encoding='utf-8'))['images'][0]['result']
    assert diag['status']==evidence['cases'][0]['status']
    assert diag['baseline3_mode']=='size_verified'
    # Offline video is a plumbing diagnostic, not new physical camera evidence.
    video=out/'validation-video.avi'
    writer=cv2.VideoWriter(video.relative_to(ROOT).as_posix(),cv2.VideoWriter_fourcc(*'MJPG'),5,(1280,720))
    assert writer.isOpened()
    for _ in range(5):writer.write(cv2.resize(images[names[0]],(1280,720)))
    writer.release()
    command(['app.py','--config',str(out/'runtime.json'),'--model','baseline3','--video',str(video),
             '--headless','--max-frames','1','--timeout','90','--output-dir',str(out/'video-app')],'video-app.log')
    video_diag=json.loads(next((out/'video-app').rglob('diagnostics.json')).read_text(encoding='utf-8'))
    vr=video_diag['images'][0]['result'];assert vr['baseline3_mode']=='size_verified' and vr['status']!='ERROR'
    if not args.pipeline_config:
        assert 'CALIBRATION_RESOLUTION_MISMATCH' in vr['inspection']['review_reasons']
    evidence.update(real_image_app='PASS',offline_video_app='PASS; not live-camera validation',
                    shared_mode_across_image_video_batch=True)
    # Broken integrated path must produce ERROR, never silently return YOLO-only results.
    broken=copy.deepcopy(config);broken['model_config']['baseline3_pipeline_config']=str(out/'missing.json')
    write('broken-runtime.json',broken)
    command(['predict.py','--config',str(out/'broken-runtime.json'),'--input_dir',str(source),
             '--model','baseline3','--counts_file',str(out/'counts.json'),'--output_dir',str(out/'error-batch')],
            'error-batch.log',expected_exit=1)
    with (out/'error-batch/images.csv').open(encoding='utf-8') as f:assert all(r['status']=='ERROR' for r in csv.DictReader(f))
    evidence['missing_pipeline_no_fallback']='PASS'
    write('acceptance.json',evidence)
    root=out.parent
    (root/'latest.json').write_bytes(json.dumps(dict(directory=out.relative_to(ROOT).as_posix())).encode())
    print('PASS',out.relative_to(ROOT))


if __name__=='__main__':main()
