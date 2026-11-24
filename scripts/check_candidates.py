"""Verify deployed candidates using validation images only, not held-out test."""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.data.development import ROOT, digest, write_json
from src.models.baseline1.features import load_image_bgr
from src.p07_bridge import load_backend
from src.p07_dashboard import render
from src.camera import Frame


def main():
    package=next(ROOT.glob('P03_*'))
    before={p.relative_to(package).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in package.rglob('*') if p.is_file()}
    data=json.loads((ROOT/'outputs/training/data/manifest.json').read_text(encoding='utf-8'))
    records=[r for r in data['records'] if r['split']=='validation']
    fixed=data['evidence']['expected_counts']
    folder=(ROOT/records[0]['path']).parent
    config=json.loads((ROOT/'config/p07.json').read_text(encoding='utf-8'))
    out=ROOT/'outputs/acceptance'/str(time.time_ns());out.mkdir(parents=True)
    evidence=dict(official_held_out_test='NOT_RUN',models={})
    for name in ('baseline1','baseline2'):
        result_dir=out/name
        command=[sys.executable,'-B','predict.py','--input_dir',str(folder),'--model',name,
                 '--output_dir',str(result_dir),'--counts_file','outputs/training/data/validation_counts.json']
        subprocess.run(command,cwd=ROOT,check=True)
        rows=list(csv.DictReader((result_dir/'images.csv').open(encoding='utf-8')))
        previous={r['image_id']:r for r in csv.DictReader((ROOT/'outputs/training'/name/'validation_predictions.csv').open(encoding='utf-8'))}
        assert len(rows)==len(records)
        for r in rows:
            old=previous[r['image_id']]
            assert r['predicted_label']==old['predicted_label']
            p=float(r['prediction_score'])
            p_fail=(p if r['predicted_label']=='FAIL' else 1-p) if name=='baseline1' else p
            assert np.isclose(p_fail,float(old['p_fail']),atol=2e-6)
            assert r['non_catalogue_count']==''
        assert not (result_dir/'objects.csv').exists() and not (result_dir/'counts.csv').exists()
        backend=load_backend('real',name,config)
        frame=load_image_bgr(str(ROOT/records[0]['path']))
        predicted=backend.infer(frame,fixed)
        altered=dict(fixed);altered['1']+=1
        unsupported=backend.infer(frame,altered)
        assert unsupported['status']=='UNSUPPORTED_COUNTS'
        assert unsupported['predicted_label'] is None and unsupported['prediction_score'] is None
        bad_config=dict(config,model_config=dict(config['model_config'],**{name+'_bundle':'models/DOES_NOT_EXIST'}))
        try:load_backend('real',name,bad_config)
        except FileNotFoundError:pass
        else:raise AssertionError('Missing model silently loaded')
        canvas=render(Frame(1,frame,0,0,'validation'),predicted,fixed,name,'real',
                      dict(version=backend.metadata['model_version'][:12],deployment='CANDIDATE',source='image',
                           device='cpu',score_meaning='P(label)' if name=='baseline1' else 'P(FAIL)'))
        cv2.imencode('.png',canvas)[1].tofile(out/(name+'.png'))
        gui_command=[sys.executable,'-B','app.py','--model',name,'--image',str(ROOT/records[0]['path']),
                     '--headless','--max-frames','1','--timeout','60','--output-dir',str(out/(name+'-headless'))]
        subprocess.run(gui_command,cwd=ROOT,check=True,timeout=120)
        evidence['models'][name]=dict(reload='PASS',batch_rows=len(rows),batch_predictions_match_training='PASS',
                                     unsupported_counts='PASS',missing_artifact='PASS',headless_app='PASS',
                                     batch_command=subprocess.list2cmdline(command),app_command=subprocess.list2cmdline(gui_command))
    after={p.relative_to(package).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in package.rglob('*') if p.is_file()}
    removed=sorted(before.keys()-after.keys())
    changed=sorted(p for p in before.keys()&after.keys() if before[p]!=after[p])
    added=sorted(after.keys()-before.keys())
    assert not removed and not changed, 'Original P03 files changed during this task'
    assert all(digest(ROOT/r['path'])==r['sha256'] for r in data['records'])
    assert all(digest(ROOT/p)==h for p,h in data['evidence']['source_files'].items())
    evidence.update(p03_original_structure_size_mtime_unchanged=True,
                    p03_added_during_task=added,p03_added_file_count=len(added),
                    p03_whole_directory_unchanged=not added,
                    development_image_and_csv_hashes_unchanged=True,
                    gui_manual_click_verification='NOT_VERIFIED: no target window exposed by desktop automation',
                    camera_latency='NOT_RUN')
    write_json(out/'acceptance.json',evidence)
    print('ACCEPTANCE',out)


if __name__=='__main__':main()
