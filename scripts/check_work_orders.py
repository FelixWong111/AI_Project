"""Execute U01-U09 on approved validation inputs, never held-out test photos."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from src.data.development import ROOT, digest, write_json
from src.models.baseline1.features import load_image_bgr
from src.p07_bridge import load_backend
from src.p07_ui import current_result
from src.work_orders import reference_counts, validate_counts
from scripts.generate_work_orders import generate


def run():
    before=dict(models={p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/'models').rglob('*') if p.is_file()},
                frozen_config=digest(ROOT/'config/baseline1.json'),
                p03_metadata={p.relative_to(ROOT).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
                              for d in ROOT.glob('P03_*') for p in d.rglob('*') if p.is_file()})
    out=ROOT/'outputs/work_orders_acceptance'/str(time.time_ns());out.mkdir(parents=True)
    manifest=json.loads((ROOT/'outputs/training/data/manifest.json').read_text(encoding='utf-8'))
    row=next(r for r in manifest['records'] if r['split']=='validation' and r['label']=='PASS')
    assert row['split']=='validation'
    path=ROOT/row['path'];frame=load_image_bgr(str(path))
    assert digest(path)==row['sha256']
    fixed=reference_counts();config=json.loads((ROOT/'config/p07.json').read_text(encoding='utf-8'))
    small=out/'validation_manifest.json'
    write_json(small,[dict(image_id=row['image_id'],split='validation')])
    variations={'extra_id5':fixed|{'5':2},'extra_id1':fixed|{'1':2},
                'zero_id1':fixed|{'1':0},'all_zero':dict.fromkeys('12345',0)}
    common=out/'common.json';write_json(common,generate(small,'validation'))
    functional={}
    for name,counts in variations.items():
        target=out/(name+'.json');write_json(target,generate(small,'validation',counts,'functional'));functional[name]=target
    evidence=dict(input_image=row['image_id'],input_split='validation',official_held_out_test='NOT_RUN',models={},
                  commands=[],functional_files={k:v.relative_to(ROOT).as_posix() for k,v in functional.items()})
    def command(args, logname, expected_code=0):
        completed=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,encoding='utf-8',timeout=120,
                                 env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        (out/logname).write_text(completed.stdout,encoding='utf-8')
        evidence['commands'].append(dict(command=subprocess.list2cmdline([sys.executable,'-B',*args]),exit_code=completed.returncode))
        assert completed.returncode==expected_code, completed.stdout
    for name in ('baseline1','baseline2'):
        backend=load_backend('real',name,dict(config,_live_scene_guard=True))
        classifier=backend.native
        extractor=classifier.features if name=='baseline1' else classifier.adapter.features
        method='extract_hog' if name=='baseline1' else 'extract'
        with patch.object(extractor,method,wraps=getattr(extractor,method)) as spy:
            good=backend.infer(frame,fixed)
            assert good['status'] in ('PASS','FAIL') and good['prediction_score'] is not None
            baseline_calls=spy.call_count
            for counts in variations.values():
                unsupported=backend.infer(frame,counts)
                assert unsupported['status']=='UNSUPPORTED_COUNTS'
                assert unsupported['predicted_label'] is None and unsupported['prediction_score'] is None
                # Direct native entry must also skip classification, not rely only on outer bridge.
                direct=classifier.infer(frame,counts)
                assert direct['status']=='UNSUPPORTED_COUNTS'
                assert spy.call_count==baseline_calls
            previous=backend.infer(frame,variations['extra_id5'])
            restored=backend.rejudge(previous,fixed)
            assert restored['status']==good['status']
            assert spy.call_count==baseline_calls+1
            second=backend.rejudge(backend.rejudge(restored,variations['extra_id5']),fixed)
            assert second['predicted_label']==restored['predicted_label']
            assert spy.call_count==baseline_calls+1
            for invalid in ({},fixed|{'1':-1},fixed|{'1':.5},fixed|{'1':True},fixed|{'0':0}):
                try:backend.infer(frame,invalid)
                except ValueError:pass
                else:raise AssertionError('Invalid counts accepted')
                assert spy.call_count==baseline_calls+1
        cached=dict(model=name,expected=fixed,result=good)
        view=current_result(name,backend,variations['extra_id5'],cached)
        assert view['status']=='UNSUPPORTED_COUNTS' and view['prediction_score'] is None
        assert current_result(name,backend,fixed,cached)==good
        other='baseline2' if name=='baseline1' else 'baseline1'
        assert current_result(other,backend,fixed,cached) is None
        # Real batch invocation, separate work order and output directory for each case.
        for case,file in dict(common=common,**functional).items():
            result_dir=out/(name+'-'+case)
            command(['predict.py','--input_dir',str(path.parent),'--model',name,'--output_dir',str(result_dir),
                     '--counts_file',str(file)],name+'-'+case+'.log')
            result=list(csv.DictReader((result_dir/'images.csv').open(encoding='utf-8')))[0]
            if case=='common':
                assert result['status']==good['status']
            else:
                assert result['status']=='UNSUPPORTED_COUNTS'
                assert result['predicted_label']==result['prediction_score']==result['non_catalogue_count']==''
        for case,file in (('common',common),('extra_id5',functional['extra_id5'])):
            app_out=out/(name+'-app-'+case)
            command(['app.py','--headless','--model',name,'--image',str(path),'--counts-file',str(file),
                     '--max-frames','1','--timeout','60','--output-dir',str(app_out)],name+'-app-'+case+'.log')
            results=list(app_out.rglob('images.csv'));assert len(results)==1
            result=next(csv.DictReader(results[0].open(encoding='utf-8')))
            assert result['status']==(good['status'] if case=='common' else 'UNSUPPORTED_COUNTS')
            if case!='common':assert result['predicted_label']==result['prediction_score']==''
        evidence['models'][name]=dict(U01_or_U02='PASS',U03_or_U04='PASS',U05='PASS',U06='PASS',U07='PASS',
                                    U08='PASS: real backend/current-view state; native mouse events covered by controller tests',
                                    U09='PASS',classification_skipped_for_four_unsupported_vectors=True,
                                    supported_result={k:good[k] for k in ('status','predicted_label','prediction_score')},
                                    model_version=backend.metadata['model_version'])
        # Frozen artifacts must actually encode the reference; this is not a config-only rewrite.
        record=json.loads((ROOT/'models'/name/'training_record.json').read_text(encoding='utf-8'))
        assert backend.fixed==record['data']['expected_counts']==fixed
        assert record['data']['manifest_sha256']==manifest['evidence']['manifest_sha256']
        for r in manifest['records']:
            derived='PASS' if r['counts'][:5]==list(fixed.values()) and r['counts'][5]==0 else 'FAIL'
            assert derived==r['label']
        print(name,'U01-U09 real validation/headless checks PASS',flush=True)
    invalid=out/'invalid.json';write_json(invalid,[dict(image_id=row['image_id'],expected_counts=fixed|{'1':-1})])
    command(['predict.py','--input_dir',str(path.parent),'--model','baseline1','--output_dir',str(out/'invalid-run'),
             '--counts_file',str(invalid)],'invalid-batch.log',expected_code=2)
    assert not (out/'invalid-run').exists()
    evidence['invalid_cli']='PASS: rejected before model load/output creation'
    for name in ('baseline3','final'):
        try:load_backend('real',name,config)
        except RuntimeError:
            evidence[name]='NOT_RUN: real detector/weights not configured; count-only contract checked in unit fixtures'
        else:evidence[name]='Available but not evaluated in this validation-only image-classifier check'
    for file,sha in before['models'].items():assert digest(ROOT/file)==sha
    assert digest(ROOT/'config/baseline1.json')==before['frozen_config']
    after={p.relative_to(ROOT).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
           for d in ROOT.glob('P03_*') for p in d.rglob('*') if p.is_file()}
    assert after==before['p03_metadata']
    evidence.update(model_artifacts_unchanged=True,frozen_preprocessing_config_unchanged=True,p03_metadata_unchanged=True,
                    gui_desktop_manual_click='NOT_RUN: headless execution and automated controller events only')
    write_json(out/'acceptance.json',evidence)
    write_json(ROOT/'outputs/work_orders_acceptance/latest.json',dict(directory=out.relative_to(ROOT).as_posix()))
    print('ACCEPTANCE',out,flush=True)


if __name__=='__main__':run()
