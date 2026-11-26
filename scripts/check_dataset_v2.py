"""Validate prepared data and real model/data loaders, without fitting or evaluation."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.dataset_v2 import (digest,save_json,verify,ROOT,prepare_development,
                                 assign_splits,relative,CLASSES)


def check(out):
    if (out/'public').is_dir():
        from collections import Counter
        from scripts.finalize_stage2 import rows,sha,DATA,ROOT as workspace
        from p08_run import evaluation_inputs
        mapping=rows(out/'manifests/source_sha256.csv')
        assert all(sha(workspace/r['destination'])==r['sha256'] for r in mapping)
        checks={}
        for split,n in [('train',360),('validation',80),('stage2_test',30)]:
            images=rows(out/'public'/split/'ground_truth_images.csv');objects=rows(out/'public'/split/'ground_truth_objects.csv')
            assert len(images)==n
            for row in images:
                counts=Counter(int(o['class_id']) for o in objects if o['image_id']==row['image_id'] and o['ignore']=='0')
                assert [counts[c] for c in range(6)]==[int(row['non_catalogue_count'])]+[int(row[name+'_count']) for name in CLASSES[1:]]
                assert 'stage3' not in row['image_path']
            checks[split]=dict(images=n,annotations=len(objects),status='PASS')
        evaluation_inputs(out/'configs/stage2_evaluation.json',check_only=True)
        return dict(curated_stage2=checks,migrated_files_sha256_verified=len(mapping),stage3='EXCLUDED',
            formal_training='NOT_RUN',inference='NOT_RUN',physical_size='NOT_VERIFIED')
    config_dir=ROOT/'tmp/ultralytics'
    (config_dir/'Ultralytics').mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(config_dir)
    os.environ.setdefault('YOLO_OFFLINE','true')
    os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib')
    import shutil
    if Path('C:/Windows/Fonts/arial.ttf').exists():
        shutil.copyfile('C:/Windows/Fonts/arial.ttf',config_dir/'Ultralytics/Arial.ttf')
    original=json.loads((out/'audit/source_files.json').read_text(encoding='utf-8'))
    print('Checking raw source hashes, annotations and lossless specialized images',flush=True)
    checks=verify(out,original)
    settings=json.loads((out/'configs/training.json').read_text(encoding='utf-8'))
    settings['allow_candidate_annotations']=True
    records,evidence=prepare_development(ROOT/settings['data_dir'],settings)
    print(f'Read development-only images: {len(records)}',flush=True)
    if any(r['split'] not in ('train','validation') for r in records):raise ValueError('Held-out image entered development loader')
    expected=json.loads((out/'manifests/dataset_statistics.json').read_text(encoding='utf-8'))['public']
    checks['baseline1_and_2_shared_data_loader']=len(records)==expected['train']['images']+expected['validation']['images']
    from src.models.baseline1.features import load_image_bgr,extract_hog
    cfg=json.loads((ROOT/'config/baseline1.json').read_text(encoding='utf-8'))
    first=next(r for r in records if r['split']=='train')
    vector=extract_hog(load_image_bgr(str(ROOT/first['path'])),tuple(cfg['image_size']),cfg['hog'])
    checks['baseline1_real_hog_input']=vector.shape==(1764,)
    from src.p07_bridge import load_backend
    config=json.loads((ROOT/'config/p07.json').read_text(encoding='utf-8'))
    models={}
    for name in ('baseline1','baseline2','baseline3'):
        print(f'Loading real {name} (no prediction)',flush=True)
        backend=load_backend('real',name,config)
        if backend.is_mock:raise ValueError('Unexpected mock model')
        models[name]=dict(loaded=True,is_mock=False,model_version=backend.metadata.get('model_version'),inference='NOT_RUN')
        if name=='baseline2':
            from src.adapters.baseline2 import read_rgb
            extracted=backend.native.adapter.features.extract([read_rgb(ROOT/first['path'])])
            checks['baseline2_real_rgb_feature_input']=tuple(extracted.shape)==(1,512)
        if name=='baseline3':checks['baseline3_current_mode']=backend.metadata.get('baseline3_mode')=='size_verified'
        del backend
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data.dataset import YOLODataset
    data=check_det_dataset(str(out/'yolo/data.yaml'),autodownload=False)
    print('Checking actual YOLO Train/Val data loader',flush=True)
    checks['yolo_mapping_matches_frozen_model']=tuple(data['names'][i] for i in range(6))==CLASSES
    yolo={}
    for split in ('train','val'):
        ds=YOLODataset(img_path=data[split],data=data,imgsz=640,augment=False,cache=False,batch_size=1)
        item=ds[0]
        yolo[split]=dict(images=len(ds),nonignored_objects=sum(len(r['cls']) for r in ds.labels),sample_tensor_shape=list(item['img'].shape))
        if len(ds)!=expected['train' if split=='train' else 'validation']['images']:raise ValueError('YOLO selected wrong split')
    checks['baseline3_yolo_real_data_loader']=True
    from p08_run import evaluation_inputs
    p08={stage:evaluation_inputs(out/'configs'/f'{stage}_evaluation.json',check_only=True)[1] for stage in ('stage2','stage3')}
    checks['p08_stage_specific_ground_truth_readable']=p08['stage2']['images']==30 and p08['stage3']['images']==90
    all_records=json.loads((out/'manifests/records.json').read_text(encoding='utf-8'))['records']
    rerun=copy.deepcopy(all_records);assign_splits(rerun)
    checks['seed42_split_reproduced']=all((a['split'],a['evaluation_stage'])==(b['split'],b['evaluation_stage']) for a,b in zip(all_records,rerun))
    if subprocess.run(['git','diff','--quiet','HEAD','--','models','config','results/baseline1','results/baseline2','results/baseline3','results/integration'],cwd=ROOT).returncode:
        raise ValueError('Protected tracked model/config/historical evidence changed')
    checks['frozen_models_configs_and_historical_results_unchanged']=True
    checks['all_public_annotations_present']=all((ROOT/r['source_annotation']).is_file() for r in all_records)
    checks['public_annotations_no_unresolved_count_conflict']=not any(r['annotation_issues'] for r in all_records)
    report=dict(checks=checks,automatic_acceptance_passed=all(checks.values()),development=evidence['summary'],
                real_models=models,yolo=yolo,p08=p08,formal_training='NOT_RUN',formal_test='NOT_RUN',
                label_visual_quality='Full human sign-off PENDING; COMPLETE means format-complete, not VERIFIED',
                protected_weights={relative(p):digest(p) for p in (ROOT/'models/baseline1/baseline1.joblib',ROOT/'models/baseline2/baseline2.json',ROOT/'models/baseline2/resnet18-f37072fd.pth',ROOT/'models/baseline3/best.pt')})
    save_json(out/'audit/interface_acceptance.json',report)
    if not all(checks.values()):raise ValueError(f'Automatic checks failed: {checks}')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'datasets/mechanical_components_v2')
    args=parser.parse_args();report=check(args.dataset.resolve())
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
