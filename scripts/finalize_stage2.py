"""Resumable final Stage 2 packaging; never trains or reads Stage 3 images."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/p07_final_integration';DATA=ROOT/'datasets/mechanical_components_v2'
OLD=ROOT/'data/mechanical_components_v2';ARCHIVE=ROOT/'archive/experiments'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvsave(p,values,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for row in values for k in row)));w.writeheader();w.writerows(values)
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def command(args,log=None):
    r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    if log:Path(log).write_text(r.stdout+r.stderr,encoding='utf8')
    if r.returncode:raise RuntimeError(f'{args}: exit {r.returncode}; see {log}')
    return r.stdout
def copy_verified(source,dest):
    source,dest=Path(source),Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(source)==sha(dest),f'Existing different artifact: {dest}'
    else:shutil.copy2(source,dest)
    assert sha(source)==sha(dest)

def frozen_runtime():
    frozen=read(OUT/'release_freeze.json')
    ui=read(OUT/'ui_revision.json') if (OUT/'ui_revision.json').exists() else dict(before={},after={})
    for path,expected in frozen['files'].items():
        actual=sha(ROOT/path)
        if actual!=expected:
            assert path=='src/p07_dashboard.py' and ui['before'][path]==expected and ui['after'][path]==actual,'Frozen inference changed: '+path
    return frozen

def audit():
    if (OUT/'initial_audit.json').exists():return
    OUT.mkdir(parents=True,exist_ok=True);backup=ROOT/'outputs/final_integration/pre_change_sources.zip';backup.parent.mkdir(parents=True,exist_ok=True)
    files=[p for folder in ('src','scripts','config','tests') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [ROOT/n for n in ('app.py','predict.py','p08_run.py','p08_object_observations.py','README.md','.gitignore','.gitattributes')]
    with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,rel(p))
    (OUT/'initial_git_status.txt').write_text(command(['git','status','--short']),encoding='utf8')
    (ROOT/'outputs/final_integration/pre_change.patch').write_text(command(['git','diff','--binary']),encoding='utf8')
    protected={rel(p):sha(p) for folder in ('models','results/dataset_v2_baseline_comparison','results/baseline3_v2_experiment')
        for p in (ROOT/folder).rglob('*') if p.is_file()}
    weights=[dict(path=rel(p),bytes=p.stat().st_size,sha256=sha(p)) for p in (ROOT/'models').rglob('*') if p.suffix in ('.pt','.pth','.joblib','.npz')]
    save(OUT/'initial_audit.json',dict(branch=command(['git','branch','--show-current']).strip(),
        protected=protected,weights=weights,source_backup=rel(backup),source_backup_sha256=sha(backup),
        manual_sha256=sha(ROOT/'Project-Man.pdf'),team_plan_sha256=sha(ROOT/'team_plan_ai.md'),
        requested_docx='NOT_FOUND; available team_plan_ai.md reviewed',
        live_physical_acceptance='NOT_VERIFIED',stage3='No Stage3 image/GT evaluation; exclude from publishable tree'))


def datasets():
    if (OUT/'dataset_migration.json').exists():return
    from PIL import Image
    public=[r for r in rows(OLD/'manifests/dataset_manifest.csv') if r['split'] in ('train','val','stage2_test')]
    # Actual source names are train/validation in some manifests; derive by stage/list.
    ids={n:{Path(line).name for line in (OLD/'splits'/f'{s}.txt').read_text(encoding='utf8').splitlines()} for n,s in [('train','train'),('validation','val'),('stage2_test','stage2_test')]}
    selected=set.union(*ids.values());public=[r for r in rows(OLD/'manifests/dataset_manifest.csv') if r['image_id'] in selected]
    assert {k:len(v) for k,v in ids.items()}=={'train':360,'validation':80,'stage2_test':30}
    assert len(public)==470
    for p in ['ground_truth_images.csv','ground_truth_objects.csv','configs/class_mapping.json','audit/annotation_resolutions.json',
              'audit/human_confirmation_20261126.json','audit/freeze_20261126.json','calibration/ruler_manifest.csv']:
        assert (OLD/p).is_file(),p
    assert all((ROOT/r['source_annotation']).is_file() for r in public)
    migration=[]
    def move(source,target,expected=None):
        source,target=Path(source),Path(target);assert target.resolve().is_relative_to(DATA.resolve())
        if source.exists():
            h=sha(source);assert expected is None or expected==h
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(target)==h;source.unlink()
            else:shutil.move(str(source),str(target))
        else:h=sha(target);assert expected is None or expected==h
        assert sha(target)==h
        migration.append(dict(source=rel(source),destination=rel(target),sha256=h,bytes=target.stat().st_size))
    for r in public:
        split=next(k for k,v in ids.items() if r['image_id'] in v);target=DATA/'public'/split/'images'/r['image_id']
        source=ROOT/r['image_path'];move(source,target,r['sha256'])
        with Image.open(target) as image:assert image.size==(int(r['width']),int(r['height']))
        label=Path(r['image_id']).with_suffix('.txt').name
        source_label=OLD/'yolo/labels'/('train' if split=='train' else 'val' if split=='validation' else 'test')/label
        move(source_label,DATA/'public'/split/'labels'/label)
        original=ROOT/r['source_annotation'];copy_verified(original,DATA/'public'/split/'annotations'/Path(original).name)
    roi_rows=read(ROOT/'results/baseline3_v2_experiment/size_development_v1/rois.json');assert len(roi_rows)==220
    for r in roi_rows:
        split='development' if r['experiment_split']=='development' else 'regression_used'
        move(ROOT/r['image_path'],DATA/'size_special'/split/r['image_id'],r['derived_sha256'])
    ruler=rows(OLD/'calibration/ruler_manifest.csv');assert len(ruler)==10
    for r in ruler:move(ROOT/r['image_path'],DATA/'calibration/ruler_images'/r['image_id'])
    pathmap={r['source']:r['destination'] for r in migration}
    def mapped(value):
        if isinstance(value,dict):return {k:mapped(v) for k,v in value.items()}
        if isinstance(value,list):return [mapped(v) for v in value]
        if isinstance(value,str):return pathmap.get(value,value)
        return value
    gt_images=[r for r in rows(OLD/'ground_truth_images.csv') if r['image_id'] in selected]
    gt_objects=[r for r in rows(OLD/'ground_truth_objects.csv') if r['image_id'] in selected]
    for r in gt_images:
        r['image_path']=pathmap[r['image_path']]
        split=next(k for k,v in ids.items() if r['image_id'] in v)
        r['annotation_path']=rel(DATA/'public'/split/'annotations'/Path(r['annotation_path']).name)
    csvsave(DATA/'annotations/ground_truth_images.csv',gt_images);csvsave(DATA/'annotations/ground_truth_objects.csv',gt_objects)
    for split,names in ids.items():
        subset=[r for r in gt_images if r['image_id'] in names]
        csvsave(DATA/'manifests'/(split+'.csv'),subset)
        (DATA/'manifests'/(split+'.txt')).write_text('\n'.join(r['image_id'] for r in subset)+'\n',encoding='utf8')
        base=DATA/'public'/split
        csvsave(base/'ground_truth_images.csv',subset);csvsave(base/'ground_truth_objects.csv',[r for r in gt_objects if r['image_id'] in names])
        orders=[dict(image_id=r['image_id'],expected_counts=dict.fromkeys('12345',1)) for r in subset]
        save(DATA/'configs'/('expected_counts_'+split+'.json'),orders)
    csvsave(DATA/'development/ground_truth_images.csv',[r for r in gt_images if r['image_id'] not in ids['stage2_test']])
    csvsave(DATA/'development/ground_truth_objects.csv',[r for r in gt_objects if r['image_id'] not in ids['stage2_test']])
    for name,split in [('validation','validation'),('stage2_evaluation','stage2_test')]:
        save(DATA/'configs'/(name+'.json'),dict(schema='inspection-evaluation/v2',dataset=rel(DATA/'public'/split),
            evaluation_stage='validation' if split=='validation' else 'stage2',input_dir=rel(DATA/'public'/split/'images'),
            ground_truth_images=rel(DATA/'public'/split/'ground_truth_images.csv'),ground_truth_objects=rel(DATA/'public'/split/'ground_truth_objects.csv'),
            counts_file=rel(DATA/'configs'/('expected_counts_'+split+'.json')),split_list=rel(DATA/'manifests'/(split+'.txt')),dataset_status=rel(DATA/'dataset_status.json')))
    cfg=read(OLD/'configs/training.json');cfg.update(data_dir=rel(DATA/'development'),prepared_dir='outputs/training_v2/prepared')
    save(DATA/'configs/training.json',cfg)
    save(DATA/'configs/data.yaml.json',dict(path='datasets/mechanical_components_v2',train='public/train/images',val='public/validation/images',test='public/stage2_test/images',
        names={i:n for i,n in enumerate(('non_catalogue','long_screw','thick_screw','thin_screw','hex_nut','flat_washer'))}))
    import yaml
    (DATA/'configs/data.yaml').write_text(yaml.safe_dump(read(DATA/'configs/data.yaml.json'),sort_keys=False),encoding='utf8')
    (DATA/'configs/data.yaml.json').unlink()
    csvsave(DATA/'manifests/source_sha256.csv',migration)
    csvsave(DATA/'manifests/source_mapping.csv',[mapped(r) for r in rows(OLD/'manifests/source_mapping.csv') if r['image_id'] in selected])
    csvsave(DATA/'size_special/manifest.csv',[mapped(r) for r in roi_rows])
    csvsave(DATA/'calibration/calibration_manifest.csv',[mapped(r) for r in ruler])
    copy_verified(ROOT/'results/baseline3_v2_experiment/size_development_v1/calibration.json',DATA/'calibration/calibration.json')
    copy_verified(OLD/'configs/class_mapping.json',DATA/'manifests/class_mapping.json')
    copy_verified(OLD/'audit/annotation_resolutions.json',DATA/'annotations/annotation_resolutions.json')
    (DATA/'annotations/annotation_rules.md').write_text('原始人工JSON为标注权威；原图像素xyxy左上原点，忽略实例不参与计数。ID0为类型或尺寸不符目录。GT由人工对象推导，expected_counts独立固定为各1。D18/F21以JSON为准，整批人工核实见原冻结确认记录。歧义/遮挡无法判定必须REVIEW，不补造框；保留原ignore字段。\n',encoding='utf8')
    state=read(OLD/'dataset_status.json');state.update(human_confirmation='datasets/mechanical_components_v2/manifests/human_confirmation.json',
        freeze_record='datasets/mechanical_components_v2/manifests/original_freeze.json',stage3='EXCLUDED',scope='Frozen Stage2 publishable subset only; reused size regression is not independent evaluation')
    copy_verified(OLD/'audit/human_confirmation_20261126.json',DATA/'manifests/human_confirmation.json')
    copy_verified(OLD/'audit/freeze_20261126.json',DATA/'manifests/original_freeze.json')
    save(DATA/'dataset_status.json',state)
    save(DATA/'manifests/dataset_statistics.json',dict(images={k:len(v) for k,v in ids.items()},
        labels={k:dict(Counter(r['ground_truth_label'] for r in gt_images if r['image_id'] in names)) for k,names in ids.items()},
        size_special=dict(development=154,regression_used=66,physical_specimens=8),rulers=10,stage3='90 images EXCLUDED; sealed locally',
        scene_independence='Existing human confirmation, not inferred from hash uniqueness'))
    # Leftovers contain sealed Stage3 images/GT and historical caches. Never publish them.
    sealed=ROOT/'data/sealed/mechanical_components_v2_legacy'
    assert sealed.resolve().is_relative_to((ROOT/'data/sealed').resolve())
    sealed.parent.mkdir(parents=True,exist_ok=True);assert not sealed.exists();shutil.move(str(OLD),str(sealed))
    save(OUT/'dataset_migration.json',dict(files=migration,public_images=470,dimension_images=220,ruler_images=10,
        sealed_local_remainder=rel(sealed),stage3_published=False,p03='NO_WRITES',image_bytes_unchanged=True))
    (DATA/'README.md').write_text('# Mechanical components v2 — Stage 2\n\n唯一正式数据目录。Train360、Validation80、Stage2 Test30，冻结划分不变。size_special/development154用于开发；regression_used66已被使用，不能作为新独立验证；仅8个实物。calibration/ruler_images含10张标尺，标定未获教师确认，不直接适用于Camo。Stage3的90张及真值不在本目录。\n\n图像/标签使用Git LFS，克隆后执行git lfs pull；实际远程上传和配额尚待push核实。GT位于annotations/及各public分区；预测程序只读取images和configs/expected_counts_*.json。manifests/source_sha256.csv记录迁移前后相同字节SHA，source_mapping.csv记录原始来源。原始数据集只读，未复制原始HEIC。\n',encoding='utf8')


def models():
    if (OUT/'model_promotion.json').exists():return
    release=ROOT/'models/release';versions=read(ROOT/'results/dataset_v2_baseline_comparison/model_versions.json')
    for model in ('baseline1','baseline2'):
        source=ROOT/'models'/(model+'_v2_candidate');target=release/model
        for p in source.iterdir():
            if p.is_file() and p.name!='scene_guard.npz':copy_verified(p,target/p.name)
        for key,v in versions[model+'_new']['files'].items():assert sha(ROOT/key)==v['sha256']
    source=ROOT/'models/baseline3_v2_candidate';target=release/'baseline3'
    for name in ('best.pt','training_record.json'):copy_verified(source/name,target/name)
    detector=read(source/'detector.json');detector['weights']='models/release/baseline3/best.pt';detector['release_status']='FROZEN_STAGE2_DETECTOR'
    save(target/'detector.json',detector)
    v1=ROOT/'results/baseline3_v2_experiment/size_development_v1';v2=ROOT/'results/baseline3_v2_experiment/size_repair_v2'
    for name in ('selection.json','calibration.json'):copy_verified(v1/name,target/name)
    size=read(v2/'candidate.json');size.update(detector_config='models/release/baseline3/detector.json',
        selection_path='models/release/baseline3/selection.json',calibration_path='models/release/baseline3/calibration.json')
    size['foreground']['background_path']='datasets/mechanical_components_v2/public/train/images/G01.jpg'
    save(target/'size.json',size)
    entries={
        'baseline1':dict(version=versions['baseline1_new']['model_version'],output_capability='image_level',bridge=None,
            model_config=dict(baseline1_config='models/release/baseline1/config.json',baseline1_bundle='models/release/baseline1/baseline1.joblib',device='cpu'),
            scene_guard=False,fixed_counts=dict.fromkeys('12345',1),origin='results/dataset_v2_baseline_comparison/selection.json'),
        'baseline2':dict(version=versions['baseline2_new']['model_version'],output_capability='image_level',bridge=None,
            model_config=dict(baseline2_bundle='models/release/baseline2',device='cpu'),scene_guard=False,fixed_counts=dict.fromkeys('12345',1),
            origin='results/dataset_v2_baseline_comparison/selection.json'),
        'baseline3':dict(version='size_repair_v2 / '+detector['weights_sha256'],output_capability='object_level',
            bridge='src.models.baseline3.size_repaired:create_backend',model_config=dict(baseline3_mode='size_repaired_development',
                size_candidate_config='models/release/baseline3/size.json'),scene_guard=False,arbitrary_counts=True,
            origin='results/baseline3_v2_experiment/size_repair_v2/candidate_freeze.json',physical_acceptance='NOT_VERIFIED')}
    for model,entry in entries.items():entry['files']={rel(p):sha(p) for p in (release/model).iterdir() if p.is_file()}
    # Bind the actual accepted geometry/decision code and common reference order.
    shared=['config/inspection.json','config/p06_part_specs_candidate.json','src/models/baseline3/geometry_repair.py',
        'src/models/baseline3/size_repaired.py','src/models/baseline3/size_candidate.py','src/models/baseline3/candidate.py']
    registry=dict(schema='frozen-model-registry/v1',created_utc=datetime.now(timezone.utc).isoformat(),models=entries,
        reserved=dict(final=dict(status='NOT_READY',reason='No independent Stage3 improved model delivered')),
        shared_files={p:sha(ROOT/p) for p in shared},training='NOT_RUN',physical_acceptance='NOT_VERIFIED')
    save(ROOT/'config/model_registry.json',registry)
    save(OUT/'model_promotion.json',dict(registry='config/model_registry.json',registry_sha256=sha(ROOT/'config/model_registry.json'),
        entries=entries,weights_and_parameters_unchanged=True))


def cleanup():
    if (OUT/'cleanup.json').exists():return
    paths=['scripts/develop_caliper_v3.py','src/models/baseline3/size_caliper.py',
        'src/models/baseline3/vision_caliper.py','tests/test_vision_caliper.py']
    target=ARCHIVE/'vision_caliper_v3/source_snapshot.zip';target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for name in paths:z.write(ROOT/name,name)
    archived={}
    with zipfile.ZipFile(target) as z:
        for name in paths:
            value=hashlib.sha256(z.read(name)).hexdigest()
            if (ROOT/name).exists():assert sha(ROOT/name)==value;(ROOT/name).unlink()
            archived[name]=dict(archive=rel(target),member=name,sha256=value)
    for name in ('last.pt','resume_last.pt'):
        source=ROOT/'models/baseline3_v2_candidate'/name;dest=ARCHIVE/'baseline3_training_checkpoints'/name
        if source.exists():copy_verified(source,dest);source.unlink()
        archived[rel(source)]=dict(archive=rel(dest),sha256=sha(dest))
    save(OUT/'cleanup.json',dict(archived=archived,deleted_without_archive=[],
        old_frozen_models_retained_in_place=['models/baseline1','models/baseline2','models/baseline3','models/baseline1_v2_candidate','models/baseline2_v2_candidate','models/baseline3_v2_candidate'],
        reason='Frozen historical profiles still reference those files. Only models/release is in production registry. Identical historical backbones are deduplicated by LFS OID.',
        production_mock='No implementation imported; test-only stubs retained in tests/'))
    (ARCHIVE/'README.md').write_text('# Historical experiments\n\nvision_caliper_v3/source_snapshot.zip保留原目录路径和精确源码字节，原始A/B/C实验仍在results/baseline3_v2_experiment/。解压到独立历史工作副本，按results/p07_final_integration/dataset_migration.json恢复当时数据路径后可复现；不进入正式GUI/CLI。baseline3_training_checkpoints保留未选中的训练checkpoint，严禁替代best.pt。\n\n已发表旧冻结权重为避免破坏历史配置保留在models/baseline1、baseline2、baseline3及*_v2_candidate，只有models/release被正式注册表加载。旧参数/预测/失败证据没有改写。\n',encoding='utf8')


def evaluate():
    """Reuse identical frozen B1/B2 predictions; run full repaired B3 exactly once."""
    from src.evaluator import evaluate as score
    final=ROOT/'results/p08_stage2_final';final.mkdir(parents=True,exist_ok=True)
    freeze=OUT/'release_freeze.json'
    if not freeze.exists():
        files=[ROOT/'config/model_registry.json',ROOT/'config/p07.json',ROOT/'config/inspection.json']
        files += [p for p in (ROOT/'src').rglob('*.py') if '__pycache__' not in p.parts]
        files += [DATA/'configs/stage2_evaluation.json',DATA/'public/stage2_test/ground_truth_images.csv',DATA/'public/stage2_test/ground_truth_objects.csv']
        save(freeze,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),files={rel(p):sha(p) for p in files},
            models=read(ROOT/'config/model_registry.json'),test_use='B1/B2 existing identical raw frozen results reused; B3 size_repair_v2 first complete Stage2 run',
            stage3='EXCLUDED',no_parameter_selection_on_test=True))
    frozen=frozen_runtime()
    oldruns=read(ROOT/'results/dataset_v2_baseline_comparison/evaluation/runs.json')
    lineage={}
    for model in ('baseline1','baseline2','baseline3'):
        folder=final/model;pred=folder/'predictions';metrics=folder/'metrics';folder.mkdir(exist_ok=True)
        if (metrics/'summary.json').exists() and (model!='baseline3' or (metrics/'object_observations.json').exists()):continue
        if model!='baseline3':
            source=ROOT/oldruns[model+'_new'];metadata=read(source/'predictions/run_config.json')
            assert metadata['model_version']==frozen['models']['models'][model]['version']
            # Raw classifier + preprocessing + fixed order are identical. The release disables scene_guard.
            assert not frozen['models']['models'][model]['scene_guard']
            if not pred.exists():shutil.copytree(source/'predictions',pred)
            provenance=dict(status='REUSED_NOT_RERUN',source=rel(source),source_predictions={rel(p):sha(p) for p in (source/'predictions').rglob('*') if p.is_file()},
                original_run_config_preserved=True,current_registry='config/model_registry.json',identity='Same weight SHA/preprocessing/fixed order/raw classifier; no scene_guard')
        else:
            if not (pred/'images.csv').exists():
                args=[sys.executable,str(ROOT/'predict.py'),'--input_dir',str(DATA/'public/stage2_test/images'),
                    '--model',model,'--counts_file',str(DATA/'configs/expected_counts_stage2_test.json'),'--output_dir',str(pred),'--record-timing']
                save(folder/'command.json',dict(command=args,started_utc=datetime.now(timezone.utc).isoformat()))
                command(args,folder/'prediction.log')
            assert len(rows(pred/'images.csv'))==30 and all(r['status']!='ERROR' for r in rows(pred/'images.csv'))
            provenance=dict(status='FIRST_FULL_SIZE_REPAIR_V2_STAGE2_RUN',formal_freeze=rel(freeze),
                historical_test='This same Stage2 set was previously used for raw YOLO and B1/B2 comparison. Not a new untouched dataset.',
                physical_size='Provisional; no teacher/caliper/Camo acceptance',no_post_test_tuning=True)
        save(folder/'provenance.json',provenance);lineage[model]=provenance
        if not (metrics/'summary.json').exists():
            score(DATA/'public/stage2_test/ground_truth_images.csv',DATA/'public/stage2_test/ground_truth_objects.csv',
                DATA/'configs/expected_counts_stage2_test.json',pred,metrics)
        if model=='baseline3' and not (metrics/'object_observations.json').exists():
            args=[sys.executable,str(ROOT/'p08_object_observations.py'),'--dataset',str(DATA/'public/stage2_test'),
                '--predictions-dir',str(pred),'--output',str(metrics/'object_observations.json')]
            save(folder/'object_command.json',args);command(args,folder/'object_scoring.log')
        print('STAGE2',model,read(metrics/'summary.json')['image_level_details']['status_counts'],flush=True)
    save(OUT/'stage2_evaluation.json',dict(directory=rel(final),models={m:read(final/m/'provenance.json') for m in ('baseline1','baseline2','baseline3')},
        source_ground_truth_sha256={rel(p):sha(p) for p in [DATA/'public/stage2_test/ground_truth_images.csv',DATA/'public/stage2_test/ground_truth_objects.csv']},
        stage3='NOT_RUN',model_selection='Fixed before this task; do not select/tune from results'))


def verify():
    if (OUT/'integration_acceptance.json').exists():return
    import cv2,numpy as np,psutil,re
    from unittest.mock import patch
    from src.p07_bridge import load_backend,validate_result
    from src.p07_ui import InferenceWorker
    from src.camera import Frame,FrameProvider
    from src.models.baseline1.features import load_image_bgr
    import app
    cv2.setNumThreads(1);profile=read(ROOT/'config/p07.json');expected=dict.fromkeys('12345',1)
    samples=['A001.jpg','A200.jpg','A004.jpg'];compat=OUT/'compatibility'
    save(compat/'expected_counts.json',[dict(image_id=i,expected_counts=expected) for i in samples])
    checks={};image=load_image_bgr(str(DATA/'public/train/images/A001.jpg'))
    for model in ('baseline1','baseline2','baseline3'):
        dest=compat/model
        if (dest/'checks.json').exists():checks[model]=read(dest/'checks.json');continue
        pred=dest/'predictions';dest.mkdir(parents=True,exist_ok=True)
        if not (pred/'images.csv').exists():
            args=[sys.executable,str(ROOT/'predict.py'),'--input_dir',str(DATA/'public/train/images'),'--model',model,
                '--counts_file',str(compat/'expected_counts.json'),'--output_dir',str(pred)]
            save(dest/'command.json',args);command(args,dest/'batch.log')
        batch={r['image_id']:r for r in rows(pred/'images.csv')};backend=load_backend('real',model,profile);worker=InferenceWorker();cases=[]
        try:
            for i,name in enumerate(samples):
                frame=load_image_bgr(str(DATA/'public/train/images'/name));packet=Frame(i,frame,time.perf_counter(),0,'image')
                worker.submit(backend,model,packet,expected,'pass',i);worker.future.result(timeout=120);r=worker.poll()['result']
                assert r['status']==batch[name]['status'] and r['status']!='ERROR'
                if model=='baseline3':
                    native=backend.native;before=native.detector.infer;calls=[]
                    native.detector.infer=lambda *a,**k:(calls.append(1) or before(*a,**k))
                    try:
                        changed=backend.rejudge(r,dict.fromkeys('12345',0));restored=backend.rejudge(changed,expected)
                    finally:native.detector.infer=before
                    assert not calls and r['objects']==changed['objects']==restored['objects'] and r['counts']==changed['counts']
                    assert r['status']==restored['status']
                    if r['non_catalogue_count']:assert r['status']==changed['status']=='FAIL'
                    diag=read(pred/'diagnostics.json');other=next(v['result'] for v in diag['images'] if v['image_id']==name)
                    for a,b in zip(r['objects'],other['objects']):
                        assert a['class_id']==b['class_id'] and np.allclose([a[k] for k in ('xmin','ymin','xmax','ymax','score')],
                            [b[k] for k in ('xmin','ymin','xmax','ymax','score')],atol=1e-6,rtol=0)
                else:
                    assert r['counts'] is None and not r['objects']
                    unsupported=backend.infer(frame,dict.fromkeys('12345',0));assert unsupported['status']=='UNSUPPORTED_COUNTS'
                    assert unsupported['prediction_score'] is None and unsupported['predicted_label'] is None
                    assert backend.infer(frame,expected)['status']==r['status']
                cases.append(dict(image_id=name,status=r['status'],gui_worker_batch_consistent=True))
        finally:worker.close()
        invalid=[]
        for frame in (None,np.zeros((0,2,3),np.uint8),np.zeros((10,10),np.uint8)):
            try:backend.infer(frame,expected)
            except (ValueError,TypeError):invalid.append('REJECTED')
            else:raise AssertionError('Invalid frame accepted')
        checks[model]=dict(real_loaded=True,metadata=backend.metadata,cases=cases,invalid_frames=invalid,
            no_production_mock=not backend.is_mock)
        save(dest/'checks.json',checks[model])
    # Two independent batch orders on a real Train image, not formal Test.
    varied=compat/'variable_order'
    if not (varied/'result.json').exists():
        for name,order in [('fixed',expected),('zeros',dict.fromkeys('12345',0))]:
            save(varied/(name+'.json'),[dict(image_id='A001.jpg',expected_counts=order)])
            if not (varied/name/'images.csv').exists():command([sys.executable,str(ROOT/'predict.py'),'--input_dir',str(DATA/'public/train/images'),
                '--model','baseline3','--counts_file',str(varied/(name+'.json')),'--output_dir',str(varied/name)],varied/(name+'.log'))
        a=rows(varied/'fixed/objects.csv');b=rows(varied/'zeros/objects.csv');assert a==b
        ca=rows(varied/'fixed/counts.csv');cb=rows(varied/'zeros/counts.csv')
        assert [r['observed_count'] for r in ca]==[r['observed_count'] for r in cb]
        assert rows(varied/'fixed/images.csv')[0]['status']=='PASS' and rows(varied/'zeros/images.csv')[0]['status']=='FAIL'
        save(varied/'result.json',dict(two_independent_batch_runs=True,boxes_scores_classes_and_observed_counts_identical=True,fixed='PASS',zeros='FAIL'))
    # Actual app controller, native models and rendering. Only OS display calls stubbed.
    gui=OUT/'real_gui';gui_record=gui/'acceptance.json'
    if not gui_record.exists():
        callback=None;step=0;deadline=time.perf_counter()+120;loaded=[];detector_calls=[]
        def loader(mode,model,cfg):
            b=load_backend(mode,model,cfg);loaded.append(dict(model=model,version=b.metadata['model_version'],is_mock=b.is_mock))
            if model=='baseline3':
                original=b.native.detector.infer
                b.native.detector.infer=lambda *a,**k:(detector_calls.append(1) or original(*a,**k))
            return b
        def register(title,fn):
            nonlocal callback
            callback=fn
        def wait(delay):
            nonlocal step
            if time.perf_counter()>deadline:raise TimeoutError('Real GUI controller validation timed out')
            sessions=list(gui.glob('session/*/latency.csv'))
            latency=rows(sessions[0]) if sessions else []
            snapshots=list(gui.glob('session/**/images.csv'))
            n=len(snapshots)
            model=('baseline1','baseline2','baseline3')[min(step//2,2)]
            ready=any(r['model']==model and r['status'] in ('PASS','FAIL','REVIEW') for r in latency)
            if step in (0,2,4) and ready:step+=1;return ord('s')
            if step==1 and n>=1:step+=1;return ord('2')
            if step==3 and n>=2:step+=1;return ord('3')
            if step==5 and n>=3:return ord('q')
            time.sleep(.01);return -1
        with patch.object(app,'load_backend',side_effect=loader),patch.object(cv2,'namedWindow'),patch.object(cv2,'resizeWindow'),\
            patch.object(cv2,'setMouseCallback',side_effect=register),patch.object(cv2,'imshow'),patch.object(cv2,'waitKey',side_effect=wait),\
            patch.object(cv2,'getWindowProperty',return_value=1),patch.object(cv2,'destroyAllWindows'):
            assert app.main(['--image',str(DATA/'public/train/images/A001.jpg'),'--output-dir',str(gui/'session')])==0
        assert [r['model'] for r in loaded]==['baseline1','baseline2','baseline3'] and all(not r['is_mock'] for r in loaded)
        assert len(list(gui.glob('session/**/images.csv')))==3
        save(gui_record,dict(real_controller_models_and_rendering=True,physical_monitor='NOT_VERIFIED; OS display calls intercepted',
            loaded=loaded,baseline3_detector_calls=len(detector_calls),snapshots=3,source='Train A001, not live camera'))
    native=load_backend('real','baseline3',profile)
    stress=OUT/'stability.json'
    if not stress.exists():
        import torch
        memory=[];statuses=[]
        for i in range(30):
            r=native.infer(image,expected);statuses.append(r['status'])
            memory.append(dict(frame=i,rss_bytes=psutil.Process().memory_info().rss,gpu_allocated_bytes=torch.cuda.memory_allocated(),
                gpu_reserved_bytes=torch.cuda.memory_reserved()))
        assert set(statuses)=={'PASS'}
        save(stress,dict(scope='30 repeated Train frames; software stability only, not fresh-camera latency or long endurance proof',samples=memory,
            rss_growth_after_warmup_bytes=memory[-1]['rss_bytes']-memory[5]['rss_bytes'],consistent_status=True))
    video=ROOT/'outputs/final_integration/train_video.avi'
    if not video.exists():
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),1.,(3024,4032));assert writer.isOpened();writer.write(image);writer.release()
    class Log:
        def capture(self,*args):return 1
    provider=FrameProvider('video',video,profile,Log()).start();provider.thread.join(timeout=20)
    try:
        packet=provider.latest;assert packet is not None
        result=native.infer(packet.image,expected);assert result['status']!='ERROR' and result['inspection']['measurement_calls']>0
    finally:provider.close()
    mismatch=native.infer(cv2.resize(image,(1280,720)),expected)
    assert mismatch['status']!='PASS' and mismatch['inspection']['measurement_calls']==0
    frozen=frozen_runtime()
    save(OUT/'integration_acceptance.json',dict(models=checks,real_gui=read(gui_record),
        variable_batch=read(varied/'result.json'),video=dict(status=result['status'],actual_size_calls=result['inspection']['measurement_calls'],released=True),
        camo_geometry_guard=dict(status=mismatch['status'],actual_size_calls=0),physical_camera=read(OUT/'camera_probe.json'),
        frozen_runtime_unchanged=True,stage3='NOT_OPENED',training='NOT_RUN',ready=False))
    print('REAL INTEGRATION VERIFIED: 3 models, GUI controller, batch, worker, rejudge, video and stability',flush=True)


def reproduction():
    if (OUT/'clean_evaluator.json').exists() and read(OUT/'clean_evaluator.json').get('all_observation_metrics')=='PASS':return
    interpreter=ROOT/'outputs/final_integration/clean_env/Scripts/python.exe'
    final=ROOT/'results/p08_stage2_final';work=ROOT/'outputs/final_integration/clean_recheck';work.mkdir(exist_ok=True)
    history=[]
    for model in ('baseline1','baseline2','baseline3'):
        args=[str(interpreter),'-m','src.evaluator','--ground-truth-images',str(DATA/'public/stage2_test/ground_truth_images.csv'),
            '--ground-truth-objects',str(DATA/'public/stage2_test/ground_truth_objects.csv'),
            '--expected-counts',str(DATA/'configs/expected_counts_stage2_test.json'),'--predictions-dir',str(final/model/'predictions'),
            '--output-dir',str(work/model)]
        if not (work/model/'summary.json').exists():command(args,work/(model+'.log'))
        assert read(work/model/'summary.json')==read(final/model/'metrics/summary.json')
        history.append(dict(model=model,command=args,status='PASS_METRICS_IDENTICAL'))
    args=[str(interpreter),str(ROOT/'scripts/verify_models.py')]
    command(args,work/'model_hashes.json')
    object_args=[str(interpreter),str(ROOT/'p08_object_observations.py'),'--dataset',str(DATA/'public/stage2_test'),
        '--predictions-dir',str(final/'baseline3/predictions'),'--output',str(work/'object_observations.json')]
    if not (work/'object_observations.json').exists():command(object_args,work/'objects.log')
    assert read(work/'object_observations.json')==read(final/'baseline3/metrics/object_observations.json')
    help_run=subprocess.run([str(interpreter),str(ROOT/'predict.py'),'--help'],cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
    (OUT/'clean_prediction_entrypoint.log').write_text(help_run.stdout+help_run.stderr,encoding='utf8')
    save(OUT/'clean_evaluator.json',dict(clean_venv=True,system_site_packages=False,third_party_installation=False,
        saved_prediction_scoring=history,hash_verification=dict(command=args,status='PASS_ALL_THREE'),
        all_observation_metrics='PASS',object_scoring_command=object_args,
        model_prediction_entrypoint=dict(status='NOT_VERIFIED_MISSING_DEPENDENCIES',exit_code=help_run.returncode),
        formal_model_inference='NOT_RERUN',stage3='NOT_OPENED',note='Pure-Python P08 scoring and SHA checks succeed independently; not a clean ML model installation'))


def clean_models():
    """Verify an actually installed isolated venv using Train-only native CLIs."""
    receipt=OUT/'clean_ml_acceptance.json'
    if receipt.exists() and read(receipt).get('status')=='PASS':return
    assert read(OUT/'clean_ml_install.json')['status']=='INSTALLED_PIP_CHECK_PASS'
    interpreter=ROOT/'outputs/final_integration/clean_env/Scripts/python.exe'
    assert 'include-system-site-packages = false' in interpreter.parents[1].joinpath('pyvenv.cfg').read_text(encoding='utf8')
    frozen_runtime();work=OUT/'clean_ml';work.mkdir(exist_ok=True);history=[]
    def run(args,name):
        log=work/(name+'.log');command(args,log);history.append(dict(command=args,log=rel(log),exit_code=0))
    run([str(interpreter),'-m','pip','check'],'pip_check')
    run([str(interpreter),str(ROOT/'scripts/verify_models.py'),'--load-models'],'model_loads')
    run([str(interpreter),str(ROOT/'predict.py'),'--help'],'predict_help')
    cases=OUT/'compatibility/expected_counts.json'
    checks={}
    for model in ('baseline1','baseline2','baseline3'):
        target=work/model
        args=[str(interpreter),str(ROOT/'predict.py'),'--input_dir',str(DATA/'public/train/images'),
              '--model',model,'--counts_file',str(cases),'--output_dir',str(target)]
        if not target.exists():run(args,model)
        actual=rows(target/'images.csv');reference=rows(OUT/'compatibility'/model/'predictions/images.csv')
        assert len(actual)==len(reference)==3
        for a,b in zip(actual,reference):
            assert a.keys()==b.keys() and a['status']!='ERROR'
            for field in a:
                if field=='prediction_score' and a[field] and b[field]:assert abs(float(a[field])-float(b[field]))<=1e-6
                else:assert a[field]==b[field],(model,field,a[field],b[field])
        if model=='baseline3':
            for name in ('objects.csv','counts.csv'):
                a=rows(target/name);b=rows(OUT/'compatibility'/model/'predictions'/name);assert len(a)==len(b)
                for x,y in zip(a,b):
                    for field in x:
                        if field in ('xmin','ymin','xmax','ymax','score'):assert abs(float(x[field])-float(y[field]))<=1e-6
                        else:assert x[field]==y[field]
            assert all(r['result']['baseline3_mode']=='size_repaired_development' for r in read(target/'diagnostics.json')['images'])
        else:assert not (target/'objects.csv').exists() and not (target/'counts.csv').exists()
        checks[model]=dict(images=[r['image_id'] for r in actual],statuses=[r['status'] for r in actual],matches_original_environment=True)
    import re
    core_tests=0
    for pattern in ('test_model_registry.py','test_work_orders.py'):
        name=Path(pattern).stem
        run([str(interpreter),'-m','unittest','discover','-s','tests','-p',pattern,'-v'],name)
        log=(work/(name+'.log')).read_text(encoding='utf8');match=re.search(r'Ran (\d+) tests',log)
        assert match and '\nOK\n' in log;core_tests+=int(match[1])
    run([str(interpreter),'-m','pip','list','--format=json'],'packages')
    save(receipt,dict(status='PASS',isolated_venv=True,system_site_packages=False,
        installation='results/p07_final_integration/clean_ml_install.json',commands=history,models=checks,
        core_interface_tests=core_tests,full_303_suite='NOT_REPEATED per final user instruction',failures=0,errors=0,scope='Train A001/A200/A004 only; no Test inference or Stage3 access',
        existing_environment='UNCHANGED',physical_acceptance='NOT_VERIFIED'))


def provenance():
    if (OUT/'source_provenance.json').exists():return
    values=[]
    for r in rows(DATA/'manifests/source_mapping.csv'):
        source=ROOT/r['source_image'];target=ROOT/r['image_path'];split=next(s for s in ('train','validation','stage2_test') if (DATA/'public'/s/'images'/r['image_id']).exists())
        annotation=DATA/'public'/split/'annotations'/Path(r['source_annotation']).name
        original_annotation=ROOT/r['source_annotation'];assert sha(annotation)==sha(original_annotation)
        values.append(dict(image_id=r['image_id'],kind=split,source=rel(source),source_sha256=sha(source),derived=rel(target),
            derived_sha256=sha(target),annotation=rel(annotation),annotation_source_sha256=sha(original_annotation)))
    for r in rows(DATA/'size_special/manifest.csv'):
        source=ROOT/r['source_path'];target=ROOT/r['image_path'];actual=sha(source);assert actual==r['source_sha256']
        values.append(dict(image_id=r['image_id'],kind='size_'+r['experiment_split'],source=rel(source),source_sha256=actual,
            derived=rel(target),derived_sha256=sha(target)))
    for r in rows(DATA/'calibration/calibration_manifest.csv'):
        source=ROOT/r['source_path'];target=ROOT/r['image_path']
        values.append(dict(image_id=r['image_id'],kind='ruler',source=rel(source),source_sha256=sha(source),derived=rel(target),derived_sha256=sha(target)))
    assert len(values)==700
    csvsave(DATA/'manifests/provenance.csv',values)
    save(OUT/'source_provenance.json',dict(source_images_sha256_verified=700,original_annotations_sha256_verified=470,
        mapping='datasets/mechanical_components_v2/manifests/provenance.csv',source_reads_only=True,stage3_images_excluded=True,
        dimension_and_ruler_conversion='HEIC source and PNG derived have separate SHA; no claim of identical file encoding'))


def handoff():
    import re,numpy as np
    from scripts.benchmark_live import environment
    from src.model_registry import portable
    final=ROOT/'results/p08_stage2_final';reg=read(ROOT/'config/model_registry.json')
    env=environment();save(OUT/'environment.json',portable(env));save(final/'run_configs/environment.json',portable(env))
    # Keep the complete pre-change source tree in Git-accessible history.
    copy_verified(ROOT/'outputs/final_integration/pre_change_sources.zip',ARCHIVE/'pre_final_integration/source_snapshot.zip')
    initial=read(OUT/'initial_audit.json');moved=read(OUT/'cleanup.json')['archived']
    for p,h in initial['protected'].items():
        if (ROOT/p).exists():assert sha(ROOT/p)==h,p
        else:
            entry=moved[p]
            assert entry['sha256']==h and sha(ROOT/entry['archive'])==h,p
    frozen=frozen_runtime()
    # Generated snapshot source PNGs are identical decoded Train pixels. Keep the
    # source once in datasets, plus genuine rendered evidence and a hash mapping.
    import cv2
    sources=read(OUT/'snapshot_source_mapping.json') if (OUT/'snapshot_source_mapping.json').exists() else []
    original=DATA/'public/train/images/A001.jpg'
    for p in OUT.rglob('source.png'):
        assert np.array_equal(cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_COLOR),cv2.imdecode(np.fromfile(original,np.uint8),cv2.IMREAD_COLOR))
        sources.append(dict(generated_snapshot=rel(p),snapshot_sha256=sha(p),canonical_source=rel(original),
            canonical_sha256=sha(original),decoded_pixels_identical=True));p.unlink()
    if sources:save(OUT/'snapshot_source_mapping.json',sources)
    comparison=[];cases=[];errors=final/'error_cases';errors.mkdir(exist_ok=True)
    support={}
    for split in ('train','validation','stage2_test'):
        images=rows(DATA/'public'/split/'ground_truth_images.csv');objects=rows(DATA/'public'/split/'ground_truth_objects.csv')
        support[split]=dict(images=len(images),scene_ids=len({r['scene_id'] for r in images}),
            conservative_scene_groups=len({r['scene_group_id'] for r in images}),
            labels=dict(Counter(r['ground_truth_label'] for r in images)),
            non_catalogue_scenes=sum(int(r['non_catalogue_count'])>0 for r in images),
            non_ignored_instances=dict(Counter(r['class_id'] for r in objects if r['ignore']=='0')))
    save(final/'comparison/dataset_support.json',support)
    for model in ('baseline1','baseline2','baseline3'):
        metrics=read(final/model/'metrics/summary.json');m=metrics['image_level_details'];cm=m['confusion_matrix']
        comparison.append(dict(model=model,n=m['input_cases'],tp=cm['tp'],fp=cm['fp'],fn=cm['fn'],tn=cm['tn'],
            accuracy_all=m['all_input_correct_fraction'],accuracy_completed=m['completed_accuracy'],
            fail_precision=m['fail']['precision'],fail_recall=m['fail']['recall'],fail_f1=m['fail']['f1'],
            support_pass=m['passed']['support_all'],support_fail=m['fail']['support_all'],review=m['status_counts'].get('REVIEW',0),
            false_pass=m['false_pass']['count'],false_fail=m['false_fail']['count'],coverage=m['eligible_coverage'],
            live_median_ms=None,live_p95_ms=None,live_status='NOT_RUN: user confirms rig/calibration incomplete; probe returned black frames'))
        for row in rows(final/model/'metrics/per_image.csv'):
            event=('False PASS' if row['ground_truth_label']=='FAIL' and row['predicted_label']=='PASS' else
                'False FAIL' if row['ground_truth_label']=='PASS' and row['predicted_label']=='FAIL' else
                'REVIEW' if row['prediction_status']=='REVIEW' else None)
            if event:cases.append(dict(model=model,event=event,image_id=row['image_id'],truth=row['ground_truth_label'],prediction=row['prediction_status']))
        # Keep historical config bytes intact; the current portable release profile
        # is separate and explicitly does not claim a new inference execution.
        save(final/'run_configs'/(model+'.json'),dict(registry='config/model_registry.json',entry=reg['models'][model],
            prediction_provenance=read(final/model/'provenance.json'),original_run_config_sha256=sha(final/model/'predictions/run_config.json'),
            input_order='datasets/mechanical_components_v2/configs/expected_counts_stage2_test.json',
            image_manifest='datasets/mechanical_components_v2/public/stage2_test/ground_truth_images.csv',
            coordinate_convention='Original image pixels, top-left xyxy',score_convention='Raw uncalibrated classifier probability; B3 rule label has null image score'))
    csvsave(final/'comparison/image_metrics.csv',comparison);save(final/'comparison/image_metrics.json',comparison)
    csvsave(errors/'image_errors.csv',cases)
    obj=read(final/'baseline3/metrics/object_observations.json');details=read(final/'baseline3/metrics/object_details.json')
    csvsave(final/'comparison/object_count_metrics.csv',[dict(class_id=c,**v) for c,v in obj['classes'].items()])
    csvsave(errors/'object_errors.csv',details['errors']);csvsave(errors/'count_differences.csv',details['count_differences'])
    # Save accurate UTF8 test logs; PowerShell redirection may have used UTF16.
    def logtext(path):
        b=Path(path).read_bytes();return b.decode('utf16') if b.startswith((b'\xff\xfe',b'\xfe\xff')) else b.decode('utf8',errors='replace')
    for name in ('software_tests_initial.log','software_tests_final.log'):
        (OUT/name).write_text(logtext(ROOT/'outputs/final_integration'/name),encoding='utf8')
    log=(OUT/'software_tests_final.log').read_text(encoding='utf8');match=re.search(r'Ran (\d+) tests',log)
    assert match and '\nOK\n' in log
    save(OUT/'software_tests.json',dict(tests=int(match[1]),failures=0,errors=0,
        source='Full current unittest discovery after production integration; archived Caliper tests are historical only',
        initial_run='308 tests, one assertion expected old final error wording; updated for NOT_READY, no algorithm changes'))
    checker=ROOT/'outputs/final_integration/dataset_check.json';save(OUT/'dataset_validation.json',json.loads(logtext(checker)))
    clean=ROOT/'outputs/final_integration/clean_install.log'
    (OUT/'clean_online_resolution.log').write_text(logtext(ROOT/'outputs/final_integration/clean_online_resolution.log'),encoding='utf8')
    save(OUT/'clean_environment.json',dict(venv_created=True,system_site_packages=False,offline_install='FAILED_NO_LOCAL_WHEELS',
        offline_log=logtext(clean),online_resolution=read(OUT/'clean_resolution_status.json'),
        actual_existing_environment='pip check PASS; real three-model entrypoints and software tests PASS',
        clean_install_and_prediction='NOT_VERIFIED',no_claim_of_clean_install_success=True))
    if (OUT/'clean_evaluator.json').exists():
        d=read(OUT/'clean_environment.json');d['clean_stdlib_verification']=read(OUT/'clean_evaluator.json');save(OUT/'clean_environment.json',d)
    clean_ml=read(OUT/'clean_ml_acceptance.json') if (OUT/'clean_ml_acceptance.json').exists() else {}
    clean_pass=clean_ml.get('status')=='PASS'
    if (OUT/'clean_ml_install.json').exists():
        d=read(OUT/'clean_environment.json');d['actual_ml_install']=read(OUT/'clean_ml_install.json')
        d['actual_ml_acceptance']=clean_ml;d['clean_install_and_prediction']='PASS' if clean_pass else 'NOT_VERIFIED';d['no_claim_of_clean_install_success']=not clean_pass
        save(OUT/'clean_environment.json',d)
    # Compare code with the task-start snapshot, not just Git HEAD: the workspace
    # already contained members' uncommitted work before this integration.
    with zipfile.ZipFile(ROOT/'outputs/final_integration/pre_change_sources.zip') as snapshot:
        before={name:hashlib.sha256(snapshot.read(name)).hexdigest() for name in snapshot.namelist()}
    current={rel(p):sha(p) for folder in ('src','scripts','config','tests') for p in (ROOT/folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    current.update({name:sha(ROOT/name) for name in ('app.py','predict.py','p08_run.py','p08_object_observations.py','README.md','.gitignore','.gitattributes')})
    changes=[dict(path=p,action='ADDED' if p not in before else 'ARCHIVED' if p not in current else 'MODIFIED',
                  before_sha256=before.get(p),after_sha256=current.get(p))
             for p in sorted(before.keys()|current.keys()) if before.get(p)!=current.get(p)]
    save(OUT/'change_inventory.json',dict(code_compared_with='archive/experiments/pre_final_integration/source_snapshot.zip',code=changes,
        formal_models='model_promotion.json',derived_data='dataset_migration.json',history=['cleanup.json','historical_runner_archive.json'],
        new_output_roots=['results/p07_final_integration','results/p08_stage2_final'],
        documentation=['README.md','docs/p07/final_handoff.md','docs/p07/demo_recording_checklist.md','docs/p07/advice_record.md',
                       'docs/live_benchmark.md','docs/ui_operations.md','docs/dataset_v2.md','docs/baseline3_integration.md',
                       'docs/baseline3_size_integration.md','results/README.md','review/decision_stage2.md'],
        note='Git status may also include pre-existing members work; dataset movement and historical artifacts are listed separately'))
    # Git inclusion, LFS attributes and exact submission inventory. No staging.
    candidate_names=command(['git','-c','core.quotepath=false','ls-files','-z','--cached','--others','--exclude-standard']).split('\0')
    candidates=[ROOT/n for n in candidate_names if n and (ROOT/n).is_file()]
    inventory=[dict(path=rel(p),bytes=p.stat().st_size) for p in candidates]
    for p in candidates:assert not p.is_relative_to(ROOT/'data') and not p.is_relative_to(ROOT/'数据集'),rel(p)
    image_files=[p for p in DATA.rglob('*') if p.suffix.lower() in ('.jpg','.png')];assert len(image_files)==700
    assert subprocess.run(['git','check-ignore','-q',rel(image_files[0])],cwd=ROOT).returncode==0
    assert subprocess.run(['git','check-ignore','-q',rel(DATA/'manifests/source_sha256.csv')],cwd=ROOT).returncode==1
    assert subprocess.run(['git','check-ignore','-q','models/release/baseline3/best.pt'],cwd=ROOT).returncode==1
    assert subprocess.run(['git','check-ignore','-q','data/sealed/mechanical_components_v2_legacy'],cwd=ROOT).returncode==0
    attrs=command(['git','check-attr','filter','--','models/release/baseline3/best.pt'])
    assert attrs.count('lfs')==1
    ordinary_large=[r for r in inventory if r['bytes']>=100*1024**2 and Path(r['path']).suffix.lower() not in ('.jpg','.png','.pt','.pth')]
    assert not ordinary_large,ordinary_large
    csvsave(OUT/'git_submission_files.csv',inventory)
    dataset_bytes=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())
    save(OUT/'git_delivery.json',dict(lfs_installed=command(['git','lfs','version']).strip(),attributes=attrs,
        candidate_files=len(inventory),candidate_bytes=sum(r['bytes'] for r in inventory),curated_image_files=700,dataset_bytes=dataset_bytes,
        dataset_images_in_git=False,dataset_image_delivery='Separate cloud drive; restore datasets/mechanical_components_v2 paths',
        largest_files=sorted(inventory,key=lambda r:r['bytes'],reverse=True)[:12],raw_and_sealed_tracked=False,
        git_add='NOT_RUN',commit_push_merge='NOT_RUN',remote_lfs_quota='NOT_VERIFIED',remote_clone='NOT_VERIFIED until user commits and pushes Git and LFS objects'))
    (OUT/'git_status_ignored.txt').write_text(command(['git','status','--short','--ignored']),encoding='utf8')
    doc=ROOT/'docs/p07/final_handoff.md';doc.parent.mkdir(parents=True,exist_ok=True)
    weight_rows=[]
    for model,e in reg['models'].items():
        for p,h in e['files'].items():
            if Path(p).suffix in ('.joblib','.pth','.pt') or (model=='baseline2' and Path(p).name=='baseline2.json'):
                weight_rows.append(f'|{model}|`{p}`|{(ROOT/p).stat().st_size:,}|`{h}`|')
    table=['|Model|全输入Accuracy|FAIL P/R/F1|TP/FP/FN/TN|False PASS/FAIL|REVIEW|Coverage|','|---|---|---|---|---|---|---|']
    for r in comparison:table.append(f"|{r['model']}|{r['accuracy_all']:.2%}|{r['fail_precision']:.4f}/{r['fail_recall']:.4f}/{r['fail_f1']:.4f}|{r['tp']}/{r['fp']}/{r['fn']}/{r['tn']}|{r['false_pass']}/{r['false_fail']}|{r['review']}|{r['coverage']:.2%}|")
    object_table=['|类别ID|实例支持量|TP/FP/FN|Precision|Recall|F1|Count MAE|','|---|---|---|---|---|---|---|']
    for c,v in obj['classes'].items():
        object_table.append(f"|{c}|{v['support']}|{v['tp']}/{v['fp']}/{v['fn']}|{v['precision']:.4f}|{v['recall']:.4f}|{v['f1']:.4f}|{v['count_mae']:.4f}|")
    clean_note=('干净venv实际安装CUDA依赖、三模型Train预测及必要核心接口测试通过，见clean_ml_acceptance.json；未重复全部303项测试。'
        if clean_pass else '干净ML安装与预测尚未通过验收，见clean_ml_install.json及clean_environment.json。')
    text=['# P07正式系统与P08交接','', '**最终状态：NOT_READY。软件集成与离线共同评价完成；实机延迟未运行，物理尺寸验收和远程Git交付仍未验证。**','',
        '依据已阅读的Project-Man.pdf（第2/3/5/7/10页相关要求）及现有team_plan_ai.md；指定Mechanical_Component_Vision_Inspection_Team_Plan.docx未在工作区找到，未将其视作已阅读材料。',
        '## 正式模型与冻结','',
        '唯一正式注册表：config/model_registry.json；app.py、predict.py、p08_run.py统一调用src.p07_bridge.load_backend。正式权重仅位于models/release/三个目录；GUI只有三个Baseline按钮和1–3快捷键。final仅保留API预留ID，明确NOT_READY，不能别名Baseline3。加载时校验SHA；错误不回退。',
        'Baseline1/2采用已冻结的新Train360训练模型。生产GUI与CLI均使用原始分类器，scene_guard关闭；不会把旧262参考特征门控套到新模型上。两者只支持[1,1,1,1,1]，无对象框/计数/ID0能力，其他合法向量返回UNSUPPORTED_COUNTS与空预测字段。严重OOD仍可能高置信误判，不能据分数宣称场景有效。',
        'Baseline3是YOLO11n epoch69 + **size_repair_v2**，conf0.35、NMS0.50、640、batch1、CUDA0、half=false；完整前景关联、几何测量、尺寸证据、计数和工单判定保持原算法。不是原YOLO-only、v1、Vision Caliper B或Hybrid C。',
        '|模型|实际权重|Bytes|SHA256|','|---|---|---|---|',*weight_rows,'',
        'release_freeze.json在本次完整Baseline3 Test推理之前生成，所有运行源码与配置SHA见results/p07_final_integration/。calibration.json和selection.json逐字节保留v2来源，路径变更记录在model_promotion.json。配置中的旧训练路径是冻结的来源记录；真正bundle路径由唯一注册表显式覆盖。',
        'Test后仅增加UI显示的类别ID，并移除app的无效Mock场景操作参数；未改变模型推理、预处理、后处理或业务判定。旧Test运行源码已精确归档，ui_revision.json及final_release_freeze.json分别记录显示变更和当前入口；没有为此重复Test。','',
        '## 文件布局','', '```text','app.py / predict.py / p08_run.py / p08_object_observations.py',
        'config/model_registry.json / p07.json / inspection.json','models/release/{baseline1,baseline2,baseline3}/',
        'src/models/baseline{1,2,3}/ / src/adapters/ / src/p07_*.py / camera.py',
        'datasets/mechanical_components_v2/{public,size_special,calibration,annotations,manifests,configs,development}/',
        'results/p08_stage2_final/{baseline1,baseline2,baseline3,comparison,error_cases,run_configs}/',
        'results/p07_final_integration/','archive/experiments/{pre_final_integration,vision_caliper_v3,development_runners,baseline3_training_checkpoints}/',
        'data/sealed/  [忽略，仅本机，不交P08]','PROGRESS.md / task_state.json','```','',
        '完整可交付文件树见results/p07_final_integration/project_tree.txt；本次代码改动对照任务开始快照，见change_inventory.json，迁移与归档文件分别见其引用清单。Git原有未提交工作不被冒充为本次新增贡献。',
        '正式数据为Train360（180/180 PASS/FAIL）、Validation80（40/40）、Stage2 Test30（15/15）。原冻结划分不变，人工场景确认与保守分组元数据保留；不把哈希唯一当作场景证明。700张交付图片=470公共+154专项开发+66已使用回归+10标尺；专项220张仅8实物。Stage3的90张及其真值均未进入交付目录。源图/原始JSON未改写，迁移SHA及路径映射见dataset_migration.json和datasets/mechanical_components_v2/manifests/source_sha256.csv。',
        'Ground Truth位于annotations/及public各分区，推理只读图像与expected_counts文件，不读取GT。ID0预期恒0，difference=observed−expected。原始JSON、YOLO标签和GT CSV均随所选分区交付。来源P03/数据集只读。','',
        '## P07验收','',
        '|项目|实际结果|证据|','|---|---|---|',
        '|三真实模型加载、静态图片、GUI worker/CLI一致|PASS|compatibility/*/checks.json|',
        '|实际app控制器三模型切换、渲染、快照|PASS（原生窗口调用拦截，非现场显示验收）|real_gui/acceptance.json|',
        '|原生桌面窗口人工交互|NOT_VERIFIED；桌面工具未返回目标窗口，已清理自启进程|desktop_gui_attempt.json|',
        '|Baseline3工单重判不再运行YOLO|PASS，框/类别/分数/观测数不变|integration_acceptance.json|',
        '|同图两工单独立batch输出|PASS，固定PASS、零向量FAIL|compatibility/variable_order/|',
        '|ID0强制FAIL、尺寸核验、REVIEW|软件/实际离线执行PASS；物理尺寸NOT_VERIFIED|P08预测及单元测试|',
        '|视频输入、设备释放、非法/空图|PASS|integration_acceptance.json|',
        '|连续处理/内存|三模型各30次重复Train图；B1另240次，末100次RSS增24KiB；非长期无泄漏证明|stability_all_models.json、baseline1_memory_followup.json|',
        '|真实模型离线Record与MP4解码|PASS，仅Train软件验证，不是演示/备份视频|recording_acceptance.json|',
        '|真实Camo采集|仅3个新读帧ID，全黑，设备已释放；真实工位NOT_VERIFIED|camera_probe.json|',
        '|断连重连/加载错误/CUDA错误|软件故障注入通过；实体断插未验收|software_tests_final.log|',
        f"|当前完整软件测试|{match[1]}项通过，0失败0错误|software_tests.json|",
        f"|干净环境安装与预测|{'PASS' if clean_pass else 'NOT_VERIFIED'}|clean_ml_acceptance.json / clean_environment.json|",'',
        '安装第三方依赖前，先在标准库干净venv中复核了三个已保存预测的P08评分、全部观察框/计数评分及三模型SHA，均与原结果一致。独立评分不需要GPU，也未重新运行Test图像推理。'+clean_note,
        f"{match[1]}项当前测试不包含已归档的10项卡尺实验测试；初次308测试的单项失败是旧final错误提示断言，已按NOT_READY更新，原日志保留。所有生产路径无Mock；测试专用Stub仅用于故障/控制器单元用例，不代表实机或模型性能。",'',
        '## 共同Stage2评价','',*table,'',
        'FAIL为正类。三个模型均为15 PASS、15 FAIL的相同30图、相同工单。Baseline1/2复用原来完全相同权重/预处理/raw分类器的预测，不重复推理；Baseline3为冻结后第一次完整尺寸v2 Stage2运行。Stage2此前已用于原YOLO及新旧分类器对照，不能称为全新未触碰测试。没有根据本次结果再选型、改阈值或改容差。',
        'Baseline3完成判定子集28/28不等于全输入100%；主表为28/30=93.33%，两个REVIEW保留。objects/counts主表使用P08 object_observations包含所有30图的观察统计；原summary.json的旧可判定子集口径仍保留并明确区分。',
        f"五类计数向量全对率={obj['five_class_vector_exact']:.2%}；ID0 TP/FP/FN={obj['classes']['0']['tp']}/{obj['classes']['0']['fp']}/{obj['classes']['0']['fn']}。每类Precision/Recall/F1/support/MAE见comparison/object_count_metrics.csv。",
        *object_table,'',
        f"Stage2非目录支持：{support['stage2_test']['non_catalogue_scenes']}个场景、{support['stage2_test']['non_ignored_instances'].get('0',0)}个实例。Train/Val/Test图像、scene_id、保守scene_group及各类实例分别保存在comparison/dataset_support.json。",
        '错误案例：'+ '; '.join(f"{r['model']} {r['event']} {r['image_id']}" for r in cases)+'.',
        '目标级错误见object_errors.csv：A198螺母框IoU约0.496未达0.50，记FP+FN；B04一个合法ID2被拒为ID0。图像级没有False FAIL不能掩盖这个正常件误拒。ID0漏检为0/9，但不能据这9个已标注实例宣称所有未见件均可拒识。details只是已有P08匹配循环的追加诊断，不是另一套Evaluator，也没有重新运行模型。','',
        '## 实机性能与资源','',
        '|系统|真实新工位帧|Live Median ms|Live P95 ms|状态|','|---|---|---|---|---|',
        '|Baseline1|0|N/A|N/A|NOT_RUN|','|Baseline2|0|N/A|N/A|NOT_RUN|','|Baseline3 v2|0|N/A|N/A|NOT_RUN|',
        '用户本次明确确认尚未完成实机布置或标定，原答复保存在physical_confirmation.json。Camera0虚拟设备能打开不代表手机已送出真实工位；这次3帧全黑不计入实机性能，之后未再采集。旧离线Train20时间不是摄像头延迟。Camo当前请求1280×720；原照片标定针对3024×4032，35.758446 px/mm不能外推。分辨率不符时尺寸模块拒绝测量并给出非PASS，未伪造尺度、MAE、公差或validated=true。',
        f"本机CPU：{env['cpu']}；GPU：{env['gpu']}；Python：{env['python'].splitlines()[0]}；CUDA：{env['cuda_runtime']}；依赖：{env['dependencies']}。Baseline1/2实际CPU，YOLO CUDA0；尺寸/前景OpenCV CPU。真实持续Camera Capture FPS与处理FPS待测，不能用1/单帧延迟冒充。",
        'benchmark_live.py默认显示，主Live边界是FrameProvider.available_at到GPU同步后P07 render、imshow、waitKey返回，包含排队/预处理/推理/后处理/尺寸/判定/显示调用；不含帧可用前曝光、Camo传输或显示器真实刷新。纯processing时间另列。分段未独立可靠计时则null，不虚构；RSS与CUDA内存可测时记录。observed_processing_fps和observed_sample_fps来自实际时间跨度；theoretical_serial_capacity_fps仅保留为理论串行容量。','',
        '## 历史保护与Git交付','',
        '未选用卡尺源码/测试和历史开发runner已无损归档ZIP，成员名与SHA见cleanup.json及historical_runner_archive.json。last.pt/resume_last.pt移到历史checkpoint目录。旧冻结模型因历史配置引用而保留原位置，仅作历史复现；LFS按OID去重相同ResNet骨干。既有results、权重与训练参数SHA保持不变，原始预测未覆盖。',
        f"本地正式数据总大小{dataset_bytes/2**30:.3f} GiB。700张图片已从Git排除，通过网盘单独共享；标注、GT、划分、类别映射、SHA、配置和README仍提交。模型.pt/.pth继续使用LFS。Git候选清单见git_submission_files.csv；远程模型LFS配额、上传和新克隆未验证，本次未git add/commit/push/merge。",
        'P08克隆源码后，先从网盘取得图片，并按datasets/mechanical_components_v2/README.md合并到同名根目录，保留文件名、目录和原始分辨率，无需修改代码。网盘链接尚未提供，不宣称已完成远程交付。原始data/、数据集/、P03、Stage3封存、环境及临时输出仍忽略。',
        '提交前建议：git lfs install；git add .gitattributes .gitignore README.md .python-version config src scripts tests datasets models archive docs results review PROGRESS.md task_state.json app.py predict.py p08_run.py p08_object_observations.py requirements*.txt；git diff --cached --stat；git lfs ls-files --size；确认数据集图片、原始数据和虚拟环境未进入暂存区。之后由你决定commit/push。','',
        '## P08克隆与运行','', '以下在你完成提交/push后执行。'+clean_note+' 不擅自降级已训练scikit-learn版本。',
        '```powershell','git lfs install','git clone -b p07/system-integration https://github.com/FelixWong111/AI_Project.git',
        'cd AI_Project','git lfs pull','# 从网盘下载图片，合并至 datasets/mechanical_components_v2/，保留目录及原文件名。','py -3.11 -m venv .venv311','.\.venv311\Scripts\Activate.ps1',
        'python -m pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126',
        'python -m pip install -r requirements.txt','python -m pip check','python scripts/verify_models.py --load-models',
        'python scripts/check_dataset_v2.py','python app.py --model baseline3 --camera 0 --capture-backend dshow',
        '# 默认P08三模型评价，会新运行；已有本次结果直接看results/p08_stage2_final，不要无必要重复正式Test：',
        'python p08_run.py --evaluation-config datasets/mechanical_components_v2/configs/stage2_evaluation.json --model all --output outputs/p08_reproduction',
        '# 单模型批处理（更换model ID即可）：',
        'python predict.py --input_dir datasets/mechanical_components_v2/public/stage2_test/images --model baseline3 --output_dir outputs/batch_unique --counts_file datasets/mechanical_components_v2/configs/expected_counts_stage2_test.json',
        '# 以下只在Camo已输出真实工位且标定就绪后执行：',
        'python scripts/benchmark_live.py --camera 0 --warmup 20 --frames 100 --models baseline1 baseline2 baseline3',
        'python -m unittest discover -s tests -v','```','',
        '只复核已有结果、不运行模型且不需要PyTorch时，可用：`python -m src.evaluator --ground-truth-images datasets/mechanical_components_v2/public/stage2_test/ground_truth_images.csv --ground-truth-objects datasets/mechanical_components_v2/public/stage2_test/ground_truth_objects.csv --expected-counts datasets/mechanical_components_v2/configs/expected_counts_stage2_test.json --predictions-dir results/p08_stage2_final/baseline3/predictions --output-dir outputs/p08_score_only_unique`；改predictions-dir即可复核其他两模型。',
        '## 未完成与人工步骤','',
        '1. 在Camo Studio确认手机连接/解锁、Camera0有真实画面；固定纯白哑光工位、灯位、相机镜头/变焦/工作距离和倾角，提供真实工位全景与背景材质记录。',
        '2. 按size_repair_v2/manual_acceptance.md在最终Camo几何下重新标定；多位置/旋转/高度采样，独立卡尺测五类及8件专项样件，核对M5实物身份和头型长度定义。现有照片尺度不覆盖Camo。',
        '3. 教师确认类目录、尺寸容差、支持工况与误差策略，保留实际记录；不能现在把validated改true。必要的标定支持属于人工资料到齐后的受控新配置，不在此次制造参数。',
        '当前v2工厂明确限定开发配置（validated=false、instructor_approved=false），不会因改一个标志自动成为已认证物理系统。资料齐后仍需P06/P07受控接入新的真实Camo配置与审核策略、保留冻结照片参考并重新验收；不要直接编辑冻结JSON或更新SHA冒充已经通过。',
        '4. 完成真实相机断插/重连、三模型切换、ID0、尺寸和U01–U06功能验收，再跑20+100新帧实机计时。没有完成不能READY。',
        '5. 按demo_recording_checklist.md拍摄<2分钟真实视频及备份；当前截图/Train编码视频不冒充实机演示。rig照片、白底材质证据与真实演示/备份仍未核实。',
        '6. 原始review/decision_stage2.md未找到；新缺项说明不能替代实施前研究决策。请P05提供当时原件；不倒填日期或编造计划。',
        '7. 核实GitHub LFS配额并提交上传，再让P08实际克隆核对。'+('干净环境验证已经完成，仍需验证远程克隆。' if clean_pass else '干净ML安装及预测尚需完成。'),
        '支持恢复：根PROGRESS.md/task_state.json；所有可执行离线阶段有幂等守卫，evaluate复用已有结果不重复Test。训练不在本任务范围，没有训练或修改权重/阈值/几何算法。']
    doc.write_text('\n'.join(text)+'\n',encoding='utf8')
    (final/'README.md').write_text('# P08 Stage 2 frozen results\n\n'+ '\n'.join(table)+'\n\nBaseline3的28/28 completed accuracy不是全输入100%；两个REVIEW保留，主Accuracy28/30。B1/B2复用同SHA原始预测，B3第一次完整size_repair_v2 Test；历史Stage2已用，非全新独立证据。Stage3未运行。\n\n每模型predictions含原始CSV与run_config，metrics由现有P08 evaluator生成；comparison/object_count_metrics.csv包含全部30图（含REVIEW预测）的目标/计数指标，error_cases保留逐图/逐对象错误。run_configs关联正式registry、原推理配置SHA与来源，未伪造B1/B2新运行。完整操作及限制见../../docs/p07/final_handoff.md。\n',encoding='utf8')
    save(OUT/'final_status.json',dict(ready=False,status='OFFLINE_COMPLETE_PHYSICAL_AND_GIT_PENDING' if clean_pass else 'OFFLINE_COMPLETE_PHYSICAL_AND_REPRODUCTION_PENDING',
        completed=['Dataset migrated/hash checked','Three genuine frozen models registered','Real offline GUI/batch/worker/video/rejudge','P08 Stage2 results',match[1]+' software tests']+(['Clean CUDA installation, Train predictions and core interface tests'] if clean_pass else []),
        not_run=['Real rig latency','Physical camera disconnect/reconnect acceptance','Real demo/backup recording'],
        not_verified=['Camo scale/caliper/instructor acceptance','Native desktop interaction','Remote LFS quota/upload/clone','Dataset cloud-drive delivery','Original pre-improvement Stage2 decision record']+([] if clean_pass else ['Clean installation and prediction']),
        p03='NO_WRITES',stage3='NOT_RUN_NOT_PUBLISHED',training='NOT_RUN',commit_push_merge='NOT_RUN'))


def main():
    sys.stdout.reconfigure(encoding='utf8',errors='replace')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['audit','datasets','models','cleanup','evaluate','verify','reproduction','clean_models','provenance','handoff'],required=True);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    state_path=ROOT/'task_state.json';state=read(state_path) if state_path.exists() else dict(task='P07 final Stage2 integration',phases={})
    record=dict(status='RUNNING',command=[sys.executable,*sys.argv],started_utc=datetime.now(timezone.utc).isoformat())
    state['phases'][args.phase]=record;save(state_path,state)
    try:globals()[args.phase]();record.update(status='COMPLETED',finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:record.update(status='ERROR',error=repr(error));raise
    finally:
        state=read(state_path);state['phases'][args.phase]=record;state['ready']=False
        clean_pass=(OUT/'clean_ml_acceptance.json').exists() and read(OUT/'clean_ml_acceptance.json').get('status')=='PASS'
        state['next']='Physical Camo/caliper/instructor/live/demo and remote Git/LFS remain unverified'+('' if clean_pass else '; full clean ML installation still unverified')
        state['resume_command']='python scripts/finalize_stage2.py --phase handoff'
        state['next_safe_check']='python scripts/verify_models.py --load-models'
        state['next_live_command']='python scripts/benchmark_live.py --camera 0 --warmup 20 --frames 100 --models baseline1 baseline2 baseline3 (only after physical setup/calibration)'
        save(state_path,state)
        history_path=OUT/'execution_history.json';history=read(history_path) if history_path.exists() else []
        history.append(dict(phase=args.phase,**record));save(history_path,history)
        (ROOT/'PROGRESS.md').write_text('# P07最终系统集成\n\n'+ '\n'.join(f"- {k}: {v['status']}" for k,v in state['phases'].items())+
            '\n\n整体NOT_READY：软件与离线评价完成；物理标定/教师确认、实机延迟、真实视频及远程LFS交付未验收。'+('干净ML安装、Train预测及核心接口测试已通过，未重复全部303项。' if clean_pass else '完整干净ML安装及预测未验收。')+'\n'+
            '\n安全检查：`'+state['next_safe_check']+'`。刷新交接：`'+state['resume_command']+'`；不会重训或重复正式Test。\n'+
            '\n资料：docs/p07/final_handoff.md、results/p08_stage2_final/、results/p07_final_integration/。\n'+
            '\n来源/权重/参数未改；未commit/push/merge；Stage3图像和真值未进入公开包或模型评价。详细状态、命令、SHA与缺项见task_state.json和验证记录。\n',encoding='utf8')
    print(args.phase,'COMPLETED',flush=True)
if __name__=='__main__':main()
