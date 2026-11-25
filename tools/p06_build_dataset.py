"""P06 deterministic development dataset build; no model training or test inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings

from PIL import Image, ImageOps

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.p06_labelme import CLASSES, DataError, from_yolo, require, to_yolo
from tools.p06_preview_annotations import digest, draw_preview, load_preview

VAL_IMAGES=('pass01.jpg','pass07.jpg','pass13.jpg','pass19.jpg','pass25.jpg','pass32.jpg',
            'fail01.jpg','fail03.jpg','fail09.jpg','fail13.jpg')
CONFIRMED_EMPTY={'fail13.jpg'}
EXPECTED_COUNTS=[3,48,48,48,49,48]


def pixel_hash(image):
    return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()


def summarize(records):
    result={}
    for split in ('train','val'):
        subset=[r for r in records if r['split']==split]
        result[split]=dict(images=len(subset),classes=[sum(r['counts'][i] for r in subset) for i in range(6)],
                           image_statuses=dict(Counter(r['image_status'] for r in subset)),
                           empty=[r['image_id'] for r in subset if not r['objects']])
    return result


def create_plan(source,val_images=VAL_IMAGES,confirmed_empty=CONFIRMED_EMPTY,*,version=1):
    require(version in (1,2),'INVALID_VERSION','Only explicit dataset versions 1/2 are supported.')
    source=Path(source).resolve()
    require(source.is_dir(),'MISSING_INPUT','Source directory is missing.')
    entries=sorted(source.iterdir())
    require(all(p.is_file() and not p.is_symlink() and p.resolve().parent==source for p in entries),
            'UNSAFE_INPUT','Only a flat directory of local files is supported.')
    images=[p for p in entries if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
    require(images,'NO_IMAGES','No supported images found.')
    names={p.name for p in images}
    require(set(val_images)<=names,'INVALID_SPLIT','Validation list includes unknown images.')
    require(set(confirmed_empty)<=names,'INVALID_EMPTY','Empty confirmations include unknown images.')
    require({p.stem for p in images}=={p.stem for p in entries if p.suffix.lower()=='.json'},
            'MISSING_PAIR','Images and Labelme JSON must pair exactly by stem.')
    require(len({p.stem.casefold() for p in images})==len(images),'AMBIGUOUS_STEM','Image stems must be unique.')
    hashes={p.name:digest(p) for p in entries}
    records=[]
    for path in images:
        image,row,objects=load_preview(path.with_suffix('.json'),confirmed_empty)
        require(row['filename']==path.name,'IMAGE_PAIR_MISMATCH','JSON references a different paired image.')
        name=path.name
        reason='Train default; retain all ID 0 examples' if name not in val_images else 'Fixed validation selection'
        if name.startswith('pass') and name in val_images: reason='Fixed spread across acquisition order; varied rearrangements/poses'
        if name=='fail01.jpg': reason='Validation: annotated long_screw missing'
        if name=='fail03.jpg': reason='Validation: annotated thin_screw missing; original EXIF=1 resolution'
        if name=='fail09.jpg': reason='Validation: annotated extra hex_nut'
        if name=='fail13.jpg': reason='Validation: user-confirmed empty background'
        records.append(dict(image_id=name,scene_id='P06_TEMP_development_'+path.stem,
                            scene_authority='P06_provisional_pending_P03_P08',
                            scene_basis='User confirmed a new physical arrangement for each image',
                            split='val' if name in val_images else 'train',reason=reason,
                            image_status=row['image_level_status'],png_name=path.stem+'.png',
                            width=image.width,height=image.height,original_width=row['original_width'],
                            original_height=row['original_height'],exif_orientation=row['exif_orientation'],
                            counts=[row[f'count_{i}'] for i in range(6)],objects=objects,
                            display_pixel_sha256=pixel_hash(image)))
        image.close()
    require(hashes=={p.name:digest(p) for p in entries},'SOURCE_CHANGED','Source changed during planning.')
    plan=dict(source=str(source),source_sha256=hashes,confirmed_empty=sorted(confirmed_empty),
              method='Fixed explicit image list; no random sampling',records=records,summary=summarize(records))
    _validate_structure(plan,allow_id0_val=version==2)
    return plan


def validate_plan(plan):
    _validate_structure(plan,allow_id0_val=False)


def _validate_structure(plan,*,allow_id0_val=False):
    records=plan['records']; scenes={}; ids=set(); hashes={}
    require(records,'EMPTY_PLAN','Plan has no images.')
    for row in records:
        require(row['split'] in {'train','val'},'INVALID_SPLIT','Only train/val are permitted; no test partition.')
        require(row['image_id'] not in ids,'DUPLICATE_IMAGE','An image appears more than once.')
        ids.add(row['image_id'])
        scene=row['scene_id']; require(bool(scene),'MISSING_SCENE_ID','Provisional scene_id is required.')
        require(scene not in scenes or scenes[scene]==row['split'],'SCENE_LEAKAGE','Scene crosses splits.')
        scenes[scene]=row['split']
        if not allow_id0_val:
            require(not(row['split']=='val' and row['counts'][0]),'ID0_IN_VAL','ID 0 must remain exclusively in train.')
        h=row['display_pixel_sha256']
        require(h not in hashes or hashes[h]==row['split'],'IMAGE_LEAKAGE','Duplicate decoded image crosses splits.')
        hashes[h]=row['split']
    require(summarize(records)==plan['summary'],'SUMMARY_MISMATCH','Plan summary does not match records.')
    require(plan['summary']['train']['images'] and plan['summary']['val']['images'],'EMPTY_SPLIT','Both splits must be nonempty.')


def validate_project(plan):
    validate_plan(plan)
    summary=plan['summary']; rows=plan['records']
    require(summary['train']['images']==41 and summary['val']['images']==10,'PROJECT_SPLIT','Expected 41 train / 10 val.')
    require([sum(r['counts'][i] for r in rows) for i in range(6)]==EXPECTED_COUNTS,
            'PROJECT_COUNTS','Original class totals differ from confirmed 244-instance development set.')
    require({r['image_id'] for r in rows if r['split']=='val'}==set(VAL_IMAGES),'PROJECT_VAL','Validation list must equal the fixed candidate.')
    require(summary['val']['image_statuses']=={'FAIL':4,'PASS':6},'PROJECT_BALANCE','Expected 6 PASS and 4 FAIL in val.')
    require(all(summary['val']['classes'][i]>0 for i in range(1,6)),'VAL_COVERAGE','Validation must cover target IDs 1-5.')
    require(summary['val']['empty']==['fail13.jpg'],'EMPTY_PLACEMENT','Confirmed empty fail13 must be in validation.')
    for name in ('fail11.jpg','fail12.jpg','fail14.jpg'):
        require(any(r['image_id']==name and r['split']=='train' and r['counts'][0]==1 for r in rows),
                'ID0_PLACEMENT',f'{name} must contain one ID 0 in train.')


def write_manifest(plan,path):
    fields=['image_id','scene_id','split','scene_authority','scene_basis','reason','image_status','png_name']+[f'count_{i}' for i in range(6)]
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in plan['records']:
            writer.writerow({**{k:row[k] for k in fields if not k.startswith('count_')},
                             **{f'count_{i}':row['counts'][i] for i in range(6)}})


def check_sources(plan):
    root=Path(plan['source'])
    require({p.name for p in root.iterdir()}==set(plan['source_sha256']),'SOURCE_CHANGED','Source file inventory changed.')
    for name,expected in plan['source_sha256'].items():
        path=root/name
        require(path.is_file() and not path.is_symlink() and path.resolve().parent==root,
                'UNSAFE_INPUT','Source reference must stay inside the development directory.')
        require(digest(path)==expected,'SOURCE_CHANGED',f'Source hash changed: {name}')


def validate_artifacts(plan,root):
    root=Path(root); totals=[0]*6; split_counts={s:[0]*6 for s in ('train','val')}; maximum=0.0
    for split in ('train','val'):
        expected={r['png_name'] for r in plan['records'] if r['split']==split}
        require({p.name for p in (root/'images'/split).glob('*.png')}==expected,'IMAGE_FILES','Output image set differs from plan.')
        require({p.stem for p in (root/'labels'/split).glob('*.txt')}=={Path(n).stem for n in expected},'LABEL_FILES','Image/TXT pairing mismatch.')
    for row in plan['records']:
        path=root/'images'/row['split']/row['png_name']
        with Image.open(path) as image:
            image.load()
            require(image.format=='PNG' and image.size==(row['width'],row['height']) and image.getexif().get(274,1)==1,
                    'PNG_GEOMETRY',f'Wrong PNG orientation/dimensions: {path.name}')
            require(pixel_hash(image.convert('RGB'))==row['display_pixel_sha256'],'PNG_PIXELS','Corrected PNG pixels differ from source display.')
        lines=(root/'labels'/row['split']/(path.stem+'.txt')).read_text(encoding='utf-8').splitlines()
        require(len(lines)==len(row['objects']),'INSTANCE_COUNT',f'Object count differs for {path.name}')
        for line,obj in zip(lines,row['objects']):
            parts=line.split(); require(len(parts)==5,'INVALID_TXT','Each TXT row must contain class + four coordinates.')
            require(parts[0]==str(obj['class_id']),'CLASS_MAPPING','TXT class differs from Labelme.')
            values=[float(v) for v in parts[1:]]
            require(all(math.isfinite(v) and 0<=v<=1 for v in values),'INVALID_TXT','Nonfinite or out-of-range YOLO coordinates.')
            require(values[2]>0 and values[3]>0,'INVALID_TXT','YOLO width/height must be positive.')
            box=from_yolo(values,row['width'],row['height'])
            error=max(abs(a-b) for a,b in zip(box,obj['xyxy'])); maximum=max(maximum,error)
            require(error<=1e-6,'ROUNDTRIP_ERROR',f'Coordinate roundtrip exceeds 1e-6 pixel: {path.name}')
            totals[obj['class_id']]+=1; split_counts[row['split']][obj['class_id']]+=1
    require(split_counts=={s:plan['summary'][s]['classes'] for s in ('train','val')},'CLASS_COUNTS','Class counts differ after export.')
    return dict(images=len(plan['records']),instances=sum(totals),classes=totals,split_classes=split_counts,
                max_roundtrip_error_pixels=maximum,source_hashes_unchanged=True)


def validate_ultralytics(root,plan):
    root=Path(root).resolve()
    settings=Path(__file__).resolve().parents[1]/'output/p06-cache/dataset-check-settings'
    settings.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(settings)
    os.environ['YOLO_OFFLINE']='true'
    mpl=settings/'matplotlib'; mpl.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(mpl)
    config=settings/'Ultralytics'; config.mkdir(exist_ok=True)
    if Path('C:/Windows/Fonts/arial.ttf').exists() and not (config/'Arial.ttf').exists():
        shutil.copyfile('C:/Windows/Fonts/arial.ttf',config/'Arial.ttf')
    import numpy as np
    import ultralytics
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import LOGGER
    stream=io.StringIO(); handler=logging.StreamHandler(stream); LOGGER.addHandler(handler)
    result=dict(version=ultralytics.__version__,training_started=False,splits={})
    try:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream),warnings.catch_warnings(record=True) as caught,contextlib.chdir(root):
            data=check_det_dataset(str(root/'dataset.yaml'),autodownload=False)
            require(data['nc']==6 and list(data['names'].values())==list(CLASSES),'ULTRALYTICS_CLASSES','Ultralytics class catalogue differs.')
            require(not data.get('test'),'TEST_REFERENCE','Training YAML must not reference a test set.')
            for split in ('train','val'):
                dataset=YOLODataset(img_path=data[split],data=data,task='detect',imgsz=640,augment=False,
                                    cache=False,rect=False,batch_size=1,prefix=f'P06 {split}: ')
                require(len(dataset)==plan['summary'][split]['images'],'LOADER_IMAGES','Ultralytics dropped input images.')
                counts=[0]*6; empty=0
                for label in dataset.labels:
                    classes=label['cls'].reshape(-1)
                    empty+=int(len(classes)==0)
                    for cls in classes: counts[int(cls)]+=1
                require(counts==plan['summary'][split]['classes'],'LOADER_CLASSES','Ultralytics class totals differ.')
                for index in range(len(dataset)):
                    sample=dataset[index]
                    require(sample['img'].ndim==3,'LOADER_TENSOR','Image tensor is malformed.')
                    require(len(sample['cls'])==len(dataset.labels[index]['cls']),'LOADER_TARGETS','Loader changed instance count.')
                cache=root/'labels'/f'{split}.cache'
                raw=np.load(cache,allow_pickle=True).item()
                results=[int(v) for v in raw['results']]
                require(results[1]==0 and results[3]==0,'LOADER_INVALID','Loader reported missing labels or corrupt inputs.')
                result['splits'][split]=dict(images=len(dataset),classes=counts,empty=empty,
                    cache_file=f'labels/{split}.cache',cache_results=results,cache_messages=raw['msgs'],
                    iterated_samples=len(dataset),augment=False,image_cache=False)
            result['python_warnings']=[str(w.message) for w in caught]
    finally:
        LOGGER.removeHandler(handler)
        (root/'validation/ultralytics.log').write_text(stream.getvalue(),encoding='utf-8')
    # Cache embeds temporary absolute paths. Do not publish stale label caches.
    for split in ('train','val'):
        (root/'labels'/f'{split}.cache').unlink()
    result['cache_policy']='Temporary label caches verified, then removed before publish; regenerate on actual training.'
    result['status']='PASS'
    return result


def roundtrip_previews(plan,root):
    root=Path(root); qa=root/'validation/roundtrip_previews'; qa.mkdir()
    samples=['pass01.jpg','fail01.jpg','fail09.jpg','fail11.jpg','fail13.jpg','fail02.jpg']
    if plan.get('version')==2:
        candidates=[r for r in plan['records'] if r.get('provenance')=='new275' and r['split']=='val']
        for condition in (lambda r:r['counts'][0]>0 and sum(r['counts'][1:])==0,
                          lambda r:r['counts'][0]>1 and sum(r['counts'][1:])>0,
                          lambda r:not r['objects'],lambda r:r['image_status']=='PASS'):
            chosen=next((r['image_id'] for r in candidates if condition(r)),None)
            if chosen and chosen not in samples: samples.append(chosen)
        samples.append('pass32.jpg')
    comparisons=[]
    previous=Path(__file__).resolve().parents[1]/'output/p06/annotation_preview/individual'
    for row in plan['records']:
        if row['image_id'] not in samples: continue
        with Image.open(root/'images'/row['split']/row['png_name']) as image:
            image=image.convert('RGB')
        objects=[]
        for index,line in enumerate((root/'labels'/row['split']/(Path(row['png_name']).stem+'.txt')).read_text().splitlines()):
            cid,*coords=line.split()
            objects.append(dict(object_id=index,class_id=int(cid),xyxy=from_yolo([float(v) for v in coords],image.width,image.height)))
        recovered=draw_preview(image,objects)
        recovered.save(qa/row['png_name'],compress_level=3)
        # Synthetic unit tests must never open real-data previews.
        real_source=Path(__file__).resolve().parents[1]/'data/p03/development'
        old=previous/row['png_name'] if Path(plan['source'])==real_source else None
        if old is not None and old.exists():
            with Image.open(old) as im: reference=im.convert('RGB')
            source='Previous Labelme preview'
        else:
            reference=draw_preview(image,row['objects']); source='Labelme coordinates from current source'
        a=ImageOps.contain(reference,(650,780)); b=ImageOps.contain(recovered,(650,780))
        from PIL import ImageDraw
        pair=Image.new('RGB',(1320,835),'white'); draw=ImageDraw.Draw(pair)
        draw.text((10,8),row['image_id']+' | LEFT Labelme / RIGHT TXT roundtrip | NOT predictions',fill='black')
        pair.paste(a,((650-a.width)//2,45)); pair.paste(b,(670+(650-b.width)//2,45))
        pair.save(qa/(Path(row['png_name']).stem+'-comparison.jpg'),quality=93)
        comparisons.append(dict(image_id=row['image_id'],reference=source))
    return comparisons


def build_dataset(plan,output,official_check=True,*,version=1):
    require(version in (1,2),'INVALID_VERSION','Unknown dataset version.')
    if version==2:
        from tools.p06_build_dataset_v2 import validate_v2_plan
        validate_v2_plan(plan)
    else:
        validate_plan(plan)
    check_sources(plan)
    out=Path(output).resolve(); source=Path(plan['source'])
    require(out!=source and source not in out.parents and out not in source.parents,'UNSAFE_OUTPUT','Input/output must not contain each other.')
    if out.exists(): raise FileExistsError(f'Refusing existing dataset: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.p06-dataset-',dir=out.parent) as temp:
        if os.name=='nt':
            # Python 3.11 Windows mkdtemp protects its ACL. Restore this workspace
            # parent's normal inherited access before creating/publishing images.
            subprocess.run(['icacls',temp,'/inheritance:e'],check=True,capture_output=True)
        stage=Path(temp)/'complete'; stage.mkdir()
        for split in ('train','val'):
            (stage/'images'/split).mkdir(parents=True); (stage/'labels'/split).mkdir(parents=True)
        (stage/'validation').mkdir()
        for index,row in enumerate(plan['records'],1):
            display,current,objects=load_preview(source/(Path(row['image_id']).stem+'.json'),plan['confirmed_empty'])
            require(objects==row['objects'] and pixel_hash(display)==row['display_pixel_sha256'],
                    'SOURCE_CHANGED','Planning/build image or annotation differs.')
            display.info.clear()
            display.save(stage/'images'/row['split']/row['png_name'],format='PNG',compress_level=3)
            lines=[str(o['class_id'])+' '+' '.join(format(v,'.12g') for v in to_yolo(o['xyxy'],display.width,display.height)) for o in objects]
            (stage/'labels'/row['split']/(Path(row['png_name']).stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
            display.close()
            if index%10==0: print(f'Built {index}/{len(plan["records"])} PNG/TXT pairs',flush=True)
        # Ultralytics 8.3.236 resolves existing relative path against cwd. Run from dataset root.
        yaml='path: .\ntrain: images/train\nval: images/val\nnc: 6\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES))
        (stage/'dataset.yaml').write_text(yaml,encoding='utf-8')
        write_manifest(plan,stage/'split_manifest.csv')
        (stage/'source_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        validation=validate_artifacts(plan,stage)
        validation['roundtrip_previews']=roundtrip_previews(plan,stage)
        validation['ultralytics']=validate_ultralytics(stage,plan) if official_check else {'status':'NOT_RUN','reason':'Synthetic unit test only'}
        check_sources(plan)
        if version==2:
            validate_v2_plan(plan)
        (stage/'validation/checks.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
        summary=plan['summary']
        report=['# P06 YOLO Train/Validation 数据集','',
                '派生图片与标签，非模型预测。本轮没有训练、调参或读取独立测试集。',
                '固定显式划分名单；无随机重采样。scene_id 为 P06_TEMP 暂定，依据用户确认每图独立重新摆放，待 P03/P08 审核，非正式 scene_id。',
                'EXIF 只通过既有 load_preview / ImageOps.exif_transpose 校正图像；JSON 框不旋转、不缩放；保存完整分辨率普通 PNG。原严格转换检查器不变。','',
                f'Train/Val：{summary["train"]["images"]}/{summary["val"]["images"]}；对象 {validation["instances"]}；类别总数 {validation["classes"]}。',
                f'Train：{summary["train"]}。',f'Validation：{summary["val"]}。',
                f'反算最大误差：{validation["max_roundtrip_error_pixels"]:.12g} 像素，验收限 1e-6 像素；源框先严格检查越界，未通过裁剪/重定位修正标注。',
                '所有输出 PNG 的 RGB 像素哈希与 EXIF 校正后源图一致，尺寸保留，EXIF Orientation 不再携带方向歧义。',
                f'{len(plan["source_sha256"])} 个源文件 SHA-256 按 source_plan.json 核对不变。',
                '详细名单/暂定 scene_id/理由见 split_manifest.csv；逐项检查见 validation/checks.json，官方加载日志见 validation/ultralytics.log。','',
                '## 可移植配置与加载','',
                '`dataset.yaml` 使用 path: . 和相对 images/train、images/val。Ultralytics 8.3.236 要求在数据集根目录运行，或调用前切换 cwd 至该目录；复制整个目录后无需写死机器绝对路径。无 test/download 键。',
                '官方数据检查使用 check_det_dataset(autodownload=False) + YOLODataset(augment=False,cache=False)，仅遍历加载样本，无模型或优化器；默认 640 的加载张量缩放不会修改原始分辨率 PNG。',
                f'官方加载状态：{validation["ultralytics"]["status"]}。标签缓存含临时绝对路径，验证后移除，正式训练时再生成。','',
                '## 已知限制','',
                '**Validation 没有 ID 0，不能用此集计算 ID 0 召回率、选择拒识阈值或声称完成 ID 0 拒识验证。**',
                'ID 0 仅三个实例且全部在 Train，非目录外类型的充分覆盖。数据仅 51 个暂定独立场景、单一背景/照明/拍摄高度；不能据此证明跨工况泛化。',
                'Validation 空背景仅 fail13 一张；图片级 PASS/FAIL 来自交接文件名，未提供参考 expected_counts，不作为模型输入。',
                '用户确认人工抽查未见明显问题，不等于全部规格与标注均已独立验收。scene_id 映射待 P03/P08 审核。',
                '技术上可准备启动训练，但需人工审核当前派生数据和证据。本轮未训练。','']
        if version==2:
            from tools.p06_build_dataset_v2 import report_v2
            report=report_v2(plan,validation)
        (stage/'dataset_report.md').write_text('\n'.join(report),encoding='utf-8')
        if out.exists(): raise FileExistsError(f'Output appeared during build: {out}')
        stage.rename(out)
    return validation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','build'])
    parser.add_argument('--input',type=Path,default=Path('data/p03/development'))
    parser.add_argument('--plan',type=Path,help='Previously generated candidate source_plan.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.output.exists(): raise FileExistsError(f'Existing output refused: {args.output}')
        if args.command=='plan':
            plan=create_plan(args.input); validate_project(plan)
            args.output.mkdir(parents=True,exist_ok=False)
            write_manifest(plan,args.output/'split_manifest.csv')
            (args.output/'source_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
            (args.output/'candidate_report.json').write_text(json.dumps(dict(method=plan['method'],summary=plan['summary'],
                constraints='PASS',validation_images=list(VAL_IMAGES),scene_authority='P06 provisional; pending P03/P08'),indent=2),encoding='utf-8')
            print(json.dumps(plan['summary'],indent=2)); return 0
        require(args.plan is not None,'MISSING_PLAN','Build requires a checked candidate --plan file.')
        plan=json.loads(args.plan.read_text(encoding='utf-8')); validate_project(plan)
        result=build_dataset(plan,args.output)
        print(json.dumps(result,indent=2)); return 0
    except (DataError,OSError,ValueError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr); return 1


if __name__=='__main__': raise SystemExit(main())
