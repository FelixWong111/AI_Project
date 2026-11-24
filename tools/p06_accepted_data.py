"""Frozen accepted P03 Train/Validation input; no Test access or repartitioning."""
import csv
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image
from src.p06_data import CLASSES,_box,_load_labelme_document,_parse_shapes,from_yolo
from tools.p06_preview_annotations import digest
from tools.p06_build_dataset import summarize,validate_ultralytics


def require(value,message):
    if not value: raise ValueError(message)


def safe_files(root):
    root=Path(root).resolve(); files=[]
    for directory,children,names in os.walk(root,followlinks=False):
        for p in [Path(directory),*(Path(directory)/n for n in children+names)]:
            require(not p.is_symlink() and not (getattr(p.lstat(),'st_file_attributes',0)&0x400)
                    and p.resolve().is_relative_to(root),'Redirected dataset path forbidden')
        files.extend(Path(directory)/n for n in names)
    return sorted(files)


def check_sources(plan):
    root=Path(plan['source']); files=safe_files(root)
    actual={p.relative_to(root).as_posix():digest(p) for p in files}
    require(actual==plan['source_sha256'],'P03 source changed since acceptance; audit again')


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def match(reference,candidate):
    require(len(reference)==len(candidate),'Object counts differ across formats')
    remaining=set(range(len(candidate)))
    for obj in reference:
        possible=[i for i in remaining if candidate[i]['class_id']==obj['class_id']]
        require(possible,'Class mapping differs across formats')
        chosen=min(possible,key=lambda i:max(abs(a-b) for a,b in zip(obj['xyxy'],candidate[i]['xyxy'])))
        require(max(abs(a-b) for a,b in zip(obj['xyxy'],candidate[chosen]['xyxy']))<=1e-6,'Box disagreement exceeds 1e-6 pixel')
        remaining.remove(chosen)


def parse_txt(text,width,height):
    objects=[]
    for line in text.splitlines():
        parts=line.split();require(len(parts)==5,'YOLO requires exactly five fields')
        cid=int(parts[0]);require(parts[0]==str(cid) and 0<=cid<6,'Unknown YOLO class ID')
        values=list(map(float,parts[1:]));require(all(math.isfinite(v) for v in values),'Nonfinite YOLO coordinate')
        precision=min(-Decimal(p).as_tuple().exponent for p in parts[1:])
        box=from_yolo(values,width,height,
                      decimal_places=precision if 9<=precision<=16 else None)
        objects.append(dict(class_id=cid,xyxy=box))
    return objects


def accepted_plan(source,acceptance):
    import yaml
    source,acceptance=Path(source).resolve(),Path(acceptance).resolve()
    audit=json.loads((acceptance/'acceptance_reviewed.json').read_text(encoding='utf-8'))
    require(audit['status']=='DATA_PASS_WITH_TOOL_COMPATIBILITY_WARNING','Expected completed read-only acceptance')
    require(Path(audit['source']).resolve()==source,'Acceptance references different source')
    hashes=json.loads((acceptance/'source_sha256.json').read_text(encoding='utf-8'))
    plan=dict(workflow='p03_accepted',source=str(source),source_sha256=hashes)
    check_sources(plan)
    config=yaml.safe_load((source/'data.yaml').read_text(encoding='utf-8-sig'))
    expected=dict(train='images/train',val='images/validation',names=dict(enumerate(CLASSES)))
    require(set(config)<={'path','train','val','nc','names'} and all(config.get(k)==v for k,v in expected.items())
            and config.get('path','.')=='.' and config.get('nc',6)==6,'Invalid source YAML or forbidden test/download reference')
    require((source/'classes.txt').read_text(encoding='utf-8-sig').splitlines()==list(CLASSES),'Source catalogue differs')
    manifest=rows(acceptance/'p06_derived_manifest_reviewed.csv')
    require(len({r['image_id'] for r in manifest})==len(manifest) and len({r['scene_id'] for r in manifest})==len(manifest),'Repeated image or provisional scene ID')
    images=rows(source/'ground_truth_images.csv'); index={r['image_id']:r for r in images}
    require(len(index)==len(images) and set(index)=={r['image_id'] for r in manifest},'Manifest/image CSV membership differs')
    gt={name:[] for name in index}; ids=set()
    for r in rows(source/'ground_truth_objects.csv'):
        key=(r['image_id'],r['object_id'])
        require(key not in ids and r['object_id'] and r['image_id'] in gt and r['ignore']=='0','Invalid object ID/ignore policy')
        ids.add(key);cid=int(r['class_id']);require(0<=cid<6,'Unknown GT class ID')
        b=[float(r[k]) for k in ('xmin','ymin','xmax','ymax')]; im=index[r['image_id']]
        _box(b,int(im['width']),int(im['height']));gt[r['image_id']].append(dict(class_id=cid,xyxy=b))
    empty={name for s in audit['splits'].values() for name in s['empty']}
    records=[]
    for r in manifest:
        name=r['image_id'];split=r['source_split']; canonical={'train':'train','validation':'val'}
        require(name==Path(name).name and '/' not in name and '\\' not in name,'Image must be local basename')
        require(split in canonical and r['split']==canonical[split],'Manifest must preserve P03 train/validation only')
        require(r['scene_id'].strip() and r['scene_id'].upper() not in {'TBD','UNKNOWN'} and r['scene_authority'],'Unconfirmed provisional scene evidence')
        indexed=index[name];require(indexed['split']==split and indexed['annotation_status']=='reviewed','P03 image status/split differs')
        path=source/'images'/split/name
        require(path.suffix.lower()=='.jpg' and digest(path)==r['sha256'],'Image differs from accepted manifest')
        with Image.open(path) as im:
            width,height=im.size
            require(im.format=='JPEG' and im.getexif().get(274,1)==1,'P03 image requires a fresh orientation audit')
            im.verify()
        # Official verify_image_label repairs a missing JPEG EOI in-place. Refuse it before linking.
        with path.open('rb') as f: f.seek(-2,2);require(f.read()==b'\xff\xd9','JPEG repair would mutate source; refuse runtime linking')
        doc=_load_labelme_document(path.with_suffix('.json'))
        require(doc.get('imagePath')==name and (doc['imageWidth'],doc['imageHeight'])==(width,height)
                and (int(indexed['width']),int(indexed['height']))==(width,height),'Image/JSON/CSV geometry differs')
        objects=_parse_shapes(doc,width,height,name,empty)
        txt=parse_txt((source/'labels'/split/Path(name).with_suffix('.txt')).read_text(encoding='utf-8-sig'),width,height)
        match(objects,txt);match(objects,gt[name])
        counts=[sum(o['class_id']==c for o in objects) for c in range(6)]
        fields=['non_catalogue_count',*[n+'_count' for n in CLASSES[1:]]]
        require(counts==[int(indexed[k]) for k in fields],'Per-image counts differ')
        records.append(dict(image_id=name,image_name=name,png_name=name,scene_id=r['scene_id'],
            scene_authority=r['scene_authority'],scene_basis=r['scene_basis'],source_split=split,split=r['split'],
            image_status=indexed.get('reference_label_all_one','UNKNOWN'),width=width,height=height,
            counts=counts,objects=objects,display_pixel_sha256=r['pixel_sha256']))
    for split in ('train','validation'):
        expected={r['image_id'] for r in records if r['source_split']==split}
        folder=source/'images'/split; labels=source/'labels'/split
        require({p.name for p in folder.glob('*.jpg')}==expected and {p.stem for p in folder.glob('*.json')}=={Path(n).stem for n in expected}
                and {p.stem for p in labels.glob('*.txt')}=={Path(n).stem for n in expected},'Split image/JSON/TXT membership differs')
    summary=summarize(records)
    require(len(records)==audit['valid_images'] and sum(sum(r['counts']) for r in records)==audit['objects'],'Accepted dataset totals differ')
    for split in ('train','val'):
        require(summary[split]['images']==audit['splits'][split]['images'] and summary[split]['classes']==audit['splits'][split]['classes'],'Accepted split totals differ')
    plan.update(records=records,summary=summary,confirmed_empty=sorted(empty),
        acceptance_sha256={p.name:digest(p) for p in [acceptance/'acceptance_reviewed.json',acceptance/'source_sha256.json',acceptance/'p06_derived_manifest_reviewed.csv']},
        limitations=['Test not accessed; cross-Test leakage and historical exposure remain P03/P08 responsibilities','Scene IDs provisional, based on user-confirmed rearrangements'])
    check_sources(plan)
    return plan


def prepare_runtime(plan,output,official_check=True):
    out=Path(output).resolve(); source=Path(plan['source'])
    require(out!=source and not out.is_relative_to(source) and not source.is_relative_to(out),'Unsafe runtime directory')
    check_sources(plan)
    def validate_existing():
        safe_files(out)
        require(json.loads((out/'source_plan.json').read_text(encoding='utf-8'))==plan,'Runtime references different accepted data')
        inventory=json.loads((out/'runtime_sha256.json').read_text(encoding='utf-8'))
        for relative,h in inventory.items(): require(digest(out/relative)==h,'Runtime file changed')
        for split in ('train','val'):
            group=[r for r in plan['records'] if r['split']==split]
            require({p.name for p in (out/'images'/split).iterdir()}=={r['image_id'] for r in group},'Unexpected runtime image')
            require({p.name for p in (out/'labels'/split).iterdir()}=={str(Path(r['image_id']).with_suffix('.txt')) for r in group},'Unexpected runtime label')
            for r in group:
                require((out/'images'/split/r['image_id']).samefile(source/'images'/r['source_split']/r['image_id']),'Runtime image link changed')
        proof=json.loads((out/'validation/official_loader.json').read_text(encoding='utf-8'))
        require(not official_check or proof['status']=='PASS','Runtime official loader not verified')
        return proof
    if out.exists(): return validate_existing()
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.p06-runtime-',dir=out.parent) as temp:
        if os.name=='nt': subprocess.run(['icacls',temp,'/inheritance:e'],check=True,capture_output=True)
        stage=Path(temp)/'complete';stage.mkdir()
        for split in ('train','val'):
            (stage/'images'/split).mkdir(parents=True);(stage/'labels'/split).mkdir(parents=True)
        (stage/'validation').mkdir()
        for r in plan['records']:
            original=source/'images'/r['source_split']/r['image_id']
            os.link(original,stage/'images'/r['split']/r['image_id'])
            label=Path(r['image_id']).with_suffix('.txt')
            shutil.copyfile(source/'labels'/r['source_split']/label,stage/'labels'/r['split']/label)
        (stage/'dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/val\nnc: 6\nnames:\n'+''.join(f'  {c}: {n}\n' for c,n in enumerate(CLASSES)),encoding='utf-8')
        (stage/'source_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        proof=validate_ultralytics(stage,plan) if official_check else dict(status='NOT_RUN',reason='Synthetic fixture only')
        (stage/'validation/official_loader.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
        included=[stage/'dataset.yaml',stage/'source_plan.json',*(stage/'images').rglob('*.jpg'),*(stage/'labels').rglob('*.txt')]
        (stage/'runtime_sha256.json').write_text(json.dumps({p.relative_to(stage).as_posix():digest(p) for p in included},indent=2))
        check_sources(plan)
        require(not out.exists(),'Runtime output appeared during preparation')
        stage.rename(out)
    return validate_existing()
