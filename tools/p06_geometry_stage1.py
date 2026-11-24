"""Bounded Train-only before/after geometry check; never evaluates Validation/Test.

Original GT boxes isolate the object. No nominal dimensions, tolerances, training,
YOLO or acceptance decisions are used in selecting boundaries.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.p06_geometry import measure_object
from tools.p06_measure_geometry import SOURCE,catalogue_records,crop_for_object,result_global,panel,save
from tools.p06_measure_candidate import load_display,sha256,write_csv,font

EXPERIMENT=ROOT/'output/p06_geometry_stage1'


def review_panel(crop,result,title):
    if not result['diagnostics'].get('partial_dimensions'):return panel(crop,result,title)
    display=json.loads(json.dumps(result))
    display['dimensions']=display['diagnostics']['partial_dimensions']
    tile=panel(crop,display,title)
    draw=ImageDraw.Draw(tile)
    draw.rectangle((0,65,460,93),fill='white')
    draw.text((5,68),'PARTIAL / REVIEW: '+result['reason'][:34],fill='#aa0000',font=font(13))
    return tile


def before_module(selection):
    path=EXPERIMENT/'before/p06_geometry.py'
    if sha256(path)!=selection['before_module_sha256']:raise ValueError('Original baseline snapshot changed')
    spec=importlib.util.spec_from_file_location('p06_stage1_original',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def compare(selection,full_train=False):
    original=before_module(selection)
    index,objects=catalogue_records('train')
    wanted=set(index) if full_train else set(selection['train_image_selection'])
    if not wanted.issubset(index):raise ValueError('Selection contains a non-Train image')
    records=[];hashes={};tiles=[];preview_ids={'fail122_004','pass01_005','pass05_003','pass11_003','pass05_001','fail78_004'}
    for name in sorted(wanted):
        path=SOURCE/'images/train'/name;hashes[name]=sha256(path)
        image,_=load_display(path)
        for row in objects[name]:
            cid=int(row['class_id'])
            if cid not in (1,2,3,5):continue
            crop,box=crop_for_object(image,row);bgr=cv2.cvtColor(np.array(crop),cv2.COLOR_RGB2BGR)
            pair={}
            for label,fn in [('before',original.measure_object),('after',measure_object)]:
                if image.size!=(3024,4032):
                    result=dict(status='UNMEASURABLE',reason='SCALE_TRANSFER_UNCONFIRMED_RESOLUTION',class_id=cid,
                                decision=None,dimensions={},diagnostics={},contours_xy=[])
                else:
                    options = {'screw_method':'stage1'} if label == 'after' else {}
                    result=fn(bgr,cid,selection['scale_px_per_mm'],selection['parameters'],**options)
                pair[label]=result_global(result,box[:2])
                if row['object_id'] in preview_ids:
                    tiles.append(review_panel(crop,result,label+' '+row['object_id']))
            records.append(dict(image_id=name,object_id=row['object_id'],class_id=cid,split='train',crop_xyxy=box,**pair))
        if sha256(path)!=hashes[name]:raise ValueError('Source changed: '+name)
    return records,hashes,tiles


def run(output,full_train=False):
    output=Path(output).resolve()
    if not output.is_relative_to(EXPERIMENT) or output.exists():raise ValueError('Need a NEW output/p06_geometry_stage1/ child directory')
    selection=json.loads((EXPERIMENT/'selection.json').read_text(encoding='utf-8'))
    # Check all inputs before publishing any result files.
    records,hashes,tiles=compare(selection,full_train)
    output.mkdir(parents=True)
    save(output/'records.json',records)
    save(output/'source_image_sha256.json',hashes)
    summary={}
    for cid in (1,2,3,5):
        rows=[r for r in records if r['class_id']==cid]
        summary[str(cid)]=dict(objects=len(rows),**{phase:dict(
            candidate_count=sum(r[phase]['status']=='MEASURED_CANDIDATE' for r in rows),
            failure_reasons=dict(Counter(r[phase]['reason'] for r in rows if r[phase]['reason']))) for phase in ('before','after')})
        summary[str(cid)]['newly_refused']=sum(r['before']['status']=='MEASURED_CANDIDATE' and r['after']['status']=='UNMEASURABLE' for r in rows)
        summary[str(cid)]['newly_available']=sum(r['before']['status']=='UNMEASURABLE' and r['after']['status']=='MEASURED_CANDIDATE' for r in rows)
        summary[str(cid)]['after_partial_review_only']=sum(bool(r['after']['diagnostics'].get('partial_dimensions')) for r in rows)
    save(output/'summary.json',summary)
    flat=[]
    for row in records:
        for phase in ('before','after'):
            r=row[phase]
            for key,d in (r['dimensions'].items() or [('none',{})]):
                flat.append(dict(image_id=row['image_id'],object_id=row['object_id'],class_id=row['class_id'],
                                 phase=phase,status=r['status'],reason=r['reason'],dimension=key,
                                 pixels=d.get('pixels',''),mm_estimate=d.get('mm_estimate',''),
                                 line_xy=json.dumps(d.get('line_xy',[]))))
            for key,d in r['diagnostics'].get('partial_dimensions',{}).items():
                flat.append(dict(image_id=row['image_id'],object_id=row['object_id'],class_id=row['class_id'],
                                 phase=phase,status='PARTIAL_REVIEW_ONLY',reason=r['reason'],dimension=key,
                                 pixels=d['pixels'],mm_estimate=d['mm_estimate'],line_xy=json.dumps(d['line_xy'])))
    write_csv(output/'comparison.csv',flat)
    if tiles:
        # Exactly two columns: the same real crop before and after.
        sheet=Image.new('RGB',(920,490*((len(tiles)+1)//2)),'white')
        for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*460,(i//2)*490))
        sheet.save(output/'before_after.jpg',quality=91)
    frozen=dict(scope='TRAIN_ONLY_STAGE1_NOT_VALIDATED',full_train=full_train,selection=selection,
                module_sha256=sha256(ROOT/'src/p06_geometry.py'),runner_sha256=sha256(__file__),
                original_snapshot_sha256=sha256(EXPERIMENT/'before/p06_geometry.py'),
                validation_run=False,test_accessed=False,nominal_size_used_for_boundaries=False,
                physical_error_validation='NOT_RUN',new_yolo_training=False)
    save(output/'experiment.json',frozen)
    print(json.dumps(dict(output=str(output),images=len(hashes),summary=summary),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full-train',action='store_true',help='Check all Train images after bounded development; no Validation/Test')
    args=parser.parse_args()
    try:run(args.output,args.full_train)
    except (ValueError,OSError) as error:parser.exit(1,f'Train-only geometry check failed: {error}\n')


if __name__=='__main__':main()
