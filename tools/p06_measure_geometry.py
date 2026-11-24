"""Train-only parameter selection, then fixed-parameter candidate metrology checks.

Uses current development GT boxes, NOT YOLO detections; never reads Test. All
outputs are separate from frozen inference/model configuration and original data.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.p06_geometry import DEFAULT_PARAMETERS,measure_object
from tools.p06_measure_candidate import font,load_display,sha256,write_csv,candidate_contour,OBJECT_ROI

SOURCE=ROOT/'data/p03/development_v2'
SCALE_CSV=ROOT/'output/p06_measure_candidate/scale_fits.csv'
GEOMETRY=ROOT/'src/p06_geometry.py'


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def new_output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT/'output') or path.exists():raise ValueError('Need a new directory under output/; never overwrite')
    path.mkdir(parents=True)
    return path


def catalogue_records(split):
    if split not in ('train','validation'):raise ValueError('Only train/validation are permitted')
    images=[r for r in read_csv(SOURCE/'ground_truth_images.csv') if r['split']==split]
    index={r['image_id']:r for r in images}
    objects=defaultdict(list)
    for r in read_csv(SOURCE/'ground_truth_objects.csv'):
        if r['image_id'] not in index:continue
        if r['ignore']!='0':raise ValueError('Unagreed ignored object')
        objects[r['image_id']].append(r)
    for name in index:
        if name!=Path(name).name or '/' in name or '\\' in name:raise ValueError('Unsafe image basename')
    return index,objects


def scales():
    rows=read_csv(SCALE_CSV)
    per={}
    for i in range(1,7):
        per[f'{i}.jpg']=tuple(float(next(r['px_per_mm'] for r in rows if r['filename']==f'{i}.jpg' and r['axis']==a)) for a in ('x','y'))
    avg=tuple(float(np.mean([per[f'{i}.jpg'][j] for i in (1,4,5,6)])) for j in (0,1))
    return avg,per


def crop_for_object(image,row,padding=.18):
    coords=np.array([float(row[k]) for k in ('xmin','ymin','xmax','ymax')])
    x1,y1,x2,y2=coords
    if not np.isfinite(coords).all() or not (0<=x1<x2<=image.width and 0<=y1<y2<=image.height):raise ValueError('Invalid original box')
    pad=max(16,int(max(x2-x1,y2-y1)*padding))
    box=(max(0,int(x1)-pad),max(0,int(y1)-pad),min(image.width,int(np.ceil(x2))+pad),min(image.height,int(np.ceil(y2))+pad))
    return image.crop(box),box


def result_global(result,offset):
    result=json.loads(json.dumps(result))
    origin=np.array(offset,float)
    for dimension in result['dimensions'].values():dimension['line_xy']=(np.array(dimension['line_xy'])+origin).tolist()
    # Approximation is for compact saved diagnostics only, never the measurement.
    result['contours_xy']=[(cv2.approxPolyDP(np.array(c,np.float32).reshape(-1,1,2),.75,True)[:,0]+origin).tolist() for c in result['contours_xy']]
    for key in ('head_shaft_boundary_xy','hexagon_xy'):
        if key in result['diagnostics']:result['diagnostics'][key]=(np.array(result['diagnostics'][key])+origin).tolist()
    key='opposite_flat_lines_xy'
    if key in result['diagnostics']:result['diagnostics'][key]=(np.array(result['diagnostics'][key])+origin).tolist()
    for value in result['diagnostics'].get('partial_dimensions',{}).values():
        value['line_xy']=(np.array(value['line_xy'])+origin).tolist()
    result['coordinate_system']='original EXIF-corrected image pixels; saved contours simplified at 0.75 px'
    return result


def ruler_crop(image):
    """Reuse the preceding experiment's object ROI; centre the isolated crop."""
    contour=candidate_contour(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2GRAY),OBJECT_ROI,120)
    if contour is None:raise ValueError('NO_RULER_PHOTO_OBJECT')
    x,y,w,h=cv2.boundingRect(contour)
    return crop_for_object(image,dict(xmin=x,ymin=y,xmax=x+w,ymax=y+h))


def panel(crop,result,title):
    annotated=crop.copy();draw=ImageDraw.Draw(annotated)
    for contour in result['contours_xy']:
        if len(contour)>1:draw.line([tuple(p) for p in contour]+[tuple(contour[0])],fill='#00cc00',width=3)
    for i,(name,dim) in enumerate(result['dimensions'].items()):
        draw.line([tuple(p) for p in dim['line_xy']],fill=('#ff00dd','#0088ff')[i%2],width=4)
    boundary=result['diagnostics'].get('head_shaft_boundary_xy')
    if boundary:draw.line([tuple(p) for p in boundary],fill='#ff9900',width=3)
    annotated.thumbnail((440,360))
    tile=Image.new('RGB',(460,490),'white');tile.paste(annotated,((460-annotated.width)//2,95))
    text=ImageDraw.Draw(tile);text.text((5,3),title,fill='black',font=font(17))
    if result['dimensions']:
        for i,(name,value) in enumerate(result['dimensions'].items()):
            text.text((5,25+21*i),f'{name}: {value["pixels"]:.1f}px / ~{value["mm_estimate"]:.2f}mm',fill='black',font=font(14))
    else:text.text((5,25),result['reason'][:56],fill='#990000',font=font(13))
    text.text((5,470),'CANDIDATE; GT crop; no dimensional ground truth',fill='black',font=font(14))
    return tile


def contact(tiles,path):
    if not tiles:return
    sheet=Image.new('RGB',(460*3,490*((len(tiles)+2)//3)),'white')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*460,(i//3)*490))
    sheet.save(path,quality=91)


def run_records(split,parameters,scale,selection=None,tiles_limit=20):
    index,objects=catalogue_records(split);records=[];tiles=[];hashes={};class_previews=Counter()
    for name in sorted(index):
        if selection is not None and name not in selection:continue
        path=SOURCE/'images'/split/name;hashes[str(path.relative_to(ROOT))]=sha256(path)
        image,meta=load_display(path)
        for row in objects[name]:
            cid=int(row['class_id'])
            if cid==0:continue
            crop,box=crop_for_object(image,row)
            if image.size!=(3024,4032):
                result=dict(status='UNMEASURABLE',reason='SCALE_TRANSFER_UNCONFIRMED_RESOLUTION',class_id=cid,
                            decision=None,dimensions={},diagnostics={},contours_xy=[])
            else:
                result=measure_object(cv2.cvtColor(np.array(crop),cv2.COLOR_RGB2BGR),cid,scale,parameters)
            if len(tiles)<tiles_limit and class_previews[cid]<max(1,tiles_limit//5):
                tiles.append(panel(crop,result,f'{split}/{name} {row["object_id"]} ID{cid}'))
                class_previews[cid]+=1
            records.append(dict(image_id=name,object_id=row['object_id'],split=split,
                                source='human_GT_box_and_class; not model prediction',crop_xyxy=box,
                                **result_global(result,box[:2])))
        if sha256(path)!=hashes[str(path.relative_to(ROOT))]:raise ValueError('Source image changed')
    return records,tiles,hashes


def tune(output):
    out=new_output(output);average,_=scales();index,objects=catalogue_records('train')
    eligible=[name for name,r in index.items() if (int(r['width']),int(r['height']))==(3024,4032)
              and {int(o['class_id']) for o in objects[name]}.issuperset({1,2,3,4,5})]
    names=sorted(eligible,key=lambda n:hashlib.sha256(n.encode()).hexdigest())[:12]
    if len(names)<12:raise ValueError('Need 12 eligible Train images')
    trials=[];best=None
    for offset in (25,35,45):
        params={**DEFAULT_PARAMETERS,'background_offset':offset}
        records,tiles,hashes=run_records('train',params,average,set(names),30)
        counts=Counter(r['status'] for r in records)
        trials.append(dict(parameters=params,measured=counts['MEASURED_CANDIDATE'],objects=len(records),
                           failure_reasons=dict(Counter(r['reason'] for r in records if r['reason']))))
        save(out/f'trial_offset{offset}.json',records)
        contact(tiles,out/f'train_offset{offset}.jpg')
        # Pure geometric availability; never optimise error against guessed sizes.
        rank=(counts['MEASURED_CANDIDATE'],-abs(offset-35))
        if best is None or rank>best[0]:best=(rank,params)
    frozen=dict(schema_version=1,status='FROZEN_CANDIDATE_GEOMETRY_NOT_PRODUCT_ACCEPTANCE',
                parameters=best[1],scale_px_per_mm=average,
                scale_source='mean of photos 1,4,5,6; independent x/y; user-approved same-geometry assumption',
                scale_fits_sha256=sha256(SCALE_CSV),module_sha256=sha256(GEOMETRY),runner_sha256=sha256(__file__),
                train_selection=names,selection_method='first 12 complete five-class full-resolution Train images sorted by SHA256(image_id)',
                tuning_objective='maximise available geometric measurements; ties prefer background_offset 35; no nominal-size optimisation',
                trials=trials,validation_used_for_selection=False,product_tolerance=None,
                source_tables_sha256={name:sha256(SOURCE/name) for name in ('ground_truth_images.csv','ground_truth_objects.csv')})
    save(out/'frozen_geometry.json',frozen);save(out/'source_image_sha256.json',hashes)
    print(json.dumps(dict(selected_parameters=best[1],trials=trials,output=str(out)),indent=2))


def evaluate(frozen_path,output):
    frozen=json.loads(Path(frozen_path).read_text(encoding='utf-8'))
    if frozen['module_sha256']!=sha256(GEOMETRY) or frozen['runner_sha256']!=sha256(__file__):raise ValueError('Code changed after Train-only freeze; tune again before Validation')
    if frozen['scale_fits_sha256']!=sha256(SCALE_CSV):raise ValueError('Scale evidence changed')
    for name,digest in frozen['source_tables_sha256'].items():
        if sha256(SOURCE/name)!=digest:raise ValueError('Source table changed after parameter freeze')
    out=new_output(output);save(out/'frozen_geometry.json',frozen)
    all_records=[];hashes={};summary={}
    for split in ('train','validation'):
        records,tiles,source_hashes=run_records(split,frozen['parameters'],frozen['scale_px_per_mm'])
        all_records.extend(records);hashes.update(source_hashes);contact(tiles,out/f'{split}_preview.jpg')
        summary[split]=summarize(records)
    # Ruler photos are check-only, processed after the Train-only parameter freeze.
    _,per=scales();measure=[];tiles=[]
    for name,cid in [('2.jpg',2),('3.jpg',3),('4.jpg',1),('5.jpg',5),('6.jpg',4)]:
        path=ROOT/'data/p01/measure'/name;hashes[str(path.relative_to(ROOT))]=sha256(path)
        image,_=load_display(path)
        crop,box=ruler_crop(image)
        result=measure_object(cv2.cvtColor(np.array(crop),cv2.COLOR_RGB2BGR),cid,per[name],frozen['parameters'])
        # This mapping is a visual geometry example, not verified procurement identity.
        measure.append(dict(image_id=name,geometry_example_id=cid,identity_status='VISUAL_TYPE_ONLY_STANDARD_UNCONFIRMED',
                            crop_xyxy=box,**result_global(result,box[:2])))
        tiles.append(panel(crop,result,'ruler photo '+name+'; visual type only'))
    contact(tiles,out/'ruler_photo_preview.jpg');save(out/'ruler_photo_measurements.json',measure)
    save(out/'measurements.json',all_records);save(out/'summary.json',summary)
    flat=[]
    for r in all_records:
        for key,dim in (r['dimensions'].items() or [('none',{})]):
            flat.append(dict(image_id=r['image_id'],object_id=r['object_id'],split=r['split'],class_id=r['class_id'],
                             status=r['status'],reason=r['reason'],dimension=key,pixels=dim.get('pixels',''),
                             mm_estimate=dim.get('mm_estimate',''),line_xy=json.dumps(dim.get('line_xy',[]))))
    write_csv(out/'measurements.csv',flat)
    for name,digest in hashes.items():
        if sha256(ROOT/name)!=digest:raise ValueError('Source changed: '+name)
    save(out/'source_image_sha256.json',hashes)
    save(out/'completion.json',dict(status='COMPLETED_CANDIDATE_CHECK',source_images_hash_unchanged=len(hashes),
        frozen_parameter_sha256=sha256(Path(frozen_path)),module_sha256=sha256(GEOMETRY),
        no_test_access=True,no_yolo_execution=True,no_product_tolerance=True,physical_accuracy='NOT_VERIFIED_NO_CALIPER_GT'))
    print(json.dumps(summary,indent=2))


def summarize(records):
    result={}
    for cid in range(1,6):
        rows=[r for r in records if r['class_id']==cid];ok=[r for r in rows if r['status']=='MEASURED_CANDIDATE']
        dimensions=defaultdict(list)
        for row in ok:
            for key,value in row['dimensions'].items():dimensions[key].append(value['mm_estimate'])
        result[str(cid)]=dict(objects=len(rows),candidate_measured=len(ok),unmeasurable=len(rows)-len(ok),
            failure_reasons=dict(Counter(r['reason'] for r in rows if r['reason'])),
            dimensions={key:dict(median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9))) for key,values in dimensions.items()})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('tune','evaluate'):
        command=sub.add_parser(mode);command.add_argument('--output',type=Path,required=True)
        if mode=='evaluate':command.add_argument('--frozen',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.mode=='tune':tune(args.output)
        else:evaluate(args.frozen,args.output)
    except (ValueError,OSError,KeyError) as error:parser.exit(1,f'Measurement failed: {error}\n')


if __name__=='__main__':main()
