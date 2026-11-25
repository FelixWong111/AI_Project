"""Fixed 16-image Train diagnostic. No tuning, training, Validation or Test images."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time
import argparse

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.p06_pipeline import EvidencePipeline,IntegratedBackend
from src.p06_inference import create_backend
from src.p06_foreground import ForegroundAssociator,iou
from src.p06_size_evidence import load_specs,SizeEvidenceProcessor
from src.p06_geometry import measure_object
from tools.p06_measure_geometry import SOURCE,catalogue_records,crop_for_object,result_global,save
from tools.p06_measure_candidate import load_display,sha256,font
from tools.p06_screw_review import OUTPUT,KEYS

NAMES=('fail123.jpg','fail124.jpg','fail02.jpg','fail100.jpg','fail101.jpg',
       'fail113.jpg','fail116.jpg','pass55.jpg','pass123.jpg','pass132.jpg',
       'fail46.jpg','pass50.jpg','pass58.jpg','pass142.jpg','pass32.jpg','fail110.jpg')


def csv_file(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def matched(objects,gt):
    available=set(range(len(gt)));pairs=[]
    for obj in sorted(objects,key=lambda r:(-r['score'],r['detection_id'])):
        choices=[(iou(obj,gt[j]),j) for j in available if obj['class_id']==gt[j]['class_id']]
        if choices:
            overlap,j=max(choices,key=lambda p:(p[0],-p[1]))
            if overlap>=.5:
                pairs.append((obj['detection_id'],gt[j]['object_id']))
                available.remove(j)
    return pairs


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/p06_train_limited_validation')
    dest=parser.parse_args().output.resolve()
    if not dest.is_relative_to(ROOT/'output'):raise ValueError('Output must be inside output/')
    if dest.exists():raise FileExistsError('Refusing existing output')
    index,objects=catalogue_records('train')
    if not set(NAMES)<=index.keys():raise ValueError('Selected image outside Train')
    if objects['fail123.jpg'] or objects['fail124.jpg']:raise ValueError('Background has GT objects')
    cfg=json.loads((ROOT/'config/p06_size_calibration_candidate.json').read_bytes())
    specs=load_specs(ROOT/'config/p06_part_specs_candidate.json')
    background,_=load_display(SOURCE/'images/train/fail123.jpg')
    bg=cv2.cvtColor(np.asarray(background),cv2.COLOR_RGB2BGR)
    fg=ForegroundAssociator(bg,reference_id='Train-fail123-candidate-background',
                            capture_geometry_id=cfg['capture_geometry_id'])
    size=SizeEvidenceProcessor(specs,cfg)
    detector=create_backend(model_id='baseline3',config={})
    dest.mkdir(parents=True)
    (dest/'predictions').mkdir()
    hashes={n:sha256(SOURCE/'images/train'/n) for n in NAMES}
    protected={str(p):sha256(p) for p in (ROOT/'config/p06_baseline3.json',SOURCE/'ground_truth_images.csv',SOURCE/'ground_truth_objects.csv')}
    save(dest/'input_manifest.json',dict(split='train',names=NAMES,image_sha256=hashes,
        method='fixed diagnostic selection, includes two empties, one unsupported resolution, ID0 and known manual regression cases',
        background='fail123.jpg',background_match='ASSUMED same fixed capture; illumination not guaranteed',calibration=cfg))
    manual=json.loads((ROOT/'data/p06_screw_marks.json').read_bytes())
    queue=json.loads((OUTPUT/'queue.json').read_bytes())
    if manual['queue_id']!=queue['queue_id']:raise ValueError('Manual queue mismatch')
    summaries=[];measurements=[];manual_rows=[];tiles=[]
    for name in NAMES:
        image,meta=load_display(SOURCE/'images/train'/name)
        frame=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        # Resolution mismatch is explicit. The unsupported image is still
        # checked for detector/size behavior without pretending background fit.
        associated=fg if frame.shape==bg.shape else None
        backend=IntegratedBackend(detector,EvidencePipeline(size,capture_geometry_id=cfg['capture_geometry_id'],
                                                           foreground=associated),{'evidence':'LIMITED_TRAIN_DIAGNOSTIC'})
        backend.synchronize();start=time.perf_counter()
        result=backend.infer(frame,dict.fromkeys('12345',1))
        backend.synchronize();elapsed=time.perf_counter()-start
        save(dest/'predictions'/f'{Path(name).stem}.json',result)
        gt=[dict(object_id=r['object_id'],class_id=int(r['class_id']),
                 **{k:float(r[k]) for k in ('xmin','ymin','xmax','ymax')}) for r in objects[name]]
        raw=result['p06_evidence']['raw_observations']['objects']
        pairs=matched(result['objects'],gt);rawpairs=matched(raw,gt)
        front=result['p06_evidence']['foreground']
        summaries.append(dict(filename=name,gt_objects=len(gt),gt_id0=sum(r['class_id']==0 for r in gt),
            raw_detections=len(raw),retained_detections=len(result['objects']),raw_matched=len(rawpairs),retained_matched=len(pairs),
            raw_unmatched_predictions=len(raw)-len(rawpairs),retained_unmatched_predictions=len(result['objects'])-len(pairs),
            retained_missed_gt=len(gt)-len(pairs),foreground_regions=len(front['regions']) if front else None,
            unresolved_candidates=len(front['candidates'])-sum(g['class_id'] is not None for g in front['candidates']) if front else None,
            background_usable=front['background_usable'] if front else False,
            suppressed=len(front['suppressed_detection_ids']) if front else 0,elapsed_seconds=elapsed))
        per_gt={}
        for r in gt:
            if r['class_id']==0:continue
            crop,box=crop_for_object(image,r)
            if image.size==(cfg['frame_width'],cfg['frame_height']):
                geometry=measure_object(cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR),r['class_id'],cfg['px_per_mm'])
                geometry=result_global(geometry,box[:2])
            else:
                geometry=dict(dimensions={},reason='SCALE_TRANSFER_UNCONFIRMED_RESOLUTION')
            per_gt[r['object_id']]=geometry
            measurements.append({**geometry,'filename':name,'object_id':r['object_id'],'class_id':r['class_id'],
                                 'source':'GT_CROP'})
        mapping={d:oid for d,oid in pairs}
        for obj in result['objects']:
            evidence=obj['size_evidence']
            geometry=evidence.get('measurement',dict(dimensions={},reason=evidence.get('reason','NO_SIZE_EVIDENCE')))
            translated=json.loads(json.dumps(geometry))
            if 'crop_xyxy' in evidence:
                origin=np.asarray(evidence['crop_xyxy'][:2])
                for dim in translated['dimensions'].values():dim['line_xy']=(np.asarray(dim['line_xy'])+origin).tolist()
            measurements.append({**translated,'filename':name,'object_id':mapping.get(obj['detection_id']),
                                 'detection_id':obj['detection_id'],'class_id':obj['class_id'],'source':'YOLO_CROP'})
        selected_manual=[t for t in queue['tasks'] if t['image_id']==name]
        for task in selected_manual:
            oid=task['object_id']
            for source in ('GT_CROP','YOLO_CROP'):
                row=next((m for m in measurements if m['filename']==name and m['source']==source and m['object_id']==oid),None)
                for key in KEYS:
                    dim=row['dimensions'].get(key) if row else None
                    item=dict(filename=name,object_id=oid,source=source,dimension=key,candidate=dim is not None,
                              reason=row.get('reason','') if row else 'NO_SAME_CLASS_IOU_MATCH')
                    if dim:
                        points=np.asarray(manual['marks'][oid][key])
                        item['difference_mm']=float(dim['mm_estimate']-np.linalg.norm((points[1]-points[0])/cfg['px_per_mm']))
                    manual_rows.append(item)
        overlay=image.copy();draw=ImageDraw.Draw(overlay)
        for r in gt:draw.rectangle([r['xmin'],r['ymin'],r['xmax'],r['ymax']],outline='#00bc66',width=6)
        for obj in result['objects']:
            draw.rectangle([obj[k] for k in ('xmin','ymin','xmax','ymax')],outline='#ed228c',width=4)
            draw.text((obj['xmin'],obj['ymin']),str(obj['class_id']),font=font(35),fill='#ed228c')
        for m in measurements:
            if m['filename']==name and m['source']=='YOLO_CROP':
                for dim in m['dimensions'].values():draw.line([tuple(p) for p in dim['line_xy']],fill='#00a0ff',width=4)
        overlay.thumbnail((480,620))
        tile=Image.new('RGB',(500,680),'white');tile.paste(overlay,((500-overlay.width)//2,50))
        ImageDraw.Draw(tile).text((8,5),name+' green=GT pink=YOLO',font=font(16),fill='black')
        tiles.append(tile)
        print(name,'GT',len(gt),'raw',len(raw),'retained',len(result['objects']),'matched',len(pairs),flush=True)
    for name in NAMES:
        if sha256(SOURCE/'images/train'/name)!=hashes[name]:raise ValueError('Source changed')
    if any(sha256(p)!=v for p,v in protected.items()):raise ValueError('Protected inputs changed')
    csv_file(dest/'image_results.csv',summaries);csv_file(dest/'manual_comparison.csv',manual_rows)
    save(dest/'measurement_records.json',measurements)
    for page in range(2):
        sheet=Image.new('RGB',(2000,1360),'white')
        for i,tile in enumerate(tiles[page*8:page*8+8]):sheet.paste(tile,((i%4)*500,(i//4)*680))
        sheet.save(dest/f'contact_sheet_{page+1}.jpg',quality=90)
    coverage=[]
    from src.p06_size_evidence import DIMENSIONS
    for source in ('GT_CROP','YOLO_CROP'):
        for cid,keys in DIMENSIONS.items():
            rows=[m for m in measurements if m['source']==source and m['class_id']==cid]
            coverage.append(dict(source=source,class_id=cid,objects=len(rows),
                **{k:sum(k in r['dimensions'] for r in rows) for k in keys},
                both=sum(all(k in r['dimensions'] for k in keys) for r in rows)))
    save(dest/'coverage.json',coverage)
    save(dest/'verification.json',dict(source_sha256_unchanged=True,protected_sha256=protected,
        evidence='SELECTED_TRAIN_DIAGNOSTIC_NOT_INDEPENDENT_ACCURACY',real_background_candidate=True,
        physical_size_accuracy='NOT_RUN',script_sha256=sha256(__file__)))


if __name__=='__main__':run()
