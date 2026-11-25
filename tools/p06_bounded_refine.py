"""Compare one bounded screw refinement against saved versions on marked Train crops."""
import argparse
import json
from pathlib import Path
import sys
import csv

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.p06_geometry import measure_object
from tools.p06_screw_review import OUTPUT,HISTORY,KEYS,compare_marks
from tools.p06_measure_geometry import result_global,save
from tools.p06_measure_candidate import sha256,font
from tools.p06_edge_audit import overlay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--review-freeze',type=Path,help='Read the development freeze before accessing held-back crops')
    args=parser.parse_args()
    dest=args.output.resolve()
    if dest.exists() or not dest.is_relative_to(ROOT/'output'):parser.error('Need new output directory')
    freeze=dict(geometry_sha256=sha256(ROOT/'src/p06_geometry.py'),
                refinement_sha256=sha256(ROOT/'src/p06_screw_edges.py'))
    if args.review_freeze:
        old=json.loads(args.review_freeze.read_bytes())
        if old!=freeze:raise ValueError('Code changed after development freeze')
    role='review' if args.review_freeze else 'development'
    q=json.loads((OUTPUT/'queue.json').read_bytes())
    marks=json.loads((ROOT/'data/p06_screw_marks.json').read_bytes())
    compare_marks(q,json.loads((OUTPUT/'comparison.json').read_bytes()),marks)
    base={r['object_id']:r for r in json.loads((OUTPUT/'comparison.json').read_bytes())}
    params=json.loads((HISTORY/'selection.json').read_bytes())['parameters']
    rows=[];pictures=[]
    for task in q['tasks']:
        if task['role']!=role:continue
        path=OUTPUT/task['src']
        if sha256(path)!=task['crop_sha256']:raise ValueError('Crop changed')
        image=Image.open(path).convert('RGB');bgr=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        result=measure_object(bgr,task['class_id'],q['scale_px_per_mm'],params,screw_method='bounded_refine')
        global_result=result_global(result,task['crop_xyxy'][:2])
        oid=task['object_id']
        for phase,prediction in [('original',base[oid]['before']),('previous_default',base[oid]['body_profile']),('bounded_refine',global_result)]:
            for key in KEYS:
                dim=prediction['dimensions'].get(key)
                row=dict(object_id=oid,class_id=task['class_id'],role=role,phase=phase,dimension=key,
                         candidate=dim is not None,reason=prediction.get('dimension_status',{}).get(key,{}).get('reason',prediction['reason']))
                if dim:
                    gt=np.asarray(marks['marks'][oid][key],float);line=np.asarray(dim['line_xy'])
                    row.update(difference_mm=float(dim['mm_estimate']-np.linalg.norm((gt[1]-gt[0])/q['scale_px_per_mm'])),
                               first_endpoint_offset_px=float(np.linalg.norm(line[0]-gt[0])),
                               second_endpoint_offset_px=float(np.linalg.norm(line[1]-gt[1])))
                rows.append(row)
        human=image.copy();d=ImageDraw.Draw(human);origin=np.asarray(task['crop_xyxy'][:2])
        for line in marks['marks'][oid].values():
            d.line([tuple(p-origin) for p in np.asarray(line)],fill='#00cc77',width=4)
        human.thumbnail((580,430))
        tile=Image.new('RGB',(1200,490),'white');tile.paste(human,((600-human.width)//2,50))
        tile.paste(overlay(image,result,oid,'bounded refinement'),(600,0))
        ImageDraw.Draw(tile).text((8,8),oid+' human',font=font(17),fill='black')
        pictures.append((oid,tile))
    dest.mkdir(parents=True)
    save(dest/'freeze.json',freeze)
    save(dest/'comparison.json',rows)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (dest/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    for oid,tile in pictures:tile.save(dest/(oid+'.jpg'),quality=89)
    summary={}
    for phase in ('original','previous_default','bounded_refine'):
        summary[phase]={}
        for key in KEYS:
            group=[r for r in rows if r['phase']==phase and r['dimension']==key];valid=[r for r in group if r['candidate']]
            summary[phase][key]=dict(n=len(valid),total=len(group),mean_absolute_mm=float(np.mean([abs(r['difference_mm']) for r in valid])) if valid else None)
    save(dest/'summary.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':main()
