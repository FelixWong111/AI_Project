"""Train comparison for shaft exterior, exposed end planes and radial aperture."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p06_geometry import measure_object
from tools.p06_measure_candidate import font, load_display, sha256
from tools.p06_measure_geometry import SOURCE, catalogue_records, crop_for_object, result_global, save
from tools.p06_screw_review import HISTORY, OUTPUT, ROOT, compact


def overlay(crop, result, name, variant):
    im = crop.copy()
    draw = ImageDraw.Draw(im)
    dims = result['dimensions']
    if 'underhead_length' in dims:
        a,b = np.asarray(dims['underhead_length']['line_xy'])
        unit = (b-a)/np.linalg.norm(b-a)
        draw.line([tuple(a-25*unit),tuple(b+25*unit)], fill='#20b3b0',width=4)
        for point,color,letter in [(a,'#fadc28','S'),(b,'#fd55a0','T')]:
            x,y = point
            draw.ellipse((x-6,y-6,x+6,y+6), fill=color,outline='black',width=2)
            draw.text((x+7,y-20),letter,font=font(22),fill=color,stroke_width=2,stroke_fill='black')
        if 'thread_major_diameter' in dims:
            draw.line([tuple(p) for p in dims['thread_major_diameter']['line_xy']],fill='#1882f0',width=4)
    elif 'thread_major_diameter' in dims:
        draw.line([tuple(p) for p in dims['thread_major_diameter']['line_xy']],fill='#1882f0',width=4)
    elif 'inner_diameter' in dims:
        for contour,color in zip(result['contours_xy'],('#1882f0','#fadc28')):
            pts = [tuple(p) for p in contour]
            draw.line(pts+[pts[0]],fill=color,width=3)
        for dim in dims.values():
            draw.line([tuple(p) for p in dim['line_xy']],fill='#ec5190',width=3)
    im.thumbnail((580,430))
    out = Image.new('RGB',(600,490),'white')
    out.paste(im,((600-im.width)//2,50))
    d = ImageDraw.Draw(out)
    d.text((8,4),f'{name}   {variant}',font=font(17),fill='black')
    d.text((8,28),result['reason'] or 'candidate',font=font(14),fill='black')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/p06_edge_audit')
    target = parser.parse_args().output.resolve()
    if not target.is_relative_to(ROOT/'output') or target.exists():
        raise FileExistsError(target)
    # Render files only after the full cohort has been measured successfully.
    hist = HISTORY/'train_full_final'
    old = {r['object_id']:r for r in json.loads((hist/'records.json').read_bytes())}
    hashes = json.loads((hist/'source_image_sha256.json').read_bytes())
    base = {r['object_id']:r for r in json.loads((OUTPUT/'comparison.json').read_bytes())}
    selection = json.loads((HISTORY/'selection.json').read_bytes())
    index, objects = catalogue_records('train')
    records, pics = [], {}
    focus = {'fail113_006','fail116_007','fail46_006','fail122_004','pass123_001','fail166_001',
             'pass05_003','pass01_005','pass11_003','fail78_004'}
    for name in sorted(index):
        path = SOURCE/'images/train'/name
        if sha256(path)!=hashes[name]:
            raise ValueError('Train source changed: '+name)
        image,_ = load_display(path)
        for obj in objects[name]:
            cid,oid = int(obj['class_id']),obj['object_id']
            if cid not in (1,2,3,5):
                continue
            crop,box = crop_for_object(image,obj)
            ref = old.pop(oid)
            if ref['crop_xyxy']!=list(box) or (ref['class_id'],ref['image_id'],ref['split'])!=(cid,name,'train'):
                raise ValueError('Historical GT or crop changed: '+oid)
            previous = ref['after'] if cid==5 else base[oid]['body_profile']
            if image.size!=(3024,4032):
                audited = compact(ref['after'])
            else:
                bgr = cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR)
                args = (bgr,cid,selection['scale_px_per_mm'],selection['parameters'])
                raw = measure_object(*args,screw_method='edge_audit',radial_washer=True)
                audited = compact(result_global(raw,box[:2]))
                if cid==5 and raw['dimensions']:
                    audited['supported_rays'] = raw['diagnostics']['radial_aperture_supported_rays']
                if oid in focus:
                    default = measure_object(*args)
                    both = Image.new('RGB',(1200,490),'white')
                    both.paste(overlay(crop,default,oid,'current candidate'),(0,0))
                    both.paste(overlay(crop,raw,oid,'edge audit'),(600,0))
                    pics[oid] = both
            records.append(dict(object_id=oid,image_id=name,class_id=cid,split='train',
                                crop_xyxy=box, previous=compact(previous), edge_audit=audited))
    if old or set(pics)!=focus:
        raise ValueError('Incomplete Train comparison')
    summary = {}
    for cid in (1,2,3,5):
        group = [r for r in records if r['class_id']==cid]
        summary[cid] = dict(total=len(group),previous=sum(bool(r['previous']['dimensions']) for r in group),
                            audited=sum(bool(r['edge_audit']['dimensions']) for r in group),
                            gained=sum(not r['previous']['dimensions'] and bool(r['edge_audit']['dimensions']) for r in group),
                            lost=sum(bool(r['previous']['dimensions']) and not r['edge_audit']['dimensions'] for r in group),
                            refused=dict(Counter(r['edge_audit']['reason'] for r in group if not r['edge_audit']['dimensions'])))
    target.mkdir()
    for oid,im in pics.items():
        im.save(target/(oid+'.jpg'),quality=92)
    save(target/'records.json',records)
    save(target/'summary.json',dict(coverage=summary,code_sha256=sha256(ROOT/'src/p06_geometry.py'),
                                    scale=selection['scale_px_per_mm'],parameters=selection['parameters']))
    print({cid:(s['previous'],s['audited']) for cid,s in summary.items()})


if __name__=='__main__':
    main()
