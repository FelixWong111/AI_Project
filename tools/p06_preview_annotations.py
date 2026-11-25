"""Labelme ground-truth previews, NOT model predictions. Never writes sources."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.p06_labelme import CLASSES, DataError, _dimensions, _load_labelme_document, _parse_shapes, require

# Red, blue, orange, purple, green, cyan. Same palette in all previews/sheets.
COLORS=((210,35,45),(20,80,210),(230,125,0),(150,45,185),(20,135,55),(0,140,165))
IMAGE_EXTENSIONS={'.jpg','.jpeg','.png','.bmp'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    try:
        return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
    except OSError:
        return ImageFont.load_default(size=size)


def load_preview(json_path, confirmed_empty=()):
    path=Path(json_path)
    require(not path.is_symlink(),'UNSAFE_JSON_PATH','Symlink JSON is not supported.')
    doc=_load_labelme_document(path)
    name=doc.get('imagePath')
    require(isinstance(name,str) and name and '/' not in name and '\\' not in name and ':' not in name,
            'UNSAFE_IMAGE_PATH','imagePath must be a local basename.')
    image_path=path.parent/name
    require(image_path.stem==path.stem and image_path.suffix.lower() in IMAGE_EXTENSIONS,
            'IMAGE_PAIR_MISMATCH','Image and JSON stems must match.')
    require(image_path.is_file(),'MISSING_IMAGE',f'Missing local image: {name}')
    require(not image_path.is_symlink() and image_path.resolve().parent==path.parent.resolve(),
            'UNSAFE_IMAGE_PATH','Image must remain in the input directory.')
    with Image.open(image_path) as raw:
        raw.load()
        original=raw.size
        orientation=raw.getexif().get(274,1)
        require(type(orientation) is int and orientation in range(1,9),
                'INVALID_EXIF','Invalid EXIF orientation; no transform will be guessed.')
        display=ImageOps.exif_transpose(raw).convert('RGB')
    width,height=display.size
    _dimensions(doc.get('imageWidth'),doc.get('imageHeight'))
    require((doc['imageWidth'],doc['imageHeight'])==(width,height),'DIMENSION_MISMATCH',
            f'JSON {doc["imageWidth"]}x{doc["imageHeight"]} does not match EXIF-corrected display {width}x{height}.')
    # Coordinates are parsed and sorted only; no rotation/scale/flip is applied.
    objects=_parse_shapes(doc,width,height,name,set(confirmed_empty))
    match=re.fullmatch(r'(pass|fail)\d+',image_path.stem,re.I)
    status=match.group(1).upper() if match else 'UNKNOWN'
    counts=Counter(obj['class_id'] for obj in objects)
    record=dict(filename=name,image_level_status=status,original_width=original[0],original_height=original[1],
                exif_orientation=orientation,display_width=width,display_height=height,
                json_width=doc['imageWidth'],json_height=doc['imageHeight'],object_count=len(objects),
                **{f'count_{i}':counts[i] for i in range(6)},
                preview_status='CONFIRMED_EMPTY' if not objects else 'PREVIEW_GENERATED',reason='')
    return display,record,objects


def draw_preview(image,objects):
    result=image.copy()
    result.info.clear()  # New raster already has display orientation; do not carry source EXIF forward.
    draw=ImageDraw.Draw(result)
    scale=max(1,min(image.size)/1000)
    stroke=max(2,round(3*scale))
    face=font(max(10,round(16*scale)))
    used=[]
    for obj in objects:
        color=COLORS[obj['class_id']]
        x1,y1,x2,y2=obj['xyxy']
        # Raster endpoints are clipped for display only; source/parsed coordinates stay unchanged.
        bounds=(max(0,round(x1)),max(0,round(y1)),min(image.width-1,round(x2)),min(image.height-1,round(y2)))
        draw.rectangle(bounds,outline=color,width=stroke)
        text=f"{obj['class_id']} {CLASSES[obj['class_id']]}"
        bbox=draw.textbbox((0,0),text,font=face)
        tw,th=bbox[2]-bbox[0]+8,bbox[3]-bbox[1]+8
        candidates=[(x1,y1-th-3),(x1,y2+3),(x2+3,y1),(x1+3,y1+3)]
        candidates=[(max(0,min(int(x),image.width-tw)),max(0,min(int(y),image.height-th))) for x,y in candidates]
        def overlap(candidate):
            x,y=candidate
            boxes=[o['xyxy'] for o in objects]+used
            return sum(max(0,min(x+tw,b[2])-max(x,b[0]))*max(0,min(y+th,b[3])-max(y,b[1])) for b in boxes)
        tx,ty=min(candidates,key=overlap)
        label_box=(tx,ty,tx+tw,ty+th)
        draw.rectangle(label_box,fill='white',outline=color,width=1)
        draw.text((tx+4,ty+4-bbox[1]),text,font=face,fill=color)
        used.append(label_box)
    return result


def _sheets(out,records):
    directory=out/'contact_sheets'; directory.mkdir()
    for group in ('PASS','FAIL','UNKNOWN'):
        rows=[r for r in records if r['image_level_status']==group]
        for offset in range(0,len(rows),16):
            page=Image.new('RGB',(2000,2520),'#e9edf2')
            draw=ImageDraw.Draw(page)
            draw.text((25,18),f'{group} | HUMAN LABELME ANNOTATIONS | page {offset//16+1}',font=font(28),fill='#162335')
            draw.text((25,58),'Not YOLO predictions. Counts are annotations, not expected quantities.',font=font(20),fill='#394a60')
            for index,row in enumerate(rows[offset:offset+16]):
                x=(index%4)*500+10; y=(index//4)*600+110
                draw.rectangle((x,y,x+480,y+580),fill='white')
                draw.text((x+10,y+8),f'{row["filename"]} | {group}',font=font(23),fill='#102238')
                if row['preview_status'] in {'PREVIEW_GENERATED','CONFIRMED_EMPTY'}:
                    with Image.open(out/'individual'/(Path(row['filename']).stem+'.png')) as im:
                        thumb=ImageOps.contain(im,(460,455))
                        page.paste(thumb,(x+10+(460-thumb.width)//2,y+45+(455-thumb.height)//2))
                else:
                    draw.text((x+15,y+220),'REFUSED: see manifest',font=font(20),fill='red')
                draw.text((x+10,y+510),' / '.join(f'{i}:{row.get(f"count_{i}","?")}' for i in range(6)),font=font(20),fill='#172c43')
                if row['preview_status']=='CONFIRMED_EMPTY':
                    draw.text((x+10,y+545),'CONFIRMED_EMPTY',font=font(21),fill='#17653c')
            page.save(directory/f'{group}-{offset//16+1:02}.jpg',quality=94,subsampling=0)


def generate(input_dir,output_dir,confirmed_empty=()):
    source=Path(input_dir).resolve(); out=Path(output_dir).resolve()
    require(source!=out and source not in out.parents and out not in source.parents,
            'UNSAFE_OUTPUT','Source/output directories must not contain each other.')
    require(source.is_dir(),'MISSING_INPUT','Input directory does not exist.')
    if out.exists(): raise FileExistsError(f'Refusing existing output directory: {out}')
    entries=sorted(source.iterdir())  # Flat input only. Never search other real-data directories.
    require(all(p.is_file() and not p.is_symlink() and p.resolve().parent==source for p in entries),
            'UNSAFE_INPUT','Use a flat directory of regular local files only.')
    images=[p for p in entries if p.suffix.lower() in IMAGE_EXTENSIONS]
    before={p.name:digest(p) for p in entries}
    out.mkdir(parents=True,exist_ok=False)
    individual=out/'individual'; individual.mkdir()
    rows=[]; errors=[]; id0=[]
    for path in images:
        row=dict(filename=path.name,image_level_status='UNKNOWN',original_width='',original_height='',
                 exif_orientation='',display_width='',display_height='',json_width='',json_height='',
                 object_count='',**{f'count_{i}':'' for i in range(6)},preview_status='REFUSED',reason='')
        match=re.fullmatch(r'(pass|fail)\d+',path.stem,re.I)
        if match: row['image_level_status']=match.group(1).upper()
        try:
            display,row,objects=load_preview(path.with_suffix('.json'),confirmed_empty)
            require(row['filename']==path.name,'IMAGE_PAIR_MISMATCH','JSON refers to a different paired image.')
            result=draw_preview(display,objects)
            result.save(individual/(path.stem+'.png'),format='PNG',compress_level=3)
            for obj in objects:
                if obj['class_id']==0:
                    id0.append(dict(filename=path.name,shape_index=obj['object_id'],xyxy=obj['xyxy']))
        except (DataError,OSError,ValueError) as exc:
            reason=f'{getattr(exc,"code",type(exc).__name__)}: {exc}'
            row.update(preview_status='REFUSED',reason=reason)
            errors.append(f'{path.name}: {reason}')
        rows.append(row)
        if len(rows)%10==0:
            print(f'Previewed {len(rows)}/{len(images)} images',flush=True)
    require(before=={p.name:digest(p) for p in entries},'SOURCE_CHANGED','Source hashes changed during rendering.')
    _sheets(out,rows)
    with (out/'preview_manifest.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['filename'])
        writer.writeheader(); writer.writerows(rows)
    success=[r for r in rows if r['preview_status']!='REFUSED']
    lines=['# EXIF 方向与标注框回画核验','',
           f'时间：{datetime.now(timezone(timedelta(hours=8))).isoformat()}。输入：`{source}`。',
           f'图片 {len(images)}；成功 {len(success)}；拒绝 {len(errors)}。全部为人工 Labelme 标注回画，不是 YOLO 预测。',
           '使用 ImageOps.exif_transpose 仅校正内存中的图像；JSON 坐标直接用于校正后的显示图，不做旋转、缩放或翻转。',
           'JSON 必须匹配显示尺寸；不匹配时不回画该图片。现有严格数据转换检查器的 EXIF 校验未关闭。',
           f'成功图片 EXIF 分布：{dict(Counter(r["exif_orientation"] for r in success))}。',
           f'成功图片对象数：{sum(r["object_count"] for r in success)}；几何校验不等于人工确认。',
           '预览 PNG 尺寸与显示尺寸一致，没有缩小；联系图缩略图等比例缩放。',
           f'源文件前后 SHA-256 一致：{len(before)} 个。未修改 JPG/JSON/原始 EXIF。','',
           '| filename | 原始尺寸 | EXIF | 显示尺寸 | JSON尺寸 | status |','|---|---|---:|---|---|---|']
    for r in rows:
        lines.append(f'| {r["filename"]} | {r["original_width"]}×{r["original_height"]} | {r["exif_orientation"]} | {r["display_width"]}×{r["display_height"]} | {r["json_width"]}×{r["json_height"]} | {r["preview_status"]} |')
    lines+=['','## 异常','',*(errors or ['无拒绝文件。']), '']
    (out/'orientation_check.md').write_text('\n'.join(lines),encoding='utf-8')
    review=['# 人工标注复核清单','',
            '所有图像是人工标注回画。程序只验证方向、结构与几何；以下复核均等待人工审核，不自动勾选。',
            '图片级 PASS/FAIL 仅来自文件名前缀，不是重新计算的课程判定。',
            '拍摄条件由用户提供：白色亚克力背景、固定打光/高度、每图重新摆放；本工具不据此自动创建 scene_id 或分区。','',
            '## 三个 ID 0（实际数量以列表为准）','']
    for obj in id0:
        review.append(f'- [ ] {obj["filename"]} / shape {obj["shape_index"]} / xyxy={obj["xyxy"]}：确认是 non_catalogue，未与目标类重复计数。')
    review+=['','## 全部图片逐项复核','',
             '- [ ] 核对六类目录与规格，不能把 FAIL 图里的所有零件当作 ID 0。',
             '- [ ] 核对漏标、错标、重复标注；框完整包围可见零件。',
             '- [ ] 核对临边、反光、遮挡目标；无法判定时反馈 P03。',
             '- [ ] 核对 EXIF 校正后方向和框位置，没有二次旋转 JSON 坐标。',
             '- [ ] fail13.jpg：用户已确认真实空背景，本轮以 CONFIRMED_EMPTY 记录；无框、无 ID 0，仍可复核预览。','']
    for r in rows:
        review.append(f'- [ ] {r["filename"]}：方向/类别/漏标/框范围；状态 {r["preview_status"]}。意见：________')
    (out/'manual_review.md').write_text('\n'.join(review)+'\n',encoding='utf-8')
    require(before=={p.name:digest(p) for p in entries},'SOURCE_CHANGED','Source hashes changed after report generation.')
    return dict(images=len(images),generated=len(success),failed=len(errors),id0_instances=len(id0),
                confirmed_empty=[r['filename'] for r in success if r['preview_status']=='CONFIRMED_EMPTY'],
                orientation_counts=dict(Counter(r['exif_orientation'] for r in success)),
                object_count=sum(r['object_count'] for r in success),output=str(out),source_hashes_unchanged=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('data/p03/development'))
    parser.add_argument('--output',type=Path,default=Path('output/p06/annotation_preview'))
    parser.add_argument('--confirmed-empty',action='append',default=[],metavar='IMAGE_ID')
    args=parser.parse_args(argv)
    try:
        summary=generate(args.input,args.output,args.confirmed_empty)
        print(json.dumps(summary,indent=2))
        return 1 if summary['failed'] else 0
    except (DataError,OSError,ValueError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
