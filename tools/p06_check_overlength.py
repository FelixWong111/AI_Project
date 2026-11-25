"""Train-only diagnostic of three user-confirmed overlength screws and controls."""
import json
from pathlib import Path
import sys
import csv
import numpy as np
import cv2
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.p06_inference import create_backend
from src.p06_foreground import iou
from src.p06_geometry import measure_object,MeasurementFailure
from tools.p06_axial_projection import project
from tools.p06_measure_geometry import SOURCE,catalogue_records,crop_for_object,save
from tools.p06_measure_candidate import load_display,sha256,font

CASES={'fail113.jpg':'fail113_001','fail125.jpg':'fail125_001','fail127.jpg':'fail127_001',
       'pass123.jpg':'pass123_001','pass132.jpg':'pass132_005','pass32.jpg':'pass32_001'}


def main():
    out=ROOT/'output/p06_overlength_train_check'
    if out.exists():raise FileExistsError(out)
    index,objects=catalogue_records('train')
    if not CASES.keys()<=index.keys():raise ValueError('Selected image outside Train')
    scale=json.loads((ROOT/'config/p06_size_calibration_candidate.json').read_bytes())['px_per_mm']
    hashes={name:sha256(SOURCE/'images/train'/name) for name in CASES}
    detector=create_backend(model_id='baseline3',config={})
    out.mkdir(parents=True)
    rows=[];records=[];tiles=[]
    for name,oid in CASES.items():
        gt=next(r for r in objects[name] if r['object_id']==oid)
        image,_=load_display(SOURCE/'images/train'/name)
        bgr=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        observations=detector.infer(bgr,dict.fromkeys('12345',1))
        save(out/(Path(name).stem+'_detections.json'),observations)
        expected=int(gt['class_id'])
        box={k:float(gt[k]) for k in ('xmin','ymin','xmax','ymax')}
        matches=[(iou(o,box),o) for o in observations['objects']]
        overlap,best=max(matches,key=lambda p:p[0]) if matches else (0,None)
        if overlap<.5:best=None
        for source,object_box in [('GT_CROP',gt),('YOLO_CROP',best)]:
            row=dict(filename=name,object_id=oid,gt_class_id=expected,source=source,
                     detected_class_id=best['class_id'] if best else None,
                     detector_iou=overlap,projection_mm=None,underhead_mm=None,reason='')
            if object_box is None:
                row['reason']='NO_IOU_MATCH';rows.append(row);continue
            crop,extent=crop_for_object(image,object_box)
            frame=cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR)
            # User confirmed screw shape. ID1 selects screw geometry only;
            # original GT0 and original detector class remain above.
            measured=measure_object(frame,1,scale)
            row['underhead_mm']=measured['dimensions'].get('underhead_length',{}).get('mm_estimate')
            row['underhead_reason']=measured['reason']
            target=np.array([float(object_box[k]) for k in ('xmin','ymin','xmax','ymax')])-np.array(extent[:2]*2)
            projection=None
            try:
                projection=project(frame,scale,np.r_[np.floor(target[:2]),np.ceil(target[2:])].astype(int))
                row['projection_mm']=projection['dimension']['mm_estimate']
            except MeasurementFailure as e:row['reason']=str(e)
            records.append(dict(**row,crop_xyxy=extent,measurement=measured,projection=projection))
            if source=='GT_CROP':
                tile=crop.copy();draw=ImageDraw.Draw(tile)
                if projection:
                    draw.line([tuple(p) for p in projection['dimension']['line_xy']],fill='#f02283',width=5)
                tile.thumbnail((450,400))
                canvas=Image.new('RGB',(480,460),'white');canvas.paste(tile,((480-tile.width)//2,55))
                d=ImageDraw.Draw(canvas);d.text((5,5),name+' GT'+str(expected),font=font(18),fill='black')
                d.text((5,30),f"whole={row['projection_mm']:.2f}mm" if row['projection_mm'] else row['reason'],font=font(16),fill='black')
                tiles.append(canvas)
            rows.append(row)
    for n,h in hashes.items():
        if sha256(SOURCE/'images/train'/n)!=h:raise ValueError('Source changed')
    save(out/'records.json',records)
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    save(out/'provenance.json',dict(source_sha256=hashes,source_unchanged=True,
        confirmation='User confirmed three selected GT0 objects are longer than catalogue long screw; exact designation unspecified',
        calibration='candidate px/mm, validated=false',scope='Seen Train cases; exact dimensions and unseen generalisation NOT_VERIFIED'))
    sheet=Image.new('RGB',(1440,920),'white')
    for i,t in enumerate(tiles):sheet.paste(t,((i%3)*480,(i//3)*460))
    sheet.save(out/'comparison.jpg',quality=90)
    text=['# 已确认超长螺钉专项检查','',
          '三颗目录外件由用户确认比合法长螺钉更长；未记录为精确 M4×40。对照为 Train 中三颗 ID1。原标注、检测类别、模型及阈值保持原值。','',
          '尺寸诊断按人工确认的螺钉形态调用几何方法。调用 ID1 仅选择螺钉算法，不改变 GT0 或检测类别。','',
          '| 图片 | GT | 路径 | 检测类别 | 整体投影 mm | 头下长度 mm | 投影失败原因 |','|---|---:|---|---|---|---|---|']
    for r in rows:
        fmt=lambda v:f'{v:.2f}' if v is not None else '无'
        text.append(f"| {r['filename']} | {r['gt_class_id']} | {r['source']} | {r['detected_class_id']} | {fmt(r['projection_mm'])} | {fmt(r['underhead_mm'])} | {r['reason']} |")
    text+=['','候选毫米数缺少卡尺真值，整体投影包含头部及姿态影响。此处可评估明显长短差异，不以结果生成产品容差或可信误差界。三颗错误件属于已见 Train，用于验证已有能力。输出文件中的检测结果仍为 REVIEW，最终 ID0→FAIL 链路由后续共享判定接入。']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(rows)


if __name__=='__main__':main()
