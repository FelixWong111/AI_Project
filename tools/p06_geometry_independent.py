"""Train-only independent screw dimensions with an explicit target box."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.p06_geometry import measure_object
from tools.p06_screw_review import OUTPUT,HISTORY,KEYS
from tools.p06_measure_geometry import catalogue_records,SOURCE,crop_for_object,result_global,save
from tools.p06_measure_candidate import load_display,sha256
from tools.p06_edge_audit import overlay
from PIL import Image,ImageDraw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full-train',action='store_true')
    parser.add_argument('--from-records',type=Path,help='Render verified records from the same module version without rerunning measurement')
    args=parser.parse_args()
    dest=args.output.resolve()
    if dest.exists() or not dest.is_relative_to(ROOT/'output'):
        parser.error('Need new output directory')
    queue=json.loads((OUTPUT/'queue.json').read_bytes())
    marks=json.loads((ROOT/'data/p06_screw_marks.json').read_bytes())
    if marks['queue_id']!=queue['queue_id']:raise ValueError('Wrong human queue')
    taskmap={t['object_id']:t for t in queue['tasks']}
    names={t['image_id'] for t in queue['tasks'] if t['role']=='development'}
    before=json.loads((HISTORY/'train_full_final/source_image_sha256.json').read_bytes())
    params=json.loads((HISTORY/'selection.json').read_bytes())['parameters']
    index,objects=catalogue_records('train')
    rows=[]
    frozen=sha256(ROOT/'src/p06_geometry.py')
    if args.from_records:
        parent=args.from_records.resolve().parent
        evidence=json.loads((parent/'summary.json').read_bytes())
        if (evidence['code_sha256']!=frozen or
                evidence['manual_sha256']!=sha256(ROOT/'data/p06_screw_marks.json')):
            raise ValueError('Saved measurements belong to another code or annotation version')
        rows=json.loads(args.from_records.read_bytes())
        if any(r['split']!='train' for r in rows):raise ValueError('Train records required')
        index={}
    for name in sorted(index):
        if not args.full_train and name not in names:continue
        path=SOURCE/'images/train'/name
        if sha256(path)!=before[name]:raise ValueError('Source changed')
        image,_=load_display(path)
        for obj in objects[name]:
            cid=int(obj['class_id'])
            if cid not in (1,2,3):continue
            oid=obj['object_id']
            if not args.full_train and oid not in taskmap:continue
            crop,box=crop_for_object(image,obj)
            target=[float(obj[k])-box[i%2] for i,k in enumerate(('xmin','ymin','xmax','ymax'))]
            if image.size!=(3024,4032):
                result=dict(dimensions={},reason='SCALE_TRANSFER_UNCONFIRMED_RESOLUTION',diagnostics={},contours_xy=[],status='UNMEASURABLE')
            else:
                result=measure_object(cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR),cid,
                                      queue['scale_px_per_mm'],params,screw_method='independent',target_box=target)
            result=result_global(result,box[:2])
            row=dict(object_id=oid,class_id=cid,image_id=name,split='train',crop_xyxy=box,
                     role=taskmap.get(oid,{}).get('role','other_train'),result=result)
            if oid in taskmap:
                errors={}
                for key,dim in result['dimensions'].items():
                    points=np.asarray(marks['marks'][oid][key],float)
                    errors[key]=dict(difference_mm=float(dim['mm_estimate']-np.linalg.norm((points[1]-points[0])/queue['scale_px_per_mm'])),
                                     endpoint_offsets_px=np.linalg.norm(np.asarray(dim['line_xy'])-points,axis=1).tolist())
                row['human_comparison']=errors
            rows.append(row)
        if sha256(path)!=before[name]:raise ValueError('Source changed')
    if sha256(ROOT/'src/p06_geometry.py')!=frozen:raise ValueError('Code changed during run')
    summary={}
    for cid in (1,2,3):
        group=[r for r in rows if r['class_id']==cid]
        summary[cid]=dict(total=len(group),**{k:sum(k in r['result']['dimensions'] for r in group) for k in KEYS},
                          both=sum(all(k in r['result']['dimensions'] for k in KEYS) for r in group),
                          reasons=dict(Counter(r['result']['reason'] for r in group)))
    manual_summary={}
    for role in ('development','review'):
        manual_summary[role]={}
        for key in KEYS:
            values=[r['human_comparison'][key]['difference_mm'] for r in rows if r['role']==role and key in r.get('human_comparison',{})]
            manual_summary[role][key]=dict(n=len(values),mean_absolute_mm=float(np.mean(np.abs(values))) if values else None)
    dest.mkdir(parents=True)
    save(dest/'records.json',rows)
    save(dest/'summary.json',dict(coverage=summary,code_sha256=frozen,scale_px_per_mm=queue['scale_px_per_mm'],
                                  manual_sha256=sha256(ROOT/'data/p06_screw_marks.json'),
                                  manual_comparison=manual_summary,
                                  scope='Train only; human image lines; no caliper accuracy'))
    for row in rows:
        oid=row['object_id']
        if oid not in taskmap:continue
        task=taskmap[oid]
        crop=Image.open(OUTPUT/task['src']).convert('RGB')
        local=json.loads(json.dumps(row['result']))
        origin=np.asarray(task['crop_xyxy'][:2])
        for dim in local['dimensions'].values():
            dim['line_xy']=(np.asarray(dim['line_xy'])-origin).tolist()
        tile=overlay(crop,local,oid,'independent')
        human=crop.copy()
        draw=ImageDraw.Draw(human)
        for key,line in marks['marks'][oid].items():
            if line:draw.line([tuple(p-origin) for p in np.asarray(line)],fill='#00d070',width=4)
        human.thumbnail((580,430))
        out=Image.new('RGB',(1200,490),'white')
        out.paste(human,((600-human.width)//2,50))
        out.paste(tile,(600,0))
        ImageDraw.Draw(out).text((10,8),oid+' human / '+row['role'],fill='black')
        out.save(dest/(oid+'.jpg'),quality=88)
    report=['# 目标隔离与独立尺寸实验','',
            '本轮保留为显式选择的 independent 实验模式，默认测量入口维持上一轮行为。原始标注框用于指定待测目标；目标连通区域明显超出框时拒测，排除其他独立连通区域。杆长与外径分别运行，失败时保留另一项已得到的候选值。','',
            '## 覆盖率','',
            '| 类别 | 总数 | 杆长 | 外径 | 两项同时 |','|---|---:|---:|---:|---:|']
    for cid,s in summary.items():
        n=s['total']
        report.append(f"| {cid} | {n} | {s[KEYS[0]]} ({s[KEYS[0]]/n:.1%}) | {s[KEYS[1]]} ({s[KEYS[1]]/n:.1%}) | {s['both']} ({s['both']/n:.1%}) |")
    report+=['','## 人工线对照','',
             '先在 18 颗开发对象检查三次候选修改，再固定测量模块处理完整 Train。六颗留后复核对象未用于本轮规则调整。均为图像点选参照，尚缺卡尺误差验证。','',
             '| 分组 | 尺寸 | 可比较数量 | 平均绝对差 mm |','|---|---|---:|---:|']
    for role,fields in manual_summary.items():
        for key,s in fields.items():
            report.append(f"| {role} | {key} | {s['n']} | {s['mean_absolute_mm']} |")
    report+=['','pass55_001 的污染裁剪在开发检查时拒测，避免继续输出偏离目标的轴线。独立尺寸返回允许外径存在而杆长失败，但候选外径仍可能落在反光边缘。严格头下边界与杆尖检查导致杆长覆盖较低。','',
             '每颗人工样本有独立 JPG 叠图，左侧绿色为人工线，右侧为本轮候选。records.json 逐尺寸记录状态、失败原因和原图坐标；summary.json 包含测量模块与人工 JSON 的哈希。','',
             '本轮仅使用 Train。当前数据不足以证明这些候选满足产品容差，保持 REVIEW 用途。垫圈算法本轮未调整。']
    (dest/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(summary)


if __name__=='__main__':main()
