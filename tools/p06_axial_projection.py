"""Bounded Train analysis of length refusals and whole-screw axial projection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import csv
import hashlib

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.p06_geometry import DEFAULT_PARAMETERS,dimension,isolate_target,silhouette,MeasurementFailure
from tools.p06_measure_geometry import SOURCE,catalogue_records
from tools.p06_measure_candidate import load_display,sha256,font
from tools.p06_screw_review import OUTPUT


def project(image,scale,box,offset=35,reference_axis=None):
    """Visible contour span along filled-silhouette PCA. No expected dimensions."""
    scale=np.asarray(scale,float)
    require_shape=np.asarray(image).shape
    if len(require_shape)!=3 or require_shape[2]!=3:
        raise ValueError('BGR image required')
    if scale.shape!=(2,) or not np.isfinite(scale).all() or min(scale)<=0:
        raise ValueError('Positive finite x/y scales required')
    params={**DEFAULT_PARAMETERS,'background_offset':offset}
    isolated=isolate_target(image,box,params)
    contour,details=silhouette(cv2.cvtColor(isolated,cv2.COLOR_BGR2GRAY),params)
    mask=np.zeros(image.shape[:2],np.uint8)
    cv2.drawContours(mask,[contour],-1,1,-1)
    y,x=np.nonzero(mask)
    points=np.c_[x,y]/scale
    center=points.mean(axis=0)
    values,vectors=np.linalg.eigh(np.cov((points-center).T))
    axis=vectors[:,-1]
    if reference_axis is not None:
        axis=np.asarray(reference_axis,float)/scale
        if not np.isfinite(axis).all() or np.linalg.norm(axis)==0:
            raise ValueError('Nonzero finite reference axis required')
        axis/=np.linalg.norm(axis)
    boundary=contour[:,0]/scale
    u=(boundary-center)@axis
    lo,hi=int(u.argmin()),int(u.argmax())
    a=(center+axis*u[lo])*scale
    b=(center+axis*u[hi])*scale
    return dict(dimension=dimension(a,b,scale),contour_xy=contour[:,0].tolist(),
                witness_xy=(boundary[[lo,hi]]*scale).tolist(),
                eigenvalue_ratio=float(values[-1]/max(values[0],1e-12)),
                contour_sha256=hashlib.sha256(contour.tobytes()).hexdigest(),
                segmentation=details,
                axis_mm=axis.tolist())


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/p06_axial_projection')
    args=parser.parse_args()
    dest=args.output.resolve()
    if dest.exists() or not dest.is_relative_to(ROOT/'output'):
        parser.error('Choose new output directory')
    q=json.loads((OUTPUT/'queue.json').read_bytes())
    manual=json.loads((ROOT/'data/p06_screw_marks.json').read_bytes())
    if manual['queue_id']!=q['queue_id']:raise ValueError('Manual queue mismatch')
    independent_path=ROOT/'output/p06_geometry_independent_review/records.json'
    independent=json.loads(independent_path.read_bytes())
    baseline={r['object_id']:r for r in json.loads((OUTPUT/'comparison.json').read_bytes())}
    index,objects=catalogue_records('train')
    objmap={r['object_id']:r for name in index for r in objects[name]}
    # Attribute the first actual length refusal. Diameter errors do not enter.
    reasons=Counter()
    transitions=Counter()
    lost_reasons=Counter()
    for row in independent:
        result=row['result'];old=baseline[row['object_id']]['body_profile']
        state=result.get('dimension_status',{}).get('underhead_length',{})
        reason=state.get('reason',result['reason'])
        if 'underhead_length' not in result['dimensions']:reasons[reason]+=1
        transitions[(bool(old['dimensions'].get('underhead_length')),bool(result['dimensions'].get('underhead_length')))]+=1
        if 'underhead_length' in old['dimensions'] and 'underhead_length' not in result['dimensions']:
            lost_reasons[reason]+=1
    endpoint_rows=[]
    for task in q['tasks']:
        oid=task['object_id'];new=next(r for r in independent if r['object_id']==oid)
        gt=np.asarray(manual['marks'][oid]['underhead_length'],float)
        for phase,result in [('body_profile',baseline[oid]['body_profile']),('independent',new['result'])]:
            dim=result['dimensions'].get('underhead_length')
            row=dict(object_id=oid,class_id=task['class_id'],role=task['role'],phase=phase,candidate=dim is not None,
                     reason=result.get('dimension_status',{}).get('underhead_length',{}).get('reason',result['reason']))
            if dim:
                line=np.asarray(dim['line_xy'])
                row.update(difference_mm=float(dim['mm_estimate']-np.linalg.norm((gt[1]-gt[0])/q['scale_px_per_mm'])),
                           shoulder_offset_px=float(np.linalg.norm(line[0]-gt[0])),tip_offset_px=float(np.linalg.norm(line[1]-gt[1])))
            endpoint_rows.append(row)
    results=[];pictures=[]
    for task in q['tasks']:
        oid=task['object_id'];path=SOURCE/'images/train'/task['image_id']
        if task['image_id'] not in index or sha256(path)!=task['image_sha256']:
            raise ValueError('Source mismatch')
        image,_=load_display(path);crop=image.crop(task['crop_xyxy']);origin=np.asarray(task['crop_xyxy'][:2])
        obj=objmap[oid]
        box=np.r_[np.floor([float(obj['xmin']),float(obj['ymin'])])-origin,
                  np.ceil([float(obj['xmax']),float(obj['ymax'])])-origin].astype(int)
        bgr=cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR)
        gt=np.asarray(manual['marks'][oid]['underhead_length'],float)
        row=dict(object_id=oid,class_id=task['class_id'],role=task['role'],variants={})
        display=crop.copy();draw=ImageDraw.Draw(display)
        draw.line([tuple(p-origin) for p in gt],fill='#00bd65',width=4)
        for offset in (25,35,45):
            try:
                candidate=project(bgr,q['scale_px_per_mm'],box,offset)
                manual_axis=project(bgr,q['scale_px_per_mm'],box,offset,gt[1]-gt[0])
                candidate['manual_axis_span_mm']=manual_axis['dimension']['mm_estimate']
                candidate['axis_angle_degrees']=float(np.degrees(np.arccos(np.clip(abs(np.dot(candidate['axis_mm'],manual_axis['axis_mm'])),0,1))))
                row['variants'][str(offset)]=candidate
                if offset==35:
                    draw.line([tuple(p) for p in candidate['dimension']['line_xy']],fill='#ed168d',width=4)
                    for x,y in candidate['witness_xy']:
                        draw.ellipse((x-5,y-5,x+5,y+5),fill='#ffcc00')
            except MeasurementFailure as e:row['variants'][str(offset)]={'reason':str(e)}
        available=[v['dimension']['mm_estimate'] for v in row['variants'].values() if 'dimension' in v]
        if available:row['offset_span_mm']=max(available)-min(available)
        results.append(row)
        display.thumbnail((470,390))
        tile=Image.new('RGB',(500,450),'white');tile.paste(display,((500-display.width)//2,50))
        d=ImageDraw.Draw(tile);d.text((6,6),oid+' green=underhead pink=whole',font=font(15),fill='black')
        v=row['variants']['35']
        text=f"{v['dimension']['mm_estimate']:.2f} mm candidate" if 'dimension' in v else v['reason']
        d.text((6,28),text,font=font(13),fill='black')
        pictures.append(tile)
    dest.mkdir(parents=True)
    save=lambda path,data:path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    save(dest/'projection_records.json',results)
    write_csv(dest/'endpoint_comparison.csv',endpoint_rows)
    save(dest/'failure_summary.json',dict(failed_lengths=sum(reasons.values()),reasons=dict(reasons),
                                         lost_from_default=dict(lost_reasons),
                                         transitions={str(k):v for k,v in transitions.items()}))
    for page in range(3):
        sheet=Image.new('RGB',(2000,900),'white')
        for i,tile in enumerate(pictures[page*8:(page+1)*8]):sheet.paste(tile,((i%4)*500,(i//4)*450))
        sheet.save(dest/f'projection_sheet_{page+1}.jpg',quality=90)
    lines=['# 杆长拒测归因与整体投影长度实验','',
           '仅使用 Train 的既有记录和人工标记。投影实验限定 24 颗，每类 8 颗；多零件隔离方法沿用上一轮。整体投影指可见轮廓在估计轴线上的跨度，包含头部及三维姿态影响。','',
           '## 杆长失败归因','',
           '| 首个杆长失败原因 | 数量 |','|---|---:|']
    lines += [f'| {reason} | {n} |' for reason,n in reasons.most_common()]
    lines+=['',f'全部 738 颗中杆长失败 {sum(reasons.values())} 颗。原因按每个尺寸的实际失败记录统计；同一对象后续未执行条件不能据此估计。','',
            '旧默认有杆长 / 新版有杆长的转移计数如下。共同成功与新增成功应与损失分开看。',str(dict(transitions)),'',
            '旧默认有输出、本轮拒测的 362 个对象，首个失败条件分布为 '+str(dict(lost_reasons))+'。这是程序执行位置的归因，没有单独移除条件重跑，不能当作每个条件的独立因果效应。','',
            '## 同一人工样本','',
            '| 分组 | 版本 | 有杆长/样本数 | 平均绝对尺寸差 mm | 起点偏移中位数 px | 杆尖偏移中位数 px |',
            '|---|---|---:|---:|---:|---:|']
    for role in ('development','review','all'):
        for phase in ('body_profile','independent'):
            group=[r for r in endpoint_rows if r['phase']==phase and (role=='all' or r['role']==role)]
            valid=[r for r in group if r['candidate']]
            values=[str(round(float(f([abs(r[k]) for r in valid])),3)) if valid else '无候选'
                    for k,f in [('difference_mm',np.mean),('shoulder_offset_px',np.median),('tip_offset_px',np.median)]]
            lines.append(f"| {role} | {phase} | {len(valid)}/{len(group)} | {' | '.join(values)} |")
    lines+=['','## 整体投影实验','',
            '| ID | 默认阈值有输出/8 | 长度范围 mm | 三阈值均成功的数量 | 阈值引起的跨度中位数 mm |',
            '|---|---:|---|---:|---:|']
    for cid in (1,2,3):
        group=[r for r in results if r['class_id']==cid]
        vals=[r['variants']['35']['dimension']['mm_estimate'] for r in group if 'dimension' in r['variants']['35']]
        stable=[r['offset_span_mm'] for r in group if all('dimension' in v for v in r['variants'].values())]
        lines.append(f"| {cid} | {len(vals)}/8 | {f'{min(vals):.2f}–{max(vals):.2f}' if vals else '无'} | {len(stable)} | {np.median(stable) if stable else '无'} |")
    common=sum(all(any(r['object_id']==t['object_id'] and r['phase']==p and r['candidate'] for r in endpoint_rows)
                   for p in ('body_profile','independent')) for t in q['tasks'])
    angles=[r['variants']['35']['axis_angle_degrees'] for r in results if 'dimension' in r['variants']['35']]
    same=sum(len({v['contour_sha256'] for v in r['variants'].values() if 'dimension' in v})==1
             for r in results if all('dimension' in v for v in r['variants'].values()))
    lines+=['',f'24 颗人工样本中，两版同时有杆长的数量为 {common}。两版的总体偏差均值不具备同对象配对精度比较的条件。','',
            f'自动 PCA 轴与人工杆轴夹角中位数 {np.median(angles):.2f} 度，最大 {max(angles):.2f} 度。三阈值均成功的样本中，{same} 个轮廓完全相同，阈值跨度为零缺少有效扰动，无法证明鲁棒性。','',
            '头部宽、杆身短时，整颗轮廓的主轴可能偏离杆轴；pass23_003 可见阴影和头部拉偏投影线。此实验暂适合筛查长短形态的可行性，尚未形成对未知规格的拒识阈值。']
    lines+=['','绿色为人工杆长，粉色为整体投影线，黄色为轮廓极值支持点。两种长度的定义不同，不能将其差值当作测量误差。manual_axis_span_mm 用人工杆轴做敏感性对照，人工轴并非自动算法输入。三阈值用于检查分割敏感性，未依据标称值选择阈值。','',
            '相同的阴影、轮廓污染和姿态可能同时影响三种阈值。稳定输出仍需人工整体投影端点验证。可先作为长短形态的候选辅助特征，不能据本实验建立未知错误规格的可靠拒识率。头部厚度未用于加减或恢复真实杆长。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    overall=dict(q,purpose='manual_whole_axial_projection')
    overall['tasks']=[dict(t,src='../p06_screw_review/'+t['src']) for t in q['tasks']]
    overall['queue_id']=hashlib.sha256(json.dumps(overall,sort_keys=True).encode()).hexdigest()
    save(dest/'overall_queue.json',overall)
    template=(ROOT/'tools/p06_screw_review.html').read_text(encoding='utf-8')
    template=template.replace("const keys = ['underhead_length','thread_major_diameter'];","const keys = ['overall_projection'];")
    template=template.replace("const titles = ['头下长度','螺纹外径'];","const titles = ['整体轴向投影'];")
    template=template.replace("index=i;field=complete(entry()[keys[0]])&&!complete(entry()[keys[1]])?1:0;","index=i;field=0;")
    start=template.index("  $('hint').textContent=")
    end=template.index("  $('length').setAttribute",start)
    hint="沿可见杆轴方向，点两端极值平面与轴线的交点：一端为包含头部的最外端，另一端为杆尖。排除阴影。可先放大；边界或轴线不清楚时选看不清。"
    template=template[:start]+"  $('hint').textContent="+json.dumps(hint,ensure_ascii=False)+";\n"+template[end:]
    template=template.replace('<button id="diameter">','<button id="diameter" hidden>')
    template=template.replace('① 头下长度','整体轴向投影')
    template=template.replace('螺钉测量线标记','整体轴向投影标记')
    template=template.replace("a.download='p06_screw_marks.json'","a.download='p06_overall_projection_marks.json'")
    template=template.replace('长度与外径可分别标为看不清','整体投影可标为看不清')
    template=template.replace('/* QUEUE */null',json.dumps(overall).replace('<',r'\u003c'))
    (dest/'mark_overall.html').write_text(template,encoding='utf-8')
    save(dest/'provenance.json',dict(script_sha256=sha256(__file__),geometry_sha256=sha256(ROOT/'src/p06_geometry.py'),
                                    manual_sha256=sha256(ROOT/'data/p06_screw_marks.json'),independent_sha256=sha256(independent_path),
                                    scope='24 Train projections, 738 saved Train results; Validation NOT_RUN; physical accuracy NOT_RUN'))
    print(dict(reasons));print(dest)


if __name__=='__main__':main()
