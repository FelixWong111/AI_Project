"""Analyse exported Train line annotations and per-dimension historical coverage."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.p06_screw_review import OUTPUT, HISTORY, KEYS, PHASES, compare_marks
from tools.p06_measure_candidate import sha256


def statistics(values):
    a = np.asarray(values,float)
    if not len(a):
        return dict(n=0)
    return dict(n=len(a),bias=float(a.mean()),median_absolute=float(np.median(abs(a))),
                mean_absolute=float(abs(a).mean()),p90_absolute=float(np.quantile(abs(a),.9)))


def coverage(rows):
    totals = {}
    for cid in (1,2,3,5):
        group = [r for r in rows if r['class_id']==cid]
        keys = KEYS if cid!=5 else ('outer_diameter','inner_diameter')
        totals[str(cid)] = dict(total=len(group))
        for phase in ('before','after','body_profile','edge_audit'):
            if phase not in group[0]:
                continue
            counts = {key:sum(key in r[phase]['dimensions'] for r in group) for key in keys}
            counts['both'] = sum(all(key in r[phase]['dimensions'] for key in keys) for r in group)
            counts['partial_review_only'] = sum(bool(r[phase].get('diagnostics',{}).get('partial_dimensions')) for r in group)
            totals[str(cid)][phase] = counts
    return totals


def write_csv(path,rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def analyse(marks_path,output):
    if output.exists() or not output.resolve().is_relative_to(ROOT/'output'):
        raise ValueError('Choose a new output directory under output/')
    queue=json.loads((OUTPUT/'queue.json').read_bytes())
    manual=json.loads(marks_path.read_bytes())
    rows=json.loads((OUTPUT/'comparison.json').read_bytes())
    audit_path=ROOT/'output/p06_edge_audit_final/records.json'
    audited={r['object_id']:r for r in json.loads(audit_path.read_bytes())}
    historical=HISTORY/'train_full_final/records.json'
    history=json.loads(historical.read_bytes())
    screws={r['object_id']:r for r in rows}
    for row in rows:
        other=audited[row['object_id']]
        if any(row[k]!=other[k] for k in ('image_id','class_id','crop_xyxy')) or row['body_profile']!=other['previous']:
            raise ValueError('Inconsistent reference cohort')
        row['edge_audit']=other['edge_audit']
    compared=compare_marks(queue,rows,manual)
    flat=[]
    for result in compared:
        oid=result['object_id']
        for phase in (*PHASES,'edge_audit'):
            flat.append(dict(object_id=oid,class_id=screws[oid]['class_id'],role=result['role'],
                             dimension=result['dimension'],human=result['human'],phase=phase,**result[phase]))
    scale=np.asarray(queue['scale_px_per_mm'])
    human_rows=[]
    for task in queue['tasks']:
        fields=manual['marks'].get(task['object_id'],{})
        result=dict(object_id=task['object_id'],class_id=task['class_id'],role=task['role'])
        lines=[]
        for key in KEYS:
            value=fields.get(key)
            if value is not None and len(value)==2:
                line=np.asarray(value,float)
                vector=line[1]-line[0]
                result[key+'_pixels']=float(np.linalg.norm(vector))
                result[key+'_candidate_mm']=float(np.linalg.norm(vector/scale))
                lines.append(vector)
        if len(lines)==2:
            result['axes_angle_degrees']=float(np.degrees(np.arccos(np.clip(abs(np.dot(*lines))/(np.linalg.norm(lines[0])*np.linalg.norm(lines[1])),0,1))))
        human_rows.append(result)
    stats=[]
    matched=[]
    for key in KEYS:
        selected=[r for r in flat if r['dimension']==key and r['human']=='marked']
        old={r['object_id']:r['length_difference_candidate_mm'] for r in selected if r['phase']=='before' and r['candidate']}
        new={r['object_id']:r['length_difference_candidate_mm'] for r in selected if r['phase']=='body_profile' and r['candidate']}
        common=sorted(old.keys() & new.keys())
        matched.append(dict(dimension=key,object_ids=common,before=statistics([old[k] for k in common]),
                            body_profile=statistics([new[k] for k in common])))
    for role in ('all','development','review'):
        for cid in (0,1,2,3):
            for key in KEYS:
                for phase in (*PHASES,'edge_audit'):
                    group=[r for r in flat if r['dimension']==key and r['phase']==phase and
                           (cid==0 or r['class_id']==cid) and (role=='all' or r['role']==role)]
                    stat=statistics([r['length_difference_candidate_mm'] for r in group if 'length_difference_candidate_mm' in r])
                    stats.append(dict(role=role,class_id=cid,dimension=key,phase=phase,objects=len(group),
                                      human_marked=sum(r['human']=='marked' for r in group),
                                      algorithm_candidates=sum(r['candidate'] for r in group),**stat))
    combined=[]
    for row in history:
        copy=dict(row)
        if row['class_id'] in (1,2,3):
            copy['body_profile']=screws[row['object_id']]['body_profile']
        copy['edge_audit']=audited[row['object_id']]['edge_audit']
        combined.append(copy)
    counts=coverage(combined)
    states={key:dict(Counter(r['human'] for r in compared if r['dimension']==key)) for key in KEYS}
    output.mkdir(parents=True)
    write_csv(output/'per_object_comparison.csv',flat)
    write_csv(output/'human_lines.csv',human_rows)
    write_csv(output/'comparison_summary.csv',stats)
    evidence=dict(annotation_sha256=sha256(marks_path),queue_sha256=sha256(OUTPUT/'queue.json'),
                  historical_sha256=sha256(historical),audit_sha256=sha256(audit_path),
                  human_states=states,coverage=counts,comparison_summary=stats,matched_comparison=matched)
    (output/'analysis.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    report=['# 人工测量线分析与分尺寸覆盖率','',
            '24 颗来自 Train，每类 8 颗。18 颗为开发组，6 颗为留后复核组。队列身份、坐标范围、有限性及端点有效性已核对。误差使用相同候选尺度换算，参照为人工图像点选；实物卡尺准确度仍待验证。','',
            '## 人工记录','',str(states),'',
            '## 与人工线的尺寸差异','',
            '以下汇总 24 颗。详细 CSV 单列开发组与留后复核组。各版本可比较的对象数量不同，应结合逐对象表检查。','',
            '| 版本 | 尺寸 | 可比较数量 | 平均有符号差 mm | 绝对差中位数 mm | 平均绝对差 mm | 绝对差 P90 mm |',
            '|---|---|---:|---:|---:|---:|---:|']
    for s in stats:
        if s['role']=='all' and s['class_id']==0 and s['n']:
            report.append(f"| {s['phase']} | {s['dimension']} | {s['n']} | {s['bias']:.3f} | {s['median_absolute']:.3f} | {s['mean_absolute']:.3f} | {s['p90_absolute']:.3f} |")
    report+=['','## 同对象对照','',
             '为了消除版本间可测对象不同的影响，仅比较原版与本轮默认候选均有输出的对象。']
    for m in matched:
        report.append(f"- {m['dimension']}，共同 {len(m['object_ids'])} 颗，平均绝对差原版 {m['before']['mean_absolute']:.3f} mm，本轮候选 {m['body_profile']['mean_absolute']:.3f} mm。")
    report+=['','共同对象的杆长没有改善，外径的差异增大。本轮候选暂不具备替代原版的证据。严格核验仅有两颗可比较，不能据此证明其整体精度。','',
             '## 错误线索','',
             '人工标记与算法叠图显示，pass55_001 的裁剪含相邻另一颗螺钉，算法轴线明显偏离目标。这属于目标隔离与轴线估计问题，应先处理污染裁剪。pass123_001 的算法头下起点沿杆身偏移，杆尖较接近人工点，说明长度损失主要来自起点。pass147_003、pass50_002 与 fail46_006 的算法直径线偏宽，应继续检查阴影、透视以及不同牙顶截面。','',
             '后续先按原目标框约束连通域，再估计杆轴。每个尺寸单独记录成功、失败及原因。固定这 18 颗开发案例后，用剩余 6 颗复核；这批均为 Train，仍需后续固定参数的 Validation 检查。']
    report+=['','## Train 分尺寸候选覆盖率','',
             '分母包含原队列内全部目标。partial_review_only 单列，未计入可用尺寸。历史实现遇到任一尺寸失败时常清空整个 dimensions，因此现有记录的单项计数可能相同；改统计方式无法恢复当时未保存的独立结果。','',
             '| 类别 | 版本 | 总目标 | 杆长或外径 | 螺纹外径或内径 | 两项同时 | 待复核局部候选 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for cid,values in counts.items():
        keys=KEYS if cid!='5' else ('outer_diameter','inner_diameter')
        for phase,c in values.items():
            if phase=='total':continue
            n=values['total']
            report.append(f"| {cid} | {phase} | {n} | {c[keys[0]]}/{n} ({c[keys[0]]/n:.1%}) | {c[keys[1]]}/{n} ({c[keys[1]]/n:.1%}) | {c['both']}/{n} ({c['both']/n:.1%}) | {c['partial_review_only']} |")
    report+=['','## 使用边界','',
             '人工线长与算法线长相近时，端点仍可能位于不同位置。杆长 CSV 同时包含头下起点和杆尖偏移；直径截线可能位于不同牙顶截面。人工测量线之间的夹角保存在 human_lines.csv，接近 90 度支持轴向与径向的一致性。','',
             '下一步按每个尺寸独立记录结果及失败原因，保留已有可靠尺寸。错误规格排除需要先确认该尺寸误差界，再与所有合法规格区间比较。当前单幅图候选尺度与人工点选差异尚不足以建立最终拒识规则。']
    (output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(states));print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--marks',type=Path,default=ROOT/'data/p06_screw_marks.json')
    parser.add_argument('--output',type=Path,default=ROOT/'output/p06_manual_geometry_analysis')
    args=parser.parse_args()
    analyse(args.marks,args.output)
