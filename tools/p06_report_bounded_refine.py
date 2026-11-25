"""Summarise paired manual-line differences for the bounded Train experiment."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.p06_measure_candidate import sha256


def run():
    development=ROOT/'output/p06_bounded_refine_dev2'
    review=ROOT/'output/p06_bounded_refine_review'
    dev_freeze=json.loads((development/'freeze.json').read_bytes())
    if dev_freeze!=json.loads((review/'freeze.json').read_bytes()):
        raise ValueError('Review used different measurement code')
    if dev_freeze!={ 'geometry_sha256':sha256(ROOT/'src/p06_geometry.py'),
                    'refinement_sha256':sha256(ROOT/'src/p06_screw_edges.py')}:
        raise ValueError('Code changed since experiment')
    text=['# 头下起点与牙顶定位：有界实验','',
          '先在既有 18 颗 Train 开发样本运行两次候选，再冻结代码，检查 6 颗 Train 留后复核样本。默认算法未替换。原版为阶段 1 之前的 before，上一轮默认为 body_profile。','',
          '候选保留最初的杆身强梯度种子，遇到宽阴影过渡时优先寻找窄边缘，并选择靠近种子宽度的真实截线。头下起点采用首个持续的宽度收缩位置；两条外露轨迹同意时采用其边缘位置。未增加必须两项均可测的条件。','',
          '## 分项候选覆盖与偏差','',
          '各版本接受的对象有差异，下表的均值仅描述各自输出，配对效果见下一节。差值参照人工图像端点，使用相同候选尺度。','',
          '| 分组 | 版本 | 杆长覆盖 | 杆长平均绝对差 mm | 外径覆盖 | 外径平均绝对差 mm |',
          '|---|---|---:|---:|---:|---:|']
    paired=[]
    for label,path in [('开发',development),('复核',review)]:
        summary=json.loads((path/'summary.json').read_bytes())
        for phase,fields in summary.items():
            a,b=fields['underhead_length'],fields['thread_major_diameter']
            text.append(f"| {label} | {phase} | {a['n']}/{a['total']} | {a['mean_absolute_mm']:.3f} | {b['n']}/{b['total']} | {b['mean_absolute_mm']:.3f} |")
        rows=json.loads((path/'comparison.json').read_bytes())
        for key in ('underhead_length','thread_major_diameter'):
            old={r['object_id']:r for r in rows if r['phase']=='original' and r['dimension']==key and r['candidate']}
            new={r['object_id']:r for r in rows if r['phase']=='bounded_refine' and r['dimension']==key and r['candidate']}
            common=sorted(old.keys() & new.keys())
            pairs=[dict(object_id=k,old_mm=old[k]['difference_mm'],new_mm=new[k]['difference_mm']) for k in common]
            paired.append(dict(role=label,dimension=key,pairs=pairs,
                               gained=sorted(new.keys()-old.keys()),lost=sorted(old.keys()-new.keys())))
    text+=['','## 同对象配对比较','',
           '| 分组 | 尺寸 | 配对数量 | 原版平均绝对差 mm | 候选平均绝对差 mm | 改善/变差/基本相同 |',
           '|---|---|---:|---:|---:|---|']
    for group in paired:
        pairs=group['pairs']
        old=np.abs([p['old_mm'] for p in pairs]);new=np.abs([p['new_mm'] for p in pairs])
        improved=int((new<old-1e-6).sum());worse=int((new>old+1e-6).sum())
        text.append(f"| {group['role']} | {group['dimension']} | {len(pairs)} | {old.mean():.3f} | {new.mean():.3f} | {improved}/{worse}/{len(pairs)-improved-worse} |")
    text+=['','## 结论与停止条件','',
           '开发组有局部改善：pass132_005 的杆长差从约 -3.57 mm 改为 -0.38 mm，fail40_002 的外径差从约 +1.67 mm 改为 -0.38 mm。也存在退步，例如 pass76_004 的外径从约 +0.26 mm 变为 -0.56 mm。','',
           '复核组三颗配对外径没有实质变化。三颗配对杆长平均绝对差从 1.031 mm 变为 1.027 mm，改进很小。新增的 pass101_001 外径差约 -0.964 mm，仍需人工检查。由此不建议替换默认算法，也不扩大到完整 Train 或 Validation。','',
           '此次结果不能证明已达到暂定容差。杆尖沿用轮廓末端，阴影、头部三维姿态和目标混入仍可影响轴线。人工点选自身也有不确定性，缺少匹配实物的卡尺真值。','',
           '每颗样本 JPG 左侧绿色为人工线，右侧为候选。comparison.csv 记录每项候选、拒测原因、尺寸差及端点偏移。下一步如继续，应先核实仍有大偏差的轮廓，或采用已评估的整体投影作辅助信息；继续增加门槛缺乏收益证据。']
    report=review/'report.md'
    report.write_text('\n'.join(text)+'\n',encoding='utf-8')
    (review/'paired_comparison.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report)


if __name__=='__main__':run()
