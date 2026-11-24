# -*- coding: utf-8 -*-
"""Rebuild the Stage 1 Chinese report from UTF-8 JSON/CSV, without rerunning data.

Run this file directly. Do not pipe non-ASCII Python source through Windows
PowerShell 5.1, whose default $OutputEncoding may replace characters with '?'.
The old Markdown is never an input to report construction.
"""
from __future__ import annotations
from collections import Counter
import argparse
import csv
import json
import math
from pathlib import Path
import re
import uuid

ROOT=Path(__file__).resolve().parents[1]
CLASSES={1:'long_screw',2:'thick_screw',3:'thin_screw',5:'flat_washer'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require(condition,message):
    if not condition:raise ValueError(message)


def cell(value):
    return str(value).replace('|',r'\|').replace('\n',' ').replace('\r',' ')


def load_evidence(experiment):
    run=experiment/'train_full_final'
    records=read_json(run/'records.json');summary=read_json(run/'summary.json')
    images=read_json(run/'source_image_sha256.json')
    seen=set();expected_csv={}
    for record in records:
        require(record['split']=='train','TRAIN_ONLY: unexpected partition')
        cid=record['class_id'];require(type(cid) is int and cid in CLASSES,'Unknown class ID')
        identity=(record['image_id'],record['object_id'])
        require(identity not in seen,'Duplicate object identity');seen.add(identity)
        require(record['image_id'] in images,'Image missing from source hash inventory')
        for phase in ('before','after'):
            result=record[phase]
            for dimension,values in (result['dimensions'].items() or [('none',{})]):
                key=(*identity,phase,dimension,result['status'])
                expected_csv[key]=(cid,result['reason'],values.get('pixels'),values.get('mm_estimate'))
            for dimension,values in result['diagnostics'].get('partial_dimensions',{}).items():
                require(result['status']=='UNMEASURABLE','Partial candidate must remain unmeasurable')
                key=(*identity,phase,dimension,'PARTIAL_REVIEW_ONLY')
                expected_csv[key]=(cid,result['reason'],values['pixels'],values['mm_estimate'])
    for cid in CLASSES:
        rows=[r for r in records if r['class_id']==cid]
        actual=dict(objects=len(rows),**{phase:dict(
            candidate_count=sum(r[phase]['status']=='MEASURED_CANDIDATE' for r in rows),
            failure_reasons=dict(Counter(r[phase]['reason'] for r in rows if r[phase]['reason'])))
            for phase in ('before','after')})
        actual['newly_refused']=sum(r['before']['status']=='MEASURED_CANDIDATE' and r['after']['status']=='UNMEASURABLE' for r in rows)
        actual['newly_available']=sum(r['before']['status']=='UNMEASURABLE' and r['after']['status']=='MEASURED_CANDIDATE' for r in rows)
        actual['after_partial_review_only']=sum(bool(r['after']['diagnostics'].get('partial_dimensions')) for r in rows)
        require(summary[str(cid)]==actual,f'SUMMARY_MISMATCH: class {cid}')
    csv_seen=set()
    with (run/'comparison.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            key=(row['image_id'],row['object_id'],row['phase'],row['dimension'],row['status'])
            require(key in expected_csv and key not in csv_seen,'CSV_MISMATCH: missing/duplicate record')
            csv_seen.add(key);cid,reason,pixels,mm=expected_csv[key]
            require(int(row['class_id'])==cid and row['reason']==reason,'CSV_MISMATCH: class/reason')
            for field,expected in [('pixels',pixels),('mm_estimate',mm)]:
                if expected is None:require(row[field]=='','CSV_MISMATCH: missing value substituted')
                else:
                    value=float(row[field]);require(math.isfinite(value) and math.isclose(value,expected,rel_tol=0,abs_tol=1e-9),'CSV_MISMATCH: numeric value')
    require(csv_seen==set(expected_csv),'CSV_MISMATCH: incomplete CSV')
    verification=read_json(experiment/'verification_final.json') if (experiment/'verification_final.json').exists() else {}
    if verification:
        require(verification['train_images']==len(images) and verification['target_records']==len(records),'Verification count mismatch')
    return records,summary,images,verification


def validate_report_text(text):
    require(all(word in text for word in ('阶段 1','测量','候选','原始记录')),'CHINESE_TEMPLATE_MISSING')
    require('\ufffd' not in text,'UNICODE_REPLACEMENT_CHARACTER')
    require(not re.search(r'\?{2,}',text),'LOSSY_QUESTION_MARKS')
    require(text.encode('utf-8',errors='strict').decode('utf-8',errors='strict')==text,'UTF8_ROUNDTRIP_FAILED')


def brief(result):
    if result['dimensions']:
        return '; '.join(f"{key}≈{d['mm_estimate']:.3f} mm" for key,d in result['dimensions'].items())
    return result['reason'] or '无完整候选'


def build_report(experiment):
    experiment=Path(experiment).resolve()
    records,summary,images,verification=load_evidence(experiment)
    selection=read_json(experiment/'selection.json')
    regressions=read_json(experiment/'additional_train_regression_cases.json') if (experiment/'additional_train_regression_cases.json').exists() else []
    screw_count=sum(summary[str(i)]['objects'] for i in (1,2,3))
    before=sum(summary[str(i)]['before']['candidate_count'] for i in (1,2,3))
    after=sum(summary[str(i)]['after']['candidate_count'] for i in (1,2,3))
    partial=sum(s['after_partial_review_only'] for s in summary.values())
    lines=['# P06 阶段 1：Train 轮廓测量修复与验证', '',
        '## 结论', '',
        '本报告从原始 JSON/CSV 结果重新生成，没有读取或重新编码损坏的 Markdown。垫圈孔洞处理取得进展；'
        '螺钉的稳定尺寸测量仍未通过验收。程序输出候选值不等于尺寸正确，新增拒绝也不等于已证明正确拦截。', '',
        '当前仅适合人工复核，不应接入自动尺寸判定。没有卡尺真值，不能宣称物理测量误差下降。', '',
        '## 数据范围与方法', '',
        f"初始开发清单有 {len(selection['train_image_selection'])} 张 Train 图。最终对照覆盖 {len(images)} 张 Train 图中的 {len(records)} 个 ID1/2/3/5 对象；ID0 和螺母不是本轮修改目标。"
        'Train 用于开发和排错，不是独立精度证明。', '',
        '初始清单：'+', '.join(cell(n) for n in selection['train_image_selection'])+'。', '',
        '额外 Train 回归案例：'+('; '.join(cell(r['object_id']) for r in regressions) if regressions else '无额外记录')+'。', '',
        '- 螺钉改用逐截面锐边、宽度与中心主轨迹及可见上包络；拒绝宽缓阴影和不一致边缘。',
        '- 头下起点要求双侧边缘可见、锐利且位置一致；截面还须明确避开头部和末端。',
        '- 只有杆径候选、头下起点未确认时，主状态仍为 UNMEASURABLE；部分线仅在 diagnostics.partial_dimensions 中供 REVIEW。',
        '- 垫圈处理中心相符的奇数层孔洞，按观测支持拟合椭圆，保留残差、支持比例和长短轴。',
        '- 新垫圈尺寸线表示可见椭圆长轴，原版表示拟合圆水平直径；定义不同，不是透视或高度补偿，不能据数值接近标称程度宣称更准。', '',
        f"候选尺度维持 x={selection['scale_px_per_mm'][0]:.10f}、y={selection['scale_px_per_mm'][1]:.10f} px/mm；采用近似共面和相同拍摄几何假设。"
        '缩放来源不明的图片拒绝毫米换算。算法未使用商品尺寸寻找、移动或补齐边界。', '',
        '## 全部 Train 的实际对照', '',
        '下表统计完整候选输出，不是检测准确率、尺寸准确率或容差通过率。', '',
        '| 类别 | 对象数 | 原完整候选 | 新完整候选 | 新仅部分杆径候选（REVIEW） | 新增完整候选 | 新增拒绝 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for cid,name in CLASSES.items():
        s=summary[str(cid)]
        lines.append(f"| {cid} {name} | {s['objects']} | {s['before']['candidate_count']} | {s['after']['candidate_count']} | {s['after_partial_review_only']} | {s['newly_available']} | {s['newly_refused']} |")
    lines+=['',f'螺钉完整候选从 {before}/{screw_count} 降为 {after}/{screw_count}，另有 {partial} 个仅部分杆径候选。'
        '检查更保守、可用性下降，仍可能误拒合法件，不能声称螺钉能力整体提升。', '',
        '## 真实记录案例', '',
        '| Train 图片／对象 | 原版本 | 最终版 |', '|---|---|---|']
    focus=('fail122_004','fail78_004','pass01_005','pass05_001','pass05_003','pass11_003','fail113_006','fail116_007')
    chosen=[r for oid in focus for r in records if r['object_id']==oid]
    if not chosen:chosen=records[:8]
    for r in chosen:
        lines.append(f"| {cell(r['image_id'])}／{cell(r['object_id'])} | {cell(brief(r['before']))} | {cell(brief(r['after']))} |")
    partial_examples=[r for r in chosen if r['after']['diagnostics'].get('partial_dimensions')]
    if partial_examples:
        lines+=['','以下仅为部分候选的原始数值，主状态仍为 UNMEASURABLE，不能当成有效完整测量：']
        for r in partial_examples:
            text='; '.join(f"{key}≈{d['mm_estimate']:.3f} mm" for key,d in r['after']['diagnostics']['partial_dimensions'].items())
            lines.append(f"- {cell(r['object_id'])}：{text}；原因 {cell(r['after']['reason'])}。")
    if any(r['object_id']=='fail116_007' for r in partial_examples):
        lines+=['','fail116_007 的部分候选仍可能选择杆旁阴影，必须保持 REVIEW，不能用来证明边界已经修复。']
    lines+=['', '## 仍然存在的失败', '', '| 类别 | 最终版失败原因及数量 |', '|---|---|']
    for cid,name in CLASSES.items():
        reasons='; '.join(f'{cell(key)}: {value}' for key,value in summary[str(cid)]['after']['failure_reasons'].items()) or '无'
        lines.append(f'| {name} | {reasons} |')
    lines+=['','头杆身份、阴影反光、裁剪邻件、孔缘与倒角仍可能造成错误输出或误拒。杆端仍取可见轮廓端点，也可能带入阴影。'
        '本轮没有放宽门槛追求输出数量，仍需少量对应实物卡尺测量。', '',
        '## 原始记录、图像及版本', '']
    for label,relative in [('原始逐对象记录','train_full_final/records.json'),('逐尺寸与失败 CSV','train_full_final/comparison.csv'),
        ('数量汇总','train_full_final/summary.json'),('六例前后对照','train_full_final/before_after.jpg'),
        ('头部混淆拦截','train_full_final/head_region_regression.jpg'),('部分候选复核','train_full_final/partial_review.jpg'),
        ('原代码快照','before/p06_geometry.py'),('最终版本哈希','stage1_frozen_code_final.json'),('工程完整性记录','verification_final.json')]:
        if (experiment/relative).is_file():lines.append(f'- [{label}]({relative})')
    lines+=['','中间诊断目录保留追溯，最终结果入口为 train_full_final。以前的 Validation 与商品差值报告保留历史定义和版本，未被本轮覆盖。', '',
        '## 测试与实际执行状态', '']
    if verification:
        lines.append(f"原阶段 1 记录：几何专项 {verification['geometry_tests']} 项、完整 P06 {verification['full_p06_tests']} 项通过。"
                     '这些是原阶段工程检查，编码修复本轮的测试另见 report_regeneration_checks.json。')
        lines.append(f"原阶段状态：Validation {verification['validation']}；Test {verification['test']}；卡尺误差 {verification['caliper_accuracy']}；YOLO 训练 {verification['yolo_training']}。")
        lines.append(f"工程记录核对了 {verification['checked_primary_and_partial_lines']} 条主／部分测量线的数值换算，并记录源图、冻结权重与团队文件未变化。计算一致性不代表物理精度。")
    else:lines.append('未提供完整工程验收记录；相应测试和物理精度不得据此宣称通过。')
    lines+=['','## 编码修复与复现', '',
        '旧报告在 PowerShell 5.1 的 ASCII 源码管道中丢失了中文，随后即使用 UTF-8 写盘也只能保存问号。'
        '现在使用磁盘上的 UTF-8 Python 源文件、显式 UTF-8 JSON/CSV 读取及 UTF-8 原子写入；不依赖控制台默认编码，不修改全局编码设置。', '',
        '重新生成前核对 JSON、汇总和 CSV 的对象、状态及数值一致性。旧 Markdown 不是输入，不能通过重新编码恢复已丢失的中文。', '',
        '```powershell',
        'conda run -n ai-p06-yolo python -B tools/p06_report_geometry_stage1.py --replace',
        '```', '',
        '本次只重建报告，不执行测量算法、不读取原始 Train/Validation/Test 图片、不训练或修改冻结配置。'
        '下一步仍是人工复核测量线与少量卡尺核验；不会自动进入新的数据检查或尺寸判定。','']
    text='\n'.join(lines);validate_report_text(text)
    return text


def write_report(experiment,output=None,replace=False):
    experiment=Path(experiment).resolve()
    target=Path(output) if output is not None else experiment/'report.md'
    if not target.is_absolute():target=experiment/target
    target=target.resolve()
    require(target.is_relative_to(experiment) and target.suffix.lower()=='.md','Report must be an .md file inside the experiment')
    if target.exists() and not replace:raise FileExistsError('Report exists; use --replace explicitly')
    text=build_report(experiment)
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_bytes(text.encode('utf-8',errors='strict'))
        validate_report_text(temp.read_text(encoding='utf-8',errors='strict'))
        if target.exists() and not replace:raise FileExistsError('Report appeared during generation')
        temp.replace(target)
    finally:
        if temp.exists():temp.unlink()
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=ROOT/'output/p06_geometry_stage1')
    parser.add_argument('--output',type=Path,help='Markdown path inside the experiment; defaults to report.md')
    parser.add_argument('--replace',action='store_true',help='Explicitly replace the report after evidence/encoding checks')
    args=parser.parse_args()
    try:
        write_report(args.experiment,args.output,args.replace)
        print('report_written_utf8')  # ASCII log remains safe under any console code page.
    except (OSError,ValueError,KeyError) as error:parser.exit(1,f'Report generation failed: {error}\n')


if __name__=='__main__':main()
