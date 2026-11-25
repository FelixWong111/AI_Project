"""Summarise the fixed Train diagnostic from saved actual results."""
import csv
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.p06_measure_geometry import SOURCE
from tools.p06_measure_candidate import load_display,font


def run():
    p=ROOT/'output/p06_train_limited_validation_final'
    def csv_rows(name):
        with (p/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    images=csv_rows('image_results.csv')
    manual=csv_rows('manual_comparison.csv')
    coverage=json.loads((p/'coverage.json').read_bytes())
    measures=json.loads((p/'measurement_records.json').read_bytes())
    counts={k:sum(int(r[k]) for r in images if r[k]!='') for k in
            ('gt_objects','gt_id0','raw_detections','retained_detections','raw_matched','retained_matched','suppressed',
             'foreground_regions','unresolved_candidates')}
    text=['# 真实 Train 有限验证','',
          '固定 16 张 Train，包含两张已标为 reviewed 的空场景、1 张尺度分辨率不匹配图片、ID0 及已知阴影和人工测量案例。使用冻结 YOLO11n 与现有参数，没有据本次结果调参。输入清单与哈希见 input_manifest.json。','',
          '## 检测与计数','',
          f"共 {counts['gt_objects']} 个 GT 实例，其中 ID0 为 {counts['gt_id0']}。原 YOLO 与流水线保留框各 {counts['raw_detections']}、{counts['retained_detections']}，均有 {counts['retained_matched']} 个同类 IoU≥0.5 的一对一匹配。本次没有抑制检测框。两张空图均无 YOLO 框。",'',
          '这些图片属于参与开发的 Train，且是定向选取的诊断案例，不能据此证明独立检测精度或未知规格拒识能力。没有出现实际重复框被成功去除的证据，去重效果仍待针对性实拍验证。','',
          '## 前景与背景条件','',
          '候选背景固定为 fail123.jpg，基于既有固定拍摄条件假设和其 reviewed 空标注。缩略图可见 fail124.jpg 的照明分布与其不同，背景匹配未独立确认。','',
          f"15 张同分辨率图片共提取 {counts['foreground_regions']} 个前景区域，形成 {counts['unresolved_candidates']} 个未解决候选组；fail02.jpg 分辨率不符，未套用该背景。未知候选数是区域关联结果，不能理解为真实物体数。",'',
          'fail124.jpg 无人工物体标注、无 YOLO 检测，却出现 1688 个前景区域。大面积变化检查未触发。说明当前候选背景、像素面积阈值和亮度差分的组合无法可靠处理这批照片。大量碎片又使检测框关联多个区域，阻止了明确对象证据输出。现阶段不应把未匹配区域自动归为 ID0 或触发正式 FAIL。','',
          '## 分尺寸候选覆盖','',
          '按每类实例独立统计。GT 路径使用人工框与类别，YOLO 路径使用实际保留检测框与类别；此批均已同类匹配。两条路径的边距与裁剪略有差异，输出更多只说明候选覆盖变化。','',
          '| 路径 | 类别 | 总数 | 杆长 / 对边 / 外径 | 螺纹外径 / 内径 | 两项同时 |',
          '|---|---|---:|---:|---:|---:|']
    from src.p06_size_evidence import DIMENSIONS
    for c in coverage:
        keys=DIMENSIONS[c['class_id']];n=c['objects']
        second=f"{c[keys[1]]}/{n}" if len(keys)>1 else '不适用'
        text.append(f"| {c['source']} | {c['class_id']} | {n} | {c[keys[0]]}/{n} | {second} | {str(c['both'])+'/'+str(n) if len(keys)>1 else '不适用'} |")
    text+=['','本次默认螺钉和垫圈测量常在一项失败后清空整组，所以两项覆盖计数仍相同。独立尺寸实验分支尚未作为默认启用。','',
           '## 已有人工线对照','',
           '16张图片中包含11颗人工标记对象。本轮只比较这些实际运行对象，其余13颗保留既有回归记录，没有额外读取对应图片。','',
           '| 路径 | 尺寸 | 有候选/11 | 平均绝对差 mm |','|---|---|---:|---:|']
    for source in ('GT_CROP','YOLO_CROP'):
        for key in ('underhead_length','thread_major_diameter'):
            values=[abs(float(r['difference_mm'])) for r in manual if r['source']==source and r['dimension']==key and r['candidate']=='True']
            text.append(f"| {source} | {key} | {len(values)}/11 | {np.mean(values):.3f} |")
    reasons=Counter(m.get('reason','') for m in measures if m['source']=='YOLO_CROP' and m.get('reason'))
    text+=['','此处参照为人工图像线，毫米值沿用候选尺度，没有卡尺真值。YOLO 路径与 GT 路径有候选的对象不同，均值不可直接作为裁剪优劣的配对结论。当前偏差不足以支持 ±0.3 mm 的自动外径规格排除。','',
           '## 失败与规格证据','',
           '| YOLO 裁剪结果的原因 | 数量 |','|---|---:|']
    text += [f'| {k} | {v} |' for k,v in reasons.most_common()]
    conclusions=Counter();relations=Counter()
    for file in (p/'predictions').glob('*.json'):
        result=json.loads(file.read_bytes())
        for obj in result['objects']:
            conclusions[obj['rejection_evidence']['conclusion']]+=1
            for comparison in obj['size_evidence'].get('comparisons',[]):
                for dimension in comparison['dimensions'].values():relations[dimension['state']]+=1
    text+=['',f'对象拒识证据结论计数：{dict(conclusions)}。',f'规格维度比较状态计数：{dict(relations)}。该计数包含每个对象与多个螺钉规格的比较，不能当作独立测量数量。','',
           '候选标定 validated=false，缺少逐尺寸可信误差依据和形态确认，实际入口保持 REVIEW。真实尺寸拒识在本次未得到验证。检测到已学习的 ID0 与确认未见错误规格件是不同的验证任务。','',
           '## 交付结论','',
           '- 可用于联调：冻结检测观察、原图坐标、计数和明确失败原因的证据接口。',
           '- 有限可用：尺寸提取与商品规格描述性比较，需要显示测量失败和 UNVERIFIED。',
           '- 当前不可直接用于正式判定：以这张候选空背景运行的前景补检，以及未经误差验证的尺寸拒识。',
           '- 现场交付前应准备同工位即时空背景并先检查空图误补。需要 P06 继续核对误补与取边，P07 负责摄像头及显示。','',
           '## 可追溯文件','',
           'image_results.csv 为逐图计数与背景统计，manual_comparison.csv 为人工线对比，measurement_records.json 为两个裁剪路径，coverage.json 为覆盖计数，predictions/ 为全部原始流水线结果。contact_sheet_1.jpg、contact_sheet_2.jpg 中绿框是 GT、粉框是 YOLO、蓝线是测量候选。','',
           '首次运行因汇总脚本重复 class_id 关键字中断，修复后在新目录完整重跑。旧 output/p06_train_limited_validation 为未完成历史记录，不能用于本轮结论。输入图片、标注 CSV 和冻结配置哈希前后不变。']
    (p/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    for name in ('fail124.jpg','fail100.jpg'):
        result=json.loads((p/'predictions'/f'{Path(name).stem}.json').read_bytes())
        im,_=load_display(SOURCE/'images/train'/name);draw=ImageDraw.Draw(im)
        for region in result['p06_evidence']['foreground']['regions']:
            draw.rectangle([region[k] for k in ('xmin','ymin','xmax','ymax')],outline='#ec2276',width=3)
        im.thumbnail((900,1200));d=ImageDraw.Draw(im)
        d.rectangle((0,0,900,40),fill='white');d.text((8,8),name+' foreground candidate regions',font=font(20),fill='black')
        im.save(p/('foreground_'+Path(name).stem+'.jpg'),quality=90)
    print(counts)


if __name__=='__main__':run()
