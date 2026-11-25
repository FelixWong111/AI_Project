"""Check supplier references and describe existing measurements. No new inference.

No geometry tuning, Test access, product decision or frozen-model config update.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.p06_labelme import CLASSES

FIELDS={1:{'underhead_length':'under_head_length_mm','thread_major_diameter':'thread_major_diameter_mm'},
        2:{'underhead_length':'under_head_length_mm','thread_major_diameter':'thread_major_diameter_mm'},
        3:{'underhead_length':'under_head_length_mm','thread_major_diameter':'thread_major_diameter_mm'},
        4:{'across_flats':'across_flats_mm'},5:{'outer_diameter':'outer_diameter_mm','inner_diameter':'inner_diameter_mm'}}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def validate_catalogue(config,package):
    package=Path(package).resolve()
    require(config['units']=='mm' and config['formal_decision_enabled'] is False,'Candidate must use mm and disable formal decisions')
    doc=package/'p06_supplier_specs.json'
    require(doc.is_file(),'Missing supplier JSON')
    require(digest(doc)==config['source']['supplier_json_sha256'],'Supplier JSON SHA256 mismatch')
    supplier=json.loads(doc.read_text(encoding='utf-8'))
    require(config['class_zero']['class_id']==0 and config['class_zero']['class_name']==CLASSES[0]
            and config['class_zero']['dimensions'] is None,'ID 0 must not have one fixed nominal size')
    ids=[item['class_id'] for item in config['items']]
    require(all(type(cid) is int for cid in ids) and sorted(ids)==[1,2,3,4,5],'Expected unique class IDs 1..5')
    source_ids=[item['class_id'] for item in supplier['items']]
    require(sorted(source_ids)==[1,2,3,4,5],'Supplier class IDs must be unique 1..5')
    images={}
    for item in config['items']:
        cid=item['class_id'];ref=next(r for r in supplier['items'] if r['class_id']==cid)
        require(item['class_name']==ref['class_name']==CLASSES[cid],'Class mapping mismatch')
        require(item['supplier_spec']==ref['nominal_spec'],'Supplier designation mismatch')
        require(item['iso_din_standard'] is None,'Standard number has not been confirmed')
        relative=Path(item['source_image'])
        require(bool(relative.parts) and not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]=='reference_images','Unsafe reference-image path')
        image=package/relative
        require(image.resolve().is_relative_to(package) and not image.is_symlink(),'Redirected reference-image path')
        require(image.is_file(),f'Missing source image: {image.name}')
        require(item['source_image']==ref['source_image_in_package'],'Source image pairing mismatch')
        actual=digest(image)
        require(actual==item['source_image_sha256']==ref['source_image_sha256'],f'Image SHA256 mismatch: {image.name}')
        require(set(item['dimensions'])==set(FIELDS[cid]),'Unsupported or missing geometric dimension')
        for name,value in item['dimensions'].items():
            field=FIELDS[cid][name]
            require(value['supplier_field']==field and finite(value['reference_mm']) and value['reference_mm']>0,'Invalid supplier field/value')
            require(value['reference_mm']==ref[field],f'Supplier value mismatch: ID{cid} {name}')
            allowance=value['provisional_allowance_mm']
            require(finite(allowance) and allowance>0,'Invalid provisional allowance')
        for field,value in item['reference_only_dimensions'].items():
            require(field in ref and finite(value) and value==ref[field],f'Supplier reference-only value mismatch: {field}')
        images[item['source_image']]=actual
    return images


def compare_record(record,config):
    cid=record['class_id']
    require(type(cid) is int and cid in range(6),'Unknown measurement class ID')
    if cid==0:return []
    item=next(r for r in config['items'] if r['class_id']==cid);rows=[]
    for key,reference in item['dimensions'].items():
        value=record.get('dimensions',{}).get(key)
        measured=record['status']=='MEASURED_CANDIDATE' and value is not None
        estimate=value['mm_estimate'] if measured else None
        require(estimate is None or (finite(estimate) and estimate>0),'Invalid/nonfinite geometric estimate')
        rows.append(dict(image_id=record['image_id'],object_id=record.get('object_id',''),split=record.get('split','ruler_example'),
                         class_id=cid,dimension=key,measurement_status=record['status'],
                         mm_estimate=estimate,supplier_reference_mm=reference['reference_mm'],
                         difference_from_supplier_reference_mm=None if estimate is None else estimate-reference['reference_mm'],
                         provisional_allowance_mm=reference['provisional_allowance_mm'],
                         reason=record.get('reason','') or ('' if measured else 'MISSING_DIMENSION'),
                         decision='REVIEW',basis='descriptive_difference_only; not caliper error or product conformance'))
    return rows


def run(output):
    output=Path(output).resolve()
    require(output.is_relative_to(ROOT/'output') and not output.exists(),'Need a new output/ directory; never overwrite')
    config_path=ROOT/'config/p06_part_specs_candidate.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    package=ROOT/'data/p06_part_specs_handoff'
    hashes=validate_catalogue(config,package)
    require(digest(package/'README_for_Codex.md')==config['source']['readme_sha256'],'README SHA256 mismatch')
    hashes.update({'p06_supplier_specs.json':digest(package/'p06_supplier_specs.json'),
                   'README_for_Codex.md':digest(package/'README_for_Codex.md')})
    run_path=ROOT/'output/p06_geometry/evaluation_02'
    source_paths=[run_path/'measurements.json',run_path/'ruler_photo_measurements.json']
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in source_paths}
    rows=[];counts=Counter()
    for path in source_paths:
        records=json.loads(path.read_text(encoding='utf-8'))
        for record in records:
            require(record.get('split','ruler_example') in ('train','validation','ruler_example'),'Forbidden measurement partition')
            rows.extend(compare_record(record,config));counts[record.get('split','ruler_example')]+=1
    require(all(digest(package/path)==sha for path,sha in hashes.items()),'Source package changed while reading')
    output.mkdir(parents=True)
    def save(name,data):(output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save('comparison.json',rows)
    save('verification.json',dict(status='PASS_REFERENCE_INTEGRITY_NOT_MEASUREMENT_ACCURACY',
        source_package_sha256=hashes,source_measurements_sha256=source_hashes,candidate_config_sha256=digest(config_path),
        source_records=dict(counts),comparison_rows=len(rows),decision='REVIEW_ONLY',
        new_geometry_or_yolo_execution=False,caliper_error_validation='NOT_RUN',instructor_confirmation='PENDING_NOT_RECORDED'))
    lines=['# P06 supplier references and provisional allowances','',
           '原始五图与 JSON 逐项人工核对一致，五张图片 SHA-256 均匹配。实际目录为 `data/p06_part_specs_handoff/`。',
           '供应商图注明单批次参考、以实物为准；304 为商家声明。商品示意图未用于像素标定。', '',
           '| ID | 测量项目 | 商品参考 mm | 用户暂定允许偏差 mm | 暂定数值区间 mm |',
           '|---|---|---:|---:|---|']
    for item in config['items']:
        for key,d in item['dimensions'].items():
            n,t=d['reference_mm'],d['provisional_allowance_mm']
            lines.append(f'| {item["class_id"]} | {key} | {n:g} | ±{t:g} | {n-t:g}–{n+t:g} |')
    lines+=['','这些区间是用户提出的候选课程允许偏差，**不是供应商制造公差、教师确认记录或图像测量不确定度**。'
            '未把它们写入冻结 YOLO 配置，未据此输出 PASS/FAIL。', '',
            '商品尺寸已明确为：大扁头十字螺钉头径 9.2/11.4/9.2 mm；M4 六角螺母对边 7 mm；垫圈 5×15×1.2 mm。'
            '商品图未标 ISO/DIN 标准号，仍不强行指定标准。此前 ISO 7045 头径和通用垫圈 5.3 mm 仅是背景资料，不作为当前商品规格。', '',
            '## 沿用既有测量，不重新拟合','',
            'comparison.json 基于 evaluation_02 的冻结测量输出，逐项保存“估计值 − 商家参考值”。'
            '差值不能解释为测量误差，也不能据此确认实物错规格。未重新读取开发图片、未调算法或尺度，未读取 Test。', '',
            '| 带尺图 | 项目 | 候选估计 mm | 相对商家参考差值 mm |', '|---|---|---:|---:|']
    for row in rows:
        if row['split']=='ruler_example':
            val='—' if row['mm_estimate'] is None else f'{row["mm_estimate"]:.3f}'
            delta='—' if row['difference_from_supplier_reference_mm'] is None else f'{row["difference_from_supplier_reference_mm"]:+.3f}'
            lines.append(f'| {row["image_id"]} | {row["dimension"]} | {val} | {delta} |')
    lines+=['','5.jpg 的孔洞歧义保留失败，未用 0、标称值或外径推算内孔来补齐。'
            '带尺图的几何类型映射仍需确认与所购实物对应；商品图只确认预期商品信息。', '',
            '## 下一步限制','',
            '- 螺钉量头下杆长和牙顶外包络，不含头；对边不等于对角距。',
            '- 头厚、螺母厚度、垫圈厚度及从螺孔估计公称螺纹直径仍不支持自动俯视测量；头径仅参考，不新增判定项目。',
            '- 轮廓/尺度不足、缺失尺寸和测量不确定度未知的情况保持 REVIEW。现有螺钉外径长尾仍存在，新增容差不会修复测量质量。',
            '- 少量实物卡尺真值仍需补充；真实误差验证为 NOT_RUN。课程正式评估前仍需教师确认目录、容差和拍摄范围。',
            '- 包装数量不是 expected_counts；ID0 没有单一规定尺寸；未实现最终规格拒识或自动六类重判。', '',
            '来源配置：`config/p06_part_specs_candidate.json`；完整性证据见 verification.json。'
            '本轮只读原始商品资料及历史数值，新增独立配置、检查工具/测试和说明；未训练、修改冻结 YOLO 或提交 Git。','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return dict(output=str(output),source_records=dict(counts),comparison_rows=len(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/p06_supplier_specs_check')
    args=parser.parse_args()
    try:print(json.dumps(run(args.output),indent=2))
    except (ValueError,OSError,KeyError) as error:parser.exit(1,f'Specification check failed: {error}\n')


if __name__=='__main__':main()
