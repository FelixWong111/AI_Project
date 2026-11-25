# P06 尺寸证据与暂定容差核验

入口为 src/p06_size_evidence.py。它接收现有 YOLO 观察值，按检测框提取带边距的裁剪，调用当前默认几何测量，再比较商品参考尺寸。可直接调用 compare_dimensions() 处理已保存的单项或多项测量结果。

## 使用

```python
import json
from pathlib import Path
from src.p06_size_evidence import load_specs, SizeEvidenceProcessor

specs = load_specs("config/p06_part_specs_candidate.json")
calibration = json.loads(
    Path("config/p06_size_calibration_candidate.json").read_text(encoding="utf-8")
)
processor = SizeEvidenceProcessor(specs, calibration)
# frame_bgr 为原始 BGR 帧，backend 为已加载的冻结 P06Backend。
observations = backend.infer(frame_bgr, expected_counts)
enriched = processor.process(
    frame_bgr,
    observations,
    capture_geometry_id=confirmed_capture_geometry_id,
)
```

调用方须核实 capture_geometry_id 对应的相机、支架、焦距、裁剪和分辨率条件。候选配置只适用于原先 3024×4032 拍摄条件，validated=false。实时摄像头需提供自身标定配置。像素尺度可用于生成候选毫米数，validated=true 则需要独立证据支持。

每个对象新增 size_evidence，包含裁剪范围、测量、标定来源、规格候选、逐尺寸比较和原因。测量及其诊断中的坐标均为裁剪局部坐标，原图坐标需加 crop_xyxy 前两项。图像和原有观察对象均复制后处理。

## 可信度与区间

几何模块的 CANDIDATE 只说明产生了测量线。默认比较标为 UNVERIFIED。可信尺寸需同时具有已验证的标定和外部提供的逐尺寸质量证据。

```python
quality = {
    detection_id: {
        "thread_major_diameter": {
            "verified": True,
            "max_error_mm": validated_error_bound,
            "evidence_id": measurement_validation_record_id
        }
    }
}
```

该误差界应来自匹配拍摄条件与测量方法的验证。它与产品允许偏差分别记录。本阶段没有从现有人工标记均值推定误差上界。

- WITHIN 表示测量区间完整落在暂定规格范围内。
- OUTSIDE 表示两个区间严格分离。
- BORDERLINE 表示测量区间跨规格边界。
- MISSING 表示该项没有测量，保留几何模块的失败原因。
- UNVERIFIED 表示有候选数值但缺少可信依据，可查看点值是否落在暂定范围内。

规格端点按包含边界处理。任一可靠尺寸处于 OUTSIDE 就能排除该规格，其余尺寸缺失时仍可保留排除证据。

## 规格候选与责任边界

螺钉测量同时比较 ID 1、2、3，螺母和垫圈分别比较自身系列。调用方通过 confirmed_families 显式列出已确认形态家族的 detection_id；默认没有形态确认。单凭 YOLO 类别无法确认未知物体的形态家族。

已确认家族且所有适用规格均被可靠尺寸排除时，得到 OUTSIDE_FAMILY_CANDIDATE。唯一剩余规格且所需尺寸区间均满足时，得到 SPEC_COMPATIBLE_CANDIDATE。其余情况为 INSUFFICIENT_EVIDENCE。这些结论随对象证据交给后续共享判定模块。

当前商品容差为用户暂定值。整个结果保持 REVIEW，模型类别、检测置信度和六类检测计数保留。P08 后续负责对象归类后的计数与最终判定。ID 0 保留检测观察，空图维持六类零检测计数。

本阶段尚未接入 P07 工厂、前景补检或关联去重，也没有完成尺寸精度验收。垫圈单项失败时，当前默认几何实现可能清空两项；核验函数可处理单项结果，测量实现的该限制继续保留。
