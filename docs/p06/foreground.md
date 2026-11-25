# P06 前景补检与候选关联

src/p06_foreground.py 提供 ForegroundAssociator，独立消费冻结 YOLO 检测观察值。使用固定工位空背景的 BGR 绝对差、形态学滤波和连通域，所有区域及检测坐标均为原始帧坐标。config/p06_foreground_candidate.json 是合成测试使用的初始参数，尚待 Train 工位验证。

```python
import json
from pathlib import Path
from src.p06_foreground import ForegroundAssociator

params = json.loads(Path("config/p06_foreground_candidate.json").read_text(encoding="utf-8"))
associator = ForegroundAssociator(
    empty_frame_bgr,
    reference_id=recorded_background_id,
    capture_geometry_id=verified_capture_geometry_id,
    roi=(left, top, right, bottom),
    config=params,
)
observations = backend.infer(frame_bgr, expected_counts)
evidence = associator.process(
    frame_bgr, observations, capture_geometry_id=current_capture_geometry_id
)
```

空背景必须与当前输入同分辨率、同相机位置、同焦距及工位。更换输入源后应新建关联器。构造时复制背景，结果保存背景像素 SHA-256、标识、ROI 和实际参数。尺寸不符或拍摄标识不符会明确报错。

## 字段及关联

- regions 为前景连通区域，保存面积、框及工位边缘接触情况。它们是图像变化候选，可能含阴影、反光或多个接触物体。
- associations 保存区域与检测框的实际前景像素交集及重叠比例。分母为区域面积与检测框在 ROI 内栅格化面积的较小值。
- candidates 保存区域与检测的连通关联组，包括所有来源 ID 和原因。
- raw_counts 与 detector_observations 保存原观察值，proposed_counts 仅统计已解决类别归属的关联组。
- counts_complete 表示当前候选组是否均有类别归属，不能据此证明物理场景无漏检。小面积物体、低对比物体或与背景相近的物体仍可能缺失。

同一区域内，所有检测框两两 IoU 均达到阈值、类别相同且区域不接触 ROI 边缘时，保留置信度最高的检测 ID，其他 ID 保存到 suppressed_detection_ids。该操作是待验证的重复框启发式，高度重叠的真实零件可能被误合并。

跨类别重叠保留 CROSS_CLASS_DUPLICATE_AMBIGUOUS。多个框不能满足两两重叠条件时，保留 MULTIPLE_OBJECTS_OR_MERGED_FOREGROUND。一个检测关联多个区域时保留歧义，因为既可能是多个物体，也可能是单个反光物体分裂。

未匹配前景保存 UNMATCHED_FOREGROUND，class_id=null。它不能直接记成 ID 0；还需要实物性与目录外证据。检测没有前景支持时也保留观察值与 REVIEW。proposed_counts 为歧义组之外的部分计数，使用时必须同时检查 counts_complete。

## 背景失效与当前边界

大面积变化触发 BACKGROUND_OR_SCENE_CHANGED，跳过正常关联及去重。当前实现没有曝光自动补偿、阴影模型、背景自动更新或运动跟踪。全局变化比例阈值无法发现所有局部照明失效。

整个模块输出 REVIEW。下一阶段再将已解决的候选送入尺寸证据模块，并把歧义及未知候选交 P08 共享判定。现有 P07 工厂、冻结检测配置及其他组员实现继续沿用。
