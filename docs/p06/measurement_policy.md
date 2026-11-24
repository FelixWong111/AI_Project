# P06 测量可信条件运行接线

当前交付配置 p06_pipeline_candidate.json 已设置 foreground_enabled=false，工厂不读取背景、不运行前景补检或该模块的去重。原检测框与计数保留，背景原因标为 FOREGROUND_DISABLED。独立前景实验与历史结果继续保留。

尺寸与 ID0 检测证据可直接沿 YOLO 对象生成。尺寸拒识仍要求可信标定、经过验证的测量误差界和形态依据；最终结果保持 REVIEW，留给 P08 共享规则消费。

## 当前生效条件

config/p06_measurement_policy.json 的 records 当前为空。已有真实测量偏差和候选尺度不足以建立可靠误差界，因此当前没有自动启用的真实尺寸可信条件。仍可显示候选线与毫米估计，并记录 UNVERIFIED。

运行时已接通 MeasurementPolicy。尺寸模块测量后先匹配验证范围，再生成 quality 与 family_confirmed 并调用原规格核验。工厂能够自动完成这条接线，调用方不必为每个 detection_id 手动设置 verified。

## 验证记录

registry 的 records 条目包含 path 与 sha256，路径相对仓库根目录或使用绝对路径。每份 UTF-8 JSON 验证记录具有以下字段：

- evidence_id，关联实际验证报告、图像清单与卡尺或人工复核结果。
- status，固定为 VALIDATED_MEASUREMENT_SCOPE。
- geometry_sha256，当前 src/p06_geometry.py 的 SHA-256。
- parameters，当前几何算法 DEFAULT_PARAMETERS 完整内容。
- calibration_id、capture_geometry_id、frame_width、frame_height、px_per_mm。
- class_ids，验证覆盖的同一几何家族类别列表。
- dimensions，尺寸名到已验证 max_error_mm 的映射。
- metric_ranges，自动可检查的诊断指标与允许区间，例如 shaft_line_fraction 的区间；必须至少有一项。
- family_confirmed，记录该范围内是否有足够依据确认几何家族。

文件哈希确认记录内容未被意外更改，不证明记录中的实验证据可靠。维护者应先完成真实验证，再把记录加入 registry。不得仅为启用功能填写较小误差界，或把人工线平均差直接当成最大误差。

匹配规则同时要求标定 validated=true、实际尺度和标定标识一致、拍摄标识及分辨率一致、测量成功、所有诊断指标处于验证区间。无匹配或多份记录同时匹配均保留未验证状态，避免任意选择有利的误差界。算法版本不符、记录哈希不符或格式非法在模型加载前报错。

quality 与 confirmed_families 的外部注入 API 仍供独立实验使用。启用 registry 的运行入口拒绝调用方同时覆盖这些标记。实际规格结果可在 size_evidence.measurement_policy_reason 中检查。

## 验证边界

tests/test_p06_measurement_policy.py 使用明确标记的合成记录验证接线，包括匹配、尺度改变、拍摄条件改变、质量指标越界、冲突记录及哈希篡改。合成误差界只用于验证程序分支。尚无真实尺寸精度或错误规格拒识通过记录。
