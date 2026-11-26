# Baseline3独立候选实验进度

更新时间UTC：2026-11-26T01:37:56.669635+00:00

- COMPLETED：requirements_environment
- COMPLETED：preserve_old
- COMPLETED：dataset_isolation
- COMPLETED：stage2_audit
- COMPLETED：development
- COMPLETED：training
- BLOCKED：size_calibration；Missing compatible physical calibration, specimen/caliper truth, instructor approval and measurement validation; no credible size MAE
- COMPLETED：validation_selection
- COMPLETED：candidate_selection
- COMPLETED：freeze
- COMPLETED：stage2_evaluation
- COMPLETED：failure_analysis
- COMPLETED：performance
- COMPLETED：p07_compatibility
- COMPLETED：acceptance
- COMPLETED：recommendation
- BLOCKED：live_benchmark；Requires user-connected Camo, physical platform and fresh arranged frames; no camera attempt or simulated live data
- COMPLETED：size_v1_inventory
- COMPLETED：size_v1_rulers
- COMPLETED：size_v1_baseline
- COMPLETED：size_v1_tuning
- COMPLETED：size_v1_holdout
- COMPLETED：size_v1_public_validation
- COMPLETED：size_v1_acceptance
- COMPLETED：size_v1_foreground

恢复：先检查task_state.json、PID、日志和checkpoint；运行下列幂等命令，不重复训练/Test：

`python scripts/experiment_baseline3_v2.py --phase auto`

Stage 3图像与真值禁止访问。app.py、默认配置和Baseline1/2不修改。

## 当前尺寸开发恢复入口

`python scripts/develop_baseline3_size.py --phase auto`

自动开发结果：`size_development_v1/`。正式物理验收、教师确认、独立卡尺误差与 Camo 标定仍未通过；不得将开发候选的 PASS 冒充正式验收。

## 尺寸候选最终开发结论

- 10张尺子1019刻度，全局35.758446 px/mm；尺寸专项220张/8实物，154开发+66保留场景。
- 完整测量160/220→162/220，正常规格通过91/100→92/100，均为开发候选范围，不是物理制造验收。
- Val：7 PASS / 46 FAIL / 27 REVIEW，False PASS=0、False FAIL=6；实际测量378次、完整305次。
- 91个不同软件用例通过，281个受保护文件SHA一致；默认GUI配置与所有旧模型不变。
- 不建议替换默认：需修复几何误测、正常件误拒绝及前景关联，并完成Camo、卡尺和教师物理验收。
- 完整记录：`size_development_v1/README.md`；已完成的正式Test未重跑，Stage 3图像/专用标注未打开。公共统一索引的读取边界见报告第1节。

## 尺寸几何修复 v2

- audit: COMPLETED
- development: COMPLETED
- head_diagnostic: COMPLETED
- train_replay: COMPLETED
- freeze: COMPLETED
- regression: COMPLETED
- acceptance: COMPLETED

结果目录：`size_repair_v2/`；恢复：`python scripts/repair_baseline3_size.py --phase auto`。旧冻结模型/结果保留；Val仅为已使用数据的回归诊断，正式物理验收待完成。

### v2 最终实际结果

- A088：2.62875mm→5.06174mm；真实外边缘181像素，换算尺度未改，整图因垫圈拟合失败仍REVIEW。
- 已使用Val回归：测量完整305/378→343/378；PASS7→23；REVIEW27→15；False PASS0→0，False FAIL6→2；全输入正确率58.75%→78.75%，Coverage66.25%→81.25%。
- 98项软件测试通过；133个v1结果文件、281个原保护文件、v1算法及新冻结配置SHA一致。
- 未训练YOLO/修改权重/放宽范围/置validated=true；P03未写入，正式Test未重跑，Stage3未打开。
- 仍未完成正式物理验收，剩余垫圈孔/椭圆问题及H034长度拒识保留；卡尺、公共/专项M5实物对应、Camo及教师确认待补。
- 记录：`size_repair_v2/README.md`；人工操作：`size_repair_v2/manual_acceptance.md`。

## Vision Caliper v3

- audit: COMPLETED
- develop: COMPLETED
- select: COMPLETED
- freeze: COMPLETED
- regress: COMPLETED
- benchmark: COMPLETED
- check: COMPLETED
- report: COMPLETED
- archive: COMPLETED

独立结果：`size_caliper_v3/`；恢复：`python scripts/develop_caliper_v3.py --phase auto`。已完成测量/逐图回归缓存将复用，不重跑旧实验或封存测试。

### v3 最终结论

- 开发集冻结选型：保留A（size_repair_v2），不替换默认GUI。
- 固定复用Val80：A/B/C完整测量343/115/342（分母378），PASS23/0/22，REVIEW15/36/16，False PASS均0，False FAIL2/4/2。
- 已使用专项保留66：完整测量52/21/52；不作为独立验证。
- 软件108项通过，真实worker/batch/video/工单重判及B1/B2加载通过。
- 346个原保护文件SHA保持一致；478个新JSON无损压缩SHA验证，当前结果约49.57MiB。
- 没有训练YOLO、访问正式Test/Stage3、修改P03或commit/push/merge。
- 可执行实验已完成；后续需要卡尺真值、Camo重新标定、教师容差确认及新独立实物验收。详见size_caliper_v3/README.md与final_verification.json。
