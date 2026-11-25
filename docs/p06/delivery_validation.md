# P06 最小代码交付验证 · 2026-11-25

交付分支为 feat/p06-baseline3。本次获取远端后，origin/main 的提交
6bdb8a181747b1d3b59495490da08240c99b2de9 是当前分支祖先。已有P07实现随main包含。
其他组员独立功能分支没有在本次自动合并。

## 交付范围

本次提交仅包含P06运行模块、候选配置、测试、说明及既有自检工具更新。
运行模块为 p06_geometry、p06_screw_edges、p06_foreground、p06_size_evidence、
p06_measurement_policy、p06_pipeline，沿用既有 p06_inference 和 p06_data。
独立测量实验入口、原图、Test、模型缓存和实验输出继续保留在本地。

沿用已受Git管理的唯一冻结权重，5465242字节，SHA-256为
4c5a1f759623a88ed2d21d639a4f3254e9da92d89107e4cdbfd6bddcf4e76b3d。
文件路径仍为 output/p06_runs/p03_baseline3_640_seed42/weights/best.pt。

## 干净检出实测

从暂存区导出全部受管理文件，使用空的独立检出目录，未复制data和本地实验输出。
沿用现有ai-p06-yolo依赖，Python3.11.18、torch2.7.1+cu126、Ultralytics8.3.236。
以下命令退出码均为0：

```powershell
python -B -m unittest discover -s tests -p "test_p06_*.py" -v
python -B -m unittest discover -s tests -p "test_p07.py" -v
python -B tools/p06_handoff_check.py --config config/p06_p07_pipeline_candidate.json --output-dir output/p06_clean_integrated_check
```

P06共182项、P07共23项通过。完整开发工作区先前193项还包含未交付的实验工具测试，
因此与最小交付测试数不同。

第三条实际加载冻结模型并使用cuda:0，CUDA张量平方和为14.0。
P07工厂和工作线程处理合成帧通过，缓存重判保留检测框及计数，且不产生新推理计时。
内层业务状态为REVIEW，foreground_enabled=false，真实测量验证记录为空。
输入为程序生成图像，该运行不评价真实零件检测或尺寸精度。

本地检查记录位于 output/p06_github_delivery_checks 和
tmp/p06_delivery_clean/output/p06_clean_integrated_check；这些生成文件不随Git提交。

## P07与P08后续

从 HANDOFF_P07.md 开始复现，现场操作后记录实际Git提交、GPU、相机、分辨率和结果。
P07摄像头和GUI现场验收为NOT_RUN。候选尺寸需要匹配工位配置及真实验证依据。
P08共享判定、未见错误规格的真实拒识以及后续final仍待集成。原检测入口继续作为对照。
