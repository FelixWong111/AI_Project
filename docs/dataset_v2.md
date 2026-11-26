> 历史来源说明：当前正式数据为datasets/mechanical_components_v2，生产模型由config/model_registry.json选择。正式操作与验收状态见docs/p07/final_handoff.md；下文旧路径/模式仅用于历史追溯。

# 新采集数据的使用入口

2026-11-26状态更新：用户已确认全部560图类别/数量/框/异常件标注，并确认90张Stage 3未参与开发。现有划分未改变，数据已READY且冻结，确认与冻结记录见`data/mechanical_components_v2/audit/human_confirmation_20261126.json`及`freeze_20261126.json`。当前训练、P08 Stage 2对照与版本建议见[原始对照指标](../results/dataset_v2_baseline_comparison/comparison_metrics.csv)。下文NOT_READY描述是原转换流程的门槛；它未被删除或跳过，而是在真实确认后解除。数据/模型训练时的NOT_RUN字段属于快照，当前评价状态单独记录，Stage 3仍未评价。

本机派生数据位于`data/mechanical_components_v2/`，完整审查报告在该目录README；可随Git分享的精简记录在[results](../results/data_manifests/mechanical_components_v2/README.md)。原始`数据集/`和派生`data/`均忽略，模型权重的Git规则不变。

转换不会训练模型，也不会替换现有模型的开发数据。默认只创建新目录；重复运行已有目录用`--verify-only`。`--refresh-candidate`仅可刷新明确NOT_READY且未冻结的本次派生文件，并先检查全部原始SHA256，不能刷新冻结数据。

```powershell
python scripts/prepare_dataset_v2.py
python scripts/prepare_dataset_v2.py --verify-only
python scripts/check_dataset_v2.py
```

560张公共图只保存一份规范图片，Baseline1/2共用YOLO图片；尺子10张和尺寸专项220张单独导出PNG，未进入公共工单。原始HEIC完整保留。HEIC工具依赖仅本地`.tools/pillow_heif`，需要重建时可安装：

```powershell
python -m pip install --target .tools --no-deps pillow-heif==1.9.0
```

Baseline1/2的数据入口是`data/mechanical_components_v2/development/`，其两份CSV只含Train/Val，不含任何测试标签。配置为`configs/training.json`（相对该数据根）。可运行以下纯数据检查，不能去掉`--prepare-only`来绕过未冻结状态：

```powershell
python -m scripts.train_baseline1 --dataset-config data/mechanical_components_v2/configs/training.json --prepare-only --audit-candidate
python -m scripts.train_baseline2 --dataset-config data/mechanical_components_v2/configs/training.json --prepare-only --audit-candidate
```

Baseline3的数据入口是`data/mechanical_components_v2/yolo/data.yaml`，仅包含Train/Val，六类映射与现有P06冻结模型一致，ID0没有删除或变成正常类别。此文件供P06下一版本候选训练使用，当前任务不执行训练；历史P06训练脚本的旧数据与冻结约束保留，不能将新数据伪装成历史已验收数据。

P08需要明确选择阶段，检查命令不运行模型、预测或评分：

```powershell
python p08_run.py --evaluation-config data/mechanical_components_v2/configs/stage2_evaluation.json --check-data-only
python p08_run.py --evaluation-config data/mechanical_components_v2/configs/stage3_evaluation.json --check-data-only
```

阶段配置指向各自的过滤Ground Truth、独立工单文件和测试清单。正式执行受`dataset_status.json`的READY及冻结检查。Stage 3保留集不可用于算法调试、选择阈值或改进方案。本次新模型对照只使用Val和Stage 2，P08通过`--config`选择独立模型配置；四模型不替换正式GUI默认版本。

全部工单复用`config/inspection.json`的固定ID1–5各1。Ground Truth来自人工JSON目标计数。`COMPLETE`表示格式完整，不等同于`VERIFIED`；两张表/JSON冲突的人工确认绑定SHA256记录在results的annotation_resolutions.json，未修改任何原始数据。没有卡尺、实物编号、标定确认的数据保持空白或待核实。
