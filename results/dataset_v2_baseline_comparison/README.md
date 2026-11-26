# 新旧Baseline模型选择实验文件

本目录仅保存P07借用P08独立评估程序比较新旧Baseline1/2所需记录，不作为P08正式报告或交付包。建议后续接入`baseline1_new`和`baseline2_new`；默认GUI及正式模型目录尚未替换。

- `comparison_metrics.csv` / `.json`：四模型已保存指标，原始分类器结果不含scene_guard。共同Validation为80图，Stage 2 Test为此前已完成的同30图对照。
- `training/`：新模型训练记录、Train/Validation清单及逐图预测；旧模型信息见`old_model_inventory.json`。
- `validation/` / `evaluation/`：原始P08预测CSV、评分JSON、计时CSV及实际命令；`runs.json`索引实际运行目录。
- `reference_data/`：Train/Validation/Stage 2划分与对应真值、工单；不含原图和Stage 3真值。
- `failure_cases/*.csv` / `*.json`、`subgroup_metrics.csv`：误判及子组统计，保留退步案例。
- `selection.json` / `model_selection.md`：Test运行前根据Validation选出的版本及当时冻结SHA256，保持原件。
- `human_confirmation.json`、`dataset_freeze.json`、`preflight/`：人工确认、数据冻结和实际检查记录。
- `environment.json`、`model_versions.json`、`reproducibility.md`：执行环境、权重身份及必要复现信息。
- `scope_cleanup.json`：本次收窄范围的实际删除、恢复及保留文件哈希；`artifact_manifest.json`是清理后的目录清单。

新权重保存在`models/baseline1_v2_candidate/`和`models/baseline2_v2_candidate/`。所有训练记录、原始评估CSV/JSON和配置快照保持不变。

`baseline1_*_gui`验证结果、`configs/*_gui.json`、`integration/`、`source_fingerprints.json`及旧测试日志仅是清理前历史记录，不代表当前GUI已接入新模型。新`scene_guard.npz`保留原件供追溯，运行时接线已撤回，不能作为现行部署配置。旧scene_guard及既有GUI功能保留。

英文/中文正式报告、12张图及生成工具已撤销；实机脚本恢复本次修改前版本。离线计时不代表实机延迟或纯模型forward时间。Stage 2未重复运行，Stage 3未访问，未commit或push。
