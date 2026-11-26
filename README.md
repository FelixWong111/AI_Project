# Mechanical Component Inspection

正式系统：新版Baseline1、新版Baseline2、Baseline3 **size_repair_v2**。唯一注册表为 [config/model_registry.json](config/model_registry.json)，GUI、CLI、P08共用同一加载入口并校验SHA；错误明确报错，没有生产Mock或其他模型回退。GUI仅显示三个Baseline，`final`为NOT_READY预留ID。

**整体NOT_READY：** 离线集成与303项软件测试通过；用户确认实机布置/标定未完成，Camera0探测的3帧全黑，真实工位延迟、Camo标定、教师容差确认、真实演示视频及原生窗口人工验收仍未完成；干净环境最小复现已通过。完整证据、指标、错误案例、Git交付及人工步骤见 [P07/P08完整交接](docs/p07/final_handoff.md)。

## 模型与工单

|ID|方法/正式目录|能力|
|---|---|---|
|baseline1|HOG + StandardScaler + MLP；`models/release/baseline1/`|固定工单整图分类|
|baseline2|冻结ResNet18 + StandardScaler + Logistic Regression；`models/release/baseline2/`|固定工单整图分类|
|baseline3|YOLO11n + v2几何尺寸 + 前景检查 + 计数/判定；`models/release/baseline3/`|原图框、0–5类别/计数、尺寸证据、PASS/FAIL/REVIEW|

来源、SHA及参数见 [模型提升记录](results/p07_final_integration/model_promotion.json)。未重训或改变冻结阈值/预处理/数值尺寸范围/几何算法。历史权重保留用于复现；未选卡尺源码和checkpoint在 [archive/experiments/](archive/experiments/README.md)，不进入正式注册表。

固定工单为ID1–5各1，唯一公共定义在 [config/inspection.json](config/inspection.json)，ID0期望恒0。GT实际数量不能反推工单。Baseline1/2仅支持固定向量，其他合法向量返回`UNSUPPORTED_COUNTS`且标签/分数为空；无框、实际数量或ID0能力。两者GUI/CLI统一使用原始分类器，scene_guard关闭；未校准分数不能证明任意无关场景有效。

Baseline3支持任意合法五类向量，工单重判不运行YOLO、不改变框/类别/分数/观测数量。确认ID0强制FAIL，不可信尺寸或未解释前景保留非PASS及原因。对象分数不是整图PASS概率。

## 数据

唯一正式根目录为 [datasets/mechanical_components_v2/](datasets/mechanical_components_v2/README.md)：Train360、Validation80、Stage2 Test30，冻结划分不变；尺寸开发154、已使用回归66、标尺10。专项220张只对应8件实物，66张不是新独立测试。Stage3的90图及真值仍本机封存，不交P08、不参加此次评价。原P03/原始数据集未改写。 **数据集图片不提交GitHub或Git LFS，改为网盘共享。** Git保留全部划分清单、GT标注、类别映射、SHA256、配置和README。P08先下载图片，按原目录和文件名放回`datasets/mechanical_components_v2/`，无需修改代码。网盘链接由P07另行提供。

GT在`annotations/`及`public/`各分区；推理只读图像和工单。真实文件名工单位于`configs/expected_counts_{train,validation,stage2_test}.json`，只含`image_id`与五类要求数量。原始JSON、YOLO标签、GT CSV、split lists与SHA来源映射均在正式数据目录。

## 环境与启动

当前已运行Windows环境：Python3.11.14、PyTorch2.7.1+cu126、TorchVision0.22.1+cu126、Ultralytics8.3.236，其他精确版本见requirements。现有环境`pip check`和真实三模型运行通过；干净环境实际安装、pip check、三模型真实加载、每模型3张Train图片预测和8项核心接口测试均通过；没有重复完整303项测试。原`.venv311`未重新安装或改动依赖。

以下克隆命令需要先由P07提交并push本次Git和LFS文件；本次未自动上传，远程LFS配额及新克隆仍未验证：

```powershell
git lfs install
git clone -b p07/system-integration https://github.com/FelixWong111/AI_Project.git
cd AI_Project
git lfs pull
# 从网盘取得数据图片，按 datasets/mechanical_components_v2/README.md 合并到同名目录
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
python scripts/verify_models.py --load-models
python scripts/check_dataset_v2.py
python app.py --model baseline1
python app.py --model baseline3 --camera 0 --capture-backend dshow
```

已有`.venv311`不用重新创建。按钮或1/2/3切模型，Camera/Image/Video、Start/Pause、Snapshot、Record保留；操作见 [UI说明](docs/ui_operations.md)。本机DSHOW Camera0=Camo，需先在Camo Studio确认真实手机画面。

## 批处理与评价

```powershell
python predict.py --input_dir datasets/mechanical_components_v2/public/stage2_test/images --model baseline1 --output_dir outputs/baseline1_unique --counts_file datasets/mechanical_components_v2/configs/expected_counts_stage2_test.json
python p08_run.py --evaluation-config datasets/mechanical_components_v2/configs/stage2_evaluation.json --check-data-only
# 已有本次结果，优先复用。以下命令会另行执行正式Test，只在需要复现时运行：
python p08_run.py --evaluation-config datasets/mechanical_components_v2/configs/stage2_evaluation.json --model all --output outputs/p08_reproduction
```

每次新输出目录，更换`--model`即可切三个模型。共同结果在 [results/p08_stage2_final/](results/p08_stage2_final/README.md)：Baseline1/2复用同SHA、同预处理、无门控的既有预测；Baseline3为完整v2首次Stage2运行。Stage2此前已用于其他基线对照，不能称全新未使用；没有据Test再次调参，Stage3保持NOT_RUN。

所有模型输出`images.csv`与`run_config.json`；仅Baseline3输出`objects.csv`、`counts.csv`及尺寸诊断。原图像素左上原点xyxy，计数覆盖0–5，difference=observed−expected，遵循原合同。继续复用P08 evaluator及object_observations，没有第二套Evaluator。完整指标、分母、REVIEW和逐图/对象错误均保留。

## 实机与交付

```powershell
# 仅在真实Camo工位及标定就绪后执行，独立运行，不需要同时开app.py
python scripts/benchmark_live.py --camera 0 --warmup 20 --frames 100 --models baseline1 baseline2 baseline3
python -m unittest discover -s tests -v
```

主Live边界包含P07渲染和显示更新调用，结果在`results/live_benchmark/<run_id>/`，见 [计时说明](docs/live_benchmark.md)。旧离线时间不当作实机成绩；照片35.758446px/mm不用于不同镜头/变焦/机位的1280×720 Camo。尺寸模块未关闭，也没有伪造validated状态。

人工步骤见 [v2物理验收](results/baseline3_v2_experiment/size_repair_v2/manual_acceptance.md)、[拍摄清单](docs/p07/demo_recording_checklist.md)。原始研究决策未找到，缺项记录在 [review/decision_stage2.md](review/decision_stage2.md)，没有倒填计划。根`PROGRESS.md`/`task_state.json`支持恢复。提交清单、LFS体积与未完成项见完整交接；本次无commit/push/merge。

## 提交本次交付

数据集图片、原始数据、虚拟环境和临时输出继续忽略；三个正式模型权重、源码、标注、预测CSV/JSON和指标保留。提交前核对 [Git清单](results/p07_final_integration/git_submission_files.csv)，本次未自动暂存、提交或推送。

```powershell
git lfs install
git add -A
git diff --cached --stat
git lfs ls-files --size
git commit -m "Finalize P07 Stage 2 handoff"
git push -u origin p07/system-integration
```
