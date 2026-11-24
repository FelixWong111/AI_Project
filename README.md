# Mechanical Component Inspection

工单规范：唯一公共参考位于 [`config/inspection.json`](config/inspection.json)，固定 ID1–ID5 各 1，ID0 恒为 0。实际数量属于 Ground Truth，不能反推工单。已生成 `outputs/work_orders/train.json`（262条）与 `validation.json`（46条）。生成、GUI/批处理统一校验、UNSUPPORTED_COUNTS 和特殊工单功能测试说明见 [工单接口文档](docs/work_orders.md)。正式 Test 仍为 NOT_RUN。

Baseline1 实时异常输入修正：P07 现启用训练特征支持范围检查，超出范围返回 REVIEW（标签/分数为空），不强制 FAIL。原始 HOG+MLP 权重和验证成绩不变；predict.py 保留原始基线，门控不参与其基线指标。详情及诊断证据见 [实时诊断说明](docs/baseline1_live_diagnosis.md)。部署 Baseline1 到 P07 时还需携带 `models/baseline1/scene_guard.npz`。其他模型不变。

机械零件视觉检测：实时摄像头、图片和视频输入，模型切换、记录和统一批处理。Baseline1/2 为实拍数据训练的固定工单分类器；Baseline3 已接入 P06 的真实六类 YOLO11n 权重和计数判定；Final 尚未配置。Baseline3 的尺度/容差及完整未知件拒识尚未提供，不能宣称完整课程功能验收通过。详见 [Baseline3接入](docs/baseline3_integration.md) 和 [功能合规审查](docs/p06/compliance_review.md)。

**Official held-out test: NOT_RUN。** 本次只训练与验证，未读取正式测试图片、测试真值或计算测试成绩。验证结果不是正式测试结果。

## 环境

Windows、Python 3.11.14，使用现有 `.venv311`，不需要 conda 环境列表中存在它。

PowerShell：

```powershell
.\.venv311\Scripts\Activate.ps1
python -m pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

CMD 激活命令为 `.venv311\Scripts\activate.bat`。也可直接调用 `.\.venv311\Scripts\python.exe`。完整三模型运行需要 PyTorch 2.7.1+cu126、TorchVision 0.22.1+cu126 与 Ultralytics 8.3.236；先按 docs/baseline3_integration.md 安装匹配CUDA构建，再安装 requirements.txt。Baseline1/2 原有依赖版本保留，历史训练CPU环境记录不改写。

## 数据与固定工单

新版测试集已整理为 [`data/test_20261124/`](data/test_20261124/README.md)：36 张照片全部直接用真实文件名，无原用例编号映射。统一工单为 expected_counts.json，真值为两份 ground_truth CSV；19 PASS、16 FAIL、1 REVIEW（34.jpg）。scene_id 等于图片名，独立场景依据用户本次确认。该目录只用于评估，不进入训练；整理不代表已执行正式 Test。具体命令和 REVIEW 规则见该目录 README。

P03 总包保持完全只读。只使用其 `01_` 开发包，`images/train` 为 262 张，`images/validation` 为 46 张。脚本按元数据和明确分区读取，不递归扫描整个 P03 图片目录。可用 `--data-dir` 显式指定开发包目录；只有唯一来源时才自动定位。

固定 expected_counts 为 `{"1":1,"2":1,"3":1,"4":1,"5":1}`。依据是 P03 的 `reference_label_all_one` 真值字段及逐类数量核对，不是把课程格式示例当成数量要求。ID 0 期望始终为 0。

训练：124 PASS / 138 FAIL；验证：24 PASS / 22 FAIL。原始 scene_id 仍为 TBD。2026-11-24 用户明确确认每张图片为不同物理场景，因此训练/验证场景数分别按 262/46 记录，确认来源保存在 config/training.json 和外部清单；这不是哈希算法证明，也没有回写或编造拍摄 SceneID。

清单校验、图片与像素哈希、排除记录、数量与框一致性记录保存在 `outputs/training/data/manifest.json`。当前排除 0 张。之后如果数据或确认有变化，需重新审查和生成新版本清单。

## 方法及候选模型

| 模型 | 方法 | 文件 | 验证 Accuracy / FAIL F1 |
|---|---|---|---|
| baseline1 | 灰度 128×128 → HOG 1764维 → 训练集 StandardScaler → MLP(32) | models/baseline1/baseline1.joblib | 100% / 1.0000 |
| baseline2 | RGB 等比例缩放+白边 224×224 → 冻结 ImageNet ResNet18 512维 → 训练集 StandardScaler → Logistic Regression | models/baseline2/baseline2.json + resnet18-f37072fd.pth | 80.43% / 0.7805 |
| baseline3 | P06 冻结 YOLO11n，0–5类检测+P07计数判定 | models/baseline3/best.pt | 历史训练指标见training_record；集成验收与正式Test分开 |
| final | Stage3研究改进入口 | 尚未交付 | NOT_RUN |

Baseline1/2 均为图像级分类器，只支持上述固定工单。输出真实 PASS/FAIL 和未校准概率；Baseline1 分数为 P(预测类别)，Baseline2 分数为 P(FAIL)。不输出或虚构框、实际数量、ID 0 数量，界面显示 N/A。其他合法工单返回 UNSUPPORTED_COUNTS，标签和分数为空。文件缺失或推理失败明确 ERROR，无生产 Mock 回退。

## 训练与复现

当前候选已经真实训练，完整记录在 `outputs/training/baseline1/`、`outputs/training/baseline2/`，模型目录也附 training_record.json。重新实验使用新输出目录，防止覆盖冻结证据：

```powershell
python -m scripts.train_baseline1 --output-dir models/baseline1-rerun
python -m scripts.train_baseline2 --output-dir models/baseline2-rerun
```

首次空模型目录的实际训练命令为 `python -m scripts.train_baseline1` 和 `python -m scripts.train_baseline2`。Baseline1 保存匹配的配置和 freeze，训练新版本时会更新 config/baseline1.json 的候选路径；Baseline2 新目录需显式更新 config/baseline2.json 的 baseline2_bundle 才用于部署。

Baseline2 默认复用 `models/baseline2/resnet18-f37072fd.pth`，可用 `--weights` 指定同 SHA256 的本地官方权重。训练不联网下载，不微调骨干。StandardScaler 仅 fit 训练集；验证集仅比较 2 个 MLP 和 3 个逻辑回归候选，选择 FAIL F1，平局先减少 False PASS，再保留较简单候选；分类阈值固定 0.5，没有反复调阈值。特征缓存带数据、预处理、骨干和依赖版本签名，位于 P03 外部。

## 启动软件

```powershell
python app.py --model baseline1
python app.py --model baseline2 --camera 0 --capture-backend dshow
```

本机用户先前确认 Camo 为 DSHOW camera 0；不同电脑需先执行 `python app.py --list-cameras`。可在顶部点击 Baseline1/2，或按 1/2 切换；Start/Pause、Image、Video、Camera、Snapshot、Record 均保留。当前先用验证集图片验收，不用测试图演示。图片模式：`python app.py --model baseline1 --image "允许的验证图片路径"`。

完整键盘与控件说明见 docs/ui_operations.md。Baseline3 使用相同GUI及批处理入口，支持自定义工单；Final 未配置时明确 ERROR。Baseline3 权重、CUDA或依赖加载失败也明确ERROR，不回退模拟结果。

## 课程批处理入口

在 PowerShell 设置当前开发包（只枚举 `01_`，不读取测试）：

```powershell
$data = Get-ChildItem -Directory -Filter 'P03_*' | ForEach-Object { Get-ChildItem -Directory -LiteralPath $_.FullName } | ForEach-Object { Get-ChildItem -Directory -LiteralPath $_.FullName -Filter '01_*' } | Select-Object -First 1
$val = Join-Path $data.FullName 'images\validation'
python predict.py --input_dir "$val" --model baseline1 --output_dir outputs/batch-baseline1 --counts_file outputs/training/data/validation_counts.json
python predict.py --input_dir "$val" --model baseline2 --output_dir outputs/batch-baseline2 --counts_file outputs/training/data/validation_counts.json
```

每次运行使用新的输出目录。工单 JSON 是只含 image_id 与 expected_counts 的列表；不含真值、框或实际数量。输出 `images.csv` 和 `run_config.json`。images.csv 的字段顺序固定为：

```text
image_id,model_id,status,predicted_label,prediction_score,score_type,non_catalogue_count
```

图像级 non_catalogue_count 为空；不生成 objects.csv/counts.csv。GUI 与 CLI 使用同一注册表和 Adapter，推理不读取 ground truth。

## 验证与资料

```powershell
python -m unittest discover -s tests -v
python tests/test_p04.py
python -m scripts.check_candidates
```

check_candidates 在全部验证图上核对重新加载后的批处理结果与训练记录，检查工单限制、缺失权重、无头 app 和 P03 未变；输出新目录到 outputs/acceptance。实际 GUI 和相机验收状态见 docs/model_training.md，不把无头检查当成真实摄像头实测。

P08 可使用 docs/model_training.md、docs/system_architecture.md、docs/model_integration.md、模型目录 training_record.json、外部数据清单、原始训练/验证预测、验收记录。完整目标级评估器仍待 P08 提供；本次不覆盖其未来接口。

## 工程目录

```text
app.py / predict.py          实时与课程批处理入口
config/                     固定工单、数据确认、模型及 UI 配置
src/data/                   开发数据只读校验
src/models/baseline1/        P04 HOG/MLP 实现及共享工具
src/adapters/               Baseline1/2 模型加载与推理
src/inference/              批处理业务
src/p07_*.py, camera.py      UI、输入、记录和后端注册
scripts/                    正式训练与验收
models/baseline1,baseline2/  真实候选权重、配置及记录
tests/                      单元测试与测试专用 Stub
docs/                       当前交接、报告资料及 history 历史来源
outputs/training/           清单、特征缓存、实验指标、原始预测
outputs/refactor_backup_20261124/  重构前源码与未提交改动备份
P03_*/                      权威原数据，完全只读，另行交付
```

旧原始开发/验证/测试目录保留用于追溯，不是训练入口。旧合成训练器、P05 smoke handoff 和旧配置已归档至重构备份，不是部署模型。Project-Man、team_plan、成员贡献和建议记录保留。

权重文件保留真实文件且不被 .gitignore 排除；交付时同时提供 models、config、src 和所需依赖。outputs 目录不自动进 Git，需要随实验资料另行交付；尤其不要漏交清单、原始预测、训练记录和备份说明。本次不自动提交或推送 Git。

P03 更新测试集后：先核定独立场景、标注和工单，封存测试版本，再用两个冻结模型调用同一批处理入口并由独立评估器读取真值；此前不进行正式测试集评估。
