> Historical handoff, retained for attribution. Current commands and status: root README.md.

# P05 与 P07 接入说明

本次基于 `main@4552cba` 接入 `baseline2`，分支为 `p05/baseline2-integration`。P07 从 `config/p07-baseline2.json` 加载 `src.p05_bridge:create_backend`。P08 可将该工厂接入自己的统一分发器，模型模块继续复用同一份代码。

## 训练与模型包

先由 P03 交原图、`ground_truth_images.csv`、`ground_truth_objects.csv`。由 P08 确认与 baseline1 共用的五类固定预期数量。下面的五个 1 是命令格式示例，正式运行时填团队确认值。

```powershell
python -m scripts.train_baseline2 --image-dir data/images --images-csv data/ground_truth_images.csv --objects-csv data/ground_truth_objects.csv --expected-counts 1 1 1 1 1 --output-dir weights/baseline2
```

首次训练下载 torchvision 官方 ImageNet ResNet18 权重。离线时用 `--weights PATH_TO_RESNET18_PTH` 指定本地副本。模型保存后 `weights/baseline2/` 包含骨干权重、`baseline2.json`、开发输入哈希和验证输出。训练器使用 train/val，按场景检查数据隔离，并从非忽略目标真值派生图像标签。固定向量不同于训练向量时，预测返回 `UNSUPPORTED_COUNTS`。

## 安装依赖

团队确认的目标解释器为 Python 3.11.14，NumPy 1.26.4、OpenCV 4.10.0.84、cv2-enumerate-cameras 1.4.0。先用 `python --version` 核对当前虚拟环境，再执行下方安装命令。

YOLO 部分拟采用 PyTorch 2.7.1、TorchVision 0.22.1、Ultralytics 8.3.236，以 YOLO11n 为 baseline3。P05 与 P06 共用这对 PyTorch/TorchVision 版本。此为待队友安装验证的目标环境，尚未冻结为已验证环境。

P07 原有依赖和 P05、YOLO 在同一个虚拟环境中安装。以下是 CPU 示例；使用 NVIDIA GPU 时由运行成员按显卡和驱动选择官方对应的 CUDA 安装源，两个包必须来自同一构建源。

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-team.txt
python -m pip check
python -m scripts.check_team_stack --device cpu --output output/team_stack_check.json
```

历史测试使用 Python 3.12 CPU、PyTorch 2.8.0 / TorchVision 0.23.0，并固定 SciPy 1.15.3 以兼容 P07 的 NumPy 1.26.4。这些历史记录保留在 `docs/p05/verification.json` 和 `training_smoke.json`，不能当作新技术栈已通过的证据。Python 3.11.14、新版本组合、CUDA 均由队友在最终运行机验证。检查脚本验证精确版本、依赖一致性、ResNet18 前向、TorchVision NMS、CLAHE 与 YOLO11n 随机初始化推理；它不证明课程零件识别有效。P06 另用正式训练权重验证预测，P08 汇总冻结环境。详见 `docs/p05/yolo_handoff.md`。

## P07 相机与离线入口

在仓库根目录运行。先准备由正式实拍数据训练的 `weights/baseline2`。

```powershell
python app.py --backend real --model baseline2 --config config/p07-baseline2.json --image data/example.png
python app.py --backend real --model baseline2 --config config/p07-baseline2.json --camera-name Camo
```

`model_config.baseline2_bundle` 是模型目录，`device` 默认 cpu。可增加 `expected_model_version` 并填模型清单内的哈希以固定部署版本。相对路径按仓库根目录解析。工厂为 P05 的 baseline2 入口，其他模型由其负责人交相应工厂，P08 统一分发。

| 接口 | 行为 |
|---|---|
| `create_backend(model_id=..., config=...)` | 按配置加载模型并校验权重与清单，提供 metadata |
| `infer(frame_bgr, expected_counts)` | 对新帧执行一次推理并保留该帧在训练向量下的预测 |
| `rejudge(result, expected_counts)` | 使用该帧保存的预测检查固定向量。改成其他向量时显示不支持，改回时恢复原标签与分数 |
| `synchronize()` | CPU 直接返回，CUDA 同步本适配器使用的设备 |
| `metadata` | 记录模型版本、模型文件总大小、预处理、固定数量、阈值及 `P(FAIL)` 的分数口径 |

数量重判保留在各帧结果中，因此恢复旧帧时能取回对应预测。`_p05_prediction` 是模块内部缓存，P07 的 CSV 导出读取统一字段。界面结果的 `counts` 与 `non_catalogue_count` 为空，`objects` 为空列表。概率表示逻辑回归估计的 `P(FAIL)`，尚未校准，阈值为 0.5。

## P08 批处理接入

```powershell
python -m scripts.p05_predict --input_dir TEST_FOLDER --model baseline2 --output_dir RESULTS --counts_file expected_counts.json --bundle-dir weights/baseline2
```

此独立入口供交接与复现，`images.csv` 和 `run_config.json` 遵循组内字段。P08 的公共 `predict.py` 在 baseline2 分支调用 `Baseline2.predict_path`，实时端通过桥接调用同一个 `Baseline2.predict_frame`。

```python
from src.adapters.baseline2 import Baseline2
adapter = Baseline2("weights/baseline2", device="cpu")
row = adapter.predict_path(input_dir / image_id, expected_counts)
run_config = adapter.run_config(per_image_expected_counts)
```

## 联调与后续交接

- P03 提供原图、标注、场景切分和数据版本。P08 确认固定预期向量后运行训练。
- P05 交完整模型包、依赖与验证记录。P07 以正式权重验证实时画面、数量切换和截图导出。
- P08 用自己的公共入口和共同测试集跑正式指标。正式实时性能需在真实相机与显示流程上采集至少 20 帧。
- `src/stage3_clahe.py` 为原有研究候选。P05/P06 根据冻结基线的真实失败证据决定启用与参数，P06 集成到 final，P08 跑独立对照。
- 原 P05 报告与决策素材位于 `docs/p05/report_draft.md` 和 `docs/p05/decision_stage2_draft.md`，由 P08 合入组报告和正式决策记录时填入实测证据。
