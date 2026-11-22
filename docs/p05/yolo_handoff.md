# P05 / P06 / P07 技术栈与交接核查

状态：按组内拟定版本编写，待最终运行机安装验证。baseline3 选用 YOLO11n；此处没有新增 P06 检测算法或替代其训练任务。

## 目标环境

Python 3.11.14；PyTorch 2.7.1；TorchVision 0.22.1；Ultralytics 8.3.236；NumPy 1.26.4；opencv-python 4.10.0.84；cv2-enumerate-cameras 1.4.0。P05 的附加依赖仍见 requirements-p05.txt。

官方版本配对依据：https://pytorch.org/get-started/previous-versions/#v2-7-1 。Ultralytics 8.3.236 的依赖要求见 https://github.com/ultralytics/ultralytics/blob/v8.3.236/pyproject.toml 。版本范围满足依赖声明，只能说明可以尝试共用环境，最终以实际安装、算子和模型执行记录为准。

## 已检查的代码与缺口

- P05 使用的 resnet18(weights=None)、ResNet18_Weights.IMAGENET1K_V1、torch.load(weights_only=True)、inference_mode 和 torchvision.transforms 均不依赖 2.8 专有接口。本次只改变目标依赖，不改变模型、白边预处理、阈值或固定数量规则。
- 原安装说明中的 2.8.0 / 0.23.0 已改为目标版本。历史 verification.json 和 training_smoke.json 仍按实际测试环境保留，不改写成新版本已通过。
- src/p05_bridge.py 仅接 baseline2。P06 尚需交真实 baseline3/final 的工厂；config/p07.json 的 bridge 为空，不能靠在界面切换 baseline3 就调用 YOLO。P08 负责统一分发，当前不要把 P05 工厂直接配置给所有模型。
- YOLO11n 官方通用权重只能作为迁移训练的起点。正式演示必须加载 P06 训练的自定义权重，交代类别映射、阈值、图像尺寸和模型哈希。

## 运行成员安装后的检查

1. 在最终使用的 Python 3.11.14 环境安装官方配套 torch/torchvision 构建；CPU 示例见 README_P05.md，GPU 成员按显卡和驱动选择 CPU/cu118/cu126/cu128 中适用的官方源。不要混装 CPU 和 CUDA 的 torch/torchvision。
2. 执行 `python -m pip install -r requirements-team.txt`，三个角色依赖一起解析，避免单独安装 Ultralytics 时无意更新 NumPy/OpenCV。
3. 执行 `python -m scripts.check_team_stack --device cpu --output output/team_stack_cpu.json`。GPU 部署再用 `--device cuda:0 --output output/team_stack_cuda.json`。返回 FAIL 时先修环境，不进入正式跑分。
4. 执行 `python -m unittest discover -s tests -v`，再按 README_P05.md 用 P03 数据训练 baseline2，并通过 P07 真实入口检查。检查脚本只验证精确版本、依赖、ResNet18 前向、NMS、CLAHE 和随机 YOLO11n 执行；它不会下载模型权重，也不证明课程性能。
5. P06 用正式权重额外验证至少一张实拍图的加载、BGR 输入、检测框和计数，保存原图 ID、输出、配置和权重哈希。P07 检查真实视频流，P08 用封存场景做算法对照。
6. 全部成功后再执行 `python -m pip freeze > output/team_environment.txt`，由 P08 归档依赖清单、检查 JSON、设备与驱动信息和正式运行 ID。不得用随机模型测试结果填课程指标。

## P06 交给 P07/P08 的接口

| 项目 | 交接要求 |
|---|---|
| 工厂 | `create_backend(*, model_id, config)`，提供 `metadata`、`infer`、`rejudge`、`synchronize`，与 src/p07_bridge.py 的调用方式一致 |
| 输入 | OpenCV uint8 H×W×3 BGR；避免对 NumPy BGR 输入再做一次无依据的 RGB 交换 |
| 类别 | 训练若使用五类连续 ID 0–4，必须明确映射到项目 ID 1–5。项目 ID 0 专指 non_catalogue，不能把训练第 0 类直接当作目录外件 |
| 目录外件 | YOLO 闭集类别不会自动识别所有未知零件。按既定方案接 OpenCV 尺寸校验与拒识；标定、尺度换算、容差及异常情况由 P01/P06 明确，保留原冻结基线 |
| 坐标 | 输出原图像素 xyxy，非归一化 xywh；每框提供唯一 detection_id、整数 class_id、有限 score，转换张量为 Python 数字 |
| 数量 | counts 字符串键 0–5；objects 与 counts 一致；non_catalogue_count 等于 counts['0']；存在目录外件时不能 PASS |
| 重判 | 改预期数量时复用当前帧的检测结果，只重算规则；不把预期数量当作检测观测，也不重新使用另一帧结果 |
| 时延与复现 | synchronize 在指定 CUDA 设备同步；metadata 记录权重哈希、实际版本、device、imgsz、conf、iou、尺寸标定、容差与拒识规则 |

## P05 改进模块的接入位置

baseline3 使用原图 BGR；候选 final 在 YOLO 输入前调用 `DetectorCLAHE(clip_limit, tile_grid_size)(frame_bgr)`。该模块不改变图像尺寸或源帧，检测框仍以原图坐标交付。几何/尺寸校验继续使用原图和标定，不擅自换成增强图。P06 保存启用开关与冻结参数，P08 在同图、同权重、同阈值、同标定下做成对对照。只有正式对照支持时才能报告改进有效。
