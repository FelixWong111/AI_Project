# 机械零件视觉检测：P05 模块与组内接入

本仓库用于八人小组的机械零件视觉检测项目。当前代码包含 P07 的相机与界面框架，以及 P05 的第二套基线和算法改进候选。本说明帮助队员安装环境、准备数据、训练 P05 模型和完成模块交接。

**当前阶段：代码与接口已做合成数据检查，正式实拍数据上的训练、算法有效性对照，以及新技术栈在最终运行机上的兼容性验证仍待完成。**

## 一、各模块负责什么

| 模块 | 负责人 | 用途与状态 |
|---|---|---|
| baseline1 | P04 | 第一套图像级基线，由 P04 交付 |
| baseline2 | P05 | 冻结 ResNet18 特征加逻辑回归，判断整张图片 PASS / FAIL；当前提供训练、批处理与 P07 接入代码 |
| baseline3 | P06 | 拟用 YOLO11n 检测零件，再结合尺寸校验与目录外件拒识；真实检测适配器和正式权重由 P06 交付 |
| final | P05 + P06 | 根据冻结基线的失败证据确定改进；当前 CLAHE 亮度增强是候选，由 P06 集成、P08 独立评估 |
| 相机、界面、演示 | P07 | 采集画面、显示结果、切换预期数量与导出记录 |
| 统一入口与评估 | P08 | 接入各模型，统一字段与运行记录，组织同图对照 |

P05 的 baseline2 是**图像级分类器**：只支持训练时约定的五类预期数量，不能输出零件框或逐类计数。更改为其他有效数量向量时返回 `UNSUPPORTED_COUNTS`。现有 P05 工厂只处理 baseline2；在界面选择 baseline3 不等于已接入真实 YOLO。

## 二、本次修正内容

- P05 安装目标从 PyTorch 2.8.0 / TorchVision 0.23.0 改为组内拟用的 **2.7.1 / 0.22.1**。
- 增加组内共用依赖清单，固定 **Ultralytics 8.3.236**，baseline3 拟用 **YOLO11n**。
- 增加安装后检查脚本和 P05/P06/P07 交接说明。
- 保留历史测试记录的真实版本，不把旧环境测试改写为新环境已通过。

## 三、目标环境

| 软件 | 目标版本 |
|---|---|
| Python | 3.11.14 |
| NumPy | 1.26.4 |
| OpenCV：opencv-python | 4.10.0.84 |
| 相机枚举：cv2-enumerate-cameras | 1.4.0 |
| PyTorch | 2.7.1 |
| TorchVision | 0.22.1 |
| Ultralytics | 8.3.236 |

P05 还使用 SciPy 1.15.3、scikit-learn 1.9.1 和 Pillow 13.0.0，见 [P05 依赖](requirements-p05.txt)。以上是安装验证目标，由最终运行成员验证后再冻结。

## 四、先安装，再检查

所有命令均在**仓库根目录**执行，即能看到 `app.py`、`src` 和 `requirements-team.txt` 的目录。先激活队伍约定的 Python 环境。

### 1. 确认 Python

```powershell
python --version
```

应显示 `Python 3.11.14`。

### 2. 安装依赖

下面是 CPU 版本的安装命令。GPU 运行成员根据显卡和驱动选择官方 CUDA 安装源，torch 与 torchvision 必须安装配套构建。

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-team.txt
python -m pip check
```

不要把本文件的 CPU 命令直接用于已经选好的 CUDA 环境。GPU 安装依据见 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/#v2-7-1)。

### 3. 运行安装后检查

```powershell
python -m scripts.check_team_stack --device cpu --output output/team_stack_cpu.json
```

GPU 部署还需执行：

```powershell
python -m scripts.check_team_stack --device cuda:0 --output output/team_stack_cuda.json
```

脚本检查版本、依赖一致性、ResNet18 前向运算、TorchVision NMS、CLAHE 和随机初始化 YOLO11n 推理。结果写入指定 JSON；显示 `FAIL` 时按其中的错误修正环境。

随机模型测试只检查程序执行，不表示已经学会识别本项目零件。正式验证还需要 P06 的自定义训练权重与实拍数据。

### 4. 运行接口测试

```powershell
python -m unittest discover -s tests -v
```

历史环境的 31 项接口测试通过；最终运行机仍需自行运行并保存结果。

## 五、P05 训练需要队友先交哪些数据

P03 提供原图和两份 CSV，P08 确认五类固定预期数量。原图文件名与 CSV 的 `image_id` 必须一致。

| 文件 | P05 读取的必要字段 |
|---|---|
| ground_truth_images.csv | image_id、scene_id、split、annotation_status |
| ground_truth_objects.csv | image_id、object_id、class_id、ignore |

同一物理场景不能跨训练、验证与测试分区。P05 训练使用 train/val，忽略 test 的图像与目标记录；REVIEW 等不可判定开发样本不进入拟合或验证计分。

项目类别编号为：0 目录外件、1 长螺钉、2 粗螺钉、3 细螺钉、4 六角螺母、5 平垫圈。具体规格以组内冻结目录为准。

## 六、训练 baseline2

```powershell
python -m scripts.train_baseline2 --image-dir data/images --images-csv data/ground_truth_images.csv --objects-csv data/ground_truth_objects.csv --expected-counts 1 1 1 1 1 --output-dir weights/baseline2
```

五个 `1` 仅表示命令示例，正式运行应替换为团队约定的类别 1–5 预期数量。输出目录必须是尚不存在的新目录；重复实验用新的输出目录保存。

首次运行会下载 torchvision 官方 ResNet18 ImageNet 权重；离线运行可增加 `--weights 本地权重路径`。训练完成会保存骨干权重、分类参数、模型版本、数据哈希和验证结果。训练与推理的预处理均为保比例缩放、白边补齐至 224×224、ImageNet 归一化。

## 七、连接 P07 界面

先完成训练，再检查 `config/p07-baseline2.json` 中的 `baseline2_bundle`、`device` 和 `expected_counts` 是否与本次模型一致。

使用图片检查：

```powershell
python app.py --backend real --model baseline2 --config config/p07-baseline2.json --image data/example.png
```

使用相机检查：

```powershell
python app.py --backend real --model baseline2 --config config/p07-baseline2.json --camera-name Camo
```

`Camo` 只是相机名称示例，换成运行机的设备名。相机选择方法见 [P07 说明](README_P07.md)。模型目录不存在时，应先训练或接收完整模型包。

## 八、P08 如何调用 P05 批处理

先创建 `expected_counts.json`，使用以下结构；数量与图片名按实际数据填写：

```json
[
  {
    "image_id": "example.png",
    "expected_counts": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1}
  }
]
```

再执行：

```powershell
python -m scripts.p05_predict --input_dir data/test_images --model baseline2 --output_dir output/baseline2_run01 --counts_file expected_counts.json --bundle-dir weights/baseline2
```

只处理 JSON 列出的图片，输出目录必须是新目录。生成 `images.csv` 与 `run_config.json`。图像级分类器不生成虚假的目标框与逐类数量。P08 的公共入口仍需调用同一份适配器，见 [P05 接入说明](README_P05.md)。

## 九、与其他成员如何交接

| 成员 | 交给谁 | 需要交付 |
|---|---|---|
| P01 / P03 | P05 / P06 / P08 | 标定、原图、标注、场景切分与数据版本 |
| P05 | P07 / P08 | 完整 baseline2 模型包、配置、依赖、命令与验证记录 |
| P05 | P06 | 改进候选、适用假设、参数与同图对照要求 |
| P06 | P07 / P08 | 真实 YOLO 适配器、正式权重、类别映射、检测阈值、尺寸与拒识规则 |
| P07 | P08 | 界面联调、真实相机运行与至少 20 帧延迟记录 |
| P08 | 全组 | 统一运行 ID、封存测试集上的对照指标及冻结环境 |

YOLO 若采用训练编号 0–4，必须明确映射到项目编号 1–5；项目的 0 专门表示目录外件。完整接口约定见 [YOLO 交接说明](docs/p05/yolo_handoff.md)。

## 十、主要文件

| 文件 | 用途 |
|---|---|
| README_P05.md | P05 模型、接口与接入细节 |
| README_P07.md | 队友的相机与界面操作说明 |
| requirements-team.txt | 汇总 P05、P07 与 YOLO 的依赖 |
| scripts/check_team_stack.py | 最终运行机安装后的检查入口 |
| scripts/train_baseline2.py | baseline2 训练入口 |
| scripts/p05_predict.py | baseline2 批处理入口 |
| src/adapters/baseline2.py | 共用图像级模型适配器 |
| src/p05_bridge.py | P05 与 P07 的接口桥接 |
| src/stage3_clahe.py | 待对照验证的 CLAHE 改进候选 |
| docs/p05/yolo_handoff.md | P05 / P06 / P07 / P08 交接约定 |
| docs/p05/verification.json | 历史测试环境、接口证据与目标环境 |

## 十一、验证记录应如何理解

历史记录使用 Python 3.12 CPU、PyTorch 2.8.0、TorchVision 0.23.0，其中合成数据训练、重载与接口检查通过。该记录保留原值，方便追溯。

组内拟用的新技术栈、正式数据训练、YOLO 真实权重、CUDA 和真实相机延迟，要在最终运行机分别验证。算法改进需由 P08 在同一组独立场景上比较 baseline3 与 final，至少覆盖正常及两类困难条件，并记录误检、漏检、整图判定与时延。最终结论按实测结果填写。
