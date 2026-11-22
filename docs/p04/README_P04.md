# P04 / SW-01 · baseline1（HOG + scikit-learn MLP，图像级基线）

> 项目：Mechanical Component Vision Inspection（机械零件视觉检测）
> owner：P04 ｜ task_id：SW-01 ｜ 组别：程序与集成组
> 依赖上游：DATA-02（P03 标注/切分）、INT-01（P08 批处理/评估接口）｜ 补位：P05

## 1. 我负责什么

`baseline1` 是三个基线里**最简单的图像级（image-level）基线**：

- 手工提取 **HOG** 梯度方向直方图特征，输入 **scikit-learn MLP** 多层感知机；
- 对**整张图**输出 PASS / FAIL 整图判定 + 一个概率分数；
- **不输出**检测框、五类计数、ID 0 框（这些是目标级 `baseline3` / `final` 的能力，见团队计划 §5.5）；
- 只支持**一条固定预期数量向量**；输入其他合法预期数时返回 `UNSUPPORTED_COUNTS`，标签与分数留空。

方法定位：快、可复现、作为弱基线与 `baseline2`（ResNet18 冻结特征 + 逻辑回归）、`baseline3`（目标检测器）做同口径对比。

## 2. 接口合规对照（团队计划 §5，字段名/顺序/枚举未改动）

| 合同项 | baseline1 实现 |
|---|---|
| 批处理命令 | `python predict.py --input_dir TEST_FOLDER --model baseline1 --output_dir RESULTS --counts_file expected_counts.json` |
| model_id | `baseline1`（另有 `baseline2` / `baseline3` / `final`，由对应 owner 实现，本入口里为 stub） |
| `expected_counts.json` | 列表，每条 `image_id` + `expected_counts`，键仅 `"1"`–`"5"`、非负整数；ID 0 恒为 0 不写入 |
| `images.csv` | `image_id, model_id, status, predicted_label, prediction_score, score_type, non_catalogue_count` |
| `objects.csv` / `counts.csv` | **图像级基线免交**，不伪造六行零计数 |
| `status` 枚举 | `PASS / FAIL / REVIEW / UNSUPPORTED_COUNTS / ERROR`（REVIEW 本基线不产生） |
| `score_type` 枚举 | `prob / logit / none / 空`；本基线用 `prob`（MLP softmax 概率，**未校准**） |
| `run_config.json` | 含 `model, model_version, output_capability, class_catalogue, thresholds, preprocessing, coordinate_convention, score_convention, hardware, rejection, size_tolerance, calibration, per_image_expected_counts`，图像级另记 `supported_fixed_counts` |
| 能力级别 | `image_level`；不支持 ID 0 定位/拒识 |

## 3. 文件结构（P04 新增部分）

```
├─ predict.py                     # 统一批处理入口（按 --model 分发；baseline1 已实现）
├─ config/
│  └─ baseline1.json              # 冻结配置：目录、固定计数、HOG/MLP 参数、阈值、分数口径
├─ src/p04/
│  ├─ catalogue.py                # 五类目录 + ID 0 non_catalogue
│  ├─ features.py                 # 图像加载、灰度/缩放、HOG 特征
│  ├─ dataset.py                  # ground_truth_images.csv 加载与样本构建
│  ├─ model.py                    # MLP 训练 / bundle 保存加载 / 预测
│  ├─ pipeline.py                 # 配置加载、特征批量提取、校验和
│  └─ evaluate.py                 # 图像级 P/R/F1、混淆矩阵、延迟 median/P95
├─ tools/
│  ├─ make_demo_data.py           # 合成演示数据（仅跑通流水线，不是项目证据）
│  └─ train_baseline1.py          # 训练 + 冻结 bundle + 指标/延迟/清单存档
├─ models/baseline1/              # 冻结产物：bundle、冻结配置、metrics、清单、freeze 记录
├─ tests/test_p04.py              # 目录/特征形状/计数校验/训练-保存-加载-预测回归
├─ docs/p04/
│  ├─ README_P04.md               # 本文件
│  └─ advice_record.md             # LLM 建议与核对记录
└─ requirements-p04.txt
```

> 公共模块 `src/inspection.py`、`src/adapters/`、`src/evaluator.py` 属 P08（INT-01/EVAL-01）交付范围；P04 只提供本基线的适配器逻辑与权重配置，不替 P08 写公共判定规则。

## 4. 快速开始（Windows PowerShell，Python 3.11 x64）

```powershell
# 1. 安装依赖（建议先建团队 venv）
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements-p04.txt

# 2. 生成合成演示数据（真实数据到位前跑通用）
.\.venv311\Scripts\python.exe tools\make_demo_data.py --out data\demo

# 3. 训练并冻结 baseline1
.\.venv311\Scripts\python.exe tools\train_baseline1.py `
    --config config\baseline1.json `
    --gt-csv data\demo\ground_truth_images.csv `
    --images-dir data\demo\images `
    --output-dir models\baseline1

# 4. 批处理推理
.\.venv311\Scripts\python.exe predict.py `
    --input_dir data\demo\images\test `
    --model baseline1 `
    --output-dir outputs\demo-run `
    --counts_file data\demo\expected_counts.json

# 5. 回归测试
.\.venv311\Scripts\python.exe tests\test_p04.py
```

## 5. 接入真实数据（P02/P03 交付后）

真值清单沿用 DATA-02 合同 `ground_truth_images.csv`：

```
image_id, scene_id, split, width, height, annotation_status,
count_1, count_2, count_3, count_4, count_5, non_catalogue_count, label
```

- `split` 取 `train` / `val`（测试集由 P03 封存、P08 跑评估，训练阶段不碰）；
- 按 `scene_id` 整组切分，同一场景的帧/角度/光照变体只能落在同一分区；
- 训练命令替换 `--gt-csv`、`--images-dir` 指向真实数据即可；
- 固定参考预期向量目前是讨论起点 `[1,1,1,1,1]`，**待组会 D07 确认**；确认后改 `config/baseline1.json` 的 `supported_fixed_counts` 并重新冻结。

## 6. 冻结与交付节点（SW-01）

| 节点 | 交付物 |
|---|---|
| 11/17 | 初版跑通（HOG+MLP 可出整图判定） |
| **11/23** | **冻结**：权重、预处理、阈值、类别映射、判定规则；保存版本/校验和与预测文件 |
| 11/26 | Stage 2 随组提交（复现包含代码、依赖、冻结模型、配置、清单、原始预测） |
| 12/9 | 手算复核一例误判：记录 `image_id`、计算过程、核对结果 |
| 12/13 | 个人报告（≤2 页，五节）独立提交 |

冻结产物在 `models/baseline1/`：

- `baseline1.joblib`：MLP + 参数 + 元数据 bundle；
- `config.frozen.json`：冻结时的配置副本；
- `metrics.json`：验证集混淆矩阵、P/R/F1、balanced accuracy、延迟 median/P95；
- `samples.json`：本次训练/验证使用的 `image_id` 清单；
- `freeze.json`：冻结时间、配置/bundle 校验和、依赖版本、训练命令。

## 7. 报告 / PPT 分工（按 §8）

- **Stage 2 组报告**：§4 写 baseline1 方法与实现；§5 出 baseline1 图像级指标与响应时间（P08 核数字口径）；§6 写 baseline1 自己的失败案例，引用 `image_id` / run ID / 预测与真值。
- **PPT**：1 页素材，现场讲 30–40 秒：做法、表现、一个可核实结论 + 运行/图像 ID。
- **个人报告（12/13）**：自己写五节——个人具体贡献 / 一个技术决策（D）/ 一个亲自核验的结果（E）/ 局限与替代解释（E/T）/ 下一步工程动作（T）。

## 8. 当前局限与状态

- 图像级基线**看不到单个零件**，无法定位缺件/多件/错规格件，也不做 ID 0 拒识；
- 只支持固定预期向量，换向量即 `UNSUPPORTED_COUNTS`；
- 分数是未校准的 MLP 输出，不能当作校准概率；
- 仓库内附带的 bundle 是**合成演示数据**训练的占位产物，仅用于跑通流水线；**Stage 2 提交前必须用 P02/P03 真实数据重新训练并冻结**，演示指标不得作为项目证据；
- 真实场景表现、困难条件（反光/低照/遮挡/相似错规格件）结果待真实数据与联调后补。
