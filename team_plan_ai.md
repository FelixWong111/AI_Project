# 机械零件视觉检测 · 组内执行规格（AI 可读版）

> **这份文件是什么**：把整份文件直接投喂给组员的 AI 助手，让它知道项目是什么、谁负责什么、接口长什么样、什么算做完。
> **版本**：v1.0 · 2026-11-11 · 组内讨论稿（2026-11-12 组会确认后升 v1.1）
> **权威优先级**：`Project-Man.pdf`（课程要求） > 本文件 > 《机械零件视觉检测_8人实验实施手册.md》 > 群聊记录。
> 发现冲突：以高优先级为准，并在回答里指出冲突位置，不要自行折中。

## 0. 给 AI 的读取约定（MUST 遵守）

1. **先复述再动手。** 读完先用 5 行复述：项目目标 / 你要做的 task_id / 输入 / 输出 / 验收条件，等人确认。
2. **只用本文件编号。** 人员 `P01–P08`；任务 `HW-01`、`DATA-01`、`DATA-02`、`SW-01`…`SW-04`、`INT-01`、`EVAL-01`、`RES-01`；联调用例 `U01–U06`；决策 `D01–D07`；场景 `S2-0001`；运行 `run_id`。不得自创编号。
3. **未知就写 `TBD`。** 禁止编造性能数字、公差、阈值、日期、教师确认结果。
4. **结论必须带证据**：`run_id` / `image_id` / 文件路径 / 命令。没实际运行就写 `NOT_RUN`，不许把"应该能跑"写成"已通过"。
5. **只能改自己 owner 范围内的文件。** 要改公共文件（§5 接口、`src/evaluator.py`、报告模板）必须先经 P08 确认。
6. **不得改动 §5 的字段名、字段顺序、枚举值。** 这是跨人、跨组的合同。
7. 时间按 `Asia/Shanghai`，年份一律 2026。
8. 你不掌握硬件实况：相机、标定、公差、教师确认类结论只能写"待 P01/P02 实测确认"。
9. 回答用 §13 的固定模板。
10. 正文用中文；文件名、字段名、模型 ID 保留英文原文。

**一句话项目**：用 USB 相机拍托盘上的五类标准件（螺丝/螺母/垫圈），判断这一盘与"预期数量"是否一致（PASS/FAIL），同时数每类多少个、框出每个零件、认出目录外的陌生件（ID 0）。三套难度不同的方法做基线，选最好的一版加一个有文献依据的改进，最后做实时界面 + 报告 + 现场演示。

## 1. 项目元数据

| 项 | 值 |
|---|---|
| 课程 | Artificial Intelligence III: AI for Intelligent Engineering — Intelligent Manufacturing Engineering |
| 项目 | Mechanical Component Vision Inspection（机械零件视觉检测） |
| 人数 | 8 人（P01–P08），**组长/提交人/答辩主答未定**（见 §11） |
| 当前阶段 | Stage 2 准备（课程 Stage 1 组队 2026-11-04 已过，登记状态需核实） |
| 演示机 | Windows；Python 3.11 x64；`venv`；Git |
| 公共库 | OpenCV、NumPy、pandas；训练用 PyTorch/torchvision、ultralytics；标注用 Labelme；界面 OpenCV 先跑通，PySide6 为候选 |
| 主接口 | `predict.py` 批处理 + `app.py` 实时界面，共用推理、坐标还原、判定 |
| 关键日期 | 2026-11-26 Stage 2 提交；2026-12-13 Stage 3 提交（同日个人报告）；2026-12-16 现场 |

## 2. 标签、判定与命名（唯一真源）

### 2.1 类别目录

| class_id | name | 规格 | 说明 |
|---|---|---|---|
| 0 | `non_catalogue` | 类型或规格不符合目录 | 用于定位/拒识，**预期数恒为 0** |
| 1 | `long_screw` | M4 × 25 | |
| 2 | `thick_screw` | M5 × 10 | |
| 3 | `thin_screw` | M4 × 10 | |
| 4 | `hex_nut` | M4 | |
| 5 | `flat_washer` | M5 | |

### 2.2 PASS/FAIL 判定

```
PASS(image) :=  对 c ∈ {1,2,3,4,5}: observed[c] == expected[c]
                且 non_catalogue_count == 0
                且 该图可判定（非 REVIEW）
否则 FAIL
```

- **FAIL 是正类（positive class）**，所有混淆矩阵按此口径。
- 已确认的目录外零件 → 一定 FAIL（不是"不支持"）。
- 不可判定（反光/遮挡/截断到无法判断）→ 状态 `REVIEW`，不进 PASS/FAIL 分母，单独报告。
- 图像级基线（`baseline1`、`baseline2`）只能处理"它支持的固定预期数"；给别的有效预期数时返回 `UNSUPPORTED_COUNTS`，标签与分数为空。

### 2.3 匹配与 IoU

- 同类框按 `IoU ≥ 0.50` 一对一匹配后统计；匹配前按分数降序，同分按 `detection_id` 升序，确保可复现。
- 忽略标注（`ignore=1`）的实例不计入分母。

### 2.4 命名与编号

| 对象 | 规则 |
|---|---|
| 图片文件名 | `S2-0001_v01.jpg`（`S2`/`S3` 表示阶段，`0001` 场景号，`v01` 帧号） |
| `scene_id` | 同一摆放、不同帧/角度/光照变体共享一个 `scene_id`（同一摆放只算一个独立场景） |
| `image_id` | 即文件名，单次运行唯一 |
| `run_id` | 【组内约定·待 11/12 确认】`<model_id>-<YYYYMMDD>-<git7>`，例 `baseline3-20261123-a1b2c3d` |
| 运行目录 | 每次运行独立目录；同一图像改预期数分别写不同结果目录 |

## 3. 人员 → 任务 → DDL（权威表）

| task_id | owner | 任务 | 主要输出 | 组内 DDL | 依赖 | 补位 |
|---|---|---|---|---|---|---|
| HW-01 | P01 | 工装、照明、尺度标定 | `docs/rig_setup.md`、工位照片、参考图、`config/calibration.json`、教师确认记录 | 11/12 方案；11/14 实拍+标定初版；12/5 困难条件复核 | — | P02 |
| DATA-01 | P02 | 样件目录、场景设计、拍摄 | `capture_manifest.csv`、原图、场景清单 | 11/14 试拍；11/18 Stage 2 数据；12/4 Stage 3 新场景 | HW-01 | P01 |
| DATA-02 | P03 | 标注、质检、切分、封存 | `ground_truth_objects.csv`、`ground_truth_images.csv`、训练/验证清单 | 11/14 标注样例；11/18 划分与真值；12/5 封存 Stage 3 测试集 | DATA-01 | P02 |
| SW-01 | P04 | HOG + MLP 图像级基线 | `baseline1` 权重/配置/预测 | 11/17 初版；11/23 冻结；12/9 误判复核 | DATA-02、INT-01 | P05 |
| SW-02 | P05 | ResNet18 冻结特征 + 逻辑回归；Stage 3 文献与改进 | `baseline2`；`review/decision_stage2.md` 研究计划；改进模块 | 11/17 初版；11/23 冻结；11/25 研究计划；12/5 改进模块 | DATA-02、INT-01 | P04（基线）/ P06（改进） |
| SW-03 | P06 | 目标检测、ID 0 拒识、尺寸核验、`final` 集成 | `baseline3`、`final`、权重/配置/失败样例 | 11/17 初版；11/20 U03 联调；11/23 冻结 baseline3；12/6 冻结 final | HW-01、DATA-02、INT-01 | P07（接口）/ P05（改进） |
| SW-04 | P07 | 相机、界面、计数判定、实时延迟、演示操作 | `app.py`、取帧/判定模块、操作说明、时延表、备份视频 | 11/14 取帧；11/20 三模型联调；11/24 录视频；12/9 双版本+延迟 | INT-01、三基线 | P06（推理）/ P02（录制） |
| INT-01 | P08 | 仓库骨架、批处理、评估器、独立对照、合稿 | `predict.py`、`src/evaluator.py`、复现包、报告 | 11/12 接口；11/20 批处理+评估器；11/23 初评；11/25 Stage 2 定稿；12/9 独立对照；12/12 Stage 3 定稿 | — | P07（批处理）/ P04（评估）/ P03（资料） |
| EVAL-01 | P08 | 三基线共同测试与对照（与 INT-01 同人） | 评估表、原始预测、复跑记录 | 11/23 初评；12/7–12/9 独立对照 | 三基线冻结 | P04 |

**组内约定（2026-11-11 确定）**：报告与 PPT 由各任务 owner 写自己那部分；P08 只做拼装、统一格式、控页数、提交，不代写；个人报告各自写。

## 4. 任务验收条件

### HW-01 · P01
- 工装可复现：相机高度/距离/倾角变化后必须重新标定；记录颜色、材质、灯位、ROI 与标定 ID。
- 交 `docs/rig_setup.md`、工位全景照、参考图、`config/calibration.json`、教师对五类目录与尺寸容差的确认记录。
- 交接验收：P02 按标记重装一次能复原，P07 能取到新帧，P06 复核两种规格样件。

### DATA-01 · P02
- 先列场景再拍：正确数量、缺件、多件、五类全对+错规格件、仅 ID 0、空台面，覆盖五类；试拍 8–12 个开发场景。
- `capture_manifest.csv` 记录标定 ID、灯档、零件类型与规格、时间、场景难度。
- 独立测试集：≥30 个独立摆放场景，按同一固定预期向量 ≥15 PASS、≥15 FAIL（同一摆放多帧只算一个）。
- 12/4 交 Stage 3 新场景：正常 + 至少两种困难条件（建议低照、相似错规格件）。

### DATA-02 · P03
- `ground_truth_objects.csv` 字段：`image_id, object_id, class_id, xmin, ymin, xmax, ymax, ignore`。
- `ground_truth_images.csv` 字段：`image_id, scene_id, split, width, height, annotation_status` + 五类计数 + `non_catalogue_count` + 按预期数派生的 PASS/FAIL。空台面保留图像行；只有 ID 0 的图有 ID 0 框。
- 回画核对：每类抽代表图核对原图与转换框，覆盖 0、空场景、仅异常件、临边框；统一左上角原点、原图 `xyxy`。
- 切分按 `scene_id` 分组：同一场景的帧/角度/光照只能落同一分区（训练/验证/测试）。
- 11/18 交训练/验证数据与说明；Stage 2 测试真值单交 P08；12/5 封存 Stage 3 测试集（图像、真值、预期数、清单 + 版本/校验值）。

### SW-01 · P04
- `baseline1`：HOG + scikit-learn MLP，整图判定，仅支持固定预期数；否则 `UNSUPPORTED_COUNTS`。
- 保存随机种子、场景列表、训练命令、配置；11/23 冻结并交一条带 `image_id` 的误判样例。
- 12/9 手算复核一例误判，记录图像 ID、计算过程、核对结果。

### SW-02 · P05
- `baseline2`：torchvision 冻结 ResNet18 特征 + 逻辑回归，与 `baseline1` 共用同一图像/计数接口。
- 11/23 冻结模型、权重、预处理与固定预期数，附典型错误与分数含义。
- 11/25 与 P08 一起写决策记录：假设、主指标、最小有意义变化、可接受代价、验证方案。
- 12/5 向 P06 交改进模块、配置、运行命令与验证证据；说明"文献思想 → 本项目适配 → 代码位置"。

### SW-03 · P06
- 训练/加载/推理/回画走通，确认框回到原图坐标；用验证集选阈值。
- 先做五类 + 已采集 ID 0 的定位；OpenCV 提取未匹配前景候选并做尺寸核验，记录候选来源、与检测框去重方式、拒识依据（同一零件不得计两次）。
- 11/20 用实拍跑通 U03：五类数量全对 + 错规格件 → ID 0 框 + 警告 + FAIL。
- 11/23 冻结 `baseline3`；12/6 冻结 `final`（集成 P05 改进，界面可切换原版/改进版）。交两个可复跑版本及命令。

### SW-04 · P07
- 演示机接真实相机：取新帧、读分辨率、记录帧 ID、释放设备；11/14 保存取帧日志或短视频。
- 界面：三模型切换；每框类别与分数；实测/预期/差值；ID 0 数量与警告；判定；延迟；图像级模型显示整图标签与分数。
- 11/20 按 §5.7 用例联调并留证据；改预期数只重算判定；断连重连可恢复。
- 12/9 测实时延迟（≥20 新帧，中位数与 P95），交 `app.py`、模块、操作说明、原始时延表、备份视频。

### INT-01 / EVAL-01 · P08
- 11/12 前定模型 ID、路径、接口、配置约定；演示机可导入依赖、加载权重；`requirements.txt` 为实测通过版本。
- 11/20 前 `predict.py` + 独立 `src/evaluator.py` 初版；用人工小用例核对空台面、仅 ID 0、错规格、缺件、忽略标注、IoU 匹配。
- 11/23 用三个冻结版本跑 Stage 2 共同测试：保存 Git 版本、权重哈希、阈值、输入清单、原始预测、评估表；在干净环境按提交命令复跑。
- 12/7–12/9 独立对照：P03 核对测试集与标注，P04 核对结果，P07 测延迟。
- 11/25、12/12 分别合稿定稿；每条结论可追到 `run_id` 或原始文件。

## 5. 统一接口合同（不可自行修改）

### 5.1 批处理命令

```powershell
python predict.py --input_dir TEST_FOLDER --model baseline3 --output_dir RESULTS --counts_file expected_counts.json
```

### 5.2 `expected_counts.json`

```json
[
  {
    "image_id": "case01.jpg",
    "expected_counts": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1}
  }
]
```

- 模型 ID：`baseline1`、`baseline2`、`baseline3`、`final`。
- `expected_counts` 只含 1–5 的非负整数；ID 0 预期固定为 0（不写进该对象）。

### 5.3 输出文件（字段顺序固定）

`images.csv`
```
image_id, model_id, status, predicted_label, prediction_score, score_type, non_catalogue_count
```
`objects.csv`
```
image_id, model_id, detection_id, class_id, xmin, ymin, xmax, ymax, score
```
`counts.csv`
```
image_id, class_id, observed_count, expected_count, difference
```
- `difference = observed_count - expected_count`；目标级模型每张图覆盖 `class_id` 0–5。
- 图像级基线（`baseline1`、`baseline2`）只交实际支持的结果，`non_catalogue_count` 可为空。
- `status` 取值【组内约定·待确认】：`PASS` | `FAIL` | `REVIEW` | `UNSUPPORTED_COUNTS` | `ERROR`。
- `score_type`【组内约定·待确认】：`prob` | `logit` | `none`；无图像级置信度的规则判定用空值。

### 5.4 `run_config.json` 必含键

`model`、`model_version`、`output_capability`（`image_level` / `object_level`）、`class_catalogue`、`thresholds`、`preprocessing`、`coordinate_convention`、`score_convention`、`hardware`、`rejection`、`size_tolerance`、`calibration`、`per_image_expected_counts`；图像级模型另记 `supported_fixed_counts`。

### 5.5 模型能力矩阵

| 模型 | 级别 | 输出 | 固定预期数 | ID 0 |
|---|---|---|---|---|
| `baseline1` | 图像级 | 整图 PASS/FAIL + 分数 | 是（否则 `UNSUPPORTED_COUNTS`） | 不支持 |
| `baseline2` | 图像级 | 整图 PASS/FAIL + 分数 | 是（否则 `UNSUPPORTED_COUNTS`） | 不支持 |
| `baseline3` | 目标级 | 框 + 五类计数 + ID 0 | 否（任意合法向量） | 支持 |
| `final` | 目标级 | 同 `baseline3` + 改进 | 否 | 支持 |

### 5.6 数据流

原图 + 预期数 + 模型/配置 → `predict.py` → `images.csv`/`objects.csv`/`counts.csv`/`run_config.json` → `evaluator.py`（真值单独输入）→ 评估表。`app.py` 与 `predict.py` 必须共用同一套推理、坐标还原与判定代码。

### 5.7 11/20 相机联调用例（U01–U06）

| 用例 | 操作 | 期望表现 |
|---|---|---|
| U01 合格 | 五类按预期摆放 | 框与计数正确，PASS |
| U02 缺件/多件 | 取走或增加一件；另测预期为 0 的类 | FAIL，显示类别与数量差 |
| U03 数量对但有错规格件 | 保留五类预期数，另放相似错规格件 | 画出 ID 0 框，警告，FAIL |
| U04 可变预期数 | 同一帧输入两套合法预期数；再对含 ID 0 的帧重复 | 框与实测数不变，按新预期数重算；含 ID 0 持续 FAIL |
| U05 空台面/仅 ID 0 | 空台面用零向量；再放一个目录外件 | 空台面零计数；仅 ID 0 有框、目标计数为 0 且 FAIL |
| U06 不支持/不可判定 | 图像级模型输入不同预期数；再测坏图 | 明确状态、界面提示、保留记录 |

## 6. 数据与评估协议

### 6.1 分区

- 训练 / 验证 / 测试按 `scene_id` 整组切分，互不重叠。
- Stage 2 测试集：≥30 独立场景，≥15 PASS、≥15 FAIL，使用**同一固定参考预期数向量**（评估前定义并文档化，`[1,1,1,1,1]` 只是讨论起点）。
- Stage 3 测试集：新拍、未被用于调参，保留版本/校验值；至少两种困难条件单独统计。

### 6.2 必报指标

| 维度 | 口径 |
|---|---|
| 整图判定 | FAIL 为正类；混淆矩阵、P/R/F1、PASS/FAIL 样本数；单列误 PASS、误 FAIL、含 ID 0 的误 PASS |
| 定位分类 | 同类框 `IoU ≥ 0.50` 一对一；1–5 各类与 ID 0 的 P/R 与有效实例数；写明排序与同分规则 |
| 数量 | 从最终保留框计数；1–5 各类 MAE；五类向量全对比例；ID 0 数量 MAE（含零数量）；不可判定单独列 |
| 拒识副作用 | 目录外漏检数、合格件被误判 ID 0 数、含异常件却被接受的场景数 |
| 现场性能 | 每个被比较系统 ≥20 张实时新帧；帧可用 → 处理与显示更新完成（含预处理/推理/后处理，GPU 计时同步）；中位数、P95、分辨率、硬件、推理设置、模型文件大小；离线时间单列 |

### 6.3 独立对照责任链（Stage 3）

P03 保管测试集 → P05/P06 交冻结代码与配置 → P08 运行对照 → P04 复核结果 → P03 核对标注与数据版本 → P07 测实时延迟。P08 卡住由 P04 接跑。

### 6.4 改进有效性判据

11/25 前在 `review/decision_stage2.md` 约定：主指标（例：ID 0 定位召回率）、最小有意义提升、允许的合格件误拒、允许的 P95 增量、样本不足时的处理。12/7–12/9 只改计划中的因素，按条件报 `参考值 / 改进值 / 绝对差 / 场景数 / 有效实例数`；多因素用消融说明贡献。

## 7. 里程碑

| 组内节点 | 必须拿得出来 |
|---|---|
| 11/12 定接口 | 岗位认领、设备清单、计数合同、样图计划、`predict.py` 骨架 |
| 11/14 小批试拍 | 白底便携工位、新相机帧、标定初版、标注小样 |
| 11/18 开发数据 | 训练/验证清单、转换后框、Stage 2 测试清单 |
| 11/20 联调 | 三模型接口、实时画面、计数、ID 0 警告、批量输出、评估器初版、U01–U06 验收 |
| 11/23–25 Stage 2 内部收件 | 冻结基线、原始预测、视频、决策记录、可复现提交包 |
| 12/5–12/9 Stage 3 内部收件 | 测试集封存、两版本冻结、对照结果、实时证据 |
| 12/10–15 汇报收件 | 12/11 收 8 人 PPT 素材与报告初稿；12/12 定稿；12/14–15 彩排 |

课程节点：

| 日期 | 交付 |
|---|---|
| 2026-11-26 Stage 2 | ≤10 页组报告（`format.docx`，九节）+ <2 分钟 MP4 + 三冻结模型与复现包 + ≤1 页 `review/decision_stage2.md`；命名 `AI Project Stage 2 Report Group XX` |
| 2026-12-13 Stage 3 | ≤15 页组报告（十二节）+ 三个未改基线 + `final` + 备份视频 + 工位照片 + 复现包 + ≤1 页 `review/decision_final.md`；命名 `AI Project Stage 3 Material Group XX` |
| 2026-12-13 个人 | 每人 ≤2 页个人报告（五节：自己做了什么 / 一个技术决策 / 亲自复核的结果 / 替代解释 / 下一步）；命名 `AI Project Student ID Individual Report Group XX`；保密贡献反馈另交 |
| 2026-12-16 现场 | 全员到场，先交打印的 Stage 3 报告；带工装、相机、电脑、电源、模型、备份视频；5 分钟汇报 + 4 分钟问答 + 8 分钟含装调实时演示 |

## 8. 报告 / PPT / 个人报告分工

**总规则：谁做的部分谁写、谁写谁讲；P08 只拼装、统一格式、控页数、提交。**

### 8.1 Stage 2 组报告（九节，≤10 页）

| 节 | 标题 | 主笔 |
|---|---|---|
| 1 | Engineering Task and Physical Setup | P01 |
| 2 | Classes, Data Sources and Annotations | P02 + P03 |
| 3 | Data Partitions and Evaluation Protocol | P03 + P08 |
| 4 | Three Baseline Methods and Implementation | P04 / P05 / P06 各写自己 |
| 5 | Baseline Results and Response Time | P04（P08 核数字口径） |
| 6 | Weaknesses and Failure Analysis | 各模型自己写失败案例，P08 汇总 |
| 7 | Initial Decision Record and Improvement Plan | P05 |
| 8 | Advice, Reproducibility and Completion Status | P08 |
| 9 | LLM Use | 各人交记录，P08 汇总 |

### 8.2 Stage 3 组报告（十二节，≤15 页）

| 节 | 标题 | 主笔 |
|---|---|---|
| 1–2 | Engineering Purpose and Final System；Frozen Baselines and Chosen Reference | P06 |
| 3 | Data, Annotations and Independent Test Protocol | P03 |
| 4–5 | Research Source and Adaptation；Hypothesis and Decision Record Revisions | P05 |
| 6 | Actual Implementation and Advice Audit | P05 + P06 |
| 7–8 | Controlled Results and Difficult Conditions；Failures and Alternative Explanations | P08 |
| 9 | Live Performance and Resource Tradeoffs | P07 |
| 10–12 | Final Decision, Limitations and Next Action；Reproducibility and Contributions；LLM Use | P08（§11 复现清单 P03 维护，§12 各人记录 P08 汇总） |

### 8.3 PPT 与个人报告

- 12/11 前每人交 **1 页可编辑素材**：图或表 + 一个可核实结论 + 运行/图像 ID + 20–40 秒说明；P02 统合；P08 核对数据。
- 组内 5 分钟汇报：每人 1–2 页讲自己那 30–40 秒（8 人约 4.5 分钟 + 开场/收尾）；建议 P01 讲工位与任务、P05 讲基线对比与研究改进、P07 实操实时演示。
- 个人报告：每人自己写 ≤2 页，12/13 交；引用同一组数字（P08 维护"数字 → run_id"口径表）。

## 9. 决策记录模板

`review/decision_stage2.md`（≤1 页）：

```yaml
decision_id: "ST2-001"
date: "2026-11-25"
owner: "P05"
weakness: "<基于哪个失败案例，附 image_id>"
evidence: ["<run_id>", "<文件路径>"]
literature: "<文献/技术研究>"
borrowed_idea: "<借鉴的思想>"
adaptation: "<如何适配到本项目>"
hypothesis: "<假设>"
primary_metric: "<主指标>"
min_meaningful_change: "<最小有意义变化>"
allowed_cost: "<允许的误拒/P95 增量>"
validation_plan: "<怎么验证>"
risks: "<风险与限制>"
```

`review/decision_final.md`（≤1 页）：追加 `implemented_version`、`run_id`、`result`、`final_choice`、`limitations`。

## 10. 仓库结构与交付物

```
.
├─ src/camera.py             # 取帧
├─ src/inspection.py         # 推理 + 坐标还原 + 判定（app.py / predict.py 共用）
├─ src/evaluator.py          # 独立评估器（只读真值）
├─ src/adapters/             # baseline1/2/3、final 适配器
├─ app.py                    # 实时界面
├─ predict.py                # 批处理入口
├─ config/calibration.json
├─ config/expected_counts.json
├─ docs/rig_setup.md
├─ data/                     # 原图、标注、清单（大文件不入 Git 或走 LFS）
├─ weights/                  # 权重 + 哈希
├─ review/tech_stack.md
├─ review/decision_stage2.md
├─ review/decision_final.md
└─ README.md                 # 复现命令
```

**复现包至少含**：代码、依赖、配置与权重、类映射、分区清单、真值与标注转换、评估器、原始预测、运行命令；保留一条实际采用或修改的 AI 建议及核对证据（与自己的操作和结果分开记录）。

## 11. 待确认事项（2026-11-12 组会）

1. D01 分工与补位认领；D02 演示机环境与依赖版本；D03 模型选型。
2. D04 标注规则（Labelme、双人对照）；D05 界面方案（OpenCV → PySide6？）；D06 工位与教师确认；D07 固定预期数向量与共用接口。
3. **组长 / 课程提交人 / 答辩主答**（课程要求 11/04 前指定组长，登记状态需核实）。
4. `format.docx` 报告模板的章节与页数口径。
5. 类目录（class_id / name / 示例图 / 特征 / 尺寸）作为正式交付物。
6. Stage 3 独立测试集"未见相似规格"的覆盖方案。
7. `status` / `score_type` / `run_id` 三个组内约定（§2.4、§5.3）。

## 12. 机器可读总表（JSON）

> 本块是给程序/AI 读取的结构化真源；上面的表格是它的可读版本。手改时两处同步。

```json
{
  "schema": "team-plan/v1",
  "version": "2026-11-11.1",
  "status": "pending_group_confirmation",
  "timezone": "Asia/Shanghai",
  "course": "Artificial Intelligence III: AI for Intelligent Engineering - Intelligent Manufacturing Engineering",
  "project": "Mechanical Component Vision Inspection",
  "group_size": 8,
  "deadlines": {
    "stage1_team_registration": "2026-11-04",
    "internal_kickoff": "2026-11-12",
    "stage2_delivery": "2026-11-26",
    "stage3_delivery": "2026-12-13",
    "individual_report": "2026-12-13",
    "onsite_demo": "2026-12-16"
  },
  "classes": [
    {"id": 0, "name": "non_catalogue", "spec": "type/spec not in catalogue"},
    {"id": 1, "name": "long_screw", "spec": "M4x25"},
    {"id": 2, "name": "thick_screw", "spec": "M5x10"},
    {"id": 3, "name": "thin_screw", "spec": "M4x10"},
    {"id": 4, "name": "hex_nut", "spec": "M4"},
    {"id": 5, "name": "flat_washer", "spec": "M5"}
  ],
  "roles": [
    {"id": "P01", "group": "hardware_data", "tasks": ["HW-01"], "backup": "P02"},
    {"id": "P02", "group": "hardware_data", "tasks": ["DATA-01"], "backup": "P01"},
    {"id": "P03", "group": "hardware_data", "tasks": ["DATA-02"], "backup": "P02"},
    {"id": "P04", "group": "software", "tasks": ["SW-01"], "backup": "P05"},
    {"id": "P05", "group": "software", "tasks": ["SW-02", "RES-01"], "backup": "P04"},
    {"id": "P06", "group": "software", "tasks": ["SW-03"], "backup": "P07"},
    {"id": "P07", "group": "software", "tasks": ["SW-04"], "backup": "P06"},
    {"id": "P08", "group": "software", "tasks": ["INT-01", "EVAL-01"], "backup": "P07"}
  ],
  "tasks": [
    {"id": "HW-01", "owner": "P01", "backup": "P02", "name": "rig, lighting, scale calibration",
     "due": ["2026-11-12", "2026-11-14", "2026-12-05"], "depends_on": [],
     "outputs": ["docs/rig_setup.md", "config/calibration.json", "rig photos", "teacher confirmation"]},
    {"id": "DATA-01", "owner": "P02", "backup": "P01", "name": "sample catalogue, scene design, capture",
     "due": ["2026-11-14", "2026-11-18", "2026-12-04"], "depends_on": ["HW-01"],
     "outputs": ["capture_manifest.csv", "raw images", "scene list"]},
    {"id": "DATA-02", "owner": "P03", "backup": "P02", "name": "annotation, QC, split, seal",
     "due": ["2026-11-14", "2026-11-18", "2026-12-05"], "depends_on": ["DATA-01"],
     "outputs": ["ground_truth_objects.csv", "ground_truth_images.csv", "train/val split"]},
    {"id": "SW-01", "owner": "P04", "backup": "P05", "name": "HOG + MLP image-level baseline",
     "due": ["2026-11-17", "2026-11-23", "2026-12-09"], "depends_on": ["DATA-02", "INT-01"],
     "outputs": ["baseline1 model/config/predictions"]},
    {"id": "SW-02", "owner": "P05", "backup": "P04", "name": "ResNet18 features + logistic regression; Stage 3 improvement",
     "due": ["2026-11-17", "2026-11-23", "2026-11-25", "2026-12-05"], "depends_on": ["DATA-02", "INT-01"],
     "outputs": ["baseline2", "review/decision_stage2.md research plan", "improvement module"]},
    {"id": "SW-03", "owner": "P06", "backup": "P07", "name": "object detection, ID0 rejection, size check, final integration",
     "due": ["2026-11-17", "2026-11-20", "2026-11-23", "2026-12-06"], "depends_on": ["HW-01", "DATA-02", "INT-01"],
     "outputs": ["baseline3", "final", "weights/config/failure samples"]},
    {"id": "SW-04", "owner": "P07", "backup": "P06", "name": "camera, UI, counting rules, latency, demo",
     "due": ["2026-11-14", "2026-11-20", "2026-11-24", "2026-12-09"], "depends_on": ["INT-01", "SW-01", "SW-02", "SW-03"],
     "outputs": ["app.py", "frame/decision modules", "manual", "latency table", "backup video"]},
    {"id": "INT-01", "owner": "P08", "backup": "P07", "name": "repo, batch pipeline, evaluator, report assembly",
     "due": ["2026-11-12", "2026-11-20", "2026-11-23", "2026-11-25", "2026-12-09", "2026-12-12"], "depends_on": [],
     "outputs": ["predict.py", "src/evaluator.py", "repro package", "reports"]},
    {"id": "EVAL-01", "owner": "P08", "backup": "P04", "name": "common test runs and independent comparison",
     "due": ["2026-11-23", "2026-12-09"], "depends_on": ["SW-01", "SW-02", "SW-03"],
     "outputs": ["evaluation tables", "raw predictions", "rerun logs"]}
  ],
  "interface": {
    "cli": "python predict.py --input_dir TEST_FOLDER --model <model_id> --output_dir RESULTS --counts_file expected_counts.json",
    "model_ids": ["baseline1", "baseline2", "baseline3", "final"],
    "expected_counts_keys": "1-5 non-negative integers; ID 0 fixed at zero",
    "images_csv": ["image_id", "model_id", "status", "predicted_label", "prediction_score", "score_type", "non_catalogue_count"],
    "objects_csv": ["image_id", "model_id", "detection_id", "class_id", "xmin", "ymin", "xmax", "ymax", "score"],
    "counts_csv": ["image_id", "class_id", "observed_count", "expected_count", "difference"],
    "difference": "observed_count - expected_count",
    "status_enum": ["PASS", "FAIL", "REVIEW", "UNSUPPORTED_COUNTS", "ERROR"],
    "pass_rule": "for c in 1..5 observed[c]==expected[c] and non_catalogue_count==0 and status != REVIEW"
  },
  "metrics": ["image-level FAIL P/R/F1 + confusion matrix", "object IoU>=0.50 per class P/R", "count MAE and full-vector exact ratio", "ID0 count MAE", "false PASS / false FAIL", "median and P95 latency over >=20 new frames", "resource usage"],
  "report_sections": {
    "stage2": [
      {"section": "1", "title": "Engineering Task and Physical Setup", "owner": "P01"},
      {"section": "2", "title": "Classes, Data Sources and Annotations", "owner": "P02+P03"},
      {"section": "3", "title": "Data Partitions and Evaluation Protocol", "owner": "P03+P08"},
      {"section": "4", "title": "Three Baseline Methods and Implementation", "owner": "P04+P05+P06"},
      {"section": "5", "title": "Baseline Results and Response Time", "owner": "P04"},
      {"section": "6", "title": "Weaknesses and Failure Analysis", "owner": "all models + P08"},
      {"section": "7", "title": "Initial Decision Record and Improvement Plan", "owner": "P05"},
      {"section": "8", "title": "Advice, Reproducibility and Completion Status", "owner": "P08"},
      {"section": "9", "title": "LLM Use", "owner": "all + P08"}
    ],
    "stage3": [
      {"section": "1-2", "title": "Engineering Purpose and Final System; Frozen Baselines and Chosen Reference", "owner": "P06"},
      {"section": "3", "title": "Data, Annotations and Independent Test Protocol", "owner": "P03"},
      {"section": "4-5", "title": "Research Source and Adaptation; Hypothesis and Decision Record Revisions", "owner": "P05"},
      {"section": "6", "title": "Actual Implementation and Advice Audit", "owner": "P05+P06"},
      {"section": "7-8", "title": "Controlled Results and Difficult Conditions; Failures and Alternative Explanations", "owner": "P08"},
      {"section": "9", "title": "Live Performance and Resource Tradeoffs", "owner": "P07"},
      {"section": "10-12", "title": "Final Decision, Limitations and Next Action; Reproducibility and Contributions; LLM Use", "owner": "P08"}
    ]
  },
  "open_items": [
    "group leader / submitter / main respondent not assigned",
    "stage1 registration status on 2026-11-04 to confirm",
    "format.docx template and page limits",
    "class catalogue deliverable (class_id/name/sample/features/dimensions)",
    "fixed reference expected-counts vector to document before evaluation",
    "unseen-spec coverage for stage 3 hidden test",
    "status/score_type/run_id conventions"
  ]
}
```

## 13. AI 回答模板

```markdown
## 我理解的任务
- task_id / owner / DDL：
- 输入（存在 / 缺失）：
- 我要产出的文件：

## 我准备怎么做
1. …
2. …

## 验收自检
- [ ] 对照 §4 的验收条件
- [ ] 对照 §5 接口字段（未改动）
- [ ] 证据：run_id / image_id / 路径

## 需要人确认
- …
```

## 14. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-11-11 | 初版：从《实施手册》+ `Project-Man.pdf` 提炼为 AI 可读结构；加入组内约定（各自负责报告/PPT、P08 只拼装） |
