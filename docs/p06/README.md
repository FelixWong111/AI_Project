# P06 / SW-03 数据工具与 GPU 验证

## 最新交付：冻结模型推理与六类计数

当前权重已完成长训练、GT 修正后复核及 640/960 对照，保留 YOLO11n 640。P06 专属工厂为 `src.p06_inference:create_backend`，使用 `config/p06_p07_detection_only.json` 可接现有 P07；仅检测/计数，状态 REVIEW，尚不判 PASS/FAIL。已完成真实 GPU 46 图及 P07 headless 验证。接口调用、生命周期和命令见 [inference.md](inference.md)，实际证据见 `output/p06_inference_checks/`。

下文环境冒烟、早期训练入口说明属于先前阶段记录。当前源 GT 已修正，旧训练入口的冻结验收哈希会拒绝直接继续；本轮没有训练或绕过该校验。

## 当前入口：P03 新交付 Train/Validation

旧 51/326 张流程涉及污染数据，模型与派生目录已经按用户批准清理，不能作为正式结果或初始化来源。当前 `tools/p06_train_baseline.py` 只接收已验收的 `data/p03/development_v2` 及对应哈希/派生清单，从官方 `output/p06-cache/yolo11n.pt` 开始。

操作与限制见 [p03_training.md](p03_training.md)。入口必须显式指定 `--dry-run`、`--check-data`、`--smoke` 或 `--train`，无参数不会启动长训练。本轮只执行 1 Epoch 工程冒烟。跨 Test 的独立性与历史曝光仍由 P03/P08 核查，不把此工程检查当成无泄漏认证。

以下章节保留历史开发记录；其中路径、数量、状态和命令不是当前正式操作说明。

## 历史：首版 51 张派生流程（非正式结果）

已按用户确认的每图独立重新摆放记录，生成 51 个 **P06_TEMP 暂定**场景标识（待 P03/P08 审核），固定 Train 41 / Validation 10。候选清单先检查约束再构建；不重采样。新增入口为 `tools/p06_build_dataset.py`，复用已有 EXIF 显示图加载、Labelme 解析和 to_yolo/from_yolo；原严格 EXIF 检查器未关闭。

```powershell
conda run -n ai-p06-yolo python -B tools/p06_build_dataset.py plan --input data/p03/development --output output/p06/yolo_dataset_candidate
conda run -n ai-p06-yolo python -B tools/p06_build_dataset.py build --plan output/p06/yolo_dataset_candidate/source_plan.json --output output/p06/yolo_dataset
conda run -n ai-p06-yolo python -B -m unittest discover -s tests -p "test_p06*.py" -v
```

这些输出已经存在，重跑必须另选新目录，不能覆盖。正式数据集位于 `output/p06/yolo_dataset/`：51 个完整分辨率、已校正方向的普通 PNG，51 个 TXT，244 实例。Train 类别计数 `[3,40,39,40,39,39]`；Validation `[0,8,9,8,10,9]`。fail13 在 Validation，TXT 为零字节。

`dataset.yaml` 使用 `path: .` 和相对图片路径，**Ultralytics 8.3.236 使用时须从数据集根目录运行**（或调用前设置 cwd 为该目录）。这样复制整个数据集后不需修改机器绝对路径。无 test/download 配置，不读取独立测试集。

临时目录完整构建并通过几何/数量/像素/官方加载检查后才整体发布。新增 7 项测试，完整 P06 为 37 项通过；官方 YOLODataset 以 augment=False、cache=False 遍历全部 51 样本，无模型训练。详情见数据集 `dataset_report.md`、`validation/checks.json`、`validation/verification.json` 和回画对比目录。修复 Windows 临时目录 ACL 继承及字体缓存路径后已重新验证；所有源文件 SHA-256 不变。

**Validation 不含 ID 0，不能计算 ID 0 召回率或用它优化拒识阈值；没有完成 ID 0 拒识验证。** 总开发集只有三个 ID 0 实例、单一拍摄工况。技术上已具备准备启动 YOLO11n 训练的数据条件，仍等待人工审核，本轮没有训练。

## 第二批当前状态（2026-11-22）

环境阻塞已通过独立 Conda 环境 `ai-p06-yolo` 解除，路径 `C:/Users/26407/anaconda3/envs/ai-p06-yolo`。Python **3.11.18**，与原建议的 3.11.14 有补丁号差异，已按本轮允许 3.11.x 实际验证。没有改动 base/yunyi 的包或系统 PATH、全局环境变量、代理。

实际直接依赖：NumPy 1.26.4、opencv-python 4.10.0.84（cv2 4.10.0）、cv2-enumerate-cameras 1.4.0、torch 2.7.1+cu126、torchvision 0.22.1+cu126、Ultralytics 8.3.236、Pillow 10.4.0。`pip check` 返回 0，输出 `No broken requirements found.`；完整 Python 包版本快照为 `requirements-p06-lock.txt`，不是跨平台 Conda 锁文件。

真实 CUDA Runtime **12.6**，`torch.cuda.is_available()=true`，GPU 为 RTX 4060 Laptop GPU。输入和结果张量都在 `cuda:0`，`sum([1,2,3]**2)=14.0`。官方 COCO 80 类 YOLO11n 在合成 384×640 图片上执行两次真实 GPU 推理（conf=.25 和 1.0），均返回空检测；boxes.xyxy 为 `(0,4)`，cls/conf 为 `(0,)`，空结果正常处理。记录模型参数、forward 输入和 Boxes 数据均为 `cuda:0`，没有 CPU 回退。

原图坐标 API 和尺寸结构通过检查；另以**明确标注为合成的已知框**验证 Ultralytics scale_boxes 从 640×640 letterbox 恢复到 384×640 原图 `(100,80,250,180)`。它不是模型预测，空检测不能证明非空目标定位质量。没有使用 P03 图片或保留测试集，没有项目模型准确率、正式训练或阈值选择。

证据：`gpu-smoke.json`（实际版本、pip check、CUDA 和推理）、`torch-download.json`（官方索引 SHA-256）、`gpu-verification.json`（命令和测试记录）、`code-review.md`（九文件审查和最小修复）。历史 base/yunyi 探测文件和第一批 verification.json 保持原样，不代表新环境状态。

下载/运行问题及处理：

- 初次官方 pip torch 下载停留在零字节，主动停止该任务专属 pip 进程（退出 1），并保留原日志 `output/p06-cache/install-cu126-interrupted.log`。随后从同一官方索引的链接分段下载单个 wheel，162 个范围都检查 Content-Range，合并后的 SHA-256 与官方值完全一致，再安装本地 wheel 和官方 torchvision。没有依赖冲突或更换第三方包源。
- 首次 GPU 推理本身通过，但配置目录父目录尚未创建，Ultralytics 回退到仓库根目录。已将本轮生成的配置移入忽略目录，并修复脚本预建父目录和验证实际配置路径；复跑通过。当前所有缓存、合成图片、wheel 和 YOLO11n 权重都在 Git 忽略的 `output/p06-cache/` 中，只有进程内设置 YOLO_CONFIG_DIR/TORCH_HOME。
- 代码审查发现转换写盘中断会留下部分正式输出。已用失败注入测试先复现，再改为同卷临时目录完整写入后发布；失败时清理本次临时目录，已有输出仍不覆盖。24 项 P06 测试在新环境通过。

复查命令（已有环境不要重新创建或覆盖；--output 必须用新文件路径）：

```powershell
conda run -n ai-p06-yolo python -m pip check
conda run -n ai-p06-yolo python -B -m unittest discover -s tests -p test_p06_data.py -v
conda run -n ai-p06-yolo python -B tools/p06_gpu_smoke.py --weights output/p06-cache/yolo11n.pt --cache output/p06-cache --output output/p06-smoke-rerun.json
```

后续先接收 P03 开发原图、Labelme JSON、真实 scene_id 和 train/val 清单，检查后回画人工复核。尚未训练本项目六标签检测器、实现公共 bridge、设置尺寸公差或做测试集评估。等待人工审核后再推进。

## 第一批记录（历史，不代表第二批环境状态）

本批仅实现 Labelme rectangle 检查、YOLO 标签转换和合成夹具验证。没有训练、真实零件推理、公共 bridge、尺寸规则或测试集调参。课程依据为 `Project-Man.pdf`，其次为 `team_plan_ai.md` 和 `README_P07.md`。P07 说明中的接入格式作为后续目标；不复制其 Mock 判定为正式实现。

## 实际环境结果（2026-11-22，Asia/Shanghai）

- 分支 `feat/p06-baseline3`，开始时工作区干净。
- `py -0p` 只登记 Python 3.9；检查已知 Conda 环境、常见安装目录及注册表，未找到可用 Python 3.11。仓库 `.venv311` 不存在。
- 默认解释器：`C:/Users/26407/anaconda3/python.exe`，Python 3.12.7、NumPy 1.26.4、Pillow 10.4.0；没有 torch/torchvision/OpenCV/Ultralytics。
- `yunyi`：Python 3.10.18、NumPy 2.2.6、torch 2.7.1+cpu；没有 torchvision/OpenCV/Ultralytics。`torch.version.cuda=null`、`cuda.is_available()=false`，CUDA 检查 FAIL；未进行 GPU 张量运算，也没有 CPU 回退冒充 GPU。
- `nvidia-smi` 退出 0：RTX 4060 Laptop GPU、8188 MiB、驱动 592.82。驱动能看到 GPU 不等于 PyTorch CUDA 已可用。
- `python -m pip --version`：pip 24.2。`pip config debug` 未发现 pip.ini 包源配置。
- 官方 PyPI 和官方 cu126 索引的 `pip index versions` 查询在网络授权后均退出 0；首次沙箱 PyPI 查询退出 1，不能把该失败当作官方缺包。查询只读取索引，没有安装。
- 阶段 A 状态：BLOCKED（缺 Python 3.11）；未创建替代版本环境，未改全局 Python、base 或 yunyi。目标环境安装、目标环境 `pip check`、YOLO11n 权重下载/加载/真实 GPU 推理均为 NOT_RUN。
- `environment-base.json`、`environment-yunyi.json` 是实际探测记录；探测命令均退出 1，正确报告环境不满足要求。

## 环境解除阻塞后的步骤（本轮未执行）

需要提供已安装的 Python 3.11 完整路径（优先 3.11.14）。不要将下面命令指向 base/yunyi。可在项目已忽略的 `.venv311` 建环境；若改用仓库外 P06 专属目录，先确认写入位置，不更改公共 `.gitignore`。

```powershell
& '已确认的Python311路径/python.exe' -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv311\Scripts\python.exe -m pip install -r docs/p06/requirements-p06.txt --index-url https://pypi.org/simple
.\.venv311\Scripts\python.exe -m pip check
.\.venv311\Scripts\python.exe -B tools/p06_check_env.py --output output/p06-env.json
```

`requirements-p06.txt` 是拟定直接依赖清单，尚未安装解析或验证，不是完整 transitive lock。Pillow 用于成熟图像解码，当前仅用其已有版本验证数据工具。环境探测脚本只检查导入、版本和真实 CUDA 张量，**不执行 YOLO 推理**；后续必须单独补 YOLO11n GPU 冒烟，验证 boxes.xyxy/cls/conf、原图尺寸、数值有限性并保存记录。通用 YOLO11n 的 COCO 类别不能解释为本项目六标签。

官方依据：

- [PyTorch 2.7.1 / torchvision 0.22.1 cu126 安装说明](https://pytorch.org/get-started/previous-versions/)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics 推理输出](https://docs.ultralytics.com/modes/predict/)

## 数据约定

映射固定且无隐式偏移：`0=non_catalogue, 1=long_screw, 2=thick_screw, 3=thin_screw, 4=hex_nut, 5=flat_washer`。拒识标签 0 的预期数恒为零。工具不生成 PASS/FAIL，不读取模型预测。

输入是一个平铺目录，每张 PNG/JPG/JPEG/BMP 与同 stem 的 Labelme JSON 配对，`imagePath` 必须是图片 basename。不支持嵌套目录、绝对路径、跨目录引用、符号链接或仅 imageData 没有原图的输入。检测同 stem 多图片，避免生成 TXT 时覆盖。

仅支持 rectangle 两点标注，允许反向拖框。原图像素采用连续 `xyxy` 边界：`0 <= xmin < xmax <= width`，`0 <= ymin < ymax <= height`；此为 P06 转换约定，最终与 P03/P08 对齐。检查实际图像解码和 JSON 宽高；有非平凡 EXIF orientation 的图片先拒绝，需 P03 规范方向并重核标注。

转换公式：`cx=(xmin+xmax)/(2W), cy=(ymin+ymax)/(2H), w=(xmax-xmin)/W, h=(ymax-ymin)/H`。TXT 保留 12 位有效数字。`from_yolo` 反算只对超边界不超过 **1e-7 像素**的序列化舍入误差作贴边修正；原始标注没有自动裁框。

拦截未知类别、非 rectangle、非法 schema、NaN/Inf、bool 坐标、零面积、越界、尺寸不符、同框重复及同框异类冲突。近似重叠框不等同于确定重复，不自行删框；仍需 P03 人工质检。

Labelme 常规空 flags 接受。出现顶层/shape 的 `annotation_status`、`ignore`、非空 flags 或非空 group_id 时，按语义未约定报错，不推测含义。此策略适用于当前交接前状态；P03/P08 确认规则后再更新。尤其不能自动清除 REVIEW/ignore 标记以通过检查。

空 shapes 默认报错。只有经人确认的空场景才用 `--confirmed-empty IMAGE_ID` 显式放行；此参数记录在操作者命令中，不代表程序能证明空场景。缺 JSON 永远不是空台面。

## CLI

以下是用法示例，`DEV_LABELME` 必须指向开发数据而非保留测试集；当前没有真实开发数据可运行。输出目录和 report 文件必须为新路径，report 不可放在源目录中，父目录须已存在。

```powershell
python -B tools/p06_prepare.py --help
python -B tools/p06_prepare.py check --input DEV_LABELME --report output/p06-check.json
python -B tools/p06_prepare.py convert --input DEV_LABELME --output output/p06-labels --report output/p06-conversion-check.json
python -B tools/p06_prepare.py convert --input DEV_LABELME --manifest DEV_MANIFEST.csv --output output/p06-partitioned
python -B tools/p06_prepare.py check --input DEV_LABELME --confirmed-empty empty.png
```

退出码：`0` 检查/转换成功；`1` 数据检查不通过；`2` CLI/路径/写入错误。错误 JSON 含文件名、错误类型、信息和处理建议；每个异常文件先报告首个问题，修复后重跑。

产物：

- `labels/unpartitioned/*.txt`：无 manifest 时的转换结果，明确未验证分区；不是正式训练集。
- `labels/train/*.txt`、`labels/val/*.txt`：有完整合法 manifest 时输出，test 项不导出。
- `objects.json`：保留原图 xyxy、object_id、class_id、ignore=0、图片尺寸、来源路径、像素哈希及已提供的 scene/split，为未来真值 CSV 转换准备；无图像判定标签。
- `conversion.json`：映射、边界约定、分区是否核验、导出/排除数量。`training_ready=false`，因为本工具不复制图片、不构造训练 yaml、不确认人工标注正确性。

原始 JSON 和图像不写入。已有输出目录（包括空目录）拒绝覆盖；任一数据错误会阻止整批导出。导出先写同卷临时目录，全部成功后才在 Windows 上整体重命名发布。I/O 中途失败会清理本次临时目录，不留下不完整的正式输出目录；已有输出仍拒绝覆盖。

## 可选分区检查

CSV 至少含 `image_id,scene_id,split`，split 只接受 `train/val/test`。只校验 P03 已提供的真实分区，不随机划分。不带 manifest 时 `partition_verified=false`；不从目录名或文件名推断是否为测试集，因此操作者仍需保证输入来源为开发集。

检查重复 image_id、缺失 scene_id/split、manifest/输入不完整、同 scene 跨区，以及解码 RGB 像素哈希相同的图片跨区（包括不同文件名/无损重编码）。这不是近似图像检索，不能识别相邻视频帧或有损变化；同场景依赖必须靠真实 scene_id 防止泄漏。带 test 的 manifest 仅用合成夹具验证了隔离逻辑，没有读取项目测试集。

## 实测验证

数据工具测试使用默认 Python 3.12.7 + Pillow 10.4.0，**不是目标 Python 3.11/CUDA 环境验收**。所有数据均为临时目录中的合成夹具。

```powershell
python -B -m unittest discover -s tests -p test_p06_data.py -v
python -B -m unittest discover -s tests -p test_p06_data.py -k serialized -v
python -B tools/p06_prepare.py --help
git diff --check
```

实际命令、退出码和测试输出保存在 `verification.json`。先写测试运行时因实现缺失退出 1；首轮 16 项通过后，1000 组序列化往返检查发现 219 组因极小浮点边界误差被拒绝。添加回归测试重现失败，再仅修复反算舍入边界，相关回归及完整测试通过。最终 23 项通过，覆盖本轮要求的 15 类验证及额外路径、EXIF、schema、防覆盖检查。CLI 真实子进程在合成数据上验证成功、检查失败和防覆盖，退出码分别 0/1/2。

手算：`W=200,H=100,xyxy=(20,10,100,60)` → `cx=60/200=.3,cy=35/100=.35,w=80/200=.4,h=50/100=.5`；类别 long_screw 的 TXT 为 `1 0.3 0.35 0.4 0.5`。反算回 `(20,10,100,60)`。

P07 全套运行测试、真实数据转换回画、项目模型性能、正式训练、GPU/YOLO11n 冒烟、尺寸标定验收均未完成。本轮未更改或执行 P07 GUI/真实摄像头。

## 真实数据交接需要 P03 提供

1. 51 张开发图原文件及配对 Labelme rectangle JSON，六标签名称严格一致；对每个可评目标含完整框。
2. `image_id,scene_id,split` 清单；未划分时明确留待人工安排，禁止以生成随机分区代替交接。
3. 已解决的 26.jpg 对应最终标注版本；33–35.jpg 的人工判定结论或明确待定状态。
4. 空场景人工确认清单；ignore、REVIEW/annotation_status、遮挡/截断和框边界约定，目录外类型/错规格说明。
5. 数据版本及校验值、图像级记录和参考 expected_counts，用于后续核对目标框派生计数。只交图像级 PASS/FAIL 不足以训练检测器。
6. 独立测试集由 P03/P08 保管；P06 开发只接收隔离标识/版本，不用其图像或标注选择模型与阈值。

下一步先解除 Python 3.11 阻塞、完成真实 CUDA/YOLO11n 冒烟，再接入开发标注做回画人工复核。训练需另外审核后开展。
