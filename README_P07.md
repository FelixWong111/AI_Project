# P07 / SW-04 机械零件视觉检测实时系统

本交付负责输入采集、OpenCV GUI、模型接入位置、结果展示、截图/录像及原始计时日志。**默认是 MOCK 推理；摄像头仍为真实输入，红色水印和 MOCK 字样只表示模拟检测结果，不能用于模型成绩或真实识别演示。**

**Camo 摄像头已实测可用（2026-11-21，用户现场确认）：使用 camera 0 可以正常使用 Camo 摄像头。** 本机 DSHOW 设备对应关系为 **0=Camo、1=HP True Vision FHD Camera（电脑内置摄像头）、2=OBS Virtual Camera**。推荐用 `--camera-name Camo` 启动，避免设备编号变化后误选内置摄像头。程序侧设备枚举与取帧记录见 `docs/p07/camera-devices.json` 和 `camo-probe.json`，其中记录的取帧分辨率为 1280×720。用户现场确认与程序探测是两类独立证据；摄像头可用不代表真实检测模型已经接入。

课程依据为 `Project-Man.pdf`，其次为 `team_plan_ai.md` v1.0。计划仍是讨论稿；本文的 CONFIRMED 仅表示文档中已有明确约束，不表示已获得组会或教师确认。初始审查见 [audit.md](docs/p07/audit.md)。

## 1. 职责与当前交付状态

P07 已提供 Camera/Image/Video → FrameProvider → P07 bridge → Presenter → GUI/Logging 主链路。没有创建或修改 P08 的 `src/inspection.py`、`predict.py`、`src/evaluator.py` 或其他成员的 `src/adapters/`。未实现训练、真实拒识、标定和评估器。

P01 提供工位、尺度与教师确认；P02 提供原始图像及采集清单；P03 提供标注、场景分区和封存版本；P04/P05/P06 提供模型及冻结配置；P08 提供共享推理/重判和独立评估。

Stage 2 状态：**Camo 摄像头输入已由用户实测确认可用，P07 Mock 推理原型可交付；课程完整 Stage 2 尚未验收。** 当前应使用 camera 0 或按名称选择 Camo。早期 camera 1 的25帧采集/录像属于电脑内置摄像头记录，不能作为 Camo 验证证据。真实三模型、零件检测联调、U01–U06 与正式性能仍待补。

## 2. 安装与启动（Windows PowerShell）

在本项目根目录执行。目标 Python 3.11 x64；系统 Python 不一定是该版本，先检查。当前工作区已建立 `.venv311`，可直接使用其中解释器。

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements-p07.txt
.\.venv311\Scripts\python.exe --version
.\.venv311\Scripts\python.exe app.py
```

如果 `py -3.11` 未注册，使用已安装的 Python 3.11 完整路径执行 `-m venv .venv311`。无需激活即可运行；需要激活时执行 `.\.venv311\Scripts\Activate.ps1`。如 PowerShell 限制脚本执行，直接使用解释器路径即可。核心仅 NumPy/OpenCV；文件选择器使用 Python 自带 tkinter。不要安装 `opencv-python-headless` 替代有 GUI 的包。

```powershell
.\.venv311\Scripts\python.exe tools/p07_demo.py
.\.venv311\Scripts\python.exe app.py --image tmp/p07-demo/mock-input.png
.\.venv311\Scripts\python.exe app.py --video tmp/p07-demo/mock-input.avi
.\.venv311\Scripts\python.exe tools/p07_probe_camera.py --indices 0 1 2
.\.venv311\Scripts\python.exe app.py --list-cameras
.\.venv311\Scripts\python.exe app.py --camera-name Camo --capture-backend dshow
```

本机已验证的按索引启动方式：

```powershell
.\.venv311\Scripts\python.exe app.py --camera 0 --capture-backend dshow
```

设备名选择会解析当前后端的索引，不会固定为 0 或 1。更换电脑、手机、后端后重新列出设备。未传输入参数时窗口保持待选择状态；NO INPUT 表示尚未选择来源。

## 3. iPhone 14 Pro → Camo 实测步骤

1. 在 Windows 安装 Camo Studio，在 iPhone 安装 Camo Camera（从各自官方分发渠道获取）；本程序不代装手机软件。
2. 用支持数据的 USB 线连接，解锁手机，按手机提示信任电脑；在 Camo Studio 中选中手机，确认能看到持续变化的实拍画面。
3. 使用固定支架、纯白哑光背景和稳定照明。高度、距离、倾角改变后交 P01 重新标定，P06 复核尺寸拒识条件。
4. 关闭占用同一摄像头的其他应用，确认 Windows 相机权限允许桌面应用访问。
5. 先执行 `app.py --list-cameras` 确认设备名称，再按需执行 `p07_probe_camera.py`。先试 `--capture-backend dshow`，失败再试 `msmf` 或 `auto`。
6. 启动 `app.py --camera-name Camo --capture-backend dshow`，核对画面确为手机现场、遮挡/移动物体能立即反映，而非黑帧、静态占位或其他摄像头。检查右侧 Device 名称、实际分辨率、帧号、采集 FPS。
7. 按 `r` 开始录像，按 `s` 保存结果包，再按 `r` 停录。拔插线确认断连提示与恢复，保留日志。此处 **MOCK 框并非从画面检测而来**；仅验证 UI。
8. P08 桥接和模型就绪后改用 `--backend real`，按 §10 做真实 U01–U06 并独立记录。

切换 P02 的 iPhone 15：不改核心代码。检查数据线/信任、Camo 设备选择、虚拟相机索引/后端、实际 `frame.shape`、分辨率与帧率、焦距/曝光/方向、工位几何与尺度标定。重新核实输出画面、延迟和尺寸核验；不得沿用旧手机实测结论。

常见问题：

- 打不开：Camo 没启动、设备索引变化、权限未开或被占用；重新探测后切换后端。
- 选成电脑自带摄像头：本机 camera 1 是 HP 内置摄像头，应改为 camera 0，或使用 `--camera-name Camo`。当前用户已确认 camera 0 可用。
- 能读但全黑：先核对右侧 Device 和 Camo Studio 的手机画面；`read_success=1` 本身不是有效实拍证明。早期黑帧录像来自 camera 1，不代表当前 Camo 的状态。
- GUI 显示缺失：检查是否误装 headless 包，或 OpenCV GUI 库不可用。
- 模型 ERROR：检查桥接导入、权重路径、配置和版本；真实模式绝不自动降级为 Mock。
- 驱动卡在 open/read：正常关闭等待最多 3 秒后明确报错；驱动阻塞不能被 Python 安全中止，需关闭设备占用/重启程序。工作线程不会从 UI 线程强制释放句柄。
- 视频 EOF：保留最后一张结果并明确 EOF 或解码失败，不自动回放冒充新帧。按 `v` 重新打开。

## 4. 界面操作与配置

窗口内使用英文标签（OpenCV 默认字体不支持中文），本文为中文说明。五条 `Expected 1..5` 滑块输入非负整数，范围由 `max_count` 设置。导入 JSON 时超过 GUI 上限会明确拒绝，先调高配置再启动；公共数据合同本身没有此上限。

| 操作 | 按键 |
|---|---|
| 检测开始/暂停 | 空格 |
| baseline1 / baseline2 / baseline3 / final | 1 / 2 / 3 / 4（推理空闲时切换） |
| Mock 场景轮换 | n |
| 打开摄像头 / 关闭输入 | c / x |
| 摄像头索引增加 / 减少 | +（或 =）/ - |
| 选择本地图片 / 视频 | i / v |
| 导入预期数量 JSON | e |
| 保存截图、原帧、结果 CSV、配置 | s |
| 录像开始/结束 | r |
| 退出并释放资源 | q / Esc / 关闭窗口 |

暂停检测仍继续取帧；已完成检测画面保持在原帧上，显示的帧号可用于识别冻结结果。修改预期数会在工作线程调用重判，不更换框与观测数；待重判完成前显示 pending 并禁止保存错配结果。模型加载也使用后台线程。使用单个推理任务和最新采集帧，不积压无限帧队列。

`config/p07.json`：camera_backend、width/height（请求值，实际以帧为准）、reconnect_seconds、max_count、record_fps、expected_counts、mock_fixed_counts、bridge、model_config。`mock_fixed_counts=[1,1,1,1,1]` **只是测试夹具，正式固定参考向量仍 TBD**。Mock 的目标观测也固定为测试场景，不依赖用户预期数。

启动参数完整列表可运行 `python app.py --help`：`--config`、`--backend mock|real`、`--model`、互斥的 `--camera/--camera-name/--image/--video`、`--list-cameras`、`--capture-backend`、`--counts-file`、`--image-id`、`--output-dir`、`--scenario`、`--headless`、`--max-frames`、`--timeout`、`--record`。`--timeout` 只控制 headless 无结果超时；`--max-frames` 统计完成处理的不同采集帧。

## 5. 程序结构

| 文件 | 用途 |
|---|---|
| app.py | CLI、事件循环、源/模型切换、计数编辑与结果导出 |
| src/camera.py | FrameProvider、后台采集、DirectShow/MSMF 回退、重连、实际 FPS |
| src/p07_devices.py | 按 OpenCV 后端枚举设备名、名称到索引的明确匹配 |
| src/p07_bridge.py | P07 本地建议桥接协议、数据校验、独立 Mock 场景 |
| src/p07_ui.py | 单推理工作线程、框/状态/计数呈现 |
| src/p07_output.py | 采集和时延 CSV、分组摘要、结果包、MP4 与参数 |
| config/p07.json / requirements-p07.txt | P07 专属配置、锁定运行依赖 |
| tools/p07_demo.py | 显式合成的离线夹具，不是训练或测试集 |
| tools/p07_verify.py | 执行测试与 CLI/GUI 冒烟，保留实际输出和退出码 |
| tools/p07_probe_camera.py | 指定索引的真实设备探测，不推断手机身份 |
| tests/test_p07.py | Mock U01–U06、输入、异常、导出、录制与释放回归 |
| docs/p07/ | 审查、自动验证记录、真实联调待办、建议核查 |

## 6. P02 / P03 输入接口

支持 OpenCV 能解码的本地彩色图像（推荐 JPEG/PNG）和视频（推荐 AVI/MJPEG 或 MP4；依赖本机解码器），内部统一为 `uint8 H×W×3 BGR`。中文路径图像通过 `np.fromfile` + `imdecode` 加载；HEIC 需上游转为可解码格式并保留对应关系。数据不复制到训练目录，不自行划分场景。

按计划 `image_id` 是完整文件名，例 `S2-0001_v01.jpg`，单次运行唯一；`scene_id` 是一次物理摆放，不同帧/角度/照明变体共享该值。P02/P03 提供采集清单、真实 ID、scene 分组、split、标定 ID、真值和数据版本；P07 不猜 scene_id，导出中未知为 null。相机/视频若没有数据集 ID，截图采用 `frame-<frame_id>.png` 的本地临时 ID，不能冒充正式 S2/S3 数据集命名。可用 `--image-id` 明确指定已登记的数据集 ID。

分辨率由 `frame.shape[:2]` 计算。P07 的 session_id 和单调 frame_id 是应用日志标识，不是手机内部编号，不是 scene_id，也不是 dataset image_id。

预期数量文件严格使用计划 §5.2 的列表：

```json
[
  {
    "image_id": "S2-0001_v01.jpg",
    "expected_counts": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1}
  }
]
```

键只能为字符串 1–5，值是非负整数，bool、浮点、负值、缺失和多余键均拒绝。ID 0 不在输入映射中，预期恒为零。加载时按 image_id 精确匹配一条记录；重复或缺失明确报错。图片默认取文件名；相机/视频导入此文件时须提供 `--image-id`。允许在 GUI 修改预期数；不会修改源 JSON。

## 7. P04 / P05 / P06 模型接口

| model_id | 能力 | 显示 |
|---|---|---|
| baseline1 | HOG + MLP，图像级，固定预期数 | 整图状态、标签、分数；框/计数/ID0 为 N/A |
| baseline2 | ResNet18 冻结特征 + Logistic Regression，图像级 | 同 baseline1 |
| baseline3 | 目标级、任意合法预期数、ID0 | 原图 xyxy 框、类别、分数、0–5 数量和差值 |
| final | P06 集成改进，保留目标级能力 | 同 baseline3 |

模型真实权重、配置路径、加载函数、预处理、阈值、分数校准和冻结校验值尚未提供，状态 BLOCKED。GUI 不自行加载通用 checkpoint 当作项目模型。缺失或不兼容权重由桥接层明确抛错，界面显示 ERROR，保留消息，不伪造预测。

要求对方交付：版本/校验值、文件大小、模型配置、依赖、工作设备、输入预处理、分数含义、坐标还原、固定数量（图像级）、ID0/尺寸核验方案（目标级）。框是原图左上原点 xyxy 像素坐标；边界包含/不包含约定由 P08 与标注方统一。Mock 采用连续边界 `0 <= xmin < xmax <= width` 等约束，不代替正式边缘约定。

`final` 原版/改进版仍使用正式 ID `final`，通过两份 `--config` 的 model_config 指向不同权重和 model_version，在不同会话运行；不新增正式模型 ID。配置具体结构由 P06/P08 确定。

## 8. P08 公共接口与建议桥接

计划规定的批处理入口是：

```powershell
python predict.py --input_dir TEST_FOLDER --model baseline3 --output_dir RESULTS --counts_file expected_counts.json
```

**当前 predict.py 缺失，此命令不可执行。** 共享 `src/inspection.py` 应负责推理、坐标还原和判定；`src/adapters/` 应接各模型；`src/evaluator.py` 应独立接真值。这些均未交付。GUI 不替代上述模块。

以下 Python 协议均为 **PROPOSED — P08 confirmation required**，仅在 `src/p07_bridge.py` 定义 P07 侧接入位置：

```python
# config/p07.json: "bridge": "your_p07_bridge:create_backend"
def create_backend(*, model_id: str, config: dict):
    ...  # 返回下列对象；工厂在此加载对应冻结版本

class Backend:
    metadata: dict
    def synchronize(self) -> None: ...
    def infer(self, frame_bgr, expected_counts: dict[str, int]) -> dict: ...
    def rejudge(self, result: dict, expected_counts: dict[str, int]) -> dict: ...
```

桥接文件由 P07/P08 对齐后提供，可用任意模块名；工厂标识 `module:factory` 从 bridge 配置读取，model_config 原样传给工厂。桥接应只映射已有 P08 函数与返回值，不复制正式规则。`infer` 调共享推理；`rejudge` 用已缓存结果调用共享判定，不能再次做目标检测。图像级重判应能从不支持数量恢复到支持数量时还原原预测，必要缓存保存在桥接对象内。CUDA 时 `synchronize` 必须同步对应设备；CPU 可以空操作。本程序在真实推理前后调用它，计时不包含模型加载。

建议结果字典：

```python
{
    "status": "PASS", "predicted_label": "PASS",
    "prediction_score": None, "score_type": None,
    "non_catalogue_count": 0,
    "counts": {"0": 0, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1},
    "objects": [{"detection_id": 0, "class_id": 1,
                 "xmin": 0, "ymin": 0, "xmax": 20, "ymax": 30, "score": 0.9}]
}
```

上面只是字段示意，实际对象列表必须与全部 counts 一致。图像级 `counts/non_catalogue_count=None`、objects 为空。ERROR 允许观测缺失；不可判定且没有可靠观测的 REVIEW 允许空框和 counts=None。一般目标结果必须包含 0–5 完整数目。缺失字段展示 N/A；不生成缺件框。桥接输出非法状态、非有限数值、越界框、重复检测 ID、框与计数不一致会转为 ERROR。

状态沿用计划枚举：PASS / FAIL / REVIEW / UNSUPPORTED_COUNTS / ERROR。PASS/FAIL 的 predicted_label 与 status 一致；其他状态标签为空。UNSUPPORTED_COUNTS 分数也为空。`score_type` 为 prob / logit / none 或空，团队草案仍待确认；prob 表示概率口径，是否校准必须由模型方说明；logit 是未校准分数，不能标作概率。规则判定无图像级置信度则空值。

目标级规则由 P08 执行：五类 observed=expected、ID0=0 且可判定才 PASS；目录外确认件触发 FAIL 和警告；不可判定保留 REVIEW，不强转 FAIL。图像级不同有效预期数返回 UNSUPPORTED_COUNTS。MockBackend 中的同形规则仅用于开发夹具，不应被正式批处理复用。

metadata 建议至少提供 `model_version`、`model_file_size_bytes`、`score_convention`、`thresholds`、`preprocessing`、`rejection`、`size_tolerance`、`calibration`，图像级另提供 `supported_fixed_counts`。未交付信息记 TBD，不能当作完整可复现正式配置。选择真实模式：

```powershell
.\.venv311\Scripts\python.exe app.py --backend real --config config/p07-real.json --model baseline3 --camera-name Camo
```

该配置文件和真实桥接尚待组员提供；不能直接运行此示例。创建它时保留 P07 配置其他键并配置 bridge、model_config，权重位置建议使用项目相对路径。

## 9. 输出格式、日志与计时

每次会话写独立 `output/p07/<session_id>/`；session_id 是 UTC 时间+随机后缀，不代替计划中仍待确认的 run_id。日志 timestamp 使用明确 UTC 偏移，可转换为 Asia/Shanghai，命名中不会混淆本地时间。

`capture.csv` 字段固定：

```text
session_id,frame_id,timestamp,camera_index,capture_backend,width,height,frame_read_ms,capture_fps,read_success
```

每次读帧/失败尝试一行，图片和视频的 camera_index 为空。width/height 是实际帧大小，失败为 0。frame_read_ms 是 read 调用边界，失败打开设备时记录零附近的失败事件，不把打开协商时间当作 read 耗时。成功读取时间的最近 60 个样本计算 `(n-1)/(t_last-t_first)`；首帧为 0。视频含回放节奏，图片仅一个采样不能测 FPS。断连时重置采集窗口。

`latency.csv`：frame_id、source、mode、model、model_version、width、height、inference_ms、processing_ms、processing_fps、status、expected_counts。所有时长使用 perf_counter。推理耗时由 bridge.infer 前后计量；重判不记为新推理样本。Processing FPS 依据完成并提交显示的新推理结果时间，不是 1000/单次推理耗时。`processing_ms` 从帧返回到 GUI 的 imshow + waitKey 返回，包含等候、预处理/推理/后处理、渲染与应用显示提交。**不能精确测量 OS 合成器/显示器呈现时刻，不是光子到屏幕延迟，也不包含手机传输的全部延迟。**

摘要 `latency_summary.json` 按模式/模型/版本/来源/分辨率分组，使用不同 frame_id，输出 Median/P95 和样本数。MOCK 永远标记 MOCK_DEBUG_ONLY；离线或少于 20 个有效实时帧标记 INSUFFICIENT_LIVE_EVIDENCE。real_headless 即便有相机也不能获得 GUI 实时证据标记。正式比较还须人工核实物理画面、模型元数据、硬件与测量边界；自动标记本身不是课程验收证书。

`session.json` 保存硬件、Python/OpenCV/NumPy、会话配置和边界；`<model>-metadata.json` 保存模型版本、文件大小和推理配置。录像 `<mode>-<timestamp>.mp4` 使用 mp4v，旁边同名 JSON 保存编码 fps、大小、帧数；帧率为配置的恒定编码速率，提交的检测画面每个新结果一帧，**播放时长不保证等于墙钟时长**。暂停推理期间不会新增录像帧。切源/切模型停止当前录像；退出正常释放 writer。

按 s 的结果包位于 `<session>/MOCK-/...` 或 `REAL-/...`，每次独立目录，包含 annotated.png、source.png、images.csv、run_config.json；目标级另有 objects.csv 和 counts.csv。Mock 原帧可以来自真实输入，但模拟结果包和配置中始终标注 evidence_mode=mock。图片/截图名称不会伪造 scene_id。

CSV 精确顺序：

```text
images.csv: image_id,model_id,status,predicted_label,prediction_score,score_type,non_catalogue_count
objects.csv: image_id,model_id,detection_id,class_id,xmin,ymin,xmax,ymax,score
counts.csv: image_id,class_id,observed_count,expected_count,difference
```

images 每个导出图一行，status 表示处理状态，predicted_label 表示系统判定，prediction_score/score_type 表示其分数与口径，non_catalogue_count 表示目录外数量。objects 每个保留框一行，detection_id 图内唯一，class_id 0–5，xyxy 是原图坐标，score 是模型提供的框分数。counts 覆盖 0–5，difference=observed-expected，ID0 expected=0。图像级不写 objects/counts，未知字段写空，不能伪造六行零计数。

run_config 必含计划 §5.4 的 model、model_version、output_capability、class_catalogue、thresholds、preprocessing、coordinate_convention、score_convention、hardware、rejection、size_tolerance、calibration、per_image_expected_counts；图像级另有 supported_fixed_counts。模型元数据尚缺的信息显式 TBD，Mock 不冒充真实权重配置。正式导出需 P08 补齐并确认 per_image_expected_counts 的具体嵌套约定。

## 10. 联调、实测与复现

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -v
.\.venv311\Scripts\python.exe tools/p07_verify.py
.\.venv311\Scripts\python.exe app.py --camera-name Camo --capture-backend dshow --max-frames 25 --record
```

自动测试覆盖 Mock U01 合格、U02 缺/多件和零预期、U03 ID0、U04 同帧重判且观测不变、U05 空/仅ID0、U06 REVIEW 和不支持数量；另含异常输入、真实模式缺失不回退、字段校验、图像/视频读取、CSV、录像回读、模拟断连重连和释放。

实际命令、输出和退出码见 [verification.json](docs/p07/verification.json)。设备探测见 [camera-probe.json](docs/p07/camera-probe.json)。真实联调状态独立记录在 [integration.csv](docs/p07/integration.csv)，所有真实 U01–U06 均为 NOT_RUN；Mock PASS 不能替代它们。

最终验证环境：Windows build 26200、Python 3.11.14 x64、OpenCV 4.10.0（发行包 4.10.0.84）、NumPy 1.26.4，新增 cv2-enumerate-cameras 1.4.0 用于后端一致的设备名枚举。设备选择修正后增加名称/后端匹配与歧义拒绝测试，23 项 unittest 全部通过；验证脚本的 7 条命令均得到预期退出码（其中真实 backend 缺失、坏图片两项预期退出 1）。GUI 冒烟确认窗口创建、视频呈现与自动退出；尚未逐一人工点击所有按键/文件对话框，不把这些交互说成已完成人工验收。

早期 camera 1 GUI 记录（历史记录）：`output/p07/20261121T125501Z-a392f39d/`，25 个应用帧、1280×720、DSHOW，MP4 回读成功，画面为黑帧、约 1 FPS。后续用户确认 camera 1 是电脑内置摄像头，因此该记录不属于 Camo 手机采集。该会话在 Python 3.9.7 / OpenCV 4.12.0 / NumPy 1.26.4 上执行，时延摘要为 MOCK_DEBUG_ONLY，不能作为正式模型性能。`docs/p07/camera-mock-preview.png` 是此历史录像抽帧，仅供核对界面水印。

当前 Camo 验证记录（2026-11-21）：设备名枚举确认本机 DSHOW 索引0为 Camo；程序侧探测成功打开并读取1280×720帧，记录见 [camera-devices.json](docs/p07/camera-devices.json) 和 [camo-probe.json](docs/p07/camo-probe.json)。用户随后现场确认“camera0可以使用camo摄像头”，因此当前 Camo 输入状态更新为 **已验证可用（用户现场确认）**。本次反馈没有提供新的录像、FPS、延迟或拔插恢复数据，不将这些项目标为通过。

后续真实联调：先由 P08 接好 bridge，在手机有效实拍画面中逐项运行 U01–U06；每个模型/版本保存独立会话与截图结果包。U04 对同帧修改预期并验证框和观测未变，含ID0始终不被接受。每个比较版本至少采 20 个真实新帧，保存原始时延和完整配置，单独记录图片/视频离线时间，补人工标注与 evaluator 结果。

Stage 2 演示建议（<2 分钟）：先展示 Camo 有效实拍画面及采集指标；说明当前是否 MOCK；切四种模型演示能力区别；按 n 展示 missing/extra/id0/empty/review；调预期数展示重判；保存截图与 MP4。真实模型未交付时必须口头及画面明确这是 UI 验证原型。当前保存的黑帧 Mock 录像不是正式课程演示视频。

已确认：当前 Camo 摄像头输入可用（camera 0，用户现场确认）。NOT_RUN：P02 的 iPhone 15 切换验证、物理 USB 拔插恢复、三个冻结模型、final 两版本、真实 U01–U06、正式 ≥20 新帧模型性能、尺寸与标定验收、独立 evaluator。自动断连测试仅为假采集设备，不冒充物理恢复。

## 11. 接口对齐总表

| 对接人 | 数据或接口 | 当前规范来源 | 当前状态 | 需要对方确认的内容 |
|---|---|---|---|---|
| P07（用户实测） | 当前 Camo 摄像头输入 | 设备枚举/取帧记录 + 2026-11-21 用户现场确认 | CONFIRMED | 本机 camera 0 可用；更换设备后重新核对，iPhone 15 尚未验证 |
| P02 | 原图/采集清单及演示手机切换 | 计划 §2.4、DATA-01 | BLOCKED | 图像/scene_id、拍摄条件、标定 ID 与 iPhone 15 切换验证 |
| P03 | 标注、scene 分组、封存版本 | 计划 DATA-02、§6 | BLOCKED | 真值完整性、框边界、ignore、split 与版本 |
| P04 | baseline1 固定计数与分数 | 计划 SW-01 / §5.5 | BLOCKED | 权重/冻结哈希、预处理、固定向量、分数含义 |
| P05 | baseline2 及改进配置 | 计划 SW-02 / §5.5 | BLOCKED | 固定向量、冻结版本、score_type、改进交付 |
| P06 | baseline3/final 与 ID0 | 课程 p2 / 计划 SW-03 | BLOCKED | 权重、原图坐标、拒识/标定/尺寸、双版本与文件大小 |
| P08 | 四模型 ID、类别与 CSV 顺序 | 课程 p2、p12 / 计划 §5 | CONFIRMED | 实现须遵守；不表示代码已到位 |
| P08 | status / score_type / run_id | 计划 §2.4、§5.3 待确认 | PROPOSED | 草案枚举和命名的组会确认 |
| P08 | Python infer/rejudge/synchronize、工厂 | 本 README §8 | PROPOSED | 映射实际 inspection API、缓存格式、异常、GPU 同步 |
| P08 | predict.py / inspection / adapters / evaluator | 计划 §5.1、§5.6、§10 | BLOCKED | 提供实现与联调入口，禁止 UI 自写正式规则 |
| P08 | run_config 嵌套格式与完整元数据 | 计划 §5.4、课程 p12 | TBD | per_image_expected_counts 结构、冻结记录、硬件详情 |

实质性开发建议和核查证据另见 [advice_record.md](docs/p07/advice_record.md)。本代码和自动化证据不代表成员已完成个人复核，也不替代报告中真实贡献说明。
