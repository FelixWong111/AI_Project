# P06 → P07：冻结检测与候选尺寸接口交接

## 2026-11-25 交付更新

本节为本次交接入口，后续旧检测专用说明继续用于原版对照。当前交付包含统一证据入口，
默认关闭前景补检。尺寸测量提供候选线和毫米估计，真实可信验证记录为零条。
实际返回 REVIEW。P08 尚需接入共享判定，不能把本版本写成已完成完整尺寸拒识或 U03 验收。

更新后先在仓库根目录执行：

```powershell
python -B -m unittest discover -s tests -p test_p06_pipeline.py -v
python -B -m unittest discover -s tests -p test_p06_measurement_policy.py -v
python -B -m unittest discover -s tests -p test_p07.py -v
python -B tools/p06_handoff_check.py --config config/p06_p07_pipeline_candidate.json --output-dir output/p06_integrated_handoff_check
```

最后一条真实加载唯一冻结权重，调用 P07 工厂与工作线程，并检查 CUDA 和缓存重判。
默认使用程序生成图像，检查工程运行。需要检查现场开发照片时添加 --image 参数。
已有输出目录会拒绝覆盖，重跑请更换目录。

图片 GUI 与摄像头启动方式：

```powershell
python -B app.py --backend real --model baseline3 --config config/p06_p07_pipeline_candidate.json --image YOUR_NON_TEST_PHOTO --output-dir output/p06_image_session
python -B app.py --backend real --model baseline3 --config config/p06_p07_pipeline_candidate.json --camera-name Camo --capture-backend dshow --output-dir output/p06_camera_session
```

相机名需按 P07 电脑实际枚举结果填写。保留原检测入口
config/p06_p07_detection_only.json，用于与统一候选入口对照。两个入口共用同一冻结 best.pt。

### P07 与 Agent 应检查的内容

1. 先完成下文环境、权重字节数及 SHA-256 核对。新运行模块只使用已有依赖。
2. 运行新增测试与真实工厂自检，保存提交号、权重哈希、退出码和 result.json。
3. 核对框与 counts 一致，修改预期数只重判缓存。状态 REVIEW 是当前输出边界。
4. 现场拍摄条件需使用新的配置副本记录。当前旧照片候选尺度为 3024×4032，实时
   1280×720 会明确提示 CALIBRATION_RESOLUTION_MISMATCH。不要仅改变尺寸字段来通过检查。
5. foreground_enabled=false 是本次明确决定。真实 Train 检查发现另一张空图产生1688个
   假前景候选，暂不启用该模块，也暂不使用其去重。跨类别重复框仍是已知限制。
6. measurement_policy.json 中 records=[]。仅在有实际误差验证、支持条件及审核记录后，
   才能加入可信测量范围；不要自行设置 validated=true 或虚构误差界。
7. GUI 原框仍为 YOLO 观察。size_evidence 和 rejection_evidence 为新增对象证据，
   p06_evidence 包含溯源信息。P07 当前通用显示与 CSV 不会自动展示所有新字段；
   需与 P08 协作决定最终显示、ID0归类及导出，P06接口本身尚未实现最终 PASS/FAIL。

统一入口见 [pipeline.md](pipeline.md)，可信条件见 [measurement_policy.md](measurement_policy.md)，
尺寸字段见 [size_evidence.md](size_evidence.md)。前景实验说明见 [foreground.md](foreground.md)。
源码和这些配置均不要求复制 P06 的本地 data 或整个 output。唯一需要的 output 文件是下文的 best.pt。

### 当前实际验证与剩余责任

已有冻结权重 GPU 冒烟与187项流水线阶段测试通过，可信策略接入后完整P06测试193项通过。
新增功能的合成测试不代表真实测量精度。固定16张Train检查中，85个GT均有同类IoU≥0.5
的检测匹配，包含15个ID0；这批属于Train，不能作为独立性能。尺寸偏差尚不足以满足
暂定容差，工位尺度/容差验收及未见错误规格件的真实拒识仍待P06协同核查。
这些历史结果为开发证据。本次交付的干净检出通过182项P06测试、23项P07回归，
真实CUDA、P07工厂/工作线程和缓存重判通过。详见 [delivery_validation.md](delivery_validation.md)。

P07负责现场摄像头、GUI与操作记录；P06继续负责测量和拒识模块问题；
P01协调标定、容差与教师确认；P08负责共享判定及独立Test。后续final集成另行安排。

供 P07 及其 Agent 按顺序执行。当前交付是开发版本的真实检测与计数，不是完整整盘判定系统；不要为“通过验收”更改权重、阈值、标签或返回状态。本文件优先于 P06 旧实验文档中的本机路径示例。

## 1. 代码版本与边界

仓库：<https://github.com/FelixWong111/AI_Project>

交付分支：`feat/p06-baseline3`。基于已获取的 `origin/main` 提交 `6bdb8a181747b1d3b59495490da08240c99b2de9`，包含已有 P07 app、摄像头、界面、桥接和测试。代码与唯一冻结 best.pt 在同一个 Git 提交中交付，不使用 Release 附件。实际提交号以分支/PR 为准，并记录到 P07 联调日志。

保留 P07 自己的修改；有未提交工作先自行提交或使用独立 checkout/worktree，不要 reset --hard 或 git clean。建议：

```powershell
git fetch origin
git switch -c p07/p06-integration origin/feat/p06-baseline3
git rev-parse HEAD
```

若已有该本地分支，不要覆盖；在 P07 自己的联调分支正常 merge `origin/feat/p06-baseline3` 并审查冲突。不要只复制 p06_inference.py，它还依赖 p06_data.py 中的六类定义。不得从 PR 直接覆盖 P07 未提交代码；本轮不自动合并 main。

## 2. 独立环境

目标为 Windows x64、Python 3.11.x。P06 本机为 3.11.18，不强制相同补丁号。不要直接改 P07 原 `.venv311`、Conda base 或其他项目环境。

先执行 `conda env list`；若 `ai-p06-yolo` 已存在，先检查它，不覆盖。不存在时：

```powershell
conda create -n ai-p06-yolo python=3.11 pip -y
conda activate ai-p06-yolo
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r docs/p06/requirements-p06.txt --index-url https://pypi.org/simple
python -m pip check
```

期望 torch 2.7.1+cu126、torchvision 0.22.1+cu126、Ultralytics 8.3.236、NumPy 1.26.4、OpenCV 4.10.0.84、Pillow 10.4.0、cv2-enumerate-cameras 1.4.0。不要安装 opencv-python-headless。完整本机 freeze 快照仅供排查，不建议对着不同包源盲装全部版本。

官方 torch 版本命令：<https://pytorch.org/get-started/previous-versions/#v2-7-1>。

### NVIDIA / CUDA 先决条件

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

记录显卡型号、驱动和运行库；有 NVIDIA 显卡不等于该 wheel 支持该 GPU。驱动过旧或架构不兼容时停止并回报，不盲目升级全部环境。不要仅凭 nvidia-smi 中的 CUDA 字样判定 Python 可以使用 GPU。

当前接口强制 CUDA 设备 0，没有 CPU/Mock 回退。无兼容 GPU 可以先执行合成接口测试，但真实检测联调必须标记 NOT_RUN；如确需 CPU，另经 P06 批准新增独立配置/验证，不要修改本冻结版本。更换设备不保证跨硬件浮点输出逐位相同，应保留实际 GPU 记录。

## 3. 获取并验证唯一权重

best.pt 已作为普通 Git 二进制文件与代码一同交付，不需要 Release、Git LFS 或单独下载。正常拉取/切换到本分支后，应已存在以下文件；路径与冻结配置保持一致：

```text
output/p06_runs/p03_baseline3_640_seed42/weights/best.pt
```

仓库文件入口：<https://github.com/FelixWong111/AI_Project/blob/feat/p06-baseline3/output/p06_runs/p03_baseline3_640_seed42/weights/best.pt>。若仅下载单文件，使用该页面的原始文件下载，不要把 HTML 页面保存为 .pt。

若文件缺失，先确认当前 Git 提交及是否使用了省略 output 的稀疏检出，再向 P06 反馈。不能拿 COCO yolo11n.pt、其他组权重或旧实验 best.pt 代替。

虽然 .gitignore 继续忽略 output/，这一个精确文件已经受 Git 管理；其他缓存、日志、权重和图像不随本次提交上传。不要整目录删除 output 来清理日志；也不要强制添加整个 output 目录。

预期字节数：`5465242`。

SHA-256：

```text
4c5a1f759623a88ed2d21d639a4f3254e9da92d89107e4cdbfd6bddcf4e76b3d
```

```powershell
$weightPath = 'output/p06_runs/p03_baseline3_640_seed42/weights/best.pt'
Get-Item -LiteralPath $weightPath | Select-Object Length
Get-FileHash -LiteralPath $weightPath -Algorithm SHA256
```

大小/哈希不符必须停止。工厂也会再次核验哈希、类别、GPU 和 Ultralytics 版本。只交付这个约 5.21 MiB 的模型，不需要 last.pt、原始开发/Test、训练数据集、缓存、旧预测或训练目录。

## 4. 不依赖数据集的最小验收

在仓库根目录、上述环境运行：

```powershell
python -B -m unittest discover -s tests -p test_p06_inference.py -v
python -B -m unittest discover -s tests -p test_p07.py -v
python -B tools/p06_handoff_check.py
```

前两条分别为 19 项 P06 接口合成测试和 23 项已有 P07 回归；并不证明模型识别能力。第三条调用真实 CUDA 模型，通过 P07 原工厂和工作线程处理一张程序生成的白色图，检查六类计数、原图坐标与缓存重判；输出 `output/p06_handoff_check/result.json`。外层 PASS 仅表示工程链路通过，内层检测结果仍应为 REVIEW，不是整盘 PASS。

已有输出目录拒绝覆盖；重复执行用 `--output-dir output/p06_handoff_check_02` 等新目录。需要检查自己的非 Test 照片时：

```powershell
python -B tools/p06_handoff_check.py --image 'C:/your/non_test_photo.jpg' --output-dir output/p06_handoff_photo_01
```

工具不依赖本机旧 data/output 验收记录。旧的 p06_verify_inference.py 不在最小交付中，也不是新电脑入口。不要为了运行它而复制 P06 整套数据和历史 output。

## 5. P07 真实图片 GUI，再摄像头

先用 P07 自己的非 Test 照片运行 headless：

```powershell
python -B app.py --backend real --model baseline3 --config config/p06_p07_detection_only.json --image 'C:/your/non_test_photo.jpg' --headless --max-frames 1 --timeout 60 --output-dir output/p06_p07_photo_check
```

成功后去掉 `--headless`，另用新输出目录打开图片 GUI，检查框、六类数量、预期数和差值。初始 expected_counts 全 1 只是 UI 初值，不是 GT 或合格承诺。

摄像头按 P07 本机 README 重新枚举；设备编号不能照搬 P06：

```powershell
python -B app.py --list-cameras
python -B app.py --backend real --model baseline3 --config config/p06_p07_detection_only.json --camera-name Camo --capture-backend dshow --output-dir output/p06_p07_camera_check
```

如果本机不是 Camo，使用真实枚举名称或正确索引。人工确认画面确为现场摄像头、移动零件能反映到新帧，再进行取帧/暂停/保存/恢复测试。不能把静态图片、Mock 或 headless 的输出当成真实 GUI 摄像头证据。

### 验收清单

- 显示 `REAL BACKEND`，无 MOCK 水印；权重版本/哈希匹配。
- 框叠加在当前被推理的原帧上，类名为六类目录，计数等于保留框数；ID 0 单独计数。
- 修改五类预期数后，观察数量与框不变，只更新 UI 中的预期与差值；这不是新检测。
- 空结果为六类零计数，不生成 ID 0 或缺件框。异常必须 ERROR，不能回退 Mock 或伪造空结果。
- 保存截图/CSV，记录 commit、权重 SHA、环境、相机/实际分辨率及原始日志。
- 状态 `REVIEW`、整图标签/分数空是当前正确行为：P06 只交付检测/计数，未执行 PASS/FAIL、尺寸与完整拒识。此 REVIEW 不可当作正式不可判定率统计。
- 若测实时表现，按 P07 的测量边界采足够真实新帧，保留首次预热与稳定运行的差别，不把单张推理速度换算为系统实时 FPS。

## 6. Agent 不得擅自修改的内容

固定 YOLO11n、640、conf=0.25、IoU=0.7、device=0、batch=1、FP32、类别相关 NMS、类别 ID 0–5 和权重哈希。不为消除重叠框擅自开启类别无关 NMS，不补造 PASS/FAIL 或图像级概率，不重训、不调参、不读取 Test。

已知局限：pass59 等存在细螺钉→粗螺钉混淆，跨类别重复框可能影响数量；960 推理对照更弱，因此保留 640。ID 0 仅学过部分非目录件，不能代表任意未知物拒识；尺寸标定/容差尚未完成。当前不是 final，也未验证独立 Test 性能。

P07 的 module:factory 实际调用和模型生命周期详见 [inference.md](inference.md)。模型初次加载同步执行，切换模型才后台加载；当前 P07 无 backend.close 钩子。不要因显存缓存未立即归零就强行改 P07 公共实现。

## 7. 反馈给 P06 的最小记录

提交号、权重哈希、Python/torch/torchvision/Ultralytics/OpenCV 版本、GPU/驱动、三条最小验收命令退出码、自检 result.json、图片 GUI/摄像头各自 PASS/FAIL/NOT_RUN、非 Test 失败截图及原始异常。不能仅写“GPU 可用”或“测试通过”。P07 新电脑环境和现场摄像头在 P06 发布时仍是 NOT_RUN，必须在 P07 电脑上实际验收。
