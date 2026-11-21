# P07 初始审查与实现边界

审查日期：2026-11-21（Asia/Shanghai）。初始目录仅有 `Project-Man.pdf`、`team_plan_ai.md`；不存在 Git 元数据、AGENTS.md、README、requirements、app.py、src/camera.py、src/inspection.py、src/adapters/、predict.py、src/evaluator.py、权重、图片或视频数据。

已完整提取课程 PDF 18 页文本并阅读相关要求；未修改 PDF 与计划稿。课程 PDF p1 允许 webcam 或 connected phone camera，p2 要求改变预期数不改变观测，p5 允许两个图像级基线，p12 规定文件格式和图像级豁免，p14 要求 ≥20 新帧及 Median/P95，p16 要求诚实保留 LLM 使用证据。

所有权：P07 新增 app.py、src/camera.py、src/p07_*、P07 专属配置、依赖、测试、tools/p07_* 和文档。P08 公共模块及其他人的模型均保持缺失，没有擅自代写。

冲突处理：

- 计划稿 §0 要求先等待确认；用户当前任务明确要求直接开发、不重复确认。按当前明确授权执行，先给五行任务复述后开发。
- 计划稿项目简介写 USB 相机；课程允许 connected phone camera，用户指定 Camo 手机虚拟相机，采用 OpenCV 通用设备索引与后端，不依赖 iPhone API。
- 计划稿是 `pending_group_confirmation`，不能把草案状态、分数枚举、固定计数或建议 Python 签名说成组会已批准。
- 初始默认 Python 3.9.7；后续发现 `env_new` 的 Python 3.11.14，建立 `.venv311` 验证。环境差异以 verification.json / session.json 为准。

最小接入方案：不实现 P08 公共判定；MockBackend 内仅有模拟测试规则，真实模式全部委托 P08 插件的 infer/rejudge/synchronize。插件函数签名是 P07 建议，不是公共接口变更。

实际设备探测：索引 1 / DSHOW 能返回 1280×720 图像；索引 0、2 及所试 MSMF 未成功。25 帧回看为黑色，物理设备身份未知，不能证明手机连接、有效零件画面或真实新场景。原始记录见 camera-probe.json 与 README 中 session 路径。

当前阻塞：P02/P03 实拍与场景清单；P01 标定和教师确认；P04/P05/P06 权重、冻结配置与版本；P08 共享推理与批处理、评估器和桥接确认。真实 U01–U06、正式模型延迟、双版本比较均 NOT_RUN。

后续设备身份修正（2026-11-21）：用户确认 camera1 是内置摄像头。新增按名称枚举后确认 DSHOW 0=Camo、1=HP True Vision FHD Camera、2=OBS Virtual Camera；MSMF 0=Camo、1=HP True Vision FHD Camera。当前 Camo 的 DSHOW 索引0能成功打开并取到1280×720帧，MSMF 此次打开失败。没有把早期 camera1 的黑帧记录改写成手机证据；早期记录保留原样，新记录分别存 camera-devices.json 和 camo-probe.json。设备名枚举使用 https://github.com/lukehugh/cv2_enumerate_cameras 的1.4.0发行包（requirements-p07.txt锁定）。
