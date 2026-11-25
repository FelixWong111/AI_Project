# Stage 2实机处理延迟测试

项目已有GUI的capture/latency日志，但没有自动执行“每模型20个新帧预热＋100个新帧测量”的独立入口，因此新增scripts/benchmark_live.py。只复用现有FrameProvider、load_backend、各adapter和validate_result，不重写模型或算法，也不依赖GUI运行。

## 运行

在项目根目录，先保持手机与Camo Studio正常连接并确认画面有效，关闭app.py及其他占用摄像头的软件。保持机位、光照、零件摆放与配置条件一致。

```powershell
.\.venv311\Scripts\Activate.ps1
python scripts/benchmark_live.py
# 或直接指定环境，无需激活：
.\.venv311\Scripts\python.exe scripts/benchmark_live.py --camera 0 --frames 100 --warmup 20 --models baseline1 baseline2 baseline3
```

默认Camera0，按baseline1→baseline2→baseline3顺序运行。分辨率和后端选择复用config/p07.json及FrameProvider，与app.py一致；实际分辨率/后端逐模型记录。模型加载完才开启该模型的采集，预热和测量各自使用阶段开始后的新read；每轮处理结束后再等下一次新read，不调用rejudge或重复缓存帧。每个模型结束释放相机，再切换下一模型。

Baseline1像GUI一样启用scene_guard；Baseline3必须为当前size_verified完整管线，拒绝yolo_frozen或绕过adapter的bridge配置。没有为速度修改标定、容差或阈值。

## 计时与指标

用perf_counter：选取新的摄像头帧、清空先前GPU任务后开始；包含backend.infer内实际执行的预处理、特征、推理、检测/几何/规格/工单与最终判定，以及结果校验和结束GPU同步。预同步在起点外，结束同步在终点内。加载、预热、等帧、哈希、进度打印和导出不计入正式处理耗时。**不包含手机曝光、Camo传输或GUI渲染，不能称为完整端到端延迟。** 不伪报没有独立计时边界的“纯模型推理”数值。

FAIL与REVIEW属于正常完成，纳入统计。ERROR及UNSUPPORTED_COUNTS不作为成功样本；失败耗时仍在逐帧表，不混进成功样本统计。Coverage=成功处理数/(正式采集样本数＋正式阶段失败read次数＋等帧超时次数)；processing_coverage=成功数/正式采集样本数。预热不计入这些指标。模型加载/运行错误单独在run_config.errors和逐帧load_or_run行保留，未开始正式阶段时不能把零失败误读成运行成功。

Median使用np.median，P95使用np.percentile(method='linear')，标准差ddof=0。Processing FPS=1000/Mean Latency，不是采集、显示或录像帧率。统计表包含请求/实际采集/成功/失败数量、覆盖率、全部ms指标、FPS、预热实际数量和最低20帧判定。

成功处理少于20帧标INSUFFICIENT；中断、加载失败、读取错误和未完成请求流程都明确保留。Ctrl+C也会尝试关闭相机并导出已有记录。若驱动阻塞导致既有FrameProvider无法正常关闭，会标ABORTED并提示错误，不能假称资源已释放，应关闭Camo设备后排查。

## 输出与P08交接

每次写入独立的results/live_benchmark/UTC时间-随机ID/：

- frame_latency.csv：预热/正式阶段、真实read ID、主机读帧完成UTC时间、处理耗时、最终状态、错误/判定原因、分辨率、排队年龄、像素哈希和几何尝试/成功对象数。
- summary.csv：三个请求Baseline的完整正式统计。
- run_config.json：硬件、版本、Git提交/运行前工作区状态、真实模型metadata/版本/哈希、实际设备、配置与工单、分辨率/后端、尺寸标定/策略状态、统计和计时边界。
- report.md：英文说明及对比表，可供P08写报告；失败和局限必须保留。
- capture_events.csv：所有read尝试，包括没被选为样本的新帧和读取失败；可复核采集ID及丢过的成功读帧。

把**整个run_id文件夹**交P08，不只截取汇总表。没有保存原始摄像头图片/视频，不读取数据集或正式Test。脚本不会覆盖旧目录，不自动commit/push。

## 新帧与物理条件限制

OpenCV读取ID证明新的采集调用，不证明手机每次都产生新的曝光；Camo可能重复冻结画面。像素相同不被丢弃或用来筛选成绩，但整段完全相同会标NOT_VERIFIED_IDENTICAL_PIXELS，不能直接认定20个有效新帧。开测前应确认实物变化能反映到Camo预览，随后恢复固定测量工位。capture_time_utc是主机read完成时间，不是手机曝光时间。

Baseline3即使启用了完整管线，也会遵守原有标定/测量保护。比如实际1280×720与3024×4032候选标定不匹配时，几何可能根本未执行；报告必须同时检查geometry_attempted_frames、geometry_measured_frames、逐帧原因和真实标定状态。不能把这种运行描述成已经验证的全几何测量性能，更不能为了得到时延表擅自把validated改true。

软件新read协议成功≥20是最低计时证据，不等于完整Stage2功能合规。真实尺寸、现场条件和工位记录仍需按baseline3_size_integration.md完成。

## 开发验收状态

已用临时目录中的明确合成夹具验证CLI、统计、同步边界、缓存帧拒绝、相机关闭和文件导出；临时数据自动删除，没有写入项目results/live_benchmark作为实机成绩。本次未自动开启实体摄像头，真实实机性能为NOT_RUN。用户实际运行后才能据输出和物理条件判断是否满足Stage2实时响应要求。
