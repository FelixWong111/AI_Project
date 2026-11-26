# Stage 2实机计时

运行`python scripts/benchmark_live.py`，复用正式注册表、FrameProvider、真实Adapter和P07渲染。默认Camera0，逐模型20个新帧预热、100个新帧测量，batch1，不测Final。先关app.py并在Camo Studio确认手机输出真实白色哑光工位。

默认启用显示。主Live边界为`FrameProvider.available_at`至处理、结果校验、GPU同步、P07渲染、`imshow/waitKey`返回，含帧可用后的排队，不含此前曝光、Camo传输或物理屏幕刷新。纯processing时间另记：预同步结束后进入infer至结果校验及完成同步；加载、预热、哈希、日志、导出不进入processing边界。

Baseline3必须是正式size_repair_v2，不能用YOLO-only。实际尺寸次数和标定兼容性单列；当前1280×720 Camo不在3024×4032照片标定范围，拒绝测量路径的耗时不能冒充完整实物尺寸验收。先补Camo标定，不能关闭核验。

每次采样必须有新read ID且available_at晚于上一轮完成，不复用缓存。FAIL/REVIEW正常计时，ERROR、断连、超时单列；黑帧明确报错。至少20成功新帧，不足则标不足。相同像素不能证明不同物理内容。

每run独立保存frame_latency.csv、capture_events.csv、summary.csv、run_config.json、report.md。NumPy median/linear P95，ddof0标准差；记录硬件/软件/设备/权重SHA和大小、实际分辨率、RSS/CUDA内存。没有可靠分段钩子的预处理/纯模型/后处理时间为null，不估算。

Capture FPS来自采集线程；observed_processing_fps/observed_sample_fps按实际采样时间跨度计算。processing_fps采用观测吞吐，theoretical_serial_capacity_fps仅为1000/平均processing毫秒的理论容量，不能当持续FPS。

```powershell
python scripts/benchmark_live.py --camera 0 --frames 100 --warmup 20 --models baseline1 baseline2 baseline3
# 诊断用，不满足含显示的课程边界：
python scripts/benchmark_live.py --no-display
```

目前Camera0三帧全黑，正式实机成绩均NOT_RUN。临时合成fixture、存储视频和旧离线Train20时间不计入实机性能。
