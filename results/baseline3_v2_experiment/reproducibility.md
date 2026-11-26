# 实验复现与恢复

本目录仅用于Baseline3模型开发、验证和选型，不代替P08正式报告；不生成英文图表。
进度与实际命令保存在task_state.json / PROGRESS.md。恢复前检查其中PID、training/progress.json、训练日志和models/baseline3_v2_candidate/resume_last.pt。
恢复命令：`python scripts/experiment_baseline3_v2.py --phase auto`。已完成阶段跳过，正在运行的PID不重复启动，已有完整P08结果直接复用。
训练使用冻结360/80分区，原生YOLO11n标准训练；仅checkpoint CSV元数据读取避开本机Polars导入错误。原始失败尝试和日志保留。恢复使用原生resume及未strip的优化器checkpoint，不是从best重新拟合。
完整模型对照位于evaluation/baseline3_old和baseline3_new，每模型仅推理一次相同30张Stage 2图。count_rule_derived只从已保存真实检测结果重判，不重新推理，也不代表完整尺寸验收。
旧完整管线保留历史未验证尺度；新管线不沿用无法证明兼容的尺度，calibration=null，因此尺寸MAE、正常规格通过率及错误规格尺寸拒识率NOT_RUN。六类ID0识别的拒绝率不等于尺寸测量精度或任意未知件拒识。
官方P08目标/计数主指标会排除模型REVIEW；object_observations.json另报包括REVIEW的真实保留框和数量，两个分母不可混淆。Accuracy、全部输入正确比例及Coverage必须一起查看。
离线forward来自Ultralytics已同步GPU的Results.speed，完整处理另用perf_counter；不将倒数当作持续摄像头FPS。旧/新测量分支因标定资料不同，完整处理时间不能归因于单独检测器加速。

## 返回现场后补做

1. P01固定Camo机位/光源/白色哑光背景，记录分辨率、距离/姿态/镜头，标注尺子端点和真实长度并复核中心及边缘尺度。
2. 对正常及三个异常规格实物建立specimen_id，卡尺重复测量，记录独立开发/验证清单；确认教师认可的规格容差及支持工况。不得用名义尺寸代替卡尺真值。
3. 用独立开发资料验证已有几何算法误差与失败率；补足测量策略、真实背景与前景关联证据。另存新配置，不改本轮冻结文件，不根据本轮Test调整算法。
4. 在相同机位/电源性能/输入条件下分别测新旧系统，每模型20新帧预热和100新帧测量。当前未验证尺寸配置的计时只能说明其当前实际分支，不证明尺寸功能通过。

```powershell
python scripts/benchmark_live.py --camera 0 --models baseline3 --warmup 20 --frames 100 --display --config results/baseline3_v2_experiment/configs/old_full.json
python scripts/benchmark_live.py --camera 0 --models baseline3 --warmup 20 --frames 100 --display --config results/baseline3_v2_experiment/configs/new_full.json
```

计时从新帧available_at至实际P07渲染和imshow/waitKey返回，包含必要GPU同步；不含此前曝光/Camo传输或物理屏幕刷新。原processing-only指标单列；Capture FPS见capture_events.csv，实际选样吞吐见observed_sample_fps。
新旧模型已按Test前配置冻结，Stage 3禁止开发访问。任何后续算法调整都需要新的独立评价证据。
