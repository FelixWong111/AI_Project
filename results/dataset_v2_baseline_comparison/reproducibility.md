# 必要复现记录

环境为`.venv311` / Python 3.11.14；完整版本及硬件见`environment.json`。固定工单为ID1–ID5各1，seed42。算法及原已选参数保持不变，没有超参数搜索。四个分类器的此次离线推理均实际使用CPU。

以下是已经执行的训练命令记录，本次收尾不再运行：

```powershell
.\.venv311\Scripts\python.exe -m scripts.train_baseline1 --dataset-config data/mechanical_components_v2/configs/training.json --output-dir models/baseline1_v2_candidate --original-only
.\.venv311\Scripts\python.exe -m scripts.train_baseline2 --dataset-config data/mechanical_components_v2/configs/training.json --output-dir models/baseline2_v2_candidate --original-only
```

Baseline1首次拟合后相对路径导出失败，日志保留于`baseline1_training.log`；同参数同seed成功重跑日志为`training/baseline1/attempt02.log`。成功轮复用已有签名特征缓存，其总运行时间不能直接与Baseline2缓存构建轮比较。`training/data_isolation_audit.json`记录两模型Scaler只拟合360张Train。

独立评估命令记录在每个运行目录`command.json`。`validation/runs.json`和`evaluation/runs.json`索引原始结果。Test运行前的选择及模型/配置SHA256见`selection.json`。正式Stage 2对照已经完成一次；本次不再次执行，也不访问Stage 3。

Validation复现入口（仅供将来经授权复核，需本地数据；使用新输出目录）：

```powershell
python p08_run.py --evaluation-config results/dataset_v2_baseline_comparison/configs/validation.json --config results/dataset_v2_baseline_comparison/configs/new_raw.json --model baseline1 --record-timing --output outputs/validation_replay/baseline1_new
```

同理可将模型改为baseline2、配置改为old_raw。`new_raw.json`/`old_raw.json`禁用GUI guard，仅比较原始分类器。已有配置快照按原字节保留，不作GUI接入指令；新guard运行时扩展已撤回。

离线`predictions/timing.csv`从已解码图像进入adapter开始，包含实际预处理、特征提取、推理、后处理及结果校验和必要设备同步，不含图片解码、模型加载、GUI或Camo传输。统计采用NumPy median / percentile(method='linear')。实机与纯forward延迟均未测量。

`predictions/images.csv`、`evaluation/summary.json`、`per_image.csv`及日志保持原始记录。`source_fingerprints.json`、GUI诊断及旧测试日志对应清理前状态；本次恢复/删除详情见`scope_cleanup.json`。未修改权重、训练阈值或原始评估成绩。
