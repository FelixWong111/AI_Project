# P08 Stage 2 frozen results

|Model|全输入Accuracy|FAIL P/R/F1|TP/FP/FN/TN|False PASS/FAIL|REVIEW|Coverage|
|---|---|---|---|---|---|---|
|baseline1|96.67%|1.0000/0.9333/0.9655|14/0/1/15|1/0|0|100.00%|
|baseline2|86.67%|0.9231/0.8000/0.8571|12/1/3/14|3/1|0|100.00%|
|baseline3|93.33%|1.0000/1.0000/1.0000|15/0/0/13|0/0|2|93.33%|

Baseline3的28/28 completed accuracy不是全输入100%；两个REVIEW保留，主Accuracy28/30。B1/B2复用同SHA原始预测，B3第一次完整size_repair_v2 Test；历史Stage2已用，非全新独立证据。Stage3未运行。

每模型predictions含原始CSV与run_config，metrics由现有P08 evaluator生成；comparison/object_count_metrics.csv包含全部30图（含REVIEW预测）的目标/计数指标，error_cases保留逐图/逐对象错误。run_configs关联正式registry、原推理配置SHA与来源，未伪造B1/B2新运行。完整操作及限制见../../docs/p07/final_handoff.md。

数据图片通过网盘共享，不在GitHub；先按 [数据恢复说明](../../datasets/mechanical_components_v2/README.md) 恢复到根目录，再运行需要图片的命令。Git中的原始预测与指标可直接查阅。完整交接和模型SHA见 [P07交接](../../docs/p07/final_handoff.md)。
