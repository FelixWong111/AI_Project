# Historical experiments

vision_caliper_v3/source_snapshot.zip保留原目录路径和精确源码字节，原始A/B/C实验仍在results/baseline3_v2_experiment/。解压到独立历史工作副本，按results/p07_final_integration/dataset_migration.json恢复当时数据路径后可复现；不进入正式GUI/CLI。baseline3_training_checkpoints保留未选中的训练checkpoint，严禁替代best.pt。

已发表旧冻结权重为避免破坏历史配置保留在models/baseline1、baseline2、baseline3及*_v2_candidate，只有models/release被正式注册表加载。旧参数/预测/失败证据没有改写。
