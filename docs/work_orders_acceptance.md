# 工单规范验收记录（2026-11-24）

实际报告：outputs/work_orders_acceptance/1795506729080232400/acceptance.json。对应目录保留每条命令、退出码、原始CSV、运行配置和无头app快照。正式 Test: NOT_RUN。

使用允许的 validation/pass102.jpg，不使用正式测试图片。Baseline1 固定工单输出 PASS、P(label)=0.81413716；Baseline2 输出 FAIL、P(FAIL)=0.52217396。后者是已保留的真实误判，不将“真实分类执行成功”混同为分类正确，也没有为验收改标签或模型。

| 用例 | 实际结果 | 证据/范围 |
|---|---|---|
| U01 Baseline1 固定工单 | PASS：真实分类执行 | 真实权重，验证图，P07门控启用，app与CLI一致 |
| U02 Baseline2 固定工单 | PASS：真实分类执行 | 实际判为FAIL；不是说该图片判对了 |
| U03 Baseline1 额外ID5要求 | PASS | UNSUPPORTED_COUNTS，空标签/分数，HOG调用数不增加 |
| U04 Baseline2 额外ID5要求 | PASS | 同上，CNN调用数不增加 |
| U05 两模型全零要求 | PASS | 两者均不支持，不当成FAIL |
| U06 恢复默认 | PASS | 从从未分类的帧恢复时真实推理；已有相同帧有效缓存可恢复，不需要重启 |
| U07 非法输入 | PASS | 缺类、负数、小数、bool、ID0等拒绝；CLI退出2，不加载模型或创建输出目录 |
| U08 模型切换与旧结果清除 | PASS（自动状态/控制器检查） | 实际后端状态检查、GUI控制器事件测试；人工桌面点击 NOT_RUN |
| U09 特殊工单批处理 | PASS | 两模型×4种合法特殊向量，8次独立运行，状态及空预测字段正确 |

额外覆盖 [2,1,1,1,1] 与 [0,1,1,1,1]；除外层ImageBackend外，直接模型桥接/Adapter入口也检查了分类跳过。实际app以无头模式对两个模型分别执行固定与特殊工单，输出与CLI一致。

Baseline3/Final 的真实权重/适配器尚未配置，因此真实检测及自定义工单现场验收均 NOT_RUN。目标级数量重判、ID0强制FAIL、框和实际数量不变，仅在明确的单元接口夹具上通过，不宣称真实检测能力完成。

52项单元测试通过，日志在 outputs/work_orders_acceptance/unit-tests-final.log。生成器测试覆盖“实际ID5=2但工单仍各1”、真实文件扩展名、显式旧编号映射、重复ID及非法输入；没有根据Ground Truth生成要求。

## 生成物

- outputs/work_orders/train.json：262条固定参考工单。
- outputs/work_orders/validation.json：46条固定参考工单。
- outputs/work_orders/functional/extra_id5.json、extra_id1.json、zero_id1.json、all_zero.json：各46条验证图特殊工单，只供功能测试，不用于共同性能指标。
- 本轮真实功能验收为避免无意义重复推理，仅从上述同一验证分区选择一张图片，每种工单单独运行，输入保留在验收目录。
- 正式Test工单本轮不重新生成，不执行Test推理或评估。

## 不变项

逐文件SHA256核对：任务开始时 models/ 下的所有工件，包括Baseline1 joblib、场景门控、Baseline2分类参数、ResNet18及训练记录，全部不变。config/baseline1.json 冻结预处理配置也未改。两个已训练模型内部工单与训练记录确认为各1，不是只改运行配置。

P03没有写入；任务前后全目录文件名、大小与mtime一致。此项只用元数据检查，不读取正式测试图片内容。未重训、未调预处理/阈值、未修改已有成绩、未提交或推送Git。

新增核心文件：config/inspection.json、src/work_orders.py、scripts/generate_work_orders.py、scripts/check_work_orders.py、tests/test_work_orders.py及本组接口文档。修改了app、UI展示、两个Adapter及桥接、批处理、开发数据准备、未来训练/测试整理脚本与配置引用；历史工件和原始预测保留。输入校验只有一份实现，没有引入新依赖。

修改清单：app.py；config/p07.json、config/training.json；src/p07_bridge.py、src/p05_bridge.py、src/p07_ui.py、src/p07_dashboard.py；src/adapters/baseline1.py、baseline2.py；src/inference/batch.py；src/data/development.py；scripts/train_candidates.py、scripts/prepare_test_dataset.py；tests/p07_stubs.py、tests/test_p07.py、tests/test_p07_real_integration.py；README.md、docs/model_integration.md。现有测试夹具改用公共参考，不代表重训。
