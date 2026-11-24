# Baseline1/2 数据泄漏专项审计

后续版本说明：本文源码指纹对应当次审计版本。之后为 P07 Baseline1 增加独立的训练支持范围门控，未改变分类器权重、训练数据或原始批处理预测；当前系统变更与功能诊断见 docs/baseline1_live_diagnosis.md，不能将历史源码指纹误称为当前全部源码指纹。

结论：**本轮可检查范围内未发现数据泄漏或推理标签回退。保留当前两个候选冻结版本，Baseline1 验证准确率 100% 不变，Baseline2 为 80.4348%，不再调参。** 正式 Test 继续 NOT_RUN。

执行命令：

```powershell
.\.venv311\Scripts\python.exe -B -m scripts.audit_model_leakage
```

归档机器记录：[audit.json](../results/audits/leakage/audit.json)，来源为 outputs/leakage_audit/1795502444356390600/audit.json。308 张图片的实际路径和训练清单在 [development.json](../results/data_manifests/development.json)；审计重新解码后的图片 SHA256、像素 SHA256 在 [image_hashes.json](../results/audits/leakage/image_hashes.json)。原记录仍保留在 outputs，公开副本只规范化本机路径前缀，源/副本 SHA 均可复核。

| 项目 | 结论 | 核验依据 |
|---|---|---|
| Train / Validation 实际范围 | PASS | 262 / 46，清单没有 test，路径逐条限制在批准的开发分区 |
| 字节重复 | PASS | 308 张逐张重新计算 SHA256，集内和跨集重复均为 0 |
| 解码像素重复 | PASS | 308 张按 EXIF 方向重新解码 RGB 并计算尺寸+像素哈希，集内和跨集重复均为 0 |
| 数据与训练前一致 | PASS | 所有开发图片和两份开发 CSV SHA256 未变，清单版本匹配 |
| StandardScaler 隔离 | PASS | 两模型保存的 mean/scale 与 X_train 直接计算值一致；Baseline1 n_samples_seen_=262；fit 调用只传 X[trainmask] 与 y[trainmask] |
| 分类器与其他拟合 | PASS | MLP/LR 仅 fit 训练数据；验证只用于 predict 和有限候选选择；无 PCA、概率校准或验证集阈值搜索 |
| 冻结 CNN | PASS | 官方骨干 SHA256 匹配，eval=True、requires_grad=False、inference_mode，未做端到端训练 |
| 历史缓存污染 | PASS | 缓存签名包含当前清单、预处理、骨干与依赖；验证集缓存逐张与重新从图片提取的特征比较一致；缓存位于 P03 之外 |
| 旧 synthetic 模型 | PASS | 已隔离归档，部署路径指向本轮模型；骨干预训练与真实训练的 LR 参数分开记录 |
| 文件名/目录/标签成为推理特征 | PASS | HOG、CNN 仅接收像素；Scaler/分类器只接收数值特征；推理不读真值。Baseline2 另用 anonymous.png 元数据逐图复测，预测不变 |
| 训练验证 / 部署 / 批处理一致 | PASS | 每模型 46/46 标签一致，概率按约定转换后在绝对容差 2e-6 内一致，特征数值容差另记于脚本 |
| GUI 加载真实后端 | PASS | app.py 与批处理共用 load_backend，实际重载对象 is_mock=False，模型版本与当前权重相符；生产源码没有 MockBackend/Stub 类 |
| GUI 人工点击和摄像头表现 | 未完成 | 无头 app 和控件回调测试通过，渲染已检查；桌面自动化未找到可操作窗口，不能声称现场验收通过 |
| 正式 Test | NOT_RUN | 审计只打开批准的开发图片和开发真值，不读测试图片或测试答案 |

每张验证图三方预测与分数见 results/audits/leakage 中的 baseline1_per_image_comparison.json、baseline2_per_image_comparison.json。原始训练阶段预测保留于 results/baseline1、results/baseline2。历史验收记录见 [original_integration_acceptance.json](../results/audits/leakage/original_integration_acceptance.json)，相同预测内容只保留一份；CUDA 环境下最新完成回归见 [baseline12](../results/integration/baseline12/)。

此审计还保存了被审查训练、数据、Adapter、注册表、批处理和 app 源码的 SHA256。模型哈希与训练结束时一致：

- Baseline1 joblib：`67f99643fb16573f20922ec4ca78bb9881e8d0c1855997d595282f75fdea8d01`。
- Baseline2 JSON：`641023153cb56693a54bad0dcc8958beda5b7cd4ef4c2a90761e9c573f1c6e55`。
- ResNet18：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

## 证据边界

场景独立性依据本次用户人工确认“每个照片的场景都不一样”。原 P03 scene_id=TBD 保持不变，尚无拍摄记录级 SceneID 对应表。哈希检查和特征比较没有证明不同照片必然为不同物理摆放；本报告不作此推断。

上述结论覆盖当前候选的实际训练与部署流程，不追溯宣称所有历史训练版本均无泄漏，也不证明新测试集与开发集之间已经独立。本次不对 Test 做交叉像素检查，因为用户明确禁止读取其图片内容。

46 张验证图已经用于参数选择，成绩不能替代未触碰测试集上的泛化评估。100% 可能与当前小样本、受控背景或采集规律有关；不能从无误判推断模型在新相机条件下也无误判。当前不人为降低成绩、不筛图、不改标签、不再优化。

P03 原有 1081 个条目大小/修改时间及开发内容哈希未变；工作期间新增 115 个新版测试包条目，详见 acceptance.json。这里只记录新增元信息，没有修改或分析该测试包。保留原数据的声明与“整个目录文件数未变”是不同结论。

候选锁定清单为 `models/candidates.json`，P08 可直接通过 Git 获取本报告、训练报告、真实模型及 [results](../results/README.md) 中的原始预测和审计 JSON。候选工件中保留的 outputs 路径是历史执行位置，归档映射见 results/catalog.json。后续只有新证据显示数据分区或标签有问题时才重新评估训练需求；本次不再训练。
