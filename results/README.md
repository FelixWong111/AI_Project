> 当前正式入口：[P08三模型最终结果](p08_stage2_final/README.md)、[P07完整交接](../docs/p07/final_handoff.md)。正式Stage2数据在datasets/mechanical_components_v2，采用Git LFS；原始及Stage3仍不上传。软件已集成，物理/实机/完整干净ML复现未验收，整体NOT_READY。以下旧档案说明保留历史时间与来源。

# 正式实验结果索引

2026-11-25新增：[Baseline3尺寸模式集成验证](integration/baseline3_size_20261125/README.md)。该子档案有独立来源/SHA清单；下述原冻结模型记录、预测和指标保持不变。接入成功不代表物理标定或Stage2验收通过。

本目录通过 Git 交付现有冻结模型的真实证据。来源仍保存在本机 outputs/，没有删除或覆盖；本轮只整理文件、提取已有指标/误判行，没有训练、推理或新增性能评估。正式 **Test: NOT_RUN**。

## P05 / P08 查阅顺序

1. 打开 [catalog.json](catalog.json) 核对当前模型版本，再看 [completeness.json](completeness.json) 了解证据缺项。
2. P05：重点看 [Baseline2 训练记录](baseline2/training_record.json)、[逐图验证预测](baseline2/validation_predictions.csv)、[指标](baseline2/metrics.json)、[误判清单](baseline2/errors.csv)。p_fail 是 P(FAIL)，不是永远代表预测类别的置信度；分数未校准。
3. P08：先用 [统一开发清单](data_manifests/development.json) 核对分区、真值和图片/像素哈希，再看各模型逐图结果、[泄漏审计](audits/leakage/audit.json) 与 integration/ 中的验收。不要将功能测试、Training 拟合成绩、Validation 选择成绩混进 Test 指标。
4. Baseline3 的 [验证推理输出](baseline3/validation_inference/) 包含 images/counts/objects/run_config；对象有置信度，整图计数规则没有概率。它们是已有真实推理记录，不是新的目标级真值评估报告。checkpoint 历史指标也不是本轮复验成绩。
5. [完整文件树](DIRECTORY_TREE.txt)、[逐文件来源与SHA256](file_manifest.json)、[全部 outputs 盘点](audits/archive/outputs_inventory.csv) 支持复核来源与去重决定。

## 目录用途与版本

| 目录 | 内容 | 版本依据 |
|---|---|---|
| baseline1/ | 原始 Training/Validation 预测、训练记录、指标、误判、独立场景门控诊断 | joblib SHA256 67f99643…d01；门控与原始分类器指标分开 |
| baseline2/ | 原始 Training/Validation 预测、LR候选、指标和误判 | model_version 0d6965ae…e1；JSON SHA256 64102315…e55 |
| baseline3/ | P06 checkpoint 训练曲线、来源、现有46张Val真实CUDA推理 | best.pt SHA256 4c5a1f75…b3d |
| data_manifests/ | 两分类器实际训练清单、缓存来源元数据、固定/特殊工单 | manifest records SHA256 44498f23…027；不冒充P06实际训练清单 |
| audits/leakage/ | 历史泄漏审计、逐图三方比较、哈希 | 指向当前相同分类器权重；源码指纹是当时版本 |
| audits/archive/ | outputs全量数量/大小/用途/版本归属/排除原因及保护检查 | 本次只读整理快照 |
| integration/ | CUDA环境、三模型控制器/批处理验收、最新完成工单验收、单元测试日志 | 各run_config和acceptance中的真实版本 |

Baseline1/2 各262张训练、46张验证：训练准确率均100%；验证准确率分别100%、80.4348%。Baseline2验证 False PASS=6、False FAIL=3；Baseline1为0/0，空 errors.csv 只有表头表示真实零误判，不补造样本。矩阵顺序PASS/FAIL、FAIL为positive class。验证集曾用于有限候选选择，不能代替独立Test。

场景独立性沿用已有人工确认；原始 scene_id=TBD 未填造，哈希无重复不证明物理场景独立。Ground Truth 是实际数量/真实标签；work_orders 是要求，不由GT反推。functional 工单只能独立运行，不计共同性能指标。

## 缺项与未执行项

- Baseline3未收到实际训练/验证图片清单、训练逐图预测、GT修正后复验及640/960对照报告；没有补造这些记录，也不把P07共用开发清单当作P06训练证据。
- Baseline3现有验证推理没有另行生成GT混淆矩阵/False PASS/False FAIL或框级指标。本轮归档不新增评估，不将13个预测ID0均触发FAIL解释为未知件识别准确率。
- 现场人工GUI点击、真实相机和20新帧延迟未验收；Final未配置；正式Test未运行。尺度标定、尺寸容差、错规格与未见目录外件能力缺口见 [功能审查](../docs/p06/compliance_review.md)。
- 没有原始图片/标注文件、模型权重、NPZ特征缓存、备份ZIP、视频或截图。图像级标签和数量作为已批准的Training/Validation审计清单保留；不含正式Test真值。单元日志中的STUB/MOCK是明确的测试夹具，不是生产模型或正式性能证据。

## 来源、去重与可移植路径

同内容只保留一份，file_manifest.json 和 catalog.json 将多个原来源映射到同一归档文件。完整记录保留原始实际命令、退出码和运行标识；其中 outputs/ 路径是历史执行位置，通过 source_to_archive 查归档位置，不要求GitHub克隆中存在 outputs/。旧实验的本机路径前缀统一为 `${PROJECT_ROOT}` / `${P06_PROJECT_ROOT}` / `${USERPROFILE}`，不发布个人用户名；数值、标签、参数、模型版本和原始文件名不变。manifest同时记录源文件和公开副本SHA，路径规范化后的文件SHA可能不同。

.gitattributes 对 results 禁用换行自动转换，确保 Windows/Linux 克隆得到相同的证据文件字节与SHA256。完整 outputs 盘点仅按路径/目录推断混合日志所属模型；精确版本以各run_config、训练记录和catalog为准。

只验证已归档文件，不调用任何模型：

```powershell
.\.venv311\Scripts\python.exe -m scripts.collect_results --verify
```

此验证在原工作区还会比对本机来源；新克隆缺少outputs时，使用下述标准库命令仅核对Git内交付文件，不需要安装推理依赖：

```powershell
python -m scripts.collect_results --verify --archive-only
```

复跑推理/审计仍需要各成员自行取得经过批准的Train/Validation原图；Git提供复核报告，不包含原始数据。原始模型仍从 models/ 加载。实验生成脚本继续写 outputs/，本任务没有修改系统集成逻辑。新实验应使用新输出目录和新归档版本，不能覆盖本次冻结证据。
