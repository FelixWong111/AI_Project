# 工程整理与恢复说明

分支始终为 p07/system-integration；起始提交 bf4fda4，起始工作区已有 P07 未提交修改。没有切分支、commit、push、reset 或 clean。改动前将源码、模型记录及当时未提交修改保存到 outputs/refactor_backup_20261124/before.zip、git-diff.patch 和 git-status.txt。

| 类型 | 处理 | 原因 |
|---|---|---|
| 移动 | src/p04 → src/models/baseline1 | 按模型职责组织，保留 P04 的算法与贡献来源，更新全部有效 import |
| 替换 | 原 Baseline1 dataset.py → src/data/development.py | 原字段格式不同、数量错误可能静默补零；新加载器严格校验 P03 字段、数量、框、状态、分区、哈希 |
| 删除并归档 | 原 Baseline1 evaluate.py、pipeline.py 及项目代码字节缓存 | 无生产调用，旧指标与加载辅助被共享训练流程替代；测试改为直接读取 JSON；源文件保留在 before.zip，不清理虚拟环境或 P03 |
| 替换 | tools/train_baseline1.py、旧 scripts/train_baseline2.py → 两个 scripts 入口及共享 train_candidates.py | 两模型统一数据与选择协议，训练证据一致，不保留两套冲突流程 |
| 替换 | predict.py | 原只实现 Baseline1、Baseline2 为 Stub；现为共享真实后端入口 |
| 移动 | 根 README_P05.md、docs/p04/README_P04.md → docs/history | 保留历史交接与贡献，不再作为当前操作说明 |
| 整理 | README_P07.md → docs/ui_operations.md | 当前 UI 操作集中说明，旧版原文已在改动前备份 |
| 移动 | models/baseline1 旧 synthetic freeze/metrics/config/samples → docs/history/baseline1_synthetic | 保留旧冻结证据，避免误当本次真实模型 |
| 隔离 | MockBackend → tests/p07_stubs.py | 合成预测仅用于单元测试，生产模型无回退路径 |
| 移除有效入口 | scripts/p05_predict.py | 与根目录批处理重复，归档后统一调用 predict.py |
| 删除并归档 | tools/make_demo_data.py、scripts/p05_smoke.py、tools/p07_verify.py、tools/p07_preview_ui.py、config/p07-baseline2.json | 已被真实训练、独立测试夹具、候选验收和 config/baseline2.json 替代；原文件在 before.zip/retired.zip |
| 隔离并精简 | P05_handoff_complete_20261123 | 旧源码与合成分类器归档到 p05_historical_handoff_without_backbone.zip；官方 ResNet18 权重单独保留在真实 models/baseline2 包，移除无用重复副本 |
| 新增 | requirements.txt | 单一运行依赖入口；旧 P04/P05 安装入口改为引用它，避免 headless 与普通 OpenCV 冲突 |
| 保留 | P03、原始开发/验证/测试、虚拟环境、Project-Man、team_plan、建议/决策记录、CLAHE 候选、相机工具、测试输入生成器 | 权威原始数据、来源和成员贡献，或仍被使用/未来扩展，不按名称冒险删除 |

P03 目录没有任何本任务写入操作。工作期间新增了一份新版测试包，因此全目录快照不同；原有条目未变，新增条目只记录元数据，没有读取测试内容。没有将新版测试包纳入开发数据或特征缓存。

正式模型不被 .gitignore 忽略。旧数据包、虚拟环境、缓存和输出被忽略，真实模型、配置和完整实验资料需要在最终交付时一并提供。模型和资料保持本地真实文件，没有只留下载说明。

恢复时先查看 change-plan.json、git-diff.patch 和 ZIP 内容，根据需要恢复个别文件，不能直接覆盖当前 P03 或其他人的新工作。这里没有创建自动恢复或破坏性清理脚本。

未进行无关 UI 风格重设计；相机、视频、快照、录制、工单修改和四模型按钮保留。旧生产 Mock 场景按钮移除，检测模型尚未交付时明确 ERROR。P08 完整目标级 evaluator 在当前仓库中不存在，本次没有覆盖其实现；公共字段和枚举保持不变。
