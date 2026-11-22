# P04 建议与核对记录（Advice / Audit Record）

> 对应团队计划 §8.1 §8（Advice, Reproducibility）与 Stage 3 §6 / §12（LLM Use）。
> 规则：每条实质性建议记录来源、摘录、接受/修改/拒绝/搁置、理由、实现状态与代码/配置/输出引用；LLM 协助与本人工作分开记录。

## 记录 A-01

- **日期**：2026-11-22
- **来源**：LLM 编程助手（AI 辅助编码）
- **建议摘录**：baseline1 采用「HOG 手工特征 + 浅层 MLP」的经典图像级分类流水线；HOG 负责形状/纹理梯度特征，MLP 做整图二分类；用类别平衡样本权重处理 PASS/FAIL 样本不均衡；模型与预处理参数打包进 joblib bundle 存档。
- **决策**：接受（Accept）
- **理由**：与课程 PDF 给出的推荐基线「handcrafted features with MLPNN」一致，也符合团队计划 SW-01「HOG + MLP 图像级基线」；方法简单、可复现、训练成本低，适合作为弱基线。
- **实现状态**：已实现
- **引用**：
  - 特征：`src/p04/features.py`（`extract_hog`）
  - 模型：`src/p04/model.py`（`fit_mlp`、`balanced_sample_weights`、`make_bundle`）
  - 配置：`config/baseline1.json`（HOG/MLP 参数、随机种子 42）
  - 训练入口：`tools/train_baseline1.py`
- **核对方式**：`tests/test_p04.py` 校验 HOG 特征维度与配置一致、bundle 保存/加载/预测往返一致；训练后在验证集输出混淆矩阵与 P/R/F1。
- **未采纳部分**：未引入更深网络或数据增强——属于 baseline2/baseline3 与 Stage 3 改进范围，不在本基线职责内。

## 记录 A-02

- **日期**：2026-11-22
- **来源**：团队计划 `team_plan_ai.md` §5 接口合同 + `Project-Man.pdf` 输出合同
- **建议摘录**：批处理命令、CSV 字段顺序、status/score_type 枚举、run_config 必含键、图像级基线免交 objects/counts 且不得伪造零计数。
- **决策**：接受（Accept），作为硬合同执行
- **理由**：跨人对接的公共接口，优先级高于个人实现偏好；字段名与顺序不得自行改动。
- **实现状态**：已实现
- **引用**：`predict.py`（images.csv 列序、状态分支、run_config 键）；`config/baseline1.json`（`output_capability: image_level`）
- **核对方式**：对照 §5.3 / §5.4 逐项核对；计数输入校验拒绝布尔、浮点、负值与多余键。

## 待补充（真实数据到位后）

- 在真实训练/验证集上的一条失败案例：`image_id`、预测 vs 真值、可能原因、工程后果（供 Stage 2 §6 使用）。
- ≥20 张新鲜帧的实时延迟 median/P95 与硬件记录（当前仅合成数据上的离线计时，不能作为现场性能证据）。
- LLM 在调试、结果解读中的其他具体使用实例与人工核对结论，按 §12 要求在个人报告中补充。
