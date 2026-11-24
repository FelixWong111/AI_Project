# 模型部署

工单接口现统一使用 config/inspection.json 与 src/work_orders.py，见 docs/work_orders.md。冻结模型内部工单不改；加载时核对公共参考。两个图像级模型在特征提取前拒绝其他合法工单，GUI立即清空旧判定和分数。检测模型 count-only rejudge 由统一包装层计算，不改变检测观测。

Baseline1：`config/baseline1.json` + `models/baseline1/baseline1.joblib` + 同目录 `freeze.json`，加载时校验 SHA256、预处理、类别映射和 sklearn 版本。训练时 StandardScaler 与 MLP 一起保存于 Pipeline。

Baseline2：`config/baseline2.json` 指向 `models/baseline2/`，其中 `baseline2.json` 保存本项目真实训练得到的 StandardScaler 与逻辑回归参数；`resnet18-f37072fd.pth` 是 torchvision 官方 ImageNet 预训练骨干，保持冻结。两者来源与作用不同。模型包可离线加载。

本项目运行 Baseline2 需要 PyTorch 和 TorchVision；Baseline1 算法本身只需要 sklearn/skimage，但完整演示环境统一安装两套依赖。模型加载不下载权重。

GUI：模型按钮或 1/2 快捷键选模型，L 重载；改变 expected 后，图像级模型仅接受训练固定工单。观测计数和 ID 0 输出为 N/A。Baseline3/Final 继续通过 bridge=module:factory 扩展，本次未交付真实检测模型；无配置时 ERROR，不伪装检测成功。

旧 `--backend auto/mock` 参数仅为旧命令兼容保留，均不能启用生产模拟推理。合成实现只存在于 tests/p07_stubs.py。旧合成训练器和旧冻结材料已归档，不能用于正式推理。

批处理使用根目录 predict.py。它按 counts_file 中的 image_id 读取图片，严格校验五个非负整数、重复 ID 和文件名路径。每次使用新的输出目录，避免覆盖证据。只读取图片、工单及模型配置，不读取真值。加载失败仍生成 ERROR 行和 run_config 错误信息，并以非零退出码结束。
