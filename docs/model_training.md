# Baseline1/2 训练、验证与 P08 交接

后续实时诊断更新：原 Baseline1 权重和本报告指标保留；P07 单独增加训练支持范围 REVIEW 门控，原始 predict.py 基线不变。系统门控与基线性能不可混报，详见 docs/baseline1_live_diagnosis.md。

本次由用户接手 Baseline1/2 的最终实现、训练与集成。P04 原 HOG/MLP 与 P05 原 ResNet18/逻辑回归贡献保留，Codex 辅助完成数据校验、工程整理、训练执行、接口与结果核验；不把这些修改伪称为原成员完成的正式实拍训练。

**Official held-out test: NOT_RUN。** 未读取正式测试照片或测试真值进行训练、特征提取、模型选择、推理或评估。测试更新期间出现的新文件只在原目录不变检查中记录了路径、大小和时间等元数据，没有读取内容。

## 数据与确认来源

| 分区 | 图片数 | 场景数 | PASS | FAIL | 排除 |
|---|---:|---:|---:|---:|---:|
| Train | 262 | 262，依据用户人工确认 | 124 | 138 | 0 |
| Validation | 46 | 46，依据用户人工确认 | 24 | 22 | 0 |
| Test | NOT_RUN | — | — | — | — |

原始 scene_id 仍为 TBD。用户于 2026-11-24 明确说明“每个照片的场景都不一样”；衍生清单保留此确认及原始字段，不把每张照片编造为拍摄记录中的 SceneID。哈希无重复只证明没有相同文件/像素，不证明物理场景独立。

固定工单为五类各 1、ID 0 为 0。依据 P03 的 `reference_label_all_one`，与六类实际数量及逐框 CSV 逐张核对一致。标注状态 reviewed 被明确接纳；REVIEW/不可判定样本在数据加载器中排除，不自动归为 FAIL。

清单：`outputs/training/data/manifest.json`，版本 SHA256：`44498f23f97b84f7bf135a5d2f277baa0e3979c91af77af15cd116337adef027`。同一份清单用于两个模型。文件哈希、解码 RGB 像素哈希、原始 SceneID、分区、标签、实际路径均可追溯。

## 方法与参数

Baseline1：EXIF 方向解码，BGR 转灰度，拉伸至 128×128；HOG orientations=9、pixels_per_cell=(16,16)、cells_per_block=(2,2)、L2-Hys、transform_sqrt=True，共 1764 维。训练集 StandardScaler 后接 MLP，一层 32 节点，ReLU、Adam、alpha=0.01、learning_rate_init=0.001、max_iter=1000、n_iter_no_change=30、early_stopping=False，按训练类别频率计算平衡样本权重，seed=42。Scaler 和 MLP 作为同一个 Pipeline 保存。

Baseline2：官方 torchvision ResNet18 IMAGENET1K_V1，保持冻结、eval、inference_mode。RGB 等比例 fit 224×224 并补白边，bilinear；ImageNet mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225)，global-average-pool 输出 512 维。不使用中心裁切，不微调 CNN。训练集 StandardScaler 后接 LogisticRegression，C=0.1、lbfgs、max_iter=3000、无 class_weight、seed=42。部署 JSON 显式保存 P(FAIL) 方向的系数，避免 sklearn 字符串类别排序引起概率反向。

模型选择预先限制为 MLP 32/64 节点两项、逻辑回归 C=0.1/1/10 三项，按验证 FAIL F1、再较少 False PASS、再简单候选选择。未搜索其他预处理或阈值；阈值固定 0.5。各候选无收敛警告。专项审计阶段不再训练或调参，不降低 Baseline1 的真实成绩。

实际环境：Python 3.11.14、NumPy 1.26.4、SciPy 1.15.3、sklearn 1.9.1、skimage 0.26.0、joblib 1.5.3、Pillow 13.0.0、OpenCV 4.10.0.84、torch 2.7.1+cpu、torchvision 0.22.1+cpu。Windows AMD64 CPU，训练数值计算限制 4 线程，逻辑处理器 24；环境未提供可核实的 CPU 型号和内存信息，不编造型号。没有 GPU 训练。

## 真实结果

FAIL 为 positive class。矩阵行是真值、列是预测，顺序 PASS / FAIL。

| 指标 | Baseline1 | Baseline2 |
|---|---:|---:|
| Training Accuracy | 100% | 100% |
| Training TP / FP / FN / TN | 138 / 0 / 0 / 124 | 138 / 0 / 0 / 124 |
| Validation Accuracy | **100%** | **80.4348%** |
| Validation Precision (FAIL) | 1.0000 | 0.8421 |
| Validation Recall (FAIL) | 1.0000 | 0.7273 |
| Validation F1 (FAIL) | 1.0000 | 0.7805 |
| Validation TP / FP / FN / TN | 22 / 0 / 0 / 24 | 16 / 3 / 6 / 21 |
| False PASS / False FAIL | 0 / 0 | 6 / 3 |
| 训练流程耗时，含特征提取与小规模候选选择 | 271.28 秒 | 216.58 秒 |
| 必需推理模型大小 | 955169 bytes，另需 config/freeze | 46875112 bytes，JSON+骨干 |

```text
Baseline1 Validation   Baseline2 Validation
[[24,  0],            [[21,  3],
 [ 0, 22]]             [ 6, 16]]
```

Baseline1 的 64 节点候选 Accuracy=95.6522%、F1=0.9545；被选中的 32 节点版本无验证误判，不编造失败案例。Baseline2 三个 C 的主要指标相同，保留 C=0.1。

Baseline2 False PASS：fail171.jpg、fail27.jpg、fail29.jpg、fail35.jpg、fail38.jpg、fail43.jpg。False FAIL：pass102.jpg、pass133.jpg、pass15.jpg。原始概率见 validation_predictions.csv。其中 fail171 的 P(FAIL)=0.49262、fail35=0.48958，接近固定阈值；fail38=0.16070，不能只用阈值附近波动解释。冻结通用 CNN 的整图表征可能难以稳定区分零件数量、细小尺寸或陌生件，训练全部正确而验证有误也提示泛化有限；这只是合理解释，需进一步图像人工复核或新独立数据验证，不能由指标直接确定原因。本轮不改阈值、不筛除误判样本。

46 张验证图较少，且已用于候选选择。100% 不能外推为任意新场景的性能，更不能写为正式 Test 100%。现场相机条件、尺度变化和目录外件定位能力没有由此得到验证。

## 模型版本与文件

- Baseline1 bundle SHA256：`67f99643fb16573f20922ec4ca78bb9881e8d0c1855997d595282f75fdea8d01`。
- Baseline2 model_version：`0d6965aeb43dc7f259f96ddddf8e083c0f48430ae5922011cff103bb859329e1`。
- Baseline2 JSON SHA256：`641023153cb56693a54bad0dcc8958beda5b7cd4ef4c2a90761e9c573f1c6e55`。
- ResNet18 SHA256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

完整参数、候选结果、文件大小/哈希、环境和数据引用在两个模型目录及 outputs/training 对应目录的 training_record.json 中。training_predictions.csv 与 validation_predictions.csv 保留全部真实 image_id 与概率。

实际训练命令：

```powershell
.\.venv311\Scripts\python.exe -B -m scripts.train_baseline1 --prepare-only
.\.venv311\Scripts\python.exe -B -m scripts.train_baseline1
.\.venv311\Scripts\python.exe -B -m scripts.train_baseline2
```

后续重复实验必须用新目录，不覆盖候选。当前用户要求不再优化，本轮不重复训练。

## 集成与验收

生产注册表同时供 app.py 和 predict.py 使用，两个真实模型能够重新加载；全部 46 张验证图的批处理标签和概率与训练记录逐张一致。固定工单正常，其他合法工单 UNSUPPORTED_COUNTS，缺失权重明确报错，无图像级虚构计数或目标文件。无头 app 对允许验证图完成真实推理和导出。

46 个 unittest 与原 Baseline1 的 4 个独立检查通过，pip check 通过。控件回调测试覆盖模型切换、导出、工单修改、录制和缺失模型恢复。截图经过实际渲染检查；桌面自动化未找到可操作窗口，不能写为人工点击或现场摄像头已验收。摄像头实时表现与至少 20 新帧延迟仍 NOT_RUN。

P03 原有文件路径、大小和修改时间未变，开发图和两份开发 CSV 的 SHA256 未变。工作期间其目录外部新增新版测试包，因此整个 P03 目录的文件总数发生变化；不能将全目录写成完全未变，也未删除这批新增文件。原有内容保护检查及新增路径清单见 acceptance.json。

专项泄漏审计由 scripts/audit_model_leakage.py 输出到 outputs/leakage_audit；最终审计结论另见 docs/data_leakage_audit.md。审计不重新训练，不读取测试照片或测试真值。

## 交给 P08 的材料

本文件、docs/data_leakage_audit.md、docs/system_architecture.md、docs/model_integration.md；models/baseline1 与 models/baseline2 的真实文件；outputs/training/data 清单及工单；各模型全部训练/验证预测和 training_record；outputs/acceptance 的验收记录；outputs/leakage_audit 的逐图审计。outputs 被 Git 忽略，打包交接时必须另行包含，不能只提交源码而漏掉证据。

本次没有提交 Git、没有改分支、没有执行正式 Test。后续待 P03 测试集封存并确认与开发场景隔离，再由 P08 独立评估；如果新证据推翻当前人工场景确认，需要重新审查分区与重训需求。Baseline3/Final 真实权重及完整目标级 evaluator 仍未交付。
