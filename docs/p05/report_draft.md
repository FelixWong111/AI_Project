# P05 报告与个人证据素材

课程实拍结果待训练和验证。每个 `TBD` 用 P08 核对的真实运行记录填写。合成验证只证明代码与接口可运行。

## Stage 2 第 4 节 · Three Baseline Methods and Implementation

Baseline 2 performs whole-image PASS/FAIL classification for the fixed expected-count vector agreed by the team. We use the ImageNet-pretrained ResNet18 backbone in evaluation mode and take its 512-dimensional global average pooling output. Input images retain their aspect ratio and receive white padding to 224 × 224 before ImageNet channel normalization. We fit a standard scaler and a regularized logistic regression classifier on the training partition. A frame is labeled FAIL if any required count differs from the verified count or a non-catalogue part is present. Validation uses separate physical scenes. The adapter reports estimated P(FAIL) with a fixed decision threshold of 0.5. It returns `UNSUPPORTED_COUNTS` for a different valid expected-count vector.

方法出处。[He et al., Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)、[torchvision ResNet18 与权重](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)、[scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。

## Stage 2 第 6 节 · Weaknesses and Failure Analysis

| 项目 | 由正式运行补齐 |
|---|---|
| 测试运行与图像 | `run_id=TBD`，`image_id=TBD`，条件 `TBD` |
| 真值与预测 | 五类真值数 `TBD`，目录外件 `TBD`，真值标签 `TBD`，预测 `TBD`，P(FAIL) `TBD` |
| 错误解释 | 根据原图和标注说明误 PASS 或误 FAIL。可核对零件尺寸与外观相似度、遮挡或低照 |
| 独立佐证 | 引用正式 `images.csv` 对应行和一张能指出错误的原图。记录其他可能解释 |

## Stage 2 第 7 节 · Initial Decision Record and Improvement Plan

将 `review/decision_stage2.md` 的定稿压缩成一段，包含 `baseline3` 具体失败证据、拟改的因素、主指标、最小有意义提升、可接受代价与独立对照安排。当前 CLAHE 为候选，依据低照漏框证据决定。

## Stage 3 第 4–5 节 · Research Source, Adaptation and Hypothesis

Zuiderveld’s contrast-limited adaptive histogram equalization enhances local contrast using tiled histograms with a contrast limit. The candidate adapts this method to the Lab lightness channel before object detection. The testable hypothesis is that local lightness enhancement increases localization recall under low illumination. The experiment uses paired runs on the same independent images and measures the additional live latency and false rejections.

方法原文与实现参数出处。[Zuiderveld, Contrast Limited Adaptive Histogram Equalization](https://doi.org/10.1016/B978-0-12-336156-1.50061-6)、[OpenCV CLAHE API](https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html)。

采用情况、假设修订、对应运行和实际结论填 `TBD`。P08 提供同图对照结果，至少记录正常和两种困难条件的指标与支持量。

## Stage 3 第 6 节 · Actual Implementation and Advice Audit

候选代码 `src/stage3_clahe.py` 的 `DetectorCLAHE` 只处理检测器输入并保留坐标。由 P06 填最终接入位置、冻结参数、配置和命令，由 P05 核对与决策一致。`final` 的集成与课程数据上的对照状态为 `NOT_RUN`。

## LLM Use 与个人复核

AI 协助编写数据读取、冻结特征、预测适配器、CLAHE 候选、草稿和 PPT。建议保留的一条核查记录是采用保比例白边预处理，并在合成样图上核对保存前后的分数和批处理/帧输入的结果。本人在 P03 正式数据到位后复跑命令，再挑一例误判核对图像 ID、标注和输出行，填入个人报告的贡献、验证、替代解释和下一步。软件工作流参考。[Kassis et al., Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。

## 30–40 秒讲稿

我负责第二套基线。先用冻结的 ResNet18 提取每张图的 512 维特征，再用逻辑回归判断固定预期数量下整盘是否合格。改进部分会根据目标级基线的错误来选。如果主要问题是低照漏检，就在检测前对亮度通道做 CLAHE，并用同一组独立场景比较召回率、误拒和实时延迟。正式汇报时用 P08 核对的实际数字替换待测项。
