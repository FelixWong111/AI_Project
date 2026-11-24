# 系统结构与接口

`app.py`（OpenCV 实时界面）和 `predict.py`（课程批处理入口）共同调用 `src/p07_bridge.py:load_backend`。注册表保留 baseline1、baseline2、baseline3、final 四个 ID；所有生产路径均为真实模型，未配置的检测模型报告 ERROR 和缺失原因。

Baseline1 的 HOG、MLP 序列化和推理位于 `src/models/baseline1/`，沿用 P04 的算法贡献。Baseline2 的冻结 ResNet18 和分类参数推理位于 `src/adapters/baseline2.py`，沿用 P05 的算法贡献；`src/p05_bridge.py` 负责帧结果缓存和工单重判。`src/adapters/baseline1.py` 校验配置、模型和 freeze 校验值。UI 不实现特征或分类算法。

`scripts/train_baseline1.py`、`scripts/train_baseline2.py` 调用同一训练流程；`src/data/development.py` 只访问指定开发包中 split=train/validation 的图片和对应真值，检验数量、标签、框、重复和分区。衍生清单、特征缓存和实验结果写入 `outputs/training/`，不会写进 P03。

输入为 uint8 BGR 帧与五类 expected_counts。图片文件统一按 EXIF 方向解码；Baseline1 灰度缩放后提取 HOG，Baseline2 转 RGB 后按声明流程提取 512 维特征。模型权重和配置不包含真实标签查表。

输出合同不变：`images.csv` 字段为 `image_id,model_id,status,predicted_label,prediction_score,score_type,non_catalogue_count`。图像级模型不支持目标框和计数，non_catalogue_count 为空，不生成 objects.csv/counts.csv。合法但不同的工单返回 UNSUPPORTED_COUNTS，标签与分数为空。

源文件缺失、模型校验不匹配、图像读取失败、推理失败均显式报告错误，不回退模拟结果。热重载检测到模型或配置变化时要求重新加载。两个基线均不支持任意工单，也不能证明 ID 0 被正确定位。

本次接手修改了团队计划原先的成员 owner 分工，依据是用户本次明确授权；公共字段和枚举未改。工作区当前未提供 P08 完整目标级 evaluator，本次只做图像级训练/验证与接口检查，不声称目标级评估完成。

`Official held-out test: NOT_RUN`。新测试集更新后，由 P08 使用封存版本进行独立评估；若其与开发集有同场景冲突，需先解决分区并评估重训需求。
