# Baseline1 实时异常诊断与系统门控

原始候选权重未修改、未重新训练，原始验证 100% 保留。正式 Test NOT_RUN，P03 未写入。本轮诊断全部使用已有训练特征、6 张训练原图及临时合成图；没有采集人物画面。

## 原因与证据

训练 PASS=124、FAIL=138，类别顺序明确为 [FAIL,PASS]，不是严重类别不平衡或单类别模型。训练诊断矩阵（行真值/列预测，PASS,FAIL）为 [[124,0],[0,138]]；这里只是拟合链路检查，不是正式评估。全部逐图概率保存在 [training_diagnostics.json](../results/baseline1/diagnostics/training_diagnostics.json)。另直接从原图重算 pass01/02/03、fail02/100/101，标签均正确，概率见同目录 diagnosis.json。

训练和部署均为 EXIF 解码、BGR 转灰度、128×128、HOG 1764维，相同 orientations=9、cell16、block2、L2-Hys、sqrt；同一保存的 StandardScaler/MLP Pipeline，Scaler n_samples_seen_=262。GUI 仍加载原 SHA256 为 67f99643fb16573f20922ec4ca78bb9881e8d0c1855997d595282f75fdea8d01 的真实权重，无默认 PASS、标签查表或生产 Mock。既有审计也已验证部署与训练逐图一致。

固定噪声图复现 PASS，P(PASS)=0.9999946，界面三位小数显示为 1.000。纯黑与纯白原始输出其实是 FAIL，P(FAIL)=0.93654。因此不是所有画面一律默认 PASS，而是闭集二分类 MLP 在分布外没有拒绝能力：sigmoid/softmax 仍会给出很大的未校准概率。噪声的标准化特征最大绝对值约19.7，超出训练支持范围仍会外推。训练和小验证集全对不能证明对人物、书本或任意背景有效，也不能排除过拟合/场景依赖。

## 修复边界

保留原始 HOG+MLP 算法与工单、输出合同，批处理 predict.py 继续报告原始基线结果。仅 P07 加入单独的系统门控：在相同 Scaler 的标准化 HOG 空间，计算到262个训练向量的最近距离；门限取训练集 leave-one-out 最近距离的最大值63.34594。门限不使用验证集、异常图或正式Test选择。没有调用任何其他模型，也没有调分类阈值。

超出支持范围时返回 REVIEW，predicted_label/prediction_score=null，不强制 FAIL；原始分类标签和完整概率保留在 raw_baseline_prediction 诊断字段。模型元数据标明 system_scene_guard、规则、门限和工件SHA。工件绑定原模型 SHA，缺失或不匹配即加载错误，不回退原始 PASS。

| 输入 | 原始 Baseline1 | P07 系统结果 |
|---|---|---|
| 3 张已知训练 PASS | PASS | PASS |
| 3 张已知训练 FAIL | FAIL | FAIL |
| 黑/白/灰空图 | FAIL，0.93654 | REVIEW |
| 固定随机噪声 | PASS，0.9999946 | REVIEW |
| 合成书本文字纹理 | FAIL，0.9999924 | REVIEW |

46张验证图的特征均在当前门限内；这是门控覆盖率检查，不用于选门限，不是系统O O D准确率。人物和真实杂乱书本现场画面本轮未采集，不能声称全部无关场景均已解决。最近距离只是经验支持检查，不是机械零件检测器；相似HOG的无关画面仍可能通过，合法的新拍摄条件也可能触发REVIEW。现场应恢复训练对应的白色桌面、稳定视角和光照。后续如果要扩大工作范围，应独立收集场景有效性数据并在训练内部开发，另行验证，不借正式Test调门控。

复现：python -m scripts.diagnose_baseline1（需本机原始开发图、特征缓存和 outputs 清单；新克隆直接查阅现有归档即可）。完整概率、距离、参数和工件哈希在 [diagnosis.json](../results/baseline1/diagnostics/diagnosis.json)。此脚本不覆盖已有门控工件、不重训MLP。软件启动与原来一致，模型1自动启用门控；其他三个模型不受影响。原始批处理与带门控系统必须分别报告。
