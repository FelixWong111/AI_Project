# Baseline3 尺寸开发候选 v1：实际结果与使用限制

结论：已完成真实尺子刻度拟合、220张专项测量、有限参数选择、保留场景检查、Train/Val完整管线比较和P07软件接入。候选恢复了一部分正常工单PASS，但出现正常件误拒绝、较多REVIEW和明显耗时增加，**不建议替换默认配置，完整Stage 2物理验收尚未通过**。

本记录只说明本次开发；不替P08撰写正式报告。没有重新训练或改变YOLO置信度/NMS，没有重跑Stage 2 Test，没有打开Stage 3图像或专用标注/评价文件，也没有用其数据训练、选参或评价。

## 1. 数据、人工确认及实验边界

`human_confirmation.json`保存本次用户确认及原文SHA。原始规格文件夹中每夹一个实物，不同照片不同摆放场景；固定机位确认不等于镜头、数字变焦和Camo视野完全相同。

共220张、8个实物；seed=42分为154张开发和66张保留场景。保留场景与开发集共享实物，不是独立实物泛化。原始HEIC与派生PNG逐张做EXIF方向转换后的RGB像素比对，220/220一致。

原冻结P06算法的固定基线曾一次性预计算220张；参数选择只读取154张开发输出，66张基线输出没有用于选参。优化后的66张测量在尺寸参数冻结后运行。未记录的外部历史使用情况UNKNOWN。

读取边界披露：初始公共目录/EXIF检查及前景开发曾读取统一dataset_manifest.csv来过滤Train/Val，该公共索引也列出封存分区元数据；封存行没有进入图像读取、选参或评价。后续代码已改用仅含440张Train/Val的development/ground_truth_images.csv，避免再次读取混合索引。严格意义上不能声称连保留集索引元数据都未曾读取。

| 实物编号 | 原始规格目录 | 官方ID | 张数 | 开发/保留 |
|---|---|---:|---:|---:|
| DIM-SPECIMEN-01 | M4×10细螺丝_ID3_20 | 3 | 20 | 14/6 |
| DIM-SPECIMEN-02 | M4×25长螺丝_ID1_20 | 1 | 20 | 14/6 |
| DIM-SPECIMEN-03 | M4×40长螺丝_ID0_40 | 0 | 40 | 28/12 |
| DIM-SPECIMEN-04 | M4六角螺母_ID4_20 | 4 | 20 | 14/6 |
| DIM-SPECIMEN-05 | M4小垫片_ID0_40 | 0 | 40 | 28/12 |
| DIM-SPECIMEN-06 | M5×10粗螺丝_ID2_20 | 2 | 20 | 14/6 |
| DIM-SPECIMEN-07 | M5平垫圈_ID5_20 | 5 | 20 | 14/6 |
| DIM-SPECIMEN-08 | M6盖型螺母_ID0_40 | 0 | 40 | 28/12 |

## 2. 十张尺子实际标定

通过刻度方向、共同基线、灰度剖面线中心、长短刻度周期提取真实刻度。以图片中cm标记及每cm的10个小间隔复核1mm间距；坐标为相对毫米刻度，未猜测绝对零点。红色为厘米长刻度、绿色为毫米刻度；原图路径/哈希保存在清单中，未复制原始数据。

| 原始照片后缀 | px/mm | 刻度数 | 跨度mm | 线性残差RMS px | P95绝对残差 px |
|---|---:|---:|---:|---:|---:|
| IMG_3332 | 35.2379 | 79 | 82 | 3.954 | 6.674 |
| IMG_3333 | 35.3903 | 82 | 81 | 4.432 | 7.439 |
| IMG_3334 | 35.2692 | 85 | 84 | 4.079 | 6.964 |
| IMG_3335 | 35.5930 | 113 | 112 | 1.866 | 3.231 |
| IMG_3336 | 35.4302 | 114 | 113 | 2.537 | 4.205 |
| IMG_3337 | 35.4919 | 114 | 113 | 1.189 | 1.887 |
| IMG_3338 | 36.7518 | 129 | 128 | 2.945 | 6.191 |
| IMG_3339 | 35.8696 | 95 | 94 | 3.220 | 5.410 |
| IMG_3340 | 36.0369 | 135 | 134 | 4.585 | 9.377 |
| IMG_3341 | 37.2537 | 73 | 72 | 7.007 | 13.046 |

共1019个识别刻度。选用全局尺度 **35.758446 px/mm**。逐照片留一验证的10mm间隔平均RMS误差 **0.1772mm**；最差照片P95绝对间隔误差 **0.7279mm/10mm**，对应7.28%的保守位置尺度包络。它不是95%置信区间，也不是零件测量误差界。

| 尺度模型 | 逐照片留一10mm间隔平均RMS mm |
|---|---:|
| global | 0.1772 |
| anisotropic | 0.1910 |
| affine | 0.2304 |
| projective | 0.2038 |
| radial1 | 0.1852 |
| radial2 | 0.1944 |

水平、垂直、斜向尺子都参加拟合。边缘/斜向及单图不同区段存在非均匀尺度变化；仿射、透视、径向模型的留一误差没有优于全局模型，未采用更复杂修正。重叠10mm区间不是独立样本，模型比较也不是外部物理验收。

EXIF：尺子均为iPhone 13 Pro Max 1.57mm超广角、数字变焦2.5802。220张专项照片同镜头，但数字变焦约2.5033–2.6341；Train/Val共440张中180张EXIF缺失，Train的H032.jpg为5.7mm镜头/1.3114变焦，必须视作物理尺度不兼容风险。`capture_exif_audit.json`保留逐图结果。该证据不能被“固定机位”覆盖。

适用条件：原始3024×4032照片，测量在原图ROI中进行。YOLO的640 letterbox只用于检测；输出框已经回到原始像素。1280×720 Camo输入会被标为尺度不兼容、零次毫米测量、非PASS。不得直接按分辨率比值换算Camo；即使分辨率相同，镜头/视野/裁剪仍需确认。

尺子平面与零件测量面有高度差风险；现有资料没有工作距离及实物高度，未伪造视差校正或误差界。当前所有mm均为provisional。

## 3. 参数选择与测量结果

完整计划`search_plan.json`在选参前写入：12次有限组合，background_offset=25/35/45、close_kernel=3/5、targeted=false/true；仅选分割及既有目标隔离参数，不修改P06核心几何实现。按对应家族完整测量数优先、再按规格内归一化波动选参。盖型螺母不适合普通六角拟合，不用其失败数驱动放松六角质量条件。

选中参数：{"1": {"background_offset": 25, "close_kernel": 5, "targeted": true}, "4": {"background_offset": 25, "close_kernel": 3, "targeted": false}, "5": {"background_offset": 25, "close_kernel": 3, "targeted": false}}。螺丝继续body_profile，螺母平行对边，垫圈内外轮廓椭圆；未启用Highlight-Mask、反光重建或Stage 3专用方法。

下表测量单位为provisional mm；括号为P05–P95，完整均值、中位数、标准差、极值、分位数及像素统计见`dimension_statistics.csv`。位置/姿态统计见`measurement_positions.csv`和`position_orientation_stability.csv`。

| 实物 | 优化后完整测量/总数 | 主要测量中位数（P05–P95）mm |
|---|---:|---|
| DIM-SPECIMEN-01 | 18/20 | thread_major_diameter: 3.89 (3.73–4.17)；underhead_length: 8.78 (8.04–9.71) |
| DIM-SPECIMEN-02 | 20/20 | thread_major_diameter: 3.78 (3.55–4.03)；underhead_length: 23.13 (21.78–24.39) |
| DIM-SPECIMEN-03 | 39/40 | thread_major_diameter: 3.97 (3.72–4.23)；underhead_length: 39.29 (36.65–40.59) |
| DIM-SPECIMEN-04 | 17/20 | across_flats: 7.05 (6.80–7.42) |
| DIM-SPECIMEN-05 | 30/40 | inner_diameter: 5.11 (4.97–5.21)；outer_diameter: 11.60 (11.03–12.00) |
| DIM-SPECIMEN-06 | 20/20 | thread_major_diameter: 5.17 (4.92–5.53)；underhead_length: 8.61 (8.07–9.44) |
| DIM-SPECIMEN-07 | 18/20 | inner_diameter: 5.10 (5.01–5.25)；outer_diameter: 14.23 (13.80–15.16) |
| DIM-SPECIMEN-08 | 0/40 | 普通六角对边测量失败；不得伪造尺寸 |

| 版本/场景 | 完整测量 | 正常件通过 | 异常规格尺寸拒识 |
|---|---:|---:|---:|
| baseline / development | 111/154 (72.08%) | 64/70 (91.43%) | 47/84 (55.95%) |
| baseline / retained_scene_validation | 49/66 (74.24%) | 27/30 (90.00%) | 21/36 (58.33%) |
| baseline / all | 160/220 (72.73%) | 91/100 (91.00%) | 68/120 (56.67%) |
| optimized / development | 113/154 (73.38%) | 65/70 (92.86%) | 48/84 (57.14%) |
| optimized / retained_scene_validation | 49/66 (74.24%) | 27/30 (90.00%) | 21/36 (58.33%) |
| optimized / all | 162/220 (73.64%) | 92/100 (92.00%) | 69/120 (57.50%) |

上述配对比较对两种算法使用同一组冻结开发接受范围，从而单独观察分割/隔离改变。原供应商标称值±历史用户候选允许量的比较另存`supplier_nominal_comparison.csv`，不能把换接受范围产生的收益全部归因于测量算法改善。

优化后正常件总通过92/100；80个可用异常长螺丝/小垫片场景中69个尺寸拒识、11个不可测，无异常件在该对应家族挑战中被接受。40张盖型螺母均不可按普通六角对边可靠测量，只能算不可测，不能算尺寸拒识；真实管线中其已学到的YOLO ID0仍强制FAIL。保留场景中正常件27/30通过、1/30误拒绝、2/30不可测；不可将这些照片称为新的独立实物。

专项测量实验根据已知文件夹规格选择对应几何方法，并把异常长螺丝/小垫片/盖型螺母分别作为长螺丝/垫圈/普通螺母家族挑战。这是有标注的离线方法诊断，69/120不是生产管线新增ID0检测率。生产推理不读取文件名或目录真值，只根据真实YOLO类别选择方法；YOLO已经输出ID0时直接FAIL，不借真值补算类别。逐规格轮廓成功数、测量失败率及接受/拒识/不可测数见measurement_summary.csv。

原问题：校准路径/分辨率及物理验收门禁使原完整流程测量次数为0；P06垫圈容易出现多片轮廓及孔歧义；螺丝头身边界/反光会使有效长度偏短；普通六角算法无法测盖型螺母。普通螺母算法只测对边，不提供可靠螺纹内孔尺寸。

## 4. 标称值、候选范围与真实物理验收

Project Manual要求尺寸/规格核验及教师确认的容差，未给出各类统一数值容差。原供应商参考/用户候选允许量保留：ID1 L25±1、D4±0.3；ID2 L10±1、D5±0.3；ID3 L10±1、D4±0.3；ID4对边7±0.5；ID5外径15±0.75、内径5±0.5mm。这些不是已经教师批准的制造公差，也不是卡尺实测。

开发候选范围仅由154张开发场景中正常实物可见几何的最小/最大值，加尺子位置尺度包络得到。保留场景/公共Val未用于调宽范围。它适合初步规格区分，不应被解释成物理制造验收标准；例如M5螺杆直径范围较宽，不能据此保证拒绝所有未见规格。所有旧validated=false及教师确认记录未修改。

| 类别 | 可见测量量 | 候选范围mm |
|---|---|---:|
| 1 | thread_major_diameter | 3.293–4.320 |
| 1 | underhead_length | 20.900–26.281 |
| 2 | thread_major_diameter | 4.590–6.390 |
| 2 | underhead_length | 7.312–10.080 |
| 3 | thread_major_diameter | 3.371–4.500 |
| 3 | underhead_length | 6.820–10.290 |
| 4 | across_flats | 6.297–7.712 |
| 5 | inner_diameter | 4.670–5.630 |
| 5 | outer_diameter | 12.811–16.246 |

## 5. 公共Train/Val真实完整系统对照

所有工单固定[1,1,1,1,1]。Train为训练样本诊断；Val已用于YOLO选择，本次只用于候选冻结后的开发验证，不能当成正式Test。原Val raw/previous_full输出直接复用，未重复推理；Train三个模式均真实运行。

Accuracy(All)的分母包括REVIEW/ERROR；Accuracy(Decided)只统计PASS/FAIL，必须同时看Coverage。FAIL为positive；REVIEW没有当FAIL或算正确。

| 数据/模式 | Accuracy All | Accuracy Decided | FAIL P/R/F1 | False PASS/FAIL | REVIEW | Coverage | PASS | 测量成功/调用 | 中位/P95 ms |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| validation_raw | 98.75% | 98.75% | 97.56%/100.00%/98.77% | 0/1 | 0 | 100.00% | 39 | 0/0 | 22.8/40.5 |
| validation_previous_full | 50.00% | 97.56% | 97.56%/100.00%/98.77% | 0/1 | 39 | 51.25% | 0 | 0/0 | 22.4/44.4 |
| validation_candidate | 58.75% | 88.68% | 86.96%/100.00%/93.02% | 0/6 | 27 | 66.25% | 7 | 305/378 | 771.8/1036.1 |
| train_raw | 99.72% | 99.72% | 99.45%/100.00%/99.72% | 0/1 | 0 | 100.00% | 179 | 0/0 | 42.1/49.1 |
| train_previous_full | 50.00% | 99.45% | 99.45%/100.00%/99.72% | 0/1 | 179 | 50.28% | 0 | 0/0 | 40.9/48.2 |
| train_candidate | 54.72% | 92.06% | 91.37%/100.00%/95.49% | 0/17 | 146 | 59.44% | 17 | 1409/1698 | 271.3/335.5 |

离线计时从已解码图像进入统一Adapter起，到预处理/YOLO/几何/前景/数量及业务判定、结果校验及GPU同步完成；不含模型加载、解码、日志导出、手机曝光、Camo传输、GUI。Val raw/previous_full复用较早运行的计时，各运行的CPU线程负载/并发状态并未做受控配对；不能把这张耗时表当作严格加速比或实机≥20新帧实验。

Val恢复7张正常PASS，但新增5张工单误拒绝（False FAIL从1到6），27张正常场景REVIEW。正常目标误改ID0使五类计数向量完全正确率由95.00%降为78.75%。原始YOLO框和原类别保留在每个对象detector_class_id中，最终class_id、counts和框对应关系一致，工单变化只重新判定。

Val的6个False FAIL：A088、H005、H018、H020、H039、H040。H005是原YOLO已有错误；A088的正常M5被P06估成2.63mm直径，H018/H020约4.53mm、H040约4.50mm低于候选下限；H039/H040正常螺母约6.26/6.28mm低于候选下限。这说明几何质量门禁与单件到公共多件场景的迁移仍有问题，不能靠事后调宽Val范围隐藏。

| Val模式/ID | GT目标 | TP/FP/FN | Precision | Recall | Count MAE |
|---|---:|---:|---:|---:|---:|
| validation_raw/0 | 25 | 25/1/0 | 96.15% | 100.00% | 0.0125 |
| validation_raw/1 | 72 | 72/0/0 | 100.00% | 100.00% | 0.0000 |
| validation_raw/2 | 77 | 77/0/0 | 100.00% | 100.00% | 0.0000 |
| validation_raw/3 | 78 | 77/5/1 | 93.90% | 98.72% | 0.0500 |
| validation_raw/4 | 73 | 72/1/1 | 98.63% | 98.63% | 0.0000 |
| validation_raw/5 | 74 | 74/0/0 | 100.00% | 100.00% | 0.0000 |
| validation_previous_full/0 | 25 | 25/1/0 | 96.15% | 100.00% | 0.0125 |
| validation_previous_full/1 | 72 | 72/0/0 | 100.00% | 100.00% | 0.0000 |
| validation_previous_full/2 | 77 | 77/0/0 | 100.00% | 100.00% | 0.0000 |
| validation_previous_full/3 | 78 | 77/5/1 | 93.90% | 98.72% | 0.0500 |
| validation_previous_full/4 | 73 | 72/1/1 | 98.63% | 98.63% | 0.0000 |
| validation_previous_full/5 | 74 | 74/0/0 | 100.00% | 100.00% | 0.0000 |
| validation_candidate/0 | 25 | 25/19/0 | 56.82% | 100.00% | 0.2375 |
| validation_candidate/1 | 72 | 70/0/2 | 100.00% | 97.22% | 0.0250 |
| validation_candidate/2 | 77 | 67/0/10 | 100.00% | 87.01% | 0.1250 |
| validation_candidate/3 | 78 | 75/4/3 | 94.94% | 96.15% | 0.0625 |
| validation_candidate/4 | 73 | 69/1/4 | 98.57% | 94.52% | 0.0375 |
| validation_candidate/5 | 74 | 74/0/0 | 100.00% | 100.00% | 0.0000 |

候选Val的ID0 TP25、FP19、FN0；原raw为TP25、FP1、FN0，新增18个正常目标误拒绝。该计数包括REVIEW图上的目标，不代表所有未知件拒识已通过。全部对象级错误和逐图退步在`failures/`保留。

## 6. 实际调用链、配置与判定

`app.py / predict.py → src.p07_bridge.load_backend → ObjectBackend → src.models.baseline3.size_candidate:create_backend → epoch69冻结YOLO → 原图ROI → P06 measure_object → 开发尺寸范围 → P06前景关联（光照校正后）→ 计数/工单/状态 → 公共输出及诊断。`

PASS：数量一致、无ID0、所有目录目标测量完整且在冻结开发范围内、前景关联完整、尺度输入条件满足。该PASS始终带DEVELOPMENT CANDIDATE及provisional标记，不是正式物理验收通过。

FAIL：检测ID0、开发尺寸规则拒识为ID0，或已观测计数不符；已知FAIL优先于其他测量不确定。REVIEW：数量匹配但存在不可测、尺度不兼容或未解析前景等真实不确定。ERROR：配置/文件/程序异常，经P07公共错误路径报告，不回退count-only或Mock。

前景有限6组合只用19张Train照片，选中threshold40、min_area30（756×1008诊断像素）；7/19关联完整。其余分片、粘连、未匹配前景导致REVIEW。前景候选不被无依据直接标成ID0；未知物体物理验证未完成。

新配置独立位于本目录`candidate.json`、`selection.json`、`calibration.json`、`p07_profile.json`和`candidate_freeze.json`。未覆盖旧生产配置；切回原启动命令即可回滚。GUI点击Sizes/Reasons或D查看测量值、候选参考、校准及失败原因，始终显示provisional。

## 7. 软件验收及Stage 2矩阵

86项初始相关软件测试通过；随后补充UI显示及旧界面回归共11项（其中6项为复跑），累计91个不同用例，0失败/错误。281个旧模型/配置/源码/实验证据SHA一致；Baseline1/2真实加载、默认工单及UNSUPPORTED_COUNTS执行复核通过，只是功能检查，不是新的性能评价。所用单张Val场景的B1/B2实际输出分别为FAIL/PASS，保留原始结果；没有重写旧验证成绩。真实GUI worker与Batch在PASS/FAIL/REVIEW各1图一致；工单重判保留对象、尺寸、计数；视频解码实际调用几何；1280×720输入停止毫米测量。实机摄像头未运行。

| 项目 | 结果 | 证据/未完成事项 |
|---|---|---|
| 原始冻结YOLO、计数、已学ID0 | PASS（软件与Val范围） | 权重/阈值不变；不是未见未知件保证 |
| 十张尺子刻度与尺度拟合 | PASS（开发图像标定） | 1019刻度；位置残差、镜头/变焦风险保留 |
| 实际尺寸方法接入 | PASS（软件） | Val378次调用；305次完整 |
| 同8实物保留场景测量 | PARTIAL | 正常27/30通过；2不可测、1误拒绝 |
| 错规格定位/ID0/强制FAIL | PARTIAL | 可测异常螺丝/小垫片可拒识；正常件也误拒，盖型测量未完成 |
| 任意未见non-catalogue与前景完整性 | NOT_PASSED | 未匹配前景只能REVIEW；缺独立物理验收 |
| 任意合法工单/重判一致性 | PASS（软件） | 不影响原框、类别、尺寸和实际计数 |
| 教师确认的公差及测量误差界 | PENDING | 没有确认记录，不把false改true |
| 卡尺精度、视差、Camo尺度迁移 | NOT_VALIDATED | 不能计算可信零件MAE；不能按分辨率套尺度 |
| 更新后完整系统独立Test | NOT_RUN | 旧Test不重用；需要新独立证据 |
| ≥20新采集实机帧延迟 | NOT_RUN | 现有结果是离线处理耗时 |
| 完整Stage 2物理合规 | NOT_PASSED | 开发成功接入不等于课程最终验收通过 |

## 8. 最小人工补充与后续修复

1. 对这8个实物各记录真实卡尺值：螺丝头下有效长度及螺杆直径；普通螺母对边；垫圈外/内径；异常件相应尺寸及盖型结构。标称规格不替代这些实测值。
2. 记录尺子平面到测量面高度差、实际工作距离，确认误差影响；没有数据时继续标provisional。
3. 在实际Camo镜头/数字变焦/分辨率下拍同一测量平面尺子，覆盖中心、边缘和横竖方向，建立独立geometry_id及尺度。无需再次确认本批文件夹实物对应关系或固定机位。
4. 请教师确认五类测量量定义、尺寸容差和验收条件，留下真实来源记录；随后用独立物理测量验证误差界，而非修改旧validated标志。
5. P06后续优先修复螺杆边缘/头身质量判断、垫圈碎片/孔歧义、多目标前景关联和盖型/未知件类型证据；在开发Train上设计新版本，并使用新的独立验证/测试证据。保留本候选退步记录，不根据这批Val继续追分。

## 9. 复现与断点续作

在项目根目录及.venv311中：
```powershell
python scripts/develop_baseline3_size.py --phase auto
python app.py --backend real --model baseline3 --config results/baseline3_v2_experiment/size_development_v1/p07_profile.json
```

auto跳过已完成阶段，从首个未完成阶段恢复；不会触发YOLO训练、Stage 2 Test或Stage 3。GUI默认配置不变。只有满足本批照片输入前提才用于尺寸诊断；Camo尚需独立标定。批处理命令及P08评价调用完整保存在每个run的command.json和execution_history.json中。

独立运行批处理（使用新的输出目录）：
```powershell
python predict.py --input_dir data/mechanical_components_v2/yolo/images/val --model baseline3 --counts_file data/mechanical_components_v2/configs/expected_counts_val.json --config results/baseline3_v2_experiment/size_development_v1/p07_profile.json --output_dir outputs/size_development_v1/manual_validation --record-timing
```

本次新增脚本scripts/develop_baseline3_size.py、模块src/models/baseline3/metrology.py和size_candidate.py、tests/test_baseline3_metrology.py；结果在本目录。src/p07_dashboard.py仅新增候选实际接受范围及未验证误差界的诊断显示，旧模式控件/业务逻辑不变。根实验PROGRESS.md/task_state.json更新进度。app.py、默认配置、P06原算法、B1/B2、冻结权重均未改；P03只读；未commit/push/merge。工作区其他已有修改不属于本次尺寸开发。

输出目录crops和compatibility.avi只是忽略的ROI/视频复现缓存，不是新数据集、标注或独立场景，也不会作为正式实机结果。保留P03及原始数据不变。
