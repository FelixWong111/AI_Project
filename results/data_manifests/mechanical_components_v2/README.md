# 新数据集审查与候选划分（2026-11-26）

**状态：CANDIDATE_NOT_READY，未正式冻结；训练和正式Test均为NOT_RUN。**
原始`数据集/`、P03、历史数据、权重、阈值及GUI均保留。数据处理前核对Project-Man第3–6、8、11–12页及team_plan_ai.md；保留Stage 2冻结基线，Stage 3保留集不参与开发。

## 收件与计划差异

| 用途 | 计划 | 实收 | JSON |
|---|---:|---:|---:|
| 公共正常工况 | 420 | 420 | 420 |
| crowded | 70 | 70 | 70 |
| edge | 70 | 70 | 70 |
| 尺子 | 10 | 10 | 不要求 |
| 尺寸专项 | 200 | 220 | 不要求 |
| 合计 | 770 | 790 | 560 |

另有7份Excel。M4小垫片、M6盖型螺母专项分别实收40张，比计划各多10张；没有复制补数。230张HEIC全部可解码，独立导出无损RGB PNG；原始HEIC与EXIF保留。
公共560张与560份LabelMe JSON一一对应；2811个rectangle标注，六类名称均合法，无缺失JSON、解码失败、越界/零面积框或完全重复框。polygon工具规则为顶点min/max水平包围框，本批无polygon。
坐标为原图连续像素xyxy，左上为原点，0≤xmin<xmax≤width、0≤ymin<ymax≤height。框不自动裁剪；无未经依据的ignore，全部ignore=0。

## 真实标签与人工确认

固定工单位于config/inspection.json：ID1–5各1，ID0=0。标签从非忽略目标标注计数推导，与文件名前缀和模型预测无关。任何ID0、缺件、多件或空场景均FAIL。
D18：JSON计数(ID1..5,ID0)=[0,1,1,0,1,2]，拍摄表=[0,1,1,1,1,1]。F21：JSON=[1,1,1,0,2,1]，拍摄表=[1,1,1,1,2,1]。用户已明确确认“已核实，两张都以JSON为准”；保留JSON和Excel，确认绑定原JSON SHA256。
E.xlsx表头明确“为2的即为加入异常尺寸件”，因此2不是两个正常件。原始表值完整保存，Ground Truth以JSON为准，未把异常件映射为正常类别。
C38实为同时缺ID4、多ID3（counts与拍摄表一致），分层scenario_type=F，原目录C保留；F08–F12等缺正常件加不同异常件，计数本身不能证明是同类替换，保留F语义与scenario_note。
annotation_status=COMPLETE仅表示人工JSON格式完整及程序检查通过，**不表示已经完成全量视觉真值核验，不设VERIFIED**。已知两处冲突已解决；仍需P03确认全量/抽样质检范围、漏标与细规格真值。

## 候选划分

seed=42；二元scene-group分配，硬约束工况与PASS/FAIL数量及A–G覆盖，软目标平衡A–G、六类实例数及有记录的异常类型。配额可因真实缺收/无效标注而调整，禁止复制凑数。
各照片独立摆放依据本次用户/采集者确认，自动scene_id来自全局唯一image_id。困难图按Excel图片编号和工况文字可靠关联，转H001.jpg等，原路径映射保留。不建立跨工况配对。

| 分区 | 图片 | PASS | FAIL | normal/crowded/edge | scene_id | scene_group |
|---|---:|---:|---:|---|---:|---:|
| train | 360 | 180 | 180 | 300/30/30 | 360 | 354 |
| validation | 80 | 40 | 40 | 60/10/10 | 80 | 80 |
| test | 120 | 60 | 60 | 60/30/30 | 120 | 120 |
| stage2 | 30 | 15 | 15 | 30/0/0 | 30 | 30 |
| stage3 | 90 | 45 | 45 | 30/30/30 | 90 | 90 |
| quarantine | 0 | 0 | 0 | 0/0/0 | 0 | 0 |

G01–G07均为空台面，保守同组且只计一个场景支持，未用7张近乎相同空背景虚增独立物理场景。其他scene_id/scene_group按采集者确认，不能据此声称独立实物。

| 分区 | ID0 | ID1 | ID2 | ID3 | ID4 | ID5 | A/B/C/D/E/F/G |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 110 | 325 | 349 | 353 | 332 | 336 | 180/36/35/44/37/18/10 |
| validation | 25 | 72 | 77 | 78 | 73 | 74 | 40/8/8/11/8/3/2 |
| test | 36 | 109 | 119 | 117 | 113 | 113 | 60/12/12/16/13/5/2 |
| stage2 | 9 | 27 | 29 | 28 | 28 | 28 | 15/3/3/3/3/2/1 |
| stage3 | 27 | 82 | 90 | 89 | 85 | 85 | 45/9/9/13/10/3/1 |

异常类型仅记录有明确拍摄表文字或D/E替换/添加栏依据的声明，不把ID0框虚构为已确认具体规格；缺失处UNKNOWN。D18拍摄表组成已被覆盖，额外ID0细类型保持UNKNOWN。各分区异常组合详见dataset_statistics.json；这统计的是含该声明的图片，不是已逐目标验证的异常实例数。

## 重复与泄漏

公共集文件SHA256完全重复组=0（两个哈希通道独立记录，不能相加为图片数）；像素完全重复组=0。pHash≤8候选127对，仅作候选，不据此合并非空独立摆放。G01–G07保守同组。
旧数据明确根目录共扫描790张，文件/解码像素交叉相同匹配=0；旧新pHash≤6候选=8，均来自一张尺寸专项盖型螺母与旧图片路径，公共560图对旧集感知候选为0。哈希不能证明物理场景或实物独立，任意裁剪与未经记录的外部使用历史仍无法完全排除。
近重复候选和选定视觉预览在audit/duplicates.json、previews/；抽查发现相似背景/相似布局但有零件位置改变，采集者确认独立摆放仍是场景依据；未对127对全部做人工签字。不得把近重复候选数当作泄漏数。
没有跨Train/Val/Test以及Stage 2/3的已确认像素重复或scene_group交叉。旧Test仅为本次授权的数据哈希审查读取，无推理、评估、调参。原开发训练清单与模型冻结证据不变。

## Stage 2/3隔离

Stage 2候选测试30张normal，15 PASS/15 FAIL，A–G均覆盖，含错规格替换、正常数量正确但额外ID0；满足组内数量目标。它不是已有模型的测试成绩。
Stage 3保留90张：normal、crowded、edge各30张，各15 PASS/15 FAIL。两阶段图像与scene_group完全不交叉；属于分层独立样本，**不是同场景配对实验**。未运行Stage 3模型、设计改进或按测试错误调参。
尚无证据表明新测试图曾用于本仓库冻结模型训练；外部未记录用途无法由代码证明，封存前请P02/P03确认。旧Stage 2模型/配置/验证结果不覆盖，新版本未READY，不自动冻结。

## 三模型与P08入口

- Baseline1/2共同数据：development/（只含Train/Validation CSV），configs/training.json；图片共用yolo/images/train和val，路径为项目相对路径。prepare入口防止未READY数据用于正式训练，--audit-candidate仅允许--prepare-only。
- Baseline3：yolo/data.yaml，现有模型真实为六类YOLO，官方ID与内部索引0–5一一对应，已核对冻结映射；不改变网络。ID0完整保留，空标注生成空TXT；training YAML仅Train/Val，完全不含Test键。旧P06历史训练脚本的分支/环境/数据验收约束仍保留，正式训练新版本需P06明确批准新候选流程。
- P08：configs/stage2_evaluation.json或stage3_evaluation.json；对应evaluation/<stage>/两份过滤GT和独立工单文件。p08_run.py支持--check-data-only，仅检查数据。正式运行要求READY且frozen，根目录全量Test不能隐式作为新阶段评价集。
- predict.py接口未改变，仍只接受图片与不含GT的工单JSON；本次不调用新Test做预测。expected_counts.json（全公共集）不是推荐的阶段评估入口，应使用阶段专用配置。

## 尺子与尺寸专项

calibration/ruler/保存10份PNG，ruler_manifest.csv记录源HEIC、分辨率与可用EXIF；尺子位置、刻度点、px/mm、相机高度保持空，需人工标定。
dimension_manifest.csv保存220张的目录声明规格及官方ID。卡尺实测、physical_sample_id、拍摄条件、开发/独立验证用途留空或UNASSIGNED；先确认实物编号和历史，再按实物整体分开发/验证，不能随机把同实物的照片拆成独立实物验证。
没有卡尺表、真实尺度误差界或验证记录，不输出任何伪造尺寸或通过结论。M6盖型螺母同时有类型与规格差异，不把全部拒识收益归于尺寸。

## 自动验收与待办

audit/acceptance.json：源文件SHA256保全、唯一ID、类别、原JSON↔GT↔YOLO逐框/数量和≤1e-6px还原、像素与scene_group隔离、工单判定、专项隔离等。audit/interface_acceptance.json：真实模型加载、B1/B2数据读取、实际YOLO数据加载、P08阶段GT读取、seed42复现、历史模型/结果未变；不等于模型性能或标定验收。
仍需：P03签署全量/抽样标注质检范围并核实漏标及细规格；复核感知候选与采集历史；提供实物编号/卡尺实测/尺子刻度与机位。当前文件可做加载及开发诊断，**正式训练前置条件尚未全部满足，状态维持NOT_READY**。

## 复现命令

从项目根目录、.venv311运行（源目录不改；新输出不覆盖历史）：

```powershell
python scripts/prepare_dataset_v2.py
python scripts/prepare_dataset_v2.py --verify-only
python -m scripts.train_baseline1 --dataset-config data/mechanical_components_v2/configs/training.json --prepare-only --audit-candidate
python -m scripts.train_baseline2 --dataset-config data/mechanical_components_v2/configs/training.json --prepare-only --audit-candidate
python scripts/check_dataset_v2.py
python p08_run.py --evaluation-config data/mechanical_components_v2/configs/stage2_evaluation.json --check-data-only
python p08_run.py --evaluation-config data/mechanical_components_v2/configs/stage3_evaluation.json --check-data-only
```

源HEIC解码使用本地.tools/pillow_heif（1.9.0），不进入模型运行requirements；若缺失，按脚本提示安装到.tools。数据/图片仅在本机data/，已被.gitignore忽略；精简文本审查在results/data_manifests/mechanical_components_v2/。
全部生成文件见manifests/generated_files.csv；原始清单与SHA256见audit/source_files.json。各文件UTF-8（CSV带BOM方便Excel），清单使用项目相对路径。LLM协助记录见docs/p07/advice_record.md。本次未训练、未修改权重、未生成性能指标、未commit/push。
