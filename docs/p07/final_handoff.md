# P07正式系统与P08交接

**最终状态：NOT_READY。软件集成与离线共同评价完成；实机延迟未运行，物理尺寸验收和远程Git交付仍未验证。**

依据已阅读的Project-Man.pdf（第2/3/5/7/10页相关要求）及现有team_plan_ai.md；指定Mechanical_Component_Vision_Inspection_Team_Plan.docx未在工作区找到，未将其视作已阅读材料。
## 正式模型与冻结

唯一正式注册表：config/model_registry.json；app.py、predict.py、p08_run.py统一调用src.p07_bridge.load_backend。正式权重仅位于models/release/三个目录；GUI只有三个Baseline按钮和1–3快捷键。final仅保留API预留ID，明确NOT_READY，不能别名Baseline3。加载时校验SHA；错误不回退。
Baseline1/2采用已冻结的新Train360训练模型。生产GUI与CLI均使用原始分类器，scene_guard关闭；不会把旧262参考特征门控套到新模型上。两者只支持[1,1,1,1,1]，无对象框/计数/ID0能力，其他合法向量返回UNSUPPORTED_COUNTS与空预测字段。严重OOD仍可能高置信误判，不能据分数宣称场景有效。
Baseline3是YOLO11n epoch69 + **size_repair_v2**，conf0.35、NMS0.50、640、batch1、CUDA0、half=false；完整前景关联、几何测量、尺寸证据、计数和工单判定保持原算法。不是原YOLO-only、v1、Vision Caliper B或Hybrid C。
|模型|实际权重|Bytes|SHA256|
|---|---|---|---|
|baseline1|`models/release/baseline1/baseline1.joblib`|955,515|`4a3ecbefa90f10a9d953936072c82bc1be8322684273c55c097defaf3a61b924`|
|baseline2|`models/release/baseline2/baseline2.json`|44,420|`ef62957cc7addd49303d004b2269190da3a43199e9fb1519ea17518914e1c33b`|
|baseline2|`models/release/baseline2/resnet18-f37072fd.pth`|46,830,571|`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`|
|baseline3|`models/release/baseline3/best.pt`|5,494,161|`1c16259c7b18587d405171c90a749fbc7b594cabfe0bdda09ac71bef82d5867d`|

release_freeze.json在本次完整Baseline3 Test推理之前生成，所有运行源码与配置SHA见results/p07_final_integration/。calibration.json和selection.json逐字节保留v2来源，路径变更记录在model_promotion.json。配置中的旧训练路径是冻结的来源记录；真正bundle路径由唯一注册表显式覆盖。
Test后仅增加UI显示的类别ID，并移除app的无效Mock场景操作参数；未改变模型推理、预处理、后处理或业务判定。旧Test运行源码已精确归档，ui_revision.json及final_release_freeze.json分别记录显示变更和当前入口；没有为此重复Test。

## 文件布局

```text
app.py / predict.py / p08_run.py / p08_object_observations.py
config/model_registry.json / p07.json / inspection.json
models/release/{baseline1,baseline2,baseline3}/
src/models/baseline{1,2,3}/ / src/adapters/ / src/p07_*.py / camera.py
datasets/mechanical_components_v2/{public,size_special,calibration,annotations,manifests,configs,development}/
results/p08_stage2_final/{baseline1,baseline2,baseline3,comparison,error_cases,run_configs}/
results/p07_final_integration/
archive/experiments/{pre_final_integration,vision_caliper_v3,development_runners,baseline3_training_checkpoints}/
data/sealed/  [忽略，仅本机，不交P08]
PROGRESS.md / task_state.json
```

完整可交付文件树见results/p07_final_integration/project_tree.txt；本次代码改动对照任务开始快照，见change_inventory.json，迁移与归档文件分别见其引用清单。Git原有未提交工作不被冒充为本次新增贡献。
正式数据为Train360（180/180 PASS/FAIL）、Validation80（40/40）、Stage2 Test30（15/15）。原冻结划分不变，人工场景确认与保守分组元数据保留；不把哈希唯一当作场景证明。700张交付图片=470公共+154专项开发+66已使用回归+10标尺；专项220张仅8实物。Stage3的90张及其真值均未进入交付目录。源图/原始JSON未改写，迁移SHA及路径映射见dataset_migration.json和datasets/mechanical_components_v2/manifests/source_sha256.csv。
Ground Truth位于annotations/及public各分区，推理只读图像与expected_counts文件，不读取GT。ID0预期恒0，difference=observed−expected。原始JSON、YOLO标签和GT CSV均随所选分区交付。来源P03/数据集只读。

## P07验收

|项目|实际结果|证据|
|---|---|---|
|三真实模型加载、静态图片、GUI worker/CLI一致|PASS|compatibility/*/checks.json|
|实际app控制器三模型切换、渲染、快照|PASS（原生窗口调用拦截，非现场显示验收）|real_gui/acceptance.json|
|原生桌面窗口人工交互|NOT_VERIFIED；桌面工具未返回目标窗口，已清理自启进程|desktop_gui_attempt.json|
|Baseline3工单重判不再运行YOLO|PASS，框/类别/分数/观测数不变|integration_acceptance.json|
|同图两工单独立batch输出|PASS，固定PASS、零向量FAIL|compatibility/variable_order/|
|ID0强制FAIL、尺寸核验、REVIEW|软件/实际离线执行PASS；物理尺寸NOT_VERIFIED|P08预测及单元测试|
|视频输入、设备释放、非法/空图|PASS|integration_acceptance.json|
|连续处理/内存|三模型各30次重复Train图；B1另240次，末100次RSS增24KiB；非长期无泄漏证明|stability_all_models.json、baseline1_memory_followup.json|
|真实模型离线Record与MP4解码|PASS，仅Train软件验证，不是演示/备份视频|recording_acceptance.json|
|真实Camo采集|仅3个新读帧ID，全黑，设备已释放；真实工位NOT_VERIFIED|camera_probe.json|
|断连重连/加载错误/CUDA错误|软件故障注入通过；实体断插未验收|software_tests_final.log|
|当前完整软件测试|303项通过，0失败0错误|software_tests.json|
|干净环境安装与预测|PASS|clean_ml_acceptance.json / clean_environment.json|

安装第三方依赖前，先在标准库干净venv中复核了三个已保存预测的P08评分、全部观察框/计数评分及三模型SHA，均与原结果一致。独立评分不需要GPU，也未重新运行Test图像推理。干净venv实际安装CUDA依赖、三模型Train预测及必要核心接口测试通过，见clean_ml_acceptance.json；未重复全部303项测试。
303项当前测试不包含已归档的10项卡尺实验测试；初次308测试的单项失败是旧final错误提示断言，已按NOT_READY更新，原日志保留。所有生产路径无Mock；测试专用Stub仅用于故障/控制器单元用例，不代表实机或模型性能。

## 共同Stage2评价

|Model|全输入Accuracy|FAIL P/R/F1|TP/FP/FN/TN|False PASS/FAIL|REVIEW|Coverage|
|---|---|---|---|---|---|---|
|baseline1|96.67%|1.0000/0.9333/0.9655|14/0/1/15|1/0|0|100.00%|
|baseline2|86.67%|0.9231/0.8000/0.8571|12/1/3/14|3/1|0|100.00%|
|baseline3|93.33%|1.0000/1.0000/1.0000|15/0/0/13|0/0|2|93.33%|

FAIL为正类。三个模型均为15 PASS、15 FAIL的相同30图、相同工单。Baseline1/2复用原来完全相同权重/预处理/raw分类器的预测，不重复推理；Baseline3为冻结后第一次完整尺寸v2 Stage2运行。Stage2此前已用于原YOLO及新旧分类器对照，不能称为全新未触碰测试。没有根据本次结果再选型、改阈值或改容差。
Baseline3完成判定子集28/28不等于全输入100%；主表为28/30=93.33%，两个REVIEW保留。objects/counts主表使用P08 object_observations包含所有30图的观察统计；原summary.json的旧可判定子集口径仍保留并明确区分。
五类计数向量全对率=96.67%；ID0 TP/FP/FN=9/1/0。每类Precision/Recall/F1/support/MAE见comparison/object_count_metrics.csv。
|类别ID|实例支持量|TP/FP/FN|Precision|Recall|F1|Count MAE|
|---|---|---|---|---|---|---|
|0|9|9/1/0|0.9000|1.0000|0.9474|0.0333|
|1|27|27/0/0|1.0000|1.0000|1.0000|0.0000|
|2|29|28/0/1|1.0000|0.9655|0.9825|0.0333|
|3|28|28/0/0|1.0000|1.0000|1.0000|0.0000|
|4|28|27/1/1|0.9643|0.9643|0.9643|0.0000|
|5|28|28/0/0|1.0000|1.0000|1.0000|0.0000|

Stage2非目录支持：8个场景、9个实例。Train/Val/Test图像、scene_id、保守scene_group及各类实例分别保存在comparison/dataset_support.json。
错误案例：baseline1 False PASS F07.jpg; baseline2 False FAIL A198.jpg; baseline2 False PASS B04.jpg; baseline2 False PASS F07.jpg; baseline2 False PASS G13.jpg; baseline3 REVIEW A084.jpg; baseline3 REVIEW A154.jpg.
目标级错误见object_errors.csv：A198螺母框IoU约0.496未达0.50，记FP+FN；B04一个合法ID2被拒为ID0。图像级没有False FAIL不能掩盖这个正常件误拒。ID0漏检为0/9，但不能据这9个已标注实例宣称所有未见件均可拒识。details只是已有P08匹配循环的追加诊断，不是另一套Evaluator，也没有重新运行模型。

## 实机性能与资源

|系统|真实新工位帧|Live Median ms|Live P95 ms|状态|
|---|---|---|---|---|
|Baseline1|0|N/A|N/A|NOT_RUN|
|Baseline2|0|N/A|N/A|NOT_RUN|
|Baseline3 v2|0|N/A|N/A|NOT_RUN|
用户本次明确确认尚未完成实机布置或标定，原答复保存在physical_confirmation.json。Camera0虚拟设备能打开不代表手机已送出真实工位；这次3帧全黑不计入实机性能，之后未再采集。旧离线Train20时间不是摄像头延迟。Camo当前请求1280×720；原照片标定针对3024×4032，35.758446 px/mm不能外推。分辨率不符时尺寸模块拒绝测量并给出非PASS，未伪造尺度、MAE、公差或validated=true。
本机CPU：Intel(R) Core(TM) i7-14650HX；GPU：[{'index': 0, 'name': 'NVIDIA GeForce RTX 4060 Laptop GPU'}]；Python：3.11.14 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 18:30:03) [MSC v.1929 64 bit (AMD64)]；CUDA：12.6；依赖：{'numpy': '1.26.4', 'torch': '2.7.1+cu126', 'torchvision': '0.22.1+cu126', 'opencv-python': '4.10.0.84', 'ultralytics': '8.3.236', 'scikit-learn': '1.9.1'}。Baseline1/2实际CPU，YOLO CUDA0；尺寸/前景OpenCV CPU。真实持续Camera Capture FPS与处理FPS待测，不能用1/单帧延迟冒充。
benchmark_live.py默认显示，主Live边界是FrameProvider.available_at到GPU同步后P07 render、imshow、waitKey返回，包含排队/预处理/推理/后处理/尺寸/判定/显示调用；不含帧可用前曝光、Camo传输或显示器真实刷新。纯processing时间另列。分段未独立可靠计时则null，不虚构；RSS与CUDA内存可测时记录。observed_processing_fps和observed_sample_fps来自实际时间跨度；theoretical_serial_capacity_fps仅保留为理论串行容量。

## 历史保护与Git交付

未选用卡尺源码/测试和历史开发runner已无损归档ZIP，成员名与SHA见cleanup.json及historical_runner_archive.json。last.pt/resume_last.pt移到历史checkpoint目录。旧冻结模型因历史配置引用而保留原位置，仅作历史复现；LFS按OID去重相同ResNet骨干。既有results、权重与训练参数SHA保持不变，原始预测未覆盖。
本地正式数据总大小2.272 GiB。700张图片已从Git排除，通过网盘单独共享；标注、GT、划分、类别映射、SHA、配置和README仍提交。模型.pt/.pth继续使用LFS。Git候选清单见git_submission_files.csv；远程模型LFS配额、上传和新克隆未验证，本次未git add/commit/push/merge。
P08克隆源码后，先从网盘取得图片，并按datasets/mechanical_components_v2/README.md合并到同名根目录，保留文件名、目录和原始分辨率，无需修改代码。网盘链接尚未提供，不宣称已完成远程交付。原始data/、数据集/、P03、Stage3封存、环境及临时输出仍忽略。
提交前建议：git lfs install；git add .gitattributes .gitignore README.md .python-version config src scripts tests datasets models archive docs results review PROGRESS.md task_state.json app.py predict.py p08_run.py p08_object_observations.py requirements*.txt；git diff --cached --stat；git lfs ls-files --size；确认数据集图片、原始数据和虚拟环境未进入暂存区。之后由你决定commit/push。

## P08克隆与运行

以下在你完成提交/push后执行。干净venv实际安装CUDA依赖、三模型Train预测及必要核心接口测试通过，见clean_ml_acceptance.json；未重复全部303项测试。 不擅自降级已训练scikit-learn版本。
```powershell
git lfs install
git clone -b p07/system-integration https://github.com/FelixWong111/AI_Project.git
cd AI_Project
git lfs pull
# 从网盘下载图片，合并至 datasets/mechanical_components_v2/，保留目录及原文件名。
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
python scripts/verify_models.py --load-models
python scripts/check_dataset_v2.py
python app.py --model baseline3 --camera 0 --capture-backend dshow
# 默认P08三模型评价，会新运行；已有本次结果直接看results/p08_stage2_final，不要无必要重复正式Test：
python p08_run.py --evaluation-config datasets/mechanical_components_v2/configs/stage2_evaluation.json --model all --output outputs/p08_reproduction
# 单模型批处理（更换model ID即可）：
python predict.py --input_dir datasets/mechanical_components_v2/public/stage2_test/images --model baseline3 --output_dir outputs/batch_unique --counts_file datasets/mechanical_components_v2/configs/expected_counts_stage2_test.json
# 以下只在Camo已输出真实工位且标定就绪后执行：
python scripts/benchmark_live.py --camera 0 --warmup 20 --frames 100 --models baseline1 baseline2 baseline3
python -m unittest discover -s tests -v
```

只复核已有结果、不运行模型且不需要PyTorch时，可用：`python -m src.evaluator --ground-truth-images datasets/mechanical_components_v2/public/stage2_test/ground_truth_images.csv --ground-truth-objects datasets/mechanical_components_v2/public/stage2_test/ground_truth_objects.csv --expected-counts datasets/mechanical_components_v2/configs/expected_counts_stage2_test.json --predictions-dir results/p08_stage2_final/baseline3/predictions --output-dir outputs/p08_score_only_unique`；改predictions-dir即可复核其他两模型。
## 未完成与人工步骤

1. 在Camo Studio确认手机连接/解锁、Camera0有真实画面；固定纯白哑光工位、灯位、相机镜头/变焦/工作距离和倾角，提供真实工位全景与背景材质记录。
2. 按size_repair_v2/manual_acceptance.md在最终Camo几何下重新标定；多位置/旋转/高度采样，独立卡尺测五类及8件专项样件，核对M5实物身份和头型长度定义。现有照片尺度不覆盖Camo。
3. 教师确认类目录、尺寸容差、支持工况与误差策略，保留实际记录；不能现在把validated改true。必要的标定支持属于人工资料到齐后的受控新配置，不在此次制造参数。
当前v2工厂明确限定开发配置（validated=false、instructor_approved=false），不会因改一个标志自动成为已认证物理系统。资料齐后仍需P06/P07受控接入新的真实Camo配置与审核策略、保留冻结照片参考并重新验收；不要直接编辑冻结JSON或更新SHA冒充已经通过。
4. 完成真实相机断插/重连、三模型切换、ID0、尺寸和U01–U06功能验收，再跑20+100新帧实机计时。没有完成不能READY。
5. 按demo_recording_checklist.md拍摄<2分钟真实视频及备份；当前截图/Train编码视频不冒充实机演示。rig照片、白底材质证据与真实演示/备份仍未核实。
6. 原始review/decision_stage2.md未找到；新缺项说明不能替代实施前研究决策。请P05提供当时原件；不倒填日期或编造计划。
7. 核实GitHub LFS配额并提交上传，再让P08实际克隆核对。干净环境验证已经完成，仍需验证远程克隆。
支持恢复：根PROGRESS.md/task_state.json；所有可执行离线阶段有幂等守卫，evaluate复用已有结果不重复Test。训练不在本任务范围，没有训练或修改权重/阈值/几何算法。
