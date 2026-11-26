# Baseline3 Vision Caliper v3：冻结对照与工程验收

结论：开发集冻结推荐为 **A（A = size_repair_v2）**。不替换默认GUI。视觉卡尺与Hybrid未获得足以替代v2的收益；完整尺寸核验保留。

## 方法与数据边界

A复用冻结v2：外侧背景支持的密集单像素横截面、螺纹外包络、头/杆轮廓分离、六角平行边和局部椭圆拟合。B新增旋转有限宽度ROI投影、平滑梯度、极性/强度、亚像素边缘配对、多截面稳健估计、轴向头下/端点卡尺、三方向螺母对边与多角度内外径卡尺。没有HALCON依赖，参考原理见 references.json。
C以开发集确定的A为各类主方法；主方法达到高质量门槛即直接采用，质量不足才运行B；现候选对触发备用后的尺寸矛盾保守REVIEW（具体局限见下文），不比较哪个更接近标称尺寸。
仅两套预声明B参数。Train360与专项开发154用于选择，freeze.json在Val80及保留66回归前生成。两组回归每方法一次，均为已使用场景，不是新的独立验证。专项220张仅8个真实样本。未访问Stage2正式Test或Stage3图像/专用真值。
35.758446 px/mm及原有数值容差不变；测量均为provisional，validated保持false；没有真实卡尺MAE、教师认可误差界或Camo尺度外推。

## 完整系统结果（REVIEW留在总分母）

|数据|方案|正确/总数|全输入准确率|PASS|REVIEW|False PASS|False FAIL|Coverage|
|---|---|---|---|---|---|---|---|---|
|Train|A|291/360|80.83%|111|60|0|9|83.33%|
|Train|B|180/360|50.00%|0|155|0|25|56.94%|
|Train|C|285/360|79.17%|105|66|0|9|81.67%|
|Reused Val|A|63/80|78.75%|23|15|0|2|81.25%|
|Reused Val|B|40/80|50.00%|0|36|0|4|55.00%|
|Reused Val|C|62/80|77.50%|22|16|0|2|80.00%|

完整FAIL Precision/Recall/F1、所有FAIL作分母的Recall、五类计数向量正确率和ID0框匹配TP/FP/FN见 comparison.csv；完成判定子集指标不能代替全输入准确率。逐图CSV/JSON、混淆矩阵、错误清单在 train/*/evaluation 与 regression/*/evaluation。

## Val逐类完整测量

|ID|A|B|C|
|---|---|---|---|
|1|68/72 (94.44%)|31/72 (43.06%)|68/72 (94.44%)|
|2|74/77 (96.10%)|23/77 (29.87%)|73/77 (94.81%)|
|3|75/82 (91.46%)|27/82 (32.93%)|75/82 (91.46%)|
|4|70/73 (95.89%)|20/73 (27.40%)|70/73 (95.89%)|
|5|56/74 (75.68%)|14/74 (18.92%)|56/74 (75.68%)|

各类失败原因见 measurement_by_class.json；实际测量分布见 measurement_distribution.csv；不是对真实尺寸误差的估计。逐实例原始证据与质量诊断保留在开发JSON.gz、regression_cases/与retained_cases/。
v2原35次不完整中，B恢复1次但新增229次不完整；C恢复1次但新增2次不完整。C的A180由PASS转为头下长度证据冲突REVIEW。未据此重新调参。

## 专项测量挑战

|数据|方案|完整|正常接受|正常误拒绝|异常拒绝|异常接受|
|---|---|---|---|---|---|---|
|Development154|A|119/154|65/70|1|53/84|0|
|Development154|B|38/154|18/70|0|20/84|0|
|Development154|C|119/154|65/70|1|53/84|0|
|Reused retained66|A|52/66|29/30|0|22/36|0|
|Reused retained66|B|21/66|7/30|1|12/36|0|
|Reused retained66|C|52/66|29/30|0|22/36|0|

此为已知家族的测量挑战，ID0家族映射仅用于量测比较，不注入生产推理；不能代替完整未知件拒识泛化验收。

## 离线完整处理速度

|方案|Train20 Median ms|P95 ms|成功数|Val80尺寸模块Median ms|相对A配对尺寸增量Median ms|
|---|---|---|---|---|---|
|A|280.6|369.3|20/20|162.8|0.0|
|B|417.5|576.9|20/20|310.0|151.6|
|C|363.2|541.5|20/20|221.8|57.0|

perf_counter before backend.infer after decoded BGR frame ready; after contract validation and CUDA completion sync. Includes preprocessing, YOLO, fresh size geometry, foreground, counts and business decision. Excludes decoding/model load/warmup/export/render.

Train20是每方法独立、新鲜完整backend.infer的离线计时，cv2线程统一1，GPU同步，2帧预热，numpy linear P95；不是实机摄像头延迟。Val80共用每图一次真实YOLO，paired_stage_timing为共享检测段加各方案处理段，不冒充独立整链计时。测量并行开发耗时不作为性能结论。

## 软件验收与保留限制

软件测试 108 项，见 software_tests.log。真实A/B/C通过P07 worker、batch、原分辨率编码视频、工单重判与P08评价接口检查；默认B1/B2真实加载及UNSUPPORTED_COUNTS检查见acceptance.json。视频为Train图片编码的软件测试，不是现场视频。
B螺丝的头下平面/端点可见性、螺母对边支持、垫圈边缘配对与拟合支持不足仍是主要缺口。C在主方法达到高质量门槛时不会运行备用方法；否则当前实现对两侧已有尺寸的不一致采取保守REVIEW，较弱的备用测量也可能否决完整主结果，这是现候选的质量策略局限，未根据回归改写。不能为了通过工单选择接近标称值的结果。失败与新增/恢复记录见regression_paired.csv。
软件验收首次运行的部分椭圆合成用例出现0.028像素代数拟合误差，超过不合理的0.01像素测试断言；修正为0.1像素软件精度断言后通过。没有改变测量算法、工程容差或实验参数。首次失败日志在software_test_history/保留。
当前数据中没有新增完整工单False PASS，不等同于未知件零风险；专项不可测与低覆盖率均如实计入，不强行采用v3。
所有冻结v2结果/模型、原P06、Baseline1/2、app.py与默认配置的SHA检查见audit.json和acceptance.json。P03无写入；没有训练、commit、push或merge。

## 复现与人工验收

```powershell
python scripts/develop_caliper_v3.py --phase auto
python app.py --backend real --model baseline3 --config results/baseline3_v2_experiment/size_repair_v2/p07_profile.json
# 独立候选，仅诊断，不能当作已验收发布版本：
python app.py --backend real --model baseline3 --config results/baseline3_v2_experiment/size_caliper_v3/configs/C.json
```

auto按进度跳过完成阶段，文件锁防止并发；逐图结果在异常中断后复用。执行命令/异常历史见execution_history.json。最终大JSON以.json.gz无损保留，SHA及体积见storage_archive.json；CSV和配置可直接读取。Python gzip.open(path,"rt",encoding="utf8")可解读，脚本自动支持压缩证据。
人工仍需：①8件实物卡尺测螺纹外径/正确头型长度、螺母对边、垫圈内外径，核对公共M5与专项M5是否同一规格；②固定Camo镜头、变焦、机位、分辨率、灯光及检测平面，在全区域多位置采集真实标尺与独立尺寸物；③估计畸变、位置/高度尺度误差，并验证旋转/反光/遮挡及邻件；④教师确认规格、长度定义和容差/误差策略，未经确认不改validated；⑤补拍未用于开发的新实物与新场景完成独立完整系统和现场延迟验收。详细操作沿用 ../size_repair_v2/manual_acceptance.md。
Vision Caliper v3是本次工程候选版本号；未将其再次包装为Stage3研究贡献。Stage2完整物理尺寸与未知件验收仍未通过，不能以软件测试或成功接入替代。
