# Baseline3 几何测量修复 v2：实际证据与剩余边界

本次保留YOLO+尺寸核验+前景+计数+工单完整链路，没有重训YOLO，没有关闭尺寸核验，没有放宽数值范围，没有把validated改true。原v1与全部旧冻结模型/实验结果保持不变。新版独立启动，正式物理验收仍待完成。

## 1. 原378次测量的73次不完整测量

| 失败环节 | 次数 |
|---|---:|
| screw/axis_or_edges | 12 |
| washer/aperture | 7 |
| nut/parallel_flats | 12 |
| ROI/isolation | 42 |

| 具体原因 | 次数 |
|---|---:|
| FOREGROUND_TOUCHES_CROP | 22 |
| TARGET_COMPONENT_EXTENDS_OUTSIDE_BOX | 14 |
| SHAFT_EDGE_AMBIGUOUS_REFLECTION_OR_SHADOW | 11 |
| OPPOSITE_FLATS_NOT_PARALLEL | 8 |
| INNER_HOLE_NOT_UNAMBIGUOUS | 4 |
| MULTIPLE_FOREGROUND_OBJECTS | 4 |
| NUT_TILT_OR_SHADOW_ASYMMETRY | 2 |
| SIX_FLATS_NOT_RESOLVED | 2 |
| CIRCULAR_EDGE_ANGULAR_COVERAGE_LOW | 2 |
| TARGET_COMPONENT_AMBIGUOUS | 2 |
| ELLIPSE_SUPPORT_DEGENERATE | 1 |
| SCREW_AXIS_AMBIGUOUS | 1 |

42次ROI/隔离失败包括：裁切边缘22、目标分量越过检测框14、多个前景4、目标关联歧义2。YOLO轴对齐框不是精确物理轮廓；邻件及其阴影进入padding后，旧算法把它们当目标的一部分或拒绝整个ROI。

原18个尺寸改ID0的目标中，M5螺丝10、细螺丝3、螺母3、长螺丝2；这不等于18个独立误判工单。原27张REVIEW均保留在准确率分母，不能从指标中排除。逐目标/逐图证据见incomplete_73.csv和original_image_diagnostics.json。

## 2. 2.63mm真实原因及修复

Closed Canny contour follows a bright axial stripe into the shaft; pooled transverse gradient chooses an internal metal/highlight transition. Strongest positive internal gradient ~22.4 vs real exterior ~7.5: relative .4*max threshold suppresses the real edge. Repetition across153 rows reproduces the same wrong physical boundary, not 153 independent confirmations.

原Canny闭轮廓沿螺杆内部反光条绕进金属面，截面粗宽约127.63px；内部反光正梯度约22.4，外边缘约7.5，被旧0.4×最大梯度门槛压掉。左-75px至右+19px仅94px，94/35.758446=2.62875mm，换算无640 letterbox或单位错误。153个截面重复支持的是同一个错误物理边界，不是153次独立正确确认。

冻结后重用A088回归：状态REVIEW，测量状态MEASURED_CANDIDATE，尺寸核验WITHIN_DEVELOPMENT_BAND。测量值：{"underhead_length": 9.704, "thread_major_diameter": 5.0617}。这些是provisional图像估计，不是卡尺真值。

证据：A088_original_measurement.jpg、A088_transverse_profile.csv、A088_forensic.json及A088_repaired_regression.json。A088已用于取证，不作为独立验证；没有用其值拟合阈值。

## 3. 实际算法修复及质量门禁

- ROI扩大至目标跨度35%，先按已知邻件归属隔离，再保留目标框内原始像素；轴对齐框相交不会自动认定实物重叠。实际轮廓重叠仍阻止可靠归属。
- 螺杆主轴使用完整支撑凸包，尺寸端点始终从原始灰度取证；凸包不冒充实物边界。选择向外持续通到背景、且局部变化足够锐利的成对边缘，拒绝内部反光和软阴影。
- 在多个轴向位置验证成对外边缘，使用重复螺纹外包络的P90，不再用单个池化截面的强反光边缘；跨段支持、宽度一致性和裁切范围不满足时明确不可测。
- 螺母对两种真实轮廓运行原平行对边方法；保留3组对边变化作为倾斜/阴影不确定性，不用单一投影短边直接拒识。原12%拟合筛选参数没有放松。
- 垫圈先隔离邻件，再在以目标为中心的局部ROI做原有内外轮廓/椭圆拟合；避免扩大背景改变Otsu分布，消除重复的相同孔拟合调用。原圆度、角覆盖和残差要求没有放松。
- 完整/可信质量检查先于ID0规则；截面重复性、对边变化及椭圆残差形成局部区间。区间仍接触接受范围时为BORDERLINE_MEASUREMENT，不能当已确认错规格。它不是卡尺绝对误差界。
- 旧单实物经验范围与已有供应商标称允许量相矛盾时为SPEC_REFERENCE_CONFLICT，不能据此宣称ID0；PASS范围及原允许量均不扩大。
- 实际测得轮廓用于前景像素守恒，解决同一金属面分片及阴影连接多件的问题；阈值40、最小面积30保持不变。轮廓之外的未匹配区域仍REVIEW，不凭空标ID0。轮廓重叠时撤销无法归属的尺寸拒识，已学ID0仍强制FAIL。

没有调用其他Baseline/Final补判，没有Highlight-Mask/反光重建，没有Stage3数据，没有把不完整测量伪装为完整。

## 4. 开发集实际结果

公共Train360：实际几何测量完整数1409/1698（82.98%）→1570/1698（92.46%）。计数/工单诊断使用同一冻结YOLO观测与实际新几何缓存，非独立模型评估；没有将缓存回放时间冒充模型延迟。

最终Train状态：{"PASS": 111, "correct": 291, "REVIEW": 60, "FAIL": 189, "false_fail": 9}；旧v1为17 PASS、197 FAIL、146 REVIEW、False FAIL17、False PASS0。

下表仅专项开发154张（同8实物），没有重跑66张旧保留场景；后者已经使用过，不再当独立验证。异常件的家族挑战根据已知目录选几何方法，仅用于开发诊断，不是生产输入读取目录标签。

| 实物编号 | 场景 | 完整测量 | 接受 | 尺寸拒识 | 不可测/边界/冲突 |
|---|---:|---:|---:|---:|---:|
| DIM-SPECIMEN-01 | 14 | 14 | 14 | 0 | 0 |
| DIM-SPECIMEN-02 | 14 | 14 | 14 | 0 | 0 |
| DIM-SPECIMEN-03 | 28 | 27 | 0 | 27 | 1 |
| DIM-SPECIMEN-04 | 14 | 13 | 13 | 0 | 1 |
| DIM-SPECIMEN-05 | 28 | 26 | 0 | 26 | 2 |
| DIM-SPECIMEN-06 | 14 | 13 | 12 | 1 | 1 |
| DIM-SPECIMEN-07 | 14 | 12 | 12 | 0 | 2 |
| DIM-SPECIMEN-08 | 28 | 0 | 0 | 0 | 28 |

## 5. 已使用Val80的冻结后回归

Val已用于先前分析/选型，本轮结果只叫回归诊断。先冻结代码和配置，再运行本轮80图；没有按本次Val结果再次改算法。正式Stage2 Test未重跑，Stage3图像/专用真值未打开。

| 模式 | 全输入正确率 | 已判定准确率 | Coverage | PASS | REVIEW | False PASS/FAIL |
|---|---:|---:|---:|---:|---:|---:|
| 原v1 | 58.75% | 88.68% | 66.25% | 7 | 27 | 0/6 |
| 修复v2 | 78.75% | 96.92% | 81.25% | 23 | 15 | 0/2 |

实际测量343/378完整，前景证据完整37/80。FAIL Precision/Recall/F1：95.24%/100.00%/97.56%。离线完整处理Median/P95：938.7/1237.5ms；不含解码、模型加载、Camo/曝光、GUI，不是实机新帧延迟。

| 类别 | 原完整/调用 | 新完整/调用 | 新主要失败原因 |
|---|---:|---:|---|
| 1 | 65/72 | 68/72 | HEAD_SHAFT_SEPARATION_FAILED;SHAFT_WIDTH_SUPPORT_INCONSISTENT: 1；SHAFT_EXTERIOR_TRACK_DISCONTINUOUS: 1；SHAFT_EXTERIOR_EDGES_UNSUPPORTED: 1 |
| 2 | 61/77 | 74/77 | HEAD_SHAFT_SEPARATION_FAILED: 2；SHAFT_EXTERIOR_EDGES_UNSUPPORTED;SHAFT_PROFILE_OUTSIDE_CROP: 1 |
| 3 | 67/82 | 75/82 | HEAD_SHAFT_SEPARATION_FAILED: 3；SHAFT_EXTERIOR_EDGES_UNSUPPORTED: 2；SHAFT_WIDTH_SUPPORT_INCONSISTENT: 1 |
| 4 | 58/73 | 70/73 | OPPOSITE_FLATS_NOT_PARALLEL: 2；OPPOSITE_FLATS_NOT_PARALLEL;SIX_FLATS_NOT_RESOLVED: 1 |
| 5 | 54/74 | 56/74 | CIRCULAR_EDGE_ANGULAR_COVERAGE_LOW: 9；ELLIPSE_SUPPORT_DEGENERATE: 6；INNER_HOLE_NOT_UNAMBIGUOUS: 3 |

回归全部退步、False PASS/FAIL及REVIEW原因保存在regression_paired.csv；对象误拒绝/漏检见regression_object_errors.csv。不得只报完成子集准确率。

## 6. 业务状态、剩余缺陷与物理边界

PASS仅用于当前开发输入前提满足、数量一致、无ID0、每个目标通过完整尺寸检查且前景已解释的场景。FAIL用于真实检测ID0、可信且无矛盾的尺寸拒识或数量不符。测量/归属/尺度/规格证据不确定为REVIEW；程序/配置故障为ERROR，无Mock或count-only回退。工单重判不会改变框、检测类别、尺寸及实际数量。

当前仍有真实几何失败、姿态/遮挡及规格证据冲突。螺母对边是可见投影量，垫圈孔受倒角/反光影响，螺丝头下长度仍需实物定义核对。旧经验上限10.08mm与正常开发图10.3–10.4mm的矛盾没有通过放宽数值掩盖。

剩余False FAIL为H005和H034：H005同时包含原YOLO的ID0误检，H034为长度拒识。A088的M5已经通过尺寸检查，整图仍因另一个垫圈ELLIPSE_SUPPORT_DEGENERATE而REVIEW，未强制改成PASS。完整记录见remaining_false_fail.json。

另用45个Train极端/中位长度案例做了多半径、双侧可见头下基准诊断：24个可获得一致可见平面，21个不能确认。Train H033/H035的长长度在这种复核下仍约11.6mm，不能凭标称10mm强行修正。该探针未作为新的生产放行规则，没有用Val再调参数；结果见head_plane_development_probe.json。公共工单实物与专项实物是否同一枚、头型和卡尺实测尚需确认，这不同于已确认的专项文件夹内实物编号关系。

标定仍为35.758446 px/mm的provisional图像标定；照片变焦/镜头差异与高度视差没有独立误差界。未标定的Camo1280×720不输出毫米尺寸或PASS。不能把开发候选PASS当教师正式验收通过。未见未知件的完整定位/拒识尚缺独立物理证据，未匹配前景不会被随意忽略。

## 7. 需要用户补充的最小实测与确认

1. 三枚正常螺丝（M4×25、M5×10、M4×10）及异常M4×40：头型、头下有效长度、总长、螺纹外径，各用卡尺重复测3次，说明标称长度从哪个平面量起。
2. M4普通螺母：三组对边距离、厚度；M6盖型螺母：基部对边、总高及盖型结构记录。M5垫圈与M4小垫片：外径、内径和厚度。不能把M4/M5公称螺纹直径当作垫圈外径。
3. Camo操作：固定Camera0的镜头、分辨率、数字变焦和裁切，关闭自动切镜；把真实毫米尺置于实际测量平面，采中心/四角的横竖刻度帧，记录实际分辨率、机位工作距离及零件测量面高度。新建独立geometry_id，不能按分辨率比例直接套照片尺度。
4. 请教师确认每类尺寸量定义、标称规格、允许容差以及错误规格/不可判定处理规则，保留真实确认来源。现有允许量仍未获教师确认，本轮未改任何validated或批准记录。
5. 修复候选冻结后采集新的独立正常/异常规格实物和场景做物理验收；旧Val、专项66、已跑Test不能重新包装成独立证据，Stage3保留集继续封存。

## 8. 软件验收与复现

软件98项通过，0失败/错误；133个v1结果文件与281个原保护文件哈希一致，v1算法冻结SHA一致。真实GUI worker/批处理和视频核心一致，工单重判不改观测；B1/B2执行和UNSUPPORTED_COUNTS功能复核通过，没有重写它们的验证成绩。实机摄像头未运行，完整Stage2物理验收NOT_PASSED。

```powershell
python scripts/repair_baseline3_size.py --phase auto
python app.py --backend real --model baseline3 --config results/baseline3_v2_experiment/size_repair_v2/p07_profile.json
```

auto按状态跳过已完成阶段，锁防止重复启动；不会训练YOLO或运行封存测试。新核心为src/models/baseline3/geometry_repair.py、size_repaired.py，脚本scripts/repair_baseline3_size.py，测试tests/test_baseline3_geometry_repair.py。旧P06源码、v1源码/配置、app.py、GUI默认配置、B1/B2和权重不变；P03未写入；未commit/push。

原始ROI缓存仅在忽略的outputs/size_repair_v2；大体积逐点测量使用JSON.gz无损保存，measurement_archive.json记录解压后SHA。开发中未完成的慢扫描尝试中止后复用了ROI，未作为成功结果；r01及质量门禁升级前的非最终结果压缩归档在development_revisions，未覆盖旧v1实验。

详细实物测量表、Camo采集步骤及教师确认事项见manual_acceptance.md。
