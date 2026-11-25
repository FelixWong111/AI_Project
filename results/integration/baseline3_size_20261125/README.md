# P07尺寸管线接入验证 · 2026-11-25

这是本次真实执行的集成验证记录，不是重新训练或正式Test评估。旧results中的模型成绩和实验数据保持不变。本子档案使用独立source_manifest.json记录来源及公开副本SHA，不改写旧冻结清单。

## 实际结果

- 全部255项自动化测试通过；合成可信证据夹具验证了PASS、已确认尺寸拒识class0、数量FAIL、REVIEW、缓存和错误保护，不能当作真实卡尺/标定证据。
- 真实CUDA尺寸管线、后台工作线程、两种模式批处理、真实无头app图片/离线视频均运行通过。
- 四张允许Validation：pass102.jpg在尺寸模式REVIEW、冻结模式PASS；fail27.jpg缺件、fail140.jpg多件、fail130.jpg目录外件两模式均FAIL。
- 每张同帧两次改工单不重新检测或测量，原始和最终观察均保持不变。
- 故意破坏管线路径时批处理返回ERROR/退出1，没有退回旧YOLO。
- Baseline1/2各46张真实Validation回归，标签及分数与冻结记录一致；两模型工单U01–U09实际检查通过。
- 核心几何、螺丝算法、权重、规格、公差、真实标定和策略记录未改；P03元信息未变。真实标定仍false，策略records仍空，未生成接受确认。

## 文件

acceptance.json：真实CUDA及接口测试、实际命令、逐样例状态和测量证据；diagnostics.json：真实尺寸模式批处理的完整公开证据；baseline12-regression.json与work-order-regression.json：真实分类器回归；unit-tests.log：本次255项执行日志；protection.json：保护核对；[尺寸详情图](../../../docs/baseline3_size_details.png)：当前渲染器显示真实Validation推理证据（不是摄像头实拍）。source_manifest中的path相对仓库根目录。

公开副本只规范化个人本机路径，不改变标签、参数、测量值或成绩。原始执行目录见source_manifest.json。不是所有对象都能测量；其中返回的mm值仍为provisional，不是精度验证。

## 尚未完成

当前有正常数量输入REVIEW并非硬编码，而是缺真实标定、接受确认、测量策略和前景条件。物理可信PASS、错规格实物拒识率、MAE/FAR/FRR、完整未知件覆盖、现场相机/新帧延迟/视频和正式Test均未完成，不得从工程接入成功推出Stage2完整通过。

最新模式、字段、人工标定步骤和合规矩阵见[集成说明](../../../docs/baseline3_size_integration.md)。
