# 实质性建议与核查

来源：本次用户提供的开发任务、Codex 开发协助。采纳摘录：“修改预期数只重新判定，不无必要地重新执行目标检测推理”。

决定：采纳并限定实现边界。P07 通过后台任务调用桥接层 rejudge；MockBackend 内有测试规则，真实模式只委托 P08，不在 UI 复制正式判定。

代码位置：app.py 的 dirty_counts 分支、src/p07_ui.py 的 InferenceWorker、src/p07_bridge.py 的 MockBackend.rejudge。

核查：tests/test_p07.py 的 test_U04_no_reinfer_or_observation_change 断言 objects 对象不变、counts 不变且含 ID0 始终 FAIL；test_worker_error_and_rejudge 核查重判没有 inference_ms。实际执行结果保留 verification.json。

限制：这些是开发过程测试，不替代组员亲自复核，也不代表正式共享接口已经确认。真实模型的同帧 U04 仍为 NOT_RUN。

## 2026-11-26 新数据处理

使用Codex协助编写只读扫描、LabelMe到课程GT/YOLO转换、seed42按场景组分层划分、数据加载适配及自动检查。采纳依据：标签从人工目标标注计数推导，工单统一复用公共配置；不以文件名/模型预测代替真值。用户明确确认每张非空公共图独立摆放，D18/F21以原JSON为准；确认与SHA256保存在新数据审查记录。

核查方式：原始文件SHA256前后对照、逐框JSON/GT/YOLO还原、组/像素隔离、实际数据加载器和P08阶段清单检查、单元测试、历史权重与结果保持不变。人工标注真实性与卡尺/标定数据仍需成员复核。未使用新正式Test进行模型预测或评价，未声称AI输出已经等同于人工标注验收。

## 2026-11-26 Baseline重训与独立对照

用户提供全量标注核实及Stage 3未使用历史确认后，记录SHA绑定证据并冻结数据。Codex协助复用原HOG+MLP、冻结ResNet18+LR训练；首轮保持旧已选参数，不搜索。用P08在相同80图Val选择候选，再对相同30图Stage 2 Test独立评价；Stage 3未推理、未调参。

保留的工程修改仅用于比较：P08独立模型配置、显式Coverage/比率分母及离线处理计时。原始预测、失败及退步案例保留。按用户收窄范围要求，撤回本次报告/图表、实机脚本扩展及新guard运行时接线；已有P08成果和默认GUI不变，不进行模型正式替换。清理只读取已保存结果，未再次训练、推理Test或访问Stage 3。


## 2026-11-26 P07最终集成

OpenAI Codex协助实现唯一注册表、SHA检查、派生数据迁移、接口/软件测试和交接索引。三模型选择来自用户明确指令及既有冻结结果，人工标注确认来自负责人，未将其归为AI亲自采集或标注。

采用并核实：相同B1/B2权重/预处理的Stage2预测复用，避免重复推理；新B3尺寸v2在冻结后运行一次。证据为results/p08_stage2_final/*/provenance.json与results/p07_final_integration/release_freeze.json。修改AI输出：补类别ID显示并记录独立UI版本，不修改检测或判定；303项软件测试、真实离线控制器/worker/batch/video/Record和SHA核对通过。

没有将Camo全黑帧、合成fixture或Train编码视频当作实机证据。教师确认、Camo物理尺度及真实视频仍待人工完成；完整干净ML安装未验证，纯Python干净环境评分/权重SHA复核通过。
