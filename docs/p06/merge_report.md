# P06 → P07 合并、环境和真实集成验收

## Git 与冲突

当前分支 p07/system-integration；P07基准05d2a43，合并源dceda24206096c7be1e518083afc38764ae4d5d9。没有reset、clean、commit或push。12个modify/delete冲突已解决；合并仍待用户提交。

| 冲突组 | 解决方式与依赖审查 |
|---|---|
| config/baseline1.json、models/baseline1/freeze.json | 保留P07精确原字节，必须与真实joblib的配置/模型SHA一致；P06不依赖这套训练配置 |
| src/models/baseline1/catalogue.py、features.py、model.py | 保留P07，P06推理只依赖它自己的六类定义和检测库，无P04导入；迁移后独立保留P06标注工具 |
| predict.py | 保留P07统一入口，通过注册表接入P06；不是换回旧P06项目目录 |
| requirements-p04.txt | 保留P07统一依赖引用，只在根requirements增加YOLO，安装同版本CUDA构建；不降级P07的sklearn/Pillow等依赖 |
| tests/test_p04.py | 保留P07链路诊断，P06测试单独迁移并改import路径；真实模型仅用Validation验收，完整自动化单元测试套件实际执行，正式Test未运行 |
| docs/history/README_P04.md及baseline1_synthetic下config.frozen.json、metrics.json、samples.json | 保留历史证据与贡献，无P06运行依赖，不当作新真实模型 |

另外恢复了合并自动删除的docs/p04/advice_record.md。对P06原模块做AST导入依赖检查，具体记录在results/integration/p06_merge/dependency-review.json；对工厂、结果校验、工作线程、导出接口做实际兼容测试，不能仅凭冲突类型机械保留文件。

发现并修复原.gitignore的data/会同时忽略src/data的问题，改为/data/并补入现有必要Python源码。原数据仍忽略。P06交付树没有原始数据集；唯一图片是历史合成UI截图。真实best.pt从原output位置迁到models，不保留重复权重。

## 实际环境

Windows、Python3.11.14；RTX4060 Laptop GPU 8GiB、驱动592.82；torch2.7.1+cu126、torchvision0.22.1+cu126、CUDA运行库12.6、ultralytics8.3.236。真实CUDA张量运算及GPU NMS通过，pip check通过。保留numpy1.26.4、OpenCV4.10.0.84、sklearn1.9.1、skimage0.26.0、Pillow13.0.0、SciPy1.15.3。

官方CUDA wheel首次下载哈希不匹配时安装停止，未跳过校验；清缓存下载模式重试官方源后安装成功。YOLO依赖从清华PyPI镜像安装，未更改CUDA/YOLO冻结版本。安装前/后环境记录分别保存在outputs/p06_merge及docs/p06/integration-requirements-lock.txt。旧P06独立环境freeze作为历史保留，不盲装覆盖P07。

## 真实模型与证据

| 模型 | 文件与SHA256 | 本次实际检查 |
|---|---|---|
| Baseline1 | models/baseline1/baseline1.joblib；67f99643fb16573f20922ec4ca78bb9881e8d0c1855997d595282f75fdea8d01 | CPU真实重载，46张Val标签和概率与原训练记录一致；固定工单保护保留 |
| Baseline2 | models/baseline2/baseline2.json；641023153cb56693a54bad0dcc8958beda5b7cd4ef4c2a90761e9c573f1c6e55；原ResNet18另保持f37072fd…哈希 | CUDA版框架中的CPU推理，46张Val与原记录一致；未重训或改变分类参数 |
| Baseline3 | models/baseline3/best.pt；4c5a1f759623a88ed2d21d639a4f3254e9da92d89107e4cdbfd6bddcf4e76b3d | CUDA真实检测，46张Val统一批处理；框、六类计数、差值一致；13张出现预测ID0，全部FAIL |

Baseline3的13个ID0例子是“模型预测到ID0后按规则FAIL”的接口证据，不是13个真值全部检出或完整未知件拒识准确率。没有据此调整阈值或类别。

真实GPU工厂、工作线程、输入原像素不变和count-only rejudge通过；Spy统计model.predict只执行1次，改变要求后的重判没有再次检测，框和实际计数完全不变。GUI实际控制器依次选择Baseline1/2/3并导出三份真实结果，只有原生窗口调用隔离，未伪造模型输出。固定样例pass102.jpg上Baseline2判为FAIL的已知误判保留，没有改标签来满足验收。

全套76项自动化测试通过。真实模型加载失败明确ERROR，不回退Mock/CPU；P06原合成接口测试明确只是测试夹具。Baseline1/2工单限制与特殊输入回归另由scripts.check_work_orders执行。所有检查只用Validation或独立接口样例，Formal Test: NOT_RUN。

原P03旧历史快照已在先前工作期间变化，旧验收脚本因此会在全部预测一致后仍对比旧快照失败；现改为每次运行开始抓取元信息并在结束比较，不回写/恢复P03。本次合并开始的独立before-state.json又单独验证原Baseline1/2全部工件SHA与P03元信息未变，不能把“修复历史检查”当作忽略本次保护。

## GUI及批处理证据

- results/integration/baseline3/acceptance.json：真实CUDA、46张批处理、ID0、三模型控制器及无头app。
- results/integration/baseline12/acceptance.json：CUDA环境下Baseline1/2各46张Val与原训练预测逐张比较。
- results/integration/p06_merge/unit-tests-final.log：76项测试通过。
- outputs/p06_merge/final-ui/：修正后真实ID0截图；类别/分数按显示分辨率绘制，count-rule结果仅分数N/A，设备显示cuda:0。
- results/integration/work_orders：特殊合法工单、非法工单、恢复默认、无头app与CLI检查；每次独立输入和输出目录。

本轮工单回归报告为results/integration/work_orders/acceptance.json，真实执行退出0，两个模型U01–U09接口检查均通过。三模型GPU批处理/控制器与76项测试均已完成；未执行正式Test或现场摄像头验收。

人工桌面窗口点击、真实摄像头及20新帧延迟为NOT_RUN，不能以截图/无头控制器当作实机验收。

## 功能完整性与课程合规

真实六类检测、定位、计数、可变工单和已检出ID0强制FAIL已集成。尺度标定、尺寸容差、未匹配前景候选、错误规格核验及经验证的未见目录外件拒识尚未提供/验证，项目也没有找到P01有效标定资料。代码没有编造毫米比例、公差或把这些标为完成。具体强制要求、团队方法约定、代码缺口、所需资料与验收方法见compliance_review.md。

models/baseline3/training_record.json保留checkpoint的训练参数、88轮历史曲线及存储指标（precision0.96978、recall0.9775、mAP50=0.98112、mAP50-95=0.59266）。这些是P06历史存储值，不是本次重新评估结果，也不能代替缺失的GT修正后复验报告。选中epoch58为P06交接说法；strip后的checkpoint自身epoch=-1，单独标明，不伪造训练重跑。

未提供文件/资料：配置引用的post-GT-fix复验报告、640/960对照报告、原训练命令日志/样本清单、P01标定/尺度参考/教师确认容差，以及开放集与错误规格验证证据。真实best.pt已完整交付，不缺权重；Final没有接入真实模型，本次不实施Stage3研究改进。

## 文件整理

新增：config/baseline3.json；models/baseline3/的真实权重、训练记录与来源；src/models/baseline3/、src/adapters/baseline3.py；src/data/p06_labelme.py；scripts/check_baseline3.py；P06原测试与新增适配器测试；本组交接、合规与验收文档。

迁移：P06根推理模块到模型包，标注工具到src/data，output中的唯一best.pt到models；旧独立GUI配置及旧检查脚本归档docs/p06/original。旧入口删除，没有删除原始P03或训练贡献证据。

修改：注册表、config/p07.json、根依赖/README、架构/接口说明；GUI仅修正设备信息、count-rule分数文字和框标签可读性；更新两个验收脚本的运行前后快照基准；P07缺失模型控制器测试明确使用缺失的Baseline3配置避免单元测试意外加载真实GPU模型。

保留：P07三模型统一CLI、工单校验、Baseline1/2全部权重/配置/验证记录、摄像头功能、快照录像、历史文档、未启用的Stage3候选模块及Final扩展入口。完整改动以git diff --cached --name-status为准，未提交。

```text
app.py / predict.py
config/{baseline1,baseline2,baseline3,inspection,p07}.json
models/{baseline1,baseline2,baseline3}/
src/adapters/{baseline1,baseline2,baseline3}.py
src/models/{baseline1,baseline3}/
src/data/{development,p06_labelme}.py
src/inference/batch.py
src/p07_*.py / src/work_orders.py / src/camera.py
scripts/check_baseline3.py / check_candidates.py / check_work_orders.py
docs/p06/ / docs/baseline3_integration.md
tests/
```

P03保持只读，原始data/P03/开发/验证/测试均不在Git跟踪集合中；真实模型在models中正常跟踪。原outputs仍被Git忽略；正式文字证据已复制到results随Git交付，无需另行打包；不要将原始数据目录加入提交。

## 正式结果归档补充

本报告描述合并任务当时的状态，不代表后续Git提交状态。当前正式证据入口：[results/README.md](../../results/README.md)。CUDA安装前/后环境分别见 results/integration/p06_merge/pip-freeze-before.txt 和 results/integration/environment-after.txt。原始截图、备份ZIP及完整本机快照保留outputs，不作为Git交付文件；GUI结果可通过已归档CSV、配置和控制器验收复核。后续归档没有重跑模型或正式Test。
