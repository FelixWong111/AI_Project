# Baseline3 真实模型集成

以P07提交05d2a43为工程基础，接入P06提交dceda24206096c7be1e518083afc38764ae4d5d9。保持合并待提交，不自动commit/push；不训练、不使用正式Test。

## 交付资源

- 权重 models/baseline3/best.pt：5465242 bytes；SHA256为4c5a1f759623a88ed2d21d639a4f3254e9da92d89107e4cdbfd6bddcf4e76b3d。
- 配置 config/baseline3.json：只修改原权重位置，不改冻结推理参数、类别或阈值。
- P06检测源迁入 src/models/baseline3/inference.py，其类别/标注工具迁入 src/data/p06_labelme.py。
- P07适配器 src/adapters/baseline3.py，注册于 src/p07_bridge.py，app.py与predict.py共用。
- models/baseline3/training_record.json：从checkpoint保存train_args、train_metrics及88轮train_results。历史指标不改写为本次实测；checkpoint epoch已strip，交接称best epoch58，单独标注来源。
- 来源/缺失资料见models/baseline3/provenance.json。原P06文档与依赖快照保留；旧独立UI配置和检查脚本归档于docs/p06/original。

冻结参数：YOLO11n、imgsz=640、conf=0.25、IoU=0.7、device=0、batch=1、half=False、agnostic_nms=False、augment=False、max_det=300。类别直接为项目ID0–5，不减1映射。目标置信度不是整盘PASS概率。

## 环境和运行

统一使用现有P07 .venv311。Baseline1使用CPU，Baseline2默认CPU，Baseline3严格CUDA0，不静默回退CPU或Mock。需要NVIDIA显卡和兼容驱动；不能只看nvidia-smi就声称PyTorch CUDA可用。

```powershell
.\.venv311\Scripts\Activate.ps1
python -m pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python app.py --model baseline3
python app.py --model baseline3 --image "允许的验证图片路径"
python predict.py --input_dir "允许的验证图片目录" --model baseline3 --output_dir outputs/baseline3-validation --counts_file outputs/work_orders/validation.json
```

保留P07的sklearn/skimage/Pillow/SciPy版本，不盲装P06旧freeze；Ultralytics固定8.3.236。实际环境验收见合并验收记录。本机初次CUDA下载哈希校验失败后停止，后续重新从官方源下载，不绕过校验。加载只使用本地校验通过的best.pt，不下载通用模型替代。

默认config/p07.json管理三模型路径，不使用P06旧GUI。1/2/3或顶部按钮切换；Final仍未配置，不复用Baseline3冒充研究改进。Baseline1/2仍只支持固定各1；Baseline3支持任意合法工单。

## 判定和输出

Baseline3生成images.csv、objects.csv、counts.csv、run_config.json。框为原图xyxy；计数含0–5六行/图，difference=observed-expected，ID0期望恒0。目标分数保留，纯数量规则整图分数为空。

原P06接口用REVIEW表示“仅检测、未做整盘判定”，并非图像不可判定检测器。P07校验真实观测后调用公共数量规则得到PASS/FAIL；检测到ID0即警告并FAIL。达到max_det且没有已检测ID0时保留REVIEW，不能把潜在截断计数当PASS。只修改expected时不再检测、不改类别、框或实际计数，异常不会伪装成空台面。

## 能力与证据边界

当前只有六类YOLO、计数及学习到的ID0。没有P01尺度/容差，没有未匹配前景候选，没有验证完成的错误规格或未见类型拒识。本次不新增这些算法或编造物理参数。GUI/metadata提示“Count-rule result / learned ID0 only / no calibrated size verification”。专项审查见docs/p06/compliance_review.md。

P06配置引用的修正标注后复验报告、640/960对照报告未随分支交付，原始训练命令日志和样本清单也未交付。checkpoint内参数和历史指标已保存，但不能独立证明历史划分无泄漏或完整错误规格拒识已通过。

scripts/check_baseline3.py只访问validation清单：真实CUDA工厂/工作线程、同帧重判、46张Val批处理、ID0规则、坐标/计数、三模型GUI控制器（真实模型，原生窗口调用隔离）和无头app。人工桌面点击、现场相机和20新帧延迟分别标记，不能用无头结果替代。

Baseline1/2回归用scripts.check_candidates和scripts.check_work_orders，对比保存模型和原始验证预测；不重训。P03只检查元信息，不写入。Formal held-out Test: NOT_RUN。

.gitignore原data/误忽略src/data，已改为/data/，仅忽略根原数据。P06分支没有原始数据集，只带历史UI合成截图和真实权重。权重移入models，不保留output中重复副本、不遗漏交付。最终仍需用户自行检查并完成合并提交。
