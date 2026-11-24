# P06 dataset v2 与旧权重对照

> 历史流程，已因数据污染撤出正式成果；数据和旧权重已清理。当前入口见 [p03_training.md](p03_training.md)，不要执行本页历史命令。

本流程只构建数据和建立验证对照，不包含训练。用户已批准 261/65 候选划分，并确认新增 275 张均独立重新摆放；scene_id 保持 P06 暂定标识。

## 构建

```powershell
conda run -n ai-p06-yolo python -B tools/p06_build_dataset_v2.py
```

唯一正式输出为 `output/p06/yolo_dataset_v2`，已有目录拒绝覆盖。输入固定为已审计的 `data/p03/development_supplement_v1/development_v1` 和已批准的 `output/p06/yolo_dataset_v2_candidate/split_manifest.csv`。

复用旧构建器的 EXIF 加载、转换、PNG 像素哈希、TXT 回算、临时构建与官方加载检查；仅显式 `version=2` 允许 Validation 包含 ID 0。默认 v1 的 41/10、244 实例和 ID 0 仅在训练集等约束保留。

v2 流程核对完整交付包哈希、候选清单哈希、逐图计数/哈希、旧分区和场景归属，失败不发布。PNG 保留完整分辨率，JSON 坐标直接作用于 EXIF 校正图，原图与原标注不改。`dataset.yaml` 相对路径以数据集根目录为 cwd 解释。

## 冻结旧权重对照

```powershell
conda run -n ai-p06-yolo python -B tools/p06_reference_v2.py
```

独立输出 `output/p06/runs/baseline1_best_on_v2_val`，已有目录拒绝覆盖。只加载首轮正式训练第 72 Epoch 的 best.pt，并核对权重 SHA 和六类映射。

首次运行因汇总代码错误访问 `model.validator` 属性而失败，失败目录保留。修复为从官方 on_val_end 回调获取实际遍历数后，使用 `--output-dir output/p06/runs/baseline1_best_on_v2_val_run02` 独立重跑。后续运行必须提供新的唯一目录，不能覆盖这两个目录。

固定预测设置：640、conf=0.25、IoU=0.7、device=0、agnostic_nms=True。保存全部 65 张预测、逐图摘要和联系图；原 10 张预测与前次跨类别 NMS 结果核对。

另用官方验证 API 分别评估 all65、old10、new55；PR/mAP 验证 conf=0.001、batch=8，不能与固定 conf=0.25 的计数指标直接混用。分组 YAML/TXT 只放在运行目录，不改正式数据集。官方标签缓存允许重建，不做缓存 Monkey Patch。

旧 10 张已参与选模与错误分析；新增 55 张是开发验证数据，不是封存测试集。ID 0 的类型/规格覆盖未知，结果不证明任意未知零件拒识。没有 supplied expected_counts，因此不推导正式整图 PASS/FAIL 指标。

## 验证

```powershell
conda run -n ai-p06-yolo python -B -m unittest discover -s tests -p 'test_p06_*.py' -v
git diff --check
```

新测试全部使用临时合成夹具，覆盖显式 v2 的 ID 0 验证集、旧规则保留、原分区保护、清单/哈希/场景错误、清单冻结后变更、构建失败不发布、坐标回算、空标签与分组计数摘要。
