# Mechanical components v2 — Stage 2

唯一正式本地数据根目录：`datasets/mechanical_components_v2/`。Train 360、Validation 80、Stage2 Test 30，冻结划分不变。尺寸开发154、已使用回归66、标尺10，共700张图片；专项220张仅对应8件实物。Stage3的90张图及其真值不在此交付目录。

## Git与网盘

Git只提交标注、Ground Truth、划分清单、类别映射、SHA256、数据配置和本README。**图片被.gitignore排除，不通过GitHub或Git LFS分发；图片通过网盘单独共享。** 模型权重仍通过Git/LFS提交。本地图片未删除、移动或重命名。网盘链接由P07另行提供，尚未记录实际上传或下载验收。

P08克隆代码并执行`git lfs pull`取得模型后，从网盘下载图片，将图片包按原目录合并至项目根目录的`datasets/mechanical_components_v2/`：

```text
public/train/images/                 360
public/validation/images/             80
public/stage2_test/images/            30
size_special/development/           154
size_special/regression_used/        66
calibration/ruler_images/             10
```

不要改名、缩放或重新编码。路径与文件名恢复正确后，无需修改代码和配置。Baseline3还需要其冻结配置引用的Train背景图，因此先恢复图片，再执行模型加载检查。

GT位于`annotations/`及各`public/`分区；YOLO labels和原始JSON标注保留。推理只读取图片和`configs/expected_counts_*.json`，不读取GT。`manifests/source_sha256.csv`、`provenance.csv`和`source_mapping.csv`保存文件SHA及来源映射。恢复图片后可按根README运行数据检查；本轮没有重新执行实验评价。

66张专项回归图已被使用，不是全新独立测试集。标定未获教师确认，原照片尺度不直接适用于Camo；P03及原始数据保持只读。
