# P03 新交付数据的 P06 训练入口

输入固定为 `data/p03/development_v2`，使用已完成只读验收的 source_sha256.json 与 p06_derived_manifest_reviewed.csv。保留 P03 Train/Validation 归属、原 CSV 的 TBD 和用户确认每图独立摆放的 P06 暂定场景依据，不读取 Test、不重新切分。

```powershell
conda run -n ai-p06-yolo python -B tools/p06_train_baseline.py --smoke --dry-run
conda run -n ai-p06-yolo python -B tools/p06_train_baseline.py --check-data
conda run -n ai-p06-yolo python -B tools/p06_train_baseline.py --smoke
```

`--dry-run` 只检查，不建立运行视图或训练目录。`--check-data` 只准备运行视图并官方加载检查。`--smoke` 仅 1 Epoch。未来正式训练必须显式 `--train`，并另行获得授权，本轮不执行。

## 输入保护与缓存

完整原件哈希与验收记录匹配，逐图重新检查 JSON、TXT、CSV、方向、尺寸、类别、场景清单及框一致性。错误数据、目录变化、Test/download 配置、错误官方权重哈希均拒绝。验收数据变化后需重新审计，不能自动更新基准哈希放行。

运行视图为 `output/p06_runtime_p03`。JPEG 使用 NTFS 硬链接，不重编码、不复制约 628 MiB 图像内容；标签是小体积独立副本，官方标签缓存只写此视图。创建链接前验证 JPEG 格式、方向、完整性和 EOI，拒绝可能触发官方 JPEG 自动修复的输入。官方加载逐张遍历 Train/Val，验证类别和实例数；不对缓存函数做 Monkey Patch。

**硬链接不是独立图片副本或写权限隔离。不得手工编辑运行视图 JPEG，也不要让图片编辑器在该目录保存文件，否则会改变对应原件。** P06流程只读取图像，每次运行前后核对原件哈希；删除链接本身不会删除原件路径。原件有变动时停止，不能依赖这个视图当数据备份。

## 精度兼容

`from_yolo` 增加显式 decimal_places 参数。对十位小数 TXT，按归一化中心及宽高的半个舍入单位传播误差：像素边界容限约为 `0.75*10^-10*尺寸`，且上限 1e-6 像素。只对反算中的极小越界贴边，随后与严格合法的 JSON/CSV 框比对。

默认旧函数行为及源 `_box`/`to_yolo` 边界规则不变；真实越界、零面积、非有限值仍报错，不更改 P03 TXT。fail176 的 xmax 舍入回归已覆盖。

## 训练和输出

默认初始化只有官方 YOLO11n COCO 预训练权重，绝不读取已删除的旧项目 best.pt。固定 640、batch 8、device 0、seed 42、workers 4、cache=False、deterministic=True、AMP/val/save/plots=True，patience=30；smoke 仅把 epochs 改为 1。其他使用 Ultralytics 8.3.236 默认设置，不继承污染实验选择的后处理设置。

默认冒烟输出 `output/p06_runs/p03_smoke_1epoch_640_seed42`，已有目录和控制台日志拒绝覆盖，后续经授权运行须提供新的 `--output-dir output/p06_runs/<unique_name>`。保留配置、完整控制台日志、有限梯度/参数变化证据、best/last、results和官方曲线；重新加载 best 独立验证实际 Val 数与 GT 类别支持。

预测摘要覆盖当前全部 Validation，显示预览仅保存最长边 800 的 JPEG 和分页图集，避免再次积累全分辨率预测副本。训练输入始终是完整 JPEG，预览不作为训练数据。

1 Epoch 只证明工程链路，不证明六类性能、拒识或无泄漏。Test 的跨集关系与历史曝光仍待 P03/P08 核定；正式模型性能结论和后续长训练需另行审核。
