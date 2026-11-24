# P06 冻结 YOLO11n 推理与六类计数

跨电脑交付请先读 [HANDOFF_P07.md](HANDOFF_P07.md)。最小 GitHub 包不包含开发数据、历史 output 或完整开发测试；新电脑使用 `tools/p06_handoff_check.py`，不是本文后半部分的本机全量验证脚本。

## 本轮交付边界

提供真实 YOLO 目标观察值：原图坐标框、类别、检测置信度及 ID 0–5 计数。无需 P08，可以独立运行，也可以通过 P07 已有 module:factory 接入。没有训练、调参、标注修改或 Test 评估。

本轮不做整盘 PASS/FAIL、尺寸校验或完整开放集拒识。为兼容 P07 已有状态协议，返回 status=REVIEW、predicted_label=None、prediction_score=None、score_type=None，并在 message 和 metadata 中明确 detection/counts only。这里 REVIEW 是“判定尚未执行”的保守接入状态，不代表输入照片已被人工认定为不可判定，不能据此计算正式 REVIEW 率。

## 实际 module:factory 调用链

已完整阅读 README_P07.md，并核对 src/p07_bridge.py、src/p07_ui.py 与 app.py，实际行为为：

1. app.py 读取整份 P07 应用配置，调用 load_backend("real", "baseline3", app_config)。
2. load_backend 将 bridge 按第一个冒号拆为模块名和工厂名，以 importlib.import_module 导入模块，然后调用：

   ```python
   factory(model_id="baseline3", config=app_config.get("model_config", {}))
   ```

   它传的是 model_config 这一整个字典，不是整份应用配置，也不是把字典用 **kwargs 展开。
3. P06 工厂签名 create_backend(*, model_id, config)，仅接受 baseline3；config 只接受可选 config_path，默认 config/p06_baseline3.json。相对模型配置路径和权重路径统一按仓库根目录解析，与调用者 cwd 无关。final、其他模型 ID、未知选项明确拒绝。
4. 返回对象必须有 metadata 字典及 infer、rejudge、synchronize 方法，P07 检查这些成员，并标记 is_mock=False。
5. 初次启动的工厂加载发生在 app.main 中、建立 Journal 和工作线程之前，是同步加载；模型切换时才通过同一个单线程池后台调用 load_backend。不能笼统宣称全部模型加载都在后台。
6. 真实推理在 P07 单工作线程内执行 synchronize() → infer(frame_bgr, expected_counts) → synchronize()。随后 validate_result 验证字段和原图框；异常变为 ERROR，不回退 Mock。计时覆盖 infer 调用，不含工厂加载，但首次推理仍可能含 YOLO 的初始化/预热开销。
7. 修改预期数时调用 rejudge(previous_result, expected_counts)，不做新检测，不记新的 inference_ms。P06 重新校验并复制 previous_result 中的对象和计数；不保存无限帧缓存，也不依赖一个可能已被新帧覆盖的“最后结果”。
8. P07 关闭时会等待推理线程退出、释放采集及日志资源，但当前协议没有 backend.close 调用，模型切换也没有显式卸载钩子。P06 不额外开线程/采集/录像/文件句柄；模型寿命随持有它的 Python 引用管理，不能保证切换后 GPU 显存立即完全释放，PyTorch 还可能保留缓存。

导入 src.p06_inference 本身不加载模型、不初始化 CUDA、不启动推理；调用工厂时才校验配置、文件、SHA-256、Ultralytics 版本、CUDA 0、模型任务及六类名称，并加载冻结权重。缺失权重或 GPU 时明确抛错，不下载替代模型、不回退 CPU。

## 输入输出

```python
from src.p06_inference import create_backend

backend = create_backend(
    model_id="baseline3",
    config={"config_path": "config/p06_baseline3.json"},
)
expected = {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1}
backend.synchronize()
result = backend.infer(frame_bgr, expected)
backend.synchronize()
repeated = backend.rejudge(result, {"1": 0, "2": 2, "3": 1, "4": 1, "5": 1})
```

frame_bgr 必须是非空 NumPy uint8 H×W×3 BGR；非连续视图可输入，模块复制为连续数组以保护调用者原帧。已经解码的数组不带 EXIF 方向信息，不再次旋转或猜测方向。expected_counts 必须恰有字符串键 1–5，值为非负整数，拒绝 bool、浮点、负值及额外键；它只为接入校验，不改变模型检测或计数。

返回结构（下面是字段示意，不是实验结果）：

```python
{
    "status": "REVIEW",
    "predicted_label": None,
    "prediction_score": None,
    "score_type": None,
    "objects": [
        {"detection_id": 0, "class_id": 2,
         "xmin": 10.0, "ymin": 5.0, "xmax": 40.0, "ymax": 35.0,
         "score": 0.8}
    ],
    "counts": {"0": 0, "1": 0, "2": 1, "3": 0, "4": 0, "5": 0},
    "non_catalogue_count": 0,
    "frame_size": {"width": 80, "height": 40},
    "message": "P06 detection/counts only; ..."
}
```

- detection_id 为图内从 0 开始的唯一整数；class_id 为 0–5 的整数。
- 使用 Ultralytics 直接还原的原图连续像素 xyxy，满足 0≤xmin<xmax≤width、0≤ymin<ymax≤height。P06 不再手工缩放、翻转或裁剪非法预测。
- score 是原始检测置信度，不是已校准概率，也不是整盘 PASS 概率。
- counts 按实际保留的对象逐类累加，键为完整字符串 0–5，non_catalogue_count 恒等于 counts["0"]。不根据文件名、预期数或图片级 PASS/FAIL 改目标类别。
- 空检测为 objects=[]、六类零计数；不生成 ID 0 或缺件占位框。模型异常直接抛错，不能伪装成空台面。
- 保留已冻结的类别相关 NMS，不增加跨类别去重。因此同一零件的粗细重复框可能仍被分别计数，这是已知局限，不能用静默删除来掩盖。
- 达到 max_det=300 时明确附加计数可能截断的提示；这个默认上限不代表已验证可准确数 300 个零件。

metadata 包含权重版本/大小/哈希、六类表、冻结阈值、坐标与分数口径、GPU 名称、未完成拒识范围及 calibration/size_tolerance=None。只支持冻结的 YOLO11n 640、conf=0.25、IoU=0.7、device=0、batch=1、FP32；其他配置明确拒绝，防止无记录地改变实验条件。

## 通过现有 P07 运行

新增独立的 config/p06_p07_detection_only.json，保留 P07 所需应用字段，仅将桥接指向 src.p06_inference:create_backend；不修改 config/p07.json、README_P07.md 或 P07 源码。配置中的五类预期数默认全 1 只是界面初始输入，不是正式 GT、固定计数模型约束或合格承诺。

在仓库根目录、ai-p06-yolo 环境执行：

```powershell
conda run -n ai-p06-yolo python -B app.py --backend real --model baseline3 --config config/p06_p07_detection_only.json --image 'C:/your/non_test_photo.jpg' --headless --max-frames 1 --timeout 60 --output-dir output/p06_manual_inference
```

这是开发图片离线链路，不是 Test 评估或正式摄像头验收。界面呈现及导出有真实框、计数、预期和差值，状态为 REVIEW。省略 --headless 可以人工打开图片界面；摄像头启动仍需用户后续确认并现场核验，当前没有代开。

不使用 .venv311 来加载本模型；它是 P07 的环境，尚未安装/验证本轮 YOLO 依赖。使用已验证的 ai-p06-yolo，不改变其他环境。

## 验证证据与限制

新电脑最小验收：`python -B tools/p06_handoff_check.py`、`python -B -m unittest discover -s tests -p test_p06_inference.py -v` 和 P07 已有测试。以下全量命令是 P06 开发工作区的历史验证说明，不要求 P07 准备数据或上传整个 output。

tools/p06_verify_inference.py 复用修正后的数据验收，只访问 development_v2。它通过真实 P07 工厂和 GPU 逐张检查 46 张 Val，比较既有 640 预测，检查输入不变、框与计数一致，再验证空结果、工作线程重判及未改动 app.py 的 headless 导出。Spy 仅统计真实 model.predict 调用次数，不替换预测。

```powershell
conda run -n ai-p06-yolo python -B tools/p06_verify_inference.py
conda run -n ai-p06-yolo python -B -m unittest discover -s tests -p 'test_p06_*.py' -v
conda run -n ai-p06-yolo python -B -m unittest discover -s tests -p test_p07.py -v
git diff --check
```

验证入口默认输出 output/p06_inference_checks，已有目录拒绝覆盖；可通过 --output-dir 指向 output 下的新目录。完整实测报告和 completion.json 位于该目录。后续执行先确认独立输出路径，不能覆盖本次证据。

这只完成 P06 观察接口和 P07 离线兼容性，不代表整盘判定、P08 公共接口、现场摄像头性能、尺寸标定或开放集拒识验收通过。pass59 等粗细错分继续保留；不重训、不在 Test 上调规则。
