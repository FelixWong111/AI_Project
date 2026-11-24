# 固定参考工单、输入校验与数量判定

唯一公共要求配置：`config/inspection.json` 的 reference_expected_counts 为 ID1–ID5 各 1，non_catalogue_expected 恒为 0。公共逻辑在 `src/work_orders.py`。GUI 默认值、工单生成器、训练数据准备与模型能力检查都使用它；保存模型内部的 supported_fixed_counts 是历史训练事实，保留不改，加载时必须与公共参考一致。

Ground Truth 的实际数量不是工单：实际 `[1,1,1,1,2]` 对固定 `[1,1,1,1,1]` 是 FAIL。不得根据实际数量生成 expected_counts，也不能通过改配置把旧模型冒充为另一个工单训练的模型。当前两个模型内部、冻结配置、训练记录和原始数据清单均确认使用各 1 工单；模型权重及预处理不变，训练和验证成绩保留。

## JSON 输入

```json
[
  {
    "image_id": "example.jpg",
    "expected_counts": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1}
  }
]
```

真实文件名含扩展名，单次运行唯一。每条只有这两个字段；ID0、实际数量、标签、框不允许进入推理输入。布尔值、小数、负数、缺类、多余类别或字段、无扩展名的用例编号都拒绝。GUI 导入与 CLI 共用同一解析器；CLI 对非法 JSON/工单以输入错误退出，不创建结果目录。

## 生成命令

在项目根目录、已激活 .venv311 后：

```powershell
python -m scripts.generate_work_orders --split train --output outputs/work_orders/train.json
python -m scripts.generate_work_orders --split validation --output outputs/work_orders/validation.json
```

本轮已生成 train 262 条、validation 46 条。默认读取经过审查的 `outputs/training/data/manifest.json`，只取 image_id 与 split，不读图片或根据实际数量/标签生成要求。既有文件不覆盖，复跑请换输出文件名。可用 `--manifest` 显式指定 CSV 或 JSON 清单。

旧用例编号只能使用明确 `case_id,image_id` 两列的 `--mapping-csv` 转换；无映射时 TEST_01 会被拒绝，不猜测文件名，重复映射/重复图片也拒绝。整理后的数据已直接使用实际图片名，不需要映射。

本轮不重生成或读取正式 Test 输入。测试最终封存后，可以显式使用：

```powershell
python -m scripts.generate_work_orders --manifest data/test_20261124/ground_truth_images.csv --split test --output outputs/work_orders/test-final.json
```

上述正式 Test 命令本轮 NOT_RUN，现有测试包及已有 JSON 不修改。

## GUI 与图像级模型

GUI 直接使用内存中的当前 expected_counts；点击数量控件不读写 JSON。config/p07.json 可选择性设置启动工单覆盖值，缺省取公共参考。自定义数量控件保留，供检测模型使用。

Baseline1/2 仅支持冻结的固定向量。其他合法工单立即呈现 UNSUPPORTED_COUNTS、Decision=N/A、Score=N/A，并提示恢复固定数量或切换 Baseline3。该状态不是 FAIL 或第三个分类类别。工单检查在 HOG/CNN/分类前执行；直接 Adapter/桥接入口也受保护。模型或输入错误仍明确报告错误，不回退预测。

修改工单后，即使旧帧异步推理尚未结束，也不能显示旧 PASS/FAIL 或分数。只有缓存的 model 与 expected 同时匹配当前设置才允许展示；切换模型或加载期间清空旧判定。恢复固定工单后可重新推理，或复用同一帧在相同工单下的有效缓存，无需重启。快照在重判未完成时不能保存旧判定为新工单结果。

P07 的 Baseline1 场景支持检查仍保留：固定工单时，合法场景才能得到真实 PASS/FAIL；分布外输入可为 REVIEW。UNSUPPORTED_COUNTS 优先于分类和场景门控，不能将它误当 OOD 或 FAIL。原始基线批处理不启用场景门控，能力限制相同。

## 目标级模型

Baseline3/Final 的真实检测结果包含实际计数、框、类别和目标分数。统一数量规则比较 observed 与任意合法 expected；ID0>0 始终 FAIL，difference=observed-expected。纯规则的整图判定不制造图像级概率；目标分数保留。

真实后端加载后通过 ObjectBackend 接口复用观测；rejudge 只计算新判定，不再调用检测器、不改变框/类别/计数、不修改观测对象。当前仍未交付真实 Baseline3/Final 权重与适配器；计数逻辑仅有明确标识的单元接口夹具检查，不能写成真实检测验收通过，也不提供生产 Mock 回退。

## 两类实验必须分开

共同性能评估统一使用 reference_expected_counts。合法但不同的数量只用于功能测试，使用 `--purpose functional --counts 1 1 1 1 2`，分别保存 JSON 和运行目录，不能混进共同性能指标。生成器拒绝 common 模式下覆盖固定工单。

```powershell
python -m scripts.generate_work_orders --split validation --purpose functional --counts 1 1 1 1 2 --output outputs/work_orders/validation-extra-id5.json
python predict.py --input_dir "验证图片目录" --model baseline1 --output_dir outputs/functional-baseline1 --counts_file outputs/work_orders/validation-extra-id5.json
```

UNSUPPORTED_COUNTS 不计成 FAIL，不进 PASS/FAIL 指标分母，应单独报告支持范围和样本数。相同图片改变工单必须另起运行与结果目录。Ground Truth 只供独立评估器使用，推理不读真值。

## 验收

`python -m scripts.check_work_orders` 仅使用一张允许的验证图，真实加载两个模型，执行四种特殊合法工单、恢复、非法输入、批处理及无头 app；以调用计数断言特殊工单没有提取 HOG/CNN 特征。使用独立运行目录保存 JSON、CSV、命令和验收报告。不会重训或读取正式 Test 图片。

`python -m unittest discover -s tests -v` 同时覆盖生成器忽略实际数量、显式映射、非法输入、GUI 旧结果屏蔽，以及检测观测不可变的纯接口约束。GUI 控制器事件可自动检查，但人工桌面点击与真实检测模型验收应分别标记，不将无头/夹具测试写成实机检测成功。
