# 界面操作

启动：`python app.py --model baseline1`。顶部模型按钮或 1/2/3/4 切换模型，L 重新加载。Baseline1/2 仅支持五类各 1 的固定工单。Baseline3/Final 没有检测权重时显示 ERROR。

顶部 Camera / Image / Video 选择输入；摄像头编号通过 +/- 或点击编号修改。C/I/V 为对应快捷键，X 关闭输入源。本机用户已确认 DSHOW camera 0 为 Camo；其他电脑应先执行 `python app.py --list-cameras`。

Start/Pause 或空格控制推理。S / Snapshot 保存当前图像、结果 CSV 和运行配置；R / Record 开始或停止 MP4 记录；Q 退出。右侧 Expected 可点击编辑，也可导入课程格式的 expected_counts.json。图像级模型遇到其他合法工单时显示 UNSUPPORTED_COUNTS。

右侧显示模型版本、PASS/FAIL、分数含义和计数能力。图像级模型的框、观测计数和 ID 0 数量为 N/A，不能把整图 FAIL 解释为定位了陌生件。模型或输入缺失时明确报错，不回退 Mock。

本轮仅用允许的验证图进行界面检查，正式测试图片不用于演示。相机延迟需要至少 20 个新帧另行采集，无头图片验收不能充当相机实测。

当前已完成真实后端和无头 app 验收、控件回调自动测试、渲染截图检查；桌面自动化没有找到本轮启动的可操作窗口，因此人工点击验收和实时摄像头验收仍待完成。
