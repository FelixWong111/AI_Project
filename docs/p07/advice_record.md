# 实质性建议与核查

来源：本次用户提供的开发任务、Codex 开发协助。采纳摘录：“修改预期数只重新判定，不无必要地重新执行目标检测推理”。

决定：采纳并限定实现边界。P07 通过后台任务调用桥接层 rejudge；MockBackend 内有测试规则，真实模式只委托 P08，不在 UI 复制正式判定。

代码位置：app.py 的 dirty_counts 分支、src/p07_ui.py 的 InferenceWorker、src/p07_bridge.py 的 MockBackend.rejudge。

核查：tests/test_p07.py 的 test_U04_no_reinfer_or_observation_change 断言 objects 对象不变、counts 不变且含 ID0 始终 FAIL；test_worker_error_and_rejudge 核查重判没有 inference_ms。实际执行结果保留 verification.json。

限制：这些是开发过程测试，不替代组员亲自复核，也不代表正式共享接口已经确认。真实模型的同帧 U04 仍为 NOT_RUN。
