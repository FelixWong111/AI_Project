# P06 统一候选证据接口

> P07接入更新：默认config/p07.json现经src.adapters.baseline3的size_verified模式加载本候选管线并进行有信任门的业务判定；原生P06候选接口仍只提供REVIEW证据。说明见[当前P07集成](../baseline3_size_integration.md)。foreground_enabled=false时实际原因为FOREGROUND_DISABLED，非BACKGROUND_REFERENCE_MISSING。


当前交付已显式关闭前景补检，foreground_enabled=false。工厂消费
measurement_policy_path 指向的验证记录注册表，按经过验证的条件自动接入尺寸可信证据。
当前真实注册表为空。详见 measurement_policy.md。下文前景关联描述保留为可选实验模式。

src.p06_pipeline:create_backend 实现 P07 已有的 module:factory 协议，提供 metadata、synchronize、infer 和 rejudge。它复用冻结检测器，顺序执行前景关联、同类重复框处理和尺寸证据生成。P08 负责最终 PASS/FAIL 与业务计数规则。

## 启动和配置

```powershell
conda run -n ai-p06-yolo python -B app.py --backend real --model baseline3 --config config/p06_p07_pipeline_candidate.json --image YOUR_DEVELOPMENT_IMAGE
```

上面的配置提供可加载的候选入口。默认 background=null，将明确报告 BACKGROUND_REFERENCE_MISSING；尺寸标定为旧开发拍摄条件的候选配置。实时 1280×720 输入会报告标定分辨率不匹配。工位就绪后应创建独立的新配置，记录实际拍摄标识、空背景和该分辨率下的标定。原冻结检测参数继续使用。

背景结构如下，file_sha256 为实际文件字节哈希。

```json
{
  "path": "path/to/empty_background.png",
  "file_sha256": "ACTUAL_SHA256",
  "reference_id": "recorded-empty-background-id",
  "roi": [0, 0, 1280, 720]
}
```

若指定背景，哈希不符或无法读取会在加载检测器前报错。背景与当前帧尺寸或拍摄标识不一致也会明确报错。空背景不能由有零件的画面自动替代。

## 返回值

顶层 objects 和 counts 为去除已判为同类重复框后的检测观察值，二者严格一致。detection_id 为重新连续编号，source_detection_id 对应原始 YOLO ID。跨类别重复框继续保留观察值和歧义。

每个对象增加 size_evidence 与 rejection_evidence。p06_evidence 保存完整原始检测、前景关联、未知候选及部分候选计数。未知前景保存在 unresolved_candidates，其 class_id=null，未伪造 YOLO 分数，也不加入顶层 ID0 检测计数。P08 消费这些候选后再确定实际对象与计数。

拒识证据结论包括 DETECTOR_NON_CATALOGUE_CANDIDATE、SIZE_OUTSIDE_CONFIRMED_FAMILY_CANDIDATE、LEGAL_SPEC_COMPATIBLE_CANDIDATE 及 INSUFFICIENT_EVIDENCE。背景可用且关联已解决后才提升对象证据。尺寸模块仍要求已验证标定、逐尺寸误差依据及形态家族确认；当前默认工厂不自动提供这些确认。

结果始终为 REVIEW，独立前景、尺寸候选与真实系统验收分别记录。测量坐标仍为裁剪局部坐标，crop_xyxy 提供到原始帧的平移量。

## 缓存和后续共享判定

rejudge 验证缓存摘要并复制已有观察证据，只更新 expected_counts，检测器、前景和几何测量均不重跑。缓存被修改时拒绝重判。摘要用于防止意外修改，不是签名或访问控制。接入 P08 后，应在共享判定中消费缓存证据。

如需在独立工具中传入已经验证的逐帧质量证据，可直接调用 EvidencePipeline.process 的 quality 和 confirmed_families，二者都以原始检测 ID 为索引。它们的格式见 size_evidence.md；工厂默认不据当前测量结果自行设定可信度。

## 当前验证

tests/test_p06_pipeline.py 验证工厂加载协议、去重计数、跨类别歧义、未知前景、ID0、标定缺失、缓存和实际几何函数的合成链路。tools/p06_verify_pipeline.py 使用冻结权重在 GPU 上处理程序生成图像，只用于运行冒烟。

尚待真实工位空背景、现场尺度验证和 P07 摄像头联调。未知前景的物体形态及规格尚不能自动确认，当前证据没有证明完整未见物体拒识能力。
