# 标注指南（PER-007 / 规格 6.2–6.3）

先标道路和遮挡，再标路面异常。目标落在不可观测区域时不得猜测形状。争议样本进入 review 队列，不直接用于高权重监督。

## 颠簸三类（训练用 YOLO 框）

Bump 学生网只吃 `pothole` / `speed_bump` / `manhole_cover` 的检测框，写在 `artifacts/bump_dataset/labels/raw/*.txt`。

不要用 CVAT / X-AnyLabeling / Label Studio 另起一套工程：那些工具能跑 SAM2，但导不出这条训练链路。本仓库的 `rpar label` 复用 **Ultralytics SAM2**（点选/框选补全 + 短窗跟踪），界面按骑行视频暂停来做。

```text
rpar label --video Video/train/inbox/VID....mp4
```

| 操作 | 作用 |
|---|---|
| 空格 | 随时暂停 / 播放 |
| 左键单击 | 正点击 → SAM2 补全框 |
| 左键拖动 | 框选 → SAM2 收紧 |
| 右键 | 负点击（去掉误切到的车/影子） |
| `1` `2` `3` | 大坑 / 减速带 / 井盖 |
| Enter | 只保存当前帧 |
| F | 保存并向前后约 1–2 秒跟踪 |
| ← → | 逐帧；Shift 步进 0.5 秒 |

人工框**不会**走 `keep_box`。标完用 `rpar train-bump --skip-propose`，不要再跑会覆盖 `labels/raw` 的 `ride-train`。

## 属性

| 字段 | 取值 | 规则 |
| --- | --- | --- |
| semantic_type | pothole, manhole_cover, speed_bump, road_joint, repair_patch, rough_broken, puddle, gravel, unknown_anomaly | 对象是什么。无法判断凹凸时用 unknown_anomaly，不要强行归类。 |
| geometry_type | concave, convex, rough, step, flat, unknown | 垂直起伏的主标签。 |
| state | normal, abnormal, unknown | 井盖必须同时标 manhole_cover；是否异常由 geometry/state 决定。 |
| severity | 0 无 / 1 轻 / 2 中 / 3 重 / unknown | 记录当前速度范围与可见性，避免把低速实感外推到 47 km/h。 |
| visibility | clear, blur, underexposed, overexposed, glare, occluded, lens_drop, unknown | 可逐帧变化；同一物理对象的语义/几何应跨帧一致。 |
| mask | polygon / RLE | 只覆盖异常真实区域，不含大面积正常道路。 |
| track_id | 整数 | 同一物理对象跨帧保持。 |

## Hard negative（必须覆盖）

- 普通黑色修补但表面平整：`repair_patch` + `flat` + `normal`，不得单独触发声音。
- 树影、标线、湿反光、车辆阴影、尾灯泛光、车灯光斑、正常井盖。

## 方向与减速带

减速带若横跨多个方向区，mask 标完整实体；左/中/右由运行时几何模块判断，标注员不在画面上画三等分。

训练/验证/测试按完整会话、日期或路线隔离，禁止把相邻帧分到不同集合。

## 公开数据（只映射，不直接合并）

不同公开集在回答不同问题：工程病害分类 ≠ 骑手冲击风险。接入外部框/掩码时用 `rpar map-label` / `rpar.ml.labelmap`，不要把 RDD、Rome、SVRDD、CNRDD 的类别表拼成一张 YOLO 标签。

- 普通井盖、裂缝、平整修补：语义可以留下，但默认 `state=normal`，不得单独触发声音。
- 井盖是否凸起/下陷必须自采补标，公开「maintenance hole」不等于异常。
- 几何主标签优先于语义：depression→`concave`，protrusion→`convex`。
