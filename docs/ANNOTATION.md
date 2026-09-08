# 标注指南（PER-007 / 规格 6.2–6.3）

先标道路和遮挡，再标路面异常。目标落在不可观测区域时不得猜测形状。争议样本进入 review 队列，不直接用于高权重监督。

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
