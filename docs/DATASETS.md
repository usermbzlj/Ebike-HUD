# 数据与模型立场（工程判断，不是下载清单）

没有一份公开数据或单一模型能直接覆盖「电动车前视、夜间、强抖、远距小坑、异常井盖、低误报 AR」。公开资源只用来预训练、教师伪标签和 hard negative。最终域是 Find X8 Pro + 左侧车把 + 自采 session。

## 分层：开集大模型只负责颠簸三类

| 层 | 现在 | 以后 |
| --- | --- | --- |
| 可行驶道路 + 车辆遮挡 | 桌面 YOLOPv2 sidecar；启发式道路只是兜底 | 手机 `roadseg-field-0.1.0` JSON 双尺度头（教师蒸馏）；PIDNet 类学生仍待实机 |
| 路面异常实例（大坑 / 下沉井盖 / 减速带） | 桌面 **YOLO-World** 开集检测（提示词，不拼公开类别表）+ `Video/train/inbox` 自采微调；无权重时启发式兜底 | 量化学生网上手机；不要把 Ultralytics AGPL 当成手机唯一路线 |
| 跟踪 / 距离 / TTC | IMU + 路平面 + 走廊 | 教师深度只做离线辅助，不在手机常驻 |
| 声音 | 颠簸三类单独阈值；正常井盖不播报 | hard negative 质量比再堆坑样本更重要 |

YOLOPv2 **不是**坑洞/井盖检测器。夜间沥青噪声不得标成 `rough_broken`。

## 标签：先几何，再语义，再风险

公开集的「病害分类」和骑手「会不会颠」不是同一个目标。接入任何外部框/掩码时必须走 `rpar.ml.labelmap`：

- 井盖 ≠ 坑。普通井盖是 hard negative。
- 裂缝 ≠ 提醒。裂缝是纹理/粗糙上下文。
- 修补、标线、树影、积水反光不得并进 pothole。
- 训练/验证/测试按整段 session 切分，禁止相邻帧泄漏。

产品枚举仍以规格为准（`concave/convex/rough/step/flat` + `pothole/manhole_cover/...`）。调研里的 depression/protrusion 只是别名，映射进现有字段，不另开一套标签。

## 自采优先

KPI 必须来自同一支架、同一手机、同一车灯、同一速度区间的 session。把新视频放到 `Video/train/inbox/` 后跑 `rpar train-bump`。公开远距坑数据可以帮「提前识别」上界，但不能替代 OPPO RideSet，也不能用来调提醒阈值。
