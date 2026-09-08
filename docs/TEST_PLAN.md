# 测试计划与验收对照

自动化（本仓库 `pytest`）：

- 坐标变换：横屏投影、左把中线补偿、地面往返
- 时间插值、CaptureResult 匹配、IMU 对齐 p95 ≤ 10 ms
- 跟踪状态机、模糊期不创建高置信新目标；遮挡多边形覆盖道路时也不新建确认实例
- 提醒：正常井盖不播报、走廊外不播报、每 ID 一次、冷却、无操控文案
- 会话清单与 SHA-256；崩溃分段 `.part` 可标记恢复；会话写入 `perception/impact_align.json`（不进提醒）
- 黄金合成视频：overlay 写出、管线不崩溃
- Oracle 几何：方向 ≥95%、5–15 m MAE ≤2.5 m
- 会话录制 + 回放 + SYNC-006 偏移扫描 + SHARE_REDACTED 导出
- Hard-negative 切片不产生语音提醒；模型包含 SHA-256 与占位 tflite
- CAM-009 变换链叠加误差 ≤8 px；CVAT 导出；主动学习队列；减震 A/B 统计；双尺度线性训练卡片
- classmap → 观测解码；道路/遮挡 jsonl；错误案例库；雨天/振动场景切片；512 MB 分卷导出
- CAP-003：16.67 ms 最小帧时长选出 1080p60；YUV 间隔中位数折算实测 FPS；并发组合候选写入探测 JSON
- CAM-012：约 4 g 残差或剧烈角速度触发紧急停止；CAP-004 最近 60 s 采样率窗
- 前台通知不得暗示路面安全；`rpar verify` 扫描会话加速度 jsonl 作 CAM-012 诊断
- M2 夜间允许信息层积水确认，固体病害/`rough_broken` 仍须为 0

黄金回归命令：

```text
python -m rpar.apps.cli golden --out artifacts/golden
python -m rpar.apps.cli field-video --out artifacts/field_video --max-frames 90
```

合成黄金输出 `overlay.mp4` 与 `metrics.json`（方向准确率、距离 MAE、首次确认距离、模糊期幽灵框代理指标）。`field-video` 扫描仓库 `Video/` 里的实拍 mp4（按时长对齐规格 25007 白天 / 25013 夜间），写出 overlay、每段 `metrics.json` 和附录 B 风格的 `m2_report.json`（夜间确认实例/提醒/`rough_broken` 必须为 0；白天要有道路多边形）。无几何 GT 时不能当作首次确认距离门槛的正式证据。夜间桥面片段不得把纹理噪声标成 `rough_broken`（见 `tests/test_heuristic_precision.py`）。M5 视觉→未来 IMU 对齐：`rpar impact-align session_...`；冲击分数不得进入提醒。

## V0.1 发布门槛（需 PKC110 实机补齐的证据）

| 项 | 最低门槛 |
| --- | --- |
| 白天中大型异常首次稳定识别 | 中位数 ≥ 15 m |
| 有路灯夜间 | 中位数 ≥ 12 m |
| 可行动区召回 | 白天 ≥85%；夜间 ≥75%；湿路 ≥70% |
| 稳定实线误标 | 白天 ≤0.5/min；夜间 ≤1.0/min |
| 误语音提醒 | ≥10 h 累计 ≤0.5/h |
| 方向 | ≥95%（道路坐标，非屏幕三等分） |
| 距离 | 5–15 m MAE ≤2.5 m；15–30 m MAE ≤5 m |
| 跟踪续接 | ≤0.5 s 模糊后恢复且无新增幽灵目标 |
| 性能 | p95 ≤150 ms；稳态 ≥10 fps；60 min 无崩溃/热关停 |
| 离线 | 飞行模式完成识别 / AR / 提醒 / 记录 / 回放 |

场景矩阵见规格 11.3：白天、逆光、夜间路灯、低照、湿路、小雨水滴、高频振动、前车遮挡、转向、热长时、存储不足/模型损坏。

## 回归策略

每次相机管线或模型变化：固定白天/夜间/湿路黄金视频 + 本仓库合成回归 + 实机 10 min 后端对比（延迟、热、功耗）。
