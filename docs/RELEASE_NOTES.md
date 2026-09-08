# Road Perception AR V0.1 发布说明

- 版本：0.1.0（对应规格 V0.1–V0.3 原型中的 M0–M3 软件基线）
- 应用 ID：`com.ebike.rpar`
- 配置 schema：`1.0`

## 本包包含

- Android 源码、Gradle 包装与 debug 构建说明
- 桌面回放控制台、会话校验、黄金回归、训练划分与模型包清单工具
- 版本化配置 `configs/rpar.defaults.yaml`
- 启发式双尺度感知引擎（可被 LiteRT 包替换）
- 用户说明、标定指南、已知限制、测试计划

## 签名

Debug 构建使用 Android 默认 debug 密钥，仅供开发安装。场测 APK 需使用团队自己的 upload key，并在此文件补充指纹。不要把密钥提交进 Git。

## 变更摘要

- M0：能力探测 JSON 契约 + 运行时探测代码（Camera2 / 传感器 / 可选 ARCore）
- M1：同步采集器、5 分钟分段录像、前台服务、会话 checksum
- M2：离线/桌面感知 AR、跟踪、距离、左中右、黄金视频
- M3：手机实时 MVP（启发式后端）、骑行/研究 UI、模型管理与安全模式
- M4 策略代码已落地（高阈值提醒、冷却、决策快照）；10 h 误报统计仍需实路

## 安装

见 `android/README.md`。首次运行允许相机与（可选）位置权限；无麦克风也可工作。
