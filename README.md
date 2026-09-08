# Road Perception AR（路面感知增强现实系统）

面向 **OPPO Find X8 Pro（PKC110）** 的 V0.1 原型：手机端实时感知闭环 + 桌面数据 / 回放 / 回归闭环。

本仓库按《Road Perception AR 开发需求规格说明书 V1.0》实现。系统是**实验性辅助感知工具**，不替代骑手观察，不输出转向、避让或制动指令。

## 现在就能用的东西

| 产物 | 说明 |
| --- | --- |
| 桌面研究控制台 | `python -m rpar.apps.cli serve` → http://127.0.0.1:8765 |
| 黄金视频回归 | `python -m rpar.apps.cli golden` 合成 1080p 道路、跑完整管线、输出 overlay + metrics |
| Android 应用 | `android/`，`assembleDebug` 已在本机通过 |
| 会话 schema | `session_<id>/` 目录 + `checksums.sha256` |
| 可替换模型包 | `models/heuristic-cv-0.1.0/`，Android `ModelManager` 校验 SHA-256 后加载 |

默认感知后端是 **双尺度启发式 CV**（远 ROI + 近 ROI），满足 PER-003「不得把整帧压成唯一 640×640」。LiteRT 模型可按同样 `RoadObservation` 接口侧载替换。

## 架构

```
手机实时闭环                         桌面数据闭环
Camera2 1080p60 + IMU/GNSS          会话导入 / QC
    → 质量门控与最近清晰帧              → 抽样 / 标注 / hard negative
    → 道路+遮挡+异常双尺度推理           → 按 session 隔离训练评估
    → 跟踪 / 距离 / TTC / 左中右        → 量化打包 manifest
    → AR 30–60 fps + 保守提醒           → 回灌 / A/B / 回滚
```

模块划分与规格 3.1 一致：`camera-core`、`sensor-core`、`sync-engine`、`quality-engine`、`perception-api`、`tracking-engine`、`geometry-engine`、`ar-render`、`alert-policy`、`recorder`、`diagnostics`。

## 快速开始（桌面）

需要 Python 3.11+（仓库 `.venv` 已用 3.12）：

```powershell
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m rpar.apps.cli serve --port 8765
```

浏览器打开 `http://127.0.0.1:8765`：

- **骑行/研究**：同一套结构化感知结果，信息密度不同
- **能力探测**：CAP-001 JSON 契约（真值必须在 PKC110 运行时探测）
- **安装标定**：左侧车把 `left_handlebar_v1` 配置档
- **回放/验收**：录制合成会话、逐帧/变速回放、事件跳转、片段导出、时间偏移扫描、黄金验收、分享脱敏

其它命令：

```text
rpar golden --out artifacts/golden
rpar simulate --out artifacts/sim_raw.mp4
rpar accept --out artifacts/acceptance
rpar record-session --out artifacts/sessions
rpar verify <session_dir>
rpar report <session_dir>
rpar share <session_dir> --out artifacts/share
rpar offset-scan <session_dir>
rpar eval --out artifacts/eval
rpar annotate <session_dir>
rpar split session_a session_b session_c
rpar package-model --id heuristic-cv-0.1.0
```

跑测试：

```powershell
$env:PYTHONPATH = "python\src"
.venv\Scripts\python.exe -m pytest tests -q
```

## 快速开始（Android / PKC110）

见 [android/README.md](android/README.md)。摘要：

```powershell
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-17.0.20.101-hotspot"
cd android
.\gradlew.bat assembleDebug
```

APK：`android/app/build/outputs/apk/debug/app-debug.apk`

真机上请先走 **能力探测 → 安装标定 → 安全模式采集 5 分钟 → 再开实时感知**。未选择有效安装配置时应用会隐藏精确米数。

## 仓库结构

```text
android/          Kotlin · Camera2 · Compose HUD · OpenGL AR · 会话录制
python/src/rpar/  桌面管线、启发式感知、回放控制台、训练脚手架
configs/          版本化参数（MOD-005，禁止把阈值写死在 UI）
schemas/          TrackedRoadObject JSON Schema
proto/            高频 IMU / FrameMeta 逻辑 schema
models/           可侧载模型包
docs/             用户说明、标定、已知限制、测试计划
tests/            坐标、同步、状态机、提醒门控、会话校验、黄金回归
```

## 安全与隐私

- 原始视频和位置默认只保存在本机（`privacy_mode: LOCAL_ONLY`）
- 没有自动上传路径
- 导出会话必须经过确认页
- 声音提醒可关；AR 不依赖提醒
- 文案检查禁止「向左避让 / 向右转向 / 制动」等操控建议

## 文档

- [用户操作说明](docs/USER_GUIDE.md)
- [安装与标定](docs/CALIBRATION.md)
- [已知限制与风险](docs/KNOWN_LIMITATIONS.md)
- [测试计划与验收](docs/TEST_PLAN.md)
- [发布说明](docs/RELEASE_NOTES.md)
