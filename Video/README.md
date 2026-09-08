# 实拍视频（本地）

把规格附录或场测 mp4 放在本目录。文件**不会**提交到 Git（见根目录 `.gitignore`）。

当前按时长对齐规格 §1.2：

| 别名 | 规格文件 | 时长指纹 |
| --- | --- | --- |
| `day_25007` | `25007.mp4` | ~68.824 s 白天 |
| `night_25013` | `25013.mp4` | ~55.014 s 有路灯夜间 |

其它 mp4 记为 `extra`。编目：

```text
python -m rpar.apps.cli field-video --catalog-only
python -m rpar.apps.cli field-video --out artifacts/field_video --max-frames 90
```

若本地有 `models/yolopv2/YOLOPv2.onnx`，默认走启发式 + YOLOPv2 混合（绿=可行驶，橙=车辆）。`--no-yolop` 可关掉。权重约 156 MB，不进 Git。

`catalog.json` 可提交：记录分辨率、帧率、亮度与 SHA 前缀，不含画面。
