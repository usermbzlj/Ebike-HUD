# YOLOPv2 sidecar (desktop)

MIT-licensed ONNX from [Kazuhito00/YOLOPv2-ONNX-Sample](https://github.com/Kazuhito00/YOLOPv2-ONNX-Sample) (`v0.0.0` / `YOLOPv2.onnx`), derived from [CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2).

Put `YOLOPv2.onnx` here (~156 MB). **Do not commit the weights.**

```powershell
gh release download v0.0.0 --repo Kazuhito00/YOLOPv2-ONNX-Sample --pattern YOLOPv2.onnx --dir models/yolopv2 --clobber
```

`rpar field-video` uses it as a **road polygon + vehicle occlusion** sidecar on top of the heuristic engine. It is not a pothole/manhole detector and is not the on-device LiteRT model.
