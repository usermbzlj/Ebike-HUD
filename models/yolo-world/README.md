# YOLO-World teacher (desktop)

Open-vocabulary detector used as the **pothole / speed bump / sunken manhole** teacher.

Put `yolov8m-worldv2.pt` here. **Do not commit the weights.**

```powershell
$env:PYTHONPATH = "python\src"
.\.venv\Scripts\python.exe -m rpar.apps.cli fetch-bump-model
```

This is not YOLOPv2. YOLOPv2 stays drivable-area + vehicles only. Phone stays heuristic until a quantized student is exported.
