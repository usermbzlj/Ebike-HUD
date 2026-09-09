# bump-world-0.1.0

Desktop **YOLO-World** open-vocabulary detector prompted for `pothole` / `speed bump` / `sunken manhole cover`, then optionally fine-tuned on phone clips dropped in `Video/train/inbox/`.

Product enums stay spec (`pothole`, `speed_bump`, `manhole_cover`). Public RDD/Rome class tables are **not** concatenated into this list.

YOLOPv2 remains drivable-area + vehicles only. Phone sidecar is `rpar export-bump-tflite` → `model.onnx` on Windows (LiteRT export is Linux/macOS) or `model.tflite` when Ultralytics allows it (gitignored). Heuristic pits are replaced only when that graph loads; a missing/broken Interpreter keeps the HUD.

- teacher: `yolov8m-worldv2.pt`
- dataset: ``
- n_clips: 0
- n_train_images: 0
- trained: False
- weights: `model.onnx`

Phone ONNX default `imgsz=640` matches desktop YOLO-World infer. Not Camera2 1080p60 GT. Fine-tune quality tracks how many real 大坑 / 下沉井盖 / 减速带 clips you deliver.
