# bump-world-0.1.0

Desktop **YOLO-World** open-vocabulary detector prompted for `pothole` / `speed bump` / `sunken manhole cover`, then optionally fine-tuned on phone clips dropped in `Video/train/inbox/`.

Product enums stay spec (`pothole`, `speed_bump`, `manhole_cover`). Public RDD/Rome class tables are **not** concatenated into this list.

YOLOPv2 remains drivable-area + vehicles only. This package is **not** a PKC110 LiteRT bump net; the phone keeps the heuristic until a quantized student is exported.

Weights (`best.pt`, teacher `.pt`) stay gitignored. After you copy clips into `Video/train/inbox/`:

```text
rpar fetch-bump-model
rpar train-bump
```

Not Camera2 1080p60 GT. Fine-tune quality tracks how many real 大坑 / 下沉井盖 / 减速带 clips you deliver.
