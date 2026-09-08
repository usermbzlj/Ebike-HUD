"""Annotation conversion: project JSONL tracks <-> CVAT-like polygon tasks (REP-003, 6.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rpar.enums import GeometryType, ObjectState, SemanticType, Severity, VisibilityClass


REQUIRED_FIELDS = [
    "semantic_type",
    "geometry_type",
    "state",
    "severity",
    "visibility",
    "track_id",
]


def tracks_to_annotation_task(tracks_jsonl: Path, out_path: Path) -> dict[str, Any]:
    items = []
    if tracks_jsonl.exists():
        for line in tracks_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(
                {
                    "frame_id": row.get("source_frame_id"),
                    "timestamp_ns": row.get("timestamp_ns"),
                    "track_id": row.get("track_id"),
                    "semantic_type": row.get("semantic_type"),
                    "geometry_type": row.get("geometry_type"),
                    "state": row.get("object_state") or row.get("state"),
                    "severity": row.get("severity"),
                    "visibility": "clear",
                    "polygon": row.get("polygon"),
                    "model_prediction": row,
                    "human_revision": None,
                    "review": False,
                }
            )
    task = {
        "schema_version": "1.0",
        "label_guide": {
            "semantic_type": [s.value for s in SemanticType],
            "geometry_type": [g.value for g in GeometryType],
            "state": [s.value for s in ObjectState],
            "severity": [int(s) for s in Severity],
            "visibility": [v.value for v in VisibilityClass],
            "rules": [
                "先标道路和遮挡，再标路面异常",
                "不可观测区域不得猜测形状",
                "井盖必须 semantic_type=manhole_cover，是否异常由 geometry/state 决定",
                "平整修补标 repair_patch + flat + normal（hard negative）",
                "无法判断凹凸时标 unknown_anomaly + unknown",
            ],
        },
        "items": items,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    return task


def apply_revision(task_path: Path, track_id: int, frame_id: int, patch: dict[str, Any]) -> None:
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    for item in task["items"]:
        if item.get("track_id") == track_id and item.get("frame_id") == frame_id:
            item["human_revision"] = {**(item.get("human_revision") or {}), **patch}
    Path(task_path).write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
