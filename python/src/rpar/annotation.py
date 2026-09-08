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
                "严重度必须记录当前速度范围和可见性",
                "同一物理对象跨帧属性一致，可见性可逐帧变化",
                "争议样本进入 review 队列",
                "训练/验证/测试按完整会话隔离，禁止相邻帧泄漏",
            ],
            "doc": "docs/ANNOTATION.md",
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


def tracks_to_cvat_xml(tracks_jsonl: Path, out_path: Path) -> Path:
    """CVAT 1.1 polygon export so an external tool can round-trip into project JSON (6.2 / REP-003)."""
    items = []
    if tracks_jsonl.exists():
        for line in tracks_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(json.loads(line))
    by_id: dict[int, list[dict[str, Any]]] = {}
    for row in items:
        tid = int(row.get("track_id") or 0)
        by_id.setdefault(tid, []).append(row)
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<annotations>",
        "  <version>1.1</version>",
        "  <meta><task><name>rpar-session</name></task></meta>",
    ]
    for tid, rows in by_id.items():
        label = str(rows[0].get("semantic_type") or "unknown_anomaly")
        parts.append(f'  <track id="{tid}" label="{_xml(label)}">')
        for row in rows:
            poly = row.get("polygon") or []
            if len(poly) < 3:
                continue
            pts = ";".join(f"{float(p[0]):.1f},{float(p[1]):.1f}" for p in poly)
            frame = int(row.get("source_frame_id") or 0)
            parts.append(
                f'    <polygon frame="{frame}" points="{pts}" outside="0" occluded="0" />'
            )
        parts.append("  </track>")
    parts.append("</annotations>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def _xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")


def hit_test_tracks(tracks: list[dict[str, Any]], x: float, y: float) -> dict[str, Any] | None:
    """REP-002 object click: first polygon containing the point, nearest last."""
    best = None
    best_a = 1e18
    for t in tracks:
        poly = t.get("polygon") or []
        if len(poly) < 3:
            continue
        if _point_in_poly(x, y, poly):
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            area = max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
            if area < best_a:
                best_a = area
                best = t
    return best


def _point_in_poly(x: float, y: float, poly: list) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside
