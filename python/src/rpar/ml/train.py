"""Training / export scaffolding (ML-001..010). Real weights are versioned separately."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rpar import SCHEMA_VERSION


@dataclass
class SplitManifest:
    schema_version: str
    created_at: str
    train_sessions: list[str]
    val_sessions: list[str]
    test_sessions: list[str]
    isolation: str = "session_date_route"
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "train_sessions": self.train_sessions,
            "val_sessions": self.val_sessions,
            "test_sessions": self.test_sessions,
            "isolation": self.isolation,
            "seed": self.seed,
        }


def split_sessions(
    session_ids: list[str],
    seed: int = 7,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> SplitManifest:
    ids = list(session_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return SplitManifest(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        train_sessions=train,
        val_sessions=val,
        test_sessions=test,
        seed=seed,
    )


def write_run_card(out: Path, **kwargs: Any) -> Path:
    card = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_version": kwargs.get("data_version"),
        "git_commit": kwargs.get("git_commit"),
        "seed": kwargs.get("seed", 0),
        "hyperparameters": kwargs.get("hyperparameters", {}),
        "label_map": kwargs.get("label_map", {}),
        "dependency_lock": kwargs.get("dependency_lock", {}),
        "scene_slices": ["day", "night", "wet", "backlight", "follow", "glare", "rough", "rain", "vibration"],
        "hard_negative_slices": [
            "tree_shadow",
            "patch",
            "marking",
            "reflection",
            "manhole_normal",
            "vehicle_shadow",
        ],
        "notes": "V0.1 ships a heuristic engine; this card is the contract for later PyTorch runs.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return out


def write_model_package(out_dir: Path, package_id: str, engine: str = "heuristic") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "semantic_type": [
            "pothole",
            "manhole_cover",
            "speed_bump",
            "road_joint",
            "repair_patch",
            "rough_broken",
            "puddle",
            "gravel",
            "unknown_anomaly",
        ],
        "geometry_type": ["concave", "convex", "rough", "step", "flat", "unknown"],
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "package_id": package_id,
        "engine": engine,
        "files": {"model.tflite": "model.tflite", "labels.json": "labels.json"},
        "input_spec": {
            "far": {"width": 768, "height": 384, "layout": "RGB", "norm": "imagenet"},
            "near": {"width": 640, "height": 480, "layout": "RGB", "norm": "imagenet"},
            "note": "Must not use a single 640x640 full-frame as the only input.",
        },
        "quantization": "none-heuristic",
        "compatible_app": ">=0.1.0",
        "labels": labels,
        "sha256": {},
    }
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    # MOD-001: package always contains a .tflite file. Heuristic engine ignores weights.
    tflite = out_dir / "model.tflite"
    if not tflite.exists():
        tflite.write_bytes(b"TFL3" + b"\x00heuristic-cv-placeholder\x00" + bytes(64))
    (out_dir / "MODEL_CARD.md").write_text(
        f"# {package_id}\n\nHeuristic dual-scale CV engine used until a calibrated LiteRT package is sideloaded.\n"
        "Android ModelManager refuses load if SHA-256 / app compatibility fail, then rolls back.\n",
        encoding="utf-8",
    )
    for p in out_dir.iterdir():
        if p.is_file() and p.name != "manifest.json":
            manifest["sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir
