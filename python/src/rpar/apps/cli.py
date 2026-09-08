"""rpar command line: serve, golden, simulate, report, split, package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpar.annotation import tracks_to_annotation_task, tracks_to_cvat_xml
from rpar.capability import write_capability_report
from rpar.capture import record_simulated_session
from rpar.config import load_config
from rpar.golden import run_acceptance_suite, run_oracle_golden, run_simulator_golden, run_video_file
from rpar.ml.eval import write_eval_bundle
from rpar.ab_compare import write_damping_ab, write_model_ab
from rpar.ml.active import write_active_queue
from rpar.ml.synth_train import write_training_bundle
from rpar.ml.seg_train import write_seg_bundle
from rpar.ml.train import split_sessions, write_model_package, write_run_card
from rpar.report import write_report
from rpar.replay import SessionReplay, scan_time_offset_ms
from rpar.error_cases import harvest_error_cases
from rpar.session import export_bundle, export_split_zip, verify_session
from rpar.share import export_share_bundle
from rpar.simulator import RoadSimulator, SimConfig, write_preview_video


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rpar", description="Road Perception AR desktop tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="open research console")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)

    g = sub.add_parser("golden", help="run synthetic golden regression")
    g.add_argument("--out", default="artifacts/golden")
    g.add_argument("--night", action="store_true")
    g.add_argument("--wet", action="store_true")

    v = sub.add_parser("video", help="run heuristic pipeline on an mp4")
    v.add_argument("path")
    v.add_argument("--out", default="artifacts/video_run")
    v.add_argument("--max-frames", type=int, default=300)
    v.add_argument("--no-yolop", action="store_true")

    fv = sub.add_parser("field-video", help="catalog + run every mp4 in Video/")
    fv.add_argument("--video-dir", default="")
    fv.add_argument("--out", default="artifacts/field_video")
    fv.add_argument("--max-frames", type=int, default=180)
    fv.add_argument("--catalog-only", action="store_true")
    fv.add_argument("--no-yolop", action="store_true", help="skip YOLOPv2 even if models/yolopv2/YOLOPv2.onnx exists")

    sim = sub.add_parser("simulate", help="write a raw synthetic preview mp4")
    sim.add_argument("--out", default="artifacts/sim_raw.mp4")
    sim.add_argument("--night", action="store_true")

    r = sub.add_parser("verify", help="verify a session bundle")
    r.add_argument("session")

    rp = sub.add_parser("report", help="write ride report json/html")
    rp.add_argument("session")
    rp.add_argument("--out", default="artifacts/report.json")

    cap = sub.add_parser("capability", help="write desktop capability stub")
    cap.add_argument("--out", default="artifacts/capability_desktop.json")

    sp = sub.add_parser("split", help="session-isolated train/val/test split")
    sp.add_argument("sessions", nargs="+")
    sp.add_argument("--out", default="artifacts/splits.json")

    pkg = sub.add_parser("package-model", help="write a model package directory")
    pkg.add_argument("--id", default="heuristic-cv-0.1.0")
    pkg.add_argument("--out", default="models/heuristic-cv-0.1.0")

    acc = sub.add_parser("accept", help="oracle + scene-sliced golden acceptance")
    acc.add_argument("--out", default="artifacts/acceptance")

    rec = sub.add_parser("record-session", help="write a full synthetic session bundle")
    rec.add_argument("--out", default="artifacts/sessions")

    off = sub.add_parser("offset-scan", help="SYNC-006 camera/IMU offset scan")
    off.add_argument("session")

    sh = sub.add_parser("share", help="export SHARE_REDACTED bundle")
    sh.add_argument("session")
    sh.add_argument("--out", default="artifacts/share")

    an = sub.add_parser("annotate", help="tracks.jsonl → annotation task")
    an.add_argument("session")
    an.add_argument("--out", default="artifacts/annotation_task.json")

    ev = sub.add_parser("eval", help="oracle + scene matrix + hard-negatives + calibration")
    ev.add_argument("--out", default="artifacts/eval")

    cl = sub.add_parser("clip", help="export a replay clip by frame index")
    cl.add_argument("session")
    cl.add_argument("--start", type=int, default=0)
    cl.add_argument("--end", type=int, default=30)
    cl.add_argument("--out", default="artifacts/clip.mp4")

    cv = sub.add_parser("cvat", help="tracks.jsonl → CVAT 1.1 XML")
    cv.add_argument("session")
    cv.add_argument("--out", default="artifacts/cvat.xml")

    al = sub.add_parser("active-learn", help="uncertain/alert/unknown annotation queue")
    al.add_argument("session")
    al.add_argument("--out", default="artifacts/active_queue.json")

    ab = sub.add_parser("damping-ab", help="CAL-006 compare two sessions")
    ab.add_argument("session_a")
    ab.add_argument("session_b")
    ab.add_argument("--out", default="artifacts/damping_ab.json")

    mab = sub.add_parser("model-ab", help="MOD-003 same-input engine A/B")
    mab.add_argument("--out", default="artifacts/model_ab.json")

    tr = sub.add_parser("train-synth", help="session-isolated linear trainer + precision cards")
    tr.add_argument("--out", default="artifacts/train_synth")
    tr.add_argument("--export-tflite", action="store_true", help="write a real TFLite graph when TensorFlow is installed")

    sg = sub.add_parser("train-seg", help="dual-scale classmap pixel trainer on synthetic GT")
    sg.add_argument("--out", default="artifacts/train_seg")
    sg.add_argument("--export-tflite", action="store_true")
    sg.add_argument("--package", default="", help="optional model package directory")

    ex = sub.add_parser("export", help="zip a session; optional size-capped volumes")
    ex.add_argument("session")
    ex.add_argument("--out", default="artifacts/export")
    ex.add_argument("--split-mb", type=float, default=0.0, help="if >0, emit .partNN.zip volumes")

    el = sub.add_parser("error-lib", help="harvest fired alerts and marks into an error-case library")
    el.add_argument("session")
    el.add_argument("--out", default="artifacts/error_lib")

    df = sub.add_parser("distill-field", help="distill YOLOPv2 teacher masks from Video/ into dual-scale classmap")
    df.add_argument("--out", default="models/roadseg-field-0.1.0")
    df.add_argument("--frames-per-clip", type=int, default=12)
    df.add_argument("--package-only", action="store_true", help="write manifest around existing seg_weights.json")

    ml = sub.add_parser("map-label", help="map a public dataset class onto RPAR semantic/geometry/state")
    ml.add_argument("source")
    ml.add_argument("raw")
    ml.add_argument("--severity", default="")

    ia = sub.add_parser("impact-align", help="M5: align confirmed tracks to future IMU (never alerts)")
    ia.add_argument("session")
    ia.add_argument("--out", default="artifacts/impact_align.json")
    ia.add_argument("--horizon", type=float, default=3.0)

    args = p.parse_args(argv)
    if args.cmd == "serve":
        from rpar.apps.server import main as serve_main

        serve_main(args.host, args.port)
        return 0
    if args.cmd == "golden":
        sc = SimConfig(duration_s=3.5, blur_windows=[(1.2, 1.55)], night=args.night, wet=args.wet)
        metrics = run_simulator_golden(Path(args.out), load_config(), sc)
        print(json.dumps(metrics, indent=2))
        return 0
    if args.cmd == "video":
        print(json.dumps(run_video_file(Path(args.path), Path(args.out), max_frames=args.max_frames, prefer_yolop=not args.no_yolop), indent=2))
        return 0
    if args.cmd == "field-video":
        from rpar.field_video import run_field_videos, write_catalog

        vdir = Path(args.video_dir) if args.video_dir else None
        if args.catalog_only:
            print(json.dumps(write_catalog(vdir), indent=2, ensure_ascii=False))
            return 0
        print(json.dumps(run_field_videos(vdir, Path(args.out), max_frames=args.max_frames, prefer_yolop=not args.no_yolop), indent=2, ensure_ascii=False)[:8000])
        return 0
    if args.cmd == "simulate":
        simu = RoadSimulator(SimConfig(night=args.night, duration_s=3.0))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_preview_video(args.out, simu)
        print(args.out)
        return 0
    if args.cmd == "verify":
        print(json.dumps(verify_session(Path(args.session)), indent=2))
        return 0
    if args.cmd == "report":
        out = Path(args.out)
        write_report(Path(args.session), out, out.with_suffix(".html"))
        print(out)
        return 0
    if args.cmd == "capability":
        write_capability_report(Path(args.out))
        print(args.out)
        return 0
    if args.cmd == "split":
        man = split_sessions(args.sessions)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(man.to_dict(), indent=2), encoding="utf-8")
        write_run_card(Path(args.out).with_name("run_card.json"), seed=man.seed)
        print(args.out)
        return 0
    if args.cmd == "package-model":
        write_model_package(Path(args.out), args.id)
        print(args.out)
        return 0
    if args.cmd == "accept":
        oracle = run_oracle_golden(Path(args.out) / "oracle", load_config())
        scenes = run_acceptance_suite(Path(args.out) / "scenes", load_config())
        print(json.dumps({"oracle": oracle, "scenes": scenes}, indent=2))
        return 0
    if args.cmd == "record-session":
        path = record_simulated_session(Path(args.out))
        print(path)
        return 0
    if args.cmd == "offset-scan":
        print(json.dumps(scan_time_offset_ms(Path(args.session)), indent=2))
        return 0
    if args.cmd == "share":
        dest = Path(args.out)
        print(json.dumps(export_share_bundle(Path(args.session), dest), indent=2))
        return 0
    if args.cmd == "annotate":
        src = Path(args.session) / "perception" / "tracks.jsonl"
        tracks_to_annotation_task(src, Path(args.out))
        print(args.out)
        return 0
    if args.cmd == "eval":
        print(json.dumps(write_eval_bundle(Path(args.out), load_config()), indent=2))
        return 0
    if args.cmd == "clip":
        rep = SessionReplay(Path(args.session))
        rep.open()
        path = rep.export_clip(Path(args.out), args.start, args.end)
        rep.close()
        print(path)
        return 0
    if args.cmd == "cvat":
        src = Path(args.session) / "perception" / "tracks.jsonl"
        print(tracks_to_cvat_xml(src, Path(args.out)))
        return 0
    if args.cmd == "active-learn":
        print(json.dumps(write_active_queue(Path(args.session), Path(args.out)), indent=2))
        return 0
    if args.cmd == "damping-ab":
        print(json.dumps(write_damping_ab(Path(args.session_a), Path(args.session_b), Path(args.out)), indent=2))
        return 0
    if args.cmd == "model-ab":
        print(json.dumps(write_model_ab(Path(args.out), load_config()), indent=2))
        return 0
    if args.cmd == "train-synth":
        print(json.dumps(write_training_bundle(Path(args.out), export_tflite=args.export_tflite), indent=2)[:2000])
        return 0
    if args.cmd == "train-seg":
        print(json.dumps(write_seg_bundle(Path(args.out), export_tflite=args.export_tflite, package_dir=Path(args.package) if args.package else None), indent=2)[:4000])
        return 0
    if args.cmd == "distill-field":
        from rpar.ml.field_distill import distill_from_videos, write_field_package

        if args.package_only:
            print(json.dumps(write_field_package(Path(args.out)), indent=2))
            return 0
        from rpar.field_video import list_clips, repo_video_dir

        clips = [Path(c["path"]) for c in list_clips(repo_video_dir()) if c.get("ok")]
        print(json.dumps(distill_from_videos(clips, Path(args.out), frames_per_clip=args.frames_per_clip), indent=2))
        return 0
    if args.cmd == "map-label":
        from rpar.ml.labelmap import map_record

        print(json.dumps(map_record(args.source, args.raw, args.severity or None), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "impact-align":
        from rpar.impact import write_impact_report

        print(json.dumps(write_impact_report(Path(args.session), Path(args.out), horizon_s=args.horizon), indent=2, ensure_ascii=False)[:8000])
        return 0
    if args.cmd == "export":
        src = Path(args.session)
        dest = Path(args.out)
        if args.split_mb and args.split_mb > 0:
            parts = export_split_zip(src, dest, max_bytes=int(args.split_mb * 1024 * 1024))
            print(json.dumps([str(p) for p in parts], indent=2))
        else:
            print(export_bundle(src, dest))
        return 0
    if args.cmd == "error-lib":
        print(json.dumps(harvest_error_cases(Path(args.session), Path(args.out)), indent=2, ensure_ascii=False)[:4000])
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
