"""rpar command line: serve, golden, simulate, report, split, package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpar.capability import write_capability_report
from rpar.config import load_config
from rpar.golden import run_simulator_golden, run_video_file
from rpar.ml.train import split_sessions, write_model_package, write_run_card
from rpar.report import write_report
from rpar.session import verify_session
from rpar.simulator import RoadSimulator, SimConfig, write_preview_video


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rpar", description="Road Perception AR desktop tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="open research console")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)

    g = sub.add_parser("golden", help="run synthetic golden regression")
    g.add_argument("--out", default="artifacts/golden")

    v = sub.add_parser("video", help="run heuristic pipeline on an mp4")
    v.add_argument("path")
    v.add_argument("--out", default="artifacts/video_run")
    v.add_argument("--max-frames", type=int, default=300)

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

    args = p.parse_args(argv)
    if args.cmd == "serve":
        from rpar.apps.server import main as serve_main

        serve_main(args.host, args.port)
        return 0
    if args.cmd == "golden":
        metrics = run_simulator_golden(Path(args.out), load_config())
        print(json.dumps(metrics, indent=2))
        return 0
    if args.cmd == "video":
        print(json.dumps(run_video_file(Path(args.path), Path(args.out)), indent=2))
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
