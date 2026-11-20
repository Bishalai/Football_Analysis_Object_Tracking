"""Command-line entry point for the football tracking pipeline.

Examples
--------
    python run_tracking.py "test_vid.mp4"
    python run_tracking.py "test_vid.mp4" --out output --max-frames 300
    python run_tracking.py "test_vid.mp4" --tracker botsort --reid osnet --no-video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.config import DetectionConfig, PipelineConfig, TrackerConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect, track and re-identify football players across "
                    "out-of-frame gaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", type=Path, help="input video file")
    p.add_argument("--out", type=Path, default=Path("output"), help="output directory")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--conf", type=float, default=None, help="detection confidence")
    p.add_argument("--imgsz", type=int, default=None, help="detection input size")
    p.add_argument("--tracker", type=str, default=None,
                   help="boxmot tracker: boosttrack, botsort, bytetrack, ...")
    p.add_argument("--max-age", type=int, default=None,
                   help="frames a lost stage-2 track stays alive")
    p.add_argument("--reid", type=str, default=None, choices=["deepsort", "osnet", "resnet18"],
                   help="appearance embedding backend")
    p.add_argument("--warmup", type=int, default=None,
                   help="frames before a new track may be re-identified")
    p.add_argument("--no-video", action="store_true", help="skip annotated video export")
    p.add_argument("--no-viz", action="store_true",
                   help="skip the analytics charts in <out>/<stem>/viz/")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.video.exists():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 2

    cfg = PipelineConfig()
    if args.conf is not None:
        cfg.detection = DetectionConfig(**{**vars(cfg.detection), "conf": args.conf})
    if args.imgsz is not None:
        cfg.detection = DetectionConfig(**{**vars(cfg.detection), "imgsz": args.imgsz})
    if args.tracker is not None or args.max_age is not None:
        tvars = vars(cfg.tracker).copy()
        if args.tracker is not None:
            tvars["spec"] = args.tracker
        if args.max_age is not None:
            tvars["max_age"] = args.max_age
        cfg.tracker = TrackerConfig(**tvars)
    if args.reid is not None:
        cfg.reid.backend = args.reid
    if args.warmup is not None:
        cfg.gallery.warmup_frames = args.warmup

    from pipeline.runner import Runner

    print(f"model : {cfg.paths.players_weights}")
    print(f"video : {args.video}")
    print(f"tracker={cfg.tracker.spec} reid={cfg.reid.backend} "
          f"conf={cfg.detection.conf} warmup={cfg.gallery.warmup_frames}")

    runner = Runner(cfg, annotate=not args.no_video)
    print(f"classes: {runner.detector.class_summary}")

    stats = runner.run(args.video, out_dir=args.out,
                       max_frames=args.max_frames, progress=not args.quiet,
                       make_viz=not args.no_viz)

    summary = runner.tracker.gallery.summary()
    print()
    print(f"processed {stats.frames} frames in {stats.elapsed_s:.1f}s "
          f"({stats.fps:.1f} fps)")
    print(f"identities: {summary}")
    print(f"outputs in: {stats.output_dir}")
    if stats.video_path:
        print(f"  video : {stats.video_path}")
    if stats.csv_path:
        print(f"  csv   : {stats.csv_path}")
    if stats.viz_files:
        print(f"  charts: {len(stats.viz_files)} pngs in {stats.viz_files[0].parent}")
    rosters = runner.tracker.gallery.team_rosters()
    for team in sorted(rosters, key=lambda t: (t is None, t)):
        print(f"  team {team}: {sorted(rosters[team])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
