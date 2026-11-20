"""Video loop: detect -> track -> re-identify -> export CSV + annotated video.

    from pipeline.runner import Runner
    Runner(cfg).run("test_vid.mp4", out_dir="output")
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .assets import ensure_assets
from .config import PipelineConfig
from .detector import BallDetector, KeypointDetector, PlayerDetector
from .homography import PitchProjector
from .team import TeamClassifier
from .tracker import FrameTracks, Stage2Tracker

# Colours are assigned per persistent id so a player keeps one colour all video.
_PALETTE = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (124, 179, 66),
    (63, 81, 181), (233, 30, 99), (0, 150, 136), (255, 193, 7),
]


@dataclass
class RunStats:
    frames: int = 0
    elapsed_s: float = 0.0
    rows: list[dict] = field(default_factory=list)
    # Everything for one input video lives in its own folder:
    #   <out_dir>/<video stem>/<stem>_tracked.mp4
    #   <out_dir>/<video stem>/<stem>_tracks.csv
    #   <out_dir>/<video stem>/viz/*.png
    output_dir: Path | None = None
    video_path: Path | None = None
    csv_path: Path | None = None
    viz_files: list[Path] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s else 0.0


class Runner:
    def __init__(self, cfg: PipelineConfig | None = None, annotate: bool = True):
        self.cfg = cfg or PipelineConfig()
        self.annotate = annotate

        # Weights are not stored in git: fetch whatever is missing from
        # Hugging Face before any model is opened. Only the files this
        # configuration actually needs are requested (the sample clip and the
        # optional ball detector are fetched by their own call sites).
        needed = ["models/players_v8_best.pt", "models/keypoints_v8m_best.pt"]
        if self.cfg.detection.use_ball_model:
            needed.append("models/ball_v8_best.pt")
        if self.cfg.reid.backend == "deepsort":
            needed.append("models/deepsort_ckpt.t7")
        elif self.cfg.reid.backend == "osnet":
            needed.append("models/osnet_x0_25_msmt17.pt")
        ensure_assets(needed)

        self.detector = PlayerDetector(self.cfg.detection, self.cfg.paths)
        self.projector = PitchProjector(self.cfg.paths)
        self.teams = TeamClassifier(self.cfg.team)
        self.tracker = Stage2Tracker(
            self.cfg.tracker, self.cfg.gallery, self.cfg.reid, self.cfg.paths,
        )

        self._keypoints: KeypointDetector | None = None
        self._ball: BallDetector | None = None
        self._ball_failed = False
        self._team_by_gid: dict[int, int] = {}
        self._color_by_gid: dict[int, tuple[int, int, int]] = {}

    def reset(self) -> None:
        """Clear all per-video state so the Runner can process another video."""
        from .gallery import IdentityGallery
        from .team import TeamClassifier

        self.teams = TeamClassifier(self.cfg.team)
        self.tracker.gallery = IdentityGallery(self.cfg.gallery)
        self.tracker.frame_idx = -1
        self.tracker._tracker = None
        self.projector.reset()
        self._team_by_gid.clear()
        self._color_by_gid.clear()
        self._ball = None
        self._ball_failed = False
        # Keep the loaded models (_keypoints, detector) -- reloading them is
        # the expensive part and they are video-agnostic.

    # ---------------- public API ---------------- #
    def run(self, video_path: str | Path, out_dir: str | Path = "output",
            max_frames: int | None = None, progress: bool = True,
            on_progress: Callable[[int, int, float], None] | None = None,
            make_viz: bool = True) -> RunStats:
        """Run the pipeline over a video.

        Outputs go into a folder specific to this input video,
        ``<out_dir>/<video stem>/`` (annotated video, tracks CSV, ``viz/``).

        Parameters
        ----------
        on_progress:
            Optional callback ``(frames_done, total_frames, fps)`` invoked every
            frame. Used by the Streamlit UI to drive a live progress bar; the
            CLI uses the `progress` print instead.
        make_viz:
            Generate the analytics charts (heatmap, possession, pass network,
            ...) into ``<out_dir>/<stem>/viz/`` after the run.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # One folder per input video, so several runs never overwrite
        # each other's outputs.
        out_dir = Path(out_dir) / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        writer = None
        video_out = None
        if self.annotate:
            video_out = out_dir / f"{video_path.stem}_tracked.mp4"
            # H.264 ("avc1") is what browsers and phone players can actually
            # decode; the old "mp4v" (MPEG-4 Part 2) file was downloadable but
            # not playable in the UI. Fall back to mp4v if this build of
            # OpenCV has no H.264 encoder.
            for fourcc in ("avc1", "mp4v"):
                candidate = cv2.VideoWriter(
                    str(video_out), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h),
                )
                if candidate.isOpened():
                    writer = candidate
                    if fourcc != "avc1":
                        print("  [warn] H.264 unavailable, falling back to mp4v "
                              "(may not play in browsers)")
                    break
                candidate.release()
            if writer is None:
                print("  [warn] could not open a video writer, skipping export")

        stats = RunStats(output_dir=out_dir)
        t_start = time.perf_counter()
        frame_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break

                tracks = self._process_frame(frame, frame_idx)
                stats.rows.extend(self._rows(tracks))
                stats.frames += 1

                if writer is not None:
                    writer.write(self._draw(frame, tracks))

                frame_idx += 1
                elapsed = time.perf_counter() - t_start
                current_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                if on_progress is not None:
                    on_progress(frame_idx, total, current_fps)
                elif progress and frame_idx % 50 == 0:
                    print(f"  frame {frame_idx:5d} | {current_fps:5.1f} fps | "
                          f"identities {self.tracker.gallery.summary()['total_identities']}")
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        stats.elapsed_s = time.perf_counter() - t_start
        stats.video_path = video_out
        stats.csv_path = out_dir / f"{video_path.stem}_tracks.csv"
        self._write_csv(stats.rows, stats.csv_path)

        # Analytics charts for this video -> <out_dir>/<stem>/viz/*.png
        if make_viz and stats.csv_path.exists() and stats.rows:
            from .analytics import generate_viz

            stats.viz_files = generate_viz(
                stats.csv_path,
                out_dir / "viz",
                cfg=self.cfg.analytics,
                paths=self.cfg.paths,
                fps=fps,
                title=video_path.stem,
            )
            if progress and stats.viz_files:
                print(f"  wrote {len(stats.viz_files)} charts -> {out_dir / 'viz'}")
        return stats

    # ---------------- per-frame ---------------- #
    def _ball_on_pitch(self, dets) -> np.ndarray:
        """Mask of ball detections that land on the pitch (not the stands)."""
        mask = dets.is_ball.copy()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return mask

        centres = np.array([
            [(dets.xyxy[i][0] + dets.xyxy[i][2]) / 2,
             (dets.xyxy[i][1] + dets.xyxy[i][3]) / 2]
            for i in idx
        ], dtype=np.float32)
        proj = self.projector.transform_points(centres)
        if proj is None:
            return mask

        # Pitch landmark extents, with a generous margin for the run-off area.
        xs = [p[0] for p in self.projector.map_positions.values()]
        ys = [p[1] for p in self.projector.map_positions.values()]
        margin = 0.15 * max(max(xs) - min(xs), max(ys) - min(ys))
        lo_x, hi_x = min(xs) - margin, max(xs) + margin
        lo_y, hi_y = min(ys) - margin, max(ys) + margin

        on_pitch = (
            (proj[:, 0] >= lo_x) & (proj[:, 0] <= hi_x)
            & (proj[:, 1] >= lo_y) & (proj[:, 1] <= hi_y)
        )
        for i, ok in zip(idx, on_pitch):
            if not ok:
                mask[i] = False
        return mask

    def _process_frame(self, frame: np.ndarray, frame_idx: int):
        dets = self.detector(frame)

        # Dedicated ball model overrides the players model's ball class.
        if self.cfg.detection.use_ball_model and not self._ball_failed:
            if self._ball is None:
                try:
                    self._ball = BallDetector(self.cfg.detection, self.cfg.paths)
                except FileNotFoundError as exc:
                    print(f"  [warn] ball model unavailable, falling back to players model: {exc}")
                    self._ball_failed = True
            if self._ball is not None:
                bxy, bconf = self._ball(frame)
                if len(bxy):
                    # Replace the ball slot in `dets` with the better box.
                    keep = ~dets.is_ball
                    n_new = len(bxy)
                    dets.xyxy = np.concatenate([dets.xyxy[keep], bxy])
                    dets.conf = np.concatenate([dets.conf[keep], bconf])
                    new_cls = np.zeros(n_new, dtype=np.int64)
                    dets.cls = np.concatenate([dets.cls[keep], new_cls])
                    dets.cls_name = np.concatenate(
                        [dets.cls_name[keep], np.array(["ball"] * n_new, dtype=object)]
                    )
                    is_ball = np.zeros(len(dets.cls), dtype=bool)
                    is_ball[-n_new:] = True
                    dets.is_ball = is_ball
                    dets.is_player = np.concatenate([dets.is_player[keep], np.zeros(n_new, dtype=bool)])
                    dets.is_goalkeeper = np.concatenate([dets.is_goalkeeper[keep], np.zeros(n_new, dtype=bool)])
                    dets.is_referee = np.concatenate([dets.is_referee[keep], np.zeros(n_new, dtype=bool)])

        # Pitch landmarks -> homography (every N frames, interpolated between).
        if frame_idx % self.cfg.detection.keypoint_interval == 0:
            if self._keypoints is None:
                try:
                    self._keypoints = KeypointDetector(self.cfg.detection, self.cfg.paths)
                except FileNotFoundError as exc:
                    print(f"  [warn] keypoints model unavailable, pitch projection off: {exc}")
            if self._keypoints is not None:
                labels, pts = self._keypoints(frame)
                self.projector.update(labels, pts)

        # Teams: bootstrap unsupervised, then EMA-maintain.
        player_boxes = dets.player_xyxy
        if not self.teams.ready:
            self.teams.sample(frame, player_boxes)

        pitch_pts = None
        if self.projector.available and len(player_boxes):
            centres = np.array([
                [(b[0] + b[2]) / 2, (b[3] - b[1]) * 0.9 + b[1]]   # feet, not torso
                for b in player_boxes
            ], dtype=np.float32)
            pitch_pts = self.projector.transform_points(centres)

        # Drop ball detections that project OUTSIDE the pitch: the players
        # model fires on white crowd specks (observed at x~1240, y~180), and
        # those are always in the stands, never on the grass.
        if self.projector.available and dets.is_ball.any():
            dets.is_ball = self._ball_on_pitch(dets)

        # Team ids are needed BEFORE tracking so the gallery can gate on them.
        team_ids = self.teams.predict_many(frame, player_boxes)
        if self.teams.ready and len(player_boxes):
            # provisional teams (before tracking) just for the EMA update
            self.teams.update(frame, player_boxes, team_ids)

        tracks = self.tracker.process(frame, dets, team_ids, pitch_pts)

        # Ball position on the pitch (needed for the tactical-map view).
        if tracks.ball_xyxy is not None and self.projector.available:
            centre = np.array([[
                (tracks.ball_xyxy[0] + tracks.ball_xyxy[2]) / 2,
                (tracks.ball_xyxy[1] + tracks.ball_xyxy[3]) / 2,
            ]], dtype=np.float32)
            proj = self.projector.transform_points(centre)
            if proj is not None:
                tracks.ball_pitch_pos = proj[0]

        # Remember the winning team per persistent id (majority over time).
        for p in tracks.players:
            if p.gid is not None and p.team is not None:
                self._team_by_gid.setdefault(p.gid, p.team)
        return tracks

    # ---------------- output ---------------- #
    @staticmethod
    def _rows(tracks: FrameTracks) -> list[dict]:
        rows = []
        for p in tracks.players:
            rows.append({
                "frame": tracks.frame_idx,
                "gid": p.gid if p.gid is not None else "",
                "local_id": p.local_id,
                "team": p.team if p.team is not None else "",
                "x1": round(float(p.xyxy[0]), 1),
                "y1": round(float(p.xyxy[1]), 1),
                "x2": round(float(p.xyxy[2]), 1),
                "y2": round(float(p.xyxy[3]), 1),
                "conf": round(float(p.conf), 3),
                "pitch_x": round(float(p.pitch_pos[0]), 1) if p.pitch_pos is not None else "",
                "pitch_y": round(float(p.pitch_pos[1]), 1) if p.pitch_pos is not None else "",
            })
        if tracks.ball_xyxy is not None:
            b = tracks.ball_xyxy
            bp = tracks.ball_pitch_pos
            rows.append({
                "frame": tracks.frame_idx, "gid": "ball", "local_id": "",
                "team": "", "x1": round(float(b[0]), 1), "y1": round(float(b[1]), 1),
                "x2": round(float(b[2]), 1), "y2": round(float(b[3]), 1),
                "conf": round(tracks.ball_conf, 3),
                "pitch_x": round(float(bp[0]), 1) if bp is not None else "",
                "pitch_y": round(float(bp[1]), 1) if bp is not None else "",
            })
        return rows

    @staticmethod
    def _write_csv(rows: list[dict], path: Path) -> None:
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {len(rows)} rows -> {path}")

    def _color(self, gid: int) -> tuple[int, int, int]:
        if gid not in self._color_by_gid:
            self._color_by_gid[gid] = _PALETTE[gid % len(_PALETTE)]
        return self._color_by_gid[gid]

    def _draw(self, frame: np.ndarray, tracks: FrameTracks) -> np.ndarray:
        img = frame

        # Player boxes + persistent id + team colour.
        for p in tracks.players:
            if p.gid is None:
                color = (160, 160, 160)          # still warming up
                label = "..."
            else:
                color = self._color(p.gid)
                team = self._team_by_gid.get(p.gid, p.team)
                label = f"#{p.gid}" + (f" T{team}" if team is not None else "")

            x1, y1, x2, y2 = (int(v) for v in p.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Ball.
        if tracks.ball_xyxy is not None:
            x1, y1, x2, y2 = (int(v) for v in tracks.ball_xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, "ball", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        # HUD.
        s = self.tracker.gallery.summary()
        hud = (f"frame {tracks.frame_idx} | players {len(tracks.players)} | "
               f"ids {s['total_identities']} (recovered {s['recovered_ids']})")
        cv2.putText(img, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return img
