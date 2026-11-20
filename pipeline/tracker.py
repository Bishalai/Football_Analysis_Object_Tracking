"""Stage 2: frame-to-frame multi-object tracking (boxmot) + Stage 3 binding.

boxmot's BoT-SORT/BoostTrack handles short-term association (occlusions,
overlaps) inside a local track-id space. Those local ids are transient --
they are regenerated whenever a player leaves and re-enters the frame.

This module ties the two stages together:

    detections ──> boxmot tracker ──> local tracks (xyxy, local_id)
                                          │
                    crops ──> embedder ────┤
                    pitch pos ─────────────┤
                    team ──────────────────┤
                                          ▼
                                  IdentityGallery.observe()
                                          │
                                          ▼
                              persistent global id (gid)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GalleryConfig, Paths, ReIDConfig, TrackerConfig
from .detector import FrameDetections
from .gallery import IdentityGallery
from .reid import Embedder, build_embedder


@dataclass
class TrackedPlayer:
    """One player in one frame, with both ids."""

    frame_idx: int
    xyxy: np.ndarray
    local_id: int            # stage-2 track id (transient)
    gid: int | None          # stage-3 persistent identity (None = warming up)
    conf: float
    team: int | None
    pitch_pos: np.ndarray | None = None
    crop: np.ndarray | None = None


@dataclass
class FrameTracks:
    frame_idx: int
    players: list[TrackedPlayer] = field(default_factory=list)
    ball_xyxy: np.ndarray | None = None
    ball_conf: float = 0.0
    ball_pitch_pos: np.ndarray | None = None

    def by_gid(self) -> dict[int, TrackedPlayer]:
        return {p.gid: p for p in self.players if p.gid is not None}


class Stage2Tracker:
    """boxmot MOT + gallery binding for one video."""

    def __init__(
        self,
        tracker_cfg: TrackerConfig | None = None,
        gallery_cfg: GalleryConfig | None = None,
        reid_cfg: ReIDConfig | None = None,
        paths: Paths | None = None,
        embedder: Embedder | None = None,
    ):
        self.cfg = tracker_cfg or TrackerConfig()
        self.paths = paths or Paths()
        self.gallery = IdentityGallery(gallery_cfg)

        # boxmot tracker: constructed lazily on first update so we can pass
        # the actual frame shape (its CMC needs it).
        self._tracker = None
        self._tracker_cls = self._load_tracker(self.cfg.spec)

        # ReID embedder (shared by gallery; boxmot's internal ReID is
        # disabled -- we feed it detections only, and do appearance ourselves).
        self.embedder = embedder or build_embedder(reid_cfg or ReIDConfig(), self.paths)

        self.frame_idx = -1

    # ---------------- public API ---------------- #
    def process(
        self,
        frame_bgr: np.ndarray,
        dets: FrameDetections,
        team_ids: list[int | None],
        pitch_positions: np.ndarray | None,
    ) -> FrameTracks:
        """Run stage 2 + stage 3 for one frame."""
        self.frame_idx += 1
        fi = self.frame_idx
        out = FrameTracks(frame_idx=fi)

        if dets.is_ball.any():
            out.ball_xyxy = dets.xyxy[dets.is_ball][0]
            out.ball_conf = float(dets.conf[dets.is_ball][0])

        player_rows = np.flatnonzero(dets.is_player)
        if player_rows.size == 0:
            self.gallery.end_frame(fi, set())
            return out

        # ---- stage 2: local MOT ----
        dets_np = np.hstack([
            dets.xyxy[player_rows].astype(np.float32),
            dets.conf[player_rows].astype(np.float32).reshape(-1, 1),
            dets.cls[player_rows].astype(np.float32).reshape(-1, 1),
        ])
        if self._tracker is None:
            self._tracker = self._tracker_cls(self.cfg)

        tracks = self._tracker.update(dets_np, frame_bgr)   # (M, 8)
        if tracks.size == 0:
            self.gallery.end_frame(fi, set())
            return out

        # tracks: [x1, y1, x2, y2, track_id, conf, cls, det_index]
        det_index = tracks[:, 7].astype(np.int64)
        local_ids = tracks[:, 4].astype(np.int64)

        # ---- appearance embeddings for the matched detections only ----
        crops = [self._crop(frame_bgr, tracks[i, :4]) for i in range(len(tracks))]
        embs = self.embedder.embed(crops)

        # ---- stage 3: bind to persistent identities ----
        seen: set[int] = set()
        for i in range(len(tracks)):
            lid = int(local_ids[i])
            seen.add(lid)

            row = player_rows[det_index[i]] if 0 <= det_index[i] < len(player_rows) else player_rows[i]
            team = team_ids[det_index[i]] if 0 <= det_index[i] < len(team_ids) else None
            pos = None if pitch_positions is None else pitch_positions[det_index[i]]

            gid = self.gallery.observe(lid, embs[i], team, pos, fi)

            out.players.append(TrackedPlayer(
                frame_idx=fi,
                xyxy=tracks[i, :4].astype(np.float32),
                local_id=lid,
                gid=gid,
                conf=float(tracks[i, 5]),
                team=team,
                pitch_pos=None if pos is None else np.asarray(pos, dtype=np.float32),
                crop=crops[i],
            ))
            _ = row  # raw row kept for callers that need the source detection

        self.gallery.end_frame(fi, seen)
        return out

    # ---------------- internals ---------------- #
    @staticmethod
    def _crop(frame: np.ndarray, xyxy: np.ndarray, pad: float = 0.10) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * pad
        x2 += bw * pad
        y1 -= bh * pad
        y2 += bh * pad
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w, x2)), int(min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((32, 16, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _load_tracker(spec: str):
        """Return a factory: factory(cfg) -> boxmot tracker instance.

        boxmot takes settings via a per-algorithm Config object; kwargs on the
        tracker itself raise TypeError, so we filter our knobs down to the
        fields that algorithm actually exposes.
        """
        import dataclasses

        import boxmot

        aliases = {
            "botsort": "BotSort", "bytetrack": "ByteTrack",
            "boosttrack": "BoostTrack", "deepocsort": "DeepOcSort",
            "ocsort": "OcSort", "strongsort": "StrongSORT",
            "hybridsort": "HybridSort", "eagermot": "EagerMot",
        }
        name = spec.strip().lower()
        cls = getattr(boxmot, aliases.get(name, name), None)
        if cls is None or not isinstance(cls, type):
            raise ValueError(f"Unknown boxmot tracker spec {spec!r}")

        cfg_cls = getattr(boxmot, cls.__name__ + "Config", None)
        if cfg_cls is None or not dataclasses.is_dataclass(cfg_cls):
            # No config object for this algorithm: fall back to defaults.
            def bare(_cfg):
                return cls()
            return bare

        fields = {f.name for f in dataclasses.fields(cfg_cls)}

        def factory(cfg):
            wanted = dict(
                max_age=cfg.max_age,
                min_hits=cfg.min_hits,
                use_cmc=cfg.use_cmc,
                use_embeddings=cfg.use_boxmot_reid,
            )
            applied = {k: v for k, v in wanted.items() if k in fields}
            return cls(cfg_cls(**applied))

        return factory
