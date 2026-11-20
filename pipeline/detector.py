"""Stage 1: object detection.

Class semantics are resolved BY NAME from the loaded model, never by index.
The trained players model is {0: ball, 1: goalkeeper, 2: player, 3: referee}
while the old trial.py assumed {0: player, 1: referee, 2: ball}.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    BALL_CLASS_NAMES,
    GOALKEEPER_CLASS_NAMES,
    PLAYER_CLASS_NAMES,
    REFEREE_CLASS_NAMES,
    DetectionConfig,
    Paths,
)


@dataclass
class FrameDetections:
    """Detections for one frame, split by semantic role."""

    xyxy: np.ndarray           # (N, 4) float32, pixel coords
    conf: np.ndarray           # (N,)
    cls: np.ndarray            # (N,) int, raw model class index
    cls_name: np.ndarray       # (N,) str, raw model class name

    # Boolean masks into the rows above.
    is_player: np.ndarray      # outfield players only (excludes GK/ref)
    is_goalkeeper: np.ndarray
    is_referee: np.ndarray
    is_ball: np.ndarray

    @property
    def n_players(self) -> int:
        return int(self.is_player.sum())

    @property
    def empty(self) -> bool:
        return self.xyxy.shape[0] == 0

    @property
    def player_xyxy(self) -> np.ndarray:
        return self.xyxy[self.is_player]

    @property
    def ball_xyxy(self) -> np.ndarray | None:
        if not self.is_ball.any():
            return None
        return self.xyxy[self.is_ball][0]


def _match(name: str, candidates: tuple[str, ...]) -> bool:
    return name.strip().lower() in candidates


class PlayerDetector:
    """Wraps the YOLO players model and splits output into semantic roles."""

    def __init__(self, cfg: DetectionConfig, paths: Paths, model_path: Path | None = None):
        from ultralytics import YOLO  # imported lazily so tests can avoid it

        self.cfg = cfg
        path = model_path or paths.players_weights
        if not path.exists():
            raise FileNotFoundError(
                f"Players weights not found: {path}\n"
                "Weights are not stored in git -- run `python -m pipeline.assets` "
                "to download them from Hugging Face."
            )
        self.model = YOLO(str(path))
        self.names: dict[int, str] = dict(self.model.names)

        # Build index sets once from the model's own class names.
        idx = {i: n.lower() for i, n in self.names.items()}
        self._player_idx = {i for i, n in idx.items() if _match(n, PLAYER_CLASS_NAMES)}
        self._gk_idx = {i for i, n in idx.items() if _match(n, GOALKEEPER_CLASS_NAMES)}
        self._ref_idx = {i for i, n in idx.items() if _match(n, REFEREE_CLASS_NAMES)}
        self._ball_idx = {i for i, n in idx.items() if _match(n, BALL_CLASS_NAMES)}

        if not self._player_idx:
            raise ValueError(
                f"Could not find a 'player' class in model names {self.names}. "
                "Update PLAYER_CLASS_NAMES in pipeline/config.py."
            )

    @property
    def class_summary(self) -> str:
        def fmt(s: set[int]) -> str:
            return ",".join(f"{i}:{self.names[i]}" for i in sorted(s)) or "-"

        return (
            f"player=[{fmt(self._player_idx)}] gk=[{fmt(self._gk_idx)}] "
            f"ref=[{fmt(self._ref_idx)}] ball=[{fmt(self._ball_idx)}]"
        )

    def __call__(self, frame_bgr: np.ndarray) -> FrameDetections:
        res = self.model.predict(
            frame_bgr,
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            verbose=False,
            device=0,
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            empty_i = np.zeros(0, dtype=np.int64)
            empty_f = np.zeros(0, dtype=np.float32)
            empty_b = np.zeros(0, dtype=bool)
            empty_s = np.zeros(0, dtype=object)
            return FrameDetections(
                xyxy=empty_f.reshape(0, 4), conf=empty_f, cls=empty_i,
                cls_name=empty_s, is_player=empty_b, is_goalkeeper=empty_b,
                is_referee=empty_b, is_ball=empty_b,
            )

        cls = res.boxes.cls.cpu().numpy().astype(np.int64)
        cls_name = np.array([self.names[int(c)] for c in cls], dtype=object)

        return FrameDetections(
            xyxy=res.boxes.xyxy.cpu().numpy().astype(np.float32),
            conf=res.boxes.conf.cpu().numpy().astype(np.float32),
            cls=cls,
            cls_name=cls_name,
            is_player=np.isin(cls, list(self._player_idx)),
            is_goalkeeper=np.isin(cls, list(self._gk_idx)),
            is_referee=np.isin(cls, list(self._ref_idx)),
            is_ball=np.isin(cls, list(self._ball_idx)),
        )


class BallDetector:
    """Dedicated ball model.

    The players model does emit a `ball` class, but at broadcast zoom it also
    fires on white crowd specks and line markings (observed: spurious balls at
    x~1240, y~180 in the stands). The dedicated ball weights are far more
    precise.
    """

    def __init__(self, cfg: DetectionConfig, paths: Paths):
        from ultralytics import YOLO

        if not paths.ball_weights.exists():
            raise FileNotFoundError(f"Ball weights not found: {paths.ball_weights}")
        self.cfg = cfg
        self.model = YOLO(str(paths.ball_weights))
        self.names: dict[int, str] = dict(self.model.names)

    def __call__(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (xyxy, conf); empty arrays when nothing is found."""
        res = self.model.predict(
            frame_bgr,
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            verbose=False,
            device=0,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return (
            res.boxes.xyxy.cpu().numpy().astype(np.float32),
            res.boxes.conf.cpu().numpy().astype(np.float32),
        )


class KeypointDetector:
    """Detects the 28 pitch landmarks used to build the homography."""

    def __init__(self, cfg: DetectionConfig, paths: Paths):
        from ultralytics import YOLO

        if not paths.keypoints_weights.exists():
            raise FileNotFoundError(f"Keypoint weights not found: {paths.keypoints_weights}")
        self.cfg = cfg
        self.model = YOLO(str(paths.keypoints_weights))
        self.names: dict[int, str] = dict(self.model.names)

    def __call__(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (labels, src_points) where src_points is (M, 2) pixel centres."""
        res = self.model.predict(
            frame_bgr,
            conf=self.cfg.keypoint_conf,
            imgsz=self.cfg.imgsz,
            verbose=False,
            device=0,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.array([], dtype=object), np.zeros((0, 2), dtype=np.float32)

        cls = res.boxes.cls.cpu().numpy().astype(np.int64)
        labels = np.array([self.names[int(c)] for c in cls], dtype=object)
        centres = res.boxes.xywh.cpu().numpy()[:, :2].astype(np.float32)
        return labels, centres
