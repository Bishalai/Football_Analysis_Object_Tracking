"""Frame -> pitch (tactical map) coordinate transform.

Extracted and stabilised from trial.py. Improvements over the original:
  * homography is temporally smoothed (EMA on the projected corners) so the
    pitch positions do not jitter frame to frame,
  * the update gate uses a proper mean displacement (not an MSE compared
    against a pixel tolerance, which mixed units),
  * returns None cleanly when too few landmarks are visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import Paths


class PitchProjector:
    """Builds and applies a homography from frame pixels to pitch pixels."""

    MIN_LANDMARKS = 4

    def __init__(self, paths: Paths, ema_alpha: float = 0.35, update_tol_px: float = 8.0):
        with open(paths.pitch_keypoints_json, "r", encoding="utf-8") as f:
            self.map_positions: dict[str, list[float]] = json.load(f)

        self.ema_alpha = ema_alpha
        self.update_tol_px = update_tol_px

        self._H: np.ndarray | None = None          # smoothed homography
        self._H_raw: np.ndarray | None = None
        self._prev_labels: list[str] | None = None
        self._prev_pts: np.ndarray | None = None

    @property
    def available(self) -> bool:
        return self._H is not None

    def reset(self) -> None:
        """Forget the current homography (new video / new camera angle)."""
        self._H = None
        self._H_raw = None
        self._prev_labels = None
        self._prev_pts = None

    @property
    def H(self) -> np.ndarray | None:
        return self._H

    def pitch_size(self) -> tuple[int, int]:
        xs = [p[0] for p in self.map_positions.values()]
        ys = [p[1] for p in self.map_positions.values()]
        return int(max(xs)) + 1, int(max(ys)) + 1

    def update(self, labels: np.ndarray, src_pts: np.ndarray) -> bool:
        """Feed this frame's detected landmarks. Returns True if H is usable."""
        labels = [str(l) for l in labels]
        valid = [i for i, l in enumerate(labels) if l in self.map_positions]
        if len(valid) < self.MIN_LANDMARKS:
            # Too few landmarks: keep the previous homography if we have one.
            return self._H is not None

        labels = [labels[i] for i in valid]
        src = np.asarray(src_pts[valid], dtype=np.float64)
        dst = np.array([self.map_positions[l] for l in labels], dtype=np.float64)

        # Gate: only recompute when landmarks actually moved beyond tolerance.
        should_update = True
        if self._prev_labels is not None and self._prev_pts is not None:
            common = set(labels) & set(self._prev_labels)
            if len(common) >= self.MIN_LANDMARKS:
                cur = {l: p for l, p in zip(labels, src)}
                prev = {l: p for l, p in zip(self._prev_labels, self._prev_pts)}
                disp = np.array([cur[l] - prev[l] for l in common])
                mean_disp = float(np.linalg.norm(disp, axis=1).mean())
                should_update = mean_disp > self.update_tol_px

        if should_update:
            H_raw, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if H_raw is not None:
                self._H_raw = H_raw
                self._H = H_raw if self._H is None else self._blend(self._H, H_raw)

        self._prev_labels = labels
        self._prev_pts = src
        return self._H is not None

    def _blend(self, old: np.ndarray, new: np.ndarray) -> np.ndarray:
        """EMA-blend two homographies.

        Averaging H's coefficients directly is meaningless (H is projective,
        defined up to scale), so we blend where each H actually sends a grid
        of frame points, then refit a homography to those averaged targets.
        """
        frame_grid = np.array(
            [[x, y] for y in np.linspace(0, 720, 5) for x in np.linspace(0, 1280, 5)],
            dtype=np.float64,
        )
        pts = frame_grid.reshape(-1, 1, 2)
        try:
            p_old = cv2.perspectiveTransform(pts, old).reshape(-1, 2)
            p_new = cv2.perspectiveTransform(pts, new).reshape(-1, 2)
        except cv2.error:
            return new

        if not (np.all(np.isfinite(p_old)) and np.all(np.isfinite(p_new))):
            return new

        p_avg = (1 - self.ema_alpha) * p_old + self.ema_alpha * p_new
        H_blend, _ = cv2.findHomography(frame_grid, p_avg, method=0)
        return H_blend if H_blend is not None else new

    def transform_points(self, pts: np.ndarray) -> np.ndarray | None:
        """Map (N, 2) frame pixels -> (N, 2) pitch pixels. None if unavailable."""
        if self._H is None or pts is None or len(pts) == 0:
            return None
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self._H).reshape(-1, 2)
        # Homography can send points behind the camera (w<0) -> garbage.
        if not np.all(np.isfinite(out)):
            return None
        return out.astype(np.float32)

    def transform_point(self, pt: np.ndarray) -> np.ndarray | None:
        out = self.transform_points(np.asarray(pt, dtype=np.float32).reshape(1, 2))
        return None if out is None else out[0]
