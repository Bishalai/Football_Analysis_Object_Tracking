"""Tactical-map rendering: project tracked players onto the pitch diagram.

Pitch coordinates (from the homography) are already in the tactical map's
pixel space -- the 28 landmark annotations in `pitch map labels position.json`
span x 15-291 / y 15-521 on a 314x546 image -- so rendering is a direct
overlay, no extra transform required.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import Paths

# Team colours chosen to read clearly on a green pitch.
_TEAM_COLORS = {0: (60, 90, 220), 1: (230, 90, 40)}   # BGR
_UNKNOWN = (200, 200, 200)
_BALL = (0, 240, 255)


class TacticalMap:
    """Renders the current frame's player positions on the pitch diagram."""

    def __init__(self, paths: Paths | None = None, scale: float = 2.0):
        paths = paths or Paths()
        self.scale = scale

        img = cv2.imread(str(paths.tactical_map))
        if img is None:
            raise FileNotFoundError(f"Tactical map not found: {paths.tactical_map}")
        self.base = img
        self.h, self.w = img.shape[:2]

    # ---------------- public API ---------------- #
    def render(
        self,
        positions: list[tuple[float, float]],
        teams: list[int | None],
        labels: list[str] | None = None,
        ball_pos: tuple[float, float] | None = None,
        trail: list[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Return a new BGR image with the given players drawn.

        Parameters
        ----------
        positions:
            Pitch-space (x, y) per player.
        teams:
            Team id per player (None -> grey).
        labels:
            Optional text per player (e.g. "#7").
        ball_pos:
            Pitch-space ball position, if known.
        trail:
            Recent ball positions, drawn as a fading line.
        """
        canvas = self.base.copy()

        if trail and len(trail) >= 2:
            pts = [(int(round(x)), int(round(y))) for x, y in trail
                   if np.isfinite(x) and np.isfinite(y)]
            for i in range(1, len(pts)):
                # Fade: older segments are darker.
                a = i / len(pts)
                color = tuple(int(c * (0.3 + 0.7 * a)) for c in _BALL)
                cv2.line(canvas, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

        r = max(3, int(4 * self.scale / 2))
        for i, (x, y) in enumerate(positions):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            px, py = int(round(x)), int(round(y))
            team = teams[i] if i < len(teams) else None
            color = _TEAM_COLORS.get(team, _UNKNOWN)
            cv2.circle(canvas, (px, py), r, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), r, (255, 255, 255), 1, cv2.LINE_AA)
            if labels and i < len(labels) and labels[i]:
                # The pitch diagram is white, so white text vanishes: draw a
                # dark outline underneath for contrast.
                cv2.putText(canvas, labels[i], (px + r + 2, py + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0),
                            2, cv2.LINE_AA)
                cv2.putText(canvas, labels[i], (px + r + 2, py + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255),
                            1, cv2.LINE_AA)

        if ball_pos is not None and np.isfinite(ball_pos[0]):
            bx, by = int(round(ball_pos[0])), int(round(ball_pos[1]))
            cv2.circle(canvas, (bx, by), max(3, r - 1), _BALL, -1, cv2.LINE_AA)

        if self.scale != 1.0:
            canvas = cv2.resize(
                canvas, (int(self.w * self.scale), int(self.h * self.scale)),
                interpolation=cv2.INTER_AREA,
            )
        return canvas

    def legend(self) -> np.ndarray:
        """Small strip image explaining the colours."""
        strip = np.full((34, 260, 3), 30, dtype=np.uint8)
        items = [("Team A", _TEAM_COLORS[0]), ("Team B", _TEAM_COLORS[1]),
                 ("Unknown", _UNKNOWN), ("Ball", _BALL)]
        for i, (name, color) in enumerate(items):
            x = 10 + i * 64
            cv2.circle(strip, (x, 17), 6, color, -1, cv2.LINE_AA)
            cv2.putText(strip, name, (x - 8, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (240, 240, 240), 1, cv2.LINE_AA)
        return strip
