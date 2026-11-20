"""Team classification from player shirt colours.

Unsupervised by default: gather shirt-patch colour samples for the first
`bootstrap_frames`, cluster into `n_teams` with k-means, then keep the
prototypes updated with an EMA so slow lighting drift is tracked.

The cluster -> {0, 1} labelling is arbitrary; we pin it once using the
initial assignment so it is stable for the whole video.
"""

from __future__ import annotations

import numpy as np

from .config import TeamConfig


class TeamClassifier:
    def __init__(self, cfg: TeamConfig | None = None):
        self.cfg = cfg or TeamConfig()
        self.prototypes: np.ndarray | None = None   # (n_teams, 3) float32, RGB
        self._samples: list[np.ndarray] = []
        self._n_frames_seen = 0

    @property
    def ready(self) -> bool:
        return self.prototypes is not None

    # ---------------- public API ---------------- #
    def sample(self, frame_bgr: np.ndarray, player_boxes: np.ndarray) -> None:
        """Collect shirt-colour samples from this frame's player boxes."""
        if self.ready:
            return
        for box in player_boxes:
            patch = self._shirt_patch(frame_bgr, box)
            if patch is not None:
                self._samples.append(patch)

        self._n_frames_seen += 1
        if self._n_frames_seen >= self.cfg.bootstrap_frames and len(self._samples) >= self.cfg.n_teams * 4:
            self._fit()

    def predict_many(self, frame_bgr: np.ndarray, player_boxes: np.ndarray) -> list[int | None]:
        """Team id per box (None when the classifier is not ready yet)."""
        if not self.ready:
            return [None] * len(player_boxes)
        out: list[int | None] = []
        for box in player_boxes:
            patch = self._shirt_patch(frame_bgr, box)
            out.append(self._predict_one(patch) if patch is not None else None)
        return out

    def update(self, frame_bgr: np.ndarray, player_boxes: np.ndarray, teams: list[int | None]) -> None:
        """EMA-update prototypes with confident assignments (lighting drift)."""
        if not self.ready:
            return
        alpha = self.cfg.ema_alpha
        sums = np.zeros_like(self.prototypes)
        counts = np.zeros(self.cfg.n_teams, dtype=np.int64)

        for box, team in zip(player_boxes, teams):
            if team is None:
                continue
            patch = self._shirt_patch(frame_bgr, box)
            if patch is None:
                continue
            # Gate: a goalkeeper/referee detected as "player" sits far from
            # both team prototypes -- do not let it drag a prototype over.
            if float(np.linalg.norm(patch - self.prototypes[team])) > self.cfg.update_max_dist:
                continue
            sums[team] += patch
            counts[team] += 1

        for t in range(self.cfg.n_teams):
            if counts[t]:
                mean = sums[t] / counts[t]
                self.prototypes[t] = (1 - alpha) * self.prototypes[t] + alpha * mean
        self.prototypes = self.prototypes.astype(np.float32)

    # ---------------- internals ---------------- #
    def _shirt_patch(self, frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """Median RGB of the torso region of a detection box.

        Two things matter for broadcast football footage:
          * boxes are tiny (~20x50 px) and loose, so a naive crop is often
            mostly grass -- we reject grass pixels (G dominant) before the
            median, and only accept the patch if enough real pixels remain,
          * kits are often two shades of the SAME hue (sky blue vs navy), so
            absolute brightness must be preserved: no unit-length
            normalisation here.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        bw, bh = x2 - x1, y2 - y1
        if bw < 6 or bh < 12:
            return None

        rx1 = int(max(0, x1 + bw * 0.28))
        rx2 = int(min(w, x2 - bw * 0.28))
        ry1 = int(max(0, y1 + bh * 0.20))
        ry2 = int(min(h, y1 + bh * 0.45))
        if rx2 <= rx1 or ry2 <= ry1:
            return None

        region = frame_bgr[ry1:ry2, rx1:rx2].astype(np.float32).reshape(-1, 3)
        if region.size == 0:
            return None

        b, g, r = region[:, 0], region[:, 1], region[:, 2]
        grass = (g > b + 8) & (g > r + 8)
        keep = region[~grass]
        if len(keep) < 6:
            return None                      # box is all background

        median_bgr = np.median(keep, axis=0)
        return median_bgr[::-1].copy().astype(np.float32)   # -> RGB

    def _fit(self) -> None:
        from sklearn.cluster import KMeans

        X = np.stack(self._samples)
        n = len(X)
        # NOTE: deliberately NOT normalising to unit length. The two kits are
        # often two shades of the same hue, and the brightness difference is
        # exactly what separates them.

        # Cluster with EXTRA capacity: goalkeepers (red/yellow) and referees
        # (black) are frequently detected as plain "player", and with exactly
        # n_teams clusters k-means happily spends one on the GK -- which then
        # forces both real teams into a single merged cluster (observed: a
        # 19:2 split). Extra clusters let the kits separate cleanly first.
        k = min(max(self.cfg.n_teams + 3, self.cfg.n_teams), n)
        km = KMeans(n_clusters=k, n_init=20, random_state=0)
        raw = km.fit_predict(X)
        centres = km.cluster_centers_

        # Drop small clusters: a GK/ref is 1-3 people, a team is ~10.
        min_size = max(8, int(0.12 * n))
        keep = [c for c in range(k) if int((raw == c).sum()) >= min_size]

        # Fall back to everything if the size filter was too aggressive.
        if len(keep) < self.cfg.n_teams:
            sizes = [int((raw == c).sum()) for c in range(k)]
            keep = list(np.argsort(sizes)[::-1][: self.cfg.n_teams])

        # Merge down to exactly n_teams by repeatedly joining the closest pair.
        groups = [[c] for c in keep]
        while len(groups) > self.cfg.n_teams:
            best, bi, bj = None, 0, 1
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    ci = centres[groups[i]].mean(axis=0)
                    cj = centres[groups[j]].mean(axis=0)
                    d = float(np.linalg.norm(ci - cj))
                    if best is None or d < best:
                        best, bi, bj = d, i, j
            groups[bi] = groups[bi] + groups[bj]
            groups.pop(bj)

        # Pin cluster -> stable team id by brightness (dark team = 0) so the
        # labelling does not flip between runs.
        order = sorted(range(len(groups)),
                       key=lambda gi: float(centres[groups[gi]].mean()))
        label_of_cluster: dict[int, int] = {}
        for stable, gi in enumerate(order):
            for c in groups[gi]:
                label_of_cluster[c] = stable

        # Samples from dropped clusters (GK/referee kits) are excluded from
        # the prototypes entirely -- they belong to neither team.
        proto = np.zeros((self.cfg.n_teams, 3), dtype=np.float32)
        counts = np.zeros(self.cfg.n_teams, dtype=np.int64)
        for i, r in enumerate(raw):
            t = label_of_cluster.get(int(r))
            if t is None:
                continue
            proto[t] += X[i]
            counts[t] += 1
        for t in range(self.cfg.n_teams):
            if counts[t]:
                proto[t] /= counts[t]
        self.prototypes = proto.astype(np.float32)

    def _predict_one(self, patch: np.ndarray) -> int | None:
        d = np.linalg.norm(self.prototypes - patch, axis=1)
        t = int(np.argmin(d))
        # Too far from either team -> a GK/referee kit, not an outfield shirt.
        if float(d[t]) > self.cfg.update_max_dist:
            return None
        return t
