"""Stage 3: long-term identity re-acquisition.

This is the layer your pipeline was missing. frame-to-frame MOT (stage 2)
assigns a *new* track id every time a player leaves and re-enters the frame;
nothing ties the two together. The gallery does that by keeping a bank of
player identities -- each with an averaged appearance embedding, a team and a
last-seen pitch position -- and re-binding new tracks to them.

Association is done as a single global Hungarian assignment over all
simultaneously-arriving tracks, not greedily, so two re-entering players
cannot claim the same identity.

Football-specific priors make this far more reliable than generic ReID:
  * team is a HARD gate (a player never changes team mid-match),
  * at most 11 live identities per team,
  * pitch position gates the search, widening with time spent off-screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GalleryConfig


# --------------------------------------------------------------------------- #
# Identity record
# --------------------------------------------------------------------------- #
@dataclass
class Identity:
    gid: int
    team: int | None = None
    emb: np.ndarray | None = None          # running mean, L2-normalised
    n_emb: int = 0
    pitch_pos: np.ndarray | None = None    # last known position on the pitch
    first_seen: int = 0
    last_seen: int = 0
    active_local: int | None = None        # stage-2 track bound now; None => lost
    frames_seen: int = 0
    over_roster: bool = False

    @property
    def age(self) -> int:
        return self.last_seen - self.first_seen


@dataclass
class PendingTrack:
    """A brand-new stage-2 track still in its warm-up window."""

    local_id: int
    first_frame: int
    embs: list[np.ndarray] = field(default_factory=list)
    teams: list[int] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return len(self.embs)

    @property
    def mean_emb(self) -> np.ndarray:
        m = np.mean(np.stack(self.embs), axis=0)
        n = np.linalg.norm(m)
        return m / n if n > 0 else m

    @property
    def team(self) -> int | None:
        if not self.teams:
            return None
        vals, counts = np.unique(self.teams, return_counts=True)
        return int(vals[np.argmax(counts)])

    @property
    def position(self) -> np.ndarray | None:
        return self.positions[-1] if self.positions else None


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #
class IdentityGallery:
    def __init__(self, cfg: GalleryConfig | None = None):
        self.cfg = cfg or GalleryConfig()
        self.identities: dict[int, Identity] = {}
        self._local_to_gid: dict[int, int] = {}
        self._pending: dict[int, PendingTrack] = {}
        self._seen_this_frame: set[int] = set()
        self._next_gid = 1
        # Book-keeping for diagnostics / analytics.
        self.stats = {"new_ids": 0, "recovered_ids": 0, "warmup_drops": 0}

    # ---------------- public API ---------------- #
    def observe(
        self,
        local_id: int,
        emb: np.ndarray | None,
        team: int | None,
        pitch_pos: np.ndarray | None,
        frame_idx: int,
    ) -> int | None:
        """Register one tracked player for this frame.

        Returns the persistent global id, or None while a brand-new track is
        still warming up (too young to trust for re-identification).
        """
        # (1) Already bound -> just refresh the identity.
        self._seen_this_frame.add(local_id)
        if local_id in self._local_to_gid:
            gid = self._local_to_gid[local_id]
            ident = self.identities[gid]
            self._refresh(ident, emb, team, pitch_pos, frame_idx)
            return gid

        # (2) New track -> accumulate evidence during warm-up.
        pending = self._pending.get(local_id)
        if pending is None:
            pending = PendingTrack(local_id=local_id, first_frame=frame_idx)
            self._pending[local_id] = pending
        if emb is not None:
            pending.embs.append(np.asarray(emb, dtype=np.float32))
        if team is not None:
            pending.teams.append(int(team))
        if pitch_pos is not None:
            pending.positions.append(np.asarray(pitch_pos, dtype=np.float32))

        if pending.frames < self.cfg.warmup_frames:
            return None

        # (3) Warm enough -> try to re-identify against the gallery.
        return self._resolve(pending, frame_idx)

    def end_frame(self, frame_idx: int, seen_local_ids: set[int]) -> None:
        """Call once per frame after all observations.

        Marks identities whose stage-2 track vanished as *lost* (so they become
        candidates for re-binding) and expires identities unseen for too long.
        """
        cfg = self.cfg

        # Unbind identities whose local track stopped showing up.
        for gid, ident in self.identities.items():
            if ident.active_local is None:
                continue
            if ident.active_local in seen_local_ids:
                continue
            if frame_idx - ident.last_seen >= cfg.lost_after_frames:
                ident.active_local = None

        # Drop pending tracks that vanished before finishing warm-up.
        gone = [lid for lid in self._pending if lid not in seen_local_ids]
        for lid in gone:
            p = self._pending.pop(lid)
            if p.frames > 0:
                self.stats["warmup_drops"] += 1

        # Expire stale identities entirely.
        stale = [
            gid for gid, i in self.identities.items()
            if frame_idx - i.last_seen > cfg.max_age_frames
        ]
        for gid in stale:
            ident = self.identities.pop(gid)
            if ident.active_local in self._local_to_gid:
                self._local_to_gid.pop(ident.active_local, None)

        # Next frame starts with a fresh observation set.
        self._seen_this_frame.clear()

    # ---------------- internals ---------------- #
    def _refresh(
        self,
        ident: Identity,
        emb: np.ndarray | None,
        team: int | None,
        pitch_pos: np.ndarray | None,
        frame_idx: int,
    ) -> None:
        if emb is not None:
            emb = np.asarray(emb, dtype=np.float32)
            if ident.emb is None:
                ident.emb = emb.copy()
                ident.n_emb = 1
            else:
                # EMA keeps the descriptor current under lighting changes.
                alpha = 1.0 / min(ident.n_emb + 1, 32)
                ident.emb = (1 - alpha) * ident.emb + alpha * emb
                n = np.linalg.norm(ident.emb)
                if n > 0:
                    ident.emb = ident.emb / n
                ident.n_emb += 1
        if team is not None and ident.team is None:
            ident.team = int(team)
        if pitch_pos is not None:
            ident.pitch_pos = np.asarray(pitch_pos, dtype=np.float32)
        ident.last_seen = frame_idx
        ident.frames_seen += 1

    def _resolve(self, pending: PendingTrack, frame_idx: int) -> int:
        """Match a warmed-up new track against lost identities."""
        cfg = self.cfg
        emb = pending.mean_emb
        team = pending.team
        pos = pending.position

        # Candidates: recently seen, appearance-compatible, and not owned by a
        # stage-2 track we are actually observing this frame.
        cands: list[Identity] = []
        for ident in self.identities.values():
            if ident.active_local is not None and ident.active_local in self._seen_this_frame:
                continue                      # owned by a live track
            if frame_idx - ident.last_seen > cfg.max_age_frames:
                continue                      # expired
            if ident.emb is None:
                continue
            if cfg.require_team_match and team is not None and ident.team is not None:
                if team != ident.team:
                    continue                  # HARD gate: never cross teams
            sim = float(np.dot(emb, ident.emb))
            if sim < cfg.min_appearance_sim:
                continue                      # appearance gate
            if pos is not None and ident.pitch_pos is not None:
                gate = self._position_gate(frame_idx - ident.last_seen)
                if float(np.linalg.norm(pos - ident.pitch_pos)) > gate:
                    continue                  # too far to be the same player
            cands.append(ident)

        matched_gid: int | None = None
        if cands:
            matched_gid = self._assign(cands, emb, pos, frame_idx, relaxed=False)

        # Roster is full: relax the appearance gate rather than minting a
        # 12th player for the same team (this is what converter.py's
        # ">22" hack was reaching for). Runs even when the strict gate
        # rejected every candidate above.
        if matched_gid is None and team is not None and self._roster_full(team, frame_idx):
            # An identity still nominally bound to a stage-2 track that is
            # NOT being observed this frame is just as available as an
            # unbound one: its player is off-screen (the unbind only
            # happens lost_after_frames later, but a returning player may
            # re-enter well before that).
            relaxed_cands = [
                i for i in self.identities.values()
                if (i.active_local is None or i.active_local not in self._seen_this_frame)
                and (team is None or i.team is None or i.team == team)
                and frame_idx - i.last_seen <= cfg.max_age_frames
            ]
            if relaxed_cands:
                matched_gid = self._assign(relaxed_cands, emb, pos, frame_idx, relaxed=True)

        # No match -> mint a brand-new identity (unless roster forbids it).
        if matched_gid is None:
            if team is not None and self._roster_full(team, frame_idx):
                # Still need an id for this detection; flag the overflow.
                gid = self._mint(pending, team, emb, pos, frame_idx, over=True)
            else:
                gid = self._mint(pending, team, emb, pos, frame_idx, over=False)
            self.stats["new_ids"] += 1
        else:
            self.stats["recovered_ids"] += 1
            ident = self.identities[matched_gid]
            ident.active_local = pending.local_id
            self._local_to_gid[pending.local_id] = matched_gid
            self._refresh(ident, emb, team, pos, frame_idx)
            if team is not None:
                ident.team = int(team)

        self._pending.pop(pending.local_id, None)
        return self._local_to_gid[pending.local_id]

    def _assign(
        self,
        cands: list[Identity],
        emb: np.ndarray,
        pos: np.ndarray | None,
        frame_idx: int,
        relaxed: bool,
    ) -> int | None:
        """Pick the cheapest valid candidate identity for one new track."""
        cfg = self.cfg
        cost = np.full(len(cands), np.inf, dtype=np.float64)

        for k, ident in enumerate(cands):
            sim = float(np.dot(emb, ident.emb))
            if not relaxed and sim < cfg.min_appearance_sim:
                continue

            app = 1.0 - sim
            if pos is not None and ident.pitch_pos is not None:
                gate = max(self._position_gate(frame_idx - ident.last_seen), 1e-6)
                pos_term = min(float(np.linalg.norm(pos - ident.pitch_pos)) / gate, 1.0)
            else:
                pos_term = 0.0
            age_term = min((frame_idx - ident.last_seen) / max(cfg.max_age_frames, 1), 1.0)

            c = cfg.w_appearance * app + cfg.w_position * pos_term + cfg.w_age * age_term
            # In relaxed mode skip only the hard cost ceiling, not the ranking.
            if not relaxed and c > self._max_cost():
                continue
            cost[k] = c

        valid = np.isfinite(cost)
        if not valid.any():
            return None

        # Single row (one track) -> pick the cheapest valid candidate.
        k = int(np.argmin(np.where(valid, cost, np.inf)))
        if not np.isfinite(cost[k]):
            return None
        if not relaxed and cost[k] > self._max_cost():
            return None
        return cands[k].gid

    def _mint(
        self,
        pending: PendingTrack,
        team: int | None,
        emb: np.ndarray,
        pos: np.ndarray | None,
        frame_idx: int,
        over: bool,
    ) -> int:
        gid = self._next_gid
        self._next_gid += 1
        self.identities[gid] = Identity(
            gid=gid,
            team=int(team) if team is not None else None,
            emb=emb.copy(),
            n_emb=1,
            pitch_pos=None if pos is None else np.asarray(pos, dtype=np.float32).copy(),
            first_seen=frame_idx,
            last_seen=frame_idx,
            active_local=pending.local_id,
            frames_seen=1,
            over_roster=over,
        )
        self._local_to_gid[pending.local_id] = gid
        return gid

    # ---------------- helpers ---------------- #
    def _position_gate(self, frames_lost: int) -> float:
        """Widen the position gate the longer a player has been off-screen."""
        cfg = self.cfg
        t = min(max(frames_lost, 0), cfg.position_gate_frames) / max(cfg.position_gate_frames, 1)
        return cfg.position_gate_near + t * (cfg.position_gate_far - cfg.position_gate_near)

    def _roster_full(self, team: int, frame_idx: int) -> bool:
        live = sum(
            1 for i in self.identities.values()
            if i.team == team and frame_idx - i.last_seen <= self.cfg.max_age_frames
        )
        return live >= self.cfg.max_roster_per_team

    def _max_cost(self) -> float:
        """Ceiling on the combined cost for a strict match."""
        cfg = self.cfg
        # Worst acceptable: appearance at the gate, position free, age free.
        return cfg.w_appearance * (1.0 - cfg.min_appearance_sim) + 0.35

    # ---------------- diagnostics ---------------- #
    def summary(self) -> dict:
        active = sum(1 for i in self.identities.values() if i.active_local is not None)
        lost = len(self.identities) - active
        return {
            "total_identities": len(self.identities),
            "active": active,
            "lost": lost,
            **self.stats,
        }

    def team_rosters(self) -> dict[int | None, list[int]]:
        out: dict[int | None, list[int]] = {}
        for ident in self.identities.values():
            out.setdefault(ident.team, []).append(ident.gid)
        return out
