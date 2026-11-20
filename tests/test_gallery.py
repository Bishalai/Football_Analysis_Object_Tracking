"""Unit tests for the Stage-3 IdentityGallery.

These run on pure numpy -- no GPU, no video, no YOLO -- so they are fast and
pin down the re-identification behaviour that the rest of the pipeline relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import GalleryConfig
from pipeline.gallery import IdentityGallery


# ------------------------------------------------------------------ helpers #
def emb(seed: int, dim: int = 16) -> np.ndarray:
    """Deterministic unit-norm descriptor."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def similar_to(base: np.ndarray, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """Descriptor close to `base` (simulates the same player, new crop)."""
    rng = np.random.default_rng(seed)
    v = base + rng.normal(scale=noise, size=base.shape).astype(np.float32)
    return v / np.linalg.norm(v)


class Sim:
    """Drives a gallery frame by frame the way the tracker will."""

    def __init__(self, cfg: GalleryConfig | None = None):
        self.g = IdentityGallery(cfg or GalleryConfig(warmup_frames=3))
        self.frame = 0

    def tick(self, tracks: dict[int, tuple[np.ndarray, int | None, np.ndarray | None]]):
        """tracks: {local_id: (embedding, team, pitch_pos)}"""
        self.frame += 1
        seen: set[int] = set()
        out = {}
        for lid, (e, team, pos) in tracks.items():
            out[lid] = self.g.observe(lid, e, team, pos, self.frame)
            seen.add(lid)
        self.g.end_frame(self.frame, seen)
        return out


def warm(sim: Sim, lid: int, e: np.ndarray, team: int, pos, n: int | None = None):
    """Feed one track for enough frames to clear warm-up and mint an identity."""
    n = n or sim.g.cfg.warmup_frames
    gid = None
    for _ in range(n):
        gid = sim.tick({lid: (e, team, pos)}).get(lid)
    return gid


# -------------------------------------------------------------------- tests #
def test_new_track_gets_new_id():
    sim = Sim()
    gid = warm(sim, lid=1, e=emb(1), team=0, pos=np.array([10.0, 10.0]))
    assert gid is not None
    assert gid == 1


def test_track_returns_same_id_after_leaving_frame():
    """THE core requirement: player exits frame, re-enters, keeps identity."""
    sim = Sim()
    e = emb(7)
    pos = np.array([50.0, 50.0])

    gid1 = warm(sim, lid=1, e=e, team=0, pos=pos)
    assert gid1 is not None

    # Player walks off-screen: no observations for a while.
    for _ in range(60):
        sim.tick({})

    # Re-enters under a brand-new stage-2 track id, slightly different crop.
    gid2 = warm(sim, lid=99, e=similar_to(e, 0.05, seed=3), team=0, pos=pos + 8)
    assert gid2 is not None, "re-entering player was not re-identified"
    assert gid2 == gid1, f"ID switched: {gid1} -> {gid2}"


def test_warmup_suppresses_transient_tracks():
    """A track that lives fewer frames than warmup must not mint an identity."""
    sim = Sim()
    cfg = sim.g.cfg
    for _ in range(cfg.warmup_frames - 1):
        sim.tick({5: (emb(2), 0, np.array([1.0, 1.0]))})
    assert sim.g.identities == {}, "short-lived track created an identity"
    assert sim.g.summary()["total_identities"] == 0


def test_team_gate_blocks_cross_team_match():
    """Same appearance but different team must never be matched."""
    sim = Sim()
    e = emb(11)
    pos = np.array([30.0, 30.0])

    gid1 = warm(sim, lid=1, e=e, team=0, pos=pos)
    for _ in range(60):
        sim.tick({})

    # Identical appearance, but the player is now on the other team.
    gid2 = warm(sim, lid=42, e=e, team=1, pos=pos)
    assert gid2 != gid1, "cross-team identity leak"


def test_two_simultaneous_reentries_do_not_collide():
    """Two players return at once; each must get its own identity back."""
    sim = Sim()
    ea, eb = emb(1), emb(2)
    pa, pb = np.array([20.0, 20.0]), np.array([180.0, 300.0])

    gid_a = warm(sim, lid=1, e=ea, team=0, pos=pa)
    gid_b = warm(sim, lid=2, e=eb, team=1, pos=pb)
    assert gid_a != gid_b

    for _ in range(45):
        sim.tick({})

    # Both return simultaneously with NEW stage-2 track ids.
    out = {}
    for _ in range(sim.g.cfg.warmup_frames + 2):
        out = sim.tick({
            71: (similar_to(ea, 0.05, seed=1), 0, pa + 5),
            72: (similar_to(eb, 0.05, seed=2), 1, pb + 5),
        })

    assert out[71] == gid_a, f"player A lost identity: {gid_a} -> {out[71]}"
    assert out[72] == gid_b, f"player B lost identity: {gid_b} -> {out[72]}"


def test_unrelated_appearance_creates_new_identity():
    sim = Sim()
    gid1 = warm(sim, lid=1, e=emb(1), team=0, pos=np.array([10.0, 10.0]))
    for _ in range(60):
        sim.tick({})
    # Orthogonal descriptor -> must not be matched.
    gid2 = warm(sim, lid=2, e=emb(999), team=0, pos=np.array([12.0, 12.0]))
    assert gid2 != gid1


def test_roster_cap_is_respected():
    """A team may never accumulate more than max_roster_per_team identities."""
    cfg = GalleryConfig(warmup_frames=2, max_roster_per_team=3, require_team_match=True)
    sim = Sim(cfg)

    made = []
    for i in range(6):
        gid = warm(sim, lid=100 + i, e=emb(i), team=0, pos=np.array([float(i), 0.0]))
        if gid is not None:
            made.append(gid)
        # Retire the previous track so it stops occupying an active slot.
        sim.tick({})

    live = [i for i in sim.g.identities.values() if i.team == 0]
    assert len(live) <= cfg.max_roster_per_team + 1, (
        f"roster overflow: {len(live)} identities for team 0"
    )


def test_position_gate_rejects_far_appearance_match():
    """A similar-looking player on the far side of the pitch must not match."""
    cfg = GalleryConfig(
        warmup_frames=3,
        position_gate_near=20.0,
        position_gate_far=40.0,
        position_gate_frames=10,
    )
    sim = Sim(cfg)
    e = emb(5)

    gid1 = warm(sim, lid=1, e=e, team=0, pos=np.array([10.0, 10.0]))
    for _ in range(30):
        sim.tick({})                       # lost -> gate is still narrow-ish

    # Reappears with matching appearance but 500 pitch units away.
    gid2 = warm(sim, lid=2, e=similar_to(e, 0.03), team=0, pos=np.array([510.0, 10.0]))
    assert gid2 != gid1, "position gate failed to reject a distant match"


def test_identity_becomes_recoverable_after_track_loss():
    """An identity bound to a live track must not be offered to others."""
    sim = Sim()
    e = emb(3)
    gid1 = warm(sim, lid=1, e=e, team=0, pos=np.array([10.0, 10.0]))
    assert gid1 is not None

    # Track 1 is still alive -> identity is owned, so a newcomer can't take it.
    out = sim.tick({1: (e, 0, np.array([10.0, 10.0])), 2: (emb(55), 0, np.array([9.0, 9.0]))})
    assert out[1] == gid1
    assert sim.g.identities[gid1].active_local == 1


def test_summary_counts():
    sim = Sim()
    warm(sim, lid=1, e=emb(1), team=0, pos=np.array([1.0, 1.0]))
    s = sim.g.summary()
    assert s["total_identities"] == 1
    assert s["active"] == 1
    assert s["lost"] == 0
    assert s["new_ids"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
