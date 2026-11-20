"""Smoke test: every ReID backend still loads, and the weights are where
``pipeline/config.py`` says they are.

The binaries no longer live in git -- they are fetched from Hugging Face into
``models/`` on first use (:mod:`pipeline.assets`). These tests therefore make
sure two things stay true after a restructure:

* the configured paths point at the flat ``models/`` layout, and
* a missing checkpoint is downloaded (or skipped, offline) instead of
  exploding with an opaque ``FileNotFoundError``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.assets import ASSETS, AssetError, ensure_assets, missing
from pipeline.config import PROJECT_ROOT, Paths, ReIDConfig
from pipeline.reid import build_embedder

BACKENDS = ["deepsort", "osnet", "resnet18"]

# backend -> asset key that must exist for it to load
_BACKEND_ASSET = {
    "deepsort": "models/deepsort_ckpt.t7",
    "osnet": "models/osnet_x0_25_msmt17.pt",
}


def _require(asset_key: str) -> None:
    """Make sure one asset is on disk, downloading it if possible."""
    if not missing(asset_key):
        return
    try:
        ensure_assets(asset_key, quiet=True)
    except AssetError as exc:  # offline / no token / hub unreachable
        pytest.skip(f"{asset_key} unavailable: {exc}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_embedder_loads_and_normalises(backend: str):
    if backend in _BACKEND_ASSET:
        _require(_BACKEND_ASSET[backend])

    crops = [(np.random.rand(128, 64, 3) * 255).astype("uint8") for _ in range(3)]
    emb = build_embedder(ReIDConfig(backend=backend), Paths())
    feats = emb.embed(crops)

    assert feats.shape == (3, emb.dim)
    norms = np.linalg.norm(feats, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"{backend} rows are not unit-norm"


def test_model_paths_use_flat_models_folder():
    """All PyTorch weights must live in models/, not scattered in run dirs."""
    p = Paths()
    assert p.deepsort_ckpt == PROJECT_ROOT / "models" / "deepsort_ckpt.t7"
    for path in (p.players_weights, p.ball_weights, p.keypoints_weights,
                 p.deepsort_ckpt, p.osnet_weights):
        assert path.parent.name == "models", f"{path} is not under models/"
        assert path.suffix in {".pt", ".t7"}, f"{path} is not a weight file"


def test_deepsort_checkpoint_is_registered_for_download():
    """The checkpoint must be in the Hugging Face manifest, under models/."""
    key = "models/deepsort_ckpt.t7"
    assert key in ASSETS
    assert ASSETS[key] == "deepsort_ckpt.t7"
    assert Paths().deepsort_ckpt.name == "deepsort_ckpt.t7"
