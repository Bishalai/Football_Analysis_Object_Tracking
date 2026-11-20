"""Download the model weights and the sample video from Hugging Face.

The GitHub repository holds **code only**. Every binary (≈330 MB) lives in

    https://huggingface.co/Ichigo369/football-tracking-weights

and is fetched on first use straight into its final location (``models/``,
``assets/``), so a plain ``git clone`` + ``pip install -r requirements.txt``
is enough to run the pipeline. Files are never re-downloaded once they exist.

    from pipeline.assets import ensure_assets
    ensure_assets()                      # everything
    ensure_assets("assets/test_vid.mp4") # just the sample clip

Command line::

    python -m pipeline.assets            # prefetch every binary
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .config import PROJECT_ROOT

#: Public Hugging Face repo holding every binary this project needs.
HF_REPO_ID = "Ichigo369/football-tracking-weights"

#: Local path (relative to the project root) -> file name in ``HF_REPO_ID``.
#: The basenames match, which keeps the mapping trivial to audit.
ASSETS: dict[str, str] = {
    # YOLOv8 detection weights (players / goalkeeper / referee / ball)
    "models/players_v8_best.pt": "players_v8_best.pt",
    # dedicated ball detector -- off by default (DetectionConfig.use_ball_model)
    "models/ball_v8_best.pt": "ball_v8_best.pt",
    # YOLOv8 pitch-keypoint weights driving the homography
    "models/keypoints_v8m_best.pt": "keypoints_v8m_best.pt",
    # DeepSORT ReID extractor (512-d, Market1501) for the default backend
    "models/deepsort_ckpt.t7": "deepsort_ckpt.t7",
    # OSNet-x0.25 for the "osnet" backend
    "models/osnet_x0_25_msmt17.pt": "osnet_x0_25_msmt17.pt",
    # demo clip used by the "Use sample video" checkbox
    "assets/test_vid.mp4": "test_vid.mp4",
}


class AssetError(RuntimeError):
    """Raised when a required binary is missing and could not be fetched."""


def asset_path(relative: str) -> Path:
    """Absolute local path for a key of :data:`ASSETS` (or any repo-relative path)."""
    return PROJECT_ROOT / relative


def _offline_hint() -> str:
    if os.environ.get("HF_HUB_OFFLINE"):
        return ("HF_HUB_OFFLINE is set, so no download was attempted. "
                "Unset it, or fetch the files manually with:\n"
                f"    hf download {HF_REPO_ID}")
    return (f"Download it manually with:\n"
            f"    hf download {HF_REPO_ID}\n"
            f"or open https://huggingface.co/{HF_REPO_ID} in a browser.")


def _download(hf_name: str, target: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise AssetError(
            "huggingface_hub is not installed -- run\n"
            "    pip install -r requirements.txt\n" + _offline_hint()
        ) from exc

    try:
        # Downloads into the HF cache; we copy from there so that models/ and
        # assets/ stay clean (no .cache/ metadata directories).
        cached = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=hf_name))
    except Exception as exc:  # noqa: BLE001 - surface any hub/network failure
        raise AssetError(
            f"Could not fetch '{hf_name}' from huggingface.co/{HF_REPO_ID}: "
            f"{type(exc).__name__}: {exc}\n" + _offline_hint()
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, target)


def ensure_assets(
    only: str | list[str] | tuple[str, ...] | None = None,
    quiet: bool = False,
) -> list[Path]:
    """Make sure the requested binaries exist locally, downloading the rest.

    Parameters
    ----------
    only:
        Restrict to these keys of :data:`ASSETS` (repo-relative paths).
        ``None`` means every asset.
    quiet:
        Suppress the per-file progress line.

    Returns
    -------
    The local paths of everything that was requested (present or fetched).
    """
    keys = list(ASSETS) if only is None else ([only] if isinstance(only, str) else list(only))

    unknown = [k for k in keys if k not in ASSETS]
    if unknown:
        raise KeyError(f"unknown asset(s) {unknown}; known: {sorted(ASSETS)}")

    resolved: list[Path] = []
    for key in keys:
        target = asset_path(key)
        if not target.exists():
            if not quiet:
                print(f"  downloading {ASSETS[key]} -> {target.relative_to(PROJECT_ROOT)}")
            _download(ASSETS[key], target)
        elif not quiet:
            print(f"  found {target.relative_to(PROJECT_ROOT)}")
        resolved.append(target)
    return resolved


def missing(only: str | list[str] | None = None) -> list[str]:
    """Keys of :data:`ASSETS` whose local file does not exist yet."""
    keys = list(ASSETS) if only is None else ([only] if isinstance(only, str) else list(only))
    return [k for k in keys if not asset_path(k).exists()]


def main(argv: list[str] | None = None) -> int:
    """``python -m pipeline.assets [repo-relative-path ...]``"""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        ensure_assets(argv or None)
    except (AssetError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"all assets present under {PROJECT_ROOT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
