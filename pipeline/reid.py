"""Appearance embeddings for player re-identification.

All backends expose the same contract:

    embed(crops_bgr) -> np.ndarray of shape (N, D), L2-normalised rows

Backends
--------
deepsort  46MB ckpt.t7 fetched from Hugging Face into models/ on first use.
          512-d, Market1501-trained. Self-contained (imports no
          ultralytics), so it is the default.
osnet     boxmot's OSNet-x0.25 encoder. Also 512-d.
resnet18  torchvision ImageNet features. Fallback if neither is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .config import ReIDConfig, Paths


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def _to_rgb(crops_bgr: list[np.ndarray]) -> list[np.ndarray]:
    import cv2

    return [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if c.ndim == 3 else c for c in crops_bgr]


class Embedder:
    """Base: subclasses implement _encode()."""

    backend: str = "base"
    dim: int = 0

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, self.dim), dtype=np.float32)
        feats = self._encode(crops_bgr)
        return _l2_normalize(np.asarray(feats, dtype=np.float32))

    def _encode(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class DeepSortEmbedder(Embedder):
    """Uses deep_sort_pytorch's Extractor from the YOLOv8-DeepSORT repo."""

    backend = "deepsort"
    dim = 512

    def __init__(self, paths: Paths, device: str = "cuda"):
        root = paths.deepsort_root
        if not root.exists():
            raise FileNotFoundError(f"deep_sort_pytorch root not found: {root}")
        if not paths.deepsort_ckpt.exists():
            raise FileNotFoundError(f"DeepSORT ReID checkpoint not found: {paths.deepsort_ckpt}")

        # Add ONLY the detect/ dir -- never the repo root, which holds a vendored
        # ultralytics 8.0.3 that would shadow the installed 8.4.x.
        root_s = str(root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)

        from deep_sort_pytorch.deep_sort.deep.feature_extractor import Extractor

        self._extractor = Extractor(str(paths.deepsort_ckpt), use_cuda=(device != "cpu"))

    def _encode(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        # Extractor expects RGB crops; it resizes to 64x128 internally.
        return self._extractor(_to_rgb(crops_bgr))


class OsnetEmbedder(Embedder):
    """boxmot's OSNet encoder via its canonical Frame/Detections types."""

    backend = "osnet"
    dim = 512

    def __init__(self, device: str = "cuda"):
        from boxmot.reid import create_reid_encoder

        self._enc = create_reid_encoder("osnet-x0-25-msmt17", device=device)
        self.dim = int(self._enc.embedding_dim)

    def _encode(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        import cv2
        import torch
        from boxmot.structures import Boxes, Detections, Frame

        # Frame.image must be a CPU-contiguous torch.uint8 RGB CHW tensor.
        frames, detections = [], []
        for i, crop in enumerate(crops_bgr):
            h, w = crop.shape[:2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.ndim == 3 else crop
            chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
            frames.append(Frame(image=torch.from_numpy(chw), sample_id=f"crop{i}"))
            detections.append(
                Detections(
                    geometry=Boxes(
                        torch.tensor([[0.0, 0.0, float(w), float(h)]], dtype=torch.float32)
                    ),
                    scores=torch.tensor([1.0], dtype=torch.float32),
                    class_ids=torch.tensor([0], dtype=torch.int64),
                    sample_id=f"crop{i}",
                )
            )
        feats = self._enc.encode(frames, detections)
        return torch.cat(feats, dim=0).cpu().numpy()


class ResNetEmbedder(Embedder):
    """ImageNet ResNet18 pooled features. Always available, no extra deps."""

    backend = "resnet18"
    dim = 512

    def __init__(self, device: str = "cuda"):
        import torch
        from torchvision.models import resnet18, ResNet18_Weights

        self._device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Identity()          # (N, 512) pooled features
        self._net = net.eval().to(self._device)
        self._torch = torch

    def _encode(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        import cv2
        import torch

        t = self._torch
        batch = []
        for c in crops_bgr:
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if c.ndim == 3 else c
            rgb = cv2.resize(rgb, (128, 256))
            x = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std
            batch.append(np.transpose(x, (2, 0, 1)).astype(np.float32))
        tensor = t.from_numpy(np.stack(batch).astype(np.float32)).to(self._device)
        with t.no_grad():
            return self._net(tensor).cpu().numpy()


def build_embedder(cfg: ReIDConfig, paths: Paths) -> Embedder:
    """Instantiate the requested backend, falling back to the others."""
    order = [cfg.backend] + [b for b in ("deepsort", "osnet", "resnet18") if b != cfg.backend]
    errors: list[str] = []
    for name in order:
        try:
            if name == "deepsort":
                return DeepSortEmbedder(paths, cfg.device)
            if name == "osnet":
                return OsnetEmbedder(cfg.device)
            if name == "resnet18":
                return ResNetEmbedder(cfg.device)
        except Exception as exc:  # noqa: BLE001 - report and fall through
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("No ReID backend could be initialised:\n  " + "\n  ".join(errors))
