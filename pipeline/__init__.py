"""Football tracking & analytics pipeline.

Stage 1  Detection   -> detector.py
Stage 2  MOT         -> tracker.py (boxmot BoT-SORT/BoostTrack, frame-to-frame)
Stage 3  ReID        -> gallery.py (long-term identity across out-of-frame gaps)
"""

__all__ = [
    "config",
    "detector",
    "homography",
    "reid",
    "team",
    "gallery",
    "tracker",
    "runner",
]

__version__ = "0.1.0"
