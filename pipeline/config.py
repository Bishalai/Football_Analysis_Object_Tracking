"""Central paths and tunable thresholds for the tracking pipeline.

Every magic number that used to live inline in trial.py / converter.py is
collected here so behaviour can be tuned in one place (and exposed in the UI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The DeepSORT ReID *code* lives under third_party/. Its checkpoint does not
# anymore -- every binary now sits in models/ and is fetched from Hugging Face
# on first use (see pipeline/assets.py).
# NOTE: only this directory is put on sys.path. The original
# YOLOv8-DeepSORT-Object-Tracking repo bundled an ultralytics 8.0.3 that
# shadows the installed 8.4.x if imported from its root -- that vendored copy
# has been archived to etc/deepsort_extra/ for exactly this reason.
DEEPSORT_ROOT = PROJECT_ROOT / "third_party"


@dataclass(frozen=True)
class Paths:
    """Resolved model / data locations.

    Nothing here ships inside the git repository: the binaries are downloaded
    from ``https://huggingface.co/Ichigo369/football-tracking-weights`` into
    ``models/`` and ``assets/`` the first time they are needed
    (:func:`pipeline.assets.ensure_assets`, called by ``Runner``).
    """

    # YOLOv8 weights, all flat under models/.
    players_weights: Path = PROJECT_ROOT / "models" / "players_v8_best.pt"
    ball_weights: Path = PROJECT_ROOT / "models" / "ball_v8_best.pt"
    keypoints_weights: Path = PROJECT_ROOT / "models" / "keypoints_v8m_best.pt"

    pitch_keypoints_json: Path = PROJECT_ROOT / "assets" / "pitch map labels position.json"
    tactical_map: Path = PROJECT_ROOT / "assets" / "tactical map.jpg"
    sample_video: Path = PROJECT_ROOT / "assets" / "test_vid.mp4"

    osnet_weights: Path = PROJECT_ROOT / "models" / "osnet_x0_25_msmt17.pt"
    deepsort_root: Path = DEEPSORT_ROOT
    deepsort_ckpt: Path = PROJECT_ROOT / "models" / "deepsort_ckpt.t7"


# Class names are resolved BY NAME from the model, never by hardcoded index:
#   playersv8 model -> {0: ball, 1: goalkeeper, 2: player, 3: referee}
#   trial.py assumed {0: player, 1: referee, 2: ball} which is wrong.
PLAYER_CLASS_NAMES = ("player",)
GOALKEEPER_CLASS_NAMES = ("goalkeeper", "goalie", "gk")
REFEREE_CLASS_NAMES = ("referee", "official")
BALL_CLASS_NAMES = ("ball", "ball_in", "ball_out")


@dataclass
class DetectionConfig:
    conf: float = 0.50
    imgsz: int = 960
    # Run the pitch-keypoint model every N frames and interpolate the
    # homography in between (camera moves slowly -> ~halves compute).
    keypoint_interval: int = 5
    keypoint_conf: float = 0.70
    # Reject detections whose implied speed exceeds a sprint ceiling.
    max_speed_px_per_frame: float = 60.0
    # Use the dedicated ball model instead of the players model's `ball`
    # class. Default OFF: on the sample clip (test_vid.mp4) the dedicated ballv8 weights are
    # badly mismatched to the footage (peak conf ~0.1, and their top hit at
    # f0 is a crowd speck), while the players model scores 0.75-0.87 on the
    # real ball. Its only flaw is occasional crowd false-positives.
    use_ball_model: bool = False


@dataclass
class TrackerConfig:
    # boxmot spec name: "boosttrack", "botsort", "bytetrack", "deepocsort", ...
    spec: str = "boosttrack"
    # Frames a lost track stays alive inside stage 2 (short occlusion).
    max_age: int = 60
    min_hits: int = 3
    use_cmc: bool = True  # camera-motion compensation: essential for broadcast pans
    # boxmot can run its own appearance ReID; we do appearance ourselves in
    # stage 3, so its default-on OSNet pass is wasted compute.
    use_boxmot_reid: bool = False


@dataclass
class ReIDConfig:
    # "deepsort" (local ckpt, 512-d), "osnet" (boxmot weights), "resnet18".
    backend: str = "deepsort"
    device: str = "cuda"


@dataclass
class GalleryConfig:
    """Stage 3: long-term identity re-acquisition across out-of-frame gaps."""

    # A new stage-2 track must survive this many frames before we try to
    # re-identify it (stops 1-frame false detections polluting identities).
    warmup_frames: int = 8

    # Appearance gate: cosine similarity must reach this to be a candidate.
    min_appearance_sim: float = 0.55

    # Cost weights (combined into a single scalar for Hungarian assignment).
    w_appearance: float = 1.0
    w_position: float = 0.30
    w_age: float = 0.10

    # Position gate (pitch units) applied when a track was recently lost.
    # Widens with age because a player can travel while off-screen.
    position_gate_near: float = 40.0     # <= 5 s ago
    position_gate_far: float = 160.0     # older than that
    position_gate_frames: int = 150      # frames to ramp near -> far

    # Retire an identity from the gallery after this many frames unseen.
    max_age_frames: int = 30 * 60        # ~60 s at 30 fps

    # Mark a stage-2 track "lost" (unbind from identity) after this long.
    lost_after_frames: int = 15          # ~0.5 s

    # Hard football constraint: never assign more than this per team.
    max_roster_per_team: int = 11

    # Team must agree for a match when both sides are confident.
    require_team_match: bool = True
    team_confident_after: int = 5        # frames of evidence before gating on team


@dataclass
class TeamConfig:
    # "kmeans" (unsupervised, no per-video setup) or "fixed" (hardcoded colors).
    mode: str = "kmeans"
    n_teams: int = 2
    # Frames of detections collected before the k-means prototypes are fixed.
    bootstrap_frames: int = 30
    # EMA rate for tracking the prototypes over time (lighting drift).
    ema_alpha: float = 0.05
    # Reject samples further than this (RGB euclidean) from the prototype
    # they were assigned to -- keeps GK/referee kits from polluting a team.
    update_max_dist: float = 70.0


@dataclass
class AnalyticsConfig:
    """Post-run charts (pipeline/analytics.py) derived from the tracks CSV."""

    # Buckets in the possession line chart (legacy poss.png style).
    n_intervals: int = 6
    # Panels in the average-positions chart (legacy form1-4.png).
    formation_intervals: int = 4
    # Pitch units within which the nearest player counts as ball owner.
    touch_radius: float = 30.0
    # Frames an owner keeps possession while the ball detector drops out.
    max_possession_gap: int = 45
    # Owner runs shorter than this are flicker and get merged away.
    min_hold: int = 4


@dataclass
class PipelineConfig:
    paths: Paths = field(default_factory=Paths)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    gallery: GalleryConfig = field(default_factory=GalleryConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
