# ⚽ Football Player Tracking & Re-Identification

Detect, track and **re-identify** football players across out-of-frame gaps,
then export an annotated video (H.264), a per-frame tracks CSV and a folder of
analytics charts (possession, passes, heatmaps, formations, …). Ships with a
Streamlit UI and a CLI.

The core problem this project solves: *a player who leaves the camera frame
and later re-enters gets a brand-new track ID from any standard MOT tracker,
breaking long-term analytics (possession, distance covered, formations).*

---

## Quick start

```bash
git clone https://github.com/Bishalai/Football_Analysis_Object_Tracking.git
cd Football_Analysis_Object_Tracking
pip install -r requirements.txt

# The repository is code-only: the weights (~330 MB) and the sample video
# are downloaded from Hugging Face on first use — no git-lfs involved.
# Optionally prefetch everything up front:
python -m pipeline.assets

# UI
streamlit run app.py

# CLI
python run_tracking.py "assets/test_vid.mp4" --out output
```

> **Python:** developed and tested on 3.10. See *Gotchas* below —
> pytest is pinned to `<9` because of a Python 3.10.0 stdlib bug.

---

## How it works

Three stages, each in its own module:

```
video frame
   │
   ▼
┌─────────────────────┐
│ 1. DETECTION        │  pipeline/detector.py
│    YOLOv8 players   │  classes resolved BY NAME, never by index
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 2. FRAME-TO-FRAME   │  pipeline/tracker.py
│    MOT (boxmot)     │  BoT-SORT / BoostTrack + camera-motion comp.
│    -> local track   │  ids that are regenerated whenever a player
│       ids           │  leaves the frame
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 3. IDENTITY GALLERY │  pipeline/gallery.py   ★ the fix
│    long-term re-ID  │  keeps a bank of player identities with an
│    -> persistent    │  averaged appearance embedding, team and last
│       global ids    │  known pitch position; re-binds returning tracks
└─────────────────────┘
```

### Why stage 3 exists

A frame-to-frame tracker only ever sees the current frame. When a player
walks off-screen the tracker drops the track; when they return it mints a
fresh ID. Stage 3 holds identities **between** appearances and re-binds them
using four signals, in order of strictness:

| Signal | Rule |
|---|---|
| **Team** | Hard gate — a player never changes team mid-match |
| **Appearance** | Cosine similarity of a 512-d ReID embedding |
| **Pitch position** | Distance gate that *widens* with time spent off-screen |
| **Roster** | At most 11 live identities per team |

All simultaneously-arriving tracks are matched in one global pass, so two
players returning at once cannot claim the same identity.

New tracks must survive `--warmup` frames (default 8) before they may claim
an identity — this stops 1-frame false detections from polluting the gallery.

### Team classification

Unsupervised, no per-video setup: shirt-colour samples are collected over the
first ~30 frames, clustered with **extra** k-means capacity, small clusters
(goalkeeper / referee kits) are discarded, and the rest are merged down to
`n_teams`. Prototypes then drift with an EMA to track lighting changes.

Two details that matter for broadcast footage:

- grass pixels are rejected before taking the median (boxes are ~20×50 px and
  otherwise mostly background),
- colours are **not** unit-normalised — the two kits are often two shades of
  the same hue, and brightness is what separates them.

---

## Repository layout

```
├── app.py                  Streamlit UI
├── run_tracking.py         CLI
├── pipeline/               the library
│   ├── config.py           all paths + tunable thresholds (single source)
│   ├── assets.py           downloads weights/video from Hugging Face ★
│   ├── detector.py         Stage 1 — YOLOv8, class-by-name resolution
│   ├── tracker.py          Stage 2 — boxmot MOT + gallery binding
│   ├── gallery.py          Stage 3 — identity re-acquisition  ★
│   ├── reid.py             pluggable embedders (deepsort/osnet/resnet18)
│   ├── team.py             unsupervised team classification
│   ├── homography.py       frame -> pitch projection (EMA-stabilised)
│   ├── viz.py              tactical-map rendering
│   ├── analytics.py        charts from the tracks CSV  (viz/*.png)
│   └── runner.py           video loop -> CSV + annotated video + charts
├── tests/                  15 tests (pure numpy, no GPU needed)
├── assets/                 pitch landmarks, tactical map (+ sample video once fetched)
├── models/                 weights — git-ignored, fetched from Hugging Face
├── third_party/            deep_sort_pytorch ReID extractor code (weights in models/)
├── output/                 per-video outputs (git-ignored)
└── etc/                    archived / superseded material (git-ignored, see below)
```

### Weights live on Hugging Face, not in git

The GitHub repository contains **code only**. Every binary is fetched from
[`Ichigo369/football-tracking-weights`](https://huggingface.co/Ichigo369/football-tracking-weights)
straight into its final path the first time it is needed (never twice):

| Local path | What | Size |
|---|---|---|
| `models/players_v8_best.pt` | YOLOv8 detector (ball/gk/player/referee) | 130 MB |
| `models/ball_v8_best.pt` | dedicated ball detector (off by default) | 84 MB |
| `models/keypoints_v8m_best.pt` | YOLOv8M pitch-keypoint model | 50 MB |
| `models/deepsort_ckpt.t7` | DeepSORT ReID extractor (default backend) | 44 MB |
| `models/osnet_x0_25_msmt17.pt` | OSNet-x0.25 (`osnet` backend) | 3 MB |
| `assets/test_vid.mp4` | sample clip for the UI checkbox | 21 MB |

```bash
python -m pipeline.assets                 # prefetch everything (~330 MB)
hf download Ichigo369/football-tracking-weights   # equivalent, manual
```

`Runner.__init__` calls `ensure_assets()` for exactly the files the current
configuration needs, so a plain clone + `pip install` is enough to run.
Offline: set `HF_HUB_OFFLINE=1` once the files are in place.

### What's in `etc/` (local only — not on GitHub)

`etc/` is git-ignored: none of it is needed at runtime, and it is only kept
locally for provenance. The repository ships just the pipeline, tests, assets
and the download logic — datasets (`playersv8/`, `ballv8/`), training runs,
metrics and notebooks are local-only too (see `.gitignore`).

| Folder | Contents |
|---|---|
| `legacy_scripts/` | the original `trial.py`, `converter.py`, plotting scripts |
| `notebooks/` | exploration notebooks |
| `data/` | raw + processed CSVs from the earlier pipeline, dataset YAMLs |
| `media/` | old result AVIs, debug probe images |
| `results/` | output of an earlier full-video run |
| `experiments/` | unused `yolov8_seg` segmentation experiment |
| `models/` | `Yolo8L Players` (training plots only — **no weights**) |
| `deepsort_extra/` | the original `YOLOv8-DeepSORT-Object-Tracking` repo shell, its vendored **ultralytics 8.0.3**, datasets and sample videos |
| `old_runs/`, `visualizations/` | earlier outputs and plots (the 15 reference charts) |

> **Why the DeepSORT repo was trimmed:** it vendored ultralytics 8.0.3, which
> silently shadows the installed 8.4.x if that directory lands on `sys.path`.
> Only `third_party/deep_sort_pytorch/` is needed, so only that was kept.

---

## CLI

```bash
python run_tracking.py INPUT.mp4 [options]

  --out DIR            output directory (default: output)
  --max-frames N       process only the first N frames
  --conf 0.5           detection confidence
  --imgsz 960          detection input size
  --tracker boosttrack boosttrack | botsort | bytetrack | ocsort | ...
  --max-age 60         frames a lost stage-2 track stays alive
  --reid deepsort      deepsort | osnet | resnet18
  --warmup 8           frames before a new track may be re-identified
  --no-video           skip annotated video export
  --no-viz             skip the analytics charts
  --quiet              suppress progress output
```

### Output layout — one folder per input video

Everything produced for a video goes into its own folder, so runs on
different videos never overwrite each other:

```
output/
└── test_vid/
    ├── test_vid_tracked.mp4     annotated video (H.264, plays in browsers)
    ├── test_vid_tracks.csv      per-frame tracks
    └── viz/                     analytics charts (PNG)
        ├── possession.png
        ├── ball_passes.png
        ├── player_touches.png
        ├── pass_network.png
        ├── passmap_team_0.png / passmap_team_1.png
        ├── heatmap_team_0.png / heatmap_team_1.png
        ├── formations.png
        ├── convex_hull.png
        └── tactical_map_positions.png
```

### Analytics charts

`pipeline/analytics.py` rebuilds the charts from the tracks CSV alone
(`generate_viz(csv, out_dir)`), so you can re-run them without re-running
tracking:

| Chart | Derived from |
|---|---|
| `possession.png` | team share of ball ownership over time |
| `ball_passes.png` | ball path coloured by the team in possession |
| `player_touches.png` | touches per player (an owner change = a touch) |
| `pass_network.png` | mean positions + pass links per team |
| `passmap_team_X.png` | one team's pass arrows |
| `heatmap_team_X.png` | positioning density per team |
| `formations.png` | average positions in time quarters |
| `convex_hull.png` | territory covered per team |
| `tactical_map_positions.png` | all mean positions on the tactical map |

**Possession / passes / touches are inferred**, not read from event data:
the player nearest the ball within `touch_radius` (30 pitch units) owns it;
an owner change is a touch; a same-team owner change within
`max_possession_gap` (45) frames is a pass. Runs shorter than `min_hold` (4)
frames are flicker and get merged away. Thresholds live in `AnalyticsConfig`
(`pipeline/config.py`). Charts are skipped gracefully when the CSV has no
pitch projection.

### CSV columns

`frame, gid, local_id, team, x1, y1, x2, y2, conf, pitch_x, pitch_y`

- `gid` — persistent identity (blank during warm-up, `ball` for the ball)
- `local_id` — stage-2 track id; changes when a player re-enters, `gid` does not
- `pitch_x/y` — position on the tactical map (blank if projection unavailable)

---

## Web UI

```bash
streamlit run app.py
```

Upload a clip (or tick **Use sample video** — fetched from Hugging Face on
first use) and press **▶ Run tracking**. The
models are loaded once via `st.cache_resource` and the `Runner` is reused
(`Runner.reset()` clears per-video state), so changing settings never reloads
weights.

| Tab | What you get |
|---|---|
| 🎬 Annotated video | inline playback (H.264, plays in the browser) + download |
| 📊 Visualisations | grid of the `viz/*.png` charts, per-chart download, or all as a `.zip` |
| 🗺️ Tactical map | scrub through projected positions with a ball trail |
| 📊 Track data | full tracks table + CSV download |
| 👕 Teams | per-team roster with first/last frame seen |

Sidebar knobs: stage-2 tracker, appearance backend, detection confidence,
re-ID warm-up, lost-track lifetime, max frames.

---

## Tests

```bash
pytest tests/ -q
```

15 tests, all pure numpy — no GPU, no video:

- `test_gallery.py` (10) — the re-identification contract: identity survives
  an out-of-frame gap, team gate blocks cross-team matches, roster cap holds,
  warm-up suppresses transients, two simultaneous re-entries don't collide,
  position gate rejects distant look-alikes.
- `test_reid.py` (5) — every embedding backend loads and emits unit-norm
  vectors (weights are downloaded on demand, or the test skips offline);
  all model paths point at the flat `models/` layout; the DeepSORT
  checkpoint is registered in the Hugging Face manifest.

---

## Tuning

Everything lives in `pipeline/config.py`:

| Knob | Default | Effect |
|---|---|---|
| `GalleryConfig.warmup_frames` | 8 | ↑ fewer spurious IDs, slower re-acquisition |
| `GalleryConfig.min_appearance_sim` | 0.55 | ↑ stricter appearance matching |
| `GalleryConfig.lost_after_frames` | 15 | how long an identity waits before becoming re-bindable |
| `GalleryConfig.position_gate_near/far` | 40 / 160 | pitch-space search radius over time off-screen |
| `GalleryConfig.max_roster_per_team` | 11 | hard football constraint |
| `TrackerConfig.max_age` | 60 | stage-2 track lifetime while occluded |
| `TeamConfig.bootstrap_frames` | 30 | frames of evidence before teams are fixed |
| `AnalyticsConfig.touch_radius` | 30 | pitch units within which a player counts as ball owner |
| `AnalyticsConfig.max_possession_gap` | 45 | frames an owner keeps the ball through detector dropouts / max gap for a "pass" |
| `AnalyticsConfig.min_hold` | 4 | owner runs shorter than this are flicker and get merged |
| `AnalyticsConfig.n_intervals` / `formation_intervals` | 6 / 4 | buckets in the possession chart / panels in the formations chart |

---

## Requirements

`pip install -r requirements.txt` — `torch`, `torchvision`, `ultralytics`,
`opencv-python`, `boxmot`, `huggingface_hub`, `numpy`, `pandas`,
`matplotlib`, `scikit-learn`, `scipy`, `streamlit`, `pytest<9`.

CUDA is used automatically when available (developed on an RTX 4060).

### Large files (Hugging Face)

No binaries are stored on GitHub — not even via git-lfs. The six weights /
sample-video files (~330 MB total) live in the public HF repo
[`Ichigo369/football-tracking-weights`](https://huggingface.co/Ichigo369/football-tracking-weights)
and are downloaded on first use by `pipeline/assets.py` into `models/` and
`assets/`. `python -m pipeline.assets` prefetches them all; deleting any
local file simply makes the next run fetch it again.

> Datasets, `etc/`, `output/`, training runs and metrics are **not** uploaded
> anywhere — see `.gitignore`.

---

## Gotchas

- **pytest is pinned to `<9`.** Python 3.10.0's `dataclasses` has a bug where
  `@dataclass(slots=True)` never assigns `init=False` fields with defaults;
  pytest 9.2's `NodeId` depends on that and fails with
  `AttributeError: 'NodeId' object has no attribute '_str_cache'`. Upgrading
  Python past 3.10.0 lets you drop the pin.
- **Class indices differ per model.** The trained players model is
  `{0: ball, 1: goalkeeper, 2: player, 3: referee}`. The pipeline resolves
  class semantics **by name** from the loaded model — never hardcode indices.
- **Weights paths.** All PyTorch weights are flat under `models/`
  (`players_v8_best.pt`, `ball_v8_best.pt`, `keypoints_v8m_best.pt`,
  `deepsort_ckpt.t7`, `osnet_x0_25_msmt17.pt`) and are downloaded from
  Hugging Face on first use — nothing binary is committed.
- **The dedicated ball model is off by default** (`use_ball_model=False`):
  it was trained on different footage and peaks at ~0.1 confidence here,
  while the players model scores 0.75–0.87 on the real ball.

### Known limitations

- Ball detections occasionally fire on white crowd specks at broadcast zoom;
  a pitch-projection filter rejects some but not all of them.
- The dedicated ball weights (`ballv8`) do not generalise to the sample clip.
- Team assignment is per-video unsupervised; heavily mixed or similar kits
  (e.g. two shades of blue) are separated by brightness, which can fail under
  strong shadows.
