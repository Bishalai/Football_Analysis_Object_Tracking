"""Streamlit UI for the football tracking & analytics pipeline.

Run with:

    streamlit run app.py

The heavy models are loaded once via ``st.cache_resource`` and the Runner is
reused across runs (``Runner.reset()`` clears per-video state), so switching
videos or settings does not reload weights.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pipeline.assets import AssetError, ensure_assets
from pipeline.config import PipelineConfig
from pipeline.runner import Runner
from pipeline.viz import TacticalMap

st.set_page_config(
    page_title="Football Tracking & Analytics",
    page_icon="⚽",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading detection / ReID models…")
def get_runner() -> Runner:
    """Load YOLO + ReID weights once per session."""
    return Runner(PipelineConfig(), annotate=True)


@st.cache_resource(show_spinner=False)
def get_tactical_map() -> TacticalMap:
    return TacticalMap()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚽ Football Tracking")
st.sidebar.caption("Detection → MOT → long-term ReID")

uploaded = st.sidebar.file_uploader(
    "Upload a video", type=["mp4", "avi", "mov", "mkv"],
    help="Broadcast or tactical-camera footage.",
)
use_sample = st.sidebar.checkbox(
    "Use sample video", value=uploaded is None,
    help="assets/test_vid.mp4 — MCI vs CHE broadcast clip.",
)

with st.sidebar.expander("⚙️ Settings", expanded=True):
    tracker = st.selectbox(
        "Stage-2 tracker", ["boosttrack", "botsort", "bytetrack", "ocsort"],
        index=0, help="boxmot algorithm for frame-to-frame association.",
    )
    reid = st.selectbox(
        "Appearance model", ["deepsort", "osnet", "resnet18"],
        index=0, help="Embeddings used by the identity gallery.",
    )
    conf = st.slider("Detection confidence", 0.10, 0.90, 0.50, 0.05)
    warmup = st.slider(
        "Re-ID warm-up (frames)", 2, 30, 8,
        help="A new track must survive this long before it may claim a "
             "persistent identity. Higher = fewer spurious IDs, slower "
             "re-acquisition.",
    )
    max_age = st.slider(
        "Lost-track lifetime", 10, 300, 60, 10,
        help="Frames a stage-2 track stays alive while occluded/off-screen.",
    )
    max_frames = st.number_input(
        "Max frames (0 = entire video)", 0, 100_000, 0, 100,
    )

run_btn = st.sidebar.button("▶  Run tracking", type="primary", width="stretch")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works**\n"
    "1. YOLOv8 detects players / GK / ref / ball\n"
    "2. BoT-SORT associates them frame-to-frame\n"
    "3. An **identity gallery** re-binds players who leave and "
    "re-enter the frame, using appearance + team + pitch position"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_video() -> Path | None:
    """Return a filesystem path for whichever source the user picked."""
    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / "football_ui"
        tmp.mkdir(parents=True, exist_ok=True)
        path = tmp / uploaded.name
        with open(path, "wb") as fh:
            fh.write(uploaded.getbuffer())
        return path
    if use_sample:
        sample = PipelineConfig().paths.sample_video
        if not sample.exists():
            # Not stored in git -- fetch it from Hugging Face on first use.
            try:
                ensure_assets("assets/test_vid.mp4", quiet=True)
            except AssetError as exc:
                st.error(f"Could not fetch the sample video.\n\n{exc}")
                return None
        if sample.exists():
            return sample
        st.error(f"Sample video missing: {sample}")
    return None


@st.cache_data(show_spinner=False)
def load_tracks_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("pitch_x", "pitch_y", "team", "conf"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def tactical_frames(df: pd.DataFrame, max_points: int = 400) -> list:
    """Build one tactical-map image per sampled frame from the CSV."""
    tmap = get_tactical_map()
    players = df[df["gid"].astype(str) != "ball"].copy()
    balls = df[df["gid"].astype(str) == "ball"].copy()

    frames = sorted(df["frame"].unique())
    step = max(1, len(frames) // max_points)
    out = []
    ball_trail: list[tuple[float, float]] = []

    for f in frames[::step]:
        pf = players[players["frame"] == f].dropna(subset=["pitch_x", "pitch_y"])
        bf = balls[balls["frame"] == f].dropna(subset=["pitch_x", "pitch_y"])

        positions = list(zip(pf["pitch_x"], pf["pitch_y"]))
        teams = [None if pd.isna(t) else int(t) for t in pf["team"]]
        labels = [f"#{int(g)}" if not pd.isna(g) else "" for g in pf["gid"]]

        ball_pos = None
        if len(bf):
            ball_pos = (float(bf.iloc[0]["pitch_x"]), float(bf.iloc[0]["pitch_y"]))
            ball_trail.append(ball_pos)
            ball_trail = ball_trail[-60:]

        out.append(tmap.render(positions, teams, labels, ball_pos, list(ball_trail)))
    return out


def rosters_table(df: pd.DataFrame) -> pd.DataFrame:
    p = df[df["gid"].astype(str) != "ball"].dropna(subset=["team"])
    if p.empty:
        return pd.DataFrame(columns=["Team", "Identity", "Frames seen"])
    g = (p.groupby(["team", "gid"])
           .agg(frames=("frame", "count"),
                first=("frame", "min"),
                last=("frame", "max"))
           .reset_index()
           .sort_values(["team", "gid"]))
    g["Team"] = g["team"].map({0: "A", 1: "B"}).fillna("?")
    g["Identity"] = "#" + g["gid"].astype(int).astype(str)
    g.rename(columns={"frames": "Frames seen", "first": "First frame",
                      "last": "Last frame"}, inplace=True)
    return g[["Team", "Identity", "Frames seen", "First frame", "Last frame"]]


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.title("⚽ Football Player Tracking & Re-Identification")

video_path = resolve_video()

if video_path is None:
    st.info("👈 Upload a video (or tick *Use sample video*) in the sidebar, "
            "then press **Run tracking**.")
    st.stop()

st.caption(f"Source: `{video_path.name}`")

if not run_btn and "last_run" not in st.session_state:
    st.markdown(
        "Press **▶ Run tracking** in the sidebar to start.\n\n"
        "The pipeline detects players, tracks them frame-to-frame, and "
        "re-identifies anyone who leaves and re-enters the frame."
    )
    st.stop()

# ----------------------------- run ---------------------------------------- #
if run_btn or "last_run" not in st.session_state:
    cfg = PipelineConfig()
    cfg.tracker.spec = tracker
    cfg.reid.backend = reid
    cfg.detection.conf = conf
    cfg.gallery.warmup_frames = warmup
    cfg.tracker.max_age = max_age

    runner = get_runner()
    runner.cfg = cfg
    runner.tracker.cfg = cfg.tracker
    runner.tracker.gallery.cfg = cfg.gallery
    runner.reset()

    out_dir = Path(tempfile.gettempdir()) / "football_ui_out"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress = st.progress(0.0, text="Starting…")
    status = st.empty()
    t0 = time.perf_counter()

    def on_progress(done: int, total: int, fps: float) -> None:
        if total > 0:
            progress.progress(min(done / total, 1.0),
                              text=f"Frame {done}/{total} · {fps:.1f} fps")
        elif done % 10 == 0:
            progress.progress(0.0, text=f"Frame {done} · {fps:.1f} fps")

    try:
        stats = runner.run(
            video_path, out_dir=out_dir,
            max_frames=int(max_frames) or None,
            progress=False, on_progress=on_progress,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
        st.error(f"Tracking failed: `{type(exc).__name__}: {exc}`")
        st.stop()

    progress.progress(1.0, text="Done")
    summary = runner.tracker.gallery.summary()

    st.session_state["last_run"] = {
        "stats": stats, "summary": summary,
        "video": (str(stats.video_path)
                  if stats.video_path and stats.video_path.exists() else None),
        "csv": (str(stats.csv_path)
                if stats.csv_path and stats.csv_path.exists() else None),
        "viz": [str(p) for p in stats.viz_files if Path(p).exists()],
        "out_dir": str(stats.output_dir) if stats.output_dir else None,
        "rosters": runner.tracker.gallery.team_rosters(),
        "elapsed": time.perf_counter() - t0,
    }
    status.empty()

run = st.session_state["last_run"]
stats = run["stats"]
summary = run["summary"]

# ----------------------------- metrics ------------------------------------ #
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Frames", f"{stats.frames}")
c2.metric("Speed", f"{stats.fps:.1f} fps")
c3.metric("Identities", summary["total_identities"])
c4.metric("Re-identified", summary["recovered_ids"])
c5.metric("Row count", f"{len(stats.rows):,}")

st.caption(
    "**Re-identified** counts how many times a track that had left the frame "
    "was successfully bound back to an existing identity — the core feature "
    "of this pipeline."
)

# ----------------------------- tabs --------------------------------------- #
tab_video, tab_viz, tab_map, tab_data, tab_teams = st.tabs(
    ["🎬 Annotated video", "📊 Visualisations", "🗺️ Tactical map",
     "📊 Track data", "👕 Teams"]
)

with tab_video:
    if run["video"]:
        if run.get("out_dir"):
            st.caption(f"Output folder: `{run['out_dir']}`")
        st.video(run["video"])
        with open(run["video"], "rb") as fh:
            st.download_button("⬇️ Download annotated video", fh,
                               file_name=Path(run["video"]).name,
                               mime="video/mp4")
    else:
        st.warning("Annotated video was not produced.")

with tab_viz:
    viz_files: list[str] = run.get("viz") or []
    if not viz_files:
        st.info("No charts were generated for this video. Charts are written "
                "to `<output>/<video name>/viz/` after each run.")
    else:
        if run.get("out_dir"):
            st.caption(f"Output folder: `{run['out_dir']}`")
        st.caption(
            "**Possession / passes / touches** are derived from ball "
            "possession (nearest player owns the ball); **heatmap, pass "
            "network, formations, convex hull** come from tracked positions."
        )
        # One download button per chart, plus the grid of images.
        cols = st.columns(3)
        for i, path in enumerate(viz_files):
            p = Path(path)
            with cols[i % 3]:
                st.image(str(p), width="stretch", caption=p.stem)
                with open(p, "rb") as fh:
                    st.download_button(
                        f"⬇️ {p.stem}", fh, file_name=p.name,
                        mime="image/png", key=f"dl_viz_{i}",
                    )
        if len(viz_files) > 1:
            import io
            import zipfile

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in viz_files:
                    zf.write(path, arcname=Path(path).name)
            st.download_button(
                "⬇️ Download all charts (.zip)", buf.getvalue(),
                file_name="charts.zip", mime="application/zip",
            )

with tab_map:
    if run["csv"]:
        df = load_tracks_csv(run["csv"])
        if df["pitch_x"].notna().any():
            frames = tactical_frames(df)
            if frames:
                idx = st.slider("Tactical frame", 0, len(frames) - 1, len(frames) // 2)
                st.image(frames[idx], channels="BGR",
                         caption=f"Pitch projection · {len(frames)} sampled frames")
                st.image(get_tactical_map().legend(), channels="BGR",
                         width="content")
            else:
                st.info("Not enough projected positions to render.")
        else:
            st.info("No pitch projection available — the pitch-keypoint "
                    "model did not find enough landmarks in this footage.")
    else:
        st.info("Run the pipeline first.")

with tab_data:
    if run["csv"]:
        df = load_tracks_csv(run["csv"])
        st.dataframe(df, width="stretch", height=420)
        with open(run["csv"], "rb") as fh:
            st.download_button("⬇️ Download tracks CSV", fh,
                               file_name=Path(run["csv"]).name,
                               mime="text/csv")
    else:
        st.info("Run the pipeline first.")

with tab_teams:
    if run["csv"]:
        df = load_tracks_csv(run["csv"])
        table = rosters_table(df)
        if table.empty:
            st.info("No team assignments yet.")
        else:
            left, right = st.columns(2)
            for col, name in ((left, "Team A"), (right, "Team B")):
                sub = table[table["Team"] == name[-1]]
                with col:
                    st.markdown(f"#### {name} — {len(sub)} identities")
                    st.dataframe(sub.drop(columns=["Team"]),
                                 width="stretch", hide_index=True)
            st.markdown("---")
            st.markdown("#### All identities")
            st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.info("Run the pipeline first.")
