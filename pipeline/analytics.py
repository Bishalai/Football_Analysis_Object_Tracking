"""Post-run visualisations: regenerate the legacy ``etc/visualizations`` charts
from the per-frame tracks CSV produced by :class:`pipeline.runner.Runner`.

Every chart reads only::

    frame,gid,local_id,team,x1,y1,x2,y2,conf,pitch_x,pitch_y

and is written into ``<output>/<video stem>/viz/`` so each input video gets a
self-contained output folder (video + CSV + charts).

The original charts in ``etc/visualizations`` were built from StatsBomb event
data (mplsoccer + Sbopen). This module rebuilds the same chart families from
tracked positions instead:

    possession.png               ball possession % per interval  (poss.png)
    heatmap_team_X.png           positioning heatmaps            (heatmap*.png)
    convex_hull.png              territory convex hulls          (convex.png)
    formations.png               average positions per quarter   (form1-4.png)
    pass_network.png             derived pass networks           (Passnet.png)
    passmap_team_X.png           derived pass arrows             (chelseapassmap/city_pass)
    ball_passes.png              ball trajectory by possessor    (BallPasses.png)
    player_touches.png           touches per player              (player_touches.png)
    tactical_map_positions.png   avg positions on the pitch map  (tactical_map.drawio.png)

Derived events (touches / passes) are inferred from ball possession: the
player nearest the ball within ``touch_radius`` owns it; a change of owner is
a touch, and a same-team change within ``pass_max_gap`` frames is a pass.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: Streamlit / CLI must not open windows

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Polygon

from .config import AnalyticsConfig, Paths

# Team colours match the legacy charts (blue / red).
TEAM_COLORS = {0: "#1f77b4", 1: "#d62728"}
UNKNOWN_COLOR = "#8c8c8c"


# --------------------------------------------------------------------------- #
# Loading / bounds
# --------------------------------------------------------------------------- #
def load_tracks(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(players, ball)`` frames with usable pitch coordinates."""
    df = pd.read_csv(csv_path)
    for col in ("pitch_x", "pitch_y", "team", "conf"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    is_ball = df["gid"].astype(str).str.lower().eq("ball")
    ball = df[is_ball].dropna(subset=["pitch_x", "pitch_y"]).copy()
    players = df[~is_ball].dropna(subset=["pitch_x", "pitch_y"]).copy()
    return players, ball


def pitch_bounds(paths: Paths | None = None) -> tuple[float, float, float, float]:
    """``(x0, x1, y0, y1)`` of the tactical-map pitch, from the landmark JSON."""
    paths = paths or Paths()
    try:
        import json

        with open(paths.pitch_keypoints_json, encoding="utf-8") as fh:
            pts = list(json.load(fh).values())
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys))
    except Exception:  # noqa: BLE001 - fall back to a regulation-looking box
        return 15.0, 291.0, 15.0, 521.0


# --------------------------------------------------------------------------- #
# Pitch drawing (portrait, matching the tactical-map pixel space)
# --------------------------------------------------------------------------- #
def _draw_pitch(ax, b: tuple[float, float, float, float],
                color: str = "white") -> None:
    x0, x1, y0, y1 = b
    w, h = x1 - x0, y1 - y0
    ym = (y0 + y1) / 2
    xc = (x0 + x1) / 2
    line = dict(color=color, linewidth=1.3, zorder=3)

    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, **line))
    ax.plot([x0, x1], [ym, ym], **line)
    ax.add_patch(plt.Circle((xc, ym), 0.129 * w, fill=False, **line))
    ax.plot([xc], [ym], marker=".", color=color, markersize=3, zorder=3)

    # Fractions measured off the landmark JSON (the map is a stylised diagram,
    # so use *its* proportions, not regulation metres).
    box_inset, box_depth = 0.25 * w, 0.115 * h
    six_inset, six_depth = 0.366 * w, 0.040 * h
    spot = 0.105 * h
    arc_r = 0.129 * w

    for top in (True, False):
        y_edge = y0 if top else y1
        sgn = 1 if top else -1
        # penalty area
        ax.add_patch(plt.Rectangle(
            (xc - (w / 2 - box_inset), min(y_edge, y_edge + sgn * box_depth)),
            2 * (w / 2 - box_inset), box_depth, fill=False, **line))
        # six-yard box
        ax.add_patch(plt.Rectangle(
            (xc - (w / 2 - six_inset), min(y_edge, y_edge + sgn * six_depth)),
            2 * (w / 2 - six_inset), six_depth, fill=False, **line))
        # goal
        gw = 0.10 * w
        ax.add_patch(plt.Rectangle(
            (xc - gw / 2, min(y_edge, y_edge + sgn * 0.018 * h)),
            gw, 0.018 * h, fill=False, linewidth=2.0, color=color, zorder=3))
        # penalty arc (only the part outside the box)
        cy = y_edge + sgn * spot
        edge = y_edge + sgn * box_depth
        frac = (edge - cy) / (sgn * arc_r)
        if 0.0 < frac < 1.0:
            a0 = np.degrees(np.arcsin(frac))
            if top:
                th1, th2 = a0, 180 - a0
            else:
                th1, th2 = 180 + a0, 360 - a0
            ax.add_patch(Arc((xc, cy), 2 * arc_r, 2 * arc_r, angle=0,
                             theta1=th1, theta2=th2, **line))

    ax.set_xlim(x0 - 0.03 * w, x1 + 0.03 * w)
    ax.set_ylim(y1 + 0.03 * h, y0 - 0.03 * h)  # inverted: y grows down-field
    ax.set_aspect("equal")
    ax.axis("off")


def _pitch_figure(figsize=(6.0, 10.0), ncols: int = 1, nrows: int = 1,
                  dark: bool = True):
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    fig.patch.set_facecolor("#101418" if dark else "white")
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("#1a1f24" if dark else "white")
    return fig, axes


def _team_label(team, team_names) -> str:
    try:
        return team_names[int(team)]
    except Exception:  # noqa: BLE001
        return f"Team {team}"


def _color(team) -> str:
    if team is None or (isinstance(team, float) and np.isnan(team)):
        return UNKNOWN_COLOR
    return TEAM_COLORS.get(int(team), UNKNOWN_COLOR)


# --------------------------------------------------------------------------- #
# Derived possession / touches / passes
# --------------------------------------------------------------------------- #
def possession_series(
    players: pd.DataFrame,
    ball: pd.DataFrame,
    cfg: AnalyticsConfig,
) -> pd.DataFrame:
    """Per-frame ball owner: ``frame, gid, team, pitch_x, pitch_y``.

    Ball detection drops out (crowd false-positives are filtered, and the ball
    is small), so a stale owner is carried forward for up to
    ``max_possession_gap`` frames. Runs shorter than ``min_hold`` frames are
    flicker (two players standing next to each other) and are merged away.
    """
    if ball.empty or players.empty:
        return pd.DataFrame(columns=["frame", "gid", "team", "pitch_x", "pitch_y"])

    by_frame = {int(f): g for f, g in players.groupby("frame")}
    raw: list[tuple[int, str, float, float, float] | None] = []

    last = None
    gap = 0
    for _, row in ball.sort_values("frame").iterrows():
        f = int(row["frame"])
        grp = by_frame.get(f)
        if grp is not None and len(grp):
            d = np.hypot(grp["pitch_x"].to_numpy() - row["pitch_x"],
                         grp["pitch_y"].to_numpy() - row["pitch_y"])
            i = int(np.argmin(d))
            if d[i] <= cfg.touch_radius:
                g = grp.iloc[i]
                last = (f, str(g["gid"]),
                        float(g["team"]) if pd.notna(g["team"]) else np.nan,
                        float(g["pitch_x"]), float(g["pitch_y"]))
                gap = 0
                raw.append(last)
                continue
        gap += 1
        if last is not None and gap <= cfg.max_possession_gap:
            # carry the previous owner through the dropout
            raw.append((f, last[1], last[2], last[3], last[4]))
        else:
            raw.append(None)
            last = None

    # Merge flicker: owner runs shorter than min_hold are absorbed into the
    # previous run (two players standing together swap the ball frame by
    # frame); an empty run (no possession) breaks ownership.
    runs: list[list] = []
    for item in raw:
        if runs and item is not None and runs[-1] and runs[-1][-1][1] == item[1]:
            runs[-1].append(item)
        else:
            runs.append([item] if item is not None else [])

    merged: list = []
    for run in runs:
        if run and merged and len(run) < cfg.min_hold and merged[-1]:
            merged[-1].extend(run)
        elif run:
            merged.append(run)

    return pd.DataFrame([
        {"frame": r[0], "gid": r[1], "team": r[2],
         "pitch_x": r[3], "pitch_y": r[4]}
        for run in merged for r in run
    ])


def touch_events(pos: pd.DataFrame,
                 pass_window: int = 45) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(touches, passes)`` derived from owner changes.

    touches: frame, gid, team, pitch_x, pitch_y -- one row per take-on.
    passes:  frame, frm, to, team -- consecutive same-team owner changes
    within ``pass_window`` frames of each other (anything slower is not
    treated as a pass).
    """
    touches = []
    passes = []
    prev = None
    for _, row in pos.iterrows():
        if prev is None or row["gid"] != prev["gid"]:
            touches.append(row.to_dict())
            if (
                prev is not None
                and pd.notna(row["team"])
                and pd.notna(prev["team"])
                and row["team"] == prev["team"]
                and row["frame"] - prev["frame"] <= pass_window
            ):
                passes.append({
                    "frame": int(row["frame"]),
                    "frm": prev["gid"],
                    "to": row["gid"],
                    "team": int(row["team"]),
                    "x_from": prev["pitch_x"], "y_from": prev["pitch_y"],
                    "x_to": row["pitch_x"], "y_to": row["pitch_y"],
                })
            prev = row
        else:
            prev = row
    return pd.DataFrame(touches), pd.DataFrame(passes)


# --------------------------------------------------------------------------- #
# Individual charts
# --------------------------------------------------------------------------- #
def chart_possession(pos: pd.DataFrame, cfg: AnalyticsConfig, fps: float,
                     team_names, out: Path, title: str) -> Path | None:
    if pos.empty:
        return None
    total = pos["frame"].max() - pos["frame"].min() + 1
    total_s = max(total / fps, 1e-6)
    bucket = max(1.0, total_s / cfg.n_intervals)

    teams = sorted(int(t) for t in pos["team"].dropna().unique())
    if not teams:
        return None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")
    for t in teams:
        sub = pos[pos["team"] == t]
        centres, pcts = [], []
        for k in range(cfg.n_intervals):
            lo = k * bucket
            hi = (k + 1) * bucket
            f_lo = pos["frame"].min() + lo * fps
            f_hi = pos["frame"].min() + hi * fps
            win = pos[(pos["frame"] >= f_lo) & (pos["frame"] < f_hi)]
            if win.empty:
                continue
            share = float((win["team"] == t).mean()) * 100
            centres.append((lo + hi) / 2)
            pcts.append(share)
        if not centres:
            continue
        overall = float((pos["team"] == t).mean()) * 100
        label = f"{_team_label(t, team_names)} ({overall:.0f}%)"
        ax.plot(centres, pcts, marker="o", linewidth=3, markersize=7,
                color=TEAM_COLORS.get(t, UNKNOWN_COLOR), label=label)
        for x, y in zip(centres, pcts):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=10,
                        color=TEAM_COLORS.get(t, UNKNOWN_COLOR))

    ax.set_ylim(0, 100)
    ax.set_xlabel("seconds")
    ax.set_ylabel("possession %")
    ax.set_title(f"Ball possession, %\n{title}", fontsize=15, loc="left")
    ax.grid(axis="y", color="#dddddd")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=len(teams), loc="lower center",
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def chart_heatmap(players: pd.DataFrame, team: int, b, cfg: AnalyticsConfig,
                  team_names, out: Path, title: str) -> Path | None:
    sub = players[players["team"] == team] if "team" in players else pd.DataFrame()
    if sub.empty or len(sub) < 10:
        return None
    x0, x1, y0, y1 = b
    nx, ny = 28, 50
    hist, _, _ = np.histogram2d(sub["pitch_y"], sub["pitch_x"],
                                bins=[ny, nx], range=[[y0, y1], [x0, x1]])
    try:
        from scipy.ndimage import gaussian_filter

        hist = gaussian_filter(hist, sigma=1.6)
    except Exception:  # noqa: BLE001 - smoothing is cosmetic
        pass

    fig, ax = _pitch_figure(figsize=(5.5, 9.5))
    im = ax.imshow(hist, origin="upper", extent=[x0, x1, y1, y0],
                   cmap="magma", aspect="equal", alpha=0.95, zorder=1)
    _draw_pitch(ax, b)
    ax.set_title(f"{_team_label(team, team_names)} — positioning heatmap\n{title}",
                 color="white", fontsize=12, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="density")
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def chart_convex(players: pd.DataFrame, b, team_names, out: Path,
                 title: str) -> Path | None:
    from scipy.spatial import ConvexHull

    teams = sorted(int(t) for t in players["team"].dropna().unique())
    if not teams:
        return None
    fig, axes = _pitch_figure(figsize=(11, 9.5), ncols=len(teams), dark=False)
    for ax, t in zip(np.atleast_1d(axes), teams):
        ax.set_facecolor("white")
        sub = players[players["team"] == t]
        pts = sub[["pitch_x", "pitch_y"]].to_numpy()
        if len(pts) >= 4:
            try:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
                ax.add_patch(Polygon(poly, closed=True, alpha=0.25,
                                     facecolor=TEAM_COLORS.get(t, UNKNOWN_COLOR),
                                     edgecolor=TEAM_COLORS.get(t, UNKNOWN_COLOR),
                                     linewidth=2))
            except Exception:  # noqa: BLE001 - degenerate point sets
                pass
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.55,
                   color=TEAM_COLORS.get(t, UNKNOWN_COLOR), zorder=3)
        _draw_pitch(ax, b, color="#333333")
        ax.set_title(_team_label(t, team_names), color="#333333", fontsize=13)
    fig.suptitle(f"Territory (convex hull) — {title}", fontsize=15, color="#333333")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def chart_formations(players: pd.DataFrame, cfg: AnalyticsConfig, b,
                     team_names, out: Path, title: str) -> Path | None:
    if players.empty:
        return None
    teams = sorted(int(t) for t in players["team"].dropna().unique())
    if not teams:
        return None
    n = cfg.formation_intervals
    lo_f, hi_f = players["frame"].min(), players["frame"].max()
    edges = np.linspace(lo_f, hi_f, n + 1)

    fig, axes = _pitch_figure(figsize=(9, 14), ncols=2, nrows=2, dark=False)
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.set_facecolor("white")
        window = players[(players["frame"] >= edges[k]) &
                         (players["frame"] < edges[k + 1])]
        for t in teams:
            sub = window[window["team"] == t]
            if sub.empty:
                continue
            grp = sub.groupby("gid").agg(x=("pitch_x", "mean"),
                                         y=("pitch_y", "mean"))
            ax.scatter(grp["x"], grp["y"], s=90,
                       color=TEAM_COLORS.get(t, UNKNOWN_COLOR),
                       edgecolor="white", linewidth=1.2, zorder=3)
            for gid, r in grp.iterrows():
                ax.annotate(f"#{int(gid)}" if str(gid).isdigit() else str(gid),
                            (r["x"], r["y"]), textcoords="offset points",
                            xytext=(7, -10), fontsize=7.5, color="#222222")
        _draw_pitch(ax, b, color="#333333")
        secs0 = (edges[k] - lo_f) / 30.0
        secs1 = (edges[k + 1] - lo_f) / 30.0
        ax.set_title(f"Q{k + 1}  ({secs0:.0f}–{secs1:.0f}s)",
                     color="#333333", fontsize=12)
    fig.suptitle(f"Average positions per interval — {title}",
                 fontsize=15, color="#333333")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def chart_pass_network(touches: pd.DataFrame, passes: pd.DataFrame, b,
                       team_names, out: Path, title: str) -> Path | None:
    if touches.empty:
        return None
    teams = sorted(int(t) for t in touches["team"].dropna().unique())
    if not teams:
        return None
    fig, axes = _pitch_figure(figsize=(11, 9.5), ncols=len(teams), dark=False)
    for ax, t in zip(np.atleast_1d(axes), teams):
        ax.set_facecolor("white")
        tc = touches[(touches["team"] == t) & touches["gid"].notna()]
        pc = passes[passes["team"] == t] if not passes.empty else passes
        if tc.empty:
            ax.set_title(_team_label(t, team_names), color="#333333")
            _draw_pitch(ax, b, color="#333333")
            continue
        counts = tc.groupby("gid").size()
        pos = tc.groupby("gid").agg(x=("pitch_x", "mean"), y=("pitch_y", "mean"))
        # A stray "nan"/None identity label would otherwise be drawn as a node.
        pos = pos[[str(g).lower() not in ("nan", "none", "") for g in pos.index]]

        if not pc.empty:
            edge_w = pc.groupby(["frm", "to"]).size()
            for (frm, to), w in edge_w.items():
                if frm in pos.index and to in pos.index:
                    a, bb = pos.loc[frm], pos.loc[to]
                    ax.plot([a["x"], bb["x"]], [a["y"], bb["y"]],
                            color=TEAM_COLORS.get(t, UNKNOWN_COLOR),
                            linewidth=0.8 + 1.6 * np.log1p(w),
                            alpha=0.55, zorder=2, solid_capstyle="round")

        for gid, r in pos.iterrows():
            size = 260 + 90 * np.log1p(int(counts.get(gid, 1)))
            ax.scatter([r["x"]], [r["y"]], s=size, color="#ffffff",
                       edgecolor=TEAM_COLORS.get(t, UNKNOWN_COLOR),
                       linewidth=3, zorder=4)
            label = f"#{int(gid)}" if str(gid).replace(".0", "").isdigit() else str(gid)
            ax.annotate(label, (r["x"], r["y"]), ha="center", va="center",
                        fontsize=7.5, color="#222222", zorder=5)
        _draw_pitch(ax, b, color="#333333")
        ax.set_title(f"{_team_label(t, team_names)} — pass network",
                     color="#333333", fontsize=13)
    fig.suptitle(f"Pass network (derived from ball possession) — {title}",
                 fontsize=15, color="#333333")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def chart_passmap(passes: pd.DataFrame, team: int, b, team_names, out: Path,
                  title: str) -> Path | None:
    if passes.empty:
        return None
    pc = passes[passes["team"] == team]
    if pc.empty:
        return None
    color = TEAM_COLORS.get(team, UNKNOWN_COLOR)
    fig, ax = _pitch_figure(figsize=(5.5, 9.5), dark=False)
    ax.set_facecolor("white")
    ax.scatter(pc["x_from"], pc["y_from"], s=18, color=color, alpha=0.5, zorder=3)
    for _, r in pc.iterrows():
        ax.annotate("", xy=(r["x_to"], r["y_to"]), xytext=(r["x_from"], r["y_from"]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                    alpha=0.65, mutation_scale=13))
    _draw_pitch(ax, b, color="#333333")
    ax.set_title(f"{_team_label(team, team_names)} — passes\n{title}",
                 color="#333333", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def chart_ball_passes(ball: pd.DataFrame, pos: pd.DataFrame, b, team_names,
                      out: Path, title: str) -> Path | None:
    if ball.empty:
        return None
    bb = ball.sort_values("frame")
    frames = bb["frame"].to_numpy()
    xs = bb["pitch_x"].to_numpy()
    ys = bb["pitch_y"].to_numpy()

    # possessor colour per ball frame
    owner = {}
    if not pos.empty:
        for _, r in pos.iterrows():
            owner[int(r["frame"])] = r["team"]

    fig, ax = _pitch_figure(figsize=(5.5, 9.5))
    ax.set_facecolor("#1a1f24")
    prev_f = None
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > 90:  # >3 s without ball: break the line
            continue
        team = owner.get(int(frames[i]))
        color = _color(team) if team is not None and not (
            isinstance(team, float) and np.isnan(team)) else "#666666"
        ax.plot(xs[i - 1:i + 1], ys[i - 1:i + 1], color=color,
                linewidth=2.0, alpha=0.85, zorder=3, solid_capstyle="round")
    ax.scatter(xs, ys, s=6, color="#ffcc00", zorder=4, alpha=0.8)
    _draw_pitch(ax, b)
    teams_seen = []
    for t in owner.values():
        if t is None or (isinstance(t, float) and np.isnan(t)):
            continue
        if t not in teams_seen:
            teams_seen.append(t)
    handles = [plt.Line2D([], [], color=TEAM_COLORS.get(t, UNKNOWN_COLOR),
                          linewidth=3, label=_team_label(t, team_names))
               for t in sorted(teams_seen, key=lambda v: (v is None, v))]
    if handles:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
                  frameon=False, fontsize=9)
    ax.set_title(f"Ball movement by possessor — {title}", color="white",
                 fontsize=12, pad=10)
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    return out


def chart_player_touches(touches: pd.DataFrame, team_names, out: Path,
                         title: str) -> Path | None:
    if touches.empty or touches["gid"].isna().all():
        return None
    tc = touches.dropna(subset=["gid"]).copy()
    # drop string "nan" / empty identities too
    tc = tc[tc["gid"].astype(str).str.strip().str.lower().isin(
        {"nan", "none", ""}) == False]  # noqa: E712
    counts = tc.groupby(["gid", "team"], dropna=False).size().reset_index(name="n")
    counts = counts.sort_values(["n", "gid"]).reset_index(drop=True)
    if counts.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(counts) + 2.2))
    fig.patch.set_facecolor("white")
    labels, colors = [], []
    for _, r in counts.iterrows():
        gid = r["gid"]
        labels.append(f"#{int(gid)}" if str(gid).replace(".0", "").isdigit() else str(gid))
        colors.append(_color(r["team"]))

    # Numeric y positions: duplicate identity labels would otherwise collapse
    # onto one bar (and push the value texts off the axes).
    y = np.arange(len(counts))
    ax.barh(y, counts["n"].to_numpy(), color=colors)
    vmax = float(counts["n"].max())
    for i, v in enumerate(counts["n"].to_numpy()):
        ax.text(v + vmax * 0.01, i, str(int(v)), va="center",
                fontsize=9, color="#444444")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(counts) - 0.4)
    ax.set_xlabel("touches (ball possessions)")
    ax.set_title(f"Player touches — {title}", fontsize=14, loc="left")
    ax.grid(axis="x", color="#e6e6e6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TEAM_COLORS.get(t, UNKNOWN_COLOR),
                             label=_team_label(t, team_names))
               for t in sorted(counts["team"].dropna().unique())]
    if handles:
        ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def chart_tactical_positions(players: pd.DataFrame, paths: Paths, team_names,
                             out: Path, title: str) -> Path | None:
    if players.empty:
        return None
    img = cv2.imread(str(paths.tactical_map))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    teams = sorted(int(t) for t in players["team"].dropna().unique())
    fig, ax = plt.subplots(figsize=(6, 10.4))
    fig.patch.set_facecolor("white")
    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])
    for t in teams:
        sub = players[players["team"] == t]
        grp = sub.groupby("gid").agg(x=("pitch_x", "mean"), y=("pitch_y", "mean"),
                                     n=("frame", "count"))
        grp = grp[grp["n"] >= 5]
        ax.scatter(grp["x"], grp["y"], s=140,
                   color=TEAM_COLORS.get(t, UNKNOWN_COLOR),
                   edgecolor="white", linewidth=2, zorder=3)
        for gid, r in grp.iterrows():
            label = f"#{int(gid)}" if str(gid).replace(".0", "").isdigit() else str(gid)
            ax.annotate(label, (r["x"], r["y"]), textcoords="offset points",
                        xytext=(8, -12), fontsize=9, color="white", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="#111111",
                                  ec="none", alpha=0.75))
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    ax.set_title(f"Average positions — {title}", fontsize=13, pad=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def generate_viz(
    csv_path: str | Path,
    out_dir: str | Path,
    cfg: AnalyticsConfig | None = None,
    paths: Paths | None = None,
    fps: float = 30.0,
    team_names: list[str] | None = None,
    title: str | None = None,
) -> list[Path]:
    """Generate every chart from ``csv_path`` into ``out_dir``.

    Returns the list of PNG paths actually written. Each chart is wrapped so
    one failure (degenerate geometry, missing tactical map, ...) still yields
    the rest of the gallery.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or AnalyticsConfig()
    paths = paths or Paths()
    team_names = team_names or ["Team A", "Team B"]
    title = title or csv_path.stem.replace("_tracks", "")

    players, ball = load_tracks(csv_path)
    if players.empty:
        return []
    b = pitch_bounds(paths)

    written: list[Path] = []

    def _make(name: str, fn) -> None:
        target = out_dir / name
        try:
            result = fn(target)
        except Exception as exc:  # noqa: BLE001 - keep the gallery going
            print(f"  [warn] viz {name} failed: {type(exc).__name__}: {exc}")
            return
        if result is not None and Path(result).exists():
            written.append(Path(result))

    pos = possession_series(players, ball, cfg) if not ball.empty else pd.DataFrame()
    touches, passes = (touch_events(pos, cfg.max_possession_gap)
                       if not pos.empty else (pd.DataFrame(), pd.DataFrame()))

    _make("possession.png",
          lambda p: chart_possession(pos, cfg, fps, team_names, p, title)
          if not pos.empty else None)
    _make("ball_passes.png",
          lambda p: chart_ball_passes(ball, pos, b, team_names, p, title)
          if not ball.empty else None)
    _make("player_touches.png",
          lambda p: chart_player_touches(touches, team_names, p, title)
          if not touches.empty else None)
    _make("pass_network.png",
          lambda p: chart_pass_network(touches, passes, b, team_names, p, title)
          if not touches.empty else None)

    for t in sorted(int(t) for t in players["team"].dropna().unique()):
        _make(f"heatmap_team_{t}.png",
              lambda p, t=t: chart_heatmap(players, t, b, cfg, team_names, p, title))
        _make(f"passmap_team_{t}.png",
              lambda p, t=t: chart_passmap(passes, t, b, team_names, p, title))

    _make("convex_hull.png",
          lambda p: chart_convex(players, b, team_names, p, title))
    _make("formations.png",
          lambda p: chart_formations(players, cfg, b, team_names, p, title))
    _make("tactical_map_positions.png",
          lambda p: chart_tactical_positions(players, paths, team_names, p, title))

    return sorted(written)
