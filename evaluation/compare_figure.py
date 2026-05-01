# evaluation/compare_figure.py
# Produces a 1x3 comparison figure for three simulation logs, all at the same scale.
#
# Usage:
#   python evaluation/compare_figure.py
#   python evaluation/compare_figure.py --save comparison.png

import argparse
import json
import math
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import args as sim_args
from environments.base_env import CircleObstacle, EllipseObstacle

# ---------------------------------------------------------------------------
# Paths and labels
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANELS = [
    ("results/mpc_constvel.csv",  "Const-Vel Predictor"),
    ("results/mpc_velacc.csv",    "Vel+Acc Predictor"),
    ("results/mpc_att.csv",       "Attractor Field Predictor"),
]

# ---------------------------------------------------------------------------
# Data loading (shared with evaluate.py)
# ---------------------------------------------------------------------------

import csv

def load_log(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Log file is empty: {csv_path}")

    def col(key, cast=float, default=None):
        out = []
        for r in rows:
            raw = r.get(key, "")
            if raw == "" or raw is None:
                out.append(default)
            else:
                out.append(cast(raw))
        return out

    data = {
        "timestep":        col("timestep", int),
        "sim_time_s":      col("sim_time_s"),
        "drone_x":         col("drone_x"),
        "drone_y":         col("drone_y"),
        "evader_x":        col("evader_x"),
        "evader_y":        col("evader_y"),
        "pred_horizon_full": [],
    }

    for r in rows:
        raw = r.get("pred_horizon_full", "")
        if raw and raw.strip():
            try:
                data["pred_horizon_full"].append(json.loads(raw))
            except (json.JSONDecodeError, ValueError):
                data["pred_horizon_full"].append(None)
        else:
            data["pred_horizon_full"].append(None)

    return data


# ---------------------------------------------------------------------------
# Bounds computation
# ---------------------------------------------------------------------------

def _data_extents(data):
    """Return (xmin, xmax, ymin, ymax) covering all trajectory + horizon points."""
    xs = list(data["drone_x"]) + list(data["evader_x"])
    ys = list(data["drone_y"]) + list(data["evader_y"])
    for traj in data["pred_horizon_full"]:
        if traj is not None:
            for pt in traj:
                xs.append(pt[0])
                ys.append(pt[1])
    return min(xs), max(xs), min(ys), max(ys)


def compute_shared_bounds(all_data, env, padding=3.0):
    xmin, xmax, ymin, ymax = math.inf, -math.inf, math.inf, -math.inf
    for data in all_data:
        x0, x1, y0, y1 = _data_extents(data)
        xmin = min(xmin, x0)
        xmax = max(xmax, x1)
        ymin = min(ymin, y0)
        ymax = max(ymax, y1)
    for obs in env.obstacles:
        r = max(obs.rx, obs.ry) if isinstance(obs, EllipseObstacle) else obs.radius
        xmin = min(xmin, obs.cx - r)
        xmax = max(xmax, obs.cx + r)
        ymin = min(ymin, obs.cy - r)
        ymax = max(ymax, obs.cy + r)
    return xmin - padding, xmax + padding, ymin - padding, ymax + padding


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_obstacles(ax, env):
    obs_fill = (80/255, 80/255, 80/255)
    obs_edge = (40/255, 40/255, 40/255)
    for obs in env.obstacles:
        if isinstance(obs, EllipseObstacle):
            patch = mpatches.Ellipse(
                (obs.cx, obs.cy), width=2*obs.rx, height=2*obs.ry,
                angle=math.degrees(obs.theta),
                facecolor=obs_fill, edgecolor=obs_edge, linewidth=1.5, zorder=2,
            )
        else:
            patch = mpatches.Circle(
                (obs.cx, obs.cy), radius=obs.radius,
                facecolor=obs_fill, edgecolor=obs_edge, linewidth=1.5, zorder=2,
            )
        ax.add_patch(patch)


def draw_trajectories(ax, data):
    ax.plot(data["drone_x"], data["drone_y"],
            color=(0.1, 0.3, 0.9), linewidth=1.5, alpha=0.7,
            label="Drone trajectory", zorder=4)
    ax.plot(data["evader_x"], data["evader_y"],
            color=(0.85, 0.1, 0.1), linewidth=1.5, alpha=0.7,
            label="Evader trajectory", zorder=4)


def draw_endpoints(ax, data):
    # Drone start / end
    ax.plot(data["drone_x"][0], data["drone_y"][0],
            marker="^", markersize=10, color=(0.1, 0.3, 0.9),
            markeredgecolor="navy", linewidth=0, zorder=6, label="Drone start")
    ax.plot(data["drone_x"][-1], data["drone_y"][-1],
            marker="^", markersize=10, color=(0.1, 0.3, 0.9),
            markeredgecolor="navy", markerfacecolor="none",
            linewidth=0, zorder=6, label="Drone end")
    # Evader start / end
    ax.plot(data["evader_x"][0], data["evader_y"][0],
            marker="o", markersize=9, color=(0.85, 0.1, 0.1),
            markeredgecolor="darkred", linewidth=0, zorder=6, label="Evader start")
    ax.plot(data["evader_x"][-1], data["evader_y"][-1],
            marker="o", markersize=9, color=(0.85, 0.1, 0.1),
            markeredgecolor="darkred", markerfacecolor="none",
            linewidth=0, zorder=6, label="Evader end")


def draw_full_horizon(ax, data, sample_interval_s=1.0):
    trajs = data["pred_horizon_full"]
    sim_times = data["sim_time_s"]
    if not any(t is not None for t in trajs):
        return

    sampled = []
    next_t = 0.0
    for i, t in enumerate(sim_times):
        if t is None:
            continue
        if t >= next_t - 1e-9:
            sampled.append(i)
            next_t += sample_interval_s

    cmap = plt.cm.plasma
    n = len(sampled)
    first = True
    for arm_idx, i in enumerate(sampled):
        traj = trajs[i]
        if traj is None:
            continue
        xs = [pt[0] for pt in traj]
        ys = [pt[1] for pt in traj]
        color = cmap(arm_idx / max(n - 1, 1))
        label = "Prediction horizon (1 s steps)" if first else "_"
        first = False
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.55, zorder=5, label=label)
        ax.plot(xs[-1], ys[-1], marker="x", markersize=4,
                color=color, linewidth=0, alpha=0.8, zorder=6, label="_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=None, help="Output filename (PNG/PDF)")
    cli = parser.parse_args()

    env = sim_args.ENV_FACTORY()

    # Load all three datasets
    all_data = []
    for rel_path, _ in PANELS:
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        all_data.append(load_log(abs_path))

    xmin, xmax, ymin, ymax = compute_shared_bounds(all_data, env)

    # Figure: 1 row × 3 columns, square panels
    panel_w = 5.0
    fig, axes = plt.subplots(1, 3, figsize=(panel_w * 3 + 0.5, panel_w + 0.8))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.10, top=0.90, wspace=0.08)

    for ax, data, (_, title) in zip(axes, all_data, PANELS):
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-10, 20)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel("x (m)", fontsize=9)

        draw_obstacles(ax, env)
        draw_full_horizon(ax, data)
        draw_trajectories(ax, data)
        draw_endpoints(ax, data)

    # Only show y-axis label on the leftmost panel
    axes[0].set_ylabel("y (m)", fontsize=9)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    # Shared legend below the figure, drawn from the last axis
    handles, labels = axes[-1].get_legend_handles_labels()
    # Also grab the obstacle patch manually for the legend
    obs_patch = mpatches.Patch(facecolor=(80/255, 80/255, 80/255),
                               edgecolor=(40/255, 40/255, 40/255), label="Obstacle")
    handles = [obs_patch] + handles
    labels = ["Obstacle"] + labels
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.01))

    if cli.save:
        out = cli.save if os.path.isabs(cli.save) else os.path.join(PROJECT_ROOT, cli.save)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
