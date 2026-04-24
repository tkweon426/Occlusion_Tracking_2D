# evaluation/evaluate.py
# Interactive post-run evaluator for occlusion tracking simulation logs.
#
# Usage:
#   python evaluation/evaluate.py results/log_20260422_204007.csv
#   python evaluation/evaluate.py results/log.csv --env circle

import argparse
import csv
import math
import os
import sys
from datetime import datetime

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import args as sim_args
from environments.base_env import CircleObstacle, EllipseObstacle
from environments.empty import make_empty_env
from environments.single_circle import make_single_circle_env
from environments.single_ellipse import make_single_ellipse_env, make_two_obs_env

ENV_CHOICES = {
    "empty":   make_empty_env,
    "circle":  make_single_circle_env,
    "ellipse": make_single_ellipse_env,
    "two_obs": make_two_obs_env,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
        "drone_psi":       col("drone_psi"),
        "evader_x":        col("evader_x"),
        "evader_y":        col("evader_y"),
        "visibility_score": col("visibility_score"),
        "tracking_error_score": col("tracking_error_score", float, None),
        "pred_evader_x":        col("pred_evader_x",        float, None),
        "pred_evader_y":        col("pred_evader_y",        float, None),
        "pred_horizon_evader_x": col("pred_horizon_evader_x", float, None),
        "pred_horizon_evader_y": col("pred_horizon_evader_y", float, None),
    }
    return data


# ---------------------------------------------------------------------------
# Axis bounds helpers
# ---------------------------------------------------------------------------

def _compute_bounds(data, env, padding=3.0):
    all_x = data["drone_x"] + data["evader_x"]
    all_y = data["drone_y"] + data["evader_y"]

    pred_x = [v for v in data["pred_evader_x"] if v is not None]
    pred_y = [v for v in data["pred_evader_y"] if v is not None]
    if pred_x:
        all_x += pred_x
        all_y += pred_y

    hor_x = [v for v in data["pred_horizon_evader_x"] if v is not None]
    hor_y = [v for v in data["pred_horizon_evader_y"] if v is not None]
    if hor_x:
        all_x += hor_x
        all_y += hor_y

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    for obs in env.obstacles:
        if isinstance(obs, EllipseObstacle):
            r = max(obs.rx, obs.ry)
        else:
            r = obs.radius
        xmin = min(xmin, obs.cx - r)
        xmax = max(xmax, obs.cx + r)
        ymin = min(ymin, obs.cy - r)
        ymax = max(ymax, obs.cy + r)

    return xmin - padding, xmax + padding, ymin - padding, ymax + padding


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_grid(ax, xmin, xmax, ymin, ymax, spacing=10.0):
    artists = []
    color = (0.78, 0.78, 0.78)
    x = math.floor(xmin / spacing) * spacing
    while x <= xmax:
        ln, = ax.plot([x, x], [ymin, ymax], color=color, linewidth=0.5, zorder=0)
        artists.append(ln)
        x += spacing
    y = math.floor(ymin / spacing) * spacing
    while y <= ymax:
        ln, = ax.plot([xmin, xmax], [y, y], color=color, linewidth=0.5, zorder=0)
        artists.append(ln)
        y += spacing
    return artists


def draw_obstacles(ax, env):
    artists = []
    obs_fill = (80 / 255, 80 / 255, 80 / 255)
    obs_edge = (40 / 255, 40 / 255, 40 / 255)
    for obs in env.obstacles:
        if isinstance(obs, EllipseObstacle):
            patch = mpatches.Ellipse(
                (obs.cx, obs.cy),
                width=2 * obs.rx,
                height=2 * obs.ry,
                angle=math.degrees(obs.theta),
                facecolor=obs_fill,
                edgecolor=obs_edge,
                linewidth=1.5,
                zorder=2,
            )
        else:
            patch = mpatches.Circle(
                (obs.cx, obs.cy),
                radius=obs.radius,
                facecolor=obs_fill,
                edgecolor=obs_edge,
                linewidth=1.5,
                zorder=2,
            )
        ax.add_patch(patch)
        artists.append(patch)
    return artists


def draw_los_samples(ax, data, env, dt, interval_s=1.0):
    """Draw LOS segments sampled every interval_s seconds."""
    n = len(data["drone_x"])
    step = max(1, round(interval_s / dt))
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    artists = []
    for i in indices:
        dx, dy = data["drone_x"][i], data["drone_y"][i]
        ex, ey = data["evader_x"][i], data["evader_y"][i]
        has_los = env.has_line_of_sight(dx, dy, ex, ey)
        color = (0, 180 / 255, 0) if has_los else (220 / 255, 0, 0)
        ln, = ax.plot([dx, ex], [dy, ey], color=color, linewidth=1.0,
                      linestyle="--", alpha=0.7, zorder=3)
        artists.append(ln)
    return artists


def draw_drone_trajectory(ax, data):
    ln, = ax.plot(data["drone_x"], data["drone_y"],
                  color=(0.1, 0.3, 0.9), linewidth=1.5, alpha=0.7,
                  label="Drone trajectory", zorder=4)
    return [ln]


def draw_evader_trajectory(ax, data):
    ln, = ax.plot(data["evader_x"], data["evader_y"],
                  color=(0.85, 0.1, 0.1), linewidth=1.5, alpha=0.7,
                  label="Evader trajectory", zorder=4)
    return [ln]


def draw_drone_start(ax, data):
    pt, = ax.plot(data["drone_x"][0], data["drone_y"][0],
                  marker="^", markersize=13, color=(0.1, 0.3, 0.9),
                  markeredgecolor="navy", linewidth=0, zorder=6,
                  label="Drone start")
    return [pt]


def draw_drone_end(ax, data):
    pt, = ax.plot(data["drone_x"][-1], data["drone_y"][-1],
                  marker="^", markersize=13, color=(0.1, 0.3, 0.9),
                  markeredgecolor="navy", markerfacecolor="none",
                  linewidth=0, zorder=6, label="Drone end")
    return [pt]


def draw_evader_start(ax, data):
    pt, = ax.plot(data["evader_x"][0], data["evader_y"][0],
                  marker="o", markersize=11, color=(0.85, 0.1, 0.1),
                  markeredgecolor="darkred", linewidth=0, zorder=6,
                  label="Evader start")
    return [pt]


def draw_evader_end(ax, data):
    pt, = ax.plot(data["evader_x"][-1], data["evader_y"][-1],
                  marker="o", markersize=11, color=(0.85, 0.1, 0.1),
                  markeredgecolor="darkred", markerfacecolor="none",
                  linewidth=0, zorder=6, label="Evader end")
    return [pt]


def draw_predicted_evader(ax, data):
    px = [v for v in data["pred_evader_x"] if v is not None]
    py = [v for v in data["pred_evader_y"] if v is not None]
    if not px:
        return []
    sc = ax.scatter(px, py, color="darkorange", s=10, alpha=0.4,
                    zorder=5, label="Predicted evader (next step)")
    return [sc]


def draw_predicted_horizon(ax, data):
    """Predicted evader position at the end of the MPC horizon, logged each timestep."""
    hx = [v for v in data["pred_horizon_evader_x"] if v is not None]
    hy = [v for v in data["pred_horizon_evader_y"] if v is not None]
    if not hx:
        return []
    sc = ax.scatter(hx, hy, color="purple", s=10, alpha=0.4,
                    zorder=5, label="Predicted evader (horizon end)")
    return [sc]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a simulation log file.")
    parser.add_argument("log", help="Path to the CSV log file")
    parser.add_argument(
        "--env",
        choices=list(ENV_CHOICES.keys()),
        default=None,
        help="Environment to display (default: uses args.py ENV_FACTORY)",
    )
    cli = parser.parse_args()

    # Resolve working directory to the project root so relative paths work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = cli.log if os.path.isabs(cli.log) else os.path.join(project_root, cli.log)

    data = load_log(log_path)
    n_steps = len(data["timestep"])
    total_time = data["sim_time_s"][-1] if data["sim_time_s"][-1] is not None else 0.0
    dt = sim_args.DT

    env = ENV_CHOICES[cli.env]() if cli.env else sim_args.ENV_FACTORY()

    has_predictions = any(v is not None for v in data["pred_evader_x"])
    has_horizon     = any(v is not None for v in data["pred_horizon_evader_x"])

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0.04, 0.05, 0.68, 0.88])
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    csv_name = os.path.basename(log_path)
    ax.set_title(
        f"Evaluation: {csv_name}    |    {n_steps} steps    |    {total_time:.2f} s",
        fontsize=11,
        pad=8,
    )

    xmin, xmax, ymin, ymax = _compute_bounds(data, env)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ------------------------------------------------------------------
    # Draw all elements and collect artist groups
    # ------------------------------------------------------------------
    grid_artists    = draw_grid(ax, xmin, xmax, ymin, ymax)
    obs_artists     = draw_obstacles(ax, env)
    drone_traj      = draw_drone_trajectory(ax, data)
    evader_traj     = draw_evader_trajectory(ax, data)
    drone_start     = draw_drone_start(ax, data)
    drone_end       = draw_drone_end(ax, data)
    evader_start    = draw_evader_start(ax, data)
    evader_end      = draw_evader_end(ax, data)
    los_artists     = draw_los_samples(ax, data, env, dt)
    pred_artists    = draw_predicted_evader(ax, data)
    horizon_artists = draw_predicted_horizon(ax, data)

    # Map toggle label → list of artists
    element_artists = {
        "Env (obstacles)":       obs_artists,
        "Drone start":           drone_start,
        "Evader start":          evader_start,
        "Drone end":             drone_end,
        "Evader end":            evader_end,
        "Drone trajectory":      drone_traj,
        "Evader trajectory":     evader_traj,
        "Line of sight":         los_artists,
        "Pred evader (next)":    pred_artists,
        "Pred evader (horizon)": horizon_artists,
    }

    # ------------------------------------------------------------------
    # Toggle panel (CheckButtons)
    # ------------------------------------------------------------------
    labels = list(element_artists.keys())
    actives = [bool(v) for v in [
        obs_artists, drone_start, evader_start, drone_end, evader_end,
        drone_traj, evader_traj, los_artists, pred_artists, horizon_artists,
    ]]

    ax_check = fig.add_axes([0.76, 0.28, 0.22, 0.62])
    ax_check.set_title("Toggle layers", fontsize=10, pad=4)
    check = CheckButtons(ax_check, labels, actives)

    # Tighten up checkbox label font size
    for txt in check.labels:
        txt.set_fontsize(9)

    def on_toggle(label):
        for artist in element_artists[label]:
            artist.set_visible(not artist.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(on_toggle)

    # ------------------------------------------------------------------
    # Screenshot button
    # ------------------------------------------------------------------
    ax_btn = fig.add_axes([0.76, 0.12, 0.22, 0.08])
    btn = Button(ax_btn, "Save screenshot", color="0.85", hovercolor="0.70")
    btn.label.set_fontsize(9)

    def on_screenshot(_event):
        screenshots_dir = os.path.join(project_root, "evaluation", "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        fname = os.path.join(
            screenshots_dir,
            f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        )
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Screenshot saved: {fname}")

    btn.on_clicked(on_screenshot)

    # ------------------------------------------------------------------
    # Info text (bottom of right panel)
    # ------------------------------------------------------------------
    info_lines = [
        f"Steps: {n_steps}",
        f"Duration: {total_time:.2f} s",
        f"dt: {dt} s",
        f"Obs: {len(env.obstacles)}",
        f"Pred (next): {'yes' if has_predictions else 'no'}",
        f"Pred (horizon): {'yes' if has_horizon else 'no'}",
        f"LOS interval: 1 s",
    ]
    fig.text(0.765, 0.06, "\n".join(info_lines), fontsize=8,
             verticalalignment="bottom", family="monospace",
             color="0.35")

    plt.show()


if __name__ == "__main__":
    main()
