# evaluation/compare_metrics.py
# Produces a 1x4 metric comparison figure overlaying three predictors on each plot.
#
# Usage:
#   python evaluation/compare_metrics.py
#   python evaluation/compare_metrics.py --save comparison_metrics.png

import argparse
import csv
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import args as sim_args

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

PANELS = [
    ("results/mpc_constvel.csv", "Const-Vel"),
    ("results/mpc_velacc.csv",   "Vel+Acc"),
    ("results/mpc_att.csv",      "Att. Field"),
]

COLORS = ["#2563EB", "#EA580C", "#16A34A"]   # blue, orange, green

GRAY = "#6B7280"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_log(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty log: {csv_path}")

    def col(key, cast=float, default=None):
        out = []
        for r in rows:
            raw = r.get(key, "")
            out.append(default if raw in ("", None) else cast(raw))
        return out

    return {
        "sim_time_s":           col("sim_time_s"),
        "visibility_score":     col("visibility_score"),
        "tracking_error_score": col("tracking_error_score"),
        "ctrl_compute_s":       col("ctrl_compute_s"),
        "drone_ax":             col("drone_ax"),
        "drone_ay":             col("drone_ay"),
    }


def accel_magnitude(ax_col, ay_col):
    return [math.hypot(ax, ay) if (ax is not None and ay is not None) else None
            for ax, ay in zip(ax_col, ay_col)]


# ---------------------------------------------------------------------------
# Axis styling
# ---------------------------------------------------------------------------

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=None, help="Output filename (PNG/PDF)")
    cli = parser.parse_args()

    all_data = []
    for rel_path, _ in PANELS:
        abs_path = os.path.join(project_root, rel_path)
        all_data.append(load_log(abs_path))

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.14, top=0.88, wspace=0.32)

    ax_vis, ax_comp, ax_terr, ax_acc = axes

    for data, (_, label), color in zip(all_data, PANELS, COLORS):
        t    = data["sim_time_s"]
        vis  = data["visibility_score"]
        terr = data["tracking_error_score"]
        comp = [x * 1000 for x in data["ctrl_compute_s"]]
        mags = accel_magnitude(data["drone_ax"], data["drone_ay"])

        # -- Visibility --
        ax_vis.fill_between(t, vis, alpha=0.08, color=color)
        ax_vis.plot(t, vis, color=color, linewidth=0.9, label=label)

        # -- Computation time --
        ax_comp.plot(t, comp, color=color, linewidth=0.7, alpha=0.85, label=label)

        # -- Tracking error (line instead of bars so three series stay legible) --
        ax_terr.plot(t, terr, color=color, linewidth=0.8, alpha=0.85, label=label)

        # -- Acceleration magnitude --
        ax_acc.plot(t, mags, color=color, linewidth=0.9, alpha=0.85, label=label)

    # Zero-reference lines
    ax_vis.axhline(0,  color=GRAY, linewidth=0.5, linestyle="--")
    ax_terr.axhline(0, color=GRAY, linewidth=0.7, linestyle="--")
    ax_acc.axhline(0,  color=GRAY, linewidth=0.5, linestyle="--")

    _style_ax(ax_vis,  "Visibility Score",           "Time (s)", "Score")
    _style_ax(ax_comp, "Controller Computation Time","Time (s)", "Time (ms)")
    _style_ax(ax_terr, "Tracking Error Score",       "Time (s)", "Score")
    _style_ax(ax_acc,  "Drone Acceleration |a|",     "Time (s)", "Acceleration (m/s²)")

    # Shared legend below the figure
    handles, labels = ax_vis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(PANELS),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))

    if cli.save:
        out = cli.save if os.path.isabs(cli.save) else os.path.join(project_root, cli.save)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
