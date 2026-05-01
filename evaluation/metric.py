# Usage:
#   python evaluation/metric.py results/log_20260422_204007.csv

import argparse
import csv
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


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
        "timestep":             col("timestep", int),
        "sim_time_s":           col("sim_time_s"),
        "visibility_score":     col("visibility_score"),
        "tracking_error_score": col("tracking_error_score"),
        "ctrl_compute_s":       col("ctrl_compute_s"),
        "drone_ax":             col("drone_ax"),
        "drone_ay":             col("drone_ay"),
    }


def stats(values, skip_none=True):
    v = [x for x in values if x is not None] if skip_none else values
    if not v:
        return dict(mean=float("nan"), mn=float("nan"), mx=float("nan"), total=float("nan"))
    arr = np.array(v, dtype=float)
    return dict(mean=float(np.mean(arr)), mn=float(np.min(arr)), mx=float(np.max(arr)), total=float(np.sum(arr)))


def accel_magnitude(ax_col, ay_col):
    return [math.hypot(ax, ay) if (ax is not None and ay is not None) else None
            for ax, ay in zip(ax_col, ay_col)]


def accel_cost_score(ax_col, ay_col, dt):
    """Mean squared acceleration magnitude (standard LQR-style control cost)."""
    mags_sq = [(ax**2 + ay**2) for ax, ay in zip(ax_col, ay_col)
               if ax is not None and ay is not None]
    return float(np.mean(mags_sq)) if mags_sq else float("nan")


def make_report_path(log_path):
    base = os.path.splitext(os.path.basename(log_path))[0]
    report_name = base.replace("log", "metric", 1) + ".pdf"
    return os.path.join(project_root, "evaluation", report_name)


BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
GRAY   = "#6B7280"

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_visibility(ax, t, vis):
    ax.fill_between(t, vis, alpha=0.15, color=GREEN)
    ax.plot(t, vis, color=GREEN, linewidth=0.9)
    ax.axhline(0, color=GRAY, linewidth=0.5, linestyle="--")
    _style_ax(ax, "Visibility Score", "Time (s)", "Score")


def plot_computation(ax, t, comp_ms):
    ax.plot(t, comp_ms, color=BLUE, linewidth=0.7, alpha=0.85)
    _style_ax(ax, "Controller Computation Time", "Time (s)", "Time (ms)")


def plot_tracking_error(ax, t, err):
    colors = [RED if e < 0 else GREEN for e in err]
    ax.bar(t, err, width=(t[1] - t[0]) if len(t) > 1 else 0.01,
           color=colors, alpha=0.6, linewidth=0)
    ax.axhline(0, color=GRAY, linewidth=0.7)
    _style_ax(ax, "Tracking Error Score", "Time (s)", "Score (negative = out of band)")


def plot_acceleration(ax, t, ax_col, ay_col, mags):
    ax.plot(t, ax_col, color=BLUE,   linewidth=0.7, alpha=0.8, label="aₓ")
    ax.plot(t, ay_col, color=RED,    linewidth=0.7, alpha=0.8, label="aᵧ")
    ax.plot(t, mags,   color=ORANGE, linewidth=0.9, alpha=0.9, label="|a|")
    ax.axhline(0, color=GRAY, linewidth=0.5, linestyle="--")
    ax.legend(fontsize=7, framealpha=0.6)
    _style_ax(ax, "Drone Acceleration", "Time (s)", "Acceleration (m/s²)")


def make_summary_text(data, dt, log_path):
    n = len(data["timestep"])
    duration = data["sim_time_s"][-1] if data["sim_time_s"] else 0.0

    vis   = stats(data["visibility_score"])
    terr  = stats(data["tracking_error_score"])
    comp  = stats([x * 1000 for x in data["ctrl_compute_s"] if x is not None])
    mags  = accel_magnitude(data["drone_ax"], data["drone_ay"])
    acc   = stats(mags)
    cost  = accel_cost_score(data["drone_ax"], data["drone_ay"], dt)

    lines = [
        f"Log file : {os.path.basename(log_path)}",
        f"Steps    : {n}",
        f"Duration : {duration:.2f} s    dt: {dt} s",
        "",
        "── Visibility Score ─────────────────────────────────────────",
        f"  Accumulated total : {vis['total']:.4f}",
        f"  Average           : {vis['mean']:.4f}",
        f"  Max               : {vis['mx']:.4f}",
        f"  Min               : {vis['mn']:.4f}",
        "",
        "── Tracking Error Score ─────────────────────────────────────",
        f"  Accumulated total : {terr['total']:.4f}",
        f"  Average           : {terr['mean']:.4f}",
        f"  Max               : {terr['mx']:.4f}",
        f"  Min               : {terr['mn']:.4f}",
        "  (0 = within 5–10 m band; negative = distance outside band)",
        "",
        "── Controller Computation Time ──────────────────────────────",
        f"  Average           : {comp['mean']:.3f} ms",
        f"  Max               : {comp['mx']:.3f} ms",
        f"  Min               : {comp['mn']:.3f} ms",
        "",
        "── Drone Acceleration (|a| = √(aₓ²+aᵧ²)) ───────────────────",
        f"  Average |a|       : {acc['mean']:.4f} m/s²",
        f"  Max |a|           : {acc['mx']:.4f} m/s²",
        f"  Min |a|           : {acc['mn']:.4f} m/s²",
        f"  Acceleration cost : {cost:.4f} m²/s⁴",
        "  (cost = mean squared |a|, lower is smoother)",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate a metric report from a simulation log.")
    parser.add_argument("log", help="Path to the CSV log file")
    cli = parser.parse_args()

    log_path = cli.log if os.path.isabs(cli.log) else os.path.join(project_root, cli.log)
    data = load_log(log_path)

    import args as sim_args
    dt = sim_args.DT

    t          = data["sim_time_s"]
    vis        = data["visibility_score"]
    terr       = data["tracking_error_score"]
    comp_ms    = [x * 1000 for x in data["ctrl_compute_s"]]
    ax_col     = data["drone_ax"]
    ay_col     = data["drone_ay"]
    mags       = accel_magnitude(ax_col, ay_col)

    report_path = make_report_path(log_path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with PdfPages(report_path) as pdf:
        fig_s, ax_s = plt.subplots(figsize=(8.5, 11))
        ax_s.axis("off")
        summary = make_summary_text(data, dt, log_path)
        ax_s.text(0.05, 0.95, summary, transform=ax_s.transAxes,
                  fontsize=9, verticalalignment="top", family="monospace",
                  bbox=dict(boxstyle="round,pad=0.6", facecolor="#F3F4F6", edgecolor="#D1D5DB"))
        ax_s.set_title("Simulation Metric Report", fontsize=13, fontweight="bold", pad=14)
        pdf.savefig(fig_s, bbox_inches="tight")
        plt.close(fig_s)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f"Metric Plots — {os.path.basename(log_path)}", fontsize=11, fontweight="bold")
        fig.subplots_adjust(hspace=0.38, wspace=0.32, left=0.08, right=0.97, top=0.92, bottom=0.08)

        plot_visibility(axes[0, 0], t, vis)
        plot_computation(axes[0, 1], t, comp_ms)
        plot_tracking_error(axes[1, 0], t, terr)
        plot_acceleration(axes[1, 1], t, ax_col, ay_col, mags)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    vis_stats = stats(data["visibility_score"])
    print(f"Visibility — mean: {vis_stats['mean']:.4f}  max: {vis_stats['mx']:.4f}  min: {vis_stats['mn']:.4f}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
