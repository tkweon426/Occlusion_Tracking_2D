# sweep.py
# Grid-search k_att and d_min for AttFieldPredictor inside FastOcclusionMPC.
# Runs headlessly (no pygame/renderer). Edit the search grid constants below.
#
# Usage:
#   python sweep.py

import itertools
import os
import sys
import numpy as np

# Ensure project root is on path when invoked from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.planar_quadrotor import TopDownQuadrotor
from models.evader import Evader
from controllers.occlusion_mpc import FastOcclusionMPC
from controllers.scripted_evader_2 import ScriptedTrajectory_2
from environments.single_ellipse import make_two_obs_env

# ---------------------------------------------------------------------------
# Search grid — edit these to change the sweep
# ---------------------------------------------------------------------------
K_ATT_VALUES = [8, 8.5, 9, 9.5, 10.0]
D_MIN_VALUES = [0.1, 0.3]
N_STEPS      = 1900   # steps per run (~50 s at dt=0.01)

# ---------------------------------------------------------------------------
# Simulation constants (must match args.py)
# ---------------------------------------------------------------------------
DT            = 0.01
DRONE_START   = (0.0, -3.5)
EVADER_START  = (0.0,  3.0)
DRONE_MASS    = 1.0
DRONE_I_ZZ    = 0.02
DRONE_RADIUS  = 0.5
EVADER_RADIUS = 0.3


def run_sim(k_att, predictor_d_min):
    env        = make_two_obs_env()
    drone      = TopDownQuadrotor(x=DRONE_START[0], y=DRONE_START[1],
                                  mass=DRONE_MASS, I_zz=DRONE_I_ZZ)
    evader     = Evader(x=EVADER_START[0], y=EVADER_START[1])
    controller = FastOcclusionMPC(env=env, sim_dt=DT,
                                  predictor_k_att=k_att,
                                  predictor_d_min=predictor_d_min)
    scripted   = ScriptedTrajectory_2(obstacle_cx=3.0, obstacle_cy=7.0)

    vis_scores = []
    for _ in range(N_STEPS):
        vx, vy       = scripted.get_velocity(evader.state)
        drone_action = controller(drone.state, evader.state)
        evader.step(vx, vy, DT)
        drone.step(drone_action, DT)

        if env.check_collision(drone.state[0], drone.state[1], DRONE_RADIUS):
            break
        if env.check_collision(evader.state[0], evader.state[1], EVADER_RADIUS):
            break

        vis_scores.append(env.los_clearance(
            drone.state[0], drone.state[1],
            evader.state[0], evader.state[1],
        ))

    return float(np.mean(vis_scores)) if vis_scores else 0.0


def main():
    combos = list(itertools.product(K_ATT_VALUES, D_MIN_VALUES))
    total  = len(combos)
    results = []

    print(f"Sweeping {total} configurations ({len(K_ATT_VALUES)} k_att × {len(D_MIN_VALUES)} d_min), "
          f"{N_STEPS} steps each\n")

    for i, (k_att, d_min) in enumerate(combos, 1):
        score = run_sim(k_att, d_min)
        results.append((k_att, d_min, score))
        print(f"[{i:>3}/{total}]  k_att={k_att:6.2f}  d_min={d_min:.2f}  →  mean_vis={score:.4f}")

    results.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 52)
    print("  Top 10 configurations by mean visibility score")
    print("=" * 52)
    print(f"  {'k_att':>8}  {'d_min':>6}  {'mean_vis':>10}")
    print("  " + "-" * 28)
    for k, d, s in results[:10]:
        print(f"  {k:8.2f}  {d:6.2f}  {s:10.4f}")

    best = results[0]
    print(f"\n  Best → k_att={best[0]}, d_min={best[1]}, mean_vis={best[2]:.4f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
