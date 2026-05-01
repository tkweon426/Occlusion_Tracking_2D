# trio_test.py
# Runs the simulation headlessly three times — once per predictor — and prints
# visibility scores (mean, max, min) for each.
#
# Usage:
#   python trio_test.py [--steps N]   (default: 1900 steps, ~19 s at dt=0.01)

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.planar_quadrotor import TopDownQuadrotor
from models.evader import Evader
from controllers.occlusion_mpc import FastOcclusionMPC
from controllers.scripted_evader_2 import ScriptedTrajectory_2
from environments.single_ellipse import make_two_obs_env
from predictors.constvel_predictor import ConstVelPredictor
from predictors.velacc_predictor import VelAccPredictor
from predictors.attfield_predictor import AttFieldPredictor

DT            = 0.01
DRONE_START   = (0.0, -3.5)
EVADER_START  = (0.0,  3.0)
DRONE_MASS    = 1.0
DRONE_I_ZZ    = 0.02
DRONE_RADIUS  = 0.5
EVADER_RADIUS = 0.3


def run_sim(predictor_name, n_steps):
    env        = make_two_obs_env()
    drone      = TopDownQuadrotor(x=DRONE_START[0], y=DRONE_START[1],
                                  mass=DRONE_MASS, I_zz=DRONE_I_ZZ)
    evader     = Evader(x=EVADER_START[0], y=EVADER_START[1])
    scripted   = ScriptedTrajectory_2(obstacle_cx=3.0, obstacle_cy=7.0)
    controller = FastOcclusionMPC(env=env, sim_dt=DT)

    if predictor_name == "constvel":
        controller._predictor = ConstVelPredictor(sim_dt=DT)
    elif predictor_name == "velacc":
        controller._predictor = VelAccPredictor(sim_dt=DT)
    elif predictor_name == "attfield":
        controller._predictor = AttFieldPredictor(sim_dt=DT, obstacles=env.obstacles,
                                                   k_att=8.5, d_min=0.3)

    vis_scores = []
    for step in range(n_steps):
        vx, vy       = scripted.get_velocity(evader.state)
        drone_action = controller(drone.state, evader.state)
        evader.step(vx, vy, DT)
        drone.step(drone_action, DT)

        if env.check_collision(drone.state[0], drone.state[1], DRONE_RADIUS):
            print(f"  [drone collision at step {step}]")
            break
        if env.check_collision(evader.state[0], evader.state[1], EVADER_RADIUS):
            print(f"  [evader collision at step {step}]")
            break

        vis_scores.append(env.los_clearance(
            drone.state[0], drone.state[1],
            evader.state[0], evader.state[1],
        ))

    return vis_scores


def main():
    parser = argparse.ArgumentParser(description="Compare three predictors headlessly.")
    parser.add_argument("--steps", type=int, default=1900,
                        help="Number of simulation steps per run (default: 1900)")
    cli = parser.parse_args()

    predictors = ["constvel", "velacc", "attfield"]

    print(f"Running {len(predictors)} simulations × {cli.steps} steps each\n")
    print(f"{'Predictor':<14}  {'Mean':>8}  {'Max':>8}  {'Min':>8}  {'Steps':>6}")
    print("-" * 52)

    for name in predictors:
        print(f"{name:<14}  ", end="", flush=True)
        scores = run_sim(name, cli.steps)
        if scores:
            arr = np.array(scores)
            print(f"{arr.mean():8.4f}  {arr.max():8.4f}  {arr.min():8.4f}  {len(scores):>6}")
        else:
            print("  no data")

    print("-" * 52)


if __name__ == "__main__":
    main()
