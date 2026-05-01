import argparse
import csv
import json
import os
import time
import numpy as np
import pygame
from datetime import datetime
from models.planar_quadrotor import TopDownQuadrotor
from models.evader import Evader
from visualization.renderer import PygameRenderer
import args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record", nargs="?", const="results/recording.mp4", metavar="FILE",
        help="Save the run as a video. Optionally specify a filename (default: results/recording.mp4).",
    )
    parser.add_argument(
        "--log", nargs="?", const="results/log.csv", metavar="FILE",
        help="Log simulation data to a CSV file. Optionally specify a filename (default: results/log_<timestamp>.csv).",
    )
    parser.add_argument(
        "--end", type=int, default=None, metavar="STEP",
        help="Terminate the simulation after this many steps.",
    )
    cli = parser.parse_args()

    if cli.log is not None and cli.log == "results/log.csv":
        cli.log = f"results/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    os.makedirs("results", exist_ok=True)

    dt = args.DT

    drone = TopDownQuadrotor(x=args.DRONE_START[0], y=args.DRONE_START[1],
                             mass=args.DRONE_MASS, I_zz=args.DRONE_I_ZZ)
    evader = Evader(x=args.EVADER_START[0], y=args.EVADER_START[1])
    renderer = PygameRenderer(width=args.RENDERER_WIDTH, height=args.RENDERER_HEIGHT,
                               scale=args.RENDERER_SCALE)
    env = args.ENV_FACTORY()

    clock = pygame.time.Clock()

    scripted = args.EVADER_CONTROLLER  # None → keyboard, object → scripted

    if scripted is None:
        print("Simulation starting. Use W/A/S/D to move the evader.")
    else:
        print("Simulation starting. Evader is following a scripted trajectory.")
    print("Close the window or press Q to quit.")

    VIDEO_FPS = 60
    frames = []
    frame_timer = 0.0
    recording = cli.record is not None
    if recording:
        print(f"Recording enabled — will save to: {cli.record}")

    logging_enabled = cli.log is not None
    log_rows = []
    if logging_enabled:
        print(f"Logging enabled — will save to: {cli.log}")

    running = True
    collided = False
    collision_msg = None
    step = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        if not collided:
            if scripted is None:
                keys = pygame.key.get_pressed()
                vx, vy = 0.0, 0.0
                if keys[pygame.K_w]: vy = evader.speed
                if keys[pygame.K_s]: vy = -evader.speed
                if keys[pygame.K_a]: vx = -evader.speed
                if keys[pygame.K_d]: vx = evader.speed
            else:
                vx, vy = scripted.get_velocity(evader.state)

            t_ctrl_start = time.perf_counter()
            drone_action = args.CONTROLLER(drone.state, evader.state)
            t_ctrl_elapsed = time.perf_counter() - t_ctrl_start

            if logging_enabled:
                theta, phi, tau_z_cmd = drone_action
                g = 9.81
                psi = drone.state[2]
                ax = g * (np.tan(theta) * np.cos(psi) - np.tan(phi) * np.sin(psi))
                ay = g * (np.tan(theta) * np.sin(psi) + np.tan(phi) * np.cos(psi))
                psi_ddot = tau_z_cmd / drone.I_zz

            evader.step(vx, vy, dt)
            drone.step(drone_action, dt)

            if env.check_collision(drone.state[0], drone.state[1], args.DRONE_RADIUS):
                collided = True
                collision_msg = "DRONE COLLISION"
            elif env.check_collision(evader.state[0], evader.state[1], args.EVADER_RADIUS):
                collided = True
                collision_msg = "EVADER COLLISION"

            if logging_enabled:
                pred_traj = getattr(args.CONTROLLER, "last_evader_traj", None)
                pred_next     = pred_traj[1]  if (pred_traj is not None and len(pred_traj) > 1) else (None, None)
                pred_horizon  = pred_traj[-1] if (pred_traj is not None and len(pred_traj) > 1) else (None, None)
                mpc_dt        = getattr(args.CONTROLLER, "dt", "")
                horizon_full  = json.dumps(pred_traj.tolist()) if pred_traj is not None else ""
                log_rows.append({
                    "timestep":        step,
                    "sim_time_s":      round(step * dt, 6),
                    "drone_x":         drone.state[0],
                    "drone_y":         drone.state[1],
                    "drone_psi":       drone.state[2],
                    "drone_vx":        drone.state[3],
                    "drone_vy":        drone.state[4],
                    "drone_psi_dot":   drone.state[5],
                    "drone_ax":        ax,
                    "drone_ay":        ay,
                    "drone_psi_ddot":  psi_ddot,
                    "ctrl_theta":      drone_action[0],
                    "ctrl_phi":        drone_action[1],
                    "ctrl_tau_z":      drone_action[2],
                    "evader_x":        evader.state[0],
                    "evader_y":        evader.state[1],
                    "evader_vx":       vx,
                    "evader_vy":       vy,
                    "ctrl_compute_s":  t_ctrl_elapsed,
                    "visibility_score": env.los_clearance(
                        drone.state[0], drone.state[1],
                        evader.state[0], evader.state[1],
                    ),
                    "tracking_error_score": (lambda d: 0 if 5.0 <= d <= 10.0 else (-(5.0 - d) if d < 5.0 else -(d - 10.0)))(
                        np.hypot(drone.state[0] - evader.state[0], drone.state[1] - evader.state[1])
                    ),
                    "pred_evader_x":        pred_next[0]    if pred_next[0]    is not None else "",
                    "pred_evader_y":        pred_next[1]    if pred_next[1]    is not None else "",
                    "pred_horizon_evader_x": pred_horizon[0] if pred_horizon[0] is not None else "",
                    "pred_horizon_evader_y": pred_horizon[1] if pred_horizon[1] is not None else "",
                    "mpc_dt":               mpc_dt,
                    "pred_horizon_full":    horizon_full,
                })

        renderer.draw(drone.state, evader.state, env=env, collision_msg=collision_msg)

        if recording:
            frame_timer += dt
            if frame_timer >= 1.0 / VIDEO_FPS:
                frame_timer -= 1.0 / VIDEO_FPS
                frame = pygame.surfarray.array3d(renderer.screen).transpose(1, 0, 2)
                frames.append(frame)

        clock.tick(int(1 / dt))
        step += 1

        if cli.end is not None and step >= cli.end:
            running = False

    renderer.quit()

    if logging_enabled and log_rows:
        print(f"Saving log with {len(log_rows)} rows to: {cli.log}")
        with open(cli.log, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"Log saved.")

    if recording and frames:
        import imageio
        print(f"Saving {len(frames)} frames at {VIDEO_FPS} fps...")
        imageio.mimwrite(cli.record, frames, fps=VIDEO_FPS)
        print(f"Video saved to: {cli.record}")

if __name__ == "__main__":
    main()
