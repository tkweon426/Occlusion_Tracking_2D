import numpy as np

from optim.paper_2d_occlusion import Paper2DOcclusionSolver


# # Paper method: MPC / receding-horizon wrapper around the multi-convex solve
_solver = Paper2DOcclusionSolver(
    horizon=20,
    dt=0.08,
    max_alt_iters=6,
    desired_range=4.0,
    min_range=2.5,
    max_range=5.5,
    w_smooth=8.0,
    w_track=120.0,
    w_occ=50.0,
    w_anchor=15000.0,
)

_prev_target = None
_prev_target_vel = np.zeros(2, dtype=float)


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _estimate_target_velocity(evader_state, dt):
    global _prev_target, _prev_target_vel

    ex, ey = evader_state[:2]
    cur = np.array([ex, ey], dtype=float)

    if _prev_target is None:
        _prev_target = cur.copy()
        return _prev_target_vel.copy()

    vel = (cur - _prev_target) / max(dt, 1e-8)
    _prev_target = cur.copy()

    # this is just target prediction plumbing for MPC, not the paper's core optimizer
    alpha = 0.25
    _prev_target_vel = (1.0 - alpha) * _prev_target_vel + alpha * vel
    return _prev_target_vel.copy()


def _pd_track_point(drone_state, goal_xy, look_at_xy):
    # low-level tracking only; the planning logic is in the paper-style optimizer above
    x, y, psi, x_dot, y_dot, psi_dot = drone_state
    gx, gy = goal_xy
    lx, ly = look_at_xy

    g = 9.81

    dx = gx - x
    dy = gy - y

    kp = 2.4
    kd = 2.0

    ax_des = kp * dx - kd * x_dot
    ay_des = kp * dy - kd * y_dot

    ax_des = np.clip(ax_des, -3.0, 3.0)
    ay_des = np.clip(ay_des, -3.0, 3.0)

    theta_cmd = np.clip(-ax_des / g, -0.35, 0.35)
    phi_cmd = np.clip(ay_des / g, -0.35, 0.35)

    psi_des = np.arctan2(ly - y, lx - x)
    psi_err = _wrap_to_pi(psi_des - psi)

    kp_psi = 2.0
    kd_psi = 0.8
    tau_z = np.clip(kp_psi * psi_err - kd_psi * psi_dot, -1.5, 1.5)

    return np.array([theta_cmd, phi_cmd, tau_z], dtype=float)


def paper_occlusion_controller(drone_state, evader_state, env, dt=0.01):
    x, y = drone_state[0], drone_state[1]
    ex, ey = evader_state[:2]

    target_vel = _estimate_target_velocity(evader_state, dt)

    sol = _solver.solve(
        drone_xy=np.array([x, y], dtype=float),
        target_xy=np.array([ex, ey], dtype=float),
        target_vel=target_vel,
        obstacles=env.obstacles,
    )

    x_plan = sol["x"]
    y_plan = sol["y"]

    # # Paper method in receding-horizon form:
    # use the first planned move from the optimized trajectory
    if len(x_plan) >= 2:
        goal_xy = (x_plan[1], y_plan[1])
    else:
        goal_xy = (x_plan[0], y_plan[0])

    look_at_xy = (ex, ey)
    return _pd_track_point(drone_state, goal_xy, look_at_xy)