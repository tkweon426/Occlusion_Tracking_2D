import numpy as np

from optim.paper_2d_occlusion import Paper2DOcclusionSolver


_solver = Paper2DOcclusionSolver()

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

    vel = (cur - _prev_target) / max(dt, 1e-6)
    _prev_target = cur.copy()

    alpha = 0.25
    _prev_target_vel = (1.0 - alpha) * _prev_target_vel + alpha * vel
    return _prev_target_vel.copy()


def _pd_track_point(drone_state, goal_xy, look_at_xy):
    x, y, psi, x_dot, y_dot, psi_dot = drone_state
    gx, gy = goal_xy
    lx, ly = look_at_xy

    g = 9.81

    dx = gx - x
    dy = gy - y

    kp = 2.8
    kd = 2.2

    ax_des = kp * dx - kd * x_dot
    ay_des = kp * dy - kd * y_dot

    ax_des = np.clip(ax_des, -4.0, 4.0)
    ay_des = np.clip(ay_des, -4.0, 4.0)

    theta_cmd = -ax_des / g
    phi_cmd = ay_des / g

    theta_cmd = np.clip(theta_cmd, -0.35, 0.35)
    phi_cmd = np.clip(phi_cmd, -0.35, 0.35)

    psi_des = np.arctan2(ly - y, lx - x)
    psi_err = _wrap_to_pi(psi_des - psi)

    kp_psi = 2.4
    kd_psi = 1.0
    tau_z = kp_psi * psi_err - kd_psi * psi_dot
    tau_z = np.clip(tau_z, -1.5, 1.5)

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

    if len(x_plan) >= 2:
        goal_xy = (x_plan[1], y_plan[1])
    else:
        goal_xy = (x_plan[0], y_plan[0])

    look_at_xy = (ex, ey)
    return _pd_track_point(drone_state, goal_xy, look_at_xy)