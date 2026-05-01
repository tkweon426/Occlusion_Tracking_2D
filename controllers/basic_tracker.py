import numpy as np

def basic_chase_controller(drone_state, evader_state):
    """
    World-frame PD controller with full theta+phi actuation.

    Rather than yaw-then-pitch, we compute a desired world-frame acceleration
    toward the evader and invert the drone kinematics to find the required
    (theta, phi) pair. This lets the drone accelerate in any direction
    regardless of which way it is currently facing.
    """
    x, y, psi, x_dot, y_dot, psi_dot = drone_state
    ex, ey = evader_state
    g = 9.81

    dx = ex - x
    dy = ey - y

    kp = 2.0
    kd = 2.8

    ax_des = kp * dx - kd * x_dot
    ay_des = kp * dy - kd * y_dot

    a_max = 8.0
    a_mag = np.hypot(ax_des, ay_des)
    if a_mag > a_max:
        ax_des *= a_max / a_mag
        ay_des *= a_max / a_mag

    # From dynamics:  [ax, ay] = g * R(psi) * [tan θ, tan φ]ᵀ
    # So:             [tan θ, tan φ] = (1/g) * R(psi)ᵀ * [ax, ay]
    c, s = np.cos(psi), np.sin(psi)
    tan_theta =  (c * ax_des + s * ay_des) / g
    tan_phi   =  (-s * ax_des + c * ay_des) / g

    theta = np.clip(np.arctan(tan_theta), -0.5, 0.5)
    phi   = np.clip(np.arctan(tan_phi),   -0.5, 0.5)

    target_angle = np.arctan2(dy, dx)
    yaw_error = (target_angle - psi + np.pi) % (2 * np.pi) - np.pi

    kp_yaw = 4.0
    kd_yaw = 2.0
    tau_z = np.clip(kp_yaw * yaw_error - kd_yaw * psi_dot, -1.0, 1.0)

    return np.array([theta, phi, tau_z])
