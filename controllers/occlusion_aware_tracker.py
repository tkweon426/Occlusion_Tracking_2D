import numpy as np


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _pd_track_point(drone_state, goal_xy, look_at_xy, env):
    x, y, psi, x_dot, y_dot, psi_dot = drone_state
    gx, gy = goal_xy
    lx, ly = look_at_xy

    g = 9.81

    dx = gx - x
    dy = gy - y

    kp = 3.8
    kd = 3.2

    ax_des = kp * dx - kd * x_dot
    ay_des = kp * dy - kd * y_dot

    # --- obstacle repulsion ---
    repulse_x = 0.0
    repulse_y = 0.0

    safety_buffer = 2.0
    repulse_gain = 18.0

    for obs in env.obstacles:
        ox, oy, r = obs.cx, obs.cy, obs.radius

        vx = x - ox
        vy = y - oy
        dist = np.hypot(vx, vy) + 1e-6

        clearance = dist - r
        trigger_dist = r + safety_buffer

        if clearance < trigger_dist:
            strength = repulse_gain * (1.0 / max(clearance, 0.15) - 1.0 / trigger_dist)
            strength = max(strength, 0.0)

            repulse_x += strength * (vx / dist)
            repulse_y += strength * (vy / dist)

    ax_des += repulse_x
    ay_des += repulse_y

    a_max = 14.0
    a_mag = np.hypot(ax_des, ay_des)
    if a_mag > a_max:
        scale = a_max / a_mag
        ax_des *= scale
        ay_des *= scale

    c, s = np.cos(psi), np.sin(psi)
    tan_theta = (c * ax_des + s * ay_des) / g
    tan_phi = (-s * ax_des + c * ay_des) / g

    theta = np.clip(np.arctan(tan_theta), -0.7, 0.7)
    phi = np.clip(np.arctan(tan_phi), -0.7, 0.7)

    target_angle = np.arctan2(ly - y, lx - x)
    yaw_error = _wrap_to_pi(target_angle - psi)

    kp_yaw = 5.0
    kd_yaw = 2.5
    tau_z = np.clip(kp_yaw * yaw_error - kd_yaw * psi_dot, -1.2, 1.2)

    return np.array([theta, phi, tau_z])


def occlusion_aware_controller(drone_state, evader_state, env):
    """
    2D occlusion-aware tracker:
    1) sample candidate observation points on a circle around the evader
    2) remove candidates that collide with obstacles
    3) remove candidates without line of sight to the evader
    4) choose the valid candidate closest to the drone
    5) use PD to track that point while yaw faces the evader
    """
    x, y, psi, x_dot, y_dot, psi_dot = drone_state
    ex, ey = evader_state

    # Desired observation radius around evader
    desired_radius = 5.5        

    # Number of candidate points around target
    num_candidates = 36
    candidate_agent_radius = 0.9

    # Prefer staying near current bearing relative to the target
    current_bearing = np.arctan2(y - ey, x - ex)

    valid_candidates = []
    fallback_candidates = []

    for k in range(num_candidates):
        angle = 2.0 * np.pi * k / num_candidates
        px = ex + desired_radius * np.cos(angle)
        py = ey + desired_radius * np.sin(angle)

        # Small smoothness/bearing penalty so motion is less jumpy
        bearing_penalty = abs(_wrap_to_pi(angle - current_bearing))
        distance_penalty = np.hypot(px - x, py - y)

        min_clearance = 1e9
        for obs in env.obstacles:
            clearance = np.hypot(px - obs.cx, py - obs.cy) - obs.radius
            min_clearance = min(min_clearance, clearance)

        clearance_penalty = 0.0
        if min_clearance < 2.0:
            clearance_penalty = 8.0 * (2.0 - min_clearance)
        score = distance_penalty + 0.6 * bearing_penalty + clearance_penalty


        # 1) collision-free candidate point
        collides = env.check_collision(px, py, agent_radius=candidate_agent_radius)
        if collides:
            continue

        # Save as fallback even if LOS is blocked
        fallback_candidates.append((score, px, py))

        # 2) LOS must be clear from candidate point to evader
        visible = env.has_line_of_sight(px, py, ex, ey)
        if visible:
            valid_candidates.append((score, px, py))

    # Best case: choose nearest smooth visible point
    if valid_candidates:
        valid_candidates.sort(key=lambda item: item[0])
        _, gx, gy = valid_candidates[0]
        return _pd_track_point(drone_state, (gx, gy), (ex, ey), env)

    # If no visible point exists, at least choose a collision-free point
    if fallback_candidates:
        fallback_candidates.sort(key=lambda item: item[0])
        _, gx, gy = fallback_candidates[0]
        return _pd_track_point(drone_state, (gx, gy), (ex, ey), env)

    # Final fallback: direct chase if everything else fails
    return _pd_track_point(drone_state, (gx, gy), (ex, ey), env)