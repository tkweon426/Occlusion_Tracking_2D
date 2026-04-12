import numpy as np

def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _segment_point_distance(px, py, x1, y1, x2, y2):
    # # Eq.(13)_LOS_trajectory
    # Distance from obstacle center p=(px,py) to a line segment joining two states.
    # In the paper, LOS is parameterized continuously by u_j in [0,1].
    # Here we use the equivalent geometric distance to the segment in 2D.

    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1

    vv = vx * vx + vy * vy
    if vv < 1e-8:
        return np.hypot(px - x1, py - y1)

    t = (wx * vx + wy * vy) / vv
    t = np.clip(t, 0.0, 1.0)

    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return np.hypot(px - proj_x, py - proj_y)


def _min_path_clearance(x1, y1, x2, y2, env, agent_radius, safety_margin):
    # # Remark_2_collision_is_special_case_of_occlusion
    # # Eq.(15)(16)_combined_occlusion_collision_constraint
    # The motion segment from drone to candidate is treated as a collision-free
    # geometric feasibility check with inflated obstacles.

    min_clearance = np.inf

    for obs in env.obstacles:
        r_eff = obs.radius + agent_radius + safety_margin
        d = _segment_point_distance(obs.cx, obs.cy, x1, y1, x2, y2)
        clearance = d - r_eff
        min_clearance = min(min_clearance, clearance)

    return min_clearance


def _candidate_visible_and_safe(px, py, ex, ey, env, agent_radius):
    # # Eq.(13)_LOS_trajectory
    # # Eq.(15)(16)_occlusion_free_constraint
    # Candidate point must:
    # 1) be collision-free itself
    # 2) maintain a clear LOS to the target

    if env.check_collision(px, py, agent_radius=agent_radius):
        return False

    if not env.has_line_of_sight(px, py, ex, ey):
        return False

    return True


def _select_feasible_observation_point(drone_state, evader_state, env):
    # # Eq.(1b)_tracking_range_constraint
    # # Eq.(9)(10)_tracking_reformulation
    # We approximate the tracking shell by sampling candidate viewpoints
    # on a circle around the target, instead of solving for d_r, alpha_r, beta_r explicitly.
    #
    # # Sec.IV_G_MPC_through_real_time_iteration
    # This is a receding-horizon-like viewpoint update:
    # choose a feasible observation point at every control cycle.

    x, y, psi, x_dot, y_dot, psi_dot = drone_state
    ex, ey = evader_state

    desired_radius = 6.0
    num_candidates = 60
    agent_radius = 0.9
    safety_margin = 0.35
    path_margin = 0.35

    current_bearing = np.arctan2(y - ey, x - ex)

    valid_candidates = []
    fallback_candidates = []

    for k in range(num_candidates):
        angle = 2.0 * np.pi * k / num_candidates

        # # Eq.(9)(10)_tracking_reformulation
        # In the paper, tracking is represented by distance/angle variables.
        # Here we generate an equivalent desired tracking point on a ring.
        px = ex + desired_radius * np.cos(angle)
        py = ey + desired_radius * np.sin(angle)

        # # Eq.(15)(16)_occlusion_free_constraint
        # Candidate itself must be collision-free and must preserve LOS.
        if not _candidate_visible_and_safe(px, py, ex, ey, env, agent_radius):
            continue

        # # Remark_2_collision_is_special_case_of_occlusion
        # Additional feasibility: path from current drone state to candidate
        # should not pass too close to inflated obstacles.
        path_clearance = _min_path_clearance(
            x, y, px, py, env,
            agent_radius=agent_radius,
            safety_margin=path_margin
        )

        # # Eq.(22a)_smoothness_objective_surrogate
        # The paper minimizes acceleration norm globally.
        # Here we use a local surrogate score:
        # shorter motion + smaller bearing change + larger path clearance.
        bearing_penalty = abs(_wrap_to_pi(angle - current_bearing))
        distance_penalty = np.hypot(px - x, py - y)
        path_penalty = 0.0 if path_clearance > 1.5 else 8.0 * (1.5 - path_clearance)

        score = distance_penalty + 0.8 * bearing_penalty + path_penalty

        fallback_candidates.append((score, px, py, path_clearance))

        # Prefer candidates with strictly positive path clearance
        if path_clearance > 0.0:
            valid_candidates.append((score, px, py, path_clearance))

    if valid_candidates:
        valid_candidates.sort(key=lambda item: item[0])
        _, gx, gy, _ = valid_candidates[0]
        return np.array([gx, gy])

    if fallback_candidates:
        fallback_candidates.sort(key=lambda item: item[0])
        _, gx, gy, _ = fallback_candidates[0]
        return np.array([gx, gy])

    # # Sec.VII_limitations_fallback_behavior
    # If nothing feasible is found, hold current position rather than
    # directly rushing toward the target.
    return np.array([x, y])


def _tracking_controller(drone_state, goal_xy, look_at_xy, env):
    # # Eq.(22a)_smoothness_objective_surrogate
    # Local PD tracking toward the selected feasible goal point.
    #
    # # Independent_Camera_Control_Assumption
    # As in Sec.IV_A, yaw is used to align with the LOS/target direction.
    #
    # This is not the exact optimizer of Eq.(22)-(29),
    # but a low-level execution layer for the selected feasible waypoint.

    x, y, psi, x_dot, y_dot, psi_dot = drone_state
    gx, gy = goal_xy
    lx, ly = look_at_xy

    g = 9.81

    dx = gx - x
    dy = gy - y

    kp = 3.0
    kd = 3.4

    ax_des = kp * dx - kd * x_dot
    ay_des = kp * dy - kd * y_dot

    # # Eq.(15)(16)_combined_occlusion_collision_constraint_execution_filter
    # Small local safety filter near inflated obstacles.
    # It is NOT the main planning logic.
    # It only prevents the executed motion from stepping inward when already too close.
    drone_radius = 0.9
    safety_margin = 0.35
    emergency_buffer = 0.7
    filter_gain = 14.0

    for obs in env.obstacles:
        ox, oy = obs.cx, obs.cy
        r_eff = obs.radius + drone_radius + safety_margin

        vx = x - ox
        vy = y - oy
        dist = np.hypot(vx, vy) + 1e-6

        nx = vx / dist
        ny = vy / dist

        surface_dist = dist - r_eff

        if surface_dist < emergency_buffer:
            # Remove acceleration component pointing toward obstacle interior
            inward_comp = ax_des * (-nx) + ay_des * (-ny)
            if inward_comp > 0.0:
                ax_des -= inward_comp * (-nx)
                ay_des -= inward_comp * (-ny)

            # Add modest outward correction
            corr = filter_gain * (emergency_buffer - surface_dist)
            ax_des += corr * nx
            ay_des += corr * ny

    # # Eq.(22a)_smoothness_objective_surrogate
    # Bounded planar acceleration
    a_max = 10.0
    a_mag = np.hypot(ax_des, ay_des)
    if a_mag > a_max:
        scale = a_max / a_mag
        ax_des *= scale
        ay_des *= scale

    # # Kinematic_Model_Assumption_Sec.IV_A
    # Convert desired world-frame planar acceleration to attitude commands.
    c = np.cos(psi)
    s = np.sin(psi)

    tan_theta = (c * ax_des + s * ay_des) / g
    tan_phi = (-s * ax_des + c * ay_des) / g

    theta = np.clip(np.arctan(tan_theta), -0.55, 0.55)
    phi = np.clip(np.arctan(tan_phi), -0.55, 0.55)

    # # Independent_Camera_Control_Assumption
    # Yaw aligns with target/LOS direction
    target_angle = np.arctan2(ly - y, lx - x)
    yaw_error = _wrap_to_pi(target_angle - psi)

    kp_yaw = 4.5
    kd_yaw = 2.2
    tau_z = np.clip(kp_yaw * yaw_error - kd_yaw * psi_dot, -1.0, 1.0)

    return np.array([theta, phi, tau_z])


def occlusion_aware_controller(drone_state, evader_state, env):
    """
    # Sec.IV_G_MPC_through_real_time_iteration
    Receding-horizon-like heuristic controller:

    1) # Eq.(9)(10)_tracking_reformulation
       Sample candidate observation points around the target.

    2) # Eq.(13)_LOS_trajectory
       # Eq.(15)(16)_occlusion_free_constraint
       Keep candidates that are collision-free and maintain clear LOS.

    3) # Remark_2_collision_is_special_case_of_occlusion
       Prefer candidates whose transition path from current drone state is also safe.

    4) # Eq.(22a)_smoothness_objective_surrogate
       Choose the lowest-cost feasible candidate.

    5) # Independent_Camera_Control_Assumption
       Track the selected candidate while yaw keeps facing the target.

    This is a paper-style heuristic approximation:
    feasible geometry first, control second.
    """

    ex, ey = evader_state

    goal_xy = _select_feasible_observation_point(drone_state, evader_state, env)
    look_at_xy = np.array([ex, ey])

    return _tracking_controller(drone_state, goal_xy, look_at_xy, env)