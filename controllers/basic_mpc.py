# controllers/basic_mpc.py
import numpy as np
from scipy.optimize import minimize


class BasicMPC:
    """
    Model Predictive Controller for the occlusion-tracking drone.

    Optimization variables are world-frame virtual accelerations [ax, ay] over
    a finite horizon. These are converted to body angles (theta, phi) via the
    same inverse-kinematics as basic_tracker.py. Yaw is controlled independently
    with a PD loop.

    Hard constraints:
      - d_min <= dist(drone, evader) <= d_max  at every horizon step
      - dist(drone_k, obs) >= obs.radius + safety_margin  per obstacle per step

    The environment is passed at construction time, so swapping to a different
    environment in args.py automatically updates the obstacle list seen here.
    """

    def __init__(
        self,
        env,
        dt=0.1,
        N=15,
        d_min=2.0,
        d_max=4.0,
        a_max=8.0,
        safety_margin=0.6,
        w_range=5.0,
        w_effort=0.1,
        kp_yaw=4.0,
        kd_yaw=2.0,
        tau_z_max=1.0,
    ):
        self.env = env
        self.dt = dt
        self.N = N
        self.d_min = d_min
        self.d_max = d_max
        self.a_max = a_max
        self.safety_margin = safety_margin
        self.w_range = w_range
        self.w_effort = w_effort
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.tau_z_max = tau_z_max

        self._d_mid = (d_min + d_max) / 2.0
        self._n_vars = 2 * N
        self._u_prev = np.zeros(self._n_vars)
        self._bounds = [(-a_max, a_max)] * self._n_vars

        # Rollout cache — invalidated at the start of each _solve_mpc call
        self._last_u_hash = None
        self._last_states = None

        self._fallback_count = 0

        self._g = 9.81

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, drone_state, evader_state):
        """
        Same signature as basic_chase_controller.

        drone_state  : [x, y, psi, x_dot, y_dot, psi_dot]
        evader_state : [x, y]
        returns      : np.array([theta, phi, tau_z])
        """
        x, y, psi, vx, vy, psi_dot = drone_state
        drone_xy_vel = np.array([x, y, vx, vy])
        evader_xy = np.asarray(evader_state[:2], dtype=float)

        u_opt, success = self._solve_mpc(drone_xy_vel, evader_xy)

        if success and np.isfinite(u_opt).all():
            ax_des, ay_des = u_opt[0], u_opt[1]
            self._shift_warm_start(u_opt)
        else:
            ax_des, ay_des = self._fallback(drone_xy_vel, evader_xy, u_opt)

        theta, phi = self._virtual_to_angles(ax_des, ay_des, psi)
        tau_z = self._yaw_control(drone_state, evader_xy)

        return np.array([theta, phi, tau_z])

    # ------------------------------------------------------------------
    # MPC internals
    # ------------------------------------------------------------------

    def _rollout(self, u_flat, drone_xy_vel):
        """
        Forward-Euler rollout of the simplified planar model.

        State: [x, y, vx, vy]
        Returns list of N+1 states (each a length-4 numpy array).
        """
        dt = self.dt
        states = np.empty((self.N + 1, 4))
        states[0] = drone_xy_vel
        for k in range(self.N):
            s = states[k]
            ax = u_flat[2 * k]
            ay = u_flat[2 * k + 1]
            states[k + 1, 0] = s[0] + dt * s[2]
            states[k + 1, 1] = s[1] + dt * s[3]
            states[k + 1, 2] = s[2] + dt * ax
            states[k + 1, 3] = s[3] + dt * ay
        return states

    def _get_states(self, u_flat, drone_xy_vel):
        """Cached rollout — reuses the last result when u_flat is unchanged."""
        key = u_flat.tobytes()
        if key != self._last_u_hash:
            self._last_states = self._rollout(u_flat, drone_xy_vel)
            self._last_u_hash = key
        return self._last_states

    def _cost(self, u_flat, drone_xy_vel, evader_xy):
        states = self._get_states(u_flat, drone_xy_vel)
        ex, ey = evader_xy
        cost = 0.0
        for k in range(1, self.N + 1):
            xk, yk = states[k, 0], states[k, 1]
            dist = np.hypot(xk - ex, yk - ey)
            cost += self.w_range * (dist - self._d_mid) ** 2
        cost += self.w_effort * np.dot(u_flat, u_flat)
        return cost

    def _all_constraints(self, u_flat, drone_xy_vel, evader_xy):
        """
        Single vector-valued constraint function — all constraints in one call.

        Returns a 1D array where every element must be >= 0 for SLSQP.
        Using one vector function instead of many scalar functions means scipy
        only needs n+1 rollouts per gradient step (vs m*(n+1) for m scalars).

        Layout per horizon step k=1..N:
          [dist_k - d_min,  d_max - dist_k,  obs_clearance_k_0, ...]
        """
        states = self._get_states(u_flat, drone_xy_vel)
        ex, ey = evader_xy
        n_obs = len(self.env.obstacles)
        cons = np.empty(self.N * (2 + n_obs))
        idx = 0
        for k in range(1, self.N + 1):
            xk, yk = states[k, 0], states[k, 1]
            dist = np.hypot(xk - ex, yk - ey)
            cons[idx]     = dist - self.d_min
            cons[idx + 1] = self.d_max - dist
            idx += 2
            for obs in self.env.obstacles:
                cons[idx] = np.hypot(xk - obs.cx, yk - obs.cy) - (obs.radius + self.safety_margin)
                idx += 1
        return cons

    def _solve_mpc(self, drone_xy_vel, evader_xy):
        """Run SLSQP and return (u_opt, success)."""
        self._last_u_hash = None  # invalidate rollout cache for this timestep

        constraint = {
            'type': 'ineq',
            'fun': self._all_constraints,
            'args': (drone_xy_vel, evader_xy),
        }

        result = minimize(
            fun=self._cost,
            x0=self._u_prev,
            args=(drone_xy_vel, evader_xy),
            method='SLSQP',
            bounds=self._bounds,
            constraints=constraint,
            options={'maxiter': 200, 'ftol': 1e-4},
        )
        return result.x, result.success

    def _shift_warm_start(self, u_opt):
        """Shift the solution left by one step to warm-start the next call."""
        u_shifted = np.empty_like(u_opt)
        u_shifted[:-2] = u_opt[2:]
        u_shifted[-2:] = u_opt[-2:]   # hold last input
        self._u_prev = u_shifted

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback(self, drone_xy_vel, evader_xy, bad_u):
        """
        Three-level fallback when the optimizer fails.

        Level 1: use optimizer's best-effort result if finite.
        Level 2: proportional pull toward d_mid.
        Level 3: zero acceleration (hover).
        """
        self._fallback_count += 1
        if self._fallback_count % 50 == 1:
            print(f"[BasicMPC] Fallback #{self._fallback_count}")

        # Level 1
        if np.isfinite(bad_u).all():
            return bad_u[0], bad_u[1]

        # Level 2
        x, y = drone_xy_vel[0], drone_xy_vel[1]
        ex, ey = evader_xy
        dx, dy = ex - x, ey - y
        dist = np.hypot(dx, dy)
        if dist > 1e-6:
            scale = (dist - self._d_mid) / dist
            kp_fb = 3.0
            ax = np.clip(kp_fb * scale * dx, -self.a_max, self.a_max)
            ay = np.clip(kp_fb * scale * dy, -self.a_max, self.a_max)
            return ax, ay

        # Level 3
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Helpers (inverse kinematics + yaw PD)
    # ------------------------------------------------------------------

    def _virtual_to_angles(self, ax, ay, psi):
        """
        Convert world-frame accelerations to body pitch/roll angles.
        Mirrors the inverse-kinematics block in basic_tracker.py.
        """
        c, s = np.cos(psi), np.sin(psi)
        tan_theta = (c * ax + s * ay) / self._g
        tan_phi   = (-s * ax + c * ay) / self._g
        theta = np.clip(np.arctan(tan_theta), -0.5, 0.5)
        phi   = np.clip(np.arctan(tan_phi),   -0.5, 0.5)
        return theta, phi

    def _yaw_control(self, drone_state, evader_xy):
        """
        PD yaw controller to face the evader.
        Mirrors the yaw block in basic_tracker.py.
        """
        x, y, psi, _, _, psi_dot = drone_state
        dx = evader_xy[0] - x
        dy = evader_xy[1] - y
        target_angle = np.arctan2(dy, dx)
        yaw_error = (target_angle - psi + np.pi) % (2 * np.pi) - np.pi
        tau_z = np.clip(
            self.kp_yaw * yaw_error - self.kd_yaw * psi_dot,
            -self.tau_z_max,
            self.tau_z_max,
        )
        return tau_z
