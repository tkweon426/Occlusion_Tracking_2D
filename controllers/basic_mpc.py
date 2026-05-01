import numpy as np
from scipy.optimize import minimize
from predictors.constvel_predictor import ConstVelPredictor


class BasicMPC:
    """
    Model Predictive Controller for the occlusion-tracking drone.

    Optimization variables are world-frame virtual accelerations [ax, ay] over
    a finite horizon. These are converted to body angles (theta, phi) via the
    same inverse-kinematics as basic_tracker.py. Yaw is controlled independently
    with a PD loop.

    Hard constraints:
      - dist(drone_k, evader_k) >= d_min  at every horizon step (lower bound)
      - dist(drone_N, evader_N) <= d_max  at terminal step only (upper bound)
      - dist(drone_k, obs) >= obs.radius + safety_margin  per obstacle per step

    Evader positions are predicted via ConstVelPredictor (constant-velocity
    assumption with EMA-smoothed finite-difference velocity estimate).

    The environment is passed at construction time, so swapping to a different
    environment in args.py automatically updates the obstacle list seen here.
    """

    def __init__(
        self,
        env,
        dt=0.1,
        N=15,
        sim_dt=0.01,
        d_min=5.0,
        d_max=10.0,
        a_max=6.87,
        safety_margin=0.6,
        w_effort=0.1,
        kp_yaw=4.0,
        kd_yaw=2.0,
        tau_z_max=1.0,
    ):
        self.env = env
        self.dt = dt
        self.N = N
        self._predictor = ConstVelPredictor(sim_dt=sim_dt)
        self.d_min = d_min
        self.d_max = d_max
        self.a_max = a_max
        self.safety_margin = safety_margin
        self.w_effort = w_effort
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.tau_z_max = tau_z_max

        self._n_vars = 2 * N
        self._u_prev = np.zeros(self._n_vars)
        self._bounds = [(-a_max, a_max)] * self._n_vars

        self._last_u_hash = None
        self._last_states = None

        self._fallback_count = 0

        self._g = 9.81

    def __call__(self, drone_state, evader_state):
        x, y, psi, vx, vy, psi_dot = drone_state
        drone_xy_vel = np.array([x, y, vx, vy])
        evader_xy = np.asarray(evader_state[:2], dtype=float)

        evader_traj = self._predictor.predict(evader_xy, self.dt, self.N)
        self.last_evader_traj = evader_traj

        u_opt, success = self._solve_mpc(drone_xy_vel, evader_traj)

        if success and np.isfinite(u_opt).all():
            ax_des, ay_des = u_opt[0], u_opt[1]
            self._shift_warm_start(u_opt)
        else:
            ax_des, ay_des = self._fallback(drone_xy_vel, evader_traj[0], u_opt)

        theta, phi = self._virtual_to_angles(ax_des, ay_des, psi)
        tau_z = self._yaw_control(drone_state, evader_traj[0])

        return np.array([theta, phi, tau_z])

    def _rollout(self, u_flat, drone_xy_vel):
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
        key = u_flat.tobytes()
        if key != self._last_u_hash:
            self._last_states = self._rollout(u_flat, drone_xy_vel)
            self._last_u_hash = key
        return self._last_states

    def _cost(self, u_flat):
        return self.w_effort * np.dot(u_flat, u_flat)

    def _all_constraints(self, u_flat, drone_xy_vel, evader_traj):
        """
        Single vector-valued constraint function — all constraints in one call.

        Returns a 1D array where every element must be >= 0 for SLSQP.
        Using one vector function instead of many scalar functions means scipy
        only needs n+1 rollouts per gradient step (vs m*(n+1) for m scalars).

        The upper bound on distance is applied only at the terminal step, not
        every step. This keeps the problem feasible when the evader has moved
        outside range — the optimizer has the full horizon to close the gap.
        """
        states = self._get_states(u_flat, drone_xy_vel)
        n_obs = len(self.env.obstacles)
        cons = np.empty(self.N * (1 + n_obs) + 1)
        idx = 0
        for k in range(1, self.N + 1):
            xk, yk = states[k, 0], states[k, 1]
            ex, ey = evader_traj[k]
            dist = np.hypot(xk - ex, yk - ey)
            cons[idx] = dist - self.d_min
            idx += 1
            for obs in self.env.obstacles:
                if hasattr(obs, 'rx'):
                    ct, st = np.cos(obs.theta), np.sin(obs.theta)
                    dx = xk - obs.cx
                    dy = yk - obs.cy
                    lx = ( ct * dx + st * dy) / (obs.rx + self.safety_margin)
                    ly = (-st * dx + ct * dy) / (obs.ry + self.safety_margin)
                    cons[idx] = np.hypot(lx, ly) - 1.0
                else:
                    cons[idx] = np.hypot(xk - obs.cx, yk - obs.cy) - (obs.radius + self.safety_margin)
                idx += 1
        # upper bound at terminal step only
        xN, yN = states[self.N, 0], states[self.N, 1]
        eNx, eNy = evader_traj[self.N]
        cons[idx] = self.d_max - np.hypot(xN - eNx, yN - eNy)
        return cons

    def _solve_mpc(self, drone_xy_vel, evader_traj):
        self._last_u_hash = None

        constraint = {
            'type': 'ineq',
            'fun': self._all_constraints,
            'args': (drone_xy_vel, evader_traj),
        }

        result = minimize(
            fun=self._cost,
            x0=self._u_prev,
            method='SLSQP',
            bounds=self._bounds,
            constraints=constraint,
            options={'maxiter': 100, 'ftol': 1e-4},
        )
        return result.x, result.success

    def _shift_warm_start(self, u_opt):
        u_shifted = np.empty_like(u_opt)
        u_shifted[:-2] = u_opt[2:]
        u_shifted[-2:] = u_opt[-2:]
        self._u_prev = u_shifted

    def _fallback(self, drone_xy_vel, evader_xy, bad_u):
        self._fallback_count += 1
        if self._fallback_count % 50 == 1:
            print(f"[BasicMPC] Fallback #{self._fallback_count}")

        if np.isfinite(bad_u).all():
            return bad_u[0], bad_u[1]

        x, y = drone_xy_vel[0], drone_xy_vel[1]
        ex, ey = evader_xy
        dx, dy = ex - x, ey - y
        dist = np.hypot(dx, dy)
        if dist > 1e-6:
            scale = (dist - (self.d_min + self.d_max) / 2.0) / dist
            kp_fb = 3.0
            ax = np.clip(kp_fb * scale * dx, -self.a_max, self.a_max)
            ay = np.clip(kp_fb * scale * dy, -self.a_max, self.a_max)
            return ax, ay

        return 0.0, 0.0

    def _virtual_to_angles(self, ax, ay, psi):
        c, s = np.cos(psi), np.sin(psi)
        tan_theta = (c * ax + s * ay) / self._g
        tan_phi   = (-s * ax + c * ay) / self._g
        theta = np.clip(np.arctan(tan_theta), -0.5, 0.5)
        phi   = np.clip(np.arctan(tan_phi),   -0.5, 0.5)
        return theta, phi

    def _yaw_control(self, drone_state, evader_xy):
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
