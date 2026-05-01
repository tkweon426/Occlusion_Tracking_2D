import numpy as np
from scipy.optimize import minimize
from predictors.constvel_predictor import ConstVelPredictor
from predictors.kalman_predictor import KalmanPredictor
from predictors.attfield_predictor import AttFieldPredictor
from predictors.velacc_predictor import VelAccPredictor

class FastOcclusionMPC:
    def __init__(
        self,
        env,
        dt=0.1,
        N=25,
        sim_dt=0.01,
        d_min=5.0,
        d_max=10.0,
        target_distance=6.0,
        a_max=6.87,
        safety_margin=0.6,
        w_effort=0.1,
        w_distance=10.0,
        w_occlusion=50.0,        # Weight of the LoS barrier
        barrier_steepness=5.0,   # How aggressively the penalty spikes when crossing the boundary
        cull_radius=25.0,
        kp_yaw=4.0,
        kd_yaw=2.0,
        tau_z_max=1.0,
        predictor_k_att=9,
        predictor_d_min=0.1,
    ):
        self.env = env
        self.dt = dt
        self.N = N
        #self._predictor = KalmanPredictor(sim_dt=sim_dt)
        #self._predictor = ConstVelPredictor(sim_dt=sim_dt)
        self._predictor = AttFieldPredictor(sim_dt=sim_dt, obstacles=self.env.obstacles, k_att=predictor_k_att, d_min=predictor_d_min)
        #self._predictor = VelAccPredictor(sim_dt=sim_dt)

        self.d_min = d_min
        self.d_max = d_max
        self.target_distance = target_distance
        self.a_max = a_max
        self.safety_margin = safety_margin

        self.w_effort = w_effort
        self.w_distance = w_distance
        self.w_occlusion = w_occlusion
        self.barrier_steepness = barrier_steepness
        self.cull_radius = cull_radius

        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.tau_z_max = tau_z_max

        self._n_vars = 2 * N
        self._u_prev = np.zeros(self._n_vars)
        self._bounds = [(-a_max, a_max)] * self._n_vars
        self._fallback_count = 0
        self._g = 9.81

        self.M = np.zeros((N, N))
        dt2 = dt * dt
        for k in range(N):
            for i in range(k):
                self.M[k, i] = dt2 * (k - i)
        self.MT = self.M.T

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
            drone_rx, drone_ry = self._get_states_vectorized(u_opt, drone_xy_vel)
            self.last_drone_traj = np.column_stack([
                np.concatenate([[x], drone_rx]),
                np.concatenate([[y], drone_ry]),
            ])
        else:
            ax_des, ay_des = self._fallback(drone_xy_vel, evader_traj[0], u_opt)
            self.last_drone_traj = None

        theta, phi = self._virtual_to_angles(ax_des, ay_des, psi)
        tau_z = self._yaw_control(drone_state, evader_traj[0])

        return np.array([theta, phi, tau_z])

    def _get_states_vectorized(self, u_flat, drone_xy_vel):
        ax = u_flat[0::2]
        ay = u_flat[1::2]
        x0, y0, vx0, vy0 = drone_xy_vel
        steps = np.arange(1, self.N + 1)
        x = (x0 + steps * self.dt * vx0) + self.M @ ax
        y = (y0 + steps * self.dt * vy0) + self.M @ ay
        return x, y

    def _cost_and_jac(self, u_flat, drone_xy_vel, evader_traj, nearby_obstacles):
        x, y = self._get_states_vectorized(u_flat, drone_xy_vel)
        ex = evader_traj[1:self.N+1, 0]
        ey = evader_traj[1:self.N+1, 1]

        dx, dy = x - ex, y - ey
        dists = np.hypot(dx, dy)
        dists[dists < 1e-6] = 1e-6

        effort_cost = self.w_effort * np.dot(u_flat, u_flat)
        dist_cost = self.w_distance * np.sum((dists - self.target_distance)**2)

        grad = 2 * self.w_effort * u_flat
        coeff = 2 * self.w_distance * (1.0 - self.target_distance / dists)
        dJ_dx = coeff * dx
        dJ_dy = coeff * dy

        occ_cost = 0.0
        if self.w_occlusion > 0:
            alpha = self.barrier_steepness

            for obs in nearby_obstacles:
                if hasattr(obs, 'rx'):
                    rx, ry = obs.rx + self.safety_margin, obs.ry + self.safety_margin
                    theta = obs.theta
                else:
                    rx, ry = obs.radius + self.safety_margin, obs.radius + self.safety_margin
                    theta = 0.0

                ct, st = np.cos(theta), np.sin(theta)

                dx_D, dy_D = x - obs.cx, y - obs.cy
                dx_E, dy_E = ex - obs.cx, ey - obs.cy

                uD = (dx_D * ct + dy_D * st) / rx
                vD = (-dx_D * st + dy_D * ct) / ry
                uE = (dx_E * ct + dy_E * st) / rx
                vE = (-dx_E * st + dy_E * ct) / ry

                delU, delV = uE - uD, vE - vD
                N2 = delU**2 + delV**2 + 1e-6
                N = np.sqrt(N2)

                # Signed area of the parallelogram formed by (drone, origin, evader) in
                # obstacle-normalized space; d = |S|/N is the perpendicular distance from
                # the origin to the drone-evader line.
                S = uD * vE - vD * uE
                S_abs = np.abs(S)
                d = S_abs / N

                # Only penalize when the obstacle center lies between drone and evader
                dot_D = (ex - x)*(obs.cx - x) + (ey - y)*(obs.cy - y)
                dot_E = (x - ex)*(obs.cx - ex) + (y - ey)*(obs.cy - ey)
                mask = (dot_D > 0) & (dot_E > 0)

                # Exponential barrier: penalty explodes as d -> 1 (LoS grazes obstacle boundary)
                P = self.w_occlusion * np.exp(-alpha * (d - 1.0))
                occ_cost += np.sum(P[mask])

                # ---- Analytic Chain Rule Jacobians ----
                duD_dx, dvD_dx = ct / rx, -st / ry
                duD_dy, dvD_dy = st / rx, ct / ry

                dS_dx = vE * duD_dx - uE * dvD_dx
                dS_dy = vE * duD_dy - uE * dvD_dy

                dN2_dx = -2 * delU * duD_dx - 2 * delV * dvD_dx
                dN2_dy = -2 * delU * duD_dy - 2 * delV * dvD_dy

                S_sign = np.sign(S)

                dd_dx = (S_sign * dS_dx * N2 - S_abs * 0.5 * dN2_dx) / (N2 * N)
                dd_dy = (S_sign * dS_dy * N2 - S_abs * 0.5 * dN2_dy) / (N2 * N)

                dP_dx = -alpha * P * dd_dx
                dP_dy = -alpha * P * dd_dy

                dJ_dx += np.where(mask, dP_dx, 0.0)
                dJ_dy += np.where(mask, dP_dy, 0.0)

        total_cost = effort_cost + dist_cost + occ_cost

        # Apply State Jacobian (M^T) to convert positional gradients to control gradients
        grad[0::2] += self.MT @ dJ_dx
        grad[1::2] += self.MT @ dJ_dy

        return total_cost, grad

    def _constraints_and_jac(self, u_flat, drone_xy_vel, evader_traj, nearby_obstacles):
        x, y = self._get_states_vectorized(u_flat, drone_xy_vel)
        ex = evader_traj[1:self.N+1, 0]
        ey = evader_traj[1:self.N+1, 1]

        dx, dy = x - ex, y - ey
        dists = np.hypot(dx, dy)
        dists[dists < 1e-6] = 1e-6

        n_obs = len(nearby_obstacles)
        total_cons = self.N + (self.N * n_obs) + 1

        C = np.empty(total_cons)
        Jac = np.zeros((total_cons, self._n_vars))

        idx = 0

        C[idx : idx+self.N] = dists - self.d_min
        dx_over_d = dx / dists
        dy_over_d = dy / dists
        Jac[idx : idx+self.N, 0::2] = dx_over_d[:, None] * self.M
        Jac[idx : idx+self.N, 1::2] = dy_over_d[:, None] * self.M
        idx += self.N

        for obs in nearby_obstacles:
            odx, ody = x - obs.cx, y - obs.cy

            if hasattr(obs, 'rx'):
                ct, st = np.cos(obs.theta), np.sin(obs.theta)
                rx_s = obs.rx + self.safety_margin
                ry_s = obs.ry + self.safety_margin

                lx = ( ct * odx + st * ody) / rx_s
                ly = (-st * odx + ct * ody) / ry_s
                ldists = np.hypot(lx, ly)
                ldists[ldists < 1e-6] = 1e-6

                C[idx : idx+self.N] = ldists - 1.0

                dlx_dx, dlx_dy = ct / rx_s, st / rx_s
                dly_dx, dly_dy = -st / ry_s, ct / ry_s

                dC_dx = (lx * dlx_dx + ly * dly_dx) / ldists
                dC_dy = (lx * dlx_dy + ly * dly_dy) / ldists

                Jac[idx : idx+self.N, 0::2] = dC_dx[:, None] * self.M
                Jac[idx : idx+self.N, 1::2] = dC_dy[:, None] * self.M

            else:
                odists = np.hypot(odx, ody)
                odists[odists < 1e-6] = 1e-6

                C[idx : idx+self.N] = odists - (obs.radius + self.safety_margin)
                Jac[idx : idx+self.N, 0::2] = (odx / odists)[:, None] * self.M
                Jac[idx : idx+self.N, 1::2] = (ody / odists)[:, None] * self.M

            idx += self.N

        C[idx] = self.d_max - dists[-1]
        Jac[idx, 0::2] = -dx_over_d[-1] * self.M[-1, :]
        Jac[idx, 1::2] = -dy_over_d[-1] * self.M[-1, :]

        return C, Jac

    def _solve_mpc(self, drone_xy_vel, evader_traj):
        x, y = drone_xy_vel[0], drone_xy_vel[1]
        nearby_obstacles = [obs for obs in self.env.obstacles if np.hypot(obs.cx - x, obs.cy - y) < self.cull_radius]

        def cost_fun(u): return self._cost_and_jac(u, drone_xy_vel, evader_traj, nearby_obstacles)[0]
        def cost_jac(u): return self._cost_and_jac(u, drone_xy_vel, evader_traj, nearby_obstacles)[1]
        def cons_fun(u): return self._constraints_and_jac(u, drone_xy_vel, evader_traj, nearby_obstacles)[0]
        def cons_jac(u): return self._constraints_and_jac(u, drone_xy_vel, evader_traj, nearby_obstacles)[1]

        constraint = {
            'type': 'ineq',
            'fun': cons_fun,
            'jac': cons_jac
        }

        result = minimize(
            fun=cost_fun,
            jac=cost_jac,
            x0=self._u_prev,
            method='SLSQP',
            bounds=self._bounds,
            constraints=constraint,
            options={'maxiter': 50, 'ftol': 1e-3},
        )
        return result.x, result.success

    def _shift_warm_start(self, u_opt):
        u_shifted = np.empty_like(u_opt)
        u_shifted[:-2] = u_opt[2:]
        u_shifted[-2:] = u_opt[-2:]
        self._u_prev = u_shifted

    def _fallback(self, drone_xy_vel, evader_xy, bad_u):
        self._fallback_count += 1
        if np.isfinite(bad_u).all(): return bad_u[0], bad_u[1]

        x, y = drone_xy_vel[0], drone_xy_vel[1]
        ex, ey = evader_xy
        dx, dy = ex - x, ey - y
        dist = np.hypot(dx, dy)
        if dist > 1e-6:
            scale = (dist - (self.d_min + self.d_max) / 2.0) / dist
            ax = np.clip(3.0 * scale * dx, -self.a_max, self.a_max)
            ay = np.clip(3.0 * scale * dy, -self.a_max, self.a_max)
            return ax, ay
        return 0.0, 0.0

    def _virtual_to_angles(self, ax, ay, psi):
        c, s = np.cos(psi), np.sin(psi)
        tan_theta = (c * ax + s * ay) / self._g
        tan_phi   = (-s * ax + c * ay) / self._g
        return np.clip(np.arctan(tan_theta), -0.5, 0.5), np.clip(np.arctan(tan_phi), -0.5, 0.5)

    def _yaw_control(self, drone_state, evader_xy):
        x, y, psi, _, _, psi_dot = drone_state
        dx, dy = evader_xy[0] - x, evader_xy[1] - y
        yaw_error = (np.arctan2(dy, dx) - psi + np.pi) % (2 * np.pi) - np.pi
        return np.clip(self.kp_yaw * yaw_error - self.kd_yaw * psi_dot, -self.tau_z_max, self.tau_z_max)
