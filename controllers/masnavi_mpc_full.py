
import numpy as np
from scipy.linalg import block_diag
from scipy.special import comb
from scipy.optimize import minimize, LinearConstraint

from predictors.constvel_predictor import ConstVelPredictor
from predictors.kalman_predictor import KalmanPredictor


class MasnaviMPCfull:
    def __init__(
        self,
        env,
        sim_dt=0.01,
        t_fin=2.0,
        num=20,
        num_samples=16,
        nvar=11,
        d_fov_min=5.0,
        d_fov_max=10.0,
        v_max=8.0,
        a_max=6.87,

        weight_smoothness=20.0,
        weight_pos=15.0,
        weight_vel=1.0,
        weight_terminal=15.0,
        desired_offset=4.0,

        rho_fov=10.0,
        rho_occ=80.0,
        rho_ineq=10.0,
        rho_scale=1.5,
        admm_iters=8,
        res_tol=0.05,
        occ_margin=0.1,
        obs_a_scale=1.3,

        kp_yaw=4.0,
        kd_yaw=2.0,
        tau_z_max=1.0,
        hard_obs=True,
        hard_obs_buffer=0.5,
        hard_obs_activation=1.0,
        hard_obs_skip_first=True,




        use_side_bias=True,
        weight_side_bias=60.0,
        side_bias_trigger_radius=5.5,
        side_bias_distance=4.5,
        side_bias_obs_cx=3.0,
        side_bias_obs_cy=7.0,
        side_bias_dir=(-1.0, -1.0),


        use_los_recovery=True,
        los_recovery_clearance=0.05,
        los_recovery_weight=45.0,
        los_recovery_distance=4.0,
        los_recovery_hysteresis=0.80,


        use_emergency_safety=True,
        emergency_safety_buffer=0.55,
        emergency_buffer=0.5,
        emergency_outward_gain=10.0,
    ):
        self.env = env
        self.t_fin = t_fin
        self.num = num
        self.num_samples = num_samples
        self.nvar = nvar
        self.d_fov_min = d_fov_min
        self.d_fov_max = d_fov_max
        self.v_max = v_max
        self.a_max = a_max

        self.weight_smoothness = weight_smoothness
        self.weight_pos = weight_pos
        self.weight_vel = weight_vel
        self.weight_terminal = weight_terminal
        self.desired_offset = desired_offset
        self.rho_fov = rho_fov
        self.rho_occ = rho_occ
        self.rho_ineq = rho_ineq
        self.rho_scale = rho_scale
        self.admm_iters = admm_iters
        self.res_tol = res_tol
        self.occ_margin = occ_margin
        self.obs_a_scale = obs_a_scale
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.tau_z_max = tau_z_max

        self.hard_obs = hard_obs
        self.hard_obs_buffer = hard_obs_buffer
        self.hard_obs_activation = hard_obs_activation
        self.hard_obs_skip_first = hard_obs_skip_first



        self.use_side_bias = use_side_bias
        self.weight_side_bias = weight_side_bias
        self.side_bias_trigger_radius = side_bias_trigger_radius
        self.side_bias_distance = side_bias_distance
        self.side_bias_obs_cx = side_bias_obs_cx
        self.side_bias_obs_cy = side_bias_obs_cy

        side_dir = np.asarray(side_bias_dir, dtype=float)
        side_norm = np.linalg.norm(side_dir)
        if side_norm < 1e-9:
            side_dir = np.array([-1.0, -1.0], dtype=float)
            side_norm = np.linalg.norm(side_dir)
        self.side_bias_dir = side_dir / side_norm




        self.use_los_recovery = use_los_recovery
        self.los_recovery_clearance = los_recovery_clearance
        self.los_recovery_weight = los_recovery_weight
        self.los_recovery_distance = los_recovery_distance
        self.los_recovery_hysteresis = los_recovery_hysteresis

        self.recovery_mode = False
        self.recovery_side_dir = self.side_bias_dir.copy()
        self.last_los_clearance = np.inf
        self.last_blocking_obstacle = None


        self.use_emergency_safety = use_emergency_safety
        self.emergency_safety_buffer = emergency_safety_buffer
        self.emergency_buffer = emergency_buffer
        self.emergency_outward_gain = emergency_outward_gain

        self._g = 9.81

        self._predictor = ConstVelPredictor(sim_dt=sim_dt)
        self._fallback_count = 0

        self._c_x_prev = np.zeros(nvar)
        self._c_y_prev = np.zeros(nvar)


        self._P, self._P_dot, self._P_ddot = self._build_bernstein_basis()

        self._PtP = self._P.T @ self._P
        self._PdotTPdot = self._P_dot.T @ self._P_dot
        self._PddotTPddot = self._P_ddot.T @ self._P_ddot

        self._H_smooth = self.weight_smoothness * self._PddotTPddot
        self._H_pos = self.weight_pos * self._PtP
        self._H_vel = self.weight_vel * self._PdotTPdot

        self._P_terminal = self._P[-1:, :]
        self._H_terminal = self.weight_terminal * (self._P_terminal.T @ self._P_terminal)

        self._A_eq = np.vstack([
            self._P[0, :],
            self._P_dot[0, :],
        ])


        self._u_vals = np.linspace(0.0, 1.0, self.num_samples)
        self._one_minus_u = 1.0 - self._u_vals
        self._sum_1_minus_u_sq = np.sum(self._one_minus_u ** 2)
        self._u_col = self._u_vals[:, None]
        self._one_minus_u_col = self._one_minus_u[:, None]


        nv = self.nvar
        A_eq_joint = block_diag(self._A_eq, self._A_eq)
        num_eq = A_eq_joint.shape[0]

        self._KKT = np.zeros((2 * nv + num_eq, 2 * nv + num_eq))
        self._KKT[2 * nv:, :2 * nv] = A_eq_joint
        self._KKT[:2 * nv, 2 * nv:] = A_eq_joint.T

    def __call__(self, drone_state, evader_state):
        x, y, psi, vx, vy, psi_dot = drone_state
        evader_xy = np.asarray(evader_state[:2], dtype=float)

        horizon_dt = self.t_fin / max(self.num - 1, 1)
        evader_traj = self._predictor.predict(evader_xy, horizon_dt, self.num - 1)
        self.last_evader_traj = evader_traj


        self._update_los_recovery_mode(drone_state, evader_xy)

        c_x, c_y, success = self._solve(drone_state, evader_traj)

        if success and np.isfinite(c_x).all() and np.isfinite(c_y).all():
            ax_des = float(self._P_ddot[0, :] @ c_x)
            ay_des = float(self._P_ddot[0, :] @ c_y)
            ax_des = np.clip(ax_des, -self.a_max, self.a_max)
            ay_des = np.clip(ay_des, -self.a_max, self.a_max)


            ax_des, ay_des = self._apply_emergency_obstacle_safety(
                ax_des,
                ay_des,
                drone_state,
            )

            self._c_x_prev = c_x.copy()
            self._c_y_prev = c_y.copy()
        else:
            drone_xy_vel = np.array([x, y, vx, vy], dtype=float)
            ax_des, ay_des = self._fallback(drone_xy_vel, evader_xy)


            ax_des, ay_des = self._apply_emergency_obstacle_safety(
                ax_des,
                ay_des,
                drone_state,
            )

        theta, phi = self._virtual_to_angles(ax_des, ay_des, psi)
        tau_z = self._yaw_control(drone_state, evader_xy)

        return np.array([theta, phi, tau_z], dtype=float)

    def _build_bernstein_basis(self):
        n = self.nvar - 1
        t = np.linspace(0.0, 1.0, self.num)

        def _basis(degree, t_arr):
            B = np.zeros((len(t_arr), degree + 1))
            for i in range(degree + 1):
                c = int(comb(degree, i, exact=True))
                B[:, i] = c * (1.0 - t_arr) ** (degree - i) * t_arr ** i
            return B

        P = _basis(n, t)

        Q9 = _basis(n - 1, t)
        P_dot_norm = np.zeros((self.num, n + 1))
        P_dot_norm[:, 0] = -n * Q9[:, 0]
        P_dot_norm[:, 1:n] = n * (Q9[:, :n - 1] - Q9[:, 1:n])
        P_dot_norm[:, n] = n * Q9[:, n - 1]
        P_dot = P_dot_norm / self.t_fin

        Q8 = _basis(n - 2, t)
        Q9_dot_norm = np.zeros((self.num, n))
        m = n - 1
        Q9_dot_norm[:, 0] = -m * Q8[:, 0]
        Q9_dot_norm[:, 1:m] = m * (Q8[:, :m - 1] - Q8[:, 1:m])
        Q9_dot_norm[:, m] = m * Q8[:, m - 1]

        P_ddot_norm = np.zeros((self.num, n + 1))
        P_ddot_norm[:, 0] = -n * Q9_dot_norm[:, 0]
        P_ddot_norm[:, 1:n] = n * (Q9_dot_norm[:, :n - 1] - Q9_dot_norm[:, 1:n])
        P_ddot_norm[:, n] = n * Q9_dot_norm[:, n - 1]
        P_ddot = P_ddot_norm / (self.t_fin ** 2)

        return P, P_dot, P_ddot

    def _solve(self, drone_state, evader_traj):
        x0, y0, psi0, vx0, vy0, _ = drone_state

        b_eq_x = np.array([x0, vx0], dtype=float)
        b_eq_y = np.array([y0, vy0], dtype=float)

        lambda_x = np.zeros(self.nvar)
        lambda_y = np.zeros(self.nvar)


        if (not np.any(self._c_x_prev)) and (not np.any(self._c_y_prev)):
            self._c_x_prev[:] = x0
            self._c_y_prev[:] = y0

        c_x = self._c_x_prev.copy()
        c_y = self._c_y_prev.copy()

        rho_fov = self.rho_fov
        rho_occ = self.rho_occ
        rho_ineq = self.rho_ineq

        n_obs = len(self.env.obstacles)

        evader_x_exp = evader_traj[:, 0][None, :]
        evader_y_exp = evader_traj[:, 1][None, :]

        for _ in range(self.admm_iters):
            x_drone = self._P @ c_x
            y_drone = self._P @ c_y


            dx = x_drone - evader_traj[:, 0]
            dy = y_drone - evader_traj[:, 1]
            alpha_r = np.arctan2(dy, dx)
            d_r_unc = np.cos(alpha_r) * dx + np.sin(alpha_r) * dy
            d_r = np.clip(d_r_unc, self.d_fov_min, self.d_fov_max)

            alpha_o = np.zeros((n_obs, self.num_samples, self.num))
            d_o = np.zeros((n_obs, self.num_samples, self.num))

            x_drone_exp = x_drone[None, :]
            y_drone_exp = y_drone[None, :]

            x_tilde = self._one_minus_u_col * x_drone_exp + self._u_col * evader_x_exp
            y_tilde = self._one_minus_u_col * y_drone_exp + self._u_col * evader_y_exp

            for oi, obs in enumerate(self.env.obstacles):
                rx_ell, ry_ell, th_ell = self._obs_axes(obs)
                ct, st = np.cos(th_ell), np.sin(th_ell)

                dx_o = x_tilde - obs.cx
                dy_o = y_tilde - obs.cy

                lx = (ct * dx_o + st * dy_o) / rx_ell
                ly = (-st * dx_o + ct * dy_o) / ry_ell
                dist_n = np.hypot(lx, ly)

                evader_lx = (ct * (evader_traj[:, 0] - obs.cx) + st * (evader_traj[:, 1] - obs.cy)) / rx_ell
                evader_ly = (-st * (evader_traj[:, 0] - obs.cx) + ct * (evader_traj[:, 1] - obs.cy)) / ry_ell
                a_n_evader = np.arctan2(evader_ly, evader_lx)

                a_n = np.where(dist_n > 1e-6, np.arctan2(ly, lx), a_n_evader[None, :])

                alpha_o[oi, :, :] = a_n
                d_o[oi, :, :] = np.maximum(1.0 + self.occ_margin, dist_n)


            s_ax = np.clip(self._P_ddot @ c_x, -self.a_max, self.a_max)
            s_ay = np.clip(self._P_ddot @ c_y, -self.a_max, self.a_max)


            H_joint, g_joint = self._build_qp(
                evader_traj, alpha_r, d_r, alpha_o, d_o,
                lambda_x, lambda_y,
                rho_fov, rho_occ, rho_ineq,
                s_ax, s_ay, evader_x_exp, evader_y_exp
            )





            if self._agent_close_to_obstacle(x0, y0):
                A_obs, lb_obs = self._build_linearized_obstacle_constraints(x_drone, y_drone)
            else:
                A_obs, lb_obs = None, None


            if A_obs is None or lb_obs is None or len(lb_obs) == 0:
                c_x_new, c_y_new, ok = self._solve_kkt(H_joint, g_joint, b_eq_x, b_eq_y)
            else:
                c_x_new, c_y_new, ok = self._solve_qp_with_hard_obs(
                    H_joint, g_joint, b_eq_x, b_eq_y,
                    c_x, c_y,
                    A_obs=A_obs, lb_obs=lb_obs
                )

            if not ok:
                return self._c_x_prev.copy(), self._c_y_prev.copy(), False

            c_x, c_y = c_x_new, c_y_new


            res_tar_x = self._P @ c_x - (evader_traj[:, 0] + d_r * np.cos(alpha_r))
            res_tar_y = self._P @ c_y - (evader_traj[:, 1] + d_r * np.sin(alpha_r))

            grad_lambda_x = rho_fov * (self._P.T @ res_tar_x)
            grad_lambda_y = rho_fov * (self._P.T @ res_tar_y)

            occ_viol_max = 0.0

            x_drone_exp = (self._P @ c_x)[None, :]
            y_drone_exp = (self._P @ c_y)[None, :]
            x_tilde = self._one_minus_u_col * x_drone_exp + self._u_col * evader_x_exp
            y_tilde = self._one_minus_u_col * y_drone_exp + self._u_col * evader_y_exp

            x_drone_scaled = self._one_minus_u_col * x_drone_exp
            y_drone_scaled = self._one_minus_u_col * y_drone_exp

            for oi, obs in enumerate(self.env.obstacles):
                rx_ell, ry_ell, th_ell = self._obs_axes(obs)
                ct, st = np.cos(th_ell), np.sin(th_ell)

                cos_a = np.cos(alpha_o[oi])
                sin_a = np.sin(alpha_o[oi])
                d = d_o[oi]

                ux = ct * rx_ell * cos_a - st * ry_ell * sin_a
                uy = st * rx_ell * cos_a + ct * ry_ell * sin_a

                b_occ_x = obs.cx - self._u_col * evader_x_exp + ux * d
                b_occ_y = obs.cy - self._u_col * evader_y_exp + uy * d

                res_occ_x = x_drone_scaled - b_occ_x
                res_occ_y = y_drone_scaled - b_occ_y

                weighted_res_occ_x = self._one_minus_u_col * res_occ_x
                weighted_res_occ_y = self._one_minus_u_col * res_occ_y

                grad_lambda_x += rho_occ * (self._P.T @ np.sum(weighted_res_occ_x, axis=0))
                grad_lambda_y += rho_occ * (self._P.T @ np.sum(weighted_res_occ_y, axis=0))

                xs_l = ct * (x_tilde - obs.cx) + st * (y_tilde - obs.cy)
                ys_l = -st * (x_tilde - obs.cx) + ct * (y_tilde - obs.cy)
                ell_val = np.sqrt((xs_l / rx_ell) ** 2 + (ys_l / ry_ell) ** 2)
                viol = float(np.max(np.maximum(0.0, (1.0 + self.occ_margin) - ell_val)))
                occ_viol_max = max(occ_viol_max, viol)

            lambda_x -= grad_lambda_x
            lambda_y -= grad_lambda_y

            fov_res = float(np.max(np.hypot(res_tar_x, res_tar_y)))
            if fov_res < self.res_tol and occ_viol_max < self.res_tol:
                break

            rho_fov *= self.rho_scale
            rho_occ *= self.rho_scale
            rho_ineq *= self.rho_scale

        return c_x, c_y, True

    def _build_qp(
        self, evader_traj, alpha_r, d_r, alpha_o, d_o,
        lambda_x, lambda_y,
        rho_fov, rho_occ, rho_ineq,
        s_ax, s_ay, evader_x_exp, evader_y_exp
    ):
        nv = self.nvar
        H_joint = np.zeros((2 * nv, 2 * nv))
        g_joint = np.zeros(2 * nv)

        H_joint[:nv, :nv] = self._H_smooth
        H_joint[nv:, nv:] = self._H_smooth

        g_joint[:nv] = -lambda_x
        g_joint[nv:] = -lambda_y


        p_des_x = evader_traj[:, 0] - self.desired_offset * np.cos(alpha_r)
        p_des_y = evader_traj[:, 1] - self.desired_offset * np.sin(alpha_r)

        dt_h = self.t_fin / max(self.num - 1, 1)
        v_des_x = np.gradient(evader_traj[:, 0], dt_h)
        v_des_y = np.gradient(evader_traj[:, 1], dt_h)

        pN_des_x = p_des_x[-1]
        pN_des_y = p_des_y[-1]


        H_joint[:nv, :nv] += self._H_pos
        H_joint[nv:, nv:] += self._H_pos
        g_joint[:nv] -= self.weight_pos * (self._P.T @ p_des_x)
        g_joint[nv:] -= self.weight_pos * (self._P.T @ p_des_y)


        H_joint[:nv, :nv] += self._H_vel
        H_joint[nv:, nv:] += self._H_vel
        g_joint[:nv] -= self.weight_vel * (self._P_dot.T @ v_des_x)
        g_joint[nv:] -= self.weight_vel * (self._P_dot.T @ v_des_y)


        H_joint[:nv, :nv] += self._H_terminal
        H_joint[nv:, nv:] += self._H_terminal
        g_joint[:nv] -= self.weight_terminal * (self._P_terminal.T[:, 0] * pN_des_x)
        g_joint[nv:] -= self.weight_terminal * (self._P_terminal.T[:, 0] * pN_des_y)




        self._add_side_bias_cost(H_joint, g_joint, evader_traj)



        self._add_los_recovery_cost(H_joint, g_joint, evader_traj)


        b_tar_x = evader_traj[:, 0] + d_r * np.cos(alpha_r)
        b_tar_y = evader_traj[:, 1] + d_r * np.sin(alpha_r)

        H_joint[:nv, :nv] += rho_fov * self._PtP
        H_joint[nv:, nv:] += rho_fov * self._PtP

        g_joint[:nv] -= rho_fov * (self._P.T @ b_tar_x)
        g_joint[nv:] -= rho_fov * (self._P.T @ b_tar_y)


        H_ineq_acc = rho_ineq * self._PddotTPddot
        H_joint[:nv, :nv] += H_ineq_acc
        H_joint[nv:, nv:] += H_ineq_acc
        g_joint[:nv] -= rho_ineq * (self._P_ddot.T @ s_ax)
        g_joint[nv:] -= rho_ineq * (self._P_ddot.T @ s_ay)


        if len(self.env.obstacles) > 0:
            H_occ_total = (rho_occ * len(self.env.obstacles) * self._sum_1_minus_u_sq) * self._PtP
            H_joint[:nv, :nv] += H_occ_total
            H_joint[nv:, nv:] += H_occ_total

        for oi, obs in enumerate(self.env.obstacles):
            rx_ell, ry_ell, th_ell = self._obs_axes(obs)
            ct, st = np.cos(th_ell), np.sin(th_ell)

            cos_a = np.cos(alpha_o[oi])
            sin_a = np.sin(alpha_o[oi])
            d = d_o[oi]

            ux = ct * rx_ell * cos_a - st * ry_ell * sin_a
            uy = st * rx_ell * cos_a + ct * ry_ell * sin_a

            b_occ_x = obs.cx - self._u_col * evader_x_exp + ux * d
            b_occ_y = obs.cy - self._u_col * evader_y_exp + uy * d

            weighted_b_occ_x = self._one_minus_u_col * b_occ_x
            weighted_b_occ_y = self._one_minus_u_col * b_occ_y

            g_joint[:nv] -= rho_occ * (self._P.T @ np.sum(weighted_b_occ_x, axis=0))
            g_joint[nv:] -= rho_occ * (self._P.T @ np.sum(weighted_b_occ_y, axis=0))

        return H_joint, g_joint

    def _update_los_recovery_mode(self, drone_state, evader_xy):
        """
        Topology-aware LOS-loss recovery trigger.

        It checks the current line segment from the tracker to the evader.
        If the LOS is blocked or nearly blocked, recovery_mode turns on.
        """
        if not self.use_los_recovery or len(self.env.obstacles) == 0:
            self.recovery_mode = False
            self.last_los_clearance = np.inf
            self.last_blocking_obstacle = None
            return

        agent_xy = np.asarray(drone_state[:2], dtype=float)
        target_xy = np.asarray(evader_xy[:2], dtype=float)

        min_clearance = np.inf
        blocking_obs = None
        u_vals = np.linspace(0.05, 0.95, 15)

        for obs in self.env.obstacles:
            rx, ry, th = self._obs_axes(obs)
            ct = np.cos(th)
            st = np.sin(th)

            for u in u_vals:
                q = (1.0 - u) * agent_xy + u * target_xy
                dx = q[0] - obs.cx
                dy = q[1] - obs.cy

                qx = ct * dx + st * dy
                qy = -st * dx + ct * dy

                ell = np.sqrt((qx / rx) ** 2 + (qy / ry) ** 2)
                clearance = (ell - 1.0) * min(rx, ry)

                if clearance < min_clearance:
                    min_clearance = clearance
                    blocking_obs = obs

        self.last_los_clearance = float(min_clearance)
        self.last_blocking_obstacle = blocking_obs

        if self.recovery_mode:
            if min_clearance > self.los_recovery_clearance + self.los_recovery_hysteresis:
                self.recovery_mode = False
        else:
            if min_clearance < self.los_recovery_clearance:
                self.recovery_mode = True

        if self.recovery_mode and blocking_obs is not None:
            self.recovery_side_dir = self._choose_recovery_side(
                agent_xy,
                target_xy,
                blocking_obs,
            )

    def _choose_recovery_side(self, agent_xy, target_xy, obs):
        """
        Choose one side of the blocking obstacle for recovery.
        """
        obs_xy = np.array([obs.cx, obs.cy], dtype=float)
        radial = target_xy - obs_xy
        radial_norm = np.linalg.norm(radial)

        if radial_norm < 1e-6:
            radial = target_xy - agent_xy
            radial_norm = np.linalg.norm(radial)

        if radial_norm < 1e-6:
            return self.side_bias_dir.copy()

        radial = radial / radial_norm
        left = np.array([-radial[1], radial[0]], dtype=float)
        right = np.array([radial[1], -radial[0]], dtype=float)

        cand_left = target_xy + self.los_recovery_distance * left
        cand_right = target_xy + self.los_recovery_distance * right

        if np.linalg.norm(agent_xy - cand_left) <= np.linalg.norm(agent_xy - cand_right):
            return left
        return right

    def _add_los_recovery_cost(self, H_joint, g_joint, evader_traj):
        """
        Strong topology-aware LOS-loss recovery cost.

        It activates only when recovery_mode is True and pulls the tracker toward
        a selected recovery viewpoint:
            p_recovery = p_evader + los_recovery_distance * recovery_side_dir
        """
        if not self.use_los_recovery:
            return
        if not self.recovery_mode:
            return
        if self.los_recovery_weight <= 0.0:
            return

        nv = self.nvar
        tx = evader_traj[:, 0]
        ty = evader_traj[:, 1]

        ref_x = tx + self.los_recovery_distance * self.recovery_side_dir[0]
        ref_y = ty + self.los_recovery_distance * self.recovery_side_dir[1]

        decay = np.linspace(1.00, 0.10, self.num)
        severity = max(
            0.0,
            (self.los_recovery_clearance - self.last_los_clearance)
            / max(self.los_recovery_clearance, 1e-6),
        )
        severity = np.clip(severity, 0.0, 2.0)

        w = self.los_recovery_weight * (1.0 + severity) * decay
        W = np.diag(w)
        H_rec = self._P.T @ W @ self._P

        H_joint[:nv, :nv] += H_rec
        H_joint[nv:, nv:] += H_rec

        g_joint[:nv] -= self._P.T @ (w * ref_x)
        g_joint[nv:] -= self._P.T @ (w * ref_y)

    def _add_side_bias_cost(self, H_joint, g_joint, evader_traj):
        """
        Soft topological side-bias for make_two_obs_env().

        Problem this solves:
            The Masnavi/LQR MPC can satisfy tracking, occlusion, collision,
            and smoothness terms but still choose the unintuitive side of the
            obstacle. This term gives the optimizer a weak hint to move the
            tracker to the lower-left side when the evader is close to the
            circle obstacle.

        Cost form:
            J_side = w(t) * ||p_agent(t) - p_ref(t)||^2

        where:
            p_ref(t) = p_evader(t) + side_bias_distance * side_bias_dir

        This is only a soft cost, not a hard constraint.
        """
        if (not self.use_side_bias) or self.weight_side_bias <= 0.0:
            return

        nv = self.nvar

        tx = evader_traj[:, 0]
        ty = evader_traj[:, 1]

        dist_to_obs = np.hypot(
            tx - self.side_bias_obs_cx,
            ty - self.side_bias_obs_cy,
        )

        active = dist_to_obs < self.side_bias_trigger_radius

        if not np.any(active):
            return

        ref_x = tx + self.side_bias_distance * self.side_bias_dir[0]
        ref_y = ty + self.side_bias_distance * self.side_bias_dir[1]


        w = np.zeros(self.num)
        ratio = 1.0 - dist_to_obs[active] / max(self.side_bias_trigger_radius, 1e-6)
        w[active] = self.weight_side_bias * ratio ** 2

        W = np.diag(w)
        H_side = self._P.T @ W @ self._P

        H_joint[:nv, :nv] += H_side
        H_joint[nv:, nv:] += H_side

        g_joint[:nv] -= self._P.T @ (w * ref_x)
        g_joint[nv:] -= self._P.T @ (w * ref_y)


    def _solve_kkt(self, H_joint, g_joint, b_eq_x, b_eq_y):
        nv = self.nvar

        H_joint = H_joint + np.eye(2 * nv) * 1e-4
        self._KKT[:2 * nv, :2 * nv] = H_joint

        b_eq_joint = np.concatenate([b_eq_x, b_eq_y])
        rhs = np.concatenate([-g_joint, b_eq_joint])

        try:
            solution = np.linalg.solve(self._KKT, rhs)
            z = solution[:2 * nv]
            return z[:nv], z[nv:], True
        except np.linalg.LinAlgError:
            return self._c_x_prev.copy(), self._c_y_prev.copy(), False

    def _obs_axes(self, obs):
        if hasattr(obs, "rx"):
            return obs.rx * self.obs_a_scale, obs.ry * self.obs_a_scale, obs.theta
        r = obs.radius * self.obs_a_scale
        return r, r, 0.0

    def _inflated_obs_axes(self, obs):
        rx, ry, th = self._obs_axes(obs)
        return float(rx) + self.hard_obs_buffer, float(ry) + self.hard_obs_buffer, th

    def _ellipse_value_and_gradient(self, x, y, obs, inflated=True):
        if inflated:
            rx, ry, th = self._inflated_obs_axes(obs)
        else:
            rx, ry, th = self._obs_axes(obs)

        ct, st = np.cos(th), np.sin(th)
        dx = float(x - obs.cx)
        dy = float(y - obs.cy)

        qx = ct * dx + st * dy
        qy = -st * dx + ct * dy

        g = (qx / rx) ** 2 + (qy / ry) ** 2 - 1.0

        dg_dqx = 2.0 * qx / (rx ** 2)
        dg_dqy = 2.0 * qy / (ry ** 2)

        grad_x = ct * dg_dqx - st * dg_dqy
        grad_y = st * dg_dqx + ct * dg_dqy

        return g, grad_x, grad_y

    def _agent_close_to_obstacle(self, x, y):
        """
        Gate for the hard obstacle constraint.

        Reference point:
            The CURRENT agent position (x, y), not the predicted horizon point
            and not the evader position.

        This function only decides whether hard constraints are sent into the QP.
        If it returns True, the normal predicted-horizon hard constraints are
        still built by _build_linearized_obstacle_constraints().
        """
        if (not self.hard_obs) or len(self.env.obstacles) == 0:
            return False

        for obs in self.env.obstacles:
            g, _, _ = self._ellipse_value_and_gradient(x, y, obs, inflated=True)



            ell = g + 1.0

            if ell < self.hard_obs_activation:
                return True

        return False

    def _build_linearized_obstacle_constraints(self, x_drone, y_drone):
        if (not self.hard_obs) or len(self.env.obstacles) == 0:
            return None, None

        rows = []
        lbs = []
        start_idx = 1 if self.hard_obs_skip_first else 0


        step_stride = 1

        for k in range(start_idx, self.num, step_stride):
            x0 = float(x_drone[k])
            y0 = float(y_drone[k])
            Pk = self._P[k, :]

            for obs in self.env.obstacles:
                g0, grad_x, grad_y = self._ellipse_value_and_gradient(
                    x0, y0, obs, inflated=True
                )

                ell0 = g0 + 1.0
                if ell0 > self.hard_obs_activation:
                    continue

                grad_norm = np.hypot(grad_x, grad_y)
                if grad_norm < 1e-8:
                    dx = x0 - obs.cx
                    dy = y0 - obs.cy
                    nrm = np.hypot(dx, dy)
                    if nrm < 1e-8:
                        grad_x, grad_y = 1.0, 0.0
                    else:
                        grad_x = dx / nrm
                        grad_y = dy / nrm

                row = np.zeros(2 * self.nvar)
                row[:self.nvar] = grad_x * Pk
                row[self.nvar:] = grad_y * Pk

                lb = grad_x * x0 + grad_y * y0 - g0

                rows.append(row)
                lbs.append(lb)

        if not rows:
            return None, None

        return np.vstack(rows), np.asarray(lbs, dtype=float)

    def _solve_qp_with_hard_obs(
        self,
        H_joint,
        g_joint,
        b_eq_x,
        b_eq_y,
        c_x_init,
        c_y_init,
        A_obs=None,
        lb_obs=None,
    ):
        nv = self.nvar


        if A_obs is None or lb_obs is None or len(lb_obs) == 0:
            return self._solve_kkt(H_joint, g_joint, b_eq_x, b_eq_y)

        H = 0.5 * (H_joint + H_joint.T) + 1e-5 * np.eye(2 * nv)
        A_eq_joint = block_diag(self._A_eq, self._A_eq)
        b_eq_joint = np.concatenate([b_eq_x, b_eq_y])

        z0 = np.concatenate([c_x_init, c_y_init])

        def obj(z):
            return 0.5 * z @ H @ z + g_joint @ z

        def grad(z):
            return H @ z + g_joint

        def hess(_z):
            return H

        constraints = [LinearConstraint(A_eq_joint, b_eq_joint, b_eq_joint)]

        ub_obs = np.full_like(lb_obs, np.inf, dtype=float)
        constraints.append(LinearConstraint(A_obs, lb_obs, ub_obs))

        try:
            res = minimize(
                obj,
                z0,
                method="trust-constr",
                jac=grad,
                hess=hess,
                constraints=constraints,
                options={
                    "maxiter": 30,
                    "gtol": 1e-3,
                    "xtol": 1e-3,
                    "barrier_tol": 1e-3,
                    "verbose": 0,
                },
            )

            if not res.success and res.status not in (1, 2):
                return self._c_x_prev.copy(), self._c_y_prev.copy(), False

            z = np.asarray(res.x, dtype=float)
            return z[:nv], z[nv:], True

        except Exception as e:
            print("Hard-QP exception:", repr(e))
            return self._c_x_prev.copy(), self._c_y_prev.copy(), False

    def _fallback(self, drone_xy_vel, evader_xy):
        self._fallback_count += 1
        if self._fallback_count % 50 == 1:
            print(f"[MasnaviMPC] MPC solve failed. Executing fallback mode #{self._fallback_count}")

        x, y, vx, vy = drone_xy_vel
        dx, dy = evader_xy[0] - x, evader_xy[1] - y
        dist = np.hypot(dx, dy)

        if dist > 1e-6:
            d_mid = (self.d_fov_min + self.d_fov_max) / 2.0
            scale = (dist - d_mid) / dist

            kp_fb = 3.0
            kd_fb = 2.0

            ax = kp_fb * scale * dx - kd_fb * vx
            ay = kp_fb * scale * dy - kd_fb * vy

            ax = np.clip(ax, -self.a_max, self.a_max)
            ay = np.clip(ay, -self.a_max, self.a_max)
            return float(ax), float(ay)

        return 0.0, 0.0

    def _apply_emergency_obstacle_safety(self, ax, ay, drone_state):
        """
        Step 1: Fast last-layer obstacle safety filter.

        This does not change the MPC optimization, side-bias cost, or any
        side-bias weights. It only modifies the final acceleration command
        if the current drone position is close to an obstacle and the command
        is pushing inward.
        """
        if not self.use_emergency_safety:
            return float(ax), float(ay)

        x, y, psi, vx, vy, psi_dot = drone_state

        a_cmd = np.array([ax, ay], dtype=float)

        safety_buffer = self.emergency_safety_buffer
        emergency_buffer = self.emergency_buffer
        outward_gain = self.emergency_outward_gain

        for obs in self.env.obstacles:
            rx, ry, th = self._obs_axes(obs)

            rx_safe = rx + safety_buffer
            ry_safe = ry + safety_buffer

            ct = np.cos(th)
            st = np.sin(th)

            dx = x - obs.cx
            dy = y - obs.cy

            qx = ct * dx + st * dy
            qy = -st * dx + ct * dy

            ell = np.sqrt((qx / rx_safe) ** 2 + (qy / ry_safe) ** 2)

            if ell >= 1.0:
                continue

            grad_local_x = qx / (rx_safe ** 2)
            grad_local_y = qy / (ry_safe ** 2)

            grad_x = ct * grad_local_x - st * grad_local_y
            grad_y = st * grad_local_x + ct * grad_local_y

            n = np.array([grad_x, grad_y], dtype=float)
            n_norm = np.linalg.norm(n)

            if n_norm < 1e-9:
                n = np.array([dx, dy], dtype=float)
                n_norm = np.linalg.norm(n)

                if n_norm < 1e-9:
                    n = np.array([1.0, 0.0], dtype=float)
                    n_norm = 1.0

            n = n / n_norm

            inward_component = np.dot(a_cmd, n)

            if inward_component < 0.0:
                a_cmd = a_cmd - inward_component * n

            violation = 1.0 - ell
            a_cmd = a_cmd + outward_gain * violation * n

            rx_emg = rx + emergency_buffer
            ry_emg = ry + emergency_buffer
            ell_emg = np.sqrt((qx / rx_emg) ** 2 + (qy / ry_emg) ** 2)

            if ell_emg < 1.0:
                a_cmd = a_cmd + 2.0 * outward_gain * (1.0 - ell_emg) * n

        ax_safe = np.clip(a_cmd[0], -self.a_max, self.a_max)
        ay_safe = np.clip(a_cmd[1], -self.a_max, self.a_max)

        return float(ax_safe), float(ay_safe)


    def _virtual_to_angles(self, ax, ay, psi):
        c, s = np.cos(psi), np.sin(psi)
        tan_theta = (c * ax + s * ay) / self._g
        tan_phi = (-s * ax + c * ay) / self._g
        theta = np.clip(np.arctan(tan_theta), -0.5, 0.5)
        phi = np.clip(np.arctan(tan_phi), -0.5, 0.5)
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
