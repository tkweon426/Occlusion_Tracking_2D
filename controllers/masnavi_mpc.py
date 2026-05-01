# controllers/masnavi_mpc.py
"""
Implementation of "Real-Time Multi-Convex Model Predictive Control for
Occlusion Free Target Tracking" (Masnavi et al., 2021)
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.special import comb

from predictors.constvel_predictor import ConstVelPredictor
from predictors.kalman_predictor import KalmanPredictor
from predictors.attfield_predictor import AttFieldPredictor


class MasnaviMPC:
    def __init__(
        self,
        env,
        sim_dt=0.01,
        t_fin=2.0,
        num=20,
        num_samples=20,
        nvar=11,
        d_fov_min=5.0,
        d_fov_max=10.0,
        v_max=8.0,
        a_max=6.87,
        weight_smoothness=20.0,
        rho_fov=10.0,
        rho_occ=100.0,
        rho_ineq=10.0,
        rho_scale=1.5,
        admm_iters=15,
        res_tol=0.05,
        occ_margin=0.1,
        obs_a_scale=1.8,
        kp_yaw=4.0,
        kd_yaw=2.0,
        tau_z_max=1.0,
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

        self._g = 9.81
        #self._predictor = KalmanPredictor(sim_dt=sim_dt)
        #self._predictor = ConstVelPredictor(sim_dt=sim_dt)
        self._predictor = AttFieldPredictor(sim_dt=sim_dt, obstacles=self.env.obstacles, k_att=10.0, d_min=0.1)
        self._fallback_count = 0

        self._c_x_prev = np.zeros(nvar)
        self._c_y_prev = np.zeros(nvar)

        self._P, self._P_dot, self._P_ddot = self._build_bernstein_basis()

        self._H_smooth = weight_smoothness * (self._P_ddot.T @ self._P_ddot)
        self._PtP = self._P.T @ self._P
        self._PdotTPdot = self._P_dot.T @ self._P_dot
        self._PddotTPddot = self._P_ddot.T @ self._P_ddot

        self._A_eq = np.vstack([
            self._P[0, :],
            self._P_dot[0, :],
        ])

        self._u_vals = np.linspace(0.0, 1.0, self.num_samples)
        self._one_minus_u = 1.0 - self._u_vals
        self._sum_1_minus_u_sq = np.sum(self._one_minus_u**2)

        self._u_col = self._u_vals[:, None]
        self._one_minus_u_col = self._one_minus_u[:, None]

        nv = self.nvar
        A_eq_joint = block_diag(self._A_eq, self._A_eq)
        num_eq = A_eq_joint.shape[0]

        self._KKT = np.zeros((2 * nv + num_eq, 2 * nv + num_eq))
        self._KKT[2*nv:, :2*nv] = A_eq_joint
        self._KKT[:2*nv, 2*nv:] = A_eq_joint.T
        # Top-left 22x22 (H_joint) is dynamically updated in _solve_kkt


    def __call__(self, drone_state, evader_state):
        x, y, psi, vx, vy, psi_dot = drone_state
        evader_xy = np.asarray(evader_state[:2], dtype=float)

        horizon_dt = self.t_fin / (self.num - 1)
        evader_traj = self._predictor.predict(evader_xy, horizon_dt, self.num - 1)
        self.last_evader_traj = evader_traj

        c_x, c_y, success = self._solve(drone_state, evader_traj)

        if success and np.isfinite(c_x).all() and np.isfinite(c_y).all():
            ax_des = float(self._P_ddot[0, :] @ c_x)
            ay_des = float(self._P_ddot[0, :] @ c_y)
            ax_des = np.clip(ax_des, -self.a_max, self.a_max)
            ay_des = np.clip(ay_des, -self.a_max, self.a_max)
            self._c_x_prev = c_x.copy()
            self._c_y_prev = c_y.copy()
        else:
            drone_xy_vel = np.array([x, y, vx, vy])
            ax_des, ay_des = self._fallback(drone_xy_vel, evader_xy)

        theta, phi = self._virtual_to_angles(ax_des, ay_des, psi)
        tau_z = self._yaw_control(drone_state, evader_xy)

        return np.array([theta, phi, tau_z])

    def _build_bernstein_basis(self):
        n = self.nvar - 1
        num = self.num
        t = np.linspace(0.0, 1.0, num)

        def _basis(degree, t_arr):
            B = np.zeros((len(t_arr), degree + 1))
            for i in range(degree + 1):
                c = int(comb(degree, i, exact=True))
                B[:, i] = c * (1.0 - t_arr) ** (degree - i) * t_arr ** i
            return B

        P = _basis(n, t)

        Q9 = _basis(n - 1, t)
        P_dot_norm = np.zeros((num, n + 1))
        P_dot_norm[:, 0] = -n * Q9[:, 0]
        P_dot_norm[:, 1:n] = n * (Q9[:, :n - 1] - Q9[:, 1:n])
        P_dot_norm[:, n] = n * Q9[:, n - 1]
        P_dot = P_dot_norm / self.t_fin

        Q8 = _basis(n - 2, t)
        Q9_dot_norm = np.zeros((num, n))
        m = n - 1
        Q9_dot_norm[:, 0] = -m * Q8[:, 0]
        Q9_dot_norm[:, 1:m] = m * (Q8[:, :m - 1] - Q8[:, 1:m])
        Q9_dot_norm[:, m] = m * Q8[:, m - 1]

        P_ddot_norm = np.zeros((num, n + 1))
        P_ddot_norm[:, 0] = -n * Q9_dot_norm[:, 0]
        P_ddot_norm[:, 1:n] = n * (Q9_dot_norm[:, :n - 1] - Q9_dot_norm[:, 1:n])
        P_ddot_norm[:, n] = n * Q9_dot_norm[:, n - 1]
        P_ddot = P_ddot_norm / (self.t_fin ** 2)

        return P, P_dot, P_ddot

    def _solve(self, drone_state, evader_traj):
        x0, y0, psi0, vx0, vy0, _ = drone_state

        b_eq_x = np.array([x0, vx0])
        b_eq_y = np.array([y0, vy0])

        lambda_x = np.zeros(self.nvar)
        lambda_y = np.zeros(self.nvar)

        c_x = self._c_x_prev.copy()
        c_y = self._c_y_prev.copy()

        rho_fov  = self.rho_fov
        rho_occ  = self.rho_occ
        rho_ineq = self.rho_ineq

        n_obs = len(self.env.obstacles)

        evader_x_exp = evader_traj[:, 0][None, :]
        evader_y_exp = evader_traj[:, 1][None, :]

        for _ in range(self.admm_iters):
            x_drone = self._P @ c_x
            y_drone = self._P @ c_y

            # STEP 1: Update xi_2 (alpha) and xi_3 (d)
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

                lx = ( ct * dx_o + st * dy_o) / rx_ell
                ly = (-st * dx_o + ct * dy_o) / ry_ell

                dist_n = np.hypot(lx, ly)

                # Fall back to evader direction when the interpolated point is near the obstacle center
                evader_lx = ( ct * (evader_traj[:, 0] - obs.cx) + st * (evader_traj[:, 1] - obs.cy)) / rx_ell
                evader_ly = (-st * (evader_traj[:, 0] - obs.cx) + ct * (evader_traj[:, 1] - obs.cy)) / ry_ell
                a_n_evader = np.arctan2(evader_ly, evader_lx)

                a_n = np.where(dist_n > 1e-6, np.arctan2(ly, lx), a_n_evader[None, :])

                alpha_o[oi, :, :] = a_n
                d_o[oi, :, :] = np.maximum(1.0 + self.occ_margin, dist_n)

            # STEP 2a: Update acceleration slack variables (xi_4)
            s_ax = np.clip(self._P_ddot @ c_x, -self.a_max, self.a_max)
            s_ay = np.clip(self._P_ddot @ c_y, -self.a_max, self.a_max)

            # STEP 2b: Solve Exact KKT for xi_1 (c_x, c_y)
            H_joint, g_joint = self._build_qp(
                evader_traj, alpha_r, d_r, alpha_o, d_o,
                lambda_x, lambda_y,
                rho_fov, rho_occ, rho_ineq,
                s_ax, s_ay, evader_x_exp, evader_y_exp
            )
            c_x_new, c_y_new, ok = self._solve_kkt(H_joint, g_joint, b_eq_x, b_eq_y)

            if not ok:
                return self._c_x_prev.copy(), self._c_y_prev.copy(), False

            c_x, c_y = c_x_new, c_y_new

            # STEP 3: Bregman Multiplier Updates
            res_tar_x = self._P @ c_x - (evader_traj[:, 0] + d_r * np.cos(alpha_r))
            res_tar_y = self._P @ c_y - (evader_traj[:, 1] + d_r * np.sin(alpha_r))

            grad_lambda_x = rho_fov * self._P.T @ res_tar_x
            grad_lambda_y = rho_fov * self._P.T @ res_tar_y

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

                xs_l =  ct * (x_tilde - obs.cx) + st * (y_tilde - obs.cy)
                ys_l = -st * (x_tilde - obs.cx) + ct * (y_tilde - obs.cy)
                ell_val = np.sqrt((xs_l / rx_ell) ** 2 + (ys_l / ry_ell) ** 2)
                viol = float(np.max(np.maximum(0.0, (1.0 + self.occ_margin) - ell_val)))
                occ_viol_max = max(occ_viol_max, viol)

            lambda_x -= grad_lambda_x
            lambda_y -= grad_lambda_y

            fov_res = float(np.max(np.hypot(res_tar_x, res_tar_y)))
            if fov_res < self.res_tol and occ_viol_max < self.res_tol:
                break

            rho_fov  *= self.rho_scale
            rho_occ  *= self.rho_scale
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

        b_tar_x = evader_traj[:, 0] + d_r * np.cos(alpha_r)
        b_tar_y = evader_traj[:, 1] + d_r * np.sin(alpha_r)

        H_joint[:nv, :nv] += rho_fov * self._PtP
        H_joint[nv:, nv:] += rho_fov * self._PtP

        g_joint[:nv] -= rho_fov * self._P.T @ b_tar_x
        g_joint[nv:] -= rho_fov * self._P.T @ b_tar_y

        H_ineq_acc = rho_ineq * self._PddotTPddot

        H_joint[:nv, :nv] += H_ineq_acc
        H_joint[nv:, nv:] += H_ineq_acc

        g_joint[:nv] -= rho_ineq * self._P_ddot.T @ s_ax
        g_joint[nv:] -= rho_ineq * self._P_ddot.T @ s_ay

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

    def _solve_kkt(self, H_joint, g_joint, b_eq_x, b_eq_y):
        nv = self.nvar

        # Ridge regularization to guarantee matrix invertibility
        H_joint += np.eye(2 * nv) * 1e-4

        self._KKT[:2*nv, :2*nv] = H_joint

        b_eq_joint = np.concatenate([b_eq_x, b_eq_y])
        rhs = np.concatenate([-g_joint, b_eq_joint])

        try:
            solution = np.linalg.solve(self._KKT, rhs)
            z = solution[:2*nv]
            return z[:nv], z[nv:], True
        except np.linalg.LinAlgError:
            return self._c_x_prev.copy(), self._c_y_prev.copy(), False

    def _obs_axes(self, obs):
        """Returns (rx, ry, theta) for any obstacle type."""
        if hasattr(obs, 'rx'):
            return obs.rx * self.obs_a_scale, obs.ry * self.obs_a_scale, obs.theta
        r = obs.radius * self.obs_a_scale
        return r, r, 0.0

    def _fallback(self, drone_xy_vel, evader_xy):
        self._fallback_count += 1
        if self._fallback_count % 50 == 1:
            print(f"[MasnaviMPC] Solver KKT failed. Executing fallback mode #{self._fallback_count}")

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
