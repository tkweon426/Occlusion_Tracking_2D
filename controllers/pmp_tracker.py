# Made by: Gabriel Oliveira - ggolivei
# AAE568: Part of PMP-Based Control Benchmark
# Note: Replay Animation of Control Path is from Online Source
# Date: 04/30/2026

from __future__ import annotations

import copy
import csv
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class OfflinePMPDebugInfo:
    planned: bool = False
    iterations: int = 0
    max_u_change: float = math.inf
    min_pred_range: float = math.inf
    max_pred_range: float = 0.0
    min_pred_los_clearance: float = math.inf
    min_pred_body_clearance: float = math.inf
    plan_steps: int = 0
    plan_time_s: float = 0.0
    target_source: str = "true_scripted"
    final_cost: float = math.inf


class PMPTrackerController:
    def __init__(
        self,
        env,
        dt: float = 0.01,
        plan_dt: float = 0.10,
        evader_controller=None,
        drone_start=(0.0, -3.5),
        evader_start=(0.0, 3.0),
        max_steps: int = 5000,
        target_source: str = "true_scripted",
        external_target_csv: Optional[str] = None,
        measurement_noise_std: float = 0.0,
        kalman_seed: int = 568,
        # Geometric tracking goal.
        desired_offset: float = 5.0,
        min_standoff: float = 5.0,
        max_standoff: float = 5.8,
        # Hamiltonian running and terminal weights.
        q_pos: float = 8.0,
        q_vel: float = 1.0,
        qf_pos: float = 20.0,
        qf_vel: float = 3.0,
        r_u: float = 0.08,
        # Smooth penalty weights for path constraints.
        w_range: float = 550.0,
        w_los: float = 420.0,
        w_body: float = 360.0,
        eps_range: float = 0.20,
        eps_los: float = 0.20,
        eps_body: float = 0.25,
        los_margin: float = 0.55,
        body_margin: float = 0.85,
        top_bias_weight: float = 2.0,
        # Full-horizon solve settings.
        max_iter: int = 80,
        step_size: float = 0.012,
        tol: float = 1.0e-3,
        a_max: float = 6.75,
        evader_speed_cap: float = 3.0,
        # Replay stabilizer
        replay_kp: float = 3.0,
        replay_kd: float = 3.2,
        safety_projection: bool = True,
        pure_offline_replay: bool = False,
        yaw_kp: float = 4.0,
        yaw_kd: float = 2.0,
        save_plan_csv: bool = True,
        plan_csv_name: str = "offline_pmp_plan.csv",
        **unused_legacy_kwargs,
    ):
        self.env = env
        self.dt = float(dt)
        self.plan_dt = float(plan_dt)
        self.evader_controller = evader_controller
        self.drone_start = tuple(drone_start)
        self.evader_start = tuple(evader_start)
        self.max_steps = int(max_steps)
        self.target_source = str(target_source)
        self.external_target_csv = external_target_csv
        self.measurement_noise_std = float(measurement_noise_std)
        self.kalman_seed = int(kalman_seed)

        self.desired_offset = float(desired_offset)
        self.min_standoff = float(min_standoff)
        self.max_standoff = float(max_standoff)
        self.a_max = float(a_max)
        self.evader_speed_cap = float(evader_speed_cap)

        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(float)
        self.Qf = np.diag([qf_pos, qf_pos, qf_vel, qf_vel]).astype(float)
        self.r_u = float(r_u)

        self.w_range = float(w_range)
        self.w_los = float(w_los)
        self.w_body = float(w_body)
        self.eps_range = float(eps_range)
        self.eps_los = float(eps_los)
        self.eps_body = float(eps_body)
        self.los_margin = float(los_margin)
        self.body_margin = float(body_margin)
        self.top_bias_weight = float(top_bias_weight)

        self.max_iter = int(max_iter)
        self.step_size = float(step_size)
        self.tol = float(tol)
        self.replay_kp = float(replay_kp)
        self.replay_kd = float(replay_kd)
        self.safety_projection = bool(safety_projection)
        self.pure_offline_replay = bool(pure_offline_replay)
        self.yaw_kp = float(yaw_kp)
        self.yaw_kd = float(yaw_kd)
        self.save_plan_csv = bool(save_plan_csv)
        self.plan_csv_name = str(plan_csv_name)

        h = self.plan_dt
        self.F = np.array(
            [[1.0, 0.0, h, 0.0],
             [0.0, 1.0, 0.0, h],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        self.G = np.array(
            [[0.5 * h * h, 0.0],
             [0.0, 0.5 * h * h],
             [h, 0.0],
             [0.0, h]],
            dtype=float,
        )
        # For Hamiltonian
        self.A = np.array(
            [[0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]],
            dtype=float,
        )
        self.B = np.array(
            [[0.0, 0.0],
             [0.0, 0.0],
             [1.0, 0.0],
             [0.0, 1.0]],
            dtype=float,
        )

        self.step_index = 0
        self.last_debug = OfflinePMPDebugInfo(target_source=self.target_source)
        self._prev_runtime_dir = None
        self._prev_runtime_ref = None

        print("[Offline PMP] Precomputing full scripted target path...")
        self.true_pe_traj, self.true_ve_traj = self._generate_true_target_path()
        self.pe_traj, self.ve_traj = self._select_planning_target(self.true_pe_traj, self.true_ve_traj)
        self.r_traj = self._build_offline_reference_trajectory(self.pe_traj, self.ve_traj)
        z0 = np.array([self.drone_start[0], self.drone_start[1], 0.0, 0.0], dtype=float)
        print(f"[Offline PMP] Solving one full-horizon adjoint TPBVP with {len(self.r_traj)} nodes...")
        self.z_plan, self.lam_plan, self.u_plan, self.last_debug = self._solve_full_horizon_tpbvp(z0)
        if self.save_plan_csv:
            self._save_plan_csv(self.plan_csv_name)
        print(
            "[Offline PMP] Plan ready: "
            f"steps={self.last_debug.plan_steps}, "
            f"T={self.last_debug.plan_time_s:.2f}s, "
            f"iters={self.last_debug.iterations}, "
            f"min_range={self.last_debug.min_pred_range:.3f}m, "
            f"max_range={self.last_debug.max_pred_range:.3f}m, "
            f"min_los={self.last_debug.min_pred_los_clearance:.3f}, "
            f"min_body={self.last_debug.min_pred_body_clearance:.3f}m"
        )

    # For Replay
    def __call__(self, drone_state, evader_state):
        plan_idx = int((self.step_index * self.dt) / self.plan_dt)
        idx = min(plan_idx, len(self.u_plan) - 1)

        x, y, psi, vx, vy, psi_dot = drone_state
        p = np.array([x, y], dtype=float)
        v = np.array([vx, vy], dtype=float)
        pe_now = np.asarray(evader_state, dtype=float)
        ve_now = self.true_ve_traj[min(idx, len(self.true_ve_traj) - 1)]

        z_ref = np.asarray(self.z_plan[idx], dtype=float).copy()
        u = np.asarray(self.u_plan[idx], dtype=float).copy()

        if self.pure_offline_replay:
            pe_plan = self.true_pe_traj[min(idx, len(self.true_pe_traj) - 1)]

            ux, uy = self._clip_vector(u, self.a_max)
            theta, phi = self._accel_to_tilt(ux, uy, psi)
            tau_z = self._yaw_control(p, pe_plan, psi, psi_dot)

            self.step_index += 1
            return np.array([theta, phi, tau_z], dtype=float)

        rel_now = p - pe_now
        rel_norm = np.linalg.norm(rel_now)
        prev_dir = self._prev_runtime_dir
        if prev_dir is None and rel_norm > 1e-9:
            prev_dir = rel_now / rel_norm
        safe_dir = self._choose_reference_direction(pe_now, ve_now, prev_dir, self._prev_runtime_ref)
        safe_ref = pe_now + (self.desired_offset + 0.80) * safe_dir
        self._prev_runtime_dir = safe_dir
        self._prev_runtime_ref = safe_ref

        # Blend toward the current visible 5 m ring
        body_here = self._min_body_clearance(p)
        los_here = self._los_clearance(p, pe_now)
        danger = 0.0
        if np.isfinite(body_here):
            danger = max(danger, np.clip((1.4 - body_here) / 1.4, 0.0, 1.0))
        if np.isfinite(los_here):
            danger = max(danger, np.clip((self.los_margin - los_here) / max(self.los_margin, 1e-6), 0.0, 1.0))
        alpha = 0.85 + 0.10 * danger
        pos_ref = (1.0 - alpha) * z_ref[0:2] + alpha * safe_ref
        vel_ref = (1.0 - alpha) * z_ref[2:4] + alpha * ve_now

        u += self.replay_kp * (pos_ref - p) + self.replay_kd * (vel_ref - v)

        if self.safety_projection:
            u = self._active_constraint_projection(u, p, v, pe_now, ve_now)
        ux, uy = self._clip_vector(u, self.a_max)

        theta, phi = self._accel_to_tilt(ux, uy, psi)
        tau_z = self._yaw_control(p, pe_now, psi, psi_dot)

        self.step_index += 1
        return np.array([theta, phi, tau_z], dtype=float)

    # Full Target Path Generation
    def _generate_true_target_path(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.evader_controller is None:
            raise ValueError("Full offline PMP requires a scripted evader_controller.")

        scripted = copy.deepcopy(self.evader_controller)
        if hasattr(scripted, "_idx"):
            scripted._idx = 0

        stride = max(1, int(round(self.plan_dt / self.dt)))
        effective_plan_dt = stride * self.dt
        if abs(effective_plan_dt - self.plan_dt) > 1e-9:
            print(
                f"[Offline PMP] Adjusting plan_dt from {self.plan_dt:.4f} "
                f"to {effective_plan_dt:.4f} so it is an integer multiple of dt."
            )
            self.plan_dt = effective_plan_dt

        pe = np.array(self.evader_start, dtype=float)
        states = []
        vels = []

        for sim_k in range(self.max_steps):
            vx, vy = scripted.get_velocity(pe)
            v = self._clip_vector(np.array([vx, vy], dtype=float), self.evader_speed_cap)

            if sim_k % stride == 0:
                states.append(pe.copy())
                vels.append(v.copy())

            pe = pe + v * self.dt

            if getattr(scripted, "done", False):
                if len(states) == 0 or np.linalg.norm(states[-1] - pe) > 1e-9:
                    states.append(pe.copy())
                    vels.append(np.zeros(2))
                # Include a short stationary tail for the terminal portion.
                for _tail in range(30):
                    states.append(pe.copy())
                    vels.append(np.zeros(2))
                break

        if len(states) == 0:
            states.append(pe.copy())
            vels.append(np.zeros(2))

        return np.asarray(states, dtype=float), np.asarray(vels, dtype=float)

    def _select_planning_target(self, true_pe, true_ve) -> Tuple[np.ndarray, np.ndarray]:
        if self.target_source == "external_csv":
            loaded = self._try_load_external_target(len(true_pe))
            if loaded is not None:
                print("[Offline PMP] Using external target CSV for planning.")
                return loaded
            print("[Offline PMP] external CSV not found/readable; falling back to true scripted target.")
        if self.target_source == "kalman_estimated":
            return self._make_noisy_kalman_target(true_pe, true_ve)
        return true_pe.copy(), true_ve.copy()

    def _try_load_external_target(self, n: int):
        if not self.external_target_csv:
            return None
        path = os.path.expanduser(self.external_target_csv)
        if not os.path.exists(path):
            return None
        try:
            data = np.genfromtxt(path, delimiter=",", names=True)
            names = data.dtype.names or ()
            if {"x", "y", "vx", "vy"}.issubset(set(names)):
                pe = np.column_stack([data["x"], data["y"]]).astype(float)
                ve = np.column_stack([data["vx"], data["vy"]]).astype(float)
            else:
                raw = np.genfromtxt(path, delimiter=",")
                if raw.ndim == 1:
                    raw = raw[None, :]
                if raw.shape[1] >= 5:
                    pe = raw[:, 1:3].astype(float)
                    ve = raw[:, 3:5].astype(float)
                elif raw.shape[1] >= 4:
                    pe = raw[:, 0:2].astype(float)
                    ve = raw[:, 2:4].astype(float)
                else:
                    return None
            return self._resample_target(pe, ve, n)
        except Exception as exc:
            print(f"[Offline PMP] Failed to read {path}: {exc}")
            return None

    def _resample_target(self, pe, ve, n):
        old = np.linspace(0.0, 1.0, len(pe))
        new = np.linspace(0.0, 1.0, n)
        pe_new = np.column_stack([np.interp(new, old, pe[:, 0]), np.interp(new, old, pe[:, 1])])
        ve_new = np.column_stack([np.interp(new, old, ve[:, 0]), np.interp(new, old, ve[:, 1])])
        return pe_new, ve_new

    def _make_noisy_kalman_target(self, true_pe, true_ve):
        rng = np.random.default_rng(self.kalman_seed)
        sigma = max(self.measurement_noise_std, 1e-9)
        y_meas = true_pe + rng.normal(0.0, sigma, size=true_pe.shape)

        h = self.plan_dt
        F = np.array([[1, 0, h, 0],
                      [0, 1, 0, h],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        q = 0.35
        Q = (q ** 2) * np.array([[h**4/4, 0, h**3/2, 0],
                                 [0, h**4/4, 0, h**3/2],
                                 [h**3/2, 0, h**2, 0],
                                 [0, h**3/2, 0, h**2]], dtype=float)
        R = (sigma ** 2) * np.eye(2)
        xhat = np.array([y_meas[0, 0], y_meas[0, 1], true_ve[0, 0], true_ve[0, 1]], dtype=float)
        P = np.diag([sigma**2, sigma**2, 2.0, 2.0])

        est = np.zeros((len(true_pe), 4), dtype=float)
        for k, y in enumerate(y_meas):
            if k > 0:
                xhat = F @ xhat
                P = F @ P @ F.T + Q
            innov = y - H @ xhat
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            xhat = xhat + K @ innov
            P = (np.eye(4) - K @ H) @ P
            xhat[2:4] = self._clip_vector(xhat[2:4], self.evader_speed_cap)
            est[k] = xhat
        return est[:, 0:2], est[:, 2:4]

    # Offline Reference Construction
    def _build_offline_reference_trajectory(self, pe_traj, ve_traj):
        n = len(pe_traj)
        r = np.zeros((n, 4), dtype=float)
        first_rel = np.asarray(self.drone_start, dtype=float) - pe_traj[0]
        first_norm = np.linalg.norm(first_rel)
        prev_dir = first_rel / first_norm if first_norm > 1e-9 else None
        prev_ref = np.asarray(self.drone_start, dtype=float)
        for k in range(n):
            pe = pe_traj[k]
            ve = ve_traj[k]
            if k == 0 and prev_dir is not None:
                direction = prev_dir
            else:
                direction = self._choose_reference_direction(pe, ve, prev_dir, prev_ref)
            p_ref = pe + self.desired_offset * direction
            r[k, 0:2] = p_ref
            # Finite difference of p_ref gives a better velocity reference
            if k > 0:
                r[k - 1, 2:4] = self._clip_vector((r[k, 0:2] - r[k - 1, 0:2]) / self.plan_dt, 2.2 * self.evader_speed_cap)
            prev_dir = direction
            prev_ref = p_ref
        if n > 1:
            r[-1, 2:4] = r[-2, 2:4]
        return r

    def _choose_reference_direction(self, pe, ve, prev_dir=None, prev_ref=None):
        candidates = []

        for ang in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            candidates.append(np.array([math.cos(ang), math.sin(ang)], dtype=float))

        speed = float(np.linalg.norm(ve))
        if speed > 1e-6:
            behind = -ve / speed
            candidates.extend([behind, self._rotate(behind, math.pi / 2), self._rotate(behind, -math.pi / 2)])
        if prev_dir is not None:
            candidates.extend([prev_dir, self._rotate(prev_dir, 0.20), self._rotate(prev_dir, -0.20)])

        for obs in self.env.obstacles:
            c = np.array([obs.cx, obs.cy], dtype=float)
            outward = pe - c
            n = np.linalg.norm(outward)
            if n > 1e-9:
                outward = outward / n
                candidates.extend([outward, self._rotate(outward, 0.45), self._rotate(outward, -0.45), np.array([0.0, 1.0])])

        best_score = -math.inf
        best_dir = np.array([0.0, 1.0])
        for dvec in candidates:
            dvec = np.asarray(dvec, dtype=float)
            norm = np.linalg.norm(dvec)
            if norm < 1e-9:
                continue
            dvec = dvec / norm
            pref = pe + self.desired_offset * dvec
            los = self._los_clearance(pref, pe)
            if not np.isfinite(los):
                los = 10.0
            body = self._min_body_clearance(pref)
            if not np.isfinite(body):
                body = 10.0
            switch_cost = 0.0 if prev_dir is None else np.linalg.norm(dvec - prev_dir)
            move_cost = 0.0 if prev_ref is None else np.linalg.norm(pref - prev_ref)

            # Near an ellipse, prefer the same vertical side as the target
            top_bonus = 0.0
            for obs in self.env.obstacles:
                if hasattr(obs, "rx") and hasattr(obs, "ry"):
                    dist_to_obs = np.linalg.norm(pe - np.array([obs.cx, obs.cy]))
                    if dist_to_obs < max(obs.rx, obs.ry) + 4.5:
                        side_sign = 1.0 if pe[1] >= obs.cy else -1.0
                        top_bonus += self.top_bias_weight * math.tanh(side_sign * (pref[1] - obs.cy) / 1.5)

            # High score means visible, outside inflated obstacle, smooth, and on
            # the preferred side near the ellipse.
            score = 15.0 * math.tanh((los - self.los_margin) / 0.60)
            score += 12.0 * math.tanh((body - 0.15) / 0.60)
            score += top_bonus
            score -= 0.35 * move_cost + 0.70 * switch_cost
            if los < 0.0:
                score -= 25.0
            if body < 0.0:
                score -= 30.0
            if score > best_score:
                best_score = score
                best_dir = dvec
        return best_dir

    # Full-Horizon PMP
    def _solve_full_horizon_tpbvp(self, z0):
        n = len(self.r_traj)
        u = self._initial_control_guess(z0)
        z = self._forward_state(z0, u)
        lam = np.zeros((n, 4), dtype=float)
        prev_cost = self._trajectory_cost(z, u)
        max_change = math.inf
        iterations = 0

        for it in range(1, self.max_iter + 1):
            iterations = it
            z = self._forward_state(z0, u)
            lam, grad_u = self._adjoint_and_gradient(z, u)
            grad_norm = np.linalg.norm(grad_u, axis=1, keepdims=True)
            grad_scaled = grad_u / np.maximum(1.0, grad_norm / self.a_max)

            alpha = self.step_size
            accepted = False
            for _ in range(8):
                u_trial = self._project_control_sequence(u - alpha * grad_scaled)
                z_trial = self._forward_state(z0, u_trial)
                trial_cost = self._trajectory_cost(z_trial, u_trial)
                if np.isfinite(trial_cost) and trial_cost <= prev_cost + 1e-9:
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                u_trial = self._project_control_sequence(0.70 * u + 0.30 * self._initial_control_guess(z0))
                z_trial = self._forward_state(z0, u_trial)
                trial_cost = self._trajectory_cost(z_trial, u_trial)

            max_change = float(np.max(np.linalg.norm(u_trial - u, axis=1)))
            u, z, prev_cost = u_trial, z_trial, trial_cost
            if max_change < self.tol:
                break

        lam, _ = self._adjoint_and_gradient(z, u)
        debug = self._compute_debug(z, u, iterations, max_change, prev_cost)
        return z, lam, u, debug

    def _initial_control_guess(self, z0):
        n = len(self.r_traj)
        u = np.zeros((n, 2), dtype=float)
        z = np.array(z0, dtype=float)
        kp, kd = 1.8, 2.4
        for k in range(n):
            if k < n - 1:
                aref = (self.r_traj[k + 1, 2:4] - self.r_traj[k, 2:4]) / self.plan_dt
            else:
                aref = np.zeros(2)
            acc = aref + kp * (self.r_traj[k, 0:2] - z[0:2]) + kd * (self.r_traj[k, 2:4] - z[2:4])
            u[k] = self._clip_vector(acc, self.a_max)
            z = self.F @ z + self.G @ u[k]
        return u

    def _forward_state(self, z0, u_traj):
        n = len(u_traj)
        z = np.zeros((n, 4), dtype=float)
        z[0] = z0
        for k in range(n - 1):
            z[k + 1] = self.F @ z[k] + self.G @ u_traj[k]
        return z

    def _adjoint_and_gradient(self, z_traj, u_traj):
        n = len(z_traj)
        lam = np.zeros((n, 4), dtype=float)
        grad_u = np.zeros_like(u_traj)
        lam[-1] = self._terminal_grad(z_traj[-1])
        for k in range(n - 2, -1, -1):
            lz = self._running_grad_z(z_traj[k], k)
            lam[k] = lz + self.F.T @ lam[k + 1]
            grad_u[k] = 2.0 * self.r_u * u_traj[k] + self.G.T @ lam[k + 1]
        grad_u[-1] = 2.0 * self.r_u * u_traj[-1]
        return lam, grad_u

    def _running_grad_z(self, z, k):
        r = self.r_traj[k]
        pe = self.pe_traj[k]
        grad = 2.0 * (self.Q @ (z - r))
        grad[0:2] += self._penalty_gradient_position(z[0:2], pe)
        return self.plan_dt * grad

    def _terminal_grad(self, z):
        return 2.0 * (self.Qf @ (z - self.r_traj[-1]))

    def _trajectory_cost(self, z_traj, u_traj):
        total = 0.0
        for k in range(len(z_traj)):
            dz = z_traj[k] - self.r_traj[k]
            total += self.plan_dt * float(dz.T @ self.Q @ dz)
            if k < len(u_traj):
                total += self.plan_dt * self.r_u * float(u_traj[k].T @ u_traj[k])
            total += self.plan_dt * self._penalty_value_position(z_traj[k, 0:2], self.pe_traj[k])
        dzf = z_traj[-1] - self.r_traj[-1]
        total += float(dzf.T @ self.Qf @ dzf)
        return float(total)

    def _project_control_sequence(self, u_seq):
        out = np.asarray(u_seq, dtype=float).copy()
        for k in range(len(out)):
            out[k] = self._clip_vector(out[k], self.a_max)
        return out

    # Soft-Constraints
    def _penalty_value_position(self, p, pe):
        rel = p - pe
        d = float(np.linalg.norm(rel) + 1e-9)
        low = max(0.0, (self.min_standoff - d) / self.eps_range)
        high = max(0.0, (d - self.max_standoff) / self.eps_range)
        value = self.w_range * (low * low + high * high)

        los = self._los_clearance(p, pe)
        if np.isfinite(los):
            v = max(0.0, (self.los_margin - los) / self.eps_los)
            value += self.w_los * v * v

        body = self._min_body_clearance(p)
        if np.isfinite(body):
            v = max(0.0, -body / self.eps_body)
            value += self.w_body * v * v
        return float(value)

    def _penalty_gradient_position(self, p, pe):
        return self._range_penalty_grad(p, pe) + self._los_penalty_grad(p, pe) + self._body_penalty_grad(p)

    def _range_penalty_grad(self, p, pe):
        rel = p - pe
        d = float(np.linalg.norm(rel) + 1e-9)
        er = rel / d
        low = max(0.0, (self.min_standoff - d) / self.eps_range)
        high = max(0.0, (d - self.max_standoff) / self.eps_range)
        dphi_dd = self.w_range * (2.0 * low * (-1.0 / self.eps_range) + 2.0 * high * (1.0 / self.eps_range))
        return dphi_dd * er

    def _los_penalty_grad(self, p, pe):
        clearance = self._los_clearance(p, pe)
        if not np.isfinite(clearance):
            return np.zeros(2)
        violation = max(0.0, (self.los_margin - clearance) / self.eps_los)
        if violation <= 0.0:
            return np.zeros(2)
        coeff = self.w_los * 2.0 * violation * (-1.0 / self.eps_los)
        grad_clear = self._finite_diff_grad(lambda pp: self._los_clearance(pp, pe), p)
        return coeff * grad_clear

    def _body_penalty_grad(self, p):
        clearance = self._min_body_clearance(p)
        if not np.isfinite(clearance):
            return np.zeros(2)
        violation = max(0.0, -clearance / self.eps_body)
        if violation <= 0.0:
            return np.zeros(2)
        coeff = self.w_body * 2.0 * violation * (-1.0 / self.eps_body)
        grad_clear = self._finite_diff_grad(self._min_body_clearance, p)
        return coeff * grad_clear

    def _los_clearance(self, p, pe):
        return float(self.env.los_clearance(p[0], p[1], pe[0], pe[1]))

    def _min_body_clearance(self, p):
        if not self.env.obstacles:
            return math.inf
        return min(self._body_clearance(p, obs) for obs in self.env.obstacles)

    def _body_clearance(self, p, obs):
        if hasattr(obs, "radius"):
            c = np.array([obs.cx, obs.cy], dtype=float)
            return float(np.linalg.norm(p - c) - (obs.radius + self.body_margin))
        if hasattr(obs, "rx") and hasattr(obs, "ry"):
            dx = p[0] - obs.cx
            dy = p[1] - obs.cy
            ct, st = math.cos(obs.theta), math.sin(obs.theta)
            lx = ct * dx + st * dy
            ly = -st * dx + ct * dy
            rx = obs.rx + self.body_margin
            ry = obs.ry + self.body_margin
            rho = math.sqrt((lx / rx) ** 2 + (ly / ry) ** 2) + 1e-12
            return float((rho - 1.0) * min(rx, ry))
        return math.inf

    def _finite_diff_grad(self, fun, p, h=2.0e-3):
        grad = np.zeros(2)
        p = np.asarray(p, dtype=float)
        for i in range(2):
            dp = np.zeros(2)
            dp[i] = h
            fp = fun(p + dp)
            fm = fun(p - dp)
            if np.isfinite(fp) and np.isfinite(fm):
                grad[i] = (fp - fm) / (2.0 * h)
        return grad

    # Replay Constraints
    def _active_constraint_projection(self, u, p, v, pe, ve):
        u = np.asarray(u, dtype=float).copy()
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        pe = np.asarray(pe, dtype=float)
        ve = np.asarray(ve, dtype=float)

        rel = p - pe
        d = float(np.linalg.norm(rel) + 1e-9)
        er = rel / d
        radial_rate = float(np.dot(v - ve, er))
        guard = 0.80
        if d < self.min_standoff + guard:
            u += 55.0 * (self.min_standoff + guard - d) * er
            if radial_rate < 0.0:
                u += -12.0 * radial_rate * er
        elif d > self.max_standoff:
            u += -15.0 * (d - self.max_standoff) * er - 4.0 * radial_rate * er

        body_clear = self._min_body_clearance(p)
        body_guard = 1.35
        if np.isfinite(body_clear) and body_clear < body_guard:
            grad_body = self._finite_diff_grad(self._min_body_clearance, p)
            n = np.linalg.norm(grad_body)
            if n > 1e-9:
                normal = grad_body / n
                closing_speed = float(np.dot(v, normal))
                u += 42.0 * (body_guard - body_clear) * normal
                if closing_speed < 0.0:
                    u += -9.0 * closing_speed * normal

        los_clear = self._los_clearance(p, pe)
        if np.isfinite(los_clear) and los_clear < self.los_margin:
            grad_los = self._finite_diff_grad(lambda pp: self._los_clearance(pp, pe), p)
            n = np.linalg.norm(grad_los)
            if n > 1e-9:
                u += 34.0 * (self.los_margin - los_clear) * (grad_los / n)

        return self._clip_vector(u, self.a_max)

    # Other Functions
    def _accel_to_tilt(self, ux, uy, psi):
        g = 9.81
        c, s = math.cos(psi), math.sin(psi)
        tan_theta = (c * ux + s * uy) / g
        tan_phi = (-s * ux + c * uy) / g
        theta = float(np.clip(math.atan(tan_theta), -0.61, 0.61))
        phi = float(np.clip(math.atan(tan_phi), -0.61, 0.61))
        return theta, phi

    def _yaw_control(self, p, pe, psi, psi_dot):
        target_angle = math.atan2(pe[1] - p[1], pe[0] - p[0])
        yaw_error = (target_angle - psi + math.pi) % (2.0 * math.pi) - math.pi
        return float(np.clip(self.yaw_kp * yaw_error - self.yaw_kd * psi_dot, -1.0, 1.0))

    @staticmethod
    def _rotate(v, angle):
        ca, sa = math.cos(angle), math.sin(angle)
        return np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1]])

    @staticmethod
    def _clip_vector(v, limit):
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        if n > limit:
            return v * (limit / n)
        return v

    def _compute_debug(self, z, u, iterations, max_change, final_cost):
        min_range = math.inf
        max_range = 0.0
        min_los = math.inf
        min_body = math.inf
        for k in range(len(z)):
            p = z[k, 0:2]
            pe = self.true_pe_traj[min(k, len(self.true_pe_traj) - 1)]
            dist = float(np.linalg.norm(p - pe))
            min_range = min(min_range, dist)
            max_range = max(max_range, dist)
            min_los = min(min_los, self._los_clearance(p, pe))
            min_body = min(min_body, self._min_body_clearance(p))
        return OfflinePMPDebugInfo(
            planned=True,
            iterations=int(iterations),
            max_u_change=float(max_change),
            min_pred_range=float(min_range),
            max_pred_range=float(max_range),
            min_pred_los_clearance=float(min_los),
            min_pred_body_clearance=float(min_body),
            plan_steps=len(z),
            plan_time_s=(len(z) - 1) * self.plan_dt,
            target_source=self.target_source,
            final_cost=float(final_cost),
        )

    def _save_plan_csv(self, filename):
        path = os.path.join(os.getcwd(), filename)
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t", "x_plan", "y_plan", "vx_plan", "vy_plan", "ux", "uy",
                    "target_x_true", "target_y_true", "target_x_used", "target_y_used",
                    "ref_x", "ref_y", "range_true", "los_clearance_true", "body_clearance",
                ])
                for k in range(len(self.z_plan)):
                    p = self.z_plan[k, 0:2]
                    pe_true = self.true_pe_traj[min(k, len(self.true_pe_traj)-1)]
                    pe_used = self.pe_traj[min(k, len(self.pe_traj)-1)]
                    ref = self.r_traj[min(k, len(self.r_traj)-1)]
                    writer.writerow([
                        k * self.plan_dt,
                        self.z_plan[k, 0], self.z_plan[k, 1], self.z_plan[k, 2], self.z_plan[k, 3],
                        self.u_plan[k, 0], self.u_plan[k, 1],
                        pe_true[0], pe_true[1], pe_used[0], pe_used[1], ref[0], ref[1],
                        np.linalg.norm(p - pe_true),
                        self._los_clearance(p, pe_true), self._min_body_clearance(p),
                    ])
            print(f"[Offline PMP] Saved plan CSV to {path}")
        except Exception as exc:
            print(f"[Offline PMP] Could not save plan CSV: {exc}")