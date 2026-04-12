import numpy as np


class Evader:
    def __init__(self, x=5.0, y=5.0):
        self.state = np.array([x, y], dtype=float)
        self.speed = 20.0

    def _obstacle_repulsion(self, env):
        x, y = self.state

        rep_x = 0.0
        rep_y = 0.0

        safety_buffer = 3.0
        repulse_gain = 45.0
        emergency_gain = 80.0

        for obs in env.obstacles:
            vx = x - obs.cx
            vy = y - obs.cy

            # ellipse-normalized distance, also works for circles since a=b=radius
            nx = vx / max(obs.a, 1e-6)
            ny = vy / max(obs.b, 1e-6)
            norm_dist = np.hypot(nx, ny) + 1e-6

            # clearance in normalized obstacle coordinates
            clearance = norm_dist - 1.0

            # outward direction
            dir_x = nx / norm_dist
            dir_y = ny / norm_dist

            scaled_buffer = safety_buffer / max(obs.a, obs.b)

            if clearance < scaled_buffer:
                strength = repulse_gain * (scaled_buffer - clearance) / max(scaled_buffer, 1e-6)
                rep_x += strength * dir_x
                rep_y += strength * dir_y

            if clearance < 0.25:
                rep_x += emergency_gain * dir_x
                rep_y += emergency_gain * dir_y

        return rep_x, rep_y

    def _normalized_clearance(self, x, y, obs):
        dx = x - obs.cx
        dy = y - obs.cy
        nx = dx / max(obs.a, 1e-6)
        ny = dy / max(obs.b, 1e-6)
        return np.hypot(nx, ny) - 1.0

    def step(self, vx, vy, dt, env):
        x, y = self.state

        rep_x, rep_y = self._obstacle_repulsion(env)

        vx_cmd = vx + rep_x
        vy_cmd = vy + rep_y

        # distance to nearest obstacle boundary in normalized coordinates
        min_clearance = 1e9
        for obs in env.obstacles:
            clearance = self._normalized_clearance(x, y, obs)
            min_clearance = min(min_clearance, clearance)

        # slow down near obstacle instead of stopping too early
        speed_limit = self.speed
        if min_clearance < 1.0:
            speed_limit = max(0.3, self.speed * max(min_clearance, 0.0) / 1.0)

        v_mag = np.hypot(vx_cmd, vy_cmd)
        if v_mag > speed_limit and v_mag > 1e-9:
            scale = speed_limit / v_mag
            vx_cmd *= scale
            vy_cmd *= scale

        next_x = x + vx_cmd * dt
        next_y = y + vy_cmd * dt

        if env.check_collision(next_x, next_y, agent_radius=0.2):
            safe = False
            for alpha in [0.5, 0.25, 0.1, 0.05]:
                test_x = x + alpha * vx_cmd * dt
                test_y = y + alpha * vy_cmd * dt
                if not env.check_collision(test_x, test_y, agent_radius=0.2):
                    next_x = test_x
                    next_y = test_y
                    safe = True
                    break

            if not safe:
                next_x = x
                next_y = y

        self.state[0] = next_x
        self.state[1] = next_y