import numpy as np


class Evader:
    def __init__(self, x=5.0, y=5.0):
        self.state = np.array([x, y], dtype=float)
        self.speed = 10.0

    def _obstacle_repulsion(self, env):
        x, y = self.state

        rep_x = 0.0
        rep_y = 0.0

        safety_buffer = 0.3     
        repulse_gain = 3.0
        emergency_gain = 10.0

        for obs in env.obstacles:
            vx = x - obs.cx
            vy = y - obs.cy
            dist = np.hypot(vx, vy) + 1e-6

            clearance = dist - obs.radius

            if clearance < safety_buffer:
                strength = repulse_gain * (safety_buffer - clearance) / safety_buffer
                rep_x += strength * (vx / dist)
                rep_y += strength * (vy / dist)

            if clearance < 1.2:
                rep_x += emergency_gain * (vx / dist)
                rep_y += emergency_gain * (vy / dist)

        return rep_x, rep_y

    def step(self, vx, vy, dt, env):
        x, y = self.state

        rep_x, rep_y = self._obstacle_repulsion(env)

        vx_cmd = vx + rep_x
        vy_cmd = vy + rep_y

        # distance to nearest obstacle boundary
        min_clearance = 1e9
        for obs in env.obstacles:
            clearance = np.hypot(x - obs.cx, y - obs.cy) - obs.radius
            min_clearance = min(min_clearance, clearance)

        # slow down near obstacle instead of stopping too early
        speed_limit = self.speed
        if min_clearance < 1.0:
            speed_limit = max(0.3, self.speed * min_clearance / 1.0)

        v_mag = np.hypot(vx_cmd, vy_cmd)
        if v_mag > speed_limit:
            scale = speed_limit / v_mag
            vx_cmd *= scale
            vy_cmd *= scale

        next_x = x + vx_cmd * dt
        next_y = y + vy_cmd * dt

        if env.check_collision(next_x, next_y, agent_radius=0.2):
            # move only a fraction instead of fully stopping
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