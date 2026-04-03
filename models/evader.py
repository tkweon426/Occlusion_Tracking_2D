import numpy as np


class Evader:
    def __init__(self, x=5.0, y=5.0):
        self.state = np.array([x, y], dtype=float)
        self.speed = 10.0

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

        v_mag = np.hypot(vx_cmd, vy_cmd)
        if v_mag > self.speed:
            scale = self.speed / v_mag
            vx_cmd *= scale
            vy_cmd *= scale

        next_x = x + vx_cmd * dt
        next_y = y + vy_cmd * dt

        if env.check_collision(next_x, next_y, agent_radius=1.0):
            rep_mag = np.hypot(rep_x, rep_y)
            if rep_mag > 1e-6:
                vx_cmd = self.speed * rep_x / rep_mag
                vy_cmd = self.speed * rep_y / rep_mag
            else:
                vx_cmd = 0.0
                vy_cmd = 0.0

            next_x = x + vx_cmd * dt
            next_y = y + vy_cmd * dt

            if env.check_collision(next_x, next_y, agent_radius=0.2):
                vx_cmd = 0.0
                vy_cmd = 0.0

        self.state[0] += vx_cmd * dt
        self.state[1] += vy_cmd * dt