import numpy as np

class Evader:
    def __init__(self, x=5.0, y=5.0):
        self.state = np.array([x, y], dtype=float)
        self.speed = 2.68  # m/s

    def step(self, vx, vy, dt):
        self.state[0] += vx * dt
        self.state[1] += vy * dt
