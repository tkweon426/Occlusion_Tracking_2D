# models/evader.py
import numpy as np

class Evader:
    def __init__(self, x=5.0, y=5.0):
        # State: [x, y]
        self.state = np.array([x, y], dtype=float)
        self.speed = 3.0 # m/s

    def step(self, vx, vy, dt):
        """Advances the evader based on commanded velocity."""
        self.state[0] += vx * dt
        self.state[1] += vy * dt