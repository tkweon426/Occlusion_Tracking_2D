# models/planar_quadrotor.py
import numpy as np
from utils.integrators import rk4_step

class TopDownQuadrotor:
    def __init__(self, x=0.0, y=0.0, mass=1.0, I_zz=0.02):
        self.m = mass
        self.I_zz = I_zz
        self.g = 9.81
        
        # State: [x, y, psi, x_dot, y_dot, psi_dot]
        self.state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=float)

    def _continuous_dynamics(self, state, action):
        x, y, psi, x_dot, y_dot, psi_dot = state
        theta, phi, tau_z = action
        
        x_ddot = self.g * (np.tan(theta) * np.cos(psi) - np.tan(phi) * np.sin(psi))
        y_ddot = self.g * (np.tan(theta) * np.sin(psi) + np.tan(phi) * np.cos(psi))
        psi_ddot = tau_z / self.I_zz
        
        return np.array([x_dot, y_dot, psi_dot, x_ddot, y_ddot, psi_ddot])

    def step(self, action, dt):
        """Advances the quadrotor state by dt using RK4."""
        self.state = rk4_step(self._continuous_dynamics, self.state, action, dt)