# utils/integrators.py
def rk4_step(dynamics_func, state, action, dt):
    """
    Standard RK4 integrator for advancing physics.
    """
    k1 = dynamics_func(state, action)
    k2 = dynamics_func(state + 0.5 * dt * k1, action)
    k3 = dynamics_func(state + 0.5 * dt * k2, action)
    k4 = dynamics_func(state + dt * k3, action)
    
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state