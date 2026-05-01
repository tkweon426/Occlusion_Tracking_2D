def rk4_step(dynamics_func, state, action, dt):
    k1 = dynamics_func(state, action)
    k2 = dynamics_func(state + 0.5 * dt * k1, action)
    k3 = dynamics_func(state + 0.5 * dt * k2, action)
    k4 = dynamics_func(state + dt * k3, action)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
