# Multi-Convex MPC for Target Tracking with Occlusion Avoidance

## Overview

This document describes the Model Predictive Control (MPC) algorithm for real-time quadrotor target tracking with obstacle avoidance. The implementation uses Bernstein polynomial trajectory representations and solves a quadratic program at each control iteration.

**Paper Reference:** "Real-Time Multi-Convex Model Predictive Control for Occlusion-Free Target Tracking with Quadrotors"

---

## Core Algorithm Structure

### 1. Trajectory Representation

**Method:** 10th-order Bernstein polynomials

For each dimension (x, y, z), the trajectory is represented as:
```
position(t) = P * c
velocity(t) = P_dot * c
acceleration(t) = P_ddot * c
```

Where:
- `P`: Bernstein basis matrix (100 × 11) for position
- `P_dot`: Derivative of basis matrix for velocity
- `P_ddot`: Second derivative of basis matrix for acceleration
- `c`: Control points (coefficients) to be optimized (11 × 1)

**Bernstein Basis (order 10):**
```
B_i(t) = C(n,i) * (1-t)^(n-i) * t^i
```
where `t` ∈ [0, 1] is normalized time, and `n = 10`

---

## Optimization Problem Formulation

### Objective Function

Minimize (weighted sum):

```
J = ||P_ddot * c||² * weight_smoothness 
    + ρ_fov * ||A_fov * c - b_fov||²
    + ρ_occ * ||A_occ * c - b_occ||²
```

Where:
- **First term:** Penalizes trajectory curvature (smoothness)
- **Second term:** FOV constraint enforcement
- **Third term:** Occlusion avoidance constraint enforcement
- `ρ_fov`, `ρ_occ`: Penalty weights (increased iteratively if constraints violated)

### Equality Constraints

Enforce initial and final conditions:

```
P(0) * c = x_init
P_dot(0) * c = v_init
P_ddot(0) * c = a_init
P_dot(T) * c = v_final
P_ddot(T) * c = a_final
```

This gives **5 equality constraints** per dimension.

### Inequality Constraints

Bound velocity and acceleration:

```
v_min ≤ P_dot * c ≤ v_max
a_min ≤ P_ddot * c ≤ a_max
```

Reformulated as:
```
P_dot * c ≤ v_max
-P_dot * c ≤ -v_min
P_ddot * c ≤ a_max
-P_ddot * c ≤ -a_min
```

---

## Constraint Details

### 2.1 Field-of-View (FOV) Constraint

**Goal:** Keep target visible in drone's camera

**Variables:**
- `d_fov`: Distance from drone to target (scalar, optimized)
- `alpha_fov`: Angle between drone-to-target and camera optical axis (scalar, optimized)

**Constraint:**
```
drone_position = target_position - d_fov * [cos(alpha_fov), sin(alpha_fov)]
```

**Distance bounds:**
```
d_fov_min ≤ d_fov ≤ d_fov_max
```

**Implementation (iterative update):**
1. Solve QP to get trajectory
2. Compute desired angle:
   ```
   alpha_fov = atan2(target_y - drone_y, target_x - drone_x)
   ```
3. Compute optimal distance:
   ```
   c1 = rho_fov * (cos²(alpha_fov) + sin²(alpha_fov))
   c2 = rho_fov * (dx * cos(alpha_fov) + dy * sin(alpha_fov))
   d_fov = clip(c2 / c1, d_fov_min, d_fov_max)
   ```
   where `dx, dy` are relative positions

---

### 2.2 Occlusion Avoidance Constraint

**Goal:** Prevent obstacles from blocking line-of-sight to target

**Method:**
1. Sample **N_samples** points along line from drone to target
2. For each sample point, define forbidden region (ellipse) around obstacle
3. Ensure drone trajectory stays outside all forbidden regions

**For each obstacle:**
- Position: `(x_obs, y_obs, z_obs)` (moves linearly with constant velocity)
- Semi-axes: `a_obs, b_obs` (ellipse dimensions in XY plane)
- Sample points along line-of-sight: `u ∈ [0, 1]`
  ```
  sampled_point = drone_pos + u * (target_pos - drone_pos)
  ```

**Occlusion constraint (ellipse):**
```
(sampled_point_x - x_obs)² / a_obs² + (sampled_point_y - y_obs)² / b_obs² > 1 + margin
```

**Linearization for QP:**
```
A_occ * c > b_occ
```
where `A_occ` and `b_occ` are constructed from sampled constraint points

**Total occlusion variables:**
```
num_occ_constraints = num_obstacles * num_samples * num_trajectory_points
                    = 6 * 100 * 100 = 60,000 constraints
```

---

## Iterative Refinement Algorithm

### Algorithm Loop (ADMM-like approach)

```
Initialize:
  rho_fov = initial_penalty
  rho_occ = initial_penalty
  max_iterations = 10-20

For iteration i = 1 to max_iterations:
  
  1. Compute trajectory via QP solver
     - Input: current rho_fov, rho_occ
     - Output: Bernstein coefficients c_x, c_y
  
  2. Update FOV variables (d_fov, alpha_fov)
     - Evaluate trajectory at all time points
     - Compute optimal d_fov and alpha_fov
     - Check residual: |d_fov - d_fov_desired|
  
  3. Update occlusion variables (alpha_occ, d_occ)
     - For each sample point and obstacle
     - Compute optimal values
     - Check residual: min(constraint_violation)
  
  4. Check convergence
     - If all residuals < threshold: DONE
     - Otherwise: increase rho_fov and rho_occ
     
  5. Update lagrange multipliers (if using full ADMM)
     - lamda_dyn_x += rho * residual_x
     - lamda_dyn_y += rho * residual_y
```

---

## Implementation Steps

### Step 1: Initialize Problem Data

```python
class ProblemData:
    # Trajectory parameters
    num = 100                    # Number of trajectory points
    num_samples = 100            # Occlusion sampling points
    num_obs = 6                  # Number of obstacles
    nvar = 11                    # Bernstein polynomial order + 1
    
    # Time
    t_fin = 5.0                  # Final time (seconds)
    t = t_fin / num              # Time step
    
    # Dynamics bounds
    v_min, v_max = [-1.0, 1.0]  # Velocity bounds (m/s)
    a_min, a_max = [-2.0, 2.0]  # Acceleration bounds (m/s²)
    
    # FOV constraints
    d_fov_min = 1.0              # Minimum distance to target
    d_fov_max = 5.0              # Maximum distance to target
    
    # Penalty weights
    weight_smoothness = 80.0
    rho_fov = 100.0
    rho_occ = 100.0
    
    # Obstacle dimensions (ellipse semi-axes)
    a_obs = [0.85] * num_obs
    b_obs = [0.85] * num_obs
```

### Step 2: Construct Bernstein Basis Matrices

```python
def bernstein_basis_10(t_array, t_min=0, t_max=1):
    """
    Compute 10th-order Bernstein polynomial basis
    
    Args:
        t_array: time points (normalized to [0,1])
        
    Returns:
        P: basis matrix (len(t_array) × 11)
        P_dot: derivative basis
        P_ddot: second derivative basis
    """
    t = (t_array - t_min) / (t_max - t_min)
    n = 10
    
    P = np.zeros((len(t), n + 1))
    for i in range(n + 1):
        # B_i(t) = C(n,i) * (1-t)^(n-i) * t^i
        coeff = binomial(n, i)
        P[:, i] = coeff * (1 - t)**(n - i) * t**i
    
    # Derivatives computed analytically
    P_dot = compute_derivative(P, t_min, t_max)
    P_ddot = compute_derivative(P_dot, t_min, t_max)
    
    return P, P_dot, P_ddot
```

### Step 3: Build Equality Constraints

```python
def build_equality_constraints(P, P_dot, P_ddot, x_init, v_init, a_init, v_final, a_final):
    """
    A_eq * c = b_eq
    """
    A_eq = np.vstack([
        P[0, :],           # Initial position
        P_dot[0, :],       # Initial velocity
        P_ddot[0, :],      # Initial acceleration
        P_dot[-1, :],      # Final velocity
        P_ddot[-1, :]      # Final acceleration
    ])
    
    b_eq = np.array([x_init, v_init, a_init, v_final, a_final])
    
    return A_eq, b_eq
```

### Step 4: Build Inequality Constraints

```python
def build_inequality_constraints(P_dot, P_ddot, v_min, v_max, a_min, a_max):
    """
    A_ineq * c <= B_ineq
    """
    A_ineq = np.vstack([
        P_dot,      # velocity <= v_max
        -P_dot,     # velocity >= v_min
        P_ddot,     # acceleration <= a_max
        -P_ddot     # acceleration >= a_min
    ])
    
    B_ineq = np.hstack([
        np.ones(P_dot.shape[0]) * v_max,
        np.ones(P_dot.shape[0]) * -v_min,
        np.ones(P_ddot.shape[0]) * a_max,
        np.ones(P_ddot.shape[0]) * -a_min
    ])
    
    return A_ineq, B_ineq
```

### Step 5: FOV Constraint Setup

```python
def compute_fov_constraint(x_target, y_target):
    """
    FOV constraint: drone looks at target
    
    Returns:
        A_fov: (num_points × nvar) matrix
        b_fov: (num_points × 1) vector
    """
    # For each trajectory point, constraint is:
    # x_drone = x_target - d_fov * cos(alpha_fov)
    # y_drone = y_target - d_fov * sin(alpha_fov)
    
    # Initially set alpha_fov and d_fov, will be updated iteratively
    A_fov = P  # Position matrix (drone must be on line from target)
    b_fov = np.column_stack([x_target, y_target])
    
    return A_fov, b_fov
```

### Step 6: Occlusion Constraint Setup

```python
def compute_occlusion_constraint(x_target, y_target, x_obs, y_obs, P, 
                                 a_obs, b_obs, num_samples):
    """
    For each obstacle, sample line-of-sight and enforce ellipse constraints
    
    Constraint: (x_sampled - x_obs)²/a² + (y_sampled - y_obs)²/b² > 1
    """
    A_occ_list = []
    b_occ_list = []
    
    for obs_idx in range(len(x_obs)):
        for sample_idx in range(num_samples):
            u = sample_idx / num_samples  # Parameter along line-of-sight
            
            # Sampled point along drone-to-target line
            # x_sampled = x_drone + u * (x_target - x_drone)
            #           = P*c + u * (x_target - P*c)
            #           = (1-u)*P*c + u*x_target
            
            P_sample = (1 - u) * P
            
            # Constraint: ||point - obs||_ellipse > 1 + margin
            # Linearized form for QP
            # c_x * P_sample_x + c_y * P_sample_y > b_occ_val
            
            A_occ_list.append(P_sample)
            b_occ_list.append(np.array([x_obs[obs_idx], y_obs[obs_idx]]))
    
    A_occ = np.vstack(A_occ_list)
    b_occ = np.vstack(b_occ_list)
    
    return A_occ, b_occ
```

### Step 7: Main QP Solver Loop

```python
def solve_mpc(drone_state, target_state, obstacle_states, params):
    """
    Main MPC solver
    
    Args:
        drone_state: dict with x, y, z, vx, vy, vz, ax, ay, az
        target_state: dict with x, y, vx, vy
        obstacle_states: list of dicts with x, y, vx, vy
        params: ProblemData
        
    Returns:
        control_input: vx, vy, ax, ay, alphadot (local frame)
    """
    
    # Predict future trajectory points
    t_array = np.linspace(0, params.t_fin, params.num)
    x_target_pred = target_state['x'] + target_state['vx'] * t_array
    y_target_pred = target_state['y'] + target_state['vy'] * t_array
    
    # Predict obstacle positions
    x_obs_pred = [obs['x'] + obs['vx'] * t_array for obs in obstacle_states]
    y_obs_pred = [obs['y'] + obs['vy'] * t_array for obs in obstacle_states]
    
    # Build QP problem
    H = params.weight_smoothness * (P_ddot.T @ P_ddot)  # Hessian
    g = np.zeros(params.nvar)  # No linear cost term initially
    
    # Solve with iterative refinement
    for iteration in range(params.max_iterations):
        
        # Update FOV constraint
        A_fov, b_fov = compute_fov_constraint(x_target_pred, y_target_pred)
        
        # Update occlusion constraint
        A_occ, b_occ = compute_occlusion_constraint(
            x_target_pred, y_target_pred, 
            x_obs_pred, y_obs_pred,
            P, params.a_obs, params.b_obs, params.num_samples
        )
        
        # Combine all constraints
        H_total = H + params.rho_fov * (A_fov.T @ A_fov) + \
                      params.rho_occ * (A_occ.T @ A_occ)
        
        g_total = -params.rho_fov * (A_fov.T @ b_fov) - \
                   params.rho_occ * (A_occ.T @ b_occ)
        
        # Solve quadratic program
        c_xy = solve_qp(H_total, g_total, 
                       A_eq, b_eq,
                       A_ineq, B_ineq)
        
        # Check convergence
        residual_fov = check_fov_residual(c_xy, P, P_dot)
        residual_occ = check_occlusion_residual(c_xy, P, x_obs_pred, y_obs_pred)
        
        if residual_fov < threshold and residual_occ < threshold:
            break
        
        # Increase penalties if not converged
        params.rho_fov *= 1.5
        params.rho_occ *= 1.5
    
    return c_xy
```

### Step 8: Extract Control Input

```python
def extract_control_input(c_x, c_y, P, P_dot, P_ddot, drone_state, target_state):
    """
    Extract control input from Bernstein coefficients
    """
    # Evaluate at current time (start of trajectory)
    x_trajectory = P @ c_x
    y_trajectory = P @ c_y
    xdot_trajectory = P_dot @ c_x
    ydot_trajectory = P_dot @ c_y
    xddot_trajectory = P_ddot @ c_x
    yddot_trajectory = P_ddot @ c_y
    
    # Use first few points (e.g., mean of first 10 points)
    n_avg = 10
    vx_drone = np.mean(xdot_trajectory[:n_avg])
    vy_drone = np.mean(ydot_trajectory[:n_avg])
    ax_drone = np.mean(xddot_trajectory[:n_avg])
    ay_drone = np.mean(yddot_trajectory[:n_avg])
    
    # Compute orientation angle (pointing towards target)
    dx = target_state['x'] - drone_state['x']
    dy = target_state['y'] - drone_state['y']
    alpha_drone = np.arctan2(dy, dx)
    
    # Compute angular velocity
    alpha_rate = np.diff(np.arctan2(
        target_state['y'] - y_trajectory,
        target_state['x'] - x_trajectory
    ))
    alphadot_drone = np.mean(alpha_rate[:n_avg])
    
    # Transform velocity to local frame (drone-centric)
    vx_local = vx_drone * np.cos(alpha_drone) + vy_drone * np.sin(alpha_drone)
    vy_local = -vx_drone * np.sin(alpha_drone) + vy_drone * np.cos(alpha_drone)
    
    return {
        'vx': vx_drone,
        'vy': vy_drone,
        'ax': ax_drone,
        'ay': ay_drone,
        'vx_local': vx_local,
        'vy_local': vy_local,
        'alphadot': alphadot_drone
    }
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num` | 100 | Trajectory planning points |
| `num_samples` | 100 | Occlusion line-of-sight samples |
| `num_obs` | 6 | Number of obstacles |
| `nvar` | 11 | Bernstein polynomial order + 1 |
| `t_fin` | 5.0 | Prediction horizon (seconds) |
| `weight_smoothness` | 80.0 | Trajectory smoothness penalty |
| `v_min/max` | [-1.0, 1.0] | Velocity bounds (m/s) |
| `a_min/max` | [-2.0, 2.0] | Acceleration bounds (m/s²) |
| `d_fov_min` | 1.0 | Minimum viewing distance (m) |
| `d_fov_max` | 5.0 | Maximum viewing distance (m) |
| `rho_fov` | 100.0 | FOV penalty (increases iteratively) |
| `rho_occ` | 100.0 | Occlusion penalty (increases iteratively) |
| `a_obs, b_obs` | 0.85 | Ellipse semi-axes for obstacles (m) |

---

## Mathematical Notes

### Bernstein Polynomial Derivatives

**First derivative:**
```
dP_i/dt = n * [B_i(t) - B_{i+1}(t)]
```

**Second derivative:**
```
d²P_i/dt² = n(n-1) * [B_i(t) - 2*B_{i+1}(t) + B_{i+2}(t)]
```

### Quadratic Programming Formulation

Standard QP form:
```
minimize:   (1/2) * x^T * H * x + g^T * x
subject to:
    A_eq * x = b_eq
    A_ineq * x <= b_ineq
```

### Penalty Method Convergence

The algorithm converges when:
```
||residual_fov|| < ε_fov
||residual_occ|| < ε_occ
```

Typical tolerance: ε = 1e-3 to 1e-4

---

## Python Implementation Notes

### QP Solver Options

1. **cvxpy** (recommended for clarity):
```python
import cvxpy as cp

c = cp.Variable(nvar)
objective = cp.Minimize(cp.quad_form(c, H) + g @ c)
constraints = [
    A_eq @ c == b_eq,
    A_ineq @ c <= b_ineq
]
problem = cp.Problem(objective, constraints)
problem.solve()
```

2. **scipy.optimize.minimize**:
```python
from scipy.optimize import minimize

result = minimize(
    fun=lambda c: 0.5 * c @ H @ c + g @ c,
    x0=c_init,
    constraints=[
        {'type': 'eq', 'fun': lambda c: A_eq @ c - b_eq},
        {'type': 'ineq', 'fun': lambda c: b_ineq - A_ineq @ c}
    ]
)
```

3. **osqp** (sparse, fast):
```python
import osqp

solver = osqp.OSQP()
solver.setup(P=H, q=g, A=A_ineq, l=-np.inf, u=b_ineq, 
             Aeq=A_eq, leq=b_eq, ueq=b_eq)
result = solver.solve()
```

---

## Computational Complexity

- **Bernstein basis construction:** O(num × nvar)
- **QP matrix assembly:** O(num² × nvar²)
- **QP solve:** O(nvar³) for dense, O(nvar²) for sparse
- **Per iteration:** ~10-50ms for 100 points, 11 coefficients
- **Typical iterations:** 5-10 for convergence

---

## Validation Checklist

Before implementing, ensure:
- [ ] Bernstein derivatives computed correctly
- [ ] Equality constraints enforce initial conditions
- [ ] Inequality constraints properly form inequality system
- [ ] FOV iterative update converges
- [ ] Occlusion sampling covers all angles
- [ ] Target prediction uses correct velocity model
- [ ] Obstacle positions update with velocities
- [ ] QP solver returns feasible solutions
- [ ] Control input extracted at correct time point
- [ ] Local-frame velocity transformation correct

---

## References

1. Bernstein polynomial basis: De Casteljau's algorithm
2. Quadratic programming: Active set methods, interior point methods
3. Penalty methods: Boyd & Parikh, "Proximal Algorithms"
4. ADMM: Distributed Optimization and Statistical Learning via ADMM
