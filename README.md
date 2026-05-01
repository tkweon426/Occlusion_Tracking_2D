# Occlusion-Aware Tracking in 2D

A 2D simulation framework for studying **occlusion-aware pursuit** — a planar quadrotor (pursuer) tracks a moving evader while navigating around obstacles that can block line-of-sight. The project compares multiple MPC-based controllers and evader predictors in a Pygame environment.

---

## Overview

The pursuer is modeled as a top-down planar quadrotor with dynamics controlled by roll/pitch/yaw commands. At each timestep the controller receives the full state of both the drone and the evader, and outputs control actions that try to:

- Keep the evader within a target distance band
- Maintain line-of-sight (LoS) despite occluding obstacles
- Avoid collisions with obstacles

The evader can be driven manually (WASD) or follow a scripted trajectory around obstacles.

---

## Project Structure

```
.
├── main.py                  # Simulation entry point
├── args.py                  # Configuration: env, controller, drone/evader params
├── controllers/             # Pursuer and evader controllers
│   ├── basic_tracker.py     # Simple PD chase controller
│   ├── basic_mpc.py         # MPC without occlusion awareness
│   ├── occlusion_mpc.py     # FastOcclusionMPC (LoS barrier in cost)
│   ├── masnavi_mpc.py       # Masnavi-style occlusion MPC
│   ├── masnavi_mpc_full.py  # Full Masnavi MPC variant
│   ├── manual_control.py    # WASD evader controller
│   └── scripted_evader_*.py # Scripted evader trajectories
├── environments/            # Obstacle world definitions
│   ├── empty.py             # No obstacles
│   ├── single_circle.py     # Single circular obstacle
│   └── single_ellipse.py    # Single/two elliptical obstacles
├── models/                  # Physical models
│   ├── planar_quadrotor.py  # Top-down quadrotor dynamics
│   └── evader.py            # Evader kinematic model
├── predictors/              # Evader motion predictors (used inside MPC)
│   ├── constvel_predictor.py
│   ├── attfield_predictor.py
│   └── velacc_predictor.py
├── evaluation/              # Metrics and comparison tools
│   ├── evaluate.py
│   ├── metric.py
│   ├── compare_metrics.py
│   └── compare_figure.py
├── utils/
│   └── integrators.py
├── environment_mac.yml      # Conda environment spec (macOS)
└── environment_windows.yml  # Conda environment spec (Windows)
```

---

## Setup

**Clone the repository:**
```bash
git clone https://github.com/tkweon426/Occlusion_Tracking_2D.git
cd Occlusion_Tracking_2D
```

**Create and activate the conda environment:**

On macOS:
```bash
conda env create -f environment_mac.yml
conda activate tracking_sim
```

On Windows:
```bash
conda env create -f environment_windows.yml
conda activate tracking_sim
```

---

## Running the Simulation

```bash
python main.py
```

Close the window or press **Q** to quit.

### Command-line Options

| Flag | Description |
|------|-------------|
| `--record [FILE]` | Save the run as a 60 fps MP4 (default: `results/recording.mp4`) |
| `--log [FILE]` | Log simulation data to a timestamped CSV (default: `results/log_<timestamp>.csv`) |
| `--end STEP` | Terminate the simulation after `STEP` timesteps |

**Examples:**
```bash
python main.py --record
python main.py --log results/my_run.csv --end 3000
python main.py --record results/test.mp4 --log --end 5000
```

---

## Configuration

All simulation parameters are set in [args.py](args.py). Key options:

**Environment** — uncomment the one you want:
```python
_env = make_empty_env()
_env = make_single_circle_env()
_env = make_single_ellipse_env()
_env = make_two_obs_env()
```

**Pursuer controller** — uncomment the one you want:
```python
CONTROLLER = FastOcclusionMPC(env=_env, sim_dt=DT)   # occlusion-aware MPC
CONTROLLER = BasicMPC(env=_env, sim_dt=DT)            # standard MPC
CONTROLLER = basic_chase_controller                   # simple PD
CONTROLLER = MasnaviMPC(env=_env, sim_dt=DT)          # Masnavi formulation
```

**Evader controller:**
```python
EVADER_CONTROLLER = ScriptedTrajectory_2(...)   # scripted path
EVADER_CONTROLLER = None                         # WASD keyboard control
```

---

## Controllers

| Controller | Description |
|------------|-------------|
| `basic_tracker` | PD controller, no obstacle avoidance |
| `BasicMPC` | MPC tracking without occlusion cost |
| `FastOcclusionMPC` | MPC with LoS barrier penalty and evader prediction |
| `MasnaviMPC` | Occlusion-aware MPC based on Masnavi et al. formulation |
| `MasnaviMPCfull` | Full variant of the Masnavi MPC |

---

## Logged Metrics

When `--log` is enabled, each row of the CSV contains:

- Drone state: position, heading, velocity, acceleration, control inputs
- Evader state: position, velocity
- `visibility_score` — LoS clearance between drone and evader
- `tracking_error_score` — reward for staying within the target distance band [5, 10] m
- `ctrl_compute_s` — controller wall-clock compute time per step
- Predicted evader trajectory (next step and horizon endpoint)

---

## Evaluation

Scripts in [evaluation/](evaluation/) can compare logged runs across controllers:

```bash
python evaluation/compare_metrics.py
python evaluation/compare_figure.py
```
