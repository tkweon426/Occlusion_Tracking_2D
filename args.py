# args.py
# Central configuration for the simulation.
# Edit this file to change the environment, controller, or any simulation settings.

# --- Simulation ---
DT = 0.01                              # timestep (seconds)

# --- Environment ---
"""
List of environments
= make_empty_env            -> Empty world
= make_single_circle_env    -> World with single circular obstacle
= make_single_ellipse_env   -> World with single elliptical obstacle
"""
from environments.empty import make_empty_env
from environments.single_circle import make_single_circle_env
from environments.single_ellipse import make_single_ellipse_env
from environments.single_ellipse import make_two_obs_env

# Create a single shared environment instance.
# To switch environments, change only this line — the controller picks up the
# new obstacle list automatically because it holds a reference to _env.
#_env = make_single_circle_env()
#_env = make_single_ellipse_env()
_env = make_two_obs_env()
ENV_FACTORY = lambda: _env            # main.py calls this once; returns the same object

# --- Controller ---
from controllers.basic_mpc import BasicMPC
from controllers.basic_tracker import basic_chase_controller
from controllers.masnavi_mpc import MasnaviMPC
from controllers.occlusion_mpc import FastOcclusionMPC
from controllers.occlusion_mpc_inv import InvOcclusionMPC
from controllers.occlusion_mpcv2 import FastOcclusionMPC_v2
from controllers.occlusion_mpcv3 import FastOcclusionMPC_v3
from controllers.masnaviLQRMPC import MasnaviLQRMPC

#CONTROLLER = BasicMPC(env=_env, sim_dt=DT)
CONTROLLER = FastOcclusionMPC(env=_env, sim_dt=DT)
#CONTROLLER = InvOcclusionMPC(env=_env, sim_dt=DT)
#CONTROLLER = FastOcclusionMPC_v2(env=_env, sim_dt=DT)
#CONTROLLER = FastOcclusionMPC_v3(env=_env, sim_dt=DT)
#CONTROLLER = basic_chase_controller
#CONTROLLER = MasnaviMPC(env=_env, sim_dt=DT)
#CONTROLLER = MasnaviLQRMPC(env=_env, sim_dt=DT)

# --- Drone ---
DRONE_START    = (0.0, -3.5)          # initial (x, y) in metres
DRONE_MASS     = 1.0                   # kg
DRONE_I_ZZ     = 0.02                  # kg·m²
DRONE_RADIUS   = 0.5                   # collision radius (metres)

# --- Evader ---
EVADER_START   = (0.0, 3.0)           # initial (x, y) in metres
EVADER_RADIUS  = 0.3                   # collision radius (metres)

# Set to None for keyboard control (W/A/S/D), or assign a ScriptedTrajectory instance.
# Example:
from controllers.scripted_evader_1 import ScriptedTrajectory
from controllers.scripted_evader_2 import ScriptedTrajectory_2
#EVADER_CONTROLLER = ScriptedTrajectory(obstacle_cx=3.0, obstacle_cy=7.0)
EVADER_CONTROLLER = ScriptedTrajectory_2(obstacle_cx=3.0, obstacle_cy=7.0)
#EVADER_CONTROLLER = None               # None → keyboard, object → scripted

# --- Renderer ---
RENDERER_WIDTH  = 1000                  # window width (pixels)
RENDERER_HEIGHT = 800              # window height (pixels)
RENDERER_SCALE  = 20.0                 # pixels per metre
