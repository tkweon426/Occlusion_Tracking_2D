# args.py
# Central configuration for the simulation.
# Edit this file to change the environment, controller, or any simulation settings.

# --- Environment ---
"""
List of environments
= make_empty_env            -> Empty world
= make_single_circle_env    -> World with single circular obstacle
"""
from environments.empty import make_empty_env
from environments.single_circle import make_single_circle_env
ENV_FACTORY = make_single_circle_env          # callable that returns a BaseEnvironment


# --- Controller ---
from controllers.basic_tracker import basic_chase_controller
CONTROLLER = basic_chase_controller    # callable: (drone_state, evader_state) -> action

# --- Simulation ---
DT = 0.01                              # timestep (seconds)

# --- Drone ---
DRONE_START    = (0.0, 0.0)           # initial (x, y) in metres
DRONE_MASS     = 1.0                   # kg
DRONE_I_ZZ     = 0.02                  # kg·m²
DRONE_RADIUS   = 0.5                   # collision radius (metres)

# --- Evader ---
EVADER_START   = (0.0, 3.0)           # initial (x, y) in metres
EVADER_RADIUS  = 0.3                   # collision radius (metres)

# Set to None for keyboard control (W/A/S/D), or assign a ScriptedTrajectory instance.
# Example:
from controllers.scripted_evader import ScriptedTrajectory
EVADER_CONTROLLER = ScriptedTrajectory(obstacle_cx=3.0, obstacle_cy=7.0)
# EVADER_CONTROLLER = None               # None → keyboard, object → scripted

# --- Renderer ---
RENDERER_WIDTH  = 1000                  # window width (pixels)
RENDERER_HEIGHT = 800              # window height (pixels)
RENDERER_SCALE  = 30.0                 # pixels per metre
