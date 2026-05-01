from environments.empty import make_empty_env
from environments.single_circle import make_single_circle_env
from environments.single_ellipse import make_single_ellipse_env
from environments.single_ellipse import make_two_obs_env

DT = 0.01

#_env = make_single_circle_env()
#_env = make_single_ellipse_env()
_env = make_two_obs_env()
ENV_FACTORY = lambda: _env  # main.py calls this once; returns the same object

from controllers.basic_mpc import BasicMPC
from controllers.basic_tracker import basic_chase_controller
from controllers.masnavi_mpc import MasnaviMPC
from controllers.occlusion_mpc import FastOcclusionMPC
from controllers.occlusion_mpc_inv import InvOcclusionMPC
from controllers.occlusion_mpcv2 import FastOcclusionMPC_v2
from controllers.occlusion_mpcv3 import FastOcclusionMPC_v3
from controllers.masnaviLQRMPC import MasnaviLQRMPC
from controllers.masnavi_mpc_full import MasnaviMPCfull

#CONTROLLER = BasicMPC(env=_env, sim_dt=DT)
CONTROLLER = FastOcclusionMPC(env=_env, sim_dt=DT)
#CONTROLLER = InvOcclusionMPC(env=_env, sim_dt=DT)
#CONTROLLER = FastOcclusionMPC_v2(env=_env, sim_dt=DT)
#CONTROLLER = FastOcclusionMPC_v3(env=_env, sim_dt=DT)
#CONTROLLER = basic_chase_controller
#CONTROLLER = MasnaviMPC(env=_env, sim_dt=DT)
#CONTROLLER = MasnaviLQRMPC(env=_env, sim_dt=DT)

DRONE_START    = (0.0, -3.5)
DRONE_MASS     = 1.0
DRONE_I_ZZ     = 0.02
DRONE_RADIUS   = 0.5

EVADER_START   = (0.0, 3.0)
EVADER_RADIUS  = 0.3

from controllers.scripted_evader_1 import ScriptedTrajectory
from controllers.scripted_evader_2 import ScriptedTrajectory_2
from controllers.scripted_evader_3 import ScriptedTrajectory_3
#EVADER_CONTROLLER = ScriptedTrajectory(obstacle_cx=3.0, obstacle_cy=7.0)
EVADER_CONTROLLER = ScriptedTrajectory_2(obstacle_cx=3.0, obstacle_cy=7.0)
#EVADER_CONTROLLER = None  # None → keyboard, object → scripted

RENDERER_WIDTH  = 1000
RENDERER_HEIGHT = 800
RENDERER_SCALE  = 20.0
