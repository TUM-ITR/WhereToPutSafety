
"""Standalone main script for 20Hz MPC coupled with 200Hz local PD+ controller."""
from pathlib import Path
import sys
import numpy as np

# Allow running the repository directly without installing it as a package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_module.params import SystemParams
from sim_module.sim_runner import (
    run_specific_architecture,
    run_all_architectures,
    run_monte_carlo,
    run_disturbance_sweep,
    run_delay_sweep,
)

def set_temp_params(params):
    """This function serves to manipulate parameters for testing"""
    ####################### Overwrite Parameters to test ############################
    #Timing: Local runs at 200 Hz (params.TA), MPC runs at 20 Hz (mpc_TA = 0.05)
    params.TA = 0.005
    params.mpc_ratio = int(10)  # 200Hz / 20 Hz = 10
    params.mpc_TA = params.TA * params.mpc_ratio
    params.tau_known = 3  # known, compensated, >=1 (for causality)
    params.tau_residual = 0 # no uncompensated delay
    params.tau = params.tau_known + params.tau_residual
    
    # Waypoints and Obstacles
    params.waypoints = [(0.18, -0.12), (0.23, 0.18), (0.6, 0.3)]
    params.scenario_obstacle_centers = [(0.0, 0.3), (0.4, 0.37)]
    params.obstacle_default_radius = [0.2, 0.145]
    # params.obstacle_default_radius = [0.2, 0.165] local get something wrong with this param
    scenario = params._rebuild_scenario()

    #Disturbance
    params.torque_disturbance_bound = 0.02 #0.1 -> violation local CBF
    params.torque_disturbance_std = params.torque_disturbance_bound/2
    
    #for more parameters: go to src/sim_module/params.py
    return params

if __name__ == "__main__":
    params = set_temp_params(SystemParams())

    # Default example: run combined architecture with bounded disturbances.
    run_specific_architecture(params, choice=3, with_disturbances=True)


    #for choice in [0, 1, 2, 3]:
    #    print(f"Testing architecture choice {choice}")
    #    run_specific_architecture(params, choice=choice, with_disturbances=True)