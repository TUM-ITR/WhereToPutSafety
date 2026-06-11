
"""Standalone test script for 20Hz MPC coupled with 200Hz local PD+ controller."""

'External Libs'
import json
import copy
import re
from dataclasses import fields, is_dataclass
from datetime import datetime
import os
import time
from typing import Optional, Tuple, Any
import numpy as np
import collections
import datetime
import pandas as pd
import matplotlib.pyplot as plt

'Internal Code'
from sim_module.params import SystemParams
from sim_module.generate_disturbance import generate_single_disturbances
from plant_module.robot_dynamics import RobotDynamics
from controller_module.local_cbf import LocalCBF
from controller_module.remote_mpc_cbf import RemoteController
from controller_module.pd_plus import PDPlusController
from predictor_module.state_predictor import StatePredictor
from sim_module.data_recorder import DataRecorder
from helper.plot_code.plotting_3dof import plot_3dof_compare_results, generate_visualizations
from helper.analysis.post_processing import post_process_single_run, print_run_summary
from helper.analysis.post_processing_MonteCarlo import (
    analyse_monte_carlo_run,
    analyse_disturbance_sweep,
    analyse_delay_sweep,
)
from helper.plot_code.plots_for_paper import (
    plot_workspace,
    plot_workspace_comparison,
    plot_min_distance_to_obstacle,
    plot_min_clearance_over_delay,
    plot_min_clearance_over_disturbance
)

def single_run(params: SystemParams, arch_name: str, base_out_dir: str, noise: Optional[np.ndarray] = None, verbose: bool = True, 
               plotting: bool = True, animation: bool = False) -> None:
    """ To run a single architecture"""
    t0 = time.time()

    params.out_dir = os.path.join(base_out_dir, arch_name) #create base_out_dir/arch_name to save data in
    params.strategy_name = arch_name
    os.makedirs(params.out_dir, exist_ok=True)

    [use_local_cbf, use_remote_cbf,params.mpc_use_CBF,params.mpc_use_slack] = _architecture_variables(arch_name)
    if(noise is not None):
        params.enable_disturbance= True
    else:
        params.enable_disturbance = False

    params.use_local_cbf = use_local_cbf
    params.use_remote_cbf = use_remote_cbf


    # Initialize dynamics
    dynamics = RobotDynamics(params,noise)
    
    # initialize remote MPC
    remote_controller = RemoteController(params, dynamics)
    
    #local_cbf = LocalCBF(params,dynamics)
    if(params.use_local_cbf):
        local_cbf = LocalCBF(params,dynamics)
    local_controller = PDPlusController(params, dynamics) #PD Plus Controller

    # Setup 200Hz predictor using local dynamics (since it interpolates delayed local steps)
    predictor = StatePredictor(params, dynamics)

    # Initiate Data Recording
    data_recorder = DataRecorder(output_dir=params.out_dir)


    # Init simulation
    dt = params.TA
    max_steps = int(params.total_time / dt)
    t_array = np.arange(max_steps) * dt

    final_target = params.waypoints[-1]
    waypoint_idx = 1 # Start by targeting waypoint 1 directly
    current_target = np.asarray(params.waypoints[waypoint_idx], dtype=float)

    # Buffers to incorporate Delay -> using double ended queues (deque)
    z_history_measured = collections.deque(maxlen=max(params.tau+1, 2)) #measurement buffer
    ddq_history_applied = collections.deque(maxlen=params.tau_residual+1) #buffer for prediction

    # Buffer for endeffector position, ensuring convergence
    ee_distance_history = []    
    target_enter_tol = getattr(params, "target_enter_tol", 0.05)
    target_hold_tol = getattr(params, "target_hold_tol", 0.07)
    target_settle_tol = getattr(params, "target_settle_tol", params.target_tol)
    target_velocity_tol = getattr(params, "target_velocity_tol", 0.1)

    pause_duration = getattr(params, "target_hold_time", 3.0)
    pause_steps_required = int(pause_duration / dt)
    settle_history_len = getattr(params, "settle_history_len", 5)

    tracking_timer_active = False
    pause_steps_remaining = 0
    ee_distance_history = [np.inf] * settle_history_len
    finalTargetReached = False

    # Setup initial state
    z_true = np.array(list(params.initial_joint_angles) + [0.0, 0.0, 0.0])
    q_d = z_true[:3].copy()
    dq_d = z_true[3:].copy()
    ddq_d = np.zeros(3)
    z_pred = np.nan * np.ones(6)
    pred_err = np.nan

    # Calculate initial error properly
    th_true_init = z_true[:3]
    _, _, _, ee_true_init = local_controller.dynamics.fk_points(th_true_init)
    if verbose: print(f"Initial EE Pos: {ee_true_init}, Initial Target: {current_target}")
    
    # Calculate torque required to hold initial position (gravity compensation)
    u_init_hold = dynamics.gravity_vector(th_true_init)
    ddq_mpc = np.zeros(3) #accelerations
    
    # Pre-fill history loops with identical states and required holding torques
    
    initial_distance_to_target = np.linalg.norm(current_target-ee_true_init)    
    for _ in range(params.tau + 1):
        z_history_measured.append(z_true.copy())
    for _ in range(params.tau_residual + 1):
        ddq_history_applied.append(np.zeros(3))
    for _ in range(5):
        ee_distance_history.append(initial_distance_to_target)
    

    #initialize arrays for saving values
    joint_angles_true = np.zeros((max_steps, 3))
    ee_states_true = np.zeros((max_steps, 4))
    U_alpha = np.zeros((max_steps - 1, 3)) #inputs
    u = u_init_hold
    ddq_mpc = None
    ddq_cbf = None        
    local_cbf_active = None
    local_cbf_slack = None
    local_cbf_feasible = None
    local_cbf_solvetime = None
    local_cbf_min = None
    mpc_cbf_active = None
    mpc_cbf_slack = None
    mpc_feasible = None
    mpc_solvetime = None
    mpc_cbf_min = None

    # Target Tracking Time Constraint
    finalTargetReached = False
    pause_steps_remaining = 0
    
    if verbose: print(f"Starting Multi-Rate Simulation: MPC @ 20Hz (sending trajectory series), Local PD+ @ 200Hz, Delay tau={params.tau} steps")
    # 200 Hz Loop
    for k in range(max_steps - 1):
        th_true = z_true[:3]
        om_true = z_true[3:]
        _, _, _, ee_true = local_controller.dynamics.fk_points(th_true)
        v_ee_true = local_controller.dynamics.jac_ee(th_true) @ om_true

        joint_angles_true[k] = th_true
        ee_states_true[k, :2] = ee_true
        ee_states_true[k, 2:] = v_ee_true

        distance_to_target = np.linalg.norm(ee_true - current_target)
        ee_distance_history.append(distance_to_target)
        ee_distance_history.pop(0)

        joint_speed = np.linalg.norm(om_true)

        inside_enter_region = distance_to_target <= target_enter_tol
        inside_hold_region = distance_to_target <= target_hold_tol
        settled_at_target = (
            all(d <= target_settle_tol for d in ee_distance_history)
            and joint_speed <= target_velocity_tol
        )

        # ------------------------------------------------------------
        # Waypoint tracking / switching logic
        # ------------------------------------------------------------

        if not finalTargetReached:

            # Case 1: timer is already running
            if tracking_timer_active:

                # If robot leaves the looser hold region, reset the timer.
                if not inside_hold_region:
                    tracking_timer_active = False
                    pause_steps_remaining = 0
                    if verbose: print(f"  [t={k * dt:.2f}s] Left tracking region of waypoint {waypoint_idx}. Resetting tracking timer.")

                else:
                    pause_steps_remaining -= 1

                    # Success either by waiting long enough or settling early.
                    target_confirmed = pause_steps_remaining <= 0 or settled_at_target

                    if target_confirmed:
                        tracking_timer_active = False
                        pause_steps_remaining = 0

                        if waypoint_idx < len(params.waypoints) - 1:
                            waypoint_idx += 1
                            current_target = np.asarray(params.waypoints[waypoint_idx], dtype=float)

                            # Reset distance history for the next waypoint.
                            ee_distance_history = [np.inf] * settle_history_len

                            if verbose: print(f"  [t={k * dt:.2f}s] Waypoint {waypoint_idx - 1} tracked. Switching to waypoint {waypoint_idx}: {current_target}")
                        else:
                            finalTargetReached = True
                            if verbose: print(f"  [t={k * dt:.2f}s] Final target tracked.")

            # Case 2: timer is not running yet
            else:
                if inside_enter_region:
                    tracking_timer_active = True
                    pause_steps_remaining = pause_steps_required

                    if verbose: print(f"  [t={k * dt:.2f}s] Entered target region of waypoint {waypoint_idx}. Starting tracking timer.")


        if (finalTargetReached):
            if verbose: print(f"Goal reached at step {k}/{max_steps-1} (t={k * params.TA:.2f}s)")
            # Fill the rest of the arrays with the current steady state to avoid dropping to zero in plots
            for remaining_k in range(k, max_steps):
                joint_angles_true[remaining_k] = th_true
                ee_states_true[remaining_k, :2] = ee_true
                ee_states_true[remaining_k, 2:] = np.zeros(2)
                if remaining_k < max_steps - 1:
                    U_alpha[remaining_k] = u #np.zeros(3)
            break

        
        # ---------------------
        # 1. 20Hz Loop  
        # ---------------------
        if k % params.mpc_ratio == 0:

            # LOCAL: update buffer with most recent measurement
            z_history_measured.append(z_true.copy())

            # REMOTE
            #1. get delayed state z_k-tau
            z_last = z_history_measured.popleft() #take left value = oldest value. Because it is tau steps long -> delay considered
            
            # 2. Predict future value
            z_pred, used_local_cbf_in_prediction = predictor.predict_future_state(z_last,u) #logged
            
            q_d = z_pred[:3]
            dq_d = z_pred[3:]
                 
            # 3. Call the MPC from the predicted state
            resultMPC= remote_controller.solve(z_pred, waypoint_idx,a_prev=ddq_d) 

            mpc_cbf_active = resultMPC.CBFactive
            mpc_cbf_slack = resultMPC.slack_sum
            mpc_feasible = resultMPC.success
            mpc_solvetime = resultMPC.solve_time
            mpc_cbf_min = resultMPC.min_cbf_margin

            
            # 4. Convert to desired position, velocity and acceleration for tracking from the PD+ controller
            if resultMPC.success:
                # MPC input is desired joint acceleration.
                # Use ONLY the first input of the optimal sequence.
                ddq_mpc = np.asarray(resultMPC.u0, dtype=float).reshape(3)
            else:
                if verbose: print(f"[t={k * dt:.2f}s] MPC failed with status {resultMPC.status}. Holding/braking reference.")
                ddq_mpc = -2.0 * dq_d  # simple damping fallback

            
            # For delay prediction, store the desired acceleration as predicted by the MPC
            predictor.add_mpc_input(ddq_mpc)

            # To consider unknown delay: save predicted value, use delayed value
            ddq_history_applied.append(ddq_mpc.copy())
            ddq_mpc = ddq_history_applied.popleft()

            # END OF REMOTE CONTROL LOOP    

        # ---------------------
        # 2. Local 200Hz PD+ Tracking
        # ---------------------

        # Local CBF
            # Use true state here
        if(use_local_cbf):

            ddq_cbf,q_d,dq_d,resultCBF = local_cbf.computeLocalCBF(z_true, ddq_mpc,u)
            local_cbf_active=resultCBF.CBFactive
            local_cbf_slack=resultCBF.slack
            local_cbf_feasible = resultCBF.success
            local_cbf_solvetime=resultCBF.solve_time
            local_cbf_min =resultCBF.min_cbf_margin
            
            ddq_d = ddq_cbf
                
        else:
            ddq_d = ddq_mpc
            
            # Integrate the desired reference at the local 200 Hz step for smoothness
            q_d = th_true + dt * om_true + 0.5 * dt**2 * ddq_d
            dq_d = om_true + dt * ddq_d

        # PD+ tracking
        u = local_controller.compute_PDplus(th_true, om_true, q_d, dq_d, ddq_d)
        
        
        U_alpha[k] = u

        # Forward Dynamics for one step
        z_true = dynamics.joint_dynamics_step(z_true, u, params.TA, add_process_noise=True)

        if k % 100 == 0:
            if verbose: print(f"[t={k * params.TA:.2f}s] EE Pos: [{ee_true[0]:.3f}, {ee_true[1]:.3f}] | target: [{current_target[0]:.3f}, {current_target[1]:.3f}] | Distance: {distance_to_target:.3f}")

        # Note down prediction logic for data recorder
        if k % params.mpc_ratio != 0:
            z_pred = np.nan * np.ones(6) # ensure NaN so it breaks line appropriately
            mpc_cbf_active = np.nan
            mpc_cbf_slack = np.nan
            mpc_feasible = np.nan
            mpc_solvetime = np.nan
            mpc_cbf_min = np.nan

        # Record metrics properly
        data_recorder.record_step(
            time=k * params.TA,
            z_real=z_true,
            z_pred=z_pred if k % params.mpc_ratio == 0 else np.nan*np.ones(6),
            u=u,
            ddq_mpc=ddq_mpc,
            ddq_local_cbf=ddq_cbf,
            z_des=np.concatenate([q_d, dq_d]) if q_d is not None else None,
            p_ref = current_target,
            q_ref = remote_controller.waypoints_jointspace[waypoint_idx],
            ee_pos_real=ee_true,
            ee_pos_des=dynamics.fk_points(q_d)[-1] if q_d is not None else np.nan*np.ones(2),
            ee_pos_pred=dynamics.fk_points(z_pred[:3])[-1] if z_pred is not None else np.nan*np.ones(2),
            local_cbf_active = local_cbf_active,
            local_cbf_slack = local_cbf_slack,
            local_cbf_feasible = local_cbf_feasible,
            local_cbf_solvetime = local_cbf_solvetime, 
            local_cbf_min = local_cbf_min,
            mpc_cbf_active = mpc_cbf_active, 
            mpc_cbf_slack = mpc_cbf_slack,
            mpc_feasible = mpc_feasible, 
            mpc_solvetime = mpc_solvetime,
            mpc_cbf_min = mpc_cbf_min,
        )

        

    # Fill final state 
    distance_to_goal = np.linalg.norm(ee_true - final_target)
    if distance_to_goal >= 0.02 or waypoint_idx < len(params.waypoints) - 1:
        th_true = z_true[:3]
        om_true = z_true[3:]
        _, _, _, ee_true = dynamics.fk_points(th_true)
        v_ee_true = dynamics.jac_ee(th_true) @ om_true
        joint_angles_true[max_steps - 1] = th_true
        ee_states_true[max_steps - 1, :2] = ee_true
        ee_states_true[max_steps - 1, 2:] = v_ee_true

    if verbose: print("Experiment completed. Generating Visualizations and Saving Data...")
    
    # Export Detailed Trajectory and Data via the Recorder
    data_recorder.save_to_csv(f"{params.scenario_name}_{arch_name}_data.csv")

    #compute single run metrics
    detailed_csv_path = os.path.join(
        params.out_dir,
        f"{params.scenario_name}_{arch_name}_data.csv",
    )

    metrics = post_process_single_run(
        detailed_csv_path=detailed_csv_path,
        params=params,
        dynamics=dynamics,
        arch_name=arch_name,
        output_dir=params.out_dir,
    )


    # Prepare reference trajectory for visualization
    pr_ref = np.zeros((max_steps, 2))
    dpr_ref = np.zeros((max_steps, 2))
    pr_ref[:] = final_target

    
    if(plotting):
        params.out_plot = "sim_plot.png"
    if(animation):
        params.out_anim = "sim_animation.gif"
    
    import pandas as pd
    df_detailed = pd.read_csv(os.path.join(params.out_dir, f"{params.scenario_name}_{arch_name}_data.csv"))
    
    # Visualization
    if(plotting):
        startViz = time.time()
        generate_visualizations(
            t_array,
            ee_states_true,
            U_alpha,
            pr_ref,
            dpr_ref,
            params,
            joint_angles_true,
            measurement_noise=np.zeros_like(joint_angles_true),
            detailed_data=df_detailed
        )

        if verbose: print(f"Creating the visualization took {round((time.time()-startViz)*1000)} ms.")

    if verbose: print(f"Data saved to: {params.out_dir}")
    elapsedTime = time.time()-t0    
    
    if verbose: print_run_summary(metrics, arch_name, elapsedTime)

    return params.out_dir, elapsedTime#, metrics
    # END OF SINGLE RUN

############## Helpers ################
def _architecture_variables(arch_name: str):
#returns boolean variables based on the chosen architecture
    use_local_cbf, use_remote_cbf = False, False
    mpc_use_CBF,mpc_use_slack = False, False
    match arch_name:
        case "LocalCBF":
            use_local_cbf = True
        case "MPC-CBF":
            use_remote_cbf = True
            mpc_use_CBF = True
        case "Combined":
            use_local_cbf, use_remote_cbf = True, True
            mpc_use_CBF = True
            mpc_use_slack = True
        case "Nominal":
            use_local_cbf, use_remote_cbf = False, False 
        case _:
            print(arch_name + " seems to be an invalid architecture name. Defaulting to nominal case (unconstrained).")

    return use_local_cbf, use_remote_cbf, mpc_use_CBF, mpc_use_slack



def fast_predict(dynamics,params,z_delayed, input_seq):
    """Simulate plant forward pass for delay compensation for tau steps (remote timing)."""
    z_curr = z_delayed.copy()
    for u in input_seq:
        z_curr = dynamics.joint_dynamics_step(z_curr, u, params.mpc_TA, add_process_noise=False)
    return z_curr    


############## Simulation Runners ################
def run_all_architectures(params:SystemParams, with_disturbances: bool = False, make_plots: bool = True, make_animation: bool = False, make_summary_plots: bool = False):

    # Create Folder for saving data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = f"data/comparison/comparison_{timestamp}"
    os.makedirs(base_out_dir, exist_ok=True)

    #save params
    params.save_meta_data(base_out_dir)

    # Generate Disturbances    
    disturbances = None
    if with_disturbances:
        disturbance_file = os.path.join(base_out_dir, "disturbances")
        if os.path.exists(disturbance_file):
            disturbances = np.load(disturbance_file)
        else:
            seed = params.random_seed
            disturbances = generate_single_disturbances(params, base_out_dir, seed, verbose =True, save_csv=False)
    
    print("=" * 60)
    print("RUNNING ARCHITECTURE 0: NOMINAL MPC (No CBF)")
    print("=" * 60)
    dir_local, elapsedTime = single_run(params,"Nominal", base_out_dir=base_out_dir,noise = disturbances,
                                        plotting = make_plots, animation=make_animation)
    # print(f"Simulating the nominal case took {elapsedTime:.2f} seconds.")
    
    print("=" * 60)
    print("RUNNING ARCHITECTURE 1: NOMINAL MPC + LOCAL CBF")
    print("=" * 60)
    dir_local, elapsedTime = single_run(params,"LocalCBF", base_out_dir=base_out_dir,noise = disturbances,
                                        plotting = make_plots, animation=make_animation)
    #print(f"Simulating the local CBF case took {elapsedTime:.2f} seconds.")

    print("\n" + "=" * 60)
    print("RUNNING ARCHITECTURE 2: REMOTE MPC-CBF + NOMINAL LOCAL")
    print("=" * 60)
    dir_local, elapsedTime = single_run(params,"MPC-CBF", base_out_dir=base_out_dir,noise = disturbances,
                                        plotting = make_plots, animation=make_animation)
    #print(f"Simulating the remote MPC-CBF case took {elapsedTime:.2f} seconds.")

    print("\n" + "=" * 60)
    print("RUNNING ARCHITECTURE 3: COMBINED SOFT REMOTE MPC-CBF + LOCAL CBF")
    print("=" * 60)
    dir_local, elapsedTime = single_run(params,"Combined", base_out_dir=base_out_dir,noise = disturbances,
                                        plotting = make_plots, animation=make_animation)
    #print(f"Simulating the combined case took {elapsedTime:.2f} seconds.")

    print(f"\nAll architectures generated under: {base_out_dir}")
    if(make_summary_plots):
        print("Creating a joint plot.")
        plot_3dof_compare_results(
            compare_root=base_out_dir,
            params=params,
            architectures=("Nominal", "LocalCBF", "MPC-CBF", "Combined"),
            output_name="architecture_comparison_summary.png",
            show_plot=False,
        )
def run_specific_architecture(params: SystemParams, choice: Optional[int] = None, with_disturbances: bool = False,
                              make_plots: bool = True, make_animation: bool = False):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    match choice:
        case 0:
            architecture_name = "Nominal"
        case 1:
            architecture_name = "LocalCBF"
        case 2:    
            architecture_name =  "MPC-CBF"
        case 3:
            architecture_name = "Combined"
        case _:
            print("No architecture was chosen. Simulating the nominal case.")
            choice = 0
            architecture_name = "Nominal"
    base_out_dir = f"data/single/{architecture_name}_{timestamp}"
    params.strategy_name = architecture_name
    
    #save params as meta data
    params.save_meta_data(base_out_dir)
    #To read:
    #params = params.read_meta_data(base_out_dir)

    # Generate Disturbances    
    disturbances = None
    if with_disturbances:
        disturbance_file = os.path.join(base_out_dir, "disturbances")
        if os.path.exists(disturbance_file):
            disturbances = np.load(disturbance_file)
        else:
            seed = 1235
            disturbances = generate_single_disturbances(params, base_out_dir, seed, verbose =True, save_csv=False)


    match choice:
            case 0:
                print("=" * 60)
                print("RUNNING ARCHITECTURE 0: NOMINAL MPC (No CBF)")
                print("=" * 60)
                dir_local, elapsedTime = single_run(params,"Nominal", base_out_dir=base_out_dir,noise = disturbances,
                                                    plotting = make_plots, animation=make_animation)
                #print(f"Simulating the nominal case took {elapsedTime} seconds.")
            case 1:
                print("=" * 60)
                print("RUNNING ARCHITECTURE 1: NOMINAL MPC + LOCAL CBF")
                print("=" * 60)
                dir_local, elapsedTime = single_run(params,"LocalCBF", base_out_dir=base_out_dir,noise = disturbances,
                                                    plotting = make_plots, animation=make_animation)
                #print(f"Simulating the local CBF case took {elapsedTime} seconds.")
            case 2:    
                print("\n" + "=" * 60)
                print("RUNNING ARCHITECTURE 2: REMOTE MPC-CBF + NOMINAL LOCAL")
                print("=" * 60)
                dir_local, elapsedTime, = single_run(params,"MPC-CBF", base_out_dir=base_out_dir,noise = disturbances,
                                                    plotting = make_plots, animation=make_animation)
                #print(f"Simulating the remote MPC-CBF case took {elapsedTime} seconds.")
            case 3:
                print("\n" + "=" * 60)
                print("RUNNING ARCHITECTURE 3: COMBINED SOFT REMOTE-CBF + LOCAL CBF")
                print("=" * 60)
                dir_local, elapsedTime = single_run(params,"Combined", base_out_dir=base_out_dir,noise = disturbances,
                                                    plotting = make_plots, animation=make_animation)
                #print(f"Simulating the combined case took {elapsedTime} seconds.")

    print(f"\nData saved under: {base_out_dir}")
        
def run_monte_carlo(params: SystemParams, num_trials: int = 50, custom_out_dir=None, architectures: list = ["Nominal"],
                    start_from_trial: int = 0, skip_baseline: bool =False, 
                    make_plots: bool = False, make_animation: bool = False, make_summary_plots: bool = True):
    """
    Ececute iterative Montecarlo Trials
    Inputs:
        params: Parameters for the simulation
        num_trials: number of trials
        custom_out_dir: string with folder, where to save results
        architectures: list of string containing the architectures to consider
    """

    #Set to True, if you want outputs from the simulation runs
    verbose = False

    t0 = time.time()

    # Create Folder for saving data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = custom_out_dir if custom_out_dir else f"data/monte_carlo/monte_carlo_{timestamp}"
    if(start_from_trial>0 and not os.path.isdir(base_out_dir)):
        print(f"Could not find {base_out_dir} to run the loop from trial {start_from_trial}.")
        print(f"Aborting")
        return None, None
    os.makedirs(base_out_dir, exist_ok=True)

    
    print("=" * 60)
    print(f"STARTING MONTE CARLO RUN: {num_trials} TRIALS")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Considered architectures:")
    for arch in architectures:
        print(f"    {arch}")
    print("=" * 60)

    # Booleans for running the architectures
    RunNominal = False
    elapsedTimeNominal = 0
    if any("Nominal" in arch for arch in architectures):
        RunNominal = True
    # Booleans for running the architectures
    RunLocalCBF = False
    elapsedTimeLocalCBF = 0
    if any("LocalCBF" in arch for arch in architectures):
        RunLocalCBF = True
    # Booleans for running the architectures
    RunMPCCBF = False
    elapsedTimeMPCCBF = 0
    if any("MPC-CBF" in arch for arch in architectures):
        RunMPCCBF = True
    # Booleans for running the architectures
    RunCombined = False
    elapsedTimeCombined = 0
    if any("Combined" in arch for arch in architectures):
        RunCombined = True

    #save params
    params.save_meta_data(base_out_dir)

    if (start_from_trial == 0 and not skip_baseline):
        # Run the baseline cases once (no disturbance)if(RunNominal):
        print("=" * 60)
        print(f"\n>>> SIMULATING BASELINE <<<\n")
        print("=" * 60)
        params.out_dir = os.path.join(base_out_dir, f"Baseline")
        baseline_out_dir = params.out_dir
        
        tStartBaseline = time.time()

        #always run the nominal baseline case
        print("     Running Architecture 0: Nominal MPC (no CBF)")
        dir_local, elapsedTimeNominal = single_run(params,"Nominal", base_out_dir=baseline_out_dir,noise = None, verbose =verbose,
                                                    plotting = make_plots, animation=make_animation)
        
        if(RunLocalCBF):
            print("     Running Architecture 1: Nominal MPC + Local CBF")
            dir_local, elapsedTimeLocalCBF = single_run(params,"LocalCBF", base_out_dir=baseline_out_dir,noise = None, verbose =verbose,
                                                    plotting = make_plots, animation=make_animation)

        if(RunMPCCBF):
            print("     Running Architecture 2: Remote MPC-CBF")
            dir_local, elapsedTimeMPCCBF = single_run(params,"MPC-CBF", base_out_dir=baseline_out_dir,noise = None, verbose =verbose,
                                                    plotting = make_plots, animation=make_animation)

        if(RunCombined):
            print("     Running Architecture 3: Combined MPC-CBF + Local CBF")
            dir_local, elapsedTimeCombined = single_run(params,"Combined", base_out_dir=baseline_out_dir,noise = None, verbose =verbose,
                                                    plotting = make_plots, animation=make_animation)
        
        tRun = elapsedTimeNominal+elapsedTimeLocalCBF+elapsedTimeMPCCBF+elapsedTimeCombined
        tEstimated = 1.2*(num_trials)*np.average(tRun)
        print(f"        Basline computation took {tRun:.2f} seconds. Estimated time for the MC Trials: {str(datetime.timedelta(seconds=round(tEstimated)))} h:min:s")
    else:
        print(f"Running the Monte Carlo Run from Trial {start_from_trial+1}.")

    # Run Monte Carlo Loop
    runTimings = []
    for trial_id in range(start_from_trial,num_trials):
        tStartTrial = time.time()
        print("=" * 60)
        print(f"\n>>> EXECUTING TRIAL {trial_id+1} <<<\n")
        print("=" * 60)

        #Make Save Path for Trial
        params.out_dir = os.path.join(base_out_dir, f"trial_{trial_id}")
        trial_out_dir = params.out_dir
        os.makedirs(params.out_dir, exist_ok=True)

        # Generate Disturbances    
        disturbance_file = os.path.join(base_out_dir, "disturbances")
        if os.path.exists(disturbance_file):
            disturbances = np.load(disturbance_file)
        else:
            seed = params.random_seed + trial_id # adapt with trial ID every run
            disturbances = generate_single_disturbances(params, output_dir=trial_out_dir, seed=seed, verbose =verbose, save_csv=False)
        
        
        if(RunNominal):
            print("     Running Architecture 0: Nominal MPC + PD+ Local (no CBF)")
            dir_local, elapsedTimeNominal = single_run(params,"Nominal", base_out_dir=trial_out_dir,noise = disturbances, verbose =verbose,
                                                    plotting = make_plots, animation=make_animation)
        
        if(RunLocalCBF):
            print("     Running Architecture 1: Nominal MPC + Local CBF")
            dir_local, elapsedTimeLocalCBF = single_run(params,"LocalCBF", base_out_dir=trial_out_dir,noise = disturbances, verbose =verbose,
                                                    plotting = make_plots, animation=make_animation)

        if(RunMPCCBF):
            print("     Running Architecture 2: Remote MPC-CBF + PD+ Local")
            dir_local, elapsedTimeMPCCBF = single_run(params,"MPC-CBF", base_out_dir=trial_out_dir,noise = disturbances, verbose =verbose,
                                                    plotting = make_plots, animation=make_animation)

        if(RunCombined):
            print("     Running Architecture 3: Combined MPC-CBF + Local CBF")
            dir_local, elapsedTimeCombined = single_run(params,"Combined", base_out_dir=trial_out_dir,noise = disturbances, verbose =verbose,
                                                    plotting = make_plots, animation=make_animation)
        tRun = elapsedTimeNominal+elapsedTimeLocalCBF+elapsedTimeMPCCBF+elapsedTimeCombined
        runTimings.append(tRun) 
        tEstimated = (num_trials-(trial_id+1))*np.average(runTimings)
        print(f"        Trial {trial_id+1} took {tRun:.2f} seconds. Estimated time left: {str(datetime.timedelta(seconds=round(tEstimated)))} h:min:s")
        
    _,summary = analyse_monte_carlo_run(base_out_dir, save_excel=True)
    tTotal = time.time()-t0
    print(f"\nAll trials finished. Outputs saved to {base_out_dir}")
    print(f"The run took {str(datetime.timedelta(seconds=round(tTotal)))} h:min:s")

    if(make_summary_plots):
        print("Creating a workspace plot for the monte carlo run.")
        legend_architectures = {"LocalCBF", "MPC-CBF", "Combined"}
        legend_ncol = 1 + sum(arch in legend_architectures for arch in architectures)
        plot_workspace(folder_path = base_out_dir, 
                    output_folder=base_out_dir,
                    architectures=architectures,
                    show_plot=False, 
                    xlim=(-0.05, 0.7),
                    ylim=(-0.15, 0.45),
                    square_limits=False,
                    save_pdf=False,
                    legend_ncol=legend_ncol)
    return summary,base_out_dir
    
def run_disturbance_sweep(
    params,
    disturbance_bounds,
    num_trials: int = 10,
    architectures=("LocalCBF", "MPC-CBF"),
    custom_out_dir=None,
    save_excel: bool = True, 
    make_plots: bool = False, 
    make_animation: bool = False, 
    make_summary_plots: bool = True
):
    """
    Run a Monte Carlo sweep over disturbance bounds.

    For each disturbance bound:
        process_noise_clip = bound
        process_noise_std  = sigma_factor * bound

    Then runs Monte Carlo for the selected architectures and analyzes each bound.
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_out_dir = custom_out_dir or f"data/disturbance_sweep/disturbance_sweep_{timestamp}"
    os.makedirs(sweep_out_dir, exist_ok=True)

    print("=" * 60)
    print("STARTING DISTURBANCE SWEEP")
    print(f"Bounds: {list(disturbance_bounds)}")
    print(f"Trials per bound: {num_trials}")
    print(f"Architectures: {list(architectures)}")
    print(f"Output: {sweep_out_dir}")
    print("=" * 60)

    sweep_dirs = []

    for bound in disturbance_bounds:
        bound = float(bound)

        params_bound = copy.deepcopy(params)
        params_bound.torque_disturbance_bound = bound
        params_bound.torque_disturbance_std = bound*0.5
        params_bound.enable_process_noise = True

        bound_label = f"bound_{bound:.5f}".replace(".", "p")
        bound_out_dir = os.path.join(sweep_out_dir, bound_label)
        os.makedirs(bound_out_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"RUNNING DISTURBANCE BOUND: {bound:.6f}")
        print(f"Standard Deviation = {params_bound.torque_disturbance_std}")
        print("=" * 60)

        # Save sweep-level params for this bound.
        params_bound.save_meta_data(bound_out_dir)

        # Run MC.
        summary,_ = run_monte_carlo(
            params=params_bound,
            num_trials=num_trials,
            custom_out_dir=bound_out_dir,
            architectures=list(architectures),
            skip_baseline=True,
            make_plots =make_plots,
            make_animation=make_animation,
            make_summary_plots= False            
        )
        summary["disturbance_bound"] = bound
        summary["torque_disturbance_bound"] = params_bound.torque_disturbance_bound
        summary["torque_disturbance_std"] = params_bound.torque_disturbance_std
         
        # Save a bound-specific summary with disturbance columns included.
        summary_path = os.path.join(bound_out_dir, "disturbance_bound_summary.csv")
        summary.to_csv(summary_path, index=False)

        sweep_dirs.append(bound_out_dir)

    print("\n" + "=" * 60)
    print("DISTURBANCE SWEEP FINISHED")
    print(f"Output: {sweep_out_dir}")
    print("=" * 60)

    if make_summary_plots:
        print("Making a plot of min clearance vs disturbance")
        plot_min_clearance_over_disturbance(
            sweep_folder_path=sweep_out_dir,
            architectures=architectures,
            obstacle_index=1, #0: obstacle 1, 1: obstacle 2
            include_safety_margin=False,
            output_name="min_clearance_over_disturbance",
            save_pdf=False
        )
    # Aggregate across all bounds.
    sweep_summary = analyse_disturbance_sweep(
        sweep_out_dir=sweep_out_dir,
        save_excel=save_excel,
    )

    return sweep_out_dir, sweep_summary

def run_delay_sweep(
    params,
    delays = range(1,11),
    num_trials: int = 10,
    architectures=("LocalCBF", "MPC-CBF"),
    custom_out_dir=None,
    save_excel: bool = True,
    make_plots: bool = False, 
    make_animation: bool = False, 
    make_summary_plots: bool = True
):
    """
    Run a sweep over delays.
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_out_dir = custom_out_dir or f"data/delay_sweep/delay_sweep_{timestamp}"
    os.makedirs(sweep_out_dir, exist_ok=True)

    print("=" * 60)
    print("STARTING Delay SWEEP")
    print(f"Considered Delays: {list(delays)}")
    print(f"Trials per Delay: {num_trials}")
    print(f"Architectures: {list(architectures)}")
    print(f"Output: {sweep_out_dir}")
    print("=" * 60)

    sweep_dirs = []

    for delay in delays:
        if(delay <=0):
            continue #Delay has to be 1 or bigger for causality
        params.tau_known = delay
        params.tau_residual = 0
        params.tau = delay

        delay_label = f"delay_{delay}"
        delay_out_dir = os.path.join(sweep_out_dir, delay_label)
        os.makedirs(delay_out_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"CURRENT DELAY: {delay}")
        print("=" * 60)

        # Save sweep-level params for this bound.
        params.save_meta_data(delay_out_dir)

        # Run MC.
        summary,_ = run_monte_carlo(
            params=params,
            num_trials=num_trials,
            custom_out_dir=delay_out_dir,
            architectures=list(architectures),  
            skip_baseline=True,
            make_plots=make_plots,
            make_animation=make_animation,
            make_summary_plots= False       
        )
        
        summary["delay"] = delay
         
        # Save a bound-specific summary with disturbance columns included.
        summary_path = os.path.join(delay_out_dir, "delay_summary.csv")
        summary.to_csv(summary_path, index=False)

        sweep_dirs.append(delay_out_dir)

    print("\n" + "=" * 60)
    print("DELAY SWEEP FINISHED")
    print(f"Output: {sweep_out_dir}")
    print("=" * 60)

    # Summary plot
    if(make_summary_plots):
        plot_min_clearance_over_delay(
            sweep_folder_path=sweep_out_dir,
            output_folder=sweep_out_dir,
            output_name="min_clearance_over_known_delay",
            architectures=architectures,
            obstacle_index=1,                 # obstacle 2:idx 1, obstacle 1: idx 0
            include_safety_margin=False,      # zero = physical collision boundary
            figsize=(3.5, 1.8),               # IEEE single-column-ish
            xlim=(0.5, 10.5),
            #ylim=(-0.02, 0.12),
            save_pdf=False,
            save_png=True,
            save_csv= False,
            dpi=800,
            show_plot=False,
        )

    # Aggregate across all bounds.
    sweep_summary = analyse_delay_sweep(
        sweep_out_dir=sweep_out_dir,
        save_excel=save_excel,
    )

    return sweep_out_dir, sweep_summary

