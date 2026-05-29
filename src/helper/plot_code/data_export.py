"""
Enhanced data saving and visualization module for 3DOF arm
Supports CSV export for post-analysis and plotting from data
"""

import os
import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def save_trajectory_to_csv(t_array: np.ndarray, 
                           joint_angles: np.ndarray,
                           ee_states: np.ndarray,
                           U_alpha: np.ndarray,
                           pr: np.ndarray,
                           dpr: np.ndarray,
                           metrics_dict: Dict,
                           output_path: str,
                           params=None,
                           measurement_noise: Optional[np.ndarray] = None,
                           process_noise: Optional[np.ndarray] = None,
                           cbf_activations: Optional[np.ndarray] = None,
                           cbf_modifications: Optional[np.ndarray] = None,
                           remote_slacks: Optional[np.ndarray] = None,
                           local_slacks: Optional[np.ndarray] = None,
                           solve_times: Optional[np.ndarray] = None) -> str:
    """
    Save complete trajectory data to CSV file for later analysis
    
    Parameters:
    -----------
    t_array : np.ndarray (N,)
        Time array
    joint_angles : np.ndarray (N, 3)
        Joint angles [theta1, theta2, theta3] at each step
    ee_states : np.ndarray (N, 4)
        End-effector states [px, py, vx, vy] at each step
    U_alpha : np.ndarray (N-1, 3)
        Joint accelerations at each step
    pr : np.ndarray (N, 2)
        Reference end-effector positions
    dpr : np.ndarray (N, 2)
        Reference end-effector velocities
    metrics_dict : Dict
        Metrics dictionary containing success, clearance, etc.
    output_path : str
        Path to output CSV file
    params : SystemParams, optional
        System parameters for reference
    measurement_noise : np.ndarray, optional
        Measurement noise magnitude at each step
    process_noise : np.ndarray, optional
        Process noise (N, 3) applied to control inputs at each step
    cbf_activations : np.ndarray, optional
        CBF activation status (bool) at each step
    cbf_modifications : np.ndarray, optional
        CBF modification magnitude at each step
    remote_slacks : np.ndarray, optional
        Remote CBF/MPC slack variable values at each step
    local_slacks : np.ndarray, optional
        Local CBF slack variable values at each step
        
    Returns:
    --------
    str : Path to the saved CSV file
    """
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Prepare data rows
    rows = []
    
    # Calculate CBF activation statistics for metadata
    cbf_activation_count = 0
    cbf_activation_rate = 0.0
    if cbf_activations is not None and len(cbf_activations) > 0:
        cbf_activation_count = int(np.sum(cbf_activations))
        cbf_activation_rate = float(np.mean(cbf_activations))
    
    # Add metadata header
    header_rows = [
        ['# Trajectory Data for 3DOF Robotic Arm with Obstacle Avoidance'],
        ['# Generated:', datetime.now().isoformat()],
        ['# Total time:', f"{t_array[-1]:.2f}s"],
        ['# Number of steps:', len(t_array)],
        ['# Experiment success:', metrics_dict.get('success', False)],
        ['# Min clearance:', f"{metrics_dict.get('min_clearance', 0):.4f}m"],
        ['# Tracking RMSE:', f"{metrics_dict.get('tracking_rmse', 0):.4f}m"],
        ['# Collision count:', metrics_dict.get('collision_count', 0)],
        ['# CBF activation count:', cbf_activation_count],
        ['# CBF activation rate:', f"{cbf_activation_rate*100:.2f}%"],
        [''],
    ]
    
    # Column headers
    column_headers = [
        'step', 'time', 
        'theta1', 'theta2', 'theta3',           # Joint angles
        'ee_x', 'ee_y', 'ee_vx', 'ee_vy',      # End-effector position and velocity
        'ref_x', 'ref_y', 'ref_vx', 'ref_vy',  # Reference position and velocity
        'alpha1', 'alpha2', 'alpha3',           # Joint accelerations
        'alpha_norm',                           # L2 norm of alpha vector
        'tracking_error_x', 'tracking_error_y',# Tracking error
        'noise_magnitude',                      # Process noise magnitude (L2 norm of w)
        'noise_w1', 'noise_w2', 'noise_w3',    # Individual joint noise (rad/s)
        'cbf_activated',                        # CBF activation status (0=False, 1=True)
        'cbf_modification',                     # CBF modification magnitude
        'remote_slack',                         # Remote CBF/MPC slack variable
        'local_slack'                           # Local CBF slack variable
    ]
    
    # Populate data rows
    for k in range(len(t_array)):
        # Calculate alpha values and norm
        alpha1 = U_alpha[k, 0] if k < len(U_alpha) else 0
        alpha2 = U_alpha[k, 1] if k < len(U_alpha) else 0
        alpha3 = U_alpha[k, 2] if k < len(U_alpha) else 0
        alpha_norm = np.linalg.norm([alpha1, alpha2, alpha3])
        
        row = {
            'step': k,
            'time': t_array[k],
            'theta1': joint_angles[k, 0],
            'theta2': joint_angles[k, 1],
            'theta3': joint_angles[k, 2],
            'ee_x': ee_states[k, 0],
            'ee_y': ee_states[k, 1],
            'ee_vx': ee_states[k, 2] if ee_states.shape[1] > 2 else 0,
            'ee_vy': ee_states[k, 3] if ee_states.shape[1] > 3 else 0,
            'ref_x': pr[min(k, len(pr)-1), 0],
            'ref_y': pr[min(k, len(pr)-1), 1],
            'ref_vx': dpr[min(k, len(dpr)-1), 0],
            'ref_vy': dpr[min(k, len(dpr)-1), 1],
            'alpha1': alpha1,
            'alpha2': alpha2,
            'alpha3': alpha3,
            'alpha_norm': alpha_norm,
            'tracking_error_x': pr[min(k, len(pr)-1), 0] - ee_states[k, 0],
            'tracking_error_y': pr[min(k, len(pr)-1), 1] - ee_states[k, 1],
            'noise_magnitude': np.linalg.norm(process_noise[k]) if process_noise is not None and k < len(process_noise) else 0.0,
            'noise_w1': process_noise[k, 0] if process_noise is not None and k < len(process_noise) else 0.0,
            'noise_w2': process_noise[k, 1] if process_noise is not None and k < len(process_noise) else 0.0,
            'noise_w3': process_noise[k, 2] if process_noise is not None and k < len(process_noise) else 0.0,
            'cbf_activated': int(cbf_activations[k]) if cbf_activations is not None and k < len(cbf_activations) else 0,
            'cbf_modification': float(cbf_modifications[k]) if cbf_modifications is not None and k < len(cbf_modifications) else 0.0,
            'remote_slack': float(remote_slacks[k]) if remote_slacks is not None and k < len(remote_slacks) else 0.0,
            'local_slack': float(local_slacks[k]) if local_slacks is not None and k < len(local_slacks) else 0.0,
        }
        rows.append(row)
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Write metadata
        for header_row in header_rows:
            f.write('# ' + ','.join(str(x) for x in header_row) + '\n')
        
        # Write data with DictWriter
        writer = csv.DictWriter(f, fieldnames=column_headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n[OK] Trajectory data saved to: {output_path}")
    print(f"  Total rows: {len(rows)}")
    print(f"  Columns: {len(column_headers)}")
    
    return output_path


def load_trajectory_from_csv(csv_path: str) -> Dict:
    """
    Load trajectory data from CSV file for plotting or analysis
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file
        
    Returns:
    --------
    Dict : Dictionary containing:
        - 't_array': time array
        - 'joint_angles': (N, 3) joint angles
        - 'ee_states': (N, 4) end-effector states
        - 'U_alpha': (N-1, 3) control inputs
        - 'pr': (N, 2) reference positions
        - 'dpr': (N, 2) reference velocities
        - 'tracking_errors': (N, 2) tracking errors
        - 'metadata': dict of metadata from header
    """
    
    metadata = {}
    
    # First find the line number where data starts (skip comment lines)
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and line.strip():  # Find first non-comment, non-empty line
                data_start = i
                break
    
    # Read with pandas, skipping comment lines
    df = pd.read_csv(csv_path, skiprows=data_start)
    
    # Extract metadata (if needed)
    # Can be parsed from CSV header comments, simplified handling here
    
    # Convert to required format
    n = len(df)
    t_array = df['time'].values
    joint_angles = df[['theta1', 'theta2', 'theta3']].values
    ee_states = df[['ee_x', 'ee_y', 'ee_vx', 'ee_vy']].values
    pr = df[['ref_x', 'ref_y']].values
    dpr = df[['ref_vx', 'ref_vy']].values
    
    # Control inputs
    U_alpha = df[['alpha1', 'alpha2', 'alpha3']].values[:-1] if n > 1 else np.array([]).reshape(0, 3)
    
    # Tracking errors
    tracking_errors = df[['tracking_error_x', 'tracking_error_y']].values if 'tracking_error_x' in df.columns else np.zeros((n, 2))
    
    return {
        't_array': t_array,
        'joint_angles': joint_angles,
        'ee_states': ee_states,
        'U_alpha': U_alpha,
        'pr': pr,
        'dpr': dpr,
        'tracking_errors': tracking_errors,
        'metadata': metadata,
    }


def plot_trajectory_from_csv(csv_path: str, output_plot_path: Optional[str] = None):
    """
    Generate plots from saved CSV trajectory data
    
    Parameters:
    -----------
    csv_path : str
        Path to input CSV file
    output_plot_path : str, optional
        Path to save output plot image
    """
    
    import matplotlib.pyplot as plt
    
    # Load data
    data = load_trajectory_from_csv(csv_path)
    t = data['t_array']
    joint_angles = data['joint_angles']
    ee_states = data['ee_states']
    U_alpha = data['U_alpha']
    pr = data['pr']
    tracking_errors = data['tracking_errors']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('3DOF Arm Trajectory Analysis', fontsize=14, fontweight='bold')
    
    # 1. End-effector position tracking
    ax = axes[0, 0]
    ax.plot(ee_states[:, 0], ee_states[:, 1], 'b-', label='Actual', linewidth=2)
    ax.plot(pr[:, 0], pr[:, 1], 'r--', label='Reference', linewidth=2)
    ax.scatter(ee_states[0, 0], ee_states[0, 1], color='g', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(ee_states[-1, 0], ee_states[-1, 1], color='r', s=100, marker='s', label='End', zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('End-Effector Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 2. Joint angles over time
    ax = axes[0, 1]
    ax.plot(t, joint_angles[:, 0], label='θ1', linewidth=2)
    ax.plot(t, joint_angles[:, 1], label='θ2', linewidth=2)
    ax.plot(t, joint_angles[:, 2], label='θ3', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Angle (rad)')
    ax.set_title('Joint Angles vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. End-effector velocity
    ax = axes[1, 0]
    ee_vel_mag = np.linalg.norm(ee_states[:, 2:4], axis=1)
    ax.plot(t, ee_vel_mag, 'g-', linewidth=2)
    ax.fill_between(t, 0, ee_vel_mag, alpha=0.3, color='g')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity Magnitude (m/s)')
    ax.set_title('End-Effector Velocity')
    ax.grid(True, alpha=0.3)
    
    # 4. Tracking error
    ax = axes[1, 1]
    tracking_error_mag = np.linalg.norm(tracking_errors, axis=1)
    ax.semilogy(t, tracking_error_mag, 'r-', linewidth=2)
    ax.fill_between(t, 1e-6, tracking_error_mag, alpha=0.3, color='r')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error (m)')
    ax.set_title('Tracking Error vs Time')
    ax.grid(True, alpha=0.3, which='both')
    
    # 5. Joint accelerations
    ax = axes[2, 0]
    t_ctrl = t[:len(U_alpha)]
    ax.plot(t_ctrl, U_alpha[:, 0], label='α1', linewidth=2)
    ax.plot(t_ctrl, U_alpha[:, 1], label='α2', linewidth=2)
    ax.plot(t_ctrl, U_alpha[:, 2], label='α3', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Acceleration (rad/s²)')
    ax.set_title('Joint Accelerations vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Control effort
    ax = axes[2, 1]
    control_effort = np.linalg.norm(U_alpha, axis=1)
    ax.bar(range(len(control_effort)), control_effort, width=1.0, alpha=0.7, color='orange')
    ax.set_xlabel('Step')
    ax.set_ylabel('Control Norm (N·m)')
    ax.set_title('Control Effort vs Step')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Plot saved to: {output_plot_path}")
    
    plt.show()
    
    return fig
