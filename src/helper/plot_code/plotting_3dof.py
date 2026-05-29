# plotting_3dof.py
"""
Specialized visualization module for 3DOF robotic arms
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from typing import Tuple, List

# ---------------------- 3DOF Kinematics ----------------------

def forward_kinematics_3r(theta1: float, theta2: float, theta3: float, 
                         L1: float, L2: float, L3: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    3R planar arm forward kinematics
    Returns: (base, joint1, joint2, end_effector) positions
    """
    base = np.array([0.0, 0.0])
    joint1 = np.array([L1*np.cos(theta1), L1*np.sin(theta1)])
    joint2 = joint1 + np.array([L2*np.cos(theta1+theta2), L2*np.sin(theta1+theta2)])
    end_effector = joint2 + np.array([L3*np.cos(theta1+theta2+theta3), L3*np.sin(theta1+theta2+theta3)])
    return base, joint1, joint2, end_effector

# ---------------------- Obstacle Drawing ----------------------

def draw_obstacles(ax, scene, edge='k', face=(0.2, 0.6, 1.0, 0.15), linewidth=2.0, label=True, zorder=0):
    """Draw circular obstacles"""
    if scene is None or getattr(scene, 'obstacles', None) is None:
        return []
    artists = []
    for i, obs in enumerate(scene.obstacles):
        if getattr(obs, 'type', 'circle') == 'circle':
            c = getattr(obs, 'center', None)
            r = getattr(obs, 'radius', None)
            if c is None or r is None:
                continue
            cx, cy = float(c[0]), float(c[1])
            rr = float(r)
            circ = patches.Circle((cx, cy), rr, edgecolor=edge, facecolor=face, linewidth=linewidth, zorder=zorder)
            ax.add_patch(circ)
            artists.append(circ)
            if label:
                ax.text(cx + 0.01, cy + 0.01, f"obs{i+1}", color=edge, fontsize=9, zorder=zorder+1)
    return artists

# ---------------------- 3DOF Specialized Plotting ----------------------

def plot_3dof_results(all_t, ee_states, U_alpha, pr, dpr, params, joint_angles, measurement_noise=None, detailed_data=None):
    """3DOF experiment results plotting"""
    desired_time = None
    desired_x = None
    desired_y = None
    q_des = None
    qd_des = None
    q_actual_detailed = None
    if detailed_data is not None and {'ee_x_des', 'ee_y_des'}.issubset(detailed_data.columns):
        desired_x = detailed_data['ee_x_des'].to_numpy()
        desired_y = detailed_data['ee_y_des'].to_numpy()
        desired_time = all_t[1:1 + len(desired_x)]
    if detailed_data is not None and {'theta1_des', 'theta2_des', 'theta3_des', 'omega1_des', 'omega2_des', 'omega3_des'}.issubset(detailed_data.columns):
        q_des = detailed_data[['theta1_des', 'theta2_des', 'theta3_des']].to_numpy()
        qd_des = detailed_data[['omega1_des', 'omega2_des', 'omega3_des']].to_numpy()
    if detailed_data is not None and {'theta1_real', 'theta2_real', 'theta3_real'}.issubset(detailed_data.columns):
        q_actual_detailed = detailed_data[['theta1_real', 'theta2_real', 'theta3_real']].to_numpy()
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f'3-DOF CBF Obstacle Avoidance: {params.scenario_name} (Delay={params.tau}s)', fontsize=16)

    # First row: Applied joint inputs (commanded accelerations)
    for i in range(3):
        axes[0, i].plot(all_t[:-1], U_alpha[:, i], 'b-', lw=2, label=f'u{i+1}')
        axes[0, i].legend(loc='lower left', markerscale=1.0); axes[0, i].grid(True)
        axes[0, i].set_xlabel('Time (s)'); axes[0, i].set_ylabel('Input (rad/s²)')
        axes[0, i].set_title(f'Joint {i+1} Input')

    # Second row: EE position & joint angles
    axes[1, 0].plot(all_t, ee_states[:, 0], 'b-', lw=2, label='x actual')
    if desired_time is not None:
        axes[1, 0].plot(desired_time, desired_x, 'k--', lw=1.5, label='x from q_d')
    # axes[1, 0].plot(all_t, pr[:, 0], 'r--', lw=2, label='x reference')  # No continuous reference trajectory
    axes[1, 0].legend(loc='lower left', markerscale=1.0); axes[1, 0].grid(True)
    axes[1, 0].set_xlabel('Time (s)'); axes[1, 0].set_ylabel('X Position (m)')
    axes[1, 0].set_title('X Position (Waypoint Navigation)')

    axes[1, 1].plot(all_t, ee_states[:, 1], 'b-', lw=2, label='y actual')
    if desired_time is not None:
        axes[1, 1].plot(desired_time, desired_y, 'k--', lw=1.5, label='y from q_d')
    # axes[1, 1].plot(all_t, pr[:, 1], 'r--', lw=2, label='y reference')  # No continuous reference trajectory
    axes[1, 1].legend(loc='lower left', markerscale=1.0); axes[1, 1].grid(True)
    axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('Y Position (m)')
    axes[1, 1].set_title('Y Position (Waypoint Navigation)')

    # Joint angle tracking against q_d when available
    if measurement_noise is not None and len(measurement_noise) > 0 and np.any(measurement_noise):
        axes[1, 2].plot(all_t, measurement_noise, 'b-', lw=2, label='Measurement Noise Magnitude')
        
        # Check if measurement_noise is multi-dimensional
        if measurement_noise.ndim > 1:
            axes[1, 2].fill_between(all_t, 0, measurement_noise[:, 0], alpha=0.3)
        else:
            axes[1, 2].fill_between(all_t, 0, measurement_noise, alpha=0.3)
            
        axes[1, 2].grid(True)
        axes[1, 2].set_xlabel('Time (s)'); axes[1, 2].set_ylabel('Noise Magnitude (rad)')
        axes[1, 2].set_title('Measurement Noise (Joint Angles)')
        axes[1, 2].legend(loc='upper left', markerscale=1.0)
    elif q_des is not None:
        q_actual = q_actual_detailed if q_actual_detailed is not None else np.asarray(joint_angles[:len(q_des)])
        colors = ['b', 'r', 'g']
        labels = ['q1', 'q2', 'q3']
        for index, color in enumerate(colors):
            axes[1, 2].plot(desired_time, q_actual[:, index], color=color, lw=2, label=labels[index])
            axes[1, 2].plot(desired_time, q_des[:, index], color=color, lw=1.5, ls='--', label=f'{labels[index]}_d')
        axes[1, 2].legend(loc='lower left', markerscale=1.0, ncol=2); axes[1, 2].grid(True)
        axes[1, 2].set_xlabel('Time (s)'); axes[1, 2].set_ylabel('Joint Angle (rad)')
        axes[1, 2].set_title('Joint Tracking: q vs q_d')
    else:
        # Display joint angles
        axes[1, 2].plot(all_t, [angles[0] for angles in joint_angles], 'b-', lw=2, label='θ1')
        axes[1, 2].plot(all_t, [angles[1] for angles in joint_angles], 'r-', lw=2, label='θ2')
        axes[1, 2].plot(all_t, [angles[2] for angles in joint_angles], 'g-', lw=2, label='θ3')
        axes[1, 2].legend(loc='lower left', markerscale=1.0); axes[1, 2].grid(True)
        axes[1, 2].set_xlabel('Time (s)'); axes[1, 2].set_ylabel('Joint Angle (rad)')
        axes[1, 2].set_title('Joint Angles')

    # Third row: joint velocity tracking, workspace trajectory, and error/safety summary
    if detailed_data is not None and {'omega1_real', 'omega2_real', 'omega3_real'}.issubset(detailed_data.columns) and qd_des is not None:
        omega_actual = detailed_data[['omega1_real', 'omega2_real', 'omega3_real']].to_numpy()
        colors = ['b', 'r', 'g']
        labels = ['q1dot', 'q2dot', 'q3dot']
        for index, color in enumerate(colors):
            axes[2, 0].plot(desired_time, omega_actual[:, index], color=color, lw=2, label=labels[index])
            axes[2, 0].plot(desired_time, qd_des[:, index], color=color, lw=1.5, ls='--', label=f'{labels[index]}_d')
        axes[2, 0].legend(loc='lower left', markerscale=1.0, ncol=2); axes[2, 0].grid(True)
        axes[2, 0].set_xlabel('Time (s)'); axes[2, 0].set_ylabel('Joint Velocity (rad/s)')
        axes[2, 0].set_title('Velocity Tracking: qdot vs qdot_d')
    else:
        axes[2, 0].plot(all_t, [angles[0] for angles in joint_angles], 'b-', lw=2, label='θ1')
        axes[2, 0].plot(all_t, [angles[1] for angles in joint_angles], 'r-', lw=2, label='θ2')
        axes[2, 0].plot(all_t, [angles[2] for angles in joint_angles], 'g-', lw=2, label='θ3')
        axes[2, 0].legend(loc='lower left', markerscale=1.0); axes[2, 0].grid(True)
        axes[2, 0].set_xlabel('Time (s)'); axes[2, 0].set_ylabel('Angle (rad)')
        axes[2, 0].set_title('Joint Angle History')

    # Workspace trajectory
    if hasattr(params, 'obstacle_scene') and params.obstacle_scene is not None:
        draw_obstacles(axes[2, 1], params.obstacle_scene, edge='red', face=(1.0, 0.5, 0.2, 0.3))
    
    # Plot joint2 trajectory
    j2_positions = []
    for angles in joint_angles:
        _, p1, p2, _ = forward_kinematics_3r(angles[0], angles[1], angles[2], params.L1, params.L2, params.L3)
        j2_positions.append(p2)
    j2_positions = np.array(j2_positions)
    axes[2, 1].plot(j2_positions[:, 0], j2_positions[:, 1], 'g-', lw=2, label='Joint2 trajectory', alpha=0.6)
    
    # Plot EE trajectory
    axes[2, 1].plot(ee_states[:, 0], ee_states[:, 1], 'b-', lw=2, label='End-effector trajectory')
    if desired_time is not None:
        axes[2, 1].plot(desired_x, desired_y, 'k--', lw=1.5, label='EE from q_d')
    
    # Plot waypoints directly from params.waypoints
    if hasattr(params, 'waypoints') and params.waypoints:
        waypoints_array = np.array(params.waypoints)
        # Start point (waypoint 0)
        axes[2, 1].scatter(waypoints_array[0, 0], waypoints_array[0, 1], color='green', s=150, 
                          marker='o', label='Start (WP0)', zorder=10, edgecolors='darkgreen', linewidths=2)
        # Intermediate waypoints (WP1, WP2, ...)
        if len(waypoints_array) > 2:
            for i in range(1, len(waypoints_array) - 1):
                axes[2, 1].scatter(waypoints_array[i, 0], waypoints_array[i, 1], color='orange', s=120, 
                                  marker='D', label=f'WP{i}' if i == 1 else '', zorder=10, 
                                  edgecolors='darkorange', linewidths=2)
                # Add text label for each intermediate waypoint
                axes[2, 1].text(waypoints_array[i, 0] + 0.02, waypoints_array[i, 1] + 0.02, 
                               f'WP{i}', fontsize=9, color='darkorange', fontweight='bold')
        # Goal point (final waypoint)
        axes[2, 1].scatter(waypoints_array[-1, 0], waypoints_array[-1, 1], color='red', s=150, 
                          marker='s', label=f'Goal (WP{len(waypoints_array)-1})', zorder=10, 
                          edgecolors='darkred', linewidths=2)
    
    axes[2, 1].set_aspect('equal', adjustable='box'); axes[2, 1].grid(True)
    axes[2, 1].set_xlim([-0.1, 1.0])  # X-axis range: -0.1 to 1.0 (same as animation)
    axes[2, 1].set_ylim([-0.5, 0.6])  # Y-axis range: -0.5 to 0.6 (same as animation)
    # Enlarge the plot area
    pos = axes[2, 1].get_position()
    axes[2, 1].set_position([pos.x0, pos.y0, pos.width*1.1, pos.height*1.1])
    axes[2, 1].legend(loc='upper right', markerscale=0.8, fontsize=7, ncol=1)
    axes[2, 1].set_xlabel('X (m)'); axes[2, 1].set_ylabel('Y (m)')
    axes[2, 1].set_title('Workspace Trajectory')

    # Obstacle distance or tracking error summary
    if hasattr(params, 'obstacle_scene') and params.obstacle_scene is not None and len(params.obstacle_scene.obstacles) > 0:
        distances = []
        for i in range(len(ee_states)):
            _, d = params.obstacle_scene.get_closest_obstacle(ee_states[i, :2])
            distances.append(d)
        axes[2, 2].plot(all_t, distances, 'b-', lw=2, label='Closest distance')
        axes[2, 2].axhline(0.0, color='r', ls='--', lw=2, label='Collision line')
        axes[2, 2].axhline(1e-3, color='orange', ls=':', lw=1, label='Safety threshold')
        axes[2, 2].set_title('Safety Distance')
    else:
        ee_err = np.linalg.norm(ee_states[:, :2] - pr[:, :2], axis=1)
        axes[2, 2].plot(all_t, ee_err, 'k-', lw=2, label='||p - p_ref||')
        if q_des is not None:
            q_actual = q_actual_detailed if q_actual_detailed is not None else np.asarray(joint_angles[:len(q_des)])
            q_err = np.linalg.norm(q_actual - q_des, axis=1)
            axes[2, 2].plot(desired_time, q_err, 'm--', lw=1.5, label='||q - q_d||')
        axes[2, 2].set_title('Tracking Error Summary')
    axes[2, 2].grid(True)
    axes[2, 2].legend(loc='lower left', markerscale=1.0)
    axes[2, 2].set_xlabel('Time (s)'); axes[2, 2].set_ylabel('Norm')

    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(params.out_dir, params.out_plot)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    #print(f"3DOF chart saved: {plot_path}")
    plt.close(fig)

def find_finish_index(all_t, ee_states, params, target_tol=None, hold_steps=20):
    """
    Find the first index at which the end effector reaches the final waypoint.

    Parameters
    ----------
    all_t : array-like
        Simulation time vector.
    ee_states : array-like
        End-effector states. The first two columns are assumed to be x/y position.
    params : object
        Parameter object containing params.waypoints.
    target_tol : float or None
        Distance tolerance for reaching the final waypoint. If None, tries to use
        params.target_tolerance and otherwise falls back to 0.02 m.
    hold_steps : int
        Number of additional samples kept after first reaching the target.
        This avoids cutting the animation too abruptly.

    Returns
    -------
    int
        Last index to include in the animation.
    """
    if not hasattr(params, "waypoints") or params.waypoints is None or len(params.waypoints) == 0:
        return len(all_t) - 1

    if target_tol is None:
        target_tol = getattr(params, "target_tolerance", 0.02)

    final_target = np.asarray(params.waypoints[-1], dtype=float)
    ee_pos = np.asarray(ee_states)[:, :2]

    distances = np.linalg.norm(ee_pos - final_target[None, :], axis=1)
    reached_indices = np.where(distances <= target_tol)[0]

    if len(reached_indices) == 0:
        return len(all_t) - 1

    finish_idx = int(reached_indices[0])
    finish_idx = min(finish_idx + hold_steps, len(all_t) - 1)

    return finish_idx

def create_3dof_animation(all_t, ee_states, joint_angles, pr, params, detailed_data=None):
    """Create 3DOF robotic arm animation"""

    finish_idx = find_finish_index(all_t, ee_states, params, target_tol=0.02, hold_steps=40)

    all_t = all_t[:finish_idx + 1]
    ee_states = ee_states[:finish_idx + 1]
    joint_angles = joint_angles[:finish_idx + 1]
    pr = pr[:finish_idx + 1]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    fig.suptitle(
        f"3DOF Robot Simulation: {params.strategy_name} Architecture",
        fontsize=13,
        y=0.97,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.10,
        top=0.86,
        wspace=0.22,
    )

    # Workspace view
    ax1.set_xlim(-0.7, 0.8); ax1.set_ylim(-0.7, 0.8)
    ax1.set_aspect('equal'); ax1.grid(True)
    
    ax1.set_title('Workspace')

    # Draw static obstacles
    if hasattr(params, 'obstacle_scene') and params.obstacle_scene is not None:
        draw_obstacles(ax1, params.obstacle_scene, edge='red', face=(1.0, 0.3, 0.3, 0.3))

    # Robot arm components
    link1_line, = ax1.plot([], [], 'b-', lw=8, label='Link 1')
    link2_line, = ax1.plot([], [], 'r-', lw=6, label='Link 2')
    link3_line, = ax1.plot([], [], 'g-', lw=4, label='Link 3')
    joint1_point, = ax1.plot([], [], 'ko', ms=8, label='Joint 1')
    joint2_point, = ax1.plot([], [], 'mo', ms=8, label='Joint 2')
    base_point, = ax1.plot([0], [0], 'ro', ms=10, label='Base')
    end_effector, = ax1.plot([], [], 'go', ms=12, label='End Effector')
    actual_trail, = ax1.plot([], [], 'b-', lw=2, alpha=0.6, label='End-effector Trajectory')
    desired_trail, = ax1.plot([], [], 'k--', lw=2, alpha=0.8, label='EE from q_d')
    desired_effector, = ax1.plot([], [], 'kx', ms=10, label='Desired EE')
    j2_trail, = ax1.plot([], [], 'g-', lw=2, alpha=0.6, label='Joint2 Trajectory')

    if detailed_data is not None:
        detailed_data = detailed_data.iloc[:finish_idx + 1].copy()
    desired_ee = None
    if detailed_data is not None and {'ee_x_des', 'ee_y_des'}.issubset(detailed_data.columns):
        desired_ee = detailed_data[['ee_x_des', 'ee_y_des']].to_numpy()

    waypoint_scatter = None
    if getattr(params, 'waypoints', None):
        waypoint_array = np.asarray(params.waypoints, dtype=float)
        waypoint_scatter = ax1.scatter(
            waypoint_array[:, 0],
            waypoint_array[:, 1],
            marker='X',
            s=120,
            c='gold',
            edgecolors='k',
            linewidths=1.0,
            label='Waypoints',
            zorder=5,
        )
        for idx, point in enumerate(waypoint_array, start=1):
            ax1.text(point[0] + 0.01, point[1] + 0.01, f"Target {idx}", fontsize=9, color='k', zorder=6)
    ax1.legend(loc='lower left', fontsize=10, markerscale=1.0)

    # Joint angle view
    ax2.set_xlim(0, all_t[-1]); ax2.set_ylim(-np.pi, np.pi); ax2.grid(True)
    ax2.set_title('Joint Angles')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Angle (rad)')
    theta1_line, = ax2.plot([], [], 'b-', lw=2, label='θ1')
    theta2_line, = ax2.plot([], [], 'r-', lw=2, label='θ2')
    theta3_line, = ax2.plot([], [], 'g-', lw=2, label='θ3')
    time_marker, = ax2.plot([], [], 'ko', ms=8)
    ax2.legend(loc='lower left', markerscale=1.0)

    # precompute joint positions
    joint_angles_arr = np.asarray(joint_angles)
    theta1_hist = joint_angles_arr[:, 0]
    theta2_hist = joint_angles_arr[:, 1]
    theta3_hist = joint_angles_arr[:, 2]

    j2_all = np.zeros((len(joint_angles_arr), 2))
    for i, angles in enumerate(joint_angles_arr):
        _, _, j2, _ = forward_kinematics_3r(
            angles[0], angles[1], angles[2],
            params.L1, params.L2, params.L3
        )
        j2_all[i, :] = j2


    def animate(frame):
        # Sample data proportionally to cover full duration
        idx = min(int(frame * len(all_t) / frames), len(all_t) - 1)
        theta1, theta2, theta3 = joint_angles[idx]
        base, j1, j2, ee = forward_kinematics_3r(theta1, theta2, theta3, params.L1, params.L2, params.L3)

        # Update robotic arm
        link1_line.set_data([base[0], j1[0]], [base[1], j1[1]])
        link2_line.set_data([j1[0], j2[0]], [j1[1], j2[1]])
        link3_line.set_data([j2[0], ee[0]], [j2[1], ee[1]])
        joint1_point.set_data([j1[0]], [j1[1]])
        joint2_point.set_data([j2[0]], [j2[1]])
        end_effector.set_data([ee[0]], [ee[1]])

        # Update trajectory trails
        trail_length = 500
        start_idx = max(0, idx - trail_length)
        
        # Update end-effector trail
        actual_trail.set_data(ee_states[start_idx:idx+1, 0], ee_states[start_idx:idx+1, 1])
        if desired_ee is not None:
            desired_start = max(0, min(start_idx - 1, len(desired_ee)))
            desired_stop = max(0, min(idx, len(desired_ee)))
            desired_trail.set_data(desired_ee[desired_start:desired_stop, 0], desired_ee[desired_start:desired_stop, 1])
            if desired_stop > 0:
                desired_effector.set_data([desired_ee[desired_stop - 1, 0]], [desired_ee[desired_stop - 1, 1]])
            else:
                desired_effector.set_data([], [])
        else:
            desired_trail.set_data([], [])
            desired_effector.set_data([], [])
        
        # Update joint2 trail
        j2_trail.set_data(j2_all[start_idx:idx+1, 0], j2_all[start_idx:idx+1, 1])

        # Update angle plot
        theta1_line.set_data(all_t[:idx+1], theta1_hist[:idx+1])
        theta2_line.set_data(all_t[:idx+1], theta2_hist[:idx+1])
        theta3_line.set_data(all_t[:idx+1], theta3_hist[:idx+1])
        time_marker.set_data([all_t[idx]], [0])

        return (link1_line, link2_line, link3_line, joint1_point, joint2_point, base_point, end_effector,
            actual_trail, desired_trail, desired_effector, j2_trail, theta1_line, theta2_line, theta3_line, time_marker)

    target_duration_s = all_t[-1]
    fps = 30
    frames = min(len(all_t), int(target_duration_s * fps))
    #frames = max(1, len(all_t) // 10)  # More frames for smoother animation
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50, blit=True, repeat=True)

    # Save animation
    anim_path = os.path.join(params.out_dir, params.out_anim)
    anim.save(anim_path, writer='pillow', fps=30, dpi = 80)
    print(f"3DOF animation saved: {anim_path}")
    plt.close(fig)

    return anim