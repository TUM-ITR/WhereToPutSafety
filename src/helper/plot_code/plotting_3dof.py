# plotting_3dof.py
"""
Specialized visualization module for 3DOF robotic arms
"""

import os
import numpy as np

from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import matplotlib.patches as patches
from typing import Tuple, List
from sim_module.params import SystemParams

################# Style ####################
def get_3dof_plot_style():
    return {
        "obstacle_facecolor": "#D9A5A5",
        "obstacle_edgecolor": "#B15C5C",
        "obstacle_text_color": "#8B0000",
        "obstacle_alpha": 0.55,

        "waypoint_marker": "D",
        "waypoint_color": "black",
        "waypoint_size": 42,

        "trajectory_color": "#0072B2",
        "distance_color": "#0072B2",
        "barrier_color": "#E69F00",

        "boundary_color": "red",
        "boundary_linestyle": "--",

        "joint_colors": ["#0072B2", "#E69F00", "#009E73"],
        "grid_alpha": 0.3,

        "arch_styles": {
            "Nominal": {
                "label": "Nominal",
                "color": "black",
                "linestyle": "-",
                "linewidth": 1.3,
            },
            "LocalCBF": {
                "label": "Local CBF",
                "color": "#0072B2",
                "linestyle": "-",
                "linewidth": 1.5,
            },
            "MPC-CBF": {
                "label": "Remote MPC-CBF",
                "color": "#E69F00",
                "linestyle": "--",
                "linewidth": 1.5,
            },
            "Combined": {
                "label": "Combined",
                "color": "#009E73",
                "linestyle": "-.",
                "linewidth": 1.5,
            },
        },
    }

def get_3dof_animation_style():
    return {
        # Paper-style obstacle / waypoint colors
        "obstacle_facecolor": "#D9A5A5",
        "obstacle_edgecolor": "#B15C5C",
        "obstacle_text_color": "#8B0000",
        "obstacle_alpha": 0.55,

        "waypoint_marker": "D",
        "waypoint_color": "black",
        "waypoint_size": 48,

        # Robot colors: same hue family, different shades
        "link1_color": "#005A8D",
        "link2_color": "#005A8D",
        "link3_color": "#005A8D",

        "joint_color": "black",
        "ee_color": "#B50ACF",
        "trail_color": "#0072B2",

        # Time-series colors
        "barrier_color": "#E69F00",
        "input_colors": ["#0072B2", "#E69F00", "#009E73"],

        "boundary_color": "red",
        "boundary_linestyle": "--",
        "grid_alpha": 0.3,
    }

# ---------------------- Computational Helpers ----------------------
################# 3DOF Kinematics ####################

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

################# Get Points on Robot (where CBFs are active) ####################
def monitored_robot_points_3r(q, params, include_base=False):
    """
    Reconstruct monitored robot points from one joint configuration.

    Uses the same convention as your post-processing:
        p1, p2, ee plus intermediate link samples.
    """
    q = np.asarray(q, dtype=float).reshape(3)
    base, p1, p2, ee = forward_kinematics_3r(
        q[0], q[1], q[2],
        params.L1, params.L2, params.L3,
    )

    points = []

    if include_base:
        points.append(base)

    points.extend([p1, p2, ee])

    n_samp = int(getattr(params, "cbf_link_samples", 3))
    grid = np.linspace(0.0, 1.0, n_samp + 1)[1:-1]

    for s in grid:
        points.append(base + s * (p1 - base))
        points.append(p1 + s * (p2 - p1))
        points.append(p2 + s * (ee - p2))

    return points

################# Distance and Barrier Value Computation ####################
def compute_distance_and_barrier_series(q_traj, params, include_base=False):
    """
    Compute two safety time series:

    d_min_raw(k):
        minimum signed physical clearance over all monitored robot points
        and all circular obstacles.

    h_min(k):
        minimum barrier value over all monitored robot points and obstacles,
        using radius + cbf_safety_margin.
    """
    q_traj = np.asarray(q_traj, dtype=float)

    d_min_raw = np.full(len(q_traj), np.nan)
    h_min = np.full(len(q_traj), np.nan)

    scene = getattr(params, "obstacle_scene", None)
    if scene is None or getattr(scene, "obstacles", None) is None:
        return d_min_raw, h_min

    margin = float(getattr(params, "cbf_safety_margin", 0.0))

    for k, q in enumerate(q_traj):
        if not np.all(np.isfinite(q)):
            continue

        points = monitored_robot_points_3r(
            q=q,
            params=params,
            include_base=include_base,
        )

        d_vals = []
        h_vals = []

        for obs in scene.obstacles:
            if getattr(obs, "type", "circle") != "circle":
                continue

            center = np.asarray(obs.center, dtype=float).reshape(2)
            r = float(obs.radius)
            r_eff = r + margin

            for p in points:
                p = np.asarray(p, dtype=float).reshape(2)
                dist = float(np.linalg.norm(p - center))

                # Raw physical signed clearance.
                d_vals.append(dist - r)

                # CBF barrier with inflated radius.
                h_vals.append(dist**2 - r_eff**2)

        if d_vals:
            d_min_raw[k] = float(np.min(d_vals))
        if h_vals:
            h_min[k] = float(np.min(h_vals))

    return d_min_raw, h_min


# ---------------------- Data Helpers ----------------------

def find_finish_index(
    all_t,
    ee_states,
    params=None,
    target_tol=None,
    hold_steps=0,
    min_static_steps=10,
    static_tol=1e-10,
    require_near_final_target=True,
):
    """
    Find the index at which a simulation has effectively finished.

    Primary logic:
        Detect the first index where the end-effector position becomes constant
        for at least `min_static_steps` consecutive steps.

    Optional safeguard:
        If `require_near_final_target=True`, only accept a static segment if the
        end effector is also close to the final waypoint.

    Fallback:
        If no static segment is found, return the last index.

    Parameters
    ----------
    all_t : array-like
        Simulation time vector.

    ee_states : array-like
        End-effector states. First two columns are interpreted as x/y position.

    params : object or None
        Optional parameter object containing `waypoints`.

    target_tol : float or None
        Distance tolerance to the final waypoint. Only used if
        `require_near_final_target=True`.

    hold_steps : int
        Number of additional samples to keep after the detected finish index.
        For your new plateau logic, this can usually be 0.

    min_static_steps : int
        Number of consecutive almost-identical steps required to classify a
        section as repetitive/static.

    static_tol : float
        Tolerance for considering two consecutive end-effector positions equal.
        Use a small positive value instead of exact equality.

    require_near_final_target : bool
        If True, only accept the plateau if the end effector is also close to the
        final waypoint.

    Returns
    -------
    int
        Last index to include.
    """
    all_t = np.asarray(all_t)
    ee_states = np.asarray(ee_states, dtype=float)

    if ee_states.ndim != 2 or ee_states.shape[0] == 0:
        return len(all_t) - 1

    n = min(len(all_t), len(ee_states))
    ee_pos = ee_states[:n, :2]

    if n <= min_static_steps:
        return n - 1

    # Difference between consecutive EE positions.
    step_diff = np.linalg.norm(np.diff(ee_pos, axis=0), axis=1)

    # static_between[k] means:
    # ee_pos[k] and ee_pos[k+1] are effectively identical.
    static_between = step_diff <= static_tol

    # Optional target proximity condition.
    if require_near_final_target:
        if params is None or not hasattr(params, "waypoints") or params.waypoints is None or len(params.waypoints) == 0:
            return n - 1

        if target_tol is None:
            target_tol = getattr(params, "target_tolerance", 0.02)

        final_target = np.asarray(params.waypoints[-1], dtype=float).reshape(2)
        dist_to_goal = np.linalg.norm(ee_pos - final_target[None, :], axis=1)
    else:
        dist_to_goal = None

    # Search for first static run of required length.
    consecutive_static = 0

    for k, is_static in enumerate(static_between):
        if is_static:
            consecutive_static += 1
        else:
            consecutive_static = 0

        if consecutive_static >= min_static_steps:
            # static_between[k - min_static_steps + 1 : k + 1] are static.
            # The first repeated state is at index:
            plateau_start_idx = k - min_static_steps + 1

            if require_near_final_target:
                if dist_to_goal[plateau_start_idx] > target_tol:
                    continue

            finish_idx = plateau_start_idx
            finish_idx = min(finish_idx + hold_steps, n - 1)
            return int(finish_idx)

    return n - 1

def _find_architecture_data_csv(compare_root, architecture):
    """
    Find the detailed *_data.csv file for one architecture in compare mode.

    Supports:
        compare_root/Nominal/*_data.csv
        compare_root/LocalCBF/*_data.csv
        compare_root/MPC-CBF/*_data.csv
        compare_root/Combined/*_data.csv
    """
    compare_root = Path(compare_root)

    candidate_dirs = [
        compare_root / architecture,
    ]

    for folder in candidate_dirs:
        if folder.exists():
            csvs = sorted(folder.glob("*_data.csv"))
            if len(csvs) > 0:
                if len(csvs) > 1:
                    print(f"Warning: multiple *_data.csv files in {folder}. Using {csvs[0].name}")
                return csvs[0]

    raise FileNotFoundError(
        f"Could not find *_data.csv for architecture '{architecture}' below {compare_root}"
    )

def _load_single_architecture_run(compare_root, architecture):
    """
    Load one architecture run from compare-mode output.

    Returns a dictionary with:
        csv_path, data, t, q, dq, ee
    """
    csv_path = _find_architecture_data_csv(compare_root, architecture)
    df = pd.read_csv(csv_path)

    required_cols = [
        "time",
        "theta1_real", "theta2_real", "theta3_real",
        "omega1_real", "omega2_real", "omega3_real",
        "ee_x_real", "ee_y_real",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in {csv_path}")

    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)

    q = df[["theta1_real", "theta2_real", "theta3_real"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    dq = df[["omega1_real", "omega2_real", "omega3_real"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    ee = df[["ee_x_real", "ee_y_real"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(t)
        & np.all(np.isfinite(q), axis=1)
        & np.all(np.isfinite(dq), axis=1)
        & np.all(np.isfinite(ee), axis=1)
    )

    return {
        "architecture": architecture,
        "csv_path": csv_path,
        "data": df.iloc[valid].reset_index(drop=True),
        "t": t[valid],
        "q": q[valid],
        "dq": dq[valid],
        "ee": ee[valid],
    }

# ---------------------- Plotting Helpers ----------------------
################# Obstacle Drawing ####################

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

################# Plot Workspace ####################
def plot_single_workspace_on_axis(
    ax,
    ee_states,
    joint_angles,
    params,
    detailed_data=None,
    style=None,
    xlim=None,
    ylim=None,
):
    """
    Plot single-run workspace trajectory with paper-style obstacles and waypoints.
    """
    if style is None:
        style = get_3dof_plot_style()

    # Obstacles
    scene = getattr(params, "obstacle_scene", None)
    if scene is not None and getattr(scene, "obstacles", None) is not None:
        for i, obs in enumerate(scene.obstacles):
            if getattr(obs, "type", "circle") != "circle":
                continue

            center = np.asarray(obs.center, dtype=float).reshape(2)
            radius = float(obs.radius)

            circ = patches.Circle(
                center,
                radius,
                facecolor=style["obstacle_facecolor"],
                edgecolor=style["obstacle_edgecolor"],
                linewidth=1.5,
                alpha=style["obstacle_alpha"],
                zorder=0,
            )
            ax.add_patch(circ)

            ax.text(
                center[0],
                center[1],
                rf"$\mathcal{{O}}_{i+1}$",
                ha="center",
                va="center",
                fontsize=9,
                color=style["obstacle_text_color"],
                zorder=1,
            )

    # Actual EE trajectory
    ee_states = np.asarray(ee_states, dtype=float)
    ax.plot(
        ee_states[:, 0],
        ee_states[:, 1],
        color=style["trajectory_color"],
        linewidth=1.7,
        label="End effector",
        zorder=3,
    )

    # Desired EE from q_d if available
    # if detailed_data is not None and {"ee_x_des", "ee_y_des"}.issubset(detailed_data.columns):
    #     ax.plot(
    #         detailed_data["ee_x_des"].to_numpy(dtype=float),
    #         detailed_data["ee_y_des"].to_numpy(dtype=float),
    #         color=style["desired_color"],
    #         linestyle=":",
    #         linewidth=1.6,
    #         label="Reference",
    #         zorder=10,
    #     )

    # Optional joint-2 trajectory, useful for link-obstacle interpretation
    joint_angles = np.asarray(joint_angles, dtype=float)
    if len(joint_angles) > 0:
        p2_positions = []
        for q in joint_angles:
            _, _, p2, _ = forward_kinematics_3r(
                q[0], q[1], q[2],
                params.L1, params.L2, params.L3,
            )
            p2_positions.append(p2)

        p2_positions = np.asarray(p2_positions)
        ax.plot(
            p2_positions[:, 0],
            p2_positions[:, 1],
            color=style["trajectory_color"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label="Joint 2",
            zorder=4,
        )

    # Waypoints
    if getattr(params, "waypoints", None) is not None:
        waypoints = np.asarray(params.waypoints, dtype=float)
        ax.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            s=style["waypoint_size"],
            c=style["waypoint_color"],
            marker=style["waypoint_marker"],
            zorder=5,
        )

        for i, (wx, wy) in enumerate(waypoints):
            ax.text(
                wx - 0.02,
                wy + 0.022,
                rf"WP$_{i}$",
                fontsize=8,
                ha="center",
                va="bottom",
                color="black",
                zorder=6,
            )

    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.grid(True, alpha=style["grid_alpha"])
    ax.set_aspect("equal", adjustable="box")

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(-0.1, 1.0)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(-0.5, 0.6)

    ax.set_title("Workspace")
    ax.legend(loc="best", fontsize=7, frameon=True)

################# Minimum Distance to Obstacle ####################
def plot_min_clearance_on_axis(ax, t, d_min_raw, style=None):
    if style is None:
        style = get_3dof_plot_style()

    ax.plot(
        t,
        d_min_raw,
        color=style["distance_color"],
        linewidth=1.3,
        label=r"$d_{\min}$",
    )
    ax.axhline(
        0.0,
        color=style["boundary_color"],
        linestyle=style["boundary_linestyle"],
        linewidth=1.0,
        label="Collision boundary",
    )

    ax.set_xlabel(r"Time $t$ (s)")
    ax.set_ylabel(r"$d_{\min}$ (m)")
    ax.set_title("Minimum clearance")
    ax.grid(True, alpha=style["grid_alpha"])
    ax.legend(loc="best", fontsize=7, frameon=True)

################# Minimum Barrier ####################
def plot_min_barrier_on_axis(ax, t, h_min, style=None):
    if style is None:
        style = get_3dof_plot_style()

    ax.plot(
        t,
        h_min,
        color=style["barrier_color"],
        linewidth=1.3,
        label=r"$h_{\min}$",
    )
    ax.axhline(
        0.0,
        color=style["boundary_color"],
        linestyle=style["boundary_linestyle"],
        linewidth=1.0,
        label="Barrier boundary",
    )

    ax.set_xlabel(r"Time $t$ (s)")
    ax.set_ylabel(r"$h_{\min}$")
    ax.set_title("Minimum barrier value")
    ax.grid(True, alpha=style["grid_alpha"])
    ax.legend(loc="best", fontsize=7, frameon=True)
    
################# Joint angles ####################
def plot_joint_angles_on_axis(ax, t, q, q_des=None, style=None):
    if style is None:
        style = get_3dof_plot_style()

    labels = [r"$q_1$", r"$q_2$", r"$q_3$"]

    for i in range(3):
        ax.plot(
            t[:len(q)],
            q[:, i],
            color=style["joint_colors"][i],
            linewidth=1.2,
            label=labels[i],
        )

        if q_des is not None:
            ax.plot(
                t[:len(q_des)],
                q_des[:, i],
                color=style["joint_colors"][i],
                linestyle="--",
                linewidth=0.9,
                alpha=0.9,
            )

    ax.set_xlabel(r"Time $t$ (s)")
    ax.set_ylabel(r"$q$ (rad)")
    ax.set_title("Joint angles")
    ax.grid(True, alpha=style["grid_alpha"])
    ax.legend(loc="best", fontsize=7, ncol=3, frameon=True)

################# Joint Velocities ####################
def plot_joint_velocities_on_axis(ax, t, dq, dq_des=None, style=None):
    if style is None:
        style = get_3dof_plot_style()

    labels = [r"$\dot q_1$", r"$\dot q_2$", r"$\dot q_3$"]

    for i in range(3):
        ax.plot(
            t[:len(dq)],
            dq[:, i],
            color=style["joint_colors"][i],
            linewidth=1.2,
            label=labels[i],
        )

        if dq_des is not None:
            ax.plot(
                t[:len(dq_des)],
                dq_des[:, i],
                color=style["joint_colors"][i],
                linestyle="--",
                linewidth=0.9,
                alpha=0.9,
            )

    ax.set_xlabel(r"Time $t$ (s)")
    ax.set_ylabel(r"$\dot q$ (rad/s)")
    ax.set_title("Joint velocities")
    ax.grid(True, alpha=style["grid_alpha"])
    ax.legend(loc="best", fontsize=7, ncol=3, frameon=True)

################# Inputs ####################
def plot_inputs_on_axis(ax, t, u, style=None):
    if style is None:
        style = get_3dof_plot_style()

    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)

    n = min(len(t), len(u))
    t = t[:n]
    u = u[:n]

    labels = [r"$u_1$", r"$u_2$", r"$u_3$"]

    for i in range(3):
        ax.plot(
            t[:len(u)],
            u[:, i],
            color=style["joint_colors"][i],
            linewidth=1.2,
            label=labels[i],
        )

    ax.set_xlabel(r"Time $t$ (s)")
    ax.set_ylabel(r"$u [Nm]$")
    ax.set_title("Inputs")
    ax.grid(True, alpha=style["grid_alpha"])
    ax.legend(loc="best", fontsize=7, ncol=3, frameon=True)

################# Compare Plotting ####################
def plot_compare_workspace_on_axis(
    ax,
    runs,
    params,
    style=None,
    xlim=None,
    ylim=None,
):
    """
    Plot all architecture end-effector paths in one workspace axis.
    """
    if style is None:
        style = get_3dof_plot_style()

    # Obstacles
    scene = getattr(params, "obstacle_scene", None)
    if scene is not None and getattr(scene, "obstacles", None) is not None:
        for i, obs in enumerate(scene.obstacles):
            if getattr(obs, "type", "circle") != "circle":
                continue

            center = np.asarray(obs.center, dtype=float).reshape(2)
            radius = float(obs.radius)

            circ = patches.Circle(
                center,
                radius,
                facecolor=style["obstacle_facecolor"],
                edgecolor=style["obstacle_edgecolor"],
                linewidth=1.5,
                alpha=style["obstacle_alpha"],
                zorder=0,
            )
            ax.add_patch(circ)

            ax.text(
                center[0],
                center[1],
                rf"$\mathcal{{O}}_{i+1}$",
                ha="center",
                va="center",
                fontsize=9,
                color=style["obstacle_text_color"],
                zorder=1,
            )

    # Architecture trajectories
    for arch, run in runs.items():
        if arch not in style["arch_styles"]:
            continue

        arch_style = style["arch_styles"][arch]
        ee = run["ee"]

        ax.plot(
            ee[:, 0],
            ee[:, 1],
            color=arch_style["color"],
            linestyle=arch_style["linestyle"],
            linewidth=arch_style["linewidth"],
            label=arch_style["label"],
            zorder=3,
        )

    # Waypoints
    if getattr(params, "waypoints", None) is not None:
        waypoints = np.asarray(params.waypoints, dtype=float)

        ax.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            s=style["waypoint_size"],
            c=style["waypoint_color"],
            marker=style["waypoint_marker"],
            zorder=5,
        )

        for i, (wx, wy) in enumerate(waypoints):
            ax.text(
                wx - 0.02,
                wy + 0.022,
                rf"WP$_{i}$",
                fontsize=8,
                ha="center",
                va="bottom",
                color="black",
                zorder=6,
            )

    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title("Workspace")
    ax.grid(True, alpha=style["grid_alpha"])
    ax.set_aspect("equal", adjustable="box")

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(-0.1, 1.0)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(-0.5, 0.6)

def plot_compare_clearance_on_axis(
    ax,
    runs,
    params,
    style=None,
):
    """
    Plot minimum physical clearance over time for all architectures.
    """
    if style is None:
        style = get_3dof_plot_style()

    for arch, run in runs.items():
        if arch not in style["arch_styles"]:
            continue

        arch_style = style["arch_styles"][arch]

        d_min_raw = run["d_min_raw"] #precomputed

        t = run["t"][:len(d_min_raw)]

        ax.plot(
            t,
            d_min_raw,
            color=arch_style["color"],
            linestyle=arch_style["linestyle"],
            linewidth=arch_style["linewidth"],
            label=arch_style["label"],
        )

    ax.axhline(
        0.0,
        color=style["boundary_color"],
        linestyle=style["boundary_linestyle"],
        linewidth=1.0,
        label="Collision boundary",
    )

    ax.set_xlabel(r"Time $t$ (s)")
    ax.set_ylabel(r"$d_{\min}$ (m)")
    ax.set_title("Minimum clearance")
    ax.grid(True, alpha=style["grid_alpha"])


def plot_compare_barrier_on_axis(
    ax,
    runs,
    params,
    style=None,
):
    """
    Plot minimum barrier value over time for all architectures.
    """
    if style is None:
        style = get_3dof_plot_style()

    for arch, run in runs.items():
        if arch not in style["arch_styles"]:
            continue

        arch_style = style["arch_styles"][arch]
        h_min = run["h_min"] #precomputed

        t = run["t"][:len(h_min)]

        ax.plot(
            t,
            h_min,
            color=arch_style["color"],
            linestyle=arch_style["linestyle"],
            linewidth=arch_style["linewidth"],
            label=arch_style["label"],
        )

    ax.axhline(
        0.0,
        color=style["boundary_color"],
        linestyle=style["boundary_linestyle"],
        linewidth=1.0,
        label="Barrier boundary",
    )

    ax.set_xlabel(r"Time $t$ (s)")
    ax.set_ylabel(r"$h_{\min}$")
    ax.set_title("Minimum barrier value")
    ax.grid(True, alpha=style["grid_alpha"])    


# ---------------------- 3DOF Specialized Plotting ----------------------

def plot_3dof_results(
    all_t,
    ee_states,
    U_alpha,
    pr,
    dpr,
    params,
    joint_angles,
    measurement_noise=None,
    detailed_data=None,
):
    """
    3DOF single-run diagnostic plot.

    Layout:
        - top-left 2x2: workspace
        - top-right: minimum physical clearance
        - middle-right: minimum barrier value
        - bottom-left: joint angles
        - bottom-middle: joint velocities
        - bottom-right: inputs
    """
    style = get_3dof_plot_style()

    all_t = np.asarray(all_t, dtype=float)
    ee_states = np.asarray(ee_states, dtype=float)
    joint_angles = np.asarray(joint_angles, dtype=float)
    U_alpha = np.asarray(U_alpha, dtype=float)

    finish_idx = find_finish_index(
        all_t=all_t,
        ee_states=ee_states,
        params=params,
        target_tol=getattr(params, "target_tol", 0.035),
        hold_steps=0,
    )
    
    # State-like signals have length len(all_t)
    all_t_plot = all_t[:finish_idx + 1]
    ee_states_plot = ee_states[:finish_idx + 1]
    joint_angles_plot = joint_angles[:finish_idx + 1]
    if detailed_data is not None:
        detailed_data_plot = detailed_data.iloc[:finish_idx + 1].copy()
    else:
        detailed_data_plot = None
    if len(U_alpha) == len(all_t) - 1:
    # Input is interval-based: u_k belongs to [t_k, t_{k+1})
        u_end = min(finish_idx, len(U_alpha))
        U_alpha_plot = U_alpha[:u_end]
        t_u = all_t[:u_end]
    else:
        # Input has same sampling convention as all_t, or some other logged length
        u_end = min(finish_idx + 1, len(U_alpha), len(all_t))
        U_alpha_plot = U_alpha[:u_end]
        t_u = all_t[:u_end]


    # ------------------------------------------------------------
    # Extract detailed trajectories if available
    # ------------------------------------------------------------
    q = joint_angles_plot
    dq = None
    q_des = None
    dq_des = None

    if detailed_data_plot is not None:
        if {"theta1_real", "theta2_real", "theta3_real"}.issubset(detailed_data_plot.columns):
            q = detailed_data_plot[["theta1_real", "theta2_real", "theta3_real"]].to_numpy(dtype=float)

        if {"omega1_real", "omega2_real", "omega3_real"}.issubset(detailed_data_plot.columns):
            dq = detailed_data_plot[["omega1_real", "omega2_real", "omega3_real"]].to_numpy(dtype=float)

        if {"theta1_des", "theta2_des", "theta3_des"}.issubset(detailed_data_plot.columns):
            q_des = detailed_data_plot[["theta1_des", "theta2_des", "theta3_des"]].to_numpy(dtype=float)

        if {"omega1_des", "omega2_des", "omega3_des"}.issubset(detailed_data_plot.columns):
            dq_des = detailed_data_plot[["omega1_des", "omega2_des", "omega3_des"]].to_numpy(dtype=float)

    if dq is None:
        dq = np.gradient(q, all_t[:len(q)], axis=0)

    # Align plotting time with detailed data length.
    t_q = all_t[:len(q)]
    t_dq = all_t[:len(dq)]
    t_u = all_t[:len(U_alpha)]

    # ------------------------------------------------------------
    # Safety series
    # ------------------------------------------------------------
    d_min_raw, h_min = compute_distance_and_barrier_series(
        q_traj=q,
        params=params,
        include_base=False,
    )

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(13.0, 9.0), constrained_layout=True)

    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.0, 1.0, 1.05],
        height_ratios=[1.0, 1.0, 0.85],
    )

    ax_ws = fig.add_subplot(gs[0:2, 0:2])
    ax_d = fig.add_subplot(gs[0, 2])
    ax_h = fig.add_subplot(gs[1, 2])
    ax_q = fig.add_subplot(gs[2, 0])
    ax_dq = fig.add_subplot(gs[2, 1])
    ax_u = fig.add_subplot(gs[2, 2])

    # ------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------
    plot_single_workspace_on_axis(
        ax=ax_ws,
        ee_states=ee_states_plot,
        joint_angles=q,
        params=params,
        detailed_data=detailed_data_plot,
        style=style,
    )

    plot_min_clearance_on_axis(
        ax=ax_d,
        t=t_q,
        d_min_raw=d_min_raw,
        style=style,
    )

    plot_min_barrier_on_axis(
        ax=ax_h,
        t=t_q,
        h_min=h_min,
        style=style,
    )

    plot_joint_angles_on_axis(
        ax=ax_q,
        t=t_q,
        q=q,
        q_des=q_des,
        style=style,
    )

    plot_joint_velocities_on_axis(
        ax=ax_dq,
        t=t_dq,
        dq=dq,
        dq_des=dq_des,
        style=style,
    )

    plot_inputs_on_axis(
        ax=ax_u,
        t=t_u,
        u=U_alpha_plot,
        style=style,
    )

    scenario_name = getattr(params, "scenario_name", "3-DOF simulation")
    strategy_name = getattr(params, "strategy_name", "")
    tau = getattr(params, "tau", None)

    if tau is not None:
        title = f"{scenario_name}: {strategy_name} (Delay={tau} steps)"
    else:
        title = f"{scenario_name}: {strategy_name}"

    fig.suptitle(title, fontsize=12)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    plot_path = os.path.join(params.out_dir, params.out_plot)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_3dof_compare_results(
    compare_root,
    params,
    architectures=("Nominal", "LocalCBF", "MPC-CBF", "Combined"),
    output_name="compare_summary.png",
    output_folder=None,
    xlim=None,
    ylim=None,
    show_plot=False,
):
    """
    Plot compare-mode results for all architectures.

    Expected folder structure:
        compare_root/
            Baseline/
                *_data.csv
            LocalCBF/
                *_data.csv
            MPC-CBF/
                *_data.csv
            Combined/
                *_data.csv

    Also supports:
        compare_root/Baseline/Nominal/*_data.csv

    Layout:
        left: workspace
        top-right: minimum physical clearance
        bottom-right: minimum barrier value
    """
    compare_root = Path(compare_root)
    style = get_3dof_plot_style()

    runs = {}

    for arch in architectures:
        try:
            run = _load_single_architecture_run(compare_root, arch)

            d_min_raw, h_min = compute_distance_and_barrier_series(
                q_traj=run["q"],
                params=params,
                include_base=False,
            )

            run["d_min_raw"] = d_min_raw
            run["h_min"] = h_min

            runs[arch] = run

            print(f"Loaded {arch}: {run['csv_path']}")

        except Exception as exc:
            print(f"Skipping {arch}: {exc}")

    if len(runs) == 0:
        raise FileNotFoundError(f"No valid architecture runs found below {compare_root}")

    fig = plt.figure(figsize=(11.0, 4.8), constrained_layout=True)

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.45, 1.0],
        height_ratios=[1.0, 1.0],
    )

    ax_ws = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 1])

    plot_compare_workspace_on_axis(
        ax=ax_ws,
        runs=runs,
        params=params,
        style=style,
        xlim=xlim,
        ylim=ylim,
    )

    plot_compare_clearance_on_axis(
        ax=ax_d,
        runs=runs,
        params=params,
        style=style,
    )

    plot_compare_barrier_on_axis(
        ax=ax_h,
        runs=runs,
        params=params,
        style=style,
    )

    # Use one shared legend for architecture labels.
    legend_handles = []
    legend_labels = []

    for arch in architectures:
        if arch not in runs:
            continue
        if arch not in style["arch_styles"]:
            continue

        arch_style = style["arch_styles"][arch]

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=arch_style["color"],
                linestyle=arch_style["linestyle"],
                linewidth=arch_style["linewidth"],
                label=arch_style["label"],
            )
        )
        legend_labels.append(arch_style["label"])

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=min(len(legend_handles), 4),
        fontsize=9,
        frameon=True,
    )

    scenario_name = getattr(params, "scenario_name", "3-DOF simulation")
    tau = getattr(params, "tau", None)

    if tau is not None:
        title = f"{scenario_name}: architecture comparison (Delay={tau} steps)"
    else:
        title = f"{scenario_name}: architecture comparison"

    fig.suptitle(title, fontsize=12, y=1.10)

    # Remove per-axis legends if you prefer only the shared figure legend.
    for ax in [ax_ws, ax_d, ax_h]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    output_path = compare_root / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved compare plot to: {output_path}")

    if show_plot:
        plt.show(block=True)
    else:
        plt.close(fig)

    return fig, (ax_ws, ax_d, ax_h), runs



def create_3dof_animation(all_t, ee_states, joint_angles, U_alpha, pr, params, detailed_data=None):
    """Create 3DOF robotic arm animation"""
    
    style = get_3dof_animation_style()

    # 1. Convert arrays
    all_t = np.asarray(all_t, dtype=float)
    ee_states = np.asarray(ee_states, dtype=float)
    joint_angles = np.asarray(joint_angles, dtype=float)
    U_alpha = np.asarray(U_alpha, dtype=float)

    # 2. Cut to finish index
    finish_idx = find_finish_index(
        all_t=all_t,
        ee_states=ee_states,
        params=params,
        target_tol=getattr(params, "goal_tolerance", 0.035),
        hold_steps=40,
    )

    all_t = all_t[:finish_idx + 1]
    ee_states = ee_states[:finish_idx + 1]
    joint_angles = joint_angles[:finish_idx + 1]
    pr = pr[:finish_idx + 1]
    #crop input:
    U_alpha = np.asarray(U_alpha, dtype=float)

    u_end = min(len(U_alpha), len(all_t))
    U_plot = U_alpha[:u_end]
    t_u = all_t[:u_end]

    if detailed_data is not None:
        detailed_data = detailed_data.iloc[:finish_idx + 1].copy()

     # 3. Compute safety series
    _, h_min = compute_distance_and_barrier_series(
        q_traj=np.asarray(joint_angles, dtype=float),
        params=params,
        include_base=False,
    )

    # Animation Layout
    fig = plt.figure(figsize=(11.0, 5.5), constrained_layout=False)

    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 0.95, 0.95],
        height_ratios=[1.0, 1.0],
    )

    ax_ws = fig.add_subplot(gs[:, 0:2])
    ax_h = fig.add_subplot(gs[0, 2:4])
    ax_u = fig.add_subplot(gs[1, 2:4])

    fig.suptitle(
        f"3DOF Robot Simulation: {getattr(params, 'strategy_name', '')} Architecture",
        fontsize=12,
    )

    # Workspace view
    ax_ws.set_xlim(-0.2, 0.8)
    ax_ws.set_ylim(-0.4, 0.6)
    ax_ws.set_aspect("equal", adjustable="box")
    ax_ws.grid(True, alpha=style["grid_alpha"])
    ax_ws.set_title("Workspace")
    ax_ws.set_xlabel(r"$x$ (m)")
    ax_ws.set_ylabel(r"$y$ (m)")

    # Draw static obstacles
    if hasattr(params, "obstacle_scene") and params.obstacle_scene is not None:
        for i, obs in enumerate(params.obstacle_scene.obstacles):
            if getattr(obs, "type", "circle") != "circle":
                continue

            center = np.asarray(obs.center, dtype=float).reshape(2)
            radius = float(obs.radius)

            circ = patches.Circle(
                center,
                radius,
                facecolor=style["obstacle_facecolor"],
                edgecolor=style["obstacle_edgecolor"],
                linewidth=1.5,
                alpha=style["obstacle_alpha"],
                zorder=0,
            )
            ax_ws.add_patch(circ)

            ax_ws.text(
                center[0],
                center[1],
                rf"$\mathcal{{O}}_{i+1}$",
                ha="center",
                va="center",
                fontsize=9,
                color=style["obstacle_text_color"],
                zorder=1,
            )

    if getattr(params, "waypoints", None):
        waypoints = np.asarray(params.waypoints, dtype=float)

        ax_ws.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            marker=style["waypoint_marker"],
            s=style["waypoint_size"],
            c=style["waypoint_color"],
            zorder=5,
        )

        for i, (wx, wy) in enumerate(waypoints):
            ax_ws.text(
                wx - 0.02,
                wy + 0.022,
                rf"WP$_{i}$",
                fontsize=8,
                ha="center",
                va="bottom",
                color="black",
                zorder=6,
            )

    # Robot arm components
    link1_line, = ax_ws.plot([], [], color=style["link1_color"], lw=7, solid_capstyle="round", zorder=8)
    link2_line, = ax_ws.plot([], [], color=style["link2_color"], lw=6, solid_capstyle="round", zorder=8)
    link3_line, = ax_ws.plot([], [], color=style["link3_color"], lw=5, solid_capstyle="round", zorder=8)

    base_point, = ax_ws.plot([], [], "o", color=style["joint_color"], ms=6, zorder=9)
    joint1_point, = ax_ws.plot([], [], "o", color=style["joint_color"], ms=5, zorder=9)
    joint2_point, = ax_ws.plot([], [], "o", color=style["joint_color"], ms=5, zorder=9)
    end_effector, = ax_ws.plot([], [], "o", color=style["ee_color"], ms=7, zorder=10)

    actual_trail, = ax_ws.plot(
        [],
        [],
        color=style["trail_color"],
        lw=1.4,
        alpha=0.85,
        zorder = 4
    )    

    # Barrier Panel
    ax_h.set_title("Minimum barrier value")
    ax_h.set_xlabel(r"Time $t$ (s)")
    ax_h.set_ylabel(r"$h_{\min}$")
    ax_h.grid(True, alpha=style["grid_alpha"])

    ax_h.axhline(
        0.0,
        color=style["boundary_color"],
        linestyle=style["boundary_linestyle"],
        linewidth=1.0,
    )

    barrier_line, = ax_h.plot(
        [],
        [],
        color=style["barrier_color"],
        linewidth=1.5,
    )

    barrier_marker, = ax_h.plot(
        [],
        [],
        marker="o",
        color=style["barrier_color"],
        markersize=5,
    )

    ax_h.set_xlim(all_t[0], all_t[-1])

    h_finite = h_min[np.isfinite(h_min)]
    if h_finite.size > 0:
        ymin = min(0.0, np.nanmin(h_finite))
        ymax = max(0.0, np.nanmax(h_finite))
        pad = 0.08 * max(ymax - ymin, 1e-6)
        ax_h.set_ylim(ymin - pad, ymax + pad)
    else:
        ax_h.set_ylim(-1.0, 1.0)

    #Input Panel
    ax_u.set_title("Inputs")
    ax_u.set_xlabel(r"Time $t$ (s)")
    ax_u.set_ylabel(r"$u$ [Nm]")
    ax_u.grid(True, alpha=style["grid_alpha"])

    input_lines = []
    input_markers = []

    labels = [r"$u_1$", r"$u_2$", r"$u_3$"]

    for i in range(3):
        line, = ax_u.plot(
            [],
            [],
            color=style["input_colors"][i],
            linewidth=1.2,
            label=labels[i],
        )
        marker, = ax_u.plot(
            [],
            [],
            marker="o",
            color=style["input_colors"][i],
            markersize=4,
        )
        input_lines.append(line)
        input_markers.append(marker)

    ax_u.set_xlim(all_t[0], all_t[-1])

    if U_plot.size > 0:
        u_min = float(np.nanmin(U_plot))
        u_max = float(np.nanmax(U_plot))
        pad = 0.08 * max(u_max - u_min, 1e-6)
        ax_u.set_ylim(u_min - pad, u_max + pad)
    else:
        ax_u.set_ylim(-1.0, 1.0)

    ax_u.legend(loc="best", fontsize=8, ncol=3, frameon=True)

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.11,
        top=0.88,
        wspace=0.45,
        hspace=0.45,
    )

    # Precompute Robot Kinematics
    base_all = np.zeros((len(joint_angles), 2))
    p1_all = np.zeros((len(joint_angles), 2))
    p2_all = np.zeros((len(joint_angles), 2))
    ee_all = np.zeros((len(joint_angles), 2))

    for k, q in enumerate(joint_angles):
        base, p1, p2, ee = forward_kinematics_3r(
            q[0], q[1], q[2],
            params.L1, params.L2, params.L3,
        )
        base_all[k] = base
        p1_all[k] = p1
        p2_all[k] = p2
        ee_all[k] = ee
    ##################### Update Animation #################################
    def update(frame):
        q = joint_angles[frame]
        # base, p1, p2, ee = forward_kinematics_3r(
        #     q[0], q[1], q[2],
        #     params.L1, params.L2, params.L3,
        # )
        base = base_all[frame]
        p1 = p1_all[frame]
        p2 = p2_all[frame]
        ee = ee_all[frame]

        # Robot links
        link1_line.set_data([base[0], p1[0]], [base[1], p1[1]])
        link2_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        link3_line.set_data([p2[0], ee[0]], [p2[1], ee[1]])

        base_point.set_data([base[0]], [base[1]])
        joint1_point.set_data([p1[0]], [p1[1]])
        joint2_point.set_data([p2[0]], [p2[1]])
        end_effector.set_data([ee[0]], [ee[1]])

        # EE trail
        actual_trail.set_data(
            ee_states[:frame + 1, 0],
            ee_states[:frame + 1, 1],
        )

        # Barrier trace
        barrier_line.set_data(
            all_t[:frame + 1],
            h_min[:frame + 1],
        )

        if np.isfinite(h_min[frame]):
            barrier_marker.set_data([all_t[frame]], [h_min[frame]])
        else:
            barrier_marker.set_data([], [])

        # Input traces
        u_frame = min(frame, len(U_plot) - 1)

        if len(U_plot) > 0:
            for i in range(3):
                input_lines[i].set_data(
                    t_u[:u_frame + 1],
                    U_plot[:u_frame + 1, i],
                )

                input_markers[i].set_data(
                    [t_u[u_frame]],
                    [U_plot[u_frame, i]],
                )

        artists = [
            link1_line,
            link2_line,
            link3_line,
            base_point,
            joint1_point,
            joint2_point,
            end_effector,
            actual_trail,
            barrier_line,
            barrier_marker,
        ]

        artists.extend(input_lines)
        artists.extend(input_markers)

        return artists

    fps = 30    
    pause_seconds = 1

    duration_s = float(all_t[-1] - all_t[0])
    n_anim_frames = max(2, int(duration_s * fps))

    frame_indices = np.linspace(
        0,
        len(all_t) - 1,
        n_anim_frames,
        dtype=int,
    ).tolist()

    pause_frames = int(fps * pause_seconds)
    frames = frame_indices + [len(all_t) - 1] * pause_frames

    #create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=True,
        repeat=True,
    )

    # Save animation
    anim_path = os.path.join(params.out_dir, params.out_anim)
    anim.save(anim_path, writer='pillow', fps=30, dpi = 120)
    print(f"3DOF animation saved: {anim_path}")
    plt.close(fig)

    return anim

def generate_visualizations(
    t_result: np.ndarray,
    ee_states: np.ndarray,
    U_alpha: np.ndarray,
    pr: np.ndarray,
    dpr: np.ndarray,
    params: SystemParams,
    joint_angles: np.ndarray,
    measurement_noise: np.ndarray,
    detailed_data=None,
) -> None:
    #print("Generating visualization...")
    try:
        plot_3dof_results(
            t_result,
            ee_states,
            U_alpha,
            pr,
            dpr,
            params,
            joint_angles,
            measurement_noise=measurement_noise,
            detailed_data=detailed_data,
        )
        #print(f"  Plot saved: {params.out_plot}")
        
        # Generate animation if out_anim is specified (not None)
        if params.out_anim:
            create_3dof_animation(t_result, ee_states, joint_angles, U_alpha, pr, params, detailed_data=detailed_data)
            print(f"  Animation saved: {params.out_anim}")
    except Exception as exc:
        print(f"  Visualization error: {exc}")