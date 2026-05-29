from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Polygon as MplPolygon
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull


# ============================================================
# Helpers
# ============================================================

def _decode_meta_object(obj: Any) -> Any:
    """
    Decode custom JSON-serialized tuples/lists/dicts from meta_data.json.
    """
    if isinstance(obj, dict):
        if "__type__" in obj:
            t = obj["__type__"]
            if t == "tuple":
                return tuple(_decode_meta_object(v) for v in obj["data"])
            elif t == "list":
                return [_decode_meta_object(v) for v in obj["data"]]
            elif t == "dict":
                return {k: _decode_meta_object(v) for k, v in obj["data"].items()}
        return {k: _decode_meta_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decode_meta_object(v) for v in obj]
    else:
        return obj


def _load_meta_data(root: Path) -> Dict[str, Any]:
    meta_path = root / "meta_data.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find meta_data.json in {root}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_raw = json.load(f)

    meta = _decode_meta_object(meta_raw)
    if "params" not in meta:
        raise ValueError("meta_data.json does not contain a top-level 'params' entry.")

    return meta["params"]


def _load_ee_path(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load time, ee_x_real, ee_y_real from a *_data.csv file.
    """
    df = pd.read_csv(csv_path)

    required_cols = ["time", "ee_x_real", "ee_y_real"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in {csv_path}")

    t = df["time"].to_numpy(dtype=float)
    x = df["ee_x_real"].to_numpy(dtype=float)
    y = df["ee_y_real"].to_numpy(dtype=float)

    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    return t[valid], x[valid], y[valid]


def _find_single_csv(folder: Path) -> Path:
    csvs = sorted(folder.glob("*_data.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No *_data.csv found in {folder}")
    if len(csvs) > 1:
        # Usually there is only one. If more exist, take the first but warn.
        print(f"Warning: multiple *_data.csv files found in {folder}. Using {csvs[0].name}")
    return csvs[0]


def _load_baseline_nominal(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_folder = root / "Baseline" / "Nominal"
    csv_path = _find_single_csv(baseline_folder)
    return _load_ee_path(csv_path)


def _load_architecture_trial_paths(
    root: Path,
    architecture: str,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load all trial paths for one architecture, e.g. 'LocalCBF' or 'RemoteMPC-CBF'.
    """
    trial_dirs = sorted([p for p in root.glob("trial_*") if p.is_dir()])
    if len(trial_dirs) == 0:
        raise FileNotFoundError(f"No trial_* directories found in {root}")

    paths = []
    for trial_dir in trial_dirs:
        arch_dir = trial_dir / architecture
        if not arch_dir.exists():
            continue

        csv_path = _find_single_csv(arch_dir)
        paths.append(_load_ee_path(csv_path))

    if len(paths) == 0:
        raise FileNotFoundError(f"No trial paths found for architecture '{architecture}'")

    return paths



def _compute_mean_path(xy_runs: np.ndarray) -> np.ndarray:
    """
    xy_runs: shape (n_runs, n_time, 2)
    Returns mean path: shape (n_time, 2)
    """
    return np.mean(xy_runs, axis=0)



def _set_square_limits(ax, all_x, all_y, pad_frac=0.05):
    """
    Set square axis limits around all data.
    """
    xmin, xmax = np.min(all_x), np.max(all_x)
    ymin, ymax = np.min(all_y), np.max(all_y)

    xspan = xmax - xmin
    yspan = ymax - ymin
    span = max(xspan, yspan)
    span = max(span, 1e-6)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    half = 0.5 * span * (1.0 + 2.0 * pad_frac)

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")

def _add_top_legend_and_adjust(
    fig,
    ax,
    handles,
    fontsize=9,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    frameon=True,
    pad=0.015,
    ):
    """
    Add a figure-level legend at the top and automatically shrink the axes
    so the legend does not overlap / dominate the plot.
    """
    legend = fig.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        fontsize=fontsize,
        frameon=frameon,
    )

    # Need a renderer before we can measure the legend
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Legend bbox in figure coordinates
    bbox_fig = legend.get_window_extent(renderer=renderer).transformed(
        fig.transFigure.inverted()
    )

    # Reserve everything below the legend
    top_of_axes = max(0.2, bbox_fig.y0 - pad)

    fig.subplots_adjust(top=top_of_axes)

    return legend

def _pad_paths_to_equal_length(
    paths: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """
        Convert a list of paths [(t,x,y), ...] into an array of shape
        (n_runs, max_len, 2), padding shorter runs with their final point.
        """
        max_len = max(len(x) for _, x, _ in paths)
        n_runs = len(paths)

        xy_runs = np.zeros((n_runs, max_len, 2), dtype=float)

        for i, (_, x, y) in enumerate(paths):
            n = len(x)
            xy_runs[i, :n, 0] = x
            xy_runs[i, :n, 1] = y

            if n < max_len:
                xy_runs[i, n:, 0] = x[-1]
                xy_runs[i, n:, 1] = y[-1]

        return xy_runs

def _compute_xwise_envelope(
    xy_runs: np.ndarray,
    n_bins: int = 300,
    min_count_per_bin: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute an x-wise envelope over all trajectory samples.

    Parameters
    ----------
    xy_runs : array, shape (n_runs, n_steps, 2)
        Trajectory data.
    n_bins : int
        Number of x bins for the envelope.
    min_count_per_bin : int
        Minimum number of points required in a bin.

    Returns
    -------
    x_grid, y_min, y_max
    """
    pts = xy_runs.reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]

    x_all = pts[:, 0]
    y_all = pts[:, 1]

    x_min = np.min(x_all)
    x_max = np.max(x_all)

    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    x_grid = 0.5 * (x_edges[:-1] + x_edges[1:])

    y_min = np.full(n_bins, np.nan)
    y_max = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (x_all >= x_edges[i]) & (x_all < x_edges[i + 1])
        if np.sum(mask) >= min_count_per_bin:
            y_min[i] = np.min(y_all[mask])
            y_max[i] = np.max(y_all[mask])

    # Fill missing bins by interpolation
    valid = np.isfinite(y_min) & np.isfinite(y_max)
    if np.sum(valid) >= 2:
        y_min = np.interp(x_grid, x_grid[valid], y_min[valid])
        y_max = np.interp(x_grid, x_grid[valid], y_max[valid])

    return x_grid, y_min, y_max

def _plot_xwise_envelope(
    ax,
    x_grid: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
    color: str,
    alpha_fill: float = 0.18,
    zorder: int = 1,
    edge_linewidth: float = 0.8,
    edge_alpha: float = 0.65,
):
    """
    Plot x-wise envelope with a transparent fill and thin upper/lower boundary lines.
    """
    mask = np.isfinite(x_grid) & np.isfinite(y_min) & np.isfinite(y_max)

    xg = x_grid[mask]
    ymin = y_min[mask]
    ymax = y_max[mask]

    ax.fill_between(
        xg,
        ymin,
        ymax,
        color=color,
        alpha=alpha_fill,
        linewidth=0.0,
        zorder=zorder,
    )

    # Thin envelope boundary lines
    if xg.size >= 2 and edge_linewidth > 0.0:
        ax.plot(
            xg,
            ymin,
            color=color,
            linewidth=edge_linewidth,
            alpha=edge_alpha,
            zorder=zorder + 0.2,
        )
        ax.plot(
            xg,
            ymax,
            color=color,
            linewidth=edge_linewidth,
            alpha=edge_alpha,
            zorder=zorder + 0.2,
        )

def _resample_path_by_arclength(
    x: np.ndarray,
    y: np.ndarray,
    n_points: int = 500,
) -> np.ndarray:
    """
    Resample one 2D path by normalized cumulative arc length.

    Returns
    -------
    xy_resampled:
        shape (n_points, 2)
    """
    xy = np.column_stack([x, y])
    valid = np.all(np.isfinite(xy), axis=1)
    xy = xy[valid]

    if xy.shape[0] < 2:
        return np.repeat(xy[:1], n_points, axis=0)

    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])

    # Remove duplicate arc-length entries caused by repeated final states.
    unique = np.concatenate([[True], np.diff(s) > 1e-12])
    s = s[unique]
    xy = xy[unique]

    if s[-1] <= 1e-12:
        return np.repeat(xy[:1], n_points, axis=0)

    s = s / s[-1]
    s_ref = np.linspace(0.0, 1.0, n_points)

    x_ref = np.interp(s_ref, s, xy[:, 0])
    y_ref = np.interp(s_ref, s, xy[:, 1])

    return np.column_stack([x_ref, y_ref])


def _resample_paths_by_arclength(
    paths: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_points: int = 500,
) -> np.ndarray:
    """
    Resample all paths by normalized arc length.

    Returns
    -------
    xy_runs:
        shape (n_runs, n_points, 2)
    """
    xy_runs = []

    for _, x, y in paths:
        xy_runs.append(_resample_path_by_arclength(x, y, n_points=n_points))

    return np.stack(xy_runs, axis=0)

# ============================================================
# Main plot function
# ============================================================

def plot_workspace(
    folder_path: str | Path,
    output_name: str = "statespace_plot",
    architectures: Tuple[str, str] = ("LocalCBF", "RemoteMPC-CBF"),
    save_pdf: bool = True,
    save_png: bool = True,
    save_svg: bool = False,
    dpi: int = 600,
    figsize: Tuple[float, float] = (4.0, 4.0),
    envelope_fill_alpha: float = 0.18,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    square_limits: bool = True,
    axis_fontsize: float = 12,
    tick_fontsize: float = 10,
    legend_fontsize: float = 10,
    waypoint_fontsize: float = 10,
    obstacle_label_fontsize: float = 10,
    title_fontsize: float = 12,
    show_title: bool = False,
    title: str | None = None,
    show_plot: bool = False,
    envelope_n_bins: int = 350,
    mean_n_points: int = 500,
):
    """
    Create a paper-style state-space plot:
      - obstacles + waypoints
      - nominal baseline as thin black line
      - envelope of all MC runs for each architecture
      - mean path for each architecture

    Expected structure in folder_path:
      meta_data.json
      Baseline/Nominal/*_data.csv
      trial_*/LocalCBF/*_data.csv
      trial_*/RemoteMPC-CBF/*_data.csv
    """

    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    # ---------------------------
    # Adjustable style dictionary
    # ---------------------------
    style = _get_workspace_style()

    # ---------------------------
    # Read metadata / scene
    # ---------------------------
    params = _load_meta_data(root)

    scenario_name = params.get("scenario_name", "Scenario")
    waypoints = params.get("waypoints", [])
    obstacle_centers = params.get("scenario_obstacle_centers", [])
    obstacle_radii = params.get("obstacle_default_radius", [])

    if len(waypoints) == 0:
        raise ValueError("No waypoints found in meta_data.json")
    if len(obstacle_centers) != len(obstacle_radii):
        raise ValueError("Obstacle centers and radii have inconsistent lengths")

    waypoints = [tuple(w) for w in waypoints]
    obstacle_centers = [tuple(c) for c in obstacle_centers]
    obstacle_radii = list(obstacle_radii)

    # ---------------------------
    # Read baseline nominal path
    # ---------------------------
    t_base, x_base, y_base = _load_baseline_nominal(root)

    # ---------------------------
    # Read MC paths + compute means / envelopes
    # ---------------------------
    arch_results = {}

    for arch in architectures:
        paths = _load_architecture_trial_paths(root, arch)
        xy_runs_space = _resample_paths_by_arclength(
            paths,
            n_points=mean_n_points,
        )

        mean_xy = _compute_mean_path(xy_runs_space)

        xy_runs_env = _pad_paths_to_equal_length(paths)
        x_env, y_env_min, y_env_max = _compute_xwise_envelope(
            xy_runs_env,
            n_bins=envelope_n_bins,
        )

        arch_results[arch] = {
            "paths": paths,
            "xy_runs": xy_runs_space,
            "mean_xy": mean_xy,
            "x_env": x_env,
            "y_env_min": y_env_min,
            "y_env_max": y_env_max,
        }

        print(f"{arch}: {len(paths)} runs loaded.")

    # ---------------------------
    # Build plot
    # ---------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Obstacles

    for i, (center, radius) in enumerate(zip(obstacle_centers, obstacle_radii)):
        circle = Circle(
            center,
            radius,
            facecolor=style["obstacle_facecolor"],
            edgecolor=style["obstacle_edgecolor"],
            linewidth=1.5,
            alpha=style["obstacle_alpha"],
            zorder=0,
        )
        ax.add_patch(circle)

        ax.text(
            center[0],
            center[1],
            rf"$\mathcal{{O}}_{i+1}$",
            ha="center",
            va="center",
            fontsize=obstacle_label_fontsize,
            color=style.get("obstacle_text_color", "#7A0000"),
            zorder=1,
        )
    
    # Waypoints
    wp_x = [w[0] for w in waypoints]
    wp_y = [w[1] for w in waypoints]
    ax.scatter(
        wp_x,
        wp_y,
        s=style["waypoint_size"],
        c=style["waypoint_color"],
        marker=style["waypoint_marker"],
        zorder=5,
    )

    for i, (wx, wy) in enumerate(waypoints):
        ax.text(
            wx,
            wy + 0.018,
            rf"WP$_{i}$",
            fontsize=waypoint_fontsize,
            ha="center",
            va="bottom",
            color="black",
            zorder=6,
        )

    # Baseline
    ax.plot(
        x_base,
        y_base,
        color=style["baseline_color"],
        linestyle=style["baseline_linestyle"],
        linewidth=style["baseline_linewidth"],
        zorder=2,
    )
    # Envelopes + mean lines
    for arch in architectures:
        arch_style = style["arch_styles"][arch]
        mean_xy = arch_results[arch]["mean_xy"]

        _plot_xwise_envelope(
            ax,
            arch_results[arch]["x_env"],
            arch_results[arch]["y_env_min"],
            arch_results[arch]["y_env_max"],
            color=arch_style["color"],
            alpha_fill=envelope_fill_alpha,
            zorder=1,
        )
        #plot means
        ax.plot(
            mean_xy[:, 0],
            mean_xy[:, 1],
            color=arch_style["color"],
            linestyle=arch_style["linestyle"],
            linewidth=arch_style["linewidth"],
            zorder=4,
        )

    # Labels
    ax.set_xlabel(r"$x$ (m)", fontsize=axis_fontsize)
    ax.set_ylabel(r"$y$ (m)", fontsize=axis_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)



    if show_title:
        if title is None:
            title = f"{scenario_name}: Mean path and Monte Carlo envelope"
        ax.set_title(title, fontsize=title_fontsize)

    # Limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if xlim is None or ylim is None:
        # Collect all geometry for automatic limits
        xs = list(x_base) + wp_x
        ys = list(y_base) + wp_y

        for center, radius in zip(obstacle_centers, obstacle_radii):
            xs.extend([center[0] - radius, center[0] + radius])
            ys.extend([center[1] - radius, center[1] + radius])

        for arch in architectures:
            mean_xy = arch_results[arch]["mean_xy"]
            xs.extend(mean_xy[:, 0].tolist())
            ys.extend(mean_xy[:, 1].tolist())

            xy_runs = arch_results[arch]["xy_runs"].reshape(-1, 2)
            xs.extend(xy_runs[:, 0].tolist())
            ys.extend(xy_runs[:, 1].tolist())

        if square_limits:
            _set_square_limits(ax, np.asarray(xs), np.asarray(ys), pad_frac=0.06)

    ax.grid(True, alpha=0.3)

    # Legend
    legend_handles = [
    Line2D([0], [0], color="black", linewidth=1.2, linestyle="-", label="Nominal"),
    Line2D([0], [0], color=style["arch_styles"]["LocalCBF"]["color"],
           linewidth=2.2, linestyle="-", label="Local CBF"),
    Line2D([0], [0], color=style["arch_styles"]["RemoteMPC-CBF"]["color"],
           linewidth=2.2, linestyle="--", label="Remote MPC-CBF"),
    ]

    _add_top_legend_and_adjust(
        fig,
        ax,
        legend_handles,
        fontsize=legend_fontsize,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=True,
        pad=0.015,
    )

    # ---------------------------
    # Save figure
    # ---------------------------
    out_base = root / output_name

    if save_pdf:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")

    print(f"Saved plot to:")
    if save_pdf:
        print(f"  {out_base.with_suffix('.pdf')}")
    if save_svg:
        print(f"  {out_base.with_suffix('.svg')}")
    if save_png:
        print(f"  {out_base.with_suffix('.png')}")


    if show_plot:
        plt.show(block=True)
    return fig, ax

# ============================================================
# Extended plotting API: single-axis, single-figure, and double-figure plots
# ============================================================

def _get_workspace_style() -> Dict[str, Any]:
    """Central style dictionary for workspace plots."""
    return {
        "obstacle_facecolor": "#D9A5A5",
        "obstacle_edgecolor": "#B15C5C",
        "obstacle_text_color": "#8B0000",
        "obstacle_alpha": 0.55,
        "waypoint_marker": "D",
        "waypoint_color": "black",
        "waypoint_size": 46,
        "baseline_color": "black",
        "baseline_linestyle": "-",
        "baseline_linewidth": 1.1,
        # Okabe-Ito colorblind-safe palette
        "arch_styles": {
            "LocalCBF": {
                "label": "Local CBF",
                "color": "#0072B2",   # blue
                "linestyle": "-",
                "linewidth": 1.7,
            },
            "RemoteMPC-CBF": {
                "label": "Remote MPC-CBF",
                "color": "#E69F00",   # orange
                "linestyle": "--",
                "linewidth": 1.7,
            },
        },
    }


def _workspace_legend_handles(style: Dict[str, Any]) -> List[Line2D]:
    """Create common legend handles for workspace plots."""
    return [
        Line2D(
            [0], [0],
            color=style["baseline_color"],
            linewidth=1.2,
            linestyle=style["baseline_linestyle"],
            label="Nominal",
        ),
        Line2D(
            [0], [0],
            color=style["arch_styles"]["LocalCBF"]["color"],
            linewidth=2.2,
            linestyle=style["arch_styles"]["LocalCBF"]["linestyle"],
            label="Local CBF",
        ),
        Line2D(
            [0], [0],
            color=style["arch_styles"]["RemoteMPC-CBF"]["color"],
            linewidth=2.2,
            linestyle=style["arch_styles"]["RemoteMPC-CBF"]["linestyle"],
            label="Remote MPC-CBF",
        ),
    ]


def plot_workspace_on_axis(
    ax,
    folder_path: str | Path,
    architectures: Tuple[str, str] = ("LocalCBF", "RemoteMPC-CBF"),
    envelope_fill_alpha: float = 0.18,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    square_limits: bool = True,
    axis_fontsize: float = 12,
    tick_fontsize: float = 10,
    waypoint_fontsize: float = 10,
    obstacle_label_fontsize: float = 10,
    title: str | None = None,
    title_fontsize: float = 12,
    show_ylabel: bool = True,
    show_yticklabels: bool = True,
    envelope_n_bins: int = 350,
    mean_n_points: int = 500,
) -> Dict[str, Any]:
    """
    Plot one Monte Carlo workspace dataset into an existing Matplotlib axis.

    This is the reusable core used by both `plot_workspace` and
    `plot_workspace_comparison`.
    """
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    style = _get_workspace_style()

    params = _load_meta_data(root)
    scenario_name = params.get("scenario_name", "Scenario")
    waypoints = [tuple(w) for w in params.get("waypoints", [])]
    obstacle_centers = [tuple(c) for c in params.get("scenario_obstacle_centers", [])]
    obstacle_radii = list(params.get("obstacle_default_radius", []))

    if len(waypoints) == 0:
        raise ValueError(f"No waypoints found in {root / 'meta_data.json'}")
    if len(obstacle_centers) != len(obstacle_radii):
        raise ValueError("Obstacle centers and radii have inconsistent lengths")

    # Baseline nominal path
    _, x_base, y_base = _load_baseline_nominal(root)

    # Architecture data: arc-length mean, x-wise envelope
    arch_results = {}
    for arch in architectures:
        paths = _load_architecture_trial_paths(root, arch)

        xy_runs_space = _resample_paths_by_arclength(paths, n_points=mean_n_points)
        mean_xy = _compute_mean_path(xy_runs_space)

        # Envelope from equal-length raw trajectories. This preserves the
        # time-sampled spatial occupancy better than using only resampled paths.
        xy_runs_env = _pad_paths_to_equal_length(paths)
        x_env, y_env_min, y_env_max = _compute_xwise_envelope(
            xy_runs_env,
            n_bins=envelope_n_bins,
        )

        arch_results[arch] = {
            "paths": paths,
            "xy_runs_space": xy_runs_space,
            "xy_runs_env": xy_runs_env,
            "mean_xy": mean_xy,
            "x_env": x_env,
            "y_env_min": y_env_min,
            "y_env_max": y_env_max,
        }

        print(f"{root.name} / {arch}: {len(paths)} runs loaded.")

    # Obstacles
    for i, (center, radius) in enumerate(zip(obstacle_centers, obstacle_radii)):
        circle = Circle(
            center,
            radius,
            facecolor=style["obstacle_facecolor"],
            edgecolor=style["obstacle_edgecolor"],
            linewidth=1.5,
            alpha=style["obstacle_alpha"],
            zorder=0,
        )
        ax.add_patch(circle)
        ax.text(
            center[0], center[1],
            rf"$\mathcal{{O}}_{i+1}$",
            ha="center",
            va="center",
            fontsize=obstacle_label_fontsize,
            color=style["obstacle_text_color"],
            zorder=1,
        )

    # Waypoints
    wp_x = [w[0] for w in waypoints]
    wp_y = [w[1] for w in waypoints]
    ax.scatter(
        wp_x,
        wp_y,
        s=style["waypoint_size"],
        c=style["waypoint_color"],
        marker=style["waypoint_marker"],
        zorder=5,
    )

    waypoint_label_dx = -0.02  # negative = move left
    waypoint_label_dy = 0.022
    for i, (wx, wy) in enumerate(waypoints):
        ax.text(
            wx+waypoint_label_dx,
            wy + waypoint_label_dy,
            rf"WP$_{i}$",
            fontsize=waypoint_fontsize,
            ha="center",
            va="bottom",
            color="black",
            zorder=6,
        )

    # Nominal baseline
    ax.plot(
        x_base,
        y_base,
        color=style["baseline_color"],
        linestyle=style["baseline_linestyle"],
        linewidth=style["baseline_linewidth"],
        zorder=2,
    )

    # Envelopes and mean lines
    for arch in architectures:
        arch_style = style["arch_styles"][arch]
        res = arch_results[arch]

        _plot_xwise_envelope(
            ax,
            res["x_env"],
            res["y_env_min"],
            res["y_env_max"],
            color=arch_style["color"],
            alpha_fill=envelope_fill_alpha,
            zorder=1,
        )

        mean_xy = res["mean_xy"]
        ax.plot(
            mean_xy[:, 0],
            mean_xy[:, 1],
            color=arch_style["color"],
            linestyle=arch_style["linestyle"],
            linewidth=arch_style["linewidth"],
            zorder=4,
        )

    # Axes formatting
    ax.set_xlabel(r"$x$ (m)", fontsize=axis_fontsize)
    if show_ylabel:
        ax.set_ylabel(r"$y$ (m)", fontsize=axis_fontsize)
    else:
        ax.set_ylabel("")
    if not show_yticklabels:
        ax.tick_params(axis="y", labelleft=False)

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    # Axis limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if square_limits and (xlim is None or ylim is None):
        xs = list(x_base) + wp_x
        ys = list(y_base) + wp_y

        for center, radius in zip(obstacle_centers, obstacle_radii):
            xs.extend([center[0] - radius, center[0] + radius])
            ys.extend([center[1] - radius, center[1] + radius])

        for arch in architectures:
            mean_xy = arch_results[arch]["mean_xy"]
            xs.extend(mean_xy[:, 0].tolist())
            ys.extend(mean_xy[:, 1].tolist())
            xy_runs_env = arch_results[arch]["xy_runs_env"].reshape(-1, 2)
            xs.extend(xy_runs_env[:, 0].tolist())
            ys.extend(xy_runs_env[:, 1].tolist())

        _set_square_limits(ax, np.asarray(xs), np.asarray(ys), pad_frac=0.06)

    return style


# This definition intentionally overrides the earlier plot_workspace definition.
def plot_workspace(
    folder_path: str | Path,
    output_name: str = "statespace_plot",
    output_folder: str | Path | None = None,
    architectures: Tuple[str, str] = ("LocalCBF", "RemoteMPC-CBF"),
    save_pdf: bool = True,
    save_png: bool = True,
    save_svg: bool = False,
    dpi: int = 600,
    figsize: Tuple[float, float] = (4.0, 4.0),
    envelope_fill_alpha: float = 0.18,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    square_limits: bool = True,
    axis_fontsize: float = 12,
    tick_fontsize: float = 10,
    legend_fontsize: float = 10,
    waypoint_fontsize: float = 10,
    obstacle_label_fontsize: float = 10,
    title_fontsize: float = 12,
    legend_ncol: int = 3,
    show_title: bool = False,
    title: str | None = None,
    show_plot: bool = False,
    envelope_n_bins: int = 350,
    mean_n_points: int = 500,
):
    """
    Create a single paper-style workspace plot and save it to either
    `output_folder` or, by default, the Monte Carlo root folder.
    """
    root = Path(folder_path)
    fig, ax = plt.subplots(figsize=figsize)

    axis_title = title if show_title else None
    style = plot_workspace_on_axis(
        ax=ax,
        folder_path=root,
        architectures=architectures,
        envelope_fill_alpha=envelope_fill_alpha,
        xlim=xlim,
        ylim=ylim,
        square_limits=square_limits,
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
        waypoint_fontsize=waypoint_fontsize,
        obstacle_label_fontsize=obstacle_label_fontsize,
        title=axis_title,
        title_fontsize=title_fontsize,
        show_ylabel=True,
        show_yticklabels=True,
        envelope_n_bins=envelope_n_bins,
        mean_n_points=mean_n_points,
    )

    legend_handles = _workspace_legend_handles(style)
    _add_top_legend_and_adjust(
        fig,
        ax,
        legend_handles,
        fontsize=legend_fontsize,
        ncol=legend_ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=True,
        pad=0.015,
    )

    out_dir = Path(output_folder) if output_folder is not None else root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / output_name

    if save_pdf:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")

    print("Saved plot to:")
    if save_pdf:
        print(f"  {out_base.with_suffix('.pdf')}")
    if save_svg:
        print(f"  {out_base.with_suffix('.svg')}")
    if save_png:
        print(f"  {out_base.with_suffix('.png')}")

    if show_plot:
        plt.show(block=True)
    else:
        plt.close(fig)

    return fig, ax


def plot_workspace_comparison(
    folder_paths: Tuple[str | Path, str | Path],
    panel_titles: Tuple[str, str] = ("Low disturbance", "High disturbance"),
    output_name: str = "statespace_comparison",
    output_folder: str | Path | None = None,
    architectures: Tuple[str, str] = ("LocalCBF", "RemoteMPC-CBF"),
    save_pdf: bool = True,
    save_png: bool = True,
    save_svg: bool = False,
    dpi: int = 600,
    figsize: Tuple[float, float] = (7.2, 3.4),
    envelope_fill_alpha: float = 0.18,
    xlim: Tuple[float, float] | None = (0.1, 0.7),
    ylim: Tuple[float, float] | None = (-0.15, 0.4),
    axis_fontsize: float = 10,
    tick_fontsize: float = 9,
    legend_fontsize: float = 10,
    waypoint_fontsize: float = 9,
    obstacle_label_fontsize: float = 10,
    title_fontsize: float = 10,
    legend_ncol: int = 3,
    envelope_n_bins: int = 350,
    mean_n_points: int = 500,
    show_plot: bool = False,
    share_axes: bool = True,
    wspace: float = 0.04,
):
    """
    Create a two-panel workspace comparison with a single shared legend.

    Parameters
    ----------
    folder_paths:
        Tuple of two Monte Carlo root folders.
    panel_titles:
        Titles for left and right panels, e.g. ("Low disturbance", "High disturbance").
    output_folder:
        Folder where images are saved. If None, saves to the parent folder of the
        first dataset folder.
    """
    folder_paths = tuple(Path(p) for p in folder_paths)
    if len(folder_paths) != 2:
        raise ValueError("plot_workspace_comparison currently expects exactly two folder paths.")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=share_axes,
        sharey=share_axes,
    )

    style = None
    for i, (ax, folder, panel_title) in enumerate(zip(axes, folder_paths, panel_titles)):
        style = plot_workspace_on_axis(
            ax=ax,
            folder_path=folder,
            architectures=architectures,
            envelope_fill_alpha=envelope_fill_alpha,
            xlim=xlim,
            ylim=ylim,
            square_limits=False,
            axis_fontsize=axis_fontsize,
            tick_fontsize=tick_fontsize,
            waypoint_fontsize=waypoint_fontsize,
            obstacle_label_fontsize=obstacle_label_fontsize,
            title=None,
            title_fontsize=title_fontsize,
            show_ylabel=(i == 0),
            show_yticklabels=(i == 0),
            envelope_n_bins=envelope_n_bins,
            mean_n_points=mean_n_points,
        )

    # Panel captions below the axes
    panel_labels = ["a)", "b)"]

    for i, (ax, panel_title) in enumerate(zip(axes, panel_titles)):
        label = panel_labels[i] if i < len(panel_labels) else f"{chr(97 + i)})"

        ax.text(
            0.5,
            -0.24,
            f"{label} {panel_title}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=title_fontsize,
        )

    legend_handles = _workspace_legend_handles(style)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        frameon=True,
    )

    fig.subplots_adjust(
        top=0.99,
        bottom=0.2,
        left=0.08,
        right=0.992,
        wspace=wspace,
    )

    if output_folder is None:
        output_folder = folder_paths[0].parent
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / output_name

    if save_pdf:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")

    print("Saved comparison plot to:")
    if save_pdf:
        print(f"  {out_base.with_suffix('.pdf')}")
    if save_svg:
        print(f"  {out_base.with_suffix('.svg')}")
    if save_png:
        print(f"  {out_base.with_suffix('.png')}")

    if show_plot:
        plt.show(block=True)
    else:
        plt.close(fig)

    return fig, axes


# ============================================================
# Minimum distance / clearance plot over time
# ============================================================

def _get_obstacle_center_and_radius(
    params: Dict[str, Any],
    obstacle_index: int = 1,
    include_safety_margin: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Get obstacle center and effective radius from metadata.

    Parameters
    ----------
    obstacle_index:
        Zero-based obstacle index. obstacle_index=1 corresponds to obstacle 2.

    include_safety_margin:
        If True, add params["cbf_safety_margin"] if available.
        If False, only use the physical obstacle radius.
    """
    obstacle_centers = [tuple(c) for c in params.get("scenario_obstacle_centers", [])]
    obstacle_radii = list(params.get("obstacle_default_radius", []))

    if obstacle_index >= len(obstacle_centers):
        raise IndexError(
            f"Requested obstacle_index={obstacle_index}, "
            f"but only {len(obstacle_centers)} obstacles are available."
        )

    center = np.asarray(obstacle_centers[obstacle_index], dtype=float).reshape(2)
    radius = float(obstacle_radii[obstacle_index])

    if include_safety_margin:
        radius += float(params.get("cbf_safety_margin", 0.0))

    return center, radius


def _compute_ee_clearance_to_obstacle(
    x: np.ndarray,
    y: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
) -> np.ndarray:
    """
    Signed clearance from end-effector to obstacle boundary.

    Positive: outside obstacle.
    Zero: on obstacle boundary.
    Negative: collision/violation.
    """
    xy = np.column_stack([x, y])
    dist_to_center = np.linalg.norm(xy - obstacle_center.reshape(1, 2), axis=1)
    return dist_to_center - obstacle_radius


def _interpolate_scalar_runs_to_common_time(
    runs: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate scalar time series to a common time grid.

    Each run is (t, y). Shorter runs are extended by holding their final value.
    Longer runs define the maximum time range.

    Returns
    -------
    t_ref:
        Common time vector.

    y_runs:
        Array of shape (n_runs, n_time).
    """
    if len(runs) == 0:
        raise ValueError("No runs provided.")

    # Estimate common sample time from all available time vectors.
    dt_candidates = []
    for t, _ in runs:
        if len(t) >= 2:
            dt = np.median(np.diff(t))
            if np.isfinite(dt) and dt > 0:
                dt_candidates.append(dt)

    if len(dt_candidates) == 0:
        raise ValueError("Could not infer a valid sample time from the data.")

    dt_ref = float(np.median(dt_candidates))
    t_max = max(float(np.nanmax(t)) for t, _ in runs if len(t) > 0)

    n_ref = int(np.floor(t_max / dt_ref)) + 1
    t_ref = np.arange(n_ref) * dt_ref

    y_runs = np.zeros((len(runs), len(t_ref)), dtype=float)

    for i, (t, y) in enumerate(runs):
        valid = np.isfinite(t) & np.isfinite(y)
        t_valid = t[valid]
        y_valid = y[valid]

        if len(t_valid) == 0:
            y_runs[i, :] = np.nan
            continue

        # Remove duplicate time stamps.
        t_unique, idx = np.unique(t_valid, return_index=True)
        y_unique = y_valid[idx]

        y_runs[i, :] = np.interp(
            t_ref,
            t_unique,
            y_unique,
            left=y_unique[0],
            right=y_unique[-1],
        )

    return t_ref, y_runs


def _load_architecture_clearance_runs(
    root: Path,
    architecture: str,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load all trial runs of one architecture and compute signed EE clearance.
    """
    paths = _load_architecture_trial_paths(root, architecture)

    runs = []
    for t, x, y in paths:
        clearance = _compute_ee_clearance_to_obstacle(
            x=x,
            y=y,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
        )
        runs.append((t, clearance))

    return runs


def _clearance_statistics_over_time(
    runs: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean/min/max clearance over time.

    Returns
    -------
    t_ref, mean_clearance, min_clearance, max_clearance
    """
    t_ref, y_runs = _interpolate_scalar_runs_to_common_time(runs)

    mean_y = np.nanmean(y_runs, axis=0)
    min_y = np.nanmin(y_runs, axis=0)
    max_y = np.nanmax(y_runs, axis=0)

    return t_ref, mean_y, min_y, max_y


def _get_distance_plot_style() -> Dict[str, Any]:
    """
    Style dictionary for clearance-over-time plots.
    """
    return {
        # Okabe-Ito inspired / colorblind-friendly
        "arch_styles": {
            "LocalCBF": {
                "label": "Local CBF",
                "color": "#0072B2",
                "linestyle": "-",
                "linewidth": 1.3,
            },
            "RemoteMPC-CBF": {
                "label": "MPC-CBF",
                "color": "#E69F00",
                "linestyle": "--",
                "linewidth": 1.3,
            },
            "Combined": {
                "label": "Combined",
                "color": "#009E73",
                "linestyle": "-.",
                "linewidth": 1.3,
            },
        },
        "boundary_color": "red",
        "boundary_linestyle": "--",
        "boundary_linewidth": 1,
        "envelope_alpha": 0.15,
        "envelope_edge_alpha": 0.45,
        "envelope_edge_linewidth": 0.6,
    }


def plot_min_distance_to_obstacle(
    folder_path: str | Path,
    output_name: str = "min_distance_to_obstacle",
    output_folder: str | Path | None = None,
    architectures: Tuple[str, ...] = ("LocalCBF", "RemoteMPC-CBF", "Combined"),
    obstacle_index: int = 1,
    include_safety_margin: bool = False,
    save_pdf: bool = True,
    save_png: bool = True,
    save_svg: bool = False,
    dpi: int = 800,
    figsize: Tuple[float, float] = (3.5, 2.4),
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    axis_fontsize: float = 8,
    tick_fontsize: float = 7,
    legend_fontsize: float = 7,
    legend_loc: str = "lower left",
    legend_ncol: int = 1,
    envelope_alpha: float = 0.15,
    envelope_edge_linewidth: float = 0.6,
    envelope_edge_alpha: float = 0.45,
    grid_alpha: float = 0.3,
    show_plot: bool = False,
):
    """
    Plot signed end-effector clearance to a selected obstacle over time.

    The plot shows, for each available architecture:
        - envelope between min and max clearance over Monte Carlo runs
        - mean clearance over Monte Carlo runs

    The red dashed line at zero is the collision boundary.

    Parameters
    ----------
    folder_path:
        Monte Carlo root folder.

    obstacle_index:
        Zero-based obstacle index. Use obstacle_index=1 for obstacle 2.

    include_safety_margin:
        If True, the plotted clearance is relative to radius + safety margin.
        If False, the plotted clearance is relative to the physical obstacle radius.

    output_folder:
        If provided, save figure there. Otherwise save in folder_path.
    """
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    params = _load_meta_data(root)
    obstacle_center, obstacle_radius = _get_obstacle_center_and_radius(
        params,
        obstacle_index=obstacle_index,
        include_safety_margin=include_safety_margin,
    )

    style = _get_distance_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    plotted_architectures = []

    for arch in architectures:
        try:
            runs = _load_architecture_clearance_runs(
                root=root,
                architecture=arch,
                obstacle_center=obstacle_center,
                obstacle_radius=obstacle_radius,
            )
        except FileNotFoundError:
            print(f"Skipping {arch}: no data found.")
            continue

        if len(runs) == 0:
            print(f"Skipping {arch}: no runs found.")
            continue

        if arch not in style["arch_styles"]:
            print(f"Skipping {arch}: no style defined.")
            continue

        t_ref, mean_y, min_y, max_y = _clearance_statistics_over_time(runs)

        arch_style = style["arch_styles"][arch]
        color = arch_style["color"]

        # Envelope
        ax.fill_between(
            t_ref,
            min_y,
            max_y,
            color=color,
            alpha=envelope_alpha,
            linewidth=0.0,
            zorder=1,
        )

        # Thin envelope boundaries
        if envelope_edge_linewidth > 0:
            ax.plot(
                t_ref,
                min_y,
                color=color,
                linewidth=envelope_edge_linewidth,
                alpha=envelope_edge_alpha,
                zorder=2,
            )
            ax.plot(
                t_ref,
                max_y,
                color=color,
                linewidth=envelope_edge_linewidth,
                alpha=envelope_edge_alpha,
                zorder=2,
            )

        # Mean
        ax.plot(
            t_ref,
            mean_y,
            color=color,
            linestyle=arch_style["linestyle"],
            linewidth=arch_style["linewidth"],
            label=f"{arch_style['label']}",
            zorder=3,
        )

        plotted_architectures.append(arch)

        print(
            f"{root.name} / {arch}: {len(runs)} runs loaded. "
            f"min envelope = {np.nanmin(min_y):.4f} m"
        )

    # Collision boundary
    ax.axhline(
        0.0,
        color=style["boundary_color"],
        linestyle=style["boundary_linestyle"],
        linewidth=style["boundary_linewidth"],
        #label="Collision Boundary",
        zorder=4,
    )

    # Axes
    ax.set_xlabel(r"Time t (s)", fontsize=axis_fontsize)
    ax.set_ylabel(r"$\overline{d_{\mathrm{min}}}$ (m)", fontsize=axis_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, alpha=grid_alpha)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    """   ax.legend(
        loc=legend_loc,
        fontsize=legend_fontsize,
        ncol=legend_ncol,
        frameon=True,
    )  """   
    fig.legend(
       # handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        frameon=True,
    )

    #fig.tight_layout()

    # Save
    out_dir = Path(output_folder) if output_folder is not None else root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / output_name

    if save_pdf:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")

    print("Saved distance plot to:")
    if save_pdf:
        print(f"  {out_base.with_suffix('.pdf')}")
    if save_svg:
        print(f"  {out_base.with_suffix('.svg')}")
    if save_png:
        print(f"  {out_base.with_suffix('.png')}")

    if show_plot:
        plt.show(block=True)
    else:
        plt.close(fig)

    return fig, ax


# ============================================================
# Delay sweep plot: minimum signed clearance over known delay
# ============================================================

def _parse_delay_value_from_folder_name(folder_name: str) -> float:
    """
    Parse delay value from folder names such as:
        delay_1
        delay=1
        tau_1
        known_delay_1
        delay_0p5

    Returns np.nan if parsing fails.
    """
    patterns = [
        r"delay_([0-9]+(?:p[0-9]+)?)",
        r"delay=([0-9]+(?:p[0-9]+)?)",
        r"tau_([0-9]+(?:p[0-9]+)?)",
        r"known_delay_([0-9]+(?:p[0-9]+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, folder_name)
        if match:
            value_str = match.group(1).replace("p", ".")
            try:
                return float(value_str)
            except ValueError:
                return np.nan

    return np.nan


def _get_delay_from_case_folder(delay_dir: Path) -> float:
    """
    Determine delay from folder name first, then fall back to meta_data.json.

    This supports folders such as delay_1, delay=1, tau_1.
    """
    delay = _parse_delay_value_from_folder_name(delay_dir.name)

    if np.isfinite(delay):
        return delay

    try:
        params = _load_meta_data(delay_dir)
        for key in ["tau_known", "tau", "delay"]:
            if key in params:
                return float(params[key])
    except Exception:
        pass

    return np.nan


def _find_delay_case_dirs(sweep_root: Path) -> List[Path]:
    """
    Return subfolders of a delay sweep that look like delay cases.
    """
    case_dirs = []

    for p in sorted(sweep_root.iterdir()):
        if not p.is_dir():
            continue

        # Keep folders that either parse as delay folders or contain MC data.
        delay = _parse_delay_value_from_folder_name(p.name)
        has_trials = any(child.is_dir() and child.name.startswith("trial_") for child in p.iterdir())

        if np.isfinite(delay) or has_trials:
            case_dirs.append(p)

    return case_dirs


def _load_trial_min_clearances_for_delay_case(
    delay_dir: Path,
    architecture: str,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
) -> np.ndarray:
    """
    For one delay folder and one architecture, compute one scalar per trial:

        min_k ||p_ee(k) - O|| - r

    Returns
    -------
    trial_mins:
        Array of shape (n_trials,).
    """
    try:
        paths = _load_architecture_trial_paths(delay_dir, architecture)
    except FileNotFoundError:
        return np.asarray([], dtype=float)

    trial_mins = []

    for _, x, y in paths:
        clearance = _compute_ee_clearance_to_obstacle(
            x=x,
            y=y,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
        )

        if clearance.size == 0 or not np.any(np.isfinite(clearance)):
            continue

        trial_mins.append(float(np.nanmin(clearance)))

    return np.asarray(trial_mins, dtype=float)


def collect_delay_sweep_min_clearance_statistics(
    sweep_folder_path: str | Path,
    architectures: Tuple[str, ...] = ("LocalCBF", "RemoteMPC-CBF", "Combined"),
    obstacle_index: int = 1,
    include_safety_margin: bool = False,
) -> pd.DataFrame:
    """
    Collect minimum signed clearances over all delay cases and architectures.

    For each delay case and architecture, computes:
        - median_min_clearance
        - min_min_clearance
        - max_min_clearance
        - mean_min_clearance
        - n_trials

    The per-trial quantity is:
        min over time of signed EE clearance to selected obstacle.
    """
    sweep_root = Path(sweep_folder_path)

    if not sweep_root.exists():
        raise FileNotFoundError(f"Sweep folder not found: {sweep_root}")

    delay_dirs = _find_delay_case_dirs(sweep_root)

    if len(delay_dirs) == 0:
        raise FileNotFoundError(f"No delay case folders found in {sweep_root}")

    rows = []

    for delay_dir in delay_dirs:
        delay = _get_delay_from_case_folder(delay_dir)

        if not np.isfinite(delay):
            print(f"Skipping {delay_dir}: could not determine delay.")
            continue

        try:
            params = _load_meta_data(delay_dir)
            obstacle_center, obstacle_radius = _get_obstacle_center_and_radius(
                params=params,
                obstacle_index=obstacle_index,
                include_safety_margin=include_safety_margin,
            )
        except Exception as exc:
            print(f"Skipping {delay_dir}: could not load obstacle data ({exc}).")
            continue

        for arch in architectures:
            trial_mins = _load_trial_min_clearances_for_delay_case(
                delay_dir=delay_dir,
                architecture=arch,
                obstacle_center=obstacle_center,
                obstacle_radius=obstacle_radius,
            )

            if trial_mins.size == 0:
                print(f"Skipping {delay_dir.name} / {arch}: no valid trials found.")
                continue

            rows.append(
                {
                    "delay": float(delay),
                    "architecture": arch,
                    "n_trials": int(trial_mins.size),
                    "median_min_clearance": float(np.nanmedian(trial_mins)),
                    "min_min_clearance": float(np.nanmin(trial_mins)),
                    "max_min_clearance": float(np.nanmax(trial_mins)),
                    "mean_min_clearance": float(np.nanmean(trial_mins)),
                    "trial_min_clearances": trial_mins,
                }
            )

            print(
                f"{delay_dir.name} / {arch}: "
                f"{trial_mins.size} trials, "
                f"median d_min = {np.nanmedian(trial_mins):.4f} m, "
                f"range = [{np.nanmin(trial_mins):.4f}, {np.nanmax(trial_mins):.4f}] m"
            )

    if len(rows) == 0:
        raise ValueError(f"No valid delay-sweep clearance statistics found in {sweep_root}")

    stats = pd.DataFrame(rows)

    architecture_order = {
        "Nominal": 0,
        "LocalCBF": 1,
        "RemoteMPC-CBF": 2,
        "Combined": 3,
    }

    stats["architecture_order"] = stats["architecture"].map(architecture_order).fillna(99)
    stats = (
        stats
        .sort_values(["delay", "architecture_order"])
        .drop(columns=["architecture_order"])
        .reset_index(drop=True)
    )

    return stats


def plot_min_clearance_over_delay(
    sweep_folder_path: str | Path,
    output_name: str = "min_clearance_over_delay",
    output_folder: str | Path | None = None,
    architectures: Tuple[str, ...] = ("LocalCBF", "RemoteMPC-CBF", "Combined"),
    obstacle_index: int = 1,
    include_safety_margin: bool = False,
    save_pdf: bool = True,
    save_png: bool = True,
    save_svg: bool = False,
    save_csv: bool = True,
    dpi: int = 800,
    figsize: Tuple[float, float] = (3.5, 2.4),
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    axis_fontsize: float = 8,
    tick_fontsize: float = 7,
    legend_fontsize: float = 7,
    legend_loc: str = "best",
    legend_ncol: int = 3,
    envelope_alpha: float = 0.15,
    envelope_edge_linewidth: float = 0.6,
    envelope_edge_alpha: float = 0.55,
    marker_size: float = 4.0,
    grid_alpha: float = 0.3,
    show_plot: bool = False,
):
    """
    Plot minimum signed clearance over known delay.

    For each delay and architecture:
        - each trial gives one scalar:
              d_min_trial = min_k ||p_ee(k) - O|| - r
        - line shows median over trials
        - shaded region shows min/max over trials

    The red dashed line at zero is the collision boundary.
    """
    sweep_root = Path(sweep_folder_path)

    stats = collect_delay_sweep_min_clearance_statistics(
        sweep_folder_path=sweep_root,
        architectures=architectures,
        obstacle_index=obstacle_index,
        include_safety_margin=include_safety_margin,
    )

    style = _get_distance_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    plotted_architectures = []

    for arch in architectures:
        arch_stats = stats[stats["architecture"] == arch].copy()

        if arch_stats.empty:
            print(f"Skipping {arch}: no statistics available.")
            continue

        if arch not in style["arch_styles"]:
            print(f"Skipping {arch}: no style defined.")
            continue

        arch_stats = arch_stats.sort_values("delay")

        x = arch_stats["delay"].to_numpy(dtype=float)
        y_med = arch_stats["median_min_clearance"].to_numpy(dtype=float)
        y_min = arch_stats["min_min_clearance"].to_numpy(dtype=float)
        y_max = arch_stats["max_min_clearance"].to_numpy(dtype=float)

        arch_style = style["arch_styles"][arch]
        color = arch_style["color"]

        # Min-max envelope over trials
        ax.fill_between(
            x,
            y_min,
            y_max,
            color=color,
            alpha=envelope_alpha,
            linewidth=0.0,
            zorder=1,
        )

        # Thin envelope boundary
        if envelope_edge_linewidth > 0:
            ax.plot(
                x,
                y_min,
                color=color,
                linewidth=envelope_edge_linewidth,
                alpha=envelope_edge_alpha,
                zorder=2,
            )
            ax.plot(
                x,
                y_max,
                color=color,
                linewidth=envelope_edge_linewidth,
                alpha=envelope_edge_alpha,
                zorder=2,
            )

        # Median line
        ax.plot(
            x,
            y_med,
            color=color,
            linestyle=arch_style["linestyle"],
            linewidth=arch_style["linewidth"],
            marker="o",
            markersize=marker_size,
            label=arch_style["label"],
            zorder=3,
        )

        plotted_architectures.append(arch)

    # Collision boundary
    ax.axhline(
        0.0,
        color=style["boundary_color"],
        linestyle=style["boundary_linestyle"],
        linewidth=style["boundary_linewidth"],
        zorder=4,
    )

    # Axes
    ax.set_xlabel(r"Known delay $\hat{\tau}$ (steps)", fontsize=axis_fontsize)
    ax.set_ylabel(r"$d_{\min}$ (m)", fontsize=axis_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, alpha=grid_alpha)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    """    ax.legend(
        loc=legend_loc,
        fontsize=legend_fontsize,
        ncol=legend_ncol,
        frameon=True,
    ) """
    fig.legend(
          #  handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            frameon=True,
        )
    #fig.tight_layout()

    # Save
    out_dir = Path(output_folder) if output_folder is not None else sweep_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / output_name

    if save_csv:
        csv_path = out_base.with_suffix(".csv")
        csv_stats = stats.drop(columns=["trial_min_clearances"], errors="ignore")
        csv_stats.to_csv(csv_path, index=False)
        print(f"Saved delay-clearance statistics to: {csv_path}")

    if save_pdf:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")

    print("Saved delay-clearance plot to:")
    if save_pdf:
        print(f"  {out_base.with_suffix('.pdf')}")
    if save_svg:
        print(f"  {out_base.with_suffix('.svg')}")
    if save_png:
        print(f"  {out_base.with_suffix('.png')}")

    if show_plot:
        plt.show(block=True)
    else:
        plt.close(fig)

    return fig, ax, stats