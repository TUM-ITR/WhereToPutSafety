"""
Monte Carlo post-processing for architecture-comparison simulations.

Expected folder structure:

    monte_carlo_root/
        meta_data.json
        trial_0/
            LocalCBF/
                run_metrics.csv
            RemoteMPC-CBF/
                run_metrics.csv
        trial_1/
            LocalCBF/
                run_metrics.csv
            RemoteMPC-CBF/
                run_metrics.csv
        ...

Outputs:

    monte_carlo_root/
        all_run_metrics.csv
        monte_carlo_architecture_summary.csv
        monte_carlo_summary.xlsx
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
#from openpyxl import load_workbook

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _parse_trial_id(path: Path) -> Optional[int]:
    """Extract trial id from a path containing trial_0, trial_01, ..."""
    for part in path.parts:
        match = re.fullmatch(r"trial_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _find_run_metric_files(
    monte_carlo_root: str | Path,
    include_baseline: bool = False,
    ) -> list[Path]:
    """
    Find all run_metrics.csv files below the Monte Carlo root.

    By default, only include files inside trial_* folders.
    This prevents Baseline/Nominal/run_metrics.csv from being counted as
    an additional Monte Carlo trial.
    """
    root = Path(monte_carlo_root)
    metric_files = sorted(root.rglob("run_metrics.csv"))

    if include_baseline:
        return metric_files

    filtered = []
    for metric_file in metric_files:
        if _parse_trial_id(metric_file) is not None:
            filtered.append(metric_file)

    return filtered


def _as_bool_series(series: pd.Series) -> pd.Series:
    """Robustly convert bool-like values to boolean."""
    if series.dtype == bool:
        return series.fillna(False)

    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "1.0", "(true,)", "[true]"])


def _ensure_column(df: pd.DataFrame, col: str, default=np.nan) -> None:
    """Add column if missing."""
    if col not in df.columns:
        df[col] = default


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _mean(series: pd.Series) -> float:
    return float(_numeric(series).mean())


def _std(series: pd.Series) -> float:
    return float(_numeric(series).std())


def _median(series: pd.Series) -> float:
    return float(_numeric(series).median())


def _min(series: pd.Series) -> float:
    return float(_numeric(series).min())


def _max(series: pd.Series) -> float:
    return float(_numeric(series).max())


# ---------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------

def collect_monte_carlo_run_metrics(
    monte_carlo_root: str | Path,
    save: bool = True,
    include_baseline: bool = False,
) -> pd.DataFrame:
    """
    Collect all per-run run_metrics.csv files into one dataframe.

    Parameters
    ----------
    monte_carlo_root:
        Root folder of one Monte Carlo run.

    save:
        If True, save all_run_metrics.csv in the root folder.

    Returns
    -------
    df_all:
        One row per trial and architecture.
    """
    root = Path(monte_carlo_root)
    metric_files = _find_run_metric_files(root, include_baseline=include_baseline)

    if len(metric_files) == 0:
        raise FileNotFoundError(f"No run_metrics.csv files found under {root}")

    rows = []

    for metric_file in metric_files:
        df = pd.read_csv(metric_file)

        if len(df) != 1:
            print(f"Warning: {metric_file} has {len(df)} rows. Expected 1 row.")

        row = df.iloc[0].to_dict()

        trial_id = _parse_trial_id(metric_file)
        if trial_id is None and not include_baseline:
            print(f"Skipping non-trial metrics file: {metric_file}")
            continue

        architecture_from_folder = metric_file.parent.name

        row["trial_id"] = trial_id
        row["architecture"] = row.get("architecture", architecture_from_folder)
        row["architecture_folder"] = architecture_from_folder
        row["metrics_file"] = str(metric_file)

        rows.append(row)

    df_all = pd.DataFrame(rows)

    # Normalize important boolean columns.
    for col in ["reached_goal", "stayed_safe"]:
        if col in df_all.columns:
            df_all[col] = _as_bool_series(df_all[col])
        else:
            df_all[col] = False

    # Derived columns.
    df_all["safe_and_reached"] = (
        _as_bool_series(df_all["reached_goal"])
        & _as_bool_series(df_all["stayed_safe"])
    )

    if "violation_step_count" in df_all.columns:
        df_all["had_safety_violation"] = (
            _numeric(df_all["violation_step_count"]).fillna(0) > 0
        )
    elif "cbf_violation_count" in df_all.columns:
        df_all["had_safety_violation"] = (
            _numeric(df_all["cbf_violation_count"]).fillna(0) > 0
        )
    else:
        df_all["had_safety_violation"] = ~df_all["stayed_safe"]

    # Compatibility aliases.
    if "remote_failure_rate" in df_all.columns and "remote_mpc_failure_rate" not in df_all.columns:
        df_all["remote_mpc_failure_rate"] = df_all["remote_failure_rate"]

    if save:
        out_path = root / "all_run_metrics.csv"
        df_all.to_csv(out_path, index=False)
        print(f"Saved collected metrics to: {out_path}")

    return df_all


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def summarize_monte_carlo_by_architecture(
    df_all: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Compute selected Monte Carlo statistics with one row per architecture.

    Only includes the criteria selected for the current study.
    """

    # Ensure all required columns exist.
    required_defaults = {
        # Boolean/success
        "reached_goal": False,
        "stayed_safe": False,
        "safe_and_reached": False,
        "had_safety_violation": False,

        # Safety
        "min_signed_clearance": np.nan,
        "min_cbf_value": np.nan,

        # Performance
        "time_to_goal": np.nan,
        "total_cost": np.nan,
        "cost_per_step": np.nan,

        # Control
        "max_abs_control_input": np.nan,
        "max_abs_control_input_change": np.nan,
        "max_abs_ddq_command": np.nan,
        "max_abs_ddq_command_change": np.nan,

        # CBF intervention
        "local_cbf_activation_rate": 0.0,
        "remote_cbf_activation_rate": 0.0,
        "local_cbf_activation_count": 0.0,
        "remote_cbf_activation_count": 0.0,
        "local_cbf_slack_sum": 0.0,
        "remote_mpc_slack_sum": 0.0,
        "local_cbf_slack_max": 0.0,
        "remote_mpc_slack_max": 0.0,

        # Failures / computation
        "local_cbf_failure_rate": 0.0,
        "remote_mpc_failure_rate": 0.0,
        "mean_mpc_solve_time": np.nan,
        "max_mpc_solve_time": np.nan,
        "mean_local_cbf_solve_time": np.nan,
        "max_local_cbf_solve_time": np.nan,
    }

    for col, default in required_defaults.items():
        _ensure_column(df_all, col, default)

    for col in ["reached_goal", "stayed_safe", "safe_and_reached", "had_safety_violation"]:
        df_all[col] = _as_bool_series(df_all[col])

    rows = []

    for architecture, group in df_all.groupby("architecture", sort=False):
        reached = _as_bool_series(group["reached_goal"])
        safe = _as_bool_series(group["stayed_safe"])
        safe_and_reached = _as_bool_series(group["safe_and_reached"])
        had_violation = _as_bool_series(group["had_safety_violation"])

        reached_group = group[reached]

        row = {
            # Core reliability
            "architecture": architecture,
            "n_runs": int(len(group)),
            "reach_rate": float(reached.mean()),
            "safe_rate": float(safe.mean()),
            "safe_and_reached_rate": float(safe_and_reached.mean()),

            # Safety
            "mean_min_signed_clearance": _mean(group["min_signed_clearance"]),
            "worst_min_signed_clearance": _min(group["min_signed_clearance"]),
            "mean_min_cbf_value": _mean(group["min_cbf_value"]),
            "worst_min_cbf_value": _min(group["min_cbf_value"]),
            "violation_run_rate": float(had_violation.mean()),

            # Performance
            "mean_time_to_goal": _mean(reached_group["time_to_goal"]) if len(reached_group) else np.nan,
            "std_time_to_goal": _std(reached_group["time_to_goal"]) if len(reached_group) else np.nan,
            "median_time_to_goal": _median(reached_group["time_to_goal"]) if len(reached_group) else np.nan,
            "mean_total_cost": _mean(group["total_cost"]),
            "std_total_cost": _std(group["total_cost"]),
            "mean_cost_per_step": _mean(group["cost_per_step"]),
            "std_cost_per_step": _std(group["cost_per_step"]),
            "median_cost_per_step": _median(group["cost_per_step"]),

            # Control effort / smoothness
            "mean_max_abs_control_input": _mean(group["max_abs_control_input"]),
            "max_max_abs_control_input": _max(group["max_abs_control_input"]),
            "mean_max_abs_control_input_change": _mean(group["max_abs_control_input_change"]),
            "mean_max_abs_ddq_command": _mean(group["max_abs_ddq_command"]),
            "mean_max_abs_ddq_command_change": _mean(group["max_abs_ddq_command_change"]),

            # CBF intervention
            "mean_local_cbf_activation_rate": _mean(group["local_cbf_activation_rate"]),
            "mean_remote_cbf_activation_rate": _mean(group["remote_cbf_activation_rate"]),
            "mean_local_cbf_activation_count": _mean(group["local_cbf_activation_count"]),
            "mean_remote_cbf_activation_count": _mean(group["remote_cbf_activation_count"]),
            "mean_local_cbf_slack_sum": _mean(group["local_cbf_slack_sum"]),
            "mean_remote_mpc_slack_sum": _mean(group["remote_mpc_slack_sum"]),
            "max_local_cbf_slack": _max(group["local_cbf_slack_max"]),
            "max_remote_mpc_slack": _max(group["remote_mpc_slack_max"]),

            # Failures / computation
            "mean_local_cbf_failure_rate": _mean(group["local_cbf_failure_rate"]),
            "mean_remote_mpc_failure_rate": _mean(group["remote_mpc_failure_rate"]),
            "mean_mpc_solve_time": _mean(group["mean_mpc_solve_time"]),
            "max_mpc_solve_time": _max(group["max_mpc_solve_time"]),
            "mean_local_cbf_solve_time": _mean(group["mean_local_cbf_solve_time"]),
            "max_local_cbf_solve_time": _max(group["max_local_cbf_solve_time"]),
        }

        rows.append(row)

    summary = pd.DataFrame(rows)

    # Stable, useful architecture ordering.
    preferred_order = ["Nominal", "LocalCBF", "RemoteMPC-CBF", "Combined"]
    summary["architecture_order"] = summary["architecture"].apply(
        lambda x: preferred_order.index(x) if x in preferred_order else len(preferred_order)
    )
    summary = summary.sort_values("architecture_order").drop(columns=["architecture_order"])

    # Exact requested column order.
    requested_columns = [
        "architecture",
        "n_runs",
        "reach_rate",
        "safe_rate",
        "safe_and_reached_rate",

        "mean_min_signed_clearance",
        "worst_min_signed_clearance",
        "mean_min_cbf_value",
        "worst_min_cbf_value",
        "violation_run_rate",

        "mean_time_to_goal",
        "std_time_to_goal",
        "median_time_to_goal",
        "mean_total_cost",
        "std_total_cost",
        "mean_cost_per_step",
        "std_cost_per_step",
        "median_cost_per_step",

        "mean_max_abs_control_input",
        "max_max_abs_control_input",
        "mean_max_abs_control_input_change",
        "mean_max_abs_ddq_command",
        "mean_max_abs_ddq_command_change",

        "mean_local_cbf_activation_rate",
        "mean_remote_cbf_activation_rate",
        "mean_local_cbf_activation_count",
        "mean_remote_cbf_activation_count",
        "mean_local_cbf_slack_sum",
        "mean_remote_mpc_slack_sum",
        "max_local_cbf_slack",
        "max_remote_mpc_slack",

        "mean_local_cbf_failure_rate",
        "mean_remote_mpc_failure_rate",
        "mean_mpc_solve_time",
        "max_mpc_solve_time",
        "mean_local_cbf_solve_time",
        "max_local_cbf_solve_time",
    ]

    summary = summary[requested_columns]

    if save_path is not None:
        save_path = Path(save_path)
        summary.to_csv(save_path, index=False)
        print(f"Saved architecture summary to: {save_path}")

    return summary


# ---------------------------------------------------------------------
# Full entry point
# ---------------------------------------------------------------------

def analyse_monte_carlo_run(
    monte_carlo_root: str | Path,
    save_excel: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect per-run metrics and compute architecture-level Monte Carlo summary.

    Returns
    -------
    df_all:
        One row per trial and architecture.

    summary:
        One row per architecture, only with selected criteria.
    """
    root = Path(monte_carlo_root)

    df_all = collect_monte_carlo_run_metrics(root, save=True)

    summary_path = root / "monte_carlo_architecture_summary.csv"
    summary = summarize_monte_carlo_by_architecture(df_all, save_path=summary_path)

    if save_excel:
        excel_path = root / "monte_carlo_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="all_run_metrics", index=False)
            summary.to_excel(writer, sheet_name="architecture_summary", index=False)
        print(f"Saved Excel summary to: {excel_path}")

    print_monte_carlo_summary(summary)

    return df_all, summary


def print_monte_carlo_summary(summary: pd.DataFrame) -> None:
    """
    Compact console printout for the selected criteria.
    """
    if summary.empty:
        print("No Monte Carlo summary data available.")
        return

    display_cols = [
        "architecture",
        "n_runs",
        "safe_rate",
        "reach_rate",
        "safe_and_reached_rate",
        "mean_cost_per_step",
        "mean_time_to_goal",
        "worst_min_signed_clearance",
        "mean_local_cbf_activation_rate",
        "mean_remote_cbf_activation_rate",
    ]

    print("\n" + "=" * 120)
    print("MONTE CARLO ARCHITECTURE SUMMARY")
    print("=" * 120)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 180,
        "display.float_format", "{:.4f}".format,
    ):
        print(summary[display_cols].to_string(index=False))

    print("=" * 120 + "\n")

def analyse_disturbance_sweep(
    sweep_out_dir: str,
    save_excel: bool = True,
):
    """
    Aggregate all disturbance-bound Monte Carlo summaries into one table.

    Expected structure:

        sweep_out_dir/
            bound_0p00000/
                monte_carlo_architecture_summary.csv
            bound_0p01000/
                monte_carlo_architecture_summary.csv
            ...

    Output:
        disturbance_sweep_summary.csv
        disturbance_sweep_summary.xlsx
    """

    selected_columns = [
        "disturbance_bound",
        "architecture",
        "safe_rate",
        "reach_rate",
        "safe_and_reached_rate",
        "mean_cost_per_step",
        "mean_time_to_goal",
        "mean_min_signed_clearance",
        "mean_local_cbf_activation_rate",
        "mean_remote_cbf_activation_rate",
        "mean_remote_mpc_slack_sum",
    ]

    rows = []

    for name in sorted(os.listdir(sweep_out_dir)):
        bound_dir = os.path.join(sweep_out_dir, name)

        if not os.path.isdir(bound_dir):
            continue

        summary_path = os.path.join(bound_dir, "monte_carlo_architecture_summary.csv")

        if not os.path.isfile(summary_path):
            print(f"Skipping {bound_dir}: no monte_carlo_architecture_summary.csv found.")
            continue

        df = pd.read_csv(summary_path)

        # Prefer explicitly saved disturbance_bound if present.
        if "disturbance_bound" not in df.columns:
            # Fallback: parse folder name like bound_0p01000 -> 0.01000
            try:
                bound_str = name.replace("bound_", "").replace("p", ".")
                df["disturbance_bound"] = float(bound_str)
            except ValueError:
                df["disturbance_bound"] = np.nan

        for col in selected_columns:
            if col not in df.columns:
                df[col] = np.nan

        rows.append(df[selected_columns])

    if len(rows) == 0:
        raise FileNotFoundError(
            f"No Monte Carlo summaries found below {sweep_out_dir}."
        )

    sweep_summary = pd.concat(rows, ignore_index=True)

    # Sort for readability.
    architecture_order = {
        "Nominal": 0,
        "LocalCBF": 1,
        "RemoteMPC-CBF": 2,
        "Combined": 3,
    }

    sweep_summary["architecture_order"] = sweep_summary["architecture"].map(
        architecture_order
    ).fillna(99)

    sweep_summary = sweep_summary.sort_values(
        ["disturbance_bound", "architecture_order"]
    ).drop(columns=["architecture_order"])

    csv_path = os.path.join(sweep_out_dir, "disturbance_sweep_summary.csv")
    sweep_summary.to_csv(csv_path, index=False)
    print(f"Saved disturbance sweep summary to: {csv_path}")

    if save_excel:
        excel_path = os.path.join(sweep_out_dir, "disturbance_sweep_summary.xlsx")
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                sweep_summary.to_excel(
                    writer,
                    sheet_name="disturbance_sweep",
                    index=False,
                )

                # Optional pivot tables for quick inspection.
                for metric in ["safe_rate", "reach_rate", "mean_cost_per_step"]:
                    pivot = sweep_summary.pivot_table(
                        index="disturbance_bound",
                        columns="architecture",
                        values=metric,
                    )
                    pivot.to_excel(writer, sheet_name=metric[:31])

            print(f"Saved disturbance sweep Excel summary to: {excel_path}")

        except ImportError:
            print("openpyxl not installed. Skipping Excel export.")

    print_disturbance_sweep_summary(sweep_summary)

    return sweep_summary

def print_disturbance_sweep_summary(sweep_summary: pd.DataFrame) -> None:
    display_cols = [
        "disturbance_bound",
        "architecture",
        "safe_rate",
        "reach_rate",
        "safe_and_reached_rate",
        "mean_cost_per_step",
        "mean_time_to_goal",
        "mean_min_signed_clearance",
    ]

    print("\n" + "=" * 120)
    print("DISTURBANCE SWEEP SUMMARY")
    print("=" * 120)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 160,
        "display.float_format", "{:.4f}".format,
    ):
        print(sweep_summary[display_cols].to_string(index=False))

    print("=" * 120 + "\n")

def analyse_delay_sweep(
    sweep_out_dir: str | Path,
    save_excel: bool = True,
):
    """
    Aggregate all known-delay Monte Carlo summaries into one table.

    Expected folder structure, for example:

        sweep_out_dir/
            delay_0/
                monte_carlo_architecture_summary.csv
            delay_1/
                monte_carlo_architecture_summary.csv
            ...
            delay_10/
                monte_carlo_architecture_summary.csv

    or alternatively:

        sweep_out_dir/
            tau_0/
            tau_1/
            ...

    Output:
        delay_sweep_summary.csv
        delay_sweep_summary.xlsx
    """

    sweep_out_dir = Path(sweep_out_dir)

    selected_columns = [
        "delay",
        "architecture",

        # reliability
        "n_runs",
        "safe_rate",
        "reach_rate",
        "safe_and_reached_rate",

        # safety
        "mean_min_signed_clearance",
        "worst_min_signed_clearance",
        "mean_min_cbf_value",
        "worst_min_cbf_value",
        "violation_run_rate",

        # performance
        "mean_time_to_goal",
        "std_time_to_goal",
        "median_time_to_goal",
        "mean_total_cost",
        "std_total_cost",
        "mean_cost_per_step",
        "std_cost_per_step",
        "median_cost_per_step",

        # control
        "mean_max_abs_control_input",
        "max_max_abs_control_input",
        "mean_max_abs_control_input_change",
        "mean_max_abs_ddq_command",
        "mean_max_abs_ddq_command_change",

        # CBF intervention
        "mean_local_cbf_activation_rate",
        "mean_remote_cbf_activation_rate",
        "mean_local_cbf_activation_count",
        "mean_remote_cbf_activation_count",
        "mean_local_cbf_slack_sum",
        "mean_remote_mpc_slack_sum",
        "max_local_cbf_slack",
        "max_remote_mpc_slack",

        # solver reliability / timing
        "mean_local_cbf_failure_rate",
        "mean_remote_mpc_failure_rate",
        "mean_mpc_solve_time",
        "max_mpc_solve_time",
        "mean_local_cbf_solve_time",
        "max_local_cbf_solve_time",
    ]

    rows = []

    for name in sorted(os.listdir(sweep_out_dir)):
        delay_dir = sweep_out_dir / name

        if not delay_dir.is_dir():
            continue

        # Prefer delay_summary.csv if you explicitly save it.
        delay_summary_path = delay_dir / "delay_summary.csv"

        # Otherwise fall back to standard MC architecture summary.
        mc_summary_path = delay_dir / "monte_carlo_architecture_summary.csv"

        if delay_summary_path.is_file():
            summary_path = delay_summary_path
        elif mc_summary_path.is_file():
            summary_path = mc_summary_path
        else:
            print(f"Skipping {delay_dir}: no delay_summary.csv or monte_carlo_architecture_summary.csv found.")
            continue

        df = pd.read_csv(summary_path)

        # Prefer explicitly saved delay column.
        if "delay" not in df.columns:
            delay = _parse_delay_from_folder_name(name)
            df["delay"] = delay

        for col in selected_columns:
            if col not in df.columns:
                df[col] = np.nan

        rows.append(df[selected_columns])

    if len(rows) == 0:
        raise FileNotFoundError(
            f"No delay sweep summaries found below {sweep_out_dir}."
        )

    sweep_summary = pd.concat(rows, ignore_index=True)

    # Convert delay to numeric if possible.
    sweep_summary["delay"] = pd.to_numeric(sweep_summary["delay"], errors="coerce")

    # Sort for readability.
    architecture_order = {
        "Nominal": 0,
        "LocalCBF": 1,
        "RemoteMPC-CBF": 2,
        "Combined": 3,
    }

    sweep_summary["architecture_order"] = (
        sweep_summary["architecture"]
        .map(architecture_order)
        .fillna(99)
    )

    sweep_summary = (
        sweep_summary
        .sort_values(["delay", "architecture_order"])
        .drop(columns=["architecture_order"])
        .reset_index(drop=True)
    )

    csv_path = sweep_out_dir / "delay_sweep_summary.csv"
    sweep_summary.to_csv(csv_path, index=False)
    print(f"Saved delay sweep summary to: {csv_path}")

    if save_excel:
        excel_path = sweep_out_dir / "delay_sweep_summary.xlsx"

        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                sweep_summary.to_excel(
                    writer,
                    sheet_name="delay_sweep",
                    index=False,
                )

                # Useful pivot tables for quick inspection.
                pivot_metrics = [
                    "safe_rate",
                    "reach_rate",
                    "safe_and_reached_rate",
                    "mean_min_signed_clearance",
                    "worst_min_signed_clearance",
                    "mean_cost_per_step",
                    "mean_time_to_goal",
                ]

                for metric in pivot_metrics:
                    if metric not in sweep_summary.columns:
                        continue

                    pivot = sweep_summary.pivot_table(
                        index="delay",
                        columns="architecture",
                        values=metric,
                    )

                    # Excel sheet names max out at 31 characters.
                    pivot.to_excel(writer, sheet_name=metric[:31])

            print(f"Saved delay sweep Excel summary to: {excel_path}")

        except ImportError:
            print("openpyxl not installed. Skipping Excel export.")

    print_delay_sweep_summary(sweep_summary)

    return sweep_summary

def _parse_delay_from_folder_name(folder_name: str) -> float:
    """
    Parse delay value from folder names such as:
        delay_0
        delay_3
        tau_5
        known_delay_10
        delay_0p5

    Returns np.nan if parsing fails.
    """
    patterns = [
        r"delay_([0-9]+(?:p[0-9]+)?)",
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

def print_delay_sweep_summary(sweep_summary: pd.DataFrame) -> None:
    display_cols = [
        "delay",
        "architecture",
        "safe_rate",
        "reach_rate",
        "safe_and_reached_rate",
        "mean_min_signed_clearance",
        "worst_min_signed_clearance",
        "mean_cost_per_step",
        "mean_time_to_goal",
    ]

    display_cols = [c for c in display_cols if c in sweep_summary.columns]

    print("\n" + "=" * 120)
    print("DELAY SWEEP SUMMARY")
    print("=" * 120)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 180,
        "display.float_format", "{:.4f}".format,
    ):
        print(sweep_summary[display_cols].to_string(index=False))

    print("=" * 120 + "\n")