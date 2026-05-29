import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Convert a column to numeric, coercing invalid entries to NaN."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Robust boolean parser.

    Handles:
        True / False
        1 / 0
        "True" / "False"
        "(True,)" / "(False,)" if accidental trailing commas occurred upstream
        NaN
    """
    if col not in df.columns:
        return pd.Series(False, index=df.index)

    s = df[col]

    if s.dtype == bool:
        return s.fillna(False)

    s_str = s.astype(str).str.strip().str.lower()

    true_values = {"true", "1", "1.0", "(true,)", "[true]"}
    false_values = {"false", "0", "0.0", "(false,)", "[false]", "nan", "none", ""}

    out = pd.Series(False, index=df.index)
    out[s_str.isin(true_values)] = True
    out[s_str.isin(false_values)] = False

    return out


def _numeric_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _wrap_angle_error(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    #needed for MPC Cost calculation
    e = q - q_ref
    return np.arctan2(np.sin(e), np.cos(e))


def _monitored_points_from_q(params, dynamics, q: np.ndarray) -> list[np.ndarray]:
    """
    Reconstruct the same type of monitored points used for obstacle checks.

    Uses:
        base, p1, p2, ee
    plus intermediate points on each link according to params.cbf_link_samples.
    """
    base, p1, p2, ee = dynamics.fk_points(q)

    points = [base, p1, p2, ee]

    n_samp = int(getattr(params, "cbf_link_samples", 3))
    grid = np.linspace(0.0, 1.0, n_samp + 1)[1:-1]

    for s in grid:
        points.append(base + s * (p1 - base))
        points.append(p1 + s * (p2 - p1))
        points.append(p2 + s * (ee - p2))

    return points


def compute_obstacle_safety_metrics(params, dynamics, q_traj: np.ndarray) -> dict[str, Any]:
    """
    Compute obstacle safety metrics over the executed trajectory.

    CBF:
        h = ||p - O||^2 - r_eff^2

    Signed clearance:
        d_clear = ||p - O|| - r_eff
    """
    if getattr(params, "obstacle_scene", None) is None:
        return {
            "min_cbf_value": np.inf,
            "min_signed_clearance": np.inf,
            "cbf_violation_count": 0,
            "violation_step_count": 0,
            "max_violation_depth": 0.0,
        }

    margin = float(getattr(params, "cbf_safety_margin", 0.0))

    min_h = np.inf
    min_clearance = np.inf
    cbf_violation_count = 0
    violation_step_count = 0
    max_violation_depth = 0.0

    for q in q_traj:
        if not np.all(np.isfinite(q)):
            continue

        step_has_violation = False
        points = _monitored_points_from_q(params, dynamics, q)

        for obs in params.obstacle_scene.obstacles:
            if getattr(obs, "type", "circle") != "circle":
                continue

            center = np.asarray(obs.center, dtype=float).reshape(2)
            r_eff = float(obs.radius) + margin

            for p in points:
                p = np.asarray(p, dtype=float).reshape(2)

                dist = float(np.linalg.norm(p - center))
                clearance = dist - r_eff
                h = dist**2 - r_eff**2

                min_h = min(min_h, h)
                min_clearance = min(min_clearance, clearance)

                if h < 0.0:
                    cbf_violation_count += 1
                    step_has_violation = True
                    max_violation_depth = max(max_violation_depth, -clearance)

        if step_has_violation:
            violation_step_count += 1

    return {
        "min_cbf_value": float(min_h),
        "min_signed_clearance": float(min_clearance),
        "cbf_violation_count": int(cbf_violation_count),
        "violation_step_count": int(violation_step_count),
        "max_violation_depth": float(max_violation_depth),
    }


def compute_executed_mpc_style_cost(
    params,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    q_ref: np.ndarray,
) -> np.ndarray:
    """
    Compute a realized stage cost along the executed trajectory.

    Uses your joint-space MPC-style terms:

        ||q - q_ref||_Qq^2
      + ||dq||_Qdq^2
      + ||ddq||_Rddq^2
      + ||ddq_k - ddq_{k-1}||_Rjerk^2

    This is not the optimizer's predicted objective value. It is better for
    comparing the realized behavior of different architectures.
    """
    Qq = np.asarray(params.Qq_mpc, dtype=float)
    Qdq = np.asarray(params.Qdq_mpc, dtype=float)
    Rddq = np.asarray(params.Rddq_mpc, dtype=float)
    Rjerk = np.asarray(params.Rjerk_mpc, dtype=float)

    n = min(len(q), len(dq), len(ddq), len(q_ref))
    costs = np.full(n, np.nan)

    ddq_prev = np.zeros(3)

    for k in range(n):
        if not (
            np.all(np.isfinite(q[k]))
            and np.all(np.isfinite(dq[k]))
            and np.all(np.isfinite(ddq[k]))
            and np.all(np.isfinite(q_ref[k]))
        ):
            continue

        q_err = _wrap_angle_error(q[k], q_ref[k])
        dq_k = dq[k]
        ddq_k = ddq[k]
        dddq_k = ddq_k - ddq_prev

        costs[k] = (
            q_err.T @ Qq @ q_err
            + dq_k.T @ Qdq @ dq_k
            + ddq_k.T @ Rddq @ ddq_k
            + dddq_k.T @ Rjerk @ dddq_k
        )

        ddq_prev = ddq_k

    return costs


def _safe_sum(df: pd.DataFrame, col: str, eval_slice) -> float:
    vals = _to_numeric(df, col).iloc[eval_slice].to_numpy(dtype=float)
    if np.all(np.isnan(vals)):
        return 0.0
    return float(np.nansum(vals))


def _safe_max(df: pd.DataFrame, col: str, eval_slice) -> float:
    vals = _to_numeric(df, col).iloc[eval_slice].to_numpy(dtype=float)
    if np.all(np.isnan(vals)):
        return 0.0
    return float(np.nanmax(vals))


def _safe_mean(df: pd.DataFrame, col: str, eval_slice) -> float:
    vals = _to_numeric(df, col).iloc[eval_slice].to_numpy(dtype=float)
    if np.all(np.isnan(vals)):
        return np.nan
    return float(np.nanmean(vals))


def _safe_valid_count(df: pd.DataFrame, col: str, eval_slice) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].iloc[eval_slice].notna().sum())


def post_process_single_run(
    detailed_csv_path: str,
    params,
    dynamics,
    arch_name: str,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """
    Post-process one simulation run generated by your DataRecorder.

    Saves:
        run_metrics.json
        run_metrics.csv

    Returns:
        metrics dictionary.
    """
    df = pd.read_csv(detailed_csv_path)

    if output_dir is None:
        output_dir = os.path.dirname(detailed_csv_path)

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Extract recorded data according to your DataRecorder columns
    # ------------------------------------------------------------
    t = _to_numeric(df, "time").to_numpy(dtype=float)

    q = _numeric_array(df, ["theta1_real", "theta2_real", "theta3_real"])
    dq = _numeric_array(df, ["omega1_real", "omega2_real", "omega3_real"])

    q_pred = _numeric_array(df, ["theta1_pred", "theta2_pred", "theta3_pred"])
    dq_pred = _numeric_array(df, ["omega1_pred", "omega2_pred", "omega3_pred"])

    q_des = _numeric_array(df, ["theta1_des", "theta2_des", "theta3_des"])
    dq_des = _numeric_array(df, ["omega1_des", "omega2_des", "omega3_des"])

    tau = _numeric_array(df, ["u1", "u2", "u3"])

    ddq_mpc = _numeric_array(df, ["ddq_mpc_1", "ddq_mpc_2", "ddq_mpc_3"])
    ddq_local_cbf = _numeric_array(df, ["ddq_local_cbf_1", "ddq_local_cbf_2", "ddq_local_cbf_3"])

    p_ref = _numeric_array(df, ["p_ref_x", "p_ref_y"])
    q_ref = _numeric_array(df, ["q_ref_1", "q_ref_2", "q_ref_3"])

    w = _numeric_array(df, ["w1", "w2", "w3"])

    ee_real = _numeric_array(df, ["ee_x_real", "ee_y_real"])
    ee_des = _numeric_array(df, ["ee_x_des", "ee_y_des"])
    ee_pred = _numeric_array(df, ["ee_x_pred", "ee_y_pred"])

    # Use filtered acceleration if available; otherwise use MPC acceleration.
    # For non-local architectures, ddq_local_cbf is usually NaN.
    if np.all(np.isnan(ddq_local_cbf)):
        ddq_exec = ddq_mpc.copy()
    else:
        ddq_exec = ddq_local_cbf.copy()

        # Fill gaps with MPC values if local CBF values are NaN at some steps.
        nan_rows = np.any(np.isnan(ddq_exec), axis=1)
        ddq_exec[nan_rows] = ddq_mpc[nan_rows]

    # Forward/backward fill reference and acceleration values because some
    # quantities may only be updated at MPC rate.
    ddq_exec = pd.DataFrame(ddq_exec).ffill().bfill().to_numpy(dtype=float)
    q_ref = pd.DataFrame(q_ref).ffill().bfill().to_numpy(dtype=float)
    p_ref = pd.DataFrame(p_ref).ffill().bfill().to_numpy(dtype=float)

    # ------------------------------------------------------------
    # Determine time to target
    # ------------------------------------------------------------
    final_target = np.asarray(params.waypoints[-1], dtype=float).reshape(2)
    final_error = np.linalg.norm(ee_real - final_target, axis=1)

    goal_tol = float(getattr(params, "goal_tolerance", 0.035))
    reached_indices = np.where(final_error <= goal_tol)[0]

    if len(reached_indices) > 0:
        reach_idx = int(reached_indices[0])
        reached_goal = True
        time_to_goal = float(t[reach_idx])
    else:
        reach_idx = len(df) - 1
        reached_goal = False
        time_to_goal = np.nan

    eval_slice = slice(0, reach_idx + 1)
    n_eval_steps = int(reach_idx + 1)

    # ------------------------------------------------------------
    # Tracking and cost
    # ------------------------------------------------------------
    ee_error_to_current_ref = np.linalg.norm(ee_real - p_ref, axis=1)
    ee_error_to_final_target = final_error

    step_costs = compute_executed_mpc_style_cost(
        params=params,
        q=q[eval_slice],
        dq=dq[eval_slice],
        ddq=ddq_exec[eval_slice],
        q_ref=q_ref[eval_slice],
    )

    total_cost = float(np.nansum(step_costs))
    cost_per_step = float(total_cost / max(n_eval_steps, 1))

    # ------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------
    safety = compute_obstacle_safety_metrics(
        params=params,
        dynamics=dynamics,
        q_traj=q[eval_slice],
    )

    stayed_safe = bool(
        safety["cbf_violation_count"] == 0
        and safety["min_cbf_value"] >= 0.0
    )

    # ------------------------------------------------------------
    # Control effort and smoothness
    # ------------------------------------------------------------
    tau_eval = tau[eval_slice]
    ddq_eval = ddq_exec[eval_slice]

    max_abs_control_input = float(np.nanmax(np.abs(tau_eval)))

    if len(tau_eval) >= 2:
        dtau = np.diff(tau_eval, axis=0)
        max_abs_control_input_change = float(np.nanmax(np.abs(dtau)))
        max_control_input_change_norm = float(np.nanmax(np.linalg.norm(dtau, axis=1)))
    else:
        max_abs_control_input_change = 0.0
        max_control_input_change_norm = 0.0

    max_abs_ddq_command = float(np.nanmax(np.abs(ddq_eval)))

    if len(ddq_eval) >= 2:
        dddq = np.diff(ddq_eval, axis=0)
        max_abs_ddq_command_change = float(np.nanmax(np.abs(dddq)))
        max_ddq_command_change_norm = float(np.nanmax(np.linalg.norm(dddq, axis=1)))
    else:
        max_abs_ddq_command_change = 0.0
        max_ddq_command_change_norm = 0.0

    # ------------------------------------------------------------
    # CBF activation / feasibility counts and rates
    # ------------------------------------------------------------
    local_active = _bool_series(df, "local_cbf_active")
    remote_active = _bool_series(df, "mpc_cbf_active")

    local_feasible = _bool_series(df, "local_cbf_feasible")
    remote_feasible = _bool_series(df, "mpc_feasible")

    # Local CBF is evaluated at local 200 Hz steps when active.
    # Use local_cbf_solvetime as validity indicator where possible.
    local_solve_time = _to_numeric(df, "local_cbf_solvetime")
    local_valid_mask = local_solve_time.iloc[eval_slice].notna()

    # Remote MPC is only evaluated at MPC trigger steps.
    # Use mpc_solvetime as the cleanest validity indicator.
    remote_solve_time = _to_numeric(df, "mpc_solvetime")
    remote_valid_mask = remote_solve_time.iloc[eval_slice].notna() #only rows with valud mpc solvetime are considered MPC calls

    # Fallback if solve times are not recorded correctly.
    # This uses finite mpc_feasible values instead.
    if remote_valid_mask.sum() == 0 and "mpc_feasible" in df.columns:
        remote_raw = df["mpc_feasible"].iloc[eval_slice].astype(str).str.strip().str.lower()
        remote_valid_mask = ~remote_raw.isin(["nan", "none", "", "(nan,)", "[nan]"])

    if local_valid_mask.sum() == 0 and "local_cbf_feasible" in df.columns:
        local_raw = df["local_cbf_feasible"].iloc[eval_slice].astype(str).str.strip().str.lower()
        local_valid_mask = ~local_raw.isin(["nan", "none", "", "(nan,)", "[nan]"])

    local_valid_count = int(local_valid_mask.sum())
    remote_valid_count = int(remote_valid_mask.sum())

    local_activation_count = int(local_active.iloc[eval_slice][local_valid_mask].sum())
    remote_activation_count = int(remote_active.iloc[eval_slice][remote_valid_mask].sum())

    local_success_count = int(local_feasible.iloc[eval_slice][local_valid_mask].sum())
    remote_success_count = int(remote_feasible.iloc[eval_slice][remote_valid_mask].sum())

    local_failure_count = int(local_valid_count - local_success_count)
    remote_failure_count = int(remote_valid_count - remote_success_count)

    local_activation_rate = float(local_activation_count / max(local_valid_count, 1))
    remote_activation_rate = float(remote_activation_count / max(remote_valid_count, 1))

    local_failure_rate = float(local_failure_count / max(local_valid_count, 1))
    remote_failure_rate = float(remote_failure_count / max(remote_valid_count, 1))

    # ------------------------------------------------------------
    # Solver times, slack, disturbance, prediction quality
    # ------------------------------------------------------------
    local_cbf_slack_sum = _safe_sum(df, "local_cbf_slack", eval_slice)
    local_cbf_slack_max = _safe_max(df, "local_cbf_slack", eval_slice)

    remote_mpc_slack_sum = _safe_sum(df, "mpc_cbf_slack", eval_slice)
    remote_mpc_slack_max = _safe_max(df, "mpc_cbf_slack", eval_slice)

    mean_local_cbf_solve_time = _safe_mean(df, "local_cbf_solvetime", eval_slice)
    max_local_cbf_solve_time = _safe_max(df, "local_cbf_solvetime", eval_slice)

    mean_mpc_solve_time = _safe_mean(df, "mpc_solvetime", eval_slice)
    max_mpc_solve_time = _safe_max(df, "mpc_solvetime", eval_slice)

    # Prediction error only at steps where prediction is recorded.
    z_real = np.hstack([q, dq])
    z_pred = np.hstack([q_pred, dq_pred])
    pred_valid = np.all(np.isfinite(z_pred), axis=1)

    if np.any(pred_valid):
        pred_err = np.linalg.norm(z_pred[pred_valid] - z_real[pred_valid], axis=1)
        mean_prediction_error = float(np.nanmean(pred_err))
        max_prediction_error = float(np.nanmax(pred_err))
    else:
        mean_prediction_error = np.nan
        max_prediction_error = np.nan

    # Disturbance magnitude
    if not np.all(np.isnan(w)):
        w_norm = np.linalg.norm(w, axis=1)
        mean_disturbance_norm = float(np.nanmean(w_norm[eval_slice]))
        max_disturbance_norm = float(np.nanmax(w_norm[eval_slice]))
    else:
        mean_disturbance_norm = np.nan
        max_disturbance_norm = np.nan

    # ------------------------------------------------------------
    # Final metrics dictionary
    # ------------------------------------------------------------
    metrics = {
        "architecture": arch_name,

        # Success / tracking
        "reached_goal": bool(reached_goal),
        "time_to_goal": time_to_goal,
        "steps_to_goal": n_eval_steps,
        "final_error_at_eval_end": float(final_error[reach_idx]),
        "mean_ee_error_to_current_ref": float(np.nanmean(ee_error_to_current_ref[eval_slice])),
        "max_ee_error_to_current_ref": float(np.nanmax(ee_error_to_current_ref[eval_slice])),
        "mean_ee_error_to_final_target": float(np.nanmean(ee_error_to_final_target[eval_slice])),
        "max_ee_error_to_final_target": float(np.nanmax(ee_error_to_final_target[eval_slice])),

        # Cost
        "total_cost": total_cost,
        "cost_per_step": cost_per_step,

        # Safety
        "stayed_safe": stayed_safe,
        "min_cbf_value": safety["min_cbf_value"],
        "min_signed_clearance": safety["min_signed_clearance"],
        "cbf_violation_count": safety["cbf_violation_count"],
        "violation_step_count": safety["violation_step_count"],
        "max_violation_depth": safety["max_violation_depth"],

        # Torque/control
        "max_abs_control_input": max_abs_control_input,
        "max_abs_control_input_change": max_abs_control_input_change,
        "max_control_input_change_norm": max_control_input_change_norm,

        # Acceleration command
        "max_abs_ddq_command": max_abs_ddq_command,
        "max_abs_ddq_command_change": max_abs_ddq_command_change,
        "max_ddq_command_change_norm": max_ddq_command_change_norm,

        # CBF activation
        "local_cbf_activation_count": local_activation_count,
        "remote_cbf_activation_count": remote_activation_count,
        "local_cbf_activation_rate": local_activation_rate,
        "remote_cbf_activation_rate": remote_activation_rate,

        # Solver feasibility/failures
        "local_cbf_failure_count": local_failure_count,
        "remote_mpc_failure_count": remote_failure_count,
        "local_cbf_failure_rate": local_failure_rate,
        "remote_mpc_failure_rate": remote_failure_rate,

        # Slack
        "local_cbf_slack_sum": local_cbf_slack_sum,
        "local_cbf_slack_max": local_cbf_slack_max,
        "remote_mpc_slack_sum": remote_mpc_slack_sum,
        "remote_mpc_slack_max": remote_mpc_slack_max,

        # Computation time
        "mean_local_cbf_solve_time": mean_local_cbf_solve_time,
        "max_local_cbf_solve_time": max_local_cbf_solve_time,
        "mean_mpc_solve_time": mean_mpc_solve_time,
        "max_mpc_solve_time": max_mpc_solve_time,

        # Prediction/noise
        "mean_prediction_error": mean_prediction_error,
        "max_prediction_error": max_prediction_error,
        "mean_disturbance_norm": mean_disturbance_norm,
        "max_disturbance_norm": max_disturbance_norm,

        # Normalization bases
        "evaluated_steps": n_eval_steps,
        "local_cbf_valid_steps": local_valid_count,
        "remote_mpc_valid_steps": remote_valid_count,
    }

    json_path = os.path.join(output_dir, "run_metrics.json")
    csv_path = os.path.join(output_dir, "run_metrics.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    return metrics

def print_run_summary(metrics: dict, arch_name: str | None = None,
    elapsed_time: float | None = None,) -> None:
    """
    Print a compact architecture-aware summary after a single simulation run.

    Parameters
    ----------
    metrics:
        Metrics dictionary returned by post_process_single_run(...).

    arch_name:
        Optional architecture name. If omitted, metrics["architecture"] is used.
    """

    def fmt_float(value, unit: str = "", precision: int = 3, nan_text: str = "n/a") -> str:
        try:
            if value is None:
                return nan_text
            value = float(value)
            if not np.isfinite(value):
                return nan_text
            return f"{value:.{precision}f}{unit}"
        except (TypeError, ValueError):
            return nan_text

    def fmt_int(value, nan_text: str = "n/a") -> str:
        try:
            if value is None:
                return nan_text
            value = float(value)
            if not np.isfinite(value):
                return nan_text
            return str(int(value))
        except (TypeError, ValueError):
            return nan_text

    def fmt_bool(value) -> str:
        return "yes" if bool(value) else "no"

    arch = arch_name or metrics.get("architecture", "Unknown")

    reached_goal = metrics.get("reached_goal", False)
    stayed_safe = metrics.get("stayed_safe", False)

    print("\n" + "=" * 72)
    print(f"Run summary: {arch}")
    print("=" * 72)

    print(
        f"Goal reached: {fmt_bool(reached_goal)}"
        f" | time: {fmt_float(metrics.get('time_to_goal'), ' s')}"
        f" | steps: {fmt_int(metrics.get('steps_to_goal'))}"
    )

    print(
        f"Safety: {'safe' if stayed_safe else 'VIOLATED'}"
        f" | min clearance: {fmt_float(metrics.get('min_signed_clearance'), ' m', 4)}"
        f" | violations: {fmt_int(metrics.get('violation_step_count'))} steps"
    )

    print(
        f"Cost: total {fmt_float(metrics.get('total_cost'), precision=2)}"
        f" | per step {fmt_float(metrics.get('cost_per_step'), precision=4)}"
    )

    print(
        f"Control: max |tau| {fmt_float(metrics.get('max_abs_control_input'), ' Nm', 3)}"
        f" | max |Δtau| {fmt_float(metrics.get('max_abs_control_input_change'), ' Nm', 3)}"
        f" | max |ddq| {fmt_float(metrics.get('max_abs_ddq_command'), ' rad/s²', 3)}"
    )

    # Architecture-aware CBF information
    if arch in {"LocalCBF", "Combined"}:
        print(
            f"Local CBF: activations {fmt_int(metrics.get('local_cbf_activation_count'))}"
            f" ({fmt_float(100.0 * metrics.get('local_cbf_activation_rate', np.nan), '%', 2)})"
            f" | failures {fmt_int(metrics.get('local_cbf_failure_count'))}"
            f" ({fmt_float(100.0 * metrics.get('local_cbf_failure_rate', np.nan), '%', 2)})"
            f" | max slack {fmt_float(metrics.get('local_cbf_slack_max'), precision=3)}"
        )

    if arch in {"RemoteMPC-CBF", "Combined"}:
        print(
            f"Remote MPC-CBF: activations {fmt_int(metrics.get('remote_cbf_activation_count'))}"
            f" ({fmt_float(100.0 * metrics.get('remote_cbf_activation_rate', np.nan), '%', 2)})"
            f" | failures {fmt_int(metrics.get('remote_mpc_failure_count'))}"
            f" / {fmt_int(metrics.get('remote_mpc_valid_steps'))} calls"
            f" ({fmt_float(100.0 * metrics.get('remote_failure_rate', metrics.get('remote_mpc_failure_rate', np.nan)), '%', 2)})"
            f" | max slack {fmt_float(metrics.get('remote_mpc_slack_max'), precision=3)}"
        )

    # Solver time information only where relevant
    solver_parts = []

    if arch in {"RemoteMPC-CBF", "Combined", "Nominal", "LocalCBF"}:
        solver_parts.append(
            f"MPC mean/max: "
            f"{fmt_float(1000.0 * metrics.get('mean_mpc_solve_time', np.nan), ' ms', 1)} / "
            f"{fmt_float(1000.0 * metrics.get('max_mpc_solve_time', np.nan), ' ms', 1)}"
        )

    if arch in {"LocalCBF", "Combined"}:
        solver_parts.append(
            f"Local CBF mean/max: "
            f"{fmt_float(1000.0 * metrics.get('mean_local_cbf_solve_time', np.nan), ' ms', 1)} / "
            f"{fmt_float(1000.0 * metrics.get('max_local_cbf_solve_time', np.nan), ' ms', 1)}"
        )

    if solver_parts:
        print("Solve time: " + " | ".join(solver_parts))

    # Optional prediction/noise diagnostics, printed only if available
    mean_pred_err = metrics.get("mean_prediction_error", np.nan)
    max_pred_err = metrics.get("max_prediction_error", np.nan)

    if np.isfinite(float(mean_pred_err)) if mean_pred_err is not None else False:
        print(
            f"Prediction error: mean {fmt_float(mean_pred_err, precision=4)}"
            f" | max {fmt_float(max_pred_err, precision=4)}"
        )
    if elapsed_time is not None:
        print(f"Wall-clock simulation time: {fmt_float(elapsed_time, ' s', 2)}")

    print("=" * 72 + "\n")