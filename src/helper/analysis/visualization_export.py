"""Visualization and export utilities for experiment results."""

from __future__ import annotations

import csv
import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from sim_module.params import SystemParams
from helper.plot_code import create_3dof_animation, plot_3dof_results, save_trajectory_to_csv


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
            create_3dof_animation(t_result, ee_states, joint_angles, pr, params, detailed_data=detailed_data)
            print(f"  Animation saved: {params.out_anim}")
    except Exception as exc:
        print(f"  Visualization error: {exc}")


def export_trajectory_data(
    scenario_name: str,
    controller_name: str,
    noise_name: str,
    time_suffix: str,
    out_dir: str,
    t_result: np.ndarray,
    joint_angles: np.ndarray,
    ee_states: np.ndarray,
    U_alpha: np.ndarray,
    pr: np.ndarray,
    dpr: np.ndarray,
    metrics_dict: dict,
    params: SystemParams,
    measurement_noise: np.ndarray,
    process_noise: np.ndarray,
    cbf_activations: np.ndarray,
    cbf_modifications: np.ndarray,
    remote_slacks: np.ndarray,
    local_slacks: np.ndarray,
) -> None:
    print("Saving trajectory data to CSV...")
    try:
        csv_filename_detailed = (
            f"{scenario_name}_{controller_name}_{noise_name}_{time_suffix}_trajectory.csv"
        )
        csv_path_detailed = os.path.join(out_dir, csv_filename_detailed)
        save_trajectory_to_csv(
            t_result,
            joint_angles,
            ee_states,
            U_alpha,
            pr,
            dpr,
            metrics_dict,
            csv_path_detailed,
            params=params,
            measurement_noise=measurement_noise,
            process_noise=process_noise,
            cbf_activations=cbf_activations,
            cbf_modifications=cbf_modifications,
            remote_slacks=remote_slacks,
            local_slacks=local_slacks,
        )
        print(f"  CSV data saved: {csv_path_detailed}")

        if noise_name == "NoNoise":
            csv_filename_simple = f"{scenario_name}_{controller_name}_trajectory.csv"
            csv_path_simple = os.path.join(out_dir, csv_filename_simple)
            save_trajectory_to_csv(
                t_result,
                joint_angles,
                ee_states,
                U_alpha,
                pr,
                dpr,
                metrics_dict,
                csv_path_simple,
                params=params,
                measurement_noise=measurement_noise,
                cbf_activations=cbf_activations,
                cbf_modifications=cbf_modifications,
                remote_slacks=remote_slacks,
                local_slacks=local_slacks,
            )
            print(f"  CSV data saved (backward compatibility): {csv_path_simple}")

            csv_filename_time = (
                f"{scenario_name}_{controller_name}_{time_suffix}_trajectory.csv"
            )
            csv_path_time = os.path.join(out_dir, csv_filename_time)
            save_trajectory_to_csv(
                t_result,
                joint_angles,
                ee_states,
                U_alpha,
                pr,
                dpr,
                metrics_dict,
                csv_path_time,
                params=params,
                measurement_noise=measurement_noise,
                cbf_activations=cbf_activations,
                cbf_modifications=cbf_modifications,
                remote_slacks=remote_slacks,
                local_slacks=local_slacks,
            )
            print(f"  CSV data saved (time suffix): {csv_path_time}")
    except Exception as exc:
        print(f"  CSV export error: {exc}")


def write_results_summary(all_results: List[dict], out_dir: str) -> None:
    if not all_results:
        return
    
    # Add timestamp to filename to prevent overwriting
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"3dof_obstacle_training_results_{timestamp}.csv"
    csv_path = os.path.join(out_dir, csv_filename)
    
    fieldnames = sorted(set().union(*(record.keys() for record in all_results)))

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {csv_path}")


def print_full_summary(
    all_results: Sequence[dict],
    scenarios: Sequence[dict],
    noise_configs: Sequence[dict],
    controllers: Sequence[Tuple[str, dict]],
    out_dir: str | None = None,
) -> None:
    print("\n" + "=" * 60)
    print("3DOF Obstacle Avoidance Training Summary")
    print("=" * 60)

    success_count = sum(1 for record in all_results if record.get("success", False))
    print(f"Successful experiments: {success_count}/{len(all_results)}")

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        for noise_cfg in noise_configs:
            print(f"  {noise_cfg['name']}:")
            scenario_results = [
                record
                for record in all_results
                if f"{scenario['name']}_{noise_cfg['name']}" in record.get("scenario_name", "")
            ]
            for record in scenario_results:
                status = "✓" if record.get("success", False) else "✗"
                print(
                    f"    {record['controller_name']}: {status} "
                    f"(clearance:{record.get('min_clearance', 0):.3f}m, "
                    f"RMSE:{record.get('tracking_rmse', 0):.3f}m)"
                )

    print("\n" + "=" * 80)
    print("Noise Robustness Analysis - Performance Comparison")
    print("=" * 80)

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 80)
        for controller_name, _ in controllers:
            print(f"\n  Controller: {controller_name}")
            no_noise = [
                record
                for record in all_results
                if f"{scenario['name']}_NoNoise" in record.get("scenario_name", "")
                and record.get("controller_name") == controller_name
            ]
            with_noise = [
                record
                for record in all_results
                if f"{scenario['name']}_LightNoise" in record.get("scenario_name", "")
                and record.get("controller_name") == controller_name
            ]

            if no_noise and with_noise:
                nn = no_noise[0]
                wn = with_noise[0]

                print(f"    {'Metric':<30} {'No Noise':>15} {'Light Noise':>15} {'Δ%':>10}")
                print(f"    {'-'*70}")
                print(f"    {'Success':<30} {nn['success']:>15} {wn['success']:>15} {'-':>10}")

                def safe_percent(new_val: float, old_val: float) -> str:
                    if old_val == 0:
                        return "N/A" if new_val == 0 else "∞"
                    return f"{((new_val - old_val) / abs(old_val) * 100):>9.1f}%"

                print(
                    f"    {'Min Clearance (m)':<30} {nn['min_clearance']:>15.4f} "
                    f"{wn['min_clearance']:>15.4f} {safe_percent(wn['min_clearance'], nn['min_clearance']):>10}"
                )
                print(
                    f"    {'Tracking RMSE (m)':<30} {nn['tracking_rmse']:>15.4f} "
                    f"{wn['tracking_rmse']:>15.4f} {safe_percent(wn['tracking_rmse'], nn['tracking_rmse']):>10}"
                )
                print(
                    f"    {'Collision Count':<30} {nn['collision_count']:>15} "
                    f"{wn['collision_count']:>15} {'-':>10}"
                )
                if wn.get("measurement_error_mean", 0) > 0:
                    print(
                        f"    {'Measurement Error (rad)':<30} {'-':>15} "
                        f"{wn['measurement_error_mean']:>15.6f} {'-':>10}"
                    )
                    if wn.get("ee_measurement_error_mean") is not None:
                        print(
                            f"    {'EE Position Error (m)':<30} {'-':>15} "
                            f"{wn['ee_measurement_error_mean']:>15.6f} {'-':>10}"
                        )
                if "control_effort_mean" in nn and "control_effort_mean" in wn:
                    print(
                        f"    {'Control Effort Mean':<30} {nn['control_effort_mean']:>15.4f} "
                        f"{wn['control_effort_mean']:>15.4f} "
                        f"{safe_percent(wn['control_effort_mean'], nn['control_effort_mean']):>10}"
                    )
                if "qp_solve_time_mean" in nn and "qp_solve_time_mean" in wn:
                    print(
                        f"    {'Solve Time Mean (ms)':<30} {nn['qp_solve_time_mean']*1000:>15.2f} "
                        f"{wn['qp_solve_time_mean']*1000:>15.2f} "
                        f"{safe_percent(wn['qp_solve_time_mean'], nn['qp_solve_time_mean']):>10}"
                    )
            else:
                print(f"    Missing comparison data for {controller_name}")

    if out_dir:
        print(f"\nAll files saved in: {out_dir}/")


def generate_comparison_animations(
    scenarios: Sequence[dict],
    noise_configs: Sequence[dict],
    controllers: Sequence[Tuple[str, dict]],
    out_dir: str,
) -> None:
    print("\n" + "=" * 60)
    print("Generate Multi-trajectory Comparison Animation")
    print("=" * 60)

    try:
        from generate_workspace_animations import create_multi_trajectory_comparison_gif

        for scenario in scenarios:
            scenario_name = scenario["name"]
            for noise_cfg in noise_configs:
                noise_name = noise_cfg["name"]
                print(
                    f"\nGenerating 4-trajectory comparison animation for {scenario_name} ({noise_name})..."
                )
                csv_paths: List[str] = []
                titles: List[str] = []
                time_suffix = f"T_{int(scenario['total_time'])}s"
                for controller_name, _ in controllers:
                    csv_file = os.path.join(
                        out_dir,
                        f"{scenario_name}_{controller_name}_{noise_name}_{time_suffix}_trajectory.csv",
                    )
                    if os.path.exists(csv_file):
                        csv_paths.append(csv_file)
                        titles.append(controller_name)
                    else:
                        print(f"  Warning: {csv_file} not found")
                if len(csv_paths) >= 2:
                    output_anim = os.path.join(
                        out_dir,
                        f"{scenario_name}_{noise_name}_4_strategies_comparison.gif",
                    )
                    try:
                        create_multi_trajectory_comparison_gif(
                            csv_paths,
                            output_gif_path=output_anim,
                            obstacle_scene=scenario["obstacle_scene"],
                            skip_frames=15,
                            fps=30,
                            titles=titles,
                        )
                        print(f"  ✓ 4-trajectory comparison animation saved: {output_anim}")
                    except Exception as exc:
                        print(f"  ✗ Animation generation failed: {exc}")
                else:
                    print(
                        f"  Skip {scenario_name} ({noise_name}) (insufficient CSV files)"
                    )

            print(f"\nGenerating backward compatible animation for {scenario_name}...")
            csv_paths = []
            titles = []
            for controller_name, _ in controllers:
                csv_file = os.path.join(out_dir, f"{scenario_name}_{controller_name}_trajectory.csv")
                if os.path.exists(csv_file):
                    csv_paths.append(csv_file)
                    titles.append(controller_name)
                else:
                    print(f"  Warning: {csv_file} not found")
            if len(csv_paths) >= 2:
                output_anim = os.path.join(out_dir, f"{scenario_name}_4_strategies_comparison.gif")
                try:
                    create_multi_trajectory_comparison_gif(
                        csv_paths,
                        output_gif_path=output_anim,
                        obstacle_scene=scenario["obstacle_scene"],
                        skip_frames=15,
                        fps=30,
                        titles=titles,
                    )
                    print(f"  ✓ Backward compatible animation saved: {output_anim}")
                except Exception as exc:
                    print(f"  ✗ Backward compatible animation generation failed: {exc}")
    except ImportError as exc:
        print(f"Animation generation module import failed: {exc}")
    except Exception as exc:
        print(f"Animation generation process error: {exc}")

    print("\nExperiment and animation generation completed!")
