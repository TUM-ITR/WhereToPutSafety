
"""Standalone main script for 20Hz MPC coupled with 200Hz local PD+ controller."""
from pathlib import Path
import sys
import argparse
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
ARCHITECTURES = ["Nominal", "LocalCBF", "MPC-CBF", "Combined"]

ARCHITECTURE_TO_CHOICE = {
    "Nominal": 0,
    "LocalCBF": 1,
    "MPC-CBF": 2,
    "Combined": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run simulations for CBF placement in a networked control system "
            "with a 3-DoF planar robot."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "compare", "monte-carlo", "delay-sweep", "disturbance-sweep"],
        help="Experiment mode to run.",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default="Combined",
        choices=ARCHITECTURES,
        help="Architecture used for single-run mode.",
    )

    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["LocalCBF", "MPC-CBF", "Combined"],
        choices=ARCHITECTURES,
        help=(
            "Architectures used for compare, Monte Carlo, and sweep modes. "
            "The compare mode always runs all architectures."
            "For single mode, use --architecture instead."
        ),
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of Monte Carlo trials for Monte Carlo and sweep modes.",
    )

    parser.add_argument(
        "--delays",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Delay values used for delay-sweep mode. "
            "If omitted, values from params are used."
        ),
    )

    parser.add_argument(
        "--disturbances",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Disturbance bounds used for disturbance-sweep mode. "
            "If omitted, values from params are used."
        ),
    )

    parser.add_argument(
        "--plot",
        dest="make_plots",
        action="store_true",
        help="Generate plots for individual simulation runs.",
    )

    parser.add_argument(
        "--no-plot",
        dest="make_plots",
        action="store_false",
        help="Do not generate plots for individual simulation runs.",
    )

    parser.add_argument(
        "--animate",
        dest="make_animation",
        action="store_true",
        help="Generate animation for individual simulation runs.",
    )

    parser.add_argument(
        "--no-animate",
        dest="make_animation",
        action="store_false",
        help="Do not generate animation for individual simulation runs.",
    )

    parser.add_argument(
        "--summary-plot",
        dest="make_summary_plots",
        action="store_true",
        help="Generate summary plots for compare, Monte Carlo, and sweep modes.",
    )

    parser.add_argument(
        "--no-summary-plot",
        dest="make_summary_plots",
        action="store_false",
        help="Do not generate summary plots.",
    )

    parser.add_argument(
        "--no-disturbances",
        dest="with_disturbances",
        action="store_false",
        help="Disable disturbances for single and compare modes.",
    )

    parser.add_argument(
        "--short",
        action="store_true",
        help="Run a short debug simulation.",
    )

    parser.set_defaults(
        make_plots=None,
        make_animation=None,
        make_summary_plots=None,
        with_disturbances=True,
    )

    return parser.parse_args()

def apply_cli_overrides(params: SystemParams, args: argparse.Namespace) -> SystemParams:
    """Apply command-line overrides to the parameter object."""

    if args.short:
        # Choose values that make a smoke test finish quickly.
        params.total_time = min(getattr(params, "total_time", 20.0), 1.0)

        if hasattr(params, "N_horizon"):
            params.N_horizon = min(params.N_horizon, 10)

    return params


def get_default_delays(params: SystemParams) -> list[int]:
    """Return default delay sweep values from params, or a fallback."""
    if hasattr(params, "delay_sweep_values"):
        return list(params.delay_sweep_values)
    if hasattr(params, "delays"):
        return list(params.delays)
    return [0, 3, 7, 10]


def get_default_disturbances(params: SystemParams) -> list[float]:
    """Return default disturbance sweep values from params, or a fallback."""
    if hasattr(params, "disturbance_sweep_values"):
        return list(params.disturbance_sweep_values)
    if hasattr(params, "disturbance_bounds"):
        return list(params.disturbance_bounds)
    return [0.005, 0.01, 0.02, 0.03]

def resolve_plot_defaults(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    """
    Resolve plotting defaults.

    Defaults:
    - single: plots on, animation on, no summary plot
    - compare: plots on, animation off, summary plot on
    - monte-carlo/sweeps: individual plots off, animation off, summary plot on
    """

    if args.mode == "single":
        make_plots_default = True
        make_animation_default = True
        make_summary_plots_default = False
    elif args.mode == "compare":
        make_plots_default = True
        make_animation_default = False
        make_summary_plots_default = True
    else:
        make_plots_default = False
        make_animation_default = False
        make_summary_plots_default = True

    make_plots = make_plots_default if args.make_plots is None else args.make_plots
    make_animation = (
        make_animation_default if args.make_animation is None else args.make_animation
    )
    make_summary_plots = (
        make_summary_plots_default
        if args.make_summary_plots is None
        else args.make_summary_plots
    )

    return make_plots, make_animation, make_summary_plots

def main() -> None:
    args = parse_args()

    params = set_temp_params(SystemParams())
    params = apply_cli_overrides(params, args)

    make_plots, make_animation, make_summary_plots = resolve_plot_defaults(args)

    print(f"Running mode: {args.mode}")

    if args.mode == "single":
        print(f"Architecture: {args.architecture}")

        run_specific_architecture(
            params=params,
            choice=ARCHITECTURE_TO_CHOICE[args.architecture],
            with_disturbances=args.with_disturbances,
            make_plots=make_plots,
            make_animation=make_animation,
        )

    elif args.mode == "compare":
        print(f"Architectures: {args.architectures}")

        run_all_architectures(
            params=params,
            with_disturbances=args.with_disturbances,
            make_plots=make_plots,
            make_animation=make_animation,
            make_summary_plots=make_summary_plots,
        )

    elif args.mode == "monte-carlo":
        print(f"Architectures: {args.architectures}")
        print(f"Trials: {args.trials}")

        run_monte_carlo(
            params=params,
            num_trials=args.trials,
            architectures=args.architectures,
            make_plots=make_plots,
            make_animation=make_animation,
            make_summary_plots=make_summary_plots,
        )

    elif args.mode == "delay-sweep":
        delays = args.delays if args.delays is not None else get_default_delays(params)

        print(f"Architectures: {args.architectures}")
        print(f"Trials per delay: {args.trials}")
        print(f"Delays: {delays}")

        run_delay_sweep(
            params=params,
            delays=delays,
            num_trials=args.trials,
            architectures=args.architectures,
            make_plots=make_plots,
            make_animation=make_animation,
            make_summary_plots=make_summary_plots,
        )

    elif args.mode == "disturbance-sweep":
        disturbances = (
            args.disturbances
            if args.disturbances is not None
            else get_default_disturbances(params)
        )

        print(f"Architectures: {args.architectures}")
        print(f"Trials per disturbance bound: {args.trials}")
        print(f"Disturbance bounds: {disturbances}")

        run_disturbance_sweep(
            params=params,
            disturbance_bounds=disturbances,
            num_trials=args.trials,
            architectures=args.architectures,
            make_plots=make_plots,
            make_animation=make_animation,
            make_summary_plots=make_summary_plots,
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


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
    main ()