# Simulation Code: Where To Put Safety? Control Barrier Function Placement in Networked Control Systems

This repository contains the Python code accompanying the paper

> [Where To Put Safety? Control Barrier Function Placement in Networked Control Systems](https://arxiv.org/abs/2603.29792)
> by S. Beger, Y. Chen and S. Hirche.

submitted to L-CSS Letters 2026.

<p align="center">
<img src="assets/robot.gif" alt="GIF of a robot simulation from this repo" width="600"/>
</p>

This repository simulates a 3 DoF planar robot conducting a pick-and-place task with obstacle avoidance.
A model predictive controller creates a desired path at 20 Hz, which a local PD+ controller tracks at 200 Hz.
We compare four safety-filter architectures:

* `Nominal`: no safety filter is considered.
* `LocalCBF`: a myopic CBF is placed locally before the PD+ controller.
* `MPC-CBF`: the remote MPC includes the CBF constraints.
* `Combined`: remote MPC-CBF and local myopic CBF are used together.

<p align="center">
<img src="assets/SystemOverview.svg" alt="Considered Networked Control System" width="450"/>
</p>
<p align="center">
  Overview of the considered networked predictive control system.
  We analyse the robustness-performance tradeoff following from the different CBF placements in the control architecture.
</p>

## Quick Start Guide

Clone the repository and create a clean Python environment:

```bash
git clone https://github.com/TUM-ITR/WhereToPutSafety.git
cd WhereToPutSafety
python -m venv .venv
```

Activate the environment.

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the default example:

```bash
python main.py
```

The default example runs the `Combined` architecture with bounded disturbances. Simulation outputs are written to timestamped folders in `data/`. Depending on the selected plotting options, generated figures or animations are written to the corresponding output folder.

## Running Simulations

The main simulation routines are implemented in `src/sim_module/sim_runner.py` and imported in `main.py`.

### Run a single architecture

Edit the bottom of `main.py` and select one architecture:

```python
params = set_temp_params(SystemParams())
run_specific_architecture(params, choice=3, with_disturbances=True)
```

The architecture choices are:

| Choice | Architecture    |
| -----: | --------------- |
|    `0` | `Nominal`       |
|    `1` | `LocalCBF`      |
|    `2` | `RemoteMPC-CBF` |
|    `3` | `Combined`      |

### Compare all architectures

```python
params = set_temp_params(SystemParams())
run_all_architectures(params, with_disturbances=True)
```

### Run Monte Carlo simulations

```python
params = set_temp_params(SystemParams())
run_monte_carlo(
    params=params,
    num_trials=50,
    architectures=["LocalCBF", "RemoteMPC-CBF", "Combined"],
)
```

### Run a disturbance sweep

```python
params = set_temp_params(SystemParams())
run_disturbance_sweep(
    params=params,
    disturbance_bounds=[0.005, 0.01, 0.02, 0.03, 0.04],
    num_trials=10,
    architectures=["LocalCBF", "RemoteMPC-CBF", "Combined"],
)
```

### Run a delay sweep

```python
params = set_temp_params(SystemParams())
run_delay_sweep(
    params=params,
    delays=range(1, 11),
    num_trials=10,
    architectures=["LocalCBF", "RemoteMPC-CBF", "Combined"],
)
```

## Repository Organization

```text
WhereToPutSafety/
├── README.md
├── requirements.txt
├── main.py
├── assets/
│   ├── robot.gif
│   └── SystemOverview.svg
├── data/
│   └── .gitignore
├── plots/
│   └── .gitignore
└── src/
    ├── controller_module/
    │   ├── local_cbf.py
    │   ├── pd_plus.py
    │   └── remote_mpc_cbf.py
    ├── plant_module/
    │   └── robot_dynamics.py
    ├── predictor_module/
    │   └── state_predictor.py
    ├── sim_module/
    │   ├── data_recorder.py
    │   ├── generate_disturbance.py
    │   ├── obstacles.py    
    │   ├── params.py
    │   └── sim_runner.py
    └── helper/
        ├── analysis/
        │   ├── plots_for_paper.py
        │   ├── post_processing.py
        │   ├── post_processing_MonteCarlo.py
        │   └── visualization_export.py
        └── plot_code/
            ├── data_export.py
            ├── generate_workspace_animations.py
            └── plotting_3dof.py
```

The main modules are:

* `controller_module`: remote MPC-CBF, local CBF, and local PD+ tracking control.
* `disturbance_module`: generation of bounded disturbance sequences.
* `plant_module`: 3 DoF planar robot dynamics and kinematics.
* `predictor_module`: delay compensation through state prediction.
* `sim_module`: simulation loop, obstacle definitions, and data recording.
* `helper/analysis`: post-processing and paper-plot routines.
* `helper/plot_code`: plotting, animation, and trajectory export utilities.

Generated simulation data is intentionally excluded from version control. The `data/` and `plots/` folders are kept as output locations, but their generated contents should not be committed.

## Installation and Requirements

The code was tested with Python 3.10. It uses the following main packages:

* `numpy`
* `scipy`
* `pandas`
* `matplotlib`
* `casadi`
* `cvxpy`
* `openpyxl`
* `pillow`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` is:

```text
numpy
scipy
pandas
matplotlib
casadi
cvxpy
openpyxl
pillow
```

The simulations can be computationally demanding because the remote MPC and the local CBF are solved repeatedly along the trajectory. For a quick test, reduce `params.total_time`, `params.N_horizon`, or the number of Monte Carlo trials.

## Reference

If you use this software in your research, please cite:

```bibtex
@article{WhereToPutSafety2026,
 title={Where To Put Safety? Control Barrier Function Placement in Networked Control Systems},
 author={Beger, Severin and Chen, Yuling and Hirche, Sandra},
 journal={arXiv preprint arXiv:2603.29792},
 year={2026}
}
```
