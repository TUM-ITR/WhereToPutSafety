from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
import os
from time import time
import numpy as np
from typing import Optional, Tuple, List, Any, Dict
from datetime import datetime
import json

import numpy as np

from sim_module.obstacles import Obstacle, ObstacleScene



@dataclass
class SystemParams:
    """Parameters for 3-DOF robot CBF-MPC simulations"""

    # ==================== Controller Strategy ====================
    strategy_name: str = ""  # Name of control strategy (e.g., "Combined_CBF", "Combined_CBF_before")

    # ==================== Output ====================
    # print the current time for output directory naming
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir: str = f"data/simulation_results_{current_time}"
    scenario_name: str = "default"
    out_anim: Optional[str] = None
    out_plot: Optional[str] = None
    # ==================== Constraints / Obstacles ====================
    obstacle_scene: Optional[Any] = None   # set by caller
    
    # Dual obstacle centers: [(x1,y1), (x2,y2)]
    # **MODIFY HERE** to change obstacle positions in workspace
    scenario_obstacle_centers: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.3), (0.4, 0.38)]
    )
    # no-stuck version: [(0.0, 0.3), (0.4, 0.38)]
    # get stucked version: [(0.0, 0.3), (0.4, 0.3)]
    obstacle_default_radius: list = field(default_factory=lambda: [0.2, 0.16])
    distance_margin: float = 0.005

    # ==================== Time / Horizon ====================
    TA: float = 0.005           # Time step for low level control loop (5 ms = 200 Hz)
    mpc_TA: float = 0.05        # Time step for remote control loop (50 ms = 20 Hz)
    mpc_ratio = int(mpc_TA/TA)       # Control Loop Timing Ratio
    total_time: float = 20.0    # Total simulation time in seconds (steps = total_time / TA)
    N_horizon: int = 30         # MPC prediction horizon
    
    # ==================== Waypoint Navigation ====================
    # Robot navigates through waypoints sequentially, stays at final waypoint after reaching it
    # Narrow passage test: wp2 near obstacles to trigger navigation challenge
    waypoints: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.18, -0.12), (0.26, 0.19), (0.77, 0.39)]
    )
    # default_factory=lambda: [(0.18, -0.12), (0.29, 0.23), (0.7, 0.4)]
    # larger noise case: [(0.18, -0.12), (0.26, 0.19), (0.77, 0.39)]
    use_reference_trajectory: bool = False  # If True, use the provided pr/dpr arrays directly instead of waypoint switching.
    target_tol = 0.005 #tolerance for having tracked a point quickly
    target_enter_tol: float =0.05 #tolerance for starting the "tracking timer"
    target_hold_tol: float =0.07 # tolerance for staying close to a target -> tracking timer continues
    target_velocity_tol: float = 0.1 #velocity l

    target_hold_time: float = 3.0 #hold time for staying close to target
    settle_history_len: float = 5

    # ==================== Robot Links (3-DOF planar arm) ====================
    L1: float = 0.40            # First link length
    L2: float = 0.30            # Second link length
    L3: float = 0.18            # Third link length
    
    # ==================== 3-DOF Joint-Space Dynamics ====================
    # Non-uniform inertia: Joint1 (base) rotates entire arm, Joint3 (wrist) only tip
    # Realistic ratio ~10:4:1 for 0.88m arm (L1=0.4m, L2=0.3m, L3=0.18m)
    # Mass distribution: Link1 ~0.8kg, Link2 ~0.6kg, Link3 ~0.3kg
    link_masses: Tuple[float, float, float] = (0.8, 0.6, 0.3)
    gravity_accel: float = 9.81
    
    # Non-uniform damping: larger joints have more friction
    # Typical ratio ~5:3:1 for small arm with low-friction bearings
    Dq3: np.ndarray = field(default_factory=lambda: np.diag([0.5, 0.3, 0.1]))  # N·m·s (decreasing from base to tip)
    
    # Zero stiffness: rigid joints without spring-back mechanism
    Kq3: np.ndarray = field(default_factory=lambda: np.diag([0.0, 0.0, 0.0]))
    
    # Preferred initial pose to avoid immediate obstacle collisions
    # Configured for EE at waypoint 1: (0.18, -0.12)
    # Optimized via inverse kinematics with obstacle avoidance
    initial_joint_angles: Tuple[float, float, float] = (
        -1.3885202265986305,
        1.9690157494651153,
        1.9117361992767938,
    )

    # ==================== CBF core parameters ====================
    # For DT-ECBF (Discrete-Time): h(x_{k+1}) - h(x_k) ≥ -γh(x_k)
    # Equivalently: h(x_{k+1}) ≥ (1-γ)h(x_k)
    # gamma_cbf represents γ ∈ (0, 1]: larger γ → faster convergence to safe set
    # Typical values: γ ∈ [0.5, 0.9] for good balance between safety and performance
    gamma_cbf: float = 0.3  
    local_cbf_slack_weight: float = 1e10 #allows for minor violations of the safety constraint without going infeasible

    # ==================== Unified CBF parameters (Local & Remote) ====================
    cbf_link_samples: int = 3           
    cbf_safety_margin: float = 0.005
    
    # Local CBF Activation Detection
    cbf_activation_threshold: float = 0.001  # Threshold for detecting CBF activation (L2 norm of modification)

    # ==================== MPC Limits ====================
    umax_joint: float = 10.0    # Joint torque limit (N·m). Small servo motor typical range
    amax_joint: float = 20.0    # Joint acceleration limit (rad/s^2)
    max_joint_velocity: float = 1e6  # Joint velocity limit (rad/s, conservative for safety)
    du_max = 20               # rate constraint

    # ==================== MPC weights ====================
    mpc_reference_space: str = "joint"  # "joint" recommended, "task" optional later

    # MPC configuration
    
    mpc_smoothness_weight: float = 50.0  # Encourage smooth torque variations
    mpc_use_slack: bool = False #decision variable, whether to use slack in the MPC
    mpc_use_CBF: bool = False # decision variable, indicating whether the CBF is used in MPC

    mpc_slack_weight: float = 1e10 #only used in combined version

    # Only "Task space"
    Qe: float = 80.0          # Stage position error weight (intermediate steps)
    Qe_terminal: float = 800.0 # Terminal position error weight (final step) 
    Qv: float = 2.0            # Joint velocity damping penalty (reduced to move faster)
    R_mpc: float = 30.0         # Control effort penalty (reduced to allow more aggressive control)

    # Only Joint Space
    Qq_mpc: np.ndarray = field(
        default_factory=lambda: np.diag([80.0, 80.0, 40.0])
    )
    Qq_terminal_mpc: np.ndarray = field(
        default_factory=lambda: np.diag([800.0, 800.0, 400.0])
    )

    Qdq_mpc: np.ndarray = field(
        default_factory=lambda: np.diag([3.0, 3.0, 1.5])
    )
    Qdq_terminal_mpc: np.ndarray = field(
        default_factory=lambda: np.diag([30.0, 30.0, 15.0])
    )

    Rddq_mpc: np.ndarray = field(
        default_factory=lambda: np.diag([0.2, 0.2, 0.1])
    )
    Rjerk_mpc: np.ndarray = field(
        default_factory=lambda: np.diag([2.0, 2.0, 1.0])
    )

    # ==================== PD+ nominal controller ====================
    pd_plus_kp_joint: float = 80.0
    pd_plus_kd_joint: float = 22.0

    # ==================== Delay ====================
    tau_known: int = 1                 # compensated communication delay in high level steps (*mpc_TA = 150ms)
    tau_residual: int = 0                 # uncompensated communication delay in high level steps
    tau: int = tau_known+tau_residual         # total roundtrip time delay
    
    # ==================== State Predictor ====================  
    use_state_predictor: bool = True    # Enable state predictor (matches main.py)
    predictor_integration_method: str = "rk4"  # Integration method: "euler" or "rk4"
    
    
    # ==================== Sweep Variables ====================
    disturbance_sweep_values = [0.005, 0.01, 0.02, 0.03]
    delay_sweep_values = [1,3,5,7,10]
    
    # 
    # ==================== Process Noise ====================
    # Model: x_{k+1} = f(x_k, u_k) +g(x_k, u_k) + [0; w] where w affects only velocity
    # Noise configuration
    # Note: Gaussian noise is used for actual value generation, which theoretically has an unbounded max value.
    # To strictly satisfy bounded noise analytical proofs (||w|| <= w_max), the noise magnitude is artificially clipped 
    # to max_norm = process_noise_clip * sqrt(3). Thus clip value should be based exactly on theoretical limits.
    enable_disturbance: bool = True
    torque_disturbance_bound: float = 0.02 #0.1 -> violation local CBF
    torque_disturbance_std: float = torque_disturbance_bound/2

    # ==================== CBF/MPC switches ====================
    use_local_cbf: bool = True         # Local CBF-QP (pre-execution filtering)
    use_remote_cbf: bool = True        # Soft CBF in MPC for each step + terminal (requires corresponding controller implementation)

    # ==================== Tolerances / success ====================
    safety_tolerance: float = 0.005
    
    # ==================== Solver / compute ====================
    solver_preference: List[str] = field(
        default_factory=lambda: ["CLARABEL", "OSQP", "SCS", "ECOS"]
    )
    max_solver_time: float = 2
    random_seed: int = 42


    ################# HELPERS TO SAVE META DATA ########################


    def _to_jsonable(self,value: Any) -> Any:
        """
        Convert common Python / NumPy objects into JSON-serializable objects.

        Uses explicit type tags for objects where reconstruction matters.
        """
        if isinstance(value, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": value.tolist(),
            }

        if isinstance(value, tuple):
            return {
                "__type__": "tuple",
                "data": [self._to_jsonable(v) for v in value],
            }

        if isinstance(value, list):
            return [self._to_jsonable(v) for v in value]

        if isinstance(value, dict):
            return {
                str(k): self._to_jsonable(v)
                for k, v in value.items()
            }

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            return float(value)

        if isinstance(value, (np.bool_,)):
            return bool(value)

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        # Last-resort fallback for objects such as obstacle_scene.
        # We do not reconstruct these automatically.
        return {
            "__type__": "non_serializable",
            "class": value.__class__.__name__,
            "repr": repr(value),
        }


    def _from_jsonable(self,value: Any) -> Any:
        """
        Reconstruct objects serialized by _to_jsonable where possible.
        """
        if isinstance(value, dict):
            value_type = value.get("__type__", None)

            if value_type == "ndarray":
                return np.asarray(value["data"], dtype=float)

            if value_type == "tuple":
                return tuple(self._from_jsonable(v) for v in value["data"])

            if value_type == "non_serializable":
                # Cannot reconstruct arbitrary objects like obstacle_scene.
                return None

            return {
                k: self._from_jsonable(v)
                for k, v in value.items()
            }

        if isinstance(value, list):
            return [self._from_jsonable(v) for v in value]

        return value


    def save_meta_data(self, folder_path: str) -> None:
        """
        Save all dataclass fields of a SystemParams object to:

            folder_path/meta_data.json

        Non-serializable objects are stored as metadata using repr(...)
        and restored as None when loading.
        """
        os.makedirs(folder_path, exist_ok=True)

        meta = {
            "__meta__": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "class_name": self.__class__.__name__,
                "format_version": 1,
            },
            "params": {},
        }

        for f in fields(self):
            name = f.name

            #skip obstacle scene (runtime object)
            if name == "obstacle_scene":
                    meta["params"][name] = None
                    continue

            value = getattr(self, name)
            meta["params"][name] = self._to_jsonable(value)

        file_path = os.path.join(folder_path, "meta_data.json")
        tmp_path = file_path + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, sort_keys=True)

        # Atomic replacement: avoids half-written JSON files if something crashes.
        os.replace(tmp_path, file_path)


    def read_meta_data(self,folder_path: str, rebuild_scenario: bool = True) -> SystemParams:
        """
        Read:

            folder_path/meta_data.json

        If the file exists and can be read, return a SystemParams object
        with the stored fields restored.

        If no file is found or reading fails, return a fresh SystemParams().
        """
        file_path = os.path.join(folder_path, "meta_data.json")

        if not os.path.isfile(file_path):
            print(f"No meta_data.json found in {folder_path}. Returning default SystemParams().")
            params = SystemParams()

            if rebuild_scenario:
                scenario = self._rebuild_scenario()
                
            return params

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            params_dict_raw = meta.get("params", {})
            params_dict = {
                key: self._from_jsonable(value)
                for key, value in params_dict_raw.items()
            }

            # Only pass fields that currently exist in SystemParams.
            # This makes loading robust if you later add/remove parameters.
            valid_field_names = {f.name for f in fields(SystemParams)}
            filtered_params_dict = {
                key: value
                for key, value in params_dict.items()
                if key in valid_field_names
            }
            # Do not restore runtime objects directly.
            # obstacle_scene should be rebuilt from parameters.
            filtered_params_dict["obstacle_scene"] = None

            params = SystemParams(**filtered_params_dict)

            if rebuild_scenario:
                scenario = self._build_default_scenario()
                params.obstacle_scene = scenario["obstacle_scene"]
                params.scenario_name = scenario["name"]

            return params

        except Exception as exc:
            print(f"Could not read meta_data.json from {folder_path}: {exc}")
            print("Returning default SystemParams().")

            params = SystemParams()

            if rebuild_scenario:
                scenario = self._build_default_scenario()
                params.obstacle_scene = scenario["obstacle_scene"]
                params.scenario_name = scenario["name"]

            return params
        
    # OBSTACLES
    def _create_3dof_dual_obstacles(self,
        center1: Tuple[float, float] = (0.0, 0.3),
        center2: Tuple[float, float] = (0.4, 0.38),
        radius1: float = 0.2,
        radius2: float = 0.16,
        ) -> ObstacleScene:
        """Create the fixed dual-circle obstacle scene used by the project."""
        
        scene = ObstacleScene()
        scene.add_obstacle(Obstacle(center=np.array(center1), radius=radius1, type="circle"))
        scene.add_obstacle(Obstacle(center=np.array(center2), radius=radius2, type="circle"))
        return scene
    
    def _build_default_scenario(self) -> Dict[str, object]:
        """Build the single supported dual-circle scenario."""
        centers = self.scenario_obstacle_centers
        radii = self.obstacle_default_radius
        obstacle_scene = self._create_3dof_dual_obstacles(
            center1=tuple(centers[0]),
            center2=tuple(centers[1]),
            radius1=float(radii[0]),
            radius2=float(radii[1]),
        )
        return {
            "name": "Dual_Obstacles_3DOF",
            "description": "Fixed dual-circle obstacle scene with waypoint tracking",
            "obstacle_scene": obstacle_scene,
            "total_time": float(self.total_time),
        }
    def _rebuild_scenario(self) -> None:
        scenario = self._build_default_scenario()
        self.obstacle_scene = scenario["obstacle_scene"]
        self.scenario_name = scenario["name"]