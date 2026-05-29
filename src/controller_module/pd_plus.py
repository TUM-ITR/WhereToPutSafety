"""Nominal PD+ torque generator for waypoint tracking."""

from __future__ import annotations

import numpy as np

from sim_module.params import SystemParams
from plant_module.robot_dynamics import RobotDynamics


class PDPlusController:
    """PD+ controller matching tau = M(q) ddq_d + C(q, dq) dq + g(q) - K e - D e_dot."""

    def __init__(self, params: SystemParams, dynamics: RobotDynamics):
        self.params = params
        self.dynamics = dynamics
        self.last_q_d = np.zeros(3)
        self.last_dq_d = np.zeros(3)
        self.last_ddq_d = np.zeros(3)
        self.last_ee_d = np.zeros(2)


    def compute_PDplus(
        self,
        th: np.ndarray,
        om: np.ndarray,
        q_d: np.ndarray,
        dq_d: np.ndarray,
        ddq_d: np.ndarray | None = None,
    ) -> np.ndarray:
        if ddq_d is None:
            ddq_d = np.zeros(3)
        self.last_q_des = np.asarray(q_d, dtype=float).copy()
        self.last_qd_dot = np.asarray(dq_d, dtype=float).copy()
        self.last_qd_ddot = np.asarray(ddq_d, dtype=float).copy()
        self.last_ee_des = self.dynamics.fk_points(self.last_q_des)[-1].copy()
        kp_joint = float(getattr(self.params, "pd_plus_kp_joint", 80.0))
        kd_joint = float(getattr(self.params, "pd_plus_kd_joint", 22.0))
        umax = float(getattr(self.params, "umax_joint", 5.0))

        M = self.dynamics.mass_matrix(th)
        C = self.dynamics.coriolis_matrix(th, om)
        G = self.dynamics.gravity_vector(th)
        
        passive_damping = self.params.Dq3 @ dq_d if self.params.Dq3 is not None else np.zeros(3)
        passive_stiffness = self.params.Kq3 @ q_d if self.params.Kq3 is not None else np.zeros(3)

        position_error = th - q_d
        velocity_error = om - dq_d
        
        # Nominal PD+ control law: tau = M*q_ddot + C*q_dot + G - Kp*e - Kd*e_dot
        tracking_feedback = M @ (-kp_joint * position_error - kd_joint * velocity_error)
        torque = M @ ddq_d + C @ om + G + passive_damping + passive_stiffness + tracking_feedback
        return np.clip(np.asarray(torque, dtype=float), -umax, umax)