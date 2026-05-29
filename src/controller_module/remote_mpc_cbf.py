"""Remote controller that hosts all MPC-based logic for the 3DOF arm."""
from __future__ import annotations
from dataclasses import dataclass

import os
import time
from typing import List, Optional, Tuple

#import cvxpy as cp
import numpy as np
import casadi as ca

from sim_module.params import SystemParams
from plant_module.robot_dynamics import RobotDynamics

@dataclass
class MPCResult:
    u0: np.ndarray
    theta_traj: np.ndarray
    omega_traj: np.ndarray
    u_traj: np.ndarray
    slack_sum: float
    solve_time: float
    status: str
    success: bool
    CBFactive: bool
    CBFrelaxed: bool
    min_cbf_margin: float


class RemoteController:
    """
    Remote controller with Nominal MPC, and optional CBF filter.    
    """

    def __init__(self, params: SystemParams, dynamics: RobotDynamics):
        # Initialize the remote MPC controller with system parameters, dynamics model, local CBF filter, and state predictor.
        self.params = params
        self.dynamics = dynamics

        self.waypoints_jointspace  = self._getWaypointsInJointspace(params.waypoints)

        self.obstacle_scene = getattr(params, "obstacle_scene", None)
       
        self.N = int(params.N_horizon)
        self.dt = float(params.mpc_TA)

        self.nx = 6 #number of states
        self.nu = 3 #number of inputs

        # Trajectories for warmstarting MPC
        self._last_U = None
        self._last_Z = None
        self._last_S = None
        self._last_lam_g = None

        self._build_forward_dynamics()
        self._build_OCP()

    def _getWaypointsInJointspace(self, waypoints):
        """
            Computation of a reasonable configuration of the robot at the desired waypoint to track
        """
        q_waypoints = []

        for point in waypoints:
            p = np.asarray(point, dtype=float).reshape(2)
            q = np.asarray(self.dynamics.inverse_kinematics(p), dtype=float).reshape(3)
            q_waypoints.append(q)

        return q_waypoints

    def _fk_points_casadi(self, q):
        """
        CasADi-native FK for base, joint 1, joint 2, end effector.
        """
        q1, q2, q3 = q[0], q[1], q[2]
        L1, L2, L3 = self.dynamics.link_lengths

        base = ca.vertcat(0.0, 0.0)

        p1 = ca.vertcat(
            L1 * ca.cos(q1),
            L1 * ca.sin(q1),
        )

        p2 = p1 + ca.vertcat(
            L2 * ca.cos(q1 + q2),
            L2 * ca.sin(q1 + q2),
        )

        ee = p2 + ca.vertcat(
            L3 * ca.cos(q1 + q2 + q3),
            L3 * ca.sin(q1 + q2 + q3),
        )

        return base, p1, p2, ee


    def _fk_ee_casadi(self, q):
        #get endeffector pose in casadi
        return self._fk_points_casadi(q)[-1] #gets the last entry of the forward kinematics = endeffector
    
    def _fk_all_points_casadi(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # compute the forward kinematics for all points, for which we need a CBF
        # Written in "Casadi native"
        l1, l2, l3 = self.dynamics.link_lengths

        base = ca.vertcat(0.0, 0.0)

        p1 = base + ca.vertcat(
            l1 * ca.cos(q[0]),
            l1 * ca.sin(q[0]),
        )

        p2 = p1 + ca.vertcat(
            l2 * ca.cos(q[0] + q[1]),
            l2 * ca.sin(q[0] + q[1]),
        )

        p3 = p2 + ca.vertcat(
            l3 * ca.cos(q[0] + q[1] + q[2]),
            l3 * ca.sin(q[0] + q[1] + q[2]),
        )

        points = []

        samples_per_link = int(getattr(self.params, "cbf_link_samples", 3))
        grid = np.linspace(0.0, 1.0, samples_per_link + 1)[1:]

        for s in grid:
            points.append(base + float(s) * (p1 - base))
        for s in grid:
            points.append(p1 + float(s) * (p2 - p1))
        for s in grid:
            points.append(p2 + float(s) * (p3 - p2))

        return points

    def _fk_all_points_numpy(self, th: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base, p1, p2, ee = self.dynamics.fk_points(th)
        n_samp = int(getattr(self.params, "cbf_link_samples", 2))
        grid = np.linspace(0.0, 1.0, n_samp+1)[1:]

        points: List[np.ndarray] = [base, p1, p2, ee]
		
        for s in grid[1:-1]:
            points.append(base + s * (p1 - base))
        for s in grid[1:-1]:
            points.append(p1 + s * (p2 - p1))
        for s in grid[1:-1]:
            points.append(p2 + s * (ee - p2))
        return points

    def _dynamics_step_casadi(self, z, a, dt=None):
        """
        Acceleration-level joint-space prediction model.

        z = [q, dq]
        a = desired joint acceleration ddq
        """
        if dt is None:
            dt = self.dt

        q = z[:3]
        dq = z[3:]

        q_next = q + dt * dq + 0.5 * dt**2 * a
        dq_next = dq + dt * a

        return ca.vertcat(q_next, dq_next)

    def _diag_param(self, name: str, fallback_scalar: float, size: int) -> ca.DM:
        """
        Read a scalar, vector, or matrix weight from params and return a CasADi DM matrix.
        """
        value = getattr(self.params, name, None)

        if value is None:
            return ca.DM.eye(size) * fallback_scalar

        arr = np.asarray(value, dtype=float)

        if arr.ndim == 0:
            return ca.DM.eye(size) * float(arr)

        if arr.ndim == 1:
            if arr.shape[0] != size:
                raise ValueError(f"{name} must have length {size}, got {arr.shape[0]}.")
            return ca.diag(ca.DM(arr))

        if arr.shape != (size, size):
            raise ValueError(f"{name} must have shape {(size, size)}, got {arr.shape}.")

        return ca.DM(arr)


    def _angle_error_casadi(self, q: ca.MX, q_ref: ca.MX) -> ca.MX:
        """
        Wrapped angle error for revolute joints.

        This avoids artificial jumps near +/- pi.
        """
        e = q - q_ref
        return ca.atan2(ca.sin(e), ca.cos(e))

    ### Dynamics in Casadi to enforce torque constraints
    def _com_jacobian_casadi(self, q, link_id: int):
        """
        CasADi version of RobotDynamics._jacobian_for_com(...).

        Returns Jv(q) in R^{2x3} for the center of mass of one link.
        """
        q1, q2, q3 = q[0], q[1], q[2]

        L1, L2, L3 = self.dynamics.link_lengths
        r1, r2, r3 = self.dynamics.link_com_lengths

        J = ca.SX.zeros(2, 3) if isinstance(q, ca.SX) else ca.MX.zeros(2, 3)

        if link_id == 1:
            J[0, 0] = -r1 * ca.sin(q1)
            J[1, 0] =  r1 * ca.cos(q1)
            return J

        if link_id == 2:
            J[0, 0] = -L1 * ca.sin(q1) - r2 * ca.sin(q1 + q2)
            J[1, 0] =  L1 * ca.cos(q1) + r2 * ca.cos(q1 + q2)
            J[0, 1] = -r2 * ca.sin(q1 + q2)
            J[1, 1] =  r2 * ca.cos(q1 + q2)
            return J

        if link_id == 3:
            J[0, 0] = (
                -L1 * ca.sin(q1)
                -L2 * ca.sin(q1 + q2)
                -r3 * ca.sin(q1 + q2 + q3)
            )
            J[1, 0] = (
                L1 * ca.cos(q1)
                + L2 * ca.cos(q1 + q2)
                + r3 * ca.cos(q1 + q2 + q3)
            )

            J[0, 1] = (
                -L2 * ca.sin(q1 + q2)
                -r3 * ca.sin(q1 + q2 + q3)
            )
            J[1, 1] = (
                L2 * ca.cos(q1 + q2)
                + r3 * ca.cos(q1 + q2 + q3)
            )

            J[0, 2] = -r3 * ca.sin(q1 + q2 + q3)
            J[1, 2] =  r3 * ca.cos(q1 + q2 + q3)
            return J

        raise ValueError(f"Unknown link_id: {link_id}")
    
    def _mass_matrix_casadi(self, q):
        """
        CasADi-native mass matrix.

        Same structure as RobotDynamics.mass_matrix:
            M(q) = sum_i m_i Jv_i(q)^T Jv_i(q) + I_i Jw_i^T Jw_i
        """
        is_sx = isinstance(q, ca.SX)
        M = ca.SX.zeros(3, 3) if is_sx else ca.MX.zeros(3, 3)

        masses = self.dynamics.link_masses
        inertias = self.dynamics.link_inertias

        for link_id in range(1, 4):
            Jv = self._com_jacobian_casadi(q, link_id)

            Jw = ca.SX.zeros(3, 1) if is_sx else ca.MX.zeros(3, 1)
            for j in range(link_id):
                Jw[j] = 1.0

            M += masses[link_id - 1] * (Jv.T @ Jv)
            M += inertias[link_id - 1] * (Jw @ Jw.T)

        return M
    
    def _gravity_vector_casadi(self, q):
        """
        CasADi-native gravity vector.

        Matches RobotDynamics.gravity_vector:
            G(q) = sum_i m_i g Jv_i(q)[1, :]
        """
        is_sx = isinstance(q, ca.SX)
        G = ca.SX.zeros(3, 1) if is_sx else ca.MX.zeros(3, 1)

        masses = self.dynamics.link_masses
        g = float(self.dynamics.gravity_accel)

        for link_id in range(1, 4):
            Jv = self._com_jacobian_casadi(q, link_id)
            G += masses[link_id - 1] * g * Jv[1, :].T

        return G
    
    def _coriolis_times_velocity_casadi(self, q, dq):
        """
        Computes C(q,dq)dq as a CasADi expression using Christoffel symbols.

        c_i = sum_{j,k} 0.5 * (
                dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i
            ) dq_j dq_k

        Returns vector c in R^3.
        """
        is_sx = isinstance(q, ca.SX)
        c = ca.SX.zeros(3, 1) if is_sx else ca.MX.zeros(3, 1)

        M = self._mass_matrix_casadi(q)

        for i in range(3):
            c_i = 0
            for j in range(3):
                for k in range(3):
                    dM_ij_dqk = ca.jacobian(M[i, j], q)[k]
                    dM_ik_dqj = ca.jacobian(M[i, k], q)[j]
                    dM_jk_dqi = ca.jacobian(M[j, k], q)[i]

                    christoffel = 0.5 * (
                        dM_ij_dqk + dM_ik_dqj - dM_jk_dqi
                    )
                    c_i += christoffel * dq[j] * dq[k]

            c[i] = c_i

        return c

  
    def _build_forward_dynamics(self):
        """
        Build CasADi-native torque-level dynamics function once.

        After this, do not call ca.jacobian(M(q), q) inside the OCP.
        Instead call self._tau_fun(q, dq, ddq)
        """
        if hasattr(self, "_tau_fun"):
            return

        q = ca.MX.sym("q", 3)
        dq = ca.MX.sym("dq", 3)
        ddq = ca.MX.sym("tau", 3)

        M = self._mass_matrix_casadi(q)
        G = self._gravity_vector_casadi(q)

        # Build C(q,dq)dq using q as a pure symbolic variable.
        C_dq = ca.MX.zeros(3, 1)

        for i in range(3):
            c_i = 0
            for j in range(3):
                for k in range(3):
                    dM_ij_dqk = ca.jacobian(M[i, j], q)[k]
                    dM_ik_dqj = ca.jacobian(M[i, k], q)[j]
                    dM_jk_dqi = ca.jacobian(M[j, k], q)[i]

                    christoffel = 0.5 * (
                        dM_ij_dqk + dM_ik_dqj - dM_jk_dqi
                    )

                    c_i += christoffel * dq[j] * dq[k]

            C_dq[i] = c_i

        D = self._diag_param("Dq3",0,3)
        K = self._diag_param("Kq3",0,3)

        q_col = ca.reshape(q, 3, 1)
        dq_col = ca.reshape(dq, 3, 1)
        ddq_col = ca.reshape(ddq, 3, 1)

        tau = M @ ddq_col + C_dq + G + D @ dq_col + K @ q_col
        
        self._tau_fun = ca.Function(
            "tau_fun",
            [q, dq, ddq],
            [tau],
        )

    ### OPTIMAL CONTROL PROBLEM
    def _build_OCP(self):
        # The decision variables are Joint ACCELERATIONS not torques
        # Two versions: tracking target in world coordinates and in joint coordinates (latter is currently in use)
        self.opti = ca.Opti()

        N = self.N

        # Decision variables
        self.Z = self.opti.variable(self.nx, N + 1)
        self.U = self.opti.variable(self.nu, N)        

        #If combined version: use slack
        if self.params.mpc_use_slack:
            self.S = self.opti.variable(N)
            self.opti.subject_to(self.S >= 0)
        else:
            self.S = None

        # Runtime parameters
        self.z0_param = self.opti.parameter(self.nx)
        if self.params.mpc_reference_space == "joint":
            self.p_ref_param = self.opti.parameter(3, N + 1) #if joint space: 3 target values
        else:
            self.p_ref_param = self.opti.parameter(2, N + 1) #if world space: 2 target values (2D Plane)
        self.u_prev_param = self.opti.parameter(self.nu)

        # Weights and limits
        if self.params.mpc_reference_space == "joint":
            Qq = self._diag_param("Qq_mpc", 80.0, 3)
            Qq_terminal = self._diag_param("Qq_terminal_mpc", 800.0, 3)

            Qdq = self._diag_param("Qdq_mpc", 3.0, 3)
            Qdq_terminal = self._diag_param("Qdq_terminal_mpc", 30.0, 3)

            Rddq = self._diag_param("Rddq_mpc", 0.2, 3)
            Rjerk = self._diag_param("Rjerk_mpc", 2.0, 3)
        else:
            Qe = float(getattr(self.params, "Qe", 100.0))
            Qe_terminal = float(getattr(self.params, "Qe_terminal", 500.0))
            Qv = float(getattr(self.params, "Qv", 10.0))
            R = float(getattr(self.params, "R_mpc", 1.0))
            smooth_weight = float(getattr(self.params, "mpc_smoothness_weight", 0.1))

        umax = float(getattr(self.params, "umax_joint", 10.0)) #torque input limit
        amax = float(getattr(self.params, "amax_joint", 20.0)) #acceleration limit
        max_vel = float(getattr(self.params, "max_joint_velocity", 2.0))        
        du_max = float(getattr(self.params, "du_max", 2.0)) #rate constraint
      
        # Angle Limit
        q_min = np.asarray(getattr(self.params, "q_min", [-np.pi, -np.pi, -np.pi]), dtype=float)
        q_max = np.asarray(getattr(self.params, "q_max", [ np.pi,  np.pi,  np.pi]), dtype=float)


        gamma = float(getattr(self.params, "gamma_cbf", 0.3))
        margin = float(getattr(self.params, "cbf_safety_margin", 0.02))
        slack_weight = float(getattr(self.params, "mpc_slack_weight", 1e5))

        # Initial condition
        self.opti.subject_to(self.Z[:, 0] == self.z0_param)

        cost = 0

        #Torque Rate Constraints
        #   REMARK: heavy on computation, commented out for speed. We recommend to rely on acceleration limits.
        #last_torque= self._tau_fun(self.Z[:3, 0],self.Z[3:, 0],self.u_prev_param)
        #self.opti.subject_to(self.opti.bounded(-du_max, self.U[:, 0] - self.u_prev_param, du_max))
        #for k in range(N-1): #rate constraint
        #    self.opti.subject_to(self.opti.bounded(-du_max, self.U[:, k + 1] - self.U[:, k], du_max))

        # other constraints
        for k in range(N):
            z_k = self.Z[:, k]
            z_next = self.Z[:, k + 1]
            u_k = self.U[:, k]

            # Dynamics
            self.opti.subject_to(z_next == self._dynamics_step_casadi(z_k, u_k,self.dt))

            # Torque limits
            #   REMARK: heavy on computation, commented out for speed. We recommend to rely on acceleration limits.
            #torque = self._tau_fun(z_k[:3],z_k[3:],u_k)
            #self.opti.subject_to(self.opti.bounded(-umax, torque, umax)) # input constraint torque
            #self.opti.subject_to(self.opti.bounded(-du_max, torque - last_torque, du_max))
            #last_torque = torque #need to update for constraints

            # Acceleration, velocity, position limits
            self.opti.subject_to(self.opti.bounded(-amax, u_k, amax))
            self.opti.subject_to(self.opti.bounded(-max_vel, z_next[3:], max_vel))
            self.opti.subject_to(self.opti.bounded(q_min, z_next[:3], q_max))
            

            # CBF constraints
            if self.params.mpc_use_CBF:            
                points_k = self._fk_all_points_casadi(z_k[:3])
                points_next = self._fk_all_points_casadi(z_next[:3])
                for obs in self.obstacle_scene.obstacles:
                    if getattr(obs, "type", "circle") != "circle":
                        continue
                    center = np.asarray(obs.center, float).reshape(2)
                    r_eff = float(obs.radius) + margin

                    for p_k, p_next in zip(points_k, points_next):
                        h_k = ca.sumsqr(p_k - center) - r_eff**2
                        h_next = ca.sumsqr(p_next - center) - r_eff**2

                        if self.params.mpc_use_slack:
                            self.opti.subject_to(
                                h_next >= (1.0 - gamma) * h_k - self.S[k]
                            )
                        else:
                            self.opti.subject_to(
                                h_next >= (1.0 - gamma) * h_k
                            )

            # Stage cost
            if self.params.mpc_reference_space == "joint":
                # Stage cost: joint-space reference tracking
                q_next = z_next[:3]
                dq_next = z_next[3:]
                q_ref_next = self.p_ref_param[:, k + 1]

                q_err = self._angle_error_casadi(q_next, q_ref_next)

                cost += ca.mtimes([q_err.T, Qq, q_err])
                cost += ca.mtimes([dq_next.T, Qdq, dq_next])
                cost += ca.mtimes([u_k.T, Rddq, u_k])

                # Penalize acceleration changes. For k=0, compare against previous applied MPC acceleration.
                if k == 0:
                    du_k = u_k - self.u_prev_param
                else:
                    du_k = u_k - self.U[:, k - 1]

                cost += ca.mtimes([du_k.T, Rjerk, du_k])
                if self.params.mpc_use_slack:
                    cost += slack_weight * ca.sumsqr(self.S[k])
            else:
                ee_next = self._fk_ee_casadi(z_next[:3])
                p_ref_next = self.p_ref_param[:, k + 1]

                cost += Qe * ca.sumsqr(ee_next - p_ref_next) #position penalty
                cost += Qv * ca.sumsqr(z_next[3:]) #velocity penalty
                cost += R * ca.sumsqr(u_k) #input penalty

                if k < N - 1:
                    cost += smooth_weight * ca.sumsqr(self.U[:, k + 1] - self.U[:, k])

                if self.params.mpc_use_slack:
                    cost += slack_weight * ca.sumsqr(self.S[k])

        # Terminal cost
        if self.params.mpc_reference_space == "joint":
            q_terminal = self.Z[:3, N]
            dq_terminal = self.Z[3:, N]
            q_ref_terminal = self.p_ref_param[:, N]

            q_terminal_err = self._angle_error_casadi(q_terminal, q_ref_terminal)

            cost += ca.mtimes([q_terminal_err.T, Qq_terminal, q_terminal_err])
            cost += ca.mtimes([dq_terminal.T, Qdq_terminal, dq_terminal])
        else:
            ee_terminal = self._fk_ee_casadi(self.Z[:3, N])
            p_ref_terminal = self.p_ref_param[:, N]
            cost += Qe_terminal * ca.sumsqr(ee_terminal - p_ref_terminal)

        self.opti.minimize(cost)

        p_opts = {
            "expand": True,
            "print_time": False,
        }

        s_opts = {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 50,
            "tol": 1e-4,
            "acceptable_tol": 1e-3,
            "acceptable_iter": 3,
            "warm_start_init_point":"yes",
            "mu_init": 1e-3
        }

        self.opti.solver("ipopt", p_opts, s_opts)

    def _evaluate_cbf_activity(
        self,
        theta_traj: np.ndarray,
        slack_traj: Optional[np.ndarray] = None,
        tol: float = 1e-4,
        ) -> tuple[bool, bool, float]:
        if not self.params.mpc_use_CBF:
            return False, False, np.inf

        if self.obstacle_scene is None:
            return False, False, np.inf

        gamma = float(getattr(self.params, "gamma_cbf", 0.3))
        margin_extra = float(getattr(self.params, "cbf_safety_margin", 0.02))

        min_margin = np.inf
        CBFactive = False

        for k in range(theta_traj.shape[0] - 1):
            points_k = self._fk_all_points_numpy(theta_traj[k])
            points_next = self._fk_all_points_numpy(theta_traj[k + 1])

            s_k = 0.0
            if slack_traj is not None:
                s_k = float(slack_traj[k])

            for obs in self.obstacle_scene.obstacles:
                if getattr(obs, "type", "circle") != "circle":
                    continue

                center = np.asarray(obs.center, dtype=float).reshape(2)
                r_eff = float(obs.radius) + margin_extra

                for p_k, p_next in zip(points_k, points_next):
                    h_k = np.sum((p_k - center) ** 2) - r_eff**2
                    h_next = np.sum((p_next - center) ** 2) - r_eff**2

                    if self.params.mpc_use_slack:
                        cbf_margin = h_next - (1.0 - gamma) * h_k + s_k
                    else:
                        cbf_margin = h_next - (1.0 - gamma) * h_k

                    min_margin = min(min_margin, cbf_margin)

                    if cbf_margin <= tol:
                        CBFactive = True

        CBFrelaxed = False
        if slack_traj is not None:
            CBFrelaxed = bool(np.any(slack_traj > tol))

        return CBFactive, CBFrelaxed, float(min_margin)

    def debug_mpc_result(self,result, z0, q_ref=None):
        q0_err = np.linalg.norm(result.theta_traj[0] - z0[:3])
        dq0_err = np.linalg.norm(result.omega_traj[0] - z0[3:])

        max_abs_u = np.max(np.abs(result.u_traj))
        max_abs_dq = np.max(np.abs(result.omega_traj))

        print("----- MPC DEBUG -----")
        print("status:", result.status, "success:", result.success)
        print("q0 consistency:", q0_err)
        print("dq0 consistency:", dq0_err)
        print("u0:", result.u0)
        print("max |u_traj|:", max_abs_u)
        print("max |omega_traj|:", max_abs_dq)

        if q_ref is not None:
            print("current q:", z0[:3])
            print("q_ref:", q_ref)
            print("initial q error:", z0[:3] - q_ref)
            print("terminal q:", result.theta_traj[-1])
            print("terminal q error:", result.theta_traj[-1] - q_ref)
        if self.params.mpc_use_slack:            
            print("Sum of slack values:", result.slack_sum)


    def solve(self, z0: np.ndarray, waypointIdx: int, a_prev: Optional[np.ndarray] = None):
            """
            Solve MPC for current state and reference trajectory.

            Parameters
            ----------
            z0:
                Current state, shape (6,).

            p_ref:
                Workspace reference trajectory.
                Accepted shapes:
                    (2,)         constant reference over horizon
                    (N+1, 2)     full reference trajectory
                    (N, 2)       will be padded to N+1
            """
            start = time.time()

            # Initialize things
            z0 = np.asarray(z0, dtype=float).reshape(6)
            if self.params.mpc_reference_space == "joint":
                nRef = 3
                p_ref = np.asarray(self.waypoints_jointspace [waypointIdx], dtype=float)
            else:
                nRef = 2
                p_ref = np.asarray(self.params.waypoints[waypointIdx], dtype=float)

            # previous acceleration    
            if a_prev is not None:
                u_prev_value = np.asarray(a_prev, dtype=float).reshape(self.nu)
            elif self._last_U is not None:
                u_prev_value = self._last_U[:, 0]
            else:
                u_prev_value = np.zeros(self.nu)

            self.opti.set_value(self.u_prev_param, u_prev_value)

            if p_ref.shape == (nRef,):
                p_ref_full = np.tile(p_ref.reshape(1, nRef), (self.N + 1, 1))
            elif p_ref.shape == (self.N, nRef):
                p_ref_full = np.vstack([p_ref[0], p_ref])
            elif p_ref.shape == (self.N + 1, nRef):
                p_ref_full = p_ref
            else:
                raise ValueError(
                    f"p_ref must have shape ({nRef},), ({self.N}, {nRef}), or ({self.N + 1}, {nRef}), "
                    f"but got {p_ref.shape}."
                )
            
            self.opti.set_value(self.z0_param, z0)
            self.opti.set_value(self.p_ref_param, p_ref_full.T)

            # Warm start
            if self._last_U is not None:
                U_guess = np.roll(self._last_U, -1, axis=1)
                U_guess[:, -1] = U_guess[:, -2]
                self.opti.set_initial(self.U, U_guess)                
            else:
                self.opti.set_initial(self.U, np.zeros((self.nu, self.N)))

            if self._last_Z is not None:
                Z_guess = np.roll(self._last_Z, -1, axis=1)
                Z_guess[:, 0] = z0
                Z_guess[:, -1] = Z_guess[:, -2]
                self.opti.set_initial(self.Z, Z_guess)
            else:
                Z_guess = np.tile(z0.reshape(6, 1), (1, self.N + 1))
                self.opti.set_initial(self.Z, Z_guess)

            #Warmstart Slack
            if self.params.mpc_use_slack:
                if self._last_S is not None:
                    S_guess = np.roll(self._last_S, -1)
                    S_guess[-1] = S_guess[-2]
                    self.opti.set_initial(self.S, S_guess)
                else:
                    self.opti.set_initial(self.S, np.zeros(self.N))
            #Warmstart dual variables
            if self._last_lam_g is not None:
                self.opti.set_initial(self.opti.lam_g, self._last_lam_g)


            # SOLVE OCP
            try:
                sol = self.opti.solve()
                status = self.opti.stats()["return_status"]
                success = True

                U_opt = sol.value(self.U)
                Z_opt = sol.value(self.Z)
                if self.params.mpc_use_slack:
                    self._last_S = sol.value(self.S)
                self._last_lam_g = sol.value(self.opti.lam_g)

                slack_sum = 0.0
                slack_traj = None

                if self.params.mpc_use_slack:
                    slack_traj = np.asarray(sol.value(self.S), dtype=float).reshape(self.N)
                    slack_sum = float(np.sum(sol.value(self.S)))

            except RuntimeError:
                status = self.opti.stats().get("return_status", "failed")
                success = False

                # Safe fallback: do not use infeasible debug values as controller output.
                U_opt = np.zeros((self.nu, self.N))
                Z_opt = np.tile(z0.reshape(6, 1), (1, self.N + 1))

                slack_sum = 0.0
                slack_traj = None
            
            if success: #only update for warm start, if successful
                self._last_U = np.asarray(U_opt, dtype=float)
                self._last_Z = np.asarray(Z_opt, dtype=float)
                self._mpc_fail_count = 0
            
            else:
                self._mpc_fail_count = getattr(self, "_mpc_fail_count", 0) + 1

                if self._mpc_fail_count >= 2:
                    print("MPC failed repeatedly: resetting warm start.")
                    self._last_U = None
                    self._last_Z = None

            solve_time = time.time() - start

            theta_traj = Z_opt[:3, :].T
            omega_traj = Z_opt[3:, :].T
            u_traj = U_opt.T
            u0 = u_traj[0, :]


            CBFactive, CBFrelaxed, min_cbf_margin = self._evaluate_cbf_activity(
                theta_traj,
                slack_traj=slack_traj,
                tol=1e-4,
                )
            
            result = MPCResult(
                u0=u0,
                theta_traj=theta_traj,
                omega_traj=omega_traj,
                u_traj=u_traj,
                slack_sum=slack_sum,
                solve_time=solve_time,
                status=status,
                success=success, 
                CBFactive=CBFactive,
                CBFrelaxed=CBFrelaxed,
                min_cbf_margin=min_cbf_margin,
            )

            #DEBUG
            #self.debug_mpc_result(result, z0, p_ref)
            return result
    