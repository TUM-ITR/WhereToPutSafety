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
class LocalCBFResult:
    u0: np.ndarray
    theta_next: np.ndarray
    omega_next: np.ndarray
    solve_time: float
    status: str
    success: bool
    CBFactive: bool
    slack: float
    min_cbf_margin: float

class LocalCBF:
    """
    Local CBF Filter   
    """

    def __init__(self, params: SystemParams, dynamics: RobotDynamics):
        # Initialize the remote MPC controller with system parameters, dynamics model, local CBF filter, and state predictor.
        self.params = params
        self.dynamics = dynamics

        self.waypoints_jointspace  = self._getWaypointsInJointspace(params.waypoints)

        self.obstacle_scene = getattr(params, "obstacle_scene", None)
       
        self.dt = float(params.TA)

        self.nx = 6 #number of states
        self.nu = 3 #number of inputs

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


        # Decision variables - only single step
        self.Z = self.opti.variable(self.nx)
        self.U = self.opti.variable(self.nu)
        self.S = self.opti.variable(1)


        # Runtime parameters
        self.z0_param = self.opti.parameter(self.nx)
        self.ddq_d_param = self.opti.parameter(3, 1)
        self.tau_prev_param = self.opti.parameter(3, 1)

        #limits       
        
        umax = float(getattr(self.params, "umax_joint", 10.0)) #torque input limit
        amax = float(getattr(self.params, "amax_joint", 20.0)) #acceleration limit
        max_vel = float(getattr(self.params, "max_joint_velocity", 2.0))        
        du_max = float(getattr(self.params, "du_max", 10.0))/self.params.mpc_ratio #rate constraint
      
        # Angle Limit
        q_min = np.asarray(getattr(self.params, "q_min", [-np.pi, -np.pi, -np.pi]), dtype=float)
        q_max = np.asarray(getattr(self.params, "q_max", [ np.pi,  np.pi,  np.pi]), dtype=float)


        gamma = float(getattr(self.params, "gamma_cbf", 0.3))
        margin = float(getattr(self.params, "cbf_safety_margin", 0.02))
        slack_weight = float(getattr(self.params, "local_cbf_slack_weight", 1e5))

        # CONSTRAINTS
        
        #Slack > 0
        self.opti.subject_to(self.S >= 0) 

        # other constraints
        z_k = self.z0_param
        z_next = self.Z
        u_k = self.U

        # Dynamics
        self.opti.subject_to(z_next == self._dynamics_step_casadi(z_k, u_k,self.dt))

        # Input and velocity limits
        #torque = self._tau_fun(z_k[:3],z_k[3:],u_k)
        #self.opti.subject_to(self.opti.bounded(-umax, torque, umax)) # input constraint torque
        #self.opti.subject_to(self.opti.bounded(-du_max, torque - self.tau_prev_param, du_max)) #rate constraint
        self.opti.subject_to(self.opti.bounded(-amax, u_k, amax)) #acceleration constraint
        self.opti.subject_to(self.opti.bounded(-max_vel, z_next[3:], max_vel)) #velocity constraint
        self.opti.subject_to(self.opti.bounded(q_min, z_next[:3], q_max)) #angle constraints
        

        # CBF constraints           
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
                self.opti.subject_to(
                    h_next >= (1.0 - gamma)*h_k - self.S
                )

        # COSTS
        ddq_err = u_k-self.ddq_d_param
        Qcbf = self._diag_param("Qcbf",1,3) #just for scaling

        cost = ca.mtimes([ddq_err.T, Qcbf, ddq_err])
        cost += slack_weight * self.S # (self.S**2)
        
        # MINIMIZER
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
        theta_now: np.ndarray,
        theta_next: np.ndarray,
        tol: float = 1e-4,
        ):

        if self.obstacle_scene is None:
            return False, None #CBF not active, no min margin

        gamma = float(getattr(self.params, "gamma_cbf", 0.3))
        margin_extra = float(getattr(self.params, "cbf_safety_margin", 0.02))

        min_margin = np.inf
        CBFactive = False

        # get points
        points_k = self._fk_all_points_numpy(theta_now)
        points_next = self._fk_all_points_numpy(theta_next)

        # calc CBF for every obstacle and point on the link
        for obs in self.obstacle_scene.obstacles:
            if getattr(obs, "type", "circle") != "circle":
                continue

            center = np.asarray(obs.center, dtype=float).reshape(2)
            r_eff = float(obs.radius) + margin_extra

            for p_k, p_next in zip(points_k, points_next):
                h_k = np.sum((p_k - center) ** 2) - r_eff**2
                h_next = np.sum((p_next - center) ** 2) - r_eff**2

                cbf_margin = h_next - (1.0 - gamma) * h_k

                min_margin = min(min_margin, cbf_margin)

                if cbf_margin <= tol:
                    CBFactive = True

        return CBFactive, float(min_margin)

    def solve(self, z0: np.ndarray, ddq_d: np.ndarray, tau_prev: np.ndarray):
            """
            Solve local CBF for current desired acceleration

            Parameters
            ----------
            z0:
                Current state, shape (6,).

            ddq_ref:
                Current desired acceleration

            tau_prev:
                Last used input command
            """
            start = time.time()

            # Initialize things
            z0 = np.asarray(z0, dtype=float).reshape(6)
            ddq_d_param = np.asarray(ddq_d, dtype=float).reshape(3)
            tau_prev_param = np.asarray(tau_prev, dtype=float).reshape(3)

            self.opti.set_value(self.z0_param, z0)
            self.opti.set_value(self.ddq_d_param, ddq_d_param)
            self.opti.set_value(self.tau_prev_param, tau_prev_param)

            try:
                sol = self.opti.solve()
                status = self.opti.stats()["return_status"]
                success = True

                U_opt = sol.value(self.U)
                Z_opt = sol.value(self.Z)
                slack = sol.value(self.S)

            except RuntimeError:
                status = self.opti.stats().get("return_status", "failed")
                success = False
                slack = None

                # Safe fallback: do not use infeasible debug values as controller output.
                U_opt = ddq_d
                Z_opt = z0
            
            solve_time = time.time() - start

            theta_next = Z_opt[:3].T
            omega_next = Z_opt[3:].T
            u0 = U_opt


            CBFactive, min_cbf_margin = self._evaluate_cbf_activity(
                z0[:3],
                theta_next,
                tol=1e-4,
                )
            
            result = LocalCBFResult(
                u0=u0,
                theta_next=theta_next,
                omega_next=omega_next,
                solve_time=solve_time,
                status=status,
                success=success, 
                CBFactive=CBFactive,
                slack = slack,
                min_cbf_margin=min_cbf_margin,
            )

            #DEBUG
            #self.debug_cbf_result(result,z0, ddq_d)
            return result
    
    def nominal_is_feasible(self, z0: np.ndarray, ddq: np.ndarray, tau_prev: np.ndarray, tol: float = 1e-6) -> bool:
        """Function to test if point is already feasible"""
        z0 = np.asarray(z0, dtype=float).reshape(6)
        ddq = np.asarray(ddq, dtype=float).reshape(3)
        tau_prev = np.asarray(tau_prev, dtype=float).reshape(3)

        q = z0[:3]
        dq = z0[3:]

        dt = self.dt

        q_next = q + dt * dq + 0.5 * dt**2 * ddq
        dq_next = dq + dt * ddq

        # Acceleration constraint
        amax = float(getattr(self.params, "amax_joint", 20.0))
        if np.any(ddq < -amax - tol) or np.any(ddq > amax + tol):
            return False

        # Velocity constraint
        max_vel = float(getattr(self.params, "max_joint_velocity", 2.0))
        if np.any(dq_next < -max_vel - tol) or np.any(dq_next > max_vel + tol):
            return False

        # Joint angle constraint
        q_min = np.asarray(getattr(self.params, "q_min", [-np.pi, -np.pi, -np.pi]), dtype=float)
        q_max = np.asarray(getattr(self.params, "q_max", [ np.pi,  np.pi,  np.pi]), dtype=float)
        if np.any(q_next < q_min - tol) or np.any(q_next > q_max + tol):
            return False

        # Torque constraint
        umax = float(getattr(self.params, "umax_joint", 10.0))
        tau = self.dynamics.computed_torque(q, dq, ddq)
        if np.any(tau < -umax - tol) or np.any(tau > umax + tol):
            return False

        # Torque-rate constraint, if you want to keep it
        du_max = float(getattr(self.params, "du_max", 10.0)) / self.params.mpc_ratio
        if np.any(tau - tau_prev < -du_max - tol) or np.any(tau - tau_prev > du_max + tol):
            return False

        # CBF constraint
        _, min_margin = self._evaluate_cbf_activity(q, q_next, tol=tol)

        if (min_margin is not None) and (min_margin < -tol):
            return False

        return True

    def computeLocalCBF(self, z0: np.ndarray, ddq_d: np.ndarray, tau_prev: np.ndarray):
        """Function to compute the local CBF"""

        #1. Test, if point is already feasible
        if self.nominal_is_feasible(z0,ddq_d,tau_prev):
            q = z0[:3]
            dq = z0[3:]
            dt = self.dt

            q_next = q + dt * dq + 0.5 * dt**2 * ddq_d
            dq_next = dq + dt * ddq_d

            result = LocalCBFResult(
                u0=ddq_d,
                theta_next=q_next,
                omega_next=dq_next,
                solve_time=0.0,
                status="Skipped_Nominal_Feasible",
                success=True,
                CBFactive=False,
                slack=0.0,
                min_cbf_margin=self.evaluate_candidate_margin(z0, ddq_d),
            )

            return ddq_d, q_next, dq_next, result
            
        try:
            resultCBF= self.solve(z0, ddq_d,tau_prev) #u is the last applied input
            success = resultCBF.success
        except RuntimeError:
            success = False

        q = z0[:3]
        dq = z0[3:]
        ddq_safe = resultCBF.u0
        dq_d = resultCBF.omega_next
        q_d = resultCBF.theta_next

        if not success:
            print("Local CBF-QP failed. Applying braking acceleration.")
            kd_brake = float(getattr(self.params, "local_cbf_brake_gain", 5.0))
            amax = float(getattr(self.params, "amax_joint", 10.0))
            ddq_safe = np.clip(-kd_brake * dq, -amax, amax)

        return ddq_safe,q_d,dq_d,resultCBF
    
    # DEBUGGING
    def debug_cbf_result(self,result, z0, ddq_d):
        
        print("----- CBF DEBUG -----")
        print("status:", result.status, ", success:", result.success)
        print("ddq0:", result.u0)

        print("Change in Acceleration:", result.u0-ddq_d)
        print("Activation Detection:", result.CBFactive)
        print("Nominal CBF margin:", self.evaluate_candidate_margin(z0, ddq_d))
        print("Safe CBF margin:", result.min_cbf_margin)
        print("Slack: ",result.slack)

    def evaluate_candidate_margin(self, z0, ddq):
        z0 = np.asarray(z0, dtype=float).reshape(6)
        ddq = np.asarray(ddq, dtype=float).reshape(3)

        z_next = np.array([
            *(z0[:3] + self.dt*z0[3:] + 0.5*self.dt**2*ddq),
            *(z0[3:] + self.dt*ddq),
        ])

        _, min_margin = self._evaluate_cbf_activity(z0,z_next)
        return min_margin