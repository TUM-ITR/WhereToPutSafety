"""State predictor used by the MPC module for delay compensation."""

import os
from typing import List, Tuple
import collections

import numpy as np

from sim_module.params import SystemParams
from plant_module.robot_dynamics import RobotDynamics
from controller_module.local_cbf import LocalCBF
from controller_module.pd_plus import PDPlusController



class StatePredictor:
	"""State predictor using iterative MPC control input for delayed measurements."""

	def __init__(self, params: SystemParams, dynamics: RobotDynamics):
		self.params = params
		self.dynamics = dynamics
		self.obstacle_scene = params.obstacle_scene
		self.dt = params.TA
		self.prediction_horizon = params.tau_known
		self.mpc_history= collections.deque(maxlen=max(params.tau_known, 1)) #buffer for prediction
		for _ in range(max(params.tau_known, 1)):
			self.mpc_history.append(np.zeros(3))
		self.z_pred = np.zeros(3)
		if(params.use_local_cbf):
			self.local_cbf = LocalCBF(params,dynamics) #Local CBF, correcting MPC input
		self.local_controller = PDPlusController(params, dynamics) #PD Plus Controller

	def add_mpc_input(self, ddq: np.ndarray):
		self.mpc_history.append(ddq.copy())

	def predict_future_state(self,z: np.ndarray, u:np.ndarray):
		"""Main function to predict the next state"""
		q = z[:3]
		dq = z[3:]
		cbf_is_active = False
		for k in range(self.prediction_horizon):
			#1. get current desired state
			ddq = self.mpc_history[k]

			#2. Compute potential next input with PD+
			z_pred, cbf_is_active_single,u = self.predict_single_step(q,dq,ddq,u)
			
			#3. if cbf was active in a single prediction -> tag this
			if(cbf_is_active_single):
				cbf_is_active = True

			#4. Overwrite old values
			q = z_pred[:3]
			dq = z_pred[3:]
		
		#Return the predicted state
		return z_pred, cbf_is_active

	def predict_single_step(self, q,dq,ddq_d,u):
		#Uses the local time step to predict one step in MPC time step mpc_TA
		cbf_is_active = False
		z= np.concatenate((q,dq))
		for _ in range(self.params.mpc_ratio):
			q = z[:3]
			dq = z[3:]
							
			q_d = q + self.dt*dq + 0.5 * (self.dt**2) * ddq_d
			dq_d = dq + self.dt* ddq_d
				
			# Compute torque a la PD+
			u = self.local_controller.compute_PDplus(q, dq, q_d, dq_d, ddq_d)

			# Forward Dynamics
			z_next = self.dynamics.joint_dynamics_step(z, u, self.dt, add_process_noise=False)

			# Save z
			z = z_next

		return z,cbf_is_active,u
	