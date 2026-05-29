"""Robot dynamics and kinematics for the 3DOF plant module."""

from typing import Optional, Tuple

import numpy as np

from sim_module.params import SystemParams


class RobotDynamics:
	"""3-DOF planar robot dynamics and kinematics."""

	def __init__(self, params: SystemParams, preloaded_noise: Optional[np.ndarray] = None):
		self.params = params
		self.link_masses = np.asarray(params.link_masses, dtype=float)
		self.link_lengths = np.asarray([params.L1, params.L2, params.L3], dtype=float)
		self.link_com_lengths = 0.5 * self.link_lengths
		self.link_inertias = (self.link_masses * self.link_lengths**2) / 12.0
		self.gravity_accel = float(getattr(params, "gravity_accel", 9.81))
		self.rng = np.random.RandomState(params.random_seed)
		self.process_noise_history = []
		self.last_process_noise = np.zeros(3)

		self.preloaded_noise = preloaded_noise
		self.noise_step_counter = 0
		self._mass_matrix_eps = 1e-6


	def _fk_com_positions(self, th: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		t1, t2, t3 = th[:3]
		l1, l2, _ = self.link_lengths
		r1, r2, r3 = self.link_com_lengths
		c1 = np.array([r1 * np.cos(t1), r1 * np.sin(t1)])
		c2 = np.array([
			l1 * np.cos(t1) + r2 * np.cos(t1 + t2),
			l1 * np.sin(t1) + r2 * np.sin(t1 + t2),
		])
		c3 = np.array([
			l1 * np.cos(t1) + l2 * np.cos(t1 + t2) + r3 * np.cos(t1 + t2 + t3),
			l1 * np.sin(t1) + l2 * np.sin(t1 + t2) + r3 * np.sin(t1 + t2 + t3),
		])
		return c1, c2, c3

	def _jacobian_for_com(self, th: np.ndarray, link_id: int) -> np.ndarray:
		t1, t2, t3 = th[:3]
		l1, l2, _ = self.link_lengths
		r1, r2, r3 = self.link_com_lengths
		J = np.zeros((2, 3))
		if link_id == 1:
			J[:, 0] = [-r1 * np.sin(t1), r1 * np.cos(t1)]
			return J
		if link_id == 2:
			J[:, 0] = [-l1 * np.sin(t1) - r2 * np.sin(t1 + t2), l1 * np.cos(t1) + r2 * np.cos(t1 + t2)]
			J[:, 1] = [-r2 * np.sin(t1 + t2), r2 * np.cos(t1 + t2)]
			return J
		J[:, 0] = [
			-l1 * np.sin(t1) - l2 * np.sin(t1 + t2) - r3 * np.sin(t1 + t2 + t3),
			l1 * np.cos(t1) + l2 * np.cos(t1 + t2) + r3 * np.cos(t1 + t2 + t3),
		]
		J[:, 1] = [
			-l2 * np.sin(t1 + t2) - r3 * np.sin(t1 + t2 + t3),
			l2 * np.cos(t1 + t2) + r3 * np.cos(t1 + t2 + t3),
		]
		J[:, 2] = [-r3 * np.sin(t1 + t2 + t3), r3 * np.cos(t1 + t2 + t3)]
		return J

	def mass_matrix(self, th: np.ndarray) -> np.ndarray:
		M = np.zeros((3, 3))
		for link_id in range(1, 4):
			Jv = self._jacobian_for_com(th, link_id)
			Jw = np.zeros(3)
			Jw[:link_id] = 1.0
			M += self.link_masses[link_id - 1] * (Jv.T @ Jv)
			M += self.link_inertias[link_id - 1] * np.outer(Jw, Jw)
		return M

	def gravity_vector(self, th: np.ndarray) -> np.ndarray:
		g_vec = np.zeros(3)
		for link_id in range(1, 4):
			Jv = self._jacobian_for_com(th, link_id)
			g_vec += self.link_masses[link_id - 1] * self.gravity_accel * Jv[1, :]
		return g_vec

	def coriolis_matrix(self, th: np.ndarray, om: np.ndarray) -> np.ndarray:
		eps = self._mass_matrix_eps
		dM_dq = []
		for idx in range(3):
			delta = np.zeros(3)
			delta[idx] = eps
			M_plus = self.mass_matrix(th + delta)
			M_minus = self.mass_matrix(th - delta)
			dM_dq.append((M_plus - M_minus) / (2.0 * eps))

		C = np.zeros((3, 3))
		for i in range(3):
			for j in range(3):
				c_ij = 0.0
				for k in range(3):
					c_ijk = 0.5 * (dM_dq[k][i, j] + dM_dq[j][i, k] - dM_dq[i][j, k])
					c_ij += c_ijk * om[k]
				C[i, j] = c_ij
		return C

	def forward_dynamics(self, th: np.ndarray, om: np.ndarray, torque: np.ndarray) -> np.ndarray:
		# Solves system dynamics: M*q_ddot + C*q_dot + G = tau
		M = self.mass_matrix(th)
		C = self.coriolis_matrix(th, om)
		G = self.gravity_vector(th)
		passive_damping = self.params.Dq3 @ om if self.params.Dq3 is not None else np.zeros(3)
		passive_stiffness = self.params.Kq3 @ th if self.params.Kq3 is not None else np.zeros(3)
		rhs = torque - C @ om - G - passive_damping - passive_stiffness
		return np.linalg.solve(M, rhs)
	
	def computed_torque(self, th: np.ndarray, om: np.ndarray, a: np.ndarray) -> np.ndarray:
		# Solves system dynamics: M*q_ddot + C*q_dot + G = tau
		M = self.mass_matrix(th)
		C = self.coriolis_matrix(th, om)
		G = self.gravity_vector(th)
		passive_damping = self.params.Dq3 @ om if self.params.Dq3 is not None else np.zeros(3)
		passive_stiffness = self.params.Kq3 @ th if self.params.Kq3 is not None else np.zeros(3)
		torque = M @ a + C @ om + G + passive_damping + passive_stiffness
		return torque

	def _state_derivative(self, z: np.ndarray, torque: np.ndarray) -> np.ndarray:
		# f(x) and g(x) combined: dz/dt = [om, q_ddot]
		th = z[:3]
		om = z[3:]
		th_dd = self.forward_dynamics(th, om, torque)
		return np.hstack([om, th_dd])

	def joint_dynamics_step(
		self,
		z: np.ndarray,
		u: np.ndarray,
		dt: float,
		add_process_noise: bool = False,
	) -> np.ndarray:
		discretization = getattr(self.params, "discretization_method", "exact")

		w = np.zeros(3)
		if add_process_noise and getattr(self.params, "enable_disturbance", False):
			if self.preloaded_noise is not None:
				if self.noise_step_counter < len(self.preloaded_noise):
					w = self.preloaded_noise[self.noise_step_counter].copy()
					self.noise_step_counter += 1
				else:
					print(f"WARNING: Preloaded noise exhausted at step {self.noise_step_counter}")
					w = np.zeros(3)
			else:
				sigma = self.params.process_noise_std
				clip_bound = getattr(self.params, "process_noise_clip", 0.01)
				max_norm = np.sqrt(3) * clip_bound
				w_raw = self.rng.normal(0, sigma, 3)
				norm = np.linalg.norm(w_raw)
				w = w_raw * (max_norm / norm) if norm > max_norm and norm > 0 else w_raw

		self.last_process_noise = w.copy()
		self.process_noise_history.append(w.copy())
		umax = float(getattr(self.params, "umax_joint", 8.0))
		torque_clipped = np.clip(u, -umax, umax)
		torque_with_noise = np.asarray(torque_clipped, dtype=float) + w
		integration_method = getattr(self.params, "predictor_integration_method", "euler").lower()
		if discretization == "euler" or integration_method == "euler":
			z_next = z + dt * self._state_derivative(z, torque_with_noise)
		else:
			k1 = self._state_derivative(z, torque_with_noise)
			k2 = self._state_derivative(z + 0.5 * dt * k1, torque_with_noise)
			k3 = self._state_derivative(z + 0.5 * dt * k2, torque_with_noise)
			k4 = self._state_derivative(z + dt * k3, torque_with_noise)
			z_next = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

		return z_next

	def fk_points(self, th: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		t1, t2, t3 = th[:3]
		L1, L2, L3 = self.params.L1, self.params.L2, self.params.L3
		base = np.array([0.0, 0.0])
		p1 = np.array([L1 * np.cos(t1), L1 * np.sin(t1)])
		p2 = p1 + np.array([L2 * np.cos(t1 + t2), L2 * np.sin(t1 + t2)])
		ee = p2 + np.array([L3 * np.cos(t1 + t2 + t3), L3 * np.sin(t1 + t2 + t3)])
		return base, p1, p2, ee

	def jac_point(self, th: np.ndarray, link_id: int, s: float = 1.0) -> np.ndarray:
		t1, t2, t3 = th[:3]
		L1, L2, L3 = self.params.L1, self.params.L2, self.params.L3

		if link_id == 1:
			J = np.zeros((2, 3))
			J[:, 0] = [-s * L1 * np.sin(t1), s * L1 * np.cos(t1)]
			return J
		if link_id == 2:
			J = np.zeros((2, 3))
			L2_actual = s * L2
			J[:, 0] = [-L1 * np.sin(t1) - L2_actual * np.sin(t1 + t2), L1 * np.cos(t1) + L2_actual * np.cos(t1 + t2)]
			J[:, 1] = [-s * L2 * np.sin(t1 + t2), s * L2 * np.cos(t1 + t2)]
			return J

		J = np.zeros((2, 3))
		J[:, 0] = [
			-self.params.L1 * np.sin(t1) - self.params.L2 * np.sin(t1 + t2) - s * L3 * np.sin(t1 + t2 + t3),
			self.params.L1 * np.cos(t1) + self.params.L2 * np.cos(t1 + t2) + s * L3 * np.cos(t1 + t2 + t3),
		]
		J[:, 1] = [
			-self.params.L2 * np.sin(t1 + t2) - s * L3 * np.sin(t1 + t2 + t3),
			self.params.L2 * np.cos(t1 + t2) + s * L3 * np.cos(t1 + t2 + t3),
		]
		J[:, 2] = [-s * L3 * np.sin(t1 + t2 + t3), s * L3 * np.cos(t1 + t2 + t3)]
		return J

	def jac_ee(self, th: np.ndarray) -> np.ndarray:
		L1, L2, L3 = self.params.L1, self.params.L2, self.params.L3
		th1, th2, th3 = th[:3]

		dx_dth1 = -L1 * np.sin(th1) - L2 * np.sin(th1 + th2) - L3 * np.sin(th1 + th2 + th3)
		dx_dth2 = -L2 * np.sin(th1 + th2) - L3 * np.sin(th1 + th2 + th3)
		dx_dth3 = -L3 * np.sin(th1 + th2 + th3)
		dy_dth1 = L1 * np.cos(th1) + L2 * np.cos(th1 + th2) + L3 * np.cos(th1 + th2 + th3)
		dy_dth2 = L2 * np.cos(th1 + th2) + L3 * np.cos(th1 + th2 + th3)
		dy_dth3 = L3 * np.cos(th1 + th2 + th3)

		return np.array([[dx_dth1, dx_dth2, dx_dth3], [dy_dth1, dy_dth2, dy_dth3]])

	def jac_dot_point(self, th: np.ndarray, om: np.ndarray, link_id: int, s: float = 1.0) -> np.ndarray:
		t1, t2, t3 = th[:3]
		L1, L2, L3 = self.params.L1, self.params.L2, self.params.L3
		th1, th2, th3 = th[:3]
		om1, om2, om3 = om[:3]

		if link_id == 1:
			L_actual = s * L1
			return np.array([
				[-L_actual * np.cos(th1) * om1, 0.0, 0.0],
				[-L_actual * np.sin(th1) * om1, 0.0, 0.0],
			])

		if link_id == 2:
			L_actual = s * L2
			sum_om12 = om1 + om2
			return np.array([
				[-L1 * np.cos(th1) * om1 - L_actual * np.cos(th1 + th2) * sum_om12, -L_actual * np.cos(th1 + th2) * sum_om12, 0.0],
				[-L1 * np.sin(th1) * om1 - L_actual * np.sin(th1 + th2) * sum_om12, -L_actual * np.sin(th1 + th2) * sum_om12, 0.0],
			])

		L_actual = s * L3
		sum_om12 = om1 + om2
		sum_om123 = om1 + om2 + om3
		return np.array([
			[
				-L1 * np.cos(th1) * om1 - L2 * np.cos(th1 + th2) * sum_om12 - L_actual * np.cos(th1 + th2 + th3) * sum_om123,
				-L2 * np.cos(th1 + th2) * sum_om12 - L_actual * np.cos(th1 + th2 + th3) * sum_om123,
				-L_actual * np.cos(th1 + th2 + th3) * sum_om123,
			],
			[
				-L1 * np.sin(th1) * om1 - L2 * np.sin(th1 + th2) * sum_om12 - L_actual * np.sin(th1 + th2 + th3) * sum_om123,
				-L2 * np.sin(th1 + th2) * sum_om12 - L_actual * np.sin(th1 + th2 + th3) * sum_om123,
				-L_actual * np.sin(th1 + th2 + th3) * sum_om123,
			],
		])
	
	def inverse_kinematics(self,point: float) -> np.ndarray:
		from scipy.optimize import minimize
		L1,L2,L3 = self.link_lengths
		px, py = point
		def objective(angles: np.ndarray) -> float:
			t1, t2, t3 = angles
			c1, s1 = np.cos(t1), np.sin(t1)
			c12, s12 = np.cos(t1 + t2), np.sin(t1 + t2)
			c123, s123 = np.cos(t1 + t2 + t3), np.sin(t1 + t2 + t3)
			ee_x = L1 * c1 + L2 * c12 + L3 * c123
			ee_y = L1 * s1 + L2 * s12 + L3 * s123
			return (ee_x - px) ** 2 + (ee_y - py) ** 2

		best_result = None
		best_error = float("inf")
		guesses = [[0.0, 0.0, 0.0], [np.pi / 4, np.pi / 4, 0.0], [-np.pi / 4, np.pi / 2, 0.0], [-0.574, 1.491, 0.755]]
		for guess in guesses:
			try:
				result = minimize(objective, guess, method="BFGS")
				if result.success and result.fun < best_error:
					best_error = result.fun
					best_result = result
					if result.fun < 1e-10:
						break
			except Exception:
				continue
		if best_result is not None and best_error < 1e-6:
			return best_result.x
		raise RuntimeError(f"Inverse kinematics failed for point {point}. Best error was {best_error:.3e}.")
		return np.array([0.0, np.pi / 2, 0.0])