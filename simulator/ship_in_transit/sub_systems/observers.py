"""
Observer and filtering utilities for ship state estimation.
"""

# ADDED/CHANGED: EKF base class and ship observer implementation

from __future__ import annotations

from typing import Optional

import numpy as np


class IExtendedKalmanFilter:
	"""
	Minimal EKF base class for nonlinear systems.

	Subclasses must implement:
	- f(x, u): state transition
	- dfdx(x, u): Jacobian of f wrt x
	- h(x, u): measurement model
	- dhdx(x, u): Jacobian of h wrt x
	"""
	def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray, dt: float):
		self.x = np.array(x0, dtype=float)
		self.P = np.array(P0, dtype=float)
		# Q og R brukes for tuning, ikke for tilfeldig støy
		self.Q = np.array(Q, dtype=float)
		self.R = np.array(R, dtype=float)
		self.dt = float(dt)

	def f(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
		raise NotImplementedError

	def dfdx(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
		raise NotImplementedError

	def h(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
		raise NotImplementedError

	def dhdx(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
		raise NotImplementedError

	def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
		F = self.dfdx(self.x, u)
		self.x = self.f(self.x, u)
		self.P = F @ self.P @ F.T + self.Q
		return self.x

	def apply_measurement_noise(self, y: np.ndarray) -> np.ndarray:
		if not hasattr(self, 'measurement_noise_std'):
			self.measurement_noise_std = 0.5
		if not hasattr(self, 'bias_noise_std'):
			self.bias_noise_std = 0.02
		if not hasattr(self, 'bias'):
			self.bias = np.zeros_like(y)
		if not hasattr(self, 'white_noise_log'):
			self.white_noise_log = []
		if not hasattr(self, 'bias_log'):
			self.bias_log = []
		if not hasattr(self, 'total_noise_log'):
			self.total_noise_log = []

		white_noise = np.random.normal(0, self.measurement_noise_std, size=y.shape)
		self.bias += np.random.normal(0, self.bias_noise_std, size=y.shape)
		total_noise = white_noise + self.bias
		y_noisy = y + total_noise

		self.white_noise_log.append(white_noise.copy())
		self.bias_log.append(self.bias.copy())
		self.total_noise_log.append(total_noise.copy())
		return y_noisy

	def update(self, y: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
		H = self.dhdx(self.x, u)
		y_hat = self.h(self.x, u)
		S = H @ self.P @ H.T + self.R
		K = self.P @ H.T @ np.linalg.inv(S)
		self.x = self.x + K @ (y - y_hat)
		I = np.eye(self.P.shape[0])
		self.P = (I - K @ H) @ self.P
		return self.x

	def reset(self, x0: Optional[np.ndarray] = None, P0: Optional[np.ndarray] = None) -> None:
		if x0 is not None:
			self.x = np.array(x0, dtype=float)
		if P0 is not None:
			self.P = np.array(P0, dtype=float)


class ShipObserverEKF(IExtendedKalmanFilter):

	"""
	EKF for a 3-DOF kinematic ship model.

	State: x = [north, east, yaw, u, v, r]
	Measurement: y = [north, east, yaw, speed]
	"""
	def __init__(
		self,
		dt: float,
		x0: Optional[np.ndarray] = None,
		P0: Optional[np.ndarray] = None,
		Q: Optional[np.ndarray] = None,
		R: Optional[np.ndarray] = None,
		speed_eps: float = 1e-6,
		velocity_decay: float = 0.08,
		sway_decay: float = 0.20,
		yaw_rate_decay: float = 0.25,
	):
		if x0 is None:
			x0 = np.zeros(6, dtype=float)
		if P0 is None:
			P0 = np.eye(6, dtype=float) * 1e-3
		if Q is None:
			Q = np.diag([0.01, 0.01, 1e-4, 0.05, 0.05, 0.01])
		if R is None:
			R = np.diag([4.0, 4.0, 0.01, 0.25])
		super().__init__(x0=x0, P0=P0, Q=Q, R=R, dt=dt)
		self.speed_eps = speed_eps
		self.velocity_decay = float(velocity_decay)
		self.sway_decay = float(sway_decay)
		self.yaw_rate_decay = float(yaw_rate_decay)

	def f(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
		north, east, yaw, u_b, v_b, r = x
		dt = self.dt
		u_b_next = u_b * max(0.0, 1.0 - self.velocity_decay * dt)
		v_b_next = v_b * max(0.0, 1.0 - self.sway_decay * dt)
		r_next = r * max(0.0, 1.0 - self.yaw_rate_decay * dt)
		north_next = north + dt * (u_b_next * np.cos(yaw) - v_b_next * np.sin(yaw))
		east_next = east + dt * (u_b_next * np.sin(yaw) + v_b_next * np.cos(yaw))
		yaw_next = yaw + dt * r_next
		return np.array([north_next, east_next, yaw_next, u_b_next, v_b_next, r_next], dtype=float)

	def dfdx(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
		_, _, yaw, u_b, v_b, _ = x
		dt = self.dt
		u_decay = max(0.0, 1.0 - self.velocity_decay * dt)
		v_decay = max(0.0, 1.0 - self.sway_decay * dt)
		r_decay = max(0.0, 1.0 - self.yaw_rate_decay * dt)
		F = np.eye(6)
		F[0, 2] = dt * (-u_b * u_decay * np.sin(yaw) - v_b * v_decay * np.cos(yaw))
		F[0, 3] = dt * u_decay * np.cos(yaw)
		F[0, 4] = -dt * v_decay * np.sin(yaw)
		F[1, 2] = dt * (u_b * u_decay * np.cos(yaw) - v_b * v_decay * np.sin(yaw))
		F[1, 3] = dt * u_decay * np.sin(yaw)
		F[1, 4] = dt * v_decay * np.cos(yaw)
		F[2, 5] = dt * r_decay
		F[3, 3] = u_decay
		F[4, 4] = v_decay
		F[5, 5] = r_decay
		return F

	def h(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
		north, east, yaw, u_b, v_b, _ = x
		speed = np.hypot(u_b, v_b)
		return np.array([north, east, yaw, speed], dtype=float)

	def dhdx(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
		_, _, _, u_b, v_b, _ = x
		speed = np.hypot(u_b, v_b)
		H = np.zeros((4, 6))
		H[0, 0] = 1.0  # north
		H[1, 1] = 1.0  # east
		H[2, 2] = 1.0  # yaw
		if speed > self.speed_eps:
			H[3, 3] = u_b / speed
			H[3, 4] = v_b / speed
		return H