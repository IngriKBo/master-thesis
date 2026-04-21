"""
AST environment variant where the RL agent perturbs observer measurement noise
instead of observer covariance tuning.
"""
from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box

from env_wrappers.sea_env_ast_v2.estimator_tuning_env import SeaEnvEstimatorTuningAST


class SeaEnvMeasurementNoiseAST(SeaEnvEstimatorTuningAST):
    """
    RL agent scales observer measurement noise instead of Q/R.

    Action: [alpha_pos, alpha_yaw, alpha_speed, alpha_bias]
      - alpha_pos   : scale for position measurement std ([0], [1])
      - alpha_yaw   : scale for heading measurement std ([2])
      - alpha_speed : scale for speed measurement std ([3])
      - alpha_bias  : scale for slowly varying bias random-walk std
    """

    def init_action_space(self):
        self.obs_tuning_range = {
            "noise_pos": np.array([0.80, 1.60], dtype=np.float32),
            "noise_yaw": np.array([0.80, 1.60], dtype=np.float32),
            "noise_speed": np.array([0.80, 1.60], dtype=np.float32),
            "noise_bias": np.array([0.80, 2.00], dtype=np.float32),
        }
        self.obs_tuning_nominal = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.obs_tuning_prev = self.obs_tuning_nominal.copy()
        self.realistic_noise_upper = np.array([1.20, 1.20, 1.20, 1.35], dtype=np.float32)
        self.action_space = Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.cumulative_noise_exposure = 0.0
        self.last_noise_average_deviation = 0.0
        self.last_noise_exposure = 0.0

    def _denormalize_action(self, action_norm):
        action_arr = np.asarray(action_norm, dtype=np.float32).reshape(-1)
        alpha_pos = self._denormalize(action_arr[0], self.obs_tuning_range["noise_pos"][0], self.obs_tuning_range["noise_pos"][1])
        alpha_yaw = self._denormalize(action_arr[1], self.obs_tuning_range["noise_yaw"][0], self.obs_tuning_range["noise_yaw"][1])
        alpha_speed = self._denormalize(action_arr[2], self.obs_tuning_range["noise_speed"][0], self.obs_tuning_range["noise_speed"][1])
        alpha_bias = self._denormalize(action_arr[3], self.obs_tuning_range["noise_bias"][0], self.obs_tuning_range["noise_bias"][1])
        return alpha_pos, alpha_yaw, alpha_speed, alpha_bias

    def _normalize_action(self, action):
        alpha_pos, alpha_yaw, alpha_speed, alpha_bias = action
        a0 = self._normalize(alpha_pos, self.obs_tuning_range["noise_pos"][0], self.obs_tuning_range["noise_pos"][1])
        a1 = self._normalize(alpha_yaw, self.obs_tuning_range["noise_yaw"][0], self.obs_tuning_range["noise_yaw"][1])
        a2 = self._normalize(alpha_speed, self.obs_tuning_range["noise_speed"][0], self.obs_tuning_range["noise_speed"][1])
        a3 = self._normalize(alpha_bias, self.obs_tuning_range["noise_bias"][0], self.obs_tuning_range["noise_bias"][1])
        return a0, a1, a2, a3

    def _apply_observer_tuning(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        low = np.array([
            self.obs_tuning_range["noise_pos"][0],
            self.obs_tuning_range["noise_yaw"][0],
            self.obs_tuning_range["noise_speed"][0],
            self.obs_tuning_range["noise_bias"][0],
        ], dtype=np.float32)
        high = np.array([
            self.obs_tuning_range["noise_pos"][1],
            self.obs_tuning_range["noise_yaw"][1],
            self.obs_tuning_range["noise_speed"][1],
            self.obs_tuning_range["noise_bias"][1],
        ], dtype=np.float32)
        clipped = np.clip(action_arr, low, high)

        average_deviation = float(np.sum(np.abs(clipped - self.obs_tuning_nominal)))
        elevated_exposure = float(np.sum(np.maximum(clipped - self.realistic_noise_upper, 0.0)))

        self.last_noise_average_deviation = average_deviation
        self.last_noise_exposure = elevated_exposure
        self.cumulative_noise_exposure += elevated_exposure
        self.obs_tuning_prev = clipped.copy()

        alpha_pos, alpha_yaw, alpha_speed, alpha_bias = tuple(float(value) for value in clipped)
        self._require_observer_attached()
        observer = self.assets[0].ship_model.observer

        if not hasattr(observer, "measurement_noise_nominal"):
            observer.measurement_noise_nominal = np.array(observer.measurement_noise_std, dtype=float)
        if not hasattr(observer, "bias_noise_nominal"):
            observer.bias_noise_nominal = np.array(observer.bias_noise_std, dtype=float)

        observer.measurement_noise_std = observer.measurement_noise_nominal.copy()
        observer.measurement_noise_std[0] = observer.measurement_noise_nominal[0] * alpha_pos
        observer.measurement_noise_std[1] = observer.measurement_noise_nominal[1] * alpha_pos
        observer.measurement_noise_std[2] = observer.measurement_noise_nominal[2] * alpha_yaw
        observer.measurement_noise_std[3] = observer.measurement_noise_nominal[3] * alpha_speed
        observer.bias_noise_std = np.array(observer.bias_noise_nominal, dtype=float) * alpha_bias

        return alpha_pos, alpha_yaw, alpha_speed, alpha_bias

    def reset(self, seed=None, options=None, route_idx=None):
        self._require_observer_attached()
        observer = self.assets[0].ship_model.observer

        if not hasattr(observer, "measurement_noise_nominal"):
            observer.measurement_noise_nominal = np.array(observer.measurement_noise_std, dtype=float)
        if not hasattr(observer, "bias_noise_nominal"):
            observer.bias_noise_nominal = np.array(observer.bias_noise_std, dtype=float)

        observer.measurement_noise_std = observer.measurement_noise_nominal.copy()
        observer.bias_noise_std = np.array(observer.bias_noise_nominal, dtype=float).copy()
        self.obs_tuning_prev = self.obs_tuning_nominal.copy()
        self.cumulative_noise_exposure = 0.0
        self.last_noise_average_deviation = 0.0
        self.last_noise_exposure = 0.0

        return super(SeaEnvEstimatorTuningAST, self).reset(seed=seed, options=options, route_idx=route_idx)

    def reward_function(
        self,
        action,
        eta=0.5,
        theta=2.0,
        cte_gain=0.4,
        cte_cap=2.0,
        average_noise_penalty=0.35,
        cumulative_noise_penalty=0.04,
        exposure_penalty=0.20,
    ):
        base_reward = len(self.action_list) * eta * theta
        reward = base_reward

        ship = self.assets[0].ship_model
        e_ct_abs = 0.0
        if ship.auto_pilot is not None:
            try:
                e_ct_abs = abs(float(ship.auto_pilot.navigate.e_ct))
            except Exception:
                e_ct_abs = 0.0

        denom = max(1.0, float(getattr(ship, "cross_track_error_tolerance", 1.0)))
        cte_ratio = e_ct_abs / denom
        reward += cte_gain * min(cte_ratio, cte_cap)

        reward -= average_noise_penalty * self.last_noise_average_deviation
        reward -= cumulative_noise_penalty * self.cumulative_noise_exposure
        reward -= exposure_penalty * self.last_noise_exposure

        collision = self.assets[0].ship_model.stop_info['collision']
        grounding_failure = self.assets[0].ship_model.stop_info['grounding_failure']
        navigation_failure = self.assets[0].ship_model.stop_info['navigation_failure']
        reaches_endpoint = self.assets[0].ship_model.stop_info['reaches_endpoint']
        outside_horizon = self.assets[0].ship_model.stop_info['outside_horizon']
        power_overload = self.assets[0].ship_model.stop_info['power_overload']

        if outside_horizon:
            reward += -50.0
        elif collision or power_overload or navigation_failure:
            reward += 5.0
        elif grounding_failure:
            reward += 25.0
        elif reaches_endpoint:
            reward += -50.0

        return reward
