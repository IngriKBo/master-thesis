"""
AST environment variant where the RL agent perturbs observer measurement noise
instead of observer covariance tuning.
"""
from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box

from env_wrappers.sea_env_ast_v2.estimator_tuning_env import SeaEnvEstimatorTuningAST
from env_wrappers.sea_env_ast_v2.env import collision_flag


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
        configured_attack_mode = getattr(self.args, "measurement_noise_attack_mode", None)
        if configured_attack_mode is None:
            configured_attack_mode = "symmetric_band" if bool(getattr(self.args, "allow_subnominal_noise", False)) else "increase_only"
        legacy_two_ship_runtime = bool(getattr(self.args, "legacy_two_ship_measurement_noise_runtime", False))

        attack_mode = str(configured_attack_mode).strip().lower()
        if attack_mode not in {"increase_only", "symmetric_band"}:
            raise ValueError(
                f"Unknown measurement noise attack mode '{configured_attack_mode}'. "
                "Expected 'increase_only' or 'symmetric_band'."
            )

        self.measurement_noise_attack_mode = attack_mode
        noise_lower_bound = np.array([1.00, 1.00, 1.00, 1.00], dtype=np.float32)
        if self.measurement_noise_attack_mode == "symmetric_band":
            noise_lower_bound = np.array([0.80, 0.80, 0.80, 0.65], dtype=np.float32)
        noise_upper_bound = np.array([1.60, 1.60, 1.60, 2.00], dtype=np.float32)
        if legacy_two_ship_runtime:
            noise_upper_bound = np.array([1.60, 1.60, 1.60, 2.00], dtype=np.float32)

        self.obs_tuning_range = {
            "noise_pos": np.array([noise_lower_bound[0], noise_upper_bound[0]], dtype=np.float32),
            "noise_yaw": np.array([noise_lower_bound[1], noise_upper_bound[1]], dtype=np.float32),
            "noise_speed": np.array([noise_lower_bound[2], noise_upper_bound[2]], dtype=np.float32),
            "noise_bias": np.array([noise_lower_bound[3], noise_upper_bound[3]], dtype=np.float32),
        }
        self.obs_tuning_nominal = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.obs_tuning_prev = self.obs_tuning_nominal.copy()
        self.realistic_noise_lower = self.obs_tuning_nominal.copy()
        if self.measurement_noise_attack_mode == "symmetric_band":
            self.realistic_noise_lower = np.array([0.80, 0.80, 0.80, 0.65], dtype=np.float32)
        self.realistic_noise_upper = np.array([1.20, 1.20, 1.20, 1.35], dtype=np.float32)
        self.action_space = Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.cumulative_noise_attack_load = 0.0
        self.cumulative_noise_band_violation = 0.0
        self.cumulative_action_change = 0.0
        self.last_noise_attack_load = 0.0
        self.last_noise_nominal_drift = 0.0
        self.last_noise_band_violation = 0.0
        self.last_action_change = 0.0

        # Backward-compatible aliases used by older analysis code.
        self.cumulative_noise_exposure = 0.0
        self.last_noise_average_deviation = 0.0
        self.last_noise_exposure = 0.0
        self.noise_penalty_deadband = np.array(
            getattr(self.args, "measurement_noise_penalty_deadband", [0.05, 0.05, 0.05, 0.08]),
            dtype=np.float32,
        )
        if self.noise_penalty_deadband.shape != (4,):
            raise ValueError(
                "measurement_noise_penalty_deadband must contain four values: "
                "[pos, yaw, speed, bias]."
            )

    def _compute_noise_metrics(self, clipped):
        clipped_arr = np.asarray(clipped, dtype=np.float32)
        delta = clipped_arr - self.obs_tuning_nominal
        abs_delta = np.maximum(np.abs(delta) - self.noise_penalty_deadband, 0.0)
        positive_delta = np.maximum(delta - self.noise_penalty_deadband, 0.0)
        nominal_drift = float(np.sum(abs_delta))
        attack_load = float(np.sum(positive_delta))
        band_violation = float(
            np.sum(np.maximum(self.realistic_noise_lower - clipped_arr, 0.0))
            + np.sum(np.maximum(clipped_arr - self.realistic_noise_upper, 0.0))
        )
        return nominal_drift, attack_load, band_violation

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

        nominal_drift, attack_load, band_violation = self._compute_noise_metrics(clipped)
        action_change = float(np.sum(np.abs(clipped - self.obs_tuning_prev)))

        self.last_noise_nominal_drift = nominal_drift
        self.last_noise_attack_load = attack_load
        self.last_noise_band_violation = band_violation
        self.last_action_change = action_change
        self.cumulative_noise_attack_load += attack_load
        self.cumulative_noise_band_violation += band_violation
        self.cumulative_action_change += action_change
        self.obs_tuning_prev = clipped.copy()

        self.last_noise_average_deviation = nominal_drift
        self.last_noise_exposure = band_violation
        self.cumulative_noise_exposure = self.cumulative_noise_band_violation

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
        self.cumulative_noise_attack_load = 0.0
        self.cumulative_noise_band_violation = 0.0
        self.cumulative_action_change = 0.0
        self.last_noise_attack_load = 0.0
        self.last_noise_nominal_drift = 0.0
        self.last_noise_band_violation = 0.0
        self.last_action_change = 0.0
        self.cumulative_noise_exposure = 0.0
        self.last_noise_average_deviation = 0.0
        self.last_noise_exposure = 0.0

        return super(SeaEnvEstimatorTuningAST, self).reset(seed=seed, options=options, route_idx=route_idx)

    def reward_function(
        self,
        action,
        eta=0.0,
        theta=1.0,
        cte_gain=1.60,
        cte_cap=3.50,
        attack_load_penalty=0.10,
        cumulative_attack_penalty=0.015,
        nominal_drift_penalty=0.06,
        band_violation_penalty=0.60,
        cumulative_band_violation_penalty=0.05,
        action_change_penalty=0.06,
        failure_reward=35.0,
        grounding_reward=50.0,
        endpoint_penalty=50.0,
    ):
        legacy_two_ship_runtime = bool(getattr(self.args, "legacy_two_ship_measurement_noise_runtime", False))
        if legacy_two_ship_runtime:
            cte_gain = 1.25
            cte_cap = 3.0
            attack_load_penalty = 0.08
            cumulative_attack_penalty = 0.01
            nominal_drift_penalty = 0.06
            band_violation_penalty = 0.60
            cumulative_band_violation_penalty = 0.04
            action_change_penalty = 0.0

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

        if self.measurement_noise_attack_mode == "increase_only":
            reward -= attack_load_penalty * self.last_noise_attack_load
            reward -= cumulative_attack_penalty * self.cumulative_noise_attack_load
        else:
            reward -= nominal_drift_penalty * self.last_noise_nominal_drift

        reward -= band_violation_penalty * self.last_noise_band_violation
        reward -= cumulative_band_violation_penalty * self.cumulative_noise_band_violation
        reward -= action_change_penalty * self.last_action_change

        collision = collision_flag(self.assets[0].ship_model.stop_info['collision'])
        grounding_failure = self.assets[0].ship_model.stop_info['grounding_failure']
        navigation_failure = self.assets[0].ship_model.stop_info['navigation_failure']
        reaches_endpoint = self.assets[0].ship_model.stop_info['reaches_endpoint']
        outside_horizon = self.assets[0].ship_model.stop_info['outside_horizon']
        power_overload = self.assets[0].ship_model.stop_info['power_overload']

        if outside_horizon:
            reward -= endpoint_penalty
        elif collision or power_overload or navigation_failure:
            reward += failure_reward
        elif grounding_failure:
            reward += grounding_reward
        elif reaches_endpoint:
            reward -= endpoint_penalty

        return reward
