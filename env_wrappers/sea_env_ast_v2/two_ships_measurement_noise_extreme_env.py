import numpy as np

from env_wrappers.sea_env_ast_v2.two_ships_measurement_noise_env import TwoShipsMeasurementNoiseEnv


class TwoShipsMeasurementNoiseExtremeEnv(TwoShipsMeasurementNoiseEnv):
    """
    Experimental two-ship measurement-noise environment with intentionally
    extreme/unrealistic observer-noise ranges. Use this variant to probe how
    much adversarial signal corruption is required before closed-loop behavior
    degrades noticeably.
    """

    def init_action_space(self):
        super().init_action_space()

        max_scale = float(getattr(self.args, "extreme_noise_max_scale", 20.0))
        max_scale = max(2.0, max_scale)
        action_gain = float(getattr(self.args, "extreme_action_gain", 1.5))
        action_gain = max(1.0, action_gain)
        self.linear_scale_penalty_gain = float(getattr(self.args, "extreme_linear_scale_penalty_gain", 0.001))
        self.cumulative_linear_scale_penalty_gain = float(getattr(self.args, "extreme_cumulative_linear_scale_penalty_gain", 0.00001))

        # Keep this environment one-sided (only amplified noise), but allow much
        # larger multipliers than the regular training environment.
        self.measurement_noise_attack_mode = "increase_only"
        self.obs_tuning_range = {
            "noise_pos": np.array([1.0, max_scale], dtype=np.float32),
            "noise_yaw": np.array([1.0, max_scale], dtype=np.float32),
            "noise_speed": np.array([1.0, max_scale], dtype=np.float32),
            "noise_bias": np.array([1.0, max_scale], dtype=np.float32),
        }
        self.obs_tuning_nominal = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.obs_tuning_prev = self.obs_tuning_nominal.copy()
        self.extreme_action_gain = action_gain

        # In this stress-test variant, "realistic band" limits are widened so
        # the agent can stay high without immediate heavy realism penalties.
        self.realistic_noise_lower = self.obs_tuning_nominal.copy()
        band_upper = max(4.0, min(max_scale, 8.0))
        self.realistic_noise_upper = np.array([band_upper, band_upper, band_upper, band_upper], dtype=np.float32)
        self.noise_penalty_deadband = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.last_linear_scale_load = 0.0
        self.cumulative_linear_scale_load = 0.0

    def step(self, action_norm):
        action_arr = np.asarray(action_norm, dtype=np.float32).reshape(-1)
        amplified_action = np.clip(self.extreme_action_gain * action_arr, -1.0, 1.0)
        return super().step(amplified_action)

    def reset(self, seed=None, options=None, route_idx=None):
        self.last_linear_scale_load = 0.0
        self.cumulative_linear_scale_load = 0.0
        return super().reset(seed=seed, options=options, route_idx=route_idx)

    def reward_function(self, action, **kwargs):
        kwargs.setdefault("attack_load_penalty", 0.0)
        kwargs.setdefault("cumulative_attack_penalty", 0.0)
        kwargs.setdefault("nominal_drift_penalty", 0.0)
        kwargs.setdefault("band_violation_penalty", 0.0)
        kwargs.setdefault("cumulative_band_violation_penalty", 0.0)
        kwargs.setdefault("action_change_penalty", 0.0)

        # Reward should depend only on actual ship behavior: path following,
        # collision, grounding, and failure. A small linear penalty discourages
        # unnecessarily large actions without adding a hard threshold.
        legacy_runtime_flag = bool(getattr(self.args, "legacy_two_ship_measurement_noise_runtime", False))
        self.args.legacy_two_ship_measurement_noise_runtime = False
        try:
            reward = super().reward_function(action, **kwargs)
        finally:
            self.args.legacy_two_ship_measurement_noise_runtime = legacy_runtime_flag

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        linear_excess = np.maximum(action_arr - 1.0, 0.0)
        self.last_linear_scale_load = float(np.sum(linear_excess))
        self.cumulative_linear_scale_load += self.last_linear_scale_load
        reward -= self.linear_scale_penalty_gain * self.last_linear_scale_load
        reward -= self.cumulative_linear_scale_penalty_gain * self.cumulative_linear_scale_load
        return reward
