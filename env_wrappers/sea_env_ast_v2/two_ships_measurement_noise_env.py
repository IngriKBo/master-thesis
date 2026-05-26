import numpy as np

from env_wrappers.sea_env_ast_v2.measurement_noise_env import SeaEnvMeasurementNoiseAST
from env_wrappers.sea_env_ast_v2.env import collision_flag, encounter_metrics


class TwoShipsMeasurementNoiseEnv(SeaEnvMeasurementNoiseAST):
    """
    Two-ship AST environment where the RL agent perturbs observer noise on the
    main ship only. The passive ship remains nominal and uses no observer.

    By default this variant does not terminate early on ship proximity. The
    simulator's native stop logic and both ships' ColAV controllers determine
    whether the route is completed, matching the nominal two-ship setup more
    closely.
    """

    def init_action_space(self):
        super().init_action_space()

        max_scale = float(getattr(self.args, 'two_ship_noise_max_scale', 5.0))
        max_scale = max(1.6, max_scale)

        lower_bounds = {
            key: float(bounds[0])
            for key, bounds in self.obs_tuning_range.items()
        }
        self.obs_tuning_range = {
            'noise_pos': np.array([lower_bounds['noise_pos'], max_scale], dtype=np.float32),
            'noise_yaw': np.array([lower_bounds['noise_yaw'], max_scale], dtype=np.float32),
            'noise_speed': np.array([lower_bounds['noise_speed'], max_scale], dtype=np.float32),
            'noise_bias': np.array([lower_bounds['noise_bias'], max_scale], dtype=np.float32),
        }
        self.obs_tuning_nominal = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.obs_tuning_prev = self.obs_tuning_nominal.copy()

        realistic_upper = np.array(
            getattr(self.args, 'two_ship_realistic_noise_upper', [2.0, 2.0, 2.0, 2.5]),
            dtype=np.float32,
        )
        if realistic_upper.shape != (4,):
            raise ValueError(
                'two_ship_realistic_noise_upper must contain four values: '
                '[pos, yaw, speed, bias].'
            )

        channel_max = np.array([max_scale, max_scale, max_scale, max_scale], dtype=np.float32)
        self.realistic_noise_upper = np.minimum(realistic_upper, channel_max)

    def __init__(self, assets, map, wave_model_config, current_model_config, wind_model_config, args, **kwargs):
        super().__init__(assets, map, wave_model_config, current_model_config, wind_model_config, args, **kwargs)
        legacy_two_ship_runtime = bool(getattr(args, 'legacy_two_ship_measurement_noise_runtime', False))
        self.collision_distance = getattr(args, 'collision_distance', 20.0)
        self.close_reward_distance = getattr(args, 'close_reward_distance', 120.0)
        self.close_reward_gain = getattr(args, 'close_reward_gain', 0.0)
        self.collision_reward = getattr(args, 'collision_reward', 120.0)
        self.dcpa_reward_gain = getattr(args, 'dcpa_reward_gain', 30.0 if legacy_two_ship_runtime else 30.0)
        self.dcpa_reward_distance = getattr(args, 'dcpa_reward_distance', 120.0)
        self.tcpa_reward_horizon = getattr(args, 'tcpa_reward_horizon', 900.0)
        self.tcpa_window_center = getattr(args, 'tcpa_window_center', 240.0)
        self.tcpa_window_width = getattr(args, 'tcpa_window_width', 180.0)
        self.terminate_on_proximity = getattr(args, 'terminate_on_proximity', False)
        self.max_steps = getattr(args, 'max_steps', None)
        self.current_step = 0

    def _encounter_reward(self, distance, dcpa, tcpa):
        if tcpa < 0.0 or tcpa > self.tcpa_reward_horizon:
            return 0.0

        dcpa_score = max(0.0, 1.0 - dcpa / max(self.dcpa_reward_distance, 1.0))
        if dcpa_score <= 0.0:
            return 0.0

        tcpa_offset = abs(tcpa - self.tcpa_window_center)
        tcpa_score = max(0.0, 1.0 - tcpa_offset / max(self.tcpa_window_width, 1.0))
        if tcpa_score <= 0.0:
            return 0.0

        proximity_bonus = 0.0
        if self.close_reward_gain > 0.0:
            proximity_bonus = self.close_reward_gain * np.exp(-distance / max(self.close_reward_distance, 1.0))

        return self.dcpa_reward_gain * dcpa_score * tcpa_score + proximity_bonus

    def step(self, action_norm):
        obs, reward, terminated, truncated, info = super().step(action_norm)

        rl_ship = self.assets[0].ship_model
        passive_ship = self.assets[1].ship_model
        encounter = encounter_metrics(rl_ship, passive_ship)
        dist = float(encounter[0])
        dcpa = float(encounter[2])
        tcpa = float(encounter[3])
        encounter_reward = self._encounter_reward(dist, dcpa, tcpa)
        reward += encounter_reward

        proximity_breach = dist < self.collision_distance
        if proximity_breach and self.terminate_on_proximity:
            terminated = True

        self.current_step += 1
        if self.max_steps is not None and self.current_step >= self.max_steps:
            terminated = True

        native_collision = (
            collision_flag(rl_ship.stop_info.get('collision', False))
            or collision_flag(passive_ship.stop_info.get('collision', False))
        )
        if native_collision and self.collision_reward > 0.0:
            reward += self.collision_reward

        info['distance'] = dist
        info['relative_bearing'] = float(encounter[1])
        info['dcpa'] = dcpa
        info['tcpa'] = tcpa
        info['encounter_reward'] = encounter_reward
        info['closeness_reward'] = encounter_reward
        info['proximity_breach'] = proximity_breach
        info['collision'] = native_collision
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, route_idx=None):
        self.current_step = 0
        return super().reset(seed=seed, options=options, route_idx=route_idx)