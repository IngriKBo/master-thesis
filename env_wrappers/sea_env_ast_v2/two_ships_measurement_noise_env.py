import numpy as np

from env_wrappers.sea_env_ast_v2.measurement_noise_env import SeaEnvMeasurementNoiseAST


class TwoShipsMeasurementNoiseEnv(SeaEnvMeasurementNoiseAST):
    """
    Two-ship AST environment where the RL agent perturbs observer noise on the
    main ship only. The passive ship remains nominal and uses no observer.

    By default this variant does not terminate early on ship proximity. The
    simulator's native stop logic and both ships' ColAV controllers determine
    whether the route is completed, matching the nominal two-ship setup more
    closely.
    """

    def __init__(self, assets, map, wave_model_config, current_model_config, wind_model_config, args, **kwargs):
        super().__init__(assets, map, wave_model_config, current_model_config, wind_model_config, args, **kwargs)
        self.collision_distance = getattr(args, 'collision_distance', 20.0)
        self.close_reward_distance = getattr(args, 'close_reward_distance', 250.0)
        self.close_reward_gain = getattr(args, 'close_reward_gain', 0.0)
        self.collision_reward = getattr(args, 'collision_reward', 100.0)
        self.terminate_on_proximity = getattr(args, 'terminate_on_proximity', False)
        self.max_steps = getattr(args, 'max_steps', None)
        self.current_step = 0

    def step(self, action_norm):
        obs, reward, terminated, truncated, info = super().step(action_norm)

        rl_ship = self.assets[0].ship_model
        passive_ship = self.assets[1].ship_model
        dist = np.hypot(
            rl_ship.north - passive_ship.north,
            rl_ship.east - passive_ship.east,
        )
        closeness_reward = 0.0
        if self.close_reward_gain > 0.0:
            closeness_reward = self.close_reward_gain * np.exp(-dist / max(self.close_reward_distance, 1.0))
        reward += closeness_reward

        proximity_breach = dist < self.collision_distance
        if proximity_breach and self.terminate_on_proximity:
            terminated = True

        self.current_step += 1
        if self.max_steps is not None and self.current_step >= self.max_steps:
            terminated = True

        native_collision = bool(rl_ship.stop_info.get('collision', False) or passive_ship.stop_info.get('collision', False))
        if native_collision and self.collision_reward > 0.0:
            reward += self.collision_reward

        info['distance'] = dist
        info['closeness_reward'] = closeness_reward
        info['proximity_breach'] = proximity_breach
        info['collision'] = native_collision
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, route_idx=None):
        self.current_step = 0
        return super().reset(seed=seed, options=options, route_idx=route_idx)