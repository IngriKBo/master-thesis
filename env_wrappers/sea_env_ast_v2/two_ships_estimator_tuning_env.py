
import gymnasium as gym

from env_wrappers.sea_env_ast_v2.estimator_tuning_env import SeaEnvEstimatorTuningAST
from env_wrappers.sea_env_ast_v2.env import collision_flag, encounter_metrics
import numpy as np

class TwoShipsEstimatorTuningEnv(SeaEnvEstimatorTuningAST):
    """
    AST environment for two ships scenario, identical estimator tuning as SeaEnvEstimatorTuningAST, with extra reward for ship proximity and collision.
    Agenten styrer kun assets[0], assets[1] er passivt skip.
    """
    def __init__(self, assets, map, wave_model_config, current_model_config, wind_model_config, args, **kwargs):
        super().__init__(assets, map, wave_model_config, current_model_config, wind_model_config, args, **kwargs)
        self.collision_distance = getattr(args, 'collision_distance', 20.0)
        self.close_reward_distance = getattr(args, 'close_reward_distance', 120.0)
        self.close_reward_gain = getattr(args, 'close_reward_gain', 0.0)
        self.collision_reward = getattr(args, 'collision_reward', 120.0)
        self.dcpa_reward_gain = getattr(args, 'dcpa_reward_gain', 30.0)
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
