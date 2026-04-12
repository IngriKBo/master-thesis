
import gymnasium as gym

from env_wrappers.sea_env_ast_v2.observer_env import SeaEnvObserverAST
import numpy as np

class ObserverTwoShipsEnv(SeaEnvObserverAST):
    """
    AST environment for two ships scenario, identisk observerstyring som SeaEnvObserverAST, men med ekstra reward for nærhet/kollisjon mellom to skip.
    Agenten styrer kun assets[0], assets[1] er passivt skip.
    """
    def __init__(self, assets, map, wave_model_config, current_model_config, wind_model_config, args, **kwargs):
        super().__init__(assets, map, wave_model_config, current_model_config, wind_model_config, args, **kwargs)
        self.collision_distance = getattr(args, 'collision_distance', 20.0)
        self.max_steps = getattr(args, 'max_steps', 1000)
        self.current_step = 0

    def step(self, action_norm):
        # Bruk observerstyring fra SeaEnvObserverAST
        obs, reward, terminated, truncated, info = super().step(action_norm)

        # Ekstra reward for nærhet/kollisjon mellom skipene
        rl_ship = self.assets[0].ship_model
        passive_ship = self.assets[1].ship_model
        dist = np.hypot(
            rl_ship.north - passive_ship.north,
            rl_ship.east - passive_ship.east
        )
        # Debug: print avstand mellom skipene for hvert steg
        print(f"[DEBUG] Step {self.current_step}: RL_ship (N={rl_ship.north:.2f}, E={rl_ship.east:.2f}), Passive_ship (N={passive_ship.north:.2f}, E={passive_ship.east:.2f}), Avstand={dist:.2f}")
        # Legg til nærhetsstraff og kollisjonsbonus
        reward += -dist * 0.01
        collision = dist < self.collision_distance
        if collision:
            reward += 100.0
            terminated = True
        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True
        info['distance'] = dist
        info['collision'] = collision
        return obs, reward, terminated, truncated, info
