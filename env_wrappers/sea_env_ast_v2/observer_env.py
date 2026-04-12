"""
AST environment variant where the RL agent perturbs the observer instead of the sea environment.
"""
from __future__ import annotations

import numpy as np

from gymnasium.spaces import Box

from env_wrappers.sea_env_ast_v2.env import SeaEnvASTv2


class SeaEnvObserverAST(SeaEnvASTv2):
    """
    RL agent tunes observer covariance instead of environmental loads.
    Action: [alpha_r_pos, alpha_r_speed, alpha_q]
      - alpha_r_pos   : scale for position measurement covariance (R[0,0], R[1,1])
      - alpha_r_speed : scale for speed measurement covariance (R[2,2])
      - alpha_q       : scale for process covariance (Q)
    """
    def init_action_space(self):
        # Action controls dimensionless covariance multipliers.
        self.obs_tuning_range = {
            "r_pos": np.array([0.01, 10.0], dtype=np.float32),
            "r_speed": np.array([0.01, 10.0], dtype=np.float32),
            "q": np.array([0.01, 10.0], dtype=np.float32)
        }
        self.obs_tuning_nominal = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.obs_tuning_prev = self.obs_tuning_nominal.copy()
        self.action_space = Box(
            low=np.array([self.obs_tuning_range["r_pos"][0], self.obs_tuning_range["r_speed"][0], self.obs_tuning_range["q"][0]], dtype=np.float32),
            high=np.array([self.obs_tuning_range["r_pos"][1], self.obs_tuning_range["r_speed"][1], self.obs_tuning_range["q"][1]], dtype=np.float32)
        )
        # Track cumulative observer tuning (noise) for the episode
        self.cumulative_noise = 0.0


    # _clip_tuning_step fjernet: ingen begrensning på endring per steg

    def _denormalize_action(self, action_norm):
        alpha_r_pos = self._denormalize(action_norm[0], self.obs_tuning_range["r_pos"][0], self.obs_tuning_range["r_pos"][1])
        alpha_r_speed = self._denormalize(action_norm[1], self.obs_tuning_range["r_speed"][0], self.obs_tuning_range["r_speed"][1])
        alpha_q = self._denormalize(action_norm[2], self.obs_tuning_range["q"][0], self.obs_tuning_range["q"][1])
        return alpha_r_pos, alpha_r_speed, alpha_q

    def _normalize_action(self, action):
        alpha_r_pos, alpha_r_speed, alpha_q = action
        a0 = self._normalize(alpha_r_pos, self.obs_tuning_range["r_pos"][0], self.obs_tuning_range["r_pos"][1])
        a1 = self._normalize(alpha_r_speed, self.obs_tuning_range["r_speed"][0], self.obs_tuning_range["r_speed"][1])
        a2 = self._normalize(alpha_q, self.obs_tuning_range["q"][0], self.obs_tuning_range["q"][1])
        return a0, a1, a2

    def _apply_observer_tuning(self, action):
        # Ingen begrensning på endring per steg, kun range
        action_arr = np.asarray(action, dtype=np.float32)
        low = np.array([
            self.obs_tuning_range["r_pos"][0],
            self.obs_tuning_range["r_speed"][0],
            self.obs_tuning_range["q"][0]
        ], dtype=np.float32)
        high = np.array([
            self.obs_tuning_range["r_pos"][1],
            self.obs_tuning_range["r_speed"][1],
            self.obs_tuning_range["q"][1]
        ], dtype=np.float32)
        clipped = np.clip(action_arr, low, high)
        self.obs_tuning_prev = clipped.copy()
        alpha_r_pos, alpha_r_speed, alpha_q = tuple(float(value) for value in clipped)
        self._require_observer_attached()
        observer = self.assets[0].ship_model.observer

        # Cache nominal covariance matrices on first use
        if not hasattr(observer, "Q_nominal"):
            observer.Q_nominal = observer.Q.copy()
        if not hasattr(observer, "R_nominal"):
            observer.R_nominal = observer.R.copy()

        # Apply tunings (keep matrices diagonal/positive)
        observer.Q = observer.Q_nominal * float(alpha_q)
        observer.R = observer.R_nominal.copy()
        observer.R[0, 0] = observer.R_nominal[0, 0] * float(alpha_r_pos)
        observer.R[1, 1] = observer.R_nominal[1, 1] * float(alpha_r_pos)
        observer.R[2, 2] = observer.R_nominal[2, 2] * float(alpha_r_speed)

        # No direct signal tampering in this mode.
        self.assets[0].ship_model.observer_noise_std = np.zeros(3, dtype=float)

        return alpha_r_pos, alpha_r_speed, alpha_q

    def _require_observer_attached(self):
        if not self.assets or self.assets[0].ship_model is None:
            raise RuntimeError("Observer scenario requires at least one ship asset with a ship model.")
        if getattr(self.assets[0].ship_model, "observer", None) is None:
            raise RuntimeError(
                "SeaEnvObserverAST requires an attached observer on assets[0].ship_model.observer. "
                "Attach ShipObserverEKF in setup before training/simulation."
            )

    def _step(self, action=None, env_args=None):
        # ADDED/CHANGED: ignore action for environment loads
        return super()._step(action=None, env_args=env_args)

    def reset(self, seed=None, options=None, route_idx=None):
        self._require_observer_attached()
        observer = self.assets[0].ship_model.observer

        if not hasattr(observer, "Q_nominal"):
            observer.Q_nominal = observer.Q.copy()
        if not hasattr(observer, "R_nominal"):
            observer.R_nominal = observer.R.copy()

        observer.Q = observer.Q_nominal.copy()
        observer.R = observer.R_nominal.copy()
        self.obs_tuning_prev = self.obs_tuning_nominal.copy()
        self.assets[0].ship_model.observer_noise_std = np.zeros(3, dtype=float)

        # Reset cumulative noise at the start of each episode
        self.cumulative_noise = 0.0

        return super().reset(seed=seed, options=options, route_idx=route_idx)

    def step(self, action_norm):
        self._require_observer_attached()

        # Record the when the action is sampled
        self.action_time_list.append(self.assets[0].ship_model.int.time)

        # Denormalize action and apply observer tuning
        action = self._denormalize_action(action_norm)
        action = self._apply_observer_tuning(action)

        # Update cumulative noise (sum of absolute deviation from nominal for all tuning params)
        self.cumulative_noise += np.sum(np.abs(np.array(action) - self.obs_tuning_nominal))

        #------------------------------ Step the simulator ------------------------------#

        # Use a default action_sampling_period if not present in args
        action_sampling_period = getattr(self.args, 'action_sampling_period', 10)
        running_time = 0
        while running_time < action_sampling_period:


            self._step()

            running_time += self.assets[0].ship_model.int.dt

            if np.all(self.ship_stop_status):
                self.terminated = True
                break

            if self.assets[0].ship_model.int.time > self.assets[0].ship_model.simulation_config.simulation_time:
                self.truncated = True
                break

        observation = self._get_obs()
        reward = self.reward_function(action)
        terminated = self.terminated
        truncated = self.truncated
        info = {}

        self.obs_list.append(observation)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.terminated_list.append(terminated)
        self.truncated_list.append(truncated)
        self.info_list.append(info)

        return observation, reward, terminated, truncated, info

    def reward_function(self,
                        action,
                        eta=0.5,
                        theta=2.0,
                        cte_gain=0.4,
                        cte_cap=2.0,
                        pos_tol_m=120.0,
                        speed_tol_mps=1.2,
                        pos_err_penalty=0.0002,
                        speed_err_penalty=0.05,
                        cumulative_noise_penalty=0.1):
        # Reward based on progress + gradual path-deviation reward + realism penalty
        base_reward = len(self.action_list) * eta * theta
        reward = base_reward

        # Gradual reward for deviating from route (not only terminal nav-failure)
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

        # Penalize unrealistic estimation error beyond tolerance (keep attacks plausible)
        pos_err = float(np.hypot(ship.estimated_north - ship.north,
                                 ship.estimated_east - ship.east))
        speed_true = float(np.hypot(ship.forward_speed, ship.sideways_speed))
        speed_err = float(abs(ship.estimated_speed - speed_true))
        reward -= pos_err_penalty * max(0.0, pos_err - pos_tol_m)**2
        reward -= speed_err_penalty * max(0.0, speed_err - speed_tol_mps)**2

        # Penalize cumulative noise (sum of all tuning deviations so far)
        reward -= cumulative_noise_penalty * self.cumulative_noise

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

    def log_RL_transition_text(self, *args, **kwargs):
        # Optional: skip in this variant
        return

    def log_RL_transition_json_csv(self, *args, **kwargs):
        # Optional: skip in this variant
        return
