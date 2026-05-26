from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulator.ship_in_transit.sub_systems.observers import ShipObserverEKF
from test_beds.ast_test.setup import get_env_assets, get_observer_noise_config, DEFAULT_OBSERVER_NOISE_PROFILE
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map

import argparse


def ship_has_collision(ship):
    collision_info = ship.stop_info.get('collision', False)
    if isinstance(collision_info, (list, tuple)):
        return bool(collision_info[0]) if len(collision_info) > 0 else False
    return bool(collision_info)


def print_separation_summary(info, collision=False, target_name=None):
    distance = info.get('distance', info.get('encounter_range'))
    if distance is None:
        return

    try:
        distance_value = float(distance)
    except (TypeError, ValueError):
        return

    label = 'Collision distance' if collision else 'Ship separation at stop'
    if target_name:
        print(f'{label} ({target_name}): {distance_value:.1f} m')
    else:
        print(f'{label}: {distance_value:.1f} m')


parser = argparse.ArgumentParser(description='Two-ship estimator tuning simulation (nominal, no RL tuning)')
parser.add_argument('--time_step', type=float, default=1.0)
parser.add_argument('--observer_time_step', type=float, default=0.2)
parser.add_argument('--engine_step_count', type=int, default=10)
parser.add_argument('--radius_of_acceptance', type=int, default=300)
parser.add_argument('--lookahead_distance', type=int, default=1000)
parser.add_argument('--nav_fail_time', type=int, default=300)
parser.add_argument('--ship_draw', type=bool, default=True)
parser.add_argument('--time_since_last_ship_drawing', type=int, default=30)
parser.add_argument('--map_gpkg_filename', type=str, default='Stangvik.gpkg')
parser.add_argument('--warm_up_time', type=int, default=2500)
parser.add_argument('--action_sampling_period', type=int, default=900)
parser.add_argument('--observer_noise_profile', type=str, default=DEFAULT_OBSERVER_NOISE_PROFILE,
                    choices=['optimistic', 'realistic', 'conservative'])
args = parser.parse_args()


env, assets, map_gdfs = get_env_assets(args=args, scenario='estimator_tuning_two_ships')

# This is an evaluation script, not training. Keep the factory-defined routes,
# otherwise env.reset() may resample the same route for both ships.
env.set_random_route_flag(False)
env.set_for_training_flag(False)

own_ship = env.assets[0].ship_model
passive_ship = env.assets[1].ship_model

# Match the single-ship nominal observer configuration.
USE_OBSERVER = True
Q_SCALE = 1.0
R_SCALE = 1.0
observer_noise_cfg = get_observer_noise_config(args.observer_noise_profile)
MEAS_NOISE_STD = observer_noise_cfg['measurement_noise_std']
BIAS_NOISE_STD = observer_noise_cfg['bias_noise_std']
BASE_Q = np.diag([0.01, 0.01, 1e-4, 0.05, 0.05, 0.01])
Q_obs = BASE_Q * Q_SCALE
R_obs = np.diag(MEAS_NOISE_STD ** 2) * R_SCALE

observer_dt = args.observer_time_step if args.observer_time_step is not None else own_ship.int.dt
own_ship.observer = ShipObserverEKF(
    dt=observer_dt,
    x0=np.array([
        own_ship.north,
        own_ship.east,
        own_ship.yaw_angle,
        own_ship.forward_speed,
        own_ship.sideways_speed,
        own_ship.yaw_rate,
    ], dtype=float),
    P0=np.eye(6) * 1e-3,
    Q=Q_obs,
    R=R_obs,
)
own_ship.observer.measurement_noise_std = MEAS_NOISE_STD
own_ship.observer.bias_noise_std = BIAS_NOISE_STD
own_ship.use_observer_for_control = USE_OBSERVER

# Ship 2 must remain nominal: no observer and no observer-driven signals.
passive_ship.observer = None
passive_ship.use_observer_for_control = False

observer_steps_per_sim = int(np.round(args.time_step / observer_dt)) if observer_dt < args.time_step else 1
nominal_action = np.array(env._normalize_action((1.0, 1.0, 1.0, 1.0)), dtype=np.float32)

true_state_log = []
est_state_log = []
innovation_log = []

obs, info = env.reset()

# Reassert the passive-ship assumptions after reset.
passive_ship = env.assets[1].ship_model
passive_ship.observer = None
passive_ship.use_observer_for_control = False

step_idx = 0
while True:
    obs, reward, terminated, truncated, info = env.step(nominal_action)

    true_state = np.array([
        own_ship.north,
        own_ship.east,
        own_ship.yaw_angle,
        np.hypot(own_ship.forward_speed, own_ship.sideways_speed),
    ])
    est_state = own_ship.observer.x[:4].copy() if hasattr(own_ship.observer, 'x') else np.zeros(4)
    innovation = true_state - est_state
    true_state_log.append(true_state)
    est_state_log.append(est_state)
    innovation_log.append(innovation)

    if USE_OBSERVER and observer_steps_per_sim > 1:
        for _ in range(observer_steps_per_sim - 1):
            meas = np.array([
                own_ship.north,
                own_ship.east,
                own_ship.yaw_angle,
                np.hypot(own_ship.forward_speed, own_ship.sideways_speed),
            ])
            noisy_meas = own_ship.observer.apply_measurement_noise(meas)
            own_ship.observer.predict()
            own_ship.observer.update(noisy_meas)

    step_idx += 1
    if terminated or truncated:
        collision = any(ship_has_collision(asset.ship_model) for asset in env.assets)
        print_separation_summary(info, collision=collision)
        print(f'Simulation stopped. terminated={terminated}, truncated={truncated}, info={info}')
        break


own_ship_results_df = pd.DataFrame().from_dict(env.assets[0].ship_model.simulation_results)
target_ship_results_df = pd.DataFrame().from_dict(env.assets[1].ship_model.simulation_results)
result_dfs = [own_ship_results_df, target_ship_results_df]

map_anim = MapAnimator(
    assets=assets,
    map_gdfs=map_gdfs,
    interval_ms=500,
    status_asset_index=0,
)
map_anim.run(fps=120, show=False, repeat=False)

polar_anim = PolarAnimator(focus_asset=assets[0], interval_ms=500)
polar_anim.run(fps=120, show=False, repeat=False)

animate_side_by_side(
    map_anim.fig,
    polar_anim.fig,
    left_frac=0.68,
    height_frac=0.92,
    gap_px=16,
    show=False,
)

plot_ship_status(assets[0], own_ship_results_df, plot_env_load=True, show=False)
plot_ship_and_real_map(assets, result_dfs, map_gdfs, show=False)

observer = env.assets[0].ship_model.observer
if hasattr(observer, 'total_noise_log') and len(observer.total_noise_log) > 0:
    total_noise = np.array(observer.total_noise_log)
    white_noise = np.array(observer.white_noise_log)
    bias_noise = np.array(observer.bias_log)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(total_noise[:, 0], label='Total north', color='tab:blue', alpha=0.5)
    axs[0].plot(white_noise[:, 0], label='White north', color='tab:cyan', linestyle='dashed', alpha=0.7)
    axs[0].plot(total_noise[:, 1], label='Total east', color='tab:orange', alpha=0.5)
    axs[0].plot(white_noise[:, 1], label='White east', color='tab:olive', linestyle='dashed', alpha=0.7)
    axs[0].plot(bias_noise[:, 0], label='Bias north (slow)', color='red', linewidth=2.5, zorder=10)
    axs[0].plot(bias_noise[:, 1], label='Bias east (slow)', color='darkred', linewidth=2.5, zorder=10)
    axs[0].set_ylabel('Position noise [m]')
    axs[0].set_title('Ship 1 measurement noise components: position')
    axs[0].legend()

    axs[1].plot(total_noise[:, 3], label='Total speed', color='tab:green', alpha=0.5)
    axs[1].plot(white_noise[:, 3], label='White speed', color='tab:olive', linestyle='dashed', alpha=0.7)
    axs[1].plot(bias_noise[:, 3], label='Bias speed (slow)', color='red', linewidth=2.5, zorder=10)
    axs[1].set_ylabel('Speed noise [m/s]')
    axs[1].set_title('Ship 1 measurement noise components: speed')
    axs[1].set_xlabel('Timestep')
    axs[1].legend()

true_state_log = np.array(true_state_log)
est_state_log = np.array(est_state_log)
innovation_log = np.array(innovation_log)
labels = ['North [m]', 'East [m]', 'Yaw [rad]', 'Speed [m/s]']

fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
for idx in range(4):
    axs[idx].plot(true_state_log[:, idx], label='True', color='tab:blue', alpha=0.7)
    axs[idx].plot(est_state_log[:, idx], label='Estimated', color='tab:orange', alpha=0.7)
    axs[idx].set_ylabel(labels[idx])
    axs[idx].legend()
axs[0].set_title('Ship 1 true vs. estimated states')
axs[-1].set_xlabel('Timestep')
plt.tight_layout()

fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for idx in range(4):
    axs[idx].plot(innovation_log[:, idx], label='Innovation (meas - est)', color='tab:green', alpha=0.7)
    axs[idx].set_ylabel(labels[idx])
    axs[idx].axhline(0, color='k', linestyle=':', linewidth=1)
    axs[idx].legend()
axs[0].set_title('Ship 1 innovation (measurement - estimate)')
axs[-1].set_xlabel('Timestep')
plt.tight_layout()

plt.show()