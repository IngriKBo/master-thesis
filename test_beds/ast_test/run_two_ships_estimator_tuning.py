from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from test_beds.ast_test.setup import get_env_assets, DEFAULT_OBSERVER_NOISE_PROFILE
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map

import argparse


def resolve_model_path(cli_model_path, model_prefix):
    if cli_model_path is not None:
        return str(Path(cli_model_path))

    trained_model_root = ROOT / 'trained_model'
    candidates = list(trained_model_root.glob(f'{model_prefix}*/model.zip'))
    if not candidates:
        raise FileNotFoundError(
            f'No trained model found for prefix {model_prefix}. Provide --model_path explicitly.'
        )
    latest_model = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest_model)


def summarize_stop_info(ship):
    stop_info = ship.stop_info
    active_flags = []
    for key in [
        'collision',
        'grounding_failure',
        'navigation_failure',
        'reaches_endpoint',
        'outside_horizon',
        'power_overload',
    ]:
        value = stop_info.get(key, False)
        if isinstance(value, (list, tuple)):
            is_active = bool(value[0])
        else:
            is_active = bool(value)
        if is_active:
            active_flags.append(f'{key}={value}')
    if not active_flags:
        return 'no stop flags set'
    return ', '.join(active_flags)


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


parser = argparse.ArgumentParser(description='Two-ship estimator tuning stress test (trained agent)')
parser.add_argument('--time_step', type=int, default=5)
parser.add_argument('--engine_step_count', type=int, default=10)
parser.add_argument('--radius_of_acceptance', type=int, default=300)
parser.add_argument('--lookahead_distance', type=int, default=1000)
parser.add_argument('--nav_fail_time', type=int, default=300)
parser.add_argument('--ship_draw', type=bool, default=True)
parser.add_argument('--time_since_last_ship_drawing', type=int, default=30)
parser.add_argument('--map_gpkg_filename', type=str, default='Stangvik.gpkg')
parser.add_argument('--warm_up_time', type=int, default=240)
parser.add_argument('--action_sampling_period', type=int, default=120)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--fixed_route', action='store_true',
                    help='Use the static setup route instead of sampling a new opposite route pair each run')
parser.add_argument('--parallel_offset_m', type=float, default=300.0,
                    help='Lateral offset applied to the passive ship route while keeping reverse-direction pairing')
parser.add_argument('--estimator_tuning_bounds_profile', type=str, default='legacy',
                    choices=['legacy', 'realistic'])
parser.add_argument('--observer_noise_profile', type=str, default=DEFAULT_OBSERVER_NOISE_PROFILE,
                    choices=['optimistic', 'realistic', 'conservative'])
args = parser.parse_args()


env, assets, map_gdfs = get_env_assets(args=args, scenario='estimator_tuning_two_ships')

env.set_random_route_flag(not args.fixed_route)
env.set_for_training_flag(not args.fixed_route)

model_load_path = resolve_model_path(args.model_path, 'AST-observer-two-ships-train')
print(f'Loading model from: {model_load_path}')

model = SAC.load(model_load_path)

observer_tuning_history = []
observer_tuning_times = []
initial_tuning = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

obs, info = env.reset()
step_idx = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    obs, reward, terminated, truncated, info = env.step(action)

    if hasattr(env, 'action_list') and len(env.action_list) > 0:
        tuning = np.asarray(env.action_list[-1], dtype=float).reshape(-1)
        observer_tuning_history.append(tuning)
        if hasattr(env, 'action_time_list') and len(env.action_time_list) > 0:
            observer_tuning_times.append(float(env.action_time_list[-1]))
        else:
            observer_tuning_times.append(float(len(observer_tuning_history)))
        print(f'Step {step_idx}: observer tuning = {tuning}')

    step_idx += 1
    if terminated or truncated:
        collision = bool(info.get('collision', False)) or any(ship_has_collision(asset.ship_model) for asset in env.assets)
        print_separation_summary(info, collision=collision)
        print(f'Simulation stopped. terminated={terminated}, truncated={truncated}, info={info}')
        for asset in env.assets:
            print(f'{asset.ship_model.name_tag} stop reason: {summarize_stop_info(asset.ship_model)}')
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

if len(observer_tuning_history) > 0:
    observer_tuning_history = np.array(observer_tuning_history)
    observer_tuning_times = np.array(observer_tuning_times)
    plt.figure(figsize=(10, 5))
    labels = ['alpha_r_pos', 'alpha_r_yaw', 'alpha_r_speed', 'alpha_q']
    for idx, label in enumerate(labels[:observer_tuning_history.shape[1]]):
        plt.step(observer_tuning_times, observer_tuning_history[:, idx], where='post', label=label)
        plt.hlines(initial_tuning[idx], observer_tuning_times[0], observer_tuning_times[-1], colors='k', linestyles='dashed', alpha=0.4)
    plt.xlabel('Simulation time [s]')
    plt.ylabel('Observer tuning (scaling)')
    plt.title('Affected ship observer tuning over time')
    plt.legend()
    plt.grid(True)

observer = env.assets[0].ship_model.observer
plt.show()