from pathlib import Path
import sys
import os
import re

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


def _checkpoint_step_key(path):
    match = re.search(r'_checkpoint_(\d+)_steps\.zip$', path.name)
    return int(match.group(1)) if match else -1


def _find_latest_model_artifact(model_dir):
    model_zip = model_dir / 'model.zip'
    if model_zip.is_file():
        return model_zip

    checkpoints = sorted(model_dir.glob('*checkpoint*_steps.zip'), key=_checkpoint_step_key)
    if checkpoints:
        return checkpoints[-1]

    zip_files = sorted(model_dir.glob('*.zip'))
    if len(zip_files) == 1:
        return zip_files[0]

    return None


def _resolve_explicit_model_path(cli_model_path):
    raw_path = Path(cli_model_path)
    candidate_paths = [raw_path] if raw_path.is_absolute() else [raw_path, ROOT / raw_path]

    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate

        if candidate.suffix.lower() == '.zip' and candidate.parent.is_dir():
            fallback = _find_latest_model_artifact(candidate.parent)
            if fallback is not None:
                print(f'Model file not found at {candidate}; using {fallback.name} from the same folder.')
                return fallback

        if candidate.is_dir():
            fallback = _find_latest_model_artifact(candidate)
            if fallback is not None:
                print(f'Using model artifact: {fallback.name}')
                return fallback

        if candidate.suffix == '':
            zipped = candidate.with_suffix('.zip')
            if zipped.is_file():
                return zipped

    raise FileNotFoundError(
        f'Could not resolve model path from {cli_model_path}. '
        'Pass either a valid .zip file or a model folder containing model.zip or checkpoint zips.'
    )


def resolve_model_path(cli_model_path, model_prefix):
    if cli_model_path is not None:
        return str(_resolve_explicit_model_path(cli_model_path))

    trained_model_root = ROOT / 'trained_model'
    candidates = sorted(trained_model_root.glob(f'{model_prefix}*/model.zip'))
    if candidates:
        return str(candidates[-1])

    checkpoint_candidates = sorted(
        trained_model_root.glob(f'{model_prefix}*/*checkpoint*_steps.zip'),
        key=_checkpoint_step_key,
    )
    if checkpoint_candidates:
        selected_checkpoint = checkpoint_candidates[-1]
        print(f'No model.zip found for prefix {model_prefix}; using checkpoint {selected_checkpoint.name}.')
        return str(selected_checkpoint)

    raise FileNotFoundError(
        f'No trained model found for prefix {model_prefix}. Provide --model_path explicitly.'
    )


def apply_runtime_noise_bounds(env, lower_bounds=None, upper_bounds=None):
    if lower_bounds is None and upper_bounds is None:
        return None

    range_keys = ['noise_pos', 'noise_yaw', 'noise_speed', 'noise_bias']
    current_lower = np.array([float(env.obs_tuning_range[key][0]) for key in range_keys], dtype=np.float32)
    current_upper = np.array([float(env.obs_tuning_range[key][1]) for key in range_keys], dtype=np.float32)

    new_lower = current_lower if lower_bounds is None else np.asarray(lower_bounds, dtype=np.float32)
    new_upper = current_upper if upper_bounds is None else np.asarray(upper_bounds, dtype=np.float32)

    if new_lower.shape != (4,) or new_upper.shape != (4,):
        raise ValueError('Runtime noise bounds must contain four values: [pos, yaw, speed, bias].')
    if np.any(new_lower > new_upper):
        raise ValueError('Each runtime lower noise bound must be less than or equal to the matching upper bound.')

    for idx, key in enumerate(range_keys):
        env.obs_tuning_range[key] = np.array([new_lower[idx], new_upper[idx]], dtype=np.float32)

    nominal = np.asarray(getattr(env, 'obs_tuning_nominal', np.ones(4, dtype=np.float32)), dtype=np.float32)
    env.obs_tuning_prev = np.clip(nominal, new_lower, new_upper).astype(np.float32)

    return {'lower': new_lower, 'upper': new_upper}


parser = argparse.ArgumentParser(description='Two-ship measurement-noise stress test (trained agent)')
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
parser.add_argument('--sim_scenario', type=str, default='extreme', choices=['normal', 'extreme'],
                    help='Simulation scenario: normal uses measurement_noise_two_ships, extreme uses measurement_noise_two_ships_extreme')
parser.add_argument('--extreme_noise_max_scale', type=float, default=20.0,
                    help='Extreme simulation only: max allowed noise scaling per channel (default: 20.0)')
parser.add_argument('--normal_noise_max_scale', type=float, default=5.0,
                    help='Normal simulation only: default max allowed noise scaling per channel when no runtime upper bounds are provided (default: 5.0)')
parser.add_argument('--extreme_action_gain', type=float, default=1.5,
                    help='Extreme simulation only: gain on normalized action before clipping (default: 1.5)')
parser.add_argument('--fixed_route', action='store_true',
                    help='Use the static setup route instead of sampling a new opposite route pair each run')
parser.add_argument('--route_source', type=str, default='training', choices=['training', 'validation'],
                    help='Route file set used when sampling routes (default: training)')
parser.add_argument('--parallel_offset_m', type=float, default=300.0,
                    help='Lateral offset applied to the passive ship route while keeping reverse-direction pairing')
parser.add_argument('--observer_noise_profile', type=str, default=DEFAULT_OBSERVER_NOISE_PROFILE,
                    choices=['optimistic', 'realistic', 'conservative'])
parser.add_argument('--measurement_noise_attack_mode', type=str, default=None,
                    choices=['increase_only', 'symmetric_band'])
parser.add_argument('--allow_subnominal_noise', action='store_true',
                    help='Legacy alias for measurement_noise_attack_mode=symmetric_band')
parser.add_argument('--measurement_noise_penalty_deadband', type=float, nargs=4,
                    default=[0.05, 0.05, 0.05, 0.08])
parser.add_argument('--runtime_noise_lower_bounds', type=float, nargs=4, default=None,
                    metavar=('POS_MIN', 'YAW_MIN', 'SPEED_MIN', 'BIAS_MIN'),
                    help='Runtime-only lower bounds for denormalizing the agent action: [pos yaw speed bias]')
parser.add_argument('--runtime_noise_upper_bounds', type=float, nargs=4, default=None,
                    metavar=('POS_MAX', 'YAW_MAX', 'SPEED_MAX', 'BIAS_MAX'),
                    help='Runtime-only upper bounds for denormalizing the agent action: [pos yaw speed bias]')
parser.add_argument('--close_reward_distance', type=float, default=120.0)
parser.add_argument('--close_reward_gain', type=float, default=0.0)
parser.add_argument('--collision_reward', type=float, default=120.0)
parser.add_argument('--dcpa_reward_gain', type=float, default=22.0)
parser.add_argument('--dcpa_reward_distance', type=float, default=120.0)
parser.add_argument('--tcpa_reward_horizon', type=float, default=900.0)
parser.add_argument('--tcpa_window_center', type=float, default=240.0)
parser.add_argument('--tcpa_window_width', type=float, default=180.0)
parser.add_argument('--stochastic_policy', action='store_true',
                    help='Sample actions from the SAC policy instead of using the deterministic mean action during evaluation')
parser.add_argument('--no_plots', action='store_true',
                    help='Skip displaying matplotlib figure windows after the simulation finishes')
args = parser.parse_args()
args.legacy_two_ship_measurement_noise_runtime = True


scenario_name = 'measurement_noise_two_ships_extreme' if args.sim_scenario == 'extreme' else 'measurement_noise_two_ships'
model_prefix = 'AST-observer-noise-two-ships-extreme-train-realistic_' if args.sim_scenario == 'extreme' else 'AST-observer-noise-two-ships-train-realistic_'

env, assets, map_gdfs = get_env_assets(args=args, scenario=scenario_name)
print(f'Simulation scenario: {args.sim_scenario} ({scenario_name})')
if args.sim_scenario == 'extreme':
    print(f'Extreme bounds      : max_scale={args.extreme_noise_max_scale}, action_gain={args.extreme_action_gain}')

runtime_upper_bounds = args.runtime_noise_upper_bounds
if args.sim_scenario == 'normal' and runtime_upper_bounds is None:
    runtime_upper_bounds = [args.normal_noise_max_scale] * 4

runtime_bounds = apply_runtime_noise_bounds(
    env,
    lower_bounds=args.runtime_noise_lower_bounds,
    upper_bounds=runtime_upper_bounds,
)
if runtime_bounds is not None:
    print(
        'Runtime noise bounds: '
        f'lower={runtime_bounds["lower"].tolist()} upper={runtime_bounds["upper"].tolist()}'
    )

env.set_random_route_flag(not args.fixed_route)
env.set_for_training_flag(args.route_source == 'training')

if args.fixed_route:
    print('Route mode: fixed default route')
else:
    print(f'Route mode: random from {args.route_source} routes')

model_load_path = resolve_model_path(
    args.model_path,
    model_prefix,
)
print(f'Loading model from: {model_load_path}')

model = SAC.load(model_load_path)

noise_history = []
noise_times = []
initial_noise = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

obs, info = env.reset()
step_idx = 0
while True:
    action, _states = model.predict(obs, deterministic=not args.stochastic_policy)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    obs, reward, terminated, truncated, info = env.step(action)

    if hasattr(env, 'action_list') and len(env.action_list) > 0:
        applied_noise = np.asarray(env.action_list[-1], dtype=float).reshape(-1)
        noise_history.append(applied_noise)
        if hasattr(env, 'action_time_list') and len(env.action_time_list) > 0:
            noise_times.append(float(env.action_time_list[-1]))
        else:
            noise_times.append(float(len(noise_history)))
        print(f'Step {step_idx}: observer noise scales = {applied_noise}')

    step_idx += 1
    if terminated or truncated:
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

if len(noise_history) > 0:
    noise_history = np.array(noise_history)
    noise_times = np.array(noise_times)
    plt.figure(figsize=(10, 5))
    labels = ['alpha_pos', 'alpha_yaw', 'alpha_speed', 'alpha_bias']
    for idx, label in enumerate(labels[:noise_history.shape[1]]):
        plt.step(noise_times, noise_history[:, idx], where='post', label=label)
        plt.hlines(initial_noise[idx], noise_times[0], noise_times[-1], colors='k', linestyles='dashed', alpha=0.4)
    plt.xlabel('Simulation time [s]')
    plt.ylabel('Noise scaling')
    plt.title('Observer noise scaling over time (two-ship RL agent)')
    plt.legend()
    plt.grid(True)

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
    axs[0].plot(bias_noise[:, 0], label='Bias north (slow)', color='red', linewidth=2.5)
    axs[0].plot(bias_noise[:, 1], label='Bias east (slow)', color='darkred', linewidth=2.5)
    axs[0].set_ylabel('Position noise [m]')
    axs[0].set_title('Affected ship observer noise components: position')
    axs[0].legend()

    axs[1].plot(total_noise[:, 2], label='Total yaw', color='tab:purple', alpha=0.5)
    axs[1].plot(white_noise[:, 2], label='White yaw', color='plum', linestyle='dashed', alpha=0.7)
    axs[1].plot(total_noise[:, 3], label='Total speed', color='tab:green', alpha=0.5)
    axs[1].plot(white_noise[:, 3], label='White speed', color='tab:olive', linestyle='dashed', alpha=0.7)
    axs[1].plot(bias_noise[:, 2], label='Bias yaw (slow)', color='red', linewidth=2.5)
    axs[1].plot(bias_noise[:, 3], label='Bias speed (slow)', color='darkred', linewidth=2.5)
    axs[1].set_ylabel('Yaw / speed noise')
    axs[1].set_xlabel('Timestep')
    axs[1].set_title('Affected ship observer noise components: yaw and speed')
    axs[1].legend()

if args.no_plots:
    plt.close('all')
else:
    plt.show()