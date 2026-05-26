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
    candidates = sorted(trained_model_root.glob(f'{model_prefix}*/model.zip'))
    if not candidates:
        raise FileNotFoundError(
            f'No trained model found for prefix {model_prefix}. Provide --model_path explicitly.'
        )
    return str(candidates[-1])


parser = argparse.ArgumentParser(description='Single-ship measurement-noise stress test (trained agent)')
parser.add_argument('--time_step', type=int, default=5)
parser.add_argument('--engine_step_count', type=int, default=10)
parser.add_argument('--radius_of_acceptance', type=int, default=300)
parser.add_argument('--lookahead_distance', type=int, default=1000)
parser.add_argument('--nav_fail_time', type=int, default=300)
parser.add_argument('--ship_draw', type=bool, default=True)
parser.add_argument('--time_since_last_ship_drawing', type=int, default=30)
parser.add_argument('--map_gpkg_filename', type=str, default='Stangvik.gpkg')
parser.add_argument('--warm_up_time', type=int, default=2500)
parser.add_argument('--action_sampling_period', type=int, default=120)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--observer_noise_profile', type=str, default=DEFAULT_OBSERVER_NOISE_PROFILE,
                    choices=['optimistic', 'realistic', 'conservative'])
parser.add_argument('--measurement_noise_attack_mode', type=str, default=None,
                    choices=['increase_only', 'symmetric_band'])
parser.add_argument('--allow_subnominal_noise', action='store_true',
                    help='Legacy alias for measurement_noise_attack_mode=symmetric_band')
parser.add_argument('--measurement_noise_penalty_deadband', type=float, nargs=4,
                    default=[0.05, 0.05, 0.05, 0.08])
parser.add_argument('--stochastic_policy', action='store_true',
                    help='Sample actions from the SAC policy instead of using the deterministic mean action during evaluation')
args = parser.parse_args()


env, assets, map_gdfs = get_env_assets(args=args, scenario='measurement_noise')

env.set_random_route_flag(False)
env.set_for_training_flag(False)

model_load_path = resolve_model_path(args.model_path, 'AST-observer-noise-one-ship-train')
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
result_dfs = [own_ship_results_df]

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
    plt.title('Observer noise scaling over time (RL agent)')
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
    axs[0].set_title('Measurement noise components: position')
    axs[0].legend()

    axs[1].plot(total_noise[:, 2], label='Total yaw', color='tab:purple', alpha=0.5)
    axs[1].plot(white_noise[:, 2], label='White yaw', color='plum', linestyle='dashed', alpha=0.7)
    axs[1].plot(total_noise[:, 3], label='Total speed', color='tab:green', alpha=0.5)
    axs[1].plot(white_noise[:, 3], label='White speed', color='tab:olive', linestyle='dashed', alpha=0.7)
    axs[1].plot(bias_noise[:, 2], label='Bias yaw (slow)', color='red', linewidth=2.5)
    axs[1].plot(bias_noise[:, 3], label='Bias speed (slow)', color='darkred', linewidth=2.5)
    axs[1].set_ylabel('Yaw / speed noise')
    axs[1].set_xlabel('Timestep')
    axs[1].set_title('Measurement noise components: yaw and speed')
    axs[1].legend()

plt.show()