from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from test_beds.ast_test.setup import get_env_assets
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map

import argparse


def summarize_stop_info(ship):
    stop_info = getattr(ship, 'stop_info', {})
    flags = []
    for key, value in stop_info.items():
        if isinstance(value, (list, tuple)):
            is_active = bool(value[0]) if len(value) > 0 else False
        else:
            is_active = bool(value)
        if is_active:
            flags.append(key)
    return ', '.join(flags) if flags else 'none'


def ship_has_collision(ship):
    collision_info = getattr(ship, 'stop_info', {}).get('collision', False)
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


parser = argparse.ArgumentParser(description='Two passive ships on opposite ends of the same route')
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
parser.add_argument('--fixed_route', action='store_true',
                    help='Use the static setup route instead of sampling a new opposite route pair each run')
parser.add_argument('--zero_disturbance', action='store_true',
                    help='Disable wave, wind, and current for a strict zero-disturbance passive baseline')
parser.add_argument('--parallel_offset_m', type=float, default=300.0,
                    help='Lateral offset applied to the passive ship route while keeping reverse-direction pairing')
args = parser.parse_args()


env, assets, map_gdfs = get_env_assets(args=args, scenario='nominal_two_ships')

env.set_random_route_flag(not args.fixed_route)
env.set_for_training_flag(not args.fixed_route)

nominal_action = np.array(
    env._normalize_action(
        (
            env.Hs_wu,
            env.U_w_bar_wu,
            env.Tp_wu,
            env.psi_ww_bar_wu,
            env.U_c_bar_wu,
            env.psi_c_bar_wu,
        )
    ),
    dtype=np.float32,
)

obs, info = env.reset()

step_idx = 0
while True:
    obs, reward, terminated, truncated, info = env.step(nominal_action)
    step_idx += 1
    if terminated or truncated:
        collision = any(ship_has_collision(asset.ship_model) for asset in env.assets)
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

plt.show()