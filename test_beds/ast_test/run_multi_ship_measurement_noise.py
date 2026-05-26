from pathlib import Path
import sys
import os
from copy import deepcopy
from itertools import combinations
from types import MethodType

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from test_beds.ast_test.setup import get_env_assets, DEFAULT_OBSERVER_NOISE_PROFILE
from env_wrappers.sea_env_ast_v2.env import SeaEnvASTv2, encounter_metrics, collision_flag
from env_wrappers.sea_env_ast_v2.measurement_noise_env import SeaEnvMeasurementNoiseAST
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.get_path import get_ship_route_path, get_ship_route_path_for_training, get_ship_route_path_for_validation
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map


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


def _select_critical_encounter(self):
    if len(self.assets) <= 1:
        return None

    own_ship = self.assets[0].ship_model
    nearest_contact = None
    best_contact = None
    nearest_distance = np.inf
    best_reward = -np.inf

    for asset_index, asset in enumerate(self.assets[1:], start=1):
        encounter = encounter_metrics(own_ship, asset.ship_model)
        distance = float(encounter[0])
        dcpa = float(encounter[2])
        tcpa = float(encounter[3])
        encounter_reward = self._encounter_reward(distance, dcpa, tcpa)

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_contact = (asset_index, asset, encounter, encounter_reward)

        if encounter_reward > best_reward:
            best_reward = encounter_reward
            best_contact = (asset_index, asset, encounter, encounter_reward)

    if best_contact is not None and best_reward > 0.0:
        return best_contact
    return nearest_contact


def _get_obs_multi(self, normalized=True):
    observation = self._base_get_obs(normalized=normalized)
    if not self.include_encounter_observation:
        return observation

    critical_contact = self._select_critical_encounter()
    if critical_contact is None:
        return observation

    encounter = critical_contact[2]
    if normalized:
        encounter_norm = self._normalize(encounter, self.encounter_range["min"], self.encounter_range["max"])
        observation["encounter"] = self._safe_clip(encounter_norm)
    else:
        observation["encounter"] = encounter.astype(np.float32)
    return observation


def _get_info_multi(self):
    critical_contact = self._select_critical_encounter()
    if critical_contact is None:
        return {}

    asset_index, asset, encounter, encounter_reward = critical_contact
    return {
        'encounter_range': float(encounter[0]),
        'relative_bearing': float(encounter[1]),
        'dcpa': float(encounter[2]),
        'tcpa': float(encounter[3]),
        'encounter_reward': float(encounter_reward),
        'critical_target_index': asset_index,
        'critical_target_name': asset.ship_model.name_tag,
    }


def _reset_multi(self, seed=None, options=None, route_idx=None):
    if getattr(self, 'observer_disabled_mode', False):
        self.obs_tuning_prev = self.obs_tuning_nominal.copy()
        self.cumulative_noise_attack_load = 0.0
        self.cumulative_noise_band_violation = 0.0
        self.last_noise_attack_load = 0.0
        self.last_noise_nominal_drift = 0.0
        self.last_noise_band_violation = 0.0
        self.cumulative_noise_exposure = 0.0
        self.last_noise_average_deviation = 0.0
        self.last_noise_exposure = 0.0
        return SeaEnvASTv2.reset(self, seed=seed, options=options, route_idx=route_idx)

    return SeaEnvMeasurementNoiseAST.reset(self, seed=seed, options=options, route_idx=route_idx)


def _step_without_observer(self, action_norm):
    self.action_time_list.append(self.assets[0].ship_model.int.time)

    nominal_action = np.array(self.obs_tuning_nominal, dtype=np.float32)
    self.obs_tuning_prev = nominal_action.copy()
    self.last_noise_attack_load = 0.0
    self.last_noise_nominal_drift = 0.0
    self.last_noise_band_violation = 0.0
    self.last_noise_average_deviation = 0.0
    self.last_noise_exposure = 0.0

    action_sampling_period = getattr(self.args, 'action_sampling_period', 10)
    running_time = 0.0
    while running_time < action_sampling_period:
        self._step()

        running_time += self.assets[0].ship_model.int.dt

        if self.assets[0].ship_model.stop:
            self.terminated = True
            break

        if np.all(self.ship_stop_status):
            self.terminated = True
            break

        if self.assets[0].ship_model.int.time > self.assets[0].ship_model.simulation_config.simulation_time:
            self.truncated = True
            break

    observation = self._get_obs()
    reward = SeaEnvMeasurementNoiseAST.reward_function(self, nominal_action)
    terminated = self.terminated
    truncated = self.truncated
    info = {}

    self.obs_list.append(observation)
    self.action_list.append(nominal_action.copy())
    self.reward_list.append(reward)
    self.terminated_list.append(terminated)
    self.truncated_list.append(truncated)
    self.info_list.append(info)

    return observation, reward, terminated, truncated, info


def _step_multi(self, action_norm):
    if getattr(self, 'observer_disabled_mode', False):
        obs, reward, terminated, truncated, info = _step_without_observer(self, action_norm)
    else:
        obs, reward, terminated, truncated, info = SeaEnvMeasurementNoiseAST.step(self, action_norm)
    allow_only_observer_involved_collisions(self)

    critical_contact = self._select_critical_encounter()
    if critical_contact is None:
        self.current_step += 1
        return obs, reward, terminated, truncated, info

    asset_index, asset, encounter, encounter_reward = critical_contact
    distance = float(encounter[0])
    dcpa = float(encounter[2])
    tcpa = float(encounter[3])
    reward += encounter_reward

    proximity_breach = distance < self.collision_distance
    if proximity_breach and self.terminate_on_proximity:
        terminated = True

    self.current_step += 1
    if self.max_steps is not None and self.current_step >= self.max_steps:
        terminated = True

    observer_collision, observer_colliders = observer_collision_summary(self)
    if observer_collision and self.collision_reward > 0.0:
        reward += self.collision_reward

    observer_ship = self.assets[0].ship_model
    if observer_ship.stop:
        terminated = True

    info['distance'] = distance
    info['relative_bearing'] = float(encounter[1])
    info['dcpa'] = dcpa
    info['tcpa'] = tcpa
    info['encounter_reward'] = encounter_reward
    info['closeness_reward'] = encounter_reward
    info['proximity_breach'] = proximity_breach
    info['collision'] = observer_collision
    info['observer_collision_partners'] = observer_colliders
    info['critical_target_index'] = asset_index
    info['critical_target_name'] = asset.ship_model.name_tag
    return obs, reward, terminated, truncated, info


def install_multi_ship_logic(env):
    env._base_get_obs = env._get_obs
    env._select_critical_encounter = MethodType(_select_critical_encounter, env)
    env._get_obs = MethodType(_get_obs_multi, env)
    env._get_info = MethodType(_get_info_multi, env)
    env.reset = MethodType(_reset_multi, env)
    env.step = MethodType(_step_multi, env)
    return env


def point_on_route(route_points, distance_along_route):
    route_points = np.asarray(route_points, dtype=float)
    segments = np.diff(route_points, axis=0)
    segment_lengths = np.sqrt((segments**2).sum(axis=1))
    cumulative_distance = np.concatenate([[0.0], np.cumsum(segment_lengths)])

    route_distance = float(np.clip(distance_along_route, 0.0, cumulative_distance[-1]))
    segment_index = int(np.searchsorted(cumulative_distance, route_distance, side='right') - 1)
    segment_index = min(max(segment_index, 0), len(segments) - 1)
    distance_into_segment = route_distance - cumulative_distance[segment_index]
    segment_length = max(float(segment_lengths[segment_index]), 1e-9)
    blend = distance_into_segment / segment_length
    point = route_points[segment_index] + blend * segments[segment_index]
    tangent = segments[segment_index] / segment_length
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    return point, tangent, normal


def build_crossing_route(own_route_points, crossing_distance, start_offset, end_offset, side_sign):
    center_point, tangent, normal = point_on_route(own_route_points, crossing_distance)
    start_point = center_point + side_sign * normal * start_offset
    exit_point = center_point - side_sign * normal * end_offset

    route_points = np.vstack([
        start_point,
        center_point,
        exit_point,
    ])
    return route_points, center_point, tangent


def load_route_points_from_path(route_path, reverse=False, waypoint_offset=0):
    route_file_points = np.loadtxt(str(route_path))
    if reverse:
        route_file_points = route_file_points[::-1].copy()
    route_file_points = route_file_points[int(waypoint_offset):, :]
    if route_file_points.shape[0] < 2:
        raise ValueError(
            f"Route '{route_path}' does not have enough waypoints after offset {waypoint_offset}."
        )
    return route_file_points[:, [1, 0]].copy()


def load_route_points(route_filename, reverse=False, waypoint_offset=0):
    route_path = get_ship_route_path(ROOT, route_filename)
    return load_route_points_from_path(route_path, reverse=reverse, waypoint_offset=waypoint_offset)


def select_distinct_route_paths(route_source, route_count, explicit_files=None):
    route_source = str(route_source).strip().lower()
    if route_source not in {'training', 'validation'}:
        raise ValueError("route_source must be either 'training' or 'validation'.")

    if route_source == 'training':
        base_dir = get_ship_route_path_for_training(ROOT)
        route_paths = get_ship_route_path_for_training(ROOT, '*', pattern='*.txt')
    else:
        base_dir = get_ship_route_path_for_validation(ROOT)
        route_paths = get_ship_route_path_for_validation(ROOT, '*', pattern='*.txt')

    if explicit_files:
        route_paths = [Path(base_dir) / route_name for route_name in explicit_files]
    else:
        route_paths = choose_spread_out_route_paths(route_paths, route_count)

    route_paths = [Path(route_path) for route_path in route_paths]
    if len(route_paths) < route_count:
        raise ValueError(
            f'Need at least {route_count} route files for distinct multi-ship mode, but only found {len(route_paths)}.'
        )

    for route_path in route_paths[:route_count]:
        if not route_path.exists():
            raise FileNotFoundError(f'Distinct multi-ship route file not found: {route_path}')

    return route_paths[:route_count]


def sample_route_polyline(route_points, samples_per_segment=25):
    route_points = np.asarray(route_points, dtype=float)
    if route_points.shape[0] < 2:
        return route_points.copy()

    sampled_points = []
    for start_point, end_point in zip(route_points[:-1], route_points[1:]):
        for fraction in np.linspace(0.0, 1.0, int(samples_per_segment), endpoint=False):
            sampled_points.append(start_point + fraction * (end_point - start_point))
    sampled_points.append(route_points[-1])
    return np.asarray(sampled_points, dtype=float)


def min_route_separation(route_points_a, route_points_b):
    sampled_a = sample_route_polyline(route_points_a)
    sampled_b = sample_route_polyline(route_points_b)
    deltas = sampled_a[:, None, :] - sampled_b[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    return float(np.min(distances))


def choose_spread_out_route_paths(route_paths, route_count):
    route_paths = [Path(route_path) for route_path in route_paths]
    if len(route_paths) <= route_count:
        return route_paths

    route_points_cache = {
        route_path: load_route_points_from_path(route_path, reverse=False, waypoint_offset=0)
        for route_path in route_paths
    }

    best_combo = None
    best_min_distance = -np.inf
    best_mean_distance = -np.inf
    for combo in combinations(route_paths, route_count):
        pair_distances = [
            min_route_separation(route_points_cache[path_a], route_points_cache[path_b])
            for path_a, path_b in combinations(combo, 2)
        ]
        min_distance = float(min(pair_distances))
        mean_distance = float(np.mean(pair_distances))
        if (
            min_distance > best_min_distance + 1e-6
            or (abs(min_distance - best_min_distance) <= 1e-6 and mean_distance > best_mean_distance)
        ):
            best_combo = combo
            best_min_distance = min_distance
            best_mean_distance = mean_distance

    if best_combo is None:
        raise ValueError('Could not choose a spread-out distinct-route combination.')

    return list(best_combo)


def make_shifted_route(route_points, east_offsets, north_offsets):
    route_points = np.asarray(route_points, dtype=float)
    east_offsets = np.asarray(east_offsets, dtype=float)
    north_offsets = np.asarray(north_offsets, dtype=float)
    if len(east_offsets) != len(route_points) or len(north_offsets) != len(route_points):
        raise ValueError('Offset arrays must match the number of route waypoints.')

    shifted = route_points.copy()
    shifted[:, 0] += east_offsets
    shifted[:, 1] += north_offsets
    return shifted


def make_constant_offset_route(route_points, east_offset, north_offset):
    route_points = np.asarray(route_points, dtype=float)
    return make_shifted_route(
        route_points,
        east_offsets=np.full(len(route_points), float(east_offset)),
        north_offsets=np.full(len(route_points), float(north_offset)),
    )


def make_parallel_offset_route(route_points, lateral_offset_m):
    route_points = np.asarray(route_points, dtype=float)
    if lateral_offset_m == 0.0 or route_points.shape[0] < 2:
        return route_points.copy()

    overall_tangent = route_points[-1] - route_points[0]
    overall_tangent_norm = float(np.linalg.norm(overall_tangent))

    if overall_tangent_norm < 1e-6:
        segments = np.diff(route_points, axis=0)
        segment_norms = np.linalg.norm(segments, axis=1)
        valid_idx = np.where(segment_norms > 1e-6)[0]
        if len(valid_idx) == 0:
            return route_points.copy()
        overall_tangent = segments[valid_idx[0]]
        overall_tangent_norm = float(segment_norms[valid_idx[0]])

    unit_tangent = overall_tangent / overall_tangent_norm
    unit_normal = np.array([-unit_tangent[1], unit_tangent[0]], dtype=float)
    return route_points + float(lateral_offset_m) * unit_normal


def route_on_land(route_points, map_obj):
    route_points = np.asarray(route_points, dtype=float)
    return map_obj.if_route_inside_obstacles(n_route=route_points[:, 0], e_route=route_points[:, 1])


def build_centered_passive_lane_family(base_route_points, map_obj, lane_spacing_m, passive_lane_count):
    base_route_points = np.asarray(base_route_points, dtype=float)
    passive_lane_count = int(passive_lane_count)
    if passive_lane_count < 1:
        raise ValueError('passive_lane_count must be at least 1.')
    if passive_lane_count % 2 != 0:
        raise ValueError('passive_lane_count must be even so the observer ship can remain centered.')

    half_span = passive_lane_count // 2
    lane_offsets = [
        lane_index * float(lane_spacing_m)
        for lane_index in range(-half_span, half_span + 1)
        if lane_index != 0
    ]

    lane_routes = []
    for lane_offset in lane_offsets:
        lane_route = make_parallel_offset_route(base_route_points, lateral_offset_m=lane_offset)
        if route_on_land(lane_route, map_obj):
            raise ValueError(
                'Could not build a centered multi-ship lane family without entering land. '
                f'Failed at offset {lane_offset:.1f} m with spacing {lane_spacing_m} m.'
            )
        lane_routes.append(lane_route)

    return lane_routes


def allow_only_observer_involved_collisions(env):
    observer_name = env.assets[0].ship_model.name_tag
    for asset_index, asset in enumerate(env.assets[1:], start=1):
        ship = asset.ship_model
        collision_info = ship.stop_info.get('collision', False)
        if not collision_flag(collision_info):
            continue

        colliders = []
        if isinstance(collision_info, (list, tuple)) and len(collision_info) > 1 and collision_info[1] is not None:
            colliders = list(collision_info[1])

        if observer_name in colliders:
            continue

        ship.stop_info['collision'] = [False, None]
        if hasattr(ship, 'collision_array') and ship.collision_array:
            ship.collision_array[-1] = False

        other_stop = any([
            bool(ship.stop_info.get('grounding_failure', False)),
            bool(ship.stop_info.get('navigation_failure', False)),
            bool(ship.stop_info.get('reaches_endpoint', False)),
            bool(ship.stop_info.get('outside_horizon', False)),
            bool(ship.stop_info.get('power_overload', False)),
        ])
        ship.stop = other_stop
        env.ship_stop_status[asset_index] = other_stop


def configure_ship_route(asset, route_points, name_tag=None, desired_speed=None, colav_mode=None):
    route_points = np.asarray(route_points, dtype=float)
    if route_points.shape[0] < 2:
        raise ValueError('Ship route must contain at least two waypoints.')

    route_for_navigation = route_points[:, [1, 0]].copy()

    start_plot_east = float(route_points[0, 0])
    start_plot_north = float(route_points[0, 1])
    delta_plot_east = float(route_points[1, 0] - route_points[0, 0])
    delta_plot_north = float(route_points[1, 1] - route_points[0, 1])
    initial_yaw_rad = float(np.arctan2(delta_plot_east, delta_plot_north))

    ship = asset.ship_model
    ship.simulation_config = ship.simulation_config._replace(
        initial_north_position_m=start_plot_east,
        initial_east_position_m=start_plot_north,
        initial_yaw_angle_rad=initial_yaw_rad,
        initial_forward_speed_m_per_s=0.0,
        initial_sideways_speed_m_per_s=0.0,
        initial_yaw_rate_rad_per_s=0.0,
    )
    if name_tag is not None:
        ship.name_tag = name_tag
    if desired_speed is not None:
        ship.desired_speed = desired_speed
    if colav_mode is not None:
        ship.colav_mode = colav_mode
    ship.north = start_plot_east
    ship.east = start_plot_north
    ship.yaw_angle = initial_yaw_rad
    ship.forward_speed = 0.0
    ship.sideways_speed = 0.0
    ship.yaw_rate = 0.0
    if hasattr(ship, 'auto_pilot') and hasattr(ship.auto_pilot, 'navigate'):
        ship.auto_pilot.navigate.route = route_for_navigation
        ship.auto_pilot.navigate.load_waypoints(route_for_navigation)
        ship.auto_pilot.navigate.record_initial_parameters()

    ship._initial_parameters['north'] = deepcopy(ship.north)
    ship._initial_parameters['east'] = deepcopy(ship.east)
    ship._initial_parameters['yaw_angle'] = deepcopy(ship.yaw_angle)
    ship._initial_parameters['forward_speed'] = 0.0
    ship._initial_parameters['sideways_speed'] = 0.0
    ship._initial_parameters['yaw_rate'] = 0.0
    ship._initial_parameters['speed'] = 0.0
    ship._initial_parameters['d_north'] = 0.0
    ship._initial_parameters['d_east'] = 0.0
    ship._initial_parameters['d_yaw'] = 0.0
    ship._initial_parameters['d_forward_speed'] = 0.0
    ship._initial_parameters['d_sideways_speed'] = 0.0
    ship._initial_parameters['d_yaw_rate'] = 0.0
    ship._initial_parameters['ship_drawings'] = [[], []]
    ship._initial_parameters['stop'] = False

    if ship.observer is not None:
        observer_state = np.array([
            ship.north,
            ship.east,
            ship.yaw_angle,
            ship.forward_speed,
            ship.sideways_speed,
            ship.yaw_rate,
        ], dtype=float)
        ship.observer.reset(x0=observer_state)

    asset.info.current_north = start_plot_east
    asset.info.current_east = start_plot_north
    asset.info.current_yaw_angle = initial_yaw_rad
    asset.info.forward_speed = 0.0
    asset.info.sideways_speed = 0.0
    if name_tag is not None:
        asset.info.name_tag = name_tag
    asset.init_copy = deepcopy(asset)
    return asset


def configure_passive_ship(asset, route_points, name_tag, desired_speed):
    return configure_ship_route(
        asset,
        route_points,
        name_tag=name_tag,
        desired_speed=desired_speed,
        colav_mode=None,
    )


def match_observer_to_passive_controls(observer_asset, passive_template_asset):
    observer_ship = observer_asset.ship_model
    passive_ship = passive_template_asset.ship_model

    observer_ship.throttle_controller = deepcopy(passive_ship.throttle_controller)
    observer_ship.auto_pilot = deepcopy(passive_ship.auto_pilot)
    observer_ship.desired_speed = float(passive_ship.desired_speed)
    observer_ship.colav_mode = passive_ship.colav_mode
    observer_ship.cross_track_error_tolerance = passive_ship.cross_track_error_tolerance
    observer_ship.nav_fail_time = passive_ship.nav_fail_time
    observer_ship._nav_fail_time = observer_ship.nav_fail_time if observer_ship.auto_pilot is not None else np.inf

    observer_asset.info.forward_speed = observer_ship.forward_speed
    observer_asset.info.sideways_speed = observer_ship.sideways_speed
    return observer_asset


def get_asset_route_waypoints(asset):
    if asset.ship_model.auto_pilot is None or not hasattr(asset.ship_model.auto_pilot, 'navigate'):
        return None

    waypoint_north = np.asarray(asset.ship_model.auto_pilot.navigate.north, dtype=float)
    waypoint_east = np.asarray(asset.ship_model.auto_pilot.navigate.east, dtype=float)
    if len(waypoint_north) == 0 or len(waypoint_east) == 0:
        return None
    return np.column_stack((waypoint_north, waypoint_east))


def print_waypoint_report(assets):
    route_waypoints = []
    for asset in assets:
        waypoints = get_asset_route_waypoints(asset)
        if waypoints is None:
            continue
        route_waypoints.append((
            asset.info.name_tag,
            waypoints,
            getattr(asset, 'route_label', None),
            bool(getattr(asset, 'route_reversed', False)),
        ))

    if not route_waypoints:
        print('No route waypoints available for waypoint report.')
        return

    print('\nExplicit route waypoints:')
    for ship_name, waypoints, route_label, route_reversed in route_waypoints:
        route_meta = ''
        if route_label is not None:
            direction_label = 'reversed' if route_reversed else 'forward'
            route_meta = f' [{route_label}, {direction_label}]'
        print(f'  {ship_name}{route_meta}:')
        for waypoint_index, (north_pos, east_pos) in enumerate(waypoints):
            print(f'    WP {waypoint_index:02d}: east={east_pos:.1f}, north={north_pos:.1f}')


def plot_multi_ship_map(assets, result_dfs, map_gdfs, focus_points=None, show=False):
    fig, ax = plt.subplots(figsize=(12, 8))

    if map_gdfs is not None:
        frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf = map_gdfs
        if not ocean_gdf.empty:
            ocean_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="none", zorder=0)
        if not land_gdf.empty:
            land_gdf.plot(ax=ax, facecolor="#e6e6e6", edgecolor="#b0b0b0", linewidth=0.4, zorder=1)
        if not water_gdf.empty:
            water_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="#5c8fc4", linewidth=0.5, alpha=0.98, zorder=2)
        if not coast_gdf.empty:
            coast_gdf.plot(ax=ax, color="#2f7f3f", linewidth=1.2, zorder=3)

        minx, miny, maxx, maxy = frame_gdf.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        dx, dy = (maxx - minx), (maxy - miny)
        if dx > 0:
            fig_h = 12 * (dy / dx)
            fig.set_size_inches(12, fig_h, forward=True)

    ship_styles = [
        {'track': '#1f5aa6', 'halo': 'white', 'route': '#6ea8fe'},
        {'track': '#d62828', 'halo': None, 'route': '#f28482'},
        {'track': '#2b9348', 'halo': None, 'route': '#95d5b2'},
        {'track': '#7b2cbf', 'halo': None, 'route': '#c8b6ff'},
    ]

    for index, asset in enumerate(assets):
        route_waypoints = get_asset_route_waypoints(asset)
        if route_waypoints is None or len(route_waypoints) == 0:
            continue

        style = ship_styles[index % len(ship_styles)]
        route_color = style['route']
        ax.plot(
            route_waypoints[:, 0],
            route_waypoints[:, 1],
            color=route_color,
            lw=2.0,
            linestyle='--',
            alpha=0.95,
            label=f'{asset.info.name_tag} planned path',
            zorder=6 + max(0, len(assets) - index),
        )
        ax.scatter(
            route_waypoints[0, 0],
            route_waypoints[0, 1],
            color=route_color,
            s=35,
            marker='o',
            zorder=7 + max(0, len(assets) - index),
        )
        ax.scatter(
            route_waypoints[-1, 0],
            route_waypoints[-1, 1],
            color=route_color,
            s=40,
            marker='X',
            zorder=7 + max(0, len(assets) - index),
        )

    for index, (asset, result_df) in enumerate(zip(assets, result_dfs)):
        if 'east position [m]' not in result_df.columns or 'north position [m]' not in result_df.columns:
            continue

        east_vals = result_df['east position [m]'].to_numpy()
        north_vals = result_df['north position [m]'].to_numpy()
        if len(east_vals) == 0:
            continue

        style = ship_styles[index % len(ship_styles)]
        track_color = style['track']
        halo_color = style['halo']
        track_width = 2.8 if index == 0 else 2.1

        if halo_color is not None:
            ax.plot(
                north_vals,
                east_vals,
                color=halo_color,
                lw=track_width + 2.4,
                alpha=1.0,
                zorder=9 + max(0, len(assets) - index),
            )
        ax.plot(
            north_vals,
            east_vals,
            color=track_color,
            lw=track_width,
            label=f'{asset.info.name_tag} actual track',
            zorder=10 + max(0, len(assets) - index),
        )

    if map_gdfs is None and focus_points is not None and len(focus_points) > 0:
        focus_points = np.asarray(focus_points, dtype=float)
        east_min = float(np.min(focus_points[:, 0]))
        east_max = float(np.max(focus_points[:, 0]))
        north_min = float(np.min(focus_points[:, 1]))
        north_max = float(np.max(focus_points[:, 1]))
        east_pad = max(1200.0, 0.15 * (east_max - east_min + 1.0))
        north_pad = max(1200.0, 0.15 * (north_max - north_min + 1.0))
        ax.set_xlim(east_min - east_pad, east_max + east_pad)
        ax.set_ylim(north_min - north_pad, north_max + north_pad)

    ax.set_title('Four-ship trajectories and planned paths')
    ax.set_xlabel('East position (m)')
    ax.set_ylabel('North position (m)')
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', frameon=True, ncol=2)

    if show:
        plt.show()

    return fig, ax


parser = argparse.ArgumentParser(description='Multi-ship measurement-noise stress test using an existing two-ship policy')
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
parser.add_argument('--observer_noise_profile', type=str, default=DEFAULT_OBSERVER_NOISE_PROFILE,
                    choices=['optimistic', 'realistic', 'conservative'])
parser.add_argument('--measurement_noise_attack_mode', type=str, default=None,
                    choices=['increase_only', 'symmetric_band'])
parser.add_argument('--allow_subnominal_noise', action='store_true',
                    help='Legacy alias for measurement_noise_attack_mode=symmetric_band')
parser.add_argument('--measurement_noise_penalty_deadband', type=float, nargs=4,
                    default=[0.05, 0.05, 0.05, 0.08])
parser.add_argument('--close_reward_distance', type=float, default=120.0)
parser.add_argument('--close_reward_gain', type=float, default=0.0)
parser.add_argument('--collision_reward', type=float, default=120.0)
parser.add_argument('--dcpa_reward_gain', type=float, default=30.0)
parser.add_argument('--dcpa_reward_distance', type=float, default=120.0)
parser.add_argument('--tcpa_reward_horizon', type=float, default=900.0)
parser.add_argument('--tcpa_window_center', type=float, default=240.0)
parser.add_argument('--tcpa_window_width', type=float, default=180.0)
parser.add_argument('--stochastic_policy', action='store_true',
                    help='Sample actions from the SAC policy instead of using the deterministic mean action during evaluation')
parser.add_argument('--lane_spacing_m', type=float, default=200.0,
                    help='Legacy argument kept for compatibility; unused in the distinct four-ship scenario')
parser.add_argument('--route_layout', type=str, default='distinct_opposing',
                    choices=['distinct_opposing'],
                    help='Use four distinct Stangvik route files: observer forward, three passive ships reversed')
parser.add_argument('--distinct_route_source', type=str, default='training',
                    choices=['training', 'validation'],
                    help='Route file set to use when route_layout=distinct_opposing')
parser.add_argument('--distinct_route_files', type=str, nargs='*', default=None,
                    help='Optional explicit route file names for distinct_opposing mode; first is observer, next three are passive ships')
parser.add_argument('--disable_agent', action='store_true',
                    help='Use nominal observer noise scaling instead of the trained RL policy')
parser.add_argument('--disable_observer', action='store_true',
                    help='Use true-state control for the observer ship instead of observer-based control')
parser.add_argument('--log_steps', action='store_true',
                    help='Print one line per RL step during the simulation loop')
parser.add_argument('--full_diagnostics', action='store_true',
                    help='Generate all animations and extra diagnostic plots after the simulation finishes')
args = parser.parse_args()


env, assets, map_gdfs = get_env_assets(args=args, scenario='measurement_noise_two_ships')
env = install_multi_ship_logic(env)
base_passive_asset = assets[1]

if args.disable_observer and args.disable_agent:
    match_observer_to_passive_controls(env.assets[0], base_passive_asset)
    print('Observer ship matched to passive nominal controls because both agent and observer are disabled.')

if args.disable_observer:
    env.observer_disabled_mode = True
    env.assets[0].ship_model.observer = None
    env.assets[0].ship_model.use_observer_for_control = False
    print('Observer disabled. Removing observer object and using true state directly for own-ship control.')

selected_route_paths = select_distinct_route_paths(
    route_source=args.distinct_route_source,
    route_count=4,
    explicit_files=args.distinct_route_files,
)

observer_route = load_route_points_from_path(selected_route_paths[0], reverse=False, waypoint_offset=0)
configure_ship_route(
    env.assets[0],
    observer_route,
    name_tag='Observer Ship',
    desired_speed=env.assets[0].ship_model.desired_speed,
    colav_mode=env.assets[0].ship_model.colav_mode,
)
env.assets[0].route_label = selected_route_paths[0].name
env.assets[0].route_reversed = False

passive_specs = [
    {
        'name': f'Passive Ship {index}',
        'route_points': load_route_points_from_path(route_path, reverse=True, waypoint_offset=0),
        'speed': 4.0,
        'route_label': route_path.name,
        'route_reversed': True,
    }
    for index, route_path in enumerate(selected_route_paths[1:], start=1)
]

passive_assets = [base_passive_asset] + [deepcopy(base_passive_asset) for _ in range(len(passive_specs) - 1)]
configured_passive_assets = [
    configure_passive_ship(
        asset,
        spec['route_points'],
        spec['name'],
        spec['speed'],
    )
    for asset, spec in zip(passive_assets, passive_specs)
]

for asset, spec in zip(configured_passive_assets, passive_specs):
    asset.route_label = spec['route_label']
    asset.route_reversed = spec['route_reversed']

env.assets = [assets[0], *configured_passive_assets]
assets = env.assets
env.ship_stop_status = [False] * len(env.assets)
env.include_encounter_observation = len(env.assets) > 1

print_waypoint_report(assets)

env.set_random_route_flag(False)
env.set_for_training_flag(False)

model_load_path = resolve_model_path(args.model_path, 'AST-observer-noise-two-ships-train')
model = None
nominal_action = np.array(env._normalize_action((1.0, 1.0, 1.0, 1.0)), dtype=np.float32)
if args.disable_observer:
    print('Observer is disabled. Measurement-noise actions are ignored in this mode.')
elif args.disable_agent:
    print('Agent disabled. Using nominal observer noise scaling [1.0, 1.0, 1.0, 1.0].')
else:
    print(f'Loading two-ship model from: {model_load_path}')
    model = SAC.load(model_load_path)


def collision_participants(ship):
    collision_info = ship.stop_info.get('collision', False)
    if not collision_flag(collision_info):
        return []
    if isinstance(collision_info, (list, tuple)) and len(collision_info) > 1 and collision_info[1] is not None:
        return list(collision_info[1])
    return []


def observer_collision_summary(env):
    observer_ship = env.assets[0].ship_model
    observer_name = env.assets[0].info.name_tag
    observer_colliders = collision_participants(observer_ship)
    if observer_colliders:
        return True, observer_colliders

    inferred_colliders = []
    for asset in env.assets[1:]:
        passive_colliders = collision_participants(asset.ship_model)
        if observer_name in passive_colliders:
            inferred_colliders.append(asset.info.name_tag)

    return bool(inferred_colliders), inferred_colliders


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

noise_history = []
noise_times = []
initial_noise = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

obs, info = env.reset()
step_idx = 0
simulation_time_limit = float(env.assets[0].ship_model.simulation_config.simulation_time)
progress_print_every_pct = 1
last_reported_progress_pct = -progress_print_every_pct
print(f'Simulation progress: 0% (0.0/{simulation_time_limit:.1f}s)', flush=True)
while True:
    if args.disable_observer or model is None:
        action = nominal_action.copy()
    else:
        action, _states = model.predict(obs, deterministic=not args.stochastic_policy)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
    obs, reward, terminated, truncated, info = env.step(action)

    current_sim_time = float(env.assets[0].ship_model.int.time)
    if simulation_time_limit > 0.0:
        progress_pct = min(100, int((100.0 * current_sim_time) / simulation_time_limit))
        if progress_pct >= last_reported_progress_pct + progress_print_every_pct:
            print(
                f'Simulation progress: {progress_pct}% '
                f'({current_sim_time:.1f}/{simulation_time_limit:.1f}s)',
                flush=True,
            )
            last_reported_progress_pct = progress_pct

    if hasattr(env, 'action_list') and len(env.action_list) > 0:
        applied_noise = np.asarray(env.action_list[-1], dtype=float).reshape(-1)
        noise_history.append(applied_noise)
        if hasattr(env, 'action_time_list') and len(env.action_time_list) > 0:
            noise_times.append(float(env.action_time_list[-1]))
        else:
            noise_times.append(float(len(noise_history)))

    if args.log_steps:
        print(
            f"Step {step_idx}: critical_target={info.get('critical_target_name', 'n/a')}, "
            f"distance={info.get('distance', np.nan):.1f}, dcpa={info.get('dcpa', np.nan):.1f}, "
            f"tcpa={info.get('tcpa', np.nan):.1f}"
        )

    step_idx += 1
    if terminated or truncated:
        observer_collision, observer_colliders = observer_collision_summary(env)
        print_separation_summary(
            info,
            collision=observer_collision,
            target_name=info.get('critical_target_name'),
        )
        print(
            f'Observer ship collision: {observer_collision}'
            + (f" with {', '.join(observer_colliders)}" if observer_colliders else '')
        )

        for asset in env.assets[1:]:
            passive_colliders = collision_participants(asset.ship_model)
            if passive_colliders:
                print(f"{asset.info.name_tag} collision: True with {', '.join(passive_colliders)}")

        print(f'Simulation stopped. terminated={terminated}, truncated={truncated}, info={info}')
        break


result_dfs = [pd.DataFrame().from_dict(asset.ship_model.simulation_results) for asset in env.assets]
own_ship_results_df = result_dfs[0]

plot_multi_ship_map(assets, result_dfs, map_gdfs, show=False)

if args.full_diagnostics:
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
        plt.title('Agent-controlled observer measurement-noise scaling over time')
        plt.legend()
        plt.grid(True)

    observer = env.assets[0].ship_model.observer
    if observer is not None and hasattr(observer, 'total_noise_log') and len(observer.total_noise_log) > 0:
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

plt.show()