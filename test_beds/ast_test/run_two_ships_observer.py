# run_two_ships_observer.py
# Basert på run_single_ship_map_observer.py, men med to skip:
# - Skip 1: RL-agent (som i original)
# - Skip 2: Motsatt path, ikke styrt av RL-agent

from pathlib import Path
import sys
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_wrappers.sea_env_ast_v2.observer_env import SeaEnvObserverAST
from env_wrappers.sea_env_ast_v2.env import AssetInfo, ShipAsset
from simulator.ship_in_transit.sub_systems.ship_model import ShipConfiguration, SimulationConfiguration, ShipModel
from simulator.ship_in_transit.sub_systems.ship_engine import MachinerySystemConfiguration, MachineryMode, MachineryModeParams, MachineryModes, SpecificFuelConsumptionBaudouin6M26Dot3, SpecificFuelConsumptionWartila6L26, RudderConfiguration
from simulator.ship_in_transit.sub_systems.LOS_guidance import LosParameters
from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from simulator.ship_in_transit.sub_systems.controllers import ThrottleControllerGains, HeadingControllerGains
from simulator.ship_in_transit.sub_systems.wave_model import WaveModelConfiguration
from simulator.ship_in_transit.sub_systems.current_model import CurrentModelConfiguration
from simulator.ship_in_transit.sub_systems.wind_model import WindModelConfiguration
from simulator.ship_in_transit.sub_systems.observers import ShipObserverEKF
from utils.get_path import get_ship_route_path, get_map_path
from utils.prepare_map import get_gdf_from_gpkg, get_polygon_from_gdf
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map
from stable_baselines3 import SAC
import argparse
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description='Ship in Transit Simulation (two ships, one RL, one passive)')
parser.add_argument('--time_step', type=int, default=5)
parser.add_argument('--engine_step_count', type=int, default=10)
parser.add_argument('--radius_of_acceptance', type=int, default=300)
parser.add_argument('--lookahead_distance', type=int, default=1000)
parser.add_argument('--nav_fail_time', type=int, default=300)
parser.add_argument('--ship_draw', type=bool, default=True)
parser.add_argument('--time_since_last_ship_drawing', default=30)
parser.add_argument('--warm_up_time', type=int, default=2500)
parser.add_argument('--action_sampling_period', type=int, default=900)
parser.add_argument('--model_path', type=str, default=None)
args = parser.parse_args()

GPKG_PATH   = get_map_path(ROOT, "basemap.gpkg")
FRAME_LAYER = "frame_3857"
OCEAN_LAYER = "ocean_3857"
LAND_LAYER  = "land_3857"
COAST_LAYER = "coast_3857"
WATER_LAYER = "water_3857"
frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf = get_gdf_from_gpkg(GPKG_PATH, FRAME_LAYER, OCEAN_LAYER, LAND_LAYER, COAST_LAYER, WATER_LAYER)
map_gdfs = frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf
map_data = get_polygon_from_gdf(land_gdf)
map = [PolygonObstacle(map_data), frame_gdf]

main_engine_capacity = 2160e3
diesel_gen_capacity = 510e3
hybrid_shaft_gen_as_generator = 'GEN'
hybrid_shaft_gen_as_motor = 'MOTOR'
hybrid_shaft_gen_as_offline = 'OFF'

ship_config = ShipConfiguration(
    coefficient_of_deadweight_to_displacement=0.7,
    bunkers=200000,
    ballast=200000,
    length_of_ship=80,
    width_of_ship=16,
    added_mass_coefficient_in_surge=0.4,
    added_mass_coefficient_in_sway=0.4,
    added_mass_coefficient_in_yaw=0.4,
    dead_weight_tonnage=3850000,
    mass_over_linear_friction_coefficient_in_surge=130,
    mass_over_linear_friction_coefficient_in_sway=18,
    mass_over_linear_friction_coefficient_in_yaw=90,
    nonlinear_friction_coefficient__in_surge=2400,
    nonlinear_friction_coefficient__in_sway=4000,
    nonlinear_friction_coefficient__in_yaw=400
)
wave_model_config = WaveModelConfiguration(
    minimum_wave_frequency=0.4,
    maximum_wave_frequency=2.5,
    wave_frequency_discrete_unit_count=50,
    minimum_spreading_angle=-np.pi,
    maximum_spreading_angle=np.pi,
    spreading_angle_discrete_unit_count=10,
    spreading_coefficient=1,
    rho=1025.0,
    timestep_size=args.time_step
)
current_model_config = CurrentModelConfiguration(
    initial_current_velocity=0.01,
    current_velocity_standard_deviation=0.0075,
    current_velocity_decay_rate=0.025,
    initial_current_direction=np.deg2rad(0.0),
    current_direction_standard_deviation=0.025,
    current_direction_decay_rate=0.025,
    timestep_size=args.time_step
)
wind_model_config = WindModelConfiguration(
    initial_mean_wind_velocity=None,
    mean_wind_velocity_decay_rate=0.025,
    mean_wind_velocity_standard_deviation=0.005,
    initial_wind_direction=np.deg2rad(0.0),
    wind_direction_decay_rate=0.025,
    wind_direction_standard_deviation=0.025,
    minimum_mean_wind_velocity=0.0,
    maximum_mean_wind_velocity=42.0,
    minimum_wind_gust_frequency=0.06,
    maximum_wind_gust_frequency=0.4,
    wind_gust_frequency_discrete_unit_count=100,
    clip_speed_nonnegative=True,
    kappa_parameter=0.0026,
    U10=10.0,
    wind_evaluation_height=5.0,
    timestep_size=args.time_step
)
pto_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=0,
    shaft_generator_state=hybrid_shaft_gen_as_generator,
    name_tag='PTO'
)
pto_mode = MachineryMode(params=pto_mode_params)
pti_mode_params = MachineryModeParams(
    main_engine_capacity=0,
    electrical_capacity=2*diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_motor,
    name_tag='PTI'
)
pti_mode = MachineryMode(params=pti_mode_params)
mec_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_offline,
    name_tag='MEC'
)
mec_mode = MachineryMode(params=mec_mode_params)
mso_modes = MachineryModes([pto_mode, mec_mode, pti_mode])
fuel_spec_me = SpecificFuelConsumptionWartila6L26()
fuel_spec_dg = SpecificFuelConsumptionBaudouin6M26Dot3()
machinery_config = MachinerySystemConfiguration(
    machinery_modes=mso_modes,
    machinery_operating_mode=1,
    linear_friction_main_engine=68,
    linear_friction_hybrid_shaft_generator=57,
    gear_ratio_between_main_engine_and_propeller=0.6,
    gear_ratio_between_hybrid_shaft_generator_and_propeller=0.6,
    propeller_inertia=6000,
    propeller_diameter=3.1,
    propeller_speed_to_torque_coefficient=7.5,
    propeller_speed_to_thrust_force_coefficient=1.7,
    hotel_load=200000,
    rated_speed_main_engine_rpm=1000,
    rudder_angle_to_sway_force_coefficient=50e3,
    rudder_angle_to_yaw_force_coefficient=500e3,
    max_rudder_angle_degrees=35,
    max_rudder_rate_degree_per_s=2.3,
    specific_fuel_consumption_coefficients_me=fuel_spec_me.fuel_consumption_coefficients(),
    specific_fuel_consumption_coefficients_dg=fuel_spec_dg.fuel_consumption_coefficients()
)

# --- Ship 1: RL-agent (original path) ---
route1_filename = 'own_ship_route.txt'
route1_path = get_ship_route_path(ROOT, route1_filename)
start_E1, start_N1 = np.loadtxt(route1_path)[0]
ship1_config = SimulationConfiguration(
    initial_north_position_m=start_E1,
    initial_east_position_m=start_N1,
    initial_yaw_angle_rad=np.deg2rad(-60.0),
    initial_forward_speed_m_per_s=0.0,
    initial_sideways_speed_m_per_s=0.0,
    initial_yaw_rate_rad_per_s=0.0,
    integration_step=args.time_step,
    simulation_time=20000,
)
ship1 = ShipModel(
    ship_config=ship_config,
    simulation_config=ship1_config,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    machinery_config=machinery_config,
    throttle_controller_gain=ThrottleControllerGains(kp_ship_speed=2.50, ki_ship_speed=0.025, kp_shaft_speed=0.05, ki_shaft_speed=0.0001),
    heading_controller_gain=HeadingControllerGains(kp=1.5, kd=75, ki=0.005),
    los_parameters=LosParameters(radius_of_acceptance=args.radius_of_acceptance, lookahead_distance=args.lookahead_distance, integral_gain=0.002, integrator_windup_limit=4000),
    name_tag='RL Ship',
    route_name=route1_path,
    engine_steps_per_time_step=args.engine_step_count,
    initial_propeller_shaft_speed_rad_per_s=0,
    initial_propeller_shaft_acc_rad_per_sec2=0,
    desired_speed=4.0,
    cross_track_error_tolerance=750,
    nav_fail_time=args.nav_fail_time,
    map_obj=map[0],
    colav_mode='sbmpc',
    print_status=True
)
ship1.observer = ShipObserverEKF(
    dt=ship1.int.dt,
    x0=np.array([
        ship1.north,
        ship1.east,
        ship1.yaw_angle,
        ship1.forward_speed,
        ship1.sideways_speed,
        ship1.yaw_rate
    ], dtype=float)
)
ship1.use_observer_for_control = True
ship1_info = AssetInfo(
    current_north=ship1.north,
    current_east=ship1.east,
    current_yaw_angle=ship1.yaw_angle,
    forward_speed=ship1.forward_speed,
    sideways_speed=ship1.sideways_speed,
    name_tag=ship1.name_tag,
    ship_length=ship1.l_ship,
    ship_width=ship1.w_ship
)
ship1_asset = ShipAsset(ship_model=ship1, info=ship1_info)

# --- Ship 2: Passive, reversed path ---
route2_filename = 'own_ship_route.txt'
route2_path = get_ship_route_path(ROOT, route2_filename)
route2_points = np.loadtxt(route2_path)
reversed_points = route2_points[::-1]
start_E2, start_N2 = reversed_points[0]
ship2_config = SimulationConfiguration(
    initial_north_position_m=start_E2,
    initial_east_position_m=start_N2,
    initial_yaw_angle_rad=np.deg2rad(120.0),  # motsatt retning
    initial_forward_speed_m_per_s=0.0,
    initial_sideways_speed_m_per_s=0.0,
    initial_yaw_rate_rad_per_s=0.0,
    integration_step=args.time_step,
    simulation_time=20000,
)
ship2 = ShipModel(
    ship_config=ship_config,
    simulation_config=ship2_config,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    machinery_config=machinery_config,
    throttle_controller_gain=ThrottleControllerGains(kp_ship_speed=2.50, ki_ship_speed=0.025, kp_shaft_speed=0.05, ki_shaft_speed=0.0001),
    heading_controller_gain=HeadingControllerGains(kp=1.5, kd=75, ki=0.005),
    los_parameters=LosParameters(radius_of_acceptance=args.radius_of_acceptance, lookahead_distance=args.lookahead_distance, integral_gain=0.002, integrator_windup_limit=4000),
    name_tag='Passive Ship',
    route_name=route2_path,
    engine_steps_per_time_step=args.engine_step_count,
    initial_propeller_shaft_speed_rad_per_s=0,
    initial_propeller_shaft_acc_rad_per_sec2=0,
    desired_speed=4.0,
    cross_track_error_tolerance=750,
    nav_fail_time=args.nav_fail_time,
    map_obj=map[0],
    colav_mode='sbmpc',
    print_status=True
)
ship2.observer = ShipObserverEKF(
    dt=ship2.int.dt,
    x0=np.array([
        ship2.north,
        ship2.east,
        ship2.yaw_angle,
        ship2.forward_speed,
        ship2.sideways_speed,
        ship2.yaw_rate
    ], dtype=float)
)
ship2.use_observer_for_control = True
ship2_info = AssetInfo(
    current_north=ship2.north,
    current_east=ship2.east,
    current_yaw_angle=ship2.yaw_angle,
    forward_speed=ship2.forward_speed,
    sideways_speed=ship2.sideways_speed,
    name_tag=ship2.name_tag,
    ship_length=ship2.l_ship,
    ship_width=ship2.w_ship
)
ship2_asset = ShipAsset(ship_model=ship2, info=ship2_info)

assets: List[ShipAsset] = [ship1_asset, ship2_asset]

env = SeaEnvObserverAST(
    assets=assets,
    map=map,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    args=args,
    include_wave=True,
    include_wind=True,
    include_current=True)

# Last inn RL-modell
if args.model_path is not None:
    model_load_path = str(Path(args.model_path))
else:
    model_load_path = str(ROOT / "trained_model" / "AST-observer-train-realistic_2026-03-19_14-42-52_04ca" / "model")
model = SAC.load(model_load_path)

obs, info = env.reset()
while True:
    # RL-agent styrer kun skip 1, skip 2 får ingen RL-action (f.eks. null-action eller fast policy)
    action, _states = model.predict(obs, deterministic=True)
    # Sett action for skip 2 til null eller fast verdi
    if isinstance(action, np.ndarray) and action.shape[0] == 1:
        action = np.concatenate([action, np.zeros_like(action)])
    elif isinstance(action, np.ndarray) and action.shape[0] == 2:
        pass  # allerede to actions
    else:
        action = np.array([action, 0])
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# Resultater og plotting
ship1_results_df = pd.DataFrame().from_dict(env.assets[0].ship_model.simulation_results)
ship2_results_df = pd.DataFrame().from_dict(env.assets[1].ship_model.simulation_results)
result_dfs = [ship1_results_df, ship2_results_df]

map_anim = MapAnimator(
    assets=assets,
    map_gdfs=map_gdfs,
    interval_ms=500,
    status_asset_index=0
)
map_anim.run(fps=120, show=False, repeat=False)

polar_anim = PolarAnimator(focus_asset=assets[0], interval_ms=500)
polar_anim.run(fps=120, show=False, repeat=False)

animate_side_by_side(map_anim.fig, polar_anim.fig, left_frac=0.68, height_frac=0.92, gap_px=16, show=False)

plot_ship_status(ship1_asset, ship1_results_df, plot_env_load=True, show=False)
plot_ship_status(ship2_asset, ship2_results_df, plot_env_load=True, show=False)
plot_ship_and_real_map(assets, result_dfs, map_gdfs, show=False)
plt.show()
