from pathlib import Path
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_wrappers.sea_env_ast_v2.estimator_tuning_env import SeaEnvEstimatorTuningAST
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
from test_beds.ast_test.setup import get_observer_noise_config, DEFAULT_OBSERVER_NOISE_PROFILE
import argparse
from typing import List

parser = argparse.ArgumentParser(description='Ship in Transit Simulation (observer, no RL tuning)')
parser.add_argument('--time_step', type=float, default=1.0)  # Simuleringens tidssteg
parser.add_argument('--observer_time_step', type=float, default=0.2, help='Tidssteg for observer (hvis raskere enn simulering)')
parser.add_argument('--engine_step_count', type=int, default=10)
parser.add_argument('--radius_of_acceptance', type=int, default=300)
parser.add_argument('--lookahead_distance', type=int, default=1000)
parser.add_argument('--nav_fail_time', type=int, default=300)
parser.add_argument('--ship_draw', type=bool, default=True)
parser.add_argument('--time_since_last_ship_drawing', default=30)
parser.add_argument('--warm_up_time', type=int, default=2500)
parser.add_argument('--observer_noise_profile', type=str, default=DEFAULT_OBSERVER_NOISE_PROFILE,
                    choices=['optimistic', 'realistic', 'conservative'])
parser.add_argument('--tuning_profile', type=str, default='extreme_model_trusting',
                    choices=['nominal', 'hard_model_trusting', 'extreme_model_trusting', 'hard_measurement_trusting', 'extreme_measurement_trusting'])
parser.add_argument('--alpha_r_pos', type=float, default=None)
parser.add_argument('--alpha_r_yaw', type=float, default=None)
parser.add_argument('--alpha_r_speed', type=float, default=None)
parser.add_argument('--alpha_q', type=float, default=None)
args = parser.parse_args()


TUNING_PROFILES = {
    'nominal': (1.0, 1.0, 1.0, 1.0),
    'hard_model_trusting': (10.0, 10.0, 10.0, 0.01),
    'extreme_model_trusting': (10.0, 10.0, 10.0, 1e-4),
    'hard_measurement_trusting': (0.05, 0.05, 0.05, 10.0),
    'extreme_measurement_trusting': (0.005, 0.005, 0.005, 10.0),
}


def resolve_test_tuning(cli_args):
    tuning = list(TUNING_PROFILES[cli_args.tuning_profile])
    overrides = [cli_args.alpha_r_pos, cli_args.alpha_r_yaw, cli_args.alpha_r_speed, cli_args.alpha_q]
    for idx, override in enumerate(overrides):
        if override is not None:
            tuning[idx] = float(override)
    return tuple(tuning)


# Map and route setup (match other run scripts)
GPKG_PATH   = get_map_path(ROOT, "Stangvik.gpkg")
FRAME_LAYER = "frame_3857"
OCEAN_LAYER = "ocean_3857"
LAND_LAYER  = "land_3857"
COAST_LAYER = "coast_3857"
WATER_LAYER = "water_3857"
frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf = get_gdf_from_gpkg(GPKG_PATH, FRAME_LAYER, OCEAN_LAYER, LAND_LAYER, COAST_LAYER, WATER_LAYER)
map_gdfs = frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf
map_data = get_polygon_from_gdf(land_gdf)
map = [PolygonObstacle(map_data), frame_gdf]

# Engine and ship config (reuse from your other scripts)
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
mso_modes = MachineryModes([
    pto_mode, mec_mode, pti_mode
])
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

# Route and ship setup
own_ship_route_filename = 'Stangvik_AST_reversed.txt'  # Samme som true_colav
own_ship_route_name = get_ship_route_path(ROOT, own_ship_route_filename)
route_data = np.loadtxt(own_ship_route_name)
start_E = route_data[0, 0]
start_N = route_data[0, 1]
own_ship_config = SimulationConfiguration(
    initial_north_position_m=start_N,
    initial_east_position_m=start_E,
    initial_yaw_angle_rad=np.deg2rad(-60.0),
    initial_forward_speed_m_per_s=0.0,
    initial_sideways_speed_m_per_s=0.0,
    initial_yaw_rate_rad_per_s=0.0,
    integration_step=args.time_step,
    simulation_time=20000,
)
own_ship_throttle_controller_gains = ThrottleControllerGains(
   kp_ship_speed=2.50, ki_ship_speed=0.025, kp_shaft_speed=0.05, ki_shaft_speed=0.0001
)
own_ship_heading_controller_gains = HeadingControllerGains(kp=1.5, kd=75, ki=0.005)
own_ship_los_guidance_parameters = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)
own_ship_desired_speed = 4.0
own_ship_cross_track_error_tolerance = 750
own_ship_initial_propeller_shaft_speed = 0
own_ship_initial_propeller_shaft_acceleration = 0
own_ship = ShipModel(
    ship_config=ship_config,
    simulation_config=own_ship_config,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    machinery_config=machinery_config,
    throttle_controller_gain=own_ship_throttle_controller_gains,
    heading_controller_gain=own_ship_heading_controller_gains,
    los_parameters=own_ship_los_guidance_parameters,
    name_tag='Own ship',
    route_name=own_ship_route_name,
    engine_steps_per_time_step=args.engine_step_count,
    initial_propeller_shaft_speed_rad_per_s=own_ship_initial_propeller_shaft_speed * np.pi /30,
    initial_propeller_shaft_acc_rad_per_sec2=own_ship_initial_propeller_shaft_acceleration * np.pi / 30,
    desired_speed=own_ship_desired_speed,
    cross_track_error_tolerance=own_ship_cross_track_error_tolerance,
    nav_fail_time=args.nav_fail_time,
    map_obj=map[0],
    colav_mode='sbmpc',
    print_status=True
)

if hasattr(own_ship, 'auto_pilot') and hasattr(own_ship.auto_pilot, 'navigate'):
    own_ship.auto_pilot.navigate.east = route_data[:, 0].tolist()
    own_ship.auto_pilot.navigate.north = route_data[:, 1].tolist()

# =====================
#  SKRU OBSERVER AV/PÅ HER:
# =====================
USE_OBSERVER = True  

# Single source of truth for observer tuning
Q_SCALE = 1.0
R_SCALE = 1.0
observer_noise_cfg = get_observer_noise_config(args.observer_noise_profile)
MEAS_NOISE_STD = observer_noise_cfg['measurement_noise_std']
BIAS_NOISE_STD = observer_noise_cfg['bias_noise_std']
BASE_Q = np.diag([0.01, 0.01, 1e-4, 0.05, 0.05, 0.01])
Q_obs = BASE_Q * Q_SCALE
R_obs = np.diag(MEAS_NOISE_STD**2) * R_SCALE

# Bestem observerens tidssteg
observer_dt = args.observer_time_step if args.observer_time_step is not None else own_ship.int.dt
own_ship.observer = ShipObserverEKF(
    dt=observer_dt,
    x0=np.array([
        own_ship.north,
        own_ship.east,
        own_ship.yaw_angle,
        own_ship.forward_speed,
        own_ship.sideways_speed,
        own_ship.yaw_rate
    ], dtype=float),
    P0=np.eye(6)*1e-3,
    Q=Q_obs,
    R=R_obs
)
own_ship.observer.measurement_noise_std = MEAS_NOISE_STD
own_ship.observer.bias_noise_std = BIAS_NOISE_STD
own_ship.use_observer_for_control = USE_OBSERVER
own_ship_info = AssetInfo(
    current_north       = own_ship.north,
    current_east        = own_ship.east,
    current_yaw_angle   = own_ship.yaw_angle,
    forward_speed       = own_ship.forward_speed,
    sideways_speed      = own_ship.sideways_speed,
    name_tag            = own_ship.name_tag,
    ship_length         = own_ship.l_ship,
    ship_width          = own_ship.w_ship
)
own_ship_asset = ShipAsset(
    ship_model=own_ship,
    info=own_ship_info
)
assets: List[ShipAsset] = [own_ship_asset]

# Environment: observer, but no RL tuning (agent does not act)
env = SeaEnvEstimatorTuningAST(
    assets=assets,
    map=map,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    args=args,
    include_wave=True,
    include_wind=True,
    include_current=True)

# Widen the local test range so failure-seeking tunings are not clipped away.
env.obs_tuning_range = {
    'r_pos': np.array([1e-3, 1e3], dtype=np.float32),
    'r_yaw': np.array([1e-3, 1e3], dtype=np.float32),
    'r_speed': np.array([1e-3, 1e3], dtype=np.float32),
    'q': np.array([1e-6, 1e3], dtype=np.float32),
}

own_ship_results_df = pd.DataFrame().from_dict(env.assets[0].ship_model.simulation_results)

obs, info = env.reset()
step_idx = 0
observer_steps_per_sim = int(np.round(args.time_step / observer_dt)) if observer_dt < args.time_step else 1

TEST_TUNING = resolve_test_tuning(args)
print(f'Using tuning profile {args.tuning_profile}: {TEST_TUNING}')
nominal_action = np.array(env._normalize_action(TEST_TUNING), dtype=np.float32)

# --- Logging for observer validation ---
true_state_log = []  # [north, east, yaw, speed]
est_state_log = []   # [north, east, yaw, speed]
innovation_log = []  # [north, east, yaw, speed]

while True:
    action = nominal_action
    # Ta et simuleringssteg (skip, miljø, etc)
    obs, reward, terminated, truncated, info = env.step(action)

    # Log true and estimated states
    true_state = np.array([
        own_ship.north,
        own_ship.east,
        own_ship.yaw_angle,
        np.hypot(own_ship.forward_speed, own_ship.sideways_speed)
    ])
    est_state = own_ship.observer.x[:4].copy() if hasattr(own_ship.observer, 'x') else np.zeros(4)
    innovation = true_state - est_state
    true_state_log.append(true_state)
    est_state_log.append(est_state)
    innovation_log.append(innovation)

    # Oppdater observeren flere ganger hvis ønsket
    if USE_OBSERVER and observer_steps_per_sim > 1:
        for _ in range(observer_steps_per_sim - 1):
            meas = np.array([
                own_ship.north,
                own_ship.east,
                own_ship.yaw_angle,
                np.hypot(own_ship.forward_speed, own_ship.sideways_speed)
            ])
            noisy_meas = own_ship.observer.apply_measurement_noise(meas)
            own_ship.observer.predict()
            own_ship.observer.update(noisy_meas)

    step_idx += 1
    if terminated or truncated:
        break

own_ship_results_df = pd.DataFrame().from_dict(env.assets[0].ship_model.simulation_results)
result_dfs = [own_ship_results_df]



map_anim = MapAnimator(
    assets=assets,
    map_gdfs=map_gdfs,
    interval_ms=500,
    status_asset_index=0
)
map_anim.run(fps=120, show=False, repeat=False)
polar_anim = PolarAnimator(focus_asset=assets[0], interval_ms=500)
polar_anim.run(fps=120, show=False, repeat=False)
animate_side_by_side(map_anim.fig, polar_anim.fig,
                     left_frac=0.68,
                     height_frac=0.92,
                     gap_px=16,
                     show=False)
plot_ship_status(own_ship_asset, own_ship_results_df, plot_env_load=True, show=False)
plot_ship_and_real_map(assets, result_dfs, map_gdfs, show=False)
 # --- Plot all noise components (white, bias, total) for observer ---
observer = env.assets[0].ship_model.observer
if hasattr(observer, 'total_noise_log') and len(observer.total_noise_log) > 0:
    total_noise = np.array(observer.total_noise_log)
    white_noise = np.array(observer.white_noise_log)
    bias_noise = np.array(observer.bias_log)
    yaw_scale = 180.0 / np.pi
    fig, axs = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    # Position noise (north/east)
    axs[0].plot(total_noise[:, 0], label="Total north", color="tab:blue", alpha=0.5)
    axs[0].plot(white_noise[:, 0], label="White north", color="tab:cyan", linestyle="dashed", alpha=0.7)
    axs[0].plot(total_noise[:, 1], label="Total east", color="tab:orange", alpha=0.5)
    axs[0].plot(white_noise[:, 1], label="White east", color="tab:olive", linestyle="dashed", alpha=0.7)
    # Plot bias (slowly varying) noise last, in front, thicker and red
    axs[0].plot(bias_noise[:, 0], label="Bias north (slow)", color="red", linewidth=2.5, zorder=10)
    axs[0].plot(bias_noise[:, 1], label="Bias east (slow)", color="darkred", linewidth=2.5, zorder=10)
    axs[0].set_ylabel("Position noise [m]")
    axs[0].set_title("Measurement noise components: position")
    axs[0].legend()

    # Heading noise in degrees for easier interpretation
    axs[1].plot(total_noise[:, 2] * yaw_scale, label="Total heading", color="tab:purple", alpha=0.5)
    axs[1].plot(white_noise[:, 2] * yaw_scale, label="White heading", color="plum", linestyle="dashed", alpha=0.7)
    axs[1].plot(bias_noise[:, 2] * yaw_scale, label="Bias heading (slow)", color="red", linewidth=2.5, zorder=10)
    axs[1].set_ylabel("Heading noise [deg]")
    axs[1].set_title("Measurement noise components: heading")
    axs[1].legend()

    # Speed noise
    axs[2].plot(total_noise[:, 3], label="Total speed", color="tab:green", alpha=0.5)
    axs[2].plot(white_noise[:, 3], label="White speed", color="tab:olive", linestyle="dashed", alpha=0.7)
    axs[2].plot(bias_noise[:, 3], label="Bias speed (slow)", color="red", linewidth=2.5, zorder=10)
    axs[2].set_ylabel("Speed noise [m/s]")
    axs[2].set_title("Measurement noise components: speed")
    axs[2].legend()
    axs[2].set_xlabel("Timestep")
plt.show()

# --- Plot observer validation: true vs. estimated and innovation ---
true_state_log = np.array(true_state_log)
est_state_log = np.array(est_state_log)
innovation_log = np.array(innovation_log)
labels = ["North [m]", "East [m]", "Yaw [rad]", "Speed [m/s]"]
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
for i in range(4):
    axs[i].plot(true_state_log[:, i], label="True", color="tab:blue", alpha=0.7)
    axs[i].plot(est_state_log[:, i], label="Estimated", color="tab:orange", alpha=0.7)
    axs[i].set_ylabel(labels[i])
    axs[i].legend()
axs[0].set_title("True vs. Estimated States (Observer Validation)")
axs[-1].set_xlabel("Timestep")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for i in range(4):
    axs[i].plot(innovation_log[:, i], label="Innovation (meas - est)", color="tab:green", alpha=0.7)
    axs[i].set_ylabel(labels[i])
    axs[i].axhline(0, color="k", linestyle=":", linewidth=1)
    axs[i].legend()
axs[0].set_title("Innovation (Measurement - Estimate)")
axs[-1].set_xlabel("Timestep")
plt.tight_layout()
plt.show()
