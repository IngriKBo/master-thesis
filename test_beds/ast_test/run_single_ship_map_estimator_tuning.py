from pathlib import Path
import sys

# Add the project root to sys.path.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

### IMPORT SIMULATOR ENVIRONMENTS
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

## IMPORT FUNCTIONS
from utils.get_path import get_ship_route_path, get_map_path
from utils.prepare_map import get_gdf_from_gpkg, get_polygon_from_gdf
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map
from test_beds.ast_test.setup import get_observer_noise_config, DEFAULT_OBSERVER_NOISE_PROFILE

## RL

from stable_baselines3 import SAC

### IMPORT TOOLS
import argparse
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


###############################################################################

# Argument Parser
parser = argparse.ArgumentParser(description='Ship in Transit Simulation (trained agent)')

## Add arguments for environments
parser.add_argument('--time_step', type=int, default=5, metavar='TIMESTEP',
                    help='ENV: time step size in second for ship transit simulator (default: 5)')
parser.add_argument('--observer_time_step', type=float, default=0.2,
                    help='ENV: time step size for observer updates (default: 0.2)')
parser.add_argument('--engine_step_count', type=int, default=10, metavar='ENGINE_STEP_COUNT',
                    help='ENV: engine integration step count in between simulation timestep (default: 10)')
parser.add_argument('--radius_of_acceptance', type=int, default=300, metavar='ROA',
                    help='ENV: radius of acceptance in meter for LOS algorithm (default: 300)')
parser.add_argument('--lookahead_distance', type=int, default=1000, metavar='LD',
                    help='ENV: lookahead distance in meter for LOS algorithm (default: 1000)')
parser.add_argument('--nav_fail_time', type=int, default=300, metavar='NAV_FAIL_TIME',
                    help='ENV: Allowed recovery time in second from navigational failure warning condition (default: 300)')
parser.add_argument('--ship_draw', type=bool, default=True, metavar='SHIP_DRAW',
                    help='ENV: record ship drawing for plotting and animation (default: True)')
parser.add_argument('--time_since_last_ship_drawing', default=30, metavar='SHIP_DRAW_TIME',
                    help='ENV: time delay in second between ship drawing record (default: 30)')

# Add arguments for AST-core
parser.add_argument('--warm_up_time', type=int, default=2500, metavar='WARM_UP_TIME',
                    help='AST: time needed in second before policy - action sampling takes place (default: 1500)')
parser.add_argument('--action_sampling_period', type=int, default=900, metavar='ACT_SAMPLING_PERIOD',
                    help='AST: time period in second between policy - action sampling (default: 900, matches training)')
parser.add_argument('--model_path', type=str, default=None, metavar='MODEL_PATH',
                    help='AST: path to model zip (or path without .zip). If omitted, uses a fixed trained_model run path')
parser.add_argument('--observer_noise_profile', type=str, default=DEFAULT_OBSERVER_NOISE_PROFILE,
                    choices=['optimistic', 'realistic', 'conservative'])

args = parser.parse_args()

# GPKG settings.
GPKG_PATH   = get_map_path(ROOT, "Stangvik.gpkg")
FRAME_LAYER = "frame_3857"
OCEAN_LAYER = "ocean_3857"
LAND_LAYER  = "land_3857"
COAST_LAYER = "coast_3857"               # optional
WATER_LAYER = "water_3857"               # optional

frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf = get_gdf_from_gpkg(GPKG_PATH, FRAME_LAYER, OCEAN_LAYER, LAND_LAYER, COAST_LAYER, WATER_LAYER)
map_gdfs = frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf

# Use east and north coordinates directly when building the polygon obstacle.
map_data = get_polygon_from_gdf(land_gdf)
map = [PolygonObstacle(map_data), frame_gdf]

# Engine configuration
main_engine_capacity = 2160e3 #4160e3
diesel_gen_capacity = 510e3 #610e3
hybrid_shaft_gen_as_generator = 'GEN'
hybrid_shaft_gen_as_motor = 'MOTOR'
hybrid_shaft_gen_as_offline = 'OFF'

# Configure the simulation
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
    initial_mean_wind_velocity=None,                    # Set to None to use a mean wind component
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
mso_modes = MachineryModes(
    [pto_mode, mec_mode, pti_mode]
)
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

### CONFIGURE THE SHIP SIMULATION MODELS
## Own ship
# --- Set up route and controllers ---
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



# --- Load route and assign east/north correctly ---
own_ship_route_filename = 'Stangvik_AST_reversed.txt'
own_ship_route_name = get_ship_route_path(ROOT, own_ship_route_filename)
route_data = np.loadtxt(own_ship_route_name)  # expecting two columns: east, north
# route_data[:,0] = east, route_data[:,1] = north
start_E = route_data[0, 0]
start_N = route_data[0, 1]
own_ship_config = SimulationConfiguration(
        initial_north_position_m=start_N,  # north from route file
        initial_east_position_m=start_E,   # east from route file
        initial_yaw_angle_rad=np.deg2rad(-60.0),
        initial_forward_speed_m_per_s=0.0,
        initial_sideways_speed_m_per_s=0.0,
        initial_yaw_rate_rad_per_s=0.0,
        integration_step=args.time_step,
        simulation_time=20000,
)

# --- Instantiate ShipModel (auto_pilot will process the route) ---

# --- Instantiate ShipModel (auto_pilot will process the route) ---
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

# --- Ensure autopilot waypoints are set as east/north ---
if hasattr(own_ship, 'auto_pilot') and hasattr(own_ship.auto_pilot, 'navigate'):
    own_ship.auto_pilot.navigate.east = route_data[:, 0].tolist()
    own_ship.auto_pilot.navigate.north = route_data[:, 1].tolist()


# Single source of truth for observer tuning/noise baseline
Q_SCALE = 1.0
R_SCALE = 1.0
observer_noise_cfg = get_observer_noise_config(args.observer_noise_profile)
MEAS_NOISE_STD = observer_noise_cfg['measurement_noise_std']
BIAS_NOISE_STD = observer_noise_cfg['bias_noise_std']
BASE_Q = np.diag([0.01, 0.01, 1e-4, 0.05, 0.05, 0.01])
Q_obs = BASE_Q * Q_SCALE
R_obs = np.diag(MEAS_NOISE_STD**2) * R_SCALE


# Attach the observer to the own ship only.
# Q and R are used for estimator tuning, not for random noise injection.
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
own_ship.use_observer_for_control = True  # ColAV follows the observer-driven signals in this script.

own_ship_info = AssetInfo(
    # dynamic state (mutable)
    current_north       = own_ship.north,
    current_east        = own_ship.east,
    current_yaw_angle   = own_ship.yaw_angle,
    forward_speed       = own_ship.forward_speed,
    sideways_speed      = own_ship.sideways_speed,

    # static properties (constants)
    name_tag            = own_ship.name_tag,
    ship_length         = own_ship.l_ship,
    ship_width          = own_ship.w_ship
)
# Wraps simulation objects based on the ship type using a dictionary
own_ship_asset = ShipAsset(
    ship_model=own_ship,
    info=own_ship_info
)

# Package the assets for reinforcement learning agent
assets: List[ShipAsset] = [own_ship_asset]

################################### ENV SPACE ###################################

# Initiate Multi-Ship Reinforcement Learning Environment Class Wrapper

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


# Position handling is correct here.

# Load trained model
if args.model_path is not None:
    model_load_path = str(Path(args.model_path))
else:
    model_load_path = str(ROOT / "trained_model" / "AST-observer-train-realistic_2026-04-13_01-44-44_d35c" / "model.zip")

model = SAC.load(model_load_path)

# Run trained model

# --- Record observer tuning history ---
observer_tuning_history = []
observer_tuning_times = []
initial_tuning = [1.0, 1.0, 1.0, 1.0]  # Reference (nominal) tuning

obs, info = env.reset()
step_idx = 0
observer_steps_per_sim = int(np.round(args.time_step / observer_dt)) if observer_dt < args.time_step else 1

# --- Logging for observer validation ---
true_state_log = []
est_state_log = []
innovation_log = []
while True:
    action, _states = model.predict(obs, deterministic=True)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    # Legacy models may emit actions in the old 3D actual-scale space; map them back to normalized space.
    if action.size == 3 and np.any((action < -1.0) | (action > 1.0)):
        action = np.array([
            env._normalize(float(action[0]), env.obs_tuning_range["r_pos"][0], env.obs_tuning_range["r_pos"][1]),
            env._normalize(1.0, env.obs_tuning_range["r_yaw"][0], env.obs_tuning_range["r_yaw"][1]),
            env._normalize(float(action[1]), env.obs_tuning_range["r_speed"][0], env.obs_tuning_range["r_speed"][1]),
            env._normalize(float(action[2]), env.obs_tuning_range["q"][0], env.obs_tuning_range["q"][1]),
        ], dtype=np.float32)
    print(f"Step {step_idx}: Agent action (normalized): {action}")
    obs, reward, terminated, truncated, info = env.step(action)

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

    if observer_steps_per_sim > 1:
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

    # Print the denormalized observer tuning chosen by the agent for this action
    if hasattr(env, 'action_list') and len(env.action_list) > 0:
        tuning = list(env.action_list[-1])
        observer_tuning_history.append(tuning)
        # Use the time at which the action was sampled
        if hasattr(env, 'action_time_list') and len(env.action_time_list) > 0:
            t = env.action_time_list[-1]
        else:
            t = len(observer_tuning_history)
        observer_tuning_times.append(t)
        if len(tuning) >= 4:
            print(f"Step {step_idx}: Observer tuning chosen by agent: alpha_r_pos={tuning[0]:.3f}, alpha_r_yaw={tuning[1]:.3f}, alpha_r_speed={tuning[2]:.3f}, alpha_q={tuning[3]:.3f} at time {t}")
        else:
            print(f"Step {step_idx}: Observer tuning chosen by agent: alpha_r_pos={tuning[0]:.3f}, alpha_r_speed={tuning[1]:.3f}, alpha_q={tuning[2]:.3f} at time {t}")
    else:
        print(f"Step {step_idx}: No observer tuning recorded.")
    step_idx += 1
    if terminated or truncated:
        print(f"[DEBUG] Simulation stopped. terminated={terminated}, truncated={truncated}, info={info}")
        break

################################## GET RESULTS ##################################

## Get the simulation results for all assets, and plot the asset simulation results
own_ship_results_df = pd.DataFrame().from_dict(env.assets[0].ship_model.simulation_results)
result_dfs = [own_ship_results_df]

# Build both animations (don’t show yet)
repeat=False
map_anim = MapAnimator(
    assets=assets,
    map_gdfs=map_gdfs,
    interval_ms=500,
    status_asset_index=0  # flags for own ship
)
map_anim.run(fps=120, show=False, repeat=False)

polar_anim = PolarAnimator(focus_asset=assets[0], interval_ms=500)
polar_anim.run(fps=120, show=False, repeat=False)

# Place windows next to each other, same height, centered
animate_side_by_side(map_anim.fig, polar_anim.fig,
                     left_frac=0.68,  # how wide the map window is
                     height_frac=0.92,
                     gap_px=16,
                     show=False)

# Plot 1: Trajectory + observer plots



plot_ship_status(own_ship_asset, own_ship_results_df, plot_env_load=True, show=False)


plot_ship_and_real_map(assets, result_dfs, map_gdfs, show=False)


# --- Plot observer tuning over time ---

import matplotlib.pyplot as plt
# Use full observer tuning history if available (logs every sim timestep)
if hasattr(env, 'observer_tuning_full_history') and len(env.observer_tuning_full_history) > 0:
    observer_tuning_history = np.array(env.observer_tuning_full_history)
    observer_tuning_times = np.array(env.observer_tuning_full_times)
else:
    observer_tuning_history = np.array(observer_tuning_history)
    observer_tuning_times = np.array(observer_tuning_times)

if observer_tuning_history.shape[0] > 0:
    plt.figure(figsize=(10, 5))
    plt.step(observer_tuning_times, observer_tuning_history[:,0], where='post', label='alpha_r_pos')
    if observer_tuning_history.shape[1] >= 4:
        plt.step(observer_tuning_times, observer_tuning_history[:,1], where='post', label='alpha_r_yaw')
        plt.step(observer_tuning_times, observer_tuning_history[:,2], where='post', label='alpha_r_speed')
        plt.step(observer_tuning_times, observer_tuning_history[:,3], where='post', label='alpha_q')
        tuning_names = ['alpha_r_pos', 'alpha_r_yaw', 'alpha_r_speed', 'alpha_q']
    else:
        plt.step(observer_tuning_times, observer_tuning_history[:,1], where='post', label='alpha_r_speed')
        plt.step(observer_tuning_times, observer_tuning_history[:,2], where='post', label='alpha_q')
        tuning_names = ['alpha_r_pos', 'alpha_r_speed', 'alpha_q']
    for i, name in enumerate(tuning_names):
        plt.hlines(initial_tuning[i], observer_tuning_times[0], observer_tuning_times[-1], colors='k', linestyles='dashed', alpha=0.5, label=f'{name} reference' if i==0 else None)
    plt.xlabel('Simulation time [s]')
    plt.ylabel('Observer tuning (scaling)')
    plt.title('Observer tuning over time (RL agent)')
    plt.legend()
    plt.grid(True)

# --- Plot all noise components (white, bias, total) for observer ---
observer = env.assets[0].ship_model.observer
if hasattr(observer, 'total_noise_log') and len(observer.total_noise_log) > 0:
    total_noise = np.array(observer.total_noise_log)
    white_noise = np.array(observer.white_noise_log)
    bias_noise = np.array(observer.bias_log)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(total_noise[:, 0], label="Total north", color="tab:blue", alpha=0.5)
    axs[0].plot(white_noise[:, 0], label="White north", color="tab:cyan", linestyle="dashed", alpha=0.7)
    axs[0].plot(total_noise[:, 1], label="Total east", color="tab:orange", alpha=0.5)
    axs[0].plot(white_noise[:, 1], label="White east", color="tab:olive", linestyle="dashed", alpha=0.7)
    axs[0].plot(bias_noise[:, 0], label="Bias north (slow)", color="red", linewidth=2.5, zorder=10)
    axs[0].plot(bias_noise[:, 1], label="Bias east (slow)", color="darkred", linewidth=2.5, zorder=10)
    axs[0].set_ylabel("Position noise [m]")
    axs[0].set_title("Measurement noise components: position")
    axs[0].legend()

    axs[1].plot(total_noise[:, 3], label="Total speed", color="tab:green", alpha=0.5)
    axs[1].plot(white_noise[:, 3], label="White speed", color="tab:olive", linestyle="dashed", alpha=0.7)
    axs[1].plot(bias_noise[:, 3], label="Bias speed (slow)", color="red", linewidth=2.5, zorder=10)
    axs[1].set_ylabel("Speed noise [m/s]")
    axs[1].set_xlabel("Timestep")
    axs[1].set_title("Measurement noise components: speed")
    axs[1].legend()

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

fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for i in range(4):
    axs[i].plot(innovation_log[:, i], label="Innovation (meas - est)", color="tab:green", alpha=0.7)
    axs[i].set_ylabel(labels[i])
    axs[i].axhline(0, color="k", linestyle=":", linewidth=1)
    axs[i].legend()
axs[0].set_title("Innovation (Measurement - Estimate)")
axs[-1].set_xlabel("Timestep")
plt.tight_layout()

# Show all figures at once after simulation and plotting are fully complete.
plt.show()
