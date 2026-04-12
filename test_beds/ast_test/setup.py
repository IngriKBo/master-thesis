from pathlib import Path
import sys

## PATH HELPER (OBLIGATORY)
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

### IMPORT SIMULATOR ENVIRONMENTS
from env_wrappers.sea_env_ast_v2.env import AssetInfo, ShipAsset
from env_wrappers.sea_env_ast_v2.observer_env import SeaEnvObserverAST
from env_wrappers.sea_env_ast_v2.observer_two_ships_env import ObserverTwoShipsEnv
from env_wrappers.sea_env_ast_v1.env import SeaEnvAST

from simulator.ship_in_transit.sub_systems.ship_model import  ShipConfiguration, SimulationConfiguration, ShipModel
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

### IMPORT TOOLS
from typing import List
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ============================================
# Environment Scenarios Registry
# Add new scenarios here as key-value pairs
# ============================================
ENVIRONMENT_SCENARIOS = {
    'observer': SeaEnvObserverAST,
    'observer_two_ships': ObserverTwoShipsEnv,
    'wave': SeaEnvAST,
    # Add more scenarios here: 'scenario_name': EnvironmentClass
}

def get_env_assets(args, print_ship_status=False, scenario='observer'):
    """
    Get environment and assets for training/simulation.
    
    Args:
        args: Command line arguments
        print_ship_status: Whether to print ship status (bool)
        scenario: Environment scenario name - 'observer' or 'wave' (str)
    
    Returns:
        env, assets, map_gdfs
    """
    
    # Validate scenario
    if scenario not in ENVIRONMENT_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(ENVIRONMENT_SCENARIOS.keys())}")
    
    env_class = ENVIRONMENT_SCENARIOS[scenario]

    # -----------------------
    # GPKG settings (edit if your layer names differ)
    # -----------------------
    GPKG_PATH   = get_map_path(ROOT, args.map_gpkg_filename)       # <-- put your file here (or absolute path)
    FRAME_LAYER = "frame_3857"
    OCEAN_LAYER = "ocean_3857"
    LAND_LAYER  = "land_3857"
    COAST_LAYER = "coast_3857"               # optional
    WATER_LAYER = "water_3857"               # optional

    frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf = get_gdf_from_gpkg(GPKG_PATH, FRAME_LAYER, OCEAN_LAYER, LAND_LAYER, COAST_LAYER, WATER_LAYER)
    map_gdfs = frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf

    map_data = get_polygon_from_gdf(land_gdf)   # list of exterior rings (E,N)
    map = [PolygonObstacle(map_data), frame_gdf]              # <-- reuse your existing simulator map type

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
    own_ship_route_filename = 'Stangvik_AST_reversed.txt'
    own_ship_route_name = get_ship_route_path(ROOT, own_ship_route_filename)

    start_E, start_N = np.loadtxt(own_ship_route_name)[0]  # expecting two columns: east, north

    own_ship_config = SimulationConfiguration(
        initial_north_position_m=start_E,
        initial_east_position_m=start_N,
        initial_yaw_angle_rad=np.deg2rad(-60.0),
        initial_forward_speed_m_per_s=0.0,
        initial_sideways_speed_m_per_s=0.0,
        initial_yaw_rate_rad_per_s=0.0,
        integration_step=args.time_step,
        simulation_time=20000,
    )
    # Set the throttle and autopilot controllers for the own ship
    own_ship_throttle_controller_gains = ThrottleControllerGains(
        kp_ship_speed=2.50, ki_ship_speed=0.025, kp_shaft_speed=0.05, ki_shaft_speed=0.0001
    )

    own_ship_heading_controller_gains = HeadingControllerGains(kp=1.5, kd=0, ki=0.00)
    own_ship_los_guidance_parameters = LosParameters(
        radius_of_acceptance=args.radius_of_acceptance,
        lookahead_distance=args.lookahead_distance,
        integral_gain=0.002,
        integrator_windup_limit=4000
    )
    own_ship_desired_speed = 4.5
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
        print_status=print_ship_status
    )


    # Attach EKF observer when training/evaluating observer scenario
    if scenario in ['observer', 'observer_two_ships']:
        own_ship.observer = ShipObserverEKF(
            dt=own_ship.int.dt,
            x0=np.array([
                own_ship.north,
                own_ship.east,
                own_ship.yaw_angle,
                own_ship.forward_speed,
                own_ship.sideways_speed,
                own_ship.yaw_rate
            ], dtype=float)
        )
        own_ship.use_observer_for_control = True

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

    if scenario == 'observer_two_ships':
        # --- Ship 2: Motsatt path av skip 1 ---
        route2_filename = 'Stangvik_AST.txt'  # motsatt rute av own_ship
        route2_path = get_ship_route_path(ROOT, route2_filename)
        start_E2, start_N2 = np.loadtxt(route2_path)[0]
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
            print_status=print_ship_status
        )
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
        assets: List[ShipAsset] = [own_ship_asset, ship2_asset]
        env = env_class(
            assets=assets,
            map=map,
            wave_model_config=wave_model_config,
            current_model_config=current_model_config,
            wind_model_config=wind_model_config,
            args=args,
            include_wave=True,
            include_wind=True,
            include_current=True)
    else:
        assets: List[ShipAsset] = [own_ship_asset]
        env = env_class(
            assets=assets,
            map=map,
            wave_model_config=wave_model_config,
            current_model_config=current_model_config,
            wind_model_config=wind_model_config,
            args=args,
            include_wave=True,
            include_wind=True,
            include_current=True)
    return env, assets, map_gdfs