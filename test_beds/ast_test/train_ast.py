
import numpy as np
from pathlib import Path
import sys

# Add the project root to sys.path.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from test_beds.ast_test.setup import get_env_assets
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from gymnasium.utils.env_checker import check_env
import argparse
import pandas as pd
import os
import time
from utils.get_path import get_trained_model_and_log_path
from utils.logger import log_ast_training_config
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


SCENARIO_MODEL_NAMES = {
    'observer': 'AST-observer-one-ship-train',
    'observer_two_ships': 'AST-observer-two-ships-train',
    'observer_noise': 'AST-observer-noise-one-ship-train',
    'observer_noise_two_ships': 'AST-observer-noise-two-ships-train',
    'observer_noise_two_ships_extreme': 'AST-observer-noise-two-ships-extreme-train',
}


def resolve_model_name(args):
    if args.model_name is not None:
        return args.model_name
    base_name = SCENARIO_MODEL_NAMES.get(args.scenario, 'AST-train')
    return f"{base_name}-{args.observer_noise_profile}"


class PercentProgressCallback(BaseCallback):
    """Print training progress in percent."""
    def __init__(self, total_timesteps: int, print_every_pct: int = 1):
        super().__init__()
        self.total_timesteps = max(1, int(total_timesteps))
        self.print_every_pct = max(1, int(print_every_pct))
        self._last_printed_pct = -1

    def _on_training_start(self) -> None:
        print(f"Training progress: 0% (0/{self.total_timesteps})", flush=True)

    def _on_step(self) -> bool:
        pct = int(100 * self.num_timesteps / self.total_timesteps)
        pct = max(0, min(100, pct))
        if pct >= self._last_printed_pct + self.print_every_pct:
            print(f"Training progress: {pct}% ({self.num_timesteps}/{self.total_timesteps})", flush=True)
            self._last_printed_pct = pct
        return True


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Ship in Transit Simulation')

    ## Add arguments for environments
    parser.add_argument('--time_step', type=int, default=5, metavar='TIMESTEP',
                        help='ENV: time step size in second for ship transit simulator (default: 5)')
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
    parser.add_argument('--time_since_last_ship_drawing', type=int, default=30, metavar='SHIP_DRAW_TIME',
                        help='ENV: time delay in second between ship drawing record (default: 30)')
    parser.add_argument('--map_gpkg_filename', type=str, default="Stangvik.gpkg", metavar='MAP_GPKG_FILENAME',
                        help='ENV: name of the .gpkg filename for the map (default: "Stangvik.gpkg")')

    # Add arguments for AST-core
    parser.add_argument('--n_episodes', type=int, default=1, metavar='N_EPISODES',
                        help='AST: number of simulation episode counts (default: 1)')
    parser.add_argument('--warm_up_time', type=int, default=240, metavar='WARM_UP_TIME',
                        help='AST: time needed in second before policy action sampling takes place (default: 240)')
    parser.add_argument('--action_sampling_period', type=int, default=120, metavar='ACT_SAMPLING_PERIOD',
                        help='AST: time period in second between policy - action sampling (default: 120)')
    parser.add_argument('--total_steps', type=int, default=25000, metavar='TOTAL_STEPS',
                        help='AST: total training timesteps (default: 25000)')
    parser.add_argument('--strict_env_check', action='store_true',
                        help='AST: abort training if gym check_env fails (default: continue with warning)')
    parser.add_argument('--model_name', type=str, default=None, metavar='MODEL_NAME',
                        help='AST: base name for this training run folder under trained_model (default: auto-generated from scenario and noise profile)')
    parser.add_argument('--observer_noise_profile', type=str, default='realistic', metavar='OBSERVER_NOISE_PROFILE',
                        help='Observer noise profile: optimistic, realistic or conservative (default: realistic)')
    parser.add_argument('--estimator_tuning_bounds_profile', type=str, default='legacy',
                        choices=['legacy', 'realistic'],
                        help='Estimator-tuning action bounds profile: legacy [0.2, 5.0] or realistic [0.5, 2.0] (default: legacy)')
    parser.add_argument('--measurement_noise_attack_mode', type=str, default=None,
                        choices=['increase_only', 'symmetric_band'],
                        help='Measurement-noise attack mode: default increase_only, or symmetric_band for experimental under/over scaling')
    parser.add_argument('--allow_subnominal_noise', action='store_true',
                        help='Legacy alias for measurement_noise_attack_mode=symmetric_band')
    parser.add_argument('--measurement_noise_penalty_deadband', type=float, nargs=4,
                        default=[0.05, 0.05, 0.05, 0.08], metavar=('POS', 'YAW', 'SPEED', 'BIAS'),
                        help='Per-channel deadband around nominal measurement noise before penalties start (default: 0.05 0.05 0.05 0.08)')
    parser.add_argument('--two_ship_noise_max_scale', type=float, default=5.0,
                        help='Regular two ship measurement noise env: max allowed noise scaling per channel (default: 5.0)')
    parser.add_argument('--two_ship_realistic_noise_upper', type=float, nargs=4,
                        default=[2.0, 2.0, 2.0, 2.5], metavar=('POS', 'YAW', 'SPEED', 'BIAS'),
                        help='Regular two ship measurement noise env: realism band upper bounds used in the penalty logic (default: 2.0 2.0 2.0 2.5)')
    parser.add_argument('--extreme_noise_max_scale', type=float, default=20.0,
                        help='Extreme env: maximum measurement-noise scaling allowed per channel (default: 20.0)')
    parser.add_argument('--extreme_action_gain', type=float, default=1.5,
                        help='Extreme env: gain applied to normalized policy action before clipping to [-1, 1] (default: 1.5)')
    parser.add_argument('--extreme_noise_threshold', type=float, default=2.0,
                        help='Extreme env: threshold where extra extreme-load reward starts (default: 2.0)')
    parser.add_argument('--extreme_linear_scale_penalty_gain', type=float, default=0.001,
                        help='Extreme env: linear per-step penalty gain for scaling above nominal (default: 0.001)')
    parser.add_argument('--extreme_cumulative_linear_scale_penalty_gain', type=float, default=0.00001,
                        help='Extreme env: cumulative linear penalty gain for scaling above nominal (default: 0.00001)')
    parser.add_argument('--close_reward_distance', type=float, default=120.0,
                        help='Optional immediate proximity reward decay distance in meters for measurement-noise scenarios (default: 120.0)')
    parser.add_argument('--close_reward_gain', type=float, default=0.0,
                        help='Optional immediate proximity reward gain for measurement-noise scenarios (default: 0.0)')
    parser.add_argument('--collision_reward', type=float, default=120.0,
                        help='Additional collision reward in two-ship measurement-noise scenarios (default: 120.0)')
    parser.add_argument('--dcpa_reward_gain', type=float, default=30.0,
                        help='Two-ship DCPA/TCPA encounter reward gain for measurement-noise scenarios (default: 30.0)')
    parser.add_argument('--dcpa_reward_distance', type=float, default=120.0,
                        help='Distance threshold in meters where low DCPA starts to be strongly rewarded (default: 120.0)')
    parser.add_argument('--tcpa_reward_horizon', type=float, default=900.0,
                        help='Maximum future TCPA horizon in seconds for encounter shaping (default: 900.0)')
    parser.add_argument('--tcpa_window_center', type=float, default=240.0,
                        help='Preferred TCPA window center in seconds for targeted attack timing (default: 240.0)')
    parser.add_argument('--tcpa_window_width', type=float, default=180.0,
                        help='Half-width of the preferred TCPA timing window in seconds (default: 180.0)')
    parser.add_argument('--parallel_offset_m', type=float, default=300.0,
                        help='Lateral offset applied to the passive ship route in two-ship scenarios while keeping reverse-direction pairing (default: 300.0)')
    parser.add_argument('--fixed_training_route', action='store_true',
                        help='Disable random route sampling during training and keep the factory default route setup')

    parser.add_argument('--scenario', type=str, default='observer_two_ships', metavar='SCENARIO',
                        help='Scenario to train on: estimator_tuning, measurement_noise, estimator_tuning_two_ships, measurement_noise_two_ships, measurement_noise_two_ships_extreme, wave, or legacy aliases such as observer_noise_two_ships (default: observer_two_ships)')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cli_args()

    # Create unique output paths under trained_model.
    model_name = resolve_model_name(args)
    model_path, log_path, tb_path = get_trained_model_and_log_path(root=ROOT, model_name=model_name)

    # Build the requested environment.
    env, assets, map_gdfs = get_env_assets(args=args, scenario=args.scenario)

    # Make route sampling for training explicit instead of relying on env defaults.
    if hasattr(env, 'set_random_route_flag'):
        env.set_random_route_flag(not args.fixed_training_route)
    if hasattr(env, 'set_for_training_flag'):
        env.set_for_training_flag(True)

    if getattr(env, 'random_route', False):
        print('Training route mode : random training trajectories enabled')
    else:
        print('Training route mode : fixed default trajectory')

    # Abort early if the selected environment does not match the intended AST scenarios.
    if type(env).__name__ not in ["SeaEnvEstimatorTuningAST", "SeaEnvMeasurementNoiseAST", "TwoShipsEstimatorTuningEnv", "TwoShipsMeasurementNoiseEnv", "TwoShipsMeasurementNoiseExtremeEnv"]:
        raise RuntimeError(
            f"Wrong environment selected: {type(env).__name__}. "
            "Expected SeaEnvEstimatorTuningAST, SeaEnvMeasurementNoiseAST, TwoShipsEstimatorTuningEnv, TwoShipsMeasurementNoiseEnv or TwoShipsMeasurementNoiseExtremeEnv."
        )
    if env.assets[0].ship_model.observer is None:
        raise RuntimeError("Observer is not attached to ship model. Aborting observer-scenario training.")

    print(f"Training environment: {type(env).__name__}")
    print(f"Model save path    : {model_path}.zip")
    print(f"Run log path       : {log_path}.txt")
    print(f"TensorBoard path   : {tb_path}")

    # This simulator is stochastic, so deterministic environment checks may fail.
    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
        if args.strict_env_check:
            print("ABORT TRAINING (strict env check enabled)")
            sys.exit(1)
        print("Continuing training (strict env check disabled).")

    # Log the training configuration and resolved environment class.
    log_ast_training_config(args=args, txt_path=log_path, env=env, also_print=True)

    # Configure the RL model.
    ast_model = SAC("MultiInputPolicy",
                    env=env,
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    learning_starts=2500,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=1,
                    action_noise=None,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    n_steps=1,
                    ent_coef="auto",
                    target_update_interval=1,
                    target_entropy="auto",
                    use_sde=False,
                    sde_sample_freq=-1,
                    use_sde_at_warmup=False,
                    stats_window_size=100,
                    tensorboard_log=tb_path,
                    policy_kwargs=None,
                    verbose=1,
                    seed=None,
                    device='cuda')

    # Train the RL model and measure wall-clock time.
    start_time = time.time()
    progress_cb = PercentProgressCallback(total_timesteps=args.total_steps, print_every_pct=1)
    # Save checkpoints regularly during training.
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=str(Path(model_path).parent),
        name_prefix=model_name + "_checkpoint"
    )
    ast_model.learn(total_timesteps=args.total_steps, callback=[progress_cb, checkpoint_callback])
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    hours, _         = divmod(minutes, 60)
    train_time = (hours, minutes, seconds)

    # Save the trained model.
    ast_model.save(model_path)

################################## LOAD THE TRAINED MODEL ##################################

    # Remove the model to demonstrate saving and loading
    del ast_model
    
    # Load the trained model
    ast_model = SAC.load(model_path)
    
    ## Run the trained model
    obs, info = env.reset()
    while True:
        action, _states = ast_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
        
####################################### GET RESULTS ########################################

    # Print RL transition
    env.log_RL_transition_text(train_time=train_time,
                           txt_path=log_path,
                           also_print=True)
    
    # Print training time
    print(f"Training is done in {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
    print(f"Saved model        : {model_path}.zip")
    print(f"Saved run log      : {log_path}.txt")
    print(f"Saved TensorBoard  : {tb_path}")

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
                        show=True)

    # Plot 1: Trajectory
    plot_ship_status(env.assets[0], own_ship_results_df, plot_env_load=True, show=False)

    # Plot 2: Status plot
    plot_ship_and_real_map(assets, result_dfs, map_gdfs, show=True)