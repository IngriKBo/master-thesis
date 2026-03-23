from pathlib import Path
import sys

## PATH HELPER (OBLIGATORY)
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

### IMPORT SIMULATOR ENVIRONMENTS
from test_beds.ast_test.setup import get_env_assets

## IMPORT FUNCTIONS
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map

## IMPORT AST RELATED TOOLS
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy, MultiInputPolicy
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import FlattenObservation, RescaleAction
from gymnasium.utils.env_checker import check_env

### IMPORT TOOLS
import argparse
import pandas as pd
import os
import time

### IMPORT UTILS
from utils.get_path import get_trained_model_and_log_path
from utils.logger import log_ast_training_config
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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
    # Argument Parser
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
    parser.add_argument('--warm_up_time', type=int, default=2500, metavar='WARM_UP_TIME',
                        help='AST: time needed in second before policy - action sampling takes place (default: 1500)')
    parser.add_argument('--action_sampling_period', type=int, default=900, metavar='ACT_SAMPLING_PERIOD',
                        help='AST: time period in second between policy - action sampling (default: 900)')
    parser.add_argument('--total_steps', type=int, default=25000, metavar='TOTAL_STEPS',
                        help='AST: total training timesteps (default: 25000)')
    parser.add_argument('--strict_env_check', action='store_true',
                        help='AST: abort training if gym check_env fails (default: continue with warning)')
    parser.add_argument('--model_name', type=str, default='AST-observer-train-realistic', metavar='MODEL_NAME',
                        help='AST: base name for this training run folder under trained_model (default: AST-observer-train-realistic)')

    # Parse args
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

###################################### TRAIN THE MODEL #####################################

    # Get the args
    args = parse_cli_args()

    # Create unique output paths under trained_model/
    model_name = args.model_name
    model_path, log_path, tb_path = get_trained_model_and_log_path(root=ROOT, model_name=model_name)
    
    # Get the assets and AST Environment Wrapper
    env, assets, map_gdfs = get_env_assets(args=args, scenario='observer')

    # Hard safety check so we never train wrong scenario by accident
    if type(env).__name__ != "SeaEnvObserverAST":
        raise RuntimeError(f"Wrong environment selected: {type(env).__name__}. Expected SeaEnvObserverAST.")
    if env.assets[0].ship_model.observer is None:
        raise RuntimeError("Observer is not attached to ship model. Aborting observer-scenario training.")

    print(f"Training environment: {type(env).__name__}")
    print(f"Model save path    : {model_path}.zip")
    print(f"Run log path       : {log_path}.txt")
    print(f"TensorBoard path   : {tb_path}")
    
    # Check env sanity. This simulator is stochastic, so deterministic-check may fail.
    try:
        check_env(env)
        print("Environment passes all chekcs!")
    except Exception as e:
        print(f"Environment has issues: {e}")
        if args.strict_env_check:
            print("ABORT TRAINING (strict env check enabled)")
            sys.exit(1)  # non-zero exit code stops the script
        print("Continuing training (strict env check disabled).")

    # Log config + actual env class to file
    log_ast_training_config(args=args, txt_path=log_path, env=env, also_print=True)
    
    # Set the Policy
    # Later
    
    # Set RL model
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
    
    # Train the RL model. Record the time
    start_time = time.time()
    progress_cb = PercentProgressCallback(total_timesteps=args.total_steps, print_every_pct=1)
    ast_model.learn(total_timesteps=args.total_steps, callback=progress_cb)
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    hours, _         = divmod(minutes, 60)
    train_time = (hours, minutes, seconds)
    
    # Save the trained model
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