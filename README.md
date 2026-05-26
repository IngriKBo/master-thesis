# MAR-AST Simulator

This repository contains the simulator and adaptive stress testing code used in the thesis experiments.

If you want the thesis level overview, start with the root README in `../../README.md`. This README is for the simulator code in `simulator/tmp-mar-ast`.

## What Is In This Repository

- ship transit simulator code
- observer based AST environments
- training and evaluation scripts
- route and map data for experiments

## Setup

Create the Conda environment:

```bash
conda env create -f mar-ast.yml
```

Then install the main Python packages if they are missing:

```bash
pip install gymnasium
pip install "stable-baselines3[extra]"
pip install tensorboard
```

If you use FMU based co-simulation, also install:

```bash
pip install libcosimpy
```

GPU training is optional. If you want CUDA support, install a CUDA compatible PyTorch build from the official PyTorch site.

## Common Commands

Train a model:

```bash
python test_beds/ast_test/train_ast.py --scenario observer_noise_two_ships
```

Run a trained two ship measurement noise model:

```bash
python test_beds/ast_test/run_two_ships_measurement_noise.py --model_path "trained_model/<model_folder>"
```

Run without plots:

```bash
python test_beds/ast_test/run_two_ships_measurement_noise.py --model_path "trained_model/<model_folder>" --no_plots
```

Open TensorBoard logs:

```bash
tensorboard --logdir "trained_model"
```

## Repository Layout

- `env_wrappers/` custom RL environments
- `test_beds/ast_test/` training and evaluation scripts
- `simulator/` ship simulator code
- `data/` route and map files
- `trained_model/` local training outputs
- `saved_model/` local saved artifacts

## Main Experiment Scripts

- `test_beds/ast_test/train_ast.py` main training entry point
- `test_beds/ast_test/run_two_ships_measurement_noise.py` main two ship noise evaluation script
- `test_beds/ast_test/run_two_ships_estimator_tuning.py` two ship estimator tuning evaluation script
- `test_beds/ast_test/run_two_ships_passive_nominal.py` passive baseline case
- `test_beds/ast_test/setup.py` shared scenario and asset setup

## Notes

- `trained_model/` and `saved_model/` are local output folders and should not be committed.
- The codebase includes both estimator tuning and measurement noise scenarios.
- The thesis work mainly focuses on the measurement noise scenarios.

## Attribution

This code builds on the MAR-AST project by Andreas King Goksøyr and on the Ship in Transit Simulator by Børge Rokseth.