# CSC580-module5CT

CSC580 Module 5CT Project

## Product

### Overview

This repository contains a toxicology prediction workflow for the Tox21 dataset.
The project uses DeepChem's MolNet loader with ECFP features, trains a weighted
Random Forest baseline, performs neural network hyperparameter search with Keras,
and evaluates weighted binary classification performance.

## Features

- `prepare_data()`: loads Tox21 splits via DeepChem and extracts task 0 labels/weights
- `build_random_forest_model()`: creates a class-balanced Random Forest baseline
- `train_random_forest_model()`: trains the Random Forest with sample weights
- `evaluate_random_forest_model()`: computes weighted train/validation/test accuracy
- `build_model()`: defines and compiles a configurable feed-forward neural network
- `train_model()`: trains the neural model with TensorBoard logging
- `evaluate_model()`: computes weighted accuracy from sigmoid predictions
- `plot_loss_curve()`: plots train and validation loss curves with Matplotlib
- `eval_tox21_hyperparams()`: evaluates one hyperparameter setting across repeated runs
- `main()`: orchestrates baseline training, grid search, best-model retraining, and test evaluation

## Getting Started

1. Create and activate a virtual environment:

```bash
python3 -m venv mod5ct
source mod5ct/bin/activate
```

2. Install dependencies for your Python version:

```bash
# Python 3.12
pip install -r requirements3.12.txt

# Python 3.10
pip install -r requirements3.10.txt
```

3. Run the project:

```bash
python main.py
```

4. (Optional) Launch TensorBoard for training logs:

```bash
tensorboard --logdir logs/tox21
```

## Notes

- Designed for coursework use in CSC580 Module 5CT.
- The pipeline uses the first Tox21 task (`TASK_INDEX = 0`) as a binary target.
- Randomness is controlled with a fixed seed (`SEED = 456`).
- Neural training uses weighted loss/metrics through per-sample task weights.
- Hyperparameter search currently explores hidden units, layer count, learning rate, dropout, epochs, and batch size.
- The Python 3.12 environment/requirements are not yet fully updated with all changes currently reflected in the Python 3.10 version.

## Outputs

1. Console output:
- Weighted Random Forest train/validation/test accuracy
- Per-configuration and averaged validation accuracy during grid search
- Best hyperparameter set and final weighted test accuracy
- TensorBoard log directory path for each neural training run

2. Artifacts:
- TensorBoard event files in `logs/tox21/<timestamp>/`
- Loss curve plot shown via Matplotlib (`Train Loss` vs `Validation Loss`)

## Additional Links

- [Code](https://github.com/Jason-csugc/CSC580-module5CT)
- [Issues](https://github.com/Jason-csugc/CSC580-module5CT/issues)
- [Pull requests](https://github.com/Jason-csugc/CSC580-module5CT/pulls)
- [Actions](https://github.com/Jason-csugc/CSC580-module5CT/actions)
- [Projects](https://github.com/Jason-csugc/CSC580-module5CT/projects)
- [Security and quality](https://github.com/Jason-csugc/CSC580-module5CT/security)