# Applied AI Midterm Exam

This repository contains the applied AI midterm project for the Dogs vs. Cats classification problem. Work is completed directly in Jupyter notebooks, with each model notebook containing its own data, model, training, and evaluation code.

> Placeholder content: final documentation, results, and figures will be added after the experiments complete.

## Repository structure

- `data/` - local Dogs vs. Cats images and generated split CSV files
- `notebooks/` - self-contained notebooks for exploration, training, and evaluation
- `models/` - saved model checkpoints and weights
- `outputs/` - plots, metrics, and prediction files

## Getting started

1. Open this repository in VS Code.
2. Create the Python environment from `environment.yml` or install `requirements.txt` in a local virtual environment.
3. Place the Kaggle dataset at `data/dogs-vs-cats-classification/`.
4. Run `notebooks/01_data_exploration.ipynb` to create local split files.
5. Run `notebooks/02_model_a_baseline.ipynb` to train and evaluate Model A locally.

Model A checkpoints are written to `models/checkpoints/`. Metrics, plots, and predictions are written to `outputs/model_a/`.
