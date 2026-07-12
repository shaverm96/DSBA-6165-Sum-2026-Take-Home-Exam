# Applied AI Midterm Exam

This repository contains the applied AI midterm project for the Dogs vs. Cats classification problem. The project includes a transfer-learning classifier baseline, a Super Resolution GAN (SRGAN), and a comparison between a classifier trained on original images and one trained on SRGAN-generated images.

> Placeholder content: final documentation, results, and figures will be added after the experiments complete.

## Repository structure

- `configs/` - YAML configuration files
- `data/` - generated data splits and derived datasets
- `notebooks/` - Jupyter notebooks for exploration, training, and evaluation
- `src/` - reusable Python modules for data, models, training, and evaluation
- `models/` - model checkpoints and saved weights
- `outputs/` - plots, metrics, and sample predictions
- `docs/` - final report and methodology documentation
- `logs/` - TensorBoard logs

## Getting started

1. Install dependencies from `requirements.txt` or `environment.yml`.
2. Place the Kaggle dataset root at `data/dogs-vs-cats-classification/`.
3. Run the data split script to generate reproducible train/val/test assignments.
