# Deep Learning

This directory contains experiment-oriented deep learning utilities for EEG analysis.

## Directory Structure

- `supervised/` — implementations for supervised EEG models, training loops, and related helpers.
- `unsupervised/` — unsupervised learning components such as autoencoders and representation-learning modules.
- `joint_architectures/` — model definitions for joint or multi-branch architectures that combine multiple objectives or modalities.
- `prepare_datasets.py` — converts public EEG datasets into the NumPy format expected by the training pipeline.
- `cross_val.py` — provides cross-validation and evaluation utilities for training and benchmarking models.
- `run_experiment.py` — entry point for launching experiments and model runs.
- `archival_cv.py` and `smoke_test_cross_val.py` — older or auxiliary scripts for experimentation and quick validation.

## Key Scripts

### `prepare_datasets.py`

This script prepares raw EEG datasets such as DEAP, DREAMER, AMIGOS, and EEGEmotions into standardized NumPy arrays. It handles dataset-specific loading, preprocessing, label extraction, and output writing so the rest of the training code can work with a consistent format.

### `cross_val.py`

This module implements model evaluation workflows, including cross-validation strategies and metric computation. It is used to train models, evaluate them under different splits, and collect performance results for experiments.

## Purpose

These modules are intended for building, training, and evaluating EEG deep learning models within the EEGProc project.
