"""Plotting helpers for EEG feature frames."""

from .plots import plot_eeg_features
from .result_figures import plot_metric_summary, plot_model_comparison, plot_per_subject_accuracy

__all__ = ["plot_eeg_features", "plot_metric_summary", "plot_model_comparison", "plot_per_subject_accuracy"]
