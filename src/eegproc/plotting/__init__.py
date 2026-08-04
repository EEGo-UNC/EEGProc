from importlib import import_module

__all__ = [
    "plot_eeg_features",
    "ResultsTables",
    "load_results",
    "class_probability_columns",
    "plot_confusion_matrix",
    "plot_metric_summary",
    "plot_per_subject_accuracy",
    "plot_subject_task_heatmap",
    "plot_hyperparameter_sweep",
    "plot_reliability",
    "plot_uncertainty",
    "plot_model_comparison",
    "build_report",
]


def __getattr__(name: str):
    if name == "plot_eeg_features":
        return import_module(".plots", __name__).plot_eeg_features
    if name in {"ResultsTables", "load_results", "class_probability_columns"}:
        module = import_module(".results_io", __name__)
        return getattr(module, name)
    if name in {
        "plot_confusion_matrix",
        "plot_metric_summary",
        "plot_per_subject_accuracy",
        "plot_subject_task_heatmap",
        "plot_hyperparameter_sweep",
        "plot_reliability",
        "plot_uncertainty",
        "plot_model_comparison",
    }:
        module = import_module(".result_figures", __name__)
        return getattr(module, name)
    if name == "build_report":
        module = import_module(".report", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")