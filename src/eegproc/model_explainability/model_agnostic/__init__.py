"""Counterfactual optimization for differentiable models through adapters."""

from importlib import import_module

__all__ = [
    "CounterfactualAdapter", "TrialDataset", "KerasInputAdapter",
    "create_keras_input_adapter", "load_trial_dataset",
    "ModelAgnosticCounterfactualOptimizer",
]


def __getattr__(name):
    if name in __all__:
        module = ".optimizer" if name == "ModelAgnosticCounterfactualOptimizer" else ".adapter"
        return getattr(import_module(module, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
