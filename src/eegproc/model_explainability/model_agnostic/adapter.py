"""Model and dataset contracts for architecture-agnostic counterfactuals.

Adapters isolate model-specific encoding, classification, decoding, and
optional validity metrics from the optimizer.  External projects can provide
an adapter factory without changing EEGProc; factories are addressed as
``package.module:function`` and receive ``model_path``, ``config``, and one
sample trial.
"""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tensorflow as tf


@dataclass
class TrialDataset:
    """Prepared trials and optional metadata needed to restore source units."""

    features: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    labels: np.ndarray | None = None
    normalization_offset: np.ndarray | None = None
    normalization_scale: np.ndarray | None = None
    channel_names: tuple[str, ...] | None = None
    band_names: tuple[str, ...] | None = None
    channel_positions: np.ndarray | None = None
    feature_order: str | None = None
    signal_unit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CounterfactualAdapter(ABC):
    """Minimal differentiable interface consumed by the generic optimizer."""

    name = "custom"
    optimization_space = "latent"
    default_state_weight = 0.1
    default_signal_weight = 0.1

    @abstractmethod
    def initial_state(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return the tensor that should be optimized for one input trial."""

    @abstractmethod
    def logits_from_state(self, state: tf.Tensor) -> tf.Tensor:
        """Return unnormalized class logits for the optimizable state."""

    @abstractmethod
    def reconstruct(
        self, state: tf.Tensor, reference_input: tf.Tensor
    ) -> Mapping[str, tf.Tensor]:
        """Map state to one or more input-shaped differentiable signals."""

    @abstractmethod
    def logits_from_input(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return class logits when an input-shaped reconstruction is evaluated."""

    def constraint(self, name: str, signal: tf.Tensor) -> tf.Tensor:
        """Return one scalar validity metric; subclasses opt in by name."""

        raise KeyError(f"Adapter {self.name!r} does not provide constraint {name!r}.")

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "optimization_space": self.optimization_space,
        }


def _resolve_object(spec: str):
    if ":" not in spec:
        raise ValueError("Object specifications must use 'package.module:name'.")
    module_name, object_name = spec.rsplit(":", 1)
    if not module_name or not object_name:
        raise ValueError("Object specifications must use 'package.module:name'.")
    return getattr(importlib.import_module(module_name), object_name)


def load_json_mapping(value: str | None) -> dict[str, Any]:
    """Load a JSON object from inline text or a JSON file path."""

    if value is None:
        return {}
    candidate = Path(value)
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Configuration JSON must contain an object.")
    return payload


def create_adapter(
    factory_spec: str,
    *,
    model_path: Path,
    config: Mapping[str, Any],
    sample_input: np.ndarray,
) -> CounterfactualAdapter:
    factory = _resolve_object(factory_spec)
    adapter = factory(
        model_path=Path(model_path),
        config=dict(config),
        sample_input=np.asarray(sample_input),
    )
    if not isinstance(adapter, CounterfactualAdapter):
        raise TypeError(
            f"Adapter factory {factory_spec!r} returned {type(adapter).__name__}, "
            "not CounterfactualAdapter."
        )
    return adapter


def load_trial_dataset(
    npz_path: Path | None,
    *,
    loader_spec: str | None = None,
    loader_config: Mapping[str, Any] | None = None,
) -> TrialDataset:
    """Load a standard NPZ or delegate dataset construction to a plug-in."""

    if loader_spec is not None:
        loaded = _resolve_object(loader_spec)(dict(loader_config or {}))
        if not isinstance(loaded, TrialDataset):
            raise TypeError(
                f"Dataset loader {loader_spec!r} must return TrialDataset."
            )
        return validate_trial_dataset(loaded)
    if npz_path is None:
        raise ValueError("Provide either an NPZ path or a dataset loader.")

    with np.load(Path(npz_path), allow_pickle=False) as data:
        required = {"features", "subject_ids", "trial_ids"}
        if not required.issubset(data.files):
            raise ValueError(f"Prepared NPZ requires keys {sorted(required)}.")

        def optional_array(name):
            return np.asarray(data[name]) if name in data.files else None

        def optional_strings(name):
            if name not in data.files:
                return None
            return tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in np.asarray(data[name]).reshape(-1)
            )

        def optional_scalar(name):
            if name not in data.files:
                return None
            values = np.asarray(data[name]).reshape(-1)
            return None if not len(values) else str(values[0])

        dataset = TrialDataset(
            features=np.asarray(data["features"]),
            subject_ids=np.asarray(data["subject_ids"]),
            trial_ids=np.asarray(data["trial_ids"]),
            labels=optional_array("labels"),
            normalization_offset=optional_array("normalization_offset"),
            normalization_scale=optional_array("normalization_scale"),
            channel_names=optional_strings("channel_names"),
            band_names=optional_strings("band_names"),
            channel_positions=optional_array("channel_positions"),
            feature_order=optional_scalar("feature_order"),
            signal_unit=optional_scalar("signal_unit"),
        )
    return validate_trial_dataset(dataset)


def validate_trial_dataset(dataset: TrialDataset) -> TrialDataset:
    x = np.asarray(dataset.features, dtype=np.float32)
    if x.ndim < 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("features must be finite, nonempty, and include a trial axis.")
    n_trials = len(x)
    for name in ("subject_ids", "trial_ids", "labels"):
        values = getattr(dataset, name)
        if values is None:
            continue
        values = np.asarray(values)
        if values.shape != (n_trials,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must contain one integer per trial.")
        setattr(dataset, name, values)
    if len(set(zip(dataset.subject_ids.tolist(), dataset.trial_ids.tolist()))) != n_trials:
        raise ValueError("Each (subject_id, trial_id) pair must be unique.")
    for name in ("normalization_offset", "normalization_scale"):
        values = getattr(dataset, name)
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] != n_trials or not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite and start with the trial axis.")
        try:
            np.broadcast_shapes(values.shape[1:], x.shape[1:])
        except ValueError as error:
            raise ValueError(
                f"{name} shape {values.shape} cannot broadcast to features {x.shape}."
            ) from error
        if name == "normalization_scale" and np.any(values <= 0):
            raise ValueError("normalization_scale values must be positive.")
        setattr(dataset, name, values)
    if dataset.channel_positions is not None:
        positions = np.asarray(dataset.channel_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
            raise ValueError("channel_positions must be finite and shaped (channels, 2).")
        dataset.channel_positions = positions
    dataset.features = x
    return dataset


def _model_output(model, inputs, *, output_key: str | None):
    output = model(inputs, training=False)
    if isinstance(output, Mapping):
        if output_key is None:
            raise ValueError("Model returned a mapping; set adapter_config.output_key.")
        output = output[output_key]
    return tf.cast(tf.convert_to_tensor(output), tf.float32)


def _as_logits(output: tf.Tensor, *, output_kind: str) -> tf.Tensor:
    if output_kind == "logits":
        return output
    if output_kind != "probabilities":
        raise ValueError("output_kind must be 'logits' or 'probabilities'.")
    probabilities = tf.clip_by_value(output, 1e-7, 1.0)
    probabilities = probabilities / tf.reduce_sum(probabilities, axis=-1, keepdims=True)
    return tf.math.log(probabilities)


class KerasInputAdapter(CounterfactualAdapter):
    """Input-space gradients for differentiable Keras models without decoders."""

    name = "keras-input"
    optimization_space = "input"
    default_state_weight = 0.1
    default_signal_weight = 0.0

    def __init__(self, model, *, output_kind: str, output_key: str | None):
        self.model = model
        self.output_kind = output_kind
        self.output_key = output_key

    def initial_state(self, inputs):
        return tf.cast(inputs, tf.float32)

    def logits_from_state(self, state):
        return self.logits_from_input(state)

    def reconstruct(self, state, reference_input):
        del reference_input
        return {"input": tf.cast(state, tf.float32)}

    def logits_from_input(self, inputs):
        return _as_logits(
            _model_output(self.model, inputs, output_key=self.output_key),
            output_kind=self.output_kind,
        )

    def metadata(self):
        return {
            **super().metadata(),
            "output_kind": self.output_kind,
            "output_key": self.output_key,
        }


def create_keras_input_adapter(*, model_path, config, sample_input):
    """Built-in factory for arbitrary differentiable input-space Keras models."""

    for module_name in config.get("registration_modules", []):
        importlib.import_module(str(module_name))
    model = tf.keras.models.load_model(Path(model_path), compile=False, safe_mode=True)
    adapter = KerasInputAdapter(
        model,
        output_kind=str(config.get("output_kind", "logits")),
        output_key=config.get("output_key"),
    )
    del sample_input
    return adapter
