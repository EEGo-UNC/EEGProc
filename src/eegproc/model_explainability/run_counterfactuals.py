"""Run deterministic window- or subject-trial EEG counterfactuals.

The runner rebuilds a completed joint-v2 model, recreates the training windows,
and follows the model's saved ``classification_level``. In trial mode it selects
all ordered windows belonging to one ``(subject_id, trial_id)`` pair, applies
the same padding/cropping layout used in training, optimizes the complete latent
trial, decodes every window independently, and evaluates one BiLSTM trial
prediction. Optimization stops at the first requested target probability by
default and saves explicit trial-level prediction metrics.

Example
-------
From the EEGProc repository root::

    python -m src.eegproc.model_explainability.run_counterfactuals \
      --run-dir runs/my_trial_model \
      --subject-id 1 --trial-id 3 \
      --learning-rate 0.005 --max-steps 200 \
      --target-probability 0.80 --overwrite

    python -m src.eegproc.model_explainability.run_counterfactuals \ 
        --raw-eeg-npy datasets/dreamer_eeg.npy \
        --raw-labels-npy datasets/dreamer_labels.npy \
        --run-dir runs/AAAI_run10_GCN/GCN/dreamer_arousal_vaevc_gcn_20260724_160700 \
        --subject-id 1 --trial-id 3 \
        --learning-rate 0.005 --max-steps 20 \
        --target-probability 0.70 --overwrite

"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))



from eegproc.deep_learning.joint_architectures.joint_v2_autoencoder_vc_train import (
    build_joint_autoencoder_variational_classifier_v2,
    load_joint_v2_training_data,
)

from eegproc.model_explainability.counterfactual_optimizer import (
    CounterfactualOptimizer,
)


LOGGER = logging.getLogger("eegproc.run_counterfactuals")

DREAMER_CHANNELS = np.asarray(
    [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
    ],
    dtype=str,
)


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "joint_v2_dreamer_arousal_cnn1d_20260720_102820"
)


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return {}

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _pick(
    key: str,
    *,
    selected_config: dict[str, Any],
    training_config: dict[str, Any],
    model_kwargs: dict[str, Any],
    default: Any,
    aliases: tuple[str, ...] = (),
) -> Any:
    for source in (selected_config, training_config, model_kwargs):
        for candidate in (key, *aliases):
            if candidate in source and source[candidate] is not None:
                return source[candidate]
    return default


def _merge_dict_setting(
    key: str,
    training_config: dict[str, Any],
    selected_config: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(training_config.get(key, {}) or {})
    merged.update(selected_config.get(key, {}) or {})
    return merged


def _load_dataset(
    args: argparse.Namespace,
    training_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use the same public data loader as the training script."""
    loader_signature = inspect.signature(load_joint_v2_training_data)

    raw_eeg = args.raw_eeg_npy or training_config.get("raw_eeg_npy")
    raw_labels = args.raw_labels_npy or training_config.get("raw_labels_npy")

    candidate_kwargs: dict[str, Any] = {
        "eeg_path": raw_eeg,
        "labels_path": raw_labels,
        "label_dimension": (
            args.label_dimension
            or training_config.get("label_dimension")
            or "arousal"
        ),
        "window_size_sec": (
            args.window_sec
            if args.window_sec is not None
            else training_config.get(
                "window_size_sec",
                training_config.get("window_sec", 4.0),
            )
        ),
        "overlap": (
            args.window_overlap
            if args.window_overlap is not None
            else training_config.get(
                "overlap",
                training_config.get("window_overlap", 0.0),
            )
        ),
        "fs": training_config.get("fs", 128.0),
        "median_label": training_config.get("median_label", 3.0),
        "zscore": training_config.get("zscore", True),
        "window_normalization": training_config.get(
            "window_normalization",
            "global_rms",
        ),
        "label_threshold_mode": training_config.get(
            "label_threshold_mode",
            "global",
        ),
        "dataset": training_config.get("dataset", "dreamer"),
    }

    # Preserve the loader's built-in DREAMER path defaults when no override is
    # available, and tolerate older loader signatures.
    loader_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if value is not None and key in loader_signature.parameters
    }

    LOGGER.info("Loading dataset with %s", loader_kwargs)
    arrays = load_joint_v2_training_data(**loader_kwargs)
    if len(arrays) != 4:
        raise ValueError(
            "load_joint_v2_training_data must return features, labels, "
            "subject IDs, and trial IDs."
        )

    features, labels, subject_ids, trial_ids = arrays
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    subject_ids = np.asarray(subject_ids)
    trial_ids = np.asarray(trial_ids)

    if features.ndim != 3:
        raise ValueError(
            "Expected features shaped (n_windows, timesteps, n_features), "
            f"got {features.shape}."
        )
    if not (
        len(features)
        == len(labels)
        == len(subject_ids)
        == len(trial_ids)
    ):
        raise ValueError("The loaded dataset arrays are not aligned.")

    return features, labels, subject_ids, trial_ids


def _resolve_classification_level(
    training_config: dict[str, Any],
    selected_config: dict[str, Any],
) -> str:
    model_kwargs = dict(training_config.get("model_kwargs", {}) or {})
    level = str(
        _pick(
            "classification_level",
            selected_config=selected_config,
            training_config=training_config,
            model_kwargs=model_kwargs,
            default="window",
        )
    ).lower()
    if level not in {"window", "trial"}:
        raise ValueError(
            "classification_level must be 'window' or 'trial'; "
            f"received {level!r}."
        )
    return level


def _resolve_trial_layout(
    *,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    training_config: dict[str, Any],
    selected_config: dict[str, Any],
) -> tuple[int | None, str]:
    """Return the training-time padded window count and crop rule."""
    if len(subject_ids) != len(trial_ids):
        raise ValueError("Subject and trial arrays must align.")

    model_kwargs = dict(training_config.get("model_kwargs", {}) or {})
    crop = str(
        _pick(
            "trial_crop",
            selected_config=selected_config,
            training_config=training_config,
            model_kwargs=model_kwargs,
            default="center",
        )
    ).lower()
    if crop not in {"start", "center", "end"}:
        raise ValueError("trial_crop must be start, center, or end.")

    configured = _pick(
        "n_windows_per_trial",
        selected_config=selected_config,
        training_config=training_config,
        model_kwargs=model_kwargs,
        default=None,
        aliases=("trial_max_windows",),
    )
    if configured is not None:
        configured = int(configured)
        if configured < 1:
            raise ValueError("n_windows_per_trial must be at least 1.")
        return configured, crop

    counts: dict[tuple[Any, Any], int] = {}
    for subject_id, trial_id in zip(subject_ids, trial_ids):
        key = (
            subject_id.item() if isinstance(subject_id, np.generic) else subject_id,
            trial_id.item() if isinstance(trial_id, np.generic) else trial_id,
        )
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        raise ValueError("No subject-trial groups were found.")
    return max(counts.values()), crop


def _build_selected_model(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    training_config: dict[str, Any],
    selected_config: dict[str, Any],
    classification_level: str,
    n_windows_per_trial: int | None,
) -> tf.keras.Model:
    """Rebuild the selected window- or trial-level joint architecture."""
    model_kwargs = dict(training_config.get("model_kwargs", {}) or {})

    encoder_kwargs = _merge_dict_setting(
        "encoder_kwargs",
        training_config,
        selected_config,
    )
    for key in (
        "t_down",
        "emb_dim",
        "dropout",
        "use_batch_norm",
        "activation",
        "conv_filters",
        "kernel_sizes",
        "pool_after_layers",
        "pool_sizes",
        "spatial_pool_sizes",
        "temporal_pool_sizes",
        "gcn_units",
    ):
        if key in selected_config:
            encoder_kwargs[key] = selected_config[key]

    decoder_kwargs = _merge_dict_setting(
        "decoder_kwargs",
        training_config,
        selected_config,
    )
    classifier_kwargs = _merge_dict_setting(
        "classifier_kwargs",
        training_config,
        selected_config,
    )
    bilstm_kwargs = _merge_dict_setting(
        "bilstm_kwargs",
        training_config,
        selected_config,
    )
    trial_bilstm_kwargs = _merge_dict_setting(
        "trial_bilstm_kwargs",
        training_config,
        selected_config,
    )

    labels_flat = labels
    if labels_flat.ndim == 2 and labels_flat.shape[-1] > 1:
        labels_flat = np.argmax(labels_flat, axis=-1)
    labels_flat = labels_flat.reshape(-1)
    n_classes = int(np.max(labels_flat)) + 1

    def pick(key: str, default: Any, aliases: tuple[str, ...] = ()) -> Any:
        return _pick(
            key,
            selected_config=selected_config,
            training_config=training_config,
            model_kwargs=model_kwargs,
            default=default,
            aliases=aliases,
        )

    builder_kwargs: dict[str, Any] = {
        "input_shape": tuple(map(int, features.shape[1:])),
        "n_classes": n_classes,
        "classification_level": classification_level,
        "n_windows_per_trial": n_windows_per_trial,
        "encoder_type": str(pick("encoder_type", "cnn1d")),
        "n_channels": int(pick("n_channels", features.shape[-1])),
        "n_bands": pick("n_bands", None),
        "learning_rate": float(pick("learning_rate", 1e-3)),
        "optimizer_name": str(pick("optimizer_name", "adamw")),
        "weight_decay": float(pick("weight_decay", 1e-4)),
        "label_smoothing": float(pick("label_smoothing", 0.0)),
        "ae_loss_weight": float(pick("ae_loss_weight", 0.5)),
        "vc_loss_weight": float(pick("vc_loss_weight", 0.5)),
        "vae_beta": float(pick("vae_beta", 1.0)),
        "vc_alpha": float(pick("vc_alpha", 1.0)),
        "vc_beta": float(pick("vc_beta", 1.0)),
        "vc_gamma": float(pick("vc_gamma", 0.0)),
        "vc_lambda": float(pick("vc_lambda", 0.0)),
        "update_discriminator": bool(pick("update_discriminator", False)),
        "use_class_weight": bool(pick("use_class_weight", True)),
        "use_subject_adversarial": bool(
            pick("use_subject_adversarial", False)
        ),
        "n_subject_classes": pick("n_subject_classes", None),
        "subject_adversarial_weight": float(
            pick("subject_adversarial_weight", 0.05)
        ),
        "subject_loss_weight": float(pick("subject_loss_weight", 1.0)),
        "subject_hidden_units": int(pick("subject_hidden_units", 64)),
        "subject_dropout": float(pick("subject_dropout", 0.0)),
        "subject_latent_mode": str(pick("subject_latent_mode", "mean")),
        "subject_mc_samples": int(pick("subject_mc_samples", 5)),
        "use_supcon": bool(pick("use_supcon", False)),
        "supcon_weight": float(pick("supcon_weight", 0.03)),
        "supcon_temperature": float(pick("supcon_temperature", 0.1)),
        "supcon_cross_subject_only": bool(
            pick("supcon_cross_subject_only", True)
        ),
        "bilstm_units": int(pick("bilstm_units", 64)),
        "n_bilstm_layers": int(
            pick("bilstm_layers", 1, aliases=("n_bilstm_layers",))
        ),
        "bilstm_dropout": float(pick("bilstm_dropout", 0.10)),
        "trial_bilstm_units": pick("trial_bilstm_units", None),
        "n_trial_bilstm_layers": pick("n_trial_bilstm_layers", None),
        "trial_bilstm_dropout": pick("trial_bilstm_dropout", None),
        "bilstm_kwargs": bilstm_kwargs,
        "trial_bilstm_kwargs": trial_bilstm_kwargs,
        "encoder_kwargs": encoder_kwargs,
        "decoder_kwargs": decoder_kwargs,
        "classifier_kwargs": classifier_kwargs,
        "classifier_head": str(pick("classifier_head", "variational")),
    }

    signature = inspect.signature(
        build_joint_autoencoder_variational_classifier_v2
    )
    supported_kwargs = {
        key: value
        for key, value in builder_kwargs.items()
        if key in signature.parameters and value is not None
    }
    model = build_joint_autoencoder_variational_classifier_v2(
        **supported_kwargs
    )

    if classification_level == "trial":
        if n_windows_per_trial is None:
            raise ValueError("Trial mode requires n_windows_per_trial.")
        dummy_numpy = np.zeros(
            (1, n_windows_per_trial, *features.shape[1:]),
            dtype=np.float32,
        )
        # The model identifies valid windows from nonzero values.
        dummy_numpy[:, 0] = np.float32(1e-6)
    else:
        dummy_numpy = np.zeros(
            (1, *features.shape[1:]),
            dtype=np.float32,
        )
    dummy = tf.convert_to_tensor(dummy_numpy)
    try:
        model(
            dummy,
            training=False,
            sample_latent=False,
            include_reconstruction=False,
            include_subject_adversarial=False,
        )
    except TypeError:
        try:
            model(dummy, training=False, sample_latent=False)
        except TypeError:
            model(dummy, training=False)

    return model

def _load_final_checkpoint(model: tf.keras.Model, run_dir: Path) -> str:
    """Load trained values from final_model.keras, with an H5 fallback."""
    keras_path = run_dir / "final_model.keras"
    if not keras_path.is_file():
        raise FileNotFoundError(f"Missing final model: {keras_path}")

    try:
        model.load_weights(keras_path)
        LOGGER.info("Loaded model weights from %s", keras_path)
        return "final_model.keras"
    except Exception as keras_error:
        weights_path = run_dir / "final_model.weights.h5"
        if not weights_path.is_file():
            raise RuntimeError(
                "The architecture was rebuilt, but Keras could not load "
                "weights from final_model.keras and final_model.weights.h5 "
                "was not found."
            ) from keras_error

        LOGGER.warning(
            "Could not load weights directly from final_model.keras: %s",
            keras_error,
        )
        model.load_weights(weights_path)
        LOGGER.info("Loaded fallback weights from %s", weights_path)
        return "final_model.weights.h5"


def _as_class_ids(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[-1] > 1:
        return np.argmax(labels, axis=-1).astype(np.int64)
    return labels.reshape(-1).astype(np.int64)


def _trial_crop_start(length: int, target: int, crop: str) -> int:
    if length <= target or crop == "start":
        return 0
    if crop == "end":
        return length - target
    return (length - target) // 2


def _select_counterfactual_sample(
    *,
    args: argparse.Namespace,
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    classification_level: str,
    n_windows_per_trial: int | None,
    trial_crop: str,
) -> dict[str, Any]:
    """Select one window or one complete ordered subject-trial."""
    class_ids = _as_class_ids(labels)

    if classification_level == "window":
        if args.subject_id is None and args.trial_id is None:
            if not 0 <= args.sample_index < len(features):
                raise IndexError(
                    f"sample_index must be in [0, {len(features) - 1}]."
                )
            index = int(args.sample_index)
        else:
            if args.subject_id is None or args.trial_id is None:
                raise ValueError(
                    "--subject-id and --trial-id must be used together."
                )
            matches = np.flatnonzero(
                (subject_ids == args.subject_id)
                & (trial_ids == args.trial_id)
            )
            if len(matches) == 0:
                raise ValueError(
                    f"No windows found for subject {args.subject_id}, "
                    f"trial {args.trial_id}."
                )
            if not 0 <= args.window_in_trial < len(matches):
                raise IndexError(
                    f"window_in_trial must be in [0, {len(matches) - 1}] "
                    "for the selected trial."
                )
            index = int(matches[args.window_in_trial])

        return {
            "inputs": features[index : index + 1],
            "window_mask": None,
            "source_indices": np.asarray([index], dtype=np.int64),
            "subject_id": subject_ids[index],
            "trial_id": trial_ids[index],
            "true_class": int(class_ids[index]),
            "true_label": labels[index],
            "n_valid_windows": 1,
            "selection_index": index,
        }

    if n_windows_per_trial is None:
        raise ValueError("Trial mode requires n_windows_per_trial.")

    if args.subject_id is None and args.trial_id is None:
        if not 0 <= args.sample_index < len(features):
            raise IndexError(
                f"sample_index must be in [0, {len(features) - 1}]."
            )
        seed_index = int(args.sample_index)
        subject_id = subject_ids[seed_index]
        trial_id = trial_ids[seed_index]
    else:
        if args.subject_id is None or args.trial_id is None:
            raise ValueError(
                "--subject-id and --trial-id must be used together."
            )
        subject_id = args.subject_id
        trial_id = args.trial_id
        seed_index = -1

    matches = np.flatnonzero(
        (subject_ids == subject_id) & (trial_ids == trial_id)
    )
    if len(matches) == 0:
        raise ValueError(
            f"No windows found for subject {subject_id}, trial {trial_id}."
        )
    unique_classes = np.unique(class_ids[matches])
    if len(unique_classes) != 1:
        raise ValueError(
            "All windows in a subject-trial must share one class; "
            f"found {unique_classes.tolist()}."
        )

    start = _trial_crop_start(
        length=len(matches),
        target=n_windows_per_trial,
        crop=trial_crop,
    )
    selected = matches[start : start + n_windows_per_trial]
    kept = len(selected)

    trial_tensor = np.zeros(
        (1, n_windows_per_trial, features.shape[1], features.shape[2]),
        dtype=np.float32,
    )
    trial_mask = np.zeros((1, n_windows_per_trial), dtype=bool)
    source_indices = np.full(n_windows_per_trial, -1, dtype=np.int64)
    trial_tensor[0, :kept] = features[selected]
    trial_mask[0, :kept] = True
    source_indices[:kept] = selected

    return {
        "inputs": trial_tensor,
        "window_mask": trial_mask,
        "source_indices": source_indices,
        "subject_id": subject_id,
        "trial_id": trial_id,
        "true_class": int(unique_classes[0]),
        "true_label": labels[matches[0]],
        "n_valid_windows": kept,
        "selection_index": seed_index,
        "original_trial_window_count": len(matches),
        "trial_crop_start": start,
    }

def _model_outputs(
    model: tf.keras.Model,
    inputs: tf.Tensor,
) -> dict[str, tf.Tensor]:
    try:
        outputs = model(
            inputs,
            training=False,
            sample_latent=False,
            include_reconstruction=True,
            include_subject_adversarial=False,
        )
    except TypeError:
        try:
            outputs = model(
                inputs,
                training=False,
                sample_latent=False,
            )
        except TypeError:
            outputs = model(inputs, training=False)

    if not isinstance(outputs, dict):
        raise TypeError("The joint model must return a dictionary.")
    for key in ("logits", "z_mean", "z_log_var", "reconstruction"):
        if key not in outputs:
            raise KeyError(f"The joint model output is missing {key!r}.")
    return outputs


def _target_class(
    probabilities: np.ndarray,
    requested_target: int | None,
) -> tuple[int, int]:
    predicted = int(np.argmax(probabilities))
    n_classes = probabilities.shape[-1]

    if requested_target is not None:
        if not 0 <= requested_target < n_classes:
            raise ValueError(
                f"target_class must be in [0, {n_classes - 1}]."
            )
        if requested_target == predicted:
            raise ValueError(
                "The requested target is already the predicted class."
            )
        return predicted, int(requested_target)

    if n_classes == 2:
        return predicted, 1 - predicted

    order = np.argsort(probabilities)[::-1]
    return predicted, int(order[1])


def _to_numpy(value: Any) -> Any:
    if tf.is_tensor(value):
        return value.numpy()
    if isinstance(value, dict):
        return {key: _to_numpy(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_numpy(child) for child in value]
    return value


def _flatten_arrays(
    value: Any,
    output: dict[str, np.ndarray],
    prefix: str,
) -> None:
    value = _to_numpy(value)

    if isinstance(value, np.ndarray):
        output[prefix] = value
    elif isinstance(value, np.generic):
        output[prefix] = np.asarray(value.item())
    elif isinstance(value, (bool, int, float)):
        output[prefix] = np.asarray(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            _flatten_arrays(child, output, f"{prefix}__{key}")
    elif isinstance(value, (list, tuple)):
        try:
            output[prefix] = np.asarray(value)
        except (TypeError, ValueError):
            for index, child in enumerate(value):
                _flatten_arrays(child, output, f"{prefix}__{index}")


def _jsonable(value: Any) -> Any:
    value = _to_numpy(value)

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size <= 32:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return repr(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one deterministic window- or subject-trial counterfactual."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help=(
            "Window index. In trial mode, its subject and trial identify the "
            "complete session unless --subject-id/--trial-id are supplied."
        ),
    )
    parser.add_argument("--subject-id", type=int, default=None)
    parser.add_argument("--trial-id", type=int, default=None)
    parser.add_argument(
        "--window-in-trial",
        type=int,
        default=0,
        help="Used only when the trained model classifies individual windows.",
    )
    parser.add_argument("--target-class", type=int, default=None)

    parser.add_argument("--raw-eeg-npy", type=Path, default=None)
    parser.add_argument("--raw-labels-npy", type=Path, default=None)
    parser.add_argument(
        "--label-dimension",
        choices=("valence", "arousal"),
        default=None,
    )
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--window-overlap", type=float, default=None)

    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--validity-weight", type=float, default=1.0)
    parser.add_argument("--signal-proximity-weight", type=float, default=0.10)
    parser.add_argument("--target-probability", type=float, default=0.80)
    parser.add_argument(
        "--signal-metric",
        choices=("mse", "mae", "rmse"),
        default="mse",
    )
    parser.add_argument(
        "--stop-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stop at the first iterate whose trial/window target probability "
            "reaches --target-probability (default: enabled)."
        ),
    )
    parser.add_argument(
        "--feature-log-interval",
        type=int,
        default=1,
        help=(
            "Save reconstructed counterfactual features every N steps. "
            "Use 0 to disable feature logging."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    training_config = _read_json(run_dir / "training_config.json")
    selected_config = _read_json(
        run_dir / "selected_config.json",
        required=False,
    )

    features, labels, subject_ids, trial_ids = _load_dataset(
        args,
        training_config,
    )
    classification_level = _resolve_classification_level(
        training_config,
        selected_config,
    )
    if classification_level == "trial":
        n_windows_per_trial, trial_crop = _resolve_trial_layout(
            subject_ids=subject_ids,
            trial_ids=trial_ids,
            training_config=training_config,
            selected_config=selected_config,
        )
    else:
        n_windows_per_trial = None
        trial_crop = "center"

    model = _build_selected_model(
        features=features,
        labels=labels,
        training_config=training_config,
        selected_config=selected_config,
        classification_level=classification_level,
        n_windows_per_trial=n_windows_per_trial,
    )
    checkpoint_name = _load_final_checkpoint(model, run_dir)

    # Freeze trained parameters while retaining gradients with respect to the
    # new counterfactual latent variable.
    model.trainable = False

    selection = _select_counterfactual_sample(
        args=args,
        features=features,
        labels=labels,
        subject_ids=subject_ids,
        trial_ids=trial_ids,
        classification_level=classification_level,
        n_windows_per_trial=n_windows_per_trial,
        trial_crop=trial_crop,
    )
    x_numpy = np.asarray(selection["inputs"], dtype=np.float32)
    x = tf.convert_to_tensor(x_numpy, dtype=tf.float32)
    window_mask_numpy = selection["window_mask"]
    window_mask = (
        None
        if window_mask_numpy is None
        else tf.convert_to_tensor(window_mask_numpy, dtype=tf.bool)
    )

    original = _model_outputs(model, x)
    original_probabilities = tf.nn.softmax(
        original["logits"],
        axis=-1,
    ).numpy()[0]
    predicted_class, target_class = _target_class(
        original_probabilities,
        args.target_class,
    )

    LOGGER.info(
        "level=%s subject=%s trial=%s true=%d valid_windows=%d",
        classification_level,
        selection["subject_id"],
        selection["trial_id"],
        selection["true_class"],
        selection["n_valid_windows"],
    )
    LOGGER.info(
        "original_%s_probabilities=%s predicted=%d target=%d",
        classification_level,
        np.array2string(original_probabilities, precision=5),
        predicted_class,
        target_class,
    )

    optimizer = CounterfactualOptimizer(
        model=model,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        validity_weight=args.validity_weight,
        signal_proximity_weight=args.signal_proximity_weight,
        target_probability=args.target_probability,
        feature_log_interval=args.feature_log_interval,
        signal_metric=args.signal_metric,
        stop_on_success=args.stop_on_success,
        seed=args.seed,
        verbose=args.verbose,
    )
    result = optimizer.optimize(
        inputs=x,
        target_class=target_class,
        window_mask=window_mask,
        true_class=selection["true_class"],
    )
    if not isinstance(result, dict):
        raise TypeError(
            "CounterfactualOptimizer.optimize must return a dictionary."
        )
    result = _to_numpy(result)

    LOGGER.info(
        "%s_metrics=%s",
        classification_level,
        json.dumps(
            _jsonable(result.get("classification_metrics", {})),
            sort_keys=True,
        ),
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "counterfactuals"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if classification_level == "trial":
        stem = (
            f"subject_{selection['subject_id']}"
            f"_trial_{selection['trial_id']}"
            f"_to_class_{target_class}"
        )
    else:
        stem = (
            f"sample_{selection['selection_index']:05d}"
            f"_subject_{selection['subject_id']}"
            f"_trial_{selection['trial_id']}"
            f"_to_class_{target_class}"
        )
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    if not args.overwrite and (npz_path.exists() or json_path.exists()):
        raise FileExistsError(
            f"Output already exists for {stem}; pass --overwrite to replace it."
        )

    sampling_rate_hz = float(training_config.get("fs", 128.0))
    channel_names = (
        DREAMER_CHANNELS
        if x_numpy.shape[-1] == len(DREAMER_CHANNELS)
        else np.asarray(
            [f"feature_{index}" for index in range(x_numpy.shape[-1])],
            dtype=str,
        )
    )
    timestep_count = x_numpy.shape[-2]

    arrays: dict[str, np.ndarray] = {
        "input_eeg": x_numpy,
        "time_seconds": np.arange(
            timestep_count,
            dtype=np.float32,
        ) / sampling_rate_hz,
        "channel_names": channel_names,
        "sampling_rate_hz": np.asarray(
            sampling_rate_hz,
            dtype=np.float32,
        ),
        "source_window_indices": np.asarray(
            selection["source_indices"],
            dtype=np.int64,
        ),
        "original_probabilities": original_probabilities,
        "original_z_mean": original["z_mean"].numpy(),
        "original_z_log_var": original["z_log_var"].numpy(),
        "original_reconstruction": original["reconstruction"].numpy(),
    }
    if window_mask_numpy is not None:
        arrays["window_mask"] = np.asarray(window_mask_numpy, dtype=bool)
    _flatten_arrays(result, arrays, "result")
    np.savez_compressed(npz_path, **arrays)

    summary = {
        "run_dir": run_dir,
        "checkpoint": checkpoint_name,
        "classification_level": classification_level,
        "selection_index": selection["selection_index"],
        "subject_id": selection["subject_id"],
        "trial_id": selection["trial_id"],
        "true_label": selection["true_label"],
        "true_class": selection["true_class"],
        "n_valid_windows": selection["n_valid_windows"],
        "n_windows_per_trial": n_windows_per_trial,
        "trial_crop": trial_crop,
        "source_window_indices": selection["source_indices"],
        "predicted_class": predicted_class,
        "target_class": target_class,
        "original_probabilities": original_probabilities,
        "sampling_rate_hz": sampling_rate_hz,
        "channel_names": channel_names,
        "optimizer_config": {
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "validity_weight": args.validity_weight,
            "signal_proximity_weight": args.signal_proximity_weight,
            "target_probability": args.target_probability,
            "signal_metric": args.signal_metric,
            "stop_on_success": args.stop_on_success,
            "feature_log_interval": args.feature_log_interval,
            "seed": args.seed,
        },
        "trial_level_metrics": (
            result.get("classification_metrics", {})
            if classification_level == "trial"
            else None
        ),
        "result": result,
        "arrays_file": npz_path,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2)

    LOGGER.info("Saved %s", npz_path)
    LOGGER.info("Saved %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
