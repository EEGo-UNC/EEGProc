"""Run one deterministic counterfactual example from a completed joint-v2 run.

The runner:

1. reads ``training_config.json`` and ``selected_config.json``;
2. rebuilds the selected joint VAE + BiLSTM + VC architecture;
3. loads the trained values from ``final_model.keras``;
4. recreates the DREAMER windows used during training;
5. selects one EEG window and chooses a target class;
6. calls ``CounterfactualOptimizer.optimize``;
7. saves the original and counterfactual arrays to NPZ and metadata to JSON.

The counterfactual optimizer is expected to expose this interface::

    optimizer = CounterfactualOptimizer(
        model=model,
        learning_rate=1e-2,
        max_steps=500,
        validity_weight=1.0,
        signal_proximity_weight=0.10,
        target_probability=0.80,
        seed=42,
        verbose=1,
    )

    result = optimizer.optimize(
        inputs=x,
        target_class=target_class,
    )

``result`` should be a dictionary. TensorFlow tensors and NumPy arrays are
stored automatically.

Run from the EEGProc repository root::

    python -m src.eegproc.model_explainability.run_counterfactuals \
        --run-dir runs/joint_v2_dreamer_arousal_cnn1d_20260720_102820 \
        --sample-index 0
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


def _build_selected_model(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    training_config: dict[str, Any],
    selected_config: dict[str, Any],
) -> tf.keras.Model:
    """Rebuild exactly the architecture selected by the training run."""
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

    labels_flat = labels
    if labels_flat.ndim == 2 and labels_flat.shape[-1] > 1:
        labels_flat = np.argmax(labels_flat, axis=-1)
    labels_flat = labels_flat.reshape(-1)
    n_classes = int(np.max(labels_flat)) + 1

    model = build_joint_autoencoder_variational_classifier_v2(
        input_shape=tuple(map(int, features.shape[1:])),
        n_classes=n_classes,
        encoder_type=str(
            _pick(
                "encoder_type",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default="cnn1d",
            )
        ),
        n_channels=int(
            _pick(
                "n_channels",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=features.shape[-1],
            )
        ),
        n_bands=_pick(
            "n_bands",
            selected_config=selected_config,
            training_config=training_config,
            model_kwargs=model_kwargs,
            default=None,
        ),
        learning_rate=float(
            _pick(
                "learning_rate",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=1e-3,
            )
        ),
        ae_loss_weight=float(
            _pick(
                "ae_loss_weight",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=0.5,
            )
        ),
        vc_loss_weight=float(
            _pick(
                "vc_loss_weight",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=0.5,
            )
        ),
        vae_beta=float(
            _pick(
                "vae_beta",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=1.0,
            )
        ),
        vc_alpha=float(
            _pick(
                "vc_alpha",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=1.0,
            )
        ),
        vc_beta=float(
            _pick(
                "vc_beta",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=1.0,
            )
        ),
        vc_gamma=float(
            _pick(
                "vc_gamma",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=0.0,
            )
        ),
        vc_lambda=float(
            _pick(
                "vc_lambda",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=0.0,
            )
        ),
        update_discriminator=bool(
            _pick(
                "update_discriminator",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=False,
            )
        ),
        bilstm_units=int(
            _pick(
                "bilstm_units",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=64,
            )
        ),
        n_bilstm_layers=int(
            _pick(
                "bilstm_layers",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=2,
                aliases=("n_bilstm_layers",),
            )
        ),
        bilstm_dropout=float(
            _pick(
                "bilstm_dropout",
                selected_config=selected_config,
                training_config=training_config,
                model_kwargs=model_kwargs,
                default=0.10,
            )
        ),
        bilstm_kwargs=bilstm_kwargs,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        classifier_kwargs=classifier_kwargs,
    )

    dummy = tf.zeros((1, *features.shape[1:]), dtype=tf.float32)
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


def _select_index(
    args: argparse.Namespace,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
) -> int:
    if args.subject_id is None and args.trial_id is None:
        if not 0 <= args.sample_index < len(subject_ids):
            raise IndexError(
                f"sample_index must be in [0, {len(subject_ids) - 1}]."
            )
        return int(args.sample_index)

    if args.subject_id is None or args.trial_id is None:
        raise ValueError("--subject-id and --trial-id must be used together.")

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
    return int(matches[args.window_in_trial])


def _model_outputs(
    model: tf.keras.Model,
    inputs: tf.Tensor,
) -> dict[str, tf.Tensor]:
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
        description="Run one deterministic latent counterfactual."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)

    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--subject-id", type=int, default=None)
    parser.add_argument("--trial-id", type=int, default=None)
    parser.add_argument("--window-in-trial", type=int, default=0)
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
    model = _build_selected_model(
        features=features,
        labels=labels,
        training_config=training_config,
        selected_config=selected_config,
    )
    checkpoint_name = _load_final_checkpoint(model, run_dir)

    # This freezes model variables but still permits gradients with respect to
    # the latent counterfactual variable inside CounterfactualOptimizer.
    model.trainable = False

    selected_index = _select_index(args, subject_ids, trial_ids)
    x_numpy = features[selected_index : selected_index + 1]
    x = tf.convert_to_tensor(x_numpy, dtype=tf.float32)

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
        "sample=%d subject=%s trial=%s true=%s",
        selected_index,
        subject_ids[selected_index],
        trial_ids[selected_index],
        np.asarray(labels[selected_index]).tolist(),
    )
    LOGGER.info(
        "probabilities=%s predicted=%d target=%d",
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
        seed=args.seed,
        verbose=args.verbose,
    )
    result = optimizer.optimize(
        inputs=x,
        target_class=target_class,
    )
    if not isinstance(result, dict):
        raise TypeError(
            "CounterfactualOptimizer.optimize must return a dictionary."
        )
    result = _to_numpy(result)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "counterfactuals"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"sample_{selected_index:05d}"
        f"_subject_{subject_ids[selected_index]}"
        f"_trial_{trial_ids[selected_index]}"
        f"_to_class_{target_class}"
    )
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    if not args.overwrite and (npz_path.exists() or json_path.exists()):
        raise FileExistsError(
            f"Output already exists for {stem}; pass --overwrite to replace it."
        )

    arrays: dict[str, np.ndarray] = {
        "input_eeg": x_numpy,
        "original_probabilities": original_probabilities,
        "original_z_mean": original["z_mean"].numpy(),
        "original_z_log_var": original["z_log_var"].numpy(),
        "original_reconstruction": original["reconstruction"].numpy(),
    }
    _flatten_arrays(result, arrays, "result")
    np.savez_compressed(npz_path, **arrays)

    summary = {
        "run_dir": run_dir,
        "checkpoint": checkpoint_name,
        "sample_index": selected_index,
        "subject_id": subject_ids[selected_index],
        "trial_id": trial_ids[selected_index],
        "true_label": labels[selected_index],
        "predicted_class": predicted_class,
        "target_class": target_class,
        "original_probabilities": original_probabilities,
        "optimizer_config": {
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "validity_weight": args.validity_weight,
            "signal_proximity_weight": args.signal_proximity_weight,
            "target_probability": args.target_probability,
            "seed": args.seed,
        },
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
