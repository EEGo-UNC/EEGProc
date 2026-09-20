"""SIC adapter and DREAMER loader for the model-agnostic counterfactual path."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import tensorflow as tf

from .adapter import CounterfactualAdapter, TrialDataset
from ..counterfactuals.loss import CounterfactualLoss


class SICCounterfactualAdapter(CounterfactualAdapter):
    """Expose SIC v11/v15 internals through the generic adapter contract."""

    name = "sic"
    optimization_space = "latent"
    default_state_weight = 0.1
    default_signal_weight = 0.1

    def __init__(self, model, *, decoder_mode: str):
        self.model = model
        self.decoder_mode = str(decoder_mode)
        if self.decoder_mode not in {"branches", "joint"}:
            raise ValueError("decoder_mode must be 'branches' or 'joint'.")
        if getattr(model, "classification_level", None) != "trial":
            raise ValueError("The SIC adapter requires a trial-level checkpoint.")
        if not getattr(model, "use_decoder", False):
            raise ValueError("The SIC latent adapter requires trained decoders.")
        self.branches = []
        for name in ("gcn_gru", "bilstm"):
            if getattr(model, f"use_{name}_branch", False):
                decoder = getattr(model, f"{name}_decoder", None)
                if decoder is None:
                    raise ValueError(f"The active SIC branch {name!r} has no decoder.")
                self.branches.append((name, int(getattr(model, f"{name}_feature_dim"))))
        if not self.branches:
            raise ValueError("The SIC checkpoint has no active encoder branches.")
        if self.decoder_mode == "joint":
            if {name for name, _ in self.branches} != {"gcn_gru", "bilstm"}:
                raise ValueError("Joint reconstruction requires both SIC branches.")
            if not getattr(model, "use_joint_reconstruction", False):
                raise ValueError("This SIC checkpoint has no joint reconstruction.")
            if getattr(model, "joint_reconstruction_fusion", None) is None:
                raise ValueError("The SIC checkpoint is missing its fusion layer.")
        self._vcsc = CounterfactualLoss()

    def restore_lazy_weights(self, model_path: Path, sample_input: np.ndarray) -> None:
        """Build lazy fusion state, then reload so its saved scalar is restored."""

        if self.decoder_mode != "joint":
            return
        fusion = self.model.joint_reconstruction_fusion
        if fusion.built:
            return
        timesteps = int(np.asarray(sample_input).shape[-2])
        branch_outputs = {}
        for name, width in self.branches:
            branch_outputs[name] = self.model.decode_branch_feature_sequence(
                name,
                tf.zeros((1, timesteps, width), dtype=tf.float32),
            )
        fusion([branch_outputs["gcn_gru"], branch_outputs["bilstm"]])
        self.model.load_weights(Path(model_path))

    def initial_state(self, inputs):
        features = self.model.get_encoder_features(inputs)
        return tf.cast(features["window_features"], tf.float32)

    def logits_from_state(self, state):
        sequence = tf.reshape(state, [1, -1, tf.shape(state)[-1]])
        embedding = self.model.trial_recurrent_classifier(sequence, training=False)
        return tf.cast(self.model.vc_target(embedding, training=False), tf.float32)

    def _branch_reconstructions(self, state, reference_input):
        outputs = {}
        offset = 0
        for name, width in self.branches:
            branch_state = state[..., offset : offset + width]
            flat = tf.reshape(
                branch_state,
                [-1, tf.shape(state)[-2], width],
            )
            decoded = self.model.decode_branch_feature_sequence(name, flat)
            outputs[name] = tf.reshape(
                tf.cast(decoded, tf.float32), tf.shape(reference_input)
            )
            offset += width
        return outputs

    def reconstruct(self, state, reference_input):
        branches = self._branch_reconstructions(state, reference_input)
        if self.decoder_mode == "branches":
            return branches
        joint = self.model.joint_reconstruction_fusion(
            [branches["gcn_gru"], branches["bilstm"]]
        )
        return {"joint": tf.cast(joint, tf.float32)}

    def constraint(self, name, signal):
        if name != "vcsc":
            return super().constraint(name, signal)
        return self._vcsc.physiological_validity(signal)

    def metadata(self):
        metadata = {
            **super().metadata(),
            "decoder_mode": self.decoder_mode,
            "outputs": (
                ["joint"]
                if self.decoder_mode == "joint"
                else [name for name, _ in self.branches]
            ),
        }
        fusion = getattr(self.model, "joint_reconstruction_fusion", None)
        if fusion is not None and fusion.built:
            metadata["joint_reconstruction_alpha"] = float(fusion.alpha.numpy())
        return metadata


def create_sic_adapter(*, model_path, config, sample_input):
    """Factory usable with ``--adapter ...:create_sic_adapter``."""

    model_module = str(
        config.get(
            "model_module",
            "eegproc.deep_learning.joint_architectures.SICModelv15.sic_model",
        )
    )
    importlib.import_module(model_module)
    model = tf.keras.models.load_model(Path(model_path), compile=False, safe_mode=True)
    adapter = SICCounterfactualAdapter(
        model,
        decoder_mode=str(config.get("decoder_mode", "joint")),
    )
    adapter.restore_lazy_weights(Path(model_path), sample_input)
    return adapter


def _normalize_windows(features, mode, epsilon=1e-6):
    """Match SIC preprocessing while retaining an invertible affine transform."""

    x = np.asarray(features, dtype=np.float32)
    if mode == "none":
        offset = np.zeros((len(x), 1, 1), dtype=np.float32)
        scale = np.ones((len(x), 1, 1), dtype=np.float32)
    elif mode == "global_rms":
        offset = np.zeros((len(x), 1, 1), dtype=np.float32)
        scale = np.sqrt(
            np.mean(np.square(x, dtype=np.float64), axis=(1, 2), keepdims=True)
        ).astype(np.float32)
        scale = np.maximum(scale, epsilon)
    elif mode == "feature_zscore":
        offset = np.mean(x, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
        scale = np.std(x, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
        scale = np.maximum(scale, epsilon)
    else:
        raise ValueError("window_normalization must be none, global_rms, or feature_zscore.")
    normalized = ((x - offset) / scale).astype(np.float32)
    return normalized, offset, scale


def _group_transform(values, window_subjects, window_trials, subjects, trials):
    grouped = []
    for subject, trial in zip(subjects.tolist(), trials.tolist()):
        selected = (window_subjects == subject) & (window_trials == trial)
        grouped.append(values[selected])
    return np.stack(grouped, axis=0).astype(np.float32)


def load_sic_raw_trials(config) -> TrialDataset:
    """Dataset-loader plug-in that preserves normalization transforms.

    ``signal_unit`` is intentionally caller supplied.  EEGProc does not claim
    that a source array is in microvolts when its provenance does not say so.
    """

    model_module = str(
        config.get(
            "model_module",
            "eegproc.deep_learning.joint_architectures.SICModelv15.sic_model",
        )
    )
    training = importlib.import_module(model_module.rsplit(".", 1)[0] + ".sic_model_train")
    eeg_path = config.get("raw_eeg_npy")
    labels_path = config.get("raw_labels_npy")
    label_dimension = config.get("label_dimension")
    if eeg_path is None or labels_path is None or label_dimension is None:
        raise ValueError(
            "SIC raw data config requires raw_eeg_npy, raw_labels_npy, and label_dimension."
        )
    window_normalization = str(config.get("window_normalization", "global_rms"))
    values = training.load_sic_training_data(
        eeg_path=Path(eeg_path),
        labels_path=Path(labels_path),
        dataset=str(config.get("dataset", "dreamer")),
        label_dimension=str(label_dimension),
        window_size_sec=float(config.get("window_sec", 1.0)),
        fs=float(config.get("fs", 128.0)),
        overlap=float(config.get("window_overlap", 0.0)),
        window_normalization="none",
        label_threshold_mode=str(config.get("label_threshold_mode", "global")),
        median_label=float(config.get("median_label", 3.0)),
        return_original_ratings=True,
    )
    raw_features, labels, window_subjects, window_trials, ratings = values
    if bool(config.get("remove_median_label", False)):
        keep = ~np.isclose(ratings, float(config.get("median_label", 3.0)))
        raw_features, labels, window_subjects, window_trials = (
            np.asarray(value)[keep]
            for value in (raw_features, labels, window_subjects, window_trials)
        )
    features, offset, scale = _normalize_windows(
        raw_features, window_normalization
    )
    grouped, grouped_labels, subjects, trials = training._group_windows_into_trials(
        features, labels, window_subjects, window_trials
    )
    grouped_offset = _group_transform(
        offset, window_subjects, window_trials, subjects, trials
    )
    grouped_scale = _group_transform(
        scale, window_subjects, window_trials, subjects, trials
    )

    channel_names = None
    band_names = None
    feature_order = config.get("feature_order")
    if str(config.get("dataset", "dreamer")) == "dreamer":
        prepare = importlib.import_module("eegproc.deep_learning.prepare_datasets")
        channel_names = tuple(prepare.DREAMER_EEG_COLS)
        band_names = ("Theta", "Alpha", "Beta")
        feature_order = feature_order or "channel-major"
    return TrialDataset(
        features=grouped,
        subject_ids=subjects,
        trial_ids=trials,
        labels=grouped_labels,
        normalization_offset=grouped_offset,
        normalization_scale=grouped_scale,
        channel_names=channel_names,
        band_names=band_names,
        feature_order=None if feature_order is None else str(feature_order),
        signal_unit=(
            None if config.get("signal_unit") is None else str(config["signal_unit"])
        ),
        metadata={
            "dataset": str(config.get("dataset", "dreamer")),
            "label_dimension": str(label_dimension),
            "window_normalization": window_normalization,
            "fs": float(config.get("fs", 128.0)),
            "window_sec": float(config.get("window_sec", 1.0)),
        },
    )
