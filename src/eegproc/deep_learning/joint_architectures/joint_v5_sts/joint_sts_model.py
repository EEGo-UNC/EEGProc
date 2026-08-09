"""Joint v5 STS: MTLFuseNet-style spatio-spectral GCN -> GRU -> classifier.

v5.0 is intentionally a classification-only baseline corresponding to the
upper spatio-spectral branch in the MTLFuseNet architecture:

    one EEG window/second
        -> split into theta/alpha/beta channel waveforms
        -> shared fixed-MI GCN over channels
        -> flatten channel x GCN features per band
        -> GRU across the ordered frequency-band sequence
        -> direct 384-D spatio-spectral latent
        -> dense classifier

For DREAMER with 1-second windows at 128 Hz and 3 bands:

    (B, 128, 42)
        -> (B, 3, 14, 128)
        -> shared GCN(32)
        -> (B, 3, 14, 32)
        -> (B, 3, 448)
        -> GRU(384)
        -> (B, 384)
        -> classifier logits

There is NO BiLSTM, VAE, decoder, feature fusion, temporal pooling, post-GRU
convolution, subject adversary, SupCon, or MLDG machinery in v5.0.

Leakage rule
------------
The mutual-information adjacency is adapted lazily inside ``fit`` from the
data passed to that fit call. Under EEGProc LOSO, each model therefore builds
A from that fold's TRAINING windows only. Validation/test subjects never enter
the graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from ...unsupervised.GNN.GraphConvMTL import GraphConvMTL
    from ...unsupervised.GNN.GCNMTL import compute_mtl_shared_mi_adjacency
except ImportError:
    from eegproc.deep_learning.unsupervised.GNN.GraphConvMTL import GraphConvMTL
    from eegproc.deep_learning.unsupervised.GNN.GCNMTL import (
        compute_mtl_shared_mi_adjacency,
    )


JOINT_STS_BUILDER_API_VERSION = 2


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Sparse categorical focal loss; gamma=0 recovers cross-entropy."""

    def __init__(
        self,
        gamma: float = 0.0,
        alpha=None,
        from_logits: bool = True,
        name: str = "sparse_categorical_focal_loss",
        **kwargs,
    ):
        super().__init__(name=name, reduction="sum_over_batch_size", **kwargs)
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.gamma = float(gamma)
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.alpha = tuple(float(v) for v in alpha)
        else:
            self.alpha = (float(alpha),)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        if self.from_logits:
            probs = tf.nn.softmax(y_pred, axis=-1)
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_true, logits=y_pred
            )
        else:
            probs = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
            ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, probs)

        rows = tf.range(tf.shape(y_true)[0], dtype=tf.int32)
        pt = tf.gather_nd(probs, tf.stack([rows, y_true], axis=1))
        focal = tf.pow(
            tf.maximum(1.0 - pt, tf.keras.backend.epsilon()), self.gamma
        )
        loss = tf.cast(ce, pt.dtype) * focal

        if self.alpha is not None:
            alpha = tf.constant(self.alpha, dtype=loss.dtype)
            if len(self.alpha) == 1:
                loss *= alpha[0]
            else:
                loss *= tf.gather(alpha, y_true)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
                "from_logits": self.from_logits,
            }
        )
        return config


def _build_optimizer(name: str, learning_rate: float, weight_decay: float):
    name = str(name).lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError("optimizer_name must be 'adam' or 'adamw'.")


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class JointSTSModel(tf.keras.Model):
    """Shared fixed-MI GCN -> spectral GRU -> dense classifier.

    The model is window-level only. For the intended DREAMER baseline each
    sample is one second shaped ``(128, 42)`` where 42 = 14 channels x 3 bands.

    The full 1-second waveform is the node feature vector. Thus each band is
    represented as ``(14 channels, 128 waveform samples)`` before graph
    convolution. The GRU recurrence is across bands, not across EEG time.
    """

    def __init__(
        self,
        *,
        timesteps: int,
        n_features: int,
        n_classes: int = 2,
        n_channels: int = 14,
        n_bands: int = 3,
        classification_level: str = "window",
        t_down: int = 1,
        temporal_pool_sizes: Sequence[int] | None = (),
        gcn_units: Sequence[int] = (32,),
        gcn_dropout: float = 0.10,
        gcn_activation: str = "relu",
        gcn_use_batch_norm: bool = False,
        spectral_gru_units: int = 384,
        spectral_gru_dropout: float = 0.0,
        graph_add_self_loops: bool = True,
        graph_symmetrize: bool = True,
        graph_epsilon: float = 1e-8,
        mi_n_neighbors: int = 3,
        mi_random_state: int = 42,
        mi_zero_diagonal: bool = False,
        mi_band_reduction: str = "mean",
        mi_max_observations: int | None = 50000,
        adjacency=None,
        classification_hidden_units: int = 128,
        classification_dropout: float = 0.30,
        activation: str = "relu",
        use_class_weight: bool = False,
        name: str = "joint_v5_sts_model",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.timesteps = int(timesteps)
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.n_channels = int(n_channels)
        self.n_bands = int(n_bands)

        self.classification_level = str(classification_level).lower()
        if self.classification_level != "window":
            raise ValueError(
                "joint_v5_sts v5.0 is window-level only. Train on 1-second "
                "windows and use cross_val trial aggregation for trial metrics."
            )

        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive.")
        if self.n_features != self.n_channels * self.n_bands:
            raise ValueError(
                f"n_features={self.n_features} must equal n_channels*n_bands="
                f"{self.n_channels * self.n_bands}."
            )

        # Kept in the signature for compatibility with existing v5 CLI/scripts,
        # but temporal downsampling is deliberately disabled in corrected v5.
        self.t_down = int(t_down)
        normalized_pool_sizes = (
            ()
            if temporal_pool_sizes is None
            else tuple(int(v) for v in temporal_pool_sizes)
        )
        if self.t_down != 1 or normalized_pool_sizes:
            raise ValueError(
                "Corrected joint_v5_sts does not temporally pool the 1-second "
                "waveform. Use t_down=1 and temporal_pool_sizes=()."
            )
        self.temporal_pool_sizes = ()

        self.gcn_units = tuple(int(v) for v in gcn_units)
        if not self.gcn_units or any(v <= 0 for v in self.gcn_units):
            raise ValueError(f"gcn_units must be positive, got {self.gcn_units}.")
        self.gcn_dropout = float(gcn_dropout)
        self.gcn_activation = str(gcn_activation)
        self.gcn_use_batch_norm = bool(gcn_use_batch_norm)

        self.spectral_gru_units = int(spectral_gru_units)
        self.spectral_gru_dropout = float(spectral_gru_dropout)
        if self.spectral_gru_units <= 0:
            raise ValueError("spectral_gru_units must be positive.")

        self.graph_add_self_loops = bool(graph_add_self_loops)
        self.graph_symmetrize = bool(graph_symmetrize)
        self.graph_epsilon = float(graph_epsilon)

        self.mi_n_neighbors = int(mi_n_neighbors)
        self.mi_random_state = int(mi_random_state)
        self.mi_zero_diagonal = bool(mi_zero_diagonal)
        self.mi_band_reduction = str(mi_band_reduction)
        self.mi_max_observations = (
            None if mi_max_observations is None else int(mi_max_observations)
        )
        if self.mi_max_observations is not None and self.mi_max_observations < 4:
            raise ValueError("mi_max_observations must be >= 4 or None.")

        self.classification_hidden_units = int(classification_hidden_units)
        self.classification_dropout_rate = float(classification_dropout)
        self.activation_name = str(activation)
        self.use_class_weight = bool(use_class_weight)

        # GCN layers depend on the fold-training MI adjacency and are installed
        # lazily before the first Keras fit.
        self.shared_gcn_layers: list[GraphConvMTL] = []
        self.shared_bn_layers: list[tf.keras.layers.Layer | None] = []
        self.shared_dropout_layers: list[tf.keras.layers.Layer] = []
        self._fitted_adjacency: np.ndarray | None = None

        # The GRU consumes one vector per frequency band:
        # (B, n_bands, n_channels * gcn_units[-1]).
        self.spectral_gru = tf.keras.layers.GRU(
            self.spectral_gru_units,
            return_sequences=False,
            dropout=self.spectral_gru_dropout,
            name="v5_spectral_gru",
        )

        self.classification_hidden = tf.keras.layers.Dense(
            self.classification_hidden_units,
            activation=self.activation_name,
            name="v5_classification_hidden",
        )
        self.classification_norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            name="v5_classification_ln",
        )
        self.classification_dropout = tf.keras.layers.Dropout(
            self.classification_dropout_rate,
            name="v5_classification_dropout",
        )
        self.logits_layer = tf.keras.layers.Dense(
            self.n_classes,
            activation=None,
            name="v5_logits",
        )

        self.requires_subject_ids = False
        self.use_subject_adversarial = False

        if adjacency is not None:
            self._install_graph_layers(np.asarray(adjacency, dtype=np.float32))

    @property
    def adjacency_is_fitted(self) -> bool:
        return bool(self.shared_gcn_layers) and self._fitted_adjacency is not None

    @property
    def spectral_feature_dim(self) -> int:
        return self.n_channels * self.gcn_units[-1]

    def _install_graph_layers(self, adjacency: np.ndarray) -> None:
        if self.shared_gcn_layers:
            raise RuntimeError(
                "Graph layers are already installed; adjacency must remain fixed "
                "for the lifetime of one LOSO-fold model."
            )

        adjacency = np.asarray(adjacency, dtype=np.float32)
        expected = (self.n_channels, self.n_channels)
        if adjacency.shape != expected:
            raise ValueError(f"adjacency must be {expected}, got {adjacency.shape}.")

        self._fitted_adjacency = adjacency.copy()

        for layer_index, units in enumerate(self.gcn_units):
            self.shared_gcn_layers.append(
                GraphConvMTL(
                    units=units,
                    n_nodes=self.n_channels,
                    adjacency=adjacency,
                    activation=self.gcn_activation,
                    add_self_loops=self.graph_add_self_loops,
                    symmetrize=self.graph_symmetrize,
                    epsilon=self.graph_epsilon,
                    name=f"v5_shared_mtl_gcn_{layer_index}",
                )
            )
            self.shared_bn_layers.append(
                tf.keras.layers.BatchNormalization(
                    name=f"v5_shared_gcn_bn_{layer_index}"
                )
                if self.gcn_use_batch_norm
                else None
            )
            self.shared_dropout_layers.append(
                tf.keras.layers.Dropout(
                    self.gcn_dropout,
                    name=f"v5_shared_gcn_dropout_{layer_index}",
                )
            )

    def adapt_graph(self, training_inputs) -> np.ndarray:
        """Estimate and install one fixed shared MI graph from TRAINING windows."""
        if isinstance(training_inputs, tf.data.Dataset):
            raise TypeError(
                "v5 graph adaptation expects an in-memory NumPy/Tensor training "
                "array so the fold-training MI graph can be computed before fit."
            )

        x = np.asarray(training_inputs, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(
                "Corrected v5 expects training windows shaped (N,T,F); "
                f"got {x.shape}."
            )
        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} channel-band features, "
                f"got {x.shape[-1]}."
            )
        if x.shape[1] != self.timesteps:
            raise ValueError(
                f"Expected {self.timesteps} samples per window, got {x.shape[1]}."
            )

        # One MI observation = one instantaneous channel-band sample.
        # This retains the existing fold-local MTL adjacency estimator while
        # the neural GCN now receives the complete 1-second waveform per node.
        observations = x.reshape(-1, self.n_features)
        if (
            self.mi_max_observations is not None
            and len(observations) > self.mi_max_observations
        ):
            rng = np.random.default_rng(self.mi_random_state)
            idx = rng.choice(
                len(observations),
                size=self.mi_max_observations,
                replace=False,
            )
            observations = observations[idx]

        adjacency = compute_mtl_shared_mi_adjacency(
            observations,
            n_channels=self.n_channels,
            n_bands=self.n_bands,
            n_neighbors=self.mi_n_neighbors,
            random_state=self.mi_random_state,
            zero_diagonal=self.mi_zero_diagonal,
            band_reduction=self.mi_band_reduction,
        )
        self._install_graph_layers(adjacency)
        return adjacency

    def fit(self, x=None, y=None, **kwargs):
        # Cross-validation creates a fresh model for each LOSO fold and calls
        # fit with only that fold's training windows. Build A at exactly that
        # point, before any validation/test prediction.
        if not self.adjacency_is_fitted:
            self.adapt_graph(x)
        if not self.use_class_weight:
            kwargs.pop("class_weight", None)
        return super().fit(x=x, y=y, **kwargs)

    def _to_band_channel_waveforms(self, windows: tf.Tensor) -> tf.Tensor:
        """(B,T,C*Bands) channel-major -> (B,Bands,C,T)."""
        windows = tf.convert_to_tensor(windows, dtype=tf.float32)
        if windows.shape.rank != 3:
            raise ValueError(
                "Corrected v5 expects (batch,timesteps,features); "
                f"got {windows.shape}."
            )

        static_t = windows.shape[1]
        static_f = windows.shape[2]
        if static_t is not None and int(static_t) != self.timesteps:
            raise ValueError(
                f"Expected {self.timesteps} timesteps, got {static_t}."
            )
        if static_f is not None and int(static_f) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {static_f}."
            )

        batch_size = tf.shape(windows)[0]
        x = tf.reshape(
            windows,
            (
                batch_size,
                self.timesteps,
                self.n_channels,
                self.n_bands,
            ),
        )
        # (B,T,C,Bands) -> (B,Bands,C,T)
        return tf.transpose(x, perm=(0, 3, 2, 1))

    def _encode_one_band(
        self,
        band_waveform: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        """Encode one band: (B,C,T) -> (B,C,gcn_units[-1])."""
        x = band_waveform
        for gcn, bn, dropout in zip(
            self.shared_gcn_layers,
            self.shared_bn_layers,
            self.shared_dropout_layers,
        ):
            x = gcn(x, training=training)
            if bn is not None:
                x = bn(x, training=training)
            x = dropout(x, training=training)
        return x

    def encode(self, inputs, training=False) -> dict[str, tf.Tensor]:
        if not self.adjacency_is_fitted:
            raise RuntimeError(
                "The MI adjacency has not been fitted. Call model.fit(...) or "
                "model.adapt_graph(training_inputs) before inference."
            )

        band_waveforms = self._to_band_channel_waveforms(inputs)

        # SAME GCN layer objects are reused for every band.
        band_gcn = [
            self._encode_one_band(
                band_waveforms[:, band_index, :, :],
                training=training,
            )
            for band_index in range(self.n_bands)
        ]

        # (B,Bands,C,U)
        band_gcn_features = tf.stack(band_gcn, axis=1)

        # (B,Bands,C*U), e.g. DREAMER (B,3,448).
        spectral_sequence = tf.reshape(
            band_gcn_features,
            (
                tf.shape(band_gcn_features)[0],
                self.n_bands,
                self.spectral_feature_dim,
            ),
        )

        # GRU recurrence is frequency-band recurrence:
        # theta -> alpha -> beta for DREAMER.
        z_ss = self.spectral_gru(spectral_sequence, training=training)

        return {
            "band_waveforms": band_waveforms,
            "band_gcn_features": band_gcn_features,
            "spectral_sequence": spectral_sequence,
            "z_ss": z_ss,
            "classification_embedding": z_ss,
        }

    def _classify_embedding(self, embedding, training=False):
        latent = self.classification_hidden(embedding)
        latent = self.classification_norm(latent)
        latent = self.classification_dropout(latent, training=training)
        logits = self.logits_layer(latent)
        return latent, logits

    def call(self, inputs, training=False):
        encoded = self.encode(inputs, training=training)
        _, logits = self._classify_embedding(encoded["z_ss"], training=training)
        return logits

    def predict_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        return self(x, training=False)

    def predict_diagnostics(self, inputs, batch_size=None):
        eeg = tf.convert_to_tensor(inputs, dtype=tf.float32)
        n = int(tf.shape(eeg)[0].numpy())
        bs = n if batch_size is None else int(batch_size)
        outputs: dict[str, list[tf.Tensor]] = {
            "z_ss": [],
            "classification_latent": [],
            "logits": [],
            "probabilities": [],
            "logit_margin": [],
        }

        for start in range(0, n, bs):
            batch = eeg[start : start + bs]
            encoded = self.encode(batch, training=False)
            latent, logits = self._classify_embedding(
                encoded["z_ss"], training=False
            )
            probs = tf.nn.softmax(logits, axis=-1)
            margin = (
                logits[:, 1] - logits[:, 0]
                if self.n_classes == 2
                else tf.math.top_k(logits, k=2).values[:, 0]
                - tf.math.top_k(logits, k=2).values[:, 1]
            )
            outputs["z_ss"].append(encoded["z_ss"])
            outputs["classification_latent"].append(latent)
            outputs["logits"].append(logits)
            outputs["probabilities"].append(probs)
            outputs["logit_margin"].append(margin)

        return {k: tf.concat(v, axis=0) for k, v in outputs.items()}

    def predict_mc_probabilities(self, inputs, n_samples=1, seed=None):
        del seed
        logits = self(inputs, training=False)
        probs = tf.nn.softmax(logits, axis=-1)
        return {
            "mean_probabilities": probs,
            "probability_samples": tf.repeat(
                probs[tf.newaxis, ...], repeats=int(n_samples), axis=0
            ),
        }

    def get_adjacency_matrices(self):
        if not self.shared_gcn_layers:
            return {}
        return {
            "mtl_mi_raw": self.shared_gcn_layers[0].raw_adjacency(),
            "mtl_mi_normalized": self.shared_gcn_layers[0].normalized_adjacency(),
        }

    def get_band_features(self, inputs, training=False):
        if not self.adjacency_is_fitted:
            raise RuntimeError("Graph has not been adapted yet.")
        band_waveforms = self._to_band_channel_waveforms(inputs)
        return {
            f"band_{band_index}": self._encode_one_band(
                band_waveforms[:, band_index, :, :],
                training=training,
            )
            for band_index in range(self.n_bands)
        }

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "timesteps": self.timesteps,
                "n_features": self.n_features,
                "n_classes": self.n_classes,
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "classification_level": self.classification_level,
                "t_down": 1,
                "temporal_pool_sizes": (),
                "gcn_units": self.gcn_units,
                "gcn_dropout": self.gcn_dropout,
                "gcn_activation": self.gcn_activation,
                "gcn_use_batch_norm": self.gcn_use_batch_norm,
                "spectral_gru_units": self.spectral_gru_units,
                "spectral_gru_dropout": self.spectral_gru_dropout,
                "graph_add_self_loops": self.graph_add_self_loops,
                "graph_symmetrize": self.graph_symmetrize,
                "graph_epsilon": self.graph_epsilon,
                "mi_n_neighbors": self.mi_n_neighbors,
                "mi_random_state": self.mi_random_state,
                "mi_zero_diagonal": self.mi_zero_diagonal,
                "mi_band_reduction": self.mi_band_reduction,
                "mi_max_observations": self.mi_max_observations,
                "adjacency": (
                    None
                    if self._fitted_adjacency is None
                    else self._fitted_adjacency.tolist()
                ),
                "classification_hidden_units": self.classification_hidden_units,
                "classification_dropout": self.classification_dropout_rate,
                "activation": self.activation_name,
                "use_class_weight": self.use_class_weight,
            }
        )
        return config


def build_joint_sts_model(
    input_shape,
    *,
    n_classes=2,
    n_channels=14,
    n_bands=3,
    classification_level="window",
    t_down=1,
    temporal_pool_sizes=(),
    gcn_units=(32,),
    gcn_dropout=0.10,
    gcn_activation="relu",
    gcn_use_batch_norm=False,
    spectral_gru_units=384,
    spectral_gru_dropout=0.0,
    graph_add_self_loops=True,
    graph_symmetrize=True,
    graph_epsilon=1e-8,
    mi_n_neighbors=3,
    mi_random_state=42,
    mi_zero_diagonal=False,
    mi_band_reduction="mean",
    mi_max_observations=50000,
    adjacency=None,
    classification_hidden_units=128,
    classification_dropout=0.30,
    activation="relu",
    focal_gamma=0.0,
    focal_alpha=None,
    use_class_weight=False,
    optimizer_name="adamw",
    classification_learning_rate=1e-4,
    weight_decay=1e-4,
    model_name="joint_v5_sts_model",
    **unused_kwargs,
):
    """Build corrected v5.0 GCN -> spectral GRU -> classifier.

    Expected input shape is ``(samples_per_window, channels*bands)``. For the
    paper-aligned DREAMER baseline use one-second windows at 128 Hz:
    ``input_shape=(128, 42)``.

    ``adjacency=None`` is intentional during LOSO. The model lazily estimates A
    from each fold's training windows at the start of ``fit``.
    """
    del unused_kwargs

    classification_level = str(classification_level).lower()
    if classification_level != "window":
        raise ValueError(
            "Corrected v5 is trained at window level. Pass "
            "classification_level='window'; use cross_val evaluation_level='trial' "
            "for trial-level reporting."
        )

    shape = tuple(int(v) for v in input_shape)
    if len(shape) != 2:
        raise ValueError(
            f"Corrected v5 input_shape must be (T,F), got {shape}."
        )
    timesteps, n_features = shape

    model = JointSTSModel(
        timesteps=timesteps,
        n_features=n_features,
        n_classes=int(n_classes),
        n_channels=int(n_channels),
        n_bands=int(n_bands),
        classification_level="window",
        t_down=int(t_down),
        temporal_pool_sizes=(
            () if temporal_pool_sizes is None else tuple(int(v) for v in temporal_pool_sizes)
        ),
        gcn_units=tuple(int(v) for v in gcn_units),
        gcn_dropout=float(gcn_dropout),
        gcn_activation=str(gcn_activation),
        gcn_use_batch_norm=bool(gcn_use_batch_norm),
        spectral_gru_units=int(spectral_gru_units),
        spectral_gru_dropout=float(spectral_gru_dropout),
        graph_add_self_loops=bool(graph_add_self_loops),
        graph_symmetrize=bool(graph_symmetrize),
        graph_epsilon=float(graph_epsilon),
        mi_n_neighbors=int(mi_n_neighbors),
        mi_random_state=int(mi_random_state),
        mi_zero_diagonal=bool(mi_zero_diagonal),
        mi_band_reduction=str(mi_band_reduction),
        mi_max_observations=(
            None if mi_max_observations is None else int(mi_max_observations)
        ),
        adjacency=adjacency,
        classification_hidden_units=int(classification_hidden_units),
        classification_dropout=float(classification_dropout),
        activation=str(activation),
        use_class_weight=bool(use_class_weight),
        name=model_name,
    )

    model.compile(
        optimizer=_build_optimizer(
            optimizer_name,
            float(classification_learning_rate),
            float(weight_decay),
        ),
        loss=SparseCategoricalFocalLoss(
            gamma=float(focal_gamma),
            alpha=focal_alpha,
            from_logits=True,
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        jit_compile=False,
    )
    return model


__all__ = [
    "JOINT_STS_BUILDER_API_VERSION",
    "JointSTSModel",
    "SparseCategoricalFocalLoss",
    "build_joint_sts_model",
]
