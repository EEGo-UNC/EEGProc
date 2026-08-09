"""Joint v4 STS: band-separated GCN -> BiLSTM -> classifier.

Classification-only architecture:
    EEG -> independent per-band GCN stacks -> concatenate band graph features
        -> BiLSTM -> temporal pooling -> dense classifier -> logits

There is no VAE, decoder, latent posterior, parallel BiLSTM encoder, or feature fusion.
"""

from __future__ import annotations

from collections.abc import Sequence
import tensorflow as tf

try:
    from ...unsupervised.Convolutions.GCN import BandSeparatedGCNEncoder
except ImportError:
    from eegproc.deep_learning.unsupervised.Convolutions.GCN import BandSeparatedGCNEncoder

JOINT_STS_BUILDER_API_VERSION = 4


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=0.0, alpha=None, from_logits=True,
                 name="sparse_categorical_focal_loss", **kwargs):
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
        focal = tf.pow(tf.maximum(1.0 - pt, tf.keras.backend.epsilon()), self.gamma)
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
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "from_logits": self.from_logits,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class JointSTSModel(tf.keras.Model):
    """Band-separated GCN followed sequentially by BiLSTM and classifier."""

    def __init__(
        self,
        graph_encoder,
        n_classes=2,
        bilstm_units=256,
        n_bilstm_layers=1,
        bilstm_dropout=0.30,
        classification_hidden_units=128,
        classification_dropout=0.30,
        activation="relu",
        name="joint_v4_sts_model",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.graph_encoder = graph_encoder
        self.n_classes = int(n_classes)
        self.bilstm_units = int(bilstm_units)
        self.n_bilstm_layers = int(n_bilstm_layers)
        self.bilstm_dropout_rate = float(bilstm_dropout)
        self.classification_hidden_units = int(classification_hidden_units)
        self.classification_dropout_rate = float(classification_dropout)
        self.activation_name = str(activation)

        self.bilstm_layers = []
        self.bilstm_norms = []
        self.bilstm_dropouts = []
        for i in range(self.n_bilstm_layers):
            self.bilstm_layers.append(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        self.bilstm_units,
                        return_sequences=True,
                        name=f"v4_lstm_{i}",
                    ),
                    merge_mode="concat",
                    name=f"v4_bilstm_{i}",
                )
            )
            self.bilstm_norms.append(
                tf.keras.layers.LayerNormalization(axis=-1, name=f"v4_bilstm_ln_{i}")
            )
            self.bilstm_dropouts.append(
                tf.keras.layers.Dropout(
                    self.bilstm_dropout_rate, name=f"v4_bilstm_dropout_{i}"
                )
            )

        self.temporal_pool = tf.keras.layers.GlobalAveragePooling1D(
            name="v4_temporal_pool"
        )
        self.classification_hidden = tf.keras.layers.Dense(
            self.classification_hidden_units,
            activation=self.activation_name,
            name="v4_classification_hidden",
        )
        self.classification_norm = tf.keras.layers.LayerNormalization(
            axis=-1, name="v4_classification_ln"
        )
        self.classification_dropout = tf.keras.layers.Dropout(
            self.classification_dropout_rate, name="v4_classification_dropout"
        )
        self.logits_layer = tf.keras.layers.Dense(
            self.n_classes, activation=None, name="v4_logits"
        )

        self.requires_subject_ids = False
        self.use_subject_adversarial = False

    def encode_sequence(self, inputs, training=False):
        graph_sequence = self.graph_encoder(inputs, training=training)
        x = graph_sequence
        for bilstm, norm, dropout in zip(
            self.bilstm_layers, self.bilstm_norms, self.bilstm_dropouts
        ):
            x = bilstm(x, training=training)
            x = norm(x)
            x = dropout(x, training=training)
        return {
            "graph_sequence": graph_sequence,
            "bilstm_sequence": x,
        }

    def call(self, inputs, training=False):
        encoded = self.encode_sequence(inputs, training=training)
        x = self.temporal_pool(encoded["bilstm_sequence"])
        x = self.classification_hidden(x)
        x = self.classification_norm(x)
        x = self.classification_dropout(x, training=training)
        return self.logits_layer(x)

    def predict_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        return self(x, training=False)

    def predict_diagnostics(self, inputs, batch_size=None):
        eeg = tf.convert_to_tensor(inputs, dtype=tf.float32)
        n = int(tf.shape(eeg)[0].numpy())
        bs = n if batch_size is None else int(batch_size)
        outputs = {
            "encoder_output": [],
            "graph_sequence": [],
            "bilstm_sequence": [],
            "classification_latent": [],
            "logits": [],
            "probabilities": [],
            "logit_margin": [],
        }
        for start in range(0, n, bs):
            batch = eeg[start:start+bs]
            encoded = self.encode_sequence(batch, training=False)
            latent = self.temporal_pool(encoded["bilstm_sequence"])
            latent = self.classification_hidden(latent)
            latent = self.classification_norm(latent)
            logits = self.logits_layer(latent)
            probs = tf.nn.softmax(logits, axis=-1)
            margin = (
                logits[:, 1] - logits[:, 0]
                if self.n_classes == 2
                else tf.math.top_k(logits, k=2).values[:, 0]
                   - tf.math.top_k(logits, k=2).values[:, 1]
            )
            batch_values = {
                "encoder_output": encoded["graph_sequence"],
                "graph_sequence": encoded["graph_sequence"],
                "bilstm_sequence": encoded["bilstm_sequence"],
                "classification_latent": latent,
                "logits": logits,
                "probabilities": probs,
                "logit_margin": margin,
            }
            for key, value in batch_values.items():
                outputs[key].append(value)
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
        return {
            "band_separated_gcn": self.graph_encoder.get_adjacency_matrices()
        }

    def get_band_features(self, inputs, training=False):
        return self.graph_encoder.get_band_features(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "graph_encoder": tf.keras.utils.serialize_keras_object(self.graph_encoder),
            "n_classes": self.n_classes,
            "bilstm_units": self.bilstm_units,
            "n_bilstm_layers": self.n_bilstm_layers,
            "bilstm_dropout": self.bilstm_dropout_rate,
            "classification_hidden_units": self.classification_hidden_units,
            "classification_dropout": self.classification_dropout_rate,
            "activation": self.activation_name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["graph_encoder"] = tf.keras.utils.deserialize_keras_object(
            config["graph_encoder"]
        )
        return cls(**config)


def _build_optimizer(name, learning_rate, weight_decay):
    name = str(name).lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
    raise ValueError("optimizer_name must be adam or adamw.")


def build_joint_sts_model(
    input_shape,
    *,
    n_classes=2,
    n_channels=14,
    n_bands=3,
    t_down=2,
    temporal_pool_sizes=(2,),
    gcn_units=(128, 64),
    spectral_emb_dim=128,
    gcn_dropout=0.20,
    gcn_activation="relu",
    gcn_use_batch_norm=False,
    graph_self_loop_bias=2.0,
    graph_identity_mix=0.0,
    graph_adjacency_reg_weight=1e-4,
    bilstm_units=256,
    n_bilstm_layers=1,
    bilstm_dropout=0.30,
    classification_hidden_units=128,
    classification_dropout=0.30,
    activation="relu",
    focal_gamma=0.0,
    focal_alpha=None,
    optimizer_name="adamw",
    classification_learning_rate=1e-4,
    weight_decay=1e-4,
    model_name="joint_v4_sts_model",
    **unused_kwargs,
):
    """Build and compile band-separated GCN -> BiLSTM -> classifier."""
    timesteps, n_features = map(int, input_shape)
    if n_features != int(n_channels) * int(n_bands):
        raise ValueError(
            f"{n_features=} must equal n_channels*n_bands="
            f"{int(n_channels) * int(n_bands)}."
        )

    graph_encoder = BandSeparatedGCNEncoder(
        timesteps=timesteps,
        t_down=int(t_down),
        n_channels=int(n_channels),
        n_bands=int(n_bands),
        gcn_units=tuple(int(v) for v in gcn_units),
        temporal_pool_sizes=(
            None if temporal_pool_sizes is None
            else tuple(int(v) for v in temporal_pool_sizes)
        ),
        emb_dim=int(spectral_emb_dim),
        dropout=float(gcn_dropout),
        activation=str(gcn_activation),
        use_batch_norm=bool(gcn_use_batch_norm),
        graph_self_loop_bias=float(graph_self_loop_bias),
        graph_identity_mix=float(graph_identity_mix),
        graph_adjacency_reg_weight=float(graph_adjacency_reg_weight),
        name="v4_band_separated_gcn",
    )

    model = JointSTSModel(
        graph_encoder=graph_encoder,
        n_classes=int(n_classes),
        bilstm_units=int(bilstm_units),
        n_bilstm_layers=int(n_bilstm_layers),
        bilstm_dropout=float(bilstm_dropout),
        classification_hidden_units=int(classification_hidden_units),
        classification_dropout=float(classification_dropout),
        activation=str(activation),
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

    _ = model(
        tf.zeros((1, timesteps, n_features), dtype=tf.float32),
        training=False,
    )
    return model
