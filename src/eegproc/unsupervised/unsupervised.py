import tensorflow as tf
from tensorflow.keras import layers, Model
from GraphConv import GraphConv

# ---------------------------------------------------------------------------
# LSTM-based encoder architectures
#
# Both encoder_lstm and encoder_bilstm follow the same two-mode design:
#
#   Encoder-only mode  (n_classes=None):
#       Output shape: (batch, ceil(timesteps / 4), emb_dim)
#       Compatible with training_autoencoder as a drop-in replacement for
#       encoder_1dcnn / encoder_2dcnn / encoder_gcn.
#
#   Classifier mode  (n_classes is an integer):
#       A GlobalAveragePooling step collapses the time axis, then a Dense
#       layer maps to n_classes. A Softmax activation is added on top when
#       include_softmax=True, which is what you normally want for training
#       with sparse_categorical_crossentropy. Set include_softmax=False if
#       you want raw logits (e.g. when using from_logits=True in your loss).
# ---------------------------------------------------------------------------


def encoder_1dcnn(
    timesteps: int,
    n_features: int,
    base_filters: int = 64,
    kernel_size: int = 7,
    emb_dim: int = 128,
    dropout: float = 0.10,
) -> tf.keras.Model:
    """Build a 1D convolutional sequence encoder over the time axis.

    Treats the feature vector at each timestep as a flat, unstructured input
    and learns temporal patterns via three Conv1D blocks. Two
    MaxPool1D layers downsample the time axis by a factor of 4 in total.
    This encoder makes no assumptions about the internal organisation of
    ``n_features`` and serves as a general-purpose temporal baseline.

    Parameters
    ----------
    timesteps : int
        Number of timesteps T.
    n_features : int
        Number of features per timestep (e.g. 84 for 14 electrodes x 6 bands).
    base_filters : int, optional
        Number of filters in the first Conv1D block. Subsequent blocks use
        ``base_filters * 2`` and ``base_filters * 4`` respectively.
        Default is 64.
    kernel_size : int, optional
        Temporal kernel size for the first Conv1D block. The second block uses
        ``kernel_size // 2 + 1``, the third uses 3. Default is 7.
    emb_dim : int, optional
        Dimensionality of the output embedding at each downsampled timestep.
        Default is 128.
    dropout : float, optional
        Dropout rate applied after each pooling operation. Default is 0.10.

    Returns
    -------
    tf.keras.Model
        Keras Model with input shape ``(batch, timesteps, n_features)`` and
        output shape ``(batch, ceil(timesteps / 4), emb_dim)``.

    Notes
    -----
    The output shape is compatible with ``training_autoencoder``, which
    accepts any encoder whose output is ``(batch, t_down, emb_dim)``.
    """
    x_in = layers.Input(shape=(timesteps, n_features), name="x")

    x = layers.Conv1D(
        base_filters, kernel_size, padding="same", activation="relu", name="enc_conv1"
    )(x_in)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_pool1")(x)
    x = layers.Dropout(dropout, name="enc_do1")(x)

    x = layers.Conv1D(
        base_filters * 2,
        kernel_size // 2 + 1,
        padding="same",
        activation="relu",
        name="enc_conv2",
    )(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_pool2")(x)
    x = layers.Dropout(dropout, name="enc_do2")(x)

    x = layers.Conv1D(
        base_filters * 4, 3, padding="same", activation="relu", name="enc_conv3"
    )(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)

    seq_emb = layers.Conv1D(
        emb_dim, 1, padding="same", activation=None, name="seq_emb"
    )(x)

    return Model(inputs=x_in, outputs=seq_emb, name="encoder_1dcnn")


def encoder_2dcnn(
    timesteps: int,
    n_channels: int = 14,
    n_bands: int = 6,
    base_filters: int = 32,
    emb_dim: int = 128,
    dropout: float = 0.10,
) -> tf.keras.Model:
    """Build a 2D convolutional sequence encoder over the electrode x band plane.

    Accepts the same flat input shape as ``encoder_1dcnn`` but internally
    reshapes it into a spatial grid of ``(n_channels, n_bands)`` at each
    timestep. Three ``TimeDistributed`` Conv2D blocks learn joint
    electrode-band patterns.

    After spatial feature extraction, a ``TimeDistributed``
    ``GlobalAveragePooling2D`` collapses the electrode x band grid into a
    single vector per timestep. Two ``MaxPool1D`` layers then downsample the
    time axis by a factor of 4, matching the temporal compression of
    ``encoder_1dcnn``. A final ``Conv1D`` projection maps the result to
    ``emb_dim``.

    Parameters
    ----------
    timesteps : int
        Number of timesteps.
    n_channels : int, optional
        Number of EEG electrode channels.
    n_bands : int, optional
        Number of frequency bands.
    base_filters : int, optional
        Number of filters in the first Conv2D block. Subsequent blocks use
        ``base_filters * 2`` and ``base_filters * 4`` respectively.
        Default is 32.
    emb_dim : int, optional
        Dimensionality of the output embedding at each downsampled timestep.
        Default is 128.
    dropout : float, optional
        Dropout rate applied after each spatial conv block and after the
        first temporal pooling step. Default is 0.10.

    Returns
    -------
    tf.keras.Model
        Keras Model with input shape ``(batch, timesteps, n_channels * n_bands)``
        and output shape ``(batch, ceil(timesteps / 4), emb_dim)``. The input
        shape is identical to ``encoder_1dcnn`` for the same feature count,
        making both encoders interchangeable as arguments to
        ``training_autoencoder``.

    Notes
    -----
    Conv2D kernel sizes are ``(3, 3)`` for the first two blocks and ``(2, 2)``
    for the third. ``same`` padding preserves the spatial grid dimensions
    throughout, so no spatial downsampling occurs and all downsampling occurs
    only on the time axis.

    ``TimeDistributed`` wraps every spatial layer so that the same weights
    are applied independently at each timestep. This weight sharing keeps the
    parameter count tractable and enforces electrode-band spatial relationship
    biases.

    Because ``TimeDistributed(Conv2D(...))`` processes ``T`` spatial grids
    per batch item, memory usage is substantially higher than
    ``encoder_1dcnn`` at equal filter counts. Reduce ``base_filters`` if
    out-of-memory errors occur during training.
    """
    n_features = n_channels * n_bands
    x_in = layers.Input(shape=(timesteps, n_features), name="x")

    x = layers.Reshape((timesteps, n_channels, n_bands, 1), name="to_grid")(x_in)

    x = layers.TimeDistributed(
        layers.Conv2D(base_filters, (3, 3), padding="same", activation="relu"),
        name="enc_conv2d_1",
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name="enc_bn2d_1")(x)
    x = layers.TimeDistributed(layers.Dropout(dropout), name="enc_do2d_1")(x)

    x = layers.TimeDistributed(
        layers.Conv2D(base_filters * 2, (3, 3), padding="same", activation="relu"),
        name="enc_conv2d_2",
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name="enc_bn2d_2")(x)
    x = layers.TimeDistributed(layers.Dropout(dropout), name="enc_do2d_2")(x)

    x = layers.TimeDistributed(
        layers.Conv2D(base_filters * 4, (2, 2), padding="same", activation="relu"),
        name="enc_conv2d_3",
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name="enc_bn2d_3")(x)

    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name="enc_gap2d")(x)

    x = layers.MaxPool1D(2, padding="same", name="enc_tpool1")(x)
    x = layers.Dropout(dropout, name="enc_tdo1")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_tpool2")(x)

    seq_emb = layers.Conv1D(
        emb_dim, 1, padding="same", activation=None, name="seq_emb"
    )(x)

    return Model(inputs=x_in, outputs=seq_emb, name="encoder_2dcnn")


def training_autoencoder(
    encoder: tf.keras.Model,
    timesteps: int,
    n_features: int,
    base_filters: int = 64,
    kernel_size: int = 7,
    dropout: float = 0.10,
    lr: float = 1e-3,
    loss: str = "mse",
) -> tf.keras.Model:
    """Attach a 1D convolutional decoder to a sequence encoder and compile an autoencoder.

    Builds a symmetric 1D decoder that mirrors the temporal structure of
    ``encoder_1dcnn``: two ``UpSampling1D`` layers restore the time axis from
    ``ceil(T / 4)`` back to ``T``, and a final ``Conv1D`` projects each
    timestep back to ``n_features``. A length-correction layer trims or pads
    the output to exactly ``timesteps`` in case upsampling introduces
    off-by-one differences.

    The decoder reads its input shape directly from ``encoder.output_shape``,
    so it is agnostic to which encoder is passed. All functions in ``unsupervised.py``
    produce ``(batch, ceil(T / 4), emb_dim)`` and are, therefore,
    fully interchangeable as arguments to this function.

    Parameters
    ----------
    encoder : tf.keras.Model
        A compiled or uncompiled Keras Model whose output shape is
        ``(batch, t_down, emb_dim)``. Usually should be the returned
        models from previous functions
    timesteps : int
        Original input sequence length T. Used by the length-correction layer
        to guarantee the reconstruction has exactly this many timesteps.
    n_features : int
        Number of features per timestep to reconstruct. Must match the feature
        dimension of the data passed to ``autoencoder.fit``.
    base_filters : int, optional
        Number of filters in the final decoder Conv1D block. Earlier blocks
        use ``base_filters * 2`` and ``base_filters * 4``. Default is 64.
    kernel_size : int, optional
        Kernel size used in the second-to-last decoder Conv1D block
        (``kernel_size // 2 + 1``). Default is 7.
    dropout : float, optional
        Dropout rate applied in the two intermediate decoder blocks.
        Default is 0.10.
    lr : float, optional
        Learning rate for the Adam optimiser. Default is 1e-3.
    loss : str, optional
        Keras loss identifier passed to ``model.compile``. Default is ``"mse"``.

    Returns
    -------
    tf.keras.Model
        Compiled Keras Model mapping ``(batch, timesteps, n_features)`` to a
        reconstruction of the same shape, optimised with Adam and monitored
        with mean absolute error alongside the primary loss.
    """
    seq_in = layers.Input(shape=encoder.output_shape[1:], name="seq_in")

    y = layers.Conv1D(
        base_filters * 4, 3, padding="same", activation="relu", name="dec_conv0"
    )(seq_in)
    y = layers.BatchNormalization(name="dec_bn0")(y)

    y = layers.UpSampling1D(2, name="dec_up1")(y)
    y = layers.Conv1D(
        base_filters * 2, 3, padding="same", activation="relu", name="dec_conv1"
    )(y)
    y = layers.BatchNormalization(name="dec_bn1")(y)
    y = layers.Dropout(dropout, name="dec_do1")(y)

    y = layers.UpSampling1D(2, name="dec_up2")(y)
    y = layers.Conv1D(
        base_filters,
        kernel_size // 2 + 1,
        padding="same",
        activation="relu",
        name="dec_conv2",
    )(y)
    y = layers.BatchNormalization(name="dec_bn2")(y)
    y = layers.Dropout(dropout, name="dec_do2")(y)

    x_hat = layers.Conv1D(n_features, 1, padding="same", activation=None, name="x_hat")(
        y
    )

    def fix_len(t):
        t = t[:, :timesteps, :]
        cur = tf.shape(t)[1]
        pad = tf.maximum(0, timesteps - cur)
        return tf.pad(t, [[0, 0], [0, pad], [0, 0]])

    x_hat = layers.Lambda(fix_len, name="fix_length")(x_hat)

    decoder = Model(seq_in, x_hat, name="decoder_seq")

    autoencoder = Model(encoder.input, decoder(encoder.output), name="autoencoder_seq")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    return autoencoder


def encoder_gcn(
    timesteps: int,
    n_channels: int = 14,
    n_bands: int = 6,
    gcn_units: tuple[int] = (32, 64),
    emb_dim: int = 128,
    dropout: float = 0.10,
) -> tf.keras.Model:
    # Partially generated with AI assistance (Claude, Anthropic) — reviewed and adapted.
    """Build a graph convolutional sequence encoder with a learned adjacency matrix.

    At each timestep the flat feature vector is reshaped into a node feature
    matrix ``X_t ∈ R^{N x F}`` (channels x bands). A stack of
    ``GraphConv`` layers — each with its own learnable adjacency
    matrix ``A_theta`` — performs neighbourhood aggregation across channels,
    allowing the model to discover functional connectivity patterns that are
    invisible to Conv1D or Conv2D encoders which assume a fixed or regular
    topology. All graph layers are applied via ``TimeDistributed`` so that the
    same weights operate independently at each timestep.

    After graph feature extraction, ``GlobalAveragePooling`` collapses the
    node dimension to a single vector per timestep. Two ``MaxPool1D`` layers
    then compress the time axis by a factor of 4, and a final ``Dense``
    projection maps each timestep to ``emb_dim``.

    Parameters
    ----------
    timesteps : int
        Number of timesteps T.
    n_channels : int, optional
        Number of EEG electrode nodes N. Default is 14 (DREAMER dataset).
    n_bands : int, optional
        Number of frequency bands F per electrode, forming the initial node
        feature dimension. Default is 6.
    gcn_units : tuple of int, optional
        Output dimensionality of each successive ``GraphConv``
        layer. The length of this tuple determines the number of GCN layers.
        Default is ``(32, 64)``.
    emb_dim : int, optional
        Dimensionality of the final output embedding per downsampled timestep.
        Default is 128.
    dropout : float, optional
        Dropout rate applied after each GCN layer and after the first temporal
        pooling step. Default is 0.10.

    Returns
    -------
    tf.keras.Model
        Keras Model with input shape ``(batch, timesteps, n_channels * n_bands)``
        and output shape ``(batch, ceil(timesteps / 4), emb_dim)``. The output
        shape will also fit in training_autoencoder.

    Notes
    -----
    Each ``GraphConv`` layer maintains its own ``A_theta ∈
    R^{N x N}`` adjacency matrix, initialized with Glorot uniform weights and
    updated end-to-end during training. With ``n_channels=14`` this adds
    only ``len(gcn_units) * 196`` parameters to the model — negligible
    overhead relative to the projection weights.

    If electrode topology priors are available (e.g. inverse scalp distances
    from the 10-20 system), each layer's adjacency matrix can be warm-started
    after construction:

        enc = encoder_gcn(...)
        prior = compute_distance_matrix()   # (14, 14)
        enc.get_layer("gcn_0").A_theta.assign(prior)

    ``TimeDistributed`` enforces that the learned graph structure is
    stationary across timesteps.
    """
    n_features = n_channels * n_bands
    x_in = layers.Input(shape=(timesteps, n_features), name="x")

    x = layers.Reshape((timesteps, n_channels, n_bands), name="to_nodes")(x_in)

    for i, units in enumerate(gcn_units):
        x = layers.TimeDistributed(
            GraphConv(units=units, n_nodes=n_channels),
            name=f"gcn_{i}",
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization(), name=f"gcn_bn_{i}")(x)
        x = layers.TimeDistributed(layers.Dropout(dropout), name=f"gcn_do_{i}")(x)

    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=2), name="gcn_pool_nodes")(x)

    x = layers.MaxPool1D(2, padding="same", name="enc_tpool1")(x)
    x = layers.Dropout(dropout, name="enc_tdo1")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_tpool2")(x)

    seq_emb = layers.Dense(emb_dim, activation=None, name="seq_emb")(x)

    return Model(inputs=x_in, outputs=seq_emb, name="encoder_gcn")


def encoder_lstm(
    timesteps: int,
    n_features: int,
    lstm_units: int = 128,
    n_lstm_layers: int = 2,
    emb_dim: int = 128,
    dropout: float = 0.10,
    n_classes: int | None = None,
    include_softmax: bool = True,
) -> tf.keras.Model:
    """Build a unidirectional LSTM sequence encoder.

    Stacks ``n_lstm_layers`` LSTM layers that each pass the full sequence
    forward through time (left-to-right only). After the LSTM stack, two
    ``MaxPool1D`` layers compress the time axis by a factor of 4 and a
    ``Dense`` layer maps each downsampled timestep to ``emb_dim``.

    The encoder can run in two modes controlled by ``n_classes``:

    * **Encoder-only mode** (``n_classes=None``): output shape is
      ``(batch, ceil(timesteps / 4), emb_dim)``, identical to
      ``encoder_1dcnn`` and compatible with ``training_autoencoder``.
    * **Classifier mode** (``n_classes`` is an integer): a
      ``GlobalAveragePooling1D`` step collapses the time axis into a single
      vector, then a ``Dense(n_classes)`` layer maps to class scores.
      A ``Softmax`` is appended when ``include_softmax=True``.

    Parameters
    ----------
    timesteps : int
        Number of timesteps T in the input sequence.
    n_features : int
        Number of features per timestep (e.g. 84 for 14 electrodes x 6 bands).
    lstm_units : int, optional
        Number of hidden units in each LSTM layer. Default is 128.
    n_lstm_layers : int, optional
        How many LSTM layers to stack. Each layer (except the last) feeds its
        full sequence output to the next. Default is 2.
    emb_dim : int, optional
        Dimensionality of the output embedding at each downsampled timestep.
        Only used in encoder-only mode. Default is 128.
    dropout : float, optional
        Dropout rate applied after each LSTM layer. Default is 0.10.
    n_classes : int or None, optional
        When set to an integer the encoder becomes a full classifier.
        A ``GlobalAveragePooling1D`` collapses the sequence and a
        ``Dense(n_classes)`` layer produces class scores. Default is None
        (encoder-only mode).
    include_softmax : bool, optional
        Whether to append a ``Softmax`` activation in classifier mode.
        Set to ``False`` if your loss uses ``from_logits=True``.
        Ignored when ``n_classes=None``. Default is True.

    Returns
    -------
    tf.keras.Model
        Keras Model with input shape ``(batch, timesteps, n_features)``.
        Output shape depends on ``n_classes``:

        * Encoder-only: ``(batch, ceil(timesteps / 4), emb_dim)``
        * Classifier:   ``(batch, n_classes)``

    Examples
    --------
    Encoder-only (use with training_autoencoder)::

        enc = encoder_lstm(timesteps=128, n_features=84)
        autoencoder = training_autoencoder(enc, timesteps=128, n_features=84)

    Classifier (use with run_cross_validation)::

        def build_model():
            model = encoder_lstm(
                timesteps=128, n_features=84,
                n_classes=3, include_softmax=True
            )
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            return model
    """
    x_in = layers.Input(shape=(timesteps, n_features), name="x")

    x = x_in
    for layer_index in range(n_lstm_layers):
        # All LSTM layers return the full sequence so the next layer or
        # pooling step can work across all timesteps.
        x = layers.LSTM(
            lstm_units,
            return_sequences=True,
            name=f"lstm_{layer_index}",
        )(x)
        x = layers.BatchNormalization(name=f"lstm_bn_{layer_index}")(x)
        x = layers.Dropout(dropout, name=f"lstm_do_{layer_index}")(x)

    # Compress the time axis by 4 (same factor as the CNN encoders) so the
    # output is compatible with training_autoencoder.
    x = layers.MaxPool1D(2, padding="same", name="enc_tpool1")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_tpool2")(x)

    if n_classes is None:
        # ---- Encoder-only mode ----------------------------------------
        # Project each downsampled timestep to emb_dim.
        output = layers.Dense(emb_dim, activation=None, name="seq_emb")(x)
        model_name = "encoder_lstm"
    else:
        # ---- Classifier mode ------------------------------------------
        # Collapse all timesteps into one vector by averaging, then classify.
        x = layers.GlobalAveragePooling1D(name="gap")(x)
        x = layers.Dense(n_classes, name="class_logits")(x)
        if include_softmax:
            output = layers.Softmax(name="class_probabilities")(x)
        else:
            output = x  # raw logits, no softmax
        model_name = "encoder_lstm_classifier"

    return Model(inputs=x_in, outputs=output, name=model_name)


def encoder_bilstm(
    timesteps: int,
    n_features: int,
    lstm_units: int = 128,
    n_bilstm_layers: int = 2,
    emb_dim: int = 128,
    dropout: float = 0.10,
    n_classes: int | None = None,
    include_softmax: bool = True,
) -> tf.keras.Model:
    """Build a bidirectional LSTM sequence encoder.

    Identical in structure to ``encoder_lstm`` but wraps each LSTM in
    ``tf.keras.layers.Bidirectional``, which processes the sequence both
    left-to-right *and* right-to-left simultaneously. The two directional
    outputs are concatenated, so each LSTM layer outputs ``lstm_units * 2``
    features per timestep instead of ``lstm_units``.

    Processing both directions lets the model use context from both the past
    *and* the future at every timestep, which is especially useful for EEG
    signals where a brain-state event at time t is often visible in the signal
    both before and after its peak.

    Parameters
    ----------
    timesteps : int
        Number of timesteps T in the input sequence.
    n_features : int
        Number of features per timestep.
    lstm_units : int, optional
        Number of hidden units in *each direction* of the LSTM. The actual
        feature dimension after each BiLSTM layer is ``lstm_units * 2`` due
        to concatenation of the two directional outputs. Default is 128.
    n_bilstm_layers : int, optional
        How many bidirectional LSTM layers to stack. Default is 2.
    emb_dim : int, optional
        Dimensionality of the output embedding per downsampled timestep in
        encoder-only mode. Default is 128.
    dropout : float, optional
        Dropout rate applied after each BiLSTM layer. Default is 0.10.
    n_classes : int or None, optional
        When set, switches the model to classifier mode: collapses the time
        axis with ``GlobalAveragePooling1D`` and appends a
        ``Dense(n_classes)`` head. Default is None (encoder-only mode).
    include_softmax : bool, optional
        Whether to append ``Softmax`` in classifier mode. Set to ``False``
        when using a loss with ``from_logits=True``. Default is True.

    Returns
    -------
    tf.keras.Model
        Keras Model with input shape ``(batch, timesteps, n_features)``.
        Output shape:

        * Encoder-only: ``(batch, ceil(timesteps / 4), emb_dim)``
        * Classifier:   ``(batch, n_classes)``

    Examples
    --------
    Classifier for a 3-class EEG task, used with LOSO cross-validation::

        def build_model():
            model = encoder_bilstm(
                timesteps=128, n_features=84,
                lstm_units=64, n_classes=3, include_softmax=True
            )
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            return model

        results = run_cross_validation(
            cv_strategy="loso",
            model_builder_function=build_model,
            feature_array=X,
            label_array=y,
            subject_id_array=subject_ids,
        )
    """
    x_in = layers.Input(shape=(timesteps, n_features), name="x")

    x = x_in
    for layer_index in range(n_bilstm_layers):
        # Bidirectional concatenates the forward and backward LSTM outputs,
        # so the output width is lstm_units * 2.
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True),
            merge_mode="concat",  # forward + backward outputs are concatenated
            name=f"bilstm_{layer_index}",
        )(x)
        x = layers.BatchNormalization(name=f"bilstm_bn_{layer_index}")(x)
        x = layers.Dropout(dropout, name=f"bilstm_do_{layer_index}")(x)

    # Compress time axis by 4 — same as the CNN encoders.
    x = layers.MaxPool1D(2, padding="same", name="enc_tpool1")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_tpool2")(x)

    if n_classes is None:
        # ---- Encoder-only mode ----------------------------------------
        output = layers.Dense(emb_dim, activation=None, name="seq_emb")(x)
        model_name = "encoder_bilstm"
    else:
        # ---- Classifier mode ------------------------------------------
        x = layers.GlobalAveragePooling1D(name="gap")(x)
        x = layers.Dense(n_classes, name="class_logits")(x)
        if include_softmax:
            output = layers.Softmax(name="class_probabilities")(x)
        else:
            output = x
        model_name = "encoder_bilstm_classifier"

    return Model(inputs=x_in, outputs=output, name=model_name)
