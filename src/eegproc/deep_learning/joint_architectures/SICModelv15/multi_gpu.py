"""Synchronous, shared-variable data parallelism for v15 source MLDG.

All devices read the same model variables. TensorFlow copies parameter reads
to each device and differentiates one global loss through all device branches.
Consequently gradients sum at the shared variables before the existing single
MLDG inner update and outer optimizer update. No local VC statistics, replica
optimizers, gradient caching, or temporal truncation are introduced.

Only trial embeddings/logits and small reconstruction/reporting tensors are
gathered; the large encoder, recurrent, and decoder activations stay on the
device that computed them. Keras Model.fit's normal graph executor can schedule
the independent device branches concurrently. Reproducibility mode instead
runs this same Python device loop eagerly and synchronously, giving the shards
a fixed execution order without moving their activations off-device.
"""

from __future__ import annotations

import tensorflow as tf


class FoldGPUMemory(tf.keras.callbacks.Callback):
    """Report actual source-training memory on each GPU in the fold."""

    def __init__(self, devices):
        super().__init__()
        self.devices = tuple(
            device.removeprefix("/device:") for device in devices if "GPU:" in device
        )

    def on_train_begin(self, logs=None):
        for device in self.devices:
            try:
                tf.config.experimental.reset_memory_stats(device)
            except (ValueError, RuntimeError):
                pass  # Allocator statistics are diagnostic, not a prerequisite.

    def on_epoch_end(self, epoch, logs=None):
        for device in self.devices:
            try:
                info = tf.config.experimental.get_memory_info(device)
            except (ValueError, RuntimeError):
                continue
            print(
                f"[MLDG epoch {epoch + 1}] {device} memory: "
                f"current={info['current'] / 2**30:.2f} GiB, "
                f"peak={info['peak'] / 2**30:.2f} GiB",
                flush=True,
            )


def _call_with_device_gradient(function, inputs, device):
    """Keep both the ordinary forward and its exact VJP on one device.

    A forward device scope alone does not explicitly place GradientTape's
    subsequently constructed backward ops. Retain a local tape and place its
    backward call too. This performs no recomputation or gradient truncation.
    Only the resulting parameter gradients are reduced at the shared weights.
    """
    structure = {}

    @tf.custom_gradient
    def call(*values):
        with tf.device(device):
            # MLDG takes one derivative of each forward. Consume this tape
            # once so its saved activations can be released after backward.
            with tf.GradientTape() as tape:
                tape.watch(values)
                result = function(*values)
                names = tuple(
                    key for key, value in result.items()
                    if tf.is_tensor(value) and value.dtype.is_floating
                )
                tensors = tuple(result[key] for key in names)
            structure["names"] = names
            structure["other"] = {
                key: value for key, value in result.items() if key not in names
            }

        def backward(*output_gradients, variables=None):
            with tf.device(device):
                variable_list = [] if variables is None else list(variables)
                gradients = tape.gradient(
                    tensors, list(values) + variable_list,
                    output_gradients=output_gradients,
                )
            input_gradients = tuple(gradients[:len(values)])
            if variables is None:
                return input_gradients
            return input_gradients, gradients[len(values):]

        return tensors, backward

    with tf.device(device):
        tensors = call(*inputs)
    return dict(structure["other"], **dict(zip(structure["names"], tensors)))


def validate_fold_devices(model, devices):
    devices = tuple(str(device) for device in devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Fold devices must be a nonempty list of distinct devices.")
    available = {device.name for device in tf.config.list_logical_devices()}
    if any(device not in available for device in devices):
        raise ValueError(f"Requested fold devices {devices}; available={sorted(available)}.")
    if len(devices) > 1:
        if not model.use_mldg or model.classification_level != "trial":
            raise ValueError("Multi-GPU v15 requires trial-level MLDG source training.")
        if model.graph_encoder is not None and model.graph_encoder.use_batch_norm:
            raise ValueError(
                "Multi-GPU v15 requires gcn_use_batch_norm=false to preserve "
                "the full-episode computation."
            )
        for name, count in (
            ("meta-train", model.mldg_meta_train_subjects),
            ("meta-test", model.mldg_meta_test_subjects),
        ):
            if count is not None and count * model.mldg_trials_per_subject < len(devices):
                raise ValueError(f"Each {name} group must contain at least one trial per GPU.")
    return devices


def encode_on_devices(model, eeg_inputs, devices, *, training):
    """Split complete trials, keeping the original order and global VC batch."""
    n_trials = tf.shape(eeg_inputs)[0]
    n_devices = len(devices)
    tf.debugging.assert_greater_equal(
        n_trials, n_devices, message="Every MLDG device needs at least one complete trial."
    )
    sizes = n_trials // n_devices + tf.cast(
        tf.range(n_devices) < n_trials % n_devices, tf.int32
    )
    trial_shards = tf.split(eeg_inputs, sizes, axis=0, num=n_devices)
    outputs = []
    for device, trials in zip(devices, trial_shards):
        with tf.device(device):
            outputs.append(_call_with_device_gradient(
                lambda value: model._encode(value, training=training),
                (tf.identity(trials),), device,
            ))

    # Never gather graph_sequence, bilstm_sequence, or classifier_sequence:
    # those activations are the reason each fold needs multiple GPUs.
    with tf.device(devices[0]):
        combined = {
            key: tf.concat([output[key] for output in outputs], axis=0)
            for key in (
                "classification_embedding", "logits", "probabilities",
                "pooled_features", "flat_windows",
            )
        }
    combined["device_shards"] = tuple(zip(devices, outputs))
    return combined


def reconstruction_on_devices(model, outputs, *, training):
    """Compute each reconstruction locally and reduce using element counts."""
    shards = outputs["device_shards"]
    components = []
    counts = []
    for device, local_outputs in shards:
        with tf.device(device):
            # Pass each differentiable input once, even in single-branch
            # ablations where combined_feature_sequence aliases that branch.
            names = tuple(key for key in (
                "flat_windows", "graph_sequence", "bilstm_sequence"
            ) if local_outputs[key] is not None)

            def reconstruct(*values, names=names):
                local = dict.fromkeys(("graph_sequence", "bilstm_sequence"))
                local.update(zip(names, values))
                local["combined_feature_sequence"] = (
                    local["graph_sequence"] if local["graph_sequence"] is not None
                    else local["bilstm_sequence"]
                )
                return model._reconstruction_components(local, training)

            components.append(_call_with_device_gradient(
                reconstruct, tuple(local_outputs[key] for key in names), device,
            ))
            counts.append(tf.size(local_outputs["flat_windows"]))

    with tf.device(shards[0][0]):
        dtype = components[0]["reconstruction_loss"].dtype
        weights = tf.cast(tf.stack(counts), dtype)
        weights /= tf.reduce_sum(weights)
        result = {}
        for key, first_value in components[0].items():
            if first_value is None:
                result[key] = None
            elif first_value.shape.rank == 0:
                result[key] = tf.add_n([
                    weights[index] * component[key]
                    for index, component in enumerate(components)
                ])
            else:
                result[key] = tf.concat([component[key] for component in components], axis=0)
        # The minimum must be taken after reducing both branch losses.
        if model.use_joint_reconstruction:
            result["joint_reconstruction_gain_vs_best_branch"] = tf.minimum(
                result["gcn_gru_reconstruction_loss"],
                result["bilstm_reconstruction_loss"],
            ) - result["joint_reconstruction_loss"]
    return result
