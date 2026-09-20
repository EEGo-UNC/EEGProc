"""Full-objective parity checks; also runnable as the GPU smoke preflight.

EEGPROC_TEST_GPUS=1 python -m src.tests.test_sic_v15_multi_gpu
Default execution uses two logical CPUs, so no CUDA hardware is needed.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf


class V15MultiGPUChecks(unittest.TestCase):
    __test__ = False  # pytest runs these in a fresh TensorFlow process below.
    @classmethod
    def setUpClass(cls):
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        if os.environ.get("EEGPROC_TEST_GPUS") == "1":
            gpus = tf.config.list_physical_devices("GPU")
            if len(gpus) < 2:
                raise RuntimeError("GPU preflight requires at least two real GPUs.")
            tf.config.set_visible_devices(gpus[:2], "GPU")
            for gpu in gpus[:2]:
                tf.config.experimental.set_memory_growth(gpu, True)
            cls.devices = tuple(d.name for d in tf.config.list_logical_devices("GPU"))
        else:
            tf.config.set_visible_devices([], "GPU")
            cpu = tf.config.list_physical_devices("CPU")[0]
            tf.config.set_logical_device_configuration(
                cpu, [tf.config.LogicalDeviceConfiguration() for _ in range(2)]
            )
            cls.devices = tuple(d.name for d in tf.config.list_logical_devices("CPU"))
        from src.eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model
        cls.builder = staticmethod(build_sic_model)

    def setUp(self):
        tf.keras.utils.set_random_seed(724)
        self.eeg = tf.constant(np.random.default_rng(18).normal(size=(12, 2, 4, 6)), tf.float32)
        # Each device's training shard has only one class; per-device VC KL
        # would therefore differ from the required full-episode objective.
        self.labels = tf.constant([0]*4 + [1]*4 + [0]*2 + [1]*2, tf.int32)
        self.subjects = tf.repeat(tf.range(6), 2)
        self.inputs = {"eeg": self.eeg, "subject_id": self.subjects,
                       "mldg_role": tf.constant([0]*8 + [1]*4)}
        self.weights = tf.constant(np.linspace(0.5, 1.5, 12), tf.float32)

    def build_model(self, parallel=False, dropout=0.0, deterministic=False):
        with tf.device(self.devices[0]):
            model = self.builder(
                (2, 4, 6), adjacency=np.eye(2), n_channels=2, n_bands=3,
                gcn_units=(4, 3), spectral_gru_units=4, bilstm_units=3,
                classifier_rnn_units=(4, 3), n_classifier_rnn_layers=2,
                n_subject_classes=6, training_method="mldg",
                mldg_meta_train_subjects=4, mldg_meta_test_subjects=2,
                mldg_trials_per_subject=2, mldg_steps_per_epoch=2,
                mldg_inner_learning_rate=1e-4, mldg_meta_test_weight=0.7,
                gcn_dropout=dropout, spectral_gru_dropout=dropout,
                bilstm_dropout=dropout, classifier_rnn_dropout=dropout,
                decoder_dropout=dropout, vc_beta=0.3, vc_gamma=0.2,
                vc_lambda=0.05, reconstruction_loss_weight=0.4,
                # Match the launch configuration. Discriminator optimization
                # samples fresh random priors and is disabled in this run.
                update_vc_discriminator=False, weight_decay=5e-5,
                use_class_weight=True,
            )
        if parallel:
            model.configure_fold_devices(
                self.devices,
                deterministic_execution=deterministic,
            )
        return model

    def assert_tensors_close(self, left, right, label=""):
        self.assertEqual(len(left), len(right))
        for index, (a, b) in enumerate(zip(left, right)):
            self.assertEqual(a is None, b is None, f"{label} gradient {index}")
            if a is not None:
                np.testing.assert_allclose(
                    tf.convert_to_tensor(a), tf.convert_to_tensor(b),
                    rtol=3e-4, atol=3e-5, err_msg=f"{label} tensor {index}",
                )

    def objective(self, model, start, stop, reconstruct):
        with tf.GradientTape() as tape:
            out = model._encode_mldg(self.eeg[start:stop], training=True)
            vc = model._vc_components(out["classification_embedding"], out["logits"],
                                      self.labels[start:stop], self.weights[start:stop], calibration=False)
            loss = model.vc_loss_weight * vc["total_loss"]
            if reconstruct:
                rec = model._reconstruction_components(out, training=True)
                sub = model._subject_components(out["pooled_features"], self.subjects[start:stop],
                                                training=True, use_grl=True)
                loss += model.reconstruction_loss_weight * rec["reconstruction_loss"]
                loss += model.subject_loss_weight * sub["subject_loss"]
                loss += model._regularization_loss(loss.dtype)
        return loss, tape.gradient(loss, model.trainable_variables)

    def test_full_objective_gradients_and_inner_update(self):
        baseline, parallel = self.build_model(), self.build_model(True)
        parallel.set_weights(baseline.get_weights())
        for stop in (8, 7):  # Includes unequal shard sizes and sample weights.
            a_loss, a_grad = self.objective(baseline, 0, stop, True)
            b_loss, b_grad = self.objective(parallel, 0, stop, True)
            self.assert_tensors_close([a_loss], [b_loss], "meta-train loss")
            self.assert_tensors_close(a_grad, b_grad, "meta-train gradient")
        for model, gradients in ((baseline, a_grad), (parallel, b_grad)):
            for variable, gradient in zip(model.trainable_variables, gradients):
                if gradient is not None:
                    variable.assign_sub(model.mldg_inner_learning_rate * tf.convert_to_tensor(gradient))
        self.assert_tensors_close(baseline.trainable_variables, parallel.trainable_variables, "inner weights")
        a_loss, a_grad = self.objective(baseline, 8, 12, False)
        b_loss, b_grad = self.objective(parallel, 8, 12, False)
        self.assert_tensors_close([a_loss], [b_loss], "meta-test loss")
        self.assert_tensors_close(a_grad, b_grad, "meta-test gradient")
        out = parallel._encode_mldg(self.eeg[:8], training=True)
        for device, shard in out["device_shards"]:
            self.assertTrue(shard["graph_sequence"].device.endswith(device.removeprefix("/device:")))
            self.assertTrue(shard["bilstm_sequence"].device.endswith(device.removeprefix("/device:")))

    def test_complete_mldg_updates_eager_and_graph(self):
        for graph in (False, True):
            baseline, parallel = self.build_model(), self.build_model(True)
            parallel.set_weights(baseline.get_weights())
            step_a, step_b = baseline._mldg_train_step, parallel._mldg_train_step
            if graph:
                step_a, step_b = tf.function(step_a), tf.function(step_b)
            for _ in range(2):
                step_a(self.inputs, self.labels, self.weights)
                step_b(self.inputs, self.labels, self.weights)
            if graph:
                graph_def = step_b.get_concrete_function(
                    self.inputs, self.labels, self.weights
                ).graph.as_graph_def()
                nodes = list(graph_def.node)
                for function in graph_def.library.function:
                    nodes.extend(function.node_def)
                for device in self.devices:
                    self.assertTrue(any(
                        node.device.endswith(device.removeprefix("/device:"))
                        and "gradient_tape" in node.name
                        and node.op in {"MatMul", "BatchMatMulV2", "CudnnRNNBackpropV3"}
                        for node in nodes
                    ), f"No compute-heavy backward operation placed on {device}")
            self.assert_tensors_close(baseline.weights, parallel.weights, f"final weights graph={graph}")
            self.assert_tensors_close(baseline.main_optimizer.variables, parallel.main_optimizer.variables, "AdamW state")
            self.assert_tensors_close([m.result() for m in baseline.metrics],
                                      [m.result() for m in parallel.metrics], "metrics")
            self.assertEqual(int(parallel.main_optimizer.iterations), 2)

    def test_fit_with_dropout_and_portable_checkpoint(self):
        model = self.build_model(True, dropout=0.1)
        # fit_sic_mldg needs source metadata, with both classes per subject.
        y = np.tile([0, 1], 6)
        model.set_source_training_metadata(self.subjects.numpy(), np.tile([0, 1], 6))
        x = model.prepare_fit_inputs(self.eeg.numpy(), self.subjects.numpy())
        model.fit(x, y, epochs=1, verbose=0)
        self.assertEqual(int(model.main_optimizer.iterations), 2)
        self.assertTrue(all(np.all(np.isfinite(v.numpy())) for v in model.weights))
        before = model(self.eeg, training=False).numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path, compile=False)
            self.assertFalse(getattr(restored, "_fold_devices", ()))
            np.testing.assert_allclose(before, restored(self.eeg, training=False), rtol=1e-5, atol=1e-6)

    def test_deterministic_fit_with_dropout_is_bitwise_repeatable(self):
        tf.config.experimental.enable_op_determinism()
        tf.config.experimental.set_synchronous_execution(True)

        def run_once():
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(1942)
            model = self.build_model(
                True,
                dropout=0.1,
                deterministic=True,
            )
            self.assertTrue(model.run_eagerly)
            y = np.tile([0, 1], 6)
            trial_ids = np.tile([0, 1], 6)
            model.set_source_training_metadata(
                self.subjects.numpy(),
                trial_ids,
            )
            x = model.prepare_fit_inputs(
                self.eeg.numpy(),
                self.subjects.numpy(),
            )
            history = model.fit(x, y, epochs=2, verbose=0)
            return (
                [np.asarray(value).copy() for value in model.weights],
                {
                    key: np.asarray(values).copy()
                    for key, values in history.history.items()
                },
            )

        first_weights, first_history = run_once()
        second_weights, second_history = run_once()
        self.assertEqual(first_history.keys(), second_history.keys())
        for name in first_history:
            np.testing.assert_array_equal(
                first_history[name],
                second_history[name],
                err_msg=f"History {name!r} is not bitwise repeatable.",
            )
        self.assertEqual(len(first_weights), len(second_weights))
        for index, (first, second) in enumerate(
            zip(first_weights, second_weights)
        ):
            np.testing.assert_array_equal(
                first,
                second,
                err_msg=f"Model tensor {index} is not bitwise repeatable.",
            )


class FoldAllocationChecks(unittest.TestCase):
    __test__ = False
    def test_subject_training_seed_is_stable_across_worker_layouts(self):
        from src.eegproc.deep_learning.cross_val import _subject_training_seed

        self.assertEqual(_subject_training_seed(42, 0), 42)
        self.assertEqual(_subject_training_seed(42, 7), 49)
        self.assertEqual(_subject_training_seed(42, np.int64(7)), 49)
        self.assertEqual(
            _subject_training_seed(42, "participant-7"),
            _subject_training_seed(42, "participant-7"),
        )
        self.assertIsNone(_subject_training_seed(None, 7))

    def test_disjoint_pairs_and_short_allocations(self):
        from src.eegproc.deep_learning.cross_val import _resolve_fold_gpu_groups
        self.assertEqual(_resolve_fold_gpu_groups(2, [0, 1, 2, 3], 2), (2, ((0, 1), (2, 3))))
        with self.assertRaises(ValueError):
            _resolve_fold_gpu_groups(4, [0, 1, 2, 3], 2)
        with self.assertRaises(ValueError):
            _resolve_fold_gpu_groups(2, [0, 1, 1, 2], 2)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c,GPU-d"}):
            self.assertEqual(_resolve_fold_gpu_groups(4, None, 2), (2, ((0, 1), (2, 3))))

    def test_child_mask_is_set_before_spawn_and_parent_restored(self):
        from src.eegproc.deep_learning.cross_val import _start_device_bound_process
        captured = {}
        class Context:
            def Process(self, **kwargs):
                captured.update(kwargs)
                return self
            def start(self):
                captured["mask"] = os.environ["CUDA_VISIBLE_DEVICES"]
        mask = "GPU-a,GPU-b,GPU-c,GPU-d"
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mask}):
            _start_device_bound_process(Context(), None, (), (2, 3), 2, "test")
            self.assertEqual(captured["mask"], "GPU-c,GPU-d")
            self.assertEqual(captured["args"][0], (0, 1))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], mask)


def test_v15_multi_gpu_in_subprocess():
    # Other EEGProc tests may already have initialized TensorFlow. Device
    # topology and thread settings must be established in a fresh process.
    with tempfile.TemporaryDirectory() as directory:
        env = dict(os.environ, MPLCONFIGDIR=directory, TF_CPP_MIN_LOG_LEVEL="3")
        result = subprocess.run(
            [sys.executable, "-m", "src.tests.test_sic_v15_multi_gpu"],
            cwd=Path(__file__).resolve().parents[2], env=env,
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    unittest.main(verbosity=2)
