"""CPU-only smoke checks for actual raw-array preparation and tracker bridging."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from eegproc.deep_learning.joint_architectures.SICModelv16.data import prepare_data
from eegproc.deep_learning.joint_architectures.SICModelv16.run_experiments import run, prediction_diagnostics, fold_subjects
from training_tracker.core import load_campaign, plan_run, scan_runs

ROOT = Path(__file__).resolve().parents[2]


class DataAndTrackerTests(unittest.TestCase):
    def test_data_pool_excludes_other_subjects_before_filtering(self):
        # Poison the excluded subjects: neither validation nor filtering may
        # inspect their EEG/ratings when a subset was explicitly requested.
        eeg = np.ones((6, 2, 14, 256), dtype="float32")
        eeg[4:] = np.nan
        labels = np.tile(np.array([1., 5.])[None, :, None], (6, 1, 3))
        labels[4:] = np.nan
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root/"eeg.npy", eeg)
            np.save(root/"labels.npy", labels)
            with patch("eegproc.deep_learning.prepare_datasets._filter_dreamer_trial",
                       side_effect=lambda raw: np.repeat(raw, 3, axis=-1)) as filtering:
                available = prepare_data(root/"eeg.npy", root/"labels.npy", root/"cache",
                    expected_subjects=6, data_subjects=[0, 1, 2, 3])
            self.assertEqual(available, [0, 1, 2, 3])
            self.assertEqual(filtering.call_count, 8)
            subjects = np.load(root/"cache/subjects.npy")
            self.assertEqual(set(subjects.tolist()), {0, 1, 2, 3})
            self.assertEqual(np.load(root/"cache/features.npy").shape, (8, 2, 128, 42))
            for target in available:
                train, validation = fold_subjects(subjects, target, 1, 42)
                self.assertEqual(len(train), 2)
                self.assertEqual(len(validation), 1)
                self.assertEqual(set(train + validation + [target]), {0, 1, 2, 3})

    def test_banded_subset_preserves_original_subject_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root/"eeg.npy", np.ones((6, 2, 42, 128), dtype="float32"))
            np.save(root/"labels.npy", np.tile(np.array([1., 5.])[None, :, None], (6, 1, 3)))
            prepare_data(root/"eeg.npy", root/"labels.npy", root/"cache",
                         expected_subjects=6, data_subjects=[1, 3])
            np.testing.assert_array_equal(np.load(root/"cache/subjects.npy"), [1, 1, 3, 3])

    def test_single_class_prediction_diagnostics(self):
        single = prediction_diagnostics([0, 1], [[.4, .6], [.3, .7]])
        self.assertTrue(single["single_class_predictions"])
        self.assertEqual(single["predicted_class_1_fraction"], 1.0)
        self.assertEqual(single["true_class_1_fraction"], .5)
        mixed = prediction_diagnostics([0, 1], [[.6, .4], [.3, .7]])
        self.assertFalse(mixed["single_class_predictions"])

    def test_raw_and_prefiltered_arrays_agree(self):
        from eegproc.deep_learning.prepare_datasets import _filter_dreamer_trial
        # Distinct rhythms on the first electrodes expose channel/band swaps.
        time = np.arange(256) / 128.0
        channels = np.stack([np.sin(2*np.pi*(6, 10, 20)[index % 3]*time)
                             for index in range(14)])
        eeg = np.tile(channels[None, None], (3, 2, 1, 1)).astype("float32")
        labels = np.tile(np.array([1., 5.])[None, :, None], (3, 1, 3))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root / "raw.npy", eeg)
            np.save(root / "labels.npy", labels)
            subjects = prepare_data(root / "raw.npy", root / "labels.npy", root / "raw_cache", expected_subjects=3)
            self.assertEqual(subjects, [0, 1, 2])
            np.testing.assert_array_equal(np.load(root / "raw.npy"), eeg)
            filtered = _filter_dreamer_trial(eeg[0, 0].T)
            energy = np.mean(filtered ** 2, axis=0).reshape(14, 3)
            np.testing.assert_array_equal(np.argmax(energy[:3], axis=-1), [0, 1, 2])
            banded = np.tile(filtered.T[None, None], (3, 2, 1, 1))
            np.save(root / "banded.npy", banded)
            prepare_data(root / "banded.npy", root / "labels.npy", root / "band_cache", expected_subjects=3)
            for name in ("features", "labels", "subjects", "trials"):
                np.testing.assert_array_equal(np.load(root / "raw_cache" / f"{name}.npy"),
                                              np.load(root / "band_cache" / f"{name}.npy"))
            self.assertEqual(np.load(root / "raw_cache/features.npy").shape, (6, 2, 128, 42))
            self.assertFalse((root / "raw_cache/filtered_trials.tmp.npy").exists())

    def test_preparation_failure_leaves_report(self):
        module = "eegproc.deep_learning.joint_architectures.SICModelv16.run_experiments"
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "failed"
            args = argparse.Namespace(configs=ROOT / "src/eegproc/deep_learning/joint_architectures/SICModelv16/configurations.json",
                config_index=0, source_epochs=2, batch_size=1, time_budget_seconds=600,
                output_dir=output, eeg=Path(folder)/"missing.npy", labels=Path(folder)/"missing_labels.npy")
            with patch(module + ".prepare_data", side_effect=ValueError("bad input")):
                with self.assertRaisesRegex(ValueError, "bad input"):
                    run(args)
            report = json.loads((output / "execution.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["stage"], "data_preparation")
            self.assertIn("bad input", (output / "failure.txt").read_text())

    def test_tracker_plan_uses_v16_and_smoke_budget(self):
        campaign = load_campaign(ROOT / "experiments/sic_v16/cnn_transformer_reconstruction/campaign.json")
        with tempfile.TemporaryDirectory() as folder:
            path = plan_run(campaign, Path(folder) / "runs", "valence")
            config = json.loads((path / "config.json").read_text())
            self.assertTrue(config["use_transformer"])
            self.assertTrue(config["use_decoder"])
            self.assertEqual(config["epochs"], 2)
            self.assertEqual(config["target_subjects"], [0, 1])
            script = (path / "submit.slurm").read_text()
            self.assertIn("experiments/sic_v16/train.py", script)
            self.assertIn("#SBATCH --time=02:00:00", script)
            self.assertNotIn("{{", script)
            self.assertEqual(scan_runs(path.parent, campaign)[0]["status"], "planned")

    def test_first4_tracker_pool_is_separate_from_old_campaign(self):
        campaign = load_campaign(ROOT / "experiments/sic_v16/first4/cnn/campaign.json")
        original = load_campaign(ROOT / "experiments/sic_v16/cnn/campaign.json")
        self.assertNotEqual(campaign["campaign"], original["campaign"])
        self.assertEqual(campaign["defaults"]["data_subjects"], [0, 1, 2, 3])
        self.assertEqual(campaign["defaults"]["target_subjects"], [0, 1, 2, 3])
        self.assertEqual(campaign["defaults"]["validation_subjects"], 1)

    def test_tracker_result_uses_validation_not_oracle(self):
        spec = importlib.util.spec_from_file_location("v16_tracked_train", ROOT / "experiments/sic_v16/train.py")
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        summary = {"validation_selected": {"status": "completed", "selection_rule": "validation",
            "folds": [{"target_subject": 0, "test_metrics": {"balanced_accuracy": .4}},
                      {"target_subject": 1, "test_metrics": {"balanced_accuracy": .6}}]},
            "oracle": {"status": "completed", "selection_score": .99}}
        result = bridge.tracker_result(summary)
        self.assertAlmostEqual(result["metrics"]["trial_balanced_accuracy"]["mean"], .5)
        self.assertFalse(result["oracle"])
        self.assertEqual([row["subject_id"] for row in result["subject_metrics"]], [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
