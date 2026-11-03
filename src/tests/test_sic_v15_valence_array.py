"""Check the fixed full-run launch and oracle/calibration routing without GPUs."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SIC = ROOT / "src/eegproc/deep_learning/joint_architectures/SICModelv15"
SCRIPT = SIC / "SLURM_scripts/full_run_v15_valence.sh"
MANIFEST = SIC / "sic_v15_valence_run_5168991_12.json"


class ValenceArrayChecks(unittest.TestCase):
    def test_only_the_pasted_run_configuration_is_available(self):
        source = SCRIPT.read_text()
        self.assertIn("#SBATCH --array=0\n", source)
        self.assertIn("full_run_v15_valence_%A_%a.out", source)
        spec = importlib.util.spec_from_file_location("oracle_sweep", SIC / "sic_v15_valence_sweep.py")
        sweep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sweep)
        records = sweep.configurations(MANIFEST)
        self.assertEqual(len(records), 1)
        config = records[0]["hyperparameters"]
        for key, expected in {
            "focal_gamma": .5, "subject_loss_weight": .3,
            "reconstruction_loss_weight": 0., "mldg_meta_train_subjects": 12,
            "mldg_meta_test_subjects": 6, "mldg_trials_per_subject": 2,
            "mldg_steps_per_epoch": 10, "vc_alpha": 2., "vc_beta": .6,
            "vc_lambda": .05, "vc_logit_scale": 16.,
            "gcn_units": [128, 64], "classifier_rnn_units": [128, 64],
            "classifier_rnn_dropout": .4, "remove_median_label": False,
        }.items():
            self.assertEqual(config[key], expected)
        self.assertFalse(any(isinstance(value, dict) for value in config.values()))

    def test_missing_or_invalid_task_fails_before_cluster_setup(self):
        for task in (None, "-1", "1", "12", "35", "abc", "01"):
            with self.subTest(task=task):
                env = {k: v for k, v in os.environ.items() if k != "SLURM_ARRAY_TASK_ID"}
                if task is not None:
                    env["SLURM_ARRAY_TASK_ID"] = task
                result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("SLURM_ARRAY_TASK_ID must be 0", result.stderr)
                self.assertNotIn("module: command not found", result.stderr)

    def test_shell_uses_selected_models_and_stops_if_source_run_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project checkout"
            venv = project / "venv312"
            binaries = venv / "bin"
            binaries.mkdir(parents=True)
            python = binaries / "python"
            python.write_text(f"#!{sys.executable}\n" + "\n".join([
                "import json, os, pathlib, sys",
                "if len(sys.argv) > 2 and sys.argv[1].endswith('sic_v15_valence_sweep.py'):",
                "    if sys.argv[2] == 'list':",
                f"        print(pathlib.Path({str(MANIFEST)!r}).read_text())",
                "    else:",
                "        with pathlib.Path(os.environ['TRAINING_CAPTURE']).open('a') as handle:",
                "            handle.write(json.dumps(sys.argv[1:]) + '\\n')",
                "        if sys.argv[2] == 'run':",
                "            sys.exit(int(os.environ.get('SOURCE_RETURN_CODE', '0')))",
                "elif sys.argv[1:3] == ['-m', 'src.tests.test_sic_v15_multi_gpu']:",
                "    pass",
                "elif sys.argv[1:] == ['-']:",
                "    pass  # GPU visibility preflight",
                "else:",
                f"    os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])",
            ]) + "\n")
            python.chmod(0o755)
            for name in ("module", "nvidia-smi", "git"):
                command = binaries / name
                command.write_text("#!/bin/bash\nexit 0\n")
                command.chmod(0o755)
            (binaries / "activate").write_text(
                f"export VIRTUAL_ENV={shlex.quote(str(venv))}\n"
                f"export PATH={shlex.quote(str(binaries))}:\"$PATH\"\n")
            libdevice = venv / "lib/python3.12/site-packages/nvidia/nvvm/libdevice/libdevice.10.bc"
            libdevice.parent.mkdir(parents=True)
            libdevice.touch()
            for code in (0, 2):
                capture = project / f"capture_{code}.jsonl"
                env = {**os.environ, "PROJECT_DIR": str(project), "VENV_DIR": str(venv),
                       "SLURM_ARRAY_JOB_ID": "100", "SLURM_JOB_ID": "200",
                       "SLURM_ARRAY_TASK_ID": "0", "TRAINING_CAPTURE": str(capture),
                       "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                       "TRAINING_SEED": "99", "SOURCE_EPOCHS": "12", "CALIBRATION_EPOCHS": "1",
                       "SOURCE_RETURN_CODE": str(code)}
                env.pop("OUTPUT_DIR", None)
                result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                calls = [json.loads(line) for line in capture.read_text().splitlines()]
                self.assertEqual(len(calls), 2 if code == 0 else 1)
                args = calls[0]
                def value(flag):
                    return args[args.index(flag) + 1]
                self.assertEqual(args[1], "run")
                self.assertEqual(value("--source-epochs"), "12")
                self.assertEqual(value("--source-batch-size"), "64")
                self.assertEqual(value("--config-index"), "0")
                self.assertTrue(value("--configs").endswith(MANIFEST.name))
                self.assertEqual(value("--seed"), "99")
                self.assertIn("--deterministic-training", args)
                self.assertIn("suite_100/task_0", value("--output-dir"))
                if code == 0:
                    self.assertEqual(calls[1][1:3], ["calibrate", value("--output-dir")])
                    self.assertEqual(calls[1][-4:], ["--calibration-epochs", "1", "--calibration-batch-size", "64"])


if __name__ == "__main__":
    unittest.main()
