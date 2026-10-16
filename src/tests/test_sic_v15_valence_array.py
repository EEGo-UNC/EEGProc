"""Check full valence array coverage and launch routing without Slurm or GPUs."""
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/eegproc/deep_learning/joint_architectures/SICModelv15/SLURM_scripts/full_run_v15_valence.sh"
TRAINER = "src.eegproc.deep_learning.joint_architectures.SICModelv15.sic_model_train"


class ValenceArrayChecks(unittest.TestCase):
    def test_all_tasks_cover_the_grid_once_in_original_order(self):
        source = SCRIPT.read_text()
        self.assertIn("#SBATCH --array=0-35%2", source)
        self.assertIn("full_run_v15_valence_%A_%a.out", source)
        self.assertIn("full_run_v15_valence_%A_%a.err", source)
        code = re.search(r"MODEL_CONFIG=\"\$\(python - <<'PY'\n(.*?)\nPY\n\)\"", source, re.S).group(1)
        expected = list(itertools.product([0.0, 0.5, 1.0, 2.0],
                                         [0.0, 0.3, 0.6], [0.0, 0.4, 0.8]))
        observed = []
        for index in range(36):
            result = subprocess.run([sys.executable, "-c", code], check=True,
                                    env={**os.environ, "TRAINING_SEED": "99", "SLURM_ARRAY_TASK_ID": str(index)},
                                    capture_output=True, text=True)
            config = json.loads(result.stdout)
            self.assertFalse(any(isinstance(v, dict) and "grid" in v for v in config.values()))
            self.assertEqual(config["gcn_units"], {"fixed": [128, 64]})
            self.assertEqual(config["classifier_rnn_units"], {"fixed": [128, 64]})
            self.assertEqual(config["mldg_seed"], 99)
            self.assertEqual(config["mi_random_state"], 99)
            observed.append(tuple(config[key] for key in
                                  ("focal_gamma", "subject_loss_weight", "reconstruction_loss_weight")))
        self.assertEqual(observed, expected)
        self.assertEqual(len(set(observed)), 36)

    def test_missing_or_invalid_task_fails_before_cluster_setup(self):
        for task in (None, "-1", "36", "abc", "01"):
            with self.subTest(task=task):
                env = {k: v for k, v in os.environ.items() if k != "SLURM_ARRAY_TASK_ID"}
                if task is not None:
                    env["SLURM_ARRAY_TASK_ID"] = task
                result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("SLURM_ARRAY_TASK_ID must be 0..35", result.stderr)
                self.assertNotIn("module: command not found", result.stderr)

    def test_shell_launches_one_configuration_per_task_in_distinct_folders(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project checkout"
            venv = project / "venv312"
            binaries = venv / "bin"
            binaries.mkdir(parents=True)
            # Replace only cluster commands and training. Execute the real shell
            # launcher and its real configuration generator end to end.
            python = binaries / "python"
            python.write_text(f"#!{sys.executable}\n" + "\n".join([
                "import json, os, pathlib, sys",
                f"if sys.argv[1:3] == ['-m', {TRAINER!r}]:",
                "    pathlib.Path(os.environ['TRAINING_CAPTURE']).write_text(json.dumps(sys.argv[1:]))",
                "elif sys.argv[1:3] == ['-m', 'src.tests.test_sic_v15_multi_gpu']:",
                "    pass",
                "elif sys.argv[1:] == ['-']:",
                "    code = sys.stdin.read()",
                "    if 'import tensorflow as tf' not in code:",
                "        exec(compile(code, '<launcher>', 'exec'), {'__name__': '__main__'})",
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
            outputs = []
            for task in (0, 1, 35):
                capture = project / f"capture_{task}.json"
                env = {**os.environ, "PROJECT_DIR": str(project), "VENV_DIR": str(venv),
                       "SLURM_ARRAY_JOB_ID": "100", "SLURM_JOB_ID": str(200 + task),
                       "SLURM_ARRAY_TASK_ID": str(task), "TRAINING_CAPTURE": str(capture),
                       "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                       "TRAINING_SEED": "99", "SOURCE_EPOCHS": "12", "CALIBRATION_EPOCHS": "1"}
                result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(capture.read_text())
                def value(flag):
                    return args[args.index(flag) + 1]
                outputs.append(value("--out-dir"))
                self.assertIn(f"suite_100/task_{task}/full", outputs[-1])
                self.assertTrue(value("--run-name").endswith(f"_task_{task}"))
                self.assertEqual(value("--source-epochs"), "12")
                self.assertEqual(value("--calibration-epochs"), "1")
                self.assertEqual(args[args.index("--calibration-level") + 1:][:2], ["3", "6"])
                self.assertEqual(value("--n-jobs"), "2")
                self.assertEqual(value("--gpus-per-fold"), "2")
                self.assertEqual(value("--hyperparameter-selection-level"), "losocv")
                self.assertIn("--deterministic-training", args)
                config = json.loads(value("--hyperparameters-json"))
                self.assertFalse(any(isinstance(v, dict) and "grid" in v for v in config.values()))
            self.assertEqual(len(set(outputs)), 3)


if __name__ == "__main__":
    unittest.main()
