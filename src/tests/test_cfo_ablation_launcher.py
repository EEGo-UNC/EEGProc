"""Exercise Slurm dispatch without loading CUDA or starting optimization."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "eegproc/model_explainability/slurm/run_cfo_ablations_arousal_1599318.sh"
VALENCE_SCRIPT = SCRIPT.with_name("run_cfo_ablations_valence_65452590.sh")


@pytest.fixture
def launcher_env(tmp_path):
    # Spaces and shell metacharacters must survive as literal data paths.
    project = tmp_path / "EEG project $literal"
    project.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    models = project / "models"
    models.mkdir()
    for subject in (0, 22):
        (models / f"heldout_{subject}.keras").write_text("checkpoint fixture")
    manifest = project / "models.json"
    manifest.write_text(json.dumps({"models": [dict(target_subject=s,
        stage="zero_shot_source_model", path=f"/cluster/heldout_{s}.keras") for s in (0, 22)]}))
    prepared = project / "trials.npz"
    prepared.write_text("not loaded during preview")
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    env.update(PROJECT_DIR=str(project), VENV_DIR=str(venv), MODELS_JSON=str(manifest),
               MODEL_DIR=str(models), TRIALS_NPZ=str(prepared), DRY_RUN="1", RESUME="0",
               RUN_ID="fixture", OUT_ROOT=str(project / "results"), TRIAL_IDS="8 10")
    return env


@pytest.mark.parametrize("subject", [0, 22])
@pytest.mark.parametrize("script,task,alpha", [(SCRIPT, "arousal", None), (VALENCE_SCRIPT, "valence", 0.49751)])
def test_ablation_launcher_matches_subject_manifest_and_parser(launcher_env, subject, script, task, alpha):
    from eegproc.model_explainability.typicality.runner import parse_args

    launcher_env["SLURM_ARRAY_TASK_ID"] = str(subject)
    result = subprocess.run(["bash", str(script)], env=launcher_env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert f"heldout_{subject}.keras" in result.stdout
    command = shlex.split(next(line.removeprefix("Command: ") for line in result.stdout.splitlines()
                              if line.startswith("Command: ")))
    args = parse_args(command[3:])
    assert args.subjects == [subject]
    assert args.task == task
    assert args.fixed_joint_alpha == alpha
    assert args.model_module.endswith(("SICModelv11" if task == "valence" else "SICModelv15") + ".sic_model")
    assert args.include_target_latent
    assert args.include_typicality_no_physiology
    assert args.trial_ids == [8, 10]
    assert args.physiological_weight > 0
    assert args.typicality_weight > 0
    assert str(args.trials_npz) == launcher_env["TRIALS_NPZ"]
    assert not Path(launcher_env["OUT_ROOT"]).exists()


@pytest.mark.parametrize("script", [SCRIPT, VALENCE_SCRIPT])
def test_ablation_launcher_rejects_out_of_range_subject(launcher_env, script):
    launcher_env["SLURM_ARRAY_TASK_ID"] = "23"
    result = subprocess.run(["bash", str(script)], env=launcher_env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "0-22" in result.stderr


@pytest.mark.parametrize("script", [SCRIPT, VALENCE_SCRIPT])
def test_ablation_launcher_does_not_overwrite_results(launcher_env, script):
    (Path(launcher_env["OUT_ROOT"]) / "fold_00").mkdir(parents=True)
    result = subprocess.run(["bash", str(script)], env=launcher_env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "output exists" in result.stderr


def test_valence_launcher_uses_matching_raw_loader(launcher_env):
    from eegproc.model_explainability.typicality.runner import parse_args

    raw = Path(launcher_env.pop("TRIALS_NPZ"))
    launcher_env.update(EEG_PATH=str(raw), LABELS_PATH=str(raw))
    result = subprocess.run(["bash", str(VALENCE_SCRIPT)], env=launcher_env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    command = shlex.split(next(line.removeprefix("Command: ") for line in result.stdout.splitlines()
                              if line.startswith("Command: ")))
    args = parse_args(command[3:])
    assert args.data_config["label_dimension"] == "valence"
    assert args.data_config["model_module"] == args.model_module
    assert args.data_config["window_normalization"] == "global_rms"
    assert "#SBATCH --array=0-22%4\n" in VALENCE_SCRIPT.read_text()


def test_valence_missing_manifest_fails_before_cluster_modules(launcher_env):
    launcher_env["MODELS_JSON"] = str(Path(launcher_env["PROJECT_DIR"]) / "missing.json")
    launcher_env["DRY_RUN"] = "0"
    result = subprocess.run(["bash", str(VALENCE_SCRIPT)], env=launcher_env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "checkpoint manifest not found" in result.stderr
    assert "CONFIG_DIR" in result.stderr
    assert "*65452590*" in result.stderr
    assert "Traceback" not in result.stderr
    assert "module: command not found" not in result.stderr
    assert not Path(launcher_env["OUT_ROOT"]).exists()


def test_valence_config_dir_override_resolves_relocated_manifest(launcher_env):
    project = Path(launcher_env["PROJECT_DIR"])
    config = project / "suite_65452590.before-pull/full/relocated_run/configuration_0001"
    config.mkdir(parents=True)
    Path(launcher_env.pop("MODELS_JSON")).rename(config / "loso_zero_shot_models.json")
    Path(launcher_env.pop("MODEL_DIR")).rename(config / "loso_zero_shot_models")
    launcher_env["CONFIG_DIR"] = str(config)
    result = subprocess.run(["bash", str(VALENCE_SCRIPT)], env=launcher_env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert str(config / "loso_zero_shot_models/heldout_0.keras") in result.stdout
    command = shlex.split(next(line.removeprefix("Command: ") for line in result.stdout.splitlines()
                              if line.startswith("Command: ")))
    assert command[command.index("--models-json") + 1] == str(config / "loso_zero_shot_models.json")
    assert command[command.index("--model-dir") + 1] == str(config / "loso_zero_shot_models")
    assert command[command.index("--fixed-joint-alpha") + 1] == "0.49751"
    assert not Path(launcher_env["OUT_ROOT"]).exists()
