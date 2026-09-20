"""Exercise Slurm dispatch without loading CUDA or starting optimization."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "eegproc/model_explainability/slurm/run_cfo_ablations_arousal_1599318.sh"


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
def test_ablation_launcher_matches_subject_manifest_and_parser(launcher_env, subject):
    from eegproc.model_explainability.typicality.runner import parse_args

    launcher_env["SLURM_ARRAY_TASK_ID"] = str(subject)
    result = subprocess.run(["bash", str(SCRIPT)], env=launcher_env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert f"heldout_{subject}.keras" in result.stdout
    command = shlex.split(next(line.removeprefix("Command: ") for line in result.stdout.splitlines()
                              if line.startswith("Command: ")))
    args = parse_args(command[3:])
    assert args.subjects == [subject]
    assert args.task == "arousal"
    assert args.include_target_latent
    assert args.trial_ids == [8, 10]
    assert args.physiological_weight > 0
    assert args.typicality_weight > 0
    assert str(args.trials_npz) == launcher_env["TRIALS_NPZ"]
    assert not Path(launcher_env["OUT_ROOT"]).exists()


def test_ablation_launcher_rejects_out_of_range_subject(launcher_env):
    launcher_env["SLURM_ARRAY_TASK_ID"] = "23"
    result = subprocess.run(["bash", str(SCRIPT)], env=launcher_env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "0-22" in result.stderr


def test_ablation_launcher_does_not_overwrite_results(launcher_env):
    (Path(launcher_env["OUT_ROOT"]) / "fold_00").mkdir(parents=True)
    result = subprocess.run(["bash", str(SCRIPT)], env=launcher_env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "output exists" in result.stderr
