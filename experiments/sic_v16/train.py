"""Bridge tracked run bundles to SICv16's existing isolated-fold controller."""
import argparse
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import traceback
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training_tracker.core import fingerprint, load_json, utc_now, write_json
from eegproc.deep_learning.joint_architectures.SICModelv16.run_experiments import run, summarize

RUNTIME_KEYS = {"target", "epochs", "target_subjects", "data_subjects", "batch_size", "validation_subjects", "seed", "time_budget_seconds"}


def tracker_result(summary):
    selected = summary["validation_selected"]
    if selected["status"] != "completed" or summary["oracle"]["status"] != "completed":
        raise ValueError("Incomplete folds cannot become a completed tracker result.")
    if not selected["folds"]:
        raise ValueError("A tracker result needs at least one completed fold.")
    subjects = [{"subject_id": row["target_subject"],
                 **{f"trial_{key}": value for key, value in row["test_metrics"].items()}}
                for row in selected["folds"]]
    metrics = {key: {"mean": statistics.mean(row[key] for row in subjects),
                     "std": statistics.pstdev(row[key] for row in subjects)}
               for key in subjects[0] if key != "subject_id"}
    return {"metrics": metrics, "subject_metrics": subjects,
            "evaluation_level": "trial", "selection": selected["selection_rule"],
            "oracle": False, "oracle_diagnostics": summary["oracle"],
            "single_class_prediction_subjects": [row["target_subject"] for row in selected["folds"]
                if row.get("prediction_diagnostics", {}).get("single_class_predictions", False)],
            "validation_selected_folds": selected["folds"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_json(args.config)
    output = args.output_dir.resolve()
    if Path("/work/users").is_dir() and output.is_relative_to(Path.home().resolve()):
        raise ValueError("Run output is still in home storage. Run recover.py --apply before training.")
    manifest_path = output / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest["config_fingerprint"] != fingerprint(config):
        raise ValueError("Tracked configuration fingerprint mismatch.")
    if args.resume and manifest["status"] == "completed":
        print("Already completed; no work needed.")
        return 0
    if not args.resume and manifest["status"] != "planned":
        raise ValueError("This bundle has already started; plan a new run instead of overwriting it.")
    if args.resume:
        manifest.setdefault("previous_attempts", []).append({k: v for k, v in manifest.items()
            if k not in {"previous_attempts"}})
        for key in ("error", "error_type", "completed_at"):
            manifest.pop(key, None)
    lock = output / ".run.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid()}, stream)
    manifest.update(status="running", started_at=utc_now(),
                    hostname=platform.node(), slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    write_json(manifest_path, manifest)
    try:
        if config["target"] != "valence":
            raise ValueError("SICv16 currently supports valence only.")
        # Keep setup failures inside the tracked lifecycle too. The synthetic
        # checks import no TensorFlow and run in their own process.
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                        str(ROOT / "src/tests"), "-p", "test_sic_v16_data.py", "-v"],
                       cwd=ROOT, check=True)
        hyperparameters = {key: value for key, value in config.items() if key not in RUNTIME_KEYS}
        fixed = output / "sic_configuration.json"
        write_json(fixed, {"base": hyperparameters,
                           "configurations": [{"name": "tracked", "overrides": {}}]})
        if args.resume:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                            str(ROOT / "src/tests"), "-p", "test_sic_v16_resume.py", "-v"],
                           cwd=ROOT, check=True)
        code = run(argparse.Namespace(configs=fixed, config_index=0,
            output_dir=output / "artifacts", eeg=args.data_dir / "dreamer_eeg.npy",
            labels=args.data_dir / "dreamer_labels.npy", source_epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]), validation_subjects=int(config["validation_subjects"]),
            seed=int(config["seed"]), time_budget_seconds=int(config["time_budget_seconds"]),
            target_subjects=config["target_subjects"], data_subjects=config.get("data_subjects"), resume=args.resume))
        if code:
            raise RuntimeError(f"SICv16 runner stopped incomplete (exit {code}); inspect artifacts/ and subject logs.")
        result = tracker_result(summarize(output / "artifacts"))
        result["data_scope"] = {"data_subjects": config.get("data_subjects"),
            "target_subjects": config["target_subjects"],
            "validation_subjects_per_fold": config["validation_subjects"]}
        write_json(output / "result.json", result)
        manifest.update(status="completed", completed_at=utc_now())
        write_json(manifest_path, manifest)
        return 0
    except Exception as exc:
        manifest.update(status="failed", completed_at=utc_now(), error_type=type(exc).__name__, error=str(exc))
        write_json(manifest_path, manifest)
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
