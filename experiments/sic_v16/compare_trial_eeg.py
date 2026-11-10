"""Compare trial ID, EEG and their combination on a verified rebuilt dataset."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def segment_log_power(windows, segments):
    """Preserve segment order; average log one-second power within each segment."""
    import numpy as np
    windows = np.asarray(windows)
    if windows.shape != (60, 128, 42) or segments not in (1, 6):
        raise ValueError("Expected 60 one-second windows and one or six segments.")
    power = np.mean(np.square(windows, dtype=np.float64), axis=1)
    return np.log(np.maximum(power, 1e-12)).reshape(segments, 60 // segments, 42).mean(axis=1).reshape(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--segments", type=int, choices=[1, 6], default=1)
    args = p.parse_args()
    import nested_logpower  # Adds the project source directory for shared preprocessing.
    from eegproc.deep_learning.joint_architectures.SICModelv16.data import prepare_data
    from permutation_audit import digest
    import numpy as np
    if args.rebuild_dir:
        rebuild = args.rebuild_dir.resolve()
    else:
        work = Path(os.environ["SIC_V16_WORK_ROOT"])
        candidates = [path.parent for path in (work / "crop-rebuild-comparison").glob("job-*/comparison.json")
                      if (path.parent / "centred_full_filter/dreamer_eeg.npy").is_file()]
        if not candidates:
            raise ValueError("No completed rebuild found; set REBUILD_DIR to its job directory.")
        rebuild = max(candidates, key=lambda path: (path / "comparison.json").stat().st_mtime)
    report = json.loads((rebuild / "crop_verification.json").read_text())
    data = rebuild / "centred_full_filter"
    if report["summary"]["trials"] != 414 or report["summary"]["identified"] != 414:
        raise ValueError("Rebuild provenance is incomplete.")
    actual = {name: digest(data / name) for name in ("dreamer_eeg.npy", "dreamer_labels.npy")}
    expected = report["centred_full_filter"]
    if actual["dreamer_eeg.npy"] != expected["eeg_sha256"] or actual["dreamer_labels.npy"] != expected["labels_sha256"]:
        raise ValueError("Rebuilt dataset hashes do not match provenance.")
    eeg = np.load(data / "dreamer_eeg.npy", mmap_mode="r")
    if eeg.shape != (23, 18, 42, 7680):
        raise ValueError("Expected full-stimulus-filtered, channel-major 42-feature data.")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    print(f"Using verified rebuilt dataset: {data}", flush=True)
    cache = root / "prepared"
    prepare_data(data / "dreamer_eeg.npy", data / "dreamer_labels.npy", cache,
                 data_subjects=list(range(23)))
    windows = np.load(cache / "features.npy", mmap_mode="r")
    matrix = np.stack([segment_log_power(trial, args.segments) for trial in windows])
    if matrix.shape != (414, 42 * args.segments) or not np.isfinite(matrix).all():
        raise ValueError("Invalid EEG feature matrix.")
    np.save(cache / "log_power.npy", matrix)
    del windows
    (cache / "features.npy").unlink()
    manifest = dict(rebuild_dir=str(rebuild), data_dir=str(data), sha256=actual,
                    preprocessing="already band-filtered source stimulus; centre crop; window RMS; mean log power within ordered segments",
                    segments=args.segments, seconds_per_segment=60 // args.segments,
                    feature_order="segment-major, then channel-major, then theta/alpha/beta",
                    arms={"trial_id": 18, "eeg": 42 * args.segments, "trial_id_plus_eeg": 18 + 42 * args.segments},
                    evaluation="identical 23 outer and 22 inner subject folds; independent C/threshold selection per arm",
                    limitation="Trial ID remains a stimulus proxy; combined features test incremental value under shared-trial evaluation.")
    (root / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results = {}
    for arm, flags in (("trial_id", ["--trial-id-only"]), ("eeg", []),
                       ("trial_id_plus_eeg", ["--trial-id-plus-eeg"])):
        subprocess.run([sys.executable, str(Path(__file__).with_name("nested_logpower.py")),
                        "--data-dir", str(data), "--feature-cache", str(cache),
                        "--output-dir", str(root / arm), *flags], check=True)
        results[arm] = json.loads((root / arm / "result.json").read_text())
    reference = results["eeg"]["folds"]
    for result in results.values():
        if len(result["folds"]) != 23:
            raise ValueError("Incomplete outer folds.")
        for left, right in zip(reference, result["folds"]):
            for key in ("test_subject", "refit_subjects", "inner_splits", "test_labels", "test_trials"):
                if left[key] != right[key]:
                    raise ValueError(f"Evaluation alignment mismatch: {key}")
    summary = {arm: {k: v for k, v in result.items() if k != "folds"} for arm, result in results.items()}
    summary["combined_minus_trial_id"] = {
        key: float(np.mean([combined[key]["balanced_accuracy"] - identity[key]["balanced_accuracy"]
                           for combined, identity in zip(results["trial_id_plus_eeg"]["folds"], results["trial_id"]["folds"])]))
        for key in ("test", "test_fixed_05")}
    (root / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    archive = root / ("segment-trial-eeg-comparison-diagnostics.tar.gz" if args.segments == 6 else "trial-eeg-comparison-diagnostics.tar.gz")
    with tarfile.open(archive, "w:gz") as bundle:
        for relative in ["provenance.json", "comparison.json", "prepared/preprocessing.json"] + [
                f"{arm}/{name}" for arm in results for name in ("config.json", "result.json")]:
            bundle.add(root / relative, arcname=relative)
    print(f"Download: {archive}", flush=True)


if __name__ == "__main__":
    main()
