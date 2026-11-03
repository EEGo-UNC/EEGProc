"""Audit array alignment, prepare compact features, or summarize permutation runs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def permute_labels(labels, subjects, seed, index):
    import numpy as np
    result = np.asarray(labels).copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    for subject in np.unique(subjects):
        mask = subjects == subject
        result[mask] = rng.permutation(result[mask])
    return result


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "summarize"])
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--permutations", type=int, default=100)
    args = p.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    def write(name, value):
        (root / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    if args.mode == "prepare":
        import numpy as np
        from nested_logpower import trial_features
        from eegproc.deep_learning.joint_architectures.SICModelv16.data import prepare_data
        cache = root / "prepared"
        prepare_data(args.data_dir / "dreamer_eeg.npy", args.data_dir / "dreamer_labels.npy",
                     cache, data_subjects=list(range(23)))
        arrays = {k: np.load(cache / f"{k}.npy", mmap_mode="r")
                  for k in ("features", "labels", "subjects", "trials")}
        eeg = np.load(args.data_dir / "dreamer_eeg.npy", mmap_mode="r")
        ratings = np.load(args.data_dir / "dreamer_labels.npy", mmap_mode="r")
        if eeg.shape[:2] != (23, 18) or ratings.shape[:2] != (23, 18):
            raise ValueError("Expected matching 23-subject, 18-trial arrays.")
        records, hashes, features = [], [], []
        for i, (subject, trial) in enumerate(zip(arrays["subjects"], arrays["trials"])):
            subject, trial = int(subject), int(trial)
            if (subject, trial) != divmod(i, 18):
                raise ValueError("Prepared subject/trial indexing changed.")
            raw = np.asarray(eeg[subject, trial])
            if not np.isfinite(raw).all() or not np.isfinite(ratings[subject, trial]).all():
                raise ValueError("Nonfinite EEG or ratings.")
            expected = int(ratings[subject, trial, 0] >= 3)
            if int(arrays["labels"][i]) != expected:
                raise ValueError("Prepared label differs from original array valence >= 3.")
            hashes.append(hashlib.sha256(raw.tobytes()).hexdigest())
            records.append(dict(subject=subject, trial=trial,
                ratings=ratings[subject, trial].tolist(), label=expected,
                eeg_rms=float(np.sqrt(np.mean(raw.astype(float)**2))), eeg_std=float(raw.std())))
            features.append(trial_features(arrays["features"][i])["log_band_power"])
        np.save(cache / "log_power.npy", np.stack(features))
        duplicates = [[i for i, h in enumerate(hashes) if h == value]
                      for value in sorted(set(hashes)) if hashes.count(value) > 1]
        write("audit.json", dict(status="array_checks_passed", eeg_shape=list(eeg.shape),
            labels_shape=list(ratings.shape), samples_per_trial=int(eeg.shape[-1]),
            nominal_seconds_at_128_hz=eeg.shape[-1]/128, records=records,
            duplicate_trial_indices=duplicates, constant_trial_indices=[i for i,r in enumerate(records) if r["eeg_std"] == 0],
            sha256={name: digest(args.data_dir / name) for name in ("dreamer_eeg.npy", "dreamer_labels.npy")},
            unresolved=["Array files alone cannot prove EEG and rating stimulus order match the original dataset.",
                        "Dimension 0 is assumed valence; verify against original export records.",
                        "128 Hz and the location of the retained segment must be verified against original conversion provenance."],
            permutation_assumption="Trial labels are exchangeable within each subject under the null; temporal/stimulus dependence may violate this assumption."))
        # Only the large derived window array is removed; compact arrays remain.
        del arrays
        import gc
        gc.collect()
        (cache / "features.npy").unlink()
        return
    if args.permutations < 1:
        p.error("Need at least one permutation.")
    runs = []
    for index in range(args.permutations + 1):
        path = root / f"run-{index}" / "result.json"
        value = json.loads(path.read_text())
        if value.get("status") != "completed" or value.get("permutation_index") != index or len(value["folds"]) != 23:
            raise ValueError(f"Incomplete or mismatched run: {path}")
        runs.append(value)
    summary = dict(permutations=args.permutations, null="labels independently shuffled within each subject, including outer test labels; full nested selection repeated")
    for key in ("mean_test_balanced_accuracy", "mean_test_fixed_05_balanced_accuracy"):
        observed = runs[0][key]
        null = [r[key] for r in runs[1:]]
        summary[key] = dict(observed=observed, null_scores=null,
            null_mean=sum(null)/len(null), upper_tail_p=(1+sum(v >= observed for v in null))/(len(null)+1))
    write("permutation_summary.json", summary)
    archive = root / "audit-permutation-diagnostics.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in [root / "audit.json", root / "permutation_summary.json", root / "prepared/preprocessing.json"]:
            bundle.add(path, arcname=str(path.relative_to(root)))
        for index in range(args.permutations + 1):
            for name in ("config.json", "result.json"):
                path = root / f"run-{index}" / name
                bundle.add(path, arcname=str(path.relative_to(root)))
    print(f"Download: {archive}", flush=True)


if __name__ == "__main__":
    main()
