"""Compare amplitude-preserving and window-RMS spectral baselines on fixed subject splits."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def trial_features(windows):
    """42 channel/band features; aggregate windows only within each trial.

    Power is mean squared band-filtered amplitude. DE is the Gaussian entropy
    estimate in nats, averaged over windows, rather than entropy of a histogram.
    Log power is the mean natural log of window power (not log of trial mean).
    """
    import numpy as np
    x = np.asarray(windows, dtype=np.float64)
    power = np.mean(x * x, axis=1)
    variance = np.var(x, axis=1)
    return {"band_power": power.mean(axis=0),
            "log_band_power": np.log(np.maximum(power, 1e-12)).mean(axis=0),
            "differential_entropy": (0.5 * np.log(2 * np.pi * np.e *
                                      np.maximum(variance, 1e-12))).mean(axis=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import numpy as np
    import sklearn
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
    from eegproc.deep_learning.joint_architectures.SICModelv16.data import prepare_data
    from eegproc.deep_learning.joint_architectures.SICModelv16.run_experiments import fold_subjects

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "data"
    prepare_data(args.data_dir / "dreamer_eeg.npy", args.data_dir / "dreamer_labels.npy",
                 cache, data_subjects=[0, 1, 2, 3])
    arrays = {k: np.load(cache / f"{k}.npy", mmap_mode="r")
              for k in ("features", "labels", "subjects", "trials")}
    y, subjects = arrays["labels"], arrays["subjects"]
    rows = [trial_features(x) for x in arrays["features"]]
    matrices = {"global_rms/" + k: np.stack([r[k] for r in rows]) for k in rows[0]}
    # Re-read and filter complete original trials, before window RMS scaling.
    # This uses the identical filter and ordering as the existing CNN loader.
    from eegproc.deep_learning.prepare_datasets import _filter_dreamer_trial
    eeg = np.load(args.data_dir / "dreamer_eeg.npy", mmap_mode="r", allow_pickle=False)
    raw_rows = []
    for subject, trial in zip(subjects, arrays["trials"]):
        original = np.asarray(eeg[int(subject), int(trial)].T, dtype=np.float32)
        filtered = _filter_dreamer_trial(original) if eeg.shape[2] == 14 else original
        count = len(filtered) // 128
        windows = np.asarray(filtered[:count * 128], dtype=np.float32).reshape(count, 128, 42)
        if not np.isfinite(windows).all():
            raise ValueError("Nonfinite amplitude-preserving input.")
        # Verify the two branches differ only in window amplitude scaling.
        rms = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=(1, 2), keepdims=True))
        normalized = windows.astype(np.float64) / np.maximum(rms, 1e-6)
        index = len(raw_rows)
        if not np.allclose(normalized, arrays["features"][index], rtol=2e-6, atol=2e-6):
            raise ValueError("Amplitude-preserving and CNN input branches do not align.")
        raw_rows.append(trial_features(windows))
    matrices.update({"no_window_scaling/" + k: np.stack([r[k] for r in raw_rows]) for k in raw_rows[0]})
    if any(not np.isfinite(x).all() for x in matrices.values()):
        raise ValueError("Nonfinite feature matrix.")
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    write("config.json", dict(seed=args.seed, data_subjects=[0, 1, 2, 3],
          target="valence >= 3", classifier="StandardScaler + balanced logistic regression",
          C_grid=grid, threshold=0.5,
          selection="highest validation balanced accuracy; smallest C on ties",
          normalization=["global_rms", "no_window_scaling"],
          scaling="StandardScaler fitted only to source training trials for every candidate",
          power_floor=1e-12, logarithm="natural; DE in nats",
          feature_order="channel-major, band-minor: theta, alpha, beta",
          aggregation="mean window power, mean log window power, or mean window Gaussian DE; 42 features per trial"))
    results = []
    for target in (0, 1, 2, 3):
        train_ids, val_ids = fold_subjects(subjects, target, 1, args.seed)
        masks = {"training": np.isin(subjects, train_ids),
                 "validation": np.isin(subjects, val_ids), "test": subjects == target}
        if [int(masks[k].sum()) for k in masks] != [36, 18, 18]:
            raise ValueError("Unexpected split sizes.")
        for feature, x in matrices.items():
            candidates = []
            for c in grid:
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=c, class_weight="balanced", solver="lbfgs", max_iter=10000,
                    random_state=args.seed))
                model.fit(x[masks["training"]], y[masks["training"]])
                score = float(balanced_accuracy_score(y[masks["validation"]],
                    model.predict(x[masks["validation"]])))
                candidates.append((score, c, model))
            score, c, model = max(candidates, key=lambda item: (item[0], -item[1]))
            row = dict(feature=feature, test_subject=target, train_subjects=train_ids,
                       validation_subjects=val_ids, selected_C=c,
                       validation_search=[dict(C=c0, balanced_accuracy=s) for s, c0, _ in candidates])
            for name, mask in masks.items():
                labels = y[mask]
                p = model.predict_proba(x[mask])[:, 1]
                predicted = (p >= 0.5).astype(int)
                row[name] = dict(accuracy=float(accuracy_score(labels, predicted)),
                    balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
                    roc_auc=float(roc_auc_score(labels, p)),
                    confusion_matrix=confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
                    labels=labels.tolist(), class_1_probabilities=p.tolist(),
                    subjects=subjects[mask].tolist(), trials=arrays["trials"][mask].tolist())
            results.append(row)
            write("folds.json", results)
            print(f"{feature} subject={target} C={c} train_BA={row['training']['balanced_accuracy']:.3f} validation_BA={score:.3f} test_BA={row['test']['balanced_accuracy']:.3f}", flush=True)
    write("result.json", dict(status="completed", sklearn=sklearn.__version__, folds=results,
          mean_test_balanced_accuracy={k: float(np.mean([r["test"]["balanced_accuracy"]
                                  for r in results if r["feature"] == k])) for k in matrices}))
    archive = output / "normalization-baseline-diagnostics.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in [output / n for n in ("config.json", "folds.json", "result.json", "data/preprocessing.json")]:
            bundle.add(path, arcname=str(path.relative_to(output)))
    print(f"Download: {archive}", flush=True)


if __name__ == "__main__":
    main()
