"""Nested subject-LOSO RMS log-power baseline with validation-selected thresholds."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def trial_features(windows):
    """42 features: mean window power or mean natural log of window power.

    Input is the same globally RMS-normalized band-filtered EEG as the CNN.
    Log power is averaged after taking logs, rather than logging trial means.
    """
    import numpy as np
    x = np.asarray(windows, dtype=np.float64)
    power = np.mean(x * x, axis=1)
    return {"band_power": power.mean(axis=0),
            "log_band_power": np.log(np.maximum(power, 1e-12)).mean(axis=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--permutation-index", type=int, default=0)
    args = parser.parse_args()
    if args.permutation_index < 0:
        parser.error("Permutation index must be nonnegative; zero means original labels.")
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
    if args.feature_cache:
        arrays = {k: np.load(args.feature_cache / f"{k}.npy", allow_pickle=False)
                  for k in ("log_power", "labels", "subjects", "trials")}
        matrices = {"log_band_power": arrays["log_power"]}
        cache.mkdir()
        import shutil
        shutil.copyfile(args.feature_cache / "preprocessing.json", cache / "preprocessing.json")
        y, subjects = arrays["labels"].copy(), arrays["subjects"]
    else:
        prepare_data(args.data_dir / "dreamer_eeg.npy", args.data_dir / "dreamer_labels.npy",
                     cache, data_subjects=list(range(23)))
        arrays = {k: np.load(cache / f"{k}.npy", mmap_mode="r")
                  for k in ("features", "labels", "subjects", "trials")}
        y, subjects = arrays["labels"], arrays["subjects"]
        rows = [trial_features(x) for x in arrays["features"]]
        matrices = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        if any(not np.isfinite(x).all() for x in matrices.values()):
            raise ValueError("Nonfinite feature matrix.")
    if args.permutation_index:
        from permutation_audit import permute_labels
        y = permute_labels(y, subjects, args.seed, args.permutation_index)
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    from sklearn.model_selection import LeaveOneGroupOut
    grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    thresholds = np.linspace(0.0, 1.0, 41)
    x = matrices["log_band_power"]
    def new_model(c):
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", solver="lbfgs", max_iter=10000,
            random_state=args.seed))
    def metrics(labels, p, threshold):
        predicted = (p >= threshold).astype(int)
        return dict(accuracy=float(accuracy_score(labels, predicted)),
                    balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
                    roc_auc=float(roc_auc_score(labels, p)),
                    confusion_matrix=confusion_matrix(labels, predicted, labels=[0, 1]).tolist())
    def subject_score(labels, p, groups, threshold):
        return float(np.mean([balanced_accuracy_score(labels[groups == g],
                          (p[groups == g] >= threshold).astype(int)) for g in np.unique(groups)]))
    write("config.json", dict(permutation_index=args.permutation_index, seed=args.seed, C_grid=grid, threshold_grid=thresholds.tolist(),
          outer_protocol="23 held-out subjects; refit on all 22 non-test subjects",
          inner_protocol="22 leave-one-subject-out validation folds per outer fold",
          selection="mean subject validation balanced accuracy; ties: smaller C, threshold nearest 0.5, lower threshold",
          feature="RMS log band power; mean natural log window power; 42 features",
          scaling="StandardScaler fitted independently on each inner training pool and final refit pool",
          diagnostic="fixed-0.5 results use the same selected C; no test-label tuning"))
    results = []
    for target in range(23):
        outer = subjects != target
        test = subjects == target
        ox, oy, groups = x[outer], y[outer], subjects[outer]
        if len(oy) != 396 or int(test.sum()) != 18:
            raise ValueError("Unexpected outer split sizes.")
        candidates = []
        inner_splits = []
        for c in grid:
            oof = np.full(len(oy), np.nan)
            coverage = np.zeros(len(oy), dtype=int)
            for train, val in LeaveOneGroupOut().split(ox, oy, groups):
                if set(groups[train]) & set(groups[val]) or target in set(groups):
                    raise ValueError("Subject leakage.")
                model = new_model(c)
                model.fit(ox[train], oy[train])
                oof[val] = model.predict_proba(ox[val])[:, 1]
                coverage[val] += 1
                if c == grid[0]:
                    inner_splits.append(dict(training_subjects=sorted(set(groups[train].tolist())),
                                             validation_subjects=sorted(set(groups[val].tolist()))))
            if not np.all(coverage == 1) or not np.isfinite(oof).all():
                raise ValueError("Every non-test trial needs exactly one finite out-of-fold prediction.")
            scores = [subject_score(oy, oof, groups, float(t)) for t in thresholds]
            best_index = max(range(len(thresholds)),
                             key=lambda i: (scores[i], -abs(float(thresholds[i]) - 0.5), -float(thresholds[i])))
            candidates.append(dict(C=c, threshold=float(thresholds[best_index]),
                                   score=scores[best_index], threshold_scores=scores,
                                   probabilities=oof.tolist()))
        best = max(candidates, key=lambda v: (v["score"], -v["C"],
                   -abs(v["threshold"] - 0.5), -v["threshold"]))
        model = new_model(best["C"])
        model.fit(ox, oy)
        p = model.predict_proba(x[test])[:, 1]
        train_p = model.predict_proba(ox)[:, 1]
        row = dict(test_subject=target, refit_subjects=sorted(set(groups.tolist())),
                   inner_splits=inner_splits, selected_C=best["C"], selected_threshold=best["threshold"],
                   inner_validation_mean_balanced_accuracy=best["score"], search=candidates,
                   inner_labels=oy.tolist(), inner_subjects=groups.tolist(),
                   inner_trials=arrays["trials"][outer].tolist(),
                   training=metrics(oy, train_p, best["threshold"]),
                   test=metrics(y[test], p, best["threshold"]),
                   test_fixed_05=metrics(y[test], p, 0.5),
                   test_labels=y[test].tolist(), test_probabilities=p.tolist(),
                   test_trials=arrays["trials"][test].tolist())
        results.append(row)
        write("folds.json", results)
        print(f"subject={target} C={best['C']} threshold={best['threshold']:.3f} inner_BA={best['score']:.3f} test_BA={row['test']['balanced_accuracy']:.3f}", flush=True)
    write("result.json", dict(status="completed", permutation_index=args.permutation_index, sklearn=sklearn.__version__, folds=results,
          mean_test_balanced_accuracy=float(np.mean([r["test"]["balanced_accuracy"] for r in results])),
          mean_test_fixed_05_balanced_accuracy=float(np.mean([r["test_fixed_05"]["balanced_accuracy"] for r in results]))))
    archive = output / "nested-logpower-diagnostics.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in [output / n for n in ("config.json", "folds.json", "result.json", "data/preprocessing.json")]:
            bundle.add(path, arcname=str(path.relative_to(output)))
    print(f"Download: {archive}", flush=True)


if __name__ == "__main__":
    main()
