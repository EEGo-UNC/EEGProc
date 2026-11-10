"""Nested within-subject whole-trial RMS log-power diagnostic."""
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
                 cache, data_subjects=list(range(23)))
    arrays = {k: np.load(cache / f"{k}.npy", mmap_mode="r")
              for k in ("features", "labels", "subjects", "trials")}
    y, subjects = arrays["labels"], arrays["subjects"]
    rows = [trial_features(x) for x in arrays["features"]]
    matrices = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    if any(not np.isfinite(x).all() for x in matrices.values()):
        raise ValueError("Nonfinite feature matrix.")
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    from sklearn.model_selection import StratifiedKFold
    grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    thresholds = np.linspace(0.0, 1.0, 41)
    x = matrices["log_band_power"]
    def splitter(labels, seed):
        count = min(3, int(np.bincount(labels.astype(int), minlength=2).min()))
        if count < 2:
            raise ValueError("At least two trials per class required in every tuning pool.")
        return StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)
    def model_for(c):
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", solver="lbfgs", max_iter=10000,
            random_state=args.seed))
    def metrics(labels, probabilities, threshold):
        predicted = (probabilities >= threshold).astype(int)
        return dict(accuracy=float(accuracy_score(labels, predicted)),
                    balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
                    roc_auc=float(roc_auc_score(labels, probabilities)),
                    confusion_matrix=confusion_matrix(labels, predicted, labels=[0, 1]).tolist())
    write("config.json", dict(seed=args.seed, C_grid=grid, threshold_grid=thresholds.tolist(),
          outer="within each subject: shuffled stratified whole-trial CV, up to 3 folds",
          inner="stratified whole-trial CV within outer training trials only, up to 3 folds",
          selection="pooled inner OOF balanced accuracy; ties smaller C, threshold nearest 0.5, lower threshold",
          scaling="fit on inner training trials; refit on outer training trials",
          feature="RMS log band power, 42 features averaged within whole trials",
          interpretation="within-subject diagnostic; trials from the same person appear in training and test; no cross-subject claim"))
    results = []
    for subject in range(23):
        mask = subjects == subject
        sx, sy, trials = x[mask], y[mask], arrays["trials"][mask]
        if len(set(trials.tolist())) != len(trials):
            raise ValueError("Duplicate trial IDs.")
        selected_predictions = np.full(len(sy), np.nan)
        fixed_predictions = np.full(len(sy), np.nan)
        coverage = np.zeros(len(sy), dtype=int)
        predicted_classes = np.zeros(len(sy), dtype=int)
        folds = []
        for fold, (train, test) in enumerate(splitter(sy, args.seed + subject).split(sx, sy)):
            inner = list(splitter(sy[train], args.seed + subject + 1000 + fold).split(sx[train], sy[train]))
            candidates = []
            for c in grid:
                oof = np.full(len(train), np.nan)
                counts = np.zeros(len(train), dtype=int)
                for itrain, ival in inner:
                    if set(train[itrain]) & set(test) or set(train[ival]) & set(test) or set(itrain) & set(ival):
                        raise ValueError("Trial leakage.")
                    model = model_for(c)
                    model.fit(sx[train[itrain]], sy[train[itrain]])
                    oof[ival] = model.predict_proba(sx[train[ival]])[:, 1]
                    counts[ival] += 1
                if not np.all(counts == 1) or not np.isfinite(oof).all():
                    raise ValueError("Incomplete inner OOF coverage.")
                scores = [float(balanced_accuracy_score(sy[train], (oof >= t).astype(int))) for t in thresholds]
                i = max(range(len(thresholds)), key=lambda i: (scores[i], -abs(float(thresholds[i])-.5), -float(thresholds[i])))
                candidates.append(dict(C=c, threshold=float(thresholds[i]), score=scores[i],
                                       threshold_scores=scores, probabilities=oof.tolist()))
            best = max(candidates, key=lambda v: (v["score"], -v["C"], -abs(v["threshold"]-.5), -v["threshold"]))
            model = model_for(best["C"])
            model.fit(sx[train], sy[train])
            p = model.predict_proba(sx[test])[:, 1]
            selected_predictions[test] = p
            fixed_predictions[test] = p
            predicted_classes[test] = (p >= best["threshold"]).astype(int)
            coverage[test] += 1
            folds.append(dict(fold=fold, training_trials=trials[train].tolist(), test_trials=trials[test].tolist(),
                inner_splits=[dict(training_trials=trials[train[a]].tolist(), validation_trials=trials[train[b]].tolist()) for a,b in inner],
                selected_C=best["C"], selected_threshold=best["threshold"], search=candidates,
                inner_labels=sy[train].tolist(), inner_trials=trials[train].tolist(),
                test=metrics(sy[test], p, best["threshold"]), test_fixed_05=metrics(sy[test], p, .5)))
        if not np.all(coverage == 1) or not np.isfinite(selected_predictions).all():
            raise ValueError("Incomplete outer OOF coverage.")
        # Aggregate held-out predictions over all 18 trials, rather than averaging tiny folds.
        selected = dict(accuracy=float(accuracy_score(sy, predicted_classes)),
                        balanced_accuracy=float(balanced_accuracy_score(sy, predicted_classes)),
                        confusion_matrix=confusion_matrix(sy, predicted_classes, labels=[0,1]).tolist())
        results.append(dict(subject=subject, folds=folds, test=selected,
            test_fixed_05=metrics(sy, fixed_predictions, .5), labels=sy.tolist(), trials=trials.tolist(),
            class_1_probabilities=selected_predictions.tolist(), selected_threshold_predictions=predicted_classes.tolist()))
        write("subjects.json", results)
        print(f"subject={subject} within_subject_BA={selected['balanced_accuracy']:.3f}", flush=True)
    write("result.json", dict(status="completed", sklearn=sklearn.__version__, subjects=results,
          mean_test_balanced_accuracy=float(np.mean([r["test"]["balanced_accuracy"] for r in results])),
          mean_test_fixed_05_balanced_accuracy=float(np.mean([r["test_fixed_05"]["balanced_accuracy"] for r in results]))))
    archive = output / "within-subject-logpower-diagnostics.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in [output / n for n in ("config.json", "subjects.json", "result.json", "data/preprocessing.json")]:
            bundle.add(path, arcname=str(path.relative_to(output)))
    print(f"Download: {archive}", flush=True)


if __name__ == "__main__":
    main()
