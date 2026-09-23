"""Historical-archive subject probe with matched, disjoint trial splits.

Each LOSO fold can learn a different coordinate basis. A probe across those
folds may identify the checkpoint instead of the subject. The CLI therefore
requires an explicit coordinate policy and reports this limitation with the
numbers. No checkpoint inference or counterfactual optimization is needed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from .artifacts import write_json, write_csv
from .results import collect_study, compatible_typicality_definition
from .core import SCORE_DEFINITION


def prepare_probe(studies, *, representation="classification_embedding"):
    studies = list(studies)
    definition = compatible_typicality_definition(studies)
    if definition["score"] != SCORE_DEFINITION:
        raise ValueError("Full-trial subject probes require a new Mahalanobis study; legacy window-moment archives are incompatible")
    if representation != "classification_embedding":
        raise ValueError("The probe uses full-trial classification_embedding vectors")
    groups, coordinate_spaces, expected_subjects = {}, {}, set()
    task = None
    for root in studies:
        root = Path(root)
        manifest = json.loads((root / "study.json").read_text())
        if task is not None and task != manifest["task"]:
            raise ValueError("Probe one emotion task at a time")
        task = manifest["task"]
        if manifest.get("artifact_format") == "json_csv_summaries":
            raise ValueError("Subject probes require saved classification embeddings, which summary-only "
                             "runs do not retain. Use a historical array archive or recompute embeddings.")
        expected_subjects.update(fold["subject_id"] for fold in manifest["folds"])
        for row in collect_study(root)[0]:
            if row["objective"] not in ("base", "typicality"):
                continue
            key = (row["subject_id"], row["trial_id"])
            pair = groups.setdefault(key, {})
            if row["objective"] in pair:
                raise ValueError("Duplicate trial/objective in probe inputs")
            pair[row["objective"]] = row
        for fold in manifest["folds"]:
            coordinate_spaces[fold["subject_id"]] = fold["sha256"]
    features = {name: [] for name in ("original", "base", "typicality")}
    keys, excluded = [], []
    for key, pair in sorted(groups.items()):
        if set(pair) != {"base", "typicality"} or any(row["status"] != "completed" for row in pair.values()):
            excluded.append(list(key))
            continue
        with np.load(Path(pair["base"]["artifact_directory"]) / "counterfactual.npz", allow_pickle=False) as data:
            original, base = data["classification_embedding"], data["classification_embedding_prime"]
        with np.load(Path(pair["typicality"]["artifact_directory"]) / "counterfactual.npz", allow_pickle=False) as data:
            constrained = data["classification_embedding_prime"]
            if not np.allclose(original, data["classification_embedding"], rtol=1e-5, atol=1e-6):
                raise ValueError("Paired arms have different original representations")
        for name, embedding in (("original", original), ("base", base), ("typicality", constrained)):
            if embedding.ndim != 2 or embedding.shape[0] != 1 or not np.isfinite(embedding).all():
                raise ValueError("Expected one finite full-trial classification embedding")
            features[name].append(embedding[0])
        keys.append(key)
    if not keys:
        raise ValueError("No matched finite counterfactual pairs are available")
    try:
        features = {name: np.stack(values) for name, values in features.items()}
    except ValueError as error:
        raise ValueError("Probe representation dimensions differ across folds") from error
    metadata = {"task": task, "representation": representation, "typicality_definition": definition,
                "n_eligible_pairs": len(groups), "n_included_pairs": len(keys), "excluded_trial_keys": excluded,
                "expected_subject_ids": sorted(expected_subjects),
                "coordinate_spaces": coordinate_spaces,
                "cohort": "all eligible trials with both finite archived endpoints, including failed class flips"}
    return np.asarray(keys), features, metadata


def fit_subject_probe(keys, representations, output, *, coordinate_policy, coordinate_spaces,
                      seed=42, test_fraction=0.3, regularization_c=1.0, metadata=None):
    if coordinate_policy not in ("shared", "fold_specific_descriptive"):
        raise ValueError("Declare shared or fold_specific_descriptive coordinate policy")
    if coordinate_policy == "shared" and len(set(coordinate_spaces.values())) != 1:
        raise ValueError("Independent LOSO checkpoints are not a shared latent coordinate system")
    keys = np.asarray(keys, dtype=int)
    if keys.ndim != 2 or keys.shape[1] != 2 or len(set(map(tuple, keys))) != len(keys):
        raise ValueError("Expected unique (subject,trial) keys")
    if not 0 < test_fraction < 1 or not np.isfinite(regularization_c) or regularization_c <= 0:
        raise ValueError("Invalid test fraction or probe regularization")
    subjects = np.unique(keys[:, 0])
    if len(subjects) < 2:
        raise ValueError("Subject identification requires at least two subjects")
    train, test = [], []
    rng = np.random.default_rng(seed)
    for subject in subjects:
        indices = rng.permutation(np.flatnonzero(keys[:, 0] == subject))
        if len(indices) < 2:
            raise ValueError(f"Subject {subject} needs at least two paired trials for disjoint training/evaluation")
        n_test = min(len(indices) - 1, max(1, int(np.ceil(len(indices) * test_fraction))))
        test.extend(indices[:n_test].tolist())
        train.extend(indices[n_test:].tolist())
    train, test = np.asarray(sorted(train)), np.asarray(sorted(test))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "probe_splits.csv", [{"subject_id": int(subject), "trial_id": int(trial),
               "split": "train" if index in train else "test"} for index, (subject, trial) in enumerate(keys)])
    rows, predictions = [], []
    for name, values in representations.items():
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or len(values) != len(keys) or not np.isfinite(values).all():
            raise ValueError("Probe features must be finite and aligned to trial keys")
        scaler = StandardScaler().fit(values[train])
        classifier = LogisticRegression(C=regularization_c, class_weight="balanced", random_state=seed, max_iter=2000)
        classifier.fit(scaler.transform(values[train]), keys[train, 0])
        probabilities = classifier.predict_proba(scaler.transform(values[test]))
        predicted = classifier.classes_[probabilities.argmax(axis=1)]
        row = {"representation": name, "balanced_accuracy_percent": 100 * float(balanced_accuracy_score(keys[test, 0], predicted)),
               "chance_percent": 100 / len(subjects), "n_subjects": len(subjects), "n_train": len(train), "n_test": len(test),
               "converged": bool(np.all(classifier.n_iter_ < classifier.max_iter))}
        rows.append(row)
        predictions.extend({"representation": name, "subject_id": int(keys[index, 0]),
                            "trial_id": int(keys[index, 1]), "predicted_subject": int(predicted[j])}
                           for j, index in enumerate(test))
    results = {**(metadata or {}), "schema_version": 1, "rows": rows, "seed": seed,
               "test_fraction": test_fraction, "regularization_c": regularization_c,
               "coordinate_policy": coordinate_policy, "coordinate_spaces": coordinate_spaces,
               "limitation": ("Fold and subject are confounded; these accuracies do not establish subject invariance."
                              if coordinate_policy == "fold_specific_descriptive" else None),
               "split_unit": "entire trial; identical split for all three representations; scaler fitted on training trials only"}
    write_json(output / "subject_identifiability.json", results)
    write_csv(output / "subject_identifiability.csv", rows)
    write_csv(output / "probe_predictions.csv", predictions)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--coordinate-policy", required=True, choices=("shared", "fold_specific_descriptive"))
    parser.add_argument("--representation", choices=("classification_embedding",), default="classification_embedding")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    args = parser.parse_args(argv)
    keys, representations, metadata = prepare_probe(args.studies, representation=args.representation)
    fit_subject_probe(keys, representations, args.out_dir, coordinate_policy=args.coordinate_policy,
                      coordinate_spaces=metadata["coordinate_spaces"], seed=args.seed,
                      test_fraction=args.test_fraction, metadata=metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
