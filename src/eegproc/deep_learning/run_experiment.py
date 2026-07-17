"""Small but *real* nested-LNSO experiment on DREAMER.

Unlike ``smoke_test_cross_val`` (which is a 1-epoch "does it run" check), this
script trains for real: more subjects, all trials, several epochs and a small
hyperparameter grid, so the figures show meaningful numbers.

It deliberately reuses the data prep and model builder from
``smoke_test_cross_val`` so there is a single source of truth, and only changes
the experiment configuration. Results are written to ``experiment_outputs/`` so
they never clobber the committed smoke-test results.

Run (from the repo root, inside the .venv that has TensorFlow)::

    python -m eegproc.deep_learning.run_experiment --subjects 6 --epochs 15

then turn the results into figures::

    python -m eegproc.plotting.report \
        --results experiment_outputs/dreamer_valence_results.json \
        --out figures --class-names "low valence,high valence"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_val import nested_lnso_cv
from .smoke_test_cross_val import (
    CSV_PATH,
    LABEL_COL,
    configure_printing,
    make_lstm_builder,
    prep_dataset,
    print_cv_result_summary,
    set_seed,
)

# Default experiment configuration (override from the CLI).
DEFAULT_SUBJECTS = 6
DEFAULT_SESSIONS = None  # None = use all 18 trials per subject
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 16
DEFAULT_LOSS = "softmax_crossentropy"
DEFAULT_SEED = 7

# A small grid: enough to make Fig 3 (hyperparameter sweep) non-trivial without
# blowing up the number of model fits.
HYPERPARAMETER_GRID = {
    "learning_rate": [1e-3, 1e-2],
    "lstm_units": [64],
    "n_lstm_layers": [1],
    "dropout": [0.3],
}


def run_experiment(
    name: str,
    subjects: int | None,
    sessions: int | None,
    epochs: int,
    batch_size: int,
    loss: str,
    out_dir: Path,
    seed: int,
) -> Path:
    configure_printing()
    set_seed(seed)

    dataset = prep_dataset(
        csv_path=CSV_PATH,
        label_col=LABEL_COL,
        max_subjects=subjects,
        max_sessions_per_subject=sessions,
    )

    print("\nExperiment dataset ready")
    print("=" * 80)
    print(f"X shape:            {dataset.X.shape}")
    print(f"subjects:           {np.unique(dataset.subject_ids)} -> {dataset.subject_id_mapping}")
    print(f"tasks (trial ids):  {sorted(np.unique(dataset.task_ids).tolist())}")
    print(f"classes:            {np.unique(dataset.y)} (label='{LABEL_COL}', threshold={dataset.label_threshold:.4f})")

    model_builder = make_lstm_builder(
        timesteps=dataset.X.shape[1],
        n_features=dataset.X.shape[2],
        n_classes=len(np.unique(dataset.y)),
        loss_name=loss,
    )

    results = nested_lnso_cv(
        model_builder_function=model_builder,
        feature_array=dataset.X,
        label_array=dataset.y,
        subject_id_array=dataset.subject_ids,
        task_id_array=dataset.task_ids,
        n_outer_subjects_to_leave_out=1,
        n_inner_subjects_to_leave_out=1,
        n_epochs=epochs,
        batch_size=batch_size,
        hyperparameters=HYPERPARAMETER_GRID,
        metrics=["accuracy", "f1", "precision", "recall"],
        selection_metric="accuracy",
        maximize_metric=True,
        log_predictions=True,
        log_variational_intervals=(loss == "variational"),
        verbose=0,
    )

    # Persist the integer-code -> original-id mapping for nicely labelled figures.
    results["subject_id_mapping"] = dataset.subject_id_mapping

    print_cv_result_summary(name=name, results=results)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{name}_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        # Wrap as {name: results} so the report labels the model with `name`.
        json.dump({name: results}, handle, indent=2)

    print(f"\nSaved experiment results: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small real nested-LNSO DREAMER experiment.")
    parser.add_argument("--name", default="dreamer_valence", help="Experiment name (used in filenames/labels).")
    parser.add_argument("--subjects", type=int, default=DEFAULT_SUBJECTS, help="Number of subjects (outer folds).")
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS, help="Trials per subject (default: all).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--loss", default=DEFAULT_LOSS, help="'softmax_crossentropy' or 'variational'.")
    parser.add_argument("--out", default="experiment_outputs", help="Output directory for the results JSON.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    run_experiment(
        name=args.name,
        subjects=args.subjects,
        sessions=args.sessions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss=args.loss,
        out_dir=Path(args.out),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
