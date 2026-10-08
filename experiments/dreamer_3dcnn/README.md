# DREAMER 3D-CNN experiment campaign

This folder turns the DREAMER 3D-CNN work into a sequence of reproducible,
tracked Longleaf runs. It keeps planning lightweight (standard-library Python),
executes training through EEGProc's LOSO evaluator, and treats downloaded run
folders as the source of truth. No database or external tracking service is
required.

## The loop

From the `sic-3dcnn` project root, ask for the next experiment:

```bash
python experiments/dreamer_3dcnn/tracker.py suggest --target valence
python experiments/dreamer_3dcnn/tracker.py plan --target valence
```

`plan` creates `experiments/dreamer_3dcnn/runs/<run-id>/` containing an exact
`config.json`, a lifecycle `manifest.json`, and a ready-to-submit
`submit.slurm`. Run folders are intentionally git-ignored, so after updating
the code checkout on Longleaf, copy that one folder to the same repo-relative
location (for example with `scp -r` or an existing sync tool), then submit the
command printed by `plan`.

The launcher assumes these defaults, all overrideable as environment variables:

- repository: `$HOME/EEGProc` (`PROJECT_DIR`)
- environment: `$PROJECT_DIR/venv312` (`VENV_DIR`)
- arrays: `$PROJECT_DIR/datasets/dreamer_eeg.npy` and
  `dreamer_labels.npy` (`DATA_DIR`)

Prepare the environment once on Longleaf:

```bash
module load python/3.12.4
python -m venv "$HOME/EEGProc/venv312"
source "$HOME/EEGProc/venv312/bin/activate"
python -m pip install -e "$HOME/EEGProc[deep-learning]"
```

After the job, download the whole run folder. It is safe to place it in a
temporary download directory; import it locally with:

```bash
python experiments/dreamer_3dcnn/tracker.py sync /path/to/downloaded/run-folder
python experiments/dreamer_3dcnn/tracker.py status
python experiments/dreamer_3dcnn/tracker.py suggest --target valence
```

Sync validates the campaign and config fingerprint, refuses conflicting
content, copies logs/results, and rebuilds `runs/index.csv` and
`runs/REPORT.md`. The run directory is git-ignored because results and Slurm
logs can be large; the planner/configuration code remains versioned.

## How “next” is chosen

Each target (`valence` and `arousal`) has an independent campaign:

1. a two-fold, two-epoch smoke test;
2. the full baseline across all 23 held-out subjects;
3. controlled one-factor experiments around the current incumbent.

The incumbent maximizes mean trial balanced accuracy with a small penalty for
between-subject standard deviation. Search dimensions are ordered in
`campaign.json`, so the campaign tests optimization/stability settings before
increasing model capacity. A proposed config is never repeated, including
planned or failed runs. Edit `search_space` when the configured candidates are
exhausted.

The model is a standalone MTLFuseNet-style 3D-CNN encoder followed by temporal
average pooling and a small dense classifier. Ratings are binarized at 3.0.
Each one-second window is standardized independently, which avoids fitting a
normalizer on held-out subjects. Hyperparameter selection uses trial-level
balanced accuracy, while accuracy, F1, macro-F1, ROC AUC, Brier score, and ECE
are retained as diagnostics.
