# SIC v16 tracker workflow

For quota failures or interrupted baselines, use [recovery instructions](RECOVERY.md).
They preserve checkpoints and resume existing runs on shared work storage.

The four real-data smoke runs have passed. See [the review](SMOKE_REVIEW.md)
for learning results, measured resources and the next full baseline step.

Use the existing tracker for planning, status, syncing and plotting. The model
and oracle selection are the same as the SICv16 Slurm array runner; this bridge
adds portable run manifests and `result.json` output. Each architecture has its
own campaign/store, with valence as its label target. Initial variants are
`cnn`, `cnn_reconstruction`, `cnn_transformer`, `cnn_transformer_reconstruction`.

## Next run: tracked real-data smoke

On Longleaf, from the repository root after pulling `SICv16-Pranav`:

```bash
module load python/3.12.4
source venv312/bin/activate
python -m pip install -e '.[deep-learning]'
python experiments/sic_v16/tracker.py --variant all plan --target valence
```

Planning uses only the standard library and executes no training. It creates
four separate bundles under `experiments/sic_v16/VARIANT/runs/RUN_ID/`, with
`config.json`, `manifest.json`, and `submit.slurm`, and prints their exact paths.
Submit each printed command, which has this form:

```bash
sbatch experiments/sic_v16/VARIANT/runs/RUN_ID/submit.slurm
```

Generated smoke scripts request two hours; later full scripts request 24 hours.
A smoke run's controller budget is 6,600 seconds. Each smoke uses two epochs,
two target subjects (0 and 1), all available source subjects except the held-out
target and validation pool, batch size one trial and one GPU. Raw 14-channel
DREAMER or band-stacked 42-feature arrays are accepted. The source files are
never modified. `DATA_DIR` defaults to `$HOME/EEGProc/datasets` when submitted
from that checkout. `PROJECT_DIR` defaults to the submission directory.

Each job first executes the new CPU-only data/bridge checks, then its fixed
training configuration. The eight model/GPU preflight tests have already
passed in Longleaf job 6610071 and are not rerun by these jobs.

## Inspect and continue

```bash
python experiments/sic_v16/tracker.py --variant all status
python experiments/sic_v16/tracker.py --variant cnn suggest --target valence
python experiments/sic_v16/tracker.py --variant cnn plot
```

If planning/running on Longleaf, `status` reads local results directly: no sync
is necessary. If planning on a laptop, copy the planned bundle to the same
repo-relative cluster directory before submission, and download each completed
bundle's top-level files afterward. Import one variant at a time:

```bash
python experiments/sic_v16/tracker.py --variant cnn sync /path/to/downloaded/cnn/run
```

Tracker sync copies top-level files only; these contain everything needed for
status, selection and plots. Keep `artifacts/` on Longleaf: it contains prepared
data, per-fold logs/checkpoints, epoch predictions and split diagnostics. A
failed job updates `manifest.json` and writes `failure.txt`; preprocessing
failures also have `artifacts/execution.json` and data/preprocessing.json when
the input could be opened.

Send back the first smoke bundles' `slurm-*.out`, `slurm-*.err`, `manifest.json`,
`result.json` or `failure.txt`, plus `artifacts/subject_*.log` and per-epoch JSONs.
No checkpoints need downloading initially. Do not also submit the legacy
`smoke.sh` array: it runs the same four experiments without tracker bundles.

## When routine runs replace this validation loop

You can use the tracker now to plan and submit this smoke run. Once a real-data
smoke completes with finite metrics, valid checkpoints and acceptable memory/
timing, routine runs use `plan -> sbatch -> status` (or `sync` for laptop stores).
No new hand-written Slurm script is needed per configuration. The tracker does
not submit jobs automatically; it prints the ready-to-submit command.

After smoke, the next suggestion is the baseline (50 epochs, all 23 subjects),
then one-factor learning-rate/dropout experiments. Do not submit that baseline
until smoke timing establishes whether its sequential folds fit the 24-hour
cap. We may need fold-level jobs or a revised matched budget first. This is a
runtime-validation gate, not a tracker limitation. Model or data errors still
need investigation when they occur.

Tracker ranking uses **validation-selected test balanced accuracy**, with a
subject-variability penalty. Test-selected common-epoch oracle performance is
retained separately in `result.json` under `oracle_diagnostics` and is never
the tracker optimization objective. Keep target sets, budgets and preprocessing
matched across the four variant campaigns.

The shared tracker's [general guide](../../docs/training-tracker.md) describes
configuration and reporting. Its older standalone example campaigns are not
part of this branch; use the SIC v16 wrapper above.
