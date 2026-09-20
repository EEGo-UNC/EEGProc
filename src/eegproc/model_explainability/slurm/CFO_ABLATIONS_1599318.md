<<<<<<< HEAD
# Four CFO ablations: DREAMER arousal, suite 1599318
=======
# Three CFO ablations: DREAMER arousal, suite 1599318
>>>>>>> 133379d27a7a807b622d6bb33f5e823136ba7493

`run_cfo_ablations_arousal_1599318.sh` uses the suite's saved winner,
`all_subjects/full_run_v15_arousal_scale64_20260918_155429/configuration_0001`.
It loads zero-shot LOSO checkpoints from the manifest and runs these arms in
one study per subject:

| Output arm | Target | Latent | Decoded | Physiology | Typicality |
| --- | ---: | ---: | ---: | ---: | ---: |
| `target_latent` | 1 | 0.1 | 0 | 0 | 0 |
| `base` — full CFO without typicality | 1 | 0.1 | 0.1 | 1 | 0 |
| `typicality` — full CFO with typicality | 1 | 0.1 | 0.1 | 1 | 1 |
<<<<<<< HEAD
| `typicality_no_physiology` — CFO with typicality, without physiological constraint | 1 | 0.1 | 0.1 | 0 | 1 |
=======
>>>>>>> 133379d27a7a807b622d6bb33f5e823136ba7493

These are configurable development defaults. `TARGET_WEIGHT`, `LATENT_WEIGHT`,
`DECODED_WEIGHT`, `PHYSIOLOGICAL_WEIGHT`, and `TYPICALITY_WEIGHT` set the full
objective's weights. The runner always zeros decoded, physiology, and
<<<<<<< HEAD
typicality contributions for `target_latent`, typicality for `base`, and
physiology for `typicality_no_physiology`. The fourth arm retains the same
target, latent, decoded, and typicality weights as the full typicality arm;
physiology does not affect its optimization or feasibility gate.
All four arms still receive all endpoint diagnostics, including VCSC and the
physiological checks.
=======
typicality contributions for `target_latent`, and typicality for `base`.
All three arms still receive all endpoint diagnostics.
>>>>>>> 133379d27a7a807b622d6bb33f5e823136ba7493

Run from the cluster EEGProc repository root after syncing the launcher and
its accompanying `typicality/runner.py` and `typicality/results.py` updates:

```bash
<<<<<<< HEAD
# First: subject 0, all its eligible trials, all four conditions.
=======
# First: subject 0, all its eligible trials, all three conditions.
>>>>>>> 133379d27a7a807b622d6bb33f5e823136ba7493
sbatch src/eegproc/model_explainability/slurm/run_cfo_ablations_arousal_1599318.sh

# Later: all 23 subjects, at most four one-GPU jobs concurrently.
sbatch --array=0-22%4 src/eegproc/model_explainability/slurm/run_cfo_ablations_arousal_1599318.sh
```

The default is one subject; the script does not submit the population run
<<<<<<< HEAD
automatically. Each subject runs its four arms sequentially on one GPU,
=======
automatically. Each subject runs its three arms sequentially on one GPU,
>>>>>>> 133379d27a7a807b622d6bb33f5e823136ba7493
calibrating references once. All arms use the same model, eligible trials,
initial latent, and trial-specific seed. Eligibility is true class 0 and
original argmax class 0, targeting class 1. A subject with no eligible trials
is explicitly reported as having performed no CFO optimizations.

Defaults are joint decoding, target probability 0.80, learning rate 0.01,
no learning-rate decay, at most 200 updates, and seed 42. The current runner's
stopping policy is retained: the first two arms stop when their target and
<<<<<<< HEAD
active physiological constraint are feasible; both typicality arms continue
their typicality phase within the same maximum update budget. In
`typicality_no_physiology`, target attainment alone opens that phase because
the physiological optimization weight is zero.
=======
active physiological constraint are feasible; the typicality arm then
continues its typicality phase within the same maximum update budget.
>>>>>>> 133379d27a7a807b622d6bb33f5e823136ba7493
`STOP_ON_SUCCESS=0` disables early target stopping for the first two arms.
Gradient-stall and typicality-improvement stopping remain separately
configurable through the variables in the script.

For a short run, restrict the optimization steps and optionally select known
eligible trial IDs. Calibration continues to use the complete dataset:

```bash
MAX_STEPS=5 sbatch src/eegproc/model_explainability/slurm/run_cfo_ablations_arousal_1599318.sh
```

Set `TRIAL_IDS="8 10"` only when those trials are the intended subset; they
must still pass eligibility. To preview the exact invocation locally without
loading modules, using a GPU, or writing results:

```bash
DRY_RUN=1 VENV_DIR=venv bash src/eegproc/model_explainability/slurm/run_cfo_ablations_arousal_1599318.sh
```

`PROJECT_DIR` defaults to the repository found from the Slurm submission
directory; `VENV_DIR` defaults to `venv312`. `CONFIG_DIR`, `MODELS_JSON`, and
`MODEL_DIR` can relocate the saved configuration. Raw DREAMER files default
to `datasets/dreamer_eeg.npy` and `datasets/dreamer_labels.npy`; a complete
prepared dataset can instead be supplied with `TRIALS_NPZ`.

Each job writes under:

```text
runs/counterfactuals/arousal_suite_1599318_ablations_JOB_ID/
  fold_00/
    study.json
    subject_0/
      calibration/
      fold.json
      trial_TRIAL_ID/
        target_latent/attempt_0001/
        base/attempt_0001/
        typicality/attempt_0001/
<<<<<<< HEAD
        typicality_no_physiology/attempt_0001/
=======
>>>>>>> 133379d27a7a807b622d6bb33f5e823136ba7493
    report/
      trial_metrics.csv
      population_counterfactuals.csv
      subject_counterfactuals.csv
      tables.tex
      results.json
```

<<<<<<< HEAD
The study manifest records the four arms; the fold manifest records their
effective loss weights. Reports include all four arms. The existing
=======
The study manifest records the three arms; the fold manifest records their
effective loss weights. Reports include all three arms. The existing
>>>>>>> 133379d27a7a807b622d6bb33f5e823136ba7493
`subject_typicality_changes.csv` remains the full-CFO base-versus-typicality
comparison. Existing paired plot/probe tools still focus on those two arms.
Failed targets remain in the reports; optimization errors cause the launcher
to fail. Existing outputs are protected. Resume with the same `OUT_ROOT` and
`RESUME=1`; the runner verifies data, model, code, and protocol fingerprints.
<<<<<<< HEAD
Use a new run directory for this four-arm protocol; a previous three-arm run
cannot be resumed as a four-arm run.
=======
>>>>>>> 133379d27a7a807b622d6bb33f5e823136ba7493

After all requested subjects finish, combine their tables locally:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.typicality.results \
  "$OUT_ROOT"/fold_* --out-dir "$OUT_ROOT/report"
```

Set `OUT_ROOT` to the completed array's directory first. Confirm all requested
folds are present before combining; the reporting command only knows about
the fold directories supplied to it.
