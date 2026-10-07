# DREAMER BiLSTM and counterfactual example

[`dreamer_bilstm_counterfactual.py`](dreamer_bilstm_counterfactual.py) connects
EEGProc's preprocessing, PSD features, BiLSTM builder, LOSOCV, and model-agnostic
counterfactual optimizer. It needs Python 3.10+ and a separately downloaded
DREAMER dataset; see the [dataset guide](../docs/source/datasets.md#dreamer).

## Run from the repository root

```bash
# Install this checkout with TensorFlow and the cross-validation dependencies.
python -m pip install -e ".[deep-learning]"

# Convert the downloaded MATLAB file once (replace these paths with your own).
eegproc-to-csv --dataset dreamer --input /path/to/DREAMER.mat \
  --output /path/to/dreamer.csv.gz

# Run every LOSO fold, then explain one held-out input from the final fold.
python examples/dreamer_bilstm_counterfactual.py /path/to/dreamer.csv.gz \
  --epochs 10 --output outputs/dreamer_example
```

An existing converted `.csv` or `.csv.gz` works directly. Conversion needs enough
RAM for `DREAMER.mat`; the example then loads the selected CSV columns into memory.
For a quick execution check, use `--epochs 1`; this still evaluates all subjects
and is not a useful performance benchmark. Reusing an output directory replaces
these example outputs.

## What each step does

1. Keep stimulus samples, sort by subject/trial/time, and label valence ratings
   1–2 as low and 3–5 as high. Including neutral (3) in high is an explicit example
   choice, not a dataset-mandated threshold. Only the 14 EEG channels are used as inputs here.
2. Process each trial separately: common-average reference, 50 Hz notch,
   detrending, and the library's six default frequency bands. Compute PSD powers
   in non-overlapping two-second windows at 128 Hz.
3. Form sequences of four PSD rows: `(4 timesteps, 84 channel-band features)`.
   This gives the BiLSTM eight seconds of temporal context. Incomplete trailing
   sequences are dropped; trials need at least eight seconds of stimulus EEG.
   The library standardizes each subject using all of that subject's unlabeled
   feature rows, **including the held-out subject's rows**. This assumes offline
   access to the subject's recordings, not training-only normalization or online
   inference on a previously unseen recording.
4. Run `strategy="fixed_loso"`: a fresh one-layer BiLSTM with 16 units in each
   direction and a two-class softmax, trained for the fixed epoch count on all
   other subjects. EEGProc reports trial metrics by averaging sequence
   probabilities. Settings are illustrative, not tuned; do not choose them by
   repeatedly inspecting these held-out scores.
5. Reload the final fold's model and optimize the first sequence from its held-out
   subject toward the opposite **predicted** class with target probability 0.8.
   The built-in `KerasInputAdapter` changes standardized PSD features while the
   model weights stay fixed. This explains one sequence, not the entire trial's
   averaged prediction. It does not reconstruct EEG or enforce physiological
   plausibility, and the probability target may not be reached.

## Outputs

- `loso_metrics.csv`: per-subject trial metrics.
- `last_fold.keras`: the final fold's classifier. Keep `n_jobs=1`: the shared
  checkpoint is intentionally overwritten in sequential fold order.
- `counterfactual.json`: source subject/trial/label, target class, original and
  final probabilities, distances, and the `counterfactual.success` flag.
- `history.csv`: optimization progress.
- `counterfactual.npz`: original (`x`) and changed (`x_prime_input`) standardized
  features, optimization states, and ordered `feature_names`. Values are not in
  original signal units.

The example uses existing public APIs; it adds no library functionality.
