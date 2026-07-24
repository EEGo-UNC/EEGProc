# EEGProc GCN-only softmax baseline

This folder adds a deterministic GCN-only classification baseline to EEGProc.
It reuses the project's existing `GCNEncoder`, DREAMER loader, and `loso_cv`
implementation, while removing the decoder, VAE sampling, BiLSTM, variational
classifier, and all VC losses.

## Architecture

```text
EEG window (T x 56)
  -> GCNEncoder
  -> temporal mean + max pooling
  -> Dense(64, ReLU)
  -> Dropout(0.5)
  -> Dense(2) logits
  -> softmax only during probability conversion/evaluation
```

The final layer intentionally returns logits. The model uses
`SparseCategoricalCrossentropy(from_logits=True)`, and EEGProc's `cross_val.py`
converts the logits to softmax probabilities exactly once.

## Install into EEGProc

Copy the entire folder to:

```text
EEGProc/src/eegproc/deep_learning/gcn_softmax/
```

From the repository root, verify the architecture:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m eegproc.deep_learning.gcn_softmax.smoke_test
```

## SLURM

Run the two-fold smoke test first:

```bash
sbatch src/eegproc/deep_learning/gcn_softmax/slurm/run_gcn_softmax_smoke.slurm
```

Then run the full DREAMER arousal LOSO experiment:

```bash
sbatch src/eegproc/deep_learning/gcn_softmax/slurm/run_gcn_softmax.slurm
```

The scripts use the Longleaf modules and L40 partition settings used by the
existing EEGProc experiments. Change the `#SBATCH` lines if your allocation or
cluster policy differs.

## Data loading

By default, the training script calls the same
`load_joint_v2_training_data(...)` function used by the joint model. Therefore,
the GCN baseline receives the same preprocessing, windows, labels, subject IDs,
and trial IDs.

Prepared arrays can also be passed explicitly:

```bash
python -m eegproc.deep_learning.gcn_softmax.train \
  --features-npy path/features.npy \
  --labels-npy path/labels.npy \
  --subjects-npy path/subjects.npy \
  --trials-npy path/trials.npy \
  --n-channels 14 \
  --n-bands 4
```

All four arrays are required in prepared-array mode.

## Configurations

- `configs/gcn_medium.json`: one recommended model, `[32, 16]` GCN layers.
- `configs/gcn_size_grid.json`: six architecture/regularization combinations
  using `[32]`, `[32, 16]`, or `[64, 32]` and dropout `0.3` or `0.5`.

To run the size grid, replace the configuration path in the full SLURM script:

```text
configs/gcn_medium.json -> configs/gcn_size_grid.json
```

## Outputs

Each run creates a timestamped directory containing:

- `command.json`
- `dataset_summary.json`
- `hyperparameters.json`
- `loso_cv_results.json`
- `loso_cv_folds.csv`
- `selected_config.json`
- `final_training_plan.json`
- `final_model.keras`
- `final_model.weights.h5`
- `final_history.csv`

The final model uses the median available cross-validation `best_epoch` value,
unless `--final-epochs` is provided. Use `--no-final-model` for diagnostic runs.

## Useful variants

Linear GCN baseline:

```json
{
  "temporal_readout": ["mean"],
  "classifier_units": [0],
  "classifier_dropout": [0.5]
}
```

Larger original-capacity GCN:

```json
{
  "gcn_units": [[64, 32]],
  "emb_dim": [32],
  "dropout": [0.3]
}
```
