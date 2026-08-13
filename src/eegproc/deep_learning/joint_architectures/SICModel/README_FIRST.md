# SIC Direct-Concatenation v10 Bundle

This bundle contains the updated SIC files and Slurm suite for builder API 10.

## Architecture guaranteed by this version

For each EEG window:

- the GCN-GRU branch emits `spectral_gru_units` features (`384` in the supplied scripts);
- the BiLSTM emits `2 * bilstm_units` features because its forward and backward outputs are concatenated (`126` when `bilstm_units=63`);
- the complete active branch outputs are concatenated directly;
- temporal pooling changes the time axis only and never projects the feature axis;
- the classifier and subject-adversarial heads receive the complete pooled representation.

The full supplied configuration therefore passes `384 + 126 = 510` features to each final head. There is no encoder `z` dimension, feature projection, decoder, reparameterization, or encoder VAE loss.

## Bundle layout

Copy the bundle contents over an existing EEGProc checkout while preserving paths:

```text
EEGProc/
├── README_SIC.md
├── SIC_MLDG_HYPERPARAMETER_REFERENCE.md
├── smoke_test_sic_direct_concat_v10.sh
├── full_run_sic_direct_concat_v10.sh
├── submit_sic_direct_concat_v10_suite.sh
└── src/eegproc/deep_learning/
    ├── cross_val.py
    ├── training_outputs.py
    ├── generalize_optimization_strats/
    │   └── MetaLearning.py
    └── joint_architectures/SICModel/
        ├── sic_model.py
        ├── sic_model_train.py
        └── sic_model_args.py
```

The surrounding EEGProc repository must already provide `requirements.txt`, `joint_models_data.py`, `supervised/variational_classifier.py`, `unsupervised/GNN/GCNMTL.py`, and the DREAMER arrays.

## Run the smoke-gated suite

From the EEGProc root:

```bash
chmod +x \
  smoke_test_sic_direct_concat_v10.sh \
  full_run_sic_direct_concat_v10.sh \
  submit_sic_direct_concat_v10_suite.sh

./submit_sic_direct_concat_v10_suite.sh full-after-smoke 2
```

Valence is the default. For arousal:

```bash
TARGET_DIMENSION=arousal \
  ./submit_sic_direct_concat_v10_suite.sh full-after-smoke 2
```

The worker scripts verify `SIC_BUILDER_API_VERSION == 10` before training.

## Slurm arrays

The smoke array has two tasks:

- `0=full`
- `1=no_bilstm`

The full array has four tasks:

- `0=full`
- `1=remove_median`
- `2=no_gcn_gru`
- `3=no_bilstm`

Use `TRAINING_METHOD=erm`, `TRAINING_METHOD=vrex`, or `TRAINING_METHOD=mldg` to select source optimization. Prediction diagnostics default to Brier score and can be changed with `PREDICTION_DIAGNOSTICS_METRIC`.
