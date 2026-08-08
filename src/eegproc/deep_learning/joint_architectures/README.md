# Joint architectures

EEGProc implementations of the repository's joint EEG modeling variants for DREAMER classification.

This package collects three related architectures for subject-independent emotion recognition:

- `joint_v1`: a first-pass joint autoencoder + variational-classifier model.
- `joint_v2_vae_vc`: an improved variational autoencoder and classifier pipeline with configurable CV and training entry points.
- `joint_v3_sts`: a fused spatiotemporal-spatiospectral model that combines BiLSTM and GCN paths with a variational latent representation.

## Directory layout

```text
EEGProc/
├── src/eegproc/deep_learning/joint_architectures/
│   ├── README.md
│   ├── joint_v2_data.py
│   ├── joint_v1/
│   │   └── joint_v1_autoencoder_vc.py
│   ├── joint_v2_vae_vc/
│   │   ├── joint_v2_autoencoder_vc.py
│   │   └── joint_v2_autoencoder_vc_train.py
│   └── joint_v3_sts/
│       ├── joint_sts_model.py
│       └── joint_sts_model_train.py
├── datasets/
│   └── dreamer_joined.csv
└── runs/
    └── joint_*
```

## Files

| File | Purpose |
| --- | --- |
| `joint_v2_data.py` | Shared DREAMER dataset loading, preprocessing, and configuration helpers. |
| `joint_v1/joint_v1_autoencoder_vc.py` | Reference joint autoencoder + variational-classifier architecture. |
| `joint_v2_vae_vc/joint_v2_autoencoder_vc.py` | Improved joint VAE/variational-classifier implementation. |
| `joint_v2_vae_vc/joint_v2_autoencoder_vc_train.py` | Training entry point for the joint V2 pipeline with CV and reporting hooks. |
| `joint_v3_sts/joint_sts_model.py` | Fused spatiotemporal-spatiospectral STS architecture. |
| `joint_v3_sts/joint_sts_model_train.py` | Training entry point for the STS pipeline with LOSO/LNSKTO support. |

## Prepare DREAMER

Run from the EEGProc repository root:

```bash
cd "$HOME/EEGProc"
source venv312/bin/activate
```

The shared dataset helpers in `joint_v2_data.py` expect the DREAMER EEG and labels files to be available under the repository's dataset paths.

## Run training

Each architecture has its own training entry point. Typical commands look like:

```bash
cd "$HOME/EEGProc"

python -m src.eegproc.deep_learning.joint_architectures.joint_v2_vae_vc.joint_v2_autoencoder_vc_train --help
python -m src.eegproc.deep_learning.joint_architectures.joint_v3_sts.joint_sts_model_train --help
```

Use the `--help` output to inspect the supported arguments for dataset selection, fold strategy, batch size, learning rate, and output directories.

## Details

The joint architectures package spans three complementary modeling ideas:

1. `joint_v1` focuses on a single shared encoder-decoder pathway with a variational classifier head.
2. `joint_v2_vae_vc` extends the idea with a more configurable training pipeline and a repository-standard training interface.
3. `joint_v3_sts` adds a richer fused representation that combines temporal and spectral information before classification.

The training entry points reuse EEGProc's shared cross-validation, thresholding, diagnostics, and result-writing utilities so the training behavior is consistent with the rest of the repository.
