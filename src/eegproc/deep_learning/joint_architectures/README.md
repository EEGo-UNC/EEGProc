# Subject Invariant Calibrator (SIC)

This directory contains the Subject Invariant Calibrator (SIC) model and its DREAMER data-loading support. SIC is designed for subject-independent valence and arousal classification, strict leave-one-subject-out (LOSO) evaluation, and zero- to multiple-shot user calibration.

## Directory layout

```text
EEGProc/
├── prepare_datasets.py
├── datasets/
│   ├── dreamer_eeg.npy
│   └── dreamer_labels.npy
└── src/eegproc/deep_learning/joint_architectures/
    ├── README.md
    ├── joint_v2_data.py
    └── sic/
        ├── sic_model.py
        ├── sic_model_args.py
        └── sic_model_train.py
```

## Files

| File | Purpose |
| --- | --- |
| `prepare_datasets.py` | Converts the raw DREAMER CSV into the NumPy arrays consumed by SIC. |
| `joint_v2_data.py` | Provides shared DREAMER loading, windowing, normalization, and data-configuration helpers. |
| `sic/sic_model.py` | Defines the SIC encoder, variational latent branches, classifier, decoder, subject adversary, and training objectives. |
| `sic/sic_model_args.py` | Defines and validates command-line arguments and experiment configurations. |
| `sic/sic_model_train.py` | Runs SIC experiments, LOSO cross-validation, user calibration, and report generation. |

## Prepare DREAMER

### 1. Arrange the raw input

Place `dreamer_joined.csv` in a directory of your choice. The directory passed to the script must contain the CSV directly:

```text
/absolute/path/to/dreamer_raw/
└── dreamer_joined.csv
```

Run all commands below from the EEGProc repository root. Activate the project environment first if required:

```bash
module load python/3.12.4

cd "$HOME/EEGProc"

python -m venv venv312

source venv312/bin/activate
```

### 2. Convert DREAMER and write directly to the SIC dataset directory

`prepare_datasets.py` has an `--output_dir` option, so the generated arrays do not need to be moved afterward. The SIC Slurm scripts use `datasets/` by default, so the recommended command is:

```bash
python prepare_datasets.py \
  --dataset dreamer \
  --dreamer_dir /absolute/path/to/dreamer_raw \
  --output_dir datasets/ \
  --verify
```

`--input_dir /absolute/path/to/dreamer_raw` can be used instead of `--dreamer_dir`, but `--dreamer_dir` makes the selected dataset explicit. The script creates the output directory when it does not already exist. `--verify` reloads the generated arrays and checks their shapes.

The command produces:

| Output | Shape | Contents |
| --- | --- | --- |
| `datasets/dreamer_eeg.npy` | `(23, 18, 42, 7680)` | `float32` EEG data: 23 subjects, 18 trials, 14 electrodes × 3 frequency bands, and 60 seconds at 128 Hz. |
| `datasets/dreamer_labels.npy` | `(23, 18, 2)` | `float32` trial labels in `[valence, arousal]` order. |

The converter excludes baseline rows, filters the complete contiguous stimulus segment, retains the middle 60 seconds, and produces theta, alpha, and beta features. Gamma is intentionally omitted, which is why the recommended directory is named ``.

If your local configuration expects the arrays directly under `datasets/`, change only the output path:

```bash
python prepare_datasets.py \
  --dataset dreamer \
  --dreamer_dir /absolute/path/to/dreamer_raw \
  --output_dir datasets \
  --verify
```

### 3. Move existing output when `--output_dir` was omitted

The default `--output_dir` is the current working directory. If the script was already run from the repository root without an output flag, move both generated files into the directory used by SIC:

```bash
mkdir -p datasets/
mv -i dreamer_eeg.npy dreamer_labels.npy datasets/
```

The `-i` option asks before replacing files that already exist. You do not need this move step when `--output_dir` was supplied.

## Details

SIC encodes every EEG window through two independent spatiotemporal paths:

- The GCN-GRU path learns spatial and spectral relationships from the source-subject mutual-information graph, then produces its own variational posterior `(mean_gcn, log_var_gcn)` and latent sample `z_gcn`.
- The BiLSTM path models forward and backward temporal context, then produces an independent variational posterior `(mean_bilstm, log_var_bilstm)` and latent sample `z_bilstm`.

The two latent samples are concatenated directly:

```text
z_joint = concatenate([z_gcn, z_bilstm])
```

There is no learned fusion projection between the branch latents and the classifier. The classifier is responsible for reducing and interpreting the concatenated feature space. The same joint latent also supports reconstruction and subject-adversarial learning, while ERM, V-REx, or MLDG controls how the source-training objective is optimized.

SIC is evaluated with strict DREAMER LOSO cross-validation. For each held-out user, the source model is first evaluated at zero shot and is then calibrated using complete target-user trials at the requested shot levels. Reports compare zero-shot and calibrated performance for both valence and arousal, with Brier score as the primary probability-calibration metric alongside classification and calibration metrics such as accuracy, balanced accuracy, F1, ROC-AUC, and ECE.
