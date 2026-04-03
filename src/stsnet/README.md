# STSNet — TensorFlow Implementation

A clean, modular, GPU-parallelisable TensorFlow 2 implementation of:

> Li et al., **"STSNet: a novel spatio-temporal-spectral network for
> subject-independent EEG-based emotion recognition"**,
> *Health Information Science and Systems*, 2023.
> https://doi.org/10.1007/s13755-023-00226-x

---

## Repository layout

```
stsnet/
├── data_representation.py   # EEG → 4-D SPD tensor (ManifoldNet) & flattened sequence (BiLSTM)
├── manifold_net.py          # ManifoldNet sub-model  (wFM layers + invariant layer → MO)
├── prepare_datasets.py      # Convert raw DEAP and DREAMER dataset files into the NumPy format
├── stsnet.py                # BiLSTMNet + FusionHead + full STSNet model + joint training
├── train_eval.py            # LOSOCV experiment runner (DEAP / DREAMER)
├── tests.py                 # Unit & smoke tests
└── README.md
```

---

## Architecture overview

```
EEG signal
   │
   ├──► 4-D SPD representation ──► ManifoldNet ──► MO (spatio-spectral features)
   │     (n_windows, n_bands, C, C)   wFM × 2
   │                                 G-transport
   │                                 Invariant layer
   │
   └──► Flattened covariance seq ──► BiLSTM ──────► HO (spatio-temporal features)
         (n_windows, C*(C+1)/2)       256 units
                                      bidirectional
                                      Eq. (10)
                                      
MH = concat(MO, HO)  ──►  FC  ──►  Softmax  ──►  class label
```

Training follows **Algorithm 1** (joint alternating optimisation):
even batches update ManifoldNet + FC; odd batches update BiLSTM + FC.

---

## Installation

The project uses a **Dev Container** (Docker-based) for a consistent, ready-to-run environment. No manual `pip install` is required.

GPU support is automatic when a CUDA-capable GPU is available in the container; multi-GPU uses `tf.distribute.MirroredStrategy`.

---

## Full setup walkthrough

### Step 1 — Open in Dev Container

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code.

1. Clone the repository and open the folder in VS Code
2. When prompted **"Reopen in Container"**, click it — or open the Command Palette (`Ctrl+Shift+P`) and run **Dev Containers: Reopen in Container**
3. VS Code will build the Docker image (first run only) and drop you into a shell inside `/app`

The container is built from `tensorflow/tensorflow:latest` with `numpy`, `scipy`, and `scikit-learn` pre-installed.

### Step 2 — Download the datasets

**DEAP**
1. Go to `https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html`
2. Register and request access (usually approved within a day)
3. Download the **"Preprocessed data in Python format"** zip
4. Unzip — you should have a folder containing `s01.dat … s32.dat`

**DREAMER**
1. File already located at `data/dreamer/DREAMER_FULL.csv`

### Step 3 — Convert datasets to NumPy

```bash
# DEAP only
python prepare_datasets.py \
    --dataset deap \
    --deap_dir /path/to/data_preprocessed_python \
    --output_dir ./data \
    --verify

# DREAMER only
python prepare_datasets.py \
    --dataset dreamer \
    --dreamer_dir /path/to/dreamer \
    --output_dir ./data \
    --verify

# Both at once
python prepare_datasets.py \
    --dataset both \
    --deap_dir /path/to/data_preprocessed_python \
    --dreamer_dir /path/to/dreamer \
    --output_dir ./data \
    --verify
```

The `--verify` flag reloads the `.npy` files and checks shapes and NaN/Inf.
Expected output shapes:

| File | Shape |
|------|-------|
| `data/deap_eeg.npy` | `(32, 40, 32, 7680)` |
| `data/deap_labels.npy` | `(32, 40, 4)` — valence, arousal, dominance, liking |
| `data/dreamer_eeg.npy` | `(23, 18, 14, 7680)` |
| `data/dreamer_labels.npy` | `(23, 18, 2)` — valence, arousal |

### Step 4 — Run tests

```bash
python tests.py
```

All 9 tests should pass. If ManifoldNet tests are slow, that's normal on CPU — the Fréchet mean iteration is the bottleneck.

### Step 5 — Run the LOSOCV experiment

```bash
# DEAP — valence (variable-length windows, matching the paper)
python train_eval.py \
    --dataset deap \
    --dimension valence \
    --eeg_path data/deap_eeg.npy \
    --label_path data/deap_labels.npy

# DEAP — arousal
python train_eval.py --dataset deap --dimension arousal \
    --eeg_path data/deap_eeg.npy --label_path data/deap_labels.npy

# DREAMER — valence
python train_eval.py --dataset dreamer --dimension valence \
    --eeg_path data/dreamer_eeg.npy --label_path data/dreamer_labels.npy

# Use fixed-length windows (FW ablation baseline)
python train_eval.py --dataset deap --dimension valence \
    --eeg_path data/deap_eeg.npy --label_path data/deap_labels.npy \
    --fixed_windows
```

Results are printed per-subject and saved as CSVs to `results/`.

---

## Quick start (programmatic)

Prepare two NumPy files:

| File | Shape | Description |
|------|-------|-------------|
| `deap_eeg.npy` | `(32, 40, 32, 7680)` | subjects × trials × channels × samples |
| `deap_labels.npy` | `(32, 40, 5)` | subjects × trials × label dimensions (valence, arousal, …) |

### 2. Run LOSOCV

```bash
# DEAP — valence dimension, variable-length windows (default)
python train_eval.py --dataset deap --dimension valence

# DEAP — arousal dimension, fixed-length windows
python train_eval.py --dataset deap --dimension arousal --fixed_windows

# DREAMER — valence
python train_eval.py --dataset dreamer --dimension valence \
    --eeg_path dreamer_eeg.npy --label_path dreamer_labels.npy
```

Results are printed per-subject and saved to `results/`.

### 3. Programmatic use

```python
import numpy as np
import tensorflow as tf
from data_representation import preprocess_dataset
from stsnet import STSNet

# --- Preprocess one dataset split ---
xd_train, bi_train, y_train = preprocess_dataset(
    eeg_data        = train_eeg,   # (n_trials, 32, 7680)
    labels          = train_labels,
    fs              = 128,
    bands           = ["theta", "alpha", "beta", "gamma"],
    n_windows       = 15,
    window_size_sec = 4.0,
    use_variable_windows = True,
)

# --- Build and train ---
model = STSNet(n_channels=32, n_classes=2)
model.fit_joint(
    tf.constant(xd_train),
    tf.constant(bi_train),
    tf.constant(y_train),
    epochs=50, lr=1e-4,
)

# --- Inference ---
logits = model((tf.constant(xd_test), tf.constant(bi_test)), training=False)
preds  = tf.argmax(logits, axis=-1).numpy()
```

### 4. Run tests

```bash
python tests.py
```

---

## Key design decisions

### SPD manifold geometry (`manifold_net.py`)

| Operation | Implementation |
|-----------|---------------|
| `covariance_to_spd` | Eigendecomposition + eigenvalue clamping at ε=1e-6 |
| `matrix_sqrt / sqrt_inv` | Via `tf.linalg.eigh`; differentiable |
| `matrix_log / exp` | Spectral decomposition; used inside wFM iterations |
| Riemannian distance² | Affine-invariant: `‖log(A^{-½} B A^{-½})‖²_F` |

### Weighted Fréchet Mean (`weighted_frechet_mean`)

Fixed-point gradient-descent iteration on the SPD manifold (10 iterations
by default). The number of iterations can be reduced (e.g. to 3–5) to
speed up training with only marginal accuracy loss.

### Variable-length windows (`data_representation.py`)

Single-link hierarchical clustering using the log-Euclidean Riemannian
distance between adjacent SPD matrices, following reference [41].
Set `use_variable_windows=False` for the FW ablation baseline.

### Multi-GPU

Pass a `tf.distribute.MirroredStrategy` to `run_fold()`, or wrap model
construction and `fit_joint()` inside `strategy.scope()`.

---

## Hyperparameters (Tables 1 & 3)

| Parameter | DEAP | DREAMER |
|-----------|------|---------|
| `n_windows` (nc) | 15 | 15 |
| BiLSTM hidden units | 256 | 256 |
| Learning rate η | 1e-4 | 1e-4 |
| Weight decay λ | 5e-4 | 5e-4 |
| Frequency bands | θ, α, β, γ | θ, α, β |
| Window size (ablation best) | 4 s | 4 s |

---

## Expected results (subject-independent LOSOCV)

| Dataset | Dimension | Accuracy |
|---------|-----------|----------|
| DEAP    | Valence   | 69.38 %  |
| DEAP    | Arousal   | 71.88 %  |
| DREAMER | Valence   | 78.26 %  |
| DREAMER | Arousal   | 82.37 %  |
