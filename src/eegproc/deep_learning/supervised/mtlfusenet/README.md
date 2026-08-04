# MTLFuseNet

> **Reference:** R. Li, C. Ren, Y. Ge, Q. Zhao, Y. Yang, Y. Shi, X. Zhang, and B. Hu, “MTLFuseNet: A novel emotion recognition model based on deep latent feature fusion of EEG signals and multi-task learning,” *Knowledge-Based Systems*, vol. 276, Art. no. 110756, 2023, doi: 10.1016/j.knosys.2023.110756.

EEGProc implementation of MTLFuseNet for subject-independent DREAMER valence and arousal classification.

## Directory layout

```text
EEGProc/
├── src/eegproc/deep_learning/supervised/mtlfusenet/
│   ├── __init__.py
│   ├── losses.py
│   ├── models.py
│   ├── mtl_model.py
│   ├── mtl_preprocess.py
│   ├── mtl_model_train.py
│   ├── preprocessing.py
│   ├── mtl_loso.py
│   └── README.md
├── scripts/
│   └── smoke_mtlfusenet_dreamer_arousal.sh
├── datasets/
│   └── dreamer_joined.csv
└── processed_trials/
    └── subj*_trial*.pkl
```

## Files

| File                                                             | Purpose                                                                                                              |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `losses.py`                                                      | Focal loss, triplet-center loss, and the weighted MTLFuseNet objective.                                              |
| `models.py`                                                      | VAE encoder/decoder, sampling layer, and graph helper components.                                                    |
| `mtl_model.py`                                                   | End-to-end VAE + GCN–GRU fusion model with custom training and evaluation steps.                                     |
| `mtl_preprocess.py`                                              | Converts DREAMER subject/trial EEG into cached `X_ST`, DE, and adjacency inputs.                                     |
| `mtl_model_train.py`                                             | EEGProc-compatible training entry point using shared LOSO, validation, trial aggregation, metrics, and output files. |
| `smoke_mtlfusenet_dreamer_arousal.sh`                            | Small Longleaf SLURM test using two LOSO folds and five epochs.                                                      |
| `preprocessing.py`                                               | Original preprocessing helpers and constants used during the replication.                                            |
| `mtl_loso.py`                                                    | Earlier standalone LOSO implementation; `mtl_model_train.py` is the repository-standard training path.               |
| `check_pairs.py`, `preprocess_dataset.py`, `test_*.py`, notebook | Development and prototype utilities; they are not required for normal training.                                      |

## Prepare DREAMER

Run from the EEGProc repository root:

```bash
cd "$HOME/EEGProc"
source venv312/bin/activate

python -m eegproc.deep_learning.supervised.mtlfusenet.mtl_preprocess \
  --csv datasets/dreamer_joined.csv \
  --out processed_trials \
  --mi-max-samples 5000
```

A complete DREAMER cache should contain 414 trial files:

```bash
find processed_trials -maxdepth 1 -name 'subj*_trial*.pkl' | wc -l
```

## Run the SLURM smoke test

Submit the included script:

```bash
cd "$HOME/EEGProc"
sbatch scripts/smoke_mtlfusenet_dreamer_arousal.sh
```

The script requests two L40 GPUs and runs two LOSO folds concurrently. It first checks the Python/CUDA environment, TensorFlow GPU visibility, local EEGProc imports, cached inputs, packed-input adapter, and one model forward pass.

Monitor the job:

```bash
squeue -u "$USER"
tail -f smoke_mtlfusenet_arousal_<JOB_ID>.out
```

Check errors:

```bash
tail -f smoke_mtlfusenet_arousal_<JOB_ID>.err
```

A successful job should finish with SLURM state `COMPLETED` and exit code `0:0`:

```bash
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Smoke-test outputs are saved under:

```text
runs/smoke/mtlfusenet/DREAMER/arousal/
```

## Run full LOSO training

### Arousal

```bash
python -m eegproc.deep_learning.supervised.mtlfusenet.mtl_model_train \
  --processed-dir processed_trials \
  --task arousal \
  --out-dir runs/mtlfusenet/DREAMER/arousal \
  --run-name dreamer_arousal_mtlfusenet \
  --epochs 50 \
  --batch-size 64 \
  --n-jobs 2 \
  --cpus-per-worker 4 \
  --selection-level trial \
  --selection-metric accuracy
```

### Valence

Use the same command with:

```bash
--task valence \
--out-dir runs/mtlfusenet/DREAMER/valence \
--run-name dreamer_valence_mtlfusenet
```

Remove `--max-folds` or leave it unset for all 23 LOSO folds. Architecture and loss settings can be passed as ordinary flags or through `--hyperparameters-json`.

## Submit the smoke test to SLURM

The smoke script is stored in `SLURM_scripts/`. Submit it from the EEGProc repository root:

```bash
cd "$HOME/EEGProc"
sbatch SLURM_scripts/smoke_mtlfusenet_dreamer_arousal.sh
```

Check whether the job is queued or running:

```bash
squeue -u "$USER" -o "%i %P %j %u %t %M %D %R"
```

Follow its standard output and error logs:

```bash
tail -f "smoke_mtlfusenet_arousal_${JOB_ID}.out"
```


## Details

MTLFuseNet combines a spatio-temporal VAE representation with a spatio-spectral GCN–GRU representation. Their fused latent vector is trained with focal classification loss, triplet-center loss, and VAE reconstruction/KL loss.

Each cached one-second window is a training sample. EEGProc averages window probabilities within each trial for trial-level selection and reporting. Setting `--prediction-latent-samples 0` uses the posterior mean for deterministic predictions; a positive value enables Monte Carlo latent averaging.

The smoke job is only a pipeline test. Its two-fold, five-epoch metrics should not be treated as the final replication result.
j