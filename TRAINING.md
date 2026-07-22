# MTLFuseNet — Training & LOSO

This is the training half of MTLFuseNet on DREAMER: turning the built-and-verified
architecture into a trained model, evaluated with **Leave-One-Subject-Out (LOSO)**
cross-validation. The model pieces (VAE branch, GCN-GRU branch, fusion, the three
losses) already live in `models.py`, `losses.py`, and `preprocessing.py`.

## What was added

| File | Purpose |
|------|---------|
| `mtl_preprocess.py` | Turn each `(subject, trial)` into training-ready arrays and **cache** them. Filter → differential-entropy features → mutual-information adjacency → normalized 9×9 grid windows. |
| `mtl_model.py` | `MTLFuseNet(tf.keras.Model)` — the whole network assembled into one trainable model: batched GCN, **learnable** triplet-center centers, and a custom `train_step` that optimizes `focal + triplet-center + VAE` loss end to end. |
| `mtl_loso.py` | The LOSO loop: train a fresh model per held-out subject, average each test trial's window predictions into one trial-level prediction, and write metrics. |
| `MTLFuseNet_training.ipynb` | Colab driver notebook that calls the three modules (GPU-friendly). |

## Data flow (per trial)

```
raw EEG (14 ch)
  ├─ spatio-temporal:  9×9 grid per timestep → 1s windows (N,9,9,128) → min-max [0,1]  ─┐
  └─ spatio-spectral:  bandpass θ/α/β → DE feats (N,3,14)  +  MI adjacency (3,14,14)    ─┤
                                                                                         ↓
   VAE(z:128)  ⊕  GRU(GCN(DE,adj)) (Z_SS:384)  =  Z_SST:512  → softmax(2)  +  centers
                          loss = 0.7·focal + 0.2·triplet-center + 0.1·VAE
```

Each 1-second window is one training sample and inherits its trial's binary label
(`score >= 3 → 1`). At test time the windows of a trial are averaged back into a
single per-trial prediction (18 trials / held-out subject).

## Run it

### Google Colab (recommended — needs a GPU)
Open `MTLFuseNet_training.ipynb`, set the `CSV` / `PROCESSED` paths to your Drive,
and run top to bottom. Preprocessing writes a cache to Drive so it's only done once.

### Locally
```bash
source .venv/bin/activate
# 1) cache features (once) — all subjects
python mtl_preprocess.py --csv datasets/dreamer_joined.csv --out processed_trials
# 2) LOSO
python mtl_loso.py --task valence --processed processed_trials --epochs 50
python mtl_loso.py --task arousal --processed processed_trials --epochs 50
```
Results land in `experiment_outputs/dreamer_{task}_results.json` (per-fold and
per-subject accuracy / F1 / precision / recall, plus mean ± std).

## Environment note
The venv shipped a **broken scipy binary** (failed to load on this macOS build),
which takes down `eegproc` and scikit-learn. Fixed by:
```bash
pip install --force-reinstall --no-cache-dir "scipy>=1.16"   # also bumps numpy
```

## Reconciliation with the paper (Li et al., KBS 2023)
Paper's DREAMER targets (subject-independent LOSO): **valence 80.43%, arousal 83.33%**.

**Confirmed matching Table 2 (DREAMER):** 3 bands θ/α/β (gamma is unavailable — DREAMER
data is bandpassed 4–30 Hz; the 4-band {θ,α,β,γ} case is DEAP), VAE 4 conv layers
128/256/256/512, spatio-temporal latent 128, spatio-spectral (GRU) 384, fusion → 512,
14×14 MI adjacency + symmetric norm, loss weights 0.7/0.2/0.1, label median 3, LOSO
22-train/1-test over all subjects. **Applied from Table 2:** learning rate `1e-4`,
`dropout 0.2` (on the fused vector before the classifier).

## Decisions to confirm with your mentor
- **Segments:** each trial currently uses *both* baseline + stimulus EEG (~260 windows).
  The paper describes stimulus-video watching (65–393 s) — consider stimulus-only in
  `mtl_preprocess.build_trial_sample`.
- **Not specified in the paper (using sensible defaults):** epochs/MaxIter, batch size,
  optimizer (Adam), focal α & γ, triplet margin (0.7 / 2.0 / 1.0 in `mtl_model.py`).
- **Focal loss form:** paper's Eq. 20 weights the negative class by `(1−α)`; the
  partner's `losses.py` applies `α` to both classes — a minor alignment to consider.
- **MI subsampling:** `mi_max_samples=5000` subsamples timesteps before the slow
  mutual-information estimate; set `None` for every sample (much slower).
