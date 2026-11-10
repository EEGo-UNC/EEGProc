# Decoder recovery experiments

Opt-in experiments branched from `CFX-update` at `b2a0b60e`. Existing SIC/CFO
training entry points, defaults, and checkpoints are unchanged. Run commands
from this repository root with its `deep-learning` dependencies installed:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m experiments.decoder_recovery --help
```

## The two methods

**Idea 1: decoded subject identification.** Pretrain a subject identifier `Q`
on real source EEG, then freeze it. Update only the existing branch decoders
and their learned fusion scalar:

```text
L_decoder = reconstruction_weight * L_rec
          + identity_weight * CE(subject, Q(R(stop_gradient(E(X)))))
```

The encoder, recurrent emotion classifier, VC head, ISA head, and identifier
remain fixed, including their nontrainable state. Decoder dropout is enabled
only during decoder training. `--identity-loss kl` instead preserves
`Q(real EEG)`'s probability distribution using `KL(Q(X) || Q(R(E(X))))`.
Both objectives retain a reconstruction term. The default is cross-entropy,
the original Idea 1. A frozen identifier remains differentiable with respect
to the decoded EEG. No reversed gradient is applied to the decoder.

**The latent coords idea.** Train a pre-ISA model with task and reconstruction
losses. Save it as `E0/R0`. Initialize a second model from exactly the same
weights and train with ISA enabled, producing `E1/R1`. For each ORIGINAL input:

```text
delta(X) = stop_gradient(E0(X) - E1(X))
decoded candidate = R0(Z1_candidate + delta(X))
emotion prediction = C1(T1(Z1_candidate))
```

Residuals are computed separately for each active branch, once per trial,
and held fixed throughout counterfactual optimization. They never enter the
emotion classifier. Both the pre-ISA branch decoders and their pre-ISA fusion
weight are used. With no latent edit, this returns `R0(E0(X))` up to roundoff.

This is the raw paired-coordinate experiment: there is no automatic alignment
or assumption that the difference contains only subject identity. Fine-tuning
can change scale, basis, emotion features, and nuisance features. It restores
information through the retained pre-ISA checkpoint and original EEG; it does
not recover missing information from `Z1` alone. If even `E0` discarded the
information, the residual cannot restore it. A separate style-code architecture
is not implemented under this experiment's name.

## Temporal reconstruction losses

Use `--reconstruction-seam-weight` for boundary-step matching and
`--reconstruction-derivative-weight` for within-window derivative matching.
`latent-coords` applies the weights to both source-training phases. Its flags
override the corresponding model config values, which default to zero. Keep
`reconstruction_loss_weight` positive for source training.

`idea1` uses the source checkpoint's weights for any omitted flags. An
explicit `0` disables the corresponding term during decoder adaptation,
so several ablations can start from the same checkpoint. Use
`--identity-weight 0` for reconstruction-only adaptation. The report and
adapted checkpoint save the effective temporal weights.

| Reconstruction loss | Seam weight | Derivative weight |
| --- | ---: | ---: |
| MSE only | 0 | 0 |
| MSE + seam | positive | 0 |
| MSE + derivative | 0 | positive |
| MSE + both | positive | positive |

Use chronological, contiguous, non-overlapping trial windows for seam
matching. See `docs/RECONSTRUCTION_TEMPORAL_INTEGRATION.md` for the objective.

## SLURM: run both ideas and matched controls

From this branch on Longleaf, using the existing `venv312` environment and
`datasets/dreamer_{eeg,labels}.npy`:

```bash
# Preview commands without loading modules, reading EEG, or writing results.
DRY_RUN=1 bash experiments/decoder_recovery/run_decoder_recovery.slurm

# Initial valence fold, held-out subject 0. This command submits the job.
sbatch --array=0 experiments/decoder_recovery/run_decoder_recovery.slurm

# After inspecting the first fold, all subjects with two concurrent GPUs.
sbatch --array=0-22%2 experiments/decoder_recovery/run_decoder_recovery.slurm

# A separate arousal suite.
TASK=arousal sbatch --array=0 experiments/decoder_recovery/run_decoder_recovery.slurm
```

The script requests one L40 GPU, 6 CPUs, 64 GB RAM, and 24 hours per subject.
It uses the project's Python 3.12.4, CUDA 12.9, and cuDNN 9.11.0 modules. It
prepares commands only until submitted; no cluster job is launched by the
implementation or local validation.

Each job runs:

1. A subject-held-out split: one test subject, four seeded validation subjects,
   and the remaining source subjects. One complete trial per source subject
   is additionally reserved for identifier/decoder validation.
2. Phase 1 without ISA contribution, then phase 2 with ISA, saving both models.
3. Real-EEG identifier training and a second independent real-EEG probe with
   a different seed. The probe never supplies a decoder gradient.
4. Equal-epoch reconstruction-only and Idea 1 decoder adaptation from the
   SAME post-ISA checkpoint.
5. Evaluation of baseline `R1`, reconstruction-only, Idea 1, and latent coords
   decoding on both withheld source trials and untouched test-subject trials.
6. Counterfactual generation for all four arms, with separate latent target
   success and decoded/re-encoded target success.

Useful overrides:

```bash
PHASE1_EPOCHS=2 PHASE2_EPOCHS=2 IDENTIFIER_EPOCHS=2 DECODER_EPOCHS=2 \
CF_MAX_STEPS=5 CF_LIMIT=2 \
sbatch --array=0 experiments/decoder_recovery/run_decoder_recovery.slurm

IDENTITY_LOSS=kl IDENTITY_WEIGHT=0.1 \
sbatch --array=0 experiments/decoder_recovery/run_decoder_recovery.slurm
```

`RECONSTRUCTION_SEAM_WEIGHT` and `RECONSTRUCTION_DERIVATIVE_WEIGHT` set
the source-training weights. `DECODER_RECONSTRUCTION_SEAM_WEIGHT` and
`DECODER_RECONSTRUCTION_DERIVATIVE_WEIGHT` override the checkpoint's
weights for both matched decoder adaptation arms. Set either variable
explicitly to `0` to ablate that term at its stage.

Other overrides: `PROJECT_DIR`, `VENV_DIR`, `EEG_PATH`, `LABELS_PATH`,
`MODEL_CONFIG`, `OUT_ROOT`, `RUN_ID`, `SEED`, `BATCH_SIZE`, `ISA_WEIGHT`,
`VALIDATION_SUBJECTS="1 2 3 4"`, `RUN_COUNTERFACTUALS=0`,
`PHYSIOLOGICAL_WEIGHT`, and `TARGET_PROBABILITY`. New runs require a new output
directory; partial outputs are never silently reused. No resume is implemented.

The included model config retains the v15 architecture and selected run's
focal/VC settings, but deliberately uses single-GPU ERM and positive
reconstruction weight `0.1`. It does not reproduce the historical MLDG run
whose reconstruction weight was zero. Supply resolved builder kwargs in
`MODEL_CONFIG` to change this. Source checkpoints are selected by validation
balanced accuracy; test-subject results never select weights. The default
counterfactual physiological weight is zero to isolate decoder recovery;
physiology diagnostics are still reported by the existing CFO implementation.

Results are under `runs/decoder_recovery/<task>/suite_<job>/fold_<subject>/`.
`comparison.json` gathers reconstruction MSE, round-trip emotion agreement,
flip rate, original/re-encoded balanced accuracy, and independent probe scores.
Each counterfactual saves its arrays, optimization history, latent success,
and actual decoded/re-encoded prediction. Existing CFO candidate selection
remains latent-based: decoded validity is measured after selection, not
guaranteed by optimization.

## Individual commands

Prepared archives contain `eeg`, `labels`, `subject_ids`, `trial_ids`, with
`eeg` shaped `(N,W,T,F)` for full trials or `(N,T,F)` for window models, and
integer task labels. Windows belonging to a trial must never cross splits.
The CLI does not silently normalize prepared data. Its preprocessing must
match the checkpoint. `prepare-fold --loader-config` reuses EEGProc's existing
loader and chronological trial grouping:

```json
{
  "eeg_path": "datasets/dreamer_eeg.npy",
  "labels_path": "datasets/dreamer_labels.npy",
  "label_dimension": "valence",
  "dataset": "dreamer",
  "window_size_sec": 1.0,
  "fs": 128.0,
  "overlap": 0.0,
  "median_label": 3.0,
  "window_normalization": "global_rms",
  "label_threshold_mode": "global"
}
```

Save that as `loader.json`, then:

```bash
python -m experiments.decoder_recovery prepare-fold \
  --loader-config loader.json --test-subjects 0 --validation-subjects 1 2 3 4 \
  --output runs/recovery_data

python -m experiments.decoder_recovery latent-coords \
  --train runs/recovery_data/train.npz --validation runs/recovery_data/validation.npz \
  --model-config experiments/decoder_recovery/model_config.json --output runs/coords

python -m experiments.decoder_recovery pretrain-identifier \
  --train runs/recovery_data/train.npz \
  --validation runs/recovery_data/identity_validation.npz --output runs/identifier

python -m experiments.decoder_recovery idea1 \
  --model runs/coords/post_isa.keras \
  --identifier runs/identifier/identifier.keras \
  --identifier-metadata runs/identifier/identifier.json \
  --train runs/recovery_data/train.npz \
  --validation runs/recovery_data/identity_validation.npz --output runs/idea1

python -m experiments.decoder_recovery evaluate \
  --model runs/coords/post_isa.keras --reference-model runs/coords/pre_isa.keras \
  --data runs/recovery_data/test.npz --output runs/coords_test

python -m experiments.decoder_recovery counterfactual \
  --model runs/coords/post_isa.keras --reference-model runs/coords/pre_isa.keras \
  --data runs/recovery_data/test.npz --output runs/coords_counterfactuals
```

Omit `--reference-model` for ordinary decoding, including an Idea 1 checkpoint.
`counterfactual` requires full trials and supports the existing DREAMER
physiology diagnostic's 14-channel, 3-band layout. `--decoder-mode branches`
supports a single-branch model. Raw phase-1/phase-2 models must have identical
branch configurations; do not pair unrelated checkpoints.

Closed-set subject accuracy is reported only for known training subjects.
Unseen test subjects are counted separately and receive no invented identity
score. A separate real-EEG style/reference path would be needed to explicitly
condition on a previously unseen subject. Higher identifier confidence alone
does not demonstrate faithful EEG or explain prediction flips. In particular,
the pretrained guide's scores should be checked against the independent probe,
task fidelity, and reconstruction/physiology diagnostics. Experiments establish
neither complete subject invariance nor an isolated causal subject component.

## Local validation

```bash
PYTHONPATH=src python -m pytest -q src/tests/test_decoder_recovery.py
bash -n experiments/decoder_recovery/run_decoder_recovery.slurm
```

Tests use tiny synthetic EEG to check gradient isolation, actual decoder
updates, checkpoint reloads, two-phase training, per-trial coordinate recovery,
counterfactual classification, split isolation, CLI outputs, and SLURM command
construction. They do not establish effectiveness on real EEG or GPU capacity
for the full dataset.
