# SICModelv15: joint decoder reconstruction

SICModelv15 carries forward the current SICModelv11 architecture and training
pipeline. Its only architectural addition is a learned convex fusion of the two
same-shaped decoder outputs in the original EEG feature space:

```text
GCN-GRU features -> GCNMTLDecoder -> x_hat_g --\
                                                convex scalar -> x_hat_joint
BiLSTM features  -> GCNMTLDecoder -> x_hat_b --/

alpha = sigmoid(mix_logit)
x_hat_joint = alpha * x_hat_g + (1 - alpha) * x_hat_b
```

`mix_logit` is a scalar Keras weight initialized so `alpha=0.5`. It is learned
end to end through the joint reconstruction MSE. Both independent branch MSEs
remain in the objective as an auxiliary safeguard:

```text
branch_mse = mean(gcn_gru_mse, bilstm_mse)
reconstruction_loss =
    (joint_mse + auxiliary_weight * branch_mse) / (1 + auxiliary_weight)
```

The default `joint_reconstruction_auxiliary_weight` is `0.25`. The normalization
keeps the reconstruction term on approximately the same scale as SICModelv11.
Single-branch ablations use their original branch MSE unchanged.

New reconstruction metrics are emitted whenever both branches and the decoder
are enabled:

- `joint_reconstruction_loss`
- `joint_decoder_r2`
- `joint_reconstruction_alpha`
- `joint_reconstruction_gain_vs_best_branch`

The gain is `min(gcn_gru_mse, bilstm_mse) - joint_mse`; positive values mean the
joint output is better than the better individual branch for that batch.

Use `model.reconstruct_joint(inputs)` or
`model.reconstruct(inputs, branch="joint")` to return the single reconstruction.
`model.reconstruct(inputs)` returns all available reconstructions under
`gcn_gru`, `bilstm`, and `joint` keys while preserving the input rank.

Relevant optional JSON settings are:

```json
{
  "use_decoder": true,
  "reconstruction_loss_weight": 0.10,
  "joint_reconstruction_auxiliary_weight": 0.25,
  "joint_reconstruction_initial_alpha": 0.5,
  "decoder_dropout": 0.10
}
```

Retrain SICModelv15 rather than loading SICModelv11 weights directly: v15 adds
the trainable joint-fusion scalar and new serialized configuration fields.

## Two GPUs per MLDG source fold

Submit the four-target DREAMER valence smoke from the cluster checkout:

```bash
sbatch src/eegproc/deep_learning/joint_architectures/SICModelv15/SLURM_scripts/smoke_val_0_3.sh
```

The job reserves four L40S GPUs and runs two independent LOSO folds at a time,
with disjoint GPU pairs `(0, 1)` and `(2, 3)`. Targets are users 0, 1, 2, and 3;
each source pool still contains all 22 other users. Its source episodes use
8 meta-train subjects and 4 meta-test subjects, with **4 trials per subject**:
32 meta-train and 16 meta-test trials per episode, split as 16/8 per GPU.
This is 33% more trials than the previous 3-trial configuration.

The smoke uses 6 source epochs, 20 MLDG steps
per epoch, 10 calibration epochs, and reconstruction weights 0.1 and 0.4.
`SOURCE_EPOCHS` and `CALIBRATION_EPOCHS` can override the epoch budgets. The
launcher uses the existing `venv312` environment and has a four-hour limit.
Logs are `smoke_val_0_3_JOBID.out/.err`; results go under
`runs/smoke/sic_v15_two_gpu/DREAMER/valence/suite_JOBID/smoke_val_0_3`.

The runtime option is `--gpus-per-fold 2`, together with `--n-jobs 2` and
`--gpu-ids 0 1 2 3`. Launchers that omit `--gpus-per-fold` still default to
one GPU per fold. Each worker's GPU mask is established before TensorFlow
is imported, including when Slurm identifies allocated GPUs by UUID.
Each target fold runs in a fresh worker process so CUDA resources from a
completed fold are released before the next fold uses that GPU pair.

Source training uses explicit device placement with shared model variables.
Complete trials are split along the batch axis, without shortening sequences.
Large encoder and decoder activations remain on their assigned device; trial
embeddings are gathered to compute the VC class means/variances over the whole
meta-train or meta-test group. Reconstruction MSE is reduced using element
counts. TensorFlow differentiates this single global objective through both
device branches into the shared variables. The original first-order MLDG
inner assignment and single outer optimizer update are unchanged.
Both forward and backward operations have explicit device placement. A local
tape retains the ordinary activations for each device's backward pass; there
is no recomputation or approximate gradient. Source epoch logs report current
and peak allocated memory separately for each GPU.

This path requires trial-level MLDG and `gcn_use_batch_norm=false`. Calibration
and prediction retain their existing single-device execution on the first GPU
of the pair. Checkpoints contain one ordinary v15 model and do not serialize
hardware assignments. Device reduction order and dropout draws may differ;
bitwise-identical training is not promised.

The smoke first runs a small **real two-GPU preflight** comparing full losses,
gradients, inner/outer updates, metrics, device placement, and checkpoint
portability against the original execution. It stops if those checks fail.
For a local check using two logical CPUs:

```bash
python -m src.tests.test_sic_v15_multi_gpu
```

The logical-CPU check cannot establish CUDA memory use or cuDNN behavior.
The actual L40S smoke remains the hardware and peak-memory validation.

## Full DREAMER runs

The full launchers are named `full_run_v15_arousal.sh` and
`full_run_v15_valence.sh` (replacing `run_sic_v15_best_v11_arousal_full.sh`
and `run_sic_v15_best_v11_valence_full.sh`). Submit from the cluster checkout:

```bash
sbatch src/eegproc/deep_learning/joint_architectures/SICModelv15/SLURM_scripts/full_run_v15_arousal.sh
sbatch src/eegproc/deep_learning/joint_architectures/SICModelv15/SLURM_scripts/full_run_v15_valence.sh
```

Each task reserves **four GPUs**, eight CPUs and 128 GB RAM, and runs two
source folds concurrently with two GPUs per fold. Every target source model
uses the other 22 users; there is no validation holdout.

The valence launcher now has **one fixed configuration**, matching
`full_run_v15_valence_5168991_12.out`: focal gamma **0.5**, subject-loss weight
**0.3**, reconstruction weight **0.0**, and all other model settings recorded
in `sic_v15_valence_run_5168991_12.json`. The single array task (`--array=0`)
uses 12 source epochs, ten MLDG steps per epoch, and a 24-hour allocation.

After every source epoch, evaluate all 18 trials of its held-out user at
threshold 0.5 and save that epoch's model. Once all 23 source fits finish,
select **one common epoch** maximizing the unweighted mean of the 23 users'
balanced accuracies; exact ties choose the earliest epoch. Reload only those
selected models for final zero-shot evaluation and one calibration epoch at
3 shots / 6 folds. The reloaded mean zero-shot balanced accuracy must match
the selected oracle score before `final_metrics.json` is written. The selected
epoch is printed beside the final metrics and saved in that file.

This is explicitly an oracle experiment: held-out labels select the common
epoch, while source gradients and MI-adjacency fitting still use source users
only. All epochs' checkpoints and predictions remain available for auditing.
`SOURCE_EPOCHS`, `CALIBRATION_EPOCHS`, `TRAINING_SEED`, `TIME_BUDGET_SECONDS`
(default 82,800 source-training seconds), and `OUTPUT_DIR` can be overridden.
Use a new output directory for each submission; the runner does not resume.

The arousal launcher retains its existing settings: ten source epochs,
12 MLDG steps per epoch, one calibration epoch at 3/6/9/12 shots, subject-loss
weight 0.2, reconstruction weight 0.6, and a 12-hour allocation.
The separate 16-configuration, six-hour oracle screening sweep remains
available as `SLURM_scripts/full_run_v15_valence_sweep.sh`; see
`VALENCE_SWEEP.md`. Its settings are independent of the fixed full run.

| Target | Meta-train / meta-test subjects | Trials per subject | Global trials, train / test | Trials per GPU, train / test |
| --- | --- | --- | --- | --- |
| Valence | 12 / 6 | 2 | 24 / 12 | 12 / 6 |
| Arousal | 12 / 6 | 2 | 24 / 12 | 12 / 6 |

Arousal retains its existing episode configuration because some subjects have
only one low-arousal trial. Four trials per subject would require two distinct
trials from each class, which the current sampler correctly rejects. More GPU
memory cannot resolve that data constraint. No repeated trials or subject
exclusions are introduced.

Both launchers run the same two-GPU correctness preflight as the smoke job and
stop if it fails. Valence logs are `full_run_v15_valence_ARRAY_ID_TASK_ID.out/.err`;
its results are under
`runs/full/sic_v15_valence_oracle_5168991_12/suite_ARRAY_ID/task_0/`.
`oracle_epoch_means.csv` contains every epoch's average; `oracle_models.json`
identifies the selected checkpoints. `final_metrics.json` records the selected
epoch and final zero-shot/3-shot means; detailed calibration results are in
its `calibration/` subdirectory.
The full datasets and CUDA memory use must still be verified on Longleaf.


## Extended valence hyperparameter sweep

`SLURM_scripts/full_run_v15_valence_focal_scale_recon_sweep.sh` adds 36
configurations crossing focal gamma, logit scale, reconstruction weights 0.0/0.4,
and 5/10/20 MLDG episodes per source epoch. It uses 24 source epochs and the
run-12 baseline for all other settings. Source jobs shard the 23 held-out users;
dependent collectors select one common oracle epoch over all users and then
calibrate the selected models. Use the bash wrapper to submit both arrays.
See [the extended sweep instructions](VALENCE_FOCAL_SCALE_RECON_SWEEP.md).
