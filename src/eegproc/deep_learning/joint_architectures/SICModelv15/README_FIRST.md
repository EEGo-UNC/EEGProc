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

Each task reserves **four GPUs**, runs **two folds concurrently with two GPUs
per fold**, and covers all 23 LOSO targets using all other 22 subjects as the
source pool. Valence submits a **36-task Slurm array**, with at most two tasks
running concurrently (`--array=0-35%2`), using up to eight GPUs. Running arousal
alongside both active valence tasks can use twelve GPUs in total.
Both launchers request 8 CPUs and 128 GB RAM per task. The valence launcher
uses 12 source epochs with 10 MLDG steps per epoch, then one calibration epoch
at 3 shots (6 folds). Its 36 configurations cross focal gamma 0.0/0.5/1.0/2.0,
subject-loss weights 0.0/0.3/0.6, and reconstruction weights 0.0/0.4/0.8.
Each array task receives one fixed configuration and its own 24-hour limit.
Reconstruction weight varies fastest, then subject loss, then focal gamma;
task 0 is (0.0, 0.0, 0.0), and task 35 is (2.0, 0.6, 0.8).
Use `sbatch --array=0-35%1 .../full_run_v15_valence.sh` to cap valence at four
GPUs, or `sbatch --array=7 .../full_run_v15_valence.sh` to rerun only task 7.
Use `squeue -r -u "$USER"` to display each array task on a separate line.
The arousal launcher uses 10 source epochs with 12 MLDG steps per epoch,
one calibration epoch at 3/6/9/12 shots, fixed subject-loss weight 0.2 and
reconstruction weight 0.6, within a 12-hour limit. Both enable deterministic
training and report zero-shot LOSO balanced accuracy. Valence's per-task
`best_hyperparameters.json` describes that task's single configuration; compare
`selection_score` across all completed tasks to select the suite's best
configuration. Each training invocation no longer sees the entire suite.

For the separate valence oracle sweep (one fixed configuration per six-hour
array task), use `SLURM_scripts/full_run_v15_valence_sweep.sh`; see
`VALENCE_SWEEP.md`. The full-run launchers above keep their own budgets
and reporting protocol.

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
`runs/full/sic_v15_valence_best_20260918/DREAMER/valence/suite_ARRAY_ID/task_TASK_ID/full`.
The full datasets and CUDA memory use must still be verified on Longleaf.
