# Validation

## Temporal reconstruction integration, November 10, 2026

Amit Chalmeti's `recon-temporal-loss` branch at `2fc2395c` is merged into the
local `CFX-update` and `codex/decoder-identity-latent-coords` branches. This
includes the training implementation at `bb153d59`, the arousal launcher
updates, and the reconstruction-comparison tool. Both temporal loss weights
retain their zero defaults. No remote branch was pushed and no cluster job
was submitted.

Targeted validation: **40 unique tests passed** with Python 3.12 / TensorFlow
on CPU, covering reconstruction temporal losses, the comparison tool,
decoder recovery, existing counterfactual temporal/envelope controls, v15
imports/calibration, and two-device objective/gradient parity. The two-device
check uses logical CPUs. The obsolete shared-runtime `SICModelv11` calibration
case was deselected because that model version is absent from this branch.

The new recovery integration test enables both seam and derivative terms on
two-window trials with both encoder branches. It runs both pre/post-ISA source
training phases, identity-guided decoder-only adaptation, and checkpoint
reload. It verifies that the temporal weights and losses survive reload, that
the emotion predictions remain unchanged during decoder adaptation, that the
encoder/classifier/identifier remain fixed, and that zero-edit paired-coordinate
decoding still reproduces the pre-ISA reconstruction.

The first test attempt encountered filesystem `ENOSPC`; after removing the
incomplete temporary Git packs from the update, all interrupted relevant
checks passed on retry. The 40-test count combines 33 initial passes, six
successful retries, and the added recovery integration test. Both launchers
pass Bash syntax validation and both branch diffs pass whitespace checks.
Scientific effectiveness and actual GPU execution remain unmeasured.

## Original decoder recovery validation

Validated locally with Python 3.12 / TensorFlow on CPU. No Longleaf job was
submitted and no real EEG effectiveness result is claimed.

| Check | Result |
|---|---|
| Decoder recovery tests | 11 passed |
| Complete launcher on synthetic raw EEG | Passed |
| SLURM syntax and command-only preview | Passed |
| Wheel build, isolated installation, SIC import | Passed |
| Existing CPU sweep worker, isolated rerun | Passed on this branch and unchanged `CFX-update` |

The full launcher test used seven synthetic subjects, four trials per subject,
42 channel-band features and 128 samples per trial, with both encoder/decoder
branches active. Each training phase ran one epoch, and each of the four arms
generated one counterfactual with one optimization step. All eight evaluation
reports, four counterfactual archives, and the comparison report were written.
This checks orchestration and contracts, not convergence or scientific value.

The automated tests verify:

- Actual identification gradients reach decoders while encoder gradients are
  blocked, for both cross-entropy and probability-preservation KL.
- Encoder/classifier/identifier parameters and state remain unchanged during
  Idea 1, and adapted checkpoints reload correctly.
- Paired pre/post-ISA checkpoints reload together. The experiment loader uses
  Keras's public architecture/weight APIs to avoid intermittent construction
  failures in the archive loader's temporary sharing scope.
- Zero-edit coordinate restoration reproduces the pre-ISA reconstruction for
  single-branch and learned joint decoding; gradients still reach edited
  latents; model parameters remain fixed; residuals are per-input.
- Counterfactual classification uses the post-ISA classifier. Re-encoded
  predictions are computed from saved decoded EEG rather than inferred from
  latent success.
- Train/validation partitions cannot share subject/trial keys; held-out target
  subjects are excluded from training; unknown subject IDs are reported as
  unknown rather than assigned a closed-set accuracy.

The broad suite run reported **441 passed, 6 skipped, 7 subtests passed, and
6 failures**. Five failures are the inherited legacy SIC/import/launcher
checks already documented in `docs/CFX_REPAIR_MERGE.md`:

- `test_imported_launchers_parse_and_calibration_help_works`
- `test_valence_array_task_has_a_valid_calibration_selection`
- `test_source_training_calibration_and_checkpoint[SICModel]`
- `test_source_training_calibration_and_checkpoint[SICModelv11]`
- `test_calibration_only_loads_version_in_fresh_workers[SICModelv11]`

The sixth, `test_real_cpu_worker_saves_reloadable_oracle_model`, exceeded its
internal training deadline during the broad run. An isolated rerun passed on
both the untouched base and this branch. Existing library implementation and
training entry points are unchanged by these experiments.

Full-size EEG training, GPU memory use, Longleaf module availability, and the
effectiveness of either recovery method remain to be measured by the prepared
SLURM run. The supplied config is a controlled single-GPU ERM experiment with
positive reconstruction training, not an exact historical MLDG reproduction.
