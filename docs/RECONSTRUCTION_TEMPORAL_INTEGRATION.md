# Temporal reconstruction losses

Amit Chalmeti's `recon-temporal-loss` branch is incorporated into this branch.
The training implementation is commit `bb153d59`; the incoming branch tip is
`2fc2395c`, which also includes the reconstruction-comparison script and the
arousal launcher's checkout/dependency checks. Existing CFX temporal-edit,
envelope-distance, and physiology controls are preserved.

## Objective and configuration

`SICModelv15` supports two optional model-config values:

- `reconstruction_derivative_weight`: match the reconstruction's within-window
  first differences to the recorded signal's first differences.
- `reconstruction_seam_weight`: match the step between consecutive decoded
  windows to the corresponding recorded step, computed separately per trial.

Both penalties are mean squared errors. They extend the reconstruction loss:

```text
L_rec = L_MSE
      + reconstruction_seam_weight * L_seam
      + reconstruction_derivative_weight * L_derivative
```

Each term uses the existing joint-plus-auxiliary-branch weighting when both
branches are active. The full source objective multiplies `L_rec` by
`reconstruction_loss_weight`. Enable the decoder and use positive effective
weights to train with these terms. Both new weights default to zero; merging
the implementation does not enable them or change existing run settings.
Model configuration and checkpoints retain both weights.

The seam term needs complete trials with chronologically ordered,
non-overlapping, contiguous windows. The production CLI rejects window-level
classification and nonzero overlap when the seam weight is positive. Prepared
arrays passed to other entry points must satisfy the same ordering and
adjacency assumptions. Derivative matching can also run on individual windows.

These losses preserve waveform dynamics and boundary steps. They do not
explicitly match Hilbert phase angles or add an envelope loss to AE training.
Counterfactual latent-edit smoothing and phase-tolerant envelope distance are
separate controls and keep their existing behavior.

For the arousal launcher, set `RECONSTRUCTION_SEAM_WEIGHT` and
`RECONSTRUCTION_DERIVATIVE_WEIGHT`. The v15 training command accepts
`--reconstruction-seam-weight` and `--reconstruction-derivative-weight`.
Either flag overrides its value in `--hyperparameters-json`, including a
grid entry. Pass `0` to ablate the corresponding term. Omit the flags to use
the model configuration, which can specify a grid for either weight.

## Checks

`src/tests/test_reconstruction_temporal_losses.py` covers numerical losses,
trial boundaries, gradients, graph execution, model integration, and config
serialization. `src/tests/test_sic_v15_multi_gpu.py` checks loss and gradient
parity across two devices, including unequal shard sizes; its local default
uses two logical CPUs. `src/tests/test_compare_reconstructions.py` checks the
incoming comparison tool.

These checks establish implementation behavior. EEG effectiveness and actual
multi-GPU training still require an experiment run.
