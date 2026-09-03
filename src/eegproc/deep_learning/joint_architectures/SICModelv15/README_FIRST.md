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
