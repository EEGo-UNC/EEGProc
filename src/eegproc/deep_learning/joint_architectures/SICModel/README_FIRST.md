# SIC joint VC-head update

This update replaces the former parallel dense-logit/VC arrangement with one
trainable `VariationalClassifier` head. That head produces the only emotion
logits, and focal loss plus the enabled VC regularizers are optimized jointly.

Copy the included `src/` tree over the matching paths in your EEGProc checkout.
The two Markdown files document the updated architecture and hyperparameters.

The existing smoke and full-run SLURM scripts do not need changes. Their
arguments remain compatible. In particular, `calibration_use_vc_target=true`
keeps the VC regularizers active during calibration, while the VC logits head
is trainable either way.

Retrain the LOSO population models after installing this update. Checkpoints
created with the removed dense logits layer are not architecture-compatible.
