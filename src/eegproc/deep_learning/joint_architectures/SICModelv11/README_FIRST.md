# SIC trial-level BiGRU update

This update changes SIC builder API version 11 from a dense classifier over an
average trial embedding to a recurrent trial classifier:

```text
one-second EEG windows
    -> parallel GCN-GRU and BiLSTM encoders
    -> one embedding per window
    -> ordered GRU/BiGRU sequence
    -> final recurrent state
    -> one trainable VariationalClassifier logits head
```

There is no averaging across a trial's window embeddings and no dense hidden
classifier stack. The supplied scripts use a tapered two-layer BiGRU with 128
units per direction in the first layer and 64 per direction in the second.
Focal loss and the enabled VC regularizers still use the same sole logits tensor.

Install the files over their matching EEGProc paths. In particular:

- `rnn_architectures.py` adds reusable `GRUClassifier`, `BiGRUClassifier`, and
  `build_sequence_summarizer()` implementations.
- `sic_model.py` uses that shared recurrent implementation.
- `sic_model_train.py` groups one-second windows into ordered complete trials.
- Both Slurm scripts now pass `--classification-level trial` and use the new
  recurrent-classifier hyperparameters.

The download preserves the `SICModelv11/` folder layout shown in the project.
It also includes `required_supervised_update/rnn_architectures.py`; copy that
file over `src/eegproc/deep_learning/supervised/rnn_architectures.py` because
the per-layer width support is implemented in the shared recurrent builders.

The main classifier settings are:

```json
{
  "classifier_rnn_type": "bigru",
  "classifier_rnn_units": {"fixed": [128, 64]},
  "n_classifier_rnn_layers": 2,
  "classifier_rnn_dropout": 0.20
}
```

A scalar width is repeated across the requested number of layers. A sequence
specifies each layer separately. The `fixed` wrapper makes `[128, 64]` one
architecture; use `{"grid": [[128], [128, 64]]}` to compare architectures.

`calibration_unfreeze_layers=1` adapts only the VC head. A value of `2` adapts
the GRU/BiGRU and VC together. `calibration_use_vc_target=true` keeps the VC
regularizers active during calibration; the VC logits head is trainable either
way.

Retrain the LOSO population models after installing this update. Checkpoints
created with the former dense classifier or mean-pooled trial representation
are not architecture-compatible with API version 11.
