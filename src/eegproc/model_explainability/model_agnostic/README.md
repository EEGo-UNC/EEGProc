# Model-agnostic counterfactuals

This package provides an adapter-based counterfactual optimizer. An adapter
creates the optimization state, classifies it, and reconstructs input-shaped
signals. Reconstructed signals are not classified again. Optional validity
constraints also belong to the adapter rather than the optimizer.

| File | Owns |
| --- | --- |
| `adapter.py` | Adapter and trial-dataset contracts, dynamic factory loading, and the generic Keras input adapter. |
| `optimizer.py` | Architecture-agnostic gradient optimization. |
| `runner.py` | Command-line loading, trial selection, optimization, and result files. |
| `sic_adapter.py` | SIC latent/joint adapter and DREAMER loader. |
| `topography.py` | Metadata-aware topographies with optional source-unit restoration. |

## SIC example

```bash
PYTHONPATH=src python -m eegproc.model_explainability.model_agnostic.runner \
  --model /path/to/loso_target_0_zero_shot.keras \
  --adapter eegproc.model_explainability.model_agnostic.sic_adapter:create_sic_adapter \
  --adapter-config '{"model_module":"eegproc.deep_learning.joint_architectures.SICModelv15.sic_model","decoder_mode":"joint"}' \
  --data-loader eegproc.model_explainability.model_agnostic.sic_adapter:load_sic_raw_trials \
  --data-config '{"raw_eeg_npy":"datasets/dreamer_eeg.npy","raw_labels_npy":"datasets/dreamer_labels.npy","dataset":"dreamer","label_dimension":"arousal","fs":128,"window_sec":1,"window_normalization":"global_rms"}' \
  --subject-id 0 --trial-id 0 \
  --max-steps 20 \
  --report-constraint vcsc \
  --out-dir runs/counterfactuals/adapter_sic_v15_subject0_trial0
```

`--report-constraint vcsc` computes VCSC only for the final reference and
counterfactual. Use `--constraint-weight vcsc=WEIGHT` to include it in the
optimization objective.

## Generic Keras example

```bash
PYTHONPATH=src python -m eegproc.model_explainability.model_agnostic.runner \
  --model /path/to/model.keras \
  --adapter eegproc.model_explainability.model_agnostic.adapter:create_keras_input_adapter \
  --adapter-config '{"output_kind":"logits","registration_modules":["my_package.models"]}' \
  --trials-npz /path/to/prepared_trials.npz \
  --subject-id 0 --trial-id 0 \
  --out-dir runs/counterfactuals/generic_input_model
```

External adapter and dataset-loader factories use
`package.module:function`. Prepared NPZ files use `features`, `subject_ids`,
`trial_ids`, and optional `labels`; normalization and channel metadata are
also supported.

## Topographies in source units

```bash
PYTHONPATH=src python -m eegproc.model_explainability.model_agnostic.topography \
  runs/.../subject_0_trial_0/counterfactual.npz \
  --branch joint --measure mean-absolute --physical-units --no-show
```

The metadata-aware SIC loader preserves each window's affine normalization
transform. The plot says `source signal units` unless the dataset provides a
specific `signal_unit`; label values as microvolts only when that provenance
is known.
