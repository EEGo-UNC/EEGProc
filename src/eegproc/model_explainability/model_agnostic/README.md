# Model-agnostic counterfactuals

Install `eegproc[deep-learning]`. The optimizer supports differentiable TensorFlow
models through `CounterfactualAdapter`. A built-in `KerasInputAdapter` optimizes
inputs directly; custom adapters can optimize latent states and reconstruct one
or more signals. Optimization updates the state, leaving model weights unchanged.

## Python API

```python
from eegproc.model_explainability.model_agnostic import (
    KerasInputAdapter, ModelAgnosticCounterfactualOptimizer,
)

adapter = KerasInputAdapter(model, output_kind="logits")
optimizer = ModelAgnosticCounterfactualOptimizer(adapter, target_probability=0.8)
result = optimizer.optimize(trial[None, ...], target_class=1)
```

`model` is a loaded Keras model; `trial` contains one input without its batch axis.
Use `output_kind="probabilities"` when the model emits probabilities. The model
must return at least two class scores per input; wrap a binary model with a
single sigmoid unit so that it returns `[1 - p, p]`. Custom
adapters implement `initial_state`, `logits_from_state`, and `reconstruct`.
They can also implement named `constraint` metrics. Reconstruction outputs must
have the input shape. Multiclass models require an explicit target class.

The result contains `summary`, `history`, and `arrays`. Signal proximity compares
each candidate reconstruction with the original reconstruction, so decoder error
is not counted as a counterfactual change.

## Command line

```bash
python -m eegproc.model_explainability.model_agnostic.runner \
  --model /path/to/model.keras \
  --adapter eegproc.model_explainability.model_agnostic.adapter:create_keras_input_adapter \
  --adapter-config '{"output_kind":"logits"}' \
  --trials-npz /path/to/prepared_trials.npz \
  --subject-id 0 --trial-id 0 \
  --out-dir /path/to/results
```

Prepared NPZ inputs contain `features`, `subject_ids`, `trial_ids`, and optional
`labels`. Features include a leading trial axis. Optional fields describe channel
names, band names, channel positions, feature order, normalization offset/scale,
and signal units. External adapter and dataset-loader factories use
`package.module:function`. Pass a dataset factory with `--data-loader`.

The runner writes JSON summaries and CSV optimization histories. Arrays remain
available through the Python API. To save an archive for plotting, use
`numpy.savez_compressed(path, **result["arrays"], ...)` and supply the channel
metadata that describes your own data.

## Topographies

```bash
python -m eegproc.model_explainability.model_agnostic.topography \
  /path/to/counterfactual.npz --branch input --no-show
```

Archives must include `channel_positions` shaped `(n_channels, 2)` in normalized
scalp coordinates. Provide `channel_names` and, for flattened channel-band data,
`band_names` and `feature_order` (`channel-major` or `band-major`). Without band
metadata, all features are interpreted as one band. No dataset montage is assumed.
Use `--physical-units` only when the archive includes the affine normalization
transform; the unit label comes from `signal_unit` or `--signal-unit`.
