# EEGProc

Featurization and Deep Learning library for EEG that is AI-friendly, lightweight, and easy to use.

EEGProc is built for researchers and developers who aim to implement EEG machine learning without reinventing the wheel. It supports the implementation clean code and reduces the margin for error involved in creating and testing a model from scratch.

## Dataset conversion

Convert downloaded AMIGOS, DREAMER, EEGEmotions-27, or DEAP data
to CSV with the base installation:

```bash
eegproc-to-csv --dataset amigos --input /path/to/AMIGOS --output /path/to/amigos_joined.csv.gz
```

See the [dataset guide](docs/source/datasets.md) for download links, supported
layouts, and examples. Recordings and generated datasets are not bundled.

## Included components

The library keeps reusable CNN/GNN encoders and decoders, RNN classifiers,
classifier heads, losses, cross-validation, and domain-generalization helpers.
Adapter-based counterfactuals work with caller-supplied models and datasets; see
[model-agnostic counterfactuals](src/eegproc/model_explainability/model_agnostic/README.md).

## Install

```bash
pip install eegproc                    # preprocessing + featurization
pip install "eegproc[deep-learning]"   # adds the cross-validation stack (TensorFlow)
```

The base install deliberately does **not** pull in TensorFlow. If you only need
filtering and features, you do not pay for a deep-learning runtime.

Requires Python 3.10 or newer.

## Featurization

```python
import pandas as pd
from eegproc import bandpass_filter, psd_bandpowers, shannons_entropy, FREQUENCY_BANDS

raw = pd.read_csv("my_eeg.csv")        # one column per electrode
fs = 128

clean = bandpass_filter(raw, fs, bands=FREQUENCY_BANDS)   # -> AF3_alpha, AF3_theta, ...
psd = psd_bandpowers(clean, fs, bands=FREQUENCY_BANDS)    # one row per window
entropy = shannons_entropy(psd)                           # -> AF3_entropy, F7_entropy
```

Featurizers compose in a pipeline: the band-energy functions consume a filtered
signal, and the entropy functions consume the corresponding energy table.

| Function | Consumes | Emits |
|---|---|---|
| `bandpass_filter` | raw signal | `{channel}_{band}` |
| `psd_bandpowers` | filtered signal | `{channel}_{band}` |
| `shannons_entropy` | PSD table | `{channel}_entropy` |
| `hjorth_params` | filtered signal | `{channel}_activity`, `_mobility`, `_complexity` |
| `wavelet_band_energy` | raw signal | `{channel}_{band}_wenergy` |
| `wavelet_entropy` | wavelet energy | `{channel}_wentropy` |
| `imf_band_energy` | raw signal | `{channel}_{band}_imfenergy` |
| `imf_entropy` | IMF energy | `{channel}_imfentropy` |

## Cross-validation

Subject-wise evaluation takes a **tidy table**: your feature columns plus
`subject`, `trial`, and a label column. Trials never straddle a fold, and each
subject's normalization is computed from that subject alone.

```python
from eegproc import feature_grouped_by_metadata, psd_bandpowers
from eegproc.deep_learning.cross_validation import cross_validate_dataframe

features = feature_grouped_by_metadata(
    eeg_df=raw,                                    # has subject/trial columns
    target_function=psd_bandpowers,
    fs=128,
    group_by_metadata_columns=["subject", "trial"],
)
features = features.merge(labels, on=["subject", "trial"])

results = cross_validate_dataframe(
    features, build_model, strategy="loso", fs=128, label_column="label",
)

for row in results["user_metrics"]:
    print(row["subject_id"], row["accuracy"])      # "P07" 0.71
```

Results are reported against your own subject identifiers, not positional indices.

Available strategies: `loso` (leave-one-subject-out), `fixed_loso` (a single fixed
configuration), `subject_calibration` (few-shot adaptation to a held-out subject),
and `nested_lnso` (nested leave-N-subjects-out).

Sessions need no special support: `trial_columns=("session", "trial")` scopes
trials per session, and `subject_columns=("subject", "session")` gives
leave-one-session-out through the same code path.

If you already hold NumPy arrays, `loso_cv` and friends take them directly.

## Package layout

- [`eegproc.preprocessing`](src/eegproc/preprocessing.py) — filtering, detrending, notch, band decomposition
- [`eegproc.featurization`](src/eegproc/featurization.py) — spectral, Hjorth, wavelet and IMF features
- [`eegproc.data`](src/eegproc/data/) — the tidy schema and the windowing assembler (no TensorFlow)
  - [`to_csv.py`](src/eegproc/data/to_csv.py) — dataset conversion
- [`eegproc.deep_learning`](src/eegproc/deep_learning/README.md)
  - [`cross_validation`](src/eegproc/deep_learning/cross_validation/) — the cross-validation strategies
  - [`supervised`](src/eegproc/deep_learning/supervised/) — RNN classifier builders, dense and variational classifier heads, and contrastive loss
  - [`unsupervised`](src/eegproc/deep_learning/unsupervised/) — CNN/GNN encoders and decoders, graph layers, and autoencoder losses
  - [`domain_generalization`](src/eegproc/deep_learning/domain_generalization/) — alternating subject groups and meta-learning strategies
  - [`training_outputs.py`](src/eegproc/deep_learning/training_outputs.py) — training callbacks, metrics, and diagnostics
  - [`prepare_datasets.py`](src/eegproc/deep_learning/prepare_datasets.py) — converters for supported public EEG datasets
- [`eegproc.model_explainability.model_agnostic`](src/eegproc/model_explainability/model_agnostic/README.md) — adapter-based counterfactuals
- [`eegproc.plotting`](src/eegproc/plotting/) — `plot_eeg_features`

## Scope

EEGProc gives you data preparation and evaluation. It does not ship model
architectures — you pass a builder that returns a compiled Keras model, and the
cross-validators handle folds, windowing, thresholds, calibration and reporting.

## Documentation

<https://eego-unc.github.io/EEGProc/>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are documented in
[CHANGELOG.md](CHANGELOG.md).

## License

GPLv2. See [LICENSE](LICENSE).
