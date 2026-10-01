# EEGProc

Featurization and deep learning library for EEG that is AI-friendly, lightweight, and easy to use.

EEGProc is built for researchers and developers who aim to implement EEG machine learning without reinventing the wheel. It supports writing clean code and reduces the margin for error involved in creating and testing a model from scratch.

## Dataset conversion

Convert downloaded AMIGOS, DREAMER, EEGEmotions-27, or DEAP data
to CSV with the base installation:

```bash
eegproc-to-csv --dataset amigos --input /path/to/AMIGOS --output /path/to/amigos_joined.csv.gz
```

See the [dataset guide](https://github.com/EEGo-UNC/EEGProc/blob/main/docs/source/datasets.md) for download links, supported
layouts, and examples. Recordings and generated datasets are not bundled.

## Included components

The library keeps reusable CNN/GNN encoders and decoders, RNN classifiers,
classifier heads, losses, cross-validation, and domain-generalization helpers.
Adapter-based counterfactuals work with caller-supplied models and datasets; see
[model-agnostic counterfactuals](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/model_explainability/model_agnostic/README.md).

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
| `hjorth_params` | filtered signal | `{channel}_{band}_activity`, `_mobility`, `_complexity` |
| `wavelet_band_energy` | raw signal | `{channel}_{band}_wenergy` |
| `wavelet_entropy` | wavelet energy | `{channel}_wentropy` |
| `imf_band_energy` | raw signal | `{channel}_{band}_imfenergy` |
| `imf_entropy` | IMF energy | `{channel}_imfentropy` |

## Cross-validation

Subject-wise evaluation takes a **tidy table**: your feature columns plus
`subject`, `trial`, and a label column. Trials never straddle a fold.
Normalization is off by default; `normalize="subject_zscore"` standardizes each
subject with its own statistics.

```python
import tensorflow as tf
from eegproc import FREQUENCY_BANDS, bandpass_filter, feature_grouped_by_metadata, psd_bandpowers
from eegproc.deep_learning.cross_validation import cross_validate_dataframe

def band_powers(signal, fs, bands, **_):
    return psd_bandpowers(bandpass_filter(signal, fs, bands=bands), fs, bands=bands)

features = feature_grouped_by_metadata(
    raw,                                           # electrodes + "subject", "trial" columns
    target_function=band_powers,
    fs=128,
    group_by_metadata_columns=["subject", "trial"],
)
features = features.merge(labels, on=["subject", "trial"])   # labels: subject, trial, label

def build_model(training_features, **hyperparameters):
    # Declaring training_features gives the builder this fold's training windows.
    _, timesteps, n_features = training_features.shape
    model = tf.keras.Sequential([
        tf.keras.layers.Input((timesteps, n_features)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

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

### Using converted datasets safely

Unless you pass `feature_columns`, every numeric column that is not a declared
subject, trial, time, or label column is used as a feature. Tables written by
`eegproc-to-csv` also hold the other ratings, ECG, sample indices, and (for
EEGEmotions) demographics, so predicting valence from them without
`feature_columns` would silently train on arousal and dominance too. Name the
columns explicitly:

```python
import pandas as pd

EEG_CHANNELS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")

df = pd.read_csv("dreamer_joined.csv.gz")
df = df[df["segment"] == "stimulus"]
df["label"] = (df["valence"] >= 3).astype(int)

results = cross_validate_dataframe(
    df, build_model, strategy="loso",
    kind="signal", fs=128, window_sec=1.0,
    subject_columns=("subject_id",), trial_columns=("trial_id",),
    time_column="sample_idx", feature_columns=EEG_CHANNELS,
    label_column="label",
)
```

## Package layout

- [`eegproc.preprocessing`](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/preprocessing.py) — filtering, detrending, notch, band decomposition
- [`eegproc.featurization`](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/featurization.py) — spectral, Hjorth, wavelet and IMF features
- [`eegproc.data`](https://github.com/EEGo-UNC/EEGProc/tree/main/src/eegproc/data) — the tidy schema and the windowing assembler (no TensorFlow)
  - [`to_csv.py`](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/data/to_csv.py) — dataset conversion
- [`eegproc.deep_learning`](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/deep_learning/README.md)
  - [`cross_validation`](https://github.com/EEGo-UNC/EEGProc/tree/main/src/eegproc/deep_learning/cross_validation) — the cross-validation strategies
  - [`supervised`](https://github.com/EEGo-UNC/EEGProc/tree/main/src/eegproc/deep_learning/supervised) — RNN classifier builders, dense and variational classifier heads, and contrastive loss
  - [`unsupervised`](https://github.com/EEGo-UNC/EEGProc/tree/main/src/eegproc/deep_learning/unsupervised) — CNN/GNN encoders and decoders, graph layers, and autoencoder losses
  - [`domain_generalization`](https://github.com/EEGo-UNC/EEGProc/tree/main/src/eegproc/deep_learning/domain_generalization) — alternating subject groups and meta-learning strategies
  - [`training_outputs.py`](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/deep_learning/training_outputs.py) — training callbacks, metrics, and diagnostics
  - [`prepare_datasets.py`](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/deep_learning/prepare_datasets.py) — converters for supported public EEG datasets
- [`eegproc.model_explainability.model_agnostic`](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/model_explainability/model_agnostic/README.md) — adapter-based counterfactuals
- [`eegproc.plotting`](https://github.com/EEGo-UNC/EEGProc/tree/main/src/eegproc/plotting) — `plot_eeg_features`

## Scope

EEGProc gives you data preparation, evaluation, and reusable model components,
not complete research models. The cross-validators take a builder that returns
a compiled Keras model and handle folds, windowing, thresholds, calibration and
reporting; the encoders, classifier heads and losses in `deep_learning` are
building blocks for such builders.

## Documentation

<https://eego-unc.github.io/EEGProc/>

## Contributing

See [CONTRIBUTING.md](https://github.com/EEGo-UNC/EEGProc/blob/main/CONTRIBUTING.md). Changes are documented in
[CHANGELOG.md](https://github.com/EEGo-UNC/EEGProc/blob/main/CHANGELOG.md).

## License

GPLv2. See [LICENSE](https://github.com/EEGo-UNC/EEGProc/blob/main/LICENSE).
