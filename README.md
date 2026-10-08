<p align="center">
  <img src="https://raw.githubusercontent.com/EEGo-UNC/EEGProc/main/docs/source/_static/eegproc-logo.png" alt="EEGProc" width="800">
</p>

<p align="center">
  <strong>Contributors:</strong>
  <a href="https://github.com/VitorInserra">@VitorInserra</a>
  · <a href="https://github.com/Pranav1006">@Pranav1006</a>
  · <a href="https://github.com/sainag7">@sainag7</a>
  · <a href="https://github.com/qwertyuiopzxcvbnmlkjhgfdsa">@Amit_Chalmeti</a>
  · <a href="https://github.com/ygadipalli">@ygadipalli</a>
</p>

# EEGProc

A lightweight Python library for EEG processing and deep
learning, and model explanations. EEGProc is meant to be friendly to AI code writers and non-technical researchers, while still providing all the technical freedom and detail needed for EEG deep learning pipelines.

Built by researchers at the **University of North
Carolina (UNC) Chapel Hill** and **Columbia University**, EEGProc has been used in research published at international
conferences. It helps researchers and developers prepare EEG data, evaluate
models, and explore their predictions with reusable, well-documented components.

We hope researchers will continue to contribute to this library, adding more model architectures, machine learning techniques, and visualization modules.

## Install

```bash
pip install eegproc                   # preprocessing, features, conversion, plotting
pip install "eegproc[deep-learning]"  # adds models, cross-validation, counterfactuals
```

Requires Python 3.10 or newer. TensorFlow is optional and is installed with the
`deep-learning` extra.

## Start with the DREAMER example

The [commented example script](https://github.com/EEGo-UNC/EEGProc/blob/main/examples/dreamer_bilstm_counterfactual.py) walks
through preprocessing, PSD feature extraction, a BiLSTM classifier,
leave-one-subject-out cross-validation (LOSOCV), and a model-agnostic
counterfactual for a held-out input.

Follow the [example README](https://github.com/EEGo-UNC/EEGProc/blob/main/examples/README.md) for installation, dataset
conversion, run commands, and an explanation of every step and output.

For Longleaf experiments with the spatiotemporal 3D-CNN, the
[DREAMER 3D-CNN campaign](experiments/dreamer_3dcnn/README.md) plans a smoke
test, tracks immutable run configurations and downloaded artifacts, maintains a
leaderboard, and recommends the next controlled hyperparameter experiment.

## What EEGProc provides

- **Preprocessing and features:** filtering, detrending, spectral band powers,
  Hjorth parameters, and Shannon, wavelet, and IMF entropy.
- **Dataset conversion:** convert downloaded DREAMER, AMIGOS, EEGEmotions-27,
  and DEAP recordings into tidy CSV tables.
- **Models and evaluation:** reusable CNN/GNN components, RNN classifiers,
  classifier heads, losses, and subject-wise cross-validation.
- **Model explanations:** adapter-based counterfactual optimization for
  differentiable models, with plots and scalp topographies.

See the [getting-started guide](https://github.com/EEGo-UNC/EEGProc/blob/main/docs/source/getting-started.md) for focused API
examples. The [dataset guide](https://github.com/EEGo-UNC/EEGProc/blob/main/docs/source/datasets.md) covers supported layouts
and download links; recordings are not bundled.

## What can you model with EEG?

Developers and researchers can use EEGProc to build and evaluate models that
estimate **valence and arousal**, **attention**, **cognitive load**, and
**engagement** from EEG. Your labeled recordings, model, and evaluation protocol
define the prediction task.

<table>
  <tr>
    <th>Affective state</th>
    <th>Cognitive state</th>
  </tr>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/EEGo-UNC/EEGProc/main/docs/source/_static/valence-arousal.png" alt="Valence–arousal diagram illustrating frustration, enjoyment, boredom, and calmness" width="244"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/EEGo-UNC/EEGProc/main/docs/source/_static/cognitive-state-targets.png" alt="EEG modeling targets: attention, cognitive load, and engagement" width="234"></td>
  </tr>
  <tr>
    <td>Model affect along valence and arousal dimensions.</td>
    <td>Estimate attention, cognitive load, and engagement.</td>
  </tr>
</table>

## A typical EEG pipeline

<p align="center">
  <img src="https://raw.githubusercontent.com/EEGo-UNC/EEGProc/main/docs/source/_static/typical-eeg-pipeline.png" alt="Typical EEG application pipeline: a task and wearable EEG feed a server and machine-learning model, with labels for training and state estimates for application feedback" width="682">
</p>

A typical workflow records EEG during a task, pairs recordings with labels such
as self-reports or task measurements, and uses preprocessing and feature
extraction to prepare model inputs. After training and subject-wise evaluation,
a model produces state estimates that an application can use for feedback.

EEGProc supplies the preprocessing, feature extraction, model-building,
evaluation, and explanation components in this workflow. Your project connects
the recording device, data storage, server, and application feedback loop.
The [DREAMER example](https://github.com/EEGo-UNC/EEGProc/blob/main/examples/README.md) demonstrates the processing, training,
evaluation, and explanation steps on a downloaded dataset.

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

## Plot EEG features over time

Use `plot_eeg_features` for stacked channel traces, channel/band selection, and
image export. The examples below use `psd` and `entropy` from the featurization
section; the four-second windows and 50% overlap match those feature calculations.

```python
from eegproc.plotting import plot_eeg_features

# Save theta and alpha band-power traces for two channels.
fig, axes = plot_eeg_features(
    psd, title="EEG band powers", channels=["AF3", "F7"],
    frequency_bands=["theta", "alpha"], seconds=4, overlap=0.5,
    save_path="bandpowers.png",
)

# Entropy has one value per channel; select channels without a band filter.
entropy_fig, entropy_axes = plot_eeg_features(
    entropy, title="EEG spectral entropy", channels=["AF3", "F7"],
    seconds=4, overlap=0.5, save_path="entropy.png",
)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/EEGo-UNC/EEGProc/main/docs/source/_static/eeg-feature-plots.png" alt="Stacked EEG feature traces showing theta-to-beta band-power ratios over time for P7, O1, and FC6" width="638">
</p>

*Illustrative EEG feature traces: theta-to-beta band-power ratios at P7, O1, and
FC6. Derived ratios can also be passed to the plotting helper as a feature table.*

## Explain predictions with scalp topographies

EEGProc's model-agnostic module can visualize the differences between an input
and its counterfactual across electrodes and frequency bands. Scalp topographies
help show where those changes are concentrated.

![Counterfactual scalp topographies for theta, alpha, and beta bands, showing amplitude differences above and RMS differences below.](https://raw.githubusercontent.com/EEGo-UNC/EEGProc/main/docs/source/_static/counterfactual-topographies.png)

*Example theta, alpha, and beta topographies: amplitude differences in the top
row and root-mean-square (RMS) differences in the bottom row.*

See the [model-agnostic counterfactual guide](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/model_explainability/model_agnostic/README.md#topographies)
for plotting commands and the channel positions, band metadata, and normalization
information needed to interpret your own results.

## Deep Learning

The [DREAMER script](https://github.com/EEGo-UNC/EEGProc/blob/main/examples/dreamer_bilstm_counterfactual.py) builds a BiLSTM
and passes its builder to `cross_validate_dataframe` for leave-one-subject-out
cross-validation. Here is the same pattern for a prepared feature CSV containing
`subject`, `trial`, a binary `label`, and EEG features in time order:

```python
import pandas as pd
from eegproc.deep_learning.supervised.rnn_architectures import BiLSTMClassifier
from eegproc.deep_learning.cross_validation import cross_validate_dataframe

features = pd.read_csv("features.csv")

def build_model(training_features):
    # Build a fresh BiLSTM using this fold's input dimensions.
    return BiLSTMClassifier(
        *training_features.shape[1:], n_classes=2, lstm_units=16, n_bilstm_layers=1,
    ).build()

results = cross_validate_dataframe(
    features, build_model, strategy="fixed_loso", fs=128,
    feature_columns=("AF3_alpha", "F7_alpha"),  # Choose your EEG features only.
    window_rows=4, normalize="subject_zscore",
    fixed_config={}, n_epochs=10, batch_size=32,
)
```

Each subject is held out once, with fixed training settings and windows kept
within trials. `subject_zscore` uses each subject's own unlabeled data, including
the held-out subject's data; omit it if that offline normalization assumption
does not fit your evaluation. Per-subject scores are in `results["user_metrics"]`.

Follow the [example README](https://github.com/EEGo-UNC/EEGProc/blob/main/examples/README.md) to run the complete DREAMER
pipeline, including preprocessing, features, model saving, and counterfactuals.

## Package layout

| Module | Purpose |
| --- | --- |
| [`eegproc.preprocessing`](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/preprocessing.py) | Filtering, detrending, and band decomposition |
| [`eegproc.featurization`](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/featurization.py) | Spectral, Hjorth, wavelet, and IMF features |
| [`eegproc.data`](https://github.com/EEGo-UNC/EEGProc/tree/main/src/eegproc/data) | Dataset conversion, table schema, and trial-safe windowing |
| [`eegproc.deep_learning`](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/deep_learning/README.md) | Reusable models, cross-validation, and domain generalization |
| [`eegproc.model_explainability`](https://github.com/EEGo-UNC/EEGProc/blob/main/src/eegproc/model_explainability/model_agnostic/README.md) | Model-agnostic counterfactuals and topographies |
| [`eegproc.plotting`](https://github.com/EEGo-UNC/EEGProc/tree/main/src/eegproc/plotting) | EEG feature plots |

## Documentation and contributing

Browse the [documentation](https://eego-unc.github.io/EEGProc/), follow the
[contribution guide](https://github.com/EEGo-UNC/EEGProc/blob/main/CONTRIBUTING.md), or read the [changelog](https://github.com/EEGo-UNC/EEGProc/blob/main/CHANGELOG.md).
If you use EEGProc in your research, see [CITATION.cff](https://github.com/EEGo-UNC/EEGProc/blob/main/CITATION.cff).

## License

GPLv2. See [LICENSE](https://github.com/EEGo-UNC/EEGProc/blob/main/LICENSE).
