Getting Started
===============

EEGProc is a vectorized library for preprocessing EEG (electroencephalogram)
data, extracting features from it, and evaluating models on it subject by
subject. It is built for researchers and developers working in neuroscience,
biomedical engineering, and machine learning.

Installation
------------

Install from PyPI (Python 3.10 or newer):

```bash
pip install eegproc                    # preprocessing, features, data conversion, plotting
pip install "eegproc[deep-learning]"   # adds cross-validation, models, counterfactuals
```

or, for the latest development version:

```bash
pip install git+https://github.com/EEGo-UNC/EEGProc.git
```

Dependencies
------------

The base install relies on:

- **NumPy**, **Pandas**, **SciPy** – numerical processing
- **PyWavelets** – wavelet features
- **PyEMD** (`EMD-signal`) – empirical mode decomposition
- **Matplotlib** – plotting utilities

The `deep-learning` extra adds **TensorFlow**, **scikit-learn**, and
**cloudpickle**.

Quick Start
-----------

1. **Import and load your EEG data:**

```python
import pandas as pd
from eegproc import bandpass_filter, FREQUENCY_BANDS

df = pd.read_csv("my_eeg_data.csv")
fs = 128  # Hz
```

2. **Filter into frequency bands:**

```python
from eegproc import bandpass_filter, FREQUENCY_BANDS

clean = bandpass_filter(df, fs, bands=FREQUENCY_BANDS)
# -> columns named {channel}_{band}, e.g. AF3_alpha
```

3. **Extract features:**

```python
from eegproc import psd_bandpowers, shannons_entropy, hjorth_params

psd = psd_bandpowers(clean, fs, bands=FREQUENCY_BANDS)   # {channel}_{band}
entropy_df = shannons_entropy(psd)                        # {channel}_entropy
hjorth_df = hjorth_params(clean, fs)                      # {channel}_{band}_activity, ...
```

`shannons_entropy` consumes the **PSD table**, not the raw signal, and returns one
value per channel describing how evenly that channel's energy is spread across
bands. The other entropy featurizers follow the same shape:
`wavelet_entropy` consumes `wavelet_band_energy`, and `imf_entropy` consumes
`imf_band_energy`.

4. **Visualize results:**

```python
from eegproc.plotting import plot_eeg_features

plot_eeg_features(entropy_df, title="Shannon Entropy per Channel", seconds=4.0)
```

Documentation Structure
-----------------------

```{toctree}
:maxdepth: 2

api/modules
```
