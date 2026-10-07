.. EEGProc documentation master file, created by
   sphinx-quickstart on Wed Oct 15 00:10:05 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EEGProc documentation
=====================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   getting-started
   datasets
   api/modules


EEGProc is a vectorized library for preprocessing EEG (electroencephalogram)
data, extracting features from it, and evaluating models on it subject by
subject. It is built for researchers and developers working in neuroscience,
biomedical engineering, and machine learning.
Check out and **star** or **fork** the project at https://github.com/EEGo-UNC/EEGProc

**Features**

- **Preprocessing**: detrending, interpolation, notch and band-pass filtering
  into frequency bands.
- **Featurization**: band power, Shannon, wavelet and IMF energy and entropy,
  and Hjorth parameters, computed on pandas DataFrames.
- **Dataset conversion**: ``eegproc-to-csv`` turns downloaded AMIGOS, DREAMER,
  EEGEmotions-27 and DEAP files into tidy CSV tables.
- **Subject-wise evaluation** (``eegproc[deep-learning]``): leave-one-subject-out,
  nested and few-shot calibration cross-validation straight from a tidy table.
- **Model components** (``eegproc[deep-learning]``): CNN, GNN and RNN encoders,
  variational classifier heads, contrastive and autoencoder losses, and
  domain-generalization helpers.
- **Counterfactual explanations** (``eegproc[deep-learning]``): adapter-based
  counterfactual optimization for differentiable models, with scalp
  topographies.

**Contributing**

Contributions are welcome! If you have ideas for new features or improvements, feel free to open an issue or submit a pull request.

**License**

This project is licensed under the GPLv2 License.
