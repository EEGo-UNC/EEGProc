# Changelog

## 2.0.0 — unreleased

### Library structure

- Keep v2's modular cross-validation package and the deprecated public
  `cross_val.py` compatibility wrapper.
- Include the DataFrame schema, feature-column parser, and trial-safe windowing
  modules referenced by the v2 API and documentation.
- Retain reusable CNN/GNN encoders and decoders, RNN classifier builders, dense
  and variational classifier heads, autoencoder and contrastive losses, and
  domain-generalization utilities.
- Keep adapter-based counterfactual optimization, generic Keras loading, the
  command-line runner, and plotting driven by supplied channel metadata.
- Remove complete research model pipelines, model-specific explainability,
  experiment reports, Slurm launchers, and smoke scripts.

### Packaging and APIs

- Require Python 3.10 or newer and declare version 2.0.0.
- Keep TensorFlow and scikit-learn in the optional `deep-learning` extra.
- Limit package discovery to `eegproc` and verify wheel contents during builds.
- Keep v2's PSD-table input and across-band output for `shannons_entropy`.
- Keep v2's figure-returning plotting API and its overlap-aware time axis.
- Preserve preprocessing and feature fixes for grouped inputs and writable
  arrays required by wavelet and IMF routines.

### Dataset conversion

- Added `eegproc-to-csv` for AMIGOS, DREAMER (MAT or split CSV), EEGEmotions-27,
  Cowen emotion mappings, and DEAP preprocessed Python files.
- Added chunked CSV/text conversion, gzip output, input validation, and dataset
  download documentation. No recordings or generated datasets are bundled.
