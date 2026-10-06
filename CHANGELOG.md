# Changelog

## Unreleased

- Port the DREAMER/MTLFuseNet spatiotemporal 3D CNN from the retired SIC
  experiment branch into the reusable v2 encoder/decoder package.

## 2.0.0 — 2026-10-04

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

### Examples

- Add a commented DREAMER example combining preprocessing, PSD features,
  fixed-setting BiLSTM LOSOCV, and a held-out input-space counterfactual.

### Dataset conversion

- Added `eegproc-to-csv` for AMIGOS, DREAMER (MAT or split CSV), EEGEmotions-27,
  Cowen emotion mappings, and DEAP preprocessed Python files.
- Added chunked CSV/text conversion, gzip output, input validation, and dataset
  download documentation. No recordings or generated datasets are bundled.

### Breaking changes from 1.0.0

- `shannons_entropy` consumes the table from `psd_bandpowers` and returns one
  value per channel, `{channel}_entropy`: how evenly that channel's energy is
  spread across bands. In 1.0.0 it took the signal with `fs`, `window_sec`,
  `overlap` and `detrend`, and returned `{channel}_{band}_entropy`.

  ```python
  # 1.0.0
  entropy = shannons_entropy(clean, fs, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)
  # 2.0.0
  entropy = shannons_entropy(psd_bandpowers(clean, fs, bands=FREQUENCY_BANDS))
  ```

- `preprocessing.detrend_df` was removed; use `apply_detrend("linear", df)`.
- The base install no longer pulls in scikit-learn, joblib, dill, multiprocess,
  pathos and their helpers; it now includes matplotlib for plotting. Install
  `eegproc[deep-learning]` for cross-validation, models and counterfactuals.
- Python 3.10 or newer is required (1.0.0 declared `>=3` but needed 3.10).

### Changed

- `generate_all_features` and `feature_grouped_by_metadata` print their
  progress line only with `verbose=True`.
- The `joblib` requirement and the unused `data` extra were dropped from the
  optional dependencies; cloudpickle is required directly instead.

### Fixed

- EEGEmotions-27 valence/arousal labels: `prepare_datasets` read
  `cowen_27_valence_arousal.csv` by row position, but the file is sorted by
  quadrant, so most trials received another emotion's ratings. Rows are now
  matched by emotion name in EEGEmotions-27 ID order.
- `RNNClassifier(loss="variational")` applied softmax twice when computing its
  loss.
- `loso_cv` and `cross_validate_dataframe` raised "Unsupported metric:
  binary_f1" when called with their default metrics.
- `GCN.GCNDecoder` and the band-separated `GCNDecoder` were registered under the
  same Keras name; the band-separated one is now
  `eegproc>BandSeparatedGCNDecoder`.
- `KerasInputAdapter` no longer requires `output_key`.
- `VariationalAutoencoderLoss` raised a shape error for the sequence latents
  that the EEGProc encoders produce; the KL term is now reduced per sample.
- `eegproc-to-csv` works on drives without hard links (exFAT, FAT, some
  network shares).
