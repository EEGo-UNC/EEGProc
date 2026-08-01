# CMHFE / CMHFE-DAN

This folder contains the raw-EEG emotion model and the LOSO training entry point used to run it.

## What It Does

CMHFE processes raw EEG windows directly, without handcrafted features.

The data flow is:

1. Raw EEG window with shape `(batch, channels, samples)`
2. Permute to `(batch, samples, channels)`
3. Four 1D convolution blocks
4. Optional max-pooling for the DANN variant
5. Transformer encoder block
6. Global average pooling
7. Two separate emotion heads:
   - valence
   - arousal
8. Optional domain classifier when DANN is enabled

The reusable implementation lives in [cmhfe_dan.py](cmhfe_dan.py).
The LOSO training script lives in [train_cmhfe_loso.py](train_cmhfe_loso.py).

## Model Structure

The shared feature extractor is built from:

- `CNN1DFeatureExtractor`
- `TransformerEncoder`
- `CMHFEFeatureExtractor`

The classification heads are separate layers:

- `EmotionHead` for valence
- `EmotionHead` for arousal
- `DomainClassifier` for the optional domain-adversarial branch

The gradient reversal behavior used by the DANN branch is implemented by `GradientReversalLayer`.

## Configuration

Most behavior is controlled through `CMHFEConfig`.

Important options include:

- EEG channels
- window length
- sampling frequency
- convolution kernel size, stride, and padding
- dropout rate
- L2 regularization
- transformer heads and hidden sizes
- whether max-pooling is enabled
- whether the domain branch is enabled
- domain-loss weight
- valence/arousal thresholds
- learning rate
- batch size

The default settings are tuned to match the architecture described in the specification, but every important value is configurable.

## Data Format

The training script expects preconverted NumPy arrays with the usual subject/trial layout:

- EEG: `(n_subjects, n_trials, n_channels, n_samples)`
- labels: `(n_subjects, n_trials, n_label_dims)`

The script does not compute FFT, PSD, wavelets, or other handcrafted features.

It builds sliding raw windows from the EEG arrays, z-scores each subject by default, then uses median-style binary thresholds for valence and arousal.

## LOSO Training

The script uses the repository's built-in LOSO helper so that each subject is held out once as the test fold.

For this model, the LOSO helper receives a label mapping:

```python
{
    "valence": y_valence,
    "arousal": y_arousal,
}
```

That lets one cross-validation run train and evaluate both emotion heads at the same time.

## Output

Each LOSO run writes a JSON summary into the results directory. The file contains:

- per-fold results
- mean scores
- standard deviation scores
- the resolved dataset and model configuration

## Example Commands

Run from the project root.

### DREAMER

```bash
python -m eegproc.deep_learning.supervised.train_cmhfe_loso \
  --dataset dreamer \
  --eeg_path path/to/dreamer_eeg.npy \
  --labels_path path/to/dreamer_labels.npy \
  --results_dir runs/cmhfe_dreamer \
  --epochs 50 \
  --batch_size 32
```

### DEAP

```bash
python -m eegproc.deep_learning.supervised.train_cmhfe_loso \
  --dataset deap \
  --eeg_path path/to/deap_eeg.npy \
  --labels_path path/to/deap_labels.npy \
  --results_dir runs/cmhfe_deap \
  --epochs 50 \
  --batch_size 32
```

### Custom data

```bash
python -m eegproc.deep_learning.supervised.train_cmhfe_loso \
  --dataset custom \
  --n_channels 14 \
  --eeg_path path/to/custom_eeg.npy \
  --labels_path path/to/custom_labels.npy \
  --results_dir runs/cmhfe_custom \
  --window_length_sec 4.0 \
  --sampling_frequency 128 \
  --valence_threshold 5.0 \
  --arousal_threshold 5.0
```

## Notes

- Use `--enable_maxpool` if you want the optional max-pooling stage before the transformer.
- For custom datasets, provide `--n_channels` explicitly.
- If you want to inspect or extend the model, start with [cmhfe_dan.py](cmhfe_dan.py).