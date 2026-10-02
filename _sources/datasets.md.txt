# Convert public datasets to CSV

Install EEGProc and use one command for the supported dataset formats:

```bash
pip install eegproc
eegproc-to-csv --dataset amigos --input /data/AMIGOS/data_preprocessed --output /data/csv/amigos_joined.csv.gz
```

From a source checkout, install it with `pip install -e .`. The equivalent module
command is `python -m eegproc.data.to_csv`. Conversion uses the base installation;
TensorFlow is not required.

## Get the datasets

Download and extract datasets separately. EEGProc includes conversion code and
links only: no recordings, videos, rating matrices, or generated CSVs are bundled.
Use each provider's access process and citation requirements.

| Dataset | Source and access | Supported input |
| --- | --- | --- |
| AMIGOS | [QMUL dataset page](https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/), also linked from the [author's dataset directory](https://sites.google.com/view/ioannispatras/datasets-and-code) | Extracted `Data_Preprocessed_P*.mat` files, recursively under the input folder, or one file |
| DREAMER | [Official Zenodo record and access request](https://zenodo.org/records/546113) | `DREAMER.mat`, or the three exported CSV files described below |
| EEGEmotions-27 | [Authors' dataset repository](https://github.com/huytungst/EEGEmotions-27) | `eeg_raw/*.txt`, `participants_info.csv`, and optional feature/label CSV |
| Cowen/Keltner emotion ratings | [KamitaniLab data instructions](https://github.com/KamitaniLab/EmotionVideoNeuralRepresentation#data-fmri-data-and-features) and [feature archive on Figshare](https://doi.org/10.6084/m9.figshare.11988351) | Extracted `features.zip`, specifically `amt/mean_score_concat/category` and `dimension` |
| DEAP | [QMUL dataset page](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), also linked from the [author's dataset directory](https://sites.google.com/view/ioannispatras/datasets-and-code) | Official preprocessed Python `s*.dat` files, or one file |

## Common behavior

- Outputs can end in `.csv` or `.csv.gz` (gzip compression).
- Existing output files require `--overwrite`. A failed conversion leaves an
  existing output intact, and input files cannot be used as the output.
- `--chunksize 100000` controls text/CSV batches. MATLAB and DEAP inputs load one
  source file at a time; `DREAMER.mat` contains the entire dataset, so enough RAM
  for that file is required. The DREAMER CSV join uses temporary disk space for
  both signal tables. Cowen loads the small rating matrices.
- EEG outputs use one row per EEG sample with `subject_id`, `trial_id`, `segment`,
  `sample_idx`, channel values, and dataset-specific labels/metadata. Subject and
  trial boundaries are preserved. No subjects are selected for a research split.
- Values are not normalized, filtered, windowed, or binarized. Original rating
  precision is retained. The native DREAMER reader aligns ECG to EEG as described
  below; the Cowen command computes a derived mapping rather than an EEG table.
- Missing metadata generates warnings. Malformed layouts and duplicate join keys
  fail explicitly. Headerless EEG text must have exactly 14 columns; headered
  files may include extra columns, which are ignored after selecting the EEG names.
- MATLAB v7.3/HDF5 files are unsupported. Use the original preprocessed MAT
  release or the supported CSV export layout.

## AMIGOS

```bash
eegproc-to-csv --dataset amigos \
  --input /data/AMIGOS/data_preprocessed \
  --output /data/csv/amigos_joined.csv
```

Reads `joined_data` and `labels_selfassessment`. Each trial can be sample-major
or channel-major (16 or 17 channels). Output retains the 14 Emotiv EEG channels
and `ECG1`, `ECG2`; GSR is omitted, matching the existing joined CSV format.
Self-assessment `[arousal, valence, dominance, ...]` is exported as
`valence, arousal, dominance` without rounding.

The first 640 samples are marked `baseline`, matching the existing converter;
remaining samples are `stimulus`. Use `--amigos-baseline-samples 0` if your input
already excludes the baseline. Trial IDs and sample indices are one-based;
sample indices restart within each segment. Empty trials are skipped with a
warning; malformed nonempty trials stop conversion.

## DREAMER

Convert the original MATLAB file:

```bash
eegproc-to-csv --dataset dreamer --input /data/DREAMER.mat \
  --output /data/csv/dreamer_joined.csv.gz
```

Output has the same 14 EEG channels, `ECG1`, `ECG2`, and three ratings as AMIGOS.
EEG values are unchanged. ECG is resampled to the EEG time grid using the file's
`ECG_SamplingRate` and `EEG_SamplingRate`, assuming the streams start together
within each segment. Duration disagreements greater than one EEG sample cause
an error; a final missing ECG sample is blank. IDs/indices are one-based and
MATLAB `stimuli` segments become `stimulus`.

The existing three-file workflow is also supported:

```text
/data/dreamer_exports/
  dreamer_eeg.csv      # subject_id, trial_id, segment, sample_idx, AF3 ... AF4
  dreamer_ecg.csv      # subject_id, trial_id, segment, sample_idx, ECG1, ECG2
  dreamer_labels.csv   # subject_id, trial_id, valence, arousal, dominance
```

```bash
eegproc-to-csv --dataset dreamer --input /data/dreamer_exports \
  --output /data/csv/dreamer_joined.csv
```

Each source CSV may instead be `.csv.gz`. The join preserves EEG row order and
existing index values, matching the original `join_dreamer.py`: ECG joins on the
four sample keys and ratings on subject/trial. Missing ECG becomes blank with a
warning; missing trial labels cause an error. Exported CSVs must already use a
common time grid: this join does not resample them. If both exports and
`DREAMER.mat` are present, the CSV exports take precedence; select the MAT file
explicitly to use the native reader.

## EEGEmotions-27

```bash
eegproc-to-csv --dataset eegemotions --input /data/EEGEmotions-27 \
  --output /data/csv/eegemotions_labeled.csv.gz
```

This combines the existing extraction, numeric subject/trial sorting, and label
attachment workflow without loading the full joined CSV. It retains:

```text
subject_id, trial_id, segment, sample_idx,
AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4,
age, gender, nation, source_file, source_file_label,
emo_label_cowen_27, emo_label_ekman_6
```

Files such as `12_5.0.txt` supply participant 12 and emotion/trial 5. Filename
IDs are preserved exactly, including zero if it exists in a downloaded release;
no automatic shift is applied. Samples start at zero within each file and
`segment` is `eeg_raw`. Demographic codes are retained as supplied.

`training/eeg_features_extracted.csv` supplies the secondary emotion system.
Use `--labels /path/to/labels.csv` to select another table containing
`ParticipantID`, `Emo_Label_Cowen(27)`, and `Emo_Label_Ekman(6)`. Without that table,
the filename label remains available and the secondary label is blank. Repeated
participant/emotion mappings use the mode (lowest label on a tie), matching the
original script; conflicting mappings emit a warning.

For a release whose filenames encode Ekman labels, add
`--source-label-kind ekman`. The default is `cowen`; choose according to your
release's metadata rather than inferring from the numeric range. These emotion
IDs are not trial-level valence/arousal self-reports.

## Cowen 27 emotion mapping

```bash
eegproc-to-csv --dataset cowen27 --input /data/Cowen27 \
  --output /data/csv/cowen_27_valence_arousal.csv
```

Accepts the `Cowen27` root with `data/features/amt/mean_score_concat`, an extracted
`features` folder, or `mean_score_concat` itself. Matching category/dimension MAT
files must contain a unique numeric vector of lengths 34 and 14, respectively.

The output preserves the previous mapping script: 27 emotion rows with weighted
valence/arousal means and standard deviations, rating weights, video counts, and
quadrants. It uses dimension indices 13 (valence) and 1 (arousal), zero-based,
and excludes seven categories outside the final 27. Quadrants use the medians of
all video ratings. **This is a derived mapping, not official participant-level
EEGEmotions ratings or official high/low thresholds.** No videos or fMRI data are
needed for this conversion.

## DEAP

```bash
eegproc-to-csv --dataset deap --input /data/DEAP/data_preprocessed_python \
  --output /data/csv/deap.csv.gz
```

Exports the first 32 EEG channels and `valence, arousal, dominance, liking`.
The 8064-sample format retains its first 384 samples as `baseline` and remaining
7680 as `stimulus`; the already-trimmed 7680-sample format contains only stimulus.
Trial and segment sample indices start at one. Other sample counts are rejected
rather than guessing where the stimulus begins. Peripheral channels are omitted.
DEAP `.dat` files use Python pickle: open only trusted official dataset files.

## Use the result

Read manageable outputs with pandas, or stream large files:

```python
import pandas as pd
from eegproc.data.to_csv import convert_dataset

rows = convert_dataset("amigos", "/data/AMIGOS", "/data/csv/amigos_joined.csv.gz")
for frame in pd.read_csv("/data/csv/amigos_joined.csv.gz", chunksize=100_000):
    stimulus = frame.loc[frame["segment"] == "stimulus"]
    # Apply your own filtering, feature extraction, or trial-safe windowing.
```

To cross-validate directly from a converted table, name the EEG columns
explicitly. Otherwise every other numeric column (the remaining ratings, ECG,
`sample_idx`, and for EEGEmotions the demographics) becomes a model input, and
predicting valence would silently train on arousal and dominance too:

```python
import pandas as pd
from eegproc.deep_learning.cross_validation import cross_validate_dataframe

EEG_CHANNELS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")

df = pd.read_csv("/data/csv/dreamer_joined.csv.gz")
df = df[df["segment"] == "stimulus"]
df["label"] = (df["valence"] >= 3).astype(int)

results = cross_validate_dataframe(
    df, build_model, strategy="loso",          # build_model: see the README
    kind="signal", fs=128, window_sec=1.0,
    subject_columns=("subject_id",), trial_columns=("trial_id",),
    time_column="sample_idx", feature_columns=EEG_CHANNELS,
    label_column="label",
)
```

For AMIGOS, also drop the `baseline` segment as above; the sample index
restarts within each segment.

The DREAMER, AMIGOS, and EEGEmotions column names match the existing
`eegproc.deep_learning.prepare_datasets` readers. For that older NumPy preparation
command, write **uncompressed** `dreamer_joined.csv`, `amigos_joined.csv`, or
`eegemotions_labeled.csv` (and the Cowen mapping where needed), using the filenames
it expects. Its DEAP reader continues to consume original `.dat` inputs.
Sampling rates are not embedded in these legacy CSV schemas; retain the dataset
metadata and pass the correct rate to downstream processing.
