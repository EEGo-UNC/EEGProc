![Coverage badge](coverage.svg)

# EEGProc: EEG Preprocessing and Featurization Library

EEGProc is a fully vectorized library designed for preprocessing and extracting features from EEG (Electroencephalogram) data. This library is optimized for performance and ease of use, making it suitable for researchers and developers working in the field of neuroscience, biomedical engineering, and machine learning.

## Documentation

[Click here to read the documentation](https://eego-unc.github.io/EEGProc/)

## Features

- **Preprocessing**: Includes functions for filtering, artifact removal, and normalization of EEG signals.
- **Featurization**: Extracts meaningful features from EEG data, such as power spectral density, band power, and more.
- **Vectorized Operations**: Fully vectorized implementation ensures high performance and scalability for working with pandas dataframes.
- **Ease of Integration**: Designed to integrate seamlessly with existing Python workflows.

## Installation

To install EEGProc, you can use pip:

```bash
pip install eegproc
```

Alternatively, you can clone the repository and install the required dependencies manually:

```bash
# Clone the repository
git clone https://github.com/VitorInserra/EEGProc.git

# Navigate to the project directory
cd EEGProc

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Preprocessing EEG Data

```python
import pandas as pd
from eegproc import bandpass_filter

# Example: Preprocess raw EEG data
data: pd.DataFrame = ...  # Load your raw EEG data as a dataframe
mask = (data['patient_index'] == 0) & (data['video_index'] == 17)
eeg_df = data.loc[mask, :]
bandpass_filtered_data: pd.DataFrame = bandpass_filter(eeg_df)
```

### Extracting Features

```python
from eegproc import psd

# Example: get Power Spectral Density from a bandpass filtered dataframe
psd_data: pd.DataFrame = psd(bandpass_filitered_data)
```

## File Structure

- `src/eegproc/`
  - `preprocessing.py`: Contains preprocessing functions.
  - `featurization.py`: Contains feature extraction functions.
- `requirements.txt`: Lists the dependencies required for the project.

## Contributing

Contributions are welcome! If you have ideas for new features or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the GPLv2 License. See the `LICENSE` file for details.
