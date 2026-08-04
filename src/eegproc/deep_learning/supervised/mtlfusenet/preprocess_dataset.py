import os
import pickle
import sys
from pathlib import Path

try:
    from .preprocessing import (
        load_dreamer_csv,
        process_trial,
        EEG_CHANNELS,
        DREAMER_BANDS,
        ELECTRODE_POSITIONS,
    )
except ImportError:
    try:
        from eegproc.deep_learning.supervised.mtlfusenet.preprocessing import (
            load_dreamer_csv,
            process_trial,
            EEG_CHANNELS,
            DREAMER_BANDS,
            ELECTRODE_POSITIONS,
        )
    except ImportError:
        CURRENT_DIR = Path(__file__).resolve().parent
        if str(CURRENT_DIR) not in sys.path:
            sys.path.insert(0, str(CURRENT_DIR))
        from preprocessing import (
            load_dreamer_csv,
            process_trial,
            EEG_CHANNELS,
            DREAMER_BANDS,
            ELECTRODE_POSITIONS,
        )

data = load_dreamer_csv("data/dreamer_joined.csv")

os.makedirs('processed_trials', exist_ok=True)

subject_trial_pairs = data[['subject_id', 'trial_id']].drop_duplicates().values

errors = []
for subj, trial in subject_trial_pairs:
    try:
        result = process_trial(data, subj, trial, EEG_CHANNELS, DREAMER_BANDS, ELECTRODE_POSITIONS)
        filename = f'processed_trials/subj{subj}_trial{trial}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        errors.append((subj, trial, str(e)))

print(f"Done. Errors: {len(errors)}")
for e in errors[:5]:
    print(e)