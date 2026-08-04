import sys
from pathlib import Path

try:
    from .preprocessing import load_dreamer_csv
except ImportError:
    try:
        from eegproc.deep_learning.supervised.mtlfusenet.preprocessing import load_dreamer_csv
    except ImportError:
        CURRENT_DIR = Path(__file__).resolve().parent
        if str(CURRENT_DIR) not in sys.path:
            sys.path.insert(0, str(CURRENT_DIR))
        from preprocessing import load_dreamer_csv


data = load_dreamer_csv("data/dreamer_joined.csv")
subject_trial_pairs = data[['subject_id', 'trial_id']].drop_duplicates().values
print(len(subject_trial_pairs))