from preprocessing import load_dreamer_csv

data = load_dreamer_csv("data/dreamer_joined.csv")
subject_trial_pairs = data[['subject_id', 'trial_id']].drop_duplicates().values
print(len(subject_trial_pairs))