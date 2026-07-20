from preprocessing import load_dreamer_csv, get_trial_eeg, compute_de_features, build_all_adjacency_matrices

data = load_dreamer_csv("data/dreamer_joined.csv")  # adjust path to wherever you put the CSV

subject_id = data['subject_id'].iloc[0]
trial_id = data['trial_id'].iloc[0]

eeg_df = get_trial_eeg(data, subject_id, trial_id)
filtered, de_features = compute_de_features(eeg_df)
adj_matrices = build_all_adjacency_matrices(filtered)

print(de_features.shape)
print(adj_matrices['theta'].shape)

from preprocessing import eeg_trial_to_spatial_tensor

X_ST = eeg_trial_to_spatial_tensor(eeg_df)
print(X_ST.shape)