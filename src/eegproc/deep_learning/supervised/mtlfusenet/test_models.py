from eegproc.deep_learning.supervised.mtlfusenet.preprocessing import (load_dreamer_csv, get_trial_eeg, compute_de_features,
                            build_all_adjacency_matrices, eeg_trial_to_spatial_tensor,
                            window_spatial_tensor)
from eegproc.deep_learning.supervised.mtlfusenet.models import build_vae

data = load_dreamer_csv("data/dreamer_joined.csv")
subject_id = data['subject_id'].iloc[0]
trial_id = data['trial_id'].iloc[0]

eeg_df = get_trial_eeg(data, subject_id, trial_id)
filtered, de_features = compute_de_features(eeg_df)
adj_matrices = build_all_adjacency_matrices(filtered)

X_ST = eeg_trial_to_spatial_tensor(eeg_df)
X_ST_windowed = window_spatial_tensor(X_ST, window_samples=128)

X_ST_normalized = (X_ST_windowed - X_ST_windowed.min()) / (X_ST_windowed.max() - X_ST_windowed.min())

vae, encoder, decoder = build_vae()
reconstruction = vae(X_ST_normalized[:1])

print(de_features.shape)
print(adj_matrices['theta'].shape)
print(X_ST_windowed.shape)
print(reconstruction.shape)