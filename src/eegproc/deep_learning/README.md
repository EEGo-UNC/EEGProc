# Deep learning components

Install `eegproc[deep-learning]` to use the TensorFlow components.

| Module | Purpose |
| --- | --- |
| [`cross_validation`](cross_validation/) | LOSO, fixed LOSO, nested evaluation, subject calibration, and the DataFrame interface. |
| [`cross_val.py`](cross_val.py) | Compatibility import for the public cross-validation API. |
| [`domain_generalization`](domain_generalization/) | Alternating subject groups and meta-learning strategies. |
| [`training_outputs.py`](training_outputs.py) | Training callbacks, metrics, and diagnostics. |
| [`supervised`](supervised/) | RNN classifier builders, dense and variational classifier heads, and contrastive loss. |
| [`unsupervised`](unsupervised/) | CNN/GNN encoders and decoders, including the DREAMER/MTLFuseNet spatiotemporal 3D CNN, graph layers, and autoencoder losses. |
| [`prepare_datasets.py`](prepare_datasets.py) | Converters for supported public EEG datasets. |

Supply your own model builder to the cross-validation functions. Model definitions
and experiment configurations live in the application that uses the library.

## DREAMER 3D CNN

`unsupervised.Convolutions.CNN3D.MTLFuseNet3DCNNEncoder` accepts windows shaped
`(batch, timesteps, 42)` from `prepare_dreamer`: 14 channels with the theta,
alpha, and beta waveforms adjacent within each channel. It restores the Emotiv
EPOC electrodes to a 9 x 9 scalp grid, applies 3D convolutions over time and
both scalp axes, and returns a temporal embedding suitable for a classifier or
fusion with `GCNMTLEncoder`. Use training-fold data only when estimating the
mutual-information adjacency for the graph branch.
