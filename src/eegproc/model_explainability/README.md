# Counterfactual explainability

The counterfactual explainability code is organized by workflow:

| Package | Purpose | Main command |
| --- | --- | --- |
| [`counterfactuals/`](counterfactuals/README.md) | SIC-specific latent counterfactual optimization, metrics, and plots. | `python -m eegproc.model_explainability.counterfactuals.runner` |
| [`model_agnostic/`](model_agnostic/README.md) | Adapter-based counterfactuals for SIC and other differentiable models. | `python -m eegproc.model_explainability.model_agnostic.runner` |
| [`typicality/`](typicality/README.md) | Paired base/typicality studies, calibration, physiology, reports, and subject probes. | `python -m eegproc.model_explainability.typicality.runner` |

The model-agnostic and typicality workflows reuse selected objective and
plotting utilities from `counterfactuals`; their runners and result formats
remain separate.
