# Model explanations

`model_agnostic` provides differentiable counterfactual optimization through an
adapter interface. The adapter owns model loading, the optimization space,
classification, reconstruction, and optional constraints.

Install `eegproc[deep-learning]`, then see
[the API and command-line examples](model_agnostic/README.md).

For the paired external time-series comparator, see
[CoMTE versus CFO](comte/README.md). Both delivered output signals pass through
the same frozen encoder–classifier. The study retains failures and reports
independent typicality, physiology, joint success, displacement, electrode
sparsity, runtime and query counts by held-out subject.
