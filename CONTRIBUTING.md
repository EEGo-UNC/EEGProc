# Contributing to EEGProc

## Setup

```bash
git clone https://github.com/EEGo-UNC/EEGProc.git
cd EEGProc
python -m venv .venv
source .venv/bin/activate
pip install -e ".[deep-learning,dev]"
python -m build
```

The base install contains preprocessing, features, the DataFrame data layer, and
plotting. TensorFlow and scikit-learn belong to the `deep-learning` extra.

Keep research models, dataset-specific counterfactual adapters, experiment
launchers, result archives, and test files outside this library branch. Preserve
reusable layers, model-building interfaces, and array/data contracts.

The cross-validation package retains v2's module structure. Keep dependencies
pointing from orchestration modules toward shared helpers; importing the public
API must not require an external research repository. `cross_val.py` is only a
compatibility wrapper.

Before submitting a change, build a wheel, verify its installed imports, and run
relevant behavior checks in an external validation environment. Describe those
checks and any API changes in the review. Keep documentation examples runnable.
