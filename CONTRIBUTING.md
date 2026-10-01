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
launchers, and result archives outside this library. Preserve reusable layers,
model-building interfaces, and array/data contracts.

The cross-validation package retains v2's module structure. Keep dependencies
pointing from orchestration modules toward shared helpers; importing the public
API must not require an external research repository. `cross_val.py` is only a
compatibility wrapper.

## Tests

```bash
pytest                                               # full suite (needs the deep-learning extra)
pytest --ignore=src/tests/test_cross_validation.py   # what CI runs on the base install
```

Tests that need TensorFlow skip themselves on a base install. CI runs the base
suite on Python 3.10–3.13 and the full suite on 3.10, 3.12 and 3.13.

Before submitting a change, run the tests, build a wheel, and check that it
installs and imports. Describe any API changes in the review. Keep documentation
examples runnable.
