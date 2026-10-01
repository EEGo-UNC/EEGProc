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
examples runnable; `src/tests/test_docs_examples.py` executes them.

## Releasing

Releases are published by `.github/workflows/release.yml` when a commit that
changes the version reaches `main`:

1. In a pull request, bump the version and date the changelog:

   ```bash
   bumpver update --patch        # or --minor / --major; updates pyproject.toml and CITATION.cff
   ```

   and rename the `## X.Y.Z — unreleased` heading in `CHANGELOG.md` to
   `## X.Y.Z — YYYY-MM-DD`.
2. Merge the pull request. The workflow runs the tests, builds the wheel and
   sdist, publishes them to PyPI through trusted publishing, creates the
   `vX.Y.Z` tag and GitHub release with that changelog section, and installs
   the published wheel on a fresh runner to check it. The docs workflow
   redeploys the documentation.

Pushes to `main` that keep the version only run the tests and the build. The
workflow refuses to publish if the version is not newer than PyPI's, the
changelog heading is undated, or `pyproject.toml`, bumpver and `CITATION.cff`
disagree.

To rehearse without touching PyPI, push a tag named `testpypi-N` (one at a
time). The same chain runs against TestPyPI with a unique `X.Y.Z.devN` version
and a draft GitHub release that is deleted afterwards.
