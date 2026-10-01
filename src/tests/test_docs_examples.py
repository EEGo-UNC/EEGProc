"""Run the documentation's Python examples verbatim on synthetic data.

Each example is read from the Markdown file and executed as written. Only
``pandas.read_csv`` (which would need the user's files) is replaced with
synthetic tables, and cross-validation trains for one epoch so the suite stays
fast. If an example stops matching the code, these tests fail.
"""

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
EEG_CHANNELS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")
FS = 128


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


def _block_containing(path: Path, text: str) -> str:
    matches = [block for block in _python_blocks(path) if text in block]
    assert len(matches) == 1, f"expected one python block in {path.name} containing {text!r}"
    return matches[0]


def _channel_table(seconds=8, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((FS * seconds, 4)), columns=["AF3", "F7", "F3", "FC5"])


def _dreamer_like_table() -> pd.DataFrame:
    """The column layout eegproc-to-csv writes for DREAMER, with a short recording."""
    rng = np.random.default_rng(1)
    parts = []
    for subject in (1, 2, 3):
        for trial in (1, 2):
            n = FS * 4
            data = {
                "subject_id": subject, "trial_id": trial,
                "segment": ["baseline"] * FS + ["stimulus"] * (n - FS),
                "sample_idx": np.r_[np.arange(1, FS + 1), np.arange(1, n - FS + 1)],
            }
            data.update({c: rng.standard_normal(n) for c in EEG_CHANNELS})
            data.update({"ECG1": rng.standard_normal(n), "ECG2": rng.standard_normal(n),
                         "valence": float(1 + 2 * ((subject + trial) % 2)),
                         "arousal": 3.0, "dominance": 2.0})
            parts.append(pd.DataFrame(data))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def fake_read_csv(monkeypatch):
    def read_csv(path, *args, **kwargs):
        return _dreamer_like_table() if "dreamer" in str(path) else _channel_table()

    monkeypatch.setattr(pd, "read_csv", read_csv)


@pytest.fixture
def fast_cross_validation(monkeypatch):
    pytest.importorskip("tensorflow", reason="requires eegproc[deep-learning]")
    import eegproc.deep_learning.cross_validation as cross_validation

    original = cross_validation.cross_validate_dataframe

    def one_epoch(*args, **kwargs):
        for key, value in {"n_epochs": 1, "batch_size": 8, "verbose": 0, "n_jobs": 1,
                           "log_predictions": False, "early_stopping_patience": None}.items():
            kwargs.setdefault(key, value)
        return original(*args, **kwargs)

    monkeypatch.setattr(cross_validation, "cross_validate_dataframe", one_epoch)


def _subject_trial_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(2)
    blocks = []
    for subject in ("P01", "P07", "P12"):
        for trial in ("a", "b"):
            blocks.append(pd.DataFrame({
                "subject": subject, "trial": trial,
                "AF3": rng.standard_normal(FS * 8), "F7": rng.standard_normal(FS * 8),
            }))
    raw = pd.concat(blocks, ignore_index=True)
    labels = raw.drop_duplicates(["subject", "trial"])[["subject", "trial"]].assign(
        label=[0, 1, 1, 0, 0, 1]
    )
    return raw, labels


def test_readme_featurization(fake_read_csv):
    namespace = {}
    exec(_block_containing(ROOT / "README.md", "shannons_entropy(psd)"), namespace)

    assert list(namespace["entropy"].columns) == [f"{c}_entropy" for c in ["AF3", "F7", "F3", "FC5"]]


def test_readme_cross_validation_and_converted_dataset(fake_read_csv, fast_cross_validation):
    raw, labels = _subject_trial_raw()
    namespace = {"raw": raw, "labels": labels}

    exec(_block_containing(ROOT / "README.md", "def build_model"), namespace)
    assert {row["subject_id"] for row in namespace["results"]["user_metrics"]} == {"P01", "P07", "P12"}

    exec(_block_containing(ROOT / "README.md", "EEG_CHANNELS = "), namespace)
    assert {row["subject_id"] for row in namespace["results"]["user_metrics"]} == {1, 2, 3}


def test_dataset_guide_cross_validation(fake_read_csv, fast_cross_validation):
    raw, labels = _subject_trial_raw()
    namespace = {"raw": raw, "labels": labels}
    exec(_block_containing(ROOT / "README.md", "def build_model"), namespace)   # defines build_model

    exec(_block_containing(ROOT / "docs" / "source" / "datasets.md", "cross_validate_dataframe"), namespace)

    assert len(namespace["results"]["user_metrics"]) == 3


def test_getting_started_quick_start(fake_read_csv):
    namespace = {}
    for block in _python_blocks(ROOT / "docs" / "source" / "getting-started.md"):
        exec(block, namespace)

    assert "AF3_entropy" in namespace["entropy_df"].columns
    assert "AF3_alpha_activity" in namespace["hjorth_df"].columns


def test_counterfactual_readme_example():
    tf = pytest.importorskip("tensorflow", reason="requires eegproc[deep-learning]")
    tf.keras.utils.set_random_seed(0)
    inputs = tf.keras.Input((8, 6))
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2)(tf.keras.layers.Flatten()(inputs)))
    trial = np.random.default_rng(0).standard_normal((8, 6)).astype("float32")
    namespace = {"model": model, "trial": trial}

    readme = ROOT / "src" / "eegproc" / "model_explainability" / "model_agnostic" / "README.md"
    exec(_block_containing(readme, "KerasInputAdapter("), namespace)

    assert namespace["result"]["summary"]["counterfactual"]["target_probability"] >= 0.8
