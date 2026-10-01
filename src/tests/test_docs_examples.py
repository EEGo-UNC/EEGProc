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
        if "dreamer" in str(path):
            return _dreamer_like_table()
        if Path(path).name == "features.csv":
            return _feature_table()
        return _channel_table()

    monkeypatch.setattr(pd, "read_csv", read_csv)


@pytest.fixture
def fast_cross_validation(monkeypatch):
    pytest.importorskip("tensorflow", reason="requires eegproc[deep-learning]")
    import eegproc.deep_learning.cross_validation as cross_validation

    original = cross_validation.cross_validate_dataframe

    def one_epoch(*args, **kwargs):
        for key, value in {"n_epochs": 1, "batch_size": 8, "verbose": 0, "n_jobs": 1,
                           "log_predictions": False}.items():
            kwargs.setdefault(key, value)
        kwargs["n_epochs"] = 1
        if kwargs.get("strategy", "loso") != "fixed_loso":
            kwargs.setdefault("early_stopping_patience", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(cross_validation, "cross_validate_dataframe", one_epoch)


def _feature_table() -> pd.DataFrame:
    """Four feature rows per trial; extra ratings must not become model inputs."""
    rng = np.random.default_rng(2)
    return pd.concat([
        pd.DataFrame({
            "subject": subject, "trial": trial, "label": label,
            "AF3_alpha": rng.uniform(size=4), "F7_alpha": rng.uniform(size=4),
            "arousal": 5.0,
        })
        for subject in ("P01", "P07", "P12")
        for trial, label in (("a", 0), ("b", 1))
    ], ignore_index=True)


def test_readme_featurization_and_plotting(fake_read_csv, tmp_path, monkeypatch):
    namespace = {}
    exec(_block_containing(ROOT / "README.md", "shannons_entropy(psd)"), namespace)

    assert list(namespace["entropy"].columns) == [f"{c}_entropy" for c in ["AF3", "F7", "F3", "FC5"]]

    # Execute the plotting recipe on those same features and verify both exports.
    monkeypatch.chdir(tmp_path)
    exec(_block_containing(ROOT / "README.md", "save_path=\"bandpowers.png\""), namespace)
    assert (tmp_path / "bandpowers.png").stat().st_size > 0
    assert (tmp_path / "entropy.png").stat().st_size > 0
    assert len(namespace["axes"]) == 4
    assert len(namespace["entropy_axes"]) == 2
    np.testing.assert_array_equal(namespace["axes"][0].lines[0].get_xdata(), [0, 2, 4])
    import matplotlib.pyplot as plt
    plt.close(namespace["fig"])
    plt.close(namespace["entropy_fig"])


def test_readme_cross_validation(fake_read_csv, fast_cross_validation):
    namespace = {}
    exec(_block_containing(ROOT / "README.md", "def build_model"), namespace)
    results = namespace["results"]
    assert {row["subject_id"] for row in results["user_metrics"]} == {"P01", "P07", "P12"}
    assert results["cv_strategy"] == "fixed_loso_no_validation"
    # A model built for this table receives just the explicitly selected features.
    model = namespace["build_model"](np.zeros((2, 4, 2), dtype="float32"))
    assert model.input_shape == (None, 4, 2)
    assert model.output_shape == (None, 2)


def test_dataset_guide_cross_validation(fake_read_csv, fast_cross_validation):
    namespace = {}
    exec(_block_containing(ROOT / "README.md", "def build_model"), namespace)
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
