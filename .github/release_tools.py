"""Release helpers for .github/workflows/release.yml.

    python .github/release_tools.py plan          # decide what to publish; writes $GITHUB_OUTPUT
    python .github/release_tools.py notes 2.0.0   # print the CHANGELOG section for a version
    python .github/release_tools.py smoke 2.0.0   # exercise an installed eegproc

The logic lives here rather than in workflow YAML so it can be tested
(src/tests/test_release_tools.py) and run locally.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INDEXES = {
    "pypi": {
        "environment": "pypi",
        "upload_url": "https://upload.pypi.org/legacy/",
        "simple_url": "https://pypi.org/simple/",
        "json_url": "https://pypi.org/pypi/eegproc/json",
        "project_url": "https://pypi.org/project/eegproc/{version}/",
    },
    "testpypi": {
        "environment": "testpypi",
        "upload_url": "https://test.pypi.org/legacy/",
        "simple_url": "https://test.pypi.org/simple/",
        "json_url": "https://test.pypi.org/pypi/eegproc/json",
        "project_url": "https://test.pypi.org/project/eegproc/{version}/",
    },
}


class ReleaseError(RuntimeError):
    """A guard failed: nothing may be published."""


# --------------------------------------------------------------------------
# Repository facts
# --------------------------------------------------------------------------

def declared_versions(root: Path = ROOT) -> dict[str, str]:
    """Every place the version is written down."""
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    citation = re.search(
        r"^version:\s*(\S+)\s*$", (root / "CITATION.cff").read_text(encoding="utf-8"), re.M
    )
    return {
        "pyproject.toml [project].version": pyproject["project"]["version"],
        "pyproject.toml [tool.bumpver].current_version": pyproject["tool"]["bumpver"]["current_version"],
        "CITATION.cff version": citation.group(1) if citation else "<missing>",
    }


def changelog_heading(changelog: str, version: str) -> str | None:
    """Return the ``## <version> ...`` heading line, or None."""
    pattern = rf"^##\s+\[?{re.escape(version)}\]?(?:\s.*)?$"
    match = re.search(pattern, changelog, re.M)
    return match.group(0) if match else None


def is_dated(heading: str) -> bool:
    return re.search(r"\d{4}-\d{2}-\d{2}\s*$", heading) is not None


def release_notes(changelog: str, version: str) -> str:
    """The body of a version's CHANGELOG section, without its heading."""
    heading = changelog_heading(changelog, version)
    if heading is None:
        raise ReleaseError(f"CHANGELOG.md has no '## {version}' section.")
    start = changelog.index(heading) + len(heading)
    following = re.search(r"^##\s", changelog[start:], re.M)
    body = changelog[start: start + following.start()] if following else changelog[start:]
    return body.strip() + "\n"


def pypi_releases(json_url: str) -> set[str] | None:
    """Published versions, or None if the project does not exist on that index."""
    try:
        with urllib.request.urlopen(json_url, timeout=30) as response:
            return set(json.load(response)["releases"])
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def remote_tag_commit(tag: str) -> str | None:
    """The commit a tag on origin points at, or None if it does not exist."""
    output = subprocess.run(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout.split()
    if not output:
        return None
    pairs = dict(zip(output[1::2], output[0::2]))
    return pairs.get(f"refs/tags/{tag}^{{}}", pairs.get(f"refs/tags/{tag}"))


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    target: str            # "pypi" or "testpypi"
    version: str           # the version that is built and uploaded
    publish: bool          # upload to the index
    release: str           # "publish", "draft", or "skip" (GitHub release)
    notes_version: str     # CHANGELOG section used for the release notes

    def outputs(self) -> dict[str, str]:
        index = INDEXES[self.target]
        return {
            "target": self.target,
            "version": self.version,
            "publish": str(self.publish).lower(),
            "release": self.release,
            "notes_version": self.notes_version,
            "environment": index["environment"],
            "upload_url": index["upload_url"],
            "simple_url": index["simple_url"],
            "project_url": index["project_url"].format(version=self.version),
        }


def decide(
    *,
    ref: str,
    sha: str,
    run_number: int,
    run_attempt: int,
    versions: dict[str, str],
    changelog: str,
    published: set[str] | None,
    tag_commit: str | None,
    release_exists: bool,
) -> Plan:
    """Decide what a release run does, or raise ReleaseError if it must not publish.

    ``published`` is the set of versions on the target index (None if the
    project is absent); ``tag_commit`` is where ``v<version>`` points on origin.
    """
    if len(set(versions.values())) != 1:
        listed = ", ".join(f"{place} = {value}" for place, value in versions.items())
        raise ReleaseError(f"Version mismatch: {listed}. Run bumpver instead of editing by hand.")
    base = next(iter(versions.values()))

    heading = changelog_heading(changelog, base)
    if heading is None:
        raise ReleaseError(f"CHANGELOG.md has no '## {base}' section.")

    if ref.startswith("refs/tags/testpypi-"):
        # Every rehearsal uploads a new, never-reused development version.
        version = f"{base}.dev{run_number * 100 + run_attempt}"
        return Plan("testpypi", version, publish=True, release="draft", notes_version=base)

    if ref != "refs/heads/main":
        raise ReleaseError(f"Releases run from main or testpypi-* tags, not {ref}.")

    already_published = published is not None and base in published
    if not already_published:
        if not is_dated(heading):
            raise ReleaseError(
                f"Date the CHANGELOG heading before releasing: '{heading}' -> "
                f"'## {base} — YYYY-MM-DD'."
            )
        if published:
            from packaging.version import Version

            newest = max(published, key=Version)
            if Version(base) <= Version(newest):
                raise ReleaseError(f"{base} is not newer than {newest}, the newest release on PyPI.")
    if tag_commit is not None and tag_commit != sha:
        raise ReleaseError(f"Tag v{base} already exists on {tag_commit}, not on {sha}.")

    return Plan(
        "pypi",
        base,
        publish=not already_published,
        release="skip" if release_exists else "publish",
        notes_version=base,
    )


# --------------------------------------------------------------------------
# Smoke test of an installed package
# --------------------------------------------------------------------------

def smoke(version: str) -> None:
    """Exercise an installed eegproc the way the README does. Must not import the source tree."""
    import shutil

    import numpy as np
    import pandas as pd

    import eegproc
    from eegproc import FREQUENCY_BANDS, bandpass_filter, psd_bandpowers, shannons_entropy

    assert "site-packages" in eegproc.__file__, f"imported from {eegproc.__file__}, not an install"
    assert eegproc.__version__ == version, f"installed {eegproc.__version__}, expected {version}"

    rng = np.random.default_rng(0)
    raw = pd.DataFrame(rng.standard_normal((128 * 8, 2)), columns=["AF3", "F7"])
    entropy = shannons_entropy(psd_bandpowers(bandpass_filter(raw, 128, bands=FREQUENCY_BANDS), 128))
    assert list(entropy.columns) == ["AF3_entropy", "F7_entropy"]

    script = shutil.which("eegproc-to-csv", path=str(Path(sys.executable).parent))
    assert script, "eegproc-to-csv was not installed"
    subprocess.run([script, "--help"], check=True, capture_output=True)

    import tensorflow as tf

    from eegproc.deep_learning.cross_validation import cross_validate_dataframe
    from eegproc.model_explainability.model_agnostic import (
        KerasInputAdapter,
        ModelAgnosticCounterfactualOptimizer,
    )

    table = pd.DataFrame({
        "subject": np.repeat(["P01", "P07", "P12"], 8),
        "trial": np.tile(np.repeat(["a", "b"], 4), 3),
        "f1": rng.standard_normal(24),
        "f2": rng.standard_normal(24),
    })
    table["label"] = (table["trial"] == "b").astype(int)

    def build_model(training_features, **_):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(training_features.shape[1:]),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    results = cross_validate_dataframe(
        table, build_model, strategy="loso", fs=1.0, label_column="label",
        n_epochs=1, batch_size=8, verbose=0, n_jobs=1,
        log_predictions=False, early_stopping_patience=None,
    )
    assert {row["subject_id"] for row in results["user_metrics"]} == {"P01", "P07", "P12"}

    inputs = tf.keras.Input((8, 6))
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2)(tf.keras.layers.Flatten()(inputs)))
    trial = rng.standard_normal((8, 6)).astype("float32")
    result = ModelAgnosticCounterfactualOptimizer(
        KerasInputAdapter(model, output_kind="logits"), target_probability=0.8,
    ).optimize(trial[None, ...], target_class=1)
    assert result["summary"]["counterfactual"]["target_probability"] >= 0.8

    print(f"eegproc {version}: smoke test passed")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _plan_from_environment() -> Plan:
    ref = os.environ["GITHUB_REF"]
    sha = os.environ["GITHUB_SHA"]
    versions = declared_versions()
    base = next(iter(versions.values()))
    target = "testpypi" if ref.startswith("refs/tags/testpypi-") else "pypi"
    on_main = target == "pypi"
    release_exists = False
    if on_main:
        release_exists = subprocess.run(
            ["gh", "release", "view", f"v{base}"], capture_output=True, cwd=ROOT
        ).returncode == 0
    return decide(
        ref=ref,
        sha=sha,
        run_number=int(os.environ["GITHUB_RUN_NUMBER"]),
        run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
        versions=versions,
        changelog=(ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        published=pypi_releases(INDEXES[target]["json_url"]) if on_main else None,
        tag_commit=remote_tag_commit(f"v{base}") if on_main else None,
        release_exists=release_exists,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    notes = commands.add_parser("notes")
    notes.add_argument("version")
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("version")
    args = parser.parse_args(argv)

    if args.command == "notes":
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        sys.stdout.write(release_notes(changelog, args.version))
        return 0
    if args.command == "smoke":
        smoke(args.version)
        return 0

    try:
        plan = _plan_from_environment()
    except ReleaseError as error:
        print(f"::error::{error}")
        return 1
    outputs = plan.outputs()
    for key, value in outputs.items():
        print(f"{key}={value}")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.writelines(f"{key}={value}\n" for key, value in outputs.items())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
