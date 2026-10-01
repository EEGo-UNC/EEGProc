"""The release workflow's decisions (.github/release_tools.py)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("tomllib", reason="release tooling runs on Python 3.11+")

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("release_tools", ROOT / ".github" / "release_tools.py")
release_tools = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = release_tools   # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(release_tools)

ReleaseError = release_tools.ReleaseError

CHANGELOG = """# Changelog

## 2.1.0 — 2026-10-02

### Fixed

- Something.

## 2.0.0 — 2026-09-30

- Older notes.
"""
SHA = "a" * 40


def _decide(**overrides):
    arguments = dict(
        ref="refs/heads/main",
        sha=SHA,
        run_number=7,
        run_attempt=1,
        versions={"pyproject": "2.1.0", "bumpver": "2.1.0", "citation": "2.1.0"},
        changelog=CHANGELOG,
        published={"1.0.0", "2.0.0"},
        tag_commit=None,
        release_exists=False,
    )
    arguments.update(overrides)
    return release_tools.decide(**arguments)


def test_repository_versions_agree():
    """pyproject, bumpver and CITATION.cff must name the same version."""
    versions = release_tools.declared_versions(ROOT)
    assert len(set(versions.values())) == 1, versions


def test_repository_changelog_has_a_section_for_the_current_version():
    version = next(iter(release_tools.declared_versions(ROOT).values()))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert release_tools.changelog_heading(changelog, version) is not None
    assert release_tools.release_notes(changelog, version).strip()


def test_new_version_on_main_is_published_and_released():
    plan = _decide()
    assert (plan.target, plan.version, plan.publish, plan.release) == ("pypi", "2.1.0", True, "publish")
    outputs = plan.outputs()
    assert outputs["environment"] == "pypi"
    assert outputs["upload_url"] == "https://upload.pypi.org/legacy/"


def test_unchanged_version_on_main_publishes_nothing():
    plan = _decide(published={"2.0.0", "2.1.0"}, release_exists=True, tag_commit=SHA)
    assert (plan.publish, plan.release) == (False, "skip")


def test_published_but_unreleased_version_only_creates_the_release():
    """Recovery after a run that uploaded but failed before the GitHub release."""
    plan = _decide(published={"2.1.0"}, release_exists=False)
    assert (plan.publish, plan.release) == (False, "publish")


def test_rehearsal_tag_builds_a_unique_development_version():
    plan = _decide(ref="refs/tags/testpypi-3", run_number=12, run_attempt=2,
                   changelog=CHANGELOG.replace("2.1.0 — 2026-10-02", "2.1.0 — unreleased"))
    assert (plan.target, plan.version, plan.publish, plan.release) == (
        "testpypi", "2.1.0.dev1202", True, "draft",
    )
    assert plan.outputs()["upload_url"] == "https://test.pypi.org/legacy/"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"versions": {"pyproject": "2.1.0", "bumpver": "2.0.0", "citation": "2.1.0"}}, "Version mismatch"),
        ({"changelog": "# Changelog\n"}, "no '## 2.1.0' section"),
        ({"changelog": CHANGELOG.replace("2.1.0 — 2026-10-02", "2.1.0 — unreleased")}, "Date the CHANGELOG"),
        ({"published": {"2.2.0"}}, "not newer than 2.2.0"),
        ({"tag_commit": "b" * 40}, "already exists"),
        ({"ref": "refs/heads/v2"}, "not refs/heads/v2"),
    ],
)
def test_guards_stop_a_bad_release(overrides, message):
    with pytest.raises(ReleaseError, match=message):
        _decide(**overrides)


def test_release_notes_are_the_versions_section_only():
    notes = release_tools.release_notes(CHANGELOG, "2.1.0")
    assert notes.startswith("### Fixed")
    assert "Older notes" not in notes
    assert release_tools.release_notes(CHANGELOG, "2.0.0").strip() == "- Older notes."
