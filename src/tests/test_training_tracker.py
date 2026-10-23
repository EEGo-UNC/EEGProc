"""Public tracker contract and integration checks using invented results."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from training_tracker import core
from training_tracker.demo import create_demo
from training_tracker.results import result_metric, subject_metrics

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def generic():
    return core.load_campaign(ROOT / "experiments/tracker_template/campaign.json")


def bundle(directory, campaign, config=None, result=None, run_id="example", status="completed"):
    config = config or campaign["defaults"]
    directory.mkdir(parents=True)
    core.write_json(directory / "config.json", config)
    core.write_json(directory / "manifest.json", {
        "campaign": campaign["campaign"], "run_id": run_id,
        "config_fingerprint": core.fingerprint(config), "stage": "full", "status": status,
    })
    if result is not None:
        core.write_json(directory / "result.json", result)


def test_generic_planning_and_template(generic, tmp_path):
    run = core.plan_run(generic, tmp_path / "runs", "regression")
    config = core.load_json(run / "config.json")
    assert config["target"] == "regression"
    assert config["epochs"] == 2
    script = (run / "submit.slurm").read_text(encoding="utf-8")
    assert "{{" not in script
    assert "dreamer" not in script.lower()
    assert "your_training_script.py" in script


def test_generic_can_skip_smoke_and_minimize(generic):
    generic.pop("smoke_overrides")
    stage, config, _ = core.suggest(generic, [], "classification")
    assert stage == "full"
    assert config["epochs"] == 50
    assert core.score({"mean": 0.1, "std": 0.0}, generic) > core.score({"mean": 0.8, "std": 0.0}, generic)


@pytest.mark.parametrize("std", [-1, float("nan"), float("inf")])
def test_bad_spread_rejected(std):
    with pytest.raises(ValueError):
        result_metric({"metrics": {"loss": {"mean": 0.2, "std": std}}}, "loss")


@pytest.mark.parametrize("index", [-1, 1, True, "0"])
def test_invalid_selected_config_rejected(index):
    with pytest.raises(ValueError):
        result_metric({"best_config_index": index, "config_results": [{}]}, "trial_accuracy")


def test_level_specific_subject_scores():
    result = {"user_metrics": [{"subject_id": "a", "evaluation_level": "window", "accuracy": 0.9,
                                "window_accuracy": 0.9, "trial_accuracy": 0.6}]}
    assert subject_metrics(result, "trial_accuracy")[0]["trial_accuracy"] == 0.6
    del result["user_metrics"][0]["trial_accuracy"]
    assert subject_metrics(result, "trial_accuracy") == []


def test_duplicate_subjects_require_explicit_aggregation():
    with pytest.raises(ValueError, match="Repeated subject"):
        subject_metrics({"metrics": {}, "subject_metrics": [{"subject_id": 1, "loss": 0.1}] * 2}, "loss")


def test_canonical_sync_status_and_report(generic, tmp_path):
    result = {"metrics": {"loss": {"mean": 0.24, "std": 0.03}},
              "subject_metrics": [{"subject_id": "person-a", "loss": 0.21}]}
    bundle(tmp_path / "download" / "example", generic, result=result)
    assert core.sync_bundles(tmp_path / "download", tmp_path / "runs", generic) == 1
    rows = core.scan_runs(tmp_path / "runs", generic)
    assert rows[0]["mean"] == 0.24
    assert rows[0]["status"] == "completed"
    assert "Classification" in (tmp_path / "runs/REPORT.md").read_text(encoding="utf-8")
    assert core.main(["--campaign", str(ROOT / "experiments/tracker_template/campaign.json"),
                      "--runs-dir", str(tmp_path / "runs"), "status"]) == 0


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "..", "a/b", "a\\b"])
def test_sync_rejects_unsafe_run_ids(generic, tmp_path, run_id):
    bundle(tmp_path / "download", generic, run_id=run_id)
    with pytest.raises(ValueError, match="Unsafe"):
        core.sync_bundles(tmp_path / "download", tmp_path / "runs", generic)


def test_tracking_imports_without_site_packages():
    script = f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import training_tracker.core; assert 'tensorflow' not in sys.modules; assert 'matplotlib' not in sys.modules"
    subprocess.run([sys.executable, "-S", "-c", script], check=True, capture_output=True)


def test_both_model_demos_plot_and_compare(tmp_path):
    from training_tracker.plotting import generate_plots
    campaigns = create_demo(tmp_path / "demo")
    campaign = core.load_campaign(campaigns[0])
    output = tmp_path / "figures"
    paths = generate_plots(campaigns[0].parent / "runs", campaign, output,
                           compare_campaigns=[campaigns[1]], chance=0.5)
    assert len(paths) == 10  # Four run summaries, four subject panels, two target comparisons.
    assert all(path.stat().st_size > 1000 for path in paths)
    manifest = core.load_json(output / "plots.json")
    assert len(manifest["runs"]) == 4
    assert manifest["skipped"] == []
    assert "SYNTHETIC" in (output / "subjects.csv").read_text(encoding="utf-8")
    import matplotlib.pyplot as plt
    assert plt.get_fignums() == []
    with pytest.raises(FileExistsError):
        create_demo(tmp_path / "demo")


def test_summary_only_plot_and_unknown_run(generic, tmp_path):
    from training_tracker.plotting import generate_plots
    bundle(tmp_path / "runs/example", generic, result={"metrics": {"loss": {"mean": 2.0, "std": 0.4}}})
    paths = generate_plots(tmp_path / "runs", generic, tmp_path / "plots", image_format="svg")
    assert len(paths) == 1
    assert "No usable per-subject" in (tmp_path / "plots/plots.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown run"):
        generate_plots(tmp_path / "runs", generic, tmp_path / "plots", run_ids=["missing"])


def test_comparison_uses_shared_subjects_and_separates_stages(generic, tmp_path, monkeypatch):
    from training_tracker.plotting import generate_plots
    from eegproc.plotting import result_figures
    seen = []
    original = result_figures.plot_model_comparison
    def capture(frame, **kwargs):
        seen.append(frame.copy())
        return original(frame, **kwargs)
    monkeypatch.setattr(result_figures, "plot_model_comparison", capture)
    for name, ids in [("first", [1, 2]), ("second", [2, 3])]:
        bundle(tmp_path / "runs" / name, generic, run_id=name,
               result={"metrics": {"loss": {"mean": 0.5, "std": 0.1}},
                       "subject_metrics": [{"subject_id": i, "loss": 0.5} for i in ids]})
    bundle(tmp_path / "runs/smoke", generic, run_id="smoke",
           result={"metrics": {"loss": {"mean": 0.1, "std": 0.0}},
                   "subject_metrics": [{"subject_id": 2, "loss": 0.1}]})
    path = tmp_path / "runs/smoke/manifest.json"
    manifest = core.load_json(path)
    manifest["stage"] = "smoke"
    core.write_json(path, manifest)
    generate_plots(tmp_path / "runs", generic, tmp_path / "plots")
    assert len(seen) == 1
    assert set(seen[0]["subject_id"]) == {"2"}
    assert set(seen[0]["stage"]) == {"full"}


def test_subject_order_and_custom_metric_direction():
    import pandas as pd
    import matplotlib.pyplot as plt
    from eegproc.plotting.result_figures import plot_per_subject_accuracy, plot_model_comparison
    frame = pd.DataFrame({"subject_id": ["1", "10", "2"], "custom_error": [0.1, 0.2, 0.3]})
    fig = plot_per_subject_accuracy(frame, metric="custom_error")
    assert [label.get_text() for label in fig.axes[0].get_xticklabels()] == ["1", "2", "10"]
    plt.close(fig)
    frame["model"] = ["best", "middle", "worst"]
    fig = plot_model_comparison(frame, metrics="custom_error", maximize=False)
    assert fig.axes[0].get_xticklabels()[0].get_text() == "best"
    plt.close(fig)


@pytest.mark.parametrize("field,value", [("targets", ["a", "a"]), ("objective", {"metric": "x", "maximize": "false"}),
                                       ("smoke_overrides", {"epochs": 50}), ("search_space", [{"name": "missing", "values": [1]}])])
def test_campaign_validation(generic, tmp_path, field, value):
    generic[field] = value
    path = tmp_path / "campaign.json"
    core.write_json(path, generic)
    with pytest.raises(ValueError):
        core.load_campaign(path)
