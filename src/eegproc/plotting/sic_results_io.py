"""Load SIC run directories into tidy DataFrames for reporting.

``subject_calibration_cv`` writes a different artifact layout than
``nested_lnso_cv``, so :mod:`eegproc.plotting.results_io` cannot read it. A SIC
run directory looks like::

    <out_dir>/<run_name>_<timestamp>/
        best_hyperparameters.json
        hyperparameter_search_results.json
        configuration_0001/
            model_config.json
            sic_overall_metrics.json
            sic_subject_summary.csv
            sic_calibration_folds.csv
            sic_prediction_diagnostics.csv
            sic_window_predictions.csv
            sic_trial_predictions.csv

:func:`load_sic_run` accepts a path at any level of that tree -- the sweep
output root, the timestamped run directory, or a ``configuration_XXXX``
directory -- and resolves it down to one configuration.

Note on prediction logs: ``sic_window_predictions.csv`` and
``sic_trial_predictions.csv`` concatenate every logged evaluation (zero-shot
plus calibrated, at every shot level) without a column identifying which phase a
row came from. They are therefore loaded but deliberately not used for
phase-specific claims such as reliability diagrams; use the per-level ECE in
``sic_overall_metrics.json`` for calibration quality instead.

Per-epoch held-out-subject metrics are deliberately not loaded. Those values
exist so a run can be inspected after the fact, not so anything can be selected
from them -- choosing an epoch budget or a hyperparameter against the held-out
subject would make the result an oracle estimate rather than a true LOSOCV one.
Hyperparameter evidence comes from ``sic_prediction_diagnostics.csv``, which is
built from source subjects only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# Phases reported for each calibration shot level, and the key holding each
# one's per-subject mean inside ``overall.calibration_levels.<shots>``.
_PHASE_MEAN_KEYS = {
    "zero_shot": "paired_zero_shot_mean_scores",
    "calibrated": "calibrated_mean_scores",
    "delta": "delta_mean_scores",
}
_PHASE_STD_KEYS = {
    "zero_shot": "paired_zero_shot_std_scores",
    "calibrated": "calibrated_std_scores",
    "delta": "delta_std_scores",
}


@dataclass(frozen=True)
class SICRun:
    """One SIC configuration's artifacts, loaded and labelled.

    Tables that the run did not produce come back empty rather than missing, so
    callers can check ``df.empty`` instead of catching ``KeyError``.
    """

    label: str
    config_dir: Path
    overall: dict
    model_config: dict
    subject_summary: pd.DataFrame
    folds: pd.DataFrame
    diagnostics: pd.DataFrame
    commit: dict = field(default_factory=dict)

    @property
    def selection_shots(self) -> int | None:
        """Shot level the run ranked configurations on, if recorded."""
        value = self.overall.get("calibration_selection_shots")
        return None if value is None else int(value)

    @property
    def shot_levels(self) -> list[int]:
        return sorted(int(key) for key in self.overall.get("calibration_levels", {}))

    @property
    def n_subjects(self) -> int | None:
        value = self.overall.get("n_subjects")
        return None if value is None else int(value)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame


def _read_commit(config_dir: Path) -> dict:
    """Read ``run_commit.txt`` written by the sweep script, if present.

    The file is written at the sweep output root, which is two levels above the
    configuration directory, so walk upward rather than guessing.
    """
    for candidate in (config_dir, *config_dir.parents):
        path = candidate / "run_commit.txt"
        if path.is_file():
            values = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator:
                    values[key.strip()] = value.strip()
            return values
    return {}


def _winner_candidates(run_dir: Path, record: dict) -> list[Path]:
    """Ways the grid winner may be addressable, most specific first.

    ``configuration_dir`` is written relative to the working directory that ran
    the training, which is the project root rather than the run directory, and
    on a cluster it may be an absolute path that does not exist locally. The
    directory name is the part that always travels, so fall back to it.
    """
    candidates: list[Path] = []

    recorded = record.get("configuration_dir")
    if recorded:
        recorded_path = Path(recorded)
        candidates.append(recorded_path)
        candidates.append(run_dir / recorded_path.name)

    configuration_id = record.get("configuration_id")
    if configuration_id is not None:
        candidates.append(run_dir / f"configuration_{int(configuration_id):04d}")

    return candidates


def _resolve_config_dir(path: Path) -> Path:
    """Resolve any level of a SIC output tree down to one configuration dir."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SIC run path does not exist: {path}")

    if (path / "sic_overall_metrics.json").is_file():
        return path

    # A timestamped run directory records which configuration won the grid.
    # Honour that rather than picking arbitrarily when several exist.
    best = path / "best_hyperparameters.json"
    if best.is_file():
        record = _read_json(best)
        for candidate in _winner_candidates(path, record):
            if (candidate / "sic_overall_metrics.json").is_file():
                return candidate

    candidates = sorted(path.glob("**/configuration_*/sic_overall_metrics.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No configuration directory with sic_overall_metrics.json under {path}. "
            "Point at the sweep output root, the timestamped run directory, or a "
            "configuration_XXXX directory."
        )
    if len(candidates) > 1:
        # Several configurations means a grid search. Without
        # best_hyperparameters.json there is no principled winner, so make the
        # ambiguity visible instead of silently taking the first.
        listed = "\n  ".join(str(item.parent) for item in candidates)
        raise ValueError(
            f"{path} contains {len(candidates)} configurations and no usable "
            f"best_hyperparameters.json. Pass one directly:\n  {listed}"
        )
    return candidates[0].parent


def load_sic_run(path: str | Path, label: str | None = None) -> SICRun:
    """Load one SIC configuration from any level of its output tree."""
    config_dir = _resolve_config_dir(Path(path))

    if label is None:
        # runs/sweep/valence/<label>/<run_name>_<timestamp>/configuration_0001
        commit = _read_commit(config_dir)
        label = commit.get("run_label") or config_dir.parent.parent.name

    return SICRun(
        label=str(label),
        config_dir=config_dir,
        overall=_read_json(config_dir / "sic_overall_metrics.json"),
        model_config=(
            _read_json(config_dir / "model_config.json")
            if (config_dir / "model_config.json").is_file()
            else {}
        ),
        subject_summary=_read_csv(config_dir / "sic_subject_summary.csv"),
        folds=_read_csv(config_dir / "sic_calibration_folds.csv"),
        diagnostics=_read_csv(config_dir / "sic_prediction_diagnostics.csv"),
        commit=_read_commit(config_dir),
    )


def load_sic_runs(
    paths: list[str | Path],
    labels: list[str] | None = None,
) -> list[SICRun]:
    """Load several runs, erroring on duplicate labels rather than overwriting."""
    if labels is not None and len(labels) != len(paths):
        raise ValueError(
            f"Got {len(paths)} paths but {len(labels)} labels; they must align."
        )

    runs = [
        load_sic_run(path, None if labels is None else labels[index])
        for index, path in enumerate(paths)
    ]

    seen = [run.label for run in runs]
    duplicates = sorted({name for name in seen if seen.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Run labels must be unique; repeated: {duplicates}. Pass --labels "
            "to name them explicitly."
        )
    return runs


def shot_level_frame(runs: list[SICRun]) -> pd.DataFrame:
    """Tidy per-shot-level metrics: one row per run/shots/phase/metric.

    Includes a ``shots=0`` row per run carrying
    ``zero_shot_all_trials_mean_scores`` -- the untouched population model
    evaluated on every target trial, which is the natural left-hand anchor of a
    shots-versus-performance curve.
    """
    rows: list[dict] = []
    for run in runs:
        for metric, value in run.overall.get(
            "zero_shot_all_trials_mean_scores", {}
        ).items():
            rows.append(
                {
                    "run": run.label,
                    "shots": 0,
                    "phase": "zero_shot",
                    "metric": metric,
                    "value": float(value),
                    "std": float(
                        run.overall.get("zero_shot_all_trials_std_scores", {}).get(
                            metric, float("nan")
                        )
                    ),
                }
            )

        for shots_key, level in run.overall.get("calibration_levels", {}).items():
            for phase, mean_key in _PHASE_MEAN_KEYS.items():
                std_scores = level.get(_PHASE_STD_KEYS[phase], {})
                for metric, value in level.get(mean_key, {}).items():
                    rows.append(
                        {
                            "run": run.label,
                            "shots": int(shots_key),
                            "phase": phase,
                            "metric": metric,
                            "value": float(value),
                            "std": float(std_scores.get(metric, float("nan"))),
                        }
                    )

    return pd.DataFrame(rows)


def subject_frame(runs: list[SICRun], shots: int, phase: str) -> pd.DataFrame:
    """Per-subject scores at one shot level, tidied across runs.

    ``sic_subject_summary.csv`` is stored wide, with columns named
    ``<shots>_shot_<phase>_<metric>`` (plus ``zero_shot_all_<metric>`` for the
    uncalibrated model). This melts the requested slice into
    ``run / target_subject / metric / value``.
    """
    if phase not in _PHASE_MEAN_KEYS:
        raise ValueError(
            f"phase must be one of {sorted(_PHASE_MEAN_KEYS)}; got {phase!r}."
        )

    prefix = (
        "zero_shot_all_"
        if shots == 0
        else f"{int(shots)}_shot_{'paired_zero' if phase == 'zero_shot' else phase}_"
    )

    rows: list[dict] = []
    for run in runs:
        summary = run.subject_summary
        if summary.empty:
            continue
        matching = [
            column for column in summary.columns if column.startswith(prefix)
        ]
        if not matching:
            raise KeyError(
                f"Run {run.label!r} has no columns starting with {prefix!r}. "
                f"Available shot levels: {run.shot_levels}."
            )
        for _, record in summary.iterrows():
            for column in matching:
                rows.append(
                    {
                        "run": run.label,
                        "target_subject": record["target_subject"],
                        "metric": column[len(prefix) :],
                        "value": float(record[column]),
                    }
                )

    return pd.DataFrame(rows)



def summary_table(runs: list[SICRun], metrics: tuple[str, ...]) -> pd.DataFrame:
    """Compact run-by-run table: one row per run/shot level, metrics as columns."""
    levels = shot_level_frame(runs)
    if levels.empty:
        return pd.DataFrame()

    wanted = levels[
        levels["metric"].isin(metrics) & levels["phase"].isin({"zero_shot", "calibrated"})
    ]
    table = wanted.pivot_table(
        index=["run", "shots", "phase"],
        columns="metric",
        values="value",
        sort=False,
    ).reset_index()
    table.columns.name = None

    ordered = [column for column in metrics if column in table.columns]
    return table[["run", "shots", "phase", *ordered]].sort_values(
        ["run", "shots", "phase"]
    )
