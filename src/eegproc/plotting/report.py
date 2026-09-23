"""Assemble a minimal, high-information figure set + metrics table from a CV run.

This is the deliverable entry point. Given a result JSON written by
legacy CV or SIC calibration runs, it produces, per model:

* **Fig 1 - Headline**: confusion matrix | outer-fold metric summary.
* **Fig 2 - Subjects**: per-subject accuracy | per-subject/per-task heatmap
  (the heatmap is included only when ``task_id`` was logged).
* **Fig 3 - Hyperparameters**: an inner-CV sweep over the hyperparameter(s) that
  actually varied (auto-detected, or chosen with ``--hyperparams``).
* **Fig 4 - Reliability**: confidence calibration whenever probabilities exist;
  predictive-interval width is added only when an interval log is present.

When the JSON holds more than one model, an extra **model_comparison** figure
compares the models' accuracies (or ``--compare-metrics``) side by side.

plus a paper-ready mean +/- std metrics table (CSV, LaTeX, Markdown).

SIC defaults to strict zero-shot on all trials. Use ``--stage post_calibration
--calibration-shots 6`` for a calibrated report or ``--stage zero_shot_paired
--calibration-shots 6`` for its paired baseline. Each selection gets a separate
subdirectory. SIC headlines use saved subject averages, while confusion and
reliability panels explicitly show pooled prediction occurrences.

CLI::

    python -m eegproc.plotting.report \
        --results smoke_test_outputs/all_smoke_tests_results.json \
        --out figures/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .results_io import ResultsTables, SIC_STAGES, class_probability_columns, load_results
from . import result_figures as rf

# Columns in the inner-CV table that are not hyperparameters.
_NON_HYPERPARAM_COLUMNS = frozenset(
    {"model", "outer_fold", "config_index", "loss", "accuracy", "f1", "precision", "recall"}
)
_SUMMARY_METRIC_ORDER = ["balanced_accuracy", "accuracy", "macro_f1", "roc_auc", "brier_score", "ece", "f1", "precision", "recall", "loss"]
_NON_HYPERPARAM_COLUMNS |= frozenset(_SUMMARY_METRIC_ORDER) | {
    "macro_precision", "macro_recall", "joint_loss", "reconstruction_loss", "decoder_r2",
    "trial_mean_scores", "window_mean_scores", "fold_metrics", "mean_scores", "std_scores",
}


def _slug(name: str) -> str:
    """Filesystem-safe model name."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_") or "model"


def _save(fig, out_dir: Path, name: str, formats: tuple[str, ...], dpi: int) -> list[Path]:
    """Save a figure to every requested format, then close it."""
    paths = [out_dir / f"{name}.{fmt}" for fmt in formats]
    try:
        fig.tight_layout()
        for path in paths:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return paths


def _varying_hyperparameters(inner_cv: pd.DataFrame) -> list[str]:
    """Hyperparameter columns that take more than one value (i.e. were swept)."""
    if inner_cv.empty:
        return []

    candidates = [c for c in inner_cv.columns if c not in _NON_HYPERPARAM_COLUMNS]
    return [c for c in candidates if inner_cv[c].nunique(dropna=True) > 1]


def _headline_figure(tables: ResultsTables, model: str, class_names, metric: str):
    predictions = _model_frame(tables.predictions, model)
    fold_metrics = _model_frame(tables.fold_metrics, model)
    summary = _model_frame(tables.summary, model)
    has_predictions = not predictions.empty
    has_metrics = not fold_metrics.empty or not summary.empty
    panels = int(has_predictions) + int(has_metrics)
    fig, axes = plt.subplots(1, panels, figsize=(14 if panels == 2 else 8, 5), squeeze=False)
    if has_predictions:
        rf.plot_confusion_matrix(predictions, model=model, class_names=class_names, ax=axes[0, 0])
        if tables.stage:
            axes[0, 0].set_title("Pooled trial predictions\n(row-normalized; not subject-averaged)")
    if has_metrics:
        metrics = tuple(dict.fromkeys((metric, "accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "brier_score")))
        rf.plot_metric_summary(fold_metrics, model=model, metrics=metrics, ax=axes[0, -1],
                               summary=summary, aggregation_unit=tables.aggregation_unit)
    fig.suptitle(f"{model} - {_evaluation_label(tables)}", fontsize=13)
    return fig


def _model_frame(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    return frame[frame["model"] == model] if not frame.empty and "model" in frame else frame


def _evaluation_label(tables: ResultsTables) -> str:
    labels = {"zero_shot_all_trials": "strict zero-shot (all trials)",
              "zero_shot_paired": "paired zero-shot", "post_calibration": "post-calibration"}
    label = labels.get(tables.stage, "headline performance")
    if tables.calibration_shots is not None:
        label += f" ({tables.calibration_shots} shots)"
    return label


def _subject_figure(tables: ResultsTables, model: str, metric: str):
    subject_labels = tables.subject_id_mapping.get(model) or None

    model_predictions = (
        tables.predictions[tables.predictions["model"] == model]
        if (not tables.predictions.empty and "model" in tables.predictions.columns)
        else tables.predictions
    )
    include_tasks = (
        not model_predictions.empty
        and "task_id" in model_predictions.columns
        and model_predictions["task_id"].notna().any()
    )

    fig, axes = plt.subplots(1, 2 if include_tasks else 1, figsize=(13 if include_tasks else 7, 5))
    axes = axes if include_tasks else [axes]

    rf.plot_per_subject_accuracy(
        tables.user_metrics, model=model, metric=metric, subject_labels=subject_labels, ax=axes[0]
    )
    if include_tasks:
        rf.plot_subject_task_heatmap(tables.predictions, model=model, ax=axes[1])

    fig.suptitle(f"{model} - subject breakdown\n{_evaluation_label(tables)}", fontsize=13)
    return fig


def _hyperparameter_figure(tables: ResultsTables, model: str, params: list[str], metric: str):
    use_params = params[:2]

    # Default to 2D plots (line/heatmap). 3D surfaces are optional and require numeric params.
    surface = False
    projection = "3d" if surface and len(use_params) == 2 else None

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection=projection)
    rf.plot_hyperparameter_sweep(
        tables.inner_cv, params=use_params, metric=metric, model=model, surface=surface, ax=ax
    )
    fig.suptitle(f"{model} - hyperparameter sweep", fontsize=13)
    return fig


def _reliability_figure(tables: ResultsTables, model: str):
    intervals = _model_frame(tables.intervals, model)
    has_intervals = not intervals.empty and {"p_pred_ci_low", "p_pred_ci_high"}.issubset(intervals.columns)
    fig, axes = plt.subplots(1, 2 if has_intervals else 1, figsize=(13 if has_intervals else 7, 5), squeeze=False)
    rf.plot_reliability(tables.predictions, model=model, ax=axes[0, 0])
    if has_intervals:
        rf.plot_uncertainty(intervals, model=model, ax=axes[0, 1])
    fig.suptitle(f"{model} - pooled reliability\n{_evaluation_label(tables)}", fontsize=13)
    return fig


def _format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Wide 'mean +/- std' table, one row per model, ordered metric columns."""
    rows = []
    for model, group in summary.groupby("model"):
        row = {"model": model}
        for _, record in group.iterrows():
            mean, std = record["mean"], record["std"]
            row[record["metric"]] = f"{mean:.4f}" + (f" ± {std:.4f}" if pd.notna(std) else "")
        rows.append(row)

    table = pd.DataFrame(rows).set_index("model")
    ordered = [m for m in _SUMMARY_METRIC_ORDER if m in table.columns]
    ordered += [m for m in table.columns if m not in ordered]
    return table[ordered]


def _to_markdown(table: pd.DataFrame) -> str:
    """Render an indexed DataFrame as a GitHub markdown table (no tabulate dep)."""
    headers = [table.index.name or "model", *table.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for index, record in table.iterrows():
        cells = [str(index), *[str(v) for v in record.tolist()]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# Single-pass character map. Sequential str.replace() is wrong here because
# escaping "\" to "\textbackslash{}" would then have its braces re-escaped by a
# later "{"/"}" pass; a per-character translation avoids re-processing.
_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "±": r"$\pm$",
}


def _latex_escape(value: str) -> str:
    """Escape LaTeX special characters in a single pass (no re-escaping)."""
    return "".join(_LATEX_SPECIALS.get(char, char) for char in str(value))

def _to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    """Render an indexed DataFrame as a self-contained LaTeX table (no jinja2 dep)."""
    columns = list(table.columns)
    col_spec = "l" + "c" * len(columns)
    header = " & ".join([_latex_escape(table.index.name or "model"), *map(_latex_escape, columns)])

    body = []
    for index, record in table.iterrows():
        cells = [_latex_escape(str(index)), *[_latex_escape(str(v)) for v in record.tolist()]]
        body.append("  " + " & ".join(cells) + r" \\")

    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\hline",
            "  " + header + r" \\",
            r"\hline",
            *body,
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _write_summary_tables(summary: pd.DataFrame, out_dir: Path, aggregation_unit: str = "outer_fold") -> None:
    if summary.empty:
        return

    summary.to_csv(out_dir / "metrics_table.csv", index=False)

    formatted = _format_summary_table(summary)
    (out_dir / "metrics_table.tex").write_text(
        _to_latex(
            formatted,
            caption=(r"Subject-averaged performance (mean $\pm$ std across subjects)."
                     if aggregation_unit == "subject" else r"Outer-fold performance (mean $\pm$ std across outer folds)."),
            label="tab:cv",
        ),
        encoding="utf-8",
    )
    (out_dir / "metrics_table.md").write_text(_to_markdown(formatted), encoding="utf-8")


def build_report(
    results_path: str | Path,
    out_dir: str | Path,
    class_names: list[str] | None = None,
    hyperparams: list[str] | None = None,
    metric: str = "accuracy",
    compare_metrics: tuple[str, ...] | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 300,
    stage: str = "zero_shot_all_trials",
    calibration_shots: int | None = None,
) -> list[Path]:
    """Generate the figure set + metrics table for every model in a result JSON.

    When the JSON holds more than one model, an extra ``model_comparison`` figure
    is written that compares ``compare_metrics`` across models.

    SIC reports select strict zero-shot by default; calibration stages require
    an explicit shot count. Each SIC evaluation gets its own output subfolder.
    Headline means/stds use the saved summaries, never pooled predictions.
    Returns every figure path written, including all requested formats.
    """
    tables = load_results(results_path, stage=stage, calibration_shots=calibration_shots)
    if not formats or dpi <= 0:
        raise ValueError("At least one output format and a positive DPI are required.")
    supported = plt.figure().canvas.get_supported_filetypes()
    plt.close(plt.gcf())
    unsupported = set(formats) - supported.keys()
    if unsupported:
        raise ValueError(f"Unsupported figure formats: {sorted(unsupported)}")
    out_dir = Path(out_dir)
    if tables.stage:
        folder = tables.stage
        if tables.calibration_shots is not None:
            folder += f"_{tables.calibration_shots}_shot"
        out_dir /= folder
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for model in tables.models:
        prefix = _slug(model)
        predictions = _model_frame(tables.predictions, model)
        subjects = _model_frame(tables.user_metrics, model)
        folds = _model_frame(tables.fold_metrics, model)
        summary = _model_frame(tables.summary, model)
        available = set(folds.columns) | set(subjects.columns)
        if not summary.empty:
            available.update(summary["metric"])
        if available and metric not in available:
            raise ValueError(f"[{model}] Metric {metric!r} is not available in the selected evaluation.")
        print(f"[{model}] {_evaluation_label(tables)}: {len(predictions)} prediction rows, "
              f"{subjects['subject_id'].nunique() if 'subject_id' in subjects else 0} subjects.")
        if not summary.empty:
            selected = summary[summary["metric"] == metric]
            if not selected.empty:
                print(f"[{model}] saved {tables.aggregation_unit}-mean {metric}={selected.iloc[0]['mean']:.8f}")

        if not predictions.empty or not folds.empty or not summary.empty:
            fig = _headline_figure(tables, model, class_names, metric)
            written.extend(_save(fig, out_dir, f"{prefix}_fig1_headline", formats, dpi))

        if not subjects.empty and {"subject_id", metric}.issubset(subjects.columns):
            fig = _subject_figure(tables, model, metric)
            written.extend(_save(fig, out_dir, f"{prefix}_fig2_subjects", formats, dpi))
        else:
            print(f"[{model}] no per-subject {metric} values; skipping Fig 2.")

        model_inner = tables.inner_cv[tables.inner_cv["model"] == model] if not tables.inner_cv.empty else tables.inner_cv
        params = hyperparams or _varying_hyperparameters(model_inner)
        if params and metric in model_inner.columns:
            fig = _hyperparameter_figure(tables, model, params, metric)
            written.extend(_save(fig, out_dir, f"{prefix}_fig3_hyperparameters", formats, dpi))
        else:
            print(f"[{model}] no swept hyperparameters found; skipping Fig 3.")

        has_probabilities = any(c in predictions for c in ("p_pred", "confidence")) or bool(class_probability_columns(predictions))
        if not predictions.empty and has_probabilities:
            fig = _reliability_figure(tables, model)
            written.extend(_save(fig, out_dir, f"{prefix}_fig4_reliability", formats, dpi))

    if len(tables.models) > 1 and not tables.fold_metrics.empty:
        fig = rf.plot_model_comparison(tables.fold_metrics, metrics=compare_metrics or (metric,))
        written.extend(_save(fig, out_dir, "model_comparison", formats, dpi))

    _write_summary_tables(tables.summary, out_dir, tables.aggregation_unit)
    if not tables.user_metrics.empty:
        tables.user_metrics.to_csv(out_dir / "subject_metrics.csv", index=False)
    metadata = {"results": str(Path(results_path).resolve()), "stage": tables.stage,
                "calibration_shots": tables.calibration_shots, "metric": metric,
                "aggregation_unit": tables.aggregation_unit,
                "prediction_rows": len(tables.predictions),
                "figure_files": [p.name for p in written]}
    (out_dir / "report_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if tables.stage:
        (out_dir / "report_notes.md").write_text(
            f"# {_evaluation_label(tables)}\n\n"
            "Headline metrics and the metrics table retain the saved mean and standard deviation across subjects. "
            "For calibration, each subject's score is already averaged across its calibration folds.\n\n"
            "Confusion matrices and reliability curves pool prediction occurrences within this stage and shot count. "
            "A trial may occur in multiple calibration evaluation folds. These pooled views do not replace the "
            "subject-averaged metrics, and the reliability curve is not the saved subject-averaged ECE.\n\n"
            "No source training, oracle-epoch, or other calibration-stage predictions are included.\n", encoding="utf-8")

    print(f"Wrote {len(written)} figure files + available metrics tables to {out_dir}/")
    return written


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")  # CLI writes files only; force a headless backend.

    parser = argparse.ArgumentParser(description="Turn CV result JSON into report figures + table.")
    parser.add_argument("--results", required=True, help="Path to the CV result JSON.")
    parser.add_argument("--out", default="figures", help="Output directory for figures/tables.")
    parser.add_argument("--class-names", default=None, help="Comma-separated class display names.")
    parser.add_argument(
        "--hyperparams",
        default=None,
        help="Comma-separated hyperparameter(s) for Fig 3 (1=line, 2=heatmap/surface). "
        "Default: auto-detect the swept ones.",
    )
    parser.add_argument("--metric", default="accuracy", help="Primary metric for headline, per-subject, sweep and comparison panels.")
    parser.add_argument("--stage", choices=SIC_STAGES, default="zero_shot_all_trials",
                        help="SIC evaluation (default: strict zero-shot on all trials).")
    parser.add_argument("--calibration-shots", type=int, default=None,
                        help="Required positive shot count for paired zero-shot or post-calibration.")
    parser.add_argument(
        "--compare-metrics",
        default=None,
        help="Comma-separated metric(s) for the multi-model comparison figure.",
    )
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI.")
    args = parser.parse_args()

    build_report(
        results_path=args.results,
        out_dir=args.out,
        class_names=_split_csv(args.class_names),
        hyperparams=_split_csv(args.hyperparams),
        metric=args.metric,
        compare_metrics=tuple(_split_csv(args.compare_metrics) or (args.metric,)),
        formats=tuple(_split_csv(args.formats) or ("png", "pdf")),
        dpi=args.dpi,
        stage=args.stage,
        calibration_shots=args.calibration_shots,
    )


if __name__ == "__main__":
    main()
