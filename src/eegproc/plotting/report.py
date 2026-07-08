"""Assemble a minimal, high-information figure set + metrics table from a CV run.

This is the deliverable entry point. Given a result JSON written by
``nested_lnso_cv`` (single- or multi-model), it produces, per model:

* **Fig 1 - Headline**: confusion matrix | outer-fold metric summary.
* **Fig 2 - Subjects**: per-subject accuracy | per-subject/per-task heatmap
  (the heatmap is included only when ``task_id`` was logged).
* **Fig 3 - Hyperparameters**: an inner-CV sweep over the hyperparameter(s) that
  actually varied (auto-detected, or chosen with ``--hyperparams``).
* **Fig 4 - Reliability**: calibration | predictive-interval width (only when a
  variational interval log is present).

plus a paper-ready mean +/- std metrics table (CSV, LaTeX, Markdown).

CLI::

    python -m eegproc.plotting.report \
        --results smoke_test_outputs/all_smoke_tests_results.json \
        --out figures/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .results_io import ResultsTables, load_results
from . import result_figures as rf

# Columns in the inner-CV table that are not hyperparameters.
_NON_HYPERPARAM_COLUMNS = frozenset(
    {"model", "outer_fold", "config_index", "loss", "accuracy", "f1", "precision", "recall"}
)
_SUMMARY_METRIC_ORDER = ["accuracy", "f1", "precision", "recall", "loss"]


def _slug(name: str) -> str:
    """Filesystem-safe model name."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_") or "model"


def _save(fig, out_dir: Path, name: str, formats: tuple[str, ...], dpi: int) -> None:
    """Save a figure to every requested format, then close it."""
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"{name}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _varying_hyperparameters(inner_cv: pd.DataFrame) -> list[str]:
    """Hyperparameter columns that take more than one value (i.e. were swept)."""
    if inner_cv.empty:
        return []

    candidates = [c for c in inner_cv.columns if c not in _NON_HYPERPARAM_COLUMNS]
    return [c for c in candidates if inner_cv[c].nunique(dropna=True) > 1]


def _headline_figure(tables: ResultsTables, model: str, class_names, metric: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rf.plot_confusion_matrix(tables.predictions, model=model, class_names=class_names, ax=axes[0])
    rf.plot_metric_summary(tables.fold_metrics, model=model, ax=axes[1])
    fig.suptitle(f"{model} - headline performance", fontsize=13)
    return fig


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

    fig.suptitle(f"{model} - subject breakdown", fontsize=13)
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rf.plot_reliability(tables.predictions, model=model, ax=axes[0])
    rf.plot_uncertainty(tables.intervals, model=model, ax=axes[1])
    fig.suptitle(f"{model} - reliability & uncertainty", fontsize=13)
    return fig


def _format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Wide 'mean +/- std' table, one row per model, ordered metric columns."""
    rows = []
    for model, group in summary.groupby("model"):
        row = {"model": model}
        for _, record in group.iterrows():
            row[record["metric"]] = f"{record['mean']:.3f} ± {record['std']:.3f}"
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


def _latex_escape(value: str) -> str:
    """Escape LaTeX special characters in model/metric names and cell values."""
    text = str(value)
    # Order matters: escape backslash first.
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("±", r"$\pm$"),
    ]
    for char, repl in replacements:
        text = text.replace(char, repl)
    return text

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


def _write_summary_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return

    summary.to_csv(out_dir / "metrics_table.csv", index=False)

    formatted = _format_summary_table(summary)
    (out_dir / "metrics_table.tex").write_text(
        _to_latex(
            formatted,
            caption=r"Outer-fold performance (mean $\pm$ std across outer folds).",
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
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    """Generate the figure set + metrics table for every model in a result JSON.

    Returns the list of figure paths written.
    """
    tables = load_results(results_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for model in tables.models:
        prefix = _slug(model)

        fig = _headline_figure(tables, model, class_names, metric)
        _save(fig, out_dir, f"{prefix}_fig1_headline", formats, dpi)
        written.append(out_dir / f"{prefix}_fig1_headline.{formats[0]}")

        fig = _subject_figure(tables, model, metric)
        _save(fig, out_dir, f"{prefix}_fig2_subjects", formats, dpi)
        written.append(out_dir / f"{prefix}_fig2_subjects.{formats[0]}")

        model_inner = tables.inner_cv[tables.inner_cv["model"] == model] if not tables.inner_cv.empty else tables.inner_cv
        params = hyperparams or _varying_hyperparameters(model_inner)
        if params:
            fig = _hyperparameter_figure(tables, model, params, metric)
            _save(fig, out_dir, f"{prefix}_fig3_hyperparameters", formats, dpi)
            written.append(out_dir / f"{prefix}_fig3_hyperparameters.{formats[0]}")
        else:
            print(f"[{model}] no swept hyperparameters found; skipping Fig 3.")

        model_intervals = tables.intervals[tables.intervals["model"] == model] if not tables.intervals.empty else tables.intervals
        if not model_intervals.empty:
            fig = _reliability_figure(tables, model)
            _save(fig, out_dir, f"{prefix}_fig4_reliability", formats, dpi)
            written.append(out_dir / f"{prefix}_fig4_reliability.{formats[0]}")

    _write_summary_tables(tables.summary, out_dir)

    print(f"Wrote {len(written)} figures + metrics table to {out_dir}/")
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
    parser.add_argument("--metric", default="accuracy", help="Metric for per-subject/sweep panels.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI.")
    args = parser.parse_args()

    build_report(
        results_path=args.results,
        out_dir=args.out,
        class_names=_split_csv(args.class_names),
        hyperparams=_split_csv(args.hyperparams),
        metric=args.metric,
        formats=tuple(_split_csv(args.formats) or ("png", "pdf")),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
