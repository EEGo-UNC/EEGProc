"""Result panels reused from sic-band-fix/src/eegproc/plotting/result_figures.py.

These shared panels accept tidy DataFrames and return matplotlib figures.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
_MODEL_COMPARISON_PALETTE = ("#4C72B0", "#55A868", "#C44E52", "#8172B2")

def _resolve_ax(ax, projection: str | None = None):
    """Return ``(fig, ax)``, creating a new figure when ``ax`` is None."""
    if ax is not None:
        return ax.figure, ax

    if projection == "3d":
        fig = plt.figure(figsize=(7, 6))
        return fig, fig.add_subplot(111, projection="3d")

    fig, ax = plt.subplots(figsize=(7, 5))
    return fig, ax

def _filter_model(df: pd.DataFrame, model: str | None) -> pd.DataFrame:
    """Restrict a frame to one model, defaulting to the only model present."""
    if df.empty:
        return df
    if "model" in df.columns:
        models = list(df["model"].unique())
        if model is not None:
            df = df[df["model"] == model]
        elif len(models) > 1:
            raise ValueError(f"Multiple models present {models}; pass model=... to select one.")
    _require_single_evaluation(df)
    return df

def _require_single_evaluation(df: pd.DataFrame) -> None:
    for column in ("stage", "calibration_shots"):
        if column in df and df[column].nunique(dropna=False) > 1:
            raise ValueError("Mixed SIC evaluations: select one stage and calibration shot count before plotting.")

def metric_lower_is_better(metric: str) -> bool:
    return metric in {"brier_score", "brier", "ece", "mse", "mae", "rmse"} or metric.endswith("loss") or metric.endswith(("_brier_score", "_ece"))

def _metric_label(metric: str) -> str:
    labels = {"balanced_accuracy": "Balanced accuracy", "roc_auc": "ROC-AUC",
              "macro_f1": "Macro F1", "brier_score": "Brier score", "ece": "ECE"}
    return labels.get(metric, metric.replace("_", " ").capitalize()) + (" ↓" if metric_lower_is_better(metric) else "")

def _score_limits(ax, values) -> None:
    """Keep ordinary scores comparable, without clipping losses or negative R²."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not len(values) or (values.min() >= 0 and values.max() <= 1):
        ax.set_ylim(0, 1.05)
    else:
        low, high = min(0.0, values.min()), max(0.0, values.max())
        pad = max((high - low) * 0.08, 0.05)
        ax.set_ylim(low - pad if low < 0 else 0, high + pad)

def plot_metric_summary(
    fold_metrics: pd.DataFrame,
    model: str | None = None,
    metrics: tuple[str, ...] = ("accuracy", "f1", "precision", "recall"),
    ax=None,
    summary: pd.DataFrame | None = None,
    aggregation_unit: str = "outer_fold",
):
    """Mean +/- std bars per metric, with individual outer-fold points overlaid.

    Shows both the headline number (bar height = mean across folds, error bar =
    std) and the fold-to-fold spread (scatter), which matters for LOSO where
    variance across subjects is large.
    """
    fold_metrics = _filter_model(fold_metrics, model)

    summary = _filter_model(summary, model) if summary is not None else pd.DataFrame()
    saved = summary.set_index("metric") if not summary.empty else pd.DataFrame()
    metrics = tuple(m for m in metrics if m in fold_metrics.columns or m in saved.index)
    if not metrics:
        raise ValueError("No requested metrics to summarize.")
    means = [float(saved.loc[m, "mean"]) if m in saved.index else fold_metrics[m].mean() for m in metrics]
    stds = [float(saved.loc[m, "std"]) if m in saved.index else fold_metrics[m].std(ddof=0) for m in metrics]

    fig, ax = _resolve_ax(ax)
    x = np.arange(len(metrics))

    errors = np.nan_to_num(stds, nan=0.0)
    ax.bar(x, means, yerr=errors, capsize=4, color="#4C72B0", alpha=0.85)

    for i, metric in enumerate(metrics):
        values = fold_metrics[metric].to_numpy(dtype=float) if metric in fold_metrics else np.array([])
        jitter = (np.random.RandomState(0).rand(len(values)) - 0.5) * 0.25
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            color="#1f1f1f",
            s=18,
            alpha=0.6,
            zorder=3,
        )
        finite_values = values[np.isfinite(values)]
        top = max(means[i] + errors[i], finite_values.max()) if len(finite_values) else means[i] + errors[i]
        ax.text(i, top + 0.025, f"{means[i]:.4f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([_metric_label(m) for m in metrics], rotation=25, ha="right")
    raw_values = [v for m in metrics if m in fold_metrics for v in fold_metrics[m]]
    _score_limits(ax, [*(np.asarray(raw_values) + 0.05), *(np.asarray(means) - errors), *(np.asarray(means) + errors + 0.05)])
    unit = {"subject": "subjects", "outer_fold": "outer folds"}.get(aggregation_unit, aggregation_unit.replace("_", " "))
    ax.set_ylabel(f"Score (mean ± std across {unit})")
    ax.set_title("Subject-averaged performance" if aggregation_unit == "subject" else f"Performance across {unit}")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    return fig

def plot_per_subject_accuracy(
    user_metrics: pd.DataFrame,
    model: str | None = None,
    metric: str = "accuracy",
    subject_labels: dict | None = None,
    ax=None,
):
    """Per-subject score as a line, ordered by subject, with the cohort mean drawn.

    Subjects are plotted in subject-id order (not sorted by score) so the x-axis is
    stable and comparable across models/metrics.
    """
    user_metrics = _filter_model(user_metrics, model)

    if user_metrics.empty:
        raise ValueError("No per-user metrics to plot.")

    per_subject = user_metrics.groupby("subject_id", sort=False)[metric].mean()
    # IDs serialized as strings still need 1, 2, ... 10 order rather than 1, 10, 2.
    def subject_key(value):
        return [(0, int(part)) if part.isdigit() else (1, part.casefold())
                for part in re.split(r"(\d+)", str(value))]
    per_subject = per_subject.reindex(sorted(per_subject.index, key=subject_key))

    if subject_labels:
        index = [subject_labels.get(str(s), s) for s in per_subject.index]
    else:
        index = per_subject.index.tolist()

    fig, ax = _resolve_ax(ax)
    x = np.arange(len(per_subject))
    ax.plot(x, per_subject.to_numpy(), marker="o", color="#55A868", linewidth=1.5)

    mean_value = float(per_subject.mean())
    ax.axhline(
        mean_value,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"mean = {mean_value:.4f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(index, rotation=45, ha="right")
    _score_limits(ax, per_subject.to_numpy())
    ax.set_xlabel("Subject")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(f"Per-subject {metric.replace('_', ' ')}")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    return fig

def plot_model_comparison(
    fold_metrics: pd.DataFrame,
    metrics: str | tuple[str, ...] = "accuracy",
    ax=None,
    maximize: bool | None = None,
):
    """Compare one or more metrics across models (mean +/- std across outer folds).

    Draws one bar group per model. With a single metric the models are sorted
    best-first and each bar is annotated; with several metrics the bars are
    grouped per model and colour-coded by metric. Error bars are the std across
    outer folds. Expects ``fold_metrics`` with a ``model`` column (i.e. a
    multi-model results file loaded via :func:`results_io.load_results`).
    """
    if fold_metrics.empty or "model" not in fold_metrics.columns:
        raise ValueError("fold_metrics must be non-empty and have a 'model' column.")
    _require_single_evaluation(fold_metrics)

    metrics = (metrics,) if isinstance(metrics, str) else tuple(metrics)
    metrics = tuple(m for m in metrics if m in fold_metrics.columns)
    if not metrics:
        raise ValueError("None of the requested metrics are present in fold_metrics.")

    means = fold_metrics.groupby("model")[list(metrics)].mean()
    stds = fold_metrics.groupby("model")[list(metrics)].std(ddof=0).fillna(0.0)

    # Order models best-first by the primary (first) metric.
    ascending = metric_lower_is_better(metrics[0]) if maximize is None else not maximize
    order = means[metrics[0]].sort_values(ascending=ascending).index
    means = means.loc[order]
    stds = stds.loc[order]
    models = list(means.index)

    fig, ax = _resolve_ax(ax)
    x = np.arange(len(models))
    n_metrics = len(metrics)
    width = 0.8 / n_metrics

    for j, metric in enumerate(metrics):
        offset = (j - (n_metrics - 1) / 2) * width
        ax.bar(
            x + offset,
            means[metric].to_numpy(),
            width,
            yerr=stds[metric].to_numpy(),
            capsize=3,
            label=metric,
            color=_MODEL_COMPARISON_PALETTE[j % len(_MODEL_COMPARISON_PALETTE)],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    _score_limits(ax, np.concatenate([(means - stds).to_numpy().ravel(), (means + stds).to_numpy().ravel()]))
    unit = "subjects" if "aggregation_unit" in fold_metrics and (fold_metrics["aggregation_unit"] == "subject").all() else "outer folds"
    ax.set_ylabel(f"Score (mean +/- std across {unit})")
    ax.set_title("Model comparison")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)

    if n_metrics > 1:
        ax.legend(fontsize=9)
    else:
        for xi, value, spread in zip(x, means[metrics[0]].to_numpy(), stds[metrics[0]].to_numpy()):
            ax.text(xi, value + spread + 0.025, f"{value:.2f}", ha="center", fontsize=9)
        low, high = ax.get_ylim()
        ax.set_ylim(low, max(high, float((means + stds).max().max()) + 0.08))

    return fig
