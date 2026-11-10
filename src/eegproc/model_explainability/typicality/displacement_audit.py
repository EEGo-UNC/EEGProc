"""Rebuild Table 2 and audit displacement from immutable endpoint archives.

No checkpoint, signal generation, training, or decoded-classification evaluation
is performed. All finite completed endpoints are included, including failed
latent target attempts. Missing endpoints stay in eligible denominators.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .artifacts import file_sha256, summary_path, write_csv, write_json
from .results import collect_study, study_roots


ARM_ORDER = ("target_latent", "base", "typicality_no_physiology", "typicality")
ARM_LABELS = {"target_latent": "Target + latent", "base": "CFO",
              "typicality_no_physiology": "Typicality, no physiology", "typicality": "CFO + typicality"}
METRICS = ("d_z", "decoder_latent_rmse", "e_rec", "delta_dec", "e_cf",
           "delta_over_reconstruction", "cf_minus_reconstruction", "paired_discrepancy_over_tau")
BASE_METRICS = ("d_z", "decoder_latent_rmse", "e_rec", "delta_dec", "e_cf")
IDENTITY = ("task", "direction", "subject_id", "trial_id")
DISTANCE_DEFINITION = "d_z=RMSE(e(Hcf),e(H)) in full-trial classification embeddings"


def _number(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar, or unavailable")
    return float(value)


def _stats(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return {"n": 0, **dict.fromkeys(("min", "q05", "q25", "median", "q75", "q95", "max"))}
    return {"n": len(values), **dict(zip(("min", "q05", "q25", "median", "q75", "q95", "max"),
                                       map(float, np.quantile(values, [0, .05, .25, .5, .75, .95, 1]))))}


def _groups(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return sorted(groups.items())


def _signature(study):
    # Fold shards and relocated input paths may differ; scientific settings,
    # the complete source/data fingerprints, and cohort filters must match.
    args = {k: v for k, v in study.get("arguments", {}).items()
            if k not in ("subjects", "models_json", "model_dir", "trials_npz", "data_config")}
    return json.dumps({"arguments": args, "source_sha256": study.get("source_sha256"),
                       "dataset_sha256": study.get("dataset_sha256"),
                       "dataset_metadata": study.get("dataset_metadata"),
                       "objectives": study.get("objectives", ["base", "typicality"]),
                       "distance_definition": study.get("distance_definition"),
                       "typicality_definition": study.get("typicality_definition"),
                       "typicality_quantile": study.get("typicality_quantile"),
                       "target_probability": study.get("target_probability")}, sort_keys=True)


def collect_audit(roots):
    roots = [p.resolve() for p in study_roots(roots)]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate study roots would double-count displacement")
    rows, folds, exclusions, inputs, studies = [], [], [], {}, []
    seen_folds, signatures = set(), {}

    def archive(path):
        path = Path(path)
        if path.is_file():
            inputs[str(path.resolve())] = file_sha256(path)

    for root in sorted(roots):
        study = json.loads((root / "study.json").read_text())
        target = study.get("target_class", 1)
        direction = study.get("direction", f"{1-target}_to_{target}")
        key = (study["task"], direction)
        signature = _signature(study)
        if key in signatures and signatures[key] != signature:
            raise ValueError("Mixed experimental protocols: audit each protocol separately")
        signatures[key] = signature
        definition = study.get("distance_definition", "")
        if not definition.startswith(DISTANCE_DEFINITION):
            raise ValueError("Unknown d_z coordinate definition; do not relabel another latent distance")
        archive(root / "study.json")
        studies.append({"root": str(root), "task": key[0], "direction": direction,
                        "objectives":study.get('objectives',['base','typicality']),
                        "arguments": study.get("arguments", {}), "distance_definition": definition,
                        "dataset_metadata": study.get("dataset_metadata", {}),
                        "source_sha256": study.get("source_sha256", {}),
                        "dataset_sha256": study.get("dataset_sha256", {})})
        collected, _, _, fold_rows = collect_study(root)
        entries = {entry["subject_id"]: entry for entry in study["folds"]}
        for fold in fold_rows:
            identity = (fold["task"], fold["direction"], fold["subject_id"])
            if identity in seen_folds:
                raise ValueError("Duplicate task/direction/subject shards")
            seen_folds.add(identity)
            decisions = fold.pop("eligibility", [])
            fold.update(n_observed=len(decisions), n_excluded=sum(not d["eligible"] for d in decisions))
            exclusions.extend(d for d in decisions if not d["eligible"])
            folds.append(fold)
            folder = root / f"subject_{fold['subject_id']}"
            for path in (folder / "fold.json", folder / "eligibility.json",
                         summary_path(folder / "observations"), summary_path(folder / "calibration/source_trials")):
                archive(path)
        for original in collected:
            row = {k: original[k] for k in (*IDENTITY, "source_class", "target_class", "objective", "status",
                                           "latent_target_success", "n_attempts", "n_error_attempts")}
            row.update(study_root=str(root), normalization=study.get("dataset_metadata", {}).get("window_normalization"),
                       metric_units="model-input units for signal RMSE; distinct frozen coordinates for each latent RMSE")
            if original["status"] == "completed":
                path = Path(original["artifact_directory"])
                result_path = path / "result.json"
                marker = json.loads((path / "complete.json").read_text())
                if marker.get("sha256", {}).get("result.json") != file_sha256(result_path):
                    raise ValueError("Committed endpoint does not authenticate result.json")
                summary = json.loads(result_path.read_text())
                checkpoint = entries[row["subject_id"]].get("sha256")
                if checkpoint and summary.get("checkpoint_sha256") != checkpoint:
                    raise ValueError("Endpoint checkpoint disagrees with its frozen fold")
                if summary.get("decoded_distance_reference") != "original_reconstruction":
                    raise ValueError("Decoded distance uses a different reference")
                decoded = summary["decoded_trials"][summary["report_output"]]
                row.update(artifact_directory=str(path), result_sha256=file_sha256(result_path),
                           checkpoint_sha256=checkpoint, selected_step=summary.get("selected_step"),
                           seed=summary.get("seed"), required_target_probability=summary.get("required_target_probability"),
                           typicality_quantile=study.get("typicality_quantile",study.get("arguments",{}).get("typicality_quantile")),
                           steps_completed=summary.get("steps_completed"), stop_reason=summary.get("stop_reason"),
                           report_output=summary["report_output"],
                           d_z=_number(summary.get("d_z"), "d_z"),
                           decoder_latent_rmse=_number(summary.get("decoder_latent_rmse"), "decoder_latent_rmse"))
                for field, name in (("e_rec", "original_reconstruction_mse"),
                                    ("delta_dec", "decoded_change_mse"),
                                    ("e_cf", "counterfactual_to_original_mse")):
                    mse = _number(decoded.get(name), name)
                    row[name] = mse
                    row[field] = float(np.sqrt(mse)) if mse is not None else None
                rec, edit, total = (row[name] for name in ("e_rec", "delta_dec", "e_cf"))
                row["triangle_checked"] = all(v is not None for v in (rec, edit, total))
                if row["triangle_checked"]:
                    tolerance = 1e-6 * max(1., rec, edit, total)
                    row.update(triangle_lower=abs(rec-edit), triangle_upper=rec+edit,
                               triangle_ok=abs(rec-edit)-tolerance <= total <= rec+edit+tolerance)
                row["delta_over_reconstruction"] = edit / rec if rec and edit is not None else None
                row["cf_minus_reconstruction"] = total-rec if rec is not None and total is not None else None
                row.update(discrepancy=original["discrepancy"], threshold=original["threshold"])
                penalty, tolerance = decoded.get("vcsc_counterfactual"), summary.get("physiological_tolerance")
                row["vcsc_pass"] = bool(penalty <= tolerance) if penalty is not None and tolerance is not None else None
                for p in (result_path, path / "complete.json"):
                    archive(p)
            rows.append(row)
    identities = [tuple(r[k] for k in (*IDENTITY, "objective")) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate trial/objective identities")
    rows.sort(key=lambda r: tuple(str(r[k]) for k in (*IDENTITY, "objective")))
    # Reconstruction is the same frozen starting point for paired objectives.
    baseline_mismatches = []
    for identity, group in _groups(rows, IDENTITY):
        recs = [r["e_rec"] for r in group if r.get("e_rec") is not None]
        if recs and not np.allclose(recs, recs[0], rtol=1e-5, atol=1e-7):
            baseline_mismatches.append(dict(zip(IDENTITY, identity)))
    anchors = {tuple(r[k] for k in IDENTITY): r for r in rows if r["objective"] == "base"}
    pairs = []
    for row in rows:
        base = anchors.get(tuple(row[k] for k in IDENTITY))
        if row["status"] != "completed" or not base or base["status"] != "completed":
            continue
        paired = {k: row[k] for k in (*IDENTITY, "objective")}
        for name in BASE_METRICS:
            a, b = row.get(name), base.get(name)
            paired[f"change_{name}"] = a-b if a is not None and b is not None else None
        tau = row["threshold"]
        if tau != base["threshold"]:
            raise ValueError("Paired objectives have different typicality thresholds")
        row["paired_discrepancy_over_tau"] = (row["discrepancy"]-base["discrepancy"])/tau if tau > 0 else None
        paired["paired_discrepancy_over_tau"] = row["paired_discrepancy_over_tau"]
        pairs.append(paired)
    validation = {"baseline_mismatches": baseline_mismatches,
                  "n_triangle_checked": sum(r.get("triangle_checked", False) for r in rows),
                  "n_triangle_violations": sum(r.get("triangle_ok") is False for r in rows),
                  "n_authenticated_endpoints": sum(r["status"] == "completed" for r in rows),
                  "n_missing_signal_triplets": sum(r["status"]=="completed" and
                      any(r.get(m) is None for m in ("e_rec","delta_dec","e_cf")) for r in rows),
                  "n_missing_latent_pairs": sum(r["status"]=="completed" and
                      any(r.get(m) is None for m in ("d_z","decoder_latent_rmse")) for r in rows),
                  "verification_scope": "Saved scalars and committed archive hashes; raw EEG distances were not recomputed."}
    return rows, pairs, folds, exclusions, studies, inputs, validation


def summarize(rows, keys=("task", "direction", "objective")):
    results = []
    for identity, group in _groups(rows, keys):
        row = dict(zip(keys, identity))
        row.update(n_eligible=len(group), n_completed=sum(r["status"] == "completed" for r in group),
                   n_error=sum(r["status"] == "error" for r in group), n_pending=sum(r["status"] == "pending" for r in group),
                   n_subjects=len({r["subject_id"] for r in group}),
                   n_unsuccessful_finite=sum(r["status"] == "completed" and not r["latent_target_success"] for r in group))
        for name in METRICS:
            row.update({f"{name}_{k}": v for k, v in _stats(r.get(name) for r in group).items()})
        for name in BASE_METRICS:
            values = [r[name] for r in group if r.get(name) is not None]
            row[f"{name}_exact_zero_n"] = sum(v == 0 for v in values)
            row[f"{name}_rounds_to_0_00_n"] = sum(float(f"{v:.2f}") == 0 for v in values)
        assessed = [r["vcsc_pass"] for r in group if r.get("vcsc_pass") is not None]
        row["vcsc_n_assessed"] = len(assessed)
        row["vcsc_pass_percent"] = 100 * sum(assessed)/len(assessed) if assessed else None
        row["n_e_cf_increased"] = sum(r.get("cf_minus_reconstruction", 0) is not None and
                                      r.get("cf_minus_reconstruction", 0) > 1e-6 for r in group)
        row["n_e_cf_decreased"] = sum(r.get("cf_minus_reconstruction", 0) is not None and
                                      r.get("cf_minus_reconstruction", 0) < -1e-6 for r in group)
        results.append(row)
    return results


def _fmt(value):
    return "--" if value is None else f"{value:.3g}"


def _interval(row, name):
    return f"{_fmt(row.get(name+'_median'))} [{_fmt(row.get(name+'_q25'))}, {_fmt(row.get(name+'_q75'))}]"


def _latex(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _table(output, population, name, columns, caption):
    header = [title for title, _ in columns]
    body = [[render(row) for _, render in columns] for row in population]
    row_end = " " + chr(92)*2
    write_csv(output / f"{name}.csv", [dict(zip(header, row)) for row in body], fieldnames=header)
    lines = [r"\begin{table}[t]", r"\centering\small", r"\caption{"+caption+"}",
             r"\label{tab:"+name+"}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{"+"l"*len(header)+"}", r"\hline",
             " & ".join(map(_latex, header))+row_end, r"\hline"]
    lines.extend(" & ".join(map(_latex, row))+row_end for row in body)
    lines.extend([r"\hline", r"\end{tabular}", "}", r"\end{table}"])
    (output / f"{name}.tex").write_text("\n".join(lines)+"\n")
    return header, body


def _correlations(rows):
    results = []
    for identity, group in _groups(rows, ("task", "direction", "objective")):
        for latent in ("d_z", "decoder_latent_rmse"):
            pairs = [(r.get(latent), r.get("delta_dec")) for r in group
                     if r.get(latent) is not None and r.get("delta_dec") is not None]
            x, y = np.array(pairs).T if pairs else (np.array([]), np.array([]))
            rho = float(spearmanr(x, y).statistic) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else None
            results.append(dict(zip(("task", "direction", "objective"), identity)) |
                           {"latent_metric": latent, "n_pairs": len(pairs), "spearman_rho": rho,
                            "interpretation": "within-objective descriptive association; clustered trials, no significance claim"})
    return results


def _figures(output, rows, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(ARM_ORDER, ("#667085", "#1261a0", "#a05fa5", "#14806f")))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    groups = sorted({(r["task"], r["direction"]) for r in rows})
    paths = []
    for task, direction in groups:
        selected = [r for r in rows if (r["task"], r["direction"]) == (task, direction)]
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
        for ax, metric, title in zip(axes, ("d_z", "decoder_latent_rmse", "delta_dec"),
                                      ("Classification embedding", "Decoder latent", "Decoded edit")):
            for arm in ARM_ORDER:
                values = np.sort([r[metric] for r in selected if r["objective"] == arm and r.get(metric) is not None])
                if len(values):
                    ax.step(values, np.arange(1, len(values)+1)/len(values), where="post",
                            color=colors[arm], label=ARM_LABELS[arm],
                            linestyle="--" if arm=="base" else "-", linewidth=2.2 if arm=="base" else 1.6)
            ax.set(xscale="symlog", xlabel="RMSE (linear below 1e-5; log above)", ylabel="Fraction of finite endpoints", title=title)
            ax.set_xscale("symlog", linthresh=1e-5)
            ax.set_ylim(0, 1.03)
            ax.grid(alpha=.18)
        axes[-1].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"{task.title()} {direction.replace('_to_', ' → ')}: displacement distributions", fontsize=13)
        stem = f"{task}_{direction}_distributions"
        for extension in ("png", "pdf"):
            path=output / f"{stem}.{extension}"; fig.savefig(path, dpi=180); paths.append(path.name)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10,4.2), constrained_layout=True)
        for ax, latent, title in zip(axes,("d_z","decoder_latent_rmse"),("Classification embedding", "Decoder latent")):
            for arm in ARM_ORDER:
                points=[r for r in selected if r["objective"]==arm and r.get(latent) is not None and r.get("delta_dec") is not None]
                ax.scatter([r[latent] for r in points],[r['delta_dec'] for r in points],s=19,
                           color=colors[arm],alpha=.65,label=ARM_LABELS[arm],edgecolors='none')
            ax.set_xscale('symlog',linthresh=1e-5); ax.set_yscale('symlog',linthresh=1e-5)
            ax.set(xlabel=title+' RMSE',ylabel='Decoded-edit RMSE',title=title+' versus signal edit')
            ax.grid(alpha=.18)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{task.title()} {direction.replace('_to_', ' → ')}: latent and decoded movement",fontsize=13)
        for extension in ('png','pdf'):
            path=output/f'{task}_{direction}_latent_decoded.{extension}'; fig.savefig(path,dpi=180); paths.append(path.name)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        for arm in ARM_ORDER:
            arm_rows = [r for r in selected if r["objective"] == arm and all(r.get(k) is not None for k in ("e_rec", "delta_dec", "e_cf"))]
            for ax, metric in zip(axes, ("delta_dec", "e_cf")):
                ax.scatter([r["e_rec"] for r in arm_rows], [r[metric] for r in arm_rows], s=19,
                           alpha=.65, color=colors[arm], label=ARM_LABELS[arm], edgecolors="none")
        axes[0].set(xlabel="Original → reconstruction RMSE", ylabel="Reconstruction → counterfactual RMSE",
                    title="Decoded edit versus reconstruction error", yscale="symlog")
        axes[0].set_yscale("symlog", linthresh=1e-5)
        axes[1].set(xlabel="Original → reconstruction RMSE", ylabel="Original → counterfactual RMSE",
                    title="Total error versus reconstruction error")
        for ax in axes:
            low, high = ax.get_xlim(); ax.plot([max(0,low), high], [max(0,low),high], "--", color="#777777", linewidth=1)
            ax.grid(alpha=.18)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{task.title()} {direction.replace('_to_', ' → ')}: signal distances", fontsize=13)
        stem=f"{task}_{direction}_signal_distances"
        for extension in ("png", "pdf"):
            path=output / f"{stem}.{extension}"; fig.savefig(path,dpi=180); paths.append(path.name)
        plt.close(fig)
    header, body = table
    fig, ax = plt.subplots(figsize=(14, 4.3)); ax.axis("off")
    tab=ax.table(cellText=body, colLabels=header, loc="center", cellLoc="left")
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.65)
    tab.auto_set_column_width(range(len(header)))
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#d5d9e0"); cell.set_linewidth(.5)
        if row==0: cell.set_facecolor("#e8eef5"); cell.set_text_props(weight="bold")
        elif row%2==0: cell.set_facecolor("#f5f7fa")
    ax.set_title("Table 2 · Counterfactual displacement and physiological diagnostics", loc="left", pad=18, fontsize=13)
    fig.text(.025,.015,"Distances: median [Q1, Q3]. All finite endpoints; VCSC is the archived penalty-tolerance check. Decoded validity deferred.",fontsize=9)
    for extension in ("png","pdf"):
        path=output/f"table2.{extension}"; fig.savefig(path,dpi=180); paths.append(path.name)
    plt.close(fig)
    return paths


def build_audit(roots, output, *, expected_subjects=None, reference_population=None, figures=True):
    output=Path(output); output.mkdir(parents=True, exist_ok=True)
    rows, pairs, folds, exclusions, studies, inputs, validation=collect_audit(roots)
    if expected_subjects is not None:
        if expected_subjects<1: raise ValueError("expected_subjects must be positive")
        present={(f["task"],f["direction"],f["subject_id"]) for f in folds}
        expected={(s["task"],s["direction"],i) for s in studies for i in range(expected_subjects)}
        if present-expected: raise ValueError("Subject outside expected coverage")
        validation["missing_folds"]=[list(v) for v in sorted(expected-present)]
    else: validation["missing_folds"]=[]
    population=summarize(rows)
    rank={arm:i for i,arm in enumerate(ARM_ORDER)}
    population.sort(key=lambda r:(r["task"],r["direction"],rank.get(r["objective"],99)))
    subject=summarize(rows, ("task","direction","subject_id","objective"))
    # Preserve declared zero-eligible participants rather than silently
    # dropping them from the subject-level audit.
    for fold in folds:
        if fold.get('n_eligible')==0:
            for arm in next(s for s in studies if (s['task'],s['direction'])==(fold['task'],fold['direction']))['objectives']:
                empty={k:fold[k] for k in ('task','direction','subject_id')}
                empty.update(objective=arm,n_eligible=0,n_completed=0,n_error=0,n_pending=0,n_subjects=1,n_unsuccessful_finite=0)
                for metric in METRICS:
                    empty.update({f'{metric}_{k}':v for k,v in _stats([]).items()})
                subject.append(empty)
    subject.sort(key=lambda r:(r['task'],r['direction'],r['subject_id'],r['objective']))
    paired_summary=[]
    for identity, group in _groups(pairs, ("task","direction","objective")):
        for name in (*[f"change_{m}" for m in BASE_METRICS], "paired_discrepancy_over_tau"):
            paired_summary.append(dict(zip(("task","direction","objective"),identity)) |
                                  {"metric":name, **_stats(r.get(name) for r in group)})
    correlations=_correlations(rows)
    cohort=[]
    for identity, group in _groups(folds,('task','direction')):
        task,direction=identity
        counts={name:sum(f.get(name,0) for f in group) if all(f.get(name) is not None for f in group) else None
                for name in ('n_observed','n_eligible','n_excluded')}
        cohort.append({'task':task,'direction':direction,'n_folds':len(group),**counts,
                       'n_zero_eligible_subjects':sum(f.get('n_eligible')==0 for f in group),
                       **{reason:sum(r['exclusion_reason']==reason for r in exclusions
                                     if (r['task'],r['direction'])==identity)
                          for reason in ('outside_source_class','originally_misclassified','trial_filter')}})
    reference_checks=[]
    if reference_population:
        reference_population=Path(reference_population)
        inputs[str(reference_population.resolve())]=file_sha256(reference_population)
        with reference_population.open(newline="") as handle: reference=list(csv.DictReader(handle))
        for ref in reference:
            matches=[r for r in population if r["task"]==ref["task"] and r["objective"]==ref["objective"]
                     and r["direction"]==ref.get("direction","0_to_1")]
            if len(matches)!=1: raise ValueError("Reference table has missing or ambiguous objective groups")
            for metric, reference_metric in (("d_z","d_z"),("delta_dec","delta_dec"),
                                             ("paired_discrepancy_over_tau","paired_delta_d_c1_over_tau")):
                for stat in ("median","q25","q75","n"):
                    key=f"{metric}_{stat}"; actual=matches[0][key]; reference_key=f"{reference_metric}_{stat}"
                    passed=actual is not None and np.isclose(actual,float(ref[reference_key]),rtol=1e-7,atol=1e-9)
                    reference_checks.append({"task":ref["task"],"objective":ref["objective"],"field":key,
                                             "archived":float(ref[reference_key]),"recomputed":actual,"passed":bool(passed)})
            actual=matches[0]['vcsc_pass_percent']; expected=float(ref['vcsc_passed_percent'])
            reference_checks.append({'task':ref['task'],'objective':ref['objective'],'field':'vcsc_pass_percent',
                                     'archived':expected,'recomputed':actual,'passed':bool(actual is not None and np.isclose(actual,expected))})
    complete=not (validation["missing_folds"] or validation['n_missing_signal_triplets'] or validation['n_missing_latent_pairs']) and all(f["status"]=="completed" for f in folds) and all(r["status"]!="pending" for r in rows)
    validation["reference_comparisons"]=len(reference_checks)
    validation["reference_mismatches"]=sum(not r["passed"] for r in reference_checks)
    validation["passed"]=not (validation["baseline_mismatches"] or validation["n_triangle_violations"] or validation["reference_mismatches"])
    for name,data in (("trial_displacements",rows),("population_displacements",population),("subject_displacements",subject),
                      ("paired_displacements",pairs),("paired_summary",paired_summary),("latent_decoded_correlations",correlations),
                      ("cohort_folds",folds),("cohort_summary",cohort),("exclusions",exclusions),("reference_checks",reference_checks)):
        write_csv(output/f"{name}.csv",data)
    identity_columns=[("Task",lambda r:r["task"].title()),("Objective",lambda r:ARM_LABELS.get(r["objective"],r["objective"])),
                      ("N finite/eligible",lambda r:f"{r['d_z_n']}/{r['n_eligible']}")]
    if len({r["direction"] for r in population})>1:
        identity_columns.insert(1,("Direction",lambda r:r["direction"]))
    table=_table(output,population,"table2",identity_columns+[
        ("d_z",lambda r:_interval(r,"d_z")),("Decoded edit",lambda r:_interval(r,"delta_dec")),
        ("VCSC (%)",lambda r:"--" if r["vcsc_pass_percent"] is None else f"{r['vcsc_pass_percent']:.2f}"),
        ("Paired D/tau",lambda r:_interval(r,"paired_discrepancy_over_tau"))],
        "Counterfactual displacement and archived physiological diagnostics. Distances are median [Q1, Q3]. N gives finite classification-displacement endpoints over eligible trials; each metric's available count is exported separately. VCSC is the raw-penalty tolerance check among assessed endpoints. Paired discrepancy changes use each trial's fold threshold and CFO baseline. Decoded classification validity is deferred.")
    _table(output,population,"table2_signal_errors",identity_columns+[
        ("Decoder latent",lambda r:_interval(r,"decoder_latent_rmse")),("Reconstruction error",lambda r:_interval(r,"e_rec")),
        ("Decoded edit",lambda r:_interval(r,"delta_dec")),("Total signal error",lambda r:_interval(r,"e_cf")),
        ("Edit / reconstruction",lambda r:_interval(r,"delta_over_reconstruction"))],
        "Displacement audit: decoder-latent RMSE is distinct from classification-embedding displacement. Signal errors refer respectively to original-to-reconstruction, reconstruction-to-counterfactual, and original-to-counterfactual. Distances are not additive.")
    image_paths=_figures(output,rows,table) if figures else []
    result={"schema_version":1,"complete":complete,"validation":validation,"n_endpoints":len(rows),
            "n_unique_trials":len({tuple(r[k] for k in IDENTITY) for r in rows}),"n_folds":len(folds),
            "population":population,"cohort":cohort,"studies":studies,"input_sha256":inputs,"figures":image_paths,
            "audit_source_sha256":{p.name:file_sha256(p) for p in (Path(__file__),Path(__file__).with_name('results.py'))},
            "decoded_validity":"deferred; no decoded prediction or re-encoding analysis performed",
            "repair_control":"optional, not required for this displacement audit; no new control generated"}
    write_json(output/"audit.json",result)
    _write_report(output,result,rows,table)
    return result


def _write_report(output,result,rows,table):
    header,body=table
    findings=[]
    for (task,direction), group in _groups(rows,('task','direction')):
        arms={r['objective']:r for r in result['population'] if (r['task'],r['direction'])==(task,direction)}
        protocols=next(s for s in result['studies'] if (s['task'],s['direction'])==(task,direction))
        args=protocols['arguments']
        findings.append(f"**{task.title()} ({direction}):** the archived typicality quantile is {_fmt(args.get('typicality_quantile'))}, the target-confidence threshold is {_fmt(args.get('target_probability'))}, and normalization is {protocols['dataset_metadata'].get('window_normalization','unavailable')}.")
        if 'typicality' in arms:
            full=arms['typicality']
            findings.append(f"The full objective has median classification-embedding displacement {_fmt(full['d_z_median'])}, decoder-latent displacement {_fmt(full['decoder_latent_rmse_median'])}, and decoded edit {_fmt(full['delta_dec_median'])}. Its median original reconstruction error is {_fmt(full['e_rec_median'])}; median total counterfactual error is {_fmt(full['e_cf_median'])}. Total signal error increases by more than 1e-6 for {full['n_e_cf_increased']}/{full['cf_minus_reconstruction_n']} assessed endpoints.")
        base={tuple(r[k] for k in IDENTITY):r for r in group if r['objective']=='base'}
        paired=[(r,base[tuple(r[k] for k in IDENTITY)]) for r in group if r['objective']=='typicality'
                and tuple(r[k] for k in IDENTITY) in base and r['status']=='completed'
                and base[tuple(r[k] for k in IDENTITY)]['status']=='completed']
        if paired:
            dz=_stats(a.get('d_z')-b.get('d_z') for a,b in paired if a.get('d_z') is not None and b.get('d_z') is not None)
            dd=_stats(a.get('delta_dec')-b.get('delta_dec') for a,b in paired if a.get('delta_dec') is not None and b.get('delta_dec') is not None)
            identical=sum(a.get('delta_dec') is not None and a.get('delta_dec')==b.get('delta_dec') for a,b in paired)
            findings.append(f"Relative to the paired CFO endpoint, the full typicality objective has median classification-displacement change {_fmt(dz['median'])} and median decoded-edit change {_fmt(dd['median'])} ({len(paired)} pairs). Decoded-edit distances are exactly equal in {identical}/{len(paired)} pairs. Paired medians are not differences between the separate arm medians.")
        if 'typicality' in arms and 'typicality_no_physiology' in arms:
            full,noph=arms['typicality'],arms['typicality_no_physiology']
            findings.append(f"Without the physiology term, median classification-embedding displacement is {_fmt(noph['d_z_median'])}, compared with {_fmt(full['d_z_median'])} for the full objective; the corresponding decoded edits are {_fmt(noph['delta_dec_median'])} and {_fmt(full['delta_dec_median'])}. Thus classification proximity and signal proximity rank these objectives differently.")
    exact_zero=sum(r.get('delta_dec')==0 for r in rows)
    rounded_zero=sum(r.get('delta_dec') is not None and float(f"{r['delta_dec']:.2f}")==0 for r in rows)
    findings.append(f"Across all available endpoints, {exact_zero} decoded-edit distances are exactly zero, while {rounded_zero} round to 0.00 at two decimal places.")
    (output/'paper_text.md').write_text('\n\n'.join(findings)+'\n')
    # Plain paragraphs can be included in an appendix without extra packages.
    (output/'paper_text.tex').write_text('\n\n'.join(_latex(p.replace('**','')) for p in findings)+'\n')
    lines=["# Table 2 displacement audit", "",f"Audited {result['n_endpoints']} eligible objective endpoints from {result['n_unique_trials']} trial identities across {result['n_folds']} task/direction/fold shards.","",
           "The original archived protocol is retained, including its source-calibrated typicality quantile. This is a saved-scalar audit, not a regeneration of EEG or independent validation of the original signals. Decoded-validity evaluation is deferred pending the reconstruction change that adds subject information.","",
           "## Cohort", "",
           *[f"- {c['task'].title()}, {c['direction']}: {c['n_observed']} task-trial observations, {c['n_eligible']} eligible; {c['outside_source_class']} outside the source class and {c['originally_misclassified']} originally misclassified. {c['n_zero_eligible_subjects']} folds have no eligible trials and remain in the subject-level export." for c in result['cohort']],"",
           "## Main findings", "", *[p+'\n' for p in findings],
           "## Table 2", "", " | ".join(header)," | ".join(["---"]*len(header)), *[" | ".join(row) for row in body],"",
           "## Signal-error comparison", ""]
    for r in result["population"]:
        lines.append(f"- {r['task'].title()}, {ARM_LABELS.get(r['objective'],r['objective'])}: original→reconstruction {_interval(r,'e_rec')}; reconstruction→counterfactual {_interval(r,'delta_dec')}; original→counterfactual {_interval(r,'e_cf')}. Median edit/reconstruction ratio {_fmt(r['delta_over_reconstruction_median'])}. Total error increased by more than 1e-6 for {r['n_e_cf_increased']}/{r['cf_minus_reconstruction_n']} assessed endpoints.")
    lines.extend(["","## Interpretation and limits","",
        "Classification-embedding RMSE and decoder-latent RMSE measure changes in different coordinates. Neither is a calibrated substitute for signal displacement. Compare objectives within each metric rather than comparing their absolute latent and signal magnitudes.","",
        "The three signal distances distinguish initial reconstruction error from the additional decoded edit and total error relative to the input. They obey triangle bounds rather than an additive decomposition. A smaller total input error does not establish that a counterfactual explains a class decision; a larger one does not establish physiological implausibility.","",
        "All finite completed endpoints enter distance distributions, including unsuccessful latent target attempts. Errors and pending attempts remain in the eligible cohort and have unavailable distances. Correlations are descriptive within-objective Spearman correlations, with no independence or significance claim across clustered trials. Subject-level and paired summaries accompany the population table.","",
        "Small values are shown with significant digits rather than being silently rounded to 0.00. Exact-zero and two-decimal-zero counts appear in population_displacements.csv. The optional target-free physiological control would address the reason for movement, but is not a prerequisite for this audit.","","## Verification","",
        f"Authenticated endpoints: {result['validation']['n_authenticated_endpoints']}. Triangle checks: {result['validation']['n_triangle_checked']}; violations: {result['validation']['n_triangle_violations']}. Reconstruction mismatches across paired arms: {len(result['validation']['baseline_mismatches'])}. Archived-summary comparisons: {result['validation']['reference_comparisons']}; mismatches: {result['validation']['reference_mismatches']}.","",
        "Input hashes, frozen checkpoint identities, code/data fingerprints, normalization metadata, selected endpoint paths, and cohort exclusions are retained in audit.json and the CSV exports. Checkpoint hashes are recorded from the archive; checkpoint weights and raw EEG were not loaded. Current code fingerprints are not asserted to equal the historical generator.",""])
    case=next((r for r in rows if (r['task'],r['subject_id'],r['trial_id'],r['objective'])==('arousal',0,16,'typicality')),None)
    if case:
        lines.extend(["## Archived case-study correction", "",f"Arousal subject 0, trial 16, typicality arm: E_rec = {case['e_rec']:.6f}, Delta_dec = {case['delta_dec']:.6f}, E_cf = {case['e_cf']:.6f} (RMSE). Their stored MSEs are {case['original_reconstruction_mse']:.6f}, {case['decoded_change_mse']:.6f}, and {case['counterfactual_to_original_mse']:.6f}. Use these replacements only for this verified trial identity.", ""])
    (output/"report.md").write_text("\n".join(lines))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies",nargs="+",type=Path)
    parser.add_argument("--out-dir",required=True,type=Path)
    parser.add_argument("--expected-subjects",type=int)
    parser.add_argument("--reference-population",type=Path)
    parser.add_argument("--no-figures",action="store_true")
    parser.add_argument("--require-complete",action="store_true")
    args=parser.parse_args(argv)
    result=build_audit(args.studies,args.out_dir,expected_subjects=args.expected_subjects,
                       reference_population=args.reference_population,figures=not args.no_figures)
    print(json.dumps({"n_endpoints":result["n_endpoints"],"complete":result["complete"],"validation":result["validation"]},indent=2))
    return int(not result["validation"]["passed"] or (args.require_complete and not result["complete"]))


if __name__=="__main__":
    raise SystemExit(main())
