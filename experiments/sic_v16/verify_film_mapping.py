"""Check source film metadata/reference mappings without equating trial IDs with films."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
TITLE_COLUMNS = ("film_title", "movie_title", "video_title", "stimulus_name", "film_name")
ID_COLUMNS = ("film_id", "movie_id", "video_id", "stimulus_id")


def read_reference(path):
    """Require explicit source trial IDs and citations; never infer index offsets."""
    if not path.is_file():
        return {}
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"trial_id", "film_title", "source_reference"} <= set(reader.fieldnames or []):
            raise ValueError("Reference needs trial_id, film_title, source_reference columns.")
        for row in reader:
            trial = int(row["trial_id"])
            if trial in result or not row["film_title"].strip() or not row["source_reference"].strip():
                raise ValueError("Duplicate trial or missing film title/reference.")
            result[trial] = dict(film_title=row["film_title"].strip(), source_reference=row["source_reference"].strip())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", required=True, type=Path)
    p.add_argument("--reference-map", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    import pandas as pd
    from permutation_audit import digest
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    columns = pd.read_csv(args.source_csv, nrows=0).columns.tolist()
    if not {"subject_id", "trial_id", "segment", "sample_idx"} <= set(columns):
        raise ValueError("Source CSV lacks trial/segment/sample identifiers.")
    metadata = [c for c in (*TITLE_COLUMNS, *ID_COLUMNS) if c in columns]
    selected = ["subject_id", "trial_id", "segment", "sample_idx", *metadata]
    groups = {}
    for chunk in pd.read_csv(args.source_csv, usecols=selected, chunksize=250000, dtype=str, keep_default_na=False):
        for (sid, tid), frame in chunk.groupby(["subject_id", "trial_id"], sort=False):
            key = (int(sid), int(tid))
            row = groups.setdefault(key, dict(values={c:set() for c in metadata}, missing={c:0 for c in metadata}, stimulus_samples=0, baseline_samples=0))
            stimulus = frame.segment.str.strip().str.lower().isin(["stimulus", "stimuli"])
            row["stimulus_samples"] += int(stimulus.sum())
            row["baseline_samples"] += int(frame.segment.str.strip().str.lower().isin(["baseline", "base"]).sum())
            for col in metadata:
                values = frame.loc[stimulus, col].str.strip()
                row["missing"][col] += int((values == "").sum())
                row["values"][col].update(v for v in values if v)
    subjects = sorted({sid for sid, _ in groups})
    if len(subjects) != 23 or any(len([t for s,t in groups if s==sid]) != 18 for sid in subjects):
        raise ValueError("Expected 23 source subjects with 18 trials each.")
    reference_path = args.reference_map or args.source_csv.with_name("dreamer_film_mapping.csv")
    reference = read_reference(reference_path)
    trial_ids = sorted({t for _,t in groups})
    problems = []
    if reference and set(reference) != set(trial_ids):
        problems.append("Reference trial IDs do not exactly match source trial IDs.")
    mappings = []
    for si, sid in enumerate(subjects):
        tids = sorted(t for s,t in groups if s==sid)
        for ti, tid in enumerate(tids):
            group = groups[sid,tid]
            values = {c: sorted(v) for c,v in group["values"].items()}
            for col in metadata:
                if len(values[col]) != 1 or group["missing"][col]:
                    problems.append(f"Missing/inconsistent {col} at subject {sid}, trial {tid}.")
            titles = {v.casefold() for c in TITLE_COLUMNS for v in values.get(c,[])}
            if len(titles)>1:
                problems.append(f"Conflicting title columns at subject {sid}, trial {tid}.")
            if tid in reference and titles and reference[tid]["film_title"].casefold() not in titles:
                problems.append(f"Reference/title mismatch at subject {sid}, trial {tid}.")
            mappings.append(dict(array_subject=si,array_trial=ti,source_subject=sid,source_trial=tid,
                source_metadata=values, reference=reference.get(tid),
                stimulus_samples=group["stimulus_samples"], baseline_samples=group["baseline_samples"]))
    for tid in trial_ids:
        for col in metadata:
            values = {v for (s,t),g in groups.items() if t==tid for v in g["values"][col]}
            if len(values)>1:
                problems.append(f"Trial {tid} has different {col} across subjects.")
    title_available = any(c in columns for c in TITLE_COLUMNS)
    status = "inconsistent" if problems else "source_titles_consistent" if title_available else "reference_supplied_only" if reference else "film_identity_unresolved"
    report = dict(status=status,source_csv=str(args.source_csv), source_sha256=digest(args.source_csv),
        source_columns=columns,film_metadata_columns=metadata, reference_path=str(reference_path),
        reference_sha256=digest(reference_path) if reference else None, problems=problems,mappings=mappings,
        limitations=["Trial index, duration and shared rating patterns are not proof of film identity.",
                     "Source CSV titles and supplied reference citations are checked for consistency, not independently authenticated.",
                     "Without source film titles or an authoritative trial-to-film reference, named films remain unresolved."])
    (output/"film_mapping.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (output/"film_mapping_reference_template.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(["trial_id","film_title","source_reference"])
        writer.writerows((tid,"","") for tid in trial_ids)
    archive=output/"film-mapping-diagnostics.tar.gz"
    with tarfile.open(archive,"w:gz") as bundle:
        for name in ("film_mapping.json","film_mapping_reference_template.csv"):
            bundle.add(output/name,arcname=name)
    print(f"Film mapping: {status}; Download: {archive}",flush=True)


if __name__=="__main__": main()
