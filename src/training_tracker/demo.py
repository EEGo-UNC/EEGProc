"""Generate deterministic, explicitly synthetic LOSO bundles without training."""
from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

from .core import fingerprint, load_json, write_json


def create_demo(output: Path, campaigns: list[Path] | None = None) -> list[Path]:
    if campaigns is None:
        root = Path(__file__).resolve().parents[2]
        campaigns = [root / "experiments" / name / "campaign.json" for name in
                     ("dreamer_3dcnn", "dreamer_3dcnn_transformer")]
    # Read before creating output; refuse to mix fake data into an existing store.
    configs = [load_json(path) for path in campaigns]
    if any(config["objective"]["metric"] != "trial_balanced_accuracy" for config in configs):
        raise ValueError("This LOSO demo expects campaigns with objective trial_balanced_accuracy.")
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for model_index, campaign in enumerate(configs):
        folder = output / f"model-{model_index + 1}"
        campaign = {**campaign, "campaign": "SYNTHETIC-" + campaign["campaign"]}
        campaign_path = folder / "campaign.json"
        write_json(campaign_path, campaign)
        paths.append(campaign_path)
        for target in campaign.get("targets", [campaign["defaults"]["target"]]):
            config = {**campaign["defaults"], "target": target}
            run_id = f"SYNTHETIC-{target}-baseline"
            run_dir = folder / "runs" / run_id
            rng = random.Random(42)
            users = []
            for subject in range(1, 24):
                score = min(0.98, max(0.05, 0.60 + rng.gauss(0, 0.13) + model_index * 0.035))
                users.append({"subject_id": subject, "fold": subject,
                              "evaluation_level": "trial", "n_trials": 18,
                              "trial_balanced_accuracy": score,
                              "window_balanced_accuracy": max(0, score - 0.04),
                              "balanced_accuracy": score})
            values = [row["trial_balanced_accuracy"] for row in users]
            summary = {"balanced_accuracy": statistics.mean(values)}
            spread = {"balanced_accuracy": statistics.pstdev(values)}
            write_json(run_dir / "config.json", config)
            write_json(run_dir / "manifest.json", {
                "schema_version": 1, "campaign": campaign["campaign"], "run_id": run_id,
                "status": "completed", "stage": "full", "synthetic": True,
                "config_fingerprint": fingerprint(config),
            })
            write_json(run_dir / "result.json", {
                "synthetic": True, "evaluation_level": "trial", "best_config_index": 0,
                "config_results": [{"trial_mean_scores": summary, "trial_std_scores": spread,
                                    "window_mean_scores": {"balanced_accuracy": summary["balanced_accuracy"] - 0.04},
                                    "window_std_scores": spread,
                                    "fold_metrics": users}],
                "user_metrics": users,
                "run_metadata": {"synthetic": True, "model_type": config.get("model_type", "3dcnn")},
            })
    (output / "README.md").write_text(
        "# SYNTHETIC tracker demonstration\n\nAll scores are invented using seed 42. "
        "These are not training results and must not be used as scientific evidence.\n",
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory; never an existing run store.")
    parser.add_argument("--campaign", type=Path, action="append", help="Source campaign; defaults to both DREAMER models in a source checkout.")
    args = parser.parse_args(argv)
    try:
        for path in create_demo(args.output_dir, args.campaign):
            print(path)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
