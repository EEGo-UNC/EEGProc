"""Existing tracker, configured for one or all four SICv16 model variants."""
import argparse
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training_tracker.core import main

VARIANTS = ("cnn", "cnn_reconstruction", "cnn_transformer", "cnn_transformer_reconstruction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="cnn")
    args, remaining = parser.parse_known_args()
    if "plan" in remaining and (os.environ.get("SIC_V16_WORK_ROOT") or Path("/work/users").is_dir()):
        from storage import default_work_root, ensure_work_stores
        ensure_work_stores(ROOT, default_work_root(), migrate=False)
    codes = []
    for variant in (VARIANTS if args.variant == "all" else (args.variant,)):
        print(f"\nSICv16 / {variant}", flush=True)
        codes.append(main(["--campaign", str(Path(__file__).with_name(variant) / "campaign.json"),
                           *remaining]))
    raise SystemExit(max(codes))
