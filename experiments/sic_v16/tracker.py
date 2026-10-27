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
    parser.add_argument("--scope", choices=("first4", "all23"), default="first4",
                        help="first4 restricts ALL data to subjects 0-3; all23 opens the original campaigns.")
    args, remaining = parser.parse_known_args()
    if "plan" in remaining and (os.environ.get("SIC_V16_WORK_ROOT") or Path("/work/users").is_dir()):
        from storage import default_work_root, ensure_work_stores
        ensure_work_stores(ROOT, default_work_root(), migrate=False, scope=args.scope)
    codes = []
    for variant in (VARIANTS if args.variant == "all" else (args.variant,)):
        print(f"\nSICv16 / {args.scope} / {variant}", flush=True)
        folder = Path(__file__).parent / ("first4" if args.scope == "first4" else "") / variant
        codes.append(main(["--campaign", str(folder / "campaign.json"),
                           *remaining]))
    raise SystemExit(max(codes))
