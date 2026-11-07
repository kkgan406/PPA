"""Frozen host + PPR fusion using audited features/native logits."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Source train/val + GenImage/CommGen15 test manifest CSV")
    p.add_argument("--cache", required=True, help="MSM + host features/native logits NPZ; see docs/cache-format.md")
    p.add_argument("--host", required=True, choices=["CLIP", "DINOv2", "CO-SPY", "Effort"])
    p.add_argument("--fusion", choices=["feature", "score"], required=True)
    p.add_argument("--output-dir", required=True, help="New/empty result directory")
    args = p.parse_args()
    from ppa.experiment import run
    run(args, args.host)


if __name__ == "__main__":
    main()
