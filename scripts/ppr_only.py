"""Source-only standalone PPR classifier; cache produced by extract.py."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="CSV with source train/val and target test observations")
    p.add_argument("--cache", required=True, help="Content-bound MSM1536 NPZ")
    p.add_argument("--output-dir", required=True, help="New/empty result directory")
    args = p.parse_args()
    from ppa.experiment import run
    run(args)


if __name__ == "__main__":
    main()
