"""Fold-0 training packages for the BDT configuration suite.

Runs vvv.ml.train.load_split_data once per category (the exact production events, weights and
splits) and saves the arrays, so every variant on NRP trains on identical inputs without the
skims. Layout: OUT/<category>/{train,monitor,test}.npz with X__<column>, w, w_diag, w_phys, y,
event (test also sample), plus OUT/<category>/meta.json.

Usage: pixi run python scripts/bdt_suite/make_packages.py CONFIG OUT --categories c1 c2 [--fold 0]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from vvv.ml.train import load_split_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--fold", type=int, default=0)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    for category in args.categories:
        d = args.out / category
        if (d / "meta.json").exists():
            print(f"{category}: exists, skipped")
            continue
        d.mkdir(parents=True, exist_ok=True)
        data = load_split_data(cfg, category, args.fold)
        for split, s in data.items():
            arrays = {f"X__{c}": v for c, v in s["X"].items()}
            arrays.update({k: s[k] for k in ("w", "w_diag", "w_phys", "y", "event")})
            if split == "test":
                arrays["sample"] = s["sample"].astype(str)
            np.savez(d / f".{split}.npz", **arrays)
            (d / f".{split}.npz").rename(d / f"{split}.npz")
        meta = {"config": str(args.config), "category": category, "fold": args.fold,
                "n": {split: int(len(s["y"])) for split, s in data.items()}}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{category}: {meta['n']}", flush=True)


def load_package(d, splits=("train", "monitor", "test")):
    """The load_split_data dict back from a package directory."""
    out = {}
    for split in splits:
        with np.load(Path(d) / f"{split}.npz", allow_pickle=False) as z:
            out[split] = {"X": {k[3:]: z[k] for k in z.files if k.startswith("X__")},
                          **{k: z[k] for k in z.files if not k.startswith("X__")}}
    return out


if __name__ == "__main__":
    main()
