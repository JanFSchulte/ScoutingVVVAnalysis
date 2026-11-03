"""Skim prescales {sample: {category: P}} from what the training draws from each sample.

For every (category, sample) the training takes, from the 3 train folds, either its share
of the class budget (train_events_per_class, water-filled over the class's samples by
expected yield xsec * N / raw_entries, as legacy) or at most entries_per_sample, and at
most eval_entries_per_sample from each of the monitor and test folds (entries_per_sample_by_class
overrides the per-sample cap for a class). The prescale keeps
`margin` times the larger of those needs, so a prescaled skim gives the training the same
events a full skim would (up to the margin). `--class-factor QCD=3` keeps 3x more of a
class, e.g. to raise its training budget later.

Usage: python scripts/make_skim_prescales.py OUT.json --training-config configs/training/production.json
"""

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import uproot
import yaml

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO.parent / "CMSSW_16_1_0_pre4/src/ScoutingVVVTools/dataset_hybrid_leptonic_jec_c"
TREES = ("boosted3", "boosted2_resolved", "boosted2_unresolved", "boosted2_leptonic_1lep", "boosted2_leptonic_2lep")
TRAIN_FRACTION = 3 / 5  # train folds of a 5-fold rotation
FOLD_FRACTION = 1 / 5


def legacy_counts():
    counts = {}
    for group in ("signal", "bkg"):
        for path in sorted(glob.glob(str(LEGACY / group / "*.root"))):
            stem = Path(path).stem
            sample = stem.rsplit("_", 1)[0] if stem.rsplit("_", 1)[-1].isdigit() else stem
            f = uproot.open(path)
            for t in TREES:
                if t in f:
                    counts.setdefault(sample, {}).setdefault(t, 0)
                    counts[sample][t] += f[t].num_entries
    return counts


def water_fill(budget, caps, shares):
    alloc = np.zeros(len(caps))
    free = np.ones(len(caps), dtype=bool)
    remaining = float(budget)
    while remaining > 0 and free.any():
        share = np.where(free, shares, 0.0)
        if share.sum() <= 0:
            break
        trial = remaining * share / share.sum()
        over = free & (alloc + trial >= caps)
        if not over.any():
            alloc += trial
            break
        remaining -= (caps[over] - alloc[over]).sum()
        alloc[over] = caps[over]
        free &= ~over
    return alloc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--training-config", default=str(REPO / "configs/training/production.json"))
    p.add_argument("--margin", type=float, default=1.5)
    p.add_argument("--class-factor", action="append", default=[], help="CLASS=F keeps F x more of a class")
    args = p.parse_args()

    cfg = json.loads(Path(args.training_config).read_text())
    samples = yaml.safe_load((REPO / "data/samples.yaml").read_text())["samples"]
    factors = {c: float(f) for c, f in (x.split("=") for x in args.class_factor)}
    counts = legacy_counts()

    prescale = {}
    summary = {}
    for tree, cc in cfg["categories"].items():
        for cls, members in cfg["class_groups"].items():
            members = [s for s in members if s not in cc.get("exclude_samples", []) and tree in counts.get(s, {})]
            if not members:
                continue
            n = np.array([counts[s][tree] for s in members], dtype=np.float64)
            budget = cc.get("train_events_per_class", {}).get(cls)
            if budget:
                yields = np.array([samples[s]["xsec_fb"] * counts[s][tree] / max(samples[s]["legacy"].get("raw_entries", 1), 1)
                                   for s in members])
                need_train = water_fill(budget, n * TRAIN_FRACTION, yields)
            else:
                per_sample_cap = cc.get("entries_per_sample_by_class", {}).get(cls, cfg["entries_per_sample"])
                need_train = np.minimum(n * TRAIN_FRACTION, per_sample_cap)
            need_fold = np.minimum(n * FOLD_FRACTION, cfg["eval_entries_per_sample"])
            factor = args.margin * factors.get(cls, 1.0)
            keep_fraction = np.minimum(1.0, factor * np.maximum(need_train / (n * TRAIN_FRACTION), need_fold / (n * FOLD_FRACTION)))
            ps = np.maximum(1, np.floor(1.0 / np.maximum(keep_fraction, 1e-12))).astype(int)
            for s, pv in zip(members, ps):
                prescale.setdefault(s, {})[tree] = int(pv)
            summary.setdefault(tree, {})[cls] = {
                "available": int(n.sum()),
                "kept": int((n / ps).sum()),
                "train_used": int(need_train.sum()),
            }

    Path(args.out).write_text(json.dumps({"margin": args.margin, "class_factors": factors, "counts": counts,
                                          "prescale": prescale, "summary": summary}, indent=1))
    print(f"{'category':24s} {'class':6s} {'available':>12s} {'kept':>11s} {'train uses':>11s}")
    for tree, by_class in summary.items():
        for cls, s in by_class.items():
            print(f"{tree:24s} {cls:6s} {s['available']:12,d} {s['kept']:11,d} {s['train_used']:11,d}")
    total = sum(s["kept"] for t in summary.values() for s in t.values())
    print(f"total kept {total / 1e6:.0f}M of {sum(sum(c.values()) for c in counts.values()) / 1e6:.0f}M; wrote {args.out}")


if __name__ == "__main__":
    main()
