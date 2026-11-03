"""Skim prescales {sample: {category: P}} so that each (sample, category) keeps about TARGET events.

Training reads at most entries_per_sample (train) + 2 x eval_entries_per_sample (monitor, test)
per sample; the expected counts come from the legacy nominal trees.

Usage: python scripts/make_skim_prescales.py OUT.json [--target 2500000]
"""

import argparse
import glob
import json
import math
from pathlib import Path

import uproot

LEGACY = Path(__file__).resolve().parents[2] / "CMSSW_16_1_0_pre4/src/ScoutingVVVTools/dataset_hybrid_leptonic_jec_c"
TREES = ("boosted3", "boosted2_resolved", "boosted2_unresolved", "boosted2_leptonic_1lep", "boosted2_leptonic_2lep")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--target", type=float, default=2.5e6)
    args = p.parse_args()
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
    prescales = {s: {t: max(1, math.ceil(n / args.target)) for t, n in c.items()} for s, c in counts.items()}
    Path(args.out).write_text(json.dumps({"target": args.target, "counts": counts, "prescale": prescales}, indent=1))
    kept = sum(n / prescales[s][t] for s, c in counts.items() for t, n in c.items())
    print(f"wrote {args.out}: {len(prescales)} samples, about {kept / 1e6:.0f}M events kept "
          f"of {sum(sum(c.values()) for c in counts.values()) / 1e6:.0f}M")


if __name__ == "__main__":
    main()
