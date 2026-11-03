"""Compare the ported category variables with a legacy conversion output.

Events are matched on (run, luminosityBlock, event) and compared where both
frameworks assign the event to the same category. A value agrees when
|new - legacy| <= 1e-5 * max(|legacy|, 1) (legacy stores float32).

Usage: python scripts/sync_variables.py LEGACY.root INPUT.root [INPUT.root ...] [--mc]
"""

import argparse

import awkward as ak
import numpy as np
import uproot
from coffea.nanoevents import NanoEventsFactory

from vvv.config import DEFAULT
from vvv.objects import CATEGORIES
from vvv.schema import VVVScoutingSchema
from vvv.selection import EventSelection, category_index, preselection
from vvv.variables import category_variables

TOLERANCE = 1e-5


def key(run, lumi, event):
    run, lumi, event = (np.asarray(x, np.uint64) for x in (run, lumi, event))
    return (run << np.uint64(44)) ^ (lumi << np.uint64(26)) ^ event


def main():
    p = argparse.ArgumentParser()
    p.add_argument("legacy")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--mc", action="store_true")
    args = p.parse_args()

    sel = EventSelection(DEFAULT)
    keys, cats = [], []
    values = {cat: {} for cat in CATEGORIES}
    for path in args.inputs:
        events = NanoEventsFactory.from_root({path: "Events"}, schemaclass=VVVScoutingSchema, mode="virtual").events()
        events = events[ak.to_numpy(preselection(events, DEFAULT))]
        obj = sel(events, is_mc=args.mc)[0]["nominal"]
        keys.append(key(events.run, events.luminosityBlock, events.event))
        cats.append(category_index(obj["categories"]))
        for cat in CATEGORIES:
            for name, v in category_variables(obj, cat, DEFAULT.objects).items():
                values[cat].setdefault(name, []).append(v)
    keys, cats = np.concatenate(keys), np.concatenate(cats)
    order = np.argsort(keys)

    legacy = uproot.open(args.legacy)
    n_bad_total = 0
    for i, cat in enumerate(CATEGORIES):
        tree = legacy[cat]
        names = sorted(values[cat])
        missing = [n for n in names if n not in tree.keys()]
        names = [n for n in names if n in tree.keys()]
        ref = tree.arrays(["run", "luminosityBlock", "event"] + names, library="np")
        k = key(ref["run"], ref["luminosityBlock"], ref["event"])
        m = np.isin(k, keys)
        pos = order[np.searchsorted(keys, k[m], sorter=order)]
        same = cats[pos] == i
        print(f"\n== {cat}: {same.sum()} events compared, {len(names)} variables"
              + (f", not in legacy tree: {missing}" if missing else ""))
        for name in names:
            a = ref[name][m][same].astype(np.float64)
            n = np.concatenate(values[cat][name])[pos][same].astype(np.float64)
            diff = np.abs(n - a)
            bad = diff > TOLERANCE * np.maximum(np.abs(a), 1.0)
            n_bad_total += int(bad.sum())
            flag = "" if not bad.any() else "  <-- MISMATCH"
            print(f"  {name:58s} max|diff| {diff.max() if len(diff) else 0:9.2e}  fails {int(bad.sum()):6d}{flag}")
    print("\nVARIABLES SYNC OK" if n_bad_total == 0 else f"\nVARIABLES SYNC: {n_bad_total} mismatching values")


if __name__ == "__main__":
    main()
