"""Compare the ttbar CR selection and branches with a legacy convert_ttbar_cr output.

Events are matched on (run, luminosityBlock, event). The selection agreement is
checked over all preselected events of the given input files; the branches of
the legacy ttbar_cr tree over the events both select. A value agrees when
|new - legacy| <= 1e-5 * max(|legacy|, 1) (legacy stores float32).

Usage: python scripts/sync_ttbar_cr.py LEGACY.root INPUT.root [INPUT.root ...] [--mc]
"""

import argparse

import awkward as ak
import numpy as np
import uproot
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoEventsFactory

from vvv.config import TTBAR_CR
from vvv.processors.ttbar_cr import TREE, TtbarCRSelection, cr_preselection, cr_variables, tree_mask
from vvv.schema import VVVScoutingSchema

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

    sel = TtbarCRSelection(TTBAR_CR)
    keys, selected, values = [], [], {}
    for path in args.inputs:
        events = NanoEventsFactory.from_root({path: "Events"}, schemaclass=VVVScoutingSchema, mode="virtual").events()
        lumi_mask = None if args.mc else LumiMask(str(TTBAR_CR.analysis.lumi_mask))
        events = events[ak.to_numpy(cr_preselection(events, TTBAR_CR, lumi_mask))]
        obj = sel(events, is_mc=args.mc)[0]["nominal"]
        print(f"{path}: {len(events)} preselected")
        keys.append(key(events.run, events.luminosityBlock, events.event))
        selected.append(ak.to_numpy(tree_mask(obj)))
        gen = events.GenPart if args.mc else None
        for name, v in cr_variables(obj, obj["met_pt"], obj["met_phi"], gen).items():
            values.setdefault(name, []).append(v)
        values.setdefault("nScoutingPrimaryVertex", []).append(ak.to_numpy(ak.num(events.ScoutingPrimaryVertex)))
    keys, selected = np.concatenate(keys), np.concatenate(selected)
    order = np.argsort(keys)

    tree = uproot.open(args.legacy)[TREE]
    names = sorted(n for n in values if n in tree.keys())
    missing = sorted(n for n in values if n not in tree.keys())
    not_ported = sorted(n for n in tree.keys() if n not in values)
    ref = tree.arrays(["run", "luminosityBlock", "event"] + names, library="np")
    k = key(ref["run"], ref["luminosityBlock"], ref["event"])
    known = np.isin(k, keys)
    pos = order[np.searchsorted(keys, k[known], sorter=order)]
    ref_selected = np.zeros(len(keys), dtype=bool)
    ref_selected[pos] = True
    agree = ref_selected == selected
    print(f"\nselection: legacy {ref_selected.sum()}, new {selected.sum()}, agreement {agree.mean():.6f} "
          f"({(~agree).sum()} differ of {len(keys)} preselected; {(~known).sum()} legacy events not in the inputs)")

    both = selected[pos]
    print(f"\n== {TREE}: {both.sum()} events compared, {len(names)} branches"
          + (f"\n   not in the legacy tree: {missing}" if missing else "")
          + (f"\n   legacy-only (not ported): {not_ported}" if not_ported else ""))
    n_bad_total = 0
    for name in names:
        a = ref[name][known][both].astype(np.float64)
        n = np.concatenate(values[name])[pos][both].astype(np.float64)
        diff = np.abs(n - a)
        bad = ~(diff <= TOLERANCE * np.maximum(np.abs(a), 1.0))
        n_bad_total += int(bad.sum())
        flag = "" if not bad.any() else "  <-- MISMATCH"
        print(f"  {name:52s} max|diff| {np.nanmax(diff) if len(diff) else 0:9.2e}  fails {int(bad.sum()):6d}{flag}")
    ok = agree.all() and n_bad_total == 0
    print("\nTTBAR CR SYNC OK" if ok else f"\nTTBAR CR SYNC: {(~agree).sum()} selection and {n_bad_total} value differences")


if __name__ == "__main__":
    main()
