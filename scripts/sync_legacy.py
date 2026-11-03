"""Compare category assignment and key quantities with a legacy conversion output.

Events are matched on (run, luminosityBlock, event); only legacy events whose
key appears in the given input files are compared. For MC every jet
configuration (nominal plus the legacy <tree>__<variation> trees) is checked.

Usage: python scripts/sync_legacy.py LEGACY.root INPUT.root [INPUT.root ...] [--mc]
"""

import argparse
import time

import awkward as ak
import numpy as np
import uproot
from coffea.nanoevents import NanoEventsFactory

from vvv.config import DEFAULT
from vvv.objects import CATEGORIES, value_at
from vvv.schema import VVVScoutingSchema
from vvv.selection import EventSelection, category_index, preselection

# legacy branch -> function of (objects, slot collection) giving one value per event.
# Slot branches (_1, _2) read ak8_sel3 in boosted3 and ak8_sel2 elsewhere.
COMPARE = {
    "ScoutingFatPFJetRecluster_pt_1": lambda o, s: value_at(o[s].pt, 0, np.nan),
    "ScoutingFatPFJetRecluster_pt_2": lambda o, s: value_at(o[s].pt, 1, np.nan),
    "ScoutingFatPFJetRecluster_msoftdrop_1": lambda o, s: value_at(o[s].msoftdrop, 0, np.nan),
    "ScoutingFatPFJetRecluster_isrScore_1": lambda o, s: value_at(o[s].isrScore, 0, np.nan),
    "ScoutingMET_pt": lambda o, s: o["met_pt"],
    "nScoutingFatPFJetRecluster_selected": lambda o, s: ak.num(o["ak8"]),
}


def slot_collection(category):
    return "ak8_sel3" if category == "boosted3" else "ak8_sel2"


def key(run, lumi, event):
    run, lumi, event = (np.asarray(x, np.uint64) for x in (run, lumi, event))
    return (run << np.uint64(44)) ^ (lumi << np.uint64(26)) ^ event


def main():
    p = argparse.ArgumentParser()
    p.add_argument("legacy")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--mc", action="store_true")
    args = p.parse_args()

    configurations = ("nominal",) + (DEFAULT.jet_variations if args.mc else ())
    sel = EventSelection(DEFAULT)
    keys = []
    cats = {c: [] for c in configurations}
    vals = {(c, b, cat): [] for c in configurations for b in COMPARE for cat in CATEGORIES}
    t0, n_in = time.time(), 0
    for path in args.inputs:
        events = NanoEventsFactory.from_root({path: "Events"}, schemaclass=VVVScoutingSchema, mode="virtual").events()
        n_in += len(events)
        events = events[ak.to_numpy(preselection(events, DEFAULT))]
        out, info = sel(events, is_mc=args.mc, configurations=configurations)
        print(f"{path}: {len(events)} preselected, rho ok {info['rho_ok'].mean():.5f}, "
              f"consensus {info['n_rho_consensus']}")
        keys.append(key(events.run, events.luminosityBlock, events.event))
        for c in configurations:
            cats[c].append(category_index(out[c]["categories"]))
            for b, fn in COMPARE.items():
                for cat in CATEGORIES:
                    vals[(c, b, cat)].append(ak.to_numpy(fn(out[c], slot_collection(cat))).astype(np.float64))
    dt = time.time() - t0
    print(f"processed {n_in} events, {len(configurations)} configurations, in {dt:.1f} s ({n_in / dt:.0f} ev/s)")
    keys = np.concatenate(keys)
    order = np.argsort(keys)

    legacy = uproot.open(args.legacy)
    all_ok = True
    for c in configurations:
        new_cat = np.concatenate(cats[c])
        ref_cat = np.full(len(keys), -1)
        trees = {}
        for i, cat in enumerate(CATEGORIES):
            name = cat if c == "nominal" else f"{cat}__{c}"
            t = legacy[name]
            branches = ["run", "luminosityBlock", "event"] + [b for b in COMPARE if b in t.keys()]
            arr = t.arrays(branches, library="np")
            k = key(arr["run"], arr["luminosityBlock"], arr["event"])
            m = np.isin(k, keys)
            pos = order[np.searchsorted(keys, k[m], sorter=order)]
            ref_cat[pos] = i
            trees[cat] = (arr, m, pos)

        agree = ref_cat == new_cat
        print(f"\n[{c}] per-event category agreement {agree.mean():.6f} "
              f"({(~agree).sum()} differ of {len(keys)} preselected)")
        print("  category                    legacy      new")
        for i, cat in enumerate(CATEGORIES):
            print(f"  {cat:24s} {int((ref_cat == i).sum()):9d} {int((new_cat == i).sum()):8d}")
        if (~agree).any():
            all_ok = False
            confusion = np.zeros((6, 6), dtype=int)
            np.add.at(confusion, (ref_cat + 1, new_cat + 1), 1)
            print("  confusion (rows legacy, cols new; first row/col = no category):")
            for row in confusion:
                print("   " + " ".join(f"{v:7d}" for v in row))

        worst = []
        for i, cat in enumerate(CATEGORIES):
            arr, m, pos = trees[cat]
            same = new_cat[pos] == i
            for b in COMPARE:
                if b not in arr:
                    continue
                a = arr[b][m][same].astype(np.float64)
                n = np.concatenate(vals[(c, b, cat)])[pos][same]
                ok = np.isfinite(a) & np.isfinite(n)
                if ok.any():
                    # Legacy stores float32: compare relative to max(|value|, 1).
                    rel = np.abs(n[ok] - a[ok]) / np.maximum(np.abs(a[ok]), 1.0)
                    worst.append((rel.max(), np.mean(rel > 1e-5), cat, b))
        worst.sort(reverse=True)
        mx, frac, cat, b = worst[0]
        print(f"  largest per-event difference: {mx:.2e} ({cat} {b}); "
              f"fraction above 1e-5: {frac:.5f}")
        all_ok &= mx < 1e-5
    print("\nSYNC OK" if all_ok else "\nSYNC DIFFERENCES FOUND")


if __name__ == "__main__":
    main()
