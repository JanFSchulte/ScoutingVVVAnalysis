"""Cross-check of scripts/make_syst_yields.py against the legacy producers on the same events.

The legacy theory_syst.py, pileup_syst.py and jet_syst_common/jet_variation_common.py are imported
from the legacy checkout (read only, no bytecode written; their module-level config reading uses the
legacy config.json files, of which only min_region_events and pdf_uncertainty_method matter here).
For every sample and category, the events of the fit-region skims of the same `vvv run` pass
(--skim-cut, --skim-theory, --skim-variations, with proba_<class>) are accumulated as the legacy
_compute_sums / accumulate_weighted_sr_yields do: the legacy threshold mask on the msoftdrop_1 window,
the legacy region_mask of each SR CSV row on the class probabilities (each configuration's own scores
and mass), the weights x pileup ratio and x generator-normalised theory member. The legacy
_entry / ratio_entry functions then make the entries (norm = 1 / lumi, generator sums = 1: the skim
columns are normalised already), which are compared field by field with the JSON files.

Usage: pixi run python scripts/sync_syst_yields.py SYST_DIR --skim-dir DIR --signal-regions CAT=CSV [...]
           --training-config JSON [--samples S ...] [--lumi 255.8] [--legacy-repo DIR]
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # never write __pycache__ into the legacy checkout
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MASS = "ScoutingFatPFJetRecluster_msoftdrop_1"
CLASSES = ["VVV", "VH", "Top", "VV", "VJets", "QCD"]
JET = ("jes", "jer", "jms", "jmr")
# Fields that differ by construction (systematics.py docstring): raw vs effective event counts.
SKIP = {"n_events"}


def legacy_modules(repo):
    sel = Path(repo) / "selections"
    for d in ("BDT", "jet_syst_common", "theory_weights", "pileup_syst"):
        sys.path.insert(0, str(sel / d))
    return (importlib.import_module("jet_variation_common"), importlib.import_module("theory_syst"),
            importlib.import_module("pileup_syst"))


def read_skim(skim_dir, sample, category, config):
    import pyarrow.dataset as ds

    path = Path(skim_dir) / sample / category / config
    if not path.exists():
        return None
    t = ds.dataset(path).to_table()
    return {c: t[c].to_numpy() for c in t.column_names}


def compare(new, ref, where, report, tol=1e-9):
    """Recursive field comparison; returns the largest relative difference."""
    worst = 0.0
    for k, v in ref.items():
        if k in SKIP or k in ("regions", "mass_range", "mass_bins"):
            continue
        if k not in new:
            report.append(f"{where}: field {k} missing")
            continue
        a, b = new[k], v
        if isinstance(b, bool) or isinstance(b, str):
            if a != b:
                report.append(f"{where}: {k} = {a!r}, legacy {b!r}")
            continue
        a, b = (list(a), list(b)) if isinstance(b, list) else ([a], [b])
        for x, y in zip(a, b):
            d = abs(x - y) / max(abs(y), 1e-300) if y != 0 else abs(x)
            worst = max(worst, d)
            if d > tol:
                report.append(f"{where}: {k} = {x!r}, legacy {y!r} (rel {d:.2e})")
    for k in new:
        if k not in ref and k not in ("regions", "sr_csv_hash", "min_region_events", "mass_fit"):
            report.append(f"{where}: extra field {k}")
    return worst


def main():
    import numpy as np
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("syst_dir", type=Path)
    p.add_argument("--skim-dir", type=Path, required=True)
    p.add_argument("--signal-regions", action="append", required=True, metavar="CAT=CSV")
    p.add_argument("--training-config", type=Path, required=True)
    p.add_argument("--samples", nargs="*")
    p.add_argument("--lumi", type=float, default=255.8)
    p.add_argument("--legacy-repo", type=Path, default=Path(__file__).resolve().parents[2] / "CMSSW_16_1_0_pre4" / "src"
                   / "ScoutingVVVTools")
    p.add_argument("--output", type=Path, help="JSON report")
    args = p.parse_args()

    jvc, th, pu = legacy_modules(args.legacy_repo)
    from vvv.samples import load_samples

    catalogue = load_samples()
    tcfg = json.loads(args.training_config.read_text())
    files = {s: json.loads((args.syst_dir / f"{s}_syst_yields.json").read_text())
             for s in ("theory", "pileup") + JET}
    samples = args.samples or sorted(files["pileup"])
    norm = 1.0 / args.lumi
    report, worst, n_cmp = [], {}, {}
    for item in args.signal_regions:
        cat, csv_path = item.split("=", 1)
        window = tcfg["categories"][cat]["thresholds"][MASS]
        sr_df = pd.read_csv(csv_path)
        axes = jvc.detect_signal_region_axes(sr_df, CLASSES)
        bin_ids = [int(round(float(v))) for v in sr_df["bin_index"]]

        def masks(ev):
            """[inclusive, SR windows...] event masks with the legacy semantics."""
            proba = np.column_stack([ev[f"proba_{c}"] for c in CLASSES])
            in_win = jvc.threshold_mask({MASS: ev[MASS]}, {MASS: window})
            return [in_win] + [in_win & jvc.region_mask(proba, sr_df.iloc[j], axes, CLASSES)
                               for j in range(len(bin_ids))]

        for sample in samples:
            nom = read_skim(args.skim_dir, sample, cat, "nominal")
            if nom is None or sample not in files["pileup"] or cat not in files["pileup"][sample]:
                if nom is not None and nom["weight"][masks(nom)[0]].sum() > 0:
                    report.append(f"{sample}/{cat}: events in the skim but no JSON entry")
                continue
            ms = masks(nom)
            w = nom["weight"].astype(np.float64)
            slot = lambda m, ww: {"sum": float(ww[m].sum()), "sumw2": float((ww[m] ** 2).sum()),  # noqa: E731
                                  "n_events": int(m.sum())}

            def check(syst, entries, new):
                incl, regions = entries[0], entries[1:]
                key = f"{sample}/{cat}/{syst}"
                d = compare(new, incl, key + "/inclusive", report)
                for bid, r in zip(bin_ids, regions):
                    d = max(d, compare(new["regions"][str(bid)], r, f"{key}/SR{bid}", report))
                worst[syst] = max(worst.get(syst, 0.0), d)
                n_cmp[syst] = n_cmp.get(syst, 0) + 1 + len(regions)

            # jet variations: each configuration's own skim (selection, scores, mass)
            for syst in JET:
                var = {d: read_skim(args.skim_dir, sample, cat, f"{syst}_{d}") for d in ("up", "down")}
                vm = {d: (masks(v) if v is not None else [np.zeros(0, bool)] * len(ms)) for d, v in var.items()}
                vw = {d: (v["weight"].astype(np.float64) if v is not None else np.zeros(0)) for d, v in var.items()}
                up_key, down_key = f"{syst}_up", f"{syst}_down"
                incl = jvc.ratio_entry(slot(ms[0], w), slot(vm["up"][0], vw["up"]), slot(vm["down"][0], vw["down"]),
                                       up_key, down_key, norm)
                regs = [jvc.ratio_entry(slot(ms[j], w), slot(vm["up"][j], vw["up"]), slot(vm["down"][j], vw["down"]),
                                        up_key, down_key, norm, incl, "fallback_to_inclusive", 10)
                        for j in range(1, len(ms))]
                check(syst, [incl] + regs, files[syst][sample][cat])

            # pileup
            wu, wd = w * nom["weight_pileup_up"], w * nom["weight_pileup_down"]

            def pu_sums(m):
                return (float(w[m].sum()), float((w[m] ** 2).sum()), float(wu[m].sum()), float(wd[m].sum()),
                        int(m.sum()), float((wu[m] ** 2).sum()), float((wd[m] ** 2).sum()))
            incl = pu._entry(pu_sums(ms[0]), (norm, 1.0, 1.0))
            regs = [pu._entry(pu_sums(m), (norm, 1.0, 1.0), incl, "fallback_to_inclusive") for m in ms[1:]]
            check("pileup", [incl] + regs, files["pileup"][sample][cat])

            # theory
            if not catalogue[sample].has_theory_weights:
                continue
            member = {k: np.column_stack([nom[f"{name}_{i}"] for i in range(n)])
                      for k, name, n in (("pdf", "LHEPdfWeight", 101), ("scale", "LHEScaleWeight", 9),
                                         ("ps", "PSWeight", 4), ("alphas", "LHEPdfWeightAlphaS", 2))
                      if f"{name}_0" in nom}
            gen = {k: np.ones(n) for k, n in (("pdf", 101), ("alphas", 2), ("scale", 9), ("ps", 4))}

            def th_sums(m):
                out = {"nom": float(w[m].sum()), "w2": float((w[m] ** 2).sum()), "n": int(m.sum()), "alphas": None,
                       "alphas2": None}
                for k, arr in member.items():
                    c = w[m][:, None] * arr[m]
                    out[k], out[k + "2"] = c.sum(axis=0), (c ** 2).sum(axis=0)
                return out
            incl = th._entry(th_sums(ms[0]), norm, gen)
            regs = [th._entry(th_sums(m), norm, gen, incl, "fallback_to_inclusive") for m in ms[1:]]
            check("theory", [incl] + regs, files["theory"][sample][cat])

    summary = {"compared_entries": n_cmp, "max_rel_diff": worst, "mismatches": report,
               "not_compared": sorted(SKIP)}
    print(json.dumps({k: v for k, v in summary.items() if k != "mismatches"}, indent=1))
    print(f"{len(report)} mismatches" + (":\n  " + "\n  ".join(report[:40]) if report else ""))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=1))
    sys.exit(1 if report else 0)


if __name__ == "__main__":
    main()
