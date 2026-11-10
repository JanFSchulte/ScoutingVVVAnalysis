"""Signal-region-independent QCD transfer factors: the mass-fit window/sideband transfer factors of
reference pass selections p(QCD) < c, scanned over c, instead of those of the optimised SR boxes
(which move with the SR choice, so the optimisation and its own ABCD uncertainty never settle).

For each threshold c the pass region is the one box {QCD: [0, c)}; everything else (the massfit
settings of the production point's qcd_est.json, the coffea MC fit-region skims, the TF_MC fit and
the sideband ratios) is as in scripts/estimate_qcd.py --skim-dir. The skims are read once. A flat
transfer factor over the thresholds the SRs use means a region-independent value exists (and that
the mass decorrelation holds there); the scan's median over the chosen threshold range is written
as the recommended pair.

Usage: pixi run python scripts/reference_tf.py QCD_EST_JSON --skim-dir DIR --training-config JSON \\
           --category CAT --output-dir DIR [--thresholds 0.1 0.03 ...] [--sr-csv CSV ...]
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from vvv.background.qcd_massfit import (MassfitAccumulator, assign_regions, estimate, load_sr_boxes,
                                        sideband_transfer_factors)
from vvv.background.skims import SkimSetup
from vvv.ml.inference import CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("qcd_est_config", type=Path)
    p.add_argument("--skim-dir", type=Path, required=True)
    p.add_argument("--training-config", type=Path, required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001])
    p.add_argument("--sr-csv", type=Path, nargs="*", default=[],
                   help="SR CSVs whose QCD acceptance marks the relevant threshold range (e.g. the iterations)")
    p.add_argument("--loosen", type=float, nargs="*", default=[],
                   help="instead of p(QCD) thresholds: the SR boxes of each --sr-csv loosened by these factors "
                        "(lower edges / f, upper edges x f, the mass fit's box_mask semantics), one pass region per box")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.loosen:
        return loosened(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # a placeholder CSV: SkimSetup needs one, the boxes are made here
    tmp = Path(tempfile.mkdtemp()) / "ref.csv"
    tmp.write_text("bin_index,QCD_low,QCD_high\n1,0,1\n")
    setup = SkimSetup(args.qcd_est_config, args.training_config, args.category, args.skim_dir, tmp, args.output_dir)
    accs = {c: MassfitAccumulator(setup.massfit, [(1, {"QCD": (0.0, c)})], setup.class_names, setup.sample_class,
                                  setup.qcd_classes, setup.window, log=lambda m: None) for c in args.thresholds}
    qcd_pass = {c: 0.0 for c in args.thresholds}
    qcd_total = 0.0
    sr_qcd = {str(f): 0.0 for f in args.sr_csv}
    sr_boxes = {str(f): load_sr_boxes(f, setup.class_names) for f in args.sr_csv}
    for sample in setup.sample_class:
        if sample in setup.exclude:
            continue
        ev = setup._read(setup.skim_dir, sample)
        if ev is None:
            continue
        for acc in accs.values():
            acc.fill(sample, ev["mass"], ev["weight"], proba=ev["proba"])
        if setup.sample_class[sample] in setup.qcd_classes:
            w, pq = ev["weight"], ev["proba"][:, CLASSES.index("QCD")]
            qcd_total += w.sum()
            for c in args.thresholds:
                qcd_pass[c] += w[pq < c].sum()
            for f, boxes in sr_boxes.items():
                sr_qcd[f] += w[assign_regions(ev["proba"], boxes, setup.class_names) > 0].sum()
    rows = []
    for c in args.thresholds:
        result = estimate(accs[c].inputs(), setup.massfit, window=setup.window, log=lambda m: None)
        tf = sideband_transfer_factors(result)
        fit = result.get("tf") or {}
        rows.append({"threshold": c, "qcd_pass_fraction": qcd_pass[c] / qcd_total,
                     "tf_model": [float(x) for x in tf["model"]], "tf_mc_true": [float(x) for x in tf["mc_true"]],
                     "tfmc_chi2_ndf": [float(fit.get("chi2", np.nan)), int(fit.get("ndf", 0))]})
        r = rows[-1]
        print(f"p(QCD) < {c:<6g} QCD pass {r['qcd_pass_fraction']:.2e}  TF model {r['tf_model'][0]:.3f} / {r['tf_model'][1]:.3f}"
              f"  MC {r['tf_mc_true'][0]:.3f} / {r['tf_mc_true'][1]:.3f}  chi2/ndf {r['tfmc_chi2_ndf'][0]:.1f}/{r['tfmc_chi2_ndf'][1]}",
              flush=True)
    sr_fracs = {f: v / qcd_total for f, v in sr_qcd.items()}
    for f, v in sr_fracs.items():
        print(f"SR union of {f}: QCD pass fraction {v:.2e}")
    # recommended: median of the model TFs over the thresholds whose QCD pass fraction lies within a
    # factor 3 of the SRs' (all thresholds if no SR CSV is given)
    if sr_fracs:
        lo, hi = min(sr_fracs.values()) / 3, max(sr_fracs.values()) * 3
        used = [r for r in rows if lo <= r["qcd_pass_fraction"] <= hi] or rows
    else:
        used = rows
    rec = [float(np.median([r["tf_model"][i] for r in used])) for i in range(2)]
    spread = [float(np.std([r["tf_model"][i] for r in used])) for i in range(2)]
    out = {"category": args.category, "scan": rows, "sr_qcd_pass_fraction": sr_fracs,
           "recommended_transfer_factors": rec, "spread_in_range": spread,
           "thresholds_used": [r["threshold"] for r in used]}
    (args.output_dir / f"{args.category}.json").write_text(json.dumps(out, indent=1))
    print(f"recommended TF {rec[0]:.3f} / {rec[1]:.3f} (spread {spread[0]:.3f} / {spread[1]:.3f} over "
          f"thresholds {out['thresholds_used']})")


def loosened(args):
    """Transfer factors of the SR boxes loosened by each factor: the SR shape, more QCD statistics."""
    from vvv.background.qcd_massfit import box_mask

    results = {}
    for csv in args.sr_csv:
        setup = SkimSetup(args.qcd_est_config, args.training_config, args.category, args.skim_dir, csv, args.output_dir)
        boxes = load_sr_boxes(csv, setup.class_names)
        accs, entries = {}, {}
        for f in args.loosen:
            loose = [(b, {a: (lo / f, min(hi * f, 1.0) if hi < 1.0 - 1e-12 else hi) for a, (lo, hi) in bounds.items()})
                     for b, bounds in boxes]
            accs[f] = (loose, MassfitAccumulator(setup.massfit, [(1, {})], setup.class_names, setup.sample_class,
                                                 setup.qcd_classes, setup.window, log=lambda m: None))
            entries[f] = 0
        for sample in setup.sample_class:
            if sample in setup.exclude:
                continue
            ev = setup._read(setup.skim_dir, sample)
            if ev is None:
                continue
            for f, (loose, acc) in accs.items():
                union = np.zeros(len(ev["mass"]), dtype=bool)
                for _, bounds in loose:
                    union |= box_mask(ev["proba"], bounds, setup.class_names)
                # the union as one region: proba set so that the single all-pass box takes exactly the union
                region = np.where(union, 1, 0)
                acc.fill(sample, ev["mass"], ev["weight"], proba=ev["proba"], region=region)
                if setup.sample_class[sample] in setup.qcd_classes:
                    entries[f] += int(union.sum())
        rows = []
        for f, (_, acc) in accs.items():
            result = estimate(acc.inputs(), setup.massfit, window=setup.window, log=lambda m: None)
            tf = sideband_transfer_factors(result)
            fit = result.get("tf") or {}
            rows.append({"factor": f, "qcd_entries": entries[f], "tf_model": [float(x) for x in tf["model"]],
                         "tf_mc_true": [float(x) for x in tf["mc_true"]],
                         "tfmc_chi2_ndf": [float(fit.get("chi2", np.nan)), int(fit.get("ndf", 0))]})
            r = rows[-1]
            print(f"{csv.parent.parent.name}: loosen x{f:<4g} QCD MC entries {r['qcd_entries']:7d}  TF model "
                  f"{r['tf_model'][0]:.3f} / {r['tf_model'][1]:.3f}  MC {r['tf_mc_true'][0]:.3f} / {r['tf_mc_true'][1]:.3f}"
                  f"  chi2/ndf {r['tfmc_chi2_ndf'][0]:.1f}/{r['tfmc_chi2_ndf'][1]}", flush=True)
        results[str(csv)] = rows
    (args.output_dir / f"{args.category}_loosened.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
