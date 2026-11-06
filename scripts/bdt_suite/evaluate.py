"""Evaluate the BDT configuration suite on the fold-0 test events of the packages.

Per (category, variant) of OUTPUT/<category>/<variant>/test_proba.npz:
- AUCs of VVV and VH against each background, in the full threshold window (plots.py definitions,
  physics weights);
- weighted Pearson correlations of every class probability with the decorrelated variables on
  QCD (training window, standardised as in training);
- Z_holdout: the production signal-region optimisation (configs/signal_regions/<category>.json,
  deterministic, ABCD sigma_B from the msoftdrop_1 control windows) run on the even-event half of
  the test events and its boxes' Z_comb evaluated on the odd half, and the reverse, averaged
  (weights x 2 x kfold); Z_insample is the optimiser's own Z_comb (optimistic).
Test predictions missing from a variant (trained without the test split) are made here.
- Z_binned: Asimov Z in signal-quantile bins of p(VVV)+p(VH) (binned_z), no optimisation.
Writes OUTPUT/<category>/<variant>/metrics.json (reused unless --redo) and OUTPUT/summary.md.

Usage: pixi run python scripts/bdt_suite/evaluate.py OUTPUT --packages DIR [--config JSON] [--redo] [--no-sr]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
SHORT = {"boosted2_leptonic_1lep": "1lep", "boosted2_leptonic_2lep": "2lep", "boosted2_resolved": "resolved",
         "boosted2_unresolved": "unresolved", "boosted3": "boosted3"}
MSD = "ScoutingFatPFJetRecluster_msoftdrop_1"


EFF_EDGES = (0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.0)


def binned_z(cfg, category, test, proba, window, n_boot=50, nonclosure=0.2):
    """Z_binned: sqrt(sum Z_i^2) of the Asimov Z (sigma_B^2 = sum w^2 + (nonclosure x QCD)^2) in
    bins of D = p(VVV) + p(VH) at the signal efficiencies EFF_EDGES (the sensitive region is the
    top ~1% of the signal), in the full threshold window, without the SR config's
    exclude_samples. No optimisation, so no bias. Returns (Z, n_boot Poisson bootstrap replicas
    with a fixed seed: the same event weights for every variant, so differences are paired)."""
    from vvv.sr.optimise import asimov_z

    classes = list(cfg["class_groups"])
    raw = json.loads((REPO / "configs" / "signal_regions" / f"{category}.json").read_text())
    m = window & ~np.isin(test["sample"], raw.get("exclude_samples", []))
    y, w = test["y"][m], test["w_phys"][m] * cfg["kfold"]
    D = proba[m, classes.index("VVV")] + proba[m, classes.index("VH")]
    is_sig = np.isin(y, [classes.index("VVV"), classes.index("VH")])
    is_qcd = y == classes.index("QCD")
    order = np.argsort(D[is_sig])
    cum = np.cumsum(w[is_sig][order]) / w[is_sig].sum()
    n_bins = len(EFF_EDGES) - 1
    edges = np.interp(1.0 - np.asarray(EFF_EDGES), cum, D[is_sig][order])
    edges[-1] = np.inf
    b = np.digitize(D, edges) - 1
    inb = (b >= 0) & (b < n_bins)

    def z(weights):
        S = np.bincount(b[inb & is_sig], weights[inb & is_sig], n_bins)
        B = np.bincount(b[inb & ~is_sig], weights[inb & ~is_sig], n_bins)
        B2 = np.bincount(b[inb & ~is_sig], weights[inb & ~is_sig] ** 2, n_bins)
        Q = np.bincount(b[inb & is_qcd], weights[inb & is_qcd], n_bins)
        return float(np.sqrt(np.sum(asimov_z(S, B, B2 + (nonclosure * Q) ** 2, stable=True) ** 2)))

    rng = np.random.default_rng(0)
    return z(w), [z(w * rng.poisson(1.0, len(w))) for _ in range(n_boot)]


def metrics(cfg, category, test, proba, sr=True, log=print):
    from vvv.ml.inference import threshold_mask
    from vvv.ml.plots import _weighted_pearson, weighted_roc
    from vvv.ml.train import standardize
    from vvv.sr.optimise import Ctx, ScanInputs, SRConfig, bin_stats, optimise

    cc = cfg["categories"][category]
    classes = list(cfg["class_groups"])
    decor = cc["decorrelate"]
    y, w = test["y"], test["w_phys"]
    full_thr = {k: v for k, v in cc["thresholds"].items() if k in decor}
    window = threshold_mask({k: test["X"][k] for k in full_thr}, full_thr) if full_thr else np.ones(len(y), bool)
    out = {}
    for s in ("VVV", "VH"):
        for b in ("QCD", "VJets", "Top", "VV"):
            si, bi = classes.index(s), classes.index(b)
            m = window & ((y == si) | (y == bi))
            score = proba[m, si] / np.clip(proba[m, si] + proba[m, bi], 1e-12, None)
            out[f"auc_{s}_vs_{b}"] = weighted_roc((y[m] == si).astype(float), score.astype(np.float64), w[m])[2]
    Z = standardize({k: test["X"][k] for k in decor}, decor, cc["clip_ranges"], cc["log_transform"])
    q = y == classes.index("QCD")
    for a, cls in enumerate(classes):
        for j, br in enumerate(decor):
            out[f"corr_QCD_p({cls})_{br.replace('ScoutingFatPFJetRecluster_', '')}"] = \
                _weighted_pearson(Z[q, j], proba[q, a], w[q])
    out["max_abs_corr_QCD"] = max(abs(v) for k, v in out.items() if k.startswith("corr_QCD"))
    out["Z_binned"], out["Z_binned_boot"] = binned_z(cfg, category, test, proba, window)
    out["Z_binned_boot_std"] = float(np.std(out["Z_binned_boot"], ddof=1))
    if sr:
        raw = json.loads((REPO / "configs" / "signal_regions" / f"{category}.json").read_text())
        raw["tie_break"] = "deterministic"
        srcfg = SRConfig.from_dict(raw)
        rest = {k: v for k, v in full_thr.items() if k != MSD}
        other = threshold_mask({k: test["X"][k] for k in rest}, rest) if rest else np.ones(len(y), bool)
        sig = [c in ("VVV", "VH") for c in classes]

        def scan_inputs(sel):
            # one fold of five, and one half of it: weights x 2 kfold
            args = (proba[sel], w[sel] * 2 * cfg["kfold"], y[sel], test["sample"][sel], classes, sig)
            if srcfg.abcd is not None:
                return ScanInputs.from_events(*args, abcd_values=test["X"][MSD][sel], abcd_cond=cc["thresholds"][MSD],
                                              control_windows=srcfg.abcd.windows)
            scan = threshold_mask({"x": test["X"][MSD][sel]}, {"x": cc["thresholds"][MSD]})
            return ScanInputs.from_events(*(a[scan] if isinstance(a, np.ndarray) else a for a in args))

        # Boxes optimised on one half of the test events, Z of those boxes on the other half (and
        # the reverse): an optimisation on the events it is scored on rewards MC fluctuations.
        t = time.time()
        even = test["event"] % 2 == 0
        halves = [scan_inputs(other & even), scan_inputs(other & ~even)]
        z_in, z_out = [], []
        for a_, b_ in ((0, 1), (1, 0)):
            bins, report = optimise(srcfg, halves[a_], cells=True, log=lambda m: None)
            z_in.append(report["Z_comb"])
            ctx = Ctx(srcfg, halves[b_].exclude(srcfg.exclude_samples, log=lambda m: None), log=lambda m: None)
            z_out.append(float(np.sqrt(sum(bin_stats(ctx, b["lo"], b["hi"])["significance"] ** 2 for b in bins))))
        out["Z_holdout"] = float(np.mean(z_out))
        out["Z_holdout_halves"] = z_out
        out["Z_insample"] = float(np.mean(z_in))
        out["sr_seconds"] = time.time() - t
    return out


def predict(vdir, category, cfg, test):
    """test_proba.npz of a model trained without the test split (large packages on NRP)."""
    import xgboost as xgb

    from vvv.ml.inference import softmax
    from vvv.ml.train import standardize

    cc = cfg["categories"][category]
    booster = xgb.Booster()
    booster.load_model(str(vdir / f"{category}_model.json"))
    X = standardize(test["X"], cc["features"], cc["clip_ranges"], cc["log_transform"])
    m = booster.inplace_predict(X, predict_type="margin").reshape(len(X), -1)
    np.savez(vdir / "test_proba.npz", proba=softmax(m.astype(np.float64)).astype(np.float32), event=test["event"])


def main():
    from make_packages import load_package

    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--packages", type=Path, required=True)
    p.add_argument("--config", type=Path, default=REPO / "configs/training/legacy_budget.json")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--no-sr", action="store_true")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    rows = {}
    for cat_dir in sorted(d for d in args.output.iterdir() if d.is_dir() and d.name in SHORT):
        category = cat_dir.name
        test = None
        for vdir in sorted(d for d in cat_dir.iterdir() if (d / "test_proba.npz").exists()
                           or (d / f"{category}_model.json").exists()):
            mfile = vdir / "metrics.json"
            if mfile.exists() and not args.redo:
                m = json.loads(mfile.read_text())
            else:
                if test is None:
                    test = load_package(args.packages / category, splits=("test",))["test"]
                if not (vdir / "test_proba.npz").exists():
                    predict(vdir, category, cfg, test)
                with np.load(vdir / "test_proba.npz") as z:
                    proba, event = z["proba"], z["event"]
                if not np.array_equal(event, test["event"]):
                    raise RuntimeError(f"{vdir}: test events differ from the package")
                m = metrics(cfg, category, test, proba.astype(np.float64), sr=not args.no_sr)
                log = json.loads((vdir / "training_log.json").read_text())
                m["rounds"] = [log["stage1"]["rounds"], log.get("stage2", {}).get("rounds", 0)]
                m["train_s"] = log.get("stage2", log["stage1"])["time_s"]
                util = vdir / "gpu_util.json"
                if util.exists():
                    m["gpu_util_mean"] = json.loads(util.read_text()).get("gpu_util_mean")
                mfile.write_text(json.dumps(m, indent=1))
            rows.setdefault(category, {})[vdir.name] = m
            print(f"{SHORT[category]:10s} {vdir.name:14s} Z_binned {m['Z_binned']:.4f} ({m['Z_binned_boot_std']:.4f})  "
                  f"Z_holdout {m.get('Z_holdout', float('nan')):.4f}  "
                  f"AUC VVV/QCD {m['auc_VVV_vs_QCD']:.4f}  max|corr| {m['max_abs_corr_QCD']:.3f}", flush=True)
    lines = []
    for category, vs in rows.items():
        seeds = [vs[s]["Z_binned"] for s in ("base", "seed43", "seed44") if s in vs]
        noise = f", seed spread of Z_binned {np.std(seeds, ddof=1):.4f} (n={len(seeds)})" if len(seeds) > 1 else ""
        lines += [f"## {SHORT[category]}{noise}", "",
                  "| variant | Z_binned (boot std) | dZ_binned vs base (paired boot std) | Z_holdout | dZ vs base | Z_insample | AUC VVV/QCD | AUC VVV/VJets | AUC VVV/Top | AUC VVV/VV | "
                  "QCD corr p(VVV)-msd | max abs QCD corr | rounds s1/s2 | train h |", "|" + "---|" * 14]
        base = vs.get("base", {}).get("Z_holdout")
        bb = vs.get("base", {}).get("Z_binned")
        for name, m in sorted(vs.items(), key=lambda kv: -kv[1].get("Z_binned", 0)):
            z = m.get("Z_holdout")
            paired = (np.std(np.subtract(m["Z_binned_boot"], vs["base"]["Z_binned_boot"]), ddof=1)
                      if bb is not None else None)
            lines.append(f"| {name} | {m['Z_binned']:.4f} ({m['Z_binned_boot_std']:.4f}) | "
                         + (f"{m['Z_binned'] - bb:+.4f} +- {paired:.4f} | " if bb is not None else " | "))
            lines[-1] += (f"{z:.4f} | {z - base:+.4f} | {m['Z_insample']:.4f} | " if z is not None and base is not None
                         else f"{z if z is not None else ''} | | {m.get('Z_insample', '')} | ")
            lines[-1] += (f"{m['auc_VVV_vs_QCD']:.4f} | {m['auc_VVV_vs_VJets']:.4f} | {m['auc_VVV_vs_Top']:.4f} | "
                          f"{m['auc_VVV_vs_VV']:.4f} | {m['corr_QCD_p(VVV)_msoftdrop_1']:+.3f} | {m['max_abs_corr_QCD']:.3f} | "
                          f"{m['rounds'][0]}/{m['rounds'][1]} | {m['train_s'] / 3600:.1f} |")
        lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines))
    print(f"wrote {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
