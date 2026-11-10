"""QCD mass-fit inputs of one category (vvv.background.qcd_massfit, port of legacy qcd_est.py abcd_mode massfit).

CONFIG is a legacy qcd_est.json (lumi, bdt_root, signal_region_csv, output_dir, root_file_name,
abcd_branches, abcd_mode = massfit, massfit[tree], exclude_samples, data_sample). The events come
from the legacy k-fold score cache of bdt_root. Writes <output-dir>/<root_file_name> (default
qcd_abcd_yields.root, the file combine.C reads: metadata, the one-bin SR bundles samples/*,
groups/*, qcd_predict, qcd_true, total_predict, total_true, and massfit/{settings, tfmc, fail, pass,
smooth, data, blinded}) and qcd_est_summary.json (TF_MC fit, n0, per-SR QCD prediction and yields,
window / sideband transfer factors, runtime, peak RSS); --plots also qcd_massfit_tf.pdf and
qcd_massfit_sr<i>.pdf.

With --skim-dir the events come from the coffea MC fit-region skims instead (vvv.background.skims:
the config supplies only the massfit settings and exclude_samples; --training-config and
--category the classes and the msoftdrop_1 window, --signal-region-csv the SR boxes, --output-dir
the output; --data-skim-dir adds data, which needs proba_<class> columns).

With --hists the binned sums come from the nominal mass histograms of `vvv run` pickles (files or
directories; vvv.background.hists), the same pass that feeds scripts/make_syst_yields.py; the other
options as with --skim-dir. Data is used when the histograms hold data_sample (default data_2024).

Usage: python scripts/estimate_qcd.py CONFIG [--output-dir DIR] [--data-sample NAME] [--no-fingerprint] [--plots]
       python scripts/estimate_qcd.py CONFIG --skim-dir DIR --training-config JSON --category CAT \
           --signal-region-csv CSV --output-dir DIR [--data-skim-dir DIR] [--plots]
       python scripts/estimate_qcd.py CONFIG --hists PKL_OR_DIR [...] --training-config JSON --category CAT \
           --signal-region-csv CSV --output-dir DIR [--plots]
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def log(message):
    print(message, flush=True)


def plot(result, out_dir, branch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    edges = result["edges"]
    c = 0.5 * (edges[:-1] + edges[1:])
    q_sw, q_sw2 = result["qcd_true"][0], result["qcd_true"][1]
    F, F2, A, A2 = q_sw[0], q_sw2[0], q_sw[1:].sum(axis=0), q_sw2[1:].sum(axis=0)
    tf = result["tf"]
    used = F > 0
    Fs = np.where(used, F, 1.0)
    ratio = np.where(used, A / Fs, np.nan)
    err = np.where(used, np.sqrt(A2 / Fs ** 2 + ratio ** 2 * F2 / Fs ** 2), np.nan)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(c, ratio, yerr=err, fmt="o", color="black", label="QCD MC, SR union / fail")
    ax.plot(c[used], (np.exp(tf["coefficients"][0]) * tf["shape"])[used], color="#bd1f01",
            label=f"TF_MC, order {len(tf['coefficients']) - 1} ($\\chi^2$/ndf = {tf['chi2']:.1f}/{tf['ndf']})")
    ax.set_yscale("log")
    ax.set_xlabel(branch)
    ax.set_ylabel("Pass / fail")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "qcd_massfit_tf.pdf", bbox_inches="tight")
    plt.close(fig)
    for i, sr in enumerate(result["sr_ids"]):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.errorbar(c, q_sw[i + 1], yerr=np.sqrt(q_sw2[i + 1]), fmt="o", color="black", label="QCD MC")
        ax.step(edges, np.r_[result["n0"][i] * F * tf["shape"], 0.0], where="post", color="#bd1f01",
                label="Nominal model n0 x F x TF_MC")
        ax.set_xlabel(branch)
        ax.set_ylabel("Events")
        ax.set_title(f"SR{sr}")
        ax.legend(frameon=False)
        fig.savefig(out_dir / f"qcd_massfit_sr{sr}.pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--output-dir", type=Path, help="default: the config's output_dir")
    p.add_argument("--data-sample", help="data sample of the data score cache (overrides the config's data_sample)")
    p.add_argument("--no-fingerprint", action="store_true", help="skip the score-cache / model fingerprint check")
    p.add_argument("--plots", action="store_true", help="also the TF_MC and per-SR model plots")
    p.add_argument("--skim-dir", type=Path, help="coffea MC fit-region skims (instead of the legacy score cache)")
    p.add_argument("--training-config", type=Path, help="with --skim-dir: class_groups and the msoftdrop_1 window")
    p.add_argument("--category", help="with --skim-dir")
    p.add_argument("--signal-region-csv", type=Path, help="with --skim-dir")
    p.add_argument("--data-skim-dir", type=Path, help="with --skim-dir: data skims with proba_<class> columns")
    p.add_argument("--hists", type=Path, nargs="+", help="vvv run histogram pickles or directories (instead of the "
                   "skims or the legacy score cache)")
    args = p.parse_args()

    from vvv.background.hists import HistSetup
    from vvv.background.legacy import LegacySetup
    from vvv.background.qcd_massfit import estimate, sr_csv_hash, summary, write_root
    from vvv.background.skims import SkimSetup

    t0 = time.time()
    if args.hists:
        if not (args.training_config and args.category and args.signal_region_csv and args.output_dir):
            p.error("--hists needs --training-config, --category, --signal-region-csv and --output-dir")
        if args.skim_dir:
            p.error("--hists and --skim-dir are exclusive")
        setup = HistSetup(args.config, args.training_config, args.category, args.hists, args.signal_region_csv,
                          args.output_dir, **({"data_sample": args.data_sample} if args.data_sample else {}))
    elif args.skim_dir:
        if not (args.training_config and args.category and args.signal_region_csv and args.output_dir):
            p.error("--skim-dir needs --training-config, --category, --signal-region-csv and --output-dir")
        setup = SkimSetup(args.config, args.training_config, args.category, args.skim_dir, args.signal_region_csv,
                          args.output_dir, data_skim_dir=args.data_skim_dir,
                          **({"data_sample": args.data_sample} if args.data_sample else {}))
    else:
        setup = LegacySetup.load(args.config, {"data_sample": args.data_sample} if args.data_sample else None)
    out_dir = args.output_dir or setup.output_dir
    log(f"QCD mass fit: config={args.config}, tree={setup.tree}, lumi={setup.lumi}, bdt_root={setup.bdt_root}, "
        f"signal_region_csv={setup.signal_region_csv}, qcd_from_mc={setup.massfit.qcd_from_mc}, output_dir={out_dir}")
    if args.hists:
        inputs = setup.inputs(log=log)
    else:
        acc = setup.accumulator(log=log)
        setup.fill_mc(acc, check_fingerprint=not args.no_fingerprint, log=log)
        if setup.data_sample:
            setup.fill_data(acc, check_fingerprint=not args.no_fingerprint, log=log)
        inputs = acc.inputs()
    t_load = time.time() - t0
    result = estimate(inputs, setup.massfit, window=setup.window, log=log)
    for i, sr in enumerate(result["sr_ids"]):
        log(f"  SR{sr}: qcd_true={result['bundles']['qcd_true']['yield'][i]:.6g}, n0={result['n0'][i]:.6g}, "
            f"qcd_predict={result['bundles']['qcd_predict']['yield'][i]:.6g}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_root(out_dir / setup.root_file_name, result, sr_csv_hash(setup.signal_region_csv), log=log)
    if args.plots and result["tf"] is not None:
        plot(result, out_dir, setup.branch)
    report = summary(result)
    report.update({"config": str(args.config), "tree": setup.tree, "bdt_root": str(setup.bdt_root),
                   "signal_region_csv": str(setup.signal_region_csv), "load_seconds": t_load,
                   "total_seconds": time.time() - t0,
                   "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6})
    (out_dir / "qcd_est_summary.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"Wrote {out_dir / setup.root_file_name} and qcd_est_summary.json "
        f"({report['total_seconds']:.0f} s, peak RSS {report['peak_rss_gb']:.1f} GB)")


if __name__ == "__main__":
    main()
