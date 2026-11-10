"""JMS/JMR tag-and-probe fit in the ttbar CR (vvv.corrections.jms_fit, port of legacy jms_jmr_fit.py).

CONFIG is configs/jms_jmr.json (data_samples, class_groups, mass/tag branches, tag_wp, fit selection,
fit_window, n_bins). The pass/fail histograms come from one of:

- --hists: `vvv ttbar-cr` output pickles (files or directories of *.pkl, summed), the "fit"
  histogram at --syst (default nominal);
- --skim-dir: `vvv ttbar-cr --skim-dir` Parquet skims (<dir>/<sample>/ttbar_cr/nominal/*.parquet),
  with the fit selection applied here (a changed selection needs no re-read of the NanoAOD);
- --legacy-root: legacy convert_ttbar_cr trees (<dir>/{data,bkg}/<sample>.root, tree ttbar_cr), with
  the legacy weights weight_pu x lumi x xsec / raw_entries from --legacy-sample-json (closure mode).

The luminosity is the sum of the data samples' lumi (catalogue, or sample.json in legacy mode).
Writes <output-dir>/jms_jmr_results.json (the format corrections/jets.py reads) and, with --plots,
jms_jmr_W_tagprobe.{pdf,png}. data/corrections/jms_jmr_results.json is only replaced by hand.

Usage: python scripts/fit_jms_jmr.py CONFIG (--hists P [P ...] | --skim-dir DIR | --legacy-root DIR) --output-dir DIR [--plots]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vvv.corrections.jms_fit import PassFail, background, edges, measure, model  # noqa: E402


def from_hists(paths, data_samples, mc_samples, syst, cfg):
    """The processor's "fit" histogram, or with fit_hist in the config that control histogram
    (same selection and pass/fail split) cut to fit_window, whose edges must match its bins."""
    import hist

    files = [f for p in map(Path, paths) for f in (sorted(p.rglob("*.pkl")) if p.is_dir() else [p])]
    n_bins = int(cfg["n_bins"])
    data, mc = PassFail(n_bins), PassFail(n_bins)
    name_h = cfg.get("fit_hist", "fit")
    lo, hi = cfg["fit_window"]
    for f in files:
        h = pickle.loads(f.read_bytes())[name_h]
        if name_h != "fit":
            axis = h.axes[-1].name
            h = h[{axis: slice(hist.loc(lo), hist.loc(hi))}]
            if not np.allclose(h.axes[-1].edges, edges(cfg)):
                raise ValueError(f"{name_h} bins {h.axes[-1].edges} do not match fit_window/n_bins {edges(cfg)}")
        for name in h.axes["dataset"]:
            target = data if name in data_samples else mc if name in mc_samples else None
            if target is not None and syst in h.axes["syst"]:
                target.add_hist(h[{"dataset": name, "syst": syst}])
    return data, mc


def from_skims(skim_dir, data_samples, mc_samples, cfg):
    import pyarrow.parquet as pq

    columns = sorted({*cfg["selection"], cfg["mass_branch"], cfg["tag_branch"], "weight"})
    data, mc = PassFail(int(cfg["n_bins"])), PassFail(int(cfg["n_bins"]))
    for name in data_samples + mc_samples:
        files = sorted((Path(skim_dir) / name / "ttbar_cr" / "nominal").glob("*.parquet"))
        if not files:
            print(f"[WARN] no skim for {name}")
            continue
        for f in files:
            table = pq.read_table(f, columns=columns).to_pydict()
            arrays = {k: np.asarray(v) for k, v in table.items()}
            (data if name in data_samples else mc).add_events(arrays, arrays["weight"], cfg)
    return data, mc


def from_legacy(root, sample_json, data_samples, mc_samples, cfg):
    """Legacy _accumulate: MC weighted by weight_pu x lumi x xsec / raw_entries, data unweighted."""
    import uproot

    info = {s["name"]: s for s in json.loads(Path(sample_json).read_text())["sample"]}
    lumi = sum(float(info[s].get("lumi", 0.0)) for s in data_samples)
    read = sorted({*cfg["selection"], cfg["mass_branch"], cfg["tag_branch"]})
    data, mc = PassFail(int(cfg["n_bins"])), PassFail(int(cfg["n_bins"]))
    for name in data_samples + mc_samples:
        is_mc = bool(info[name].get("is_MC"))
        f = Path(root) / ("bkg" if is_mc else "data") / f"{name}.root"
        if not f.exists():
            print(f"[WARN] missing {f}")
            continue
        a = uproot.open(f)["ttbar_cr"].arrays(read + (["weight_pu"] if is_mc else []), library="np")
        if is_mc:
            xs, raw = float(info[name].get("xsection", 0)), float(info[name].get("raw_entries", 0))
            if xs <= 0 or raw <= 0:
                continue
            mc.add_events(a, np.asarray(a["weight_pu"], float) * (lumi * xs / raw), cfg)
        else:
            data.add_events(a, np.ones(len(a[cfg["mass_branch"]])), cfg)
    return data, mc, lumi


def plot(data, mc, result, cfg, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    bins = edges(cfg)
    centers = 0.5 * (bins[:-1] + bins[1:])
    lo, hi = cfg["fit_window"]
    xf = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    panels = [
        (ax[0, 0], data, "pass", result["pass"]["data"], "black", "Data, Pass (W-tagged)"),
        (ax[0, 1], mc, "pass", result["pass"]["mc"], "#5790fc", "MC, Pass (W-tagged)"),
        (ax[1, 0], data, "fail", None, "black", "Data, Fail"),
        (ax[1, 1], mc, "fail", None, "#5790fc", "MC, Fail"),
    ]
    for a, h, tag, fit, colour, title in panels:
        counts, errors = h.counts(tag)
        a.errorbar(centers, counts, yerr=errors, fmt="o", color=colour, ms=4)
        if fit is not None:
            a.plot(xf, model(xf, *fit["popt"]), "-", color="#e42536", lw=2, label="S+B fit")
            a.plot(xf, background(xf, fit["B"], fit["C"], fit["m0"]), "--", color="gray", lw=1.3, label="bkg")
            a.axvline(fit["mu"], color="#e42536", ls=":", lw=1)
            a.text(0.04, 0.95, f"$\\mu$={fit['mu']:.2f}$\\pm${fit['mu_err']:.2f}\n$\\sigma$={fit['sigma']:.2f}$\\pm${fit['sigma_err']:.2f}",
                   transform=a.transAxes, va="top", fontsize=12, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
            a.legend(loc="upper right", fontsize=11)
        a.set_title(title, fontsize=13)
        a.set_ylim(bottom=0)
    for a in ax[1]:
        a.set_xlabel(r"Probe $m_{\mathrm{SD}}$ [GeV]")
    fig.suptitle(f"JMS = {result['JMS']:.3f} $\\pm$ {result['JMS_err']:.3f}      "
                 f"JMR = {result['JMR']:.3f} $\\pm$ {result['JMR_err']:.3f}", fontsize=15, y=0.99)
    hep.cms.label("Preliminary", data=True, com=13.6, year="2024", lumi=round(result["lumi_fb"], 1), ax=ax[0, 0])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / "jms_jmr_W_tagprobe.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("config")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hists", nargs="+")
    src.add_argument("--skim-dir")
    src.add_argument("--legacy-root")
    p.add_argument("--legacy-sample-json", default=str(Path(__file__).resolve().parents[2] / "CMSSW_16_1_0_pre4/src/ScoutingVVVTools/src/sample.json"))
    p.add_argument("--data-samples", nargs="+", help="override the config's data_samples")
    p.add_argument("--mc-samples", nargs="+", help="override the config's class_groups samples")
    p.add_argument("--syst", default="nominal", help="jet configuration of the --hists input")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    data_samples = args.data_samples or list(cfg["data_samples"])
    mc_samples = args.mc_samples or [s for group in cfg["class_groups"].values() for s in group]
    if args.legacy_root:
        data, mc, lumi = from_legacy(args.legacy_root, args.legacy_sample_json, data_samples, mc_samples, cfg)
    else:
        from vvv.samples import load_samples

        catalogue = load_samples()
        lumi = sum(float(catalogue[s].lumi_fb or 0.0) for s in data_samples)
        if args.hists:
            data, mc = from_hists(args.hists, data_samples, mc_samples, args.syst, cfg)
        else:
            data, mc = from_skims(args.skim_dir, data_samples, mc_samples, cfg)

    print(f"Lumi={lumi:.2f}/fb  tag WP({cfg['tag_branch']})>{cfg['tag_wp']}  window={tuple(cfg['fit_window'])} bins={cfg['n_bins']}")
    print(f"  PASS: data={data.sumw['pass'].sum():.0f}  MC={mc.sumw['pass'].sum():.1f}")
    print(f"  FAIL: data={data.sumw['fail'].sum():.0f}  MC={mc.sumw['fail'].sum():.1f}")
    result = measure(data, mc, cfg, lumi)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jms_jmr_results.json").write_text(json.dumps(result, indent=2))
    for key, label in (("data", "Data"), ("mc", "MC")):
        f = result["pass"][key]
        print(f"  {label + ':':6s} mu={f['mu']:6.2f}+/-{f['mu_err']:4.2f}  sigma={f['sigma']:5.2f}+/-{f['sigma_err']:4.2f}  "
              f"chi2/ndf={f['chi2']:.1f}/{f['ndf']}")
    print(f"  JMS = {result['JMS']:.4f} +/- {result['JMS_err']:.4f}")
    print(f"  JMR = {result['JMR']:.4f} +/- {result['JMR_err']:.4f}")
    print(f"Saved: {out_dir / 'jms_jmr_results.json'}")
    if args.plots:
        plot(data, mc, result, cfg, out_dir)


if __name__ == "__main__":
    main()
