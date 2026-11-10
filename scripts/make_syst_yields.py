"""The six *_syst_yields.json inputs of the legacy combine.C (theory_syst_json, pileup_syst_json,
jes_syst_json, jer_syst_json, jms_syst_json, jmr_syst_json) from `vvv run` histogram pickles
(vvv.stats.systematics: port of legacy theory_syst.py, pileup_syst.py and jes/jer/jms/jmr_syst.py).

HISTS are pickles or directories of pickles (searched recursively); a pickle may hold several datasets,
and a dataset spread over several pickles is added up. Every MC dataset gets pileup and jet entries,
those with has_theory_weights (data/samples.yaml) theory entries, in each category given with
--signal-regions (the CSV the histograms were filled with; checked against the SHA-256 the pickles
record). The msoftdrop_1 window of each category comes from --training-config. Writes
<output-dir>/<syst>_syst_yields.{json,csv} and syst_yields_provenance.json.

Usage: pixi run python scripts/make_syst_yields.py HISTS... --signal-regions CAT=CSV [...]
           --training-config configs/training/baseline.json --output-dir DIR
           [--samples S ...] [--systs theory,pileup,jes,jer,jms,jmr] [--lumi 255.8]
           [--min-region-events 10] [--pdf-method hessian_symmetric] [--mass-fit 20,250,10]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def log(message):
    print(message, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("hists", nargs="+", type=Path)
    p.add_argument("--signal-regions", action="append", required=True, metavar="CAT=CSV")
    p.add_argument("--training-config", type=Path, required=True, help="the msoftdrop_1 threshold window per category")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples", nargs="*", help="only these MC samples (default: every MC dataset of the histograms)")
    p.add_argument("--systs", default="theory,pileup,jes,jer,jms,jmr")
    p.add_argument("--lumi", type=float, help="fb^-1 of the histograms, for pickles that do not record it "
                   "(default 255.8, the vvv run default)")
    p.add_argument("--min-region-events", type=int, default=10, help="legacy min_region_events (n_eff fallback)")
    p.add_argument("--pdf-method", default="hessian_symmetric", choices=["hessian_symmetric", "mc_replica"])
    p.add_argument("--mass-fit", help="LOW,HIGH,WIDTH: also the mass-binned entries (legacy mass_fit config; the "
                   "bins must be the qcd_est massfit bins). Default: none, as the legacy production")
    args = p.parse_args()

    from vvv.samples import load_samples
    from vvv.stats.systematics import MASS_BRANCH, SYSTS, HistOutputs, SystConfig, build, summarise, write

    t0 = time.time()
    systs = tuple(s for s in args.systs.split(",") if s)
    if not set(systs) <= set(SYSTS):
        p.error(f"--systs: unknown {sorted(set(systs) - set(SYSTS))}")
    tcfg = json.loads(args.training_config.read_text())
    categories = {}
    for item in args.signal_regions:
        cat, csv_path = item.split("=", 1)
        if cat not in tcfg["categories"]:
            p.error(f"{cat} is not a category of {args.training_config}")
        categories[cat] = (Path(csv_path), tuple(tcfg["categories"][cat]["thresholds"][MASS_BRANCH]))
    mass_fit = None
    if args.mass_fit:
        lo, hi, width = (float(v) for v in args.mass_fit.split(","))
        mass_fit = ((lo, hi), width)

    outputs = HistOutputs.load(args.hists, log=log)
    if outputs.lumi is not None and args.lumi is not None and abs(outputs.lumi - args.lumi) > 1e-9:
        p.error(f"--lumi {args.lumi} differs from the {outputs.lumi} fb^-1 recorded in the pickles")
    lumi = outputs.lumi if outputs.lumi is not None else (args.lumi or 255.8)
    log(f"Systematic yields: {', '.join(systs)}; lumi {lumi} fb^-1 ({'recorded' if outputs.lumi else '--lumi'}), "
        f"min_region_events {args.min_region_events}, pdf_method {args.pdf_method}, "
        f"mass_fit {args.mass_fit or 'none'}")
    cfg = SystConfig(lumi, min_region_events=args.min_region_events, pdf_method=args.pdf_method, mass_fit=mass_fit)
    result = build(outputs, categories, cfg, load_samples(), samples=args.samples, systs=systs, log=log)
    for syst in systs:
        summarise(result[syst], syst, log=log)
    write(result, args.output_dir, log=log)
    (args.output_dir / "syst_yields_provenance.json").write_text(json.dumps({
        "hists": outputs.files, "lumi_fb": lumi, "signal_regions": {c: str(v[0]) for c, v in categories.items()},
        "windows": {c: list(v[1]) for c, v in categories.items()}, "training_config": str(args.training_config),
        "min_region_events": args.min_region_events, "pdf_method": args.pdf_method, "mass_fit": args.mass_fit,
        "seconds": time.time() - t0}, indent=1))
    log(f"Done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
