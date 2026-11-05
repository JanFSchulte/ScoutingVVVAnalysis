"""Optimise the signal regions of one category (vvv.sr.optimise, port of legacy signal_region_hist.py).

CONFIG is a legacy sr_config.json (lumi, n_signal_regions, score_axes, min_*, grid widths, beam and
branch-and-bound parameters, exclude_samples, abcd_uncertainty). The events come from a legacy
k-fold score cache (--legacy-bdt-root, default the config's bdt_root). Writes
<output-dir>/signal_region.csv in the legacy format and signal_region_report.json (per-SR yields
and sigma_B terms, Z_comb before / after refinement, candidate pool, branch-and-bound nodes,
runtime, peak RSS). --cells runs the coarse stage on occupied coarse-grid cells (much faster,
same boxes up to floating-point summation order).

Usage: python scripts/optimise_signal_regions.py CONFIG [--legacy-bdt-root DIR] [--output-dir DIR]
                                                     [--cells] [--threads N]
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--legacy-bdt-root", type=Path, help="legacy k-fold tree (default: the config's bdt_root)")
    p.add_argument("--sample-config", type=Path, help="legacy sample.json (default: from the k-fold config)")
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--cells", action="store_true", help="coarse stage on coarse-grid cells")
    p.add_argument("--threads", type=int, help="default: the config's max_threads")
    p.add_argument("--no-fingerprint", action="store_true", help="skip the score-cache / model fingerprint check")
    args = p.parse_args()

    from vvv.sr.legacy import load_legacy_inputs
    from vvv.sr.optimise import SRConfig, optimise, write_csv

    t0 = time.time()
    raw = json.loads(args.config.read_text())
    cfg = SRConfig.from_dict(raw)
    bdt_root = args.legacy_bdt_root or raw.get("bdt_root")
    if bdt_root is None:
        p.error("no --legacy-bdt-root and no bdt_root in the config")
    log(f"SR optimisation: config={args.config}, lumi={cfg.lumi}, n_signal_regions={cfg.n_signal_regions}, "
        f"bdt_root={bdt_root}, cells={args.cells}")
    inputs = load_legacy_inputs(cfg, bdt_root, sample_config=args.sample_config,
                                check_fingerprint=not args.no_fingerprint, log=log)
    t_load = time.time() - t0
    bins, report = optimise(cfg, inputs, cells=args.cells, threads=args.threads, log=log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    axis_names = report["axes"]
    write_csv(args.output_dir / "signal_region.csv", bins, axis_names)
    report.update({"config": str(args.config), "bdt_root": str(bdt_root), "total_seconds": time.time() - t0,
                   "load_seconds": t_load,
                   "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6})
    (args.output_dir / "signal_region_report.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"Wrote {args.output_dir / 'signal_region.csv'} and signal_region_report.json "
        f"({report['total_seconds']:.0f} s, peak RSS {report['peak_rss_gb']:.1f} GB)")


if __name__ == "__main__":
    main()
