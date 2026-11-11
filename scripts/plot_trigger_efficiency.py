"""Trigger efficiency curves from the `vvv trigger-efficiency` pickles.

Usage:
    pixi run python scripts/plot_trigger_efficiency.py PICKLE_DIR --training-config configs/training/baseline.json \
        --output-dir DIR [--classes VVV VH] [--triggers JetHT SingleMuon ...]

Per signal class (the class_groups of the training config, samples merged with their MC weights),
region and variable: the efficiency passed / all per bin, one curve per trigger, with a weighted
binomial uncertainty. Data samples in PICKLE_DIR (events of the reference path only, unweighted)
are drawn as points with Clopper-Pearson 68 % intervals, with the data/MC ratio below.
Writes DIR/<class>/<region>/<variable>.png and .pdf, and DIR/efficiencies.json (per bin and
integrated).
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

from vvv.processors.trigger_efficiency import REGIONS, TRIGGERS, VARIABLES  # noqa: E402
from vvv.samples import load_samples  # noqa: E402


def load(directory):
    hists, references = {}, set()
    for path in sorted(Path(directory).glob("*.pkl")):
        with open(path, "rb") as handle:
            out = pickle.load(handle)
        references.add(out["reference"])
        for v in VARIABLES:
            hists[v] = out[v] if v not in hists else hists[v] + out[v]
    if len(references) > 1:
        raise ValueError(f"{directory}: pickles with different reference paths {sorted(references)}")
    return hists, references.pop() if references else None


def counts(h, datasets, region, trigger):
    """(sum w, sum w^2) per bin of the given datasets."""
    present = [d for d in datasets if d in h.axes["dataset"]]
    if not present:
        n = h.axes["x"].size
        return np.zeros(n), np.zeros(n)
    sub = h[{"dataset": present, "region": region, "trigger": trigger}][{"dataset": sum}]
    return sub.values(), sub.variances()


def mc_efficiency(passed, passed_w2, total, total_w2):
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(total > 0, passed / total, np.nan)
        var = ((1 - eff) ** 2 * passed_w2 + eff ** 2 * np.maximum(total_w2 - passed_w2, 0.0)) / total ** 2
    err = np.sqrt(np.where(total > 0, var, np.nan))
    return eff, err, err


def data_efficiency(passed, total):
    k, n = np.round(passed), np.round(total)
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(n > 0, k / n, np.nan)
        lo = np.where(k > 0, beta.ppf(0.16, k, n - k + 1), 0.0)
        hi = np.where(k < n, beta.ppf(0.84, k + 1, n - k), 1.0)
    return eff, np.where(n > 0, eff - lo, np.nan), np.where(n > 0, hi - eff, np.nan)


def curves(h, datasets, region, triggers, is_data):
    total, total_w2 = counts(h, datasets, region, "all")
    out = {}
    for trigger in triggers:
        passed, passed_w2 = counts(h, datasets, region, trigger)
        eff, lo, hi = data_efficiency(passed, total) if is_data else mc_efficiency(passed, passed_w2, total, total_w2)
        integrated = float(passed.sum() / total.sum()) if total.sum() > 0 else None
        out[trigger] = {"eff": eff, "err_lo": lo, "err_hi": hi, "integrated": integrated, "denominator": float(total.sum())}
    return out


def plot(path, axis, mc, data, label, reference):
    centres = axis.centers
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": (3, 1)})
    hep.cms.label("Private work", data=data is not None, ax=ax, loc=0, com=13.6)
    for i, (trigger, c) in enumerate(mc.items()):
        colour = f"C{i}"
        ax.errorbar(centres, c["eff"], yerr=[c["err_lo"], c["err_hi"]], drawstyle="steps-mid", color=colour,
                    label=f"{label} MC {trigger}")
        if data is not None:
            d = data[trigger]
            ax.errorbar(centres, d["eff"], yerr=[d["err_lo"], d["err_hi"]], fmt="o", ms=3, color=colour,
                        label=f"Data {trigger}")
            with np.errstate(divide="ignore", invalid="ignore"):
                rax.errorbar(centres, d["eff"] / c["eff"], yerr=[d["err_lo"] / c["eff"], d["err_hi"] / c["eff"]],
                             fmt="o", ms=3, color=colour)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Efficiency")
    ax.legend(fontsize="x-small", ncol=2, title=f"MC: passed / all; data reference: {reference}" if data else None)
    rax.axhline(1.0, color="k", lw=0.8)
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("Data / MC")
    rax.set_xlabel(axis.label)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_dir")
    p.add_argument("--training-config", required=True, help="training config with the class_groups")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--classes", nargs="+", default=["VVV", "VH"])
    p.add_argument("--triggers", nargs="+", default=[t for t in TRIGGERS if t != "all"])
    args = p.parse_args()

    hists, reference = load(args.pickle_dir)
    groups = json.loads(Path(args.training_config).read_text())["class_groups"]
    samples = load_samples()
    present = set(hists[next(iter(VARIABLES))].axes["dataset"])
    data_sets = sorted(d for d in present if not samples[d].is_mc)
    out_dir = Path(args.output_dir)
    summary = {"pickle_dir": str(Path(args.pickle_dir).resolve()), "reference": reference, "data": data_sets,
               "classes": {c: [s for s in groups[c] if s in present] for c in args.classes}, "efficiencies": {}}
    for cls in args.classes:
        for region in REGIONS:
            for variable, h in hists.items():
                mc = curves(h, groups[cls], region, args.triggers, is_data=False)
                data = curves(h, data_sets, region, args.triggers, is_data=True) if data_sets else None
                plot(out_dir / cls / region / variable, h.axes["x"], mc, data, cls, reference)
                entry = {"edges": h.axes["x"].edges.tolist(), "mc": mc, "data": data}
                summary["efficiencies"].setdefault(cls, {}).setdefault(region, {})[variable] = entry
    (out_dir / "efficiencies.json").write_text(json.dumps(
        summary, indent=1, default=lambda x: [None if not np.isfinite(v) else float(v) for v in x]))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
