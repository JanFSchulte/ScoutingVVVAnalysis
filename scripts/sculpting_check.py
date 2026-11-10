"""msoftdrop_1 sculpting by the signal-region selection: background (and signal) mass shapes in the
SR boxes compared with the whole category, for one or more (skims, SR) configurations.

For every configuration NAME=SKIM_DIR:SR_CSV the events of the category's fit-region skims
(20-250 GeV) are assigned to the SR boxes (the mass fit's box semantics). Per SR and for the SR
union it fills the normalised msoftdrop_1 shape of each background class and of the signal, and
reports the fraction of the background in the W/Z peak window (65-105 GeV) and the
Kolmogorov distance of the background shape to the inclusive category shape. Writes OUT/<category>
.json and a .pdf with one panel per configuration (SR union, per class).

Usage: pixi run python scripts/sculpting_check.py OUT --category CAT --training-config JSON \\
           --config NAME=SKIM_DIR:SR_CSV [--config ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from vvv.background.qcd_massfit import assign_regions, load_sr_boxes
from vvv.ml.inference import CLASSES

MSD = "ScoutingFatPFJetRecluster_msoftdrop_1"
EDGES = np.arange(20.0, 251.0, 10.0)
PEAK = (65.0, 105.0)


def load(skim_dir, category, groups):
    parts = []
    for idx, (cls, members) in enumerate(groups.items()):
        for sample in members:
            path = Path(skim_dir) / sample / category / "nominal"
            if not path.exists():
                continue
            t = ds.dataset(path).to_table(columns=[f"proba_{c}" for c in CLASSES] + ["weight", MSD])
            parts.append({"proba": np.column_stack([t[f"proba_{c}"].to_numpy() for c in CLASSES]),
                          "w": t["weight"].to_numpy(), "m": t[MSD].to_numpy(), "y": np.full(t.num_rows, idx)})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def shape(m, w):
    h = np.histogram(m, EDGES, weights=w)[0]
    return h / h.sum() if h.sum() > 0 else h


def stats(m, w, ref):
    s = shape(m, w)
    peak = w[(m >= PEAK[0]) & (m < PEAK[1])].sum() / w.sum() if w.sum() > 0 else float("nan")
    return {"peak_fraction": float(peak), "ks_to_inclusive": float(np.abs(np.cumsum(s) - np.cumsum(ref)).max()),
            "yield": float(w.sum()), "entries": int(len(w))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--category", required=True)
    p.add_argument("--training-config", type=Path, required=True)
    p.add_argument("--config", action="append", required=True, help="NAME=SKIM_DIR:SR_CSV")
    args = p.parse_args()
    groups = json.loads(args.training_config.read_text())["class_groups"]
    names = list(groups)
    bkg = [names.index(c) for c in ("QCD", "VJets", "Top", "VV")]
    sig = [names.index(c) for c in ("VVV", "VH")]
    result, shapes = {}, {}
    for item in args.config:
        name, rest = item.split("=", 1)
        skim_dir, csv = rest.split(":", 1)
        ev = load(skim_dir, args.category, groups)
        region = assign_regions(ev["proba"], load_sr_boxes(csv, names), names)
        is_b = np.isin(ev["y"], bkg)
        ref = shape(ev["m"][is_b], ev["w"][is_b])
        res = {"inclusive_background": stats(ev["m"][is_b], ev["w"][is_b], ref)}
        for r in range(1, region.max() + 1):
            sel = (region == r) & is_b
            res[f"SR{r}_background"] = stats(ev["m"][sel], ev["w"][sel], ref)
        u = region > 0
        res["SR_union_background"] = stats(ev["m"][u & is_b], ev["w"][u & is_b], ref)
        res["SR_union_signal"] = stats(ev["m"][u & np.isin(ev["y"], sig)], ev["w"][u & np.isin(ev["y"], sig)], ref)
        for c in bkg:
            sel = u & (ev["y"] == c)
            res[f"SR_union_{names[c]}"] = stats(ev["m"][sel], ev["w"][sel], shape(ev["m"][ev["y"] == c], ev["w"][ev["y"] == c]))
        result[name] = res
        shapes[name] = {"inclusive background": ref, "SR union background": shape(ev["m"][u & is_b], ev["w"][u & is_b]),
                        "SR union signal": shape(ev["m"][u & np.isin(ev["y"], sig)], ev["w"][u & np.isin(ev["y"], sig)])}
        print(name, json.dumps({k: round(v["peak_fraction"], 3) for k, v in res.items()}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.category}.json").write_text(json.dumps(result, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(shapes), figsize=(5 * len(shapes), 4), squeeze=False, sharey=True)
    for ax, (name, sh) in zip(axes[0], shapes.items()):
        for label, h in sh.items():
            ax.stairs(h, EDGES, label=label)
        ax.axvspan(*PEAK, color="grey", alpha=0.15)
        ax.set_title(f"{args.category}: {name}")
        ax.set_xlabel("msoftdrop_1 [GeV]")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("normalised")
    fig.tight_layout()
    fig.savefig(args.out / f"{args.category}.pdf")
    print(f"wrote {args.out / args.category}.json/.pdf")


if __name__ == "__main__":
    main()
