"""Pileup reweighting (port of legacy selections/weight/weight.C).

weight(nTrueInt) = data(bin) / mc(bin), with both profiles normalised to
their in-range integral; bins without MC get weight 0, and nTrueInt outside the
histogram range gets weight 0 (the data profile carries no probability there).
"nominal", "down" and "up" use the 69.2, 66.0 and 72.4 mb data profiles.
"""

import correctionlib.schemav2 as cs
import numpy as np
import uproot

from vvv.config import DATA_DIR

DATA_PROFILES = {
    "nominal": DATA_DIR / "pileup" / "dataPileupHistogram-2024GHI_processed-69200ub.root",
    "down": DATA_DIR / "pileup" / "dataPileupHistogram-2024GHI_processed-66000ub.root",
    "up": DATA_DIR / "pileup" / "dataPileupHistogram-2024GHI_processed-72400ub.root",
}


def load_data_profiles(paths=DATA_PROFILES, histogram="pileup"):
    edges, profiles = None, {}
    for syst, path in paths.items():
        h = uproot.open(path)[histogram]
        if edges is None:
            edges = h.axis().edges()
        elif not np.array_equal(edges, h.axis().edges()):
            raise ValueError(f"{path}: binning differs from the nominal profile")
        profiles[syst] = h.values(flow=False)
    return edges, profiles


def pileup_weights(mc_counts, data_profiles):
    """Per-bin weights for each systematic from in-range MC counts."""
    mc = np.asarray(mc_counts, dtype=np.float64)
    if mc.sum() <= 0:
        raise ValueError("MC pileup histogram has no entries in range")
    mc = mc / mc.sum()
    out = {}
    for syst, values in data_profiles.items():
        data = np.asarray(values, dtype=np.float64)
        data = data / data.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            out[syst] = np.where(mc > 0, data / np.where(mc > 0, mc, 1.0), 0.0)
    return out


def mean_weights(mc_counts, n_total, weights):
    """Mean weight over all generated events (out-of-range events weigh 0)."""
    mc = np.asarray(mc_counts, dtype=np.float64)
    return {syst: float(np.sum(mc * w) / n_total) for syst, w in weights.items()}


def make_correction(name, edges, weights, description=""):
    """correctionlib Correction: inputs (nTrueInt, systematic) -> weight."""
    content = []
    keys = []
    for syst, w in weights.items():
        keys.append(syst)
        content.append(
            cs.CategoryItem(
                key=syst,
                value=cs.Binning(
                    nodetype="binning",
                    input="nTrueInt",
                    edges=[float(e) for e in edges],
                    content=[float(x) for x in w],
                    flow=0.0,
                ),
            )
        )
    return cs.Correction(
        name=name,
        description=description or "Pileup weight data/MC in Pileup_nTrueInt",
        version=1,
        inputs=[
            cs.Variable(name="nTrueInt", type="real", description="Pileup_nTrueInt"),
            cs.Variable(name="systematic", type="string", description=" / ".join(keys)),
        ],
        output=cs.Variable(name="weight", type="real"),
        data=cs.Category(nodetype="category", input="systematic", content=content),
    )


def correction_set(corrections):
    return cs.CorrectionSet(schema_version=2, corrections=list(corrections))


def from_legacy_csv(path):
    """(edges, {"nominal", "down", "up"} weights) from a legacy pileup CSV."""
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    edges = np.append(table[:, 0], table[-1, 1])
    return edges, {"nominal": table[:, 2], "down": table[:, 3], "up": table[:, 4]}
