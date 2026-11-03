"""Event weights and weight-based variations.

Nominal MC weight per unit luminosity (fb^-1), as legacy kfold.py / gen_weight.py:

    w = xsec_fb * genWeight / sum(genEventSumw) * w_PU / <w_PU>

where <w_PU> is the mean pileup weight over all generated events. Theory
members are normalised to their generator-level value (Runs-tree sums), so
they change the selection efficiency only; the inclusive cross-section
uncertainty enters the datacards separately.
"""

import awkward as ak
import numpy as np

from vvv.processors.metadata import PU_EDGES

N_PDF = 101
N_SCALE = 9
N_PS = 4
N_ALPHAS = 2


def _lookup(edges, values, x):
    """Legacy lookupPileupWeight: [low, high) bins, last upper edge included, 0 outside."""
    x = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.where(x == edges[-1], len(values) - 1, idx)
    inside = (idx >= 0) & (idx < len(values))
    return np.where(inside, np.asarray(values)[np.clip(idx, 0, len(values) - 1)], 0.0)


class SampleWeights:
    def __init__(self, sample, metadata):
        self.sample = sample
        self.metadata = metadata
        if sample.is_mc:
            runs = metadata["runs"]
            self.norm = sample.xsec_fb / runs["genEventSumw"]
            self.pu = {k: np.asarray(v) for k, v in metadata["pileup_weights"].items()}
            self.pu_mean = metadata["pileup_mean"]
            self.theory_sums = {k: np.asarray(runs[k]) for k in ("LHEScaleSumw", "LHEPdfSumw", "PSSumw") if k in runs}

    def pileup(self, n_true_int, syst="nominal"):
        return _lookup(PU_EDGES, self.pu[syst], n_true_int) / self.pu_mean[syst]

    def nominal(self, events):
        """Per-event weight for 1 fb^-1 (data: 1)."""
        if not self.sample.is_mc:
            return np.ones(len(events))
        gen = ak.to_numpy(events.genWeight).astype(np.float64)
        return self.norm * gen * self.pileup(ak.to_numpy(events.Pileup.nTrueInt))

    def variations(self, events):
        """Multiplicative weight ratios to the nominal, per weight systematic.

        Returns {name: array(n_events)} for pileup up/down and
        {name: array(n_events, n_members)} for the theory member sets.
        """
        if not self.sample.is_mc:
            return {}
        n_true = ak.to_numpy(events.Pileup.nTrueInt)
        nominal_pu = self.pileup(n_true)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = {
                f"pileup_{d}": np.where(nominal_pu > 0, self.pileup(n_true, d) / nominal_pu, 0.0)
                for d in ("up", "down")
            }
        if not self.sample.has_theory_weights:
            return out
        if "LHEScaleWeight" in events.fields and "LHEScaleSumw" in self.theory_sums:
            scale = _standard_scale(ak.to_numpy(ak.fill_none(ak.pad_none(events.LHEScaleWeight, N_SCALE, clip=True), 1.0)), ak.to_numpy(ak.num(events.LHEScaleWeight)))
            stored = self.theory_sums["LHEScaleSumw"][:N_SCALE]
            sums = np.ones((1, N_SCALE))
            sums[0, : len(stored)] = stored
            sums = _standard_scale(sums, np.array([len(stored)]))[0]
            out["LHEScaleWeight"] = scale / sums
        if "LHEPdfWeight" in events.fields and "LHEPdfSumw" in self.theory_sums:
            n_members = N_PDF + N_ALPHAS
            pdf = ak.to_numpy(ak.fill_none(ak.pad_none(events.LHEPdfWeight, n_members, clip=True), 1.0))
            sums = np.ones(n_members)
            stored = self.theory_sums["LHEPdfSumw"][:n_members]
            sums[: len(stored)] = stored
            ratio = pdf / sums
            out["LHEPdfWeight"] = ratio[:, :N_PDF]
            if len(self.theory_sums["LHEPdfSumw"]) >= n_members:
                out["LHEPdfWeightAlphaS"] = ratio[:, N_PDF:]
        if "PSWeight" in events.fields and "PSSumw" in self.theory_sums:
            ps = ak.to_numpy(ak.fill_none(ak.pad_none(events.PSWeight, N_PS, clip=True), 1.0))
            sums = np.ones(N_PS)
            sums[: min(N_PS, len(self.theory_sums["PSSumw"]))] = self.theory_sums["PSSumw"][:N_PS]
            out["PSWeight"] = ps / sums
        return out


def _standard_scale(values, counts):
    """9-member (muR, muF) layout; an 8-member set (nominal omitted) gets 1.0 at index 4."""
    values = np.array(values, dtype=np.float64)
    eight = counts == N_SCALE - 1
    if eight.any():
        shifted = values[eight].copy()
        shifted[:, 5:] = values[eight][:, 4:8]
        shifted[:, 4] = 1.0
        values[eight] = shifted
    return values
