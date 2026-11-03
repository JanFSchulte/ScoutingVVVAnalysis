"""From NanoEvents to corrected objects and category masks, per jet configuration."""

import awkward as ak
import numpy as np

from vvv.config import AnalysisConfig
from vvv.corrections.jets import JetCorrector
from vvv.objects import build_objects, categories


def input_collections(events, cfg: AnalysisConfig):
    """Input collections as legacy reads them: truncated, AK4 raw-pT filtered."""
    c = cfg.inputs
    ak4 = events.ScoutingPFJetRecluster2[:, : c.max_ak4]
    ak4 = ak4[ak4.pt * (1.0 - ak4.rawFactor) >= c.ak4_min_raw_pt]
    ak8 = events.ScoutingFatPFJetRecluster[:, : c.max_ak8]
    electrons = events.ScoutingElectron[:, : c.max_electrons]
    muons = events.ScoutingMuonVtx[:, : c.max_muons]
    return ak4, ak8, electrons, muons


def preselection(events, cfg: AnalysisConfig, lumi_mask=None):
    """Legacy event_preselection, evaluated on the stored (uncorrected) inputs."""
    prefix, path = cfg.trigger.split("_", 1)
    mask = (ak.num(events.ScoutingFatPFJetRecluster) > 1) & events[prefix][path]
    if lumi_mask is not None:
        mask = mask & lumi_mask(events.run, events.luminosityBlock)
    return mask


def with_kinematics(collection, pt, mass, msoftdrop=None):
    collection = ak.with_field(collection, pt, "pt")
    collection = ak.with_field(collection, mass, "mass")
    if msoftdrop is not None:
        collection = ak.with_field(collection, msoftdrop, "msoftdrop")
    return collection


class EventSelection:
    """Corrections + objects + categories for every requested jet configuration."""

    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg
        self.corrector = JetCorrector(cfg.jets, cfg.rho)

    def __call__(self, events, is_mc, configurations=("nominal",)):
        ak4, ak8, electrons, muons = input_collections(events, self.cfg)
        rho = None
        if self.cfg.jets.rho_source == "branch":
            rho = events[self.cfg.jets.rho_branch]
        state = self.corrector.prepare(ak4, ak8, events.ScoutingMET, events.event, is_mc, rho=rho)

        out = {}
        for name in configurations:
            if name != "nominal" and not is_mc:
                continue
            k = self.corrector.apply(state, name)
            ak4_c = with_kinematics(ak4, k["ak4_pt"], k["ak4_mass"])
            ak8_c = with_kinematics(ak8, k["ak8_pt"], k["ak8_mass"], k["ak8_msoftdrop"])
            obj = build_objects(ak4_c, ak8_c, electrons, muons, self.cfg.objects)
            obj["met_pt"] = k["met_pt"]
            obj["met_phi"] = k["met_phi"]
            obj["categories"] = categories(obj, self.cfg.objects)
            out[name] = obj
        return out, {"rho": state.rho, "rho_ok": state.rho_ok, "n_rho_consensus": state.n_rho_consensus}


def category_index(cats):
    """-1 if no category, else the index in CATEGORIES (they are exclusive)."""
    idx = np.full(len(next(iter(cats.values()))), -1)
    for i, mask in enumerate(cats.values()):
        m = ak.to_numpy(mask)
        if np.any((idx >= 0) & m):
            raise RuntimeError("categories overlap")
        idx[m] = i
    return idx
