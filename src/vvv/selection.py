"""From NanoEvents to corrected objects and category masks, per jet configuration."""

import awkward as ak
import correctionlib
import numpy as np

from vvv.config import AnalysisConfig
from vvv.corrections.jets import JetCorrector
from vvv.objects import CATEGORIES, build_objects, categories, identified_leptons, scouting_jet_id


def input_collections(events, cfg: AnalysisConfig):
    """Input collections as legacy reads them: truncated, AK4 raw-pT filtered."""
    c = cfg.inputs
    ak4 = events.ScoutingPFJetRecluster2[:, : c.max_ak4]
    ak4 = ak4[ak4.pt * (1.0 - ak4.rawFactor) >= c.ak4_min_raw_pt]
    ak8 = events.ScoutingFatPFJetRecluster[:, : c.max_ak8]
    electrons = events.ScoutingElectron[:, : c.max_electrons]
    muons = events.ScoutingMuonVtx[:, : c.max_muons]
    return ak4, ak8, electrons, muons


def trigger_bit(events, path):
    """The decision of one trigger path (DST_PFScouting_JetHT is events.DST.PFScouting_JetHT)."""
    prefix, name = path.split("_", 1)
    return ak.to_numpy(events[prefix][name]).astype(bool)


def trigger_paths(cfg: AnalysisConfig):
    """Every trigger path of the configuration once, in configuration order."""
    return tuple(dict.fromkeys(p for paths in cfg.triggers.values() for p in paths))


def trigger_columns(events, cfg: AnalysisConfig):
    """{path: decision} of every configured path (skim columns)."""
    return {path: trigger_bit(events, path) for path in trigger_paths(cfg)}


def category_triggers(events, cfg: AnalysisConfig):
    """{category: OR of its trigger paths} per event."""
    bits = trigger_columns(events, cfg)
    return {c: np.logical_or.reduce([bits[p] for p in paths]) for c, paths in cfg.triggers.items()}


def preselection(events, cfg: AnalysisConfig, lumi_mask=None):
    """Legacy event_preselection (>= 2 stored AK8 jets) with the OR of every configured trigger
    path, evaluated on the stored (uncorrected) inputs; each category then requires its own
    paths (EventSelection)."""
    fired = np.logical_or.reduce(list(trigger_columns(events, cfg).values()))
    mask = (ak.num(events.ScoutingFatPFJetRecluster) > 1) & fired
    if lumi_mask is not None:
        mask = mask & lumi_mask(events.run, events.luminosityBlock)
    return mask


def with_kinematics(collection, pt, eta, phi, mass, msoftdrop=None):
    for name, values in (("pt", pt), ("eta", eta), ("phi", phi), ("mass", mass)):
        collection = ak.with_field(collection, values, name)
    if msoftdrop is not None:
        collection = ak.with_field(collection, msoftdrop, "msoftdrop")
    return collection


def corrected_collections(corrector, state, ak4, ak8, name):
    """(AK4, AK8, corrected kinematics) of one jet configuration."""
    k = corrector.apply(state, name)
    ak4_c = with_kinematics(ak4, k["ak4_pt"], k["ak4_eta"], k["ak4_phi"], k["ak4_mass"])
    ak8_c = with_kinematics(ak8, k["ak8_pt"], k["ak8_eta"], k["ak8_phi"], k["ak8_mass"], k["ak8_msoftdrop"])
    return ak4_c, ak8_c, k


class JetVetoMap:
    """The JERC jet veto map: True for events to reject.

    Run 3 prescription: reject the event if any AK4 jet with fully corrected pT above
    jet_veto_min_pt, passing the jet ID and with (chEmEF + neEmEF) below jet_veto_max_em_fraction
    lies in a vetoed (eta, phi) bin (map value > 0), for data and MC alike. Evaluated per jet
    configuration, since the pT threshold sees the jet variations.
    """

    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg
        self.map = correctionlib.CorrectionSet.from_file(str(cfg.jet_veto_file))[cfg.jet_veto_map]

    def __call__(self, ak4):
        c = self.cfg
        em = ak4.chEmEF + ak4.neEmEF
        jets = ak4[(ak4.pt > c.jet_veto_min_pt) & scouting_jet_id(ak4, c.objects) & (em < c.jet_veto_max_em_fraction)]
        eta = ak.to_numpy(ak.flatten(jets.eta)).astype(np.float64)
        phi = ak.to_numpy(ak.flatten(jets.phi)).astype(np.float64)
        inputs = {"type": c.jet_veto_type, "eta": eta, "phi": phi}
        vetoed = self.map.evaluate(*(inputs[i.name] for i in self.map.inputs)) > 0
        return ak.to_numpy(ak.any(ak.unflatten(vetoed, ak.num(jets)), axis=1))


class EventSelection:
    """Corrections + objects + categories (with the jet veto map and their trigger paths) for
    every requested jet configuration."""

    def __init__(self, cfg: AnalysisConfig):
        if set(cfg.triggers) != set(CATEGORIES):
            raise ValueError(f"trigger paths are configured for {sorted(cfg.triggers)}, the categories are {list(CATEGORIES)}")
        self.cfg = cfg
        self.corrector = JetCorrector(cfg.jets, cfg.rho)
        self.jet_veto = JetVetoMap(cfg)

    def __call__(self, events, is_mc, configurations=("nominal",), apply_triggers=True):
        """apply_triggers=False gives the categories without their trigger requirement (the
        denominators of trigger efficiencies); the jet veto map applies in both cases."""
        triggered = category_triggers(events, self.cfg) if apply_triggers else None
        ak4, ak8, electrons, muons = input_collections(events, self.cfg)
        identified = identified_leptons(electrons, muons, self.cfg.objects)
        rho = None
        if self.cfg.jets.rho_source == "branch":
            rho = events[self.cfg.jets.rho_branch]
        state = self.corrector.prepare(ak4, ak8, events.ScoutingMET, events.event, is_mc, identified, rho=rho,
                                       gen_jets=events.GenJet if is_mc else None)

        out = {}
        for name in configurations:
            if name != "nominal" and not is_mc:
                continue
            ak4_c, ak8_c, k = corrected_collections(self.corrector, state, ak4, ak8, name)
            obj = build_objects(ak4_c, ak8_c, identified, self.cfg.objects)
            obj["met_pt"] = k["met_pt"]
            obj["met_phi"] = k["met_phi"]
            obj["jet_veto"] = self.jet_veto(ak4_c)
            kept = ~obj["jet_veto"]
            cats = categories(obj, self.cfg.objects)
            obj["categories"] = {c: m & kept & (triggered[c] if apply_triggers else True) for c, m in cats.items()}
            out[name] = obj
        return out, {"rho": state.rho, "rho_ok": state.rho_ok, "n_rho_consensus": state.n_rho_consensus,
                     "ak4_jer_matched": state.ak4_jer_matched}


def category_index(cats):
    """-1 if no category, else the index in CATEGORIES (they are exclusive)."""
    idx = np.full(len(next(iter(cats.values()))), -1)
    for i, mask in enumerate(cats.values()):
        m = ak.to_numpy(mask)
        if np.any((idx >= 0) & m):
            raise RuntimeError("categories overlap")
        idx[m] = i
    return idx
