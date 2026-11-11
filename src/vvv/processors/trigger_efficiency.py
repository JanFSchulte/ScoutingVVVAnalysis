"""Trigger efficiencies of the analysis selection versus reconstructed objects.

Events: the preselection without its trigger requirement (>= 2 stored AK8 jets, and the golden
JSON for data) with the nominal jet corrections. Regions: "preselection" (all those events kept
by the jet veto map) and the five categories without their trigger requirement (EventSelection
apply_triggers=False; the jet veto map applies). Per
region, the denominator ("all") and the events that fired each entry of TRIGGERS are filled
versus

- ak8_ht: the scalar pT sum of the selected AK8 jets (pT > 175 GeV and |eta| < 2.5 after JEC/JER);
- mu_pt, mu_eta and ele_pt, ele_eta: the leading muon and the leading electron of lep_veto_c2
  (the veto leptons cleaned against the two V-candidate AK8 jets), in events that have one.

MC: every event, with the full MC weight (efficiency = passed / all). Data: only the events that
fired the reference path, unweighted (efficiency = passed and reference / reference). The data
production keeps only JetHT or SingleMuon events (config.DATA_PRODUCTION_TRIGGERS), so a reference
outside that filter, such as the default DST_PFScouting_ZeroBias, finds almost no events in the
current data: it needs a production that keeps the events of the reference path.
"""

import awkward as ak
import hist
import numpy as np
from coffea import processor
from coffea.lumi_tools import LumiMask

from vvv.config import AnalysisConfig
from vvv.objects import CATEGORIES
from vvv.selection import EventSelection, trigger_bit
from vvv.weights import SampleWeights

DEFAULT_REFERENCE = "DST_PFScouting_ZeroBias"
REGIONS = ("preselection",) + CATEGORIES
# Numerators: name -> OR of these paths; "all" is the denominator.
TRIGGERS = {
    "all": (),
    "JetHT": ("DST_PFScouting_JetHT",),
    "SingleMuon": ("DST_PFScouting_SingleMuon",),
    "JetHT_or_SingleMuon": ("DST_PFScouting_JetHT", "DST_PFScouting_SingleMuon"),
    "DoubleMuon": ("DST_PFScouting_DoubleMuon",),
    "DoubleEG": ("DST_PFScouting_DoubleEG",),
    "SinglePhotonEB": ("DST_PFScouting_SinglePhotonEB",),
}
VARIABLES = {
    "ak8_ht": hist.axis.Regular(54, 300.0, 3000.0, name="x", label=r"$H_T$ of the selected AK8 jets [GeV]"),
    "mu_pt": hist.axis.Regular(49, 10.0, 500.0, name="x", label=r"Leading muon $p_T$ [GeV]"),
    "mu_eta": hist.axis.Regular(24, -2.4, 2.4, name="x", label=r"Leading muon $\eta$"),
    "ele_pt": hist.axis.Regular(49, 10.0, 500.0, name="x", label=r"Leading electron $p_T$ [GeV]"),
    "ele_eta": hist.axis.Regular(25, -2.5, 2.5, name="x", label=r"Leading electron $\eta$"),
}


def efficiency_hist(variable):
    return hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.StrCategory(list(REGIONS), name="region"),
        hist.axis.StrCategory(list(TRIGGERS), name="trigger"),
        VARIABLES[variable],
        storage=hist.storage.Weight(),
    )


def _leading(leptons, field):
    return ak.to_numpy(ak.fill_none(ak.firsts(leptons[field]), np.nan)).astype(np.float64)


class TriggerEfficiencyProcessor(processor.ProcessorABC):
    def __init__(self, cfg: AnalysisConfig, samples, metadata, reference=DEFAULT_REFERENCE):
        """samples: {name: Sample}; metadata: {name: metadata record} (MC); reference: the
        reference path of the data."""
        self.cfg = cfg
        self.samples = samples
        self.metadata = metadata
        self.reference = reference
        self._selection = None

    # Built lazily on the worker: correctionlib objects do not pickle.
    @property
    def selection(self):
        if self._selection is None:
            self._selection = EventSelection(self.cfg)
        return self._selection

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_selection"] = None
        return state

    def process(self, events):
        dataset = events.metadata["dataset"]
        sample = self.samples[dataset]
        out = {"cutflow": {dataset: {"all": len(events)}}, **{v: efficiency_hist(v) for v in VARIABLES}}
        mask = ak.to_numpy(ak.num(events.ScoutingFatPFJetRecluster) > 1)
        if not sample.is_mc:
            mask &= ak.to_numpy(LumiMask(str(self.cfg.lumi_mask))(events.run, events.luminosityBlock))
            out["cutflow"][dataset]["preselection"] = int(mask.sum())
            mask &= trigger_bit(events, self.reference)
            out["cutflow"][dataset]["reference"] = int(mask.sum())
        else:
            out["cutflow"][dataset]["preselection"] = int(mask.sum())
        events = events[mask]
        if len(events) == 0:
            return out

        weight = SampleWeights(sample, self.metadata[dataset]).nominal(events) if sample.is_mc else np.ones(len(events))
        obj = self.selection(events, sample.is_mc, ("nominal",), apply_triggers=False)[0]["nominal"]
        fired = {
            name: (np.logical_or.reduce([trigger_bit(events, p) for p in paths]) if paths
                   else np.ones(len(events), dtype=bool))
            for name, paths in TRIGGERS.items()
        }
        leptons = obj["lep_veto_c2"]
        muons, electrons = leptons[~leptons.is_electron], leptons[leptons.is_electron]
        values = {
            "ak8_ht": ak.to_numpy(ak.sum(obj["ak8"].pt, axis=1)).astype(np.float64),
            "mu_pt": _leading(muons, "pt"),
            "mu_eta": _leading(muons, "eta"),
            "ele_pt": _leading(electrons, "pt"),
            "ele_eta": _leading(electrons, "eta"),
        }
        categories = {c: ak.to_numpy(m) for c, m in obj["categories"].items()}
        for category, in_category in categories.items():
            out["cutflow"][dataset][category] = int(in_category.sum())
        out["cutflow"][dataset]["jet_veto_rejected"] = int(obj["jet_veto"].sum())
        for region, in_region in {"preselection": ~obj["jet_veto"], **categories}.items():
            for variable, x in values.items():
                has = in_region & np.isfinite(x)
                for name, passed in fired.items():
                    sel = has & passed
                    if sel.any():
                        out[variable].fill(dataset=dataset, region=region, trigger=name, x=x[sel], weight=weight[sel])
        return out

    def postprocess(self, accumulator):
        return accumulator
