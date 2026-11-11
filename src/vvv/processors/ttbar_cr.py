"""Semileptonic ttbar control region: the tag-and-probe sample of the JMS/JMR fit.

Port of the legacy selections/convert_ttbar_cr conversion (selection.json and
branch.json, tree ttbar_cr). Collections, in legacy names:

- ak8: every stored AK8 jet (first 20), pT-sorted, no cut;
- ak4: AK4 jet ID (as the analysis) and deltaR(ak8) > 0.8;
- mu_vtx: ScoutingMuonVtx with pT > 10, |eta| < 2.4, deltaR(ak8) > 0.8, no ID
  (the ID cuts are applied by the fit, on the stored branches);
- probe: the AK8 jets with |eta| < 2.4 and |dphi(mu_vtx[0])| > 1.5708.

Tree selection: >= 1 mu_vtx, >= 1 ak4, >= 1 ak8, and the jet veto map. The jet
corrections are the analysis ones (lepton subtraction, JEC, rho recovery, JER, Type-1
MET) without JMS/JMR.

Per chunk and jet configuration (nominal, and for MC optionally JES/JER up/down):

- skim (optional, nominal): the legacy ttbar_cr branches, the event weight and
  the pileup weight ratios, to Parquet (<skim_dir>/<dataset>/ttbar_cr/nominal/);
- histograms of the events passing the fit selection (configs/jms_jmr.json),
  split by the probe W tag (pass/fail): the JMS/JMR fit histogram (fit window
  and binning of the config) and the CR control distributions.
"""

import awkward as ak
import hist
import numpy as np
from coffea import processor
from coffea.lumi_tools import LumiMask

from vvv.config import TtbarCRConfig
from vvv.corrections.jets import JetCorrector
from vvv.ml.inference import threshold_mask
from vvv.objects import ak4_jet_id, kinematic_mask, min_delta_r, p4_pt_eta, sort_by_pt
from vvv.processors.analysis import _chunk_id
from vvv.objects import identified_leptons
from vvv.selection import JetVetoMap, corrected_collections, input_collections
from vvv.variables import AK4, AK8, DEF, Kin, _max_value, _nth_max, _safe_div, _slot, _wrap, passthrough
from vvv.weights import SampleWeights

TREE = "ttbar_cr"
MU = "ScoutingMuonVtx"
GEN_MAX = 1024  # legacy GenPart max_size
PROBS = ("Xud", "Xcs", "Xqq", "Xbb", "Xcc", "Xss", "QCD")
MU_FLOAT_FIELDS = ("pt", "eta", "phi", "trackIso", "ecalIso", "hcalIso", "normchi2", "trk_dxy", "trk_dz")
MU_INT_FIELDS = (
    "charge", "type", "nValidStandAloneMuonHits", "nValidPixelHits", "nTrackerLayersWithMeasurement",
    "nStandAloneMuonMatchedStations", "nRecoMuonMatchedStations", "nValidRecoMuonHits",
)
# Unfilled slots keep the branch default: def (-99) for float branches, 0 for int and bool ones.
INT_DEFAULT = 0
# Control distributions: branch -> (bins, lo, hi, label).
CONTROL = {
    "Probe_msoftdrop_1": (50, 0.0, 250.0, r"Probe $m_{SD}$ [GeV]"),
    "Probe_pt_1": (40, 200.0, 1000.0, r"Probe $p_T$ [GeV]"),
    "Probe_WvsQCD_1": (50, 0.0, 1.0, "Probe W vs QCD"),
    "LeptonicW_pt": (40, 150.0, 950.0, r"$p_T(\mu + p_T^{miss})$ [GeV]"),
    "MuonMET_mT": (40, 0.0, 400.0, r"$m_T(\mu, p_T^{miss})$ [GeV]"),
    "ScoutingMuonVtx_pt_1": (40, 30.0, 430.0, r"Muon $p_T$ [GeV]"),
    "ScoutingMET_pt": (40, 0.0, 800.0, r"$p_T^{miss}$ [GeV]"),
    "ScoutingPFJetRecluster2_selected_HT": (40, 0.0, 2000.0, r"AK4 $H_T$ [GeV]"),
    "ScoutingPFJetRecluster2_scoutUParT_probb_max": (30, 0.85, 1.0, "max AK4 b score"),
    "nScoutingPFJetRecluster2_selected": (10, 0.0, 10.0, "AK4 jets"),
}


def cr_preselection(events, cfg: TtbarCRConfig, lumi_mask=None):
    """Legacy event_preselection: the SingleMuon trigger and a stored muon (before truncation)."""
    prefix, path = cfg.trigger.split("_", 1)
    mask = events[prefix][path] & (ak.num(events[MU]) >= 1)
    if lumi_mask is not None:
        mask = mask & lumi_mask(events.run, events.luminosityBlock)
    return mask


def _massless(objects):
    return Kin.of(ak.zip({"pt": objects.pt, "eta": objects.eta, "phi": objects.phi, "mass": ak.zeros_like(objects.pt)}))


def cr_objects(ak4_in, ak8_in, muons_in, cfg: TtbarCRConfig):
    """The legacy CR collections from corrected AK4/AK8 jets and stored muons."""
    obj_cfg = cfg.analysis.objects
    ak8 = sort_by_pt(ak8_in)
    ak4 = sort_by_pt(ak4_in)
    ak4 = ak4[ak4_jet_id(ak4, obj_cfg) & (min_delta_r(ak4, ak8) > obj_cfg.clean_dr)]
    muons = sort_by_pt(muons_in)
    muons = muons[kinematic_mask(muons, cfg.mu_min_pt, cfg.mu_max_abs_eta) & (min_delta_r(muons, ak8) > obj_cfg.clean_dr)]

    # deltaPhi of TLorentzVectors: Phi_mpi_pi(phi(jet) - phi(mu_vtx[0])).
    has_mu = ak.num(muons) >= 1
    mu_phi = ak.fill_none(ak.firsts(_massless(muons).phi), 0.0)
    dphi = abs(_wrap(Kin.of(ak8).phi - mu_phi))
    _, eta = p4_pt_eta(ak8)
    in_probe = has_mu & (abs(eta) < cfg.probe_max_abs_eta) & (dphi > cfg.probe_min_abs_dphi)
    return {"ak8": ak8, "ak4": ak4, "mu_vtx": muons, "probe": ak8[in_probe], "probe_dphi": dphi[in_probe]}


def tree_mask(obj):
    """>= 1 of each of mu_vtx, ak4 and ak8, in events kept by the jet veto map."""
    return (ak.num(obj["mu_vtx"]) >= 1) & (ak.num(obj["ak4"]) >= 1) & (ak.num(obj["ak8"]) >= 1) & ~obj["jet_veto"]


def _gen_value(gen_values, idx, default):
    """Legacy value_at(GenPart, idx, field, default) per object: default outside [0, nGenPart)."""
    counts = ak.to_numpy(ak.num(idx))
    n_gen = ak.to_numpy(ak.num(gen_values))
    starts = np.cumsum(n_gen) - n_gen
    flat_idx = ak.to_numpy(ak.flatten(idx)).astype(np.int64)
    evt = np.repeat(np.arange(len(counts)), counts)
    ok = (flat_idx >= 0) & (flat_idx < np.minimum(n_gen, GEN_MAX)[evt])
    flat_gen = ak.to_numpy(ak.flatten(gen_values)).astype(np.int64)
    out = np.full(len(flat_idx), default, dtype=np.int64)
    out[ok] = flat_gen[starts[evt][ok] + flat_idx[ok]]
    return ak.unflatten(out, counts)


def _gen_ids(objects, gen):
    """(pdgId, mother pdgId) of each object's genPartIdx, 0 if none."""
    pdg = _gen_value(gen.pdgId, objects.genPartIdx, 0)
    mother = _gen_value(gen.genPartIdxMother, objects.genPartIdx, -1)
    return pdg, _gen_value(gen.pdgId, mother, 0)


def _w_vs_qcd(jets):
    w = jets.scoutGlobalParT_prob_Xud + jets.scoutGlobalParT_prob_Xcs
    return _safe_div(w, w + jets.scoutGlobalParT_prob_QCD, 0.0)


def _v_vs_qcd(jets):
    v = sum(jets[f"scoutGlobalParT_prob_{p}"] for p in PROBS if p != "QCD")
    return _safe_div(v, v + jets.scoutGlobalParT_prob_QCD, 0.0)


def _f64(values):
    return ak.values_astype(values, np.float64)


def cr_variables(obj, met_pt, met_phi, gen=None):
    """The legacy ttbar_cr branches (branch.json formulas) per event; gen: GenPart (MC)."""
    ak8, ak4, mu, probe = obj["ak8"], obj["ak4"], obj["mu_vtx"], obj["probe"]
    met_pt = np.asarray(met_pt, dtype=np.float64)
    met_phi = np.asarray(met_phi, dtype=np.float64)
    out = {
        "nScoutingMuon_selected": ak.num(mu),
        "nScoutingPFJetRecluster2_selected": ak.num(ak4),
        "nScoutingFatPFJetRecluster_selected": ak.num(ak8),
        "n_probe": ak.num(probe),
        "ScoutingMET_pt": met_pt,
        "ScoutingMET_phi": met_phi,
        "ScoutingPFJetRecluster2_selected_HT": ak.sum(Kin.of(ak4).pt, axis=1),
    }
    mu_kin = _massless(mu)
    mu_pt = ak.fill_none(ak.firsts(mu_kin.pt), np.nan)
    mu_phi = ak.fill_none(ak.firsts(mu_kin.phi), np.nan)
    out["MuonMET_mT"] = np.sqrt(2 * mu_pt * met_pt * (1 - np.cos(mu_phi - met_phi)))
    out["LeptonicW_pt"] = np.sqrt(
        (mu_pt * np.cos(mu_phi) + met_pt * np.cos(met_phi)) ** 2 + (mu_pt * np.sin(mu_phi) + met_pt * np.sin(met_phi)) ** 2
    )

    def slots(prefix, values, n, default=DEF):
        for i in range(n):
            out[f"{prefix}_{i + 1}"] = _slot(values, i, default)

    for name in MU_FLOAT_FIELDS:
        slots(f"{MU}_{name}", mu[name], 2)
    for name in MU_INT_FIELDS:
        slots(f"{MU}_{name}", mu[name], 2, INT_DEFAULT)
    iso = _f64(mu.trackIso) + _f64(mu.ecalIso) + _f64(mu.hcalIso)
    slots(f"{MU}_relIso", _safe_div(iso, _f64(mu.pt), -1.0), 2)
    slots(f"{MU}_trkRelIso", _safe_div(_f64(mu.trackIso), _f64(mu.pt), -1.0), 2)
    medium = (
        (mu.nValidStandAloneMuonHits > 0) & (mu.normchi2 < 3) & (mu.nTrackerLayersWithMeasurement > 5)
        & (mu.nValidPixelHits > 0) & (mu.nRecoMuonMatchedStations >= 2)
    )
    slots(f"{MU}_mediumId", ak.values_astype(medium, np.int32), 2, INT_DEFAULT)

    for name in ("pt", "eta", "phi", "mass", "scoutUParT_probb"):
        slots(f"{AK4}_{name}", ak4[name], 4)
    for name in ("pt", "eta", "phi", "mass", "msoftdrop"):
        slots(f"{AK8}_{name}", ak8[name], 2)
    slots(f"{AK8}_WvsQCD", _w_vs_qcd(ak8), 2)
    slots(f"{AK8}_VvsQCD", _v_vs_qcd(ak8), 2)

    for name in ("pt", "eta", "phi", "mass", "msoftdrop"):
        slots(f"Probe_{name}", probe[name], 1)
    slots("Probe_dphi_mu", obj["probe_dphi"], 1)
    slots("Probe_WvsQCD", _w_vs_qcd(probe), 1)
    for p in PROBS:
        slots(f"Probe_prob_{p}", probe[f"scoutGlobalParT_prob_{p}"], 1)

    probb = ak4.scoutUParT_probb
    out[f"{AK4}_scoutUParT_probb_max"] = _max_value(probb, DEF)
    out[f"{AK4}_scoutUParT_probb_second_max"] = _nth_max(probb, 2, DEF)

    if gen is not None:
        pdg, mother = _gen_ids(mu, gen)
        slots(f"{MU}_genPartPdgId", pdg, 2, INT_DEFAULT)
        slots(f"{MU}_genPartMotherPdgId", mother, 2, INT_DEFAULT)
        pdg, mother = _gen_ids(ak4, gen)
        slots(f"{AK4}_partonFlavour", ak4.partonFlavour, 4, INT_DEFAULT)
        slots(f"{AK4}_genPartPdgId", pdg, 4, INT_DEFAULT)
        slots(f"{AK4}_genPartMotherPdgId", mother, 4, INT_DEFAULT)
        for prefix, jets, n in ((AK8, ak8, 2), ("Probe", probe, 1)):
            pdg, mother = _gen_ids(jets, gen)
            slots(f"{prefix}_topWCategory", jets.topWCategory, n, INT_DEFAULT)
            slots(f"{prefix}_partonFlavour", jets.partonFlavour, n, INT_DEFAULT)
            slots(f"{prefix}_genPartPdgId", pdg, n, INT_DEFAULT)
            slots(f"{prefix}_genPartMotherPdgId", mother, n, INT_DEFAULT)
    return {k: ak.to_numpy(v) for k, v in out.items()}


class TtbarCRSelection:
    """Corrections (no JMS/JMR) + CR objects for every requested jet configuration, with the
    analysis lepton subtraction from the jets and the jet veto map."""

    def __init__(self, cfg: TtbarCRConfig):
        from dataclasses import replace

        self.cfg = cfg
        self.corrector = JetCorrector(replace(cfg.analysis.jets, apply_jms_jmr=False), cfg.analysis.rho)
        self.jet_veto = JetVetoMap(cfg.analysis)

    def __call__(self, events, is_mc, configurations=("nominal",)):
        ak4, ak8, electrons, muons = input_collections(events, self.cfg.analysis)
        identified = identified_leptons(electrons, muons, self.cfg.analysis.objects)
        rho = None
        if self.cfg.analysis.jets.rho_source == "branch":
            rho = events[self.cfg.analysis.jets.rho_branch]
        state = self.corrector.prepare(ak4, ak8, events.ScoutingMET, events.event, is_mc, identified, rho=rho,
                                       gen_jets=events.GenJet if is_mc else None)
        out = {}
        for name in configurations:
            if name != "nominal" and not is_mc:
                continue
            ak4_c, ak8_c, k = corrected_collections(self.corrector, state, ak4, ak8, name)
            obj = cr_objects(ak4_c, ak8_c, muons, self.cfg)
            obj["met_pt"], obj["met_phi"] = k["met_pt"], k["met_phi"]
            obj["jet_veto"] = self.jet_veto(ak4_c)
            out[name] = obj
        return out, {"rho_ok": state.rho_ok}


def fit_hist(fit_cfg):
    lo, hi = fit_cfg["fit_window"]
    return hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.StrCategory([], name="syst", growth=True),
        hist.axis.StrCategory(["pass", "fail"], name="tag"),
        hist.axis.Regular(int(fit_cfg["n_bins"]), lo, hi, name="mass", label=r"Probe $m_{SD}$ [GeV]"),
        storage=hist.storage.Weight(),
    )


def control_hist(name):
    bins, lo, hi, label = CONTROL[name]
    return hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.StrCategory([], name="syst", growth=True),
        hist.axis.StrCategory(["pass", "fail"], name="tag"),
        hist.axis.Regular(bins, lo, hi, name="x", label=label),
        storage=hist.storage.Weight(),
    )


class TtbarCRProcessor(processor.ProcessorABC):
    def __init__(self, cfg: TtbarCRConfig, samples, metadata, lumi_fb, fit_cfg, skim_dir=None, jet_variations=False):
        """
        samples: {name: Sample}; metadata: {name: metadata record} (MC);
        fit_cfg: the JMS/JMR fit config (configs/jms_jmr.json): selection, tag_branch, tag_wp,
        fit_window, n_bins; the histograms are filled for the events passing its selection.
        """
        self.cfg = cfg
        self.samples = samples
        self.metadata = metadata
        self.lumi_fb = lumi_fb
        self.fit_cfg = fit_cfg
        self.skim_dir = skim_dir
        self.jet_variations = jet_variations
        self._selection = None

    # Built lazily on the worker: correctionlib objects do not pickle.
    @property
    def selection(self):
        if self._selection is None:
            self._selection = TtbarCRSelection(self.cfg)
        return self._selection

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_selection"] = None
        return state

    def process(self, events):
        dataset = events.metadata["dataset"]
        sample = self.samples[dataset]
        out = {
            "cutflow": {dataset: {"all": len(events)}},
            "fit": fit_hist(self.fit_cfg),
            **{name: control_hist(name) for name in CONTROL},
            "rho_failures": {dataset: 0},
        }
        lumi_mask = None if sample.is_mc else LumiMask(str(self.cfg.analysis.lumi_mask))
        events = events[ak.to_numpy(cr_preselection(events, self.cfg, lumi_mask))]
        out["cutflow"][dataset]["preselection"] = len(events)
        if len(events) == 0:
            return out

        weights = SampleWeights(sample, self.metadata.get(dataset, {}))
        w_nominal = weights.nominal(events) * (self.lumi_fb if sample.is_mc else 1.0)
        configurations = ("nominal",) + (self.cfg.jet_variations if sample.is_mc and self.jet_variations else ())
        selected, info = self.selection(events, sample.is_mc, configurations)
        out["rho_failures"][dataset] = int((~info["rho_ok"]).sum())
        gen = events.GenPart if sample.is_mc and "GenPart" in events.fields else None

        for config_name, obj in selected.items():
            mask = ak.to_numpy(tree_mask(obj))
            if config_name == "nominal":
                out["cutflow"][dataset]["jet_veto_rejected"] = int(obj["jet_veto"].sum())
                out["cutflow"][dataset][TREE] = int(mask.sum())
            if not mask.any():
                continue
            columns = {k: v[mask] for k, v in cr_variables(obj, obj["met_pt"], obj["met_phi"], gen).items()}
            w = w_nominal[mask]
            if self.skim_dir is not None and config_name == "nominal":
                ids = {k: v[mask] for k, v in passthrough(events, sample.is_mc).items()}
                trig = {f"DST_{p}": ak.to_numpy(events.DST[p])[mask] for p in ("PFScouting_SingleMuon", "PFScouting_JetHT")}
                ids["nScoutingPrimaryVertex"] = ak.to_numpy(ak.num(events.ScoutingPrimaryVertex))[mask]
                variations = {k: v[mask] for k, v in weights.variations(events).items() if v.ndim == 1}
                self._write_skim(dataset, events.metadata, {**ids, **trig, **columns}, w, variations)
            self._fill(out, dataset, config_name, columns, w)
        return out

    def _fill(self, out, dataset, config_name, columns, w):
        keep = threshold_mask(columns, self.fit_cfg["selection"])
        if config_name == "nominal":
            out["cutflow"][dataset]["fit_selection"] = int(keep.sum())
        if not keep.any():
            return
        tag = np.where(columns[self.fit_cfg["tag_branch"]][keep] > float(self.fit_cfg["tag_wp"]), "pass", "fail")
        fill = dict(dataset=dataset, syst=config_name, tag=tag, weight=w[keep])
        out["fit"].fill(mass=np.asarray(columns[self.fit_cfg["mass_branch"]][keep], dtype=np.float64), **fill)
        for name in CONTROL:
            out[name].fill(x=np.asarray(columns[name][keep], dtype=np.float64), **fill)

    def _write_skim(self, dataset, metadata, columns, w, w_variations):
        import os

        import pyarrow as pa
        import pyarrow.parquet as pq

        table = {k: np.asarray(v) for k, v in columns.items()}
        table["weight"] = w
        for name, ratio in w_variations.items():
            table[f"weight_{name}"] = ratio
        path = self.skim_dir / dataset / TREE / "nominal"
        path.mkdir(parents=True, exist_ok=True)
        chunk_id = _chunk_id(metadata)
        # Write-then-rename, as the analysis skims.
        tmp = path / f".{chunk_id}.parquet.tmp"
        pq.write_table(pa.table(table), tmp, compression="zstd")
        os.replace(tmp, path / f"{chunk_id}.parquet")

    def postprocess(self, accumulator):
        return accumulator
