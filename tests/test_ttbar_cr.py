import awkward as ak
import numpy as np
import pytest
from coffea.nanoevents.methods import candidate

from vvv.config import TTBAR_CR
from vvv.corrections.jms_fit import PassFail, edges, fit_peak, measure, model
from vvv.processors.ttbar_cr import MU_INT_FIELDS, PROBS, _gen_value, cr_objects, cr_variables, tree_mask

ak.behavior.update(candidate.behavior)

# Inputs of the official scouting jet ID (passing), plus the b score.
AK4_ID = {"chHEF": 0.6, "neHEF": 0.1, "chEmEF": 0.0, "neEmEF": 0.1, "muEF": 0.0, "chHadMultiplicity": 5,
          "elMultiplicity": 0, "muMultiplicity": 0, "neHadMultiplicity": 3, "phMultiplicity": 2,
          "hfHadMultiplicity": 0, "hfEMMultiplicity": 0, "scoutUParT_probb": 0.9}
AK8_EXTRA = {"msoftdrop": 80.0, **{f"scoutGlobalParT_prob_{p}": 0.1 for p in PROBS}}
MU_EXTRA = {"trackIso": 1.0, "ecalIso": 0.5, "hcalIso": 0.5, "normchi2": 1.0, "trk_dxy": 0.01, "trk_dz": 0.02,
            **{f: 8 for f in MU_INT_FIELDS}}


def objects(extra, *events):
    """Jagged PtEtaPhiM records from lists of (pt, eta, phi, mass), with constant extra fields."""
    fields = ("pt", "eta", "phi", "mass")
    record = {f: ak.Array([[o[i] for o in ev] for ev in events]) for i, f in enumerate(fields)}
    for name, value in extra.items():
        record[name] = ak.Array([[value for _ in ev] for ev in events])
    return ak.zip(record, with_name="PtEtaPhiMLorentzVector")


def example():
    # Event 0: AK8 at (eta 0, phi 0) is the probe (dphi to the muon = 3); the harder AK8 at
    # eta 3 is in ak8 but not a probe. AK4: one within 0.8 of an AK8 (removed), one too soft.
    # Event 1: the only muon is within 0.8 of the AK8, so no muon and no probe.
    ak8 = objects(AK8_EXTRA, [(300.0, 0.0, 0.0, 90.0), (400.0, 3.0, 3.0, 90.0)], [(300.0, 0.0, 0.0, 90.0)])
    ak4 = objects(AK4_ID, [(60.0, 0.0, 0.5, 5.0), (50.0, -1.5, -1.5, 5.0), (25.0, 1.0, 1.0, 5.0)], [(50.0, -1.5, -1.5, 5.0)])
    mu = objects({**MU_EXTRA, "charge": -1}, [(50.0, 0.0, 3.0, 0.1)], [(50.0, 0.3, 0.3, 0.1)])
    return cr_objects(ak4, ak8, mu, TTBAR_CR)


def test_cr_collections_and_probe():
    obj = example()
    assert ak.num(obj["ak8"]).tolist() == [2, 1]
    assert obj["ak8"].pt.tolist()[0] == [400.0, 300.0]
    assert obj["ak4"].pt.tolist() == [[50.0], [50.0]]
    assert ak.num(obj["mu_vtx"]).tolist() == [1, 0]
    assert obj["probe"].pt.tolist() == [[300.0], []]
    assert obj["probe_dphi"].tolist()[0] == pytest.approx([3.0])
    obj["jet_veto"] = np.array([False, False])
    assert tree_mask(obj).tolist() == [True, False]
    obj["jet_veto"] = np.array([True, False])  # rejected by the jet veto map
    assert tree_mask(obj).tolist() == [False, False]


def test_cr_variables_slots_and_leptonic_w():
    obj = example()
    v = cr_variables(obj, np.array([100.0, 10.0]), np.array([3.0, 0.0]))
    assert v["n_probe"].tolist() == [1, 0]
    assert v["Probe_pt_1"].tolist() == [300.0, -99.0]
    # Unfilled slots: -99 for float branches, 0 for int and bool ones (legacy branch defaults).
    assert v["ScoutingMuonVtx_pt_2"][0] == -99.0
    assert v["ScoutingMuonVtx_charge_2"][0] == 0
    assert v["ScoutingMuonVtx_mediumId_1"][0] == 1
    assert v["ScoutingMuonVtx_relIso_1"][0] == pytest.approx(2.0 / 50.0)
    assert v["LeptonicW_pt"][0] == pytest.approx(150.0)  # muon and MET collinear at phi = 3
    assert v["MuonMET_mT"][0] == pytest.approx(0.0, abs=1e-6)
    assert v["Probe_WvsQCD_1"][0] == pytest.approx(2.0 / 3.0)
    assert v["ScoutingPFJetRecluster2_scoutUParT_probb_second_max"][0] == -99.0


def test_gen_value_bounds():
    gen = ak.Array([[11, 22, 33], []])
    idx = ak.Array([[2, -1, 3], [0]])
    assert _gen_value(gen, idx, 0).tolist() == [[33, 0, 0], [0]]


def test_fit_recovers_peak_and_ratios():
    cfg = {"fit_window": [50, 120], "n_bins": 17, "selection": {}, "mass_branch": "m", "tag_branch": "t",
           "tag_wp": 0.2, "w_mass_ref": 80.38}
    bins = edges(cfg)
    centers = 0.5 * (bins[:-1] + bins[1:])
    truth = [400.0, 84.0, 8.0, 30.0, 0.02, 50.0]
    counts = model(centers, *truth)
    fit = fit_peak(centers, counts, np.sqrt(counts), (50, 120), "MC")
    assert fit["mu"] == pytest.approx(84.0, rel=1e-6)
    assert fit["sigma"] == pytest.approx(8.0, rel=1e-6)
    assert fit["ndf"] == 12

    def passfail(mu, sigma):
        h = PassFail(17)
        h.sumw["pass"] = model(centers, 400.0, mu, sigma, 30.0, 0.02, 50.0)
        h.sumw2["pass"] = h.sumw["pass"].copy()
        return h

    result = measure(passfail(88.2, 10.0), passfail(84.0, 8.0), cfg, 1.0)
    assert result["JMS"] == pytest.approx(1.05, rel=1e-6)
    assert result["JMR"] == pytest.approx(1.25, rel=1e-6)
    assert set(result) >= {"JMS", "JMS_err", "JMR", "JMR_err", "pass", "yields"}


def test_passfail_histogram_like_legacy():
    cfg = {"fit_window": [50, 120], "n_bins": 17, "selection": {"x": [0, None]}, "mass_branch": "m",
           "tag_branch": "t", "tag_wp": 0.2}
    cols = {"m": np.array([50.0, 120.0, 80.0, 80.0, 80.0]), "t": np.array([0.5, 0.5, 0.2, 0.9, 0.9]),
            "x": np.array([1.0, 1.0, 1.0, -999.0, -1.0])}
    h = PassFail(17)
    h.add_events(cols, np.full(5, 2.0), cfg)
    # np.histogram includes the upper edge; tag > wp is pass; the sentinel and x <= 0 fail the selection.
    assert h.sumw["pass"].sum() == 4.0 and h.sumw["pass"][-1] == 2.0
    assert h.sumw["fail"].sum() == 2.0
    assert h.sumw2["pass"].sum() == 8.0
