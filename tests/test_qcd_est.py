import json

import hist
import numpy as np
import pytest
import uproot

from vvv.background.legacy import LegacySetup
from vvv.background.qcd_massfit import (MassfitAccumulator, MassfitConfig, MassfitInputs, assign_regions, estimate,
                                        fit_tfmc, mass_bin, sideband_transfer_factors, sr_csv_hash, summary,
                                        write_root)

CLASSES = ["VVV", "VH", "Top", "VV", "VJets", "QCD"]
BOXES = [(1, {"VVV": (0.5, 1.0)}), (2, {"VVV": (0.2, 0.5), "VV": (0.3, 0.6)})]
MSD = "ScoutingFatPFJetRecluster_msoftdrop_1"


def _events(seed=3, n=20000):
    rng = np.random.default_rng(seed)
    out = {}
    for c in CLASSES:
        proba = rng.dirichlet(np.ones(len(CLASSES)) * 0.7, n)
        mass = rng.uniform(15.0, 260.0, n)
        weight = rng.exponential(2.0, n) * (50.0 if c == "QCD" else 1.0)
        weight[:5] *= -1.0  # signed generator weights
        out[f"s_{c}"] = (c, proba, mass, weight)
    return out


def _accumulate(cfg, events, window=(50, 150)):
    acc = MassfitAccumulator(cfg, BOXES, CLASSES, {s: e[0] for s, e in events.items()}, ["QCD"], window, log=lambda m: None)
    for s, (_, proba, mass, weight) in events.items():
        half = len(mass) // 2  # two chunks: filling is additive
        acc.fill(s, mass[:half], weight[:half], proba=proba[:half])
        acc.fill(s, mass[half:], weight[half:], proba=proba[half:])
    return acc


def test_mass_bin_edges():
    edges = MassfitConfig().edges
    assert len(edges) == 24 and edges[0] == 20 and edges[-1] == 250
    idx = mass_bin(np.array([19.9, 20.0, 29.99, 30.0, 250.0, 250.1, np.nan, -999.0]), edges)
    assert idx.tolist() == [-1, 0, 0, 1, 22, -1, -1, -1]


def test_regions_overlap_raises():
    proba = np.array([[0.6, 0, 0, 0.4, 0, 0], [0.3, 0, 0, 0.4, 0, 0], [0.1, 0, 0, 0.9, 0, 0], [1.0, 0, 0, 0, 0, 0]])
    assert assign_regions(proba, BOXES, CLASSES).tolist() == [1, 2, 0, 1]  # upper edge 1 includes 1
    with pytest.raises(RuntimeError):
        assign_regions(proba, BOXES + [(3, {"VVV": (0.55, 0.7)})], CLASSES)


def test_tfmc_recovers_exp_polynomial():
    edges = MassfitConfig().edges
    c = 0.5 * (edges[:-1] + edges[1:])
    x = 2 * (c - c.min()) / (c.max() - c.min()) - 1
    F = np.full(len(c), 1e4)
    A = F * np.exp(-3.0 + 0.4 * x - 0.2 * x ** 2)
    tf = fit_tfmc(A, A * 1e-6, F, F * 1e-6, edges, 2)
    assert np.allclose(tf["coefficients"], [-3.0, 0.4, -0.2], atol=1e-9)
    assert tf["chi2"] < 1e-12 and tf["ndf"] == 20 and len(tf["eig"]) == 2
    assert np.allclose(tf["shape"], np.exp(0.4 * x - 0.2 * x ** 2))


def test_estimate_against_direct_sums():
    events = _events()
    cfg = MassfitConfig.from_dict({"range": [20, 250], "width": 10, "residual_order": 0, "smooth_templates": {"VV": 2.0}},
                                  default_blind_window=[50, 150])
    res = estimate(_accumulate(cfg, events).inputs(), cfg, window=(50, 150), log=lambda m: None)
    _, proba, mass, weight = events["s_QCD"]
    region = assign_regions(proba, BOXES, CLASSES)
    in_a = (mass > 50) & (mass < 150)
    for i in range(2):
        m = (region == i + 1) & in_a
        assert res["bundles"]["qcd_true"]["yield"][i] == pytest.approx(weight[m].sum(), rel=1e-12)
        assert res["bundles"]["qcd_true"]["n_entries"][i] == m.sum()
    F = res["qcd_true"][0][0]
    fit = (mass >= 20) & (mass <= 250)
    assert F.sum() == pytest.approx(weight[fit & (region == 0)].sum(), rel=1e-12)
    # n0 normalises the nominal model to the SR's QCD MC over the fit range
    for i in range(2):
        model = res["n0"][i] * F * res["tf"]["shape"]
        assert model.sum() == pytest.approx(res["qcd_true"][0][i + 1].sum(), rel=1e-12)
        assert res["bundles"]["qcd_predict"]["yield"][i] == pytest.approx(model[res["in_window"]].sum(), rel=1e-12)
    assert res["in_window"].sum() == 10  # bin centres 55 ... 145
    # sample / class bins clamped at 0, the QCD sum not
    assert all((v[0] >= 0).all() for v in res["samples"].values())
    # smoothed VV templates: the loosened box minus the other SR's box, a superset of the SR box
    sm, box = res["smooth"]["VV"][2], res["groups"]["VV"][2]
    assert (sm >= box[1:]).all() and sm.sum() > box[1:].sum()
    # medians of |w| over the window events at any score
    _, _, m_vv, w_vv = events["s_VV"]
    assert res["bundles"]["groups/vv"]["mean_event_weight"] == pytest.approx(
        np.median(np.abs(w_vv[(m_vv > 50) & (m_vv < 150)])))
    tfs = sideband_transfer_factors(res)
    assert len(tfs["model"]) == 2 and all(v > 0 for v in tfs["model"])
    json.dumps(summary(res), default=float)


def test_qcd_from_mc_and_data_blinding(tmp_path):
    events = _events(seed=5, n=5000)
    cfg = MassfitConfig.from_dict({"range": [20, 250], "width": 10, "qcd_from_mc": True},
                                  default_blind_window=[50, 150])
    acc = _accumulate(cfg, events)
    _, proba, mass, _ = events["s_QCD"]
    acc.fill_data(mass, proba=proba)
    res = estimate(acc.inputs(), cfg, window=(50, 150), log=lambda m: None)
    assert res["tf"] is None
    assert np.array_equal(res["bundles"]["qcd_predict"]["yield"], res["bundles"]["qcd_true"]["yield"])
    counts, blinded = res["data"]["counts"], res["data"]["blinded"]
    assert np.flatnonzero(blinded).tolist() == list(range(3, 13))  # 50-150 GeV
    assert (counts[1:, blinded] == 0).all() and counts[0, blinded].sum() > 0
    csv = tmp_path / "sr.csv"
    csv.write_text("bin_index,VVV_low,VVV_high\n1,0.5,1.0\n2,0.2,0.5\n")
    path = tmp_path / "out.root"
    write_root(path, res, sr_csv_hash(csv), log=lambda m: None)
    with uproot.open(path) as f:
        assert f.classname_of("metadata/signal_regions") == "TTree"  # combine.C cannot read an RNTuple
        assert f["metadata/signal_regions"]["bin_index"].array(library="np").tolist() == [1, 2]
        assert f["massfit/settings/qcd_from_mc"].values()[0] == 1.0 and "massfit/tfmc" not in f
        assert f["massfit/settings/has_data"].values()[0] == 1.0
        assert np.array_equal(f["massfit/pass/data/sr1/yield"].values(), counts[1])
        assert f["groups/qcd/sr2/yield"].values()[0] == pytest.approx(res["bundles"]["groups/qcd"]["yield"][1])
        assert f["qcd_predict/covariance_total"].values().shape == (2, 2)


def test_from_hist_matches_events():
    events = _events(seed=7, n=4000)
    # no event on a window edge or outside [20, 250): the histogram path is then exact for the sums
    events = {s: (c, p[(m >= 20) & (m < 250)], m[(m >= 20) & (m < 250)], w[(m >= 20) & (m < 250)])
              for s, (c, p, m, w) in events.items()}
    cfg = MassfitConfig.from_dict({"range": [20, 250], "width": 10, "residual_order": 0})
    ref = estimate(_accumulate(cfg, events).inputs(), cfg, log=lambda m: None)
    h = hist.Hist(hist.axis.StrCategory([], name="dataset", growth=True),
                  hist.axis.IntCategory([], name="region", growth=True),
                  hist.axis.StrCategory([], name="syst", growth=True),
                  hist.axis.Regular(23, 20.0, 250.0, name="mass"), storage=hist.storage.Weight())
    for s, (_, proba, mass, weight) in events.items():
        region = np.array([0, 1, 2])[assign_regions(proba, BOXES, CLASSES)]
        h.fill(dataset=s, region=region, syst="nominal", mass=mass, weight=weight)
    inputs = MassfitInputs.from_hist(h, {s: e[0] for s, e in events.items()}, [1, 2], CLASSES, ["QCD"], log=lambda m: None)
    res = estimate(inputs, cfg, log=lambda m: None)
    assert np.allclose(res["qcd_true"][0], ref["qcd_true"][0], rtol=1e-12)
    assert np.allclose(res["tf"]["coefficients"], ref["tf"]["coefficients"], rtol=1e-9)
    assert np.allclose(res["bundles"]["qcd_predict"]["yield"], ref["bundles"]["qcd_predict"]["yield"], rtol=1e-9)
    assert np.allclose(res["bundles"]["groups/vv"]["yield"], ref["bundles"]["groups/vv"]["yield"], rtol=1e-12)
    assert res["approximations"]


def test_legacy_loader(tmp_path):
    """A minimal legacy layout: selections/BDT marker files, a k-fold bdt_root with a score cache."""
    bdt = tmp_path / "selections" / "BDT"
    root = bdt / "study" / "cat"
    for f in range(2):
        (root / f"fold{f}").mkdir(parents=True)
        (root / f"fold{f}" / "cat_model.json").write_text(f"model{f}")
    (bdt / "kfold.py").write_text("")
    (bdt / "train.py").write_text("")
    groups = {c: [f"s_{c}"] for c in CLASSES}
    (bdt / "sample.json").write_text(json.dumps({"sample": [
        {"name": f"s_{c}", "is_signal": c in ("VVV", "VH"), "is_MC": True} for c in CLASSES]}))
    (root / "fold0" / "test_ranges.json").write_text(json.dumps({"tree_name": "cat", "kfold": 2}))
    (root / "fold0" / "config.json").write_text(json.dumps({"class_groups": groups, "sample_config": "sample.json"}))
    (root / "fold0" / "selection.json").write_text(json.dumps({"cat": {"thresholds": {MSD: [50, 150], "pt": [175, None]}}}))
    from vvv.background.legacy import model_fingerprint

    fp = model_fingerprint(root, "cat", {}, 2)
    (root / "kfold_scores").mkdir()
    events = _events(seed=11, n=3000)
    for s, (_, proba, mass, weight) in events.items():
        pt = np.where(np.arange(len(mass)) % 4 == 0, 100.0, 300.0)
        np.savez(root / "kfold_scores" / f"{s}.npz", proba=proba, weight=weight / 2.0, model_fingerprint=np.array(fp),
                 **{f"b_{MSD}": mass.astype(np.float32), "b_pt": pt.astype(np.float32)})
    (tmp_path / "sr.csv").write_text("bin_index,VVV_low,VVV_high,VV_low,VV_high\n1,0.5,1.0,0.0,1.0\n2,0.2,0.5,0.3,0.6\n")
    cfg = {"lumi": 2.0, "bdt_root": str(root), "signal_region_csv": "sr.csv", "abcd_branches": {"cat": [MSD]},
           "abcd_mode": "massfit", "exclude_samples": ["s_VH"],
           "massfit": {"cat": {"range": [20, 250], "width": 10, "residual_order": 0, "smooth_templates": {"VV": 2}}}}
    (tmp_path / "qcd_est.json").write_text(json.dumps(cfg))
    setup = LegacySetup.load(tmp_path / "qcd_est.json")
    assert setup.window == (50, 150) and setup.massfit.blind_window == (50.0, 150.0)
    acc = setup.accumulator(log=lambda m: None)
    setup.fill_mc(acc, log=lambda m: None)
    res = estimate(acc.inputs(), setup.massfit, window=setup.window, log=lambda m: None)
    _, proba, mass, weight = events["s_QCD"]
    keep = np.arange(len(mass)) % 4 != 0
    region = assign_regions(proba[keep], BOXES, CLASSES)
    m32 = mass[keep].astype(np.float32).astype(float)
    sel = (region == 1) & (m32 > 50) & (m32 < 150)
    assert res["bundles"]["qcd_true"]["yield"][0] == pytest.approx(weight[keep][sel].sum(), rel=1e-12)  # x lumi / 2
    assert res["bundles"]["groups/vh"]["yield"].sum() == 0.0 and res["bundles"]["samples/s_VH"]["mean_event_weight"] == 0
