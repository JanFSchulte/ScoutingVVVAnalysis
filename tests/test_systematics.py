import json
import pickle

import hist
import numpy as np
import pytest

from vvv.processors.analysis import mass_hist, member_hist
from vvv.samples import Sample
from vvv.stats.systematics import (SCALE_VALID, CellHists, HistOutputs, SystConfig, build, category_entries,
                                   mass_fit_bins, theory_ratios, window_bins, write)

LUMI = 100.0
CAT = "boosted3"
JET = [f"{s}_{d}" for s in ("jes", "jer", "jms", "jmr") for d in ("up", "down")]
MASS_AXIS = hist.axis.Regular(23, 20.0, 250.0, name="mass")


def _events(seed, n):
    rng = np.random.default_rng(seed)
    region = rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])
    mass = rng.uniform(20.0, 250.0, n)
    w = rng.exponential(1.0, n) * np.where(rng.random(n) < 0.1, -1.0, 1.0)
    return region, mass, w


def _outputs(n=4000, n_sr2=None, seed=1, theory=True, alphas=True):
    """Hand-built vvv run histograms of one MC sample 'sig': nominal, jet variations (own events),
    pileup and the theory member sets. Returns (HistOutputs, the event arrays)."""
    rng = np.random.default_rng(seed)
    region, mass, w = _events(seed, n)
    if n_sr2 is not None:  # thin out SR 2 to n_sr2 events
        keep = (region != 2) | (np.cumsum(region == 2) <= n_sr2)
        region, mass, w = region[keep], mass[keep], w[keep]
    ev = {"region": region, "mass": mass, "w": w}
    h = mass_hist(MASS_AXIS)
    h.fill(dataset="sig", category=CAT, region=region, syst="nominal", mass=mass, weight=w)
    for k, name in enumerate(JET):  # each variation: its own selection (drop, migrate) and mass
        sel = rng.random(len(w)) < 0.95 + 0.01 * k
        r2 = np.where(rng.random(len(w)) < 0.05, rng.choice([0, 1, 2], len(w)), region)
        m2 = mass * (1.0 + 0.02 * (k - 4))
        ev[name] = (r2[sel], m2[sel], w[sel])
        h.fill(dataset="sig", category=CAT, region=r2[sel], syst=name, mass=m2[sel], weight=w[sel])
    for d in ("up", "down"):
        ev[f"pileup_{d}"] = rng.uniform(0.8, 1.2, len(w))
        h.fill(dataset="sig", category=CAT, region=region, syst=f"pileup_{d}", mass=mass, weight=w * ev[f"pileup_{d}"])
    out = {"mass": h, "lumi_fb": LUMI}
    if theory:
        sets = {"LHEPdfWeight": 101, "LHEScaleWeight": 9, "PSWeight": 4}
        if alphas:
            sets["LHEPdfWeightAlphaS"] = 2
        for name, k in sets.items():
            ratio = rng.normal(1.0, 0.1, (len(w), k))
            ev[name] = ratio
            hm = member_hist(MASS_AXIS, k)
            hm.fill(dataset="sig", category=CAT, region=np.repeat(region, k), member=np.tile(np.arange(k), len(w)),
                    mass=np.repeat(mass, k), weight=(w[:, None] * ratio).ravel())
            out[name] = hm
    return out, ev


def _load(tmp_path, out):
    path = tmp_path / "h.pkl"
    path.write_bytes(pickle.dumps(out))
    return HistOutputs.load([tmp_path], log=lambda m: None)


def _sums(mask, w):
    return float(w[mask].sum()), float((w[mask] ** 2).sum())


def _window(mass):
    return (mass > 50.0) & (mass < 150.0)


def test_window_and_fit_bins():
    edges = MASS_AXIS.edges
    assert window_bins(edges, (50, 150)).tolist() == list(range(3, 13))
    fit, groups = mass_fit_bins(edges, (20, 250), 10)
    assert len(groups) == 23 and fit[0] == 20 and fit[-1] == 250
    fit, groups = mass_fit_bins(edges, (40, 160), 20)
    assert [g.tolist() for g in groups][:2] == [[2, 3], [4, 5]]
    with pytest.raises(ValueError):
        window_bins(edges, (55, 150))
    with pytest.raises(ValueError):
        mass_fit_bins(edges, (20, 250), 15)


def test_jet_and_pileup_ratios(tmp_path):
    out, ev = _outputs()
    outputs = _load(tmp_path, out)
    cell = CellHists.from_outputs(outputs, "sig", CAT, [1, 2])
    e = category_entries(cell, [1, 2], "abc", SystConfig(LUMI), systs=("jes", "pileup"))
    reg, m, w = ev["region"], ev["mass"], ev["w"]
    win = _window(m)
    nom = _sums(win, w)
    r2, m2, w2 = ev["jes_up"]
    up = _sums(_window(m2), w2)
    assert e["jes"]["jes_up"] == pytest.approx(up[0] / nom[0], rel=1e-12)
    assert e["jes"]["yield"] == pytest.approx(nom[0] / LUMI, rel=1e-12)
    assert e["jes"]["sumw2_jes_up"] == pytest.approx(up[1] / LUMI ** 2, rel=1e-12)
    assert e["jes"]["n_eff"] == pytest.approx(nom[0] ** 2 / nom[1], rel=1e-12)
    # SR 1 window: each configuration's own region assignment and mass
    sr1 = _sums(win & (reg == 1), w)
    rd, md, wd = ev["jes_down"]
    assert e["jes"]["regions"]["1"]["jes_down"] == pytest.approx(_sums(_window(md) & (rd == 1), wd)[0] / sr1[0], rel=1e-12)
    pu = _sums(win & (reg == 2), w * ev["pileup_up"])
    assert e["pileup"]["regions"]["2"]["pu_up"] == pytest.approx(pu[0] / _sums(win & (reg == 2), w)[0], rel=1e-12)
    assert e["pileup"]["regions"]["2"]["yield_pu_up"] == pytest.approx(pu[0] / LUMI, rel=1e-12)
    assert e["pileup"]["sr_csv_hash"] == "abc" and e["pileup"]["min_region_events"] == 10
    assert "mass_fit" not in e["jes"] and "mass_range" not in e["jes"]["regions"]["1"]


def test_fallback_to_inclusive(tmp_path):
    out, ev = _outputs(n_sr2=12)  # ~4 events of SR 2 in the window: n_eff < 10
    cell = CellHists.from_outputs(_load(tmp_path, out), "sig", CAT, [1, 2])
    e = category_entries(cell, [1, 2], "abc", SystConfig(LUMI), systs=("jms", "theory"), has_theory=True)
    sr2 = e["jms"]["regions"]["2"]
    assert sr2["fallback_to_inclusive"] and sr2["jms_up"] == e["jms"]["jms_up"]
    assert sr2["n_eff"] < 10 and sr2["yield"] != e["jms"]["yield"]  # own sums, inclusive ratios
    assert "fallback_to_inclusive" not in e["jms"]["regions"]["1"]
    assert e["theory"]["regions"]["2"]["fallback_to_inclusive"]
    assert e["theory"]["regions"]["2"]["scale_up"] == e["theory"]["scale_up"]
    # a lower threshold keeps the region's own ratios
    e0 = category_entries(cell, [1, 2], "abc", SystConfig(LUMI, min_region_events=0), systs=("jms",))
    assert "fallback_to_inclusive" not in e0["jms"]["regions"]["2"]


def _legacy_theory(nom, pdf, scale, ps, alphas):
    """theory_syst._ratios_from_sums (hessian_symmetric) with gen = 1, written out independently."""
    central = pdf[0] / nom
    members = pdf[1:101] / nom
    sigma_pdf = np.sqrt(np.sum((members - central) ** 2))
    sigma_as = 0.5 * abs(alphas[1] / nom - alphas[0] / nom) if alphas is not None else 0.0
    tot = np.sqrt(sigma_pdf ** 2 + sigma_as ** 2)
    s = scale[[0, 1, 3, 4, 5, 7, 8]] / nom
    return {"pdf_up": central + tot, "pdf_down": central - tot, "scale_up": s.max(), "scale_down": s.min(),
            "ps_isr_up": ps[2] / nom, "ps_isr_down": ps[0] / nom, "ps_fsr_up": ps[3] / nom, "ps_fsr_down": ps[1] / nom}


@pytest.mark.parametrize("alphas", [True, False])
def test_theory(tmp_path, alphas):
    out, ev = _outputs(alphas=alphas)
    cell = CellHists.from_outputs(_load(tmp_path, out), "sig", CAT, [1, 2])
    e = category_entries(cell, [1, 2], "abc", SystConfig(LUMI), systs=("theory",), has_theory=True)["theory"]
    reg, m, w = ev["region"], ev["mass"], ev["w"]
    sel = _window(m) & (reg == 1)
    sw = lambda name: (w[sel][:, None] * ev[name][sel]).sum(axis=0)  # noqa: E731
    ref = _legacy_theory(w[sel].sum(), sw("LHEPdfWeight"), sw("LHEScaleWeight"), sw("PSWeight"),
                         sw("LHEPdfWeightAlphaS") if alphas else None)
    r1 = e["regions"]["1"]
    for k, v in ref.items():
        assert r1[k] == pytest.approx(v, rel=1e-10), k
    assert r1["alphas_available"] is alphas and (("yield_alphas" in r1) is alphas)
    assert len(r1["yield_pdf"]) == 101 and len(r1["sumw2_scale"]) == 9 and len(r1["yield_ps"]) == 4
    assert r1["yield_scale"][4] == pytest.approx(sw("LHEScaleWeight")[4] / LUMI, rel=1e-12)
    assert r1["pdf_method"] == "hessian_symmetric" and r1["generator_normalised"]


def test_theory_replica_and_envelope():
    nom = 10.0
    pdf = np.r_[10.0, np.linspace(9.0, 11.0, 100)]
    scale = np.array([9.0, 9.5, 5.0, 10.2, 10.0, 10.1, 20.0, 10.5, 9.8])  # corners 2 and 6 dropped
    members = {"pdf": (pdf, None), "scale": (scale, None), "ps": (np.array([9.0, 9.5, 11.0, 10.5]), None)}
    r = theory_ratios(nom, members, "mc_replica")
    srt = np.sort(pdf[1:] / nom)
    assert r["pdf_up"] == pytest.approx(srt[83]) and r["pdf_down"] == pytest.approx(srt[15])
    assert r["scale_up"] == pytest.approx(1.05) and r["scale_down"] == pytest.approx(0.9)
    assert (r["ps_isr_up"], r["ps_isr_down"], r["ps_fsr_up"], r["ps_fsr_down"]) == (1.1, 0.9, 1.05, 0.95)
    assert not r["alphas_available"] and SCALE_VALID == [0, 1, 3, 4, 5, 7, 8]


def test_mass_bins(tmp_path):
    out, ev = _outputs()
    cell = CellHists.from_outputs(_load(tmp_path, out), "sig", CAT, [1, 2])
    e = category_entries(cell, [1, 2], "abc", SystConfig(LUMI, mass_fit=((20, 250), 10)), systs=("jmr",))["jmr"]
    assert e["mass_fit"]["edges"] == [float(v) for v in range(20, 251, 10)]
    r1 = e["regions"]["1"]
    assert len(r1["mass_bins"]) == 23
    reg, m, w = ev["region"], ev["mass"], ev["w"]
    assert r1["mass_range"]["yield"] == pytest.approx(w[reg == 1].sum() / LUMI, rel=1e-12)
    b = r1["mass_bins"][5]  # 70-80 GeV
    sel = (reg == 1) & (m >= 70) & (m < 80)
    assert b["yield"] == pytest.approx(w[sel].sum() / LUMI, rel=1e-12)
    rv, mv, wv = ev["jmr_up"]
    assert b["yield_jmr_up"] == pytest.approx(wv[(rv == 1) & (mv >= 70) & (mv < 80)].sum() / LUMI, rel=1e-12)
    assert all(bb["jmr_up"] == r1["mass_range"]["jmr_up"] for bb in r1["mass_bins"] if bb.get("fallback_to_mass_range"))


def test_build_and_write(tmp_path):
    out, _ = _outputs()
    (tmp_path / "in").mkdir()
    csv_path = tmp_path / "sr.csv"
    csv_path.write_text("bin_index,VVV_low,VVV_high\n2,0.5,1.0\n1,0.2,0.5\n")
    import hashlib

    out["signal_regions"] = {CAT: {"csv": str(csv_path), "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}}
    outputs = _load(tmp_path / "in", out)
    catalogue = {"sig": Sample("sig", (), "signal", 1, has_theory_weights=True),
                 "bkg": Sample("bkg", (), "background", 2)}
    res = build(outputs, {CAT: (csv_path, (50, 150))}, SystConfig(outputs.lumi), catalogue, log=lambda m: None)
    assert set(res) == {"theory", "pileup", "jes", "jer", "jms", "jmr"}
    assert list(res["jes"]["sig"][CAT]["regions"]) == ["2", "1"]  # CSV order, as legacy
    paths = write(res, tmp_path / "out", log=lambda m: None)
    d = json.loads(paths["pileup"].read_text())
    assert list(d) == ["sig"] and "pu_up" in d["sig"][CAT]
    assert (tmp_path / "out" / "theory_syst_yields.csv").read_text().startswith("sample,tree,region,systematic")
    other = tmp_path / "sr_other.csv"
    other.write_text("bin_index,VVV_low,VVV_high\n1,0.3,1.0\n2,0.2,0.3\n")
    with pytest.raises(ValueError, match="filled with"):
        build(outputs, {CAT: (other, (50, 150))}, SystConfig(outputs.lumi), catalogue, log=lambda m: None)
