"""Lepton subtraction, hybrid JER, scouting jet ID and jet veto map (hand-built inputs)."""

import gzip
import json

import awkward as ak
import numpy as np
import pytest

from vvv.config import DEFAULT
from vvv.corrections.jets import JetCorrector, matched_gen_jets, subtract_leptons
from vvv.objects import scouting_jet_id
from vvv.selection import JetVetoMap


def _p4(pt, eta, phi, mass):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return px, py, pz, np.sqrt(px * px + py * py + pz * pz + mass * mass)


def _leptons(rows):
    return ak.Array([[{"pt": pt, "eta": eta, "phi": phi, "is_electron": ele} for pt, eta, phi, ele in event]
                     for event in rows])


def test_subtract_leptons():
    # Event 0: jets A (with a muon candidate) and B; event 1: jet C.
    pt = np.array([100.0, 80.0, 50.0])
    eta = np.array([0.0, 0.3, 1.0])
    phi = np.array([0.0, 0.0, 2.0])
    mass = np.array([10.0, 8.0, 5.0])
    counts = np.array([2, 1])
    has_muon = np.array([True, False, False])
    leptons = _leptons([
        [(40.0, 0.05, 0.0, True),    # electron: within 0.4 of A (0.05) and B (0.25): the closest, A
         (20.0, 0.30, 0.02, False)],  # muon: B is closer but has no muon candidate, so A (0.30)
        [(70.0, 1.0, 2.0, False)],    # muon near C, C has no muon candidate: not subtracted
    ])
    out = subtract_leptons(pt, eta, phi, mass, counts, has_muon, leptons, 0.4, 5.191)
    assert out.subtracted.tolist() == [True, False, False]
    jx, jy, jz, je = _p4(100.0, 0.0, 0.0, 10.0)
    e1 = _p4(40.0, 0.05, 0.0, 0.0)
    e2 = _p4(20.0, 0.30, 0.02, 0.0)
    rx, ry, rz, re = jx - e1[0] - e2[0], jy - e1[1] - e2[1], jz - e1[2] - e2[2], je - e1[3] - e2[3]
    assert out.pt[0] == pytest.approx(np.hypot(rx, ry))
    assert out.eta[0] == pytest.approx(np.arcsinh(rz / np.hypot(rx, ry)))
    assert out.phi[0] == pytest.approx(np.arctan2(ry, rx))
    assert out.mass[0] == pytest.approx(np.sqrt(max(re * re - rx * rx - ry * ry - rz * rz, 0.0)))  # spacelike -> 0
    assert out.muon_energy[0] == pytest.approx(e2[3])
    # Unchanged jets keep their stored values exactly.
    for got, stored in ((out.pt, pt), (out.eta, eta), (out.phi, phi), (out.mass, mass)):
        assert np.array_equal(got[1:], stored[1:])


def test_subtract_leptons_removes_a_lepton_only_jet():
    out = subtract_leptons(np.array([30.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1]),
                           np.array([False]), _leptons([[(31.0, 0.0, 0.0, True)]]), 0.4, 5.191)
    assert out.subtracted[0] and out.pt[0] == 0.0 and out.mass[0] == 0.0


def test_subtract_leptons_removes_a_remainder_beyond_the_jme_range():
    # Jet (pT 40, eta 2) and an electron (pT 39.5, eta 1.65, dR 0.35): the remainder has pT 0.5 GeV
    # and |eta| ~ 5.2, beyond the payload range.
    jet_pt, jet_eta = 40.0, 2.0
    e_pt, e_eta = 39.5, 1.65
    out = subtract_leptons(np.array([jet_pt]), np.array([jet_eta]), np.array([0.0]), np.array([30.0]), np.array([1]),
                           np.array([False]), _leptons([[(e_pt, e_eta, 0.0, True)]]), 0.4, 5.191)
    assert out.subtracted[0] and abs(out.eta[0]) > 5.191 and out.pt[0] == 0.0


def test_subtract_leptons_without_leptons():
    pt = np.array([100.0, 50.0])
    out = subtract_leptons(pt, np.zeros(2), np.zeros(2), np.ones(2), np.array([2]), np.ones(2, bool), _leptons([[]]), 0.8,
                           5.191)
    assert not out.subtracted.any() and np.array_equal(out.pt, pt)


def test_matched_gen_jets():
    gen = ak.Array([[{"pt": 50.0, "eta": 0.1, "phi": 0.2}, {"pt": 30.0, "eta": -1.0, "phi": 1.0}], []])
    idx = ak.Array([[1, -1, 5], [0]])
    pt, eta, phi = matched_gen_jets(idx, gen)
    assert pt.tolist() == [30.0, -1.0, -1.0, -1.0] and eta[0] == -1.0 and phi[0] == 1.0


@pytest.fixture(scope="module")
def corrector():
    return JetCorrector(DEFAULT.jets, DEFAULT.rho)


def test_hybrid_jer(corrector):
    resolution_c, sf_c, _ = corrector.jer_ak4
    pt, eta, rho = np.full(4, 100.0), np.full(4, 0.5), np.full(4, 20.0)
    sigma = resolution_c.evaluate(0.5, 100.0, 20.0)
    sf = sf_c.evaluate(0.5, 100.0)
    # Matched; no generator jet; pT outside 3 sigma; deltaR above R/2 = 0.2.
    gen_pt = np.array([99.0, -1.0, 100.0 * (1 - 4 * sigma), 99.0])
    gen_eta = np.array([0.5, 0.0, 0.5, 0.75])
    factors, matched = corrector._jer_factors(corrector.jer_ak4, 0.4, pt, eta, np.zeros(4), rho,
                                              np.full(4, 12345, dtype=np.int64), gen_pt, gen_eta, np.zeros(4))
    assert matched.tolist() == [True, False, False, False]
    assert factors[0, 0] == pytest.approx(1 + (sf - 1) * (100.0 - 99.0) / 100.0)
    # The unmatched jets share the stochastic draw of their (pT, eta, rho, event).
    assert factors[1, 0] == pytest.approx(factors[2, 0]) and factors[1, 0] == pytest.approx(factors[3, 0])
    assert factors[1, 0] != pytest.approx(factors[0, 0])


def _jets(**fields):
    base = {"pt": 50.0, "eta": 0.0, "phi": 0.0, "chHEF": 0.5, "neHEF": 0.2, "chEmEF": 0.0, "neEmEF": 0.2,
            "muEF": 0.0, "chHadMultiplicity": 5, "elMultiplicity": 0, "muMultiplicity": 0, "neHadMultiplicity": 2,
            "phMultiplicity": 3, "hfHadMultiplicity": 0, "hfEMMultiplicity": 0}
    n = len(next(iter(fields.values())))
    rows = [{k: (fields[k][i] if k in fields else v) for k, v in base.items()} for i in range(n)]
    return ak.Array([rows])


def test_scouting_jet_id():
    jets = _jets(neHEF=[0.2, 0.995, 0.2, 0.2, 0.2, 0.2], muEF=[0.0, 0.0, 0.85, 0.0, 0.0, 0.0],
                 chHadMultiplicity=[5, 5, 5, 0, 0, 1], muMultiplicity=[0, 0, 0, 0, 1, 0],
                 neHadMultiplicity=[2, 2, 2, 2, 2, 0], phMultiplicity=[3, 3, 3, 3, 3, 0])
    # Charged multiplicity counts muons (jet 4 passes with a muon and no charged hadron);
    # jet 5 has a single constituent.
    assert ak.to_list(scouting_jet_id(jets, DEFAULT.objects)) == [[True, False, False, False, True, False]]


def _bin_centres(vetoed, max_abs_eta=2.5):
    """(eta, phi) at the centre of the first bin with |eta| < max_abs_eta of the configured map
    that is (not) vetoed (|eta| < 2.6: every jet-ID requirement applies)."""
    data = json.load(gzip.open(DEFAULT.jet_veto_file))
    correction = next(c for c in data["corrections"] if c["name"] == DEFAULT.jet_veto_map)
    node = next(x for x in correction["data"]["content"] if x["key"] == DEFAULT.jet_veto_type)["value"]
    eta_edges, phi_edges = (np.asarray(e) for e in node["edges"])
    content = np.asarray(node["content"]).reshape(len(eta_edges) - 1, len(phi_edges) - 1)  # row-major, eta first
    eta_centres = 0.5 * (eta_edges[1:] + eta_edges[:-1])
    content[np.abs(eta_centres) >= max_abs_eta, :] = -1
    i, j = np.argwhere((content > 0) if vetoed else (content == 0))[0]
    return eta_centres[i], 0.5 * (phi_edges[j] + phi_edges[j + 1])


def test_jet_veto_map():
    eta, phi = _bin_centres(vetoed=True)
    eta0, phi0 = _bin_centres(vetoed=False)
    veto = JetVetoMap(DEFAULT)
    assert veto.map.evaluate(DEFAULT.jet_veto_type, eta, phi) > 0
    assert veto.map.evaluate(DEFAULT.jet_veto_type, eta0, phi0) == 0
    # One jet per event: in a vetoed bin and selected; below 15 GeV; failing the jet ID; in an
    # allowed bin. (The EM-fraction requirement is not tested separately: with chEmEF = 0, as in
    # every scouting jet, the jet ID's neEmEF < 0.9 for |eta| < 2.6 already implies it.)
    events = ak.concatenate([
        _jets(pt=[20.0], eta=[eta], phi=[phi]),
        _jets(pt=[14.0], eta=[eta], phi=[phi]),
        _jets(pt=[20.0], eta=[eta], phi=[phi], neHEF=[0.995]),
        _jets(pt=[20.0], eta=[eta0], phi=[phi0]),
    ])
    assert veto(events).tolist() == [True, False, False, False]


def test_prepare_end_to_end(corrector):
    """prepare() on hand-built jets: the production JEC of unchanged AK4 jets, the re-evaluated JEC
    of a subtracted one, the Type-1 muon bookkeeping, the AK8 muon-candidate rule, gen matching."""
    def rec(rows):
        return ak.Array([[dict(r) for r in event] for event in rows])

    base4 = {"mass": 5.0, "area": 0.5, "chEmEF": 0.0, "neEmEF": 0.2}
    ak4 = rec([[{**base4, "pt": 120.0, "eta": 0.2, "phi": 0.1, "rawFactor": 0.2, "muEF": 0.5, "muMultiplicity": 1, "genJetIdx": 0},
                {**base4, "pt": 60.0, "eta": -1.0, "phi": 2.0, "rawFactor": 0.1, "muEF": 0.0, "muMultiplicity": 0, "genJetIdx": 1}]])
    ak8 = rec([[{"pt": 400.0, "eta": 0.2, "phi": 0.1, "mass": 90.0, "msoftdrop": 80.0, "area": 2.0, "nMuons": 0},
                {"pt": 300.0, "eta": -1.0, "phi": 2.0, "mass": 85.0, "msoftdrop": 75.0, "area": 2.0, "nMuons": 0}]])
    met = ak.Array([{"pt": 50.0, "phi": 0.0}])
    event = ak.Array(np.array([123456789], dtype=np.uint64))
    # A muon inside AK4 jet 0 (which has a muon candidate) and inside AK8 jet 0 (which has none);
    # an electron inside AK8 jet 1 and AK4 jet 1.
    leptons = _leptons([[(30.0, 0.25, 0.1, False), (40.0, -1.05, 2.0, True)]])
    gen = rec([[{"pt": 70.0, "eta": 0.2, "phi": 0.1}, {"pt": 15.0, "eta": -1.0, "phi": 2.0}]])
    rho = ak.Array([20.0])
    for is_mc in (False, True):
        s = corrector.prepare(ak4, ak8, met, event, is_mc, leptons, rho=rho, gen_jets=gen if is_mc else None)
        raw0, raw1 = 120.0 * 0.8, 60.0 * 0.9
        # AK4 jet 0: muon subtracted, JEC re-evaluated on the remainder; jet 1: electron subtracted.
        assert s.ak4_raw_pt[0] < raw0 and s.ak4_raw_pt[1] < raw1
        for j in (0, 1):
            expected = corrector.jec_ak4.evaluate(0.5, s.ak4_eta[j], s.ak4_phi[j], s.ak4_raw_pt[j], 20.0)
            assert s.ak4_jec[j] == pytest.approx(expected)
        # Type-1: the muon energy left in jet 0 is muEF x E(raw) - E(subtracted muon).
        e_raw = np.sqrt((raw0 * np.cosh(0.2)) ** 2 + (5.0 * 0.8) ** 2)
        e_mu = 30.0 * np.cosh(0.25)
        e_rem = np.sqrt((s.ak4_raw_pt[0] * np.cosh(s.ak4_eta[0])) ** 2 + s.ak4_raw_mass[0] ** 2)
        assert s.ak4_raw_pt_nomu[0] == pytest.approx(s.ak4_raw_pt[0] * (1 - max(0.5 * e_raw - e_mu, 0.0) / e_rem))
        # AK8: no muon candidates, so only the electron is subtracted (from jet 1).
        assert s.ak8_raw_pt[0] == 400.0 and s.ak8_raw_pt[1] < 300.0
        assert s.ak8_raw_msd.tolist() == [80.0, 75.0]
        if is_mc:
            # AK4 jet 0 matches its generator jet only if within 3 sigma of the remainder's corrected pT.
            pt_jec = s.ak4_raw_pt * s.ak4_jec
            sigma = corrector.jer_ak4[0].evaluate(s.ak4_eta[0], pt_jec[0], 20.0)
            assert s.ak4_jer_matched[0] == (abs(pt_jec[0] - 70.0) < 3 * sigma * pt_jec[0])
        else:
            assert np.all(s.ak4_jer == 1.0) and not s.ak4_jer_matched.any()


def test_prepare_keeps_the_production_jec_of_unchanged_jets(corrector):
    ak4 = ak.Array([[{"pt": 80.0, "eta": 0.5, "phi": 1.0, "mass": 6.0, "rawFactor": 0.25, "area": 0.5, "muEF": 0.1,
                      "chEmEF": 0.0, "neEmEF": 0.3, "muMultiplicity": 1}]])
    ak8 = ak.Array([[{"pt": 300.0, "eta": -1.0, "phi": -2.0, "mass": 85.0, "msoftdrop": 75.0, "area": 2.0, "nMuons": 0}]])
    s = corrector.prepare(ak4, ak8, ak.Array([{"pt": 30.0, "phi": 0.0}]), ak.Array(np.array([7], dtype=np.uint64)),
                          False, _leptons([[]]), rho=ak.Array([25.0]))
    assert s.ak4_jec[0] == 1.0 / 0.75 and s.ak4_raw_pt[0] == 80.0 * 0.75
    assert s.ak4_raw_pt_nomu[0] == pytest.approx(80.0 * 0.75 * 0.9)
    assert s.ak8_jec[0] == pytest.approx(corrector.jec_ak8.evaluate(2.0, -1.0, -2.0, 300.0, 25.0))
