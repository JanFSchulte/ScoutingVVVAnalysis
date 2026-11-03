import awkward as ak
import numpy as np
import pytest
from coffea.nanoevents.methods import candidate

from vvv.variables import (
    DEF,
    Kin,
    _event_shapes,
    _nth_max,
    _safe_div,
    best_wz_dijet,
    symmetric_eigenvalues3,
)

ak.behavior.update(candidate.behavior)


def jets(*events):
    """Jagged PtEtaPhiM records from lists of (pt, eta, phi, mass)."""
    fields = ("pt", "eta", "phi", "mass")
    return ak.zip(
        {f: ak.Array([[j[i] for j in ev] for ev in events]) for i, f in enumerate(fields)},
        with_name="PtEtaPhiMLorentzVector",
    )


def test_safe_div_fallback_on_zero_denominator():
    out = _safe_div(ak.Array([1.0, 2.0]), ak.Array([0.0, 4.0]), -99.0)
    assert out.tolist() == [-99.0, 0.5]


def test_nth_max_default_when_too_few():
    values = ak.Array([[0.1, 0.7, 0.3], [0.4], []])
    assert _nth_max(values, 2, DEF).tolist() == pytest.approx([0.3, DEF, DEF])


def test_wz_dijet_prefers_w_mass_and_small_dr():
    # Jets 0+1 form ~80 GeV at small dR; jet 2 is far away and pairs badly.
    j = jets(
        [(60.0, 0.0, 0.0, 5.0), (60.0, 0.0, 1.4, 5.0), (40.0, 2.0, -2.5, 5.0)],
        [(50.0, 0.0, 0.0, 5.0)],
    )
    found, i, k = best_wz_dijet(j, 4)
    assert found.tolist() == [True, False]
    assert (i.tolist()[0], k.tolist()[0]) == (0, 1)
    m = (Kin.of(j[:1, :1]) + Kin.of(j[:1, 1:2])).mass
    assert 70 < ak.flatten(m)[0] < 90


def test_wz_dijet_only_first_four_jets():
    good = (60.0, 0.0, 0.0, 5.0), (60.0, 0.0, 1.4, 5.0)
    filler = [(100.0, 2.0 * s, 3.0 * s, 1.0) for s in (-1, 1)]
    j = jets(filler + [(30.0, -1.0, 1.0, 1.0)] + list(good))
    found, i, k = best_wz_dijet(j, 4)
    assert found.tolist() == [True]
    assert max(i.tolist()[0], k.tolist()[0]) < 4


def test_eigenvalues_match_numpy():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(50, 3, 3))
    m = m + np.transpose(m, (0, 2, 1))
    l1, l2, l3 = symmetric_eigenvalues3(m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], m[:, 0, 1], m[:, 0, 2], m[:, 1, 2])
    ref = np.sort(np.linalg.eigvalsh(m), axis=1)[:, ::-1]
    assert np.allclose(np.stack([l1, l2, l3], axis=1), ref, atol=1e-9)


def test_sphericity_back_to_back_and_isotropic():
    # Two back-to-back objects: S = 0. Three orthogonal equal momenta: S = 1, A = 0.5.
    back = jets([(100.0, 0.0, 0.0, 0.0), (100.0, 0.0, np.pi, 0.0)])
    k = Kin.of(back)
    s, a, p = _event_shapes([k[:, 0], k[:, 1]])
    assert s[0] == pytest.approx(0.0, abs=1e-12)

    px = Kin(ak.Array([100.0]), ak.Array([0.0]), ak.Array([0.0]), ak.Array([100.0]))
    py = Kin(ak.Array([0.0]), ak.Array([100.0]), ak.Array([0.0]), ak.Array([100.0]))
    pz = Kin(ak.Array([0.0]), ak.Array([0.0]), ak.Array([100.0]), ak.Array([100.0]))
    s, a, p = _event_shapes([px, py, pz])
    assert s[0] == pytest.approx(1.0) and a[0] == pytest.approx(0.5) and p[0] == pytest.approx(0.0, abs=1e-12)


def test_event_shapes_need_two_objects():
    one = Kin(ak.Array([10.0]), ak.Array([0.0]), ak.Array([0.0]), ak.Array([10.0]))
    zero = Kin(ak.Array([0.0]), ak.Array([0.0]), ak.Array([0.0]), ak.Array([0.0]))
    s, a, p = _event_shapes([one, zero])
    assert (s[0], a[0], p[0]) == (-1.0, -1.0, -1.0)
