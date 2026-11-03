import awkward as ak
import numpy as np
import pytest
import vector

from vvv.config import DEFAULT
from vvv.corrections.jets import jet_gaussian
from vvv.ml.inference import region_index, threshold_mask
from vvv.objects import _opposite_sign, best_sfos_pair, p4_pt_eta
from vvv.weights import _lookup, _standard_scale

vector.register_awkward()


def test_jet_gaussian_matches_legacy_cpp():
    # Reference values from convert_branch.C jetGaussian compiled with c++ -O2.
    got = jet_gaussian(
        np.array([2125471, 1, 2**64 - 1], dtype=np.uint64),
        np.array([241.70323, 175.5, 1000.25], dtype=np.float32),
        np.array([-2.5, 0.1, 2.4], dtype=np.float32),
        np.array([1.234, -3.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(got, [-0.50666441796385153, 0.28501213930380942, -0.5931579303471729], rtol=1e-14)


def test_p4_eta_round_trip_moves_values_on_the_cut():
    jets = ak.zip({"pt": [[241.70323]], "eta": [[-2.5]], "phi": [[0.3]]})
    _, eta = p4_pt_eta(jets)
    assert abs(eta[0, 0]) == pytest.approx(2.5, abs=1e-12)


def test_legacy_electron_charge_pairing():
    cfg = DEFAULT.objects
    int_min = np.iinfo(np.int32).min
    q1 = ak.Array([int_min, int_min, int_min, 1])
    q2 = ak.Array([1, -1, int_min, -1])
    # A track-less electron (INT_MIN) is "opposite sign" to any charged one, not to another.
    assert ak.to_list(_opposite_sign(q1, q2, cfg)) == [True, True, False, True]


def test_sfos_pair_closest_to_z():
    cfg = DEFAULT.objects

    def lep(pt, eta, phi, q, ele):
        return {"pt": pt, "eta": eta, "phi": phi, "mass": 0.0, "charge": q, "is_electron": ele}

    leps = ak.Array(
        [[lep(60, 0.0, 0.0, 1, False), lep(40, 0.5, 2.8, -1, False), lep(20, -1.0, 1.0, -1, True)]],
        with_name="Momentum4D",
    )
    found, p4 = best_sfos_pair(leps, cfg)
    assert ak.to_list(found) == [True]
    expected = (leps[0, 0] + leps[0, 1]).mass
    assert p4.mass[0] == pytest.approx(expected)


def test_pileup_lookup_edges():
    edges = np.array([0.0, 1.0, 2.0])
    values = np.array([0.5, 2.0])
    got = _lookup(edges, values, np.array([-0.1, 0.0, 0.99, 1.0, 2.0, 2.01]))
    np.testing.assert_array_equal(got, [0.0, 0.5, 0.5, 2.0, 2.0, 0.0])


def test_eight_member_scale_set_gets_nominal_at_index_4():
    values = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 1.0]])
    got = _standard_scale(values, np.array([8]))
    np.testing.assert_array_equal(got, [[1, 2, 3, 4, 1, 5, 6, 7, 8]])


def test_threshold_sentinels_fail():
    cols = {"a": np.array([-999.0, 60.0, 160.0, 100.0])}
    np.testing.assert_array_equal(threshold_mask(cols, {"a": [50, 150]}), [False, True, False, True])


def test_region_upper_edge_one_is_inclusive():
    proba = np.array([[1.0, 0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0.8]])
    regions = [(1, {"VVV": (0.6, 1.0)}), (2, {"VVV": (0.3, 0.6)})]
    np.testing.assert_array_equal(region_index(proba, regions), [1, 2, 0])
