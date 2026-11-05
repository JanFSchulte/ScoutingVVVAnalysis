"""Signal-region optimiser (vvv.sr.optimise) on small synthetic inputs."""

import csv

import numpy as np
import pytest

from vvv.ml.inference import load_signal_regions, region_index
from vvv.sr.optimise import (
    ScanInputs,
    SRConfig,
    abcd_qcd_variance,
    asimov_z,
    asimov_z_with_error,
    box_mask,
    optimise,
    overlap,
    write_csv,
)

CLASSES = ["Sig", "Bkg", "QCD"]


def test_asimov_z():
    s, b = 5.0, 20.0
    exact = np.sqrt(2 * ((s + b) * np.log1p(s / b) - s))
    assert asimov_z(s, b, 0.0) == pytest.approx(exact)
    assert asimov_z(s, b, 1e-14 * b * b) == pytest.approx(exact)  # v <= 1e-12 B^2: the v -> 0 form
    v = 4.0
    cowan = np.sqrt(2 * ((s + b) * np.log((s + b) * (b + v) / (b * b + (s + b) * v))
                         - b * b / v * np.log1p(v * s / (b * (b + v)))))
    assert asimov_z(s, b, v) == pytest.approx(cowan)
    assert asimov_z(s, b, v) < exact
    assert asimov_z(s, b, -1.0) == pytest.approx(exact)  # negative variance clipped to 0
    assert asimov_z(0.0, b, v) == 0.0 and asimov_z(s, 0.0, v) == 0.0 and asimov_z(-1.0, b, v) == 0.0
    z = asimov_z(np.array([s, 0.0, s]), np.array([b, b, -1.0]), np.array([v, v, v]))
    assert z.shape == (3,) and z[0] == pytest.approx(cowan) and z[1] == z[2] == 0.0
    Z, sZ = asimov_z_with_error(s, b, 0.5, 2.0)
    assert Z == pytest.approx(cowan) and sZ > 0.0
    assert asimov_z_with_error(0.0, b, 0.5, 2.0) == (0.0, 0.0)


def test_box_mask_upper_edge():
    axes = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 0.5], [0.4999, 0.5]])
    assert box_mask(axes, [0.5, 0.0], [1.0, 1.0]).tolist() == [False, True, True, False]  # hi = 1 includes 1
    assert box_mask(axes, [0.0, 0.0], [0.5, 1.0]).tolist() == [True, False, False, True]  # [lo, hi)
    assert box_mask(axes, [0.0, 0.0], [1.0, 0.5]).tolist() == [False] * 4
    assert not overlap([0.0, 0.0], [0.5, 1.0], [0.5, 0.0], [1.0, 1.0])  # abutting edges
    assert overlap([0.0, 0.0], [0.6, 1.0], [0.5, 0.0], [1.0, 1.0])


def test_abcd_variance():
    tf = [1.5, 2.0]
    # One window: max(TF^2 c_var, TF QCD).
    assert abcd_qcd_variance(tf[:1], [10.0], 1.0) == pytest.approx(22.5)
    assert abcd_qcd_variance(tf[:1], [0.0], 3.0) == pytest.approx(4.5)  # floor TF x QCD
    # Two windows: inverse-variance combination.
    v1, v2 = 2.25 * 10.0, 4.0 * 20.0
    assert abcd_qcd_variance(tf, [10.0, 20.0], 1.0) == pytest.approx(1.0 / (1 / v1 + 1 / v2))
    assert abcd_qcd_variance(tf, [0.0, 20.0], 0.0) == 0.0  # any V_k = 0 -> 0
    arr = abcd_qcd_variance(tf, [np.array([10.0, 0.0]), np.array([20.0, 20.0])], np.array([1.0, 0.0]))
    assert arr == pytest.approx([1.0 / (1 / v1 + 1 / v2), 0.0])


def _toy(n=20000, seed=3):
    """Signal at high p(Sig), background spread out, QCD near p(QCD) = 1."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.2, 0.5, 0.3])
    alpha = np.where(y[:, None] == np.arange(3)[None, :], 8.0, 1.0)
    proba = np.array([rng.dirichlet(a) for a in alpha])
    w = np.where(y == 0, 0.01, np.where(y == 1, 0.2, 0.5))
    sample = np.array(["sig", "bkg", "qcd"])[y]
    msd = rng.uniform(20.0, 250.0, size=n)
    return proba, w, y, sample, msd


def _cfg(**kw):
    d = {"lumi": 1.0, "n_signal_regions": 2, "score_axes": "all", "min_bkg_weight": 5.0,
         "coarse_bin_width": 0.05, "fine_bin_width": 0.01, "fine_refine_window": 0.05,
         "beam_width": 16, "top_intervals_per_axis": 4, "coordinate_rounds": 4, "fine_refine_passes": 2,
         "max_selection_candidates": 100, "selection_reps_per_axis": 10, "branch_bound_max_nodes": 100000,
         "branch_bound_time_limit_seconds": 30, "max_threads": 2, "progress_every_seconds": 0}
    d.update(kw)
    return SRConfig.from_dict(d)


def test_optimise_finds_signal_corner(tmp_path):
    proba, w, y, sample, _ = _toy()
    inputs = ScanInputs(proba, w, y, sample, CLASSES, [True, False, False])
    bins, report = optimise(_cfg(), inputs, log=lambda m: None)
    assert len(bins) == 2 and report["branch_bound"]["completed"]
    best = bins[0]
    assert best["lo"][0] >= 0.3  # a high p(Sig) cut
    assert best["B"] >= 5.0 and best["S"] / best["B"] > inputs.weight[y == 0].sum() / inputs.weight[y > 0].sum()
    assert not overlap(bins[0]["lo"], bins[0]["hi"], bins[1]["lo"], bins[1]["hi"])
    assert report["Z_comb"] >= report["Z_comb_before_refinement"] - 1e-9
    # Same result on coarse cells.
    bins_c, _ = optimise(_cfg(), inputs, cells=True, log=lambda m: None)
    assert [b["S_entries"] for b in bins_c] == [b["S_entries"] for b in bins]
    assert [b["significance"] for b in bins_c] == pytest.approx([b["significance"] for b in bins], rel=1e-9)
    # CSV readable by vvv.ml.inference, the same events in each box.
    path = tmp_path / "signal_region.csv"
    write_csv(path, bins, report["axes"])
    header = next(csv.reader(open(path)))
    assert header[:9] == ["bin_index", "significance", "significance_error", "S", "S_err", "S_entries",
                          "B", "B_err", "B_entries"] and header[9:11] == ["Sig_low", "Sig_high"]
    import vvv.ml.inference as inf

    regions = load_signal_regions(path)
    assert [r[0] for r in regions] == [1, 2]
    old = inf.CLASSES
    try:
        inf.CLASSES = tuple(CLASSES)
        regions = load_signal_regions(path)
        idx = region_index(proba, regions)
    finally:
        inf.CLASSES = old
    for b in bins:
        assert int(np.count_nonzero((idx == b["bin_index"]) & (y == 0))) == b["S_entries"]


def test_optimise_abcd():
    proba, w, y, sample, msd = _toy()
    inputs = ScanInputs.from_events(proba, w, y, sample, CLASSES, [True, False, False],
                                    abcd_values=msd, abcd_cond=[50, 150], control_windows=[(20, 50), (150, 250)])
    assert len(inputs.weight) == np.count_nonzero((msd > 50) & (msd < 150))
    assert len(inputs.control["weight"]) == np.count_nonzero((msd < 50) | (msd > 150))
    cfg = _cfg(abcd_uncertainty={"branch": "msd", "control_window": [[20, 50], [150, 250]], "nonclosure": 0.2,
                                 "transfer_factors": [0.5, 0.5], "min_control_weight": 5.0})
    bins, report = optimise(cfg, inputs, log=lambda m: None)
    assert report["control_events_per_window"] == [int(np.count_nonzero(msd < 50)), int(np.count_nonzero(msd > 150))]
    for b in bins:
        assert b["control_count"] >= 5.0
        assert b["sigma_B"] ** 2 == pytest.approx(b["sigma_mc_nonqcd"] ** 2 + b["sigma_abcd"] ** 2)
        assert b["sigma_abcd"] ** 2 >= (0.2 * b["QCD"]) ** 2
    bins_c, _ = optimise(cfg, inputs, cells=True, log=lambda m: None)
    assert [b["significance"] for b in bins_c] == pytest.approx([b["significance"] for b in bins], rel=1e-9)


def test_exclude_samples():
    proba, w, y, sample, _ = _toy(n=2000)
    inputs = ScanInputs(proba, w, y, sample, CLASSES, [True, False, False]).exclude(["qcd"], log=lambda m: None)
    assert "qcd" not in set(inputs.sample) and len(inputs.weight) == np.count_nonzero(y != 2)
