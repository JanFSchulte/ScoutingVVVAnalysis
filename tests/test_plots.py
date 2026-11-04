"""Training diagnostic plots on a tiny synthetic fold."""

import json
from pathlib import Path

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")
pytest.importorskip("matplotlib")

from vvv.ml.plots import plot_fold, weighted_roc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LEGACY_FILES = (
    [f"{stem}{s}.pdf" for s in ("_cls", "_decorr") for stem in (
        "decor_branch_shapes_by_signal_score", "decor_corr_test", "decor_corr_train", "decor_score_vs_branch",
        "importance", "roc_vh", "roc_vvv")]
    + [f"score_{sig}_vs_{bkg}{s}.pdf" for s in ("_cls", "_decorr") for sig in ("vh", "vvv")
       for bkg in ("qcd", "top", "vjets", "vv")]
    + ["feature_corr.pdf", "loss_classification.pdf", "loss_decorrelation.pdf", "loss_mlogloss.pdf", "loss_total.pdf"]
)


def test_weighted_roc():
    y = np.array([1, 1, 0, 0], dtype=float)
    assert weighted_roc(y, np.array([0.9, 0.8, 0.2, 0.1]), np.ones(4))[2] == pytest.approx(1.0)
    assert weighted_roc(y, np.full(4, 0.5), np.ones(4))[2] == pytest.approx(0.5)


def test_plot_fold_writes_the_legacy_files(tmp_path):
    cfg = json.loads((REPO / "configs/training/production.json").read_text())
    category = "boosted3"
    cc = cfg["categories"][category]
    features, decor = cc["features"], cc["decorrelate"]
    n_classes = len(cfg["class_groups"])
    rng = np.random.default_rng(1)
    samples = [members[0] for members in cfg["class_groups"].values()]

    def split(n):
        y = rng.integers(0, n_classes, n).astype(np.int32)
        X = {c: rng.normal(1.0, 0.3, n).astype(np.float32) + 0.2 * y for c in features + decor}
        X["ScoutingFatPFJetRecluster_msoftdrop_1"] = rng.uniform(0, 250, n).astype(np.float32)
        w = rng.uniform(0.5, 1.5, n)
        return {"X": X, "w": w, "w_diag": w, "y": y, "event": np.arange(n), "sample": np.array(samples, dtype=object)[y]}

    data = {"train": split(3000), "monitor": split(1000), "test": split(2000)}
    Xtr = np.column_stack([data["train"]["X"][f] for f in features])
    dtrain = xgb.DMatrix(Xtr, label=data["train"]["y"], weight=data["train"]["w"], feature_names=features)
    params = {"objective": "multi:softprob", "num_class": n_classes, "max_depth": 3}
    stage1 = xgb.train(params, dtrain, num_boost_round=5)
    final = xgb.train(params, dtrain, num_boost_round=3, xgb_model=stage1.copy())
    h = {"round": [5, 8], "classification": [2.0, 1.5], "mlogloss": [1.0, 0.9], "decorrelation": [0.0, 0.3], "total": [2.0, 1.8]}
    log = {"loss_components": {"train": h, "monitor": h}}

    written, summary = plot_fold(tmp_path, category, cfg, stage1, final, data, log)
    names = {p.name for p in written if p.parent == tmp_path}
    assert set(LEGACY_FILES) <= names
    assert (tmp_path / "branches" / f"{features[0]}_by_sample.pdf").exists()
    assert set(summary) == {"stage1", "stage2"} and "decor_corr_test" in summary["stage2"]
