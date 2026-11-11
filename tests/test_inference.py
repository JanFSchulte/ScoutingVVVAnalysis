"""K-fold model directories and the event -> fold dispatch (vvv.ml.inference)."""

import json

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from vvv.ml.inference import CLASSES, CategoryModel, fold_directories, fold_of  # noqa: E402

TREE = "boosted3"
FEATURES = ["a", "b"]


def _write_fold(base, r, k, offset, legacy=False):
    """One rotation whose booster predicts class (offset % 6) for every event."""
    d = base / TREE / f"fold{r}"
    d.mkdir(parents=True)
    rng = np.random.default_rng(r)
    x = rng.normal(size=(60, 2))
    y = np.full(60, offset % len(CLASSES))
    y[: len(CLASSES)] = np.arange(len(CLASSES))  # every class present
    booster = xgb.train({"objective": "multi:softprob", "num_class": len(CLASSES), "max_depth": 1},
                        xgb.DMatrix(x, label=y, feature_names=FEATURES), num_boost_round=5)
    booster.save_model(str(d / f"{TREE}_model.json"))
    (d / "branch.json").write_text(json.dumps({TREE: [{"name": f} for f in FEATURES]}))
    (d / "selection.json").write_text(json.dumps({TREE: {"clip_ranges": {}, "log_transform": [], "thresholds": {}}}))
    if legacy:
        (d / "test_ranges.json").write_text(json.dumps({"kfold": k, "fold": r, "tree_name": TREE}))
    else:
        (d / "config.json").write_text(json.dumps({"kfold": k}))
        (d / "training_log.json").write_text(json.dumps({"fold": r}))


@pytest.mark.parametrize("legacy", [False, True])
def test_load_and_dispatch(tmp_path, legacy):
    for r in range(3):
        _write_fold(tmp_path, r, 3, offset=r, legacy=legacy)
    model = CategoryModel.load(tmp_path, TREE)
    assert model.k == 3
    events = np.arange(12, dtype=np.uint64)
    proba = model.predict({"a": np.zeros(12), "b": np.zeros(12)}, events)
    # The booster of rotation r favours class r: each event is scored by the model of event % 3.
    np.testing.assert_array_equal(np.argmax(proba, axis=1), fold_of(events, 3))


def test_missing_rotation_is_refused(tmp_path):
    for r in (0, 1, 3):
        _write_fold(tmp_path, r, 4, offset=r)
    with pytest.raises(ValueError, match="exactly"):
        fold_directories(tmp_path / TREE)


def test_extra_rotation_is_refused(tmp_path):
    for r in range(3):
        _write_fold(tmp_path, r, 2, offset=r)
    with pytest.raises(ValueError, match="exactly"):
        fold_directories(tmp_path / TREE)


def test_recorded_fold_must_match_its_directory(tmp_path):
    for r in range(2):
        _write_fold(tmp_path, r, 2, offset=r)
    (tmp_path / TREE / "fold1" / "training_log.json").write_text(json.dumps({"fold": 0}))
    with pytest.raises(ValueError, match="expected 1"):
        fold_directories(tmp_path / TREE)


def test_kfold_must_be_recorded(tmp_path):
    _write_fold(tmp_path, 0, 1, offset=0)
    (tmp_path / TREE / "fold0" / "config.json").write_text(json.dumps({}))
    with pytest.raises(FileNotFoundError, match="no kfold"):
        fold_directories(tmp_path / TREE)
