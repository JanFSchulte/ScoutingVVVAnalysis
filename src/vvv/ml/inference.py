"""K-fold XGBoost inference and the legacy BDT threshold cuts.

A model directory (legacy train.py k-fold output, <dir>/<tree>/fold<r>/) holds
per rotation the booster <tree>_model.json, the feature list branch.json and
selection.json (clip_ranges, log_transform, thresholds). An event of fold r
(event number % K, for MC and data alike) is scored with the model of rotation r,
which neither trained nor monitored on it. K is the kfold the training recorded,
and the directories must be exactly fold0..fold<K-1> (fold_directories).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

SENTINEL = -990.0
CLASSES = ("VVV", "VH", "Top", "VV", "VJets", "QCD")


def fold_of(event, k):
    return (np.asarray(event, dtype=np.uint64) % np.uint64(k)).astype(np.int64)


def threshold_mask(columns, thresholds, apply_to_sentinel=True):
    """Legacy filter_X: per-branch conditions; sentinel values (< -990) fail."""

    def cond_mask(col, cond):
        if cond is None:
            return np.ones(len(col), dtype=bool)
        if isinstance(cond, (int, float)):
            return col > float(cond)
        if isinstance(cond, (list, tuple)) and len(cond) == 2 and not isinstance(cond[0], (list, dict, tuple)):
            lo, hi = cond
            m = np.ones(len(col), dtype=bool)
            if lo is not None:
                m &= col > lo
            if hi is not None:
                m &= col < hi
            return m
        if isinstance(cond, (list, tuple)):
            return np.logical_or.reduce([cond_mask(col, c) for c in cond])
        if isinstance(cond, dict):
            for key, reduce in (("&", np.logical_and), ("and", np.logical_and), ("|", np.logical_or), ("or", np.logical_or)):
                if key in cond:
                    return reduce.reduce([cond_mask(col, c) for c in cond[key]])
        raise ValueError(f"unsupported threshold condition {cond!r}")

    n = len(next(iter(columns.values())))
    mask = np.ones(n, dtype=bool)
    for branch, cond in thresholds.items():
        col = np.asarray(columns[branch], dtype=np.float64)
        sentinel = col < SENTINEL
        if apply_to_sentinel:
            mask &= ~sentinel & cond_mask(col, cond)
        else:
            mask &= cond_mask(col, cond) | sentinel
    return mask


def standardize(columns, features, clip_ranges, log_transform):
    """Legacy standardize_X as an (n_events, n_features) float array."""
    log_set = set(log_transform)
    out = np.empty((len(next(iter(columns.values()))), len(features)), dtype=np.float64)
    for j, name in enumerate(features):
        # Legacy features are float32 branches; rounding through float32 keeps the
        # booster's split comparisons identical.
        arr = np.asarray(columns[name], dtype=np.float32).astype(np.float64)
        valid = ~(arr < SENTINEL)
        lo, hi = clip_ranges.get(name, (None, None))
        if lo is not None:
            arr[valid & (arr < lo)] = lo
        if hi is not None:
            arr[valid & (arr > hi)] = hi
        if name in log_set:
            pos = valid & (arr > 0)
            arr[pos] = np.log(arr[pos])
        out[:, j] = arr
    return out


FOLD_DIR = re.compile(r"fold(\d+)")


def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def training_kfold(base):
    """K of the k-fold training under base, from fold0: the kfold of config.json (vvv.ml.train)
    or of test_ranges.json (legacy train.py); where both record it, they must agree."""
    recorded = {name: int(meta["kfold"]) for name in ("config.json", "test_ranges.json")
                if "kfold" in (meta := _read_json(base / "fold0" / name))}
    if not recorded:
        raise FileNotFoundError(f"{base}/fold0: no kfold in config.json or test_ranges.json")
    if len(set(recorded.values())) > 1:
        raise ValueError(f"{base}/fold0: the recorded kfold differs: {recorded}")
    return next(iter(recorded.values()))


def fold_directories(base):
    """[fold0, ..., fold<K-1>] of the training under base.

    Events are dispatched by event % K to the model at that position, so the directories must be
    exactly those K rotations (a missing or extra one would score events with a model that
    trained on them), and a rotation index recorded in a fold (training_log.json of vvv.ml.train,
    test_ranges.json of legacy train.py) must match its directory.
    """
    k = training_kfold(base)
    found = sorted((p.name for p in base.iterdir() if p.is_dir() and FOLD_DIR.fullmatch(p.name)),
                   key=lambda name: int(name[4:]))
    expected = [f"fold{r}" for r in range(k)]
    if found != expected:
        raise ValueError(f"{base}: fold directories {found}, a {k}-fold training needs exactly {expected}")
    for r, name in enumerate(expected):
        for meta_name in ("training_log.json", "test_ranges.json"):
            meta = _read_json(base / name / meta_name)
            if "fold" in meta and int(meta["fold"]) != r:
                raise ValueError(f"{base / name / meta_name}: records fold {meta['fold']}, expected {r}")
    return [base / name for name in expected]


def softmax(margins):
    m = margins - margins.max(axis=1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class CategoryModel:
    tree: str
    features: list[str]
    clip_ranges: dict
    log_transform: list[str]
    thresholds: dict
    boosters: list[xgb.Booster]

    @classmethod
    def load(cls, model_dir, tree):
        base = Path(model_dir) / tree
        folds = fold_directories(base)
        branch = json.loads((folds[0] / "branch.json").read_text())[tree]
        selection = json.loads((folds[0] / "selection.json").read_text())[tree]
        boosters = []
        for fold in folds:
            booster = xgb.Booster()
            booster.load_model(str(fold / f"{tree}_model.json"))
            boosters.append(booster)
        # The boosters' own feature lists are authoritative: legacy drops the mass-decorrelation
        # branches (msoftdrop_1, sphere_mass, mass_fraction_1) from branch.json at training.
        features = boosters[0].feature_names or [b["name"] for b in branch]
        if any(b.feature_names != boosters[0].feature_names for b in boosters):
            raise ValueError(f"{base}: folds were trained on different features")
        return cls(
            tree=tree,
            features=list(features),
            clip_ranges={k: tuple(v) for k, v in selection.get("clip_ranges", {}).items()},
            log_transform=list(selection.get("log_transform", [])),
            thresholds=selection.get("thresholds", {}),
            boosters=boosters,
        )

    @property
    def k(self):
        return len(self.boosters)

    def predict(self, columns, event):
        """Softmax class probabilities (n_events, 6), each event scored by its fold's model."""
        x = standardize(columns, self.features, self.clip_ranges, self.log_transform)
        folds = fold_of(event, self.k)
        proba = np.zeros((len(x), len(CLASSES)))
        for r, booster in enumerate(self.boosters):
            sel = folds == r
            if sel.any():
                dmat = xgb.DMatrix(x[sel], feature_names=self.features)
                margins = booster.predict(dmat, output_margin=True).reshape(sel.sum(), len(CLASSES))
                proba[sel] = softmax(margins)
        return proba


def load_signal_regions(csv_path):
    """Legacy signal_region.csv: [(bin_index, {class: (low, high)})], sorted by bin_index."""
    import csv

    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    regions = []
    for row in rows:
        bounds = {
            cls: (float(row[f"{cls}_low"]), float(row[f"{cls}_high"]))
            for cls in CLASSES
            if f"{cls}_low" in row and f"{cls}_high" in row
        }
        regions.append((int(round(float(row["bin_index"]))), bounds))
    return sorted(regions)


def region_index(proba, regions):
    """bin_index of the SR box containing each event (0: none), as qcd_est._region_mask:
    [low, high) per class axis, with an upper edge at 1 including 1."""
    idx = np.zeros(len(proba), dtype=np.int64)
    for bin_index, bounds in regions:
        inside = idx == 0
        for cls, (lo, hi) in bounds.items():
            c = proba[:, CLASSES.index(cls)]
            inside &= (c >= lo) & ((c < hi) if hi < 1.0 - 1e-12 else True)
        idx[inside] = bin_index
    return idx
