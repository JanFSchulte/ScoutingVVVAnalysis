import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vvv.ml.inference import CLASSES
from vvv.sr.inputs import MASS_BRANCH, load_skim_inputs
from vvv.sr.optimise import SRConfig


def _setup(tmp_path):
    rng = np.random.default_rng(1)
    groups = {c: [f"s_{c}"] for c in CLASSES}
    catalogue = {f"s_{c}": SimpleNamespace(kind="signal" if c in ("VVV", "VH") else "background") for c in CLASSES}
    for c in CLASSES:
        n = 200
        proba = rng.dirichlet(np.ones(len(CLASSES)), n)
        table = {f"proba_{k}": proba[:, i] for i, k in enumerate(CLASSES)}
        table["weight"] = np.full(n, 2.0)
        table[MASS_BRANCH] = np.linspace(20, 249, n)
        d = tmp_path / "skims" / f"s_{c}" / "cat" / "nominal"
        d.mkdir(parents=True)
        pq.write_table(pa.table(table), d / "part-00000.parquet")
        (tmp_path / "skims" / f"s_{c}" / "_DONE.json").write_text("{}")
    tcfg = tmp_path / "training.json"
    tcfg.write_text(json.dumps({"class_groups": groups, "categories": {"cat": {"thresholds": {MASS_BRANCH: [50, 150]}}}}))
    return tcfg, catalogue


def test_scan_and_control_split(tmp_path):
    tcfg, catalogue = _setup(tmp_path)
    msd = np.linspace(20, 249, 200)
    plain = load_skim_inputs(SRConfig.from_dict({"lumi": 1.0}), tmp_path / "skims", "cat", tcfg, catalogue, log=lambda m: None)
    assert len(plain.weight) == 6 * np.count_nonzero((msd > 50) & (msd < 150))
    assert plain.signal_classes == [True, True, False, False, False, False]
    assert np.all(plain.weight == 2.0)  # stored weights are already x lumi
    abcd = SRConfig.from_dict({"lumi": 1.0, "abcd_uncertainty": {
        "branch": MASS_BRANCH, "control_window": [[20.0, 50.0], [150.0, 250.0]], "nonclosure": 0.2}})
    split = load_skim_inputs(abcd, tmp_path / "skims", "cat", tcfg, catalogue, log=lambda m: None)
    assert len(split.weight) == len(plain.weight)
    low = np.count_nonzero((msd > 20) & (msd < 50))
    high = np.count_nonzero((msd > 150) & (msd < 250))
    assert np.bincount(split.control["window"]).tolist() == [6 * low, 6 * high]
