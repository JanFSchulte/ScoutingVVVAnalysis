"""Inputs of the SR optimiser from a legacy k-fold score cache (closure tests).

A legacy k-fold tree (<bdt_root>/fold<r>/, train.py kfold = K) has kfold_scores/<sample>.npz per
MC sample (score_kfold.py): proba (N, C) out-of-fold, weight (pre-lumi physics weight), class_idx
and b_<branch> columns, for the events passing the non-mass thresholds. As legacy
signal_region.py / signal_region_hist.py: samples in class_groups order, weight x lumi, every
threshold of fold0/selection.json applied (sentinel < -990 fails); with abcd_uncertainty every
threshold but the ABCD branch's, the scan events passing that one and the control events failing
it inside a control window (low < value < high). A class is signal if all its samples are.
"""

import glob
import hashlib
import json
from pathlib import Path

import numpy as np

from vvv.ml.inference import threshold_mask
from vvv.sr.optimise import ScanInputs


def _json(path):
    return json.loads(Path(path).read_text())


def _legacy_bdt_dir(bdt_root):
    """legacy selections/BDT (train.py's working directory, base of relative config paths)."""
    for parent in Path(bdt_root).resolve().parents:
        if (parent / "kfold.py").exists() and (parent / "train.py").exists():
            return parent
    raise FileNotFoundError(f"no legacy selections/BDT directory above {bdt_root}")


def _fingerprint(bdt_root, tree, cfg, k):
    """legacy KFoldModels.fingerprint: SHA-256 (16 hex) of the K model files."""
    pattern = cfg.get("model_pattern", "{output_root}/{tree_name}_model")
    h = hashlib.sha256()
    for fold in range(k):
        base = pattern.format(output_root=str(Path(bdt_root) / f"fold{fold}"), tree_name=tree)
        files = sorted(glob.glob(base + ".*"))
        if not files:
            raise FileNotFoundError(f"no model file {base}.*")
        for path in files:
            h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def load_legacy_inputs(cfg, bdt_root, sample_config=None, check_fingerprint=True, log=print):
    """ScanInputs (weights x cfg.lumi) of a legacy k-fold bdt_root for the SRConfig cfg."""
    bdt_root = Path(bdt_root)
    fold0 = bdt_root / "fold0"
    meta = _json(fold0 / "test_ranges.json")
    if "kfold" not in meta:
        raise RuntimeError(f"{bdt_root} is not a k-fold training (no kfold in fold0/test_ranges.json)")
    tree, k = meta["tree_name"], int(meta["kfold"])
    tcfg = _json(fold0 / "config.json")
    thresholds = {b: (tuple(v) if isinstance(v, list) else v)
                  for b, v in _json(fold0 / "selection.json")[tree].get("thresholds", {}).items()}
    if sample_config is None:
        sample_config = tcfg["sample_config"]
        if not Path(sample_config).is_absolute():
            sample_config = _legacy_bdt_dir(bdt_root) / sample_config
    info = {s["name"]: s for s in _json(sample_config)["sample"]}
    groups = tcfg["class_groups"]
    class_names = list(groups)
    signal = [bool(m) and all(info[s]["is_signal"] for s in m) for m in groups.values()]
    samples = [s for m in groups.values() for s in m]
    fingerprint = _fingerprint(bdt_root, tree, tcfg, k) if check_fingerprint else None

    abcd = cfg.abcd
    if abcd is not None and abcd.branch not in thresholds:
        raise KeyError(f"abcd_uncertainty branch {abcd.branch} has no threshold in selection.json")
    others = {b: c for b, c in thresholds.items() if abcd is None or b != abcd.branch}
    log(f"Loading the k-fold score cache: {k} folds, {len(samples)} samples"
        + (f" (every threshold but {abcd.branch})" if abcd is not None else ""))
    parts = {"scan": [], "control": []}
    for sample in samples:
        path = bdt_root / "kfold_scores" / f"{sample}.npz"
        z = np.load(path, allow_pickle=False)
        if fingerprint is not None and str(z["model_fingerprint"]) != fingerprint:
            raise RuntimeError(f"{path} was scored with other models")
        cols = {b: z[f"b_{b}"] for b in thresholds}
        mask = threshold_mask(cols, others) if others else np.ones(len(z["weight"]), dtype=bool)
        if abcd is None:
            sel = {"scan": mask}
        else:
            v = np.asarray(cols[abcd.branch], dtype=float)
            passed = threshold_mask({abcd.branch: v}, {abcd.branch: thresholds[abcd.branch]})
            window = np.full(len(v), -1, dtype=int)
            for i, (low, high) in enumerate(abcd.windows):
                window[mask & ~passed & (v > low) & (v < high)] = i
            sel = {"scan": mask & passed, "control": window >= 0}
        proba, weight, y = z["proba"], z["weight"], z["class_idx"]
        for key, m in sel.items():
            part = {"proba": proba[m], "weight": weight[m], "class_idx": y[m].astype(int),
                    "sample": np.full(int(m.sum()), sample, dtype=object)}
            if key == "control":
                part["window"] = window[m]
            parts[key].append(part)
        log(f"  {sample}: {int(mask.sum())} of {len(mask)} cached events (n_entries={int(z['n_entries'])})")
        del z, proba, weight, y, cols

    def cat(lst):
        out = {key: np.concatenate([p[key] for p in lst]) for key in lst[0]}
        out["weight"] = out["weight"] * cfg.lumi
        return out

    scan = cat(parts["scan"])
    control = cat(parts["control"]) if abcd is not None else None
    log(f"After thresholds: {len(scan['weight'])} events (all MC entries, k-fold scores)"
        + ("" if control is None else "; control windows " + ", ".join(
            f"{lw:g} < {abcd.branch} < {hw:g}: {int(np.count_nonzero(control['window'] == i))} events"
            for i, (lw, hw) in enumerate(abcd.windows))))
    return ScanInputs(scan["proba"], scan["weight"], scan["class_idx"], scan["sample"], class_names, signal, control)
