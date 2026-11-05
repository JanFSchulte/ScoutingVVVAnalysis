"""Legacy inputs of the QCD mass fit: qcd_est.json and the k-fold score caches (closure tests).

As legacy qcd_est.py with a k-fold bdt_root: the MC events of every class_groups sample from
<bdt_root>/kfold_scores/<sample>.npz passing every fold0/selection.json threshold but the ABCD
branch's (sentinel < -990 fails), weight = cached weight x lumi, out-of-fold proba; the samples of
exclude_samples are not read (legacy reads and drops them: zero bundles either way). Data
(data_sample) from <bdt_root>/kfold_scores/<sample>/*.npz with the same thresholds. A class is QCD
if its name contains "qcd". Samples are read one at a time into a MassfitAccumulator.

Deliberate simplifications: relative paths in the config resolve against the config's directory
(legacy: against background_estimation/); the data caches are globbed instead of being matched to
the current data file list and its entry counts (legacy kfold.load_data_scores opens every ROOT
file for that check); the non-k-fold mode (re-inference of a test split) is not ported.
"""

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vvv.background.qcd_massfit import MassfitAccumulator, MassfitConfig, load_sr_boxes
from vvv.ml.inference import threshold_mask


def _json(path):
    return json.loads(Path(path).read_text())


def _legacy_bdt_dir(bdt_root):
    for parent in Path(bdt_root).resolve().parents:
        if (parent / "kfold.py").exists() and (parent / "train.py").exists():
            return parent
    raise FileNotFoundError(f"no legacy selections/BDT directory above {bdt_root}")


def model_fingerprint(bdt_root, tree, tcfg, k):
    """legacy KFoldModels.fingerprint: SHA-256 (16 hex) of the K model files."""
    pattern = tcfg.get("model_pattern", "{output_root}/{tree_name}_model")
    h = hashlib.sha256()
    for fold in range(k):
        base = pattern.format(output_root=str(Path(bdt_root) / f"fold{fold}"), tree_name=tree)
        files = sorted(glob.glob(base + ".*"))
        if not files:
            raise FileNotFoundError(f"no model file {base}.*")
        for path in files:
            h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


@dataclass
class LegacySetup:
    """A resolved legacy qcd_est.json (abcd_mode massfit, k-fold bdt_root)."""

    raw: dict
    lumi: float
    bdt_root: Path
    tree: str
    k: int
    tcfg: dict
    branch: str
    window: object
    other_thresholds: dict
    class_names: list
    class_groups: dict
    qcd_classes: list
    sample_class: dict
    signal_samples: set
    exclude: set
    massfit: MassfitConfig
    data_sample: str | None
    signal_region_csv: Path
    output_dir: Path
    root_file_name: str

    @classmethod
    def load(cls, config_path, overrides=None):
        config_path = Path(config_path)
        raw = dict(_json(config_path), **(overrides or {}))
        res = lambda p: Path(p) if Path(p).is_absolute() else (config_path.parent / p).resolve()  # noqa: E731
        if str(raw.get("abcd_mode", "union")).lower() != "massfit":
            raise ValueError(f"only abcd_mode 'massfit' is ported, got {raw.get('abcd_mode')!r}")
        bdt_root = res(raw["bdt_root"])
        meta = _json(bdt_root / "fold0" / "test_ranges.json")
        if "kfold" not in meta:
            raise RuntimeError(f"{bdt_root} is not a k-fold training (the non-k-fold test split is not ported)")
        tree, k = meta["tree_name"], int(meta["kfold"])
        tcfg = _json(bdt_root / "fold0" / "config.json")
        thresholds = {b: (tuple(v) if isinstance(v, list) else v)
                      for b, v in _json(bdt_root / "fold0" / "selection.json")[tree].get("thresholds", {}).items()}
        branches = raw["abcd_branches"][tree]
        branches = [branches] if isinstance(branches, str) else list(dict.fromkeys(branches))
        if len(branches) != 1:
            raise RuntimeError("abcd_mode 'massfit' needs exactly one ABCD branch per tree (the fit variable)")
        branch = branches[0]
        if branch not in thresholds:
            raise RuntimeError(f"ABCD branch {branch} is missing from the selection.json thresholds")
        if float(raw.get("qcd_predict_scale_multipliers", {}).get(tree, 1.0)) != 1.0:
            raise RuntimeError("qcd_predict_scale_multipliers is not supported with abcd_mode 'massfit'")
        window = thresholds[branch]
        mf = raw.get("massfit", {}).get(tree)
        if not isinstance(mf, dict):
            raise RuntimeError(f"abcd_mode 'massfit' needs massfit['{tree}']")
        massfit = MassfitConfig.from_dict(mf, default_blind_window=window)
        sample_config = tcfg["sample_config"]
        if not Path(sample_config).is_absolute():
            sample_config = _legacy_bdt_dir(bdt_root) / sample_config
        info = {s["name"]: s for s in _json(sample_config)["sample"]}
        groups = tcfg["class_groups"]
        class_names = list(groups)
        qcd = [c for c in class_names if "qcd" in c.lower()]
        if not qcd:
            raise RuntimeError("class_groups must contain at least one QCD class")
        sample_class = {s: c for c, members in groups.items() for s in members}
        exclude = set(raw.get("exclude_samples", []))
        if exclude - set(sample_class):
            raise RuntimeError(f"exclude_samples names samples not in class_groups: {sorted(exclude - set(sample_class))}")
        for c in massfit.smooth_templates:
            if c not in groups or c in qcd or all(info[s]["is_signal"] for s in groups[c]):
                raise RuntimeError(f"massfit smooth_templates: '{c}' is not a non-QCD background class")
        data_sample = raw.get("data_sample")
        if data_sample is not None:
            if data_sample not in info or info[data_sample]["is_MC"]:
                raise RuntimeError(f"data_sample '{data_sample}' is not a data sample of the sample config")
            if massfit.blind_window is None:
                raise RuntimeError("data_sample needs massfit blind_window")
        return cls(raw, float(raw["lumi"]), bdt_root, tree, k, tcfg, branch, window,
                   {b: c for b, c in thresholds.items() if b != branch}, class_names, groups, qcd, sample_class,
                   {s for s in sample_class if info[s]["is_signal"]}, exclude, massfit, data_sample,
                   res(raw["signal_region_csv"]), res(raw.get("output_dir", "./output")),
                   raw.get("root_file_name", "qcd_abcd_yields.root"))

    def accumulator(self, log=print):
        boxes = load_sr_boxes(self.signal_region_csv, self.class_names)
        unknown = sorted(set(self.massfit.sr_slopes) - {b[0] for b in boxes})
        if unknown:
            raise RuntimeError(f"massfit sr_slopes names SRs not in the SR CSV: {unknown}")
        return MassfitAccumulator(self.massfit, boxes, self.class_names, self.sample_class, self.qcd_classes,
                                  self.window, log=log)

    def _read(self, path, fingerprint, keys):
        z = np.load(path, allow_pickle=False)
        if fingerprint is not None and str(z["model_fingerprint"]) != fingerprint:
            raise RuntimeError(f"{path} was scored with other models")
        needed = set(self.other_thresholds) | {self.branch}
        missing = [b for b in needed if f"b_{b}" not in z.files]
        if missing:
            raise KeyError(f"{path} lacks the branches {missing}")
        mask = None
        for b, cond in self.other_thresholds.items():  # one column at a time: the largest caches hold ~60M rows
            m = threshold_mask({b: z[f"b_{b}"]}, {b: cond})
            mask = m if mask is None else mask & m
        if mask is None:
            mask = np.ones(len(z[f"b_{self.branch}"]), dtype=bool)
        out = {k: z[k][mask] for k in keys}
        out["mass"] = z[f"b_{self.branch}"][mask].astype(np.float64)
        return out, int(mask.sum()), len(mask)

    def fill_mc(self, acc, check_fingerprint=True, log=print):
        """Fill acc with every class_groups sample (sample by sample)."""
        fp = model_fingerprint(self.bdt_root, self.tree, self.tcfg, self.k) if check_fingerprint else None
        n_tot = 0
        for sample in self.sample_class:
            if sample in self.exclude:
                log(f"  {sample}: excluded (exclude_samples)")
                continue
            ev, n_pass, n_all = self._read(self.bdt_root / "kfold_scores" / f"{sample}.npz", fp, ("proba", "weight"))
            acc.fill(sample, ev["mass"], ev["weight"] * self.lumi, proba=ev["proba"])
            n_tot += n_pass
            log(f"  {sample}: {n_pass} of {n_all} cached events")
            del ev
        log(f"Loaded MC events (k-fold scores, after thresholds): {n_tot}")

    def fill_data(self, acc, check_fingerprint=True, log=print):
        fp = model_fingerprint(self.bdt_root, self.tree, self.tcfg, self.k) if check_fingerprint else None
        files = sorted((self.bdt_root / "kfold_scores" / self.data_sample).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no data score cache under {self.bdt_root}/kfold_scores/{self.data_sample}")
        n_tot = n_all = 0
        for path in files:
            ev, n_pass, n = self._read(path, fp, ("proba",))
            acc.fill_data(ev["mass"], proba=ev["proba"])
            n_tot, n_all = n_tot + n_pass, n_all + n
        log(f"  {self.data_sample}: {n_tot} of {n_all} cached data events in {len(files)} files")
