"""QCD mass-fit inputs from the `vvv run` histograms (counterpart of skims.SkimSetup and legacy.LegacySetup).

The massfit settings and exclude_samples come from a legacy qcd_est.json, the classes and the
msoftdrop_1 window from the coffea training config, the SR boxes from the signal_region.csv the
histograms were filled with (its SHA-256 is checked against the one the pickles record), and the
binned sums from the nominal mass[dataset, category, region, syst, mass] histogram of the same pass
that feeds scripts/make_syst_yields.py (MassfitInputs.from_hist). The smoothed templates
(massfit smooth_templates {class: f}) come from mass_smooth[dataset, category, factor, region, mass],
filled by `vvv run` for the loosening factors given with --smooth-factor (default 2). Data
(data_sample) is used when the histograms hold it.
"""

import json
from pathlib import Path

import numpy as np

from vvv.background.qcd_massfit import MassfitConfig, MassfitInputs, load_sr_boxes
from vvv.ml.inference import CLASSES

MASS_BRANCH = "ScoutingFatPFJetRecluster_msoftdrop_1"


class HistSetup:
    def __init__(self, qcd_est_config, training_config, category, hists, signal_region_csv, output_dir,
                 data_sample="data_2024", catalogue=None):
        from vvv.samples import load_samples

        raw = json.loads(Path(qcd_est_config).read_text())
        tcfg = json.loads(Path(training_config).read_text())
        if str(raw.get("abcd_mode", "union")).lower() != "massfit":
            raise ValueError(f"only abcd_mode 'massfit' is ported, got {raw.get('abcd_mode')!r}")
        cc = tcfg["categories"][category]
        self.raw, self.tree, self.branch = raw, category, MASS_BRANCH
        self.window = tuple(cc["thresholds"][MASS_BRANCH])
        self.massfit = MassfitConfig.from_dict(raw["massfit"][category], default_blind_window=self.window)
        self.class_groups = tcfg["class_groups"]
        self.class_names = list(self.class_groups)
        if self.class_names != list(CLASSES):
            raise ValueError(f"class_groups {self.class_names} are not in the model's class order {CLASSES}")
        self.qcd_classes = [c for c in self.class_names if "qcd" in c.lower()]
        self.exclude = set(raw.get("exclude_samples", []))
        self.sample_class = {s: c for c, members in self.class_groups.items() for s in members}
        catalogue = catalogue or load_samples()
        for c in self.massfit.smooth_templates:
            if c not in self.class_groups or c in self.qcd_classes or all(
                    catalogue[s].kind == "signal" for s in self.class_groups[c]):
                raise RuntimeError(f"massfit smooth_templates: '{c}' is not a non-QCD background class")
        self.lumi = float(raw.get("lumi", 0))  # informational: the histograms are already x lumi
        self.hist_paths = [Path(h) for h in hists]
        self.bdt_root = self.hist_paths[0]
        self.signal_region_csv = Path(signal_region_csv)
        self.output_dir = Path(output_dir)
        self.root_file_name = raw.get("root_file_name", "qcd_abcd_yields.root")
        self.data_sample = data_sample
        self.outputs = None

    def inputs(self, log=print):
        from vvv.stats.systematics import HistOutputs

        self.outputs = out = HistOutputs.load(self.hist_paths, log=log)
        out.check_signal_regions(self.tree, self.signal_region_csv)
        if out.lumi is not None and self.lumi and abs(out.lumi - self.lumi) > 1e-6:
            log(f"[WARN] histograms normalised to {out.lumi} fb^-1, the qcd_est config says {self.lumi}")
        sr_ids = [b[0] for b in load_sr_boxes(self.signal_region_csv, self.class_names)]
        h = out.hists["mass"]
        datasets = set(h.axes["dataset"])
        for s in sorted(set(self.sample_class) - datasets - self.exclude):
            log(f"  {s}: no histograms")
        smooth = {}
        for cls, factor in self.massfit.smooth_templates.items():
            hs = out.hists.get("mass_smooth")
            label = f"{float(factor):g}"  # processors.analysis.factor_label
            if hs is None or label not in hs.axes["factor"]:
                raise RuntimeError(f"massfit smooth_templates {cls}: no mass_smooth histogram for factor {label} "
                                   f"(vvv run --smooth-factor {label})")
            n_bins = h.axes["mass"].size
            sw, sw2 = np.zeros((len(sr_ids), n_bins)), np.zeros((len(sr_ids), n_bins))
            present = list(hs.axes["region"])
            for s, c in self.sample_class.items():
                if c != cls or s in self.exclude or s not in hs.axes["dataset"]:
                    continue
                v = hs[{"dataset": s, "category": self.tree, "factor": label}].view(flow=False)
                for i, sr in enumerate(sr_ids):
                    if sr in present:
                        sw[i] += v["value"][present.index(sr)]
                        sw2[i] += v["variance"][present.index(sr)]
            n = np.where(sw2 > 0, sw ** 2 / np.where(sw2 > 0, sw2, 1.0), 0.0)
            smooth[cls] = (sw, sw2, n)
        data = [self.data_sample] if self.data_sample in datasets else []
        inputs = MassfitInputs.from_hist(h, self.sample_class, sr_ids, self.class_names, self.qcd_classes,
                                         window=self.window, category=self.tree, data_datasets=data,
                                         blind_window=self.massfit.blind_window, smooth=smooth, log=log)
        for s in self.exclude & set(self.sample_class):  # zero bundles, as legacy and the skim setup
            log(f"  {s}: excluded (exclude_samples)")
            inputs.sums[s] = tuple(np.zeros_like(a) for a in inputs.sums[s])
            inputs.a_sums[s] = tuple(np.zeros_like(a) for a in inputs.a_sums[s])
            inputs.abs_w[s] = np.zeros(0)
        if smooth:
            inputs.approximations.append("effective MC entry counts of the smoothed templates")
        if not data:
            log(f"  no {self.data_sample} histograms: MC only")
            self.data_sample = None
        return inputs
