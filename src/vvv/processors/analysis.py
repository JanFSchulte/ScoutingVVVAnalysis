"""The analysis processor: corrections, categories, variables, weights,
classifier, signal regions and histograms in one pass over NanoEvents.

Per chunk and jet configuration (nominal + the eight jet variations for MC),
for every category:

- skim (optional): the category's variables, weights and weight-variation
  ratios are written to Parquet (<skim_dir>/<dataset>/<category>/<config>/),
  the training input and the replacement of the legacy converted trees; with skim_cut
  only the events that can reach a mass-fit histogram are written (the data skims), and
  for categories with a model, their out-of-fold class probabilities proba_<class> (the
  input of the signal-region optimisation);
- histograms (optional, needs models): the K-fold classifier scores the events
  passing the legacy BDT thresholds (all but the mass-fit branch), each event
  gets its SR bin_index (0 = outside every SR box), and the mass-fit histogram
  (dataset, category, region, syst, mass) is filled for the configuration and,
  at nominal, for every weight variation; theory members fill their own
  histograms with a member axis.
"""

import hashlib
import os
from pathlib import Path

import awkward as ak
import hist
import numpy as np
from coffea import processor
from coffea.lumi_tools import LumiMask

from vvv.config import AnalysisConfig
from vvv.ml.inference import CLASSES, CategoryModel, region_index, threshold_mask
from vvv.objects import CATEGORIES
from vvv.selection import EventSelection, preselection
from vvv.variables import category_variables, passthrough
from vvv.weights import N_ALPHAS, N_PDF, N_PS, N_SCALE, SampleWeights

MASS_BRANCH = "ScoutingFatPFJetRecluster_msoftdrop_1"
THEORY_SETS = {"LHEScaleWeight": N_SCALE, "LHEPdfWeight": N_PDF, "LHEPdfWeightAlphaS": N_ALPHAS, "PSWeight": N_PS}


def mass_hist(mass_axis):
    return hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.StrCategory(list(CATEGORIES), name="category"),
        hist.axis.IntCategory([], name="region", growth=True),
        hist.axis.StrCategory([], name="syst", growth=True),
        mass_axis,
        storage=hist.storage.Weight(),
    )


def member_hist(mass_axis, n_members):
    return hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.StrCategory(list(CATEGORIES), name="category"),
        hist.axis.IntCategory([], name="region", growth=True),
        hist.axis.Integer(0, n_members, name="member"),
        mass_axis,
        storage=hist.storage.Weight(),
    )


def _prescale_mask(event, prescale):
    """Keep 1 in prescale events by a hash of the event number, independent of event % K folds."""
    if prescale <= 1:
        return np.ones(len(event), dtype=bool)
    from vvv.corrections.jets import _splitmix64

    return (_splitmix64(np.asarray(event, dtype=np.uint64)) % np.uint64(prescale)) == 0


class AnalysisProcessor(processor.ProcessorABC):
    def __init__(
        self,
        cfg: AnalysisConfig,
        samples,
        metadata,
        lumi_fb,
        models=None,
        signal_regions=None,
        mass_axis=None,
        skim_dir=None,
        skim_variations=False,
        skim_prescale=None,
        skim_theory=False,
        skim_cut=None,
        jet_variations=True,
        blind=True,
        blind_window=(50.0, 150.0),
    ):
        """
        samples: {name: Sample}; metadata: {name: metadata record} (MC);
        models: {category: model_dir} of legacy-format K-fold trainings;
        signal_regions: {category: [(bin_index, {class: (lo, hi)})]};
        skim_cut: {category: thresholds}: skim only events passing the thresholds other than
        the mass-fit branch, with the mass-fit branch inside the mass axis (what _fill keeps).
        """
        self.cfg = cfg
        self.samples = samples
        self.metadata = metadata
        self.lumi_fb = lumi_fb
        self.model_dirs = models or {}
        self.signal_regions = signal_regions or {}
        self.mass_axis = mass_axis or hist.axis.Regular(23, 20.0, 250.0, name="mass", label=r"$m_{SD}^{j1}$ [GeV]")
        self.skim_dir = Path(skim_dir) if skim_dir else None
        self.skim_variations = skim_variations
        # {sample: {category: P}}: keep 1 in P events (hash of the event number), weight x P.
        self.skim_prescale = skim_prescale or {}
        self.skim_theory = skim_theory
        self.skim_cut = {cat: {k: v for k, v in thr.items() if k != MASS_BRANCH} for cat, thr in (skim_cut or {}).items()}
        self.jet_variations = jet_variations
        self.blind = blind
        self.blind_window = blind_window
        self._selection = None
        self._models = None

    # Built lazily on the worker: correctionlib and xgboost objects do not pickle.
    @property
    def selection(self):
        if self._selection is None:
            self._selection = EventSelection(self.cfg)
        return self._selection

    @property
    def models(self):
        if self._models is None:
            self._models = {cat: CategoryModel.load(d, cat) for cat, d in self.model_dirs.items()}
        return self._models

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_selection"] = None
        state["_models"] = None
        return state

    def process(self, events):
        dataset = events.metadata["dataset"]
        sample = self.samples[dataset]
        out = {
            "cutflow": {dataset: {"all": len(events)}},
            "mass": mass_hist(self.mass_axis),
            **{name: member_hist(self.mass_axis, n) for name, n in THEORY_SETS.items()},
            "rho_failures": {dataset: 0},
            "rho_consensus": {dataset: 0},
        }
        lumi_mask = None if sample.is_mc else LumiMask(str(self.cfg.lumi_mask))
        events = events[ak.to_numpy(preselection(events, self.cfg, lumi_mask))]
        out["cutflow"][dataset]["preselection"] = len(events)
        if len(events) == 0:
            return out

        weights = SampleWeights(sample, self.metadata.get(dataset, {}))
        w_nominal = weights.nominal(events) * (self.lumi_fb if sample.is_mc else 1.0)
        w_variations = weights.variations(events)

        # Jet variations matter only for histograms (or explicitly skimmed variations).
        vary = sample.is_mc and self.jet_variations and (bool(self.model_dirs) or self.skim_variations)
        configurations = ("nominal",) + (self.cfg.jet_variations if vary else ())
        selected, info = self.selection(events, sample.is_mc, configurations)
        out["rho_failures"][dataset] = int((~info["rho_ok"]).sum())
        out["rho_consensus"][dataset] = int(info["n_rho_consensus"])
        event_number = ak.to_numpy(events.event)
        ids = passthrough(events, sample.is_mc)
        chunk_id = _chunk_id(events.metadata)

        for config_name, obj in selected.items():
            for category in CATEGORIES:
                mask = ak.to_numpy(obj["categories"][category])
                if config_name == "nominal":
                    out["cutflow"][dataset][category] = int(mask.sum())
                if not mask.any():
                    continue
                columns = {
                    k: ak.to_numpy(v)[mask] for k, v in category_variables(obj, category, self.cfg.objects).items()
                }
                w = w_nominal[mask]
                if self.skim_dir is not None and (config_name == "nominal" or self.skim_variations):
                    prescale = int(self.skim_prescale.get(dataset, {}).get(category, 1))
                    keep = _prescale_mask(event_number[mask], prescale)
                    if category in self.skim_cut:
                        keep &= self._fit_region(columns, category)
                        if config_name == "nominal":
                            out["cutflow"][dataset][f"{category}_skim"] = int(keep.sum())
                    variations = {}
                    if config_name == "nominal":
                        variations = {k: v[mask][keep] for k, v in w_variations.items()
                                      if self.skim_theory or k not in THEORY_SETS}
                    skim = {k: v[keep] for k, v in {**{k: v[mask] for k, v in ids.items()}, **columns}.items()}
                    if category in self.skim_cut and category in self.model_dirs:
                        proba = self.models[category].predict(skim, event_number[mask][keep])
                        skim.update({f"proba_{c}": proba[:, i] for i, c in enumerate(CLASSES)})
                    self._write_skim(dataset, category, config_name, chunk_id, skim, w[keep] * prescale, variations)
                if category in self.model_dirs:
                    self._fill(out, dataset, sample, category, config_name, columns, event_number[mask], w,
                               {k: v[mask] for k, v in w_variations.items()} if config_name == "nominal" else {})
        return out

    def _fit_region(self, columns, category):
        lo, hi = self.mass_axis.edges[0], self.mass_axis.edges[-1]
        mass = np.asarray(columns[MASS_BRANCH], dtype=np.float64)
        return threshold_mask(columns, self.skim_cut[category]) & (mass >= lo) & (mass < hi)

    def _fill(self, out, dataset, sample, category, config_name, columns, event, w, w_variations):
        model = self.models[category]
        thresholds = {k: v for k, v in model.thresholds.items() if k != MASS_BRANCH}
        keep = threshold_mask(columns, thresholds)
        if not keep.any():
            return
        cols = {k: v[keep] for k, v in columns.items()}
        proba = model.predict(cols, event[keep])
        region = region_index(proba, self.signal_regions.get(category, []))
        mass = np.asarray(cols[MASS_BRANCH], dtype=np.float64)
        weight = w[keep]
        if not sample.is_mc and self.blind:
            lo, hi = self.blind_window
            unblinded = ~((region > 0) & (mass > lo) & (mass < hi))
            region, mass, weight = region[unblinded], mass[unblinded], weight[unblinded]
            if not len(mass):
                return

        fill = dict(dataset=dataset, category=category, region=region, mass=mass)
        out["mass"].fill(syst=config_name, weight=weight, **fill)
        for name, ratio in w_variations.items():
            ratio = ratio[keep]
            if name in THEORY_SETS:
                n_members = ratio.shape[1]
                out[name].fill(
                    dataset=dataset,
                    category=category,
                    region=np.repeat(region, n_members),
                    member=np.tile(np.arange(n_members), len(region)),
                    mass=np.repeat(mass, n_members),
                    weight=(weight[:, None] * ratio).ravel(),
                )
            else:
                out["mass"].fill(syst=name, weight=weight * ratio, **fill)

    def _write_skim(self, dataset, category, config_name, chunk_id, columns, w, w_variations):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = {k: np.asarray(v) for k, v in columns.items()}
        table["weight"] = w
        for name, ratio in w_variations.items():
            if ratio.ndim == 1:
                table[f"weight_{name}"] = ratio
            else:
                for m in range(ratio.shape[1]):
                    table[f"{name}_{m}"] = ratio[:, m]
        path = self.skim_dir / dataset / category / config_name
        path.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a worker killed mid-write leaves only a dot file, which readers ignore.
        tmp = path / f".{chunk_id}.parquet.tmp"
        pq.write_table(pa.table(table), tmp, compression="zstd")
        os.replace(tmp, path / f"{chunk_id}.parquet")

    def postprocess(self, accumulator):
        return accumulator


def _chunk_id(metadata):
    """Name of a chunk's skim files: fixed by its file and entry range, so a retried or
    re-run chunk overwrites its own files instead of duplicating its events."""
    key = f"{metadata.get('fileuuid', metadata.get('filename'))}:{metadata['entrystart']}:{metadata['entrystop']}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]
