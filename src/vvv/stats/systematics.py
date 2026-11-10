"""Systematic-uncertainty inputs of the legacy combine.C from the `vvv run` histograms.

Port of the legacy yield-ratio producers theory_syst.py (mode 8), pileup_syst.py (mode 9) and
jes/jer/jms/jmr_syst.py (modes 11-14, selections/jet_syst_common/jet_variation_common.py). They
re-read the converted trees; here every number comes from the histograms of one `vvv run` pass:

- mass[dataset, category, region, syst, mass]: syst = nominal, the eight jet variations
  (<jes|jer|jms|jmr>_<up|down>, each with its own selection, classifier scores and msoftdrop_1) and
  pileup_up / pileup_down (nominal events, weight x the varied / nominal pileup weight, each
  normalised to its own all-generated-event mean);
- LHEScaleWeight [9], LHEPdfWeight [101], LHEPdfWeightAlphaS [2], PSWeight [4] as
  [dataset, category, region, member, mass]: nominal events, weight x member / its generator-level
  mean (Runs-tree sums), so every variation changes the selection efficiency only.

Region 0 is outside every SR box and region i the SR with bin_index i; every filled event passed the
category thresholds except the msoftdrop_1 window. The legacy regions are, per sample and category:

- inclusive: the post-threshold selection, all regions in the msoftdrop_1 threshold window;
- regions[<bin_index>]: the ABCD A region, SR box i in the window;
- with mass_fit (optional, legacy per-tree config mass_fit = {branch, range, width}): mass_range
  (SR box i over the fit range) and mass_bins (one entry per fit bin).

Every entry holds the legacy ratios and the sums combine.C adds up over the samples of a datacard
process (yields per fb^-1, i.e. the histogram sums / lumi, and their sums of squared weights):

- jes/jer/jms/jmr: <s>_up = sum w (variation) / sum w (nominal) in the same region, each event in the
  region its own configuration puts it in;
- pileup: pu_up / pu_down = sum w x r_up / sum w;
- theory: member ratios Y_i / Y_0; PDF hessian_symmetric sigma = sqrt(sum_{i=1..100} (r_i - r_0)^2)
  (mc_replica: half the 16th-84th interval of the sorted members) with alpha_s (r_up - r_down) / 2 in
  quadrature, pdf_up/down = centre +- sigma; scale: envelope (max / min) of the 7 (mu_R, mu_F)
  members without the anti-correlated corners 2 and 6; PS: ps_isr_up / down = PSWeight [2] / [0]
  (isr.murfac 0.5 / 2), ps_fsr_up / down = [3] / [1].

Fallback rule (min_region_events, legacy 10): a region keeps its own ratios only if its nominal sum
and every varied sum the nuisance uses (theory: the 101 PDF members, the alpha_s members when
stored, the 7 envelope scale members, the 4 PS members; pileup and jets: up and down) are positive
with n_eff = (sum w)^2 / sum w^2 >= min_region_events; otherwise it takes the ratios of the next
coarser region (mass bin -> mass_range -> inclusive, SR window -> inclusive), flagged
fallback_to_mass_range / fallback_to_inclusive. combine.C applies the same rule again to the
summed sums of each process.

Deliberate differences from legacy (the histograms keep sum w and sum w^2 only):

- n_events is the effective count round(n_eff), not the raw MC event count, and nom_sum is the sum
  of the lumi-normalised weights instead of the unnormalised genWeight / <genWeight> x w_PU (combine.C
  reads neither);
- the msoftdrop_1 window is a set of whole mass bins: events exactly on a window edge (50 < m < 150
  in legacy) fall into the bin whose lower edge they sit on.
"""

import csv
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MASS_BRANCH = "ScoutingFatPFJetRecluster_msoftdrop_1"
JET_SYSTS = ("jes", "jer", "jms", "jmr")
SYSTS = ("theory", "pileup") + JET_SYSTS
MEMBER_SETS = {"pdf": "LHEPdfWeight", "alphas": "LHEPdfWeightAlphaS", "scale": "LHEScaleWeight", "ps": "PSWeight"}
N_MEMBERS = {"pdf": 101, "alphas": 2, "scale": 9, "ps": 4}
SCALE_VALID = [0, 1, 3, 4, 5, 7, 8]
PDF_METHODS = ("hessian_symmetric", "mc_replica")
RATIO_FIELDS = ("generator_normalised", "pdf_method", "pdf_alphas_unc", "alphas_available", "pdf_up", "pdf_down",
                "pdf_central", "scale_up", "scale_down", "ps_isr_up", "ps_isr_down", "ps_fsr_up", "ps_fsr_down")
THEORY_CSV = (("pdf", "pdf_up", "pdf_down"), ("scale", "scale_up", "scale_down"),
              ("ps_isr", "ps_isr_up", "ps_isr_down"), ("ps_fsr", "ps_fsr_up", "ps_fsr_down"))


def output_name(syst):
    return f"{syst}_syst_yields"


def sr_csv_hash(csv_path):
    """Legacy sr_csv_hash: the first 13 hex digits of the SHA-256 of the SR CSV bytes."""
    return hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()[:13]


def sr_bin_ids(csv_path):
    """bin_index values of a signal_region.csv in file order (legacy bin_ids)."""
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"signal region CSV is empty: {csv_path}")
    return [int(round(float(r["bin_index"]))) for r in rows]


def n_eff(s, s2):
    """Effective number of events (sum w)^2 / sum w^2 of signed weights (0 without events)."""
    return float(s * s / s2) if s2 > 0.0 else 0.0


# --- histogram input ----------------------------------------------------------


def _pickles(paths):
    out = []
    for p in map(Path, paths):
        out += sorted(p.rglob("*.pkl")) if p.is_dir() else [p]
    if not out:
        raise FileNotFoundError(f"no histogram pickles in {[str(p) for p in paths]}")
    return out


@dataclass
class HistOutputs:
    """The histograms of one or more `vvv run` output pickles, added up (a dataset split over several
    pickles, e.g. batches, is summed), with their provenance (lumi, SR CSV hashes) when recorded."""

    hists: dict
    lumi: float | None = None
    signal_regions: dict = field(default_factory=dict)
    files: list = field(default_factory=list)

    NAMES = ("mass", "mass_smooth") + tuple(MEMBER_SETS.values())

    @classmethod
    def load(cls, paths, log=print):
        hists, lumis, regions, files = {}, set(), {}, []
        for path in _pickles(paths):
            with open(path, "rb") as handle:
                out = pickle.load(handle)
            files.append(str(path))
            for name in cls.NAMES:
                if name in out:
                    hists[name] = out[name] if name not in hists else hists[name] + out[name]
            if "lumi_fb" in out:
                lumis.add(float(out["lumi_fb"]))
            for cat, info in out.get("signal_regions", {}).items():
                if regions.setdefault(cat, info["sha256"]) != info["sha256"]:
                    raise ValueError(f"{path}: the {cat} histograms were made with other signal regions than "
                                     f"those of the other pickles")
        if "mass" not in hists:
            raise ValueError("the pickles hold no mass histogram (vvv run without --model-dir?)")
        if len(lumis) > 1:
            raise ValueError(f"the pickles are normalised to different luminosities: {sorted(lumis)}")
        log(f"Loaded {len(files)} histogram pickles: {len(list(hists['mass'].axes['dataset']))} datasets")
        return cls(hists, lumis.pop() if lumis else None, regions, files)

    def datasets(self, name="mass"):
        return list(self.hists[name].axes["dataset"]) if name in self.hists else []

    def check_signal_regions(self, category, csv_path):
        """The SR CSV must be the one the histograms were filled with (when the pickles record it)."""
        recorded = self.signal_regions.get(category)
        if recorded is not None and recorded != hashlib.sha256(Path(csv_path).read_bytes()).hexdigest():
            raise ValueError(f"{csv_path} is not the signal_region.csv the {category} histograms were filled with")


def _region_slices(h, sr_ids, where):
    """Indices of region 0 and the SRs (sr_ids order) on the region axis; -1 for a region without
    entries. A filled region outside the CSV means other signal regions: an error."""
    present = list(h.axes["region"])
    # The region axis is shared by the categories: only a region with entries in this one counts.
    view = h.view(flow=False)
    filled = {r for i, r in enumerate(present) if np.any(view["variance"][i] != 0) or np.any(view["value"][i] != 0)}
    unknown = sorted(filled - {0, *sr_ids})
    if unknown:
        raise ValueError(f"{where}: regions {unknown} are not in the SR CSV (histograms of other signal regions?)")
    return [present.index(r) if r in present else -1 for r in [0, *sr_ids]]


def _project(view, idx, n_mass, extra=()):
    """[region, *extra, mass] array of a sliced Weight-storage view, zero for absent regions."""
    out_v = np.zeros((len(idx),) + tuple(extra) + (n_mass,))
    out_w2 = np.zeros_like(out_v)
    for k, i in enumerate(idx):
        if i >= 0:
            out_v[k] = view["value"][i]
            out_w2[k] = view["variance"][i]
    return out_v, out_w2


@dataclass
class CellHists:
    """Binned sums of one sample in one category: nominal and every variation as (sw, sw2) arrays
    [region, mass] (region 0 = outside the SR boxes, k = the k-th SR of the CSV); the theory member
    sets as [region, member, mass]; None for a variation or member set the histograms lack."""

    edges: np.ndarray
    sums: dict

    @classmethod
    def from_outputs(cls, outputs, dataset, category, sr_ids):
        h = outputs.hists["mass"]
        edges = np.asarray(h.axes["mass"].edges, dtype=float)
        sums = {}
        if dataset not in h.axes["dataset"]:
            return cls(edges, sums)
        hc = h[{"dataset": dataset, "category": category}]
        idx = _region_slices(hc, sr_ids, f"{dataset}/{category}")
        syst_ax = list(hc.axes["syst"])
        view = hc.view(flow=False)
        for k, syst in enumerate(syst_ax):
            sums[syst] = _project(view[:, k, :], idx, len(edges) - 1)
        for key, name in MEMBER_SETS.items():
            hm = outputs.hists.get(name)
            if hm is None or dataset not in hm.axes["dataset"]:
                continue
            hmc = hm[{"dataset": dataset, "category": category}]
            sums[key] = _project(hmc.view(flow=False), _region_slices(hmc, sr_ids, f"{name} {dataset}/{category}"),
                                 len(edges) - 1, (hmc.axes["member"].size,))
        return cls(edges, sums)

    def has(self, key):
        return key in self.sums

    def total(self, key, regions, bins):
        """(sum w, sum w^2) of one variation over the given region rows and mass bins (arrays over the
        members for a member set)."""
        sw, sw2 = self.sums[key]
        sel = (np.asarray(regions)[:, None], np.asarray(bins)[None, :]) if sw.ndim == 2 else None
        if sw.ndim == 2:
            return float(sw[sel].sum()), float(sw2[sel].sum())
        sub, sub2 = sw[np.asarray(regions)][..., np.asarray(bins)], sw2[np.asarray(regions)][..., np.asarray(bins)]
        return sub.sum(axis=(0, 2)), sub2.sum(axis=(0, 2))


def window_bins(edges, window):
    """Mass bins inside the msoftdrop_1 threshold window (low, high): its edges must be bin edges."""
    lo, hi = window
    for v in (lo, hi):
        if v is not None and not np.any(np.isclose(edges, v)):
            raise ValueError(f"the msoftdrop_1 window edge {v} is not a mass-bin edge of the histograms {edges.tolist()}")
    centres = 0.5 * (edges[:-1] + edges[1:])
    inside = np.ones(len(centres), dtype=bool)
    if lo is not None:
        inside &= centres > lo
    if hi is not None:
        inside &= centres < hi
    return np.flatnonzero(inside)


def mass_fit_bins(edges, mass_range, width):
    """Fit-bin edges (legacy mass_fit_context: range in steps of width) and, per fit bin, the histogram
    bins it merges; the fit edges must be histogram edges."""
    fit = np.arange(float(mass_range[0]), float(mass_range[1]) + 0.5 * width, float(width))
    if len(fit) < 3 or abs(fit[-1] - float(mass_range[1])) > 1e-9 * width:
        raise ValueError("mass_fit: width must divide the range into at least two bins")
    pos = [int(np.argmin(np.abs(edges - e))) for e in fit]
    if not all(np.isclose(edges[p], e) for p, e in zip(pos, fit)):
        raise ValueError(f"mass_fit edges {fit.tolist()} are not edges of the histograms' mass axis {edges.tolist()}")
    return fit, [np.arange(pos[b], pos[b + 1]) for b in range(len(fit) - 1)]


# --- entries (legacy semantics) ---------------------------------------------------


def _enough(pairs, min_events):
    return all(s > 0.0 and n_eff(s, s2) >= min_events for s, s2 in pairs)


def updown_entry(syst, nom, up, down, lumi, up_key, down_key, fallback=None, flag=None, min_events=0):
    """jet_variation_common.ratio_entry / pileup_syst._entry: (sum w, sum w^2) of the nominal, up and
    down weights of one region -> its entry."""
    w, w2 = nom
    entry = {"nom_sum": w, "n_events": int(round(n_eff(w, w2)))}
    if w > 0.0 and (fallback is None or _enough((nom, up, down), min_events)):
        entry[up_key], entry[down_key] = up[0] / w, down[0] / w
    else:
        entry[up_key], entry[down_key] = fallback[up_key], fallback[down_key]
        entry[flag] = True
    entry.update({"n_eff": n_eff(w, w2), "yield": w / lumi, "sumw2": w2 / lumi ** 2,
                  f"yield_{up_key}": up[0] / lumi, f"yield_{down_key}": down[0] / lumi,
                  f"sumw2_{up_key}": up[1] / lumi ** 2, f"sumw2_{down_key}": down[1] / lumi ** 2})
    return entry


def theory_ratios(nom, members, pdf_method):
    """theory_syst._ratios_from_sums on generator-normalised member sums."""
    r = {k: np.asarray(v[0], dtype=float) / nom for k, v in members.items() if v is not None}
    member_ratios, central = r["pdf"][1:101], float(r["pdf"][0])
    if pdf_method == "hessian_symmetric":
        sigma_pdf, centre = float(np.sqrt(np.sum((member_ratios - central) ** 2))), central
    else:
        srt = np.sort(member_ratios)
        lo, hi = float(srt[15]), float(srt[83])
        centre, sigma_pdf = 0.5 * (hi + lo), 0.5 * (hi - lo)
    sigma_as = 0.5 * abs(float(r["alphas"][1]) - float(r["alphas"][0])) if "alphas" in r else 0.0
    sigma = float(np.sqrt(sigma_pdf ** 2 + sigma_as ** 2))
    scale = r["scale"][SCALE_VALID]
    ps = r["ps"]
    return {"generator_normalised": True, "pdf_method": pdf_method, "pdf_alphas_unc": float(sigma_as),
            "alphas_available": "alphas" in r, "pdf_up": centre + sigma, "pdf_down": centre - sigma,
            "pdf_central": centre, "scale_up": float(scale.max()), "scale_down": float(scale.min()),
            "ps_isr_up": float(ps[2]), "ps_isr_down": float(ps[0]), "ps_fsr_up": float(ps[3]),
            "ps_fsr_down": float(ps[1])}


def theory_entry(nom, members, lumi, pdf_method, fallback=None, flag=None, min_events=0):
    """theory_syst._entry: nominal (sum w, sum w^2) and the member sums {pdf, alphas or None, scale, ps}
    of one region -> its entry."""
    w, w2 = nom
    pairs = [nom]
    for key, valid in (("pdf", slice(None)), ("scale", SCALE_VALID), ("ps", slice(None)), ("alphas", slice(None))):
        if members.get(key) is not None:
            pairs += list(zip(members[key][0][valid], members[key][1][valid]))
    if w > 0.0 and (fallback is None or _enough(pairs, min_events)):
        entry = {"n_events": int(round(n_eff(w, w2))), "nom_sum": w, **theory_ratios(w, members, pdf_method)}
    else:
        entry = {"n_events": int(round(n_eff(w, w2))), "nom_sum": w, **{k: fallback[k] for k in RATIO_FIELDS},
                 flag: True}
    entry.update({"n_eff": n_eff(w, w2), "yield": w / lumi, "sumw2": w2 / lumi ** 2})
    for key in ("pdf", "scale", "ps", "alphas"):
        if members.get(key) is None:
            continue
        entry[f"yield_{key}"] = (np.asarray(members[key][0]) / lumi).tolist()
        entry[f"sumw2_{key}"] = (np.asarray(members[key][1]) / lumi ** 2).tolist()
    return entry


# --- per sample and category ------------------------------------------------------


@dataclass
class SystConfig:
    """lumi: fb^-1 the histograms are normalised to; window: the msoftdrop_1 threshold (low, high);
    mass_fit: (range, width) for the mass-binned entries or None (legacy production: None)."""

    lumi: float
    window: tuple = (50.0, 150.0)
    min_region_events: int = 10
    pdf_method: str = "hessian_symmetric"
    mass_fit: tuple | None = None

    def __post_init__(self):
        if self.pdf_method not in PDF_METHODS:
            raise ValueError(f"pdf_method must be one of {PDF_METHODS}, got {self.pdf_method!r}")
        if not self.lumi > 0:
            raise ValueError("lumi must be positive")


def _levels(cell, sr_ids, cfg):
    """Region / bin index sets: inclusive, per SR window, and with mass_fit per SR fit range and fit bin."""
    win = window_bins(cell.edges, cfg.window)
    all_regions = np.arange(len(sr_ids) + 1)
    out = {"inclusive": (all_regions, win), "windows": [(np.array([k + 1]), win) for k in range(len(sr_ids))]}
    if cfg.mass_fit is not None:
        fit, groups = mass_fit_bins(cell.edges, *cfg.mass_fit)
        rng = np.concatenate(groups)
        out["fit_edges"] = fit
        out["mass"] = [((np.array([k + 1]), rng), [(np.array([k + 1]), g) for g in groups]) for k in range(len(sr_ids))]
    return out


def _sample_entry(make, levels, sr_ids, csv_hash, cfg, mass_branch=MASS_BRANCH):
    """Legacy top-level entry of one sample and category from make(level, fallback, flag) -> entry."""
    incl = make(levels["inclusive"], None, None)
    entry = dict(incl)
    regions = {}
    for k, bid in enumerate(sr_ids):
        regions[str(bid)] = make(levels["windows"][k], incl, "fallback_to_inclusive")
        if "mass" in levels:
            rng, bins = levels["mass"][k]
            mr = make(rng, incl, "fallback_to_inclusive")
            regions[str(bid)]["mass_range"] = mr
            regions[str(bid)]["mass_bins"] = [make(b, mr, "fallback_to_mass_range") for b in bins]
    entry["regions"] = regions
    entry["sr_csv_hash"] = csv_hash
    if "mass" in levels:
        entry["mass_fit"] = {"branch": mass_branch, "edges": [float(e) for e in levels["fit_edges"]]}
    entry["min_region_events"] = cfg.min_region_events
    return entry


def category_entries(cell, sr_ids, csv_hash, cfg, systs=SYSTS, has_theory=False, where=""):
    """{syst: entry} of one sample and category (no entry without a positive inclusive nominal yield,
    as legacy; theory only for samples with theory weights)."""
    if not cell.has("nominal"):
        return {}
    levels = _levels(cell, sr_ids, cfg)
    if not cell.total("nominal", *levels["inclusive"])[0] > 0.0:
        return {}
    out = {}
    for syst in systs:
        if syst == "theory":
            if not has_theory:
                continue
            missing = [MEMBER_SETS[k] for k in ("pdf", "scale", "ps") if not cell.has(k)]
            if missing:
                raise ValueError(f"{where}: a theory-weight sample without the histograms {missing}")

            def make(level, fallback, flag):
                members = {k: (cell.total(k, *level) if cell.has(k) else None) for k in ("pdf", "scale", "ps", "alphas")}
                for k, v in members.items():
                    if v is not None and len(v[0]) != N_MEMBERS[k]:
                        raise ValueError(f"{where}: {MEMBER_SETS[k]} has {len(v[0])} members, expected {N_MEMBERS[k]}")
                return theory_entry(cell.total("nominal", *level), members, cfg.lumi, cfg.pdf_method, fallback, flag,
                                    cfg.min_region_events)
        else:
            src, up_key, down_key = ("pileup", "pu_up", "pu_down") if syst == "pileup" else (syst, f"{syst}_up",
                                                                                          f"{syst}_down")
            names = (f"{src}_up", f"{src}_down")
            missing = [n for n in names if not cell.has(n)]
            if missing:
                raise ValueError(f"{where}: no {missing} histograms (vvv run with --no-jet-variations?)")

            def make(level, fallback, flag, names=names, up_key=up_key, down_key=down_key):
                return updown_entry(syst, cell.total("nominal", *level), cell.total(names[0], *level),
                                    cell.total(names[1], *level), cfg.lumi, up_key, down_key, fallback, flag,
                                    cfg.min_region_events)
        out[syst] = _sample_entry(make, levels, sr_ids, csv_hash, cfg)
    return out


def build(outputs, categories, cfg, catalogue, samples=None, systs=SYSTS, log=print):
    """{syst: {sample: {category: entry}}} for the MC samples of the histograms (or `samples`).

    categories: {category: (signal_region.csv, msoftdrop_1 window)}."""
    datasets = outputs.datasets()
    mc = [d for d in datasets if d in catalogue and catalogue[d].is_mc]
    unknown = sorted(set(datasets) - set(catalogue))
    if unknown:
        raise ValueError(f"datasets not in the sample catalogue: {unknown}")
    if samples:
        missing = sorted(set(samples) - set(mc))
        if missing:
            raise ValueError(f"samples without histograms: {missing}")
        mc = [s for s in mc if s in set(samples)]
    result = {s: {} for s in systs}
    for category, (csv_path, window) in categories.items():
        outputs.check_signal_regions(category, csv_path)
        sr_ids, csv_hash = sr_bin_ids(csv_path), sr_csv_hash(csv_path)
        ccfg = SystConfig(cfg.lumi, tuple(window), cfg.min_region_events, cfg.pdf_method, cfg.mass_fit)
        n = 0
        for sample in mc:
            cell = CellHists.from_outputs(outputs, sample, category, sr_ids)
            entries = category_entries(cell, sr_ids, csv_hash, ccfg, systs, catalogue[sample].has_theory_weights,
                                       f"{sample}/{category}")
            for syst, entry in entries.items():
                result[syst].setdefault(sample, {})[category] = entry
            n += bool(entries)
        log(f"  {category}: {len(sr_ids)} SRs {sr_ids} (SR CSV hash {csv_hash}), {n} of {len(mc)} MC samples with "
            f"a positive inclusive yield")
    return result


def summarise(result, syst, log=print):
    """Logs the inclusive and per-SR ratios and the fallbacks of one systematic."""
    keys = ("pdf_up", "pdf_down", "scale_up", "scale_down") if syst == "theory" else \
        (("pu_up", "pu_down") if syst == "pileup" else (f"{syst}_up", f"{syst}_down"))
    n_cells = n_fb = 0
    for sample, cats in result.items():
        for cat, e in cats.items():
            for r in e["regions"].values():
                n_cells += 1
                n_fb += bool(r.get("fallback_to_inclusive"))
    log(f"  {syst}: {len(result)} samples, {n_cells} SR entries, {n_fb} take the inclusive ratios "
        f"(n_eff < min_region_events); ratio keys {', '.join(keys)}")


def write(result, out_dir, log=print):
    """<syst>_syst_yields.json (legacy layout {sample: {category: entry}}) and the legacy flat CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for syst, nested in result.items():
        path = out_dir / f"{output_name(syst)}.json"
        path.write_text(json.dumps(nested, indent=2, default=float))
        with open(out_dir / f"{output_name(syst)}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            if syst == "theory":
                w.writerow(["sample", "tree", "region", "systematic", "up_ratio", "down_ratio"])
                for sample in sorted(nested):
                    for tree in sorted(nested[sample]):
                        e = nested[sample][tree]
                        for region, rr in [("inclusive", e), *e["regions"].items()]:
                            for name, up, down in THEORY_CSV:
                                w.writerow([sample, tree, region, name, f"{rr[up]:.6f}", f"{rr[down]:.6f}"])
            else:
                up, down = ("pu_up", "pu_down") if syst == "pileup" else (f"{syst}_up", f"{syst}_down")
                w.writerow(["sample", "tree", "region", up, down, "n_events"])
                for sample in sorted(nested):
                    for tree in sorted(nested[sample]):
                        e = nested[sample][tree]
                        for region, rr in [("inclusive", e), *e["regions"].items()]:
                            w.writerow([sample, tree, region, f"{rr[up]:.6f}", f"{rr[down]:.6f}", rr["n_events"]])
        paths[syst] = path
        log(f"Wrote {path}")
    return paths
