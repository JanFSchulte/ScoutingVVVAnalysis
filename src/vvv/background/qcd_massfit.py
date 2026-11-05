"""QCD mass-fit inputs (port of legacy background_estimation/qcd_est.py, abcd_mode "massfit").

qcd_est.py does not fit data: it prepares the inputs of the pass/fail fit that combine.C builds
(writeMassfitCard). With msoftdrop_1 binned over the fit range (20-250 GeV in 10 GeV bins):

- every event passing every threshold but the msoftdrop_1 window is "pass" (inside SR box i,
  [low, high) per score axis, an upper edge at 1 including 1) or "fail" (outside every box), at any
  msoftdrop_1 in the fit range: sum w, sum w^2 and the MC entry count per sample, class and the QCD
  sum, [region, mass bin] (negative sample / class bins, from signed generator weights, set to 0);
- TF_MC = exp(c0 + c1 x + ... + ck x^k): fitted to the QCD MC ratio R = A / F of the SR union over
  the fail region, x the bin centre mapped to [-1, 1] over the bins with F > 0 (weighted least
  squares of ln R, then Gauss-Newton on the chi2 of R); stored are x, shape = exp(c1 x + ...),
  the decorrelated exponent shifts sqrt(lambda_e) v_e . (x, x^2, ...) of the covariance of
  c1..ck, chi2 and ndf;
- n0_i = QCD MC in SR i / sum (F x shape) over the bins with F > 0 (the nominal per-SR
  normalisation); combine models the pass-bin QCD as qcdF_j x n_i x TF_MC shape x a Bernstein
  residual with free fail yields qcdF_j;
- smooth_templates {class: f}: per SR the mass shape of the class's events in the SR box loosened
  by f (lower bounds / f, upper bounds x f capped at 1) and outside the other SR boxes;
- data (optional): counts like the MC, the pass bins overlapping the blind window never filled;
- one-bin SR bundles (the A region: SR box and msoftdrop_1 inside its threshold window) per
  sample, class, qcd_true and totals, with qcd_predict = the nominal model n0_i x F x shape in the
  window bins (or, with qcd_from_mc, the MC QCD of the A region).

The inputs are binned sums (MassfitInputs), filled from events (MassfitAccumulator: per sample,
so a category never has to fit in memory at once) or from a coffea mass histogram
(MassfitInputs.from_hist). estimate() returns the legacy quantities and write_root() the file
combine.C reads (qcd_abcd_yields.root layout).
"""

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from vvv.ml.inference import threshold_mask

ONE_BIN = np.array([0.0, 1.0])


def slugify(text):
    """Legacy _slugify: lower-case alphanumerics, everything else '_' (group directory names)."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


# --- config -----------------------------------------------------------------


@dataclass
class MassfitConfig:
    """qcd_est.json massfit[tree] (defaults as legacy)."""

    range: tuple = (20.0, 250.0)
    width: float = 10.0
    tfmc_order: int = 2
    residual_order: int = 1
    tfmc_free: bool = True
    bern_range: tuple = (0.0, 10.0)
    qcd_from_mc: bool = False
    sr_slopes: list = field(default_factory=list)
    smooth_templates: dict = field(default_factory=dict)
    blind_window: tuple | None = None

    @classmethod
    def from_dict(cls, d, default_blind_window=None):
        rng, width = d.get("range"), d.get("width")
        if (not isinstance(rng, list) or len(rng) != 2 or not rng[0] < rng[1]
                or not isinstance(width, (int, float)) or not width > 0):
            raise ValueError(f"massfit needs range [low, high] with low < high and width > 0, got {d}")
        bern = d.get("bern_range", [0.0, 10.0])
        slopes = d.get("sr_slopes", [])
        if (not isinstance(slopes, list) or len(set(slopes)) != len(slopes)
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in slopes)):
            raise ValueError("massfit sr_slopes must be a list of distinct SR bin indices")
        smooth = d.get("smooth_templates", {})
        if not isinstance(smooth, dict) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and v > 1 for v in smooth.values()):
            raise ValueError("massfit smooth_templates must map class names to loosening factors > 1")
        blind = d.get("blind_window", default_blind_window)
        blind_ok = (isinstance(blind, (list, tuple)) and len(blind) == 2
                    and all(isinstance(v, (int, float)) for v in blind) and blind[0] < blind[1])
        out = cls((float(rng[0]), float(rng[1])), float(width), int(d.get("tfmc_order", 2)),
                  int(d.get("residual_order", 1)), bool(d.get("tfmc_free", True)),
                  (float(bern[0]), float(bern[1])), bool(d.get("qcd_from_mc", False)), sorted(slopes),
                  {str(k): float(v) for k, v in smooth.items()},
                  (float(blind[0]), float(blind[1])) if blind_ok else None)
        if out.tfmc_order < 0 or out.residual_order < 0 or not out.bern_range[0] < out.bern_range[1]:
            raise ValueError("massfit orders must be >= 0 and bern_range [low, high] with low < high")
        if out.qcd_from_mc and out.sr_slopes:
            raise ValueError("massfit sr_slopes need the QCD fit (qcd_from_mc is set)")
        out.edges  # validates the binning
        return out

    @property
    def edges(self):
        lo, hi = self.range
        e = np.arange(lo, hi + 0.5 * self.width, self.width)
        if len(e) < 3 or abs(e[-1] - hi) > 1e-9 * self.width:
            raise ValueError("massfit width must divide the range into at least two bins")
        return e


# --- signal regions ---------------------------------------------------------


def load_sr_boxes(csv_path, class_names):
    """Legacy signal_region.csv as [(bin_index, {axis: (low, high)})] sorted by bin_index (axes:
    the <class>_low/_high column pairs in column order; none: every class but the last)."""
    import csv

    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"signal region CSV is empty: {csv_path}")
    axes = [c[:-4] for c in cols if c.endswith("_low") and f"{c[:-4]}_high" in cols]
    for a in axes:
        if a not in class_names:
            raise KeyError(f"signal region axis {a!r} is not a class: {class_names}")
    axes = axes or list(class_names[:max(1, len(class_names) - 1)])
    ids = [float(r["bin_index"]) for r in rows]
    if any(not math.isfinite(v) or v != round(v) or v <= 0 for v in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("signal region bin_index values must be unique positive integers")
    boxes = [(int(round(float(r["bin_index"]))), {a: (float(r[f"{a}_low"]), float(r[f"{a}_high"])) for a in axes})
             for r in rows]
    return sorted(boxes, key=lambda b: b[0])


def sr_csv_hash(csv_path):
    """metadata/sr_csv_hash: the first 13 hex digits of the SHA-256 of the CSV bytes (exact in a double)."""
    with open(csv_path, "rb") as fh:
        return float(int(hashlib.sha256(fh.read()).hexdigest()[:13], 16))


def box_mask(proba, bounds, class_names, factor=1.0):
    """Events in a box (legacy _region_mask; factor > 1: _loosened_region_mask, lower bounds / factor,
    upper bounds x factor capped at 1, an open upper bound staying open)."""
    mask = np.ones(len(proba), dtype=bool)
    for axis, (low, high) in bounds.items():
        s = proba[:, class_names.index(axis)]
        low = low / factor
        if factor != 1.0 and high < 1.0 - 1e-12:
            high = min(high * factor, 1.0)
        mask &= (s >= low) & (s < high) if high < 1.0 - 1e-12 else s >= low
    return mask


def assign_regions(proba, boxes, class_names):
    """Region per event: 0 = outside every SR box, i + 1 = inside box i (boxes order); overlaps are an error."""
    region = np.zeros(len(proba), dtype=np.int64)
    member = np.zeros(len(proba), dtype=np.int64)
    for i, (_, bounds) in enumerate(boxes):
        m = box_mask(proba, bounds, class_names)
        region[m] = i + 1
        member += m
    if np.any(member > 1):
        raise RuntimeError("signal region definitions overlap on the current event set")
    return region


def mass_bin(values, edges):
    """Legacy _massfit_bin_index: np.histogram semantics (the last bin includes its upper edge), -1 outside."""
    values = np.asarray(values, dtype=np.float64)
    idx = np.searchsorted(edges, values, side="right") - 1
    idx[values == edges[-1]] = len(edges) - 2
    idx[~np.isfinite(values) | (values < edges[0]) | (values > edges[-1])] = -1
    return idx


# --- binned inputs ----------------------------------------------------------


@dataclass
class MassfitInputs:
    """Binned inputs of one category.

    sums[sample] = (sw, sw2, n), arrays [region, mass bin] (region 0 = fail, i + 1 = SR i in sr_ids
    order), raw (negative bins kept); a_sums[sample] = (sw, sw2, n) [SR] in the A region (SR box and
    the mass threshold window); abcd[sample] = [[sw, sw2] of the A union, B, C, D] (the legacy ABCD
    totals, informational); abs_w[sample] = |w| of the events inside the mass window at any score
    (for the median event weight); smooth[class] = (sw, sw2, n) [SR, mass bin]; data = counts
    [region, mass bin] or None; blinded [mass bin] (pass bins never filled with data)."""

    edges: np.ndarray
    sr_ids: list
    class_names: list
    sample_class: dict
    qcd_classes: list
    sums: dict
    a_sums: dict
    abcd: dict
    abs_w: dict
    smooth: dict = field(default_factory=dict)
    data: np.ndarray | None = None
    blinded: np.ndarray | None = None
    approximations: list = field(default_factory=list)

    @property
    def samples(self):
        return sorted(self.sample_class)

    @classmethod
    def from_hist(cls, h, sample_class, sr_ids, class_names, qcd_classes, window=(50.0, 150.0), category=None,
                  syst="nominal", data_datasets=(), blind_window=None, smooth=None, log=print):
        """Inputs from a coffea mass histogram (axes dataset, [category], region = SR bin_index with 0 =
        fail, [syst], mass; Weight storage). Approximations against the event-level inputs, recorded in
        .approximations: the A region is the mass bins inside the window (events exactly on a window
        edge differ), MC entry counts are the effective counts sw^2 / sw2, the median |w| is replaced by
        the effective weight sw2 / sw over the window bins, there are no ABCD totals, and smoothed
        templates need `smooth` ({class: (sw, sw2, n) [SR, mass bin]}) from elsewhere."""
        names = [a.name for a in h.axes]
        edges = np.asarray(h.axes["mass"].edges, dtype=float)
        sel = {}
        if category is not None and "category" in names:
            sel["category"] = category
        if "syst" in names:
            sel["syst"] = syst
        hh = h[sel] if sel else h
        regions = [0] + list(sr_ids)
        centres = 0.5 * (edges[:-1] + edges[1:])
        in_win = (centres > window[0]) & (centres < window[1])
        n_sr, n_bins = len(sr_ids), len(edges) - 1
        datasets = list(hh.axes["dataset"])
        sums, a_sums, abcd, abs_w = {}, {}, {}, {}

        def binned(ds):
            sw, sw2 = np.zeros((n_sr + 1, n_bins)), np.zeros((n_sr + 1, n_bins))
            present = set(hh.axes["region"])
            for r, reg in enumerate(regions):
                if ds in datasets and reg in present:
                    v = hh[{"dataset": ds, "region": hist_loc(reg)}]
                    sw[r], sw2[r] = v.values(), v.variances()
            return sw, sw2

        for sample in sample_class:
            sw, sw2 = binned(sample)
            n = np.where(sw2 > 0, sw ** 2 / np.where(sw2 > 0, sw2, 1.0), 0.0)
            sums[sample] = (sw, sw2, n)
            a_sums[sample] = (sw[1:, in_win].sum(axis=1), sw2[1:, in_win].sum(axis=1), n[1:, in_win].sum(axis=1))
            tot, tot2 = sw[:, in_win].sum(), sw2[:, in_win].sum()
            abs_w[sample] = np.array([tot2 / tot]) if tot > 0 else np.zeros(0)
            abcd[sample] = np.zeros((4, 2))
        data = blinded = None
        if data_datasets:
            if blind_window is None:
                raise ValueError("data needs a blind window")
            blinded = (edges[:-1] < blind_window[1]) & (edges[1:] > blind_window[0])
            data = sum(binned(ds)[0] for ds in data_datasets)
            data[1:, blinded] = 0.0
        approx = ["A region from mass bins", "effective MC entry counts", "effective event weight for the median",
                  "no ABCD totals"]
        log(f"Mass-fit inputs from a histogram: {len(sample_class)} samples, {n_sr} SRs, {n_bins} mass bins "
            f"(approximations: {', '.join(approx)})")
        return cls(edges, list(sr_ids), list(class_names), dict(sample_class), list(qcd_classes), sums, a_sums,
                   abcd, abs_w, dict(smooth or {}), data, blinded, approx)


def hist_loc(value):
    import hist

    return hist.loc(value)


class MassfitAccumulator:
    """Event-level filling of MassfitInputs, one sample (or chunk) at a time.

    window: the mass threshold condition of the A region (legacy selection.json, e.g. [50, 150]:
    50 < m < 150). Events are expected to pass every other threshold already."""

    def __init__(self, cfg, boxes, class_names, sample_class, qcd_classes, window, log=print):
        self.cfg, self.boxes, self.class_names = cfg, boxes, list(class_names)
        self.window = tuple(window) if isinstance(window, list) else window
        self.edges = cfg.edges
        n_sr, n_bins = len(boxes), len(self.edges) - 1
        self.shape = (n_sr + 1, n_bins)
        self.sample_class = dict(sample_class)
        self.qcd_classes = list(qcd_classes)
        z = lambda *s: np.zeros(s)  # noqa: E731
        self.sums = {s: (z(*self.shape), z(*self.shape), z(*self.shape)) for s in sample_class}
        self.a_sums = {s: (z(n_sr), z(n_sr), z(n_sr)) for s in sample_class}
        self.abcd = {s: z(4, 2) for s in sample_class}
        self.abs_w = {s: [] for s in sample_class}
        self.smooth = {c: (z(n_sr, n_bins), z(n_sr, n_bins), z(n_sr, n_bins)) for c in cfg.smooth_templates}
        self.data = None
        self.blinded = None
        self.log = log

    def _window_masks(self, mass):
        """Legacy _abcd_pass_fail_masks: valid (finite, not a sentinel) and inside / outside the window."""
        mass = np.asarray(mass, dtype=np.float64)
        valid = np.isfinite(mass) & ~(mass < -990)
        inside = threshold_mask({"m": mass}, {"m": self.window}) if self.window is not None else valid
        return valid & inside, valid & ~inside

    def fill(self, sample, mass, weight, proba=None, region=None):
        """Add MC events of one sample: msoftdrop_1, weight (x lumi), and the class probabilities
        (needed for smooth_templates) or the region (0 = fail, i + 1 = SR i)."""
        cls = self.sample_class[sample]
        mass = np.asarray(mass, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)
        if region is None:
            region = assign_regions(proba, self.boxes, self.class_names)
        region = np.asarray(region, dtype=np.int64)
        n_sr, n_bins = len(self.boxes), self.shape[1]
        mbin = mass_bin(mass, self.edges)
        ok = mbin >= 0
        flat = region[ok] * n_bins + mbin[ok]
        size = (n_sr + 1) * n_bins
        sw, sw2, n = self.sums[sample]
        w = weight[ok]
        sw += np.bincount(flat, weights=w, minlength=size).reshape(self.shape)
        sw2 += np.bincount(flat, weights=w ** 2, minlength=size).reshape(self.shape)
        n += np.bincount(flat, minlength=size).reshape(self.shape)

        passed, failed = self._window_masks(mass)
        a_sw, a_sw2, a_n = self.a_sums[sample]
        in_a = passed & (region > 0)
        r = region[in_a] - 1
        a_sw += np.bincount(r, weights=weight[in_a], minlength=n_sr)
        a_sw2 += np.bincount(r, weights=weight[in_a] ** 2, minlength=n_sr)
        a_n += np.bincount(r, minlength=n_sr)
        union = region > 0
        for k, m in enumerate((union & passed, ~union & passed, union & failed, ~union & failed)):
            self.abcd[sample][k] += (weight[m].sum(), (weight[m] ** 2).sum())
        self.abs_w[sample].append(np.abs(weight[passed]))

        if cls in self.smooth:
            if proba is None:
                raise ValueError(f"smooth_templates of {cls} need the class probabilities")
            ssw, ssw2, sn = self.smooth[cls]
            for i, (_, bounds) in enumerate(self.boxes):
                sel = ok & box_mask(proba, bounds, self.class_names, self.cfg.smooth_templates[cls]) \
                    & ((region == 0) | (region == i + 1))
                ssw[i] += np.bincount(mbin[sel], weights=weight[sel], minlength=n_bins)
                ssw2[i] += np.bincount(mbin[sel], weights=weight[sel] ** 2, minlength=n_bins)
                sn[i] += np.bincount(mbin[sel], minlength=n_bins)

    def fill_data(self, mass, proba=None, region=None):
        """Add data events (legacy _massfit_data): events in blinded pass bins are dropped."""
        if self.cfg.blind_window is None:
            raise ValueError("data needs massfit blind_window (or a mass threshold window)")
        if region is None:
            region = assign_regions(proba, self.boxes, self.class_names)
        region = np.asarray(region, dtype=np.int64)
        low, high = self.cfg.blind_window
        self.blinded = (self.edges[:-1] < high) & (self.edges[1:] > low)
        mbin = mass_bin(mass, self.edges)
        keep = (mbin >= 0) & ~((region > 0) & self.blinded[np.maximum(mbin, 0)])
        n_bins = self.shape[1]
        counts = np.bincount(region[keep] * n_bins + mbin[keep], minlength=self.shape[0] * n_bins)
        self.data = (0 if self.data is None else self.data) + counts.reshape(self.shape).astype(float)

    def inputs(self):
        return MassfitInputs(self.edges, [b[0] for b in self.boxes], self.class_names, self.sample_class,
                             self.qcd_classes, self.sums, self.a_sums, self.abcd,
                             {s: np.concatenate(v) if v else np.zeros(0) for s, v in self.abs_w.items()},
                             self.smooth, self.data, self.blinded)


# --- the estimate -----------------------------------------------------------


def fit_tfmc(A, A2, F, F2, edges, order):
    """Legacy _massfit_tfmc: TF_MC = exp(c0 + c1 x + ... + ck x^k) fitted to R = A / F."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    used = F > 0
    if used.sum() < 2:
        raise RuntimeError("massfit: fewer than two mass bins with QCD MC in the fail region")
    lo, hi = float(centres[used].min()), float(centres[used].max())
    x = 2.0 * (centres - lo) / (hi - lo) - 1.0
    Fs = np.where(used, F, 1.0)
    r = A / Fs
    var = A2 / Fs ** 2 + r ** 2 * F2 / Fs ** 2
    ok = used & (A > 0) & (var > 0)
    if ok.sum() <= order + 1:
        raise RuntimeError(f"massfit: TF_MC of order {order} needs more than {order + 1} bins with QCD MC in the "
                           f"SR boxes, found {int(ok.sum())}")
    X = np.vander(x[ok], order + 1, increasing=True)
    rv, vv = r[ok], var[ok]
    wl = rv ** 2 / vv
    beta = np.linalg.lstsq(X * np.sqrt(wl)[:, None], np.log(rv) * np.sqrt(wl), rcond=None)[0]
    for _ in range(50):
        model = np.exp(X @ beta)
        J = (model / np.sqrt(vv))[:, None] * X
        step = np.linalg.lstsq(J, (rv - model) / np.sqrt(vv), rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    model = np.exp(X @ beta)
    J = (model / np.sqrt(vv))[:, None] * X
    chi2 = float(np.sum((rv - model) ** 2 / vv))
    cov = np.linalg.inv(J.T @ J)
    X_all = np.vander(x, order + 1, increasing=True)[:, 1:]
    shape = np.exp(X_all @ beta[1:]) if order > 0 else np.ones_like(F)
    eig = []
    if order > 0:
        lam, vec = np.linalg.eigh(cov[1:, 1:])
        eig = [X_all @ (math.sqrt(lam[e]) * vec[:, e]) for e in range(len(lam)) if lam[e] > 0]
    return {"x": x, "shape": shape, "eig": eig, "coefficients": beta, "covariance": cov, "chi2": chi2,
            "ndf": int(ok.sum() - order - 1)}


def _clamp(values, label, log):
    neg = values < 0.0
    if np.any(neg):
        log(f"[WARN] massfit '{label}': {int(neg.sum())} negative mass bins (signed generator weights) set to 0")
    return np.where(neg, 0.0, values)


def _median(arrays):
    a = np.concatenate([np.asarray(x) for x in arrays]) if arrays else np.zeros(0)
    return float(np.median(a)) if a.size else 0.0


def estimate(inputs, cfg, window=(50.0, 150.0), log=print):
    """The legacy qcd_est massfit quantities from binned inputs: mass-binned bundles (samples, groups,
    qcd_true, smooth, data), the TF_MC fit and n0 (unless qcd_from_mc), and the one-bin SR bundles
    (samples, groups, qcd_predict, qcd_true, total_predict, total_true)."""
    edges, sr_ids, classes = inputs.edges, inputs.sr_ids, inputs.class_names
    if not np.allclose(edges, cfg.edges):
        raise ValueError(f"input mass bins {edges} differ from the config's {cfg.edges}")
    n_sr = len(sr_ids)
    unknown = sorted(set(cfg.sr_slopes) - set(sr_ids))
    if unknown:
        raise RuntimeError(f"massfit sr_slopes names SRs not in the SR CSV: {unknown}")
    for c in cfg.smooth_templates:
        if c not in classes or c in inputs.qcd_classes:
            raise RuntimeError(f"massfit smooth_templates: '{c}' is not a non-QCD background class")
        if c not in inputs.smooth:
            raise RuntimeError(f"massfit smooth_templates: no smoothed inputs for '{c}'")
    samples = inputs.samples
    by_class = {c: [s for s in samples if inputs.sample_class[s] == c] for c in classes}
    qcd_samples = [s for s in samples if inputs.sample_class[s] in inputs.qcd_classes]

    def add(names, src):
        return tuple(sum((src[s][k] for s in names), np.zeros_like(src[samples[0]][k])) for k in range(3))

    groups = {c: add(by_class[c], inputs.sums) for c in classes}
    q_sw, q_sw2, q_n = add(qcd_samples, inputs.sums)
    F, F2 = q_sw[0], q_sw2[0]
    A, A2 = q_sw[1:].sum(axis=0), q_sw2[1:].sum(axis=0)
    used = F > 0
    centres = 0.5 * (edges[:-1] + edges[1:])
    in_window = threshold_mask({"c": centres}, {"c": tuple(window)})
    tf, n0, window_prediction = None, np.zeros(n_sr), None
    if not cfg.qcd_from_mc:
        tf = fit_tfmc(A, A2, F, F2, edges, cfg.tfmc_order)
        for i in range(n_sr):
            qcd_sr = float(q_sw[i + 1][used].sum())
            if qcd_sr <= 0.0:
                raise RuntimeError(f"massfit: SR {i + 1} holds no QCD MC in the fit range, its normalisation is undefined")
            n0[i] = qcd_sr / float((F * tf["shape"])[used].sum())
        window_prediction = np.array([float((n0[i] * F * tf["shape"])[in_window].sum()) for i in range(n_sr)])
        log(f"massfit: {len(centres)} bins of {edges[1] - edges[0]:g} in {edges[0]:g}-{edges[-1]:g}, "
            f"{int(used.sum())} with fail-region QCD ({float(F.sum()):.6g}); TF_MC order {cfg.tfmc_order}: "
            f"chi2/ndf = {tf['chi2']:.2f}/{tf['ndf']}, {len(tf['eig'])} shape eigen-directions")
    else:
        log("massfit: QCD from MC (qcd_from_mc, no TF_MC fit): QCD MC in the SR boxes "
            + ", ".join(f"SR{sr_ids[i]} {float(q_sw[i + 1].sum()):.6g} ({int(q_n[i + 1].sum())} MC events)"
                        for i in range(n_sr)) + f", fail region {float(F.sum()):.6g}")
    smooth = {}
    for c in cfg.smooth_templates:
        sw, sw2, n = inputs.smooth[c]
        smooth[c] = (_clamp(sw, f"smooth/{c}", log), sw2, n)

    # one-bin SR bundles (A region)
    def a_bundle(names):
        sw, sw2, n = add(names, inputs.a_sums)
        return {"yield": sw, "var": sw2, "n_entries": n, "mean_event_weight": _median([inputs.abs_w[s] for s in names])}

    sample_b = {s: a_bundle([s]) for s in samples}
    group_b = {c: a_bundle(by_class[c]) for c in classes}
    qcd_true = a_bundle(qcd_samples)
    all_b = a_bundle(samples)
    abcd = sum((inputs.abcd[s] for s in qcd_samples), np.zeros((4, 2)))
    qa, qb, qc, qd = abcd[:, 0]
    if np.any(abcd) and (qb <= 0.0 or qd <= 0.0):
        log("[WARN] QCD B or D total is not positive (legacy qcd_est raises here; informational only in massfit mode)")
    qcd_a_total = float(qcd_true["yield"].sum())  # = the QCD of the A union (the SR boxes are disjoint)
    pred_qcd = qcd_true["yield"].copy() if cfg.qcd_from_mc else window_prediction
    qcd_scale = float(pred_qcd.sum() / qcd_a_total) if qcd_a_total > 0.0 else 0.0
    non_qcd = [c for c in classes if c not in inputs.qcd_classes]
    zeros = np.zeros(n_sr)
    pred_total = sum((group_b[c]["yield"] for c in non_qcd), zeros) + pred_qcd
    pred_total_var = sum((group_b[c]["var"] for c in non_qcd), zeros)
    bundles = {f"samples/{s}": sample_b[s] for s in samples}
    bundles.update({f"groups/{slugify(c)}": group_b[c] for c in classes})
    bundles["qcd_predict"] = {"yield": pred_qcd, "var": zeros, "n_entries": qcd_true["n_entries"],
                              "mean_event_weight": qcd_scale * qcd_true["mean_event_weight"],
                              "covariance": np.zeros((n_sr, n_sr)), "abcd_scale": qcd_scale, "abcd_scale_error": 0.0}
    bundles["qcd_true"] = qcd_true
    bundles["total_predict"] = {"yield": pred_total, "var": pred_total_var, "n_entries": all_b["n_entries"],
                                "mean_event_weight": all_b["mean_event_weight"]}
    bundles["total_true"] = {"yield": sum((group_b[c]["yield"] for c in classes), zeros),
                             "var": sum((group_b[c]["var"] for c in classes), zeros),
                             "n_entries": all_b["n_entries"], "mean_event_weight": all_b["mean_event_weight"]}
    raw_union = qb * qc / qd if qd > 0 else float("nan")
    return {
        "edges": edges, "sr_ids": list(sr_ids), "class_names": list(classes), "cfg": cfg, "tf": tf, "n0": n0,
        "window_prediction": window_prediction, "in_window": in_window,
        "samples": {s: (_clamp(inputs.sums[s][0], f"samples/{s}", log),) + tuple(inputs.sums[s][1:]) for s in samples},
        "groups": {c: (_clamp(groups[c][0], f"groups/{c}", log),) + tuple(groups[c][1:]) for c in classes},
        "qcd_true": (q_sw, q_sw2, q_n), "smooth": smooth,
        "data": None if inputs.data is None else {"counts": inputs.data, "blinded": inputs.blinded},
        "bundles": bundles,
        "abcd": {"A_union": qa, "B": qb, "C": qc, "D": qd, "variances": abcd[:, 1].tolist(),
                 "raw_pred_union": raw_union, "raw_scale": raw_union / qa if qa > 0 else float("nan")},
        "approximations": list(inputs.approximations),
    }


def sideband_transfer_factors(result, signal=(50.0, 150.0), sidebands=((20.0, 50.0), (150.0, 250.0))):
    """Window / sideband ratios of the QCD (build/kfold_powheg/tf_check.py, the SR scan's transfer_factors):
    of the nominal TF_MC model sum_i n0_i x F x shape (None with qcd_from_mc) and of the MC-true QCD in
    the SR boxes; bins by centre, strictly inside each window."""
    edges = result["edges"]
    c = 0.5 * (edges[:-1] + edges[1:])

    def ratios(v):
        s = lambda lo, hi: float(v[(c > lo) & (c < hi)].sum())  # noqa: E731
        win = s(*signal)
        return [win / s(*sb) if s(*sb) > 0 else float("nan") for sb in sidebands]

    q_sw = result["qcd_true"][0]
    out = {"mc_true": ratios(q_sw[1:].sum(axis=0))}
    out["model"] = None if result["tf"] is None else ratios(result["n0"].sum() * q_sw[0] * result["tf"]["shape"])
    return out


# --- output -----------------------------------------------------------------


def write_root(path, result, csv_hash, log=print):
    """qcd_abcd_yields.root as legacy write_root_output (abcd_mode massfit): the file combine.C reads."""
    import uproot

    sr_ids = np.asarray(result["sr_ids"], dtype=np.int32)
    cfg, edges = result["cfg"], result["edges"]
    sr_edges = np.arange(len(sr_ids) + 1, dtype=float)
    with uproot.recreate(path) as f:
        f["metadata/signal_regions"] = {"bin_index": sr_ids}
        f["metadata/sr_csv_hash"] = (np.array([float(csv_hash)]), ONE_BIN)
        for prefix, b in result["bundles"].items():
            vals = np.asarray(b["yield"], dtype=float).copy()
            neg = vals < 0.0
            if np.any(neg):
                log(f"[WARN] ROOT bundle '{prefix}': negative SR yields {vals[neg].tolist()} (signed generator weights) "
                    f"written as 0")
                vals[neg] = 0.0
            var = np.asarray(b["var"], dtype=float)
            cov = b.get("covariance")
            cov = np.diag(np.maximum(var, 0.0)) if cov is None else np.asarray(cov, dtype=float)
            err = np.sqrt(np.maximum(var, 0.0))
            for i, sr in enumerate(sr_ids):
                p = f"{prefix}/sr{int(sr)}"
                f[f"{p}/yield"] = (np.array([vals[i]]), ONE_BIN)
                f[f"{p}/stat_error"] = (np.array([err[i]]), ONE_BIN)
                f[f"{p}/scale_error"] = (np.array([0.0]), ONE_BIN)
                f[f"{p}/n_entries"] = (np.array([float(b["n_entries"][i])]), ONE_BIN)
            f[f"{prefix}/covariance_total"] = (cov, sr_edges, sr_edges)
            f[f"{prefix}/mean_event_weight"] = (np.array([float(b["mean_event_weight"])]), ONE_BIN)
            if "abcd_scale" in b:
                f[f"{prefix}/abcd_scale"] = (np.array([float(b["abcd_scale"])]), ONE_BIN)
                f[f"{prefix}/abcd_scale_error"] = (np.array([float(b["abcd_scale_error"])]), ONE_BIN)

        tf = result["tf"]
        if cfg.qcd_from_mc:
            settings = {"enabled": 1.0, "qcd_from_mc": 1.0}
        else:
            settings = {"enabled": 1.0, "tfmc_order": cfg.tfmc_order, "residual_order": cfg.residual_order,
                        "tfmc_free": 1.0 if cfg.tfmc_free else 0.0, "bern_low": cfg.bern_range[0],
                        "bern_high": cfg.bern_range[1], "tfmc_chi2": tf["chi2"], "tfmc_ndf": tf["ndf"],
                        "n_eig": len(tf["eig"])}
        for key, value in settings.items():
            f[f"massfit/settings/{key}"] = (np.array([float(value)]), ONE_BIN)
        if tf is not None:
            f["massfit/tfmc/x"] = (np.asarray(tf["x"], dtype=float), edges)
            f["massfit/tfmc/shape"] = (np.asarray(tf["shape"], dtype=float), edges)
            for k, shift in enumerate(tf["eig"]):
                f[f"massfit/tfmc/eig{k}"] = (np.asarray(shift, dtype=float), edges)
            f["massfit/tfmc/n0"] = (np.asarray(result["n0"], dtype=float), sr_edges)
        if cfg.sr_slopes:
            f["massfit/sr_slope"] = (np.array([1.0 if int(v) in cfg.sr_slopes else 0.0 for v in sr_ids]), sr_edges)

        def write_mass(prefix, sw, sw2, n):
            for region, p in [(0, f"massfit/fail/{prefix}")] + [
                    (i + 1, f"massfit/pass/{prefix}/sr{int(s)}") for i, s in enumerate(sr_ids)]:
                f[f"{p}/yield"] = (np.asarray(sw[region], dtype=float), edges)
                f[f"{p}/sumw2"] = (np.asarray(sw2[region], dtype=float), edges)
                f[f"{p}/n_entries"] = (np.asarray(n[region], dtype=float), edges)

        for s in sorted(result["samples"]):
            write_mass(f"samples/{s}", *result["samples"][s])
        for c in result["class_names"]:
            write_mass(f"groups/{slugify(c)}", *result["groups"][c])
        write_mass("qcd_true", *result["qcd_true"])
        for c, (sw, sw2, n) in result["smooth"].items():
            base = f"massfit/smooth/{slugify(c)}"
            f[f"{base}/factor"] = (np.array([cfg.smooth_templates[c]]), ONE_BIN)
            for i, s in enumerate(sr_ids):
                f[f"{base}/sr{int(s)}/yield"] = (np.asarray(sw[i], dtype=float), edges)
                f[f"{base}/sr{int(s)}/sumw2"] = (np.asarray(sw2[i], dtype=float), edges)
                f[f"{base}/sr{int(s)}/n_entries"] = (np.asarray(n[i], dtype=float), edges)
        if result["data"] is not None:
            counts, blinded = result["data"]["counts"], result["data"]["blinded"]
            low, high = cfg.blind_window
            for key, value in (("has_data", 1.0), ("blind_low", low), ("blind_high", high)):
                f[f"massfit/settings/{key}"] = (np.array([float(value)]), ONE_BIN)
            f["massfit/fail/data/yield"] = (np.asarray(counts[0], dtype=float), edges)
            for i, s in enumerate(sr_ids):
                f[f"massfit/pass/data/sr{int(s)}/yield"] = (np.asarray(counts[i + 1], dtype=float), edges)
                f[f"massfit/blinded/sr{int(s)}"] = (blinded.astype(float), edges)
    log(f"Wrote ROOT file: {path}")


def summary(result):
    """JSON-ready summary: fit, normalisations, per-SR QCD prediction and yields, transfer factors."""
    tf, cfg = result["tf"], result["cfg"]
    b = result["bundles"]
    per_sr = []
    for i, sr in enumerate(result["sr_ids"]):
        row = {"bin_index": sr, "qcd_true": float(b["qcd_true"]["yield"][i]),
               "qcd_true_stat": float(math.sqrt(max(b["qcd_true"]["var"][i], 0.0))),
               "qcd_true_entries": int(b["qcd_true"]["n_entries"][i]), "qcd_predict": float(b["qcd_predict"]["yield"][i]),
               "n0": float(result["n0"][i]),
               "groups": {c: {"yield": float(b[f"groups/{slugify(c)}"]["yield"][i]),
                              "stat_error": float(math.sqrt(max(b[f"groups/{slugify(c)}"]["var"][i], 0.0))),
                              "n_entries": int(b[f"groups/{slugify(c)}"]["n_entries"][i])}
                          for c in result["class_names"]},
               "pass_qcd_mc_fit_range": float(result["qcd_true"][0][i + 1].sum())}
        per_sr.append(row)
    out = {"qcd_from_mc": cfg.qcd_from_mc, "edges": result["edges"].tolist(), "signal_regions": per_sr,
           "fail_qcd_mc": float(result["qcd_true"][0][0].sum()), "abcd": result["abcd"],
           "qcd_predict_abcd_scale": float(b["qcd_predict"]["abcd_scale"]),
           "transfer_factors": sideband_transfer_factors(result), "approximations": result["approximations"],
           "has_data": result["data"] is not None}
    if tf is not None:
        out["tfmc"] = {"order": cfg.tfmc_order, "coefficients": tf["coefficients"].tolist(),
                       "covariance": tf["covariance"].tolist(), "chi2": tf["chi2"], "ndf": tf["ndf"],
                       "x": tf["x"].tolist(), "shape": tf["shape"].tolist(), "eig": [e.tolist() for e in tf["eig"]]}
    if result["data"] is not None:
        out["data"] = {"fail": float(result["data"]["counts"][0].sum()),
                       "pass_unblinded": [float(v.sum()) for v in result["data"]["counts"][1:]],
                       "blinded_bins": np.flatnonzero(result["data"]["blinded"]).tolist()}
    return out
