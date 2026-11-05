"""Signal-region optimiser (port of legacy selections/signal_region/signal_region_hist.py).

Finds n_signal_regions non-overlapping boxes [lo, hi) in the class-probability space
(an upper edge at 1 includes 1) maximising sum Z^2, Z the Asimov significance with the
background uncertainty sigma_B^2 (Cowan):

1. Coarse: coordinate-descent beam search on the coarse grid, run n_signal_regions times,
   each pass restricted to boxes disjoint from the previous passes' best boxes (diversity).
   Every accepted box evaluated is a candidate; the per-pass bests form a feasible chain.
2. Selection: the candidates capped (top Z^2 + per dominant axis + chain), a multi-start
   greedy incumbent, then an exact branch-and-bound maximising (#regions, sum Z^2).
3. Fine: each selected box's edges re-optimised on the fine grid within
   +/- fine_refine_window, one SR after the other, keeping the boxes disjoint.
   bin_index = selection order.

sigma_B^2 is the sum w^2 of the background MC; with abcd_uncertainty the QCD part is replaced
by the ABCD variance of a QCD prediction from msoftdrop control windows (abcd_qcd_variance)
plus (nonclosure x QCD)^2, and boxes need min_control_weight expected control events.

Every coarse-stage box has its edges on the coarse grid, so with cells=True the events are
first summed into the occupied coarse cells and the coarse stage runs on those (exact up to
floating-point summation order); the fine stage and the final yields use the events.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12
SEED_QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


# --- config -----------------------------------------------------------------


@dataclass
class AbcdConfig:
    branch: str
    windows: list
    nonclosure: float
    transfer_factors: list | None = None
    min_control_weight: float = 0.0

    @classmethod
    def from_dict(cls, d):
        wins = d["control_window"]
        wins = [tuple(float(v) for v in w) for w in wins] if wins and isinstance(wins[0], (list, tuple)) \
            else [tuple(float(v) for v in wins)]
        tf = None if d.get("transfer_factors") is None else [float(v) for v in d["transfer_factors"]]
        out = cls(str(d["branch"]), wins, float(d["nonclosure"]), tf, float(d.get("min_control_weight", 0.0)))
        if (not wins or any(len(w) != 2 or not w[0] < w[1] for w in wins)
                or out.nonclosure < 0.0 or out.min_control_weight < 0.0):
            raise ValueError(f"abcd_uncertainty needs control windows with low < high, nonclosure >= 0 and "
                             f"min_control_weight >= 0, got {d}")
        ws = sorted(wins)
        if any(a[1] > b[0] for a, b in zip(ws, ws[1:])):
            raise ValueError(f"abcd_uncertainty control windows overlap: {wins}")
        if tf is not None and (len(tf) != len(wins) or any(not v > 0.0 for v in tf)):
            raise ValueError(f"abcd_uncertainty transfer_factors needs one positive value per window, got {tf}")
        return out


@dataclass
class SRConfig:
    """The legacy sr_config.json keys (defaults as signal_region_hist.py; input paths ignored)."""

    lumi: float
    n_signal_regions: int = 4
    score_axes: object = "independent"
    merge_score_axes: list = field(default_factory=list)
    min_bkg_weight: float = 5.0
    min_signal_weight: float = 0.0
    min_signal_entries: int = 0
    min_bkg_entries: int = 0
    coarse_bin_width: float = 0.05
    fine_bin_width: float = 0.01
    fine_refine_window: float = 0.05
    qcd_coarse_bin_width: float | None = None
    qcd_fine_bin_width: float | None = None
    qcd_fine_refine_window: float | None = None
    beam_width: int = 64
    top_intervals_per_axis: int = 8
    coordinate_rounds: int = 6
    fine_refine_passes: int = 2
    max_selection_candidates: int = 400
    selection_reps_per_axis: int = 60
    branch_bound_max_nodes: int = 2000000
    branch_bound_time_limit_seconds: float = 60.0
    max_threads: int = 8
    progress_every_seconds: float = 30.0
    zero_count_floor: bool = False
    exclude_samples: list = field(default_factory=list)
    abcd: AbcdConfig | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d):
        c = cls(
            lumi=float(d["lumi"]),
            n_signal_regions=max(1, int(d.get("n_signal_regions", d.get("N", 4)))),
            score_axes=d.get("score_axes", "independent"),
            merge_score_axes=list(d.get("merge_score_axes", [])),
            min_bkg_weight=float(d.get("min_bkg_weight", 5.0)),
            min_signal_weight=float(d.get("min_signal_weight", 0.0)),
            min_signal_entries=max(0, int(d.get("min_signal_entries", 0))),
            min_bkg_entries=max(0, int(d.get("min_bkg_entries", 0))),
            coarse_bin_width=float(d.get("coarse_bin_width", 0.05)),
            fine_bin_width=float(d.get("fine_bin_width", 0.01)),
            fine_refine_window=float(d.get("fine_refine_window", 0.05)),
            beam_width=max(1, int(d.get("beam_width", 64))),
            top_intervals_per_axis=max(1, int(d.get("top_intervals_per_axis", 8))),
            coordinate_rounds=max(1, int(d.get("coordinate_rounds", 6))),
            fine_refine_passes=max(1, int(d.get("fine_refine_passes", 2))),
            max_selection_candidates=max(1, int(d.get("max_selection_candidates", 400))),
            selection_reps_per_axis=max(0, int(d.get("selection_reps_per_axis", 60))),
            branch_bound_max_nodes=max(0, int(d.get("branch_bound_max_nodes", 2000000))),
            branch_bound_time_limit_seconds=max(0.0, float(d.get("branch_bound_time_limit_seconds", 60.0))),
            max_threads=max(1, int(d.get("max_threads", 8))),
            progress_every_seconds=float(d.get("progress_every_seconds", 30.0)),
            zero_count_floor=bool(d.get("zero_count_floor", False)),
            exclude_samples=list(d.get("exclude_samples", [])),
            abcd=AbcdConfig.from_dict(d["abcd_uncertainty"]) if d.get("abcd_uncertainty") is not None else None,
            raw=dict(d),
        )
        c.qcd_coarse_bin_width = float(d.get("qcd_coarse_bin_width", c.coarse_bin_width))
        c.qcd_fine_bin_width = float(d.get("qcd_fine_bin_width", c.fine_bin_width))
        c.qcd_fine_refine_window = float(d.get("qcd_fine_refine_window", c.fine_refine_window))
        return c

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @property
    def need_counts(self):
        return self.min_signal_entries > 0 or self.min_bkg_entries > 0


def resolve_axes(cfg, class_names):
    """(axis names, class-index groups summed per axis): score_axes, then merge_score_axes."""
    n = len(class_names)
    setting = cfg.score_axes if cfg.score_axes is not None else "independent"
    if isinstance(setting, str) and setting.strip().lower() in ("independent", "first_n_minus_one", "first_n-1"):
        base = list(range(max(1, n - 1)))
    elif isinstance(setting, str) and setting.strip().lower() in ("all", "*"):
        base = list(range(n))
    else:
        names = [s.strip() for s in setting.split(",") if s.strip()] if isinstance(setting, str) else list(setting)
        base = []
        for item in names:
            idx = int(item) if isinstance(item, (int, np.integer)) else class_names.index(str(item))
            if not 0 <= idx < n:
                raise IndexError(f"score_axes index {idx} outside [0, {n})")
            if idx not in base:
                base.append(idx)
    base_names = [class_names[i] for i in base]
    pos = {nm: i for i, nm in enumerate(base_names)}
    groups, used = [], set()
    for merge in cfg.merge_score_axes:
        p = [pos[nm] for nm in merge]
        if used & set(p):
            raise ValueError(f"merge_score_axes group {merge!r} overlaps another group")
        used |= set(p)
        groups.append(("+".join(merge), [base[i] for i in p], min(p)))
    groups += [(nm, [base[i]], i) for i, nm in enumerate(base_names) if i not in used]
    groups.sort(key=lambda g: g[2])
    return [g[0] for g in groups], [g[1] for g in groups]


# --- inputs -----------------------------------------------------------------


@dataclass
class ScanInputs:
    """Events of the scan and (ABCD only) of the control windows.

    proba (N, C) class probabilities, weight (N,) physics weight x lumi, class_idx (N,),
    sample (N,) sample names; control: the same keys plus window (index of the control
    window of each event). class_names and signal_classes (bool per class) define S and B.
    """

    proba: np.ndarray
    weight: np.ndarray
    class_idx: np.ndarray
    sample: np.ndarray
    class_names: list
    signal_classes: list
    control: dict | None = None

    @classmethod
    def from_events(cls, proba, weight, class_idx, sample, class_names, signal_classes,
                    abcd_values=None, abcd_cond=None, control_windows=()):
        """Split events into the scan (abcd_values passes abcd_cond, legacy threshold semantics)
        and the control windows (fails abcd_cond, low < value < high), e.g. msoftdrop_1 of
        coffea outputs. Without abcd_values every event is a scan event."""
        from vvv.ml.inference import threshold_mask

        proba, weight = np.asarray(proba, dtype=float), np.asarray(weight, dtype=float)
        class_idx, sample = np.asarray(class_idx).astype(int), np.asarray(sample)
        if abcd_values is None:
            return cls(proba, weight, class_idx, sample, list(class_names), list(signal_classes))
        v = np.asarray(abcd_values, dtype=float)
        passed = threshold_mask({"x": v}, {"x": abcd_cond})
        window = np.full(len(v), -1, dtype=int)
        for k, (low, high) in enumerate(control_windows):
            window[~passed & (v > low) & (v < high)] = k
        c = window >= 0
        control = {"proba": proba[c], "weight": weight[c], "class_idx": class_idx[c], "sample": sample[c],
                   "window": window[c]}
        return cls(proba[passed], weight[passed], class_idx[passed], sample[passed], list(class_names),
                   list(signal_classes), control)

    def exclude(self, samples, log=print):
        """Drop the events of the given samples (legacy exclude_samples) from scan and control."""
        if not samples:
            return self
        drop = sorted(samples)
        keep = ~np.isin(np.asarray(self.sample, dtype=str), drop)
        log(f"Excluding samples {drop}: {int(np.count_nonzero(~keep))} events "
            f"(weight {float(self.weight[~keep].sum()):.6g})")
        control = self.control
        if control is not None:
            ck = ~np.isin(np.asarray(control["sample"], dtype=str), drop)
            log(f"Excluding samples {drop}: {int(np.count_nonzero(~ck))} events "
                f"(weight {float(control['weight'][~ck].sum()):.6g})")
            control = {k: v[ck] for k, v in control.items()}
        return ScanInputs(self.proba[keep], self.weight[keep], self.class_idx[keep], self.sample[keep],
                          self.class_names, self.signal_classes, control)


# --- significance -----------------------------------------------------------


def asimov_z(S, B, B_var):
    """Asimov discovery significance with background uncertainty sigma_B^2 = B_var (Cowan):
    Z = sqrt(2[(S+B) ln((S+B)(B+v)/(B^2+(S+B)v)) - (B^2/v) ln(1 + vS/(B(B+v)))]), the
    v -> 0 limit sqrt(2[(S+B) ln(1+S/B) - S]) for v <= 1e-12 B^2. Vectorised; 0 where S <= 0
    or B <= 0."""
    S, B, V = np.broadcast_arrays(np.asarray(S, dtype=float), np.asarray(B, dtype=float),
                                  np.maximum(np.asarray(B_var, dtype=float), 0.0))
    Z = np.zeros(S.shape, dtype=float)
    ok = (S > 0.0) & (B > 0.0)
    if np.any(ok):
        s, b, v = S[ok], B[ok], V[ok]
        exact = v <= 1e-12 * b * b
        v_safe = np.where(exact, 1.0, v)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            f_exact = (s + b) * np.log1p(s / b) - s
            f_unc = ((s + b) * np.log((s + b) * (b + v_safe) / (b * b + (s + b) * v_safe))
                     - (b * b / v_safe) * np.log1p(v_safe * s / (b * (b + v_safe))))
        f = np.where(exact, f_exact, f_unc)
        Z[ok] = np.sqrt(2.0 * np.maximum(np.nan_to_num(f, nan=0.0), 0.0))
    return Z if Z.ndim else float(Z)


def asimov_z_with_error(S, B, sS, sB):
    """(Z, sigma_Z): Z = asimov_z(S, B, sB^2), sigma_Z from the MC statistics sS and sB of S
    and B by central finite differences (sigma_B held fixed)."""
    Z = float(asimov_z(S, B, sB * sB))
    if Z <= 0.0:
        return 0.0, 0.0
    hS = 1e-6 * max(abs(S), 1e-12)
    hB = 1e-6 * max(abs(B), 1e-12)
    v = sB * sB
    dZ_dS = (float(asimov_z(S + hS, B, v)) - float(asimov_z(S - hS, B, v))) / (2.0 * hS)
    dZ_dB = (float(asimov_z(S, B + hB, v)) - float(asimov_z(S, max(B - hB, 1e-300), v))) / (2.0 * hB)
    return Z, float(np.sqrt((dZ_dS * sS) ** 2 + (dZ_dB * sB) ** 2))


def abcd_qcd_variance(tf, c_vars, qcd):
    """Variance of the ABCD QCD prediction from the per-window control variances c_vars (control
    count of every process + non-QCD MC w^2) and the QCD yield (floats or arrays):
    V_k = max(TF_k^2 c_var_k, TF_k QCD) and their inverse-variance combination, 0 if any V_k is 0."""
    tf = np.asarray(tf, dtype=float)
    qcd_pos = np.maximum(qcd, 0.0)
    v = [np.maximum(tf[k] ** 2 * np.maximum(cv, 0.0), tf[k] * qcd_pos) for k, cv in enumerate(c_vars)]
    if len(v) == 1:
        return v[0]
    inv = sum(1.0 / np.where(vk > 0.0, vk, 1.0) for vk in v)
    exact = np.logical_or.reduce([vk <= 0.0 for vk in v])
    return np.where(exact, 0.0, 1.0 / inv)


# --- geometry ---------------------------------------------------------------


def _hi_open(h):
    return float(h) >= 1.0 - EPS


def box_mask(axes, lo, hi):
    """Rows of axes (n, D) inside [lo, hi); an upper edge at 1 includes 1."""
    m = np.ones(axes.shape[0], dtype=bool)
    for d in range(axes.shape[1]):
        v = axes[:, d]
        if _hi_open(hi[d]):
            m &= v >= lo[d]
        else:
            m &= (v >= lo[d]) & (v < hi[d])
    return m


def _mask_except(axes, lo, hi, skip):
    m = np.ones(axes.shape[0], dtype=bool)
    for dd in range(axes.shape[1]):
        if dd == skip:
            continue
        v = axes[:, dd]
        m &= (v >= lo[dd]) if _hi_open(hi[dd]) else ((v >= lo[dd]) & (v < hi[dd]))
    return m


def overlap(lo1, hi1, lo2, hi2):
    """Half-open boxes overlap (abutting edges do not)."""
    return all(lo1[d] < hi2[d] and lo2[d] < hi1[d] for d in range(len(lo1)))


def region_key(lo, hi):
    return tuple(round(float(v), 10) for v in lo), tuple(round(float(v), 10) for v in hi)


def _valid(lo, hi):
    return all(float(lo[d]) < float(hi[d]) - EPS for d in range(len(lo)))


def grid_edges(width):
    edges = np.round(np.arange(0.0, 1.0 + width / 2.0, width), 6)
    if edges[-1] < 1.0 - EPS:
        edges = np.r_[edges, 1.0]
    edges[-1] = 1.0
    edges[0] = 0.0
    return np.unique(edges)


# --- scan rows (events or coarse cells) --------------------------------------


class Rows:
    """Scan rows with per-row sums: s (signal w), b (background w), b2_all (background w^2),
    b2 (w^2 entering sigma_B: all background, or non-QCD background with ABCD), q (QCD w),
    floor_* (per zero-count-floor class), and the control rows (axes, var, win, w).
    Events keep the original arrays so sums follow the legacy expressions exactly."""

    cells = False

    def __init__(self, axes, y, w, is_sig, is_bkg, b2, q, floor_masks, control):
        self.axes, self.y, self.w = axes, y, w
        self.is_sig, self.is_bkg = is_sig, is_bkg
        self.s = np.where(is_sig, w, 0.0)
        self.b = np.where(is_bkg, w, 0.0)
        self.b2, self.q = b2, q
        self.floor_masks = floor_masks
        self.control = control  # dict(axes, var, win, w) or None
        self.n = axes.shape[0]

    def rect_stats(self, m):
        """(S, B, sigma_B MC of all background, S entries, B entries) of the rows m."""
        ms, mb = m & self.is_sig, m & self.is_bkg
        wS, wB = self.w[ms], self.w[mb]
        return float(wS.sum()), float(wB.sum()), float(np.sqrt((wB ** 2).sum())), int(ms.sum()), int(mb.sum())

    def floor_empty(self, m, c):
        mr = m & self.floor_masks[c]
        return not mr.any() or float(self.w[mr].sum()) <= 0.0

    def sweep(self, ctx, d, lo, hi, edges):
        """Histograms on axis d (bins edges) of the rows inside [lo, hi) on the other axes:
        s, b, b2, q, counts ns / nb (entry thresholds), floor [(n, w)] per floor class, and of the
        control rows cvar (per window) and cw. None if no row passes."""
        cfg = ctx.cfg
        m = _mask_except(self.axes, lo, hi, d)
        if not m.any():
            return None
        v_d = self.axes[:, d]

        def h(mask, weights=None):
            return np.histogram(v_d[mask], bins=edges, weights=None if weights is None else weights[mask])[0]

        out = {"s": h(m, self.s), "b": h(m, self.b), "b2": h(m, self.b2)}
        if cfg.need_counts:
            out["ns"], out["nb"] = h(m & self.is_sig), h(m & self.is_bkg)
        if cfg.zero_count_floor:
            out["floor"] = [(h(m & mc), h(m & mc, self.w)) for mc in self.floor_masks]
        if ctx.abcd:
            c = self.control
            mc_ = _mask_except(c["axes"], lo, hi, d)
            cv = c["axes"][:, d]
            out["cvar"] = []
            for k in range(len(cfg.abcd.windows)):
                sel = mc_ & (c["win"] == k)
                out["cvar"].append(np.histogram(cv[sel], bins=edges, weights=c["var"][sel])[0])
            out["q"] = h(m, self.q)
            if ctx.min_control > 0.0:
                out["cw"] = np.histogram(cv[mc_], bins=edges, weights=c["w"][mc_])[0]
        return out


class CellRows(Rows):
    """Events summed into occupied cells of per-axis grids (cell coordinate = its lower edge);
    exact for boxes whose edges lie on those grids."""

    cells = True

    def __init__(self, ev, axis_edges, nwin=0):
        idx, cell_axes = _cell_index(ev.axes, axis_edges)
        key = _cell_key(idx, axis_edges)
        uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
        n = len(uniq)

        def agg(x):
            return np.bincount(inv, weights=np.asarray(x, dtype=float), minlength=n)

        self.axes = cell_axes[first]
        self.grid = axis_edges
        self.idx = [np.ascontiguousarray(idx[first, d]).astype(np.uint16) for d in range(len(axis_edges))]
        self.index = _axis_index(self.idx, axis_edges)
        self.n = n
        self.s, self.b, self.b2, self.q = agg(ev.s), agg(ev.b), agg(ev.b2), agg(ev.q)
        self.b2_all = agg(ev.b * ev.b)
        self.ns, self.nb = agg(ev.is_sig), agg(ev.is_bkg)
        self.floor_n = [agg(mc) for mc in ev.floor_masks]
        self.floor_w = [agg(np.where(mc, ev.w, 0.0)) for mc in ev.floor_masks]
        self.control = None
        if ev.control is not None:
            c = ev.control
            cidx, caxes = _cell_index(c["axes"], axis_edges)
            ckey = _cell_key(cidx, axis_edges) * 64 + c["win"]
            cu, cfirst, cinv = np.unique(ckey, return_index=True, return_inverse=True)
            self.control = {"axes": caxes[cfirst], "win": c["win"][cfirst],
                            "idx": [np.ascontiguousarray(cidx[cfirst, d]).astype(np.uint16)
                                    for d in range(len(axis_edges))],
                            "index": None,
                            "var": np.bincount(cinv, weights=c["var"], minlength=len(cu)),
                            "w": np.bincount(cinv, weights=c["w"], minlength=len(cu))}

        if self.control is not None:
            self.control["index"] = _axis_index(self.control["idx"], axis_edges)
        if self.control is not None:
            c = self.control
            c["var_win"] = [np.where(c["win"] == k, c["var"], 0.0) for k in range(nwin)]

    def rect_stats(self, m):
        return (float(self.s[m].sum()), float(self.b[m].sum()), float(np.sqrt(self.b2_all[m].sum())),
                int(round(self.ns[m].sum())), int(round(self.nb[m].sum())))

    def floor_empty(self, m, c):
        return not self.floor_n[c][m].sum() > 0 or float(self.floor_w[c][m].sum()) <= 0.0

    def _index_range(self, d, lo, hi):
        grid = self.grid[d]
        a = int(np.searchsorted(grid, lo - EPS))
        if not abs(grid[a] - lo) < 1e-9:
            raise ValueError(f"box edge {lo} not on the cell grid of axis {d}")
        if _hi_open(hi):
            return a, grid.size  # the cell at 1 included
        b = int(np.searchsorted(grid, hi - EPS))
        if not abs(grid[b] - hi) < 1e-9:
            raise ValueError(f"box edge {hi} not on the cell grid of axis {d}")
        return a, b

    @staticmethod
    def _select(idx, index, ranges, skip):
        """Sorted row numbers inside the index ranges on every axis but skip: the rows of the
        axis with the fewest rows in range (from its index), filtered on the others."""
        axes = [dd for dd in range(len(ranges)) if dd != skip]
        if not axes:
            return np.arange(len(idx[0]))
        counts = [index[dd][1][ranges[dd][1]] - index[dd][1][ranges[dd][0]] for dd in axes]
        first = axes[int(np.argmin(counts))]
        order, start = index[first]
        sel = order[start[ranges[first][0]]:start[ranges[first][1]]]
        for dd in axes:
            if dd != first and sel.size:
                x = idx[dd][sel]
                sel = sel[(x >= ranges[dd][0]) & (x < ranges[dd][1])]
        return np.sort(sel)

    def sweep(self, ctx, d, lo, hi, edges):
        """As Rows.sweep, by bincounts of the cell indices (edges must be the cell grid of axis d)."""
        if not (edges.size == self.grid[d].size and np.array_equal(edges, self.grid[d])):
            raise ValueError("cell sweeps need the cell grid as bins")
        cfg = ctx.cfg
        nb = edges.size - 1
        ranges = [self._index_range(dd, lo[dd], hi[dd]) for dd in range(len(lo))]
        sel = self._select(self.idx, self.index, ranges, d)
        if sel.size == 0:
            return None
        bins = np.minimum(self.idx[d][sel], nb - 1).astype(np.intp)

        def h(col, count=False):
            out = np.bincount(bins, weights=col[sel], minlength=nb)
            return np.rint(out).astype(np.int64) if count else out

        out = {"s": h(self.s), "b": h(self.b), "b2": h(self.b2)}
        if cfg.need_counts:
            out["ns"], out["nb"] = h(self.ns, True), h(self.nb, True)
        if cfg.zero_count_floor:
            out["floor"] = [(h(n_, True), h(w_)) for n_, w_ in zip(self.floor_n, self.floor_w)]
        if ctx.abcd:
            c = self.control
            csel = self._select(c["idx"], c["index"], ranges, d)
            cbins = np.minimum(c["idx"][d][csel], nb - 1).astype(np.intp)
            out["cvar"] = [np.bincount(cbins, weights=v[csel], minlength=nb) for v in c["var_win"]]
            out["q"] = h(self.q)
            out["cw"] = np.bincount(cbins, weights=c["w"][csel], minlength=nb)
        return out


def _cell_index(axes, axis_edges):
    idx = np.empty(axes.shape, dtype=np.int64)
    coords = np.empty(axes.shape, dtype=float)
    for d, edges in enumerate(axis_edges):
        v = axes[:, d]
        if np.any((v < 0.0) | (v > 1.0)):
            raise ValueError(f"scores outside [0, 1] on axis {d}: the cell aggregation needs scores in [0, 1]")
        i = np.searchsorted(edges, v, side="right") - 1
        idx[:, d] = i
        coords[:, d] = edges[i]
    return idx, coords


def _axis_index(idx, axis_edges):
    """Per axis (rows sorted by cell index, offset of each index value in that order)."""
    out = []
    for x, edges in zip(idx, axis_edges):
        order = np.argsort(x, kind="stable").astype(np.int64)
        out.append((order, np.searchsorted(x[order], np.arange(edges.size + 1))))
    return out


def _cell_key(idx, axis_edges):
    key = np.zeros(idx.shape[0], dtype=np.int64)
    for d, edges in enumerate(axis_edges):
        key = key * len(edges) + idx[:, d]
    return key


# --- context ----------------------------------------------------------------


class Ctx:
    """Scan context: config, axes, the event rows (and coarse rows), ABCD constants, logging."""

    def __init__(self, cfg, inputs, cells=False, threads=None, log=print):
        self.cfg, self.log = cfg, log
        self.threads = max(1, int(threads or cfg.max_threads))
        self.class_names = list(inputs.class_names)
        self.signal_classes = [i for i, s in enumerate(inputs.signal_classes) if s]
        self.background_classes = [i for i, s in enumerate(inputs.signal_classes) if not s]
        self.axis_names, groups = resolve_axes(cfg, self.class_names)
        proba = np.asarray(inputs.proba, dtype=float)
        if proba.shape[1] != len(self.class_names):
            raise RuntimeError(f"proba has {proba.shape[1]} classes, expected {len(self.class_names)}")
        axes = np.column_stack([proba[:, g].sum(axis=1) for g in groups])
        self.D = axes.shape[1]
        y = np.asarray(inputs.class_idx, dtype=int)
        w = np.asarray(inputs.weight, dtype=float)
        is_sig = np.isin(y, self.signal_classes)
        is_bkg = np.isin(y, self.background_classes)
        self.abcd = cfg.abcd is not None and inputs.control is not None
        if cfg.abcd is not None and inputs.control is None:
            raise RuntimeError("abcd_uncertainty needs control-window events")
        self.qcd_classes = [c for c in self.background_classes if "qcd" in self.class_names[c].lower()]
        mc_bkg = is_bkg & ~np.isin(y, self.qcd_classes) if self.abcd else is_bkg
        b2 = np.where(mc_bkg, w * w, 0.0)
        q = np.where(np.isin(y, self.qcd_classes), w, 0.0)
        control = None
        if self.abcd:
            if not self.qcd_classes:
                raise RuntimeError("abcd_uncertainty needs a QCD class")
            c = inputs.control
            cy = np.asarray(c["class_idx"], dtype=int)
            cw = np.asarray(c["weight"], dtype=float)
            c_qcd = np.isin(cy, self.qcd_classes)
            c_mc_bkg = np.isin(cy, self.background_classes) & ~c_qcd
            cp = np.asarray(c["proba"], dtype=float)
            # Per control event: its expected count (every process, signal included) plus w^2 of
            # the non-QCD background MC, its window, and its expected count alone.
            control = {"axes": np.column_stack([cp[:, g].sum(axis=1) for g in groups]),
                       "var": cw + np.where(c_mc_bkg, cw * cw, 0.0),
                       "win": np.asarray(c["window"], dtype=int), "w": cw}
            nwin = len(cfg.abcd.windows)
            if cfg.abcd.transfer_factors is not None:
                self.tf = np.asarray(cfg.abcd.transfer_factors, dtype=float)
            else:  # QCD MC of the scan / of the control window
                self.tf = np.array([float(q.sum()) / float(cw[c_qcd & (control["win"] == k)].sum())
                                    for k in range(nwin)])
            self.nc2 = cfg.abcd.nonclosure ** 2
            self.min_control = cfg.abcd.min_control_weight
        # Zero-count floor: per background class (not QCD with ABCD) alpha^2, alpha = median |w|.
        self.floor_classes = [c for c in self.background_classes if not (self.abcd and c in self.qcd_classes)]
        floor_masks = [y == c for c in self.floor_classes]
        self.floor_alpha2 = [(float(np.median(np.abs(w[mc]))) ** 2) if mc.any() else 0.0 for mc in floor_masks]
        self.events = Rows(axes, y, w, is_sig, is_bkg, b2, q, floor_masks, control)
        self.S_total = float(self.events.s.sum())
        self.B_total = float(self.events.b.sum())

        qa = self.qcd_axis()
        coarse = grid_edges(cfg.coarse_bin_width)
        coarse_qcd = grid_edges(cfg.qcd_coarse_bin_width) if qa >= 0 else coarse
        self.coarse_edges = [coarse_qcd if d == qa else coarse for d in range(self.D)]
        fine = grid_edges(cfg.fine_bin_width)
        fine_qcd = grid_edges(cfg.qcd_fine_bin_width) if qa >= 0 else fine
        self.fine_edges = [fine_qcd if d == qa else fine for d in range(self.D)]
        self.fine_windows = [cfg.qcd_fine_refine_window if d == qa else cfg.fine_refine_window
                             for d in range(self.D)]
        self.coarse = (CellRows(self.events, self.coarse_edges, len(cfg.abcd.windows) if self.abcd else 0)
                       if cells else self.events)
        self.forbidden = []
        self._t0 = time.monotonic()
        self._last = self._t0

    def qcd_axis(self):
        for i, name in enumerate(self.axis_names):
            if name.strip().upper() == "QCD":
                return i
        return -1

    def progress(self, message, force=False):
        now = time.monotonic()
        if force or self.cfg.progress_every_seconds <= 0.0 or now - self._last >= self.cfg.progress_every_seconds:
            self.log(f"  [{now - self._t0:.1f}s] {message}")
            self._last = now

    def map(self, fn, items):
        items = list(items)
        if self.threads <= 1 or len(items) <= 1:
            return [fn(it) for it in items]
        with ThreadPoolExecutor(max_workers=self.threads) as ex:
            return list(ex.map(fn, items))

    # ---- box statistics ----
    def floor_variance(self, rows, m):
        if not self.cfg.zero_count_floor:
            return 0.0
        return sum(a2 for c, a2 in enumerate(self.floor_alpha2) if rows.floor_empty(m, c))

    def control_box(self, rows, lo, hi):
        """Per-window control variances and the expected control count of the box."""
        c = rows.control
        cm = box_mask(c["axes"], lo, hi)
        c_vars = [float(np.sum(c["var"][cm & (c["win"] == k)])) for k in range(len(self.cfg.abcd.windows))]
        return c_vars, float(np.sum(c["w"][cm]))

    def abcd_terms(self, rows, lo, hi, m):
        """ABCD variance of the box's QCD prediction (+ nonclosure term) and its control count."""
        c_vars, c_count = self.control_box(rows, lo, hi)
        qcd = float(rows.q[m].sum())
        return float(abcd_qcd_variance(self.tf, c_vars, qcd)) + self.nc2 * qcd * qcd, c_count


# --- candidates -------------------------------------------------------------


def evaluate_region(ctx, rows, lo, hi, stats=None):
    """Candidate dict of the box [lo, hi), or None if it fails the constraints or overlaps a
    forbidden box. stats = (S, B, B2, S_e, B_e) from top_intervals_on_axis skips the recompute."""
    cfg = ctx.cfg
    lo = [float(v) for v in lo]
    hi = [float(v) for v in hi]
    if not _valid(lo, hi):
        return None
    for flo, fhi in ctx.forbidden:
        if overlap(lo, hi, flo, fhi):
            return None
    if stats is None:
        m = box_mask(rows.axes, lo, hi)
        S, B, sB, S_e, B_e = rows.rect_stats(m)
        if ctx.abcd:
            abcd_var, c_count = ctx.abcd_terms(rows, lo, hi, m)
            B2 = float(rows.b2[m].sum()) + ctx.floor_variance(rows, m) + abcd_var
            # The interval sweep applies min_control_weight to the values it supplies.
            if ctx.min_control > 0.0 and c_count < ctx.min_control:
                return None
        else:
            B2 = sB * sB + ctx.floor_variance(rows, m)
    else:
        S, B, B2 = float(stats[0]), float(stats[1]), float(stats[2])
        S_e, B_e = (int(stats[3]), int(stats[4])) if cfg.need_counts else (-1, -1)
    if B < cfg.min_bkg_weight or S <= cfg.min_signal_weight:
        return None
    if (cfg.min_signal_entries > 0 and S_e < cfg.min_signal_entries) or \
            (cfg.min_bkg_entries > 0 and B_e < cfg.min_bkg_entries):
        return None
    Z = float(asimov_z(S, B, B2))
    if Z <= 0.0:
        return None
    return {"lo": lo, "hi": hi, "S": S, "B": B, "Z": Z, "S_entries": S_e, "B_entries": B_e}


def top_intervals_on_axis(ctx, rows, d, lo, hi, edges, top_n):
    """Top-Z intervals [edges[a], edges[b]) on axis d with the other axes of [lo, hi) fixed, as
    [(low, high, S, B, B2, Z, S_entries, B_entries)] (entries -1 without entry thresholds)."""
    cfg = ctx.cfg
    edges = np.unique(np.clip(np.asarray(edges, dtype=float), 0.0, 1.0))
    if edges.size < 2:
        return []
    h = rows.sweep(ctx, d, lo, hi, edges)
    if h is None:
        return []
    pS = np.r_[0.0, np.cumsum(h["s"])]
    pB = np.r_[0.0, np.cumsum(h["b"])]
    pB2 = np.r_[0.0, np.cumsum(h["b2"])]
    K = pS.size
    a_idx = np.arange(K).reshape(-1, 1)
    b_idx = np.arange(K).reshape(1, -1)
    tri = b_idx > a_idx
    S_mat = pS[b_idx] - pS[a_idx]
    B_mat = pB[b_idx] - pB[a_idx]
    B2_mat = pB2[b_idx] - pB2[a_idx]
    if cfg.need_counts:
        pNs = np.r_[0, np.cumsum(h["ns"])]
        pNb = np.r_[0, np.cumsum(h["nb"])]
    if cfg.zero_count_floor:
        for (hN, hW), alpha2 in zip(h["floor"], ctx.floor_alpha2):
            pN = np.r_[0, np.cumsum(hN)]
            pW = np.r_[0.0, np.cumsum(hW)]
            empty = ((pN[b_idx] - pN[a_idx]) == 0) | ((pW[b_idx] - pW[a_idx]) <= 0.0)
            B2_mat = B2_mat + np.where(empty, alpha2, 0.0)
    if ctx.abcd:
        c_mats = []
        for hC in h["cvar"]:
            pC = np.r_[0.0, np.cumsum(hC)]
            c_mats.append(pC[b_idx] - pC[a_idx])
        pQ = np.r_[0.0, np.cumsum(h["q"])]
        Q_mat = pQ[b_idx] - pQ[a_idx]
        B2_mat = B2_mat + abcd_qcd_variance(ctx.tf, c_mats, Q_mat) + ctx.nc2 * Q_mat * Q_mat
    valid = tri & (B_mat >= cfg.min_bkg_weight) & (S_mat > cfg.min_signal_weight)
    if ctx.abcd and ctx.min_control > 0.0:
        pN = np.r_[0.0, np.cumsum(h["cw"])]
        valid &= (pN[b_idx] - pN[a_idx]) >= ctx.min_control
    if not valid.any():
        return []
    Bsafe = np.where(valid, B_mat, 1.0)
    Ssafe = np.where(valid, S_mat, 0.0)
    Z2 = np.where(valid, asimov_z(Ssafe, Bsafe, np.where(valid, B2_mat, 0.0)) ** 2, -np.inf)
    valid_count = int(np.count_nonzero(np.isfinite(Z2) & (Z2 > 0.0)))
    if valid_count == 0:
        return []
    take = min(max(1, int(top_n)), valid_count)
    flat = Z2.ravel()
    if take >= flat.size:
        order = np.argsort(flat)[::-1]
    else:
        order = np.argpartition(flat, -take)[-take:]
        order = order[np.argsort(flat[order])[::-1]]
    out, seen = [], set()
    for f in order:
        if not np.isfinite(flat[f]) or flat[f] <= 0.0:
            continue
        a, b = int(f // K), int(f % K)
        if not valid[a, b] or (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((float(edges[a]), float(edges[b]), float(S_mat[a, b]), float(B_mat[a, b]),
                    float(B2_mat[a, b]), float(np.sqrt(Z2[a, b])),
                    int(pNs[b] - pNs[a]) if cfg.need_counts else -1,
                    int(pNb[b] - pNb[a]) if cfg.need_counts else -1))
        if len(out) >= take:
            break
    return out


# --- coarse beam search -----------------------------------------------------


def build_seeds(D, forbidden, axis_edges):
    """Seed boxes: the full box, per axis high bands [e, 1) and low bands [0, e) at grid
    quantiles SEED_QUANTILES, and the complement bands of the forbidden boxes."""
    seeds, seen = [], set()

    def add(lo, hi):
        if _valid(lo, hi) and region_key(lo, hi) not in seen:
            seen.add(region_key(lo, hi))
            seeds.append((list(lo), list(hi)))

    add([0.0] * D, [1.0] * D)
    for d in range(D):
        last = len(axis_edges[d]) - 1
        for qn in SEED_QUANTILES:
            e = float(axis_edges[d][int(round(qn * last))])
            if e > EPS:
                lo = [0.0] * D; hi = [1.0] * D; lo[d] = e; add(lo, hi)  # noqa: E702
            if e < 1.0 - EPS:
                lo = [0.0] * D; hi = [1.0] * D; hi[d] = e; add(lo, hi)  # noqa: E702
    for flo, fhi in forbidden:
        for d in range(D):
            if flo[d] > EPS:
                lo = [0.0] * D; hi = [1.0] * D; hi[d] = float(flo[d]); add(lo, hi)  # noqa: E702
            if fhi[d] < 1.0 - EPS:
                lo = [0.0] * D; hi = [1.0] * D; lo[d] = float(fhi[d]); add(lo, hi)  # noqa: E702
    return seeds


def coarse_beam_search(ctx, forbidden=()):
    """Coordinate-descent beam search on the coarse grid, restricted to boxes disjoint from
    forbidden. Returns every accepted box evaluated (the pool), not just the final beam."""
    cfg, rows = ctx.cfg, ctx.coarse
    ctx.forbidden = list(forbidden)
    pool, initial = {}, []
    seeds = build_seeds(ctx.D, ctx.forbidden, ctx.coarse_edges)
    for item in ctx.map(lambda s: evaluate_region(ctx, rows, s[0], s[1]), seeds):
        if item is not None and region_key(item["lo"], item["hi"]) not in pool:
            pool[region_key(item["lo"], item["hi"])] = item
            initial.append(item)
    if not initial:
        ctx.forbidden = []
        return []
    beam = sorted(initial, key=lambda it: -it["Z"])[:cfg.beam_width]
    ctx.progress(f"Coarse beam start: edges/axis={ctx.coarse_edges[0].size}, seeds={len(initial)}, "
                 f"beam={len(beam)}, best_Z={beam[0]['Z']:.4f}", force=True)

    def task(t):
        item, d = t
        children = []
        for low, high, S, B, B2, _Z, S_e, B_e in top_intervals_on_axis(
                ctx, rows, d, item["lo"], item["hi"], ctx.coarse_edges[d], cfg.top_intervals_per_axis):
            lo, hi = list(item["lo"]), list(item["hi"])
            lo[d], hi[d] = low, high
            children.append((lo, hi, (S, B, B2, S_e, B_e)))
        return children

    prev = None
    for r in range(cfg.coordinate_rounds):
        produced = []
        for children in ctx.map(task, [(it, d) for it in beam for d in range(ctx.D)]):
            for lo, hi, stats in children:
                key = region_key(lo, hi)
                if key in pool:
                    produced.append(pool[key])
                    continue
                item = evaluate_region(ctx, rows, lo, hi, stats)
                if item is not None:
                    pool[key] = item
                    produced.append(item)
        merged = {}
        for item in beam + produced:
            key = region_key(item["lo"], item["hi"])
            if key not in merged or item["Z"] > merged[key]["Z"]:
                merged[key] = item
        beam = sorted(merged.values(), key=lambda it: -it["Z"])[:cfg.beam_width]
        ctx.progress(f"Coarse round {r + 1}/{cfg.coordinate_rounds} done: pool={len(pool)}, beam={len(beam)}, "
                     f"best_Z={beam[0]['Z'] if beam else 0.0:.4f}", force=True)
        keys = tuple(region_key(it["lo"], it["hi"]) for it in beam)
        if keys == prev:
            ctx.progress("Coarse beam converged (beam unchanged)", force=True)
            break
        prev = keys
    ctx.forbidden = []
    return list(pool.values())


def find_regions(ctx, target_n, report=None):
    """Candidate pool and disjoint chain from target_n coarse passes, each forbidding the
    previous passes' best boxes; the chain is then completed greedily from the pool."""
    forbidden, pool, chain = [], {}, []
    for k in range(target_n):
        cpool = coarse_beam_search(ctx, forbidden)
        if not cpool:
            ctx.progress(f"Diversity pass {k + 1}/{target_n}: no disjoint region remains", force=True)
            break
        for it in cpool:
            key = region_key(it["lo"], it["hi"])
            if key not in pool or it["Z"] > pool[key]["Z"]:
                pool[key] = it
        best = max(cpool, key=lambda it: it["Z"])
        chain.append(best)
        forbidden.append((best["lo"], best["hi"]))
        ctx.progress(f"Diversity pass {k + 1}/{target_n}: best_Z={best['Z']:.4f}, chain={len(chain)}, "
                     f"pool={len(pool)}", force=True)
        if report is not None:
            report.append({"pass": k + 1, "best_Z": best["Z"], "pass_pool": len(cpool), "pool": len(pool),
                           "lo": best["lo"], "hi": best["hi"]})
    keys = {region_key(c["lo"], c["hi"]) for c in chain}
    for it in sorted(pool.values(), key=lambda x: -x["Z"]):
        if len(chain) >= target_n:
            break
        key = region_key(it["lo"], it["hi"])
        if key not in keys and all(not overlap(it["lo"], it["hi"], c["lo"], c["hi"]) for c in chain):
            chain.append(it)
            keys.add(key)
    ctx.progress(f"Diversity done: pool={len(pool)}, completed_chain={len(chain)}/{target_n}", force=True)
    return list(pool.values()), chain


# --- selection --------------------------------------------------------------


def select_branch_bound(ctx, candidates, target_n, incumbent=None):
    """Up to target_n non-overlapping candidates maximising (#regions, sum Z^2): capped pool,
    multi-start greedy incumbent, exact bitset branch-and-bound (node / time caps).
    Returns (picked boxes, summary)."""
    cfg = ctx.cfg
    items_sorted = sorted(candidates, key=lambda it: -(it["Z"] ** 2))
    if len(items_sorted) < target_n:
        raise RuntimeError(f"Only {len(items_sorted)} candidate signal regions are available; requested {target_n}")
    # Diversity-preserving cap: top Z^2, the top reps per dominant (largest lower cut) axis, the chain.
    kept = list(items_sorted[:cfg.max_selection_candidates])
    present = {region_key(it["lo"], it["hi"]) for it in kept}

    def augment(extra):
        for it in extra:
            key = region_key(it["lo"], it["hi"])
            if key not in present:
                kept.append(it)
                present.add(key)

    if cfg.selection_reps_per_axis > 0:
        by_axis = {}
        for it in items_sorted:
            d = int(np.argmax(it["lo"]))
            if it["lo"][d] > EPS:
                bucket = by_axis.setdefault(d, [])
                if len(bucket) < cfg.selection_reps_per_axis:
                    bucket.append(it)
        for d in by_axis:
            augment(by_axis[d])
    if incumbent:
        augment(incumbent)
    if len(kept) != len(items_sorted):
        ctx.log(f"  Selection pool: {len(items_sorted)} -> {len(kept)} (top-Z + per-axis diversity + incumbent)")
    items = sorted(kept, key=lambda it: -(it["Z"] ** 2))
    n = len(items)
    los = [it["lo"] for it in items]
    his = [it["hi"] for it in items]
    Z2 = np.array([it["Z"] ** 2 for it in items], dtype=float)
    key_to_idx = {region_key(it["lo"], it["hi"]): i for i, it in enumerate(items)}

    # Multi-start greedy: each of the top M as first pick; best set per size.
    best_by_size = {}

    def record(picks):
        for size in range(1, len(picks) + 1):
            sub = tuple(picks[:size])
            score = float(np.sum(Z2[list(sub)]))
            if size not in best_by_size or score > best_by_size[size][0]:
                best_by_size[size] = (score, sub)

    def greedy(first):
        picks = [first]
        for i in range(n):
            if len(picks) >= target_n:
                break
            if i != first and all(not overlap(los[i], his[i], los[j], his[j]) for j in picks):
                picks.append(i)
        return picks

    for first in range(min(n, max(8 * target_n, 64))):
        record(greedy(first))
    if incumbent:
        chain_idx = [key_to_idx[region_key(it["lo"], it["hi"])] for it in incumbent
                     if region_key(it["lo"], it["hi"]) in key_to_idx]
        if chain_idx:
            record(chain_idx)
    if not best_by_size:
        raise RuntimeError("Global selection found no valid signal-region set")

    # Branch-and-bound on compatibility bitsets (index order = Z^2 order).
    LO, HI = np.asarray(los, dtype=float), np.asarray(his, dtype=float)
    ov = np.ones((n, n), dtype=bool)
    for d in range(ctx.D):
        ov &= (LO[:, d][:, None] < HI[:, d][None, :]) & (LO[:, d][None, :] < HI[:, d][:, None])
    compat_mat = ~ov
    np.fill_diagonal(compat_mat, False)
    compat = []
    for i in range(n):
        m = 0
        for j in np.flatnonzero(compat_mat[i]):
            m |= 1 << int(j)
        compat.append(m)
    seed_size = max(best_by_size)
    best = [seed_size, float(best_by_size[seed_size][0]), best_by_size[seed_size][1]]
    incumbent_z = float(np.sqrt(best[1]))
    root_upper = float(np.sum(np.sort(Z2)[::-1][:target_n]))
    nodes, stopped = [0], [False]
    t0 = time.monotonic()
    max_nodes, time_limit = cfg.branch_bound_max_nodes, cfg.branch_bound_time_limit_seconds

    def dfs(avail, picks, score):
        if stopped[0]:
            return
        nodes[0] += 1
        if max_nodes > 0 and nodes[0] >= max_nodes:
            stopped[0] = True
            return
        if time_limit > 0.0 and (nodes[0] & 0x3FFF) == 0 and time.monotonic() - t0 >= time_limit:
            stopped[0] = True
            return
        s = len(picks)
        if s > best[0] or (s == best[0] and score > best[1] + 1e-12):
            best[0], best[1], best[2] = s, float(score), picks
        if s >= target_n:
            return
        cap = target_n - s
        bsize = s + min(cap, bin(avail).count("1"))
        if bsize < best[0]:
            return
        bscore, need, tmp = score, cap, avail
        while tmp and need > 0:
            b = tmp & (-tmp)
            tmp ^= b
            bscore += Z2[b.bit_length() - 1]
            need -= 1
        if bsize == best[0] and bscore <= best[1] + 1e-12:
            return
        tmp = avail
        while tmp:
            b = tmp & (-tmp)
            i = b.bit_length() - 1
            tmp ^= b
            if stopped[0]:
                return
            dfs(tmp & compat[i], picks + (i,), score + Z2[i])

    ctx.progress(f"Branch-and-bound start: candidates={n}, target_bins={target_n}, incumbent_size={best[0]}, "
                 f"incumbent_Z={incumbent_z:.4f}", force=True)
    dfs((1 << n) - 1, tuple(), 0.0)
    size, score, picks = best
    if not picks:
        raise RuntimeError("Global selection found no valid signal-region set")
    if size < target_n:
        ctx.log(f"Warning: Pool supports only {size} mutually non-overlapping regions; selecting {size} "
                f"(requested {target_n})")
    completed = not stopped[0]
    if not completed:
        ctx.log(f"Warning: Branch-and-bound stopped early (node cap {max_nodes} or time limit {time_limit}s); "
                f"returning best incumbent (may be sub-optimal)")
    summary = {"completed": completed, "nodes": int(nodes[0]), "candidate_count": n,
               "candidate_pool": len(items_sorted), "incumbent_size": seed_size, "incumbent_Z": incumbent_z,
               "objective_sum_z2": float(score),
               "objective_upper_bound_sum_z2": float(score) if completed else float(max(score, root_upper)),
               "seconds": time.monotonic() - t0}
    return [items[i] for i in picks], summary


# --- fine refinement --------------------------------------------------------


def fine_refine(ctx, los, his):
    """Each box in turn: per pass and axis, adopt the best fine-grid interval within
    +/- fine_refine_window of the current edges that keeps it disjoint from the others."""
    cfg, rows = ctx.cfg, ctx.events
    los = [list(map(float, x)) for x in los]
    his = [list(map(float, x)) for x in his]
    n = len(los)
    ctx.forbidden = []

    def local_edges(b_lo, b_hi, d):
        base, win = ctx.fine_edges[d], ctx.fine_windows[d]
        keep = np.zeros(base.size, dtype=bool)
        for b in (b_lo, b_hi):
            keep |= np.abs(base - b) <= win + EPS
        return np.clip(np.unique(np.r_[base[keep], b_lo, b_hi, 0.0, 1.0]), 0.0, 1.0)

    ctx.progress(f"Fine refinement of {n} selected regions: window=+/-{cfg.fine_refine_window}, "
                 f"passes={cfg.fine_refine_passes}", force=True)
    for i in range(n):
        for _ in range(cfg.fine_refine_passes):
            changed = False
            for d in range(ctx.D):
                edges = local_edges(los[i][d], his[i][d], d)
                for low, high, S, B, _B2, _Z, S_e, B_e in top_intervals_on_axis(
                        ctx, rows, d, los[i], his[i], edges, cfg.top_intervals_per_axis):
                    if B < cfg.min_bkg_weight or S <= cfg.min_signal_weight:
                        continue
                    lo_alt, hi_alt = list(los[i]), list(his[i])
                    lo_alt[d], hi_alt[d] = low, high
                    if not _valid(lo_alt, hi_alt):
                        continue
                    if cfg.need_counts and (S_e < cfg.min_signal_entries or B_e < cfg.min_bkg_entries):
                        continue
                    if all(not overlap(lo_alt, hi_alt, los[j], his[j]) for j in range(n) if j != i):
                        if abs(low - los[i][d]) > 1e-12 or abs(high - his[i][d]) > 1e-12:
                            changed = True
                        los[i][d], his[i][d] = low, high
                        break
            if not changed:
                break
        ctx.progress(f"  Refined SR{i + 1}/{n}", force=True)
    return los, his


# --- per-bin results --------------------------------------------------------


def bin_stats(ctx, lo, hi):
    """Yields and significance of one box on the events (the CSV row plus sigma_B terms)."""
    ev = ctx.events
    m = box_mask(ev.axes, lo, hi)
    wS, wB = ev.w[m & ev.is_sig], ev.w[m & ev.is_bkg]
    S, B = float(wS.sum()), float(wB.sum())
    sS, sB = float(np.sqrt((wS ** 2).sum())), float(np.sqrt((wB ** 2).sum()))
    floor = ctx.floor_variance(ev, m)
    sB_z = float(np.sqrt(sB ** 2 + floor)) if floor > 0.0 else sB
    out = {"lo": list(lo), "hi": list(hi), "S": S, "S_err": sS, "S_entries": int((m & ev.is_sig).sum()),
           "B": B, "B_err": sB, "B_entries": int((m & ev.is_bkg).sum()), "floor_variance": floor}
    if ctx.abcd:
        mc_var = float(ev.b2[m].sum())
        abcd_var, c_count = ctx.abcd_terms(ev, lo, hi, m)
        sB_z = float(np.sqrt(mc_var + floor + abcd_var))
        c = ev.control
        cm = box_mask(c["axes"], lo, hi)
        out.update({"QCD": float(ev.q[m].sum()), "sigma_abcd": abcd_var ** 0.5, "sigma_mc_nonqcd": mc_var ** 0.5,
                    "control_count": c_count,
                    "control_count_per_window": [float(c["w"][cm & (c["win"] == k)].sum())
                                                 for k in range(len(ctx.cfg.abcd.windows))]})
    out["sigma_B"] = sB_z
    out["significance"], out["significance_error"] = asimov_z_with_error(S, B, sS, sB_z)
    out["backgrounds"] = {ctx.class_names[c]: float(ev.w[m & (ev.y == c)].sum()) for c in ctx.background_classes}
    return out


# --- driver -----------------------------------------------------------------


def optimise(cfg, inputs, cells=False, threads=None, log=print):
    """Run the optimisation. Returns (bins, report): bins in bin_index order (bin_stats dicts),
    report the JSON-able summary (counts, Z before / after refinement, pool, B&B)."""
    t0 = time.monotonic()
    inputs = inputs.exclude(cfg.exclude_samples, log=log)
    ctx = Ctx(cfg, inputs, cells=cells, threads=threads, log=log)
    report = {"n_scan_events": int(ctx.events.n), "S_total": ctx.S_total, "B_total": ctx.B_total,
              "axes": ctx.axis_names, "cells": bool(cells), "threads": ctx.threads}
    if ctx.abcd:
        win = ctx.events.control["win"]
        report["control_events_per_window"] = [int(np.count_nonzero(win == k)) for k in range(len(cfg.abcd.windows))]
        report["transfer_factors"] = [float(v) for v in ctx.tf]
        log("ABCD-aware sigma_B ON: control windows " + ", ".join(
            f"{lw:g} < {cfg.abcd.branch} < {hw:g} (TF {tf:.4g})" for (lw, hw), tf in zip(cfg.abcd.windows, ctx.tf))
            + f", nonclosure = {cfg.abcd.nonclosure:g}, min_control_weight = {cfg.abcd.min_control_weight:g}")
    if cells:
        report["coarse_cells"] = int(ctx.coarse.n)
        if ctx.abcd:
            report["coarse_control_cells"] = int(len(ctx.coarse.control["w"]))
        log(f"  Coarse cells: {ctx.coarse.n} of {ctx.events.n} events" + (
            f", control {len(ctx.coarse.control['w'])} of {len(ctx.events.control['w'])}" if ctx.abcd else ""))
    log(f"  S_total={ctx.S_total:.4g}, B_total={ctx.B_total:.4g}")
    log(f"  Scan dimensions D={ctx.D}, axes={ctx.axis_names}")
    target_n = cfg.n_signal_regions

    t = time.monotonic()
    passes = []
    candidates, chain = find_regions(ctx, target_n, report=passes)
    report.update({"diversity_passes": passes, "candidate_pool": len(candidates), "chain": len(chain),
                   "coarse_seconds": time.monotonic() - t})
    log(f"  Candidate pool: {len(candidates)}, disjoint chain: {len(chain)}")
    if not chain:
        raise RuntimeError("No signal region found; lower min_bkg_weight or check inputs")

    picked, summary = select_branch_bound(ctx, candidates, target_n, incumbent=chain)
    report["branch_bound"] = summary
    report["coarse_regions"] = [{"lo": it["lo"], "hi": it["hi"], "S": it["S"], "B": it["B"], "Z": it["Z"]}
                                for it in picked]
    report["Z_comb_before_refinement"] = float(np.sqrt(summary["objective_sum_z2"]))

    t = time.monotonic()
    los, his = fine_refine(ctx, [it["lo"] for it in picked], [it["hi"] for it in picked])
    report["fine_seconds"] = time.monotonic() - t
    for a in range(len(los)):
        for b in range(a + 1, len(los)):
            if overlap(los[a], his[a], los[b], his[b]):
                raise RuntimeError(f"Selected signal regions overlap ({a + 1},{b + 1})")

    bins = []
    for k, (lo, hi) in enumerate(zip(los, his)):
        b = {"bin_index": k + 1, **bin_stats(ctx, lo, hi)}
        bins.append(b)
        if ctx.abcd:
            log(f"  Bin {k + 1} ABCD: QCD={b['QCD']:.4g}, sigma_ABCD={b['sigma_abcd']:.4g}, "
                f"sigma_MC(non-QCD)={b['sigma_mc_nonqcd']:.4g}, sigma_B={b['sigma_B']:.4g}; "
                f"control count={b['control_count']:.4g}")
        log(f"  Bin {k + 1}: Z={b['significance']:.4f}+/-{b['significance_error']:.4f}, "
            f"S={b['S']:.4g}+/-{b['S_err']:.4g}, B={b['B']:.4g}+/-{b['B_err']:.4g}")
    sum_z2 = float(sum(b["significance"] ** 2 for b in bins))
    report["Z_comb"] = float(np.sqrt(sum_z2))
    report["sum_z2"] = sum_z2
    report["seconds"] = time.monotonic() - t0
    log(f"  Selected {len(bins)} signal regions, sum(Z^2)={sum_z2:.6g}, Z_comb={report['Z_comb']:.4f}, "
        f"nodes={summary['nodes']}")
    report["bins"] = bins
    return bins, report


CSV_BASE = ("bin_index", "significance", "significance_error", "S", "S_err", "S_entries", "B", "B_err", "B_entries")


def write_csv(path, bins, axis_names):
    """Legacy signal_region.csv (read by vvv.ml.inference.load_signal_regions)."""
    import csv

    cols = list(CSV_BASE) + [f"{a}_{s}" for a in axis_names for s in ("low", "high")]
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(cols)
        for b in bins:
            row = [b[c] for c in CSV_BASE]
            for d in range(len(axis_names)):
                row += [float(b["lo"][d]), float(b["hi"][d])]
            wr.writerow([repr(v) if isinstance(v, float) else v for v in row])
