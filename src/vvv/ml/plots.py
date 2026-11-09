"""Diagnostic plots of one trained fold (port of legacy plot_results / plot_branch_distributions).

Written into the fold directory with the legacy file names. Model plots come twice, `_cls`
for the stage-1 model and `_decorr` for the final (decorrelated) one:

- roc_<signal>, score_<signal>_vs_<background>, importance (gain): full evaluation
  thresholds (the decorrelate-variable windows, e.g. msoftdrop_1 in 50-150, on top of the
  training thresholds), weights re-balanced per class after that cut, as legacy eval_splits;
- decor_corr_{train,test}: weighted Pearson r of each class's own probability on its events,
  and of every other class probability on QCD events, with the standardised decorrelate
  variables; decor_score_vs_branch and decor_branch_shapes_by_signal_score (multi-page):
  test split at training thresholds, decorrelate variables clipped only;
- once: feature_corr (train, full thresholds), loss_{classification,mlogloss,total,decorrelation}
  (train and the monitor fold, "Eval" in legacy), branches/<branch>[_by_sample].pdf (train +
  test, full thresholds, clipped, before the log transform).

Differences from legacy: ROC curves and AUCs are computed here (no scikit-learn), each split
is subsampled to MAX_PLOT_EVENTS events (fixed seed), and the train-split diagnostics use the
class-balanced weights before the training-weight cap, as legacy. The loss curves come from
training_log.json "loss_components" (recorded every metric_eval_every rounds); for folds
trained before it was recorded, loss_history_from_models recomputes them from the saved model.
"""

from pathlib import Path

import numpy as np

MAX_PLOT_EVENTS = 3_000_000
EPS = 1e-12
DECOR_EFFICIENCIES = (1.0, 0.5, 0.1, 0.01)


def _slugify(text):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def signal_classes(class_groups):
    """Indices of the classes whose samples are all signal (legacy CLASS_TYPES "single")."""
    from vvv.samples import load_samples

    catalogue = load_samples()
    return [i for i, members in enumerate(class_groups.values())
            if members and all(catalogue[s].kind == "signal" for s in members if s in catalogue)]


def _subsample_split(d, max_events, seed):
    n = len(d["y"])
    if not max_events or n <= max_events:
        return d
    idx = np.sort(np.random.default_rng(seed).choice(n, size=max_events, replace=False))
    return {k: ({c: v[idx] for c, v in val.items()} if isinstance(val, dict) else val[idx]) for k, val in d.items()}


def _weighted_pearson(x, y, w):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.abs(np.asarray(w, float))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[m], y[m], w[m]
    sw = w.sum()
    if sw <= EPS:
        return 0.0
    x0, y0 = x - (w * x).sum() / (sw + EPS), y - (w * y).sum() / (sw + EPS)
    cov, vx, vy = ((w * x0 * y0).sum(), (w * x0 * x0).sum(), (w * y0 * y0).sum())
    return float(cov / sw / (np.sqrt(vx / sw * vy / sw) + EPS))


def weighted_roc(y_bin, score, w):
    """(fpr, tpr, auc) of a weighted binary ROC, ties grouped (as sklearn roc_curve)."""
    order = np.argsort(-score, kind="mergesort")
    s, yb, ww = score[order], y_bin[order], w[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(ww * yb)[last]
    fps = np.cumsum(ww * (1 - yb))[last]
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def _weighted_bin_average(x, y, w, min_bins=200, max_bins=600):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if x.size < 2 or not x.max() > x.min():
        return np.array([]), np.array([])
    n_bins = max(2, min(min(max_bins, max(min_bins, int(np.sqrt(x.size)))), x.size))
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    sw, _ = np.histogram(x, bins=edges, weights=w)
    swy, _ = np.histogram(x, bins=edges, weights=w * y)
    ok = sw > 0
    return (0.5 * (edges[:-1] + edges[1:]))[ok], swy[ok] / sw[ok]


def _weighted_score_threshold(score, w, efficiency):
    m = np.isfinite(score) & np.isfinite(w) & (w > 0)
    score, w = score[m], w[m]
    if score.size == 0:
        return float("nan")
    if efficiency >= 1.0:
        return -float("inf")
    order = np.argsort(score)
    idx = int(np.searchsorted(np.cumsum(w[order]), (1.0 - efficiency) * w.sum(), side="left"))
    return float(score[order][min(max(idx, 0), score.size - 1)])


def _format_eff(eff):
    if eff >= 0.999999:
        return "100%"
    return f"{100.0 * eff:.0f}%" if eff >= 0.01 else f"{100.0 * eff:.2g}%"


def _clip_only(values, name, clip_ranges):
    arr = np.asarray(values, dtype=np.float64).copy()
    valid = ~(arr < -990)
    lo, hi = clip_ranges.get(name, (None, None))
    if lo is not None:
        arr[valid & (arr < lo)] = lo
    if hi is not None:
        arr[valid & (arr > hi)] = hi
    return arr


def plot_fold(out, category, cfg, stage1, final, data, log, max_events=MAX_PLOT_EVENTS):
    """Write the legacy diagnostic PDFs of one fold into `out`. Returns the files written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    import xgboost as xgb
    from matplotlib.backends.backend_pdf import PdfPages

    from vvv.ml.train import _balance, standardize, threshold_mask

    # Legacy style.
    plt.style.use(hep.style.CMS)
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["mathtext.rm"] = "serif"

    out = Path(out)
    cc = cfg["categories"][category]
    features, decor = cc["features"], cc["decorrelate"]
    class_names = list(cfg["class_groups"])
    n_classes = len(class_names)
    sig_idx = signal_classes(cfg["class_groups"])
    bkg_idx = [i for i in range(n_classes) if i not in sig_idx]
    qcd_idx = [i for i, c in enumerate(class_names) if "qcd" in c.lower()]
    palette = plt.colormaps["tab10"].resampled(max(n_classes, 2))(np.arange(max(n_classes, 2)))
    written = []

    def savefig(stem, fig, tight=True):
        path = out / f"{stem}.pdf"
        if tight:
            fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    # Splits: training thresholds (decor diagnostics) and full thresholds (eval plots).
    full_cuts = {k: v for k, v in cc["thresholds"].items() if k in decor}
    splits = {}
    for i, name in enumerate(("train", "test")):
        d = _subsample_split(data[name], max_events, 42 + i)
        w = np.asarray(d.get("w_diag", d["w"]), dtype=np.float64)
        X = standardize(d["X"], features + decor, cc["clip_ranges"], cc["log_transform"])
        m = threshold_mask({k: d["X"][k] for k in full_cuts}, full_cuts) if full_cuts else np.ones(len(w), bool)
        splits[name] = {"X": X, "y": d["y"], "w": w, "raw": d["X"], "sample": d["sample"], "eval": m,
                        "w_eval": _balance(w[m], d["y"][m], n_classes)}
    n_f = len(features)
    tr, te = splits["train"], splits["test"]
    decor_raw_test = {b: _clip_only(te["raw"][b], b, cc["clip_ranges"]) for b in decor}

    def roc_pairs():
        if sig_idx and bkg_idx:
            return [(s, b) for s in sig_idx for b in bkg_idx], [(s, bkg_idx) for s in sig_idx]
        return ([(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)],
                [(i, [j for j in range(n_classes) if j != i]) for i in range(n_classes)])

    pairs, groups = roc_pairs()

    def heatmap(matrix, row_labels, col_labels, stem, aspect, annotate, cbar_label=None):
        fig, ax = plt.subplots(figsize=(max(6.5, 0.55 * len(col_labels) + 4.0), max(5.0, 0.48 * len(row_labels) + 3.0)))
        mesh = ax.pcolormesh(np.arange(matrix.shape[1] + 1), np.arange(matrix.shape[0] + 1), matrix, cmap="bwr",
                             vmin=-1.0, vmax=1.0, shading="flat", edgecolors="white", linewidth=0.35, antialiased=False)
        ax.set_xlim(0, matrix.shape[1])
        ax.set_ylim(0, matrix.shape[0])
        ax.invert_yaxis()
        ax.set_aspect(aspect)
        ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
        ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
        ax.set_xticklabels(col_labels, rotation=90 if aspect == "equal" else 45,
                           ha="center" if aspect == "equal" else "right", fontsize=10)
        ax.set_yticklabels(row_labels, fontsize=10)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        if cbar_label:
            cbar.set_label(cbar_label)
        if annotate:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j + 0.5, i + 0.5, f"{matrix[i, j]:+.2f}", ha="center", va="center",
                            color="white" if abs(matrix[i, j]) > 0.5 else "black", fontsize=10)
        savefig(stem, fig)

    def multipage(stem):
        path = out / f"{stem}.pdf"
        written.append(path)
        return PdfPages(path)

    def plot_model(booster, suffix, stage_tag):
        p_tr = booster.predict(xgb.DMatrix(tr["X"][:, :n_f], feature_names=features)).reshape(-1, n_classes)
        p_te = booster.predict(xgb.DMatrix(te["X"][:, :n_f], feature_names=features)).reshape(-1, n_classes)
        pe_tr, pe_te = p_tr[tr["eval"]], p_te[te["eval"]]
        ye_tr, ye_te = tr["y"][tr["eval"]], te["y"][te["eval"]]
        we_tr, we_te = tr["w_eval"], te["w_eval"]
        summary = {}

        def pair_score(p, s, b):
            return p[:, s] / np.clip(p[:, s] + p[:, b], EPS, None)

        # ROC curves: signal efficiency vs background efficiency (log), test solid, train dashed.
        for s, bkgs in groups:
            fig, ax = plt.subplots(figsize=(10, 10))
            any_curve = False
            for b in bkgs:
                for tag, p, y, w, style in (("Test", pe_te, ye_te, we_te, "-"), ("Train", pe_tr, ye_tr, we_tr, "--")):
                    m = (y == s) | (y == b)
                    yb = (y[m] == s).astype(float)
                    if yb.sum() == 0 or yb.sum() == len(yb):
                        continue
                    fpr, tpr, auc = weighted_roc(yb, pair_score(p, s, b)[m], w[m])
                    ax.plot(tpr, fpr, color=palette[b], linestyle=style, label=f"{tag} vs {class_names[b]} AUC={auc:.3f}")
                    summary[f"auc_{tag.lower()}_{class_names[s]}_vs_{class_names[b]}"] = auc
                    any_curve = True
            if not any_curve:
                plt.close(fig)
                continue
            ax.set_xlabel(rf"$\epsilon_{{\rm {class_names[s]}}}$", fontsize=20)
            ax.set_ylabel(r"$\epsilon_{\rm bkg}$", fontsize=20)
            ax.set_yscale("log")
            ax.set_ylim(1e-6, 1)
            ax.set_xlim(0, 1)
            ax.legend(loc="lower right", fontsize=12)
            savefig(f"roc_{_slugify(class_names[s])}{suffix}", fig)

        # Feature importance (gain), log scale.
        gain = booster.get_score(importance_type="gain")
        imp = np.array([float(gain.get(f, gain.get(f"f{i}", 0.0))) for i, f in enumerate(features)])
        pos = imp > 0 if np.any(imp > 0) else np.ones_like(imp, bool)
        names = [features[i] for i in np.flatnonzero(pos)]
        vals = imp[pos]
        order = np.argsort(vals)
        names, vals = [names[i] for i in order], vals[order]
        longest = max((len(n) for n in names), default=10)
        fig, ax = plt.subplots(figsize=(max(7.0, min(13.0, 4.8 + 0.055 * longest)), max(4.0, 0.24 * len(names) + 1.4)))
        ax.barh(np.arange(len(names)), np.maximum(vals, 1e-12), color="steelblue", edgecolor="none", alpha=0.9)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_title(f"{category} Feature Importance [{stage_tag}]", fontsize=16)
        ax.set_xlabel("Gain", fontsize=12)
        if np.any(vals > 0):
            ax.set_xscale("log")
            ax.set_xlim(max(vals[vals > 0].min() / 2.0, 1e-12), vals.max() * 2.0)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)
        fig.subplots_adjust(left=min(0.45, max(0.16, 0.0065 * longest + 0.06)), right=0.98, top=0.94, bottom=0.08)
        savefig(f"importance{suffix}", fig, tight=False)

        # Score distributions of every (signal, background) pair.
        bins = np.linspace(0, 1, 31)
        for s, b in pairs:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(0, 1)
            for p, y, w, cls, kw in (
                (pe_tr, ye_tr, we_tr, b, {"histtype": "bar", "alpha": 0.5, "label": f"Train {class_names[b]}"}),
                (pe_tr, ye_tr, we_tr, s, {"histtype": "bar", "alpha": 0.5, "label": f"Train {class_names[s]}"}),
                (pe_te, ye_te, we_te, b, {"histtype": "step", "linewidth": 2, "color": "lime", "label": f"Test {class_names[b]}"}),
                (pe_te, ye_te, we_te, s, {"histtype": "step", "linewidth": 2, "color": "red", "label": f"Test {class_names[s]}"}),
            ):
                m = y == cls
                if np.any(m) and w[m].sum() > 0:
                    ax.hist(pair_score(p, s, b)[m], bins=bins, weights=w[m], density=True, **kw)
            ax.set_xlabel("BDT Score")
            ax.set_yscale("log")
            ax.set_ylim(1e-2)
            ax.set_ylabel("Density")
            ax.legend()
            savefig(f"score_{_slugify(class_names[s])}_vs_{_slugify(class_names[b])}{suffix}", fig)

        if not decor:
            return summary
        # Correlation of the scores with the decorrelate variables. Rows: each class's own
        # probability on its events, and on QCD events every other class probability.
        rows = [(c, c) for c in range(n_classes)] + [(q, a) for q in qcd_idx for a in range(n_classes) if a != q]
        row_labels = [class_names[c] if a == c else f"{class_names[c]}: p({class_names[a]})" for c, a in rows]
        for tag, sp, p in (("train", tr, p_tr), ("test", te, p_te)):
            R = np.zeros((len(rows), len(decor)))
            for r, (c, a) in enumerate(rows):
                m = sp["y"] == c
                if np.any(m):
                    for j in range(len(decor)):
                        R[r, j] = _weighted_pearson(sp["X"][m, n_f + j], p[m, a], sp["w"][m])
            heatmap(R, row_labels, decor, f"decor_corr_{tag}{suffix}", "auto", True, "weighted Pearson r")
            summary[f"decor_corr_{tag}"] = {lab: dict(zip(decor, map(float, R[i]))) for i, lab in enumerate(row_labels)}

        # Score vs decorrelate variable (test): QCD events every class probability, others their own.
        with multipage(f"decor_score_vs_branch{suffix}") as pdf:
            for c, cname in enumerate(class_names):
                cm = te["y"] == c
                if not np.any(cm):
                    continue
                for a in (range(n_classes) if c in qcd_idx else [c]):
                    for b in decor:
                        x, s = decor_raw_test[b], p_te[:, a]
                        m = cm & np.isfinite(x) & (x > -990.0) & np.isfinite(s) & (te["w"] > 0)
                        if not np.any(m):
                            continue
                        fig, ax = plt.subplots(figsize=(8.5, 6.2))
                        ax.scatter(x[m], s[m], s=2.0, alpha=0.12, color=palette[c], edgecolors="none",
                                   rasterized=True, label=f"Test {cname}")
                        cx, avg = _weighted_bin_average(x[m], s[m], te["w"][m])
                        if cx.size:
                            ax.plot(cx, avg, color="black", linewidth=1.8, label="weighted average")
                        ax.set_title(f"{category} {stage_tag}: {cname}", fontsize=15)
                        ax.set_xlabel(b)
                        ax.set_ylabel(f"p({class_names[a]})")
                        ax.set_ylim(0.0, 1.0)
                        ax.grid(True, linestyle="--", alpha=0.35)
                        ax.legend(loc="best", fontsize=10)
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

        # Shapes of the decorrelate variables after cuts of given signal efficiency (test).
        effs = cc.get("decor_efficiencies", DECOR_EFFICIENCIES)
        colors = plt.colormaps["viridis"].resampled(max(len(effs), 2))(np.arange(max(len(effs), 2)))
        with multipage(f"decor_branch_shapes_by_signal_score{suffix}") as pdf:
            for c, cname in enumerate(class_names):
                cm = te["y"] == c
                if not np.any(cm):
                    continue
                for b in decor:
                    x = decor_raw_test[b]
                    base_x = cm & np.isfinite(x) & (x > -990.0) & (te["w"] > 0)
                    if base_x.sum() < 2 or not x[base_x].max() > x[base_x].min():
                        continue
                    edges = np.linspace(x[base_x].min(), x[base_x].max(), 61)
                    for s in sig_idx:
                        score = p_te[:, s]
                        base = base_x & np.isfinite(score)
                        fig, ax = plt.subplots(figsize=(8.5, 6.2))
                        for k, eff in enumerate(effs):
                            if eff >= 0.999999:
                                cut, label = base, f"eff={_format_eff(eff)} (no cut)"
                            else:
                                thr = _weighted_score_threshold(score[base], te["w"][base], eff)
                                cut, label = base & (score > thr), f"eff={_format_eff(eff)}, p({class_names[s]})>{thr:.3f}"
                            if np.any(cut) and te["w"][cut].sum() > 0:
                                ax.hist(x[cut], bins=edges, weights=te["w"][cut], density=True, histtype="step",
                                        linewidth=2.0, color=colors[k], label=label)
                        ax.set_title(f"{category} {stage_tag}: {cname}, cut on p({class_names[s]})", fontsize=15)
                        ax.set_xlabel(b)
                        ax.set_ylabel("A.U.")
                        ax.grid(True, linestyle="--", alpha=0.35)
                        ax.legend(loc="best", fontsize=9)
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
        return summary

    summary = {"stage1": plot_model(stage1, "_cls", "stage1")}
    rounds1 = stage1.num_boosted_rounds()
    if final.num_boosted_rounds() > rounds1:
        summary["stage2"] = plot_model(final, "_decorr", "stage2")

    # Loss curves (train and monitor fold), stage boundary dotted.
    history = log.get("loss_components")
    if history:
        for key, ylabel, stem in (("classification", "classification_loss", "loss_classification"),
                                  ("mlogloss", "mlogloss", "loss_mlogloss"),
                                  ("total", "total_loss", "loss_total"),
                                  ("decorrelation", "decorrelation_loss", "loss_decorrelation")):
            fig, ax = plt.subplots(figsize=(8, 5))
            n_rounds, top = 1, 0.0
            for split, label in (("train", "Train"), ("monitor", "Eval")):
                h = history.get(split)
                if not h or not h.get(key):
                    continue
                vals = np.asarray(h[key], dtype=float)
                ax.plot(h["round"], vals, label=label)
                n_rounds = max(n_rounds, max(h["round"]))
                if np.any(np.isfinite(vals)):
                    top = max(top, float(np.nanmax(vals)))
            if 0 < rounds1 < n_rounds:
                ax.axvline(rounds1 + 0.5, color="gray", linestyle=":", alpha=0.7, label="stage 2 start")
            ax.set_xlabel("Boosting Round")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            ax.set_xlim(1, n_rounds)
            ax.set_ylim(0.0, (top if top > 0 else 1.0) * 1.05)
            savefig(stem, fig)

    # Correlations of the (standardised) features, train split at full thresholds.
    Xf = tr["X"][tr["eval"]][:, :n_f]
    ok = np.nanstd(Xf, axis=0) > 0
    corr = np.corrcoef(Xf[:, ok], rowvar=False)
    names = [f for f, k in zip(features, ok) if k]
    heatmap(np.atleast_2d(corr), names, names, "feature_corr", "equal", False)

    written += plot_branch_distributions(out, cfg, category, splits)
    return written, summary


def plot_branch_distributions(out, cfg, category, splits, n_bins=200):
    """branches/<branch>.pdf (per class) and <branch>_by_sample.pdf: train + test, full
    thresholds, clip_ranges applied, before the log transform."""
    import matplotlib.pyplot as plt

    cc = cfg["categories"][category]
    out_dir = Path(out) / "branches"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(cfg["class_groups"])
    parts = [splits["train"], splits["test"]]
    y = np.concatenate([p["y"][p["eval"]] for p in parts])
    w = np.concatenate([p["w_eval"] for p in parts])
    s = np.concatenate([p["sample"][p["eval"]] for p in parts])
    class_palette = plt.colormaps["tab10"].resampled(max(len(class_names), 2))(np.arange(max(len(class_names), 2)))
    ordered = [x for members in cfg["class_groups"].values() for x in members if np.any(s == x)]
    colors = [c for cm in ("tab20", "tab20b", "tab20c") for c in plt.colormaps[cm].colors]
    written = []
    for col in cc["features"] + cc["decorrelate"]:
        v = np.concatenate([_clip_only(p["raw"][col][p["eval"]], col, cc["clip_ranges"]) for p in parts])
        valid = v > -990
        if not np.any(valid) or not v[valid].max() > v[valid].min():
            continue
        lo, hi = float(v[valid].min()), float(v[valid].max())
        bins = np.linspace(lo, hi, n_bins + 1)
        for stem, groups, figsize, lw in ((col, [(class_names[i], y == i, class_palette[i]) for i in range(len(class_names))], (8, 6), 2),
                                          (f"{col}_by_sample", [(x, s == x, colors[i % len(colors)]) for i, x in enumerate(ordered)], (9, 6), 1.5)):
            fig, ax = plt.subplots(figsize=figsize)
            plotted = False
            for label, m, color in groups:
                m = valid & m
                if np.any(m) and w[m].sum() > 0:
                    ax.hist(v[m], bins=bins, weights=w[m], density=True, histtype="step", linewidth=lw, color=color, label=label)
                    plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xlim(lo, hi)
            ax.set_xlabel(col)
            ax.set_ylabel("A.U.")
            ax.legend(**({"fontsize": 8, "ncol": 2, "loc": "best"} if stem.endswith("_by_sample") else {}))
            fig.tight_layout()
            path = out_dir / f"{stem}.pdf"
            fig.savefig(path)
            plt.close(fig)
            written.append(path)
    return written


def loss_history_from_models(final, rounds1, data, cfg, category, scale, every=50, device="cpu"):
    """Legacy loss history (train and monitor fold) recomputed from the saved model every
    `every` rounds, for folds trained before training_log.json recorded it."""
    import xgboost as xgb

    from vvv.ml.train import DecorState, cross_entropy_sum, standardize

    cc = cfg["categories"][category]
    features, decor = cc["features"], cc["decorrelate"]
    n_classes = len(cfg["class_groups"])
    qcd = [i for i, c in enumerate(cfg["class_groups"]) if "qcd" in c.lower()]
    lam = cc.get("decor_lambda", cfg.get("decor_lambda", 0))
    n_total = final.num_boosted_rounds()
    grid = sorted(set(list(range(every, n_total + 1, every)) + [rounds1, n_total]))
    history = {}
    for split in ("train", "monitor"):
        d = data[split]
        X = standardize(d["X"], features + decor, cc["clip_ranges"], cc["log_transform"])
        dm = xgb.DMatrix(X[:, : len(features)], feature_names=features)
        dec = DecorState(X[:, len(features):], d["w"], np.isin(d["y"], qcd), cfg, device) if decor and qcd and scale else None
        h = {"round": [], "classification": [], "mlogloss": [], "decorrelation": [], "total": []}

        def partial(a, b):
            return final.predict(dm, output_margin=True, iteration_range=(a, b)).reshape(-1, n_classes)

        # Margins accumulate over round ranges; every partial prediction carries the base score once.
        head = dm.slice(np.arange(min(1000, dm.num_row())))
        base = (final.predict(head, output_margin=True, iteration_range=(0, 1))
                + final.predict(head, output_margin=True, iteration_range=(1, 2))
                - final.predict(head, output_margin=True, iteration_range=(0, 2))).reshape(-1, n_classes).mean(axis=0)
        margin, prev = None, 0
        for r in grid:
            margin = partial(0, r) if margin is None else margin + partial(prev, r) - base
            prev = r
            ce = cross_entropy_sum(margin, d["y"], d["w"])
            dl = lam * scale * dec(margin, with_grad=False)[0] if (dec is not None and r > rounds1) else 0.0
            for k, v in (("round", r), ("classification", ce), ("mlogloss", ce / d["w"].sum()), ("decorrelation", dl), ("total", ce + dl)):
                h[k].append(float(v))
        history[split] = h
    return history
