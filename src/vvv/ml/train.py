"""K-fold multiclass BDT training with mass decorrelation (port of legacy selections/BDT/train.py).

One call of train_fold trains one (category, fold) on one GPU:

1. Data: the category's Parquet skims of every sample in the class groups.
   Fold of an event = event % K. Rotation r: test = fold r, monitor = fold
   (r - 1) % K, train = the other K - 2 folds.
2. Weights, as legacy: per sample the physics weight scaled to the full
   sample (x N_sample / n_read), made positive with the sample's signed total
   kept, training thresholds applied, every class balanced to N_pass / n_classes,
   then capped at 20 x the class mean with the class total kept (train and monitor).
3. Stage 1: multi:softprob, monitored every 5 rounds on the monitor fold
   (weighted cross-entropy sum); the learning rate halves at every check that
   does not improve, down to the floor, after which 4 misses stop the training.
   The booster is cut at the best round.
4. Stage 2: continues from stage 1 with a custom objective, cross-entropy +
   lambda * decor_scale * smooth Cramer-von Mises penalty of every class
   score's log-odds against the mass variables, on QCD events, with decor_scale
   = (stage-1 train cross-entropy + regularisation) / penalty. The penalty and
   its gradient run in torch on the GPU.

Deliberate differences from legacy: folds by event number (legacy: contiguous
entry blocks), random instead of contiguous per-sample subsampling, and a GPU
'hist' tree builder (legacy ran CPU hist; XGBoost 1.7 ignored device=cuda).
"""

import json
import time
import zlib
from pathlib import Path

import numpy as np

SENTINEL = -990.0
EPS = 1e-12
HESS_FLOOR = 1e-6
EFF_NORM_FLOOR = 1e-4


# --- data -------------------------------------------------------------------


def _read_sample(skim_dir, sample, category, columns):
    import pyarrow.dataset as ds

    path = Path(skim_dir) / sample / category / "nominal"
    if not path.exists():
        return None
    table = ds.dataset(str(path), format="parquet").to_table(columns=columns)
    return {c: table.column(c).to_numpy() for c in columns}


def threshold_mask(columns, thresholds):
    from vvv.ml.inference import threshold_mask as legacy_mask

    return legacy_mask(columns, thresholds, apply_to_sentinel=True)


def _water_fill(budget, caps, shares):
    """Split budget in proportion to shares, capping each at caps (legacy _apply_class_budget)."""
    alloc = np.zeros(len(caps))
    free = np.ones(len(caps), dtype=bool)
    remaining = float(budget)
    while remaining > 0 and free.any():
        share = np.where(free, shares, 0.0)
        if share.sum() <= 0:
            break
        trial = remaining * share / share.sum()
        over = free & (alloc + trial >= caps)
        if not over.any():
            alloc += trial
            break
        remaining -= (caps[over] - alloc[over]).sum()
        alloc[over] = caps[over]
        free &= ~over
    return np.floor(alloc).astype(np.int64)


def _subsample(n, k, seed_key):
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng([42, zlib.crc32(seed_key.encode())])
    return np.sort(rng.choice(n, size=k, replace=False))


def load_split_data(cfg, category, fold):
    """Train / monitor / test arrays of one rotation, with legacy weights and labels."""
    cc = cfg["categories"][category]
    k = cfg["kfold"]
    features = cc["features"]
    decor = cc["decorrelate"]
    thresholds = cc["thresholds"]
    train_thresholds = {}
    for name, cond in thresholds.items():
        if name in decor:
            continue
        train_thresholds[name] = cond
    for name in decor:
        if "msoftdrop" in name:
            train_thresholds[name] = [0, cfg["decor_msoftdrop_training_max"]]
    columns = sorted(set(features) | set(decor) | set(thresholds) | {"event", "weight"})
    classes = list(cfg["class_groups"])

    splits = {"train": [], "monitor": [], "test": []}
    for label, cls in enumerate(classes):
        samples = [s for s in cfg["class_groups"][cls] if s not in cc.get("exclude_samples", [])]
        loaded = []
        for sample in samples:
            data = _read_sample(cfg["skim_dir"], sample, category, columns)
            if data is None or len(data["event"]) == 0:
                print(f"  {category}: no skim entries for {sample}, skipped", flush=True)
                continue
            loaded.append((sample, data))
        if not loaded:
            raise RuntimeError(f"{category}: class {cls} has no training events")

        for split, wanted in (("train", None), ("monitor", (fold - 1) % k), ("test", fold)):
            per_sample = []
            for sample, data in loaded:
                folds = (data["event"].astype(np.uint64) % np.uint64(k)).astype(np.int64)
                sel = np.flatnonzero(folds == wanted) if wanted is not None else np.flatnonzero(
                    (folds != fold) & (folds != (fold - 1) % k)
                )
                per_sample.append((sample, data, sel))
            # Per-sample entry caps; a class budget is water-filled over its samples.
            n_avail = np.array([len(sel) for _, _, sel in per_sample], dtype=np.float64)
            if split == "train":
                budget = cc.get("train_events_per_class", {}).get(cls)
                if budget:
                    shares = np.array([max(float(np.sum(np.abs(d["weight"]))), 0.0) for _, d, _ in per_sample])
                    caps = _water_fill(budget, n_avail, shares)
                    caps = np.minimum(np.maximum(caps, np.minimum(8192, n_avail)), n_avail).astype(np.int64)
                else:
                    caps = np.minimum(n_avail, cfg["entries_per_sample"]).astype(np.int64)
            else:
                caps = np.minimum(n_avail, cfg["eval_entries_per_sample"]).astype(np.int64)

            for (sample, data, sel), cap in zip(per_sample, caps):
                if len(sel) == 0:
                    continue
                pick = sel[_subsample(len(sel), int(cap), f"{category}/{sample}/{split}")]
                w = data["weight"][pick].astype(np.float64) * (len(data["event"]) / max(len(pick), 1))
                total, abs_total = w.sum(), np.abs(w).sum()
                if total <= 0:
                    raise RuntimeError(f"{category}/{sample}/{split}: non-positive signed weight sum")
                w = np.abs(w) * (total / abs_total)
                cols = {c: data[c][pick] for c in columns}
                keep = threshold_mask(cols, train_thresholds)
                splits[split].append({
                    "X": {c: cols[c][keep] for c in features + decor},
                    "w": w[keep],
                    "y": np.full(int(keep.sum()), label, dtype=np.int32),
                    "event": cols["event"][keep],
                    "sample": np.full(int(keep.sum()), sample, dtype=object),
                })

    out = {}
    for split, parts in splits.items():
        X = {c: np.concatenate([p["X"][c] for p in parts]) for c in features + decor}
        w = np.concatenate([p["w"] for p in parts])
        y = np.concatenate([p["y"] for p in parts])
        w = _balance(w, y, len(classes))
        if split in ("train", "monitor") and cfg["categories"][category].get("training_weight_cap"):
            w = _cap(w, y, len(classes), cfg["categories"][category]["training_weight_cap"])
        out[split] = {"X": X, "w": w, "y": y,
                      "event": np.concatenate([p["event"] for p in parts]),
                      "sample": np.concatenate([p["sample"] for p in parts])}
    if len(out["train"]["y"]):
        order = np.random.default_rng(42).permutation(len(out["train"]["y"]))
        tr = out["train"]
        out["train"] = {k_: ({c: v[order] for c, v in val.items()} if isinstance(val, dict) else val[order])
                        for k_, val in tr.items()}
    return out


def _balance(w, y, n_classes):
    """Each class scaled to a total of N / n_classes (legacy class balancing after cuts)."""
    target = len(y) / n_classes
    out = w.copy()
    for c in range(n_classes):
        m = y == c
        s = w[m].sum()
        if s > 0:
            out[m] = w[m] * (target / s)
    return out


def _cap(w, y, n_classes, cap):
    """Legacy _cap_training_weights: limit = cap x class mean, class total kept."""
    out = w.copy()
    for c in range(n_classes):
        m = (y == c) & (w > 0)
        if not m.any():
            continue
        wc = w[m]
        total = wc.sum()
        limit = cap * wc.mean()
        if wc.max() <= limit:
            continue
        lo, hi = 1.0, limit / wc.min()
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if np.minimum(mid * wc, limit).sum() < total:
                lo = mid
            else:
                hi = mid
        capped = np.minimum(hi * wc, limit)
        out[m] = capped * (total / capped.sum())
    return out


def standardize(X, features, clip_ranges, log_transform):
    from vvv.ml.inference import standardize as legacy_standardize

    return legacy_standardize(X, features, clip_ranges, log_transform)


# --- decorrelation penalty (torch) ---------------------------------------------


def threshold_grid(tail_min_prob=2e-4, tail_n=10, n_thresholds=15):
    low = np.geomspace(tail_min_prob, 0.02, tail_n + 1)[:-1]
    p = np.unique(np.concatenate([low, np.linspace(0.02, 0.98, n_thresholds), 1.0 - low]))
    return np.log(p / (1.0 - p))


class DecorState:
    """Bin memberships of the QCD events of one split for the smooth CvM penalty."""

    def __init__(self, Z, w, is_qcd, cfg, device):
        import torch

        self.torch = torch
        self.device = device
        # float64 matches legacy numpy; float32 is much faster on T4s (decor_dtype in the config).
        dt = torch.float64 if cfg.get("decor_dtype", "float64") == "float64" else torch.float32
        self.dtype = dt
        n_bins = cfg["decor_n_bins"]
        self.idx = np.flatnonzero(is_qcd)
        wq = torch.as_tensor(w[self.idx], dtype=dt, device=device)
        self.w = wq
        self.total_w = wq.sum()
        self.features = []
        for j in range(Z.shape[1]):
            z_all = Z[:, j]
            edges = np.linspace(z_all.min(), z_all.max(), n_bins + 1)
            tau = cfg["decor_bin_tau_scale"] * (edges[1] - edges[0])
            z = torch.as_tensor(z_all[self.idx], dtype=dt, device=device)[:, None]
            e = torch.as_tensor(edges, dtype=dt, device=device)
            if tau > 0:
                m = torch.sigmoid((z - e[None, :-1]) / tau) * torch.sigmoid((e[None, 1:] - z) / tau)
            else:
                m = torch.zeros(len(self.idx), n_bins, dtype=dt, device=device)
            row = m.sum(dim=1, keepdim=True)
            hard = torch.clamp(torch.bucketize(z[:, 0], e[1:-1]), 0, n_bins - 1)
            hard_m = torch.nn.functional.one_hot(hard, n_bins).to(dt)
            m = torch.where(row > EPS, m / torch.clamp(row, min=EPS), hard_m)
            bin_tot = (wq[:, None] * m).sum(dim=0)
            keep = bin_tot > EPS
            m, bin_tot = m[:, keep], bin_tot[keep]
            self.features.append({
                "wm_t": (wq[:, None] * m).T.contiguous(),  # (B, n)
                "m_over_bt": m / bin_tot[None, :],  # (n, B)
                "rho": bin_tot / self.total_w,
            })
        self.c = torch.as_tensor(threshold_grid(cfg["decor_tail_min_prob"], cfg["decor_tail_n"], cfg["decor_n_thresholds"]),
                                 dtype=dt, device=device)
        self.score_tau = cfg["decor_score_tau"]

    def __call__(self, margins, with_grad=True):
        """(raw penalty, grad (n_qcd, K), hess (n_qcd, K)) for margins of the QCD events."""
        torch = self.torch
        dt = self.dtype
        z = torch.as_tensor(margins[self.idx], dtype=dt, device=self.device)
        n, K = z.shape
        T = float(len(self.c))
        w, W = self.w, self.total_w
        total = 0.0
        grad = torch.zeros_like(z) if with_grad else None
        hess = torch.zeros_like(z) if with_grad else None
        for k in range(K):
            others = torch.cat([z[:, :k], z[:, k + 1:]], dim=1)
            lse = torch.logsumexp(others, dim=1)
            s = z[:, k] - lse
            sig = torch.sigmoid((s[:, None] - self.c[None, :]) / self.score_tau)  # (n, T)
            g = (w[:, None] * sig).sum(dim=0) / W
            var = g * (1.0 - g)
            a = 1.0 / torch.clamp(var, min=EFF_NORM_FLOOR)
            a_d = torch.where(var > EFF_NORM_FLOOR, -(1.0 - 2.0 * g) * a * a, torch.zeros_like(a))
            if with_grad:
                dsig = sig * (1.0 - sig) / self.score_tau
                w_dsig = w[:, None] * dsig
                s_grad = torch.zeros(n, dtype=dt, device=self.device)
                s_hess = torch.zeros(n, dtype=dt, device=self.device)
            for f in self.features:
                local = (f["wm_t"] @ sig) / (f["rho"] * W)[:, None]  # (B, T)
                delta = local - g[None, :]
                wd = f["rho"][:, None] * delta * a[None, :]
                total += float((wd * delta).sum() / T)
                if with_grad:
                    local_term = f["m_over_bt"] @ wd  # (n, T)
                    global_term = wd.sum(dim=0) / W - a_d * (f["rho"][:, None] * delta * delta).sum(dim=0) / (2.0 * W)
                    coeff = local_term - global_term[None, :]
                    s_grad += (2.0 * w_dsig * coeff).sum(dim=1) / T
                    s_hess += (2.0 * (w_dsig * coeff) ** 2).sum(dim=1) / T
            if with_grad:
                q = torch.softmax(others, dim=1)
                q = torch.cat([q[:, :k], torch.zeros(n, 1, dtype=q.dtype, device=q.device), q[:, k:]], dim=1)
                grad -= s_grad[:, None] * q
                grad[:, k] += s_grad
                hess += s_hess[:, None] * q * q
                hess[:, k] += s_hess
        if not with_grad:
            return total, None, None
        return total, grad.double().cpu().numpy(), torch.clamp(hess, min=HESS_FLOOR).double().cpu().numpy()


# --- training ---------------------------------------------------------------------


def softmax(m):
    m = m - m.max(axis=1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_sum(margins, y, w):
    p = softmax(margins)
    return float(np.sum(w * -np.log(p[np.arange(len(y)), y] + EPS)))


def _callback(loss_fn, cfg, cc, lr0, tree_offset, record):
    """Legacy _DetailedLossMonitor: check every 5 rounds, halve lr on a miss, stop at the floor."""
    import xgboost as xgb

    every = cfg["metric_eval_every"]

    class LossMonitor(xgb.callback.TrainingCallback):
        def __init__(self):
            self.lr = lr0
            self.best = np.inf
            self.best_iter = 0
            self.stale = 0
            self.lr_stale = 0

        def after_iteration(self, model, epoch, evals_log):
            record["eta"].append(self.lr)
            if epoch % every:
                return False
            loss = loss_fn(model, tree_offset + epoch + 1)
            record["loss"].append((epoch, loss, self.lr))
            if loss < self.best - 1e-12:
                self.best, self.best_iter, self.stale, self.lr_stale = loss, epoch, 0, 0
                return False
            self.stale += 1
            if self.lr > cc["min_learning_rate"]:
                self.lr_stale += 1
                if self.lr_stale * every >= cc["lr_reduce_patience"]:
                    self.lr = max(self.lr / 2.0, cc["min_learning_rate"])
                    model.set_param("eta", self.lr)
                    self.lr_stale = self.stale = 0
                return False
            return self.stale * every >= cc["early_stopping_rounds"]

    return LossMonitor()


def _regularisation(booster, n_rounds, etas, cc, n_classes):
    """Legacy stage-1 regularisation loss: sum over trees of gamma*leaves + l2/l1 of leaf/eta."""
    model = json.loads(booster.save_raw("json"))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    total = 0.0
    for i, tree in enumerate(trees[: n_rounds * n_classes]):
        eta = etas[i // n_classes]
        left = np.asarray(tree["left_children"])
        leaves = np.asarray(tree["base_weights"])[left == -1] / eta
        total += cc["gamma"] * len(leaves) + 0.5 * cc["reg_lambda"] * np.sum(leaves**2) + cc["reg_alpha"] * np.sum(np.abs(leaves))
    return total


def _device_name(device):
    if device != "cuda":
        return "cpu"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("device cuda requested but no GPU is visible")
    return torch.cuda.get_device_name(0)


def train_fold(cfg, category, fold, device="cuda"):
    import xgboost as xgb

    t0 = time.time()
    cc = cfg["categories"][category]
    out = Path(cfg["output_dir"]) / category / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    classes = list(cfg["class_groups"])
    n_classes = len(classes)
    qcd = [i for i, c in enumerate(classes) if "qcd" in c.lower()]
    features, decor = cc["features"], cc["decorrelate"]

    data = load_split_data(cfg, category, fold)
    log = {"category": category, "fold": fold, "n": {s: int(len(d["y"])) for s, d in data.items()},
           "device": _device_name(device), "xgboost": xgb.__version__}
    print(f"{category} fold{fold}: training on {log['device']}", flush=True)
    print(f"{category} fold{fold}: {log['n']} events after cuts ({time.time() - t0:.0f} s)", flush=True)

    def matrices(split):
        d = data[split]
        X = standardize(d["X"], features + decor, cc["clip_ranges"], cc["log_transform"])
        return X[:, : len(features)], X[:, len(features):], d["y"], d["w"]

    Xtr, Ztr, ytr, wtr = matrices("train")
    Xmo, Zmo, ymo, wmo = matrices("monitor")
    dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr, feature_names=features)
    dmon = xgb.DMatrix(Xmo, label=ymo, weight=wmo, feature_names=features)

    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "base_score": 0.5,
        "max_depth": cc["max_depth"],
        "gamma": cc["gamma"],
        "reg_lambda": cc["reg_lambda"],
        "reg_alpha": cc["reg_alpha"],
        "min_child_weight": cc["min_child_weight"],
        "subsample": cc["subsample"],
        "colsample_bytree": cc["colsample_bytree"],
        "colsample_bynode": cc["colsample_bynode"],
        "tree_method": "hist",
        "device": device,
        "seed": 42,
        "disable_default_eval_metric": 1,
        "eta": cc["learning_rate"],
    }

    def margins(model, d, rounds):
        return model.predict(d, output_margin=True, iteration_range=(0, rounds)).reshape(-1, n_classes)

    # Stage 1
    rec1 = {"eta": [], "loss": []}
    cb1 = _callback(lambda m, r: cross_entropy_sum(margins(m, dmon, r), ymo, wmo), cfg, cc, cc["learning_rate"], 0, rec1)
    stage1 = xgb.train(params, dtrain, num_boost_round=cc["n_estimators"], callbacks=[cb1], verbose_eval=False)
    rounds1 = cb1.best_iter + 1
    stage1 = stage1[:rounds1]
    stage1.save_model(str(out / f"{category}_model_stage1.json"))
    log["stage1"] = {"rounds": rounds1, "best_monitor_loss": cb1.best, "time_s": time.time() - t0}
    print(f"{category} fold{fold}: stage 1 {rounds1} rounds, monitor loss {cb1.best:.6g} ({time.time() - t0:.0f} s)", flush=True)

    final = stage1
    if cfg.get("decor_lambda", 0) > 0 and decor and qcd:
        torch_device = "cuda" if device == "cuda" else "cpu"
        dec_tr = DecorState(Ztr, wtr, np.isin(ytr, qcd), cfg, torch_device)
        dec_mo = DecorState(Zmo, wmo, np.isin(ymo, qcd), cfg, torch_device)
        m_tr = margins(stage1, dtrain, rounds1)
        reg = _regularisation(stage1, rounds1, rec1["eta"], cc, n_classes)
        raw_tr, _, _ = dec_tr(m_tr, with_grad=False)
        scale = (cross_entropy_sum(m_tr, ytr, wtr) + reg) / max(raw_tr, EPS)
        lam = cfg["decor_lambda"]
        log["decor_scale"] = scale
        print(f"{category} fold{fold}: decorrelation scale {scale:.6g} (raw {raw_tr:.6g}, reg {reg:.6g})", flush=True)
        onehot = np.eye(n_classes)[ytr]

        def objective(predt, _dtrain):
            m = predt.reshape(-1, n_classes).astype(np.float64)
            p = softmax(m)
            grad = (p - onehot) * wtr[:, None]
            hess = np.maximum(2.0 * p * (1.0 - p) * wtr[:, None], HESS_FLOOR)
            _, g_dec, h_dec = dec_tr(m)
            grad[dec_tr.idx] += lam * scale * g_dec
            hess[dec_tr.idx] = np.maximum(hess[dec_tr.idx] + lam * scale * h_dec, HESS_FLOOR)
            return grad.astype(np.float32), hess.astype(np.float32)

        def total_loss(model, r):
            m = margins(model, dmon, r)
            return cross_entropy_sum(m, ymo, wmo) + lam * scale * dec_mo(m, with_grad=False)[0]

        rec2 = {"eta": [], "loss": []}
        cb2 = _callback(total_loss, cfg, cc, cc["learning_rate_decorr"], rounds1, rec2)
        params2 = dict(params, eta=cc["learning_rate_decorr"])
        stage2 = xgb.train(params2, dtrain, num_boost_round=cc["n_estimators_decorr"], obj=objective,
                           xgb_model=stage1, callbacks=[cb2], verbose_eval=False)
        final = stage2[: rounds1 + cb2.best_iter + 1]
        log["stage2"] = {"rounds": cb2.best_iter + 1, "best_monitor_loss": cb2.best, "time_s": time.time() - t0}
        print(f"{category} fold{fold}: stage 2 {cb2.best_iter + 1} rounds ({time.time() - t0:.0f} s)", flush=True)
        log["loss_stage2"] = rec2["loss"]
    log["loss_stage1"] = rec1["loss"]

    # Legacy layout, read by vvv.ml.inference.CategoryModel.
    (out / "branch.json").write_text(json.dumps({category: [{"name": f} for f in features]}, indent=1))
    (out / "selection.json").write_text(json.dumps({category: {
        "clip_ranges": cc["clip_ranges"], "log_transform": cc["log_transform"], "thresholds": cc["thresholds"]}}, indent=1))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "training_log.json").write_text(json.dumps(log, indent=1, default=float))
    final.save_model(str(out / f"{category}_model.json"))  # last: marks the fold as done
    return f"{final.num_boosted_rounds()} rounds in {time.time() - t0:.0f} s"
