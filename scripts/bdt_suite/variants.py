"""BDT configuration variants of the suite: overrides of the production training config.

Each variant is {"cfg": {...}, "cc": {...}, "weight_cap": x, "class_scale": {...}}:
- cfg: top-level config keys (decorrelation settings);
- cc: category keys (XGBoost parameters, rounds, seed);
- weight_cap: replaces the training weight cap (None: no cap), from the uncapped w_diag;
- class_scale: multiplies the train/monitor weights of classes after the cap.
"base" is the production config; seed variants measure the training noise.
"""

VARIANTS = {
    "base": {},
    "seed43": {"cc": {"seed": 43}},
    "seed44": {"cc": {"seed": 44}},
    # both stages stop at their round cap in production
    "s1_3000": {"cc": {"n_estimators": 3000}},
    "s2_3000": {"cc": {"n_estimators_decorr": 3000}},
    "s12_3000": {"cc": {"n_estimators": 3000, "n_estimators_decorr": 3000}},
    "lr0p1": {"cc": {"learning_rate": 0.1, "learning_rate_decorr": 0.1}},
    # capacity
    "depth6": {"cc": {"max_depth": 6}},
    "depth8": {"cc": {"max_depth": 8, "min_child_weight": 100}},
    "mcw10": {"cc": {"min_child_weight": 10}},
    "lessreg": {"cc": {"gamma": 0, "reg_lambda": 1, "reg_alpha": 0}},
    "nosubsample": {"cc": {"subsample": 1.0, "colsample_bytree": 1.0, "colsample_bynode": 1.0}},
    "colsample0p6": {"cc": {"colsample_bytree": 0.6, "colsample_bynode": 0.6}},
    # decorrelation strength (production decor_lambda 2)
    "lam1": {"cfg": {"decor_lambda": 1}},
    "lam4": {"cfg": {"decor_lambda": 4}},
    "lam8": {"cfg": {"decor_lambda": 8}},
    "decor_f32": {"cfg": {"decor_dtype": "float32"}},
    # weights
    "nocap": {"weight_cap": None},
    "cap50": {"weight_cap": 50},
    "vvv_x2": {"class_scale": {"VVV": 2.0}},
    "sig_x2": {"class_scale": {"VVV": 2.0, "VH": 2.0}},
    # the suite runs with loss_device "cuda" (vvv.ml.train.TorchLoss); this one with numpy
    "numpy_loss": {"cfg": {"loss_device": "cpu"}},
    # plumbing test only
    "smoke": {"cc": {"n_estimators": 10, "n_estimators_decorr": 5}},
}


def apply(cfg, category, data, name):
    """(cfg, data) for variant name; inputs are not modified."""
    import copy

    import numpy as np

    from vvv.ml.train import _cap

    v = VARIANTS[name]
    cfg = copy.deepcopy(cfg)
    cfg["loss_device"] = "cuda"
    cfg.update(v.get("cfg", {}))
    cc = cfg["categories"][category]
    cc.update(v.get("cc", {}))
    classes = list(cfg["class_groups"])
    data = dict(data)
    for split in ("train", "monitor"):
        s = dict(data[split])
        if "weight_cap" in v:
            cap = v["weight_cap"]
            cc["training_weight_cap"] = cap
            s["w"] = s["w_diag"].copy() if cap is None else _cap(s["w_diag"], s["y"], len(classes), cap)
        for cls, f in v.get("class_scale", {}).items():
            s["w"] = np.where(s["y"] == classes.index(cls), s["w"] * f, s["w"])
        data[split] = s
    return cfg, data
