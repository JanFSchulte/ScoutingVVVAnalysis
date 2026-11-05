"""Training config (JSON) from the legacy production k-fold configs.

Usage: python scripts/make_training_config.py OUT.json --skim-dir DIR --output-dir DIR [--class-scale QCD=2]

--class-scale CLS=F trains on F x more events of a class than legacy: its class budget and,
where it has none, its per-sample cap (entries_per_sample_by_class) are scaled by F.
"""

import argparse
import json
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[2] / "CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/BDT/study_boosted3"
PRODUCTION = {
    "boosted3": "kfold_finedecor_jec_powheg",
    "boosted2_resolved": "kfold_finedecor_jec_powheg_resolved",
    "boosted2_unresolved": "kfold_finedecor_jec_powheg_unresolved",
    "boosted2_leptonic_1lep": "kfold_finedecor_jec_powheg_1lep",
    "boosted2_leptonic_2lep": "kfold_finedecor_jec_powheg_2lep",
}
GLOBAL_KEYS = (
    "kfold", "entries_per_sample", "eval_entries_per_sample", "random_state", "decor_lambda",
    "decor_n_bins", "decor_n_thresholds", "decor_score_tau", "decor_bin_tau_scale",
    "decor_msoftdrop_training_max", "decor_tail_min_prob", "decor_tail_n", "metric_eval_every",
)
TREE_KEYS = (
    "n_estimators", "n_estimators_decorr", "max_depth", "gamma", "learning_rate", "learning_rate_decorr",
    "min_learning_rate", "lr_reduce_patience", "reg_lambda", "reg_alpha", "min_child_weight", "subsample",
    "colsample_bytree", "colsample_bynode", "early_stopping_rounds", "decorrelate", "training_weight_cap",
)


# Not used (a deliberate change from legacy): QCD below HT 200 GeV reaches the selection only through
# pileup, so these samples are neither skimmed nor trained on.
NOT_USED = ("qcd_ht40to70", "qcd_ht70to100", "qcd_ht100to200")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--skim-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--class-scale", action="append", default=[], help="CLS=F trains on F x more of a class")
    args = p.parse_args()
    scales = {c: float(f) for c, f in (x.split("=") for x in args.class_scale)}

    cfg, class_groups = None, None
    categories = {}
    for tree, study in PRODUCTION.items():
        legacy = json.loads((LEGACY / study / "bdt_config.json").read_text())
        if cfg is None:
            cfg = {k: legacy[k] for k in GLOBAL_KEYS}
            class_groups = {c: [x for x in members if x not in NOT_USED] for c, members in legacy["class_groups"].items()}
        elif any(legacy[k] != cfg[k] for k in GLOBAL_KEYS):
            raise ValueError(f"{study}: global settings differ from the other categories")
        fold0 = LEGACY / study / tree / "fold0"
        branches = [b["name"] for b in json.loads((fold0 / "branch.json").read_text())[tree]]
        selection = json.loads((fold0 / "selection.json").read_text())[tree]
        cc = {k: legacy[tree][k] for k in TREE_KEYS}
        cc["features"] = [b for b in branches if b not in cc["decorrelate"]]
        cc["clip_ranges"] = selection.get("clip_ranges", {})
        cc["log_transform"] = selection.get("log_transform", [])
        cc["thresholds"] = selection.get("thresholds", {})
        budget = legacy["train_events_per_class"]
        cc["train_events_per_class"] = budget if isinstance(budget, dict) else {c: budget for c in class_groups}
        for cls, f in scales.items():
            if cls in cc["train_events_per_class"]:
                cc["train_events_per_class"][cls] = int(cc["train_events_per_class"][cls] * f)
            else:
                cc.setdefault("entries_per_sample_by_class", {})[cls] = int(cfg["entries_per_sample"] * f)
        members = {s for v in legacy["class_groups"].values() for s in v}
        cc["exclude_samples"] = sorted({s for v in class_groups.values() for s in v} - members)
        categories[tree] = cc

    cfg.update({
        "skim_dir": args.skim_dir,
        "output_dir": args.output_dir,
        "class_groups": class_groups,
        "categories": categories,
        "source": {tree: str(LEGACY / study) for tree, study in PRODUCTION.items()},
        "class_scale": scales,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cfg, indent=1))
    print(f"wrote {args.out}: {', '.join(f'{t} ({len(c['features'])} features)' for t, c in categories.items())}")


if __name__ == "__main__":
    main()
