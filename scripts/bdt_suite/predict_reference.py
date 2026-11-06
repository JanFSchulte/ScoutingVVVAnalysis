"""Test-split predictions of an existing training (e.g. the production fold-0 models) in the
suite layout, OUTPUT/<category>/<name>/, so evaluate.py scores it next to the variants.

Usage: pixi run python scripts/bdt_suite/predict_reference.py MODEL_DIR OUTPUT --packages DIR \\
           --categories c1 c2 [--name production]
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main():
    import xgboost as xgb

    from make_packages import load_package
    from vvv.ml.inference import softmax, standardize

    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--packages", type=Path, required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--name", default="production")
    p.add_argument("--fold", type=int, default=0)
    args = p.parse_args()
    import json

    for category in args.categories:
        fold_dir = args.model_dir / category / f"fold{args.fold}"
        sel = json.loads((fold_dir / "selection.json").read_text())[category]
        booster = xgb.Booster()
        booster.load_model(str(fold_dir / f"{category}_model.json"))
        test = load_package(args.packages / category, splits=("test",))["test"]
        X = standardize(test["X"], booster.feature_names, sel["clip_ranges"], sel["log_transform"])
        m = booster.inplace_predict(X, predict_type="margin").reshape(len(X), -1)
        out = args.output / category / args.name
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "test_proba.npz", proba=softmax(m.astype(np.float64)).astype(np.float32), event=test["event"])
        shutil.copy(fold_dir / "training_log.json", out / "training_log.json")
        print(f"{category}: {out}", flush=True)


if __name__ == "__main__":
    main()
