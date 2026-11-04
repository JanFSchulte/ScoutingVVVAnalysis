"""Diagnostic plots of finished training folds (vvv.ml.plots), written into each fold directory.

train_fold makes them at the end of every training; this redoes them, e.g. for folds trained
before the plots existed or after a plotting failure. For each fold with <category>_model.json
it reads the fold's config.json, rebuilds the fold's data with load_split_data, loads both
models and writes the PDFs. Folds whose plots exist are skipped unless --overwrite. If the
fold's training_log.json has no loss components (folds trained before they were recorded),
the loss history is recomputed from the model every --loss-every rounds and stored there.
CPU only. Also writes plot_summary.json (AUCs, decorrelation correlations) per fold.

Usage: python scripts/plot_training.py OUTPUT_DIR [--categories ...] [--folds ...] [--jobs 5] [--overwrite]
"""

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def plot_one(fold_dir, category, fold, loss_every, max_events):
    import resource

    import xgboost as xgb

    from vvv.ml.plots import loss_history_from_models, plot_fold
    from vvv.ml.train import load_split_data

    t0 = time.time()
    cfg = json.loads((fold_dir / "config.json").read_text())
    log_path = fold_dir / "training_log.json"
    log = json.loads(log_path.read_text())
    stage1, final = xgb.Booster(), xgb.Booster()
    stage1.load_model(str(fold_dir / f"{category}_model_stage1.json"))
    final.load_model(str(fold_dir / f"{category}_model.json"))
    for b in (stage1, final):
        b.set_param({"device": "cpu", "nthread": 16})
    data = load_split_data(cfg, category, fold)
    if "loss_components" not in log:
        log["loss_components"] = loss_history_from_models(final, stage1.num_boosted_rounds(), data, cfg, category,
                                                          log.get("decor_scale", 0.0), every=loss_every)
        log_path.write_text(json.dumps(log, indent=1, default=float))
    written, summary = plot_fold(fold_dir, category, cfg, stage1, final, data, log, max_events=max_events)
    (fold_dir / "plot_summary.json").write_text(json.dumps(summary, indent=1))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return f"{category} fold{fold}: {len(written)} files in {time.time() - t0:.0f} s, peak RSS {peak:.1f} GB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", type=Path)
    p.add_argument("--categories", nargs="*")
    p.add_argument("--folds", nargs="*", type=int)
    p.add_argument("--jobs", type=int, default=5, help="folds in parallel (each needs up to ~30 GB)")
    p.add_argument("--loss-every", type=int, default=50)
    p.add_argument("--max-events", type=int, default=None, help="events per split (default vvv.ml.plots.MAX_PLOT_EVENTS)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from vvv.ml.plots import MAX_PLOT_EVENTS

    todo = []
    for model in sorted(args.output_dir.glob("*/fold*/*_model.json")):
        fold_dir, category = model.parent, model.parent.parent.name
        fold = int(fold_dir.name[len("fold"):])
        if model.name != f"{category}_model.json":
            continue
        if (args.categories and category not in args.categories) or (args.folds is not None and fold not in args.folds):
            continue
        if not args.overwrite and (fold_dir / "plot_summary.json").exists():
            print(f"{category} fold{fold}: plots exist, skipped")
            continue
        todo.append((fold_dir, category, fold))
    print(f"{len(todo)} folds to plot", flush=True)
    failed = 0
    with ProcessPoolExecutor(max(1, args.jobs)) as pool:
        futures = {pool.submit(plot_one, d, c, f, args.loss_every, args.max_events or MAX_PLOT_EVENTS): (c, f)
                   for d, c, f in todo}
        for fut, (c, f) in futures.items():
            try:
                print(fut.result(), flush=True)
            except Exception:  # noqa: BLE001 - report it, go on with the other folds
                failed += 1
                print(f"{c} fold{f}: FAILED\n{traceback.format_exc()}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
