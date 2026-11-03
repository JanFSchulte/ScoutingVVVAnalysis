"""Ray entrypoint: train every (category, fold) of a training config on its own GPU.

Runs on the AF Ray cluster (submitted by scripts/submit_training.py). Each
(category, fold) is one Ray task holding one GPU; the driver itself needs none.
Workers past the first can be taken back by the facility, so a task that dies
is retried, and a finished fold (its model file present) is never redone, so
resubmitting the same config resumes the training.

Usage: python -m vvv.ml.ray_driver CONFIG.json [--categories c1 c2 ...] [--folds 0 1 ...]
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def main():
    import ray

    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--categories", nargs="*")
    p.add_argument("--folds", nargs="*", type=int)
    p.add_argument("--cpus-per-task", type=int, default=None)
    p.add_argument("--local", action="store_true", help="run tasks in this process on CPU (tests)")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    categories = args.categories or list(cfg["categories"])
    folds = args.folds if args.folds is not None else list(range(cfg["kfold"]))
    jobs = [(c, r) for c in categories for r in folds if not _done(out, c, r)]
    print(f"{len(jobs)} (category, fold) trainings to run, output {out}", flush=True)
    if not jobs:
        return

    if args.local:
        from vvv.ml.train import train_fold

        for c, r in jobs:
            train_fold(cfg, c, r, device="cpu")
        return

    ray.init()
    # GPU workers start on demand; ask for the GPU and one CPU (XGBoost still uses the worker's cores).
    cpus = args.cpus_per_task or 1
    task = ray.remote(num_gpus=1, num_cpus=cpus, max_retries=3, retry_exceptions=False)(_run)
    print(f"cluster: {ray.cluster_resources()}; {cpus} CPUs per task", flush=True)
    pending = {task.remote(cfg, c, r): (c, r) for c, r in jobs}
    failed = []
    t0 = time.time()
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        c, r = pending.pop(ready[0])
        try:
            result = ray.get(ready[0])
            print(f"[{time.time() - t0:7.0f} s] {c} fold{r}: {result}", flush=True)
        except Exception:  # noqa: BLE001 - report every failure, keep the others running
            failed.append((c, r))
            print(f"[{time.time() - t0:7.0f} s] {c} fold{r} FAILED\n{traceback.format_exc()}", flush=True)
    if failed:
        print(f"{len(failed)} trainings failed: {failed}; resubmit to retry them", flush=True)
        sys.exit(1)
    print("all trainings done", flush=True)


def _done(out, category, fold):
    return (out / category / f"fold{fold}" / f"{category}_model.json").exists()


def _run(cfg, category, fold):
    # Runs on a GPU worker; the package is importable from the job's working directory.
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    from vvv.ml.train import train_fold

    return train_fold(cfg, category, fold, device="cuda")


if __name__ == "__main__":
    main()
