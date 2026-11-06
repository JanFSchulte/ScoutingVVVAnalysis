"""Ray entrypoint of the BDT configuration suite: one GPU task per (category, variant).

Each task copies its category's package (make_packages.py) from the inputs to local disk,
applies the variant (variants.py), trains fold 0 with vvv.ml.train.train_fold (no plots),
predicts the test split on the GPU (if the package has it; else evaluate.py does) and copies <category>/<variant>/ (model, training_log.json,
test_proba.npz, gpu_util.json) to the output. A (category, variant) already in the output is
skipped. Inputs and output are AF_INPUTS / AF_OUTPUT (NRP: copied in / back by the facility),
else --inputs / --output (Purdue, plain paths).

Usage: python suite_job.py TASKS.json [--inputs DIR --output DIR]
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback


def _fs(uri):
    import pyarrow.fs as pafs

    return pafs.FileSystem.from_uri(uri)


def _exists(uri):
    import pyarrow.fs as pafs

    fs, path = _fs(uri)
    return fs.get_file_info(path).type != pafs.FileType.NotFound


def _gpu_sampler(samples, stop):
    while not stop.wait(30):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            u, mem = (float(x) for x in r.stdout.split(",")[:2])
            samples.append((time.time(), u, mem))
        except Exception:  # noqa: BLE001 - monitoring only
            pass


def run_task(cfg, category, variant, inputs, output):
    import numpy as np
    import pyarrow.fs as pafs

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    from make_packages import load_package
    from variants import apply

    from vvv.ml.train import softmax, standardize, train_fold

    t0 = time.time()
    dest = f"{output}/{category}/{variant}"
    if _exists(f"{dest}/{category}_model.json"):
        return f"{category}/{variant}: already done"
    local = f"/tmp/suite_inputs/{category}"
    if not os.path.exists(f"{local}/meta.json"):
        os.makedirs(local, exist_ok=True)
        pafs.copy_files(f"{inputs}/{category}", local)
    t_copy = time.time() - t0
    samples, stop = [], threading.Event()
    threading.Thread(target=_gpu_sampler, args=(samples, stop), daemon=True).start()
    try:
        has_test = os.path.exists(f"{local}/test.npz")  # else evaluate.py predicts at Purdue
        data = load_package(local, splits=("train", "monitor", "test") if has_test else ("train", "monitor"))
        cfg, data = apply(cfg, category, data, variant)
        out = f"/tmp/suite_out/{category}/{variant}"
        cfg["output_dir"] = f"/tmp/suite_out/{category}/{variant}_root"
        device = os.environ.get("SUITE_DEVICE", "cuda")  # cpu: local smoke tests
        final = train_fold(cfg, category, 0, device=device, data=data, plots=False)
        cc = cfg["categories"][category]
        import xgboost as xgb

        os.makedirs(out, exist_ok=True)
        if has_test:
            test = data["test"]
            X = standardize(test["X"], cc["features"], cc["clip_ranges"], cc["log_transform"])
            final.set_param({"device": device})
            m = final.inplace_predict(X, predict_type="margin").reshape(len(X), -1)
            np.savez(f"{out}/test_proba.npz", proba=softmax(m.astype(np.float64)).astype(np.float32), event=test["event"])
        fold_dir = f"{cfg['output_dir']}/{category}/fold0"
        for name in (f"{category}_model.json", "training_log.json"):
            os.replace(f"{fold_dir}/{name}", f"{out}/{name}")
    finally:
        stop.set()
    util = [u for _, u, _ in samples]
    info = {"variant": variant, "category": category, "seconds": time.time() - t0, "copy_s": t_copy,
            "gpu_util_mean": float(np.mean(util)) if util else None, "gpu_samples": samples, "xgboost": xgb.__version__}
    with open(f"{out}/gpu_util.json", "w") as f:
        json.dump(info, f)
    fs, path = _fs(dest)
    fs.create_dir(path, recursive=True)
    pafs.copy_files(out, dest)
    return (f"{category}/{variant}: {time.time() - t0:.0f} s (copy {t_copy:.0f} s), "
            f"GPU util mean {info['gpu_util_mean']}")


def main():
    import ray

    p = argparse.ArgumentParser()
    p.add_argument("tasks")
    p.add_argument("--inputs", default=os.environ.get("AF_INPUTS"))
    p.add_argument("--output", default=os.environ.get("AF_OUTPUT"))
    args = p.parse_args()
    spec = json.load(open(args.tasks))
    cfg = json.load(open(spec["config"]))
    tasks = spec["tasks"]
    print(f"{len(tasks)} tasks, inputs {args.inputs}, output {args.output}", flush=True)
    ray.init()
    remote = ray.remote(num_gpus=1, num_cpus=1, max_retries=2, retry_exceptions=False)(run_task)
    pending = {remote.remote(cfg, t["category"], t["variant"], args.inputs, args.output): t for t in tasks}
    failed = []
    t0 = time.time()
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        t = pending.pop(ready[0])
        try:
            print(f"[{time.time() - t0:7.0f} s] {ray.get(ready[0])}", flush=True)
        except Exception:  # noqa: BLE001 - keep the others running
            failed.append(t)
            print(f"[{time.time() - t0:7.0f} s] {t} FAILED\n{traceback.format_exc()}", flush=True)
    print(f"done, {len(failed)} failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
