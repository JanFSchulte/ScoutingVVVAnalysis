"""Submit a K-fold BDT training to the Purdue AF Ray cluster.

Each (category, fold) trains on its own GPU (scripts run in the AF global
environment, which has Ray and a CUDA build of XGBoost). Resubmitting the same
config resumes: folds whose model exists are skipped.

Usage (from the repository root, in a session without a GPU):

    /work/pixi/global/.pixi/envs/default/bin/python scripts/submit_training.py configs/training/production.json
    ... --workers 8 --categories boosted3 --folds 0 1     # subsets
    ... --status JOB_ID | --logs JOB_ID | --stop JOB_ID
"""

import argparse
import json
import os
import sys
from pathlib import Path

GATEWAY = "http://ray-train-gateway:8265"
REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?")
    p.add_argument("--workers", default="8", help="most GPU workers (af-max-workers)")
    p.add_argument("--min-gpu-memory-gb", default="16", help="16 = T4s only")
    p.add_argument("--cpus-per-worker", default="8")
    p.add_argument("--ram-per-worker-gb", default="64")
    p.add_argument("--categories", nargs="*")
    p.add_argument("--folds", nargs="*")
    p.add_argument("--status")
    p.add_argument("--logs")
    p.add_argument("--stop")
    args = p.parse_args()

    from ray.job_submission import JobSubmissionClient

    if args.status or args.logs or args.stop:
        client = JobSubmissionClient(GATEWAY)
        if args.status:
            print(client.get_job_status(args.status))
        if args.logs:
            print(client.get_job_logs(args.logs))
        if args.stop:
            print(client.stop_job(args.stop))
        return

    config = Path(args.config).resolve()
    cfg = json.loads(config.read_text())
    if not str(Path(cfg["output_dir"]).resolve()).startswith(("/work/", "/depot/")):
        sys.exit("output_dir must be on /work or /depot: the Ray cluster has no /home")
    headers = {
        "af-max-workers": str(args.workers),
        "af-min-gpu-memory-gb": str(args.min_gpu_memory_gb),
        "af-cpus-per-worker": str(args.cpus_per_worker),
        "af-ram-per-worker-gb": str(args.ram_per_worker_gb),
    }
    # The client adds the preset auth token to the dict it is given: pass a copy.
    client = JobSubmissionClient(GATEWAY, headers=dict(headers))
    extra = []
    if args.categories:
        extra += ["--categories", *args.categories]
    if args.folds:
        extra += ["--folds", *args.folds]
    job = client.submit_job(
        entrypoint=" ".join(["python", "-m", "vvv.ml.ray_driver", str(config), *extra]),
        runtime_env={
            "working_dir": str(REPO),
            "excludes": [".pixi", ".git", "data/filelists", "metadata", "*.parquet", "*.pkl"],
            "env_vars": {"PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"},
        },
    )
    print(f"submitted {job} with {headers}")
    print(f"status: {sys.executable} {__file__} --status {job}")
    print(f"logs:   {sys.executable} {__file__} --logs {job}")


if __name__ == "__main__":
    main()
