"""Submit the BDT configuration suite (suite_job.py) to the AF Ray cluster, on NRP or at Purdue.

The job's working directory holds only what a task imports (vvv/ml/train.py, the suite scripts,
the training config and the task list). On NRP the package directory is copied in by the
facility (af-inputs) and the output copied back to OUTPUT when the job ends.

Usage (AF global environment, from the repository root):
    python scripts/bdt_suite/submit_suite.py --site nrp --workers 20 \\
        --categories boosted2_leptonic_1lep boosted3 --variants base seed43 ... \\
        --inputs /depot/cms/users/$USER/VVV/bdt_suite/packages --output /work/users/$USER/vvv/bdt_suite/run1
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

GATEWAY = "http://ray-train-gateway:8265"
REPO = Path(__file__).resolve().parents[2]


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from variants import VARIANTS

    p = argparse.ArgumentParser()
    p.add_argument("--site", choices=("nrp", "purdue"), default="nrp")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--config", default=str(REPO / "configs/training/legacy_budget.json"))
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    unknown = set(args.variants) - set(VARIANTS)
    if unknown:
        sys.exit(f"unknown variants {sorted(unknown)}")

    stage = Path(tempfile.mkdtemp(prefix="bdt_suite_"))
    for rel in ("src/vvv/__init__.py", "src/vvv/ml/__init__.py", "src/vvv/ml/train.py", "src/vvv/ml/inference.py"):
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, stage / rel)
    for f in Path(__file__).parent.glob("*.py"):
        shutil.copy(f, stage / f.name)
    shutil.copy(args.config, stage / "config.json")
    # Longest first (1lep/resolved/unresolved, more rounds), so the short ones fill the end.
    slow = {"boosted2_unresolved": 4, "boosted2_resolved": 3, "boosted2_leptonic_1lep": 2, "boosted3": 1}
    tasks = sorted(({"category": c, "variant": v} for c in args.categories for v in args.variants),
                   key=lambda t: -(slow.get(t["category"], 0) * 10 + ("3000" in t["variant"]) * 5 + ("depth8" in t["variant"])))
    (stage / "tasks.json").write_text(json.dumps({"config": "config.json", "tasks": tasks}, indent=1))

    from ray.job_submission import JobSubmissionClient

    if args.site == "nrp":
        headers = {"af-site": "nrp", "af-max-workers": str(args.workers),
                   "af-inputs": args.inputs, "af-output": args.output}
        entry = "python suite_job.py tasks.json"
    else:
        headers = {"af-max-workers": str(args.workers), "af-min-gpu-memory-gb": "16",
                   "af-cpus-per-worker": "8", "af-ram-per-worker-gb": "64"}
        entry = f"python suite_job.py tasks.json --inputs {args.inputs} --output {args.output}"
    client = JobSubmissionClient(GATEWAY, headers=dict(headers))
    job = client.submit_job(entrypoint=entry, runtime_env={
        "working_dir": str(stage), "env_vars": {"PYTHONPATH": "src:.", "PYTHONUNBUFFERED": "1"}})
    print(f"submitted {job}: {len(tasks)} tasks, {headers}")


if __name__ == "__main__":
    main()
