"""Run the BDT configuration suite as one single-GPU NRP job per (category, variant).

Each job reads only its category's inputs (INPUTS_ROOT/<category>, a directory holding
<category>/) and writes OUTPUT/<category>__<variant>/<category>/<variant>/, so no two jobs share
an output directory. At most --max-inflight jobs are pending or running at a time. A job that
ends without its model (its cluster did not start in time, or it failed) is resubmitted, at most
--attempts times. A task is done when its model file is back on AF storage. State and a log are
kept in OUTPUT/orchestrate_state.json, so a restart picks up where it left off.

Usage (AF global environment, from the repository root):
    python scripts/bdt_suite/orchestrate.py OUTPUT --inputs-root /work/users/$USER/vvv/bdt_suite \\
        --categories boosted2_leptonic_1lep boosted3 boosted2_leptonic_2lep [--variants ...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
GATEWAY = "http://ray-train-gateway:8265"
# inputs directory per category under --inputs-root (each under the NRP copy limit of ~3 GB)
INPUTS = {"boosted2_leptonic_1lep": "inputs_1lep_notest", "boosted2_leptonic_2lep": "inputs_boosted2_leptonic_2lep",
          "boosted3": "inputs_boosted3"}


def main():
    from ray.job_submission import JobSubmissionClient
    from submit_suite import REPO, submit
    from variants import VARIANTS

    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--inputs-root", type=Path, required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--variants", nargs="+", default=[v for v in VARIANTS if v not in ("smoke", "numpy_loss")])
    p.add_argument("--config", default=str(REPO / "configs/training/legacy_budget.json"))
    p.add_argument("--max-inflight", type=int, default=30)
    p.add_argument("--attempts", type=int, default=8)
    p.add_argument("--poll", type=int, default=120)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_file = args.output / "orchestrate_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    slow = {"boosted2_leptonic_1lep": 2, "boosted3": 1}
    tasks = sorted(((c, v) for c in args.categories for v in args.variants),
                   key=lambda t: -(slow.get(t[0], 0) * 10 + ("3000" in t[1]) * 5 + ("depth8" in t[1])))
    client = JobSubmissionClient(GATEWAY)

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def out_dir(c, v):
        return args.output / f"{c}__{v}"

    def done(c, v):
        return (out_dir(c, v) / c / v / f"{c}_model.json").exists()

    while True:
        inflight, open_tasks = 0, 0
        for c, v in tasks:
            key = f"{c}__{v}"
            st = state.setdefault(key, {"jobs": [], "status": None})
            if done(c, v):
                if st["status"] != "done":
                    st["status"] = "done"
                    log(f"{key}: done")
                continue
            open_tasks += 1
            if st["jobs"] and st["status"] in ("PENDING", "RUNNING", "SUBMITTED"):
                try:
                    s = str(client.get_job_status(st["jobs"][-1]))
                except Exception as err:  # noqa: BLE001 - gateway hiccup: ask again next poll
                    log(f"{key}: status error {err}")
                    s = st["status"]
                s = s.split(".")[-1]
                if s != st["status"]:
                    log(f"{key}: {st['jobs'][-1]} {s}")
                st["status"] = s
                if s in ("PENDING", "RUNNING", "SUBMITTED"):
                    inflight += 1
                    continue
                if s == "SUCCEEDED":
                    # outputs are copied back after the job ends: give them a few polls
                    st.setdefault("succeeded_polls", 0)
                    st["succeeded_polls"] += 1
                    if st["succeeded_polls"] < 5:
                        st["status"] = "RUNNING"
                        inflight += 1
                        continue
                    log(f"{key}: SUCCEEDED but no model came back")
        for c, v in tasks:
            key = f"{c}__{v}"
            st = state[key]
            if inflight >= args.max_inflight:
                break
            if st["status"] in ("done", "PENDING", "RUNNING", "SUBMITTED", "gave up"):
                continue
            if len(st["jobs"]) >= args.attempts:
                st["status"] = "gave up"
                log(f"{key}: gave up after {len(st['jobs'])} attempts")
                continue
            out_dir(c, v).mkdir(parents=True, exist_ok=True)
            try:
                job = submit("nrp", 1, args.config, [c], [v], str(args.inputs_root / INPUTS[c]), str(out_dir(c, v)), quiet=True)
            except Exception as err:  # noqa: BLE001
                log(f"{key}: submit failed {err}")
                continue
            st["jobs"].append(job)
            st["status"] = "SUBMITTED"
            st.pop("succeeded_polls", None)
            inflight += 1
            log(f"{key}: submitted {job} (attempt {len(st['jobs'])})")
        state_file.write_text(json.dumps(state, indent=1))
        if open_tasks == 0:
            log("all tasks done")
            return
        if all(state[f"{c}__{v}"]["status"] in ("done", "gave up") for c, v in tasks):
            log("finished; some tasks gave up")
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
