"""Performance metrics of the running workflows, appended as snapshots and summarised.

Sources:
- orchestrator runs (scripts/bdt_suite/orchestrate.py): submission times from the log, start/end
  times and final message of every attempt from the Ray job API, and per task the outputs'
  gpu_util.json (input copy time, GPU utilisation samples) and training_log.json (device, stage
  times, rounds, events);
- other Ray jobs given with --job NAME=ID (Purdue runs);
- a Dask skim production log (vvv run --batch-files): finished batches/samples and failures, and
  the Gateway cluster's worker activity.

Writes METRICS_DIR/snapshots.jsonl (one line per call) and METRICS_DIR/report.md.

Usage: python collect_metrics.py METRICS_DIR --orchestrator NAME=OUTPUT_DIR:LOG ... \\
           [--job NAME=RAY_JOB_ID ...] [--dask-log NAME=LOG ...]
"""

import argparse
import datetime as dt
import json
import re
import statistics as st
import time
from pathlib import Path

GATEWAY = "http://ray-train-gateway:8265"


def _ts(day, hms):
    return dt.datetime.combine(day, dt.time.fromisoformat(hms), tzinfo=dt.timezone.utc).timestamp()


def orchestrator(client, out_dir, log_path):
    """Per task: attempts (submitted, started, ended, status, message) and run metrics."""
    log = Path(log_path).read_text().splitlines()
    day = dt.datetime.fromtimestamp(Path(log_path).stat().st_ctime, dt.timezone.utc).date()
    # Ray's start_time is the submission time; the orchestrator log has the status transitions
    # (at its polling resolution, 2 min by default)
    submitted, running, prev = {}, {}, None
    for line in log:
        m = re.match(r"(\d\d:\d\d:\d\d) (\S+): (?:submitted )?(raysubmit_\w+)(?: (\w+))?", line)
        if not m:
            continue
        t = _ts(day, m[1])
        if prev is not None and t < prev - 3600:  # the log crossed midnight
            day += dt.timedelta(days=1)
            t = _ts(day, m[1])
        prev = t
        if "submitted" in line:
            submitted[m[3]] = (m[2], t)
        elif m[4] == "RUNNING":
            running.setdefault(m[3], t)
    state = json.loads((Path(out_dir) / "orchestrate_state.json").read_text())
    tasks = {}
    for key, s in state.items():
        attempts = []
        for job in s["jobs"]:
            try:
                info = client.get_job_info(job)
                start = running.get(job)
                end = info.end_time / 1000 if info.end_time else None
                status, message = str(info.status).split(".")[-1], info.message
            except Exception as err:  # noqa: BLE001 - record and go on
                start = end = None
                status, message = "UNKNOWN", str(err)
            sub = submitted.get(job, (key, None))[1]
            attempts.append({"job": job, "submitted": sub, "start": start, "end": end, "status": status,
                             "message": message, "started": start is not None})
        for a, nxt in zip(attempts, attempts[1:]):
            if a["end"] is None and a["status"] in ("STOPPED", "FAILED"):
                a["end"] = nxt["submitted"]  # Ray keeps no end time for a cluster that never started
        c, v = key.split("__", 1)
        res = Path(out_dir) / key / c / v
        run = {}
        if (res / "gpu_util.json").exists():
            g = json.loads((res / "gpu_util.json").read_text())
            run.update({k: g.get(k) for k in ("seconds", "copy_s", "gpu_util_mean")})
            samples = g.get("gpu_samples") or []
            run["gpu_util_min_30min"] = _min_window([u for _, u, _ in samples], 60)
            run["gpu_mem_max_mib"] = max((m for _, _, m in samples), default=None)
        if (res / "training_log.json").exists():
            t = json.loads((res / "training_log.json").read_text())
            run.update({"device": t.get("device"), "n_train": t["n"].get("train"),
                        "stage1_rounds": t["stage1"]["rounds"], "stage1_s": t["stage1"]["time_s"],
                        "stage2_rounds": t.get("stage2", {}).get("rounds"),
                        "total_train_s": t.get("stage2", t["stage1"])["time_s"]})
        tasks[key] = {"status": s["status"], "attempts": attempts, "run": run}
    return tasks


def _min_window(u, n):
    """Lowest mean over n consecutive samples (30-s samples: n=60 is NRP's 30-min idle window)."""
    if len(u) < n:
        return st.mean(u) if u else None
    return min(st.mean(u[i:i + n]) for i in range(len(u) - n + 1))


def dask_log(path):
    text = Path(path).read_text().replace("\r", "\n")
    lines = text.splitlines()
    return {"batches_done": sum(1 for l in lines if re.match(r"\S+ batch \d+/\d+: \{", l)),
            "samples_done": sum(1 for l in lines if re.match(r"[A-Za-z0-9_-]+: \{", l)),
            "batches_failed": sum(1 for l in lines if re.match(r"\S+ batch \d+/\d+: FAILED", l)),
            "last": next((l[:80] for l in reversed(lines) if re.match(r"\S+( batch \d+/\d+)?: \{", l)), None)}


def dask_workers():
    try:
        from dask_gateway import Gateway

        gw = Gateway()
        clusters = gw.list_clusters()
        if not clusters:
            return None
        client = gw.connect(clusters[0].name).get_client()
        info = client.scheduler_info()["workers"]
        busy = sum(1 for tasks in client.processing().values() if tasks)
        return {"workers": len(info), "busy": busy}
    except Exception as err:  # noqa: BLE001
        return {"error": str(err)[:200]}


def report(snapshot, path):
    lines = [f"# Workflow performance ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(snapshot['time']))})", ""]
    for name, tasks in snapshot.get("orchestrators", {}).items():
        atts = [a for t in tasks.values() for a in t["attempts"]]
        started = [a for a in atts if a["started"]]
        never = [a for a in atts if not a["started"] and a["status"] in ("STOPPED", "FAILED")]
        pending = [a for a in atts if not a["started"] and a["status"] == "PENDING"]
        waits = [a["start"] - a["submitted"] for a in started if a["submitted"] and a["start"]]
        lost = [a["end"] - a["submitted"] for a in never if a["submitted"] and a["end"]]
        runs = [t["run"] for t in tasks.values() if t["run"].get("seconds")]
        statuses = {}
        for t in tasks.values():
            statuses[t["status"]] = statuses.get(t["status"], 0) + 1
        lines += [f"## {name}", "",
                  f"- tasks: {len(tasks)} ({', '.join(f'{k} {v}' for k, v in sorted(statuses.items(), key=str))})",
                  f"- attempts: {len(atts)}; started {len(started)}; pending now {len(pending)} "
                  + (f"(oldest {max(snapshot['time'] - a['submitted'] for a in pending if a['submitted']) / 60:.0f} min); " if pending else "; ")
                  + f"never started {len(never)} "
                  f"(messages: {sorted({a['message'] for a in never})})",
                  f"- submit -> start: " + (f"median {st.median(waits) / 60:.1f} min, max {max(waits) / 60:.1f} min (n={len(waits)})" if waits else "n/a"),
                  f"- time lost in attempts that never started: " + (f"{sum(lost) / 3600:.1f} h total, median {st.median(lost) / 60:.1f} min" if lost else "none"),
                  ""]
        if runs:
            lines += ["| task | attempts | wait min | device | copy s | stage 1 rounds / min | stage 2 rounds / total min | GPU util mean | min 30-min mean | GPU mem MiB |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for key, t in sorted(tasks.items()):
                r = t["run"]
                if not r.get("seconds"):
                    continue
                a = [x for x in t["attempts"] if x["started"]]
                wait = (a[-1]["start"] - a[-1]["submitted"]) / 60 if a and a[-1]["submitted"] and a[-1]["start"] else float("nan")
                lines.append(f"| {key} | {len(t['attempts'])} | {wait:.0f} | {r.get('device', '')} | {r.get('copy_s', 0):.0f} | "
                             f"{r.get('stage1_rounds')} / {r.get('stage1_s', 0) / 60:.0f} | {r.get('stage2_rounds')} / "
                             f"{r.get('total_train_s', 0) / 60:.0f} | {r.get('gpu_util_mean') or float('nan'):.0f} | "
                             f"{r.get('gpu_util_min_30min') or float('nan'):.0f} | {r.get('gpu_mem_max_mib') or 0:.0f} |")
            lines.append("")
    for name, d in snapshot.get("dask", {}).items():
        lines += [f"## {name} (Dask)", "", f"- now: {d}"]
        hist = snapshot.get("dask_rate", {}).get(name)
        if hist:
            lines.append(f"- rate over the last {hist['hours']:.1f} h: {hist['batches_per_h']:.1f} batches/h, {hist['samples_per_h']:.1f} samples/h")
        lines.append("")
    for name, j in snapshot.get("jobs", {}).items():
        lines += [f"- Ray job {name}: {j}"]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    from ray.job_submission import JobSubmissionClient

    p = argparse.ArgumentParser()
    p.add_argument("metrics_dir", type=Path)
    p.add_argument("--orchestrator", action="append", default=[], help="NAME=OUTPUT_DIR:LOG")
    p.add_argument("--job", action="append", default=[], help="NAME=RAY_JOB_ID")
    p.add_argument("--dask-log", action="append", default=[], help="NAME=LOG")
    args = p.parse_args()
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    client = JobSubmissionClient(GATEWAY)
    snap = {"time": time.time(), "orchestrators": {}, "jobs": {}, "dask": {}}
    for item in args.orchestrator:
        name, rest = item.split("=", 1)
        out, log = rest.split(":", 1)
        snap["orchestrators"][name] = orchestrator(client, out, log)
    for item in args.job:
        name, job = item.split("=", 1)
        info = client.get_job_info(job)
        snap["jobs"][name] = {"status": str(info.status).split(".")[-1], "start": info.start_time, "end": info.end_time,
                              "message": info.message}
    workers = dask_workers() if args.dask_log else None
    for item in args.dask_log:
        name, log = item.split("=", 1)
        snap["dask"][name] = {**dask_log(log), "cluster": workers}
    hist_file = args.metrics_dir / "snapshots.jsonl"
    history = [json.loads(l) for l in hist_file.read_text().splitlines()] if hist_file.exists() else []
    snap["dask_rate"] = {}
    for name, d in snap["dask"].items():
        old = [h for h in history if name in h.get("dask", {}) and snap["time"] - h["time"] <= 3 * 3600]
        if old:
            h0 = old[0]
            hours = (snap["time"] - h0["time"]) / 3600
            if hours > 0.1:
                snap["dask_rate"][name] = {"hours": hours,
                                           "batches_per_h": (d["batches_done"] - h0["dask"][name]["batches_done"]) / hours,
                                           "samples_per_h": (d["samples_done"] - h0["dask"][name]["samples_done"]) / hours}
    with open(hist_file, "a") as f:
        f.write(json.dumps(snap, default=str) + "\n")
    report(snap, args.metrics_dir / "report.md")


if __name__ == "__main__":
    main()
