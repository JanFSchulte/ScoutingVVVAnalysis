"""Expected sensitivity of the binned msoftdrop_1 mass fit for several variants of qcd_abcd_yields.root
inputs, with the legacy combine.C (CMS Combine, Asimov data only).

SPEC is a JSON file (example: configs/fitcheck/decor_vs_nodecor_lep.json):

  legacy_repo        ScoutingVVVTools checkout (combine/combine.C, combine/config.json, src/sample.json,
                     combine/xsec_unc.json); never written to
  cmssw_src          CMSSW src area with HiggsAnalysis/CombinedLimit built (combine, combineCards.py)
  categories         channel names, e.g. ["boosted2_leptonic_1lep", "boosted2_leptonic_2lep"]
  combinations       {name: [category, ...]}: combined fits, run in addition to every category alone
  setups             {name: [enabled_nuisances]}: e.g. stat = lumi + autoMCStats
  qcd_est_config     {category: legacy qcd_est.json} (massfit settings, exclude_samples); default: the
                     qcd_est.json next to the "source" sr_config.json of configs/signal_regions/<category>.json
  variants           {name: {training_config, and either root_files {category: qcd_abcd_yields.root}
                     or skim_dir + sr_dir (<sr_dir>/<category>/signal_region.csv), from which
                     scripts/estimate_qcd.py --skim-dir makes the inputs}}
  validation         optional list of legacy combine.json files (production points): rerun into
                     OUT/validation/<name> and compared with the recorded significance.csv / limits.csv
  combine_overrides  optional keys set on every combine.C config (lumi_unc, trigger_unc, ...)

combine.C reads its class registry from <bdt_root>/config.json (class_groups, sample_config). The
coffea inputs have no legacy model directory, so every variant gets a stub OUT/registry/<variant>/
config.json with the training config's class_groups (those that estimate_qcd.py used) and the
legacy sample.json. Every combine.C config starts from a copy of the legacy combine/config.json
(OUT/config/legacy_combine_config.json), with the syst-yield JSON keys removed.

Steps (--steps, default all): build (compile combine.C into OUT/bin), validate, inputs, combine,
summary. Outputs: OUT/inputs/<variant>/<category>/, OUT/runs/<variant>/<setup>/<scope>/ (combine.json,
job.log, CSVs, work/ with the cards), OUT/summary.json and OUT/summary.md.

Usage: pixi run python scripts/fitcheck_massfit.py SPEC --out DIR [--steps build,validate,inputs,combine,summary]
           [--jobs N] [--force] [--notes MD]
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SYST_JSON_KEYS = ("theory_syst_json", "pileup_syst_json", "jes_syst_json", "jer_syst_json", "jms_syst_json",
                  "jmr_syst_json")
STEPS = ("build", "validate", "inputs", "combine", "summary")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def cms_shell(spec, command, cwd, log_path=None):
    """Runs command in bash after cmsenv in spec['cmssw_src']; fails when combine is not on PATH."""
    script = (f"source /cvmfs/cms.cern.ch/cmsset_default.sh && cd {spec['cmssw_src']} && "
              f"eval \"$(scramv1 runtime -sh)\" && command -v combine >/dev/null && "
              f"command -v combineCards.py >/dev/null || {{ echo 'cmsenv failed: no combine on PATH' >&2; exit 97; }}; "
              f"cd {cwd} && {command}")
    out = open(log_path, "w") if log_path else None
    try:
        # A clean environment (no pixi variables), and stdin from /dev/null: with stdin a socket, bash
        # (also scram's inner `env -i bash -c`) sources ~/.bashrc, whose AF banner then breaks scram.
        env = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME") if k in os.environ}
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        r = subprocess.run(["/bin/bash", "-c", script], stdin=subprocess.DEVNULL, stdout=out or subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, env=env)
    finally:
        if out:
            out.close()
    return r


def build(spec, out, force):
    binary = out / "bin" / "combine_fitcheck"
    if binary.exists() and not force:
        return binary
    binary.parent.mkdir(parents=True, exist_ok=True)
    src = Path(spec["legacy_repo"]) / "combine" / "combine.C"
    log(f"compiling {src} -> {binary}")
    r = cms_shell(spec, f"c++ -O2 -DNDEBUG -std=c++17 $(root-config --cflags) {src} -o {binary} "
                        f"$(root-config --libs)", out)
    if r.returncode != 0:
        sys.exit(f"compilation failed:\n{r.stdout}")
    head = subprocess.run(["git", "-C", spec["legacy_repo"], "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    md5 = subprocess.run(["md5sum", str(src)], capture_output=True, text=True).stdout.split()[0]
    (binary.parent / "combine_fitcheck.source").write_text(f"{src}\nlegacy HEAD {head}\nmd5 {md5}\n")
    return binary


def run_combine(spec, binary, run_dir, cfg, force):
    """Writes run_dir/combine.json and runs combine.C there (skipped when significance.csv exists)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "significance.csv").exists() and (run_dir / "limits.csv").exists() and not force:
        return 0
    cfg = dict(cfg, output_dir=str(run_dir))
    (run_dir / "combine.json").write_text(json.dumps(cfg, indent=2))
    t0 = time.time()
    r = cms_shell(spec, f"COMBINE_CONFIG_PATH={run_dir / 'combine.json'} {binary}", run_dir, run_dir / "job.log")
    log(f"  {run_dir}: status {r.returncode} ({time.time() - t0:.0f} s)")
    return r.returncode


def read_csv(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def results(run_dir):
    """Expected significance and limit quantiles of the combined signal scenario, plus per channel."""
    sig = {r["name"]: float(r["significance"]) for r in read_csv(run_dir / "significance.csv")
           if r["scope"] == "combined"}
    lim = {r["name"]: r for r in read_csv(run_dir / "limits.csv") if r["scope"] == "combined"}
    out = {}
    if "combined" in sig:
        q = lim.get("combined", {})
        out = {"significance": sig["combined"],
               "limit": {k: float(v) for k, v in q.items() if k.startswith("exp_")}}
    chans = {}
    for r in read_csv(run_dir / "significance_by_channel.csv"):
        if r["scope"] == "combined" and r["name"] == "combined":
            chans.setdefault(r["channel"], {})["significance"] = float(r["significance"])
    for r in read_csv(run_dir / "limits_by_channel.csv"):
        if r["scope"] == "combined" and r["name"] == "combined":
            chans.setdefault(r["channel"], {})["limit_exp_50"] = float(r["exp_50"])
    out["by_channel"] = chans
    return out


def validate(spec, binary, out, force, jobs):
    entries = spec.get("validation", [])
    todo = []
    for path in entries:
        path = Path(path)
        name = path.parent.parent.name + "_" + path.parent.name
        cfg = json.loads(path.read_text())
        todo.append((name, path, cfg))
    with ThreadPoolExecutor(jobs) as pool:
        status = list(pool.map(lambda t: run_combine(spec, binary, out / "validation" / t[0], t[2], force), todo))
    report = []
    for (name, path, cfg), st in zip(todo, status):
        new = results(out / "validation" / name)
        ref = results(path.parent / "combine")
        row = {"point": str(path.parent), "status": st, "enabled_nuisances": cfg.get("enabled_nuisances"),
               "recorded": {k: ref.get(k) for k in ("significance", "limit")},
               "rerun": {k: new.get(k) for k in ("significance", "limit")}}
        if new.get("significance") is not None and ref.get("significance") is not None:
            row["significance_rel_diff"] = new["significance"] / ref["significance"] - 1
            row["limit_exp_50_rel_diff"] = new["limit"]["exp_50"] / ref["limit"]["exp_50"] - 1
        report.append(row)
        log(f"validation {name}: Z {ref.get('significance')} recorded, {new.get('significance')} rerun; "
            f"r95 {ref.get('limit', {}).get('exp_50')} recorded, {new.get('limit', {}).get('exp_50')} rerun")
    (out / "validation" / "validation.json").write_text(json.dumps(report, indent=1))
    return report


def qcd_est_config(spec, category):
    if category in spec.get("qcd_est_config", {}):
        return Path(spec["qcd_est_config"][category])
    source = json.loads((REPO / "configs" / "signal_regions" / f"{category}.json").read_text())["source"]
    rel = source.split(None, 1)[1] if source.startswith("legacy ") else source
    sr_cfg = Path(spec["legacy_repo"]) / "selections" / "BDT" / "study_boosted3" / rel
    return sr_cfg.parent / "qcd_est.json"


def make_inputs(spec, out, force):
    roots = {}
    for variant, v in spec["variants"].items():
        for cat in spec["categories"]:
            if "root_files" in v:
                roots[(variant, cat)] = Path(v["root_files"][cat])
                continue
            od = out / "inputs" / variant / cat
            root = od / "qcd_abcd_yields.root"
            roots[(variant, cat)] = root
            if root.exists() and not force:
                continue
            od.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(REPO / "scripts" / "estimate_qcd.py"), str(qcd_est_config(spec, cat)),
                   "--skim-dir", v["skim_dir"], "--training-config", str(REPO / v["training_config"]),
                   "--category", cat, "--signal-region-csv", str(Path(v["sr_dir"]) / cat / "signal_region.csv"),
                   "--output-dir", str(od)]
            log(f"inputs {variant}/{cat}: {' '.join(cmd[1:])}")
            with open(od / "estimate_qcd.log", "w") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
            if r.returncode != 0:
                sys.exit(f"estimate_qcd.py failed for {variant}/{cat}: see {od / 'estimate_qcd.log'}")
    return roots


def registry(spec, out, variant):
    """Stub bdt_root for combine.C: class_groups of the training config, sample_config = legacy sample.json."""
    tcfg = json.loads((REPO / spec["variants"][variant]["training_config"]).read_text())
    d = out / "registry" / variant
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({
        "class_groups": tcfg["class_groups"],
        "sample_config": str(Path(spec["legacy_repo"]) / "src" / "sample.json"),
        "note": f"combine.C class registry stub from {spec['variants'][variant]['training_config']}"}, indent=1))
    return d


def base_config(spec, out):
    legacy = Path(spec["legacy_repo"]) / "combine" / "config.json"
    copy = out / "config" / "legacy_combine_config.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(legacy, copy)
    cfg = json.loads(copy.read_text())
    for k in SYST_JSON_KEYS:
        cfg.pop(k, None)
    cfg.update({"xsec_unc_json": str(Path(spec["legacy_repo"]) / "combine" / "xsec_unc.json"),
                "keep_work": True, "scenario_scopes": ["combined"], "data_mode": "asimov"})
    cfg.update(spec.get("combine_overrides", {}))
    return cfg


def scopes(spec):
    out = {cat: [cat] for cat in spec["categories"]}
    out.update(spec.get("combinations", {}))
    return out


def run_all(spec, binary, out, roots, force, jobs):
    base = base_config(spec, out)
    tasks = []
    for variant in spec["variants"]:
        reg = registry(spec, out, variant)
        for setup, nuis in spec["setups"].items():
            for scope, cats in scopes(spec).items():
                cfg = dict(base, enabled_nuisances=nuis,
                           channels=[{"name": c, "root_file": str(roots[(variant, c)]), "bdt_root": str(reg)}
                                     for c in cats])
                tasks.append((out / "runs" / variant / setup / scope, cfg))
    log(f"{len(tasks)} combine.C runs, {jobs} in parallel")
    with ThreadPoolExecutor(jobs) as pool:
        status = list(pool.map(lambda t: run_combine(spec, binary, t[0], t[1], force), tasks))
    failed = [str(t[0]) for t, s in zip(tasks, status) if s != 0]
    if failed:
        sys.exit("combine.C failed in: " + ", ".join(failed))


def counting_z(spec, variant, cat):
    """Z_comb = sqrt(sum Z_i^2) of the SR optimisation (the counting figure of merit), if known."""
    v = spec["variants"][variant]
    if "sr_dir" not in v:
        return None
    rows = read_csv(Path(v["sr_dir"]) / cat / "signal_region.csv")
    return math.sqrt(sum(float(r["significance"]) ** 2 for r in rows)) if rows else None


def summary(spec, out, validation, notes=None):
    variants = list(spec["variants"])
    table = []
    for setup in spec["setups"]:
        for scope, cats in scopes(spec).items():
            row = {"setup": setup, "scope": scope, "nuisances": spec["setups"][setup]}
            for v in variants:
                res = results(out / "runs" / v / setup / scope)
                zc = [counting_z(spec, v, c) for c in cats]
                row[v] = {"significance": res.get("significance"), "limit": res.get("limit"),
                          "by_channel": res.get("by_channel") if len(cats) > 1 else None,
                          "counting_z_comb": (math.sqrt(sum(z * z for z in zc)) if None not in zc else None)}
            table.append(row)
    doc = {"spec": spec, "validation": validation, "results": table,
           "method": "combine.C (legacy) mass-binned cards (writeMassfitCard, qcd_from_mc), Asimov data "
                     "(-t -1): -M Significance --expectSignal 1, -M AsymptoticLimits --run expected; "
                     "signal = all signal classes (VVV + VH) with one strength r"}
    (out / "summary.json").write_text(json.dumps(doc, indent=1, default=str))

    a, b = variants[0], variants[-1]
    lines = [f"# Mass-fit expected sensitivity: {a} vs {b}", "",
             "combine.C mass-binned cards, Asimov (a-priori) only; Z = expected significance for r = 1, "
             "r95 = median expected 95% CL upper limit on r. Z_count = counting Z_comb of the SR optimisation.", ""]
    if validation:
        lines += ["## Validation (legacy production points rerun)", "",
                  "| point | Z recorded | Z rerun | r95 recorded | r95 rerun |", "|---|---|---|---|---|"]
        for r in validation:
            rec, new = r["recorded"], r["rerun"]
            lines.append(f"| {Path(r['point']).parent.name}/{Path(r['point']).name} | {rec['significance']:.4f} | "
                         f"{new['significance']:.4f} | {rec['limit']['exp_50']:.4f} | {new['limit']['exp_50']:.4f} |")
        lines.append("")
    for setup in spec["setups"]:
        lines += [f"## Setup {setup}: {', '.join(spec['setups'][setup])}", "",
                  f"| scope | Z_count {a} | Z_count {b} | Z {a} | Z {b} | dZ | r95 {a} | r95 {b} | r95 ratio |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in table:
            if row["setup"] != setup:
                continue
            ra, rb = row[a], row[b]
            za, zb = ra["significance"], rb["significance"]
            la, lb = (ra["limit"] or {}).get("exp_50"), (rb["limit"] or {}).get("exp_50")
            f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
            lines.append(f"| {row['scope']} | {f(ra['counting_z_comb'])} | {f(rb['counting_z_comb'])} | {f(za)} | "
                         f"{f(zb)} | {f(None if None in (za, zb) else zb - za)} | {f(la)} | {f(lb)} | "
                         f"{f(None if None in (la, lb) else lb / la)} |")
        lines.append("")
    if notes:
        lines += ["", notes.read_text().rstrip(), ""]
    (out / "summary.md").write_text("\n".join(lines))
    log(f"wrote {out / 'summary.json'} and {out / 'summary.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("spec", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", default=",".join(STEPS))
    p.add_argument("--jobs", type=int, default=6, help="combine.C runs in parallel")
    p.add_argument("--force", action="store_true", help="redo steps whose outputs exist")
    p.add_argument("--notes", type=Path, help="markdown appended to summary.md (interpretation, caveats)")
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    steps = args.steps.split(",")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    binary = build(spec, out, args.force and "build" in steps) if {"build", "validate", "combine"} & set(steps) \
        else out / "bin" / "combine_fitcheck"
    validation = None
    if "validate" in steps:
        validation = validate(spec, binary, out, args.force, args.jobs)
    elif (out / "validation" / "validation.json").exists():
        validation = json.loads((out / "validation" / "validation.json").read_text())
    if {"inputs", "combine"} & set(steps):
        roots = make_inputs(spec, out, args.force and "inputs" in steps)
    if "combine" in steps:
        run_all(spec, binary, out, roots, args.force, args.jobs)
    if "summary" in steps:
        summary(spec, out, validation, args.notes)


if __name__ == "__main__":
    main()
