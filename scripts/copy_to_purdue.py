"""Copy samples whose files sit at a remote site to Purdue EOS, then point their file lists there.

Files go from the global redirector to root://eos.cms.rcac.purdue.edu//store/user/<user>/mirror/<lfn
below /store/user/>, with an adler32 check, a few transfers at a time. Files already there with
the source size are skipped, so a rerun resumes. Once every file of a sample is copied, its
data/filelists/<sample>.json lists the Purdue copies (the original list is kept under "source_files"),
and `--redirector auto` then reads them from EOS.

Usage: python scripts/copy_to_purdue.py SAMPLE... [--parallel 8] [--user jschulte] [--dry-run]
"""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GLOBAL = "root://cms-xrd-global.cern.ch/"
EOS = "root://eos.cms.rcac.purdue.edu/"


def _size(host, lfn, timeout=120):
    out = subprocess.run(["xrdfs", host, "stat", lfn], capture_output=True, text=True, timeout=timeout)
    for line in out.stdout.splitlines():
        if line.startswith("Size:"):
            return int(line.split()[1])
    return None


def _copy(src_lfn, dst_lfn, retries=3):
    try:
        if _size("eos.cms.rcac.purdue.edu", dst_lfn) is not None:
            return "exists"
    except subprocess.TimeoutExpired:
        pass
    err = "timeout"
    for _ in range(retries):
        cmd = ["xrdcp", "--nopbar", "--force", "--posc", "--cksum", "adler32", "--path",
               GLOBAL + src_lfn, EOS + dst_lfn]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            continue
        if res.returncode == 0:
            return "copied"
        err = (res.stderr.strip().splitlines() or ["?"])[-1]
    return f"FAILED {err}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("samples", nargs="+")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--user", default="jschulte", help="CERN username of the Purdue EOS area")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    for sample in args.samples:
        path = REPO / "data" / "filelists" / f"{sample}.json"
        record = json.loads(path.read_text())
        source = record.get("source_files", record["files"])
        prefix = f"/store/user/{args.user}/mirror/"
        pairs = [(f, prefix + f[len("/store/user/"):]) for f in source]
        if args.dry_run:
            print(f"{sample}: {len(pairs)} files, e.g. {pairs[0][0]} -> {EOS}{pairs[0][1]}")
            continue
        with ThreadPoolExecutor(args.parallel) as pool:
            results = list(pool.map(lambda pair: _copy(*pair), pairs))
        failed = [src for (src, _), r in zip(pairs, results) if r.startswith("FAILED")]
        n_new = sum(r == "copied" for r in results)
        print(f"{sample}: {n_new} copied, {len(pairs) - n_new - len(failed)} already there, {len(failed)} failed", flush=True)
        if failed:
            for src in failed[:5]:
                print(f"  failed: {src}")
            continue
        record["source_files"] = source
        record["files"] = [dst for _, dst in pairs]
        path.write_text(json.dumps(record, indent=1))
        print(f"  {path.name} now lists the Purdue copies", flush=True)


if __name__ == "__main__":
    main()
