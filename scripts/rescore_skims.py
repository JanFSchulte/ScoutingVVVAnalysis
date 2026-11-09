"""Out-of-fold scores of a new K-fold training, recomputed on existing fit-region skims.

The skims (vvv run --skim-cut --model-dir) carry every classifier input, so a retrained model
needs no new pass over NanoAOD: each <SRC>/<sample>/<category>/nominal/*.parquet is copied to
DEST with its proba_<class> columns replaced by the new model's (each event scored by the
model of its fold, event % k), and _DONE.json is copied (marked with the model directory).
Only the given category is written, which is all the SR optimisation of that category reads.

Usage: pixi run python scripts/rescore_skims.py SRC DEST --category CAT --model-dir DIR [--threads N]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vvv.ml.inference import CLASSES, CategoryModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--category", required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    model = CategoryModel.load(args.model_dir, args.category)
    samples = sorted(d.parent for d in args.src.glob("*/_DONE.json"))
    jobs = [(s, f) for s in samples for f in sorted((s / args.category / "nominal").glob("*.parquet"))]
    print(f"{len(samples)} samples, {len(jobs)} files", flush=True)

    def rescore(job):
        sample, f = job
        t = pq.read_table(f)
        if t.num_rows:
            cols = {c: t[c].to_numpy() for c in model.features}
            proba = model.predict(cols, t["event"].to_numpy())
            for i, c in enumerate(CLASSES):
                t = t.set_column(t.schema.get_field_index(f"proba_{c}"), f"proba_{c}", pa.array(proba[:, i]))
        out = args.dest / f.relative_to(args.src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out.with_name(f".{out.name}.tmp"), compression="zstd")
        out.with_name(f".{out.name}.tmp").rename(out)
        return t.num_rows

    with ThreadPoolExecutor(args.threads) as ex:
        n = sum(ex.map(rescore, jobs))
    for s in samples:
        done = json.loads((s / "_DONE.json").read_text())
        done["rescored"] = {"category": args.category, "model_dir": str(args.model_dir)}
        (args.dest / s.name).mkdir(parents=True, exist_ok=True)
        (args.dest / s.name / "_DONE.json").write_text(json.dumps(done, indent=1))
    print(f"rescored {n} events", flush=True)


if __name__ == "__main__":
    main()
