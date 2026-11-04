"""Merge the per-chunk skim files into a few large files per (sample, category, variation).

`vvv run --skim-dir` writes one Parquet file per chunk and category (about 200k files for the
training skims). This rewrites each <sample>/<category>/<variation>/ directory as part-<i>.parquet
files of at most --rows-per-file rows, in the sorted order of the chunk files, checks the row
count, and copies _DONE.json. Directories already compacted (same row count) are skipped, so a
rerun resumes. With --remove-source a source directory is deleted only after its check passed.

Usage: python scripts/compact_skims.py SRC DEST [--workers 8] [--rows-per-file 5000000] [--remove-source]
"""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _rows(files):
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def compact(src_dir, dest_dir, rows_per_file):
    files = sorted(src_dir.glob("*.parquet"))
    n_src = _rows(files)
    done = sorted(dest_dir.glob("part-*.parquet"))
    if done and _rows(done) == n_src:
        return src_dir, n_src, "skipped"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in done:
        f.unlink()
    schema = pa.unify_schemas([pq.read_schema(f) for f in files]) if files else None
    part, writer, in_part = 0, None, 0
    for f in files:
        table = pq.read_table(f).cast(schema)
        if table.num_rows == 0:
            continue
        if writer is None or in_part >= rows_per_file:
            if writer is not None:
                writer.close()
                part += 1
            tmp = dest_dir / f".part-{part:03d}.parquet.tmp"
            writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            in_part = 0
        writer.write_table(table, row_group_size=1_000_000)
        in_part += table.num_rows
    if writer is not None:
        writer.close()
    for tmp in sorted(dest_dir.glob(".part-*.parquet.tmp")):
        tmp.rename(dest_dir / tmp.name[1:-len(".tmp")])
    n_dest = _rows(sorted(dest_dir.glob("part-*.parquet")))
    if n_dest != n_src:
        raise RuntimeError(f"{src_dir}: {n_dest} rows written, {n_src} in the source")
    return src_dir, n_src, "compacted"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rows-per-file", type=int, default=5_000_000)
    p.add_argument("--remove-source", action="store_true")
    args = p.parse_args()

    leaf_dirs = sorted({f.parent for f in args.src.glob("*/*/*/*.parquet")})
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(compact, d, args.dest / d.relative_to(args.src), args.rows_per_file) for d in leaf_dirs]
        for fut in futures:
            src_dir, n, status = fut.result()
            print(f"{src_dir.relative_to(args.src)}: {n} rows {status}", flush=True)
            if args.remove_source:
                shutil.rmtree(src_dir)
    for done in sorted(args.src.glob("*/_DONE.json")):
        target = args.dest / done.relative_to(args.src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(done, target)
    print(f"done: {len(leaf_dirs)} directories")


if __name__ == "__main__":
    main()
