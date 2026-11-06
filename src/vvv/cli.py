"""Command-line interface: vvv <command> --help."""

import json
import os
import pickle
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="Scouting VVV analysis on coffea.")

LEGACY_ROOT = Path(__file__).resolve().parents[3] / "CMSSW_16_1_0_pre4" / "src" / "ScoutingVVVTools"
FILELIST_DIR = Path(__file__).resolve().parents[2] / "data" / "filelists"
METADATA_DIR = Path(__file__).resolve().parents[2] / "metadata"
XCACHE = "root://xcache.cms.rcac.purdue.edu/"
GLOBAL = "root://cms-xrd-global.cern.ch/"
# "auto" (legacy input_xrootd_prefixes): Purdue EOS for the files stored there, else the global
# redirector. XCache times out on these user files.
AUTO_REDIRECTORS = {"/store/user/jschulte/": "root://eos.cms.rcac.purdue.edu/"}


def _work_dir():
    return Path("/work/users") / os.environ.get("NB_USER", os.environ.get("USER", "")) / "vvv"


def _gateway_client():
    from dask_gateway import Gateway

    gw = Gateway()
    clusters = gw.list_clusters()
    if not clusters:
        raise typer.BadParameter("no running Dask Gateway cluster; create one first (one per user)")
    return gw.connect(clusters[0].name).get_client()


def _executor(workers: int, gateway: bool, client=None):
    from coffea import processor

    if not gateway:
        if workers <= 1:
            return processor.IterativeExecutor()
        return processor.FuturesExecutor(workers=workers)
    return processor.DaskExecutor(client=client or _gateway_client())


def _map_fn(workers: int, client=None):
    """map() over files: on the Dask cluster, else a local thread pool (I/O bound)."""
    if client is not None:
        return lambda fn, items: client.gather(client.map(fn, list(items), pure=False, retries=3))
    from concurrent.futures import ThreadPoolExecutor

    def fn_map(fn, items):
        with ThreadPoolExecutor(max(workers, 1)) as pool:
            return list(pool.map(fn, items))
    return fn_map


def _load_filelist(sample: str, redirector: str | None):
    files = json.loads((FILELIST_DIR / f"{sample}.json").read_text())["files"]
    if not redirector:
        return files
    out = []
    for f in files:
        lfn = f[f.index("/store/"):]
        if redirector == "auto":
            prefix = next((r for p, r in AUTO_REDIRECTORS.items() if lfn.startswith(p)), GLOBAL)
        else:
            prefix = redirector
        out.append(prefix + lfn)
    return out


def _drop_excluded(sample: str, files: list[str]):
    """files minus those listed in data/filelists/<sample>.excluded.json (unreadable or broken),
    applied after batching so batch boundaries stay fixed; returns (kept, dropped)."""
    path = FILELIST_DIR / f"{sample}.excluded.json"
    excluded = set(json.loads(path.read_text())["files"]) if path.exists() else set()
    kept = [f for f in files if f[f.index("/store/"):] not in excluded]
    return kept, [f for f in files if f[f.index("/store/"):] in excluded]


@app.command("import-samples")
def import_samples(
    sample_json: Path = LEGACY_ROOT / "src" / "sample.json",
    bdt_config: Path = LEGACY_ROOT / "selections/BDT/study_boosted3/kfold_finedecor_jec_powheg/bdt_config.json",
):
    """Write data/samples.yaml from the legacy sample.json (run once)."""
    from vvv.samples import import_legacy

    out = import_legacy(sample_json, json.loads(bdt_config.read_text())["class_groups"])
    typer.echo(f"wrote {out}")


@app.command("import-filelists")
def import_filelists(legacy_dataset: Path = LEGACY_ROOT / "dataset_hybrid_leptonic_jec_c"):
    """File lists from the legacy conversion snapshots (<group>_tmp/<sample>.files)."""
    FILELIST_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for snapshot in sorted(legacy_dataset.glob("*_tmp/*.files")):
        files = [line.strip() for line in snapshot.read_text().splitlines() if line.strip()]
        files = ["/store/" + f.split("/store/", 1)[1] for f in files]
        (FILELIST_DIR / f"{snapshot.stem}.json").write_text(json.dumps({"source": str(snapshot), "files": files}, indent=1))
        n += 1
    typer.echo(f"wrote {n} file lists to {FILELIST_DIR}")


@app.command()
def metadata(
    samples: list[str] = typer.Argument(..., help="sample names"),
    workers: int = 8,
    gateway: bool = False,
    redirector: str = typer.Option("auto", help='"auto" (EOS / global), or a prefix such as ' + XCACHE),
    metadata_dir: Path = METADATA_DIR,
    max_files: int = typer.Option(None, help="first N files only (tests; normalisation is then partial)"),
    overwrite: bool = typer.Option(False, help="redo samples whose metadata exists"),
    xrootd_timeout: int = 300,
):
    """Sums of weights, theory sums and pileup profile per MC sample -> metadata/<sample>.json.

    One sample at a time; samples with a record are skipped (resume), and a failing sample
    is reported at the end without stopping the others.
    """
    from coffea import processor

    from vvv.processors.metadata import MetadataProcessor, runs_sums, summarise
    from vvv.schema import VVVScoutingSchema

    metadata_dir.mkdir(parents=True, exist_ok=True)
    client = _gateway_client() if gateway else None
    # Every file must be read: a skipped file would bias the normalisation, so no skipbadfiles here.
    runner = processor.Runner(executor=_executor(workers, gateway, client), schema=VVVScoutingSchema,
                              chunksize=500_000, xrootdtimeout=xrootd_timeout)
    failed = []
    for name in samples:
        path = metadata_dir / f"{name}.json"
        if path.exists() and not overwrite:
            typer.echo(f"{name}: done, skipped")
            continue
        files = _load_filelist(name, redirector)[:max_files]
        try:
            out = runner({name: {"files": {f: "Events" for f in files}}}, MetadataProcessor(), treename="Events")
            record = summarise(name, out[name], runs_sums(files, timeout=xrootd_timeout, map_fn=_map_fn(workers, client)))
        except Exception as err:  # noqa: BLE001 - report it, go on with the other samples
            failed.append(name)
            typer.echo(f"{name}: FAILED {type(err).__name__}: {err}")
            continue
        if record["runs"]["genEventCount"] != record["n_events"]:
            typer.echo(f"WARNING {name}: Runs genEventCount {record['runs']['genEventCount']} != {record['n_events']} events read")
        path.write_text(json.dumps(record, indent=1))
        typer.echo(f"{name}: {record['n_events']} events, sumw {record['runs']['genEventSumw']:.6g}")
    if failed:
        typer.echo(f"{len(failed)} samples failed (rerun to retry): {' '.join(failed)}")
        raise typer.Exit(1)


@app.command()
def run(
    samples: list[str] = typer.Argument(..., help="sample names"),
    output: Path = typer.Option(None, help="output pickle (default /work/users/$USER/vvv/hists/<first>.pkl); "
                                "with --skim-dir, the directory of the per-sample pickles"),
    skim_dir: Path = typer.Option(None, help="write Parquet skims here (e.g. /work/users/$USER/vvv/skims)"),
    skim_prescale: Path = typer.Option(None, help="JSON {sample: {category: P}} from scripts/make_skim_prescales.py"),
    skim_theory: bool = typer.Option(False, help="also write the theory-weight columns to the skims"),
    skim_cut: Path = typer.Option(None, help="training config (configs/training/*.json): skim only events passing "
                                  "its category thresholds bar the msoftdrop_1 window, with msoftdrop_1 in the "
                                  "mass-fit range (data skims)"),
    jet_variations: bool = typer.Option(True, help="with --model-dir, also run the eight jet variations (MC)"),
    every_nth_file: int = typer.Option(None, help="use every N-th file of each sample (e.g. a small unthresholded "
                                       "data skim for plots); recorded in _DONE.json"),
    model_dir: list[str] = typer.Option([], help="CATEGORY=DIR of a legacy-format K-fold training"),
    signal_regions: list[str] = typer.Option([], help="CATEGORY=signal_region.csv"),
    lumi: float = typer.Option(255.8, help="fb^-1 for MC normalisation"),
    workers: int = 8,
    gateway: bool = False,
    chunksize: int = 200_000,
    maxchunks: int = typer.Option(None, help="limit the number of chunks (testing)"),
    max_files: int = typer.Option(None, help="use only the first N files per sample (testing)"),
    redirector: str = typer.Option("auto", help='"auto" (EOS / global), or a prefix such as ' + XCACHE),
    metadata_dir: Path = METADATA_DIR,
    unblind: bool = False,
    xrootd_timeout: int = 300,
    batch_files: int = typer.Option(None, help="with --skim-dir: process each sample in batches of N files, "
                                    "each resumable on its own (large samples over unreliable links)"),
):
    """Process samples: skims and/or mass-fit histograms for nominal and all variations."""
    from coffea import processor

    from vvv.config import DEFAULT
    from vvv.ml.inference import load_signal_regions
    from vvv.processors.analysis import AnalysisProcessor
    from vvv.samples import load_samples
    from vvv.schema import VVVScoutingSchema

    catalogue = load_samples()
    meta = {}
    for name in samples:
        if catalogue[name].is_mc:
            path = metadata_dir / f"{name}.json"
            if not path.exists():
                raise typer.BadParameter(f"{name}: run `vvv metadata {name}` first")
            meta[name] = json.loads(path.read_text())
    models = dict(item.split("=", 1) for item in model_dir)
    regions = {cat: load_signal_regions(p) for cat, p in (item.split("=", 1) for item in signal_regions)}
    proc = AnalysisProcessor(DEFAULT, catalogue, meta, lumi, models=models, signal_regions=regions,
                             skim_dir=skim_dir, blind=not unblind,
                             skim_prescale=json.loads(skim_prescale.read_text())["prescale"] if skim_prescale else None,
                             skim_theory=skim_theory,
                             skim_cut={c: v["thresholds"] for c, v in json.loads(skim_cut.read_text())["categories"].items()}
                             if skim_cut else None, jet_variations=jet_variations)
    if skim_dir is None:
        fileset = {
            name: {"files": {f: "Events" for f in _drop_excluded(name, _load_filelist(name, redirector)[::every_nth_file or 1][:max_files])[0]}}
            for name in samples
        }
        runner = processor.Runner(executor=_executor(workers, gateway), schema=VVVScoutingSchema,
                                  chunksize=chunksize, maxchunks=maxchunks, skipbadfiles=True,
                                  xrootdtimeout=xrootd_timeout)
        out = runner(fileset, proc, treename="Events")
        output = output or _work_dir() / "hists" / f"{samples[0]}.pkl"
        _dump(out, output)
        return

    # Skims: one sample at a time and no skipped files, so every skim covers its whole
    # sample. A finished sample gets <skim_dir>/<sample>/_DONE.json and is skipped on a rerun;
    # chunk files are named by file and entry range, so re-running a failed sample overwrites
    # its partial output instead of duplicating it. With --batch-files the same holds per batch
    # of files (<skim_dir>/<sample>/_batches/b<N>_<i>.json), and _DONE.json follows the last batch.
    client = _gateway_client() if gateway else None
    runner = processor.Runner(executor=_executor(workers, gateway, client), schema=VVVScoutingSchema,
                              chunksize=chunksize, maxchunks=maxchunks, xrootdtimeout=xrootd_timeout)
    hist_dir = output or _work_dir() / "hists" / skim_dir.name
    complete = maxchunks is None and max_files is None
    failed = []
    for name in samples:
        done = skim_dir / name / "_DONE.json"
        if done.exists():
            typer.echo(f"{name}: done, skipped")
            continue
        files = _load_filelist(name, redirector)[::every_nth_file or 1][:max_files]
        extra = {k: v for k, v in (("every_nth_file", every_nth_file), ("skim_cut", str(skim_cut) if skim_cut else None)) if v}
        if not batch_files:
            files, dropped = _drop_excluded(name, files)
            if dropped:
                extra["excluded_files"] = dropped
            try:
                out = runner({name: {"files": {f: "Events" for f in files}}}, proc, treename="Events")
            except Exception as err:  # noqa: BLE001 - report it, go on with the other samples
                failed.append(name)
                typer.echo(f"{name}: FAILED {type(err).__name__}: {err}")
                continue
            _dump(out, hist_dir / f"{name}.pkl")
            if complete:
                _write_done(done, len(files), chunksize, out["cutflow"][name], **extra)
            typer.echo(f"{name}: {json.dumps(out['cutflow'][name])}")
            continue

        batches = [files[i:i + batch_files] for i in range(0, len(files), batch_files)]
        markers = [skim_dir / name / "_batches" / f"b{batch_files}_{i:05d}.json" for i in range(len(batches))]
        n_failed = 0
        for i, (batch, marker) in enumerate(zip(batches, markers)):
            if marker.exists():
                continue
            batch, dropped = _drop_excluded(name, batch)
            try:
                out = runner({name: {"files": {f: "Events" for f in batch}}}, proc, treename="Events")
            except Exception as err:  # noqa: BLE001 - report it, go on with the next batch
                n_failed += 1
                typer.echo(f"{name} batch {i}/{len(batches)}: FAILED {type(err).__name__}: {err}")
                continue
            _dump(out, hist_dir / name / f"b{batch_files}_{i:05d}.pkl")
            if complete:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"files": batch, "excluded_files": dropped, "cutflow": out["cutflow"][name]},
                                             indent=1))
            typer.echo(f"{name} batch {i}/{len(batches)}: {json.dumps(out['cutflow'][name])}")
        if n_failed:
            failed.append(name)
            typer.echo(f"{name}: {n_failed} of {len(batches)} batches failed")
        elif complete:
            cutflow, dropped = {}, []
            for marker in markers:
                info = json.loads(marker.read_text())
                dropped += info.get("excluded_files", [])
                for key, n in info["cutflow"].items():
                    cutflow[key] = cutflow.get(key, 0) + n
            if dropped:
                extra["excluded_files"] = dropped
            _write_done(done, len(files) - len(dropped), chunksize, cutflow, batch_files=batch_files, **extra)
            typer.echo(f"{name}: {json.dumps(cutflow)}")
    if failed:
        typer.echo(f"{len(failed)} samples failed (rerun to retry): {' '.join(failed)}")
        raise typer.Exit(1)


@app.command("ttbar-cr")
def ttbar_cr(
    samples: list[str] = typer.Argument(..., help="sample names (configs/jms_jmr.json lists the CR samples)"),
    output_dir: Path = typer.Option(None, help="per-sample output pickles (default /work/users/$USER/vvv/ttbar_cr)"),
    skim_dir: Path = typer.Option(None, help="also write the ttbar_cr tree as Parquet here (for refits)"),
    fit_config: Path = typer.Option(Path(__file__).resolve().parents[2] / "configs" / "jms_jmr.json",
                                    help="JMS/JMR fit config: the histograms are filled for its selection"),
    jet_variations: bool = typer.Option(False, help="also fill the histograms for JES/JER up/down (MC)"),
    lumi: float = typer.Option(None, help="fb^-1 for MC normalisation (default: data_2024 luminosity)"),
    workers: int = 8,
    gateway: bool = False,
    chunksize: int = 200_000,
    maxchunks: int = typer.Option(None, help="limit the number of chunks (testing)"),
    max_files: int = typer.Option(None, help="use only the first N files per sample (testing)"),
    batch_files: int = typer.Option(None, help="process each sample in batches of N files, one pickle per batch; "
                                    "finished batches are skipped on a rerun"),
    redirector: str = typer.Option("auto", help='"auto" (EOS / global), or a prefix such as ' + XCACHE),
    metadata_dir: Path = METADATA_DIR,
    xrootd_timeout: int = 300,
):
    """Semileptonic ttbar control region: probe-jet histograms and skims for the JMS/JMR fit.

    Jet corrections as the analysis but without JMS/JMR, the SingleMuon trigger, the golden JSON
    for data. Output <output-dir>/<sample>.pkl (or <sample>/b<N>_<i>.pkl with --batch-files):
    cutflow, the pass/fail fit histogram and the CR control distributions;
    scripts/fit_jms_jmr.py fits them.
    """
    from coffea import processor

    from vvv.config import TTBAR_CR
    from vvv.processors.ttbar_cr import TtbarCRProcessor
    from vvv.samples import load_samples
    from vvv.schema import VVVScoutingSchema

    catalogue = load_samples()
    meta = {}
    for name in samples:
        if catalogue[name].is_mc:
            path = metadata_dir / f"{name}.json"
            if not path.exists():
                raise typer.BadParameter(f"{name}: run `vvv metadata {name}` first")
            meta[name] = json.loads(path.read_text())
    lumi = lumi if lumi is not None else catalogue["data_2024"].lumi_fb
    proc = TtbarCRProcessor(TTBAR_CR, catalogue, meta, lumi, json.loads(fit_config.read_text()),
                            skim_dir=skim_dir, jet_variations=jet_variations)
    client = _gateway_client() if gateway else None
    runner = processor.Runner(executor=_executor(workers, gateway, client), schema=VVVScoutingSchema,
                              chunksize=chunksize, maxchunks=maxchunks, skipbadfiles=True,
                              xrootdtimeout=xrootd_timeout)
    output_dir = output_dir or _work_dir() / "ttbar_cr"
    for name in samples:
        files = _load_filelist(name, redirector)[:max_files]
        batches = [files[i:i + batch_files] for i in range(0, len(files), batch_files)] if batch_files else [files]
        for i, batch in enumerate(batches):
            path = output_dir / name / f"b{batch_files}_{i:05d}.pkl" if batch_files else output_dir / f"{name}.pkl"
            if batch_files and path.exists():
                continue
            # After batching, so the batch boundaries do not move (as in `run`).
            batch, dropped = _drop_excluded(name, batch)
            out = runner({name: {"files": {f: "Events" for f in batch}}}, proc, treename="Events")
            out["excluded_files"] = dropped
            _dump(out, path)
            typer.echo(f"{name}{f' batch {i}/{len(batches)}' if batch_files else ''}: {json.dumps(out['cutflow'][name])}")


def _write_done(done: Path, n_files: int, chunksize: int, cutflow: dict, **extra):
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({"n_files": n_files, "chunksize": chunksize, **extra, "cutflow": cutflow}, indent=1))


def _dump(out, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        pickle.dump(out, handle)
    typer.echo(f"wrote {output}")


if __name__ == "__main__":
    app()
