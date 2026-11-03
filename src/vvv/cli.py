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
):
    """Sums of weights, theory sums and pileup profile per MC sample -> metadata/<sample>.json."""
    from coffea import processor

    from vvv.processors.metadata import MetadataProcessor, runs_sums, summarise
    from vvv.schema import VVVScoutingSchema

    metadata_dir.mkdir(parents=True, exist_ok=True)
    client = _gateway_client() if gateway else None
    runner = processor.Runner(executor=_executor(workers, gateway, client), schema=VVVScoutingSchema, chunksize=500_000)
    # Every file must be read: a skipped file would bias the normalisation, so no skipbadfiles here.
    filesets = {name: _load_filelist(name, redirector)[:max_files] for name in samples}
    out = runner({name: {"files": {f: "Events" for f in files}} for name, files in filesets.items()},
                 MetadataProcessor(), treename="Events")
    for name, files in filesets.items():
        record = summarise(name, out[name], runs_sums(files, map_fn=_map_fn(workers, client)))
        if record["runs"]["genEventCount"] != record["n_events"]:
            typer.echo(f"WARNING {name}: Runs genEventCount {record['runs']['genEventCount']} != {record['n_events']} events read")
        (metadata_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
        typer.echo(f"{name}: {record['n_events']} events, sumw {record['runs']['genEventSumw']:.6g}")


@app.command()
def run(
    samples: list[str] = typer.Argument(..., help="sample names"),
    output: Path = typer.Option(None, help="output pickle (default /work/users/$USER/vvv/hists/<first>.pkl)"),
    skim_dir: Path = typer.Option(None, help="write Parquet skims here (e.g. /work/users/$USER/vvv/skims)"),
    skim_prescale: Path = typer.Option(None, help="JSON {sample: {category: P}} from scripts/make_skim_prescales.py"),
    skim_theory: bool = typer.Option(False, help="also write the theory-weight columns to the skims"),
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
                             skim_theory=skim_theory)
    fileset = {
        name: {"files": {f: "Events" for f in _load_filelist(name, redirector)[:max_files]}} for name in samples
    }
    runner = processor.Runner(executor=_executor(workers, gateway), schema=VVVScoutingSchema,
                              chunksize=chunksize, maxchunks=maxchunks, skipbadfiles=True)
    out = runner(fileset, proc, treename="Events")
    output = output or _work_dir() / "hists" / f"{samples[0]}.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        pickle.dump(out, handle)
    typer.echo(json.dumps(out["cutflow"], indent=1))
    typer.echo(f"wrote {output}")


if __name__ == "__main__":
    app()
