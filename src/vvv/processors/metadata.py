"""Per-sample metadata over all generated events.

Replaces the legacy write-back into sample.json (raw_entries, genweight_mean,
weight_pu*_mean) and the per-sample pileup CSVs of weight.C: one pass over the
Events tree for the Pileup_nTrueInt profile, plus the Runs-tree sums of weights
and theory-weight sums used to normalise the theory variations.
"""

import awkward as ak
import numpy as np
import uproot
from coffea import processor

from vvv.corrections.pileup import load_data_profiles, mean_weights, pileup_weights

PU_EDGES = np.linspace(0.0, 200.0, 201)


class MetadataProcessor(processor.ProcessorABC):
    def process(self, events):
        dataset = events.metadata["dataset"]
        out = {"n_events": len(events), "pileup": np.zeros(len(PU_EDGES) - 1)}
        if "Pileup" in events.fields:
            counts, _ = np.histogram(ak.to_numpy(events.Pileup.nTrueInt), bins=PU_EDGES)
            out["pileup"] = counts.astype(np.float64)
            out["sum_genweight"] = float(ak.sum(events.genWeight))
        return {dataset: out}

    def postprocess(self, accumulator):
        return accumulator


def runs_file_sums(path, timeout=60):
    """Runs-tree sums of one file; theory sums are genEventSumw-weighted (combine with runs_sums)."""
    runs = uproot.open(path, timeout=timeout)["Runs"].arrays(library="np")
    sumw = runs["genEventSumw"]
    out = {"genEventSumw": float(sumw.sum()), "genEventCount": int(runs["genEventCount"].sum()), "weighted": {}}
    for name in ("LHEScaleSumw", "LHEPdfSumw", "PSSumw"):
        if name not in runs:
            continue
        acc = None
        for w, values in zip(sumw, runs[name]):
            term = w * np.asarray(values, dtype=np.float64)
            if acc is not None and len(acc) != len(term):
                raise ValueError(f"{path}: {name} length {len(term)} != {len(acc)}")
            acc = term if acc is None else acc + term
        out["weighted"][name] = acc
    return out


def runs_sums(files, timeout=60, map_fn=map):
    """Sums of the Runs tree over files (map_fn spreads the per-file reads, e.g. over Dask)."""
    total = {"genEventSumw": 0.0, "genEventCount": 0, "n_files": 0}
    weighted = {}
    for path, one in zip(files, map_fn(lambda f: runs_file_sums(f, timeout), files)):
        total["genEventSumw"] += one["genEventSumw"]
        total["genEventCount"] += one["genEventCount"]
        total["n_files"] += 1
        for name, acc in one["weighted"].items():
            if name in weighted and len(weighted[name]) != len(acc):
                raise ValueError(f"{path}: {name} length {len(acc)} != {len(weighted[name])}")
            weighted[name] = acc if name not in weighted else weighted[name] + acc
    for name, acc in weighted.items():
        total[name] = (acc / total["genEventSumw"]).tolist()
    return total


def summarise(sample, events_output, runs):
    """The metadata record of one MC sample."""
    profiles_edges, profiles = load_data_profiles()
    if not np.array_equal(profiles_edges, PU_EDGES):
        raise ValueError("data pileup profiles do not use the 0-200 unit binning")
    weights = pileup_weights(events_output["pileup"], profiles)
    return {
        "sample": sample,
        "n_events": int(events_output["n_events"]),
        "sum_genweight": events_output.get("sum_genweight"),
        "runs": runs,
        "pileup_mc": events_output["pileup"].tolist(),
        "pileup_weights": {k: v.tolist() for k, v in weights.items()},
        "pileup_mean": mean_weights(events_output["pileup"], events_output["n_events"], weights),
    }
