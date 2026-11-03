"""Sample catalogue (data/samples.yaml).

The catalogue is imported once from the legacy src/sample.json with
``vvv import-samples`` and is never written by processing: metadata derived
from the files (sums of weights, pileup profile) lives in metadata/<sample>.json.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vvv.config import DATA_DIR

SAMPLES_FILE = DATA_DIR / "samples.yaml"

# Legacy keys copied for cross-checks only; never used for normalisation.
LEGACY_METADATA_KEYS = ("raw_entries", "genweight_mean", "weight_pu_mean", "weight_pu_up_mean", "weight_pu_down_mean")


@dataclass(frozen=True)
class Sample:
    name: str
    das: tuple[str, ...]
    kind: str  # "data", "signal" or "background"
    sample_id: int
    process_class: str | None = None  # VVV, VH, Top, VV, VJets, QCD
    xsec_fb: float | None = None
    lumi_fb: float | None = None
    has_theory_weights: bool = False
    legacy: dict = field(default_factory=dict, compare=False)

    @property
    def is_mc(self):
        return self.kind != "data"


def load_samples(path: Path = SAMPLES_FILE) -> dict[str, Sample]:
    raw = yaml.safe_load(Path(path).read_text())
    return {
        name: Sample(name=name, das=tuple(entry.pop("das")), **entry)
        for name, entry in raw["samples"].items()
    }


def import_legacy(sample_json: Path, class_groups: dict[str, list[str]], out: Path = SAMPLES_FILE):
    """Write samples.yaml from the legacy sample.json and BDT class groups."""
    legacy = json.loads(Path(sample_json).read_text())["sample"]
    class_of = {s: cls for cls, members in class_groups.items() for s in members}
    samples = {}
    for entry in legacy:
        paths = entry["path"]
        das = [paths] if isinstance(paths, str) else list(paths)
        kind = "data" if not entry["is_MC"] else ("signal" if entry["is_signal"] else "background")
        record = {
            "das": das,
            "kind": kind,
            "sample_id": entry["sample_ID"],
            "process_class": class_of.get(entry["name"]),
            "has_theory_weights": bool(entry.get("has_theory_weights", False)),
            "legacy": {k: entry[k] for k in LEGACY_METADATA_KEYS if k in entry},
        }
        if kind == "data":
            record["lumi_fb"] = entry["lumi"]
        else:
            record["xsec_fb"] = entry["xsection"]
        samples[entry["name"]] = record
    header = (
        "# Imported from legacy ScoutingVVVTools src/sample.json by `vvv import-samples`.\n"
        "# Cross sections in fb (XSDB pb x 1000, VV scaled to NNLO, V+jets NLO/LO k-factors applied).\n"
    )
    out.write_text(header + yaml.safe_dump({"samples": samples}, sort_keys=False, width=200))
    return out
