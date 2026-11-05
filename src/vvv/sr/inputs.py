"""Inputs of the SR optimiser from the coffea MC fit-region skims.

`vvv run --skim-cut <training config> --model-dir ...` writes, per MC sample and category, the
events passing every category threshold but the msoftdrop_1 window, with 20 <= msoftdrop_1 < 250
GeV, their out-of-fold proba_<class> and the weight (x lumi, x skim prescale). The scan events are
those inside the msoftdrop_1 threshold window, the ABCD control events those outside it and
strictly inside a control window (as the legacy loader). A class is signal if all its samples are.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from vvv.ml.inference import CLASSES, threshold_mask
from vvv.sr.optimise import ScanInputs

MASS_BRANCH = "ScoutingFatPFJetRecluster_msoftdrop_1"


def load_skim_inputs(cfg, skim_dir, category, training_config, catalogue=None, log=print):
    """ScanInputs of one category from <skim_dir>/<sample>/<category>/nominal/ for the samples of
    the training config's class_groups bar the category's exclude_samples (weights as stored:
    already x lumi)."""
    from vvv.samples import load_samples

    catalogue = catalogue or load_samples()
    tcfg = json.loads(Path(training_config).read_text())
    groups = tcfg["class_groups"]
    if list(groups) != list(CLASSES):
        raise ValueError(f"class_groups {list(groups)} are not in the model's class order {CLASSES}")
    window = tcfg["categories"][category]["thresholds"][MASS_BRANCH]
    excluded = set(tcfg["categories"][category].get("exclude_samples", []))
    signal = [all(catalogue[s].kind == "signal" for s in members) for members in groups.values()]
    columns = [f"proba_{c}" for c in CLASSES] + ["weight", MASS_BRANCH]
    parts = []
    for idx, members in enumerate(groups.values()):
        for sample in members:
            if sample in excluded:
                continue
            if not (Path(skim_dir) / sample / "_DONE.json").exists():
                raise FileNotFoundError(f"{skim_dir}/{sample}: no _DONE.json (skim incomplete)")
            path = Path(skim_dir) / sample / category / "nominal"
            if not path.exists():
                log(f"  {sample}: no {category} events")
                continue
            t = ds.dataset(path).to_table(columns=columns)
            parts.append({"proba": np.column_stack([t[c].to_numpy() for c in columns[:len(CLASSES)]]),
                          "weight": t["weight"].to_numpy(), "msd": t[MASS_BRANCH].to_numpy(),
                          "class_idx": np.full(t.num_rows, idx), "sample": np.full(t.num_rows, sample, dtype=object)})
            log(f"  {sample}: {t.num_rows} events")
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    abcd = cfg.abcd
    if abcd is not None:
        return ScanInputs.from_events(cat["proba"], cat["weight"], cat["class_idx"], cat["sample"], list(groups),
                                      signal, abcd_values=cat["msd"], abcd_cond=window,
                                      control_windows=abcd.windows)
    scan = threshold_mask({"x": cat["msd"]}, {"x": window})
    return ScanInputs.from_events(cat["proba"][scan], cat["weight"][scan], cat["class_idx"][scan],
                                  cat["sample"][scan], list(groups), signal)
