# vvv: Scouting VVV analysis on coffea

Re-implementation of the conversion, systematics and histogram stages of
ScoutingVVVTools (`../CMSSW_16_1_0_pre4/src/ScoutingVVVTools`, git 421d4d7) with
coffea 2026.9, awkward, correctionlib and hist. What is ported and why is in
`specification.md`; the phased plan and decisions are in `DEVELOPMENT_PLAN.md`.

## Setup

```bash
cd /depot/cms/private/users/schul105/VVV/analysis/coffea
pixi install            # environment in .pixi/ (visible to Slurm and Dask Gateway workers)
pixi run test           # unit tests
```

Remote reads need a VOMS proxy that workers can read:

```bash
export X509_USER_PROXY=/work/users/$USER/x509up
voms-proxy-init -voms cms -valid 192:00
```

## Running

```bash
pixi run vvv import-samples          # once: data/samples.yaml from legacy sample.json
pixi run vvv import-filelists        # once: data/filelists/ from legacy conversion snapshots
pixi run vvv metadata www            # sums of weights, theory sums, pileup profile -> metadata/www.json
pixi run vvv run www \
    --model-dir boosted3=<k-fold training dir> \
    --signal-regions boosted3=<signal_region.csv> \
    --skim-dir /work/users/$USER/vvv/skims
```

`vvv run` makes one pass per sample: jet corrections (with rho recovery), the
five categories for nominal and the eight jet variations, the classifier inputs,
weights and weight variations, and, for categories with a model, K-fold scores,
signal-region assignment and the mass-fit histograms:

- `mass[dataset, category, region, syst, mass]`: syst = nominal, jet variations, pileup up/down;
- `LHEScaleWeight`, `LHEPdfWeight`, `LHEPdfWeightAlphaS`, `PSWeight`
  `[dataset, category, region, member, mass]`: members normalised to their generator-level sums;
- `cutflow`, `rho_failures`, `rho_consensus`.

MC is normalised to `--lumi` (default 255.8 fb^-1, the projection target). Data
in an SR box inside the 50-150 GeV mass window is dropped unless `--unblind`.
`--gateway` runs on the user's Dask Gateway cluster (create it with
`pixi_project` set to this directory). `--redirector /eos/purdue` reads Purdue
EOS files through the session mount, for local tests only.

## Validation against legacy

| What | Reference | Result |
|---|---|---|
| Category assignment, nominal + 8 jet variations | legacy batch `www_0` (WWW, 3 files, 76,712 preselected events) | 100% event-by-event agreement in all 9 configurations (`scripts/sync_legacy.py`) |
| Corrected AK8 pT/msd, MET, ISR score | same | bit-identical (float32), incl. JER-smeared MC |
| ~110 classifier/threshold/decorrelation variables, 5 categories | same | all within 1e-5 x max(|ref|, 1); most bit-identical (`scripts/sync_variables.py`) |
| Metadata: event count, genweight mean, pileup weights and means | legacy sample.json, pileup CSV (all 23 WWW files) | identical; pileup weights to 5e-11 |
| Per-event pileup weight, genWeight | legacy trees | 2e-7 (float32), exact |
| Classifier scores, production k-fold boosters | legacy `model_io.predict_model_proba` on legacy trees | 1e-7; 6 of 22k events differ by up to 2e-3 (a float32 ulp in HT/aplanarity crosses a split) |
| Data | not yet validated | needs a grid proxy (the data files are not at Purdue) |

## Legacy behaviours reproduced on purpose

- Leptons are massless; kinematic cuts use the pT/eta of a `TLorentzVector`
  rebuilt from (pT, eta, phi, m), which moves stored |eta| = 2.5 values to
  either side of the cut.
- Track-less electrons carry charge 2147483647; legacy casts it through float
  to int (INT_MIN), so they pair "opposite-sign" with any charged electron in
  the 2-lepton SFOS search. `ObjectConfig.legacy_electron_charge = False`
  switches to physical charges (95 -> 88 2-lepton WWW events on one file).
- Inputs are truncated to 30 AK4 / 20 AK8 / 30 leptons, and AK4 jets with
  raw pT < 19.9 GeV are dropped, before any correction.
- JMR < 1 gives no smearing, so the JMR variations equal nominal with the
  current `jms_jmr_results.json`.

## Layout

```
src/vvv/schema.py              NanoEvents schema for the ScoutingNano collections
src/vvv/corrections/jets.py    JEC, rho recovery, JER, JMS/JMR, Type-1 MET
src/vvv/corrections/pileup.py  pileup weights (correctionlib export, legacy CSV import)
src/vvv/objects.py             objects, ISR score, cleaning, categories
src/vvv/variables.py           per-category classifier and selection variables
src/vvv/selection.py           inputs -> corrected objects -> categories, per jet configuration
src/vvv/weights.py             normalisation, pileup and theory weight variations
src/vvv/ml/inference.py        K-fold XGBoost inference, BDT thresholds, SR boxes
src/vvv/processors/            metadata and analysis processors
src/vvv/cli.py                 `vvv` command line
scripts/sync_*.py              event-by-event comparisons with legacy outputs
```
