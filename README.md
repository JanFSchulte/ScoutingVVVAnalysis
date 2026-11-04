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
`--gateway` runs on the user's Dask Gateway cluster. Create it with `pixi_project` set to
the world-readable mirror (`scripts/sync_work_mirror.sh`) and with env
`X509_USER_PROXY=/work/users/$USER/x509up` and
`X509_CERT_DIR=/cvmfs/cms.cern.ch/grid/etc/grid-security/certificates`: the workers have no
CA directory of their own, and without it every read from a non-Purdue site fails as
"Operation expired" (GSI cannot verify the server). The default `--redirector auto` reads
`/store/user/jschulte/` from Purdue EOS (`root://eos.cms.rcac.purdue.edu/`) and every
other file through `root://cms-xrd-global.cern.ch/`, as the legacy conversion does;
XCache timed out on these user files on 2026-11-03. `--redirector /eos/purdue` reads
Purdue EOS files through the session mount, for local tests only. `--max-files N` limits a test run;
`--maxchunks` alone still preprocesses every file of the sample.

## Training (XGBoost on GPU via Ray)

The k-fold, mass-decorrelated multiclass BDTs (port of legacy `selections/BDT/train.py`)
train on the AF Ray cluster, one GPU task per (category, fold). They run in the AF global
environment, which has Ray, a CUDA build of XGBoost and torch; submit from a session
without a GPU.

```bash
# 1. Training config from the legacy production k-fold configs; QCD x2 over legacy
#    (class budget in boosted3, per-sample cap 1.5M elsewhere) for the decorrelation
pixi run python scripts/make_training_config.py configs/training/production.json \
    --skim-dir /depot/cms/users/$USER/VVV/skims --output-dir /work/users/$USER/vvv/models/production \
    --class-scale QCD=2
# 2. Training skims: nominal only, each (sample, category) prescaled to keep 1.5x what the
#    training config draws from it
pixi run python scripts/make_skim_prescales.py configs/skim_prescales.json
pixi run vvv metadata <samples...> --gateway
pixi run vvv run <samples...> --gateway --skim-dir /work/users/$USER/vvv/skims \
    --skim-prescale configs/skim_prescales.json
# 2b. One file per chunk is ~200k files and ~100 GB, over the /work quota: merge them onto /depot
pixi run python scripts/compact_skims.py /work/users/$USER/vvv/skims /depot/cms/users/$USER/VVV/skims --remove-source
# 3. Submit (resubmitting the same config resumes; finished folds are skipped)
/work/pixi/global/.pixi/envs/default/bin/python scripts/submit_training.py configs/training/production.json --workers 8
/work/pixi/global/.pixi/envs/default/bin/python scripts/submit_training.py --logs <job id>
```

The models land in the legacy layout (`<output_dir>/<category>/fold<r>/<category>_model.json`,
`branch.json`, `selection.json`, plus `training_log.json`), so `vvv run --model-dir
<category>=<output_dir>` uses them directly. Events are assigned to folds by `event % 5`, for
MC and data alike.

Every fold ends with the legacy diagnostic plots (`src/vvv/ml/plots.py`), written into its fold
directory with the legacy names: `roc_*`, `score_*_vs_*`, `importance`, `decor_corr_{train,test}`,
`decor_score_vs_branch`, `decor_branch_shapes_by_signal_score` (each `_cls` for stage 1 and
`_decorr` for the final model), `feature_corr`, `loss_*`, `branches/`, and `plot_summary.json`
(AUCs and score/mass correlations). A plotting failure is logged and does not fail the fold.
To (re)make them on CPU, e.g. after a failure or for folds trained before the plots existed
(their loss curves are recomputed from the model every `--loss-every` rounds):

```bash
/work/pixi/global/.pixi/envs/default/bin/python scripts/plot_training.py /work/users/$USER/vvv/models/production \
    [--categories boosted3] [--folds 0 1] [--overwrite]
```

What follows legacy exactly: features, thresholds, preprocessing, the weight chain
(sample normalisation, sign-preserving |w|, class balancing after cuts, cap at 20 x the class
mean), the stage-1 learning-rate schedule and early stopping (checks every 5 rounds), and the
stage-2 smooth Cramer-von Mises penalty on all six class log-odds against msoftdrop_1,
log(sphere_mass) and mass_fraction_1 for QCD, with its decor_scale calibration.
Deliberate differences:
- folds by event number instead of contiguous entry blocks;
- random instead of contiguous per-sample subsampling;
- GPU `hist` tree building (legacy XGBoost 1.7.5 ignored `device="cuda"` and ran on CPU).

On a T4, one stage-2 objective evaluation for 3M QCD events takes 4.5 s in float64, about
2 h for 1500 rounds (legacy: 12-72 h per fold on 32 CPUs). `decor_dtype: "float32"` is 4x
faster but moves the penalty by 14% at that size (`scripts/bench_decor.py`).

## Validation against legacy

| What | Reference | Result |
|---|---|---|
| Category assignment, nominal + 8 jet variations | legacy batch `www_0` (WWW, 3 files, 76,712 preselected events) | 100% event-by-event agreement in all 9 configurations (`scripts/sync_legacy.py`) |
| Corrected AK8 pT/msd, MET, ISR score | same | bit-identical (float32), incl. JER-smeared MC |
| ~110 classifier/threshold/decorrelation variables, 5 categories | same | all within 1e-5 x max(|ref|, 1); most bit-identical (`scripts/sync_variables.py`) |
| Metadata: event count, genweight mean, pileup weights and means | legacy sample.json, pileup CSV (all 23 WWW files) | identical; pileup weights to 5e-11 |
| Per-event pileup weight, genWeight | legacy trees | 2e-7 (float32), exact |
| Classifier scores, production k-fold boosters | legacy `model_io.predict_model_proba` on legacy trees | 1e-7; 6 of 22k events differ by up to 2e-3 (a float32 ulp in HT/aplanarity crosses a split) |
| Data: categories and all variables | legacy batch `data_2024_0` (2 files of 2024G, 66,729 golden events) | 100% event-by-event agreement; variables within tolerance (`sync_*.py` without `--mc`) |

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
