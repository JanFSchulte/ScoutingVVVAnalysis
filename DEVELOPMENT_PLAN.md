# Development plan: Scouting VVV in coffea

This plan covers how to build what `specification.md` specifies.

The guiding rule is **validate against legacy at every stage**. The legacy chain (`convert_branch.C` and its JSON expression engine, about 15.6k unique lines of C++ and 31k lines of Python) is the reference. Each phase ends with an event-by-event or yield-level sync test, matched on (run, lumi, event). The port never becomes "the analysis" before it reproduces the legacy numbers. Differences that are *intended* fixes go in a changelog, not into tolerance knobs.

## Status (2026-11-03)

- Phase 0: done. Pixi project, schema, samples.yaml, file lists (from legacy snapshots), legacy reference = batch `www_0` of `dataset_hybrid_leptonic_jec_c`.
- Phases 1-2: done for MC. Categories and corrections agree event by event with legacy for nominal and all 8 jet variations. The smearing turned out bit-reproducible, so MC is synced exactly, not only statistically. Data (2 files of 2024G) is synced exactly too.
- Phase 3: variables done and synced. Parquet skims are written (nominal by default).
- Phase 5: the histogram processor is implemented and plumbing-tested. Still open:
  - the adapter from histograms to the legacy `combine.C` inputs (`*_syst_yields.json`, `qcd_abcd_yields.root` mass-fit directory);
  - a full-statistics run on Dask Gateway.
- Training (Phase 4, deferred) has a blocker: the production k-fold models were trained on `dataset_hybrid_leptonic_jec` (before the 2026-11-02 cleaning) with contiguous-block folds. Scores for the `_jec_c` selection therefore need a retraining. The event-number fold scheme can be introduced with that retraining.

## Decisions (2026-11-03)

- Lumi for SR/limits: 255.8 fb^-1 projection.
- rho: reconstruct now; switch to the stored branch once the next ScoutingNano production has it.
- Folds: by event number.
- Smearing: statistical equivalence with legacy is enough.
- QCD: mass fit only.
- First port: Phases 0-3 and 5. Phases 4 and 6 (training, SR optimisation, datacards) stay legacy for now.
- Location: this `/depot` directory.
- Training QCD statistics (2026-11-03): x2 over legacy (boosted3 class budget 6M; per-sample cap 1.5M in the boosted2 categories) to test whether the mass decorrelation improves. Compare against a legacy-budget training before adopting.

## Architecture

```
vvv/                         # python package (pixi project in this dir)
  samples.yaml               # immutable: names, DAS paths, xsec, groups (imported from legacy sample.json)
  config/                    # cuts, WPs, SR boxes, syst lists: one typed config (pydantic/dataclass), no path literals
  schema.py                  # ScoutingNanoAODSchema subclass: Recluster2/FatRecluster/MuonVtx -> Jet/FatJet/Muon mixins
  corrections/
    jec.py  rho.py  jer.py  jmsjmr.py  met.py  pileup.py  theory.py
  objects.py                 # jets, leptons, ISR score, cleaning (c2/c3)
  categories.py              # 5 exclusive masks + CRs, built with PackedSelection
  variables.py               # classifier features, dijet chi2 pairing, event shapes
  systematics.py             # variation loop: yields (name, shifted_events) for nominal + jes/jer/jms/jmr
  processors/
    metadata.py  skim.py  histos.py  ttbar_cr.py
  ml/        train.py  kfold.py  infer.py
  stats/     sr_opt.py  massfit_tf.py  datacards.py
  plotting/  datamc.py
  cli.py                     # Typer: vvv metadata|skim|train|hist|cards|plot
tests/                       # pytest: unit + sync tests on scouting_nano_MC_1.root
```

Key design choices compared with legacy:

- **Categories are masks, not trees.** One pass builds every category with `PackedSelection`, which also gives cutflows for free.
- **Jet systematics are a loop over shifted collections.** Weight systematics go through `coffea.analysis_tools.Weights`. This replaces 8 x 5 variation trees and six separate yield scripts that each re-read the data.
- **Two-pass data flow.**
  - Pass 1, skim: features to Parquet, roughly 100x smaller than the current 3.1 TB of ROOT. Its only consumer is ML training.
  - Pass 2, histogram: inference and every histogram and variation filled at once. Iterating on SR boxes or binning then only re-runs pass 2. With cached score columns it can run on the skims alone.
- **No write-back into shared config.** Derived metadata lives in `metadata/`, keyed by a file-list hash.
- **The fold assignment is a function of (run, event).** It no longer depends on entry order, so models survive re-skims.

## Phases

Rough effort assumes one person part-time, with an agent doing most of the coding.

### Phase 0: Infrastructure (about 1 week)

- Set up a pixi project in this directory, starting from the `/work/pixi/global` manifest. Add `xgboost`, `pydantic` and `typer`. Run `pixi install`.
  - It lives on `/depot`, so Slurm jobs and Dask Gateway workers can both read it.
  - Note: `/depot/cms/private` is not world-readable, so it cannot appear as a Jupyter kernel. If that matters, move the environment to `/work/users/schul105/`.
- Write `samples.yaml` from the legacy `sample.json`. Add a test that fails on any drift in dataset names.
- Schema: check whether `ScoutingNanoAODSchema` maps `ScoutingPFJetRecluster2`, `ScoutingFatPFJetRecluster` and `ScoutingMuonVtx`, including the `genPartIdx` cross-references. If not, subclass it and add a mixins table.
- Data access:
  - Set up the VOMS proxy at `X509_USER_PROXY=/work/users/schul105/x509up`.
  - Confirm phys03 files can be read through XCache.
  - Build file lists with `dasgoclient` and cache them per sample.
- **Legacy reference sample.** Run the legacy `convert_branch` on `scouting_nano_MC_1.root` (and one data file) with the `dataset_hybrid_leptonic_jec_c` config, nominal plus variations. Freeze its output as the sync target.

### Phase 1: Objects and categories, before corrections (about 1-2 weeks)

- Port the object IDs, ISR score, c2/c3 cleaning, SFOS pairing and the five category masks.
- Sync test: run legacy with corrections disabled, or compare at the raw level where possible. Check per-event category assignment and object multiplicities. Target 100% agreement.
- Benchmark events/s per core against `convert_branch`.

### Phase 2: Corrections (about 2-3 weeks; highest risk)

- **JEC** via correctionlib compound evaluation. The payloads are already JSON.
- **rho recovery:** vectorised root-finding over all events at once. The bracket comes from the JEC rho bin edges; use bisection or regula falsi in numpy over the event axis. The consistency check across jets matches legacy, and failures are counted rather than raised.
- **JER:** use `JERSmear` from correctionlib with a deterministic per-jet seed. Validate statistically (distributions and yields), not per jet.
- **JMS/JMR:** Gaussian smearing with a deterministic per-jet random stream; statistical validation only.
- **Type-1 MET.**
- **Pileup:** build a correctionlib JSON from data profiles, and keep the legacy CSVs as cross-checks.
- Sync test: deterministic pieces (JEC, rho, JES, JMS scale, MET from non-smeared inputs, data) agree per jet with legacy to 1e-5 relative. Smeared MC quantities agree in distribution (KS / chi2 per variable) and in per-category yields within statistical uncertainty. Re-run the Phase 1 category sync after corrections.
- Port the ttbar CR processor and refit JMS/JMR after the JEC. This is the open physics item that legacy has flagged as stale.

### Phase 3: Derived variables and skim (about 2 weeks)

- Port all classifier features (`variables.py`). This replaces about 267 KB of `branch.json` expressions. It is the biggest pile of small things; generate tests per feature.
- Explicit combinatorics: dijet chi2 pairing over `ak.combinations(ak4[:, :4], 2)`, the SFOS pair, and dR/dphi extrema over AK8 pairs.
- Skim processor: write Parquet per (sample, category, variation-group). Run the full production on Dask Gateway (`pixi_project` = this directory, `env=dict(os.environ)` for the proxy). Use `query_storage_usage` before writing.
- Sync test:
  - Feature-by-feature comparison to legacy trees.
  - Yields per category per sample.
  - The legacy XGBoost models applied to the *coffea* features must reproduce legacy scores. This is the strongest end-to-end check.

### Phase 4: ML (about 1-2 weeks; deferred)

- Port the K-fold training (including the Cramer-von Mises stage-2 objective) to read Parquet. Reuse the legacy training code where it is sound, and remove the 7 x `standardize_X` / 5 x weight-split duplicates.
- Run on GPU through the Ray cluster (`submit_ray_job`), or Slurm with `--gpus-per-node=1`.
- Validation:
  - First, with legacy models: nothing to validate beyond Phase 3.
  - Then retrain and compare ROC/AUC and decorrelation metrics. Scores differ only if the fold scheme changes.
- In-worker inference: models load once per worker, and K-fold dispatch happens on the event number.

### Phase 5: Histograms and systematics (about 2 weeks)

- Histogram processor:
  - SR box evaluation as masks over score columns.
  - Fill the msd1 x region x syst histogram, with data blinded.
  - Fill the theory/pileup weight variations and jet variations in the same pass.
- Sync test: per-SR yields and kappa values against the legacy `*_syst_yields.json`, including the n_eff < 10 fallback rule.

### Phase 6: SR optimisation, background estimate, datacards (about 2-3 weeks; deferred)

- SR box optimisation: port the beam search plus branch-and-bound. Its inputs become histograms or score arrays instead of re-inference.
- Mass-fit transfer factor (TF_MC fit, Bernstein residual). Mass fit only; no union/per-SR ABCD.
- Datacard writer in Python, replacing `combine.C`'s card writing; keep the Combine command structure.
- Sync test: identical datacards (modulo formatting), and Asimov significance and limits within 1% of legacy (reference: a-priori Z 0.925, r < 2.27 at 255.8 fb^-1).

### Phase 7: Cut-over

- Data/MC plots and the remaining study scripts.
- Freeze legacy at a tag. Document how to reproduce each legacy result with the new CLI.

## Cross-cutting

- **Tests:** unit tests for corrections and combinatorics on hand-built awkward arrays; sync tests on the frozen reference; one 1% "smoke production" on Dask Gateway in CI-like manual runs.
- **Reproducibility:** each output records the git hash, config hash and file-list hash.
- **Performance targets:** the skim processor needs at least legacy throughput per core.
  - The obvious gain is reading only needed columns.
  - The bigger gain is removing the 3.1 TB of intermediates and the per-systematic re-reads.
- **Out of scope** unless asked: new physics (SFs, a new JER), changing the categories, new taggers.

## Biggest risks

1. rho recovery (exact, deterministic) and statistical validation of the stochastic smearing.
2. The volume of derived features: about 160 per tree, defined in a custom expression language.
3. XRootD flakiness: 21 unreadable data files today. Use coffea's `skipbadfiles`, and keep a recorded list of the files actually processed, which feeds the lumi.
