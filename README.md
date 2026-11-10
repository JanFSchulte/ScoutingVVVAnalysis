# ScoutingVVVAnalysis: Scouting VVV analysis on coffea

This is a port of the CMS Run-3 scouting VVV search from the ScoutingVVVTools legacy chain to the
coffea 2026.9 stack (awkward, correctionlib, hist, XGBoost). The legacy chain is ScoutingVVVTools
(`CMSSW_16_1_0_pre4/src/ScoutingVVVTools`, git 421d4d7), made of `convert_branch.C`, its JSON
expression engine, the Python BDT and the `combine.C` scripts.

The legacy chain stays the reference: every ported stage is checked against legacy outputs, event
by event where possible. Intended differences are listed as decisions rather than absorbed into
tolerances.

- `specification.md`: what the analysis does (samples, objects, categories, corrections,
  classifier, statistics).
- `DEVELOPMENT_PLAN.md`: phases, status and the dated decisions.

## Installation

The environment is managed with [pixi](https://pixi.sh). Everything is pinned in `pixi.toml` and
`pixi.lock`: Python 3.13, coffea 2026.9, XRootD 5.9.2, XGBoost and the rest.

```bash
git clone https://github.com/JanFSchulte/ScoutingVVVAnalysis.git
cd ScoutingVVVAnalysis
pixi install           # creates .pixi/ in the checkout
pixi run test          # unit tests (pytest -q tests)
pixi run vvv --help    # command line
```

Requirements:

- linux-64 (the only platform in the lock file).
- For remote reads, a CMS VOMS proxy stored where the workers can read it:

  ```bash
  export X509_USER_PROXY=/work/users/$USER/x509up    # not /tmp: workers cannot see it
  voms-proxy-init -voms cms -valid 192:00
  ```

- For classifier training, a GPU. On the Purdue AF, training is submitted to the AF Ray cluster
  from a session without a GPU (see [Training](#training)). That uses the AF global environment
  (`/work/pixi/global`, which has Ray and a CUDA build of XGBoost) and not this project's
  environment.

### On the Purdue Analysis Facility

- **Location.** Check out under `/work/users/$USER/` or `/depot/cms/...` so that Dask Gateway
  workers can read the environment. pixi refuses project commands under `/home`.
- **Dask Gateway.** Create the cluster with `pixi_project=<checkout>` and the environment
  variables `X509_USER_PROXY=/work/users/$USER/x509up` and
  `X509_CERT_DIR=/cvmfs/cms.cern.ch/grid/etc/grid-security/certificates`. Without the CA
  directory every read from a non-Purdue site fails as "Operation expired".
- **Private checkouts.** If the checkout is in a directory that is not world-readable (e.g.
  `/depot/cms/private/...`), mirror it to `/work` with `scripts/sync_work_mirror.sh` and point
  `pixi_project` at the mirror. Do not re-sync the mirror while a production is running.

### Data shipped in the repository

- `data/samples.yaml`: the sample catalogue (names, DAS paths, cross sections, groups), imported
  from the legacy `sample.json`.
- `data/filelists/`: per-sample file lists, from the legacy conversion snapshots.
- `data/corrections/`, `data/pileup/`, `data/lumi/`: JEC/JER/JMS/JMR payloads, pileup profiles
  and the golden JSON / luminosity inputs.
- `metadata/<sample>.json`: sums of weights, theory-weight sums and pileup profiles, keyed by a
  file-list hash. `vvv metadata` regenerates them.

The commands `vvv import-samples` and `vvv import-filelists`, the `sync_*.py` scripts and the
"legacy" modes of the SR and QCD tools read the legacy checkout, expected at
`../CMSSW_16_1_0_pre4/src/ScoutingVVVTools`. The analysis itself does not need that checkout.

The training configs (`configs/training/*.json`) contain absolute skim and model paths of the
current production; regenerate them with `scripts/make_training_config.py` for other locations.

## Running

### Metadata and the analysis pass

```bash
pixi run vvv metadata www            # sums of weights, theory sums, pileup profile -> metadata/www.json
pixi run vvv run www \
    --model-dir boosted3=<k-fold training dir> \
    --signal-regions boosted3=<signal_region.csv> \
    --skim-dir /work/users/$USER/vvv/skims
```

`vvv run` makes one pass per sample. It applies the jet corrections (including rho recovery),
builds the five categories for the nominal jets and the eight jet variations, computes the
classifier inputs, weights and weight variations, and, for categories with a model, the K-fold
scores, signal-region assignment and mass-fit histograms:

- `mass[dataset, category, region, syst, mass]`: syst = nominal, jet variations, pileup up/down.
- `LHEScaleWeight`, `LHEPdfWeight`, `LHEPdfWeightAlphaS`, `PSWeight`
  `[dataset, category, region, member, mass]`: members normalised to their generator-level sums.
- `mass_smooth[dataset, category, factor, region, mass]`: nominal MC in each SR box loosened by
  `--smooth-factor` (default 2, the legacy `smooth_templates` value) and outside the other boxes,
  for the smoothed mass-fit templates.
- `cutflow`, `rho_failures`, `rho_consensus`, and the provenance `lumi_fb`, `signal_regions`
  (CSV path and SHA-256 per category) and `models`.

Region 0 is outside every SR box and region i the SR with bin_index i; every filled event passed the
category thresholds except the msoftdrop_1 window.

Options:

- **Luminosity.** MC is normalised to `--lumi` (default 255.8 fb^-1, the projection target).
- **Blinding.** Data in an SR box inside the 50-150 GeV mass window is dropped unless `--unblind`.
- **Dask.** `--gateway` runs on the user's Dask Gateway cluster.
- **Redirector.** The default `--redirector auto` reads `/store/user/jschulte/` from Purdue EOS
  and every other file through `root://cms-xrd-global.cern.ch/`, as the legacy conversion does.
  `--redirector /eos/purdue` reads through the session mount, for local tests only.
- **Test runs.** `--max-files N` limits a test run. `--maxchunks` alone still preprocesses every
  file of the sample.

### Skims

All of data_2024 skimmed in full would be about 600 GB, so there are two reduced data skims:

- **Fit-region skim (~107 GB).** `--skim-cut configs/training/<config>.json` keeps only the
  events that can reach a mass-fit histogram: each category's thresholds except the msoftdrop_1
  window, with msoftdrop_1 inside the mass axis (20-250 GeV). With `--model-dir`, the skim also
  stores the out-of-fold `proba_<class>` columns.
- **Plot skim (~15 GB).** `--every-nth-file 40` without the cut, for data/MC plots.

Both are staged on /work and merged onto /depot while the run is going (`scripts/drain_skims.py`).
`drain_skims.py --check` compares the rows with the cutflow and looks for duplicate events.
Changing the thresholds or the fit range requires a re-skim.

### ttbar control region and JMS/JMR fit

```bash
pixi run vvv ttbar-cr ttbar_semilep ... data_2024 --skim-dir /work/users/$USER/vvv/skims_ttbar_cr
pixi run python scripts/fit_jms_jmr.py configs/jms_jmr.json \
    --skim-dir /work/users/$USER/vvv/skims_ttbar_cr --output-dir <dir> --plots
```

`vvv ttbar-cr` is the port of the legacy `convert_ttbar_cr` conversion (tree `ttbar_cr`). It
applies the SingleMuon trigger, the golden JSON for data, and the analysis jet corrections (JEC,
rho recovery, JER, Type-1 MET) without JMS/JMR. The collections are:

- `ak8`: all AK8 jets, pT-sorted;
- `ak4`: analysis jet ID and deltaR(ak8) > 0.8;
- `mu_vtx`: muons with pT > 10 GeV, |eta| < 2.4, deltaR(ak8) > 0.8;
- `probe`: AK8 jets with |eta| < 2.4 and |dphi(mu_vtx[0])| > 1.5708.

The tree requires >= 1 of each of mu_vtx, ak4 and ak8. The output per sample is a pickle with the
cutflow, the pass/fail probe-mass histogram of the fit and the CR control distributions. These
are filled for the events that pass the fit selection of `configs/jms_jmr.json`, split by the probe
W tag. With `--skim-dir`, the whole `ttbar_cr` tree (legacy branch names, `weight`, pileup ratios)
is also written to Parquet, so a changed fit selection needs no new pass over the NanoAOD.
`--jet-variations` adds JES/JER up/down to the histograms. `--batch-files N` writes one
resumable pickle per batch of files.

`scripts/fit_jms_jmr.py` is the port of `jms_jmr_fit.py`. It fits a Gaussian plus exponential to the
W peak (50-120 GeV, 17 bins) of the pass region, for data and MC, and writes
`jms_jmr_results.json` in the format `corrections/jets.py` reads (JMS = mu_data/mu_MC, JMR =
sigma_data/sigma_MC). Its inputs are the skims, the pickles (`--hists`, fixed to the processor's
fit selection), or legacy `convert_ttbar_cr` trees (`--legacy-root`, with the legacy weights, for
closure). The fit never writes `data/corrections/`; adopting a new fit means copying the file there
by hand.

### Training

The classifiers are k-fold, mass-decorrelated multiclass BDTs, a port of the legacy
`selections/BDT/train.py`. They train on the AF Ray cluster, one GPU task per (category, fold).

```bash
# 1. Training config from the legacy production k-fold configs, with the legacy budgets
#    (--class-scale QCD=2 decorrelated worse, see DEVELOPMENT_PLAN.md)
pixi run python scripts/make_training_config.py configs/training/legacy_budget.json \
    --skim-dir /depot/cms/users/$USER/VVV/skims --output-dir /work/users/$USER/vvv/models/legacy_budget
# 2. Training skims: nominal only, each (sample, category) prescaled to keep 1.5x what the
#    training config draws from it
pixi run python scripts/make_skim_prescales.py configs/skim_prescales.json
pixi run vvv metadata <samples...> --gateway
pixi run vvv run <samples...> --gateway --skim-dir /work/users/$USER/vvv/skims \
    --skim-prescale configs/skim_prescales.json
# 2b. One file per chunk is ~200k files and ~100 GB, over the /work quota: merge them onto /depot
pixi run python scripts/compact_skims.py /work/users/$USER/vvv/skims /depot/cms/users/$USER/VVV/skims --remove-source
# 3. Submit (resubmitting the same config resumes; finished folds are skipped)
/work/pixi/global/.pixi/envs/default/bin/python scripts/submit_training.py configs/training/legacy_budget.json --workers 8
/work/pixi/global/.pixi/envs/default/bin/python scripts/submit_training.py --logs <job id>
```

Models land in the legacy layout: `<output_dir>/<category>/fold<r>/<category>_model.json`,
`branch.json`, `selection.json`, plus `training_log.json`. `vvv run --model-dir
<category>=<output_dir>` uses them directly. Events are assigned to folds by `event % 5`, for MC
and data alike.

Every fold ends with the legacy diagnostic plots (`src/vvv/ml/plots.py`): ROCs, score
distributions, importances, decorrelation diagnostics, loss curves and `plot_summary.json` (AUCs
and score/mass correlations). `scripts/plot_training.py` remakes them on CPU.

What follows legacy exactly:

- features, thresholds and preprocessing;
- the weight chain (sample normalisation, sign-preserving |w|, class balancing after cuts, cap
  at 20 x the class mean);
- the stage-1 learning-rate schedule and early stopping;
- the stage-2 smooth Cramer-von Mises penalty on all six class log-odds against msoftdrop_1,
  log(sphere_mass) and mass_fraction_1 for QCD, with its decor_scale calibration.

Deliberate differences:

- folds by event number instead of contiguous entry blocks;
- random instead of contiguous per-sample subsampling;
- GPU `hist` tree building (legacy XGBoost 1.7.5 ignored `device="cuda"` and ran on CPU);
- low-HT QCD (HT < 200 GeV) is not used: it reaches the selection only through pileup.

### Signal-region optimisation and QCD estimate

```bash
# SR boxes: beam search + branch-and-bound over the 6 class probabilities, Asimov Z with sigma_B
pixi run python scripts/optimise_signal_regions.py configs/signal_regions/boosted3.json \
    --skim-dir /depot/cms/users/$USER/VVV/skims_mc_fit --training-config configs/training/legacy_budget.json \
    --category boosted3 --deterministic --output-dir <dir>
# QCD mass-fit inputs: fail/SR mass templates, TF_MC fit, sideband transfer factors -> qcd_abcd_yields.root
pixi run python scripts/estimate_qcd.py <config> --output-dir <dir> [--plots]
```

Both tools also have a legacy mode, which reads the legacy BDT outputs or histograms and was used
for the closure tests below. `estimate_qcd.py` takes its events from the legacy score cache, the
fit-region skims (`--skim-dir`) or the `vvv run` histograms (`--hists`, see below).

### Systematic uncertainties and the combine.C inputs

One `vvv run` pass over all MC with every category's model and SR CSV gives both the mass-fit inputs
and the six `*_syst_yields.json` files the legacy `combine.C` reads (`theory_syst_json`,
`pileup_syst_json`, `jes_syst_json`, `jer_syst_json`, `jms_syst_json`, `jmr_syst_json`):

```bash
M=/work/users/$USER/vvv/models/baseline; S=/work/users/$USER/vvv/sr/baseline
CATS="boosted3 boosted2_resolved boosted2_unresolved boosted2_leptonic_1lep boosted2_leptonic_2lep"
ARGS=$(for c in $CATS; do echo --model-dir $c=$M --signal-regions $c=$S/$c/signal_region.csv; done)
H=/work/users/$USER/vvv/hists/baseline_syst
SAMPLES=$(python3 -c "import json; print(' '.join(s for v in json.load(open('configs/training/baseline.json'))['class_groups'].values() for s in v))")
# 1. histograms of all MC, one pickle per sample: a rerun skips the finished samples (one vvv run call
#    per sample; a single call over all samples would be one Runner without resume)
for s in $SAMPLES; do
  [ -f $H/$s.pkl ] || pixi run vvv run $s $ARGS --gateway --output $H/$s.pkl || echo "FAILED $s"
done
# 2. the six *_syst_yields.json (+ CSV) for combine.C (a directory: every pickle in it, added up)
pixi run python scripts/make_syst_yields.py $H \
    $(for c in $CATS; do echo --signal-regions $c=$S/$c/signal_region.csv; done) \
    --training-config configs/training/baseline.json --output-dir /work/users/$USER/vvv/syst/baseline
# 3. qcd_abcd_yields.root from the nominal histograms of the same pass
pixi run python scripts/estimate_qcd.py <legacy qcd_est.json> --hists $H \
    --training-config configs/training/baseline.json --category <cat> \
    --signal-region-csv $S/<cat>/signal_region.csv --output-dir /work/users/$USER/vvv/qcd_est/<run>/<cat>
# 4. combine.C with every nuisance: a fitcheck spec whose variant has root_files {cat: qcd_abcd_yields.root}
#    and syst_dir = the step-2 output, setups {"all": []} (an empty enabled_nuisances list = all)
pixi run python scripts/fitcheck_massfit.py <spec.json> --out <dir> --steps build,combine,summary
```

In hist mode, `vvv run` skips unreadable files (`skipbadfiles`) while the normalisation uses the sums
of all files, so check the log for skipped files.

`make_syst_yields.py` (`vvv.stats.systematics`) is the port of the legacy `theory_syst.py`,
`pileup_syst.py` and `jes/jer/jms/jmr_syst.py`. It writes the legacy JSON layout
`{sample: {category: entry}}`. An entry has the inclusive ratios and sums at the top level,
`regions[<bin_index>]`, `sr_csv_hash` and `min_region_events`, and, with `--mass-fit 20,250,10`,
also per SR `mass_range` and `mass_bins` and `mass_fit`.

The regions:

- **inclusive:** the post-threshold selection, all regions inside the msoftdrop_1 window;
- **SR:** the SR box inside the window;
- **mass_range / mass_bins:** the SR box over the fit range and per fit bin.

What every entry holds:

- **Ratios.** Jet variations: `<s>_up/down` = Y(variation, its own selection, scores and mass) / Y(nominal)
  in the same region. Pileup: `pu_up/down`. Theory:
  - PDF: Hessian quadrature over the 100 members, with alpha_s (Y_up - Y_down)/2 in quadrature
    (`hessian_symmetric`; `mc_replica` optional);
  - scale: the 7-point envelope without the corners 2 and 6;
  - PS: ISR [2]/[0], FSR [3]/[1].
- **Sums.** The yields per fb^-1 and their sums of squared weights (`yield`, `sumw2`, `yield_<var>`,
  `sumw2_<var>`, with the 101/9/4/2 theory members). `combine.C` adds these up per process.
- **Fallback.** A region whose nominal sum or any varied sum the nuisance uses has
  n_eff = (sum w)^2 / sum w^2 < 10 takes the ratios of the next coarser region (flag
  `fallback_to_inclusive` / `fallback_to_mass_range`). `combine.C` applies the same rule again to the
  merged process sums.

combine.C turns these into lnN rows, one per nuisance and theory_scale per process. Its
mass-fit cards apply the SR-window kappa to every mass bin of the SR. With `--mass-fit` (bins equal
to the qcd_est massfit bins) they use per-mass-bin kappas instead. The legacy production used the
SR-window kappas. `scripts/fitcheck_massfit.py` drives combine.C; a variant with `syst_dir` uses the six files.

Deliberate differences from legacy: `n_events` is round(n_eff) instead of the raw MC count, and
`nom_sum` is the lumi-weighted sum. combine.C reads neither, because the histograms keep only sum w and
sum w^2. The msoftdrop_1 window is whole 10 GeV bins: an event exactly at 50.0 or 150.0 GeV is
counted in the bin whose lower edge it sits on.

`estimate_qcd.py --hists` has the same approximations as `MassfitInputs.from_hist`. `n_entries`
holds effective counts, `mean_event_weight` holds the effective weight sum w^2 / sum w instead of
the median |w|, and there are no ABCD totals. The mass-fit cards use none of these. Everything else
(yields, sumw2, smoothed templates, TF_MC inputs, metadata/signal_regions as a TTree) is the same
as `--skim-dir` on the same events.

## Status

### Implemented and validated against legacy

| Stage | Code | Validation |
|---|---|---|
| Schema, samples, file lists, metadata | `schema.py`, `samples.py`, `processors/metadata.py` | event counts, genweight means, pileup weights identical to legacy (pileup to 5e-11) |
| Objects and the 5 categories (boosted3, boosted2 resolved/unresolved, leptonic 1lep/2lep), nominal + 8 jet variations | `objects.py`, `selection.py` | 100% event-by-event agreement with legacy batch `www_0` (76,712 events) in all 9 configurations, and with 2 files of 2024G data |
| Jet corrections: JEC, rho recovery, JER, JMS/JMR, Type-1 MET | `corrections/jets.py` | AK8 pT/msd, MET and ISR score bit-identical (float32), incl. JER-smeared MC |
| Pileup and theory weights | `corrections/pileup.py`, `weights.py` | per-event pileup weight 2e-7, genWeight exact |
| ~110 classifier/threshold/decorrelation variables | `variables.py` | within 1e-5 x max(\|ref\|, 1), most bit-identical |
| K-fold inference | `ml/inference.py` | legacy boosters on coffea features reproduce legacy scores to 1e-7 |
| Analysis processor: skims, scores, SR assignment, mass histograms with jet/pileup/theory variations, blinding | `processors/analysis.py`, `cli.py` | plumbing-tested; per-SR yields not yet synced (see to-do) |
| K-fold mass-decorrelated BDT training on Ray GPUs, diagnostics | `ml/train.py`, `ml/ray_driver.py`, `ml/plots.py` | AUCs match legacy; QCD score-mass correlation +0.04 vs legacy +0.02 (see to-do) |
| SR optimisation | `sr/optimise.py`, `scripts/optimise_signal_regions.py` | legacy mode bit-identical to legacy on all production points; deterministic tie-break mode for new SRs |
| QCD mass-fit inputs (TF_MC fit, templates, transfer factors) | `background/qcd_massfit.py`, `scripts/estimate_qcd.py` | closure with legacy to 2e-12 on all 5 production points |
| ttbar CR (probe selection, CR branches, histograms, skims) and JMS/JMR tag-and-probe fit | `processors/ttbar_cr.py`, `corrections/jms_fit.py`, `scripts/fit_jms_jmr.py`, `scripts/sync_ttbar_cr.py` | 100% event-by-event selection agreement with legacy `convert_branch` on 2 ttbar_semilep files (19,521 of 132,677 preselected events) and 1 data_2024 file (400 of 1,118); all 128 (MC) / 100 (data) branches within 1e-5, 115 / 87 bit-identical; the fit reproduces the legacy results JSON bit for bit on the same trees |
| Data skims (fit region and 1/40 plot skim), staged drain to /depot | `scripts/drain_skims.py` | row counts against cutflow, duplicate check |
| Systematic inputs of combine.C (theory, pileup, JES/JER/JMS/JMR `*_syst_yields.json`) from the histograms | `stats/systematics.py`, `scripts/make_syst_yields.py`, `scripts/sync_syst_yields.py` | every field (n_events aside) equal to the legacy producers' own entry functions run on the event-level skims of the same pass (2 files each of www, ttbar_semilep; 46 entries per systematic): jets and pileup to 1e-15, theory to 3e-12; read by combine.C with all nuisances enabled |
| QCD mass-fit inputs from the histograms (`estimate_qcd.py --hists`) | `background/hists.py` | identical to `--skim-dir` on the same events (all yields, sumw2, smoothed templates to 1e-12), all 5 categories; n_entries / mean_event_weight approximated |

### Current production (2026-11-06)

- data_2024 fit-region and plot skims: running on Dask Gateway.
- Legacy-budget BDTs: resolved and unresolved done; boosted3, 1lep and 2lep training.
- Next: MC fit-region skims with out-of-fold scores from these models.

## To do

Analysis chain:

- [ ] MC fit-region skims with the legacy-budget models (`vvv run --skim-cut ... --model-dir ...`
      for every sample).
- [ ] SR optimisation on the coffea skims for all five categories. Unresolved needs ~68 GB in
      cells mode, so it needs a memory reduction or a large worker.
- [ ] Recompute the QCD transfer factors on the new SRs with `estimate_qcd.py`, then re-optimise
      with them (the SRs are currently optimised with the legacy TFs).
- [ ] Datacard writer in Python, replacing `combine.C` (`writeMassfitCard`). Keep the Combine
      command structure. Sync target: identical cards and Asimov Z / limits within 1% of legacy
      (a-priori Z 0.925, r < 2.27 at 255.8 fb^-1).
- [x] Adapter from the histograms to the legacy `combine.C` inputs (`scripts/make_syst_yields.py`:
      the six `*_syst_yields.json`, n_eff < 10 fallback rule; `estimate_qcd.py --hists`).
- [ ] Full MC histogram production (`vvv run` with all five models and SR CSVs), then the syst
      JSONs, `estimate_qcd.py --hists` and the combine.C fit with all nuisances.
- [ ] Phase-5 per-SR yield sync against a legacy production with the same SRs (the legacy
      theory/pileup outputs are for older SRs; JES/JER/JMS/JMR were never produced in legacy).
- [ ] Decide whether the cards use per-mass-bin kappas (`make_syst_yields.py --mass-fit`) or, as the
      legacy production, the SR-window kappa on every mass bin.

Physics and validation:

- [x] ttbar control-region processor and JMS/JMR fit (`vvv ttbar-cr`, `scripts/fit_jms_jmr.py`).
- [ ] Full-statistics CR production and JMS/JMR refit after the JEC (the legacy values are stale),
      then replace `data/corrections/jms_jmr_results.json`.
- [ ] Data/MC plots from the plot skim.
- [ ] Remaining decorrelation gap to legacy (QCD p(VVV)-msd1 correlation ~+0.04 vs ~+0.02). The
      stage-2 penalty is still falling at the 1496-round cap.
- [ ] Switch from reconstructed rho to the stored branch once the next ScoutingNano production
      has it.

Infrastructure:

- [ ] Remove the absolute user paths from `configs/training/*.json` (skim and model directories,
      legacy references).
- [ ] CI for the unit tests; a 1% smoke production on Dask Gateway.
- [ ] Cut-over: freeze legacy at a tag and document how each legacy result is reproduced with
      the `vvv` CLI.

## Legacy behaviours reproduced on purpose

- Leptons are massless. Kinematic cuts use the pT/eta of a `TLorentzVector` rebuilt from (pT,
  eta, phi, m), which moves stored |eta| = 2.5 values to either side of the cut.
- Track-less electrons carry charge 2147483647. Legacy casts it through float to int (INT_MIN),
  so they pair "opposite-sign" with any charged electron in the 2-lepton SFOS search.
  `ObjectConfig.legacy_electron_charge = False` switches to physical charges (95 -> 88 2-lepton
  WWW events on one file).
- Inputs are truncated to 30 AK4 / 20 AK8 / 30 leptons, and AK4 jets with raw pT < 19.9 GeV are
  dropped, before any correction.
- JMR < 1 gives no smearing, so the JMR variations equal nominal with the current
  `jms_jmr_results.json`.

### ttbar CR: deliberate differences from legacy

- Data is `data_2024` (2024G-I, the SR data sample) instead of the legacy `2024G_UParT_v6`.
- The diboson samples are those of the analysis VV class (the Powheg samples and the aMC@NLO FXFX
  WZto4Q/ZZto4Q), instead of the legacy inclusive `ww`, `wz`, `zz`, which had no events
  (raw_entries = 0) and were skipped by the legacy fit.
- MC weights in the fit are the analysis weights (signed genWeight, w_PU / <w_PU>, Runs-tree
  sums). The legacy fit used weight_pu x lumi x xsec / raw_entries, without genWeight, so
  negative-weight events (genWeight = -337 in ttbar) counted as positive.
- The legacy theory-weight and gen-level index branches are not written. Gen pdgIds of the matched
  particles and topWCategory/partonFlavour are.

## Layout

```
src/vvv/schema.py              NanoEvents schema for the ScoutingNano collections
src/vvv/corrections/jets.py    JEC, rho recovery, JER, JMS/JMR, Type-1 MET
src/vvv/corrections/jms_fit.py JMS/JMR tag-and-probe fit of the W peak
src/vvv/corrections/pileup.py  pileup weights (correctionlib export, legacy CSV import)
src/vvv/objects.py             objects, ISR score, cleaning, categories
src/vvv/variables.py           per-category classifier and selection variables
src/vvv/selection.py           inputs -> corrected objects -> categories, per jet configuration
src/vvv/weights.py             normalisation, pileup and theory weight variations
src/vvv/ml/                    K-fold inference, training (Ray driver), diagnostic plots
src/vvv/sr/                    SR box optimisation (inputs from skims or legacy BDT outputs)
src/vvv/background/            QCD mass-fit transfer factors and templates (legacy cache, skims, histograms)
src/vvv/stats/                 systematic yield ratios for combine.C (*_syst_yields.json)
src/vvv/processors/            metadata, analysis and ttbar CR processors
src/vvv/cli.py                 `vvv` command line
configs/                       training, skim prescale, SR and JMS/JMR fit configs
scripts/                       production, training, SR/QCD tools, sync_*.py legacy comparisons
data/, metadata/               samples, file lists, corrections, derived sample metadata
tests/                         pytest unit tests
```
