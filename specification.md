# Scouting VVV (2024) - coffea re-implementation

Search for triboson (WWW, WWZ, WZZ, ZZZ) and VH(H->WW) production in fully and semi-hadronic final states with 2024 PF-scouting data, in five exclusive AK8-based categories.
This document specifies *what* the coffea port must reproduce, using ScoutingVVVTools at git HEAD 421d4d7 (`../CMSSW_16_1_0_pre4/src/ScoutingVVVTools`, "legacy" below) as the reference. The step-by-step build plan is in `DEVELOPMENT_PLAN.md`.

## Scope of the first port (confirmed)

- Ported now: conversion (metadata + skim processor, corrections, objects, categories, variables), systematics, histogram processor.
- Kept from legacy for now: XGBoost training (`selections/BDT/train.py`), SR optimisation, `combine.C`. The new code reads legacy models and SR boxes and writes histograms in the format the legacy statistics stage needs.
- Later: training, SR optimisation, datacards, plotting.
- Location: this directory on `/depot` (no Jupyter kernel needed).

## Data Samples

The source of truth for dataset names, cross sections and sample groups is the legacy `src/sample.json`: 2 data entries and 98 MC entries. Dataset names are not copied here, to avoid drift. The port imports that file once into an immutable `samples.yaml`. Metadata the legacy code writes back into it (`raw_entries`, `genweight_mean`, `weight_pu*_mean`) is not imported: it gets recomputed in a separate metadata step (see Workflow).

| Dataset (DAS, `instance=prod/phys03`)                                    | Tier                       | Era / campaign           | Role(s)                                            |
| ------------------------------------------------------------------------ | -------------------------- | ------------------------ | -------------------------------------------------- |
| `/ScoutingPFRun3/yiyangz-ScoutingNano_Data_2024{G,H,I}_part*_v6/v7/USER` (verbatim list in sample.json `data_2024`) | USER (ScoutingNano)        | 2024G-I                  | Data (blinded in the SR)                           |
| `2024G_UParT_v6` entry in sample.json                                    | USER (ScoutingNano)        | 2024G                    | Data/MC plots, ttbar CR                            |
| `/.../jschulte-ScoutingNano_MC_*_v5/v6/v7.../USER`: www, wwz, wzz, zzz (aMC@NLO) | USER (ScoutingNano, MC)    | Run3Winter24-like        | Signal (VVV class)                                 |
| VH(H->WW) powhegMINLO, about 28 samples                                  | USER (ScoutingNano, MC)    | same                     | Signal (VH class)                                  |
| QCD HT-binned, top (TT, tW), VV (powheg plus aMC WZ/ZZto4Q), W/Z+jets LO MLM | USER (ScoutingNano, MC)    | same                     | Backgrounds (Top, VV, VJets, QCD)                  |

- Luminosity / certification: `Cert_Collisions2024_378981_386951_Golden.json`.
  - Processed lumi is 54.2056 fb^-1 (brilcalc, normtag_PHYSICS), used for data/MC comparisons.
  - SR yields, datacards and limits are normalised to the 255.8 fb^-1 projection (confirmed).
- Triggers: `DST_PFScouting_JetHT` for the analysis. `DST_PFScouting_SingleMuon` for the ttbar CR.
- MC normalisation: `w = L x xsec x genWeight / sum(Runs/genEventSumw) x w_PU / <w_PU>`.
  - xsec comes from sample.json (XSDB, fb). VV is scaled to NNLO (MATRIX, `combine/xsec_unc.json`). V+jets gets per-sample NLO/LO k-factors of 1.41-1.62.
  - Behaviour change, deliberate: the port uses `sum genEventSumw` from the Runs tree directly instead of `raw_entries x genweight_mean`. These are equivalent when all files are read. When files are missing, compute the sum from the files actually processed.
- Files are read over XRootD through `root://xcache.cms.rcac.purdue.edu/`.
  - Files under `/store/user/jschulte` can also be read from `root://eos.cms.rcac.purdue.edu/`.
  - (question: confirm XCache serves the phys03 USER files. Fallback: the Purdue EOS door / global redirector.)

## Histograms

Every histogram is a single `hist.Hist` with categorical axes `dataset`, `category` (the 5 trees plus CRs), `region` and `syst`. Storage is `Weight()`.

1. AK8 leading soft-drop mass (mass-fit / ABCD axis)
   - x: `msoftdrop_1` of the leading selected AK8 after JEC, JER, JMS and JMR. Label `$m_{SD}^{j1}$ [GeV]`. Binning as in the legacy massfit configs, 50-150. One entry per event.
   - region axis: each SR box, "fail" (outside all boxes), and sidebands.
   - Weights: the full MC weight; data unweighted. Blinded bins are masked downstream, not dropped at fill time.
2. Classifier scores
   - x: the six softmax outputs (VVV, VH, Top, VV, VJets, QCD), one 1D histogram each, in [0, 1]. One entry per event.
3. Data/MC control distributions
   - The input features listed in legacy `selections/BDT/branch.json` (58-67 per category), plus jet/lepton kinematics.
   - Same weights as above. Blinded by default in the SR.
4. Cutflow
   - Events surviving each step, per category. Raw and weighted counts.
5. Pileup
   - `Pileup_nTrueInt` (MC). This feeds the pileup-weight derivation.

## Analysis Steps

1. Event preselection
   - Golden JSON (data), `DST_PFScouting_JetHT`, `nScoutingFatPFJetRecluster >= 2`. Legacy applies this before the JEC; keep that.
2. Jet corrections
   - Applied before the object selection; see Corrections.
3. Objects
   - AK8 (`ScoutingFatPFJetRecluster`): pT > 175, |eta| < 2.5, sorted by pT.
   - AK4 (`ScoutingPFJetRecluster2`): raw pT > 19.9. After corrections: pT > 30, |eta| < 2.5, nConstituents > 1, neHEF < 0.99, neEmEF < 0.90, chHadMultiplicity > 0, muEF < 0.80.
   - Electron veto: pT > 10, |eta| < 2.5, the legacy cut-based EB/EE ID, (trk + ecal + hcal iso)/pT < 0.20.
   - Muon veto (`ScoutingMuonVtx`): pT > 10, |eta| < 2.4, the legacy ID (chi2 < 3, > 5 tracker layers, >= 1 pixel hit, >= 1 SA hit, >= 2 matched stations, type in {2, 6, 10, 14}), trkIso/pT < 0.15.
4. Per-AK8 ISR score
   - `-0.0097*min(|msd-80.4|,|msd-91.2|) + 2.5884*SumX/(SumX+QCD) - 2.2984*maxratio + 3.3657`.
   - SumX = Xud + Xcs + Xqq + Xbb + Xcc + Xss.
   - maxratio = the largest pT(AK4, no cuts)/pT(AK8) over AK4 within dR < 0.8.
5. Category-dependent cleaning
   - c3: AK4 and leptons cleaned (dR > 0.8) against the 3 leading AK8.
   - c2: cleaned against the 2 highest-ISR-score AK8 among the leading 3.
6. Categories (exclusive; the port must reproduce the legacy assignment event by event). minScore3 is the minimum ISR score over the 3 leading AK8.
   - boosted3: >= 3 AK8, minScore3 > 0.469, 0 c3 leptons.
   - boosted2_resolved / boosted2_unresolved: exactly 2 AK8, or >= 3 AK8 with minScore3 <= 0.469; 0 c2 leptons; >= 2 resolved c2 AK4 for resolved, < 2 for unresolved.
   - boosted2_leptonic_1lep: >= 2 AK8, exactly 1 c2 lepton, and not boosted3-like.
   - boosted2_leptonic_2lep: >= 2 AK8, an SFOS pair among the first 3 c2 leptons (closest to mZ, no window), and not boosted3-like.
7. Analysis cuts (legacy "BDT thresholds")
   - msd1 in [50, 150], msd2/msd3 in [40, 150], pT_i > 175, |eta_i| < 2.4.
   - b-veto: max over AK4 (dR0p4-cleaned) of (b + bb + lepb)/all < 0.5.
8. Derived variables
   - Everything that feeds the classifier: ParT ratios, multijet masses and pTs, mass fractions, pt_balance, sphere_mass, dR/|dphi| extrema, HT, MET/HT, event shapes (sphericity, aplanarity, planarity), b-tag counts, LeptonicW mT/pT, DileptonZ.
   - Resolved dijet: over the first 4 resolved AK4, choose the pair minimising `chi2 = (min(|mjj-80.4|,|mjj-91.2|)/12.7)^2 + (dRjj/0.60)^2`. This is an explicit pair loop (`ak.combinations`, n = 2).
   - MC truth labels (nGenHadronicWZ, ...) for studies only.
9. Classifier
   - XGBoost 6-class softmax per category, applied with K-fold (K = 5): each event is scored by the model that did not train on it; data by `event % K`.
   - Stage-2 mass-decorrelation penalty (Cramer-von Mises, lambda = 2) against msd1, sphere_mass, mass_fraction_1.
   - Fold assignment by event number for MC and data alike (confirmed). This replaces the legacy contiguous blocks and requires one retrain.
10. Signal regions
    - N = 3-4 non-overlapping boxes in score space, found by maximising sum(Z_Asimov^2) with min_bkg_weight 5.

## Corrections and Systematics

- JEC: `Winter24HLT_V1_MC_L1L2L3Res_{AK4,AK8}PFHLT` (+ L1FastJet) from the legacy `selections/convert/jec/Run3Winter24/*_jerc.json.gz`, through correctionlib. Data use the same payload; no residuals.
  - AK4 raw = pt x (1 - rawFactor).
  - AK8 is corrected from raw.
- rho is not stored in the current production. Reconstruct it per event by inverting the AK4 JEC (legacy regula falsi, 1 GeV consistency), vectorised over events.
  - A future ScoutingNano production will store rho. Keep the reconstruction behind a switch (`rho_source: reconstruct | branch`) so the stored branch can be used directly, with the reconstruction kept as a cross-check.
- JER (MC): `Summer23BPixPrompt23_RunD_JRV1_MC_{PtResolution,ScaleFactor}_AK4PFPuppi` (offline proxy, 2024_Summer24 POG) plus `JERSmear`, GenPt = -1. Applied to AK4 and AK8.
  - Smearing must be statistically equivalent to legacy, not bit-identical: compare the distributions of corrected pT/msd and per-category yields within statistical uncertainty. Use a deterministic per-jet seed built from (run, event, jet index) so results are reproducible.
- JES: flat +-10% on pT, mass and msd, applied after JER.
- JMS/JMR: from `jms_jmr_results.json`, fitted in the ttbar CR (Gaussian + exponential W peak). The smearing only needs to be statistically equivalent, so the legacy splitMix64 hash is not ported. The values must be refitted after the JEC with the ported ttbar CR.
- Type-1 MET: built from AK4 with corrected muon-subtracted pT > 15 and EMF < 0.9. Propagate all jet variations to MET.
- Pileup: data/MC nTrueInt ratio with 69.2 / 66.0 / 72.4 mb data profiles (2024GHI), stored as a correctionlib JSON instead of CSV.
- Theory (weights only, acceptance-only after normalising to generator sums from `Runs`):
  - PDF: NNPDF3.1 Hessian, 100 members.
  - mu_R/mu_F: 7-point envelope (drop indices 2 and 6).
  - PS: ISR (2, 0) and FSR (3, 1).
- Systematic list:
  - Weight-based: pileup, theory_pdf, theory_scale, theory_ps_isr, theory_ps_fsr.
  - Shape (the selection reruns on shifted objects): jes, jer, jms, jmr.
  - lnN: lumi 1.6%, trigger 2%, xsec per process.
  - Statistical: autoMCStats 10.
  - Background estimate: ABCD/mass-fit parameters.
- Not applied, same as legacy: tagging, lepton and trigger SFs.

## Workflow

1. Metadata (once per sample)
   - Use coffea preprocess on the file list.
   - Sum `genEventSumw`, `LHEScaleSumw`, `LHEPdfSumw` and `PSSumw` from `Runs`.
   - Fill the MC nTrueInt histogram and compute <w_PU>.
   - Write `metadata/<sample>.json`. Nothing is written back into the sample config.
2. Skim processor (nominal + jet variations, one pass)
   - Source filter: preselection (step 1).
   - Branches read: those listed in the legacy `selections/convert/branch.json` "input", loaded lazily by NanoEvents.
     - MC-only: `genWeight`, `Pileup_nTrueInt`, `LHE*Weight`, `PSWeight`, `GenPart_*`, `*_genPartIdx`, `*_partonFlavour`.
   - Output: per-category Parquet with the classifier features, msd1, weights and the needed weight-variation columns.
     - Jet-variation features go in separate column groups.
     - This replaces the 5 x 9 ROOT trees.
3. Training
   - K-fold XGBoost on the Parquet skims (GPU: Ray cluster or Slurm, `--gpus-per-node=1`).
4. Histogram processor
   - Applies the K-fold models in the workers, evaluates the SR boxes, and fills the histograms above for nominal plus every variation, in one pass.
   - This replaces modes 3-5 and 8-14, which each re-read the full dataset.
5. Statistics inputs
   - hist -> ROOT shapes (uproot), plus datacard writer.

## Statistical Analysis

- Binned likelihood in Combine (v11.1.0).
- Main approach: mass-fit QCD estimate.
  - Per msd1 bin, free fail-region QCD (rateParam).
  - pass = fail x n_sr x TF_MC (exp-polynomial) x Bernstein residual.
- Outputs: Asimov significance (`-t -1`), AsymptoticLimits, per channel and combined.
- Only the mass-fit method is implemented. The union and per-SR ABCD variants are not ported (confirmed).

## Tools

- coffea 2026.9 (global env)
  - NanoEvents with `ScoutingNanoAODSchema`, subclassed if the `*Recluster*` collections are not mapped.
  - Processor executed on Dask Gateway, or the local Dask executor for tests.
- awkward, vector: all object and combinatoric logic. No per-event Python loops.
- correctionlib: JEC, JER, JERSmear, pileup, and JMS/JMR as a custom correction.
- hist, mplhep: histograms and CMS-style plots.
- xgboost: training (GPU) and in-worker inference. Optionally ONNX later.
- uproot: ROOT shape output for Combine.
- Combine: fits and limits. Initially driven by porting the card-writing part of `combine.C` to Python.
- pytest: sync tests against legacy outputs on `scouting_nano_MC_1.root`.
