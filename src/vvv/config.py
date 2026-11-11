"""Analysis constants.

Values follow the legacy ScoutingVVVTools configuration at git 421d4d7:
selections/convert/{config,selection,branch}.json and the constants at the top
of selections/convert/convert_branch.C, except the deliberate differences listed in
README.md (trigger paths per category, the JER payloads and hybrid smearing, the official
scouting jet ID, the jet veto map and the lepton subtraction from jets).
"""

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CORRECTIONS_DIR = DATA_DIR / "corrections"
# Copies of dated releases of the CAT JME payloads (/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/
# <campaign>/<date>/), kept with their changes.md: the CVMFS "latest" links move with new releases.
CAT_JME_DIR = CORRECTIONS_DIR / "cat" / "JME"
JME_2024 = CAT_JME_DIR / "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15" / "2026-07-16"
JME_2024_SCOUTING = CAT_JME_DIR / "Run3-24Scouting-Summer24-ScoutingNanoAODv15" / "2026-09-15"

W_MASS = 80.4
Z_MASS = 91.2


@dataclass(frozen=True)
class InputConfig:
    # Legacy reads at most this many objects per collection (branch.json max_size).
    max_ak4: int = 30
    max_ak8: int = 20
    max_electrons: int = 30
    max_muons: int = 30
    # AK4 jets with raw pT below this are dropped before any correction.
    ak4_min_raw_pt: float = 19.9


@dataclass(frozen=True)
class JetCorrectionConfig:
    jec_ak4_file: Path = CORRECTIONS_DIR / "jetHLT_jerc.json.gz"
    jec_ak8_file: Path = CORRECTIONS_DIR / "fatJetHLT_jerc.json.gz"
    jec_ak4: str = "Winter24HLT_V1_MC_L1L2L3Res_AK4PFHLT"
    jec_ak8: str = "Winter24HLT_V1_MC_L1L2L3Res_AK8PFHLT"
    jec_ak4_l1: str = "Winter24HLT_V1_MC_L1FastJet_AK4PFHLT"
    jec_ak8_l1: str = "Winter24HLT_V1_MC_L1FastJet_AK8PFHLT"
    # JER: no scouting JER exists; the offline PUPPI JER of the Summer24 MC campaign, AK4 for AK4
    # and AK8 for AK8 (ScaleFactor with a fractional SFUncertainty: SF x (1 +- unc)).
    jer_ak4_file: Path = JME_2024 / "jet_jerc.json.gz"
    jer_ak8_file: Path = JME_2024 / "fatJet_jerc.json.gz"
    jer_tag: str = "Summer24Prompt24_JRV2_MC"
    jer_ak4_collection: str = "AK4PFPuppi"
    jer_ak8_collection: str = "AK8PFPuppi"
    jer_smear_file: Path = CAT_JME_DIR / "JER-Smearing" / "2026-09-18" / "jer_smear.json.gz"
    # Hybrid smearing: a generator jet matches if dR < R/2 and |pT - pT_gen| < 3 sigma_JER pT.
    ak4_radius: float = 0.4
    ak8_radius: float = 0.8
    jer_match_max_sigma: float = 3.0
    # The |eta| range of the JME payloads (JER, jet veto map): a lepton-subtraction remainder
    # pointing beyond it is no physical jet and is removed.
    jme_max_abs_eta: float = 5.191
    jes_shift: float = 0.1
    jms_jmr_file: Path = CORRECTIONS_DIR / "jms_jmr_results.json"
    apply_jms_jmr: bool = True
    type1_min_pt: float = 15.0
    type1_max_em_fraction: float = 0.9
    # "reconstruct" inverts the AK4 production JEC; "branch" reads rho_branch.
    rho_source: str = "reconstruct"
    rho_branch: str = "ScoutingRho_fixedGridRhoFastjetAll"


@dataclass(frozen=True)
class RhoRecoveryConfig:
    rho_max: float = 70.0  # L1FastJet clamps rho to [0, 70]
    tolerance: float = 1.0  # GeV, allowed spread between jets
    bracket_margin: float = 0.5
    min_factor: float = 1e-3  # jets at the L1FastJet floor carry no rho
    min_sensitivity: float = 1e-4  # min relative JEC change over the rho range
    solver_iterations: int = 60
    bin_edge_probe: float = 1e-3
    bin_probe_rho: float = 30.0


@dataclass(frozen=True)
class ObjectConfig:
    ak8_min_pt: float = 175.0
    ak8_max_abs_eta: float = 2.5
    ak4_min_pt: float = 30.0
    ak4_max_abs_eta: float = 2.5
    # Official 2024 scouting jet ID (JMAR; CAT Run3-24Scouting payload).
    jet_id_file: Path = JME_2024_SCOUTING / "jetid.json.gz"
    jet_id: str = "ScoutingPFJetRecluster_TightLeptonVeto"
    clean_dr: float = 0.8
    clean_dr_loose: float = 0.4
    ele_min_pt: float = 10.0
    ele_max_abs_eta: float = 2.5
    ele_max_rel_iso: float = 0.20
    mu_min_pt: float = 10.0
    mu_max_abs_eta: float = 2.4
    mu_max_rel_iso: float = 0.15
    # ISR-vs-W score of an AK8 jet (selection.json, all categories).
    isr_mass_coef: float = -0.0097
    isr_tag_coef: float = 2.5884
    isr_ratio_coef: float = -2.2984
    isr_offset: float = 3.3657
    isr_ratio_dr: float = 0.8
    isr_ratio_default: float = 1e6
    boosted3_min_score: float = 0.469
    sfos_max_leptons: int = 3
    # Legacy reads the electron charge as a float and casts it to int, so the
    # no-track sentinel (2147483647) becomes INT_MIN and the int product of two
    # charges overflows. True reproduces that pairing; False uses the charge
    # sign and skips electrons without a track.
    legacy_electron_charge: bool = True


# Scouting paths of the data production filter (ScoutingNanoProduction, data config: a
# TriggerResultsFilter on DST_PFScouting_JetHT_v* OR DST_PFScouting_SingleMuon_v*, then >= 1 AK8
# jet). MC is produced without a trigger filter. A category may therefore require only these
# paths: an event that passes only another path is in the MC but can never be in the data.
DATA_PRODUCTION_TRIGGERS = ("DST_PFScouting_JetHT", "DST_PFScouting_SingleMuon")

JETHT = ("DST_PFScouting_JetHT",)
JETHT_OR_SINGLEMUON = ("DST_PFScouting_JetHT", "DST_PFScouting_SingleMuon")


def _category_triggers():
    """Trigger paths per category (an event passes if any of them fired).

    The lepton categories take JetHT or the lepton paths the data production keeps. Their
    target is JetHT or the single-lepton paths (1lep: SingleMuon and SinglePhotonEB, the single-EG
    scouting path) and JetHT or the dilepton paths (2lep: DoubleMuon, DoubleEG); the paths missing
    here need a production that keeps their events (DATA_PRODUCTION_TRIGGERS).
    """
    return {
        "boosted3": JETHT,
        "boosted2_resolved": JETHT,
        "boosted2_unresolved": JETHT,
        "boosted2_leptonic_1lep": JETHT_OR_SINGLEMUON,
        "boosted2_leptonic_2lep": JETHT_OR_SINGLEMUON,
    }


@dataclass(frozen=True)
class AnalysisConfig:
    inputs: InputConfig = field(default_factory=InputConfig)
    jets: JetCorrectionConfig = field(default_factory=JetCorrectionConfig)
    rho: RhoRecoveryConfig = field(default_factory=RhoRecoveryConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    # {category: trigger paths}; NanoEvents reads DST_PFScouting_JetHT as events.DST.PFScouting_JetHT.
    triggers: dict = field(default_factory=_category_triggers)
    lumi_mask: Path = DATA_DIR / "lumi" / "Cert_Collisions2024_378981_386951_Golden.json"
    # Jet veto map (JERC, Run 3): the event is rejected if any AK4 jet with corrected pT above
    # jet_veto_min_pt, passing the jet ID and with (chEmEF + neEmEF) below jet_veto_max_em_fraction,
    # lies in a vetoed (eta, phi) bin. Data and MC alike.
    jet_veto_file: Path = JME_2024 / "jetvetomaps.json.gz"
    jet_veto_map: str = "Summer24Prompt24_RunBCDEFGHI_V1"
    jet_veto_type: str = "jetvetomap"
    jet_veto_min_pt: float = 15.0
    jet_veto_max_em_fraction: float = 0.9
    jet_variations: tuple[str, ...] = (
        "jes_up",
        "jes_down",
        "jer_up",
        "jer_down",
        "jms_up",
        "jms_down",
        "jmr_up",
        "jmr_down",
    )

    def __post_init__(self):
        for category, paths in self.triggers.items():
            if not paths:
                raise ValueError(f"category {category!r} has no trigger path")
            outside = [p for p in paths if p not in DATA_PRODUCTION_TRIGGERS]
            if outside:
                raise ValueError(
                    f"category {category!r} requires {outside}, which the data production does not keep "
                    f"(its filter is {' OR '.join(DATA_PRODUCTION_TRIGGERS)}): those events exist only in the MC"
                )


DEFAULT = AnalysisConfig()


@dataclass(frozen=True)
class TtbarCRConfig:
    """Semileptonic ttbar control region (legacy selections/convert_ttbar_cr).

    Same inputs, jet corrections and object IDs as the analysis (AnalysisConfig),
    except that the AK8 soft-drop mass gets no JMS/JMR: this region measures them.
    """

    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    trigger: str = "DST_PFScouting_SingleMuon"
    mu_min_pt: float = 10.0
    mu_max_abs_eta: float = 2.4
    probe_max_abs_eta: float = 2.4
    probe_min_abs_dphi: float = 1.5708  # legacy literal, not pi/2
    # JMS/JMR variations would equal nominal without JMS/JMR.
    jet_variations: tuple[str, ...] = ("jes_up", "jes_down", "jer_up", "jer_down")


TTBAR_CR = TtbarCRConfig()
