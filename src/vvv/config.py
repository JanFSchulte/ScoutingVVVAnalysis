"""Analysis constants.

Values follow the legacy ScoutingVVVTools configuration at git 421d4d7:
selections/convert/{config,selection,branch}.json and the constants at the top
of selections/convert/convert_branch.C.
"""

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CORRECTIONS_DIR = DATA_DIR / "corrections"
JME_POG_DIR = Path("/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME")

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
    jer_file: Path = JME_POG_DIR / "2024_Summer24" / "jet_jerc.json.gz"
    jer_smear_file: Path = JME_POG_DIR / "jer_smear.json.gz"
    jer_resolution: str = "Summer23BPixPrompt23_RunD_JRV1_MC_PtResolution_AK4PFPuppi"
    jer_scale_factor: str = "Summer23BPixPrompt23_RunD_JRV1_MC_ScaleFactor_AK4PFPuppi"
    # The PtResolution Rho axis ends here and returns 1.0 beyond it.
    jer_rho_upper_edge: float = 52.05
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


@dataclass(frozen=True)
class AnalysisConfig:
    inputs: InputConfig = field(default_factory=InputConfig)
    jets: JetCorrectionConfig = field(default_factory=JetCorrectionConfig)
    rho: RhoRecoveryConfig = field(default_factory=RhoRecoveryConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    trigger: str = "DST_PFScouting_JetHT"  # NanoEvents: events.DST.PFScouting_JetHT
    lumi_mask: Path = DATA_DIR / "lumi" / "Cert_Collisions2024_378981_386951_Golden.json"
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
