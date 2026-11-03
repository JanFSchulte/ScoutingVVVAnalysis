"""Object definitions, the per-AK8 ISR score, cleaning and the five categories.

Port of the "collections" and "tree_selection" blocks of the legacy
selections/convert/selection.json. Collection names in comments are the legacy
ones (ak8, ak8_sel2, ak4_resolved_c2, lep_veto_c3, ...).
"""

import awkward as ak
import numpy as np

from vvv.config import W_MASS, Z_MASS, ObjectConfig

INT32_MIN = np.iinfo(np.int32).min
INT32_SENTINEL = np.iinfo(np.int32).max


def p4_pt_eta(objects):
    """pT and eta as legacy cuts on them: from a TLorentzVector built with
    SetPtEtaPhiM in double precision. The round trip moves values stored
    exactly on a cut (scouting stores eta with a 10-bit mantissa, so
    |eta| = 2.5 occurs) to either side of it.
    """
    counts = ak.num(objects)
    pt = ak.to_numpy(ak.flatten(objects.pt)).astype(np.float64)
    eta = ak.to_numpy(ak.flatten(objects.eta)).astype(np.float64)
    phi = ak.to_numpy(ak.flatten(objects.phi)).astype(np.float64)
    pt = np.abs(pt)
    x, y, z = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    perp = np.sqrt(x * x + y * y)
    mag = np.sqrt(x * x + y * y + z * z)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.where(mag == 0.0, 1.0, z / np.where(mag == 0.0, 1.0, mag))
        p4_eta = np.where(
            cos_theta * cos_theta < 1.0,
            -0.5 * np.log((1.0 - cos_theta) / (1.0 + cos_theta)),
            np.where(z == 0, 0.0, np.where(z > 0, 1e10, -1e10)),
        )
    return ak.unflatten(perp, counts), ak.unflatten(p4_eta, counts)


def kinematic_mask(objects, min_pt, max_abs_eta):
    pt, eta = p4_pt_eta(objects)
    return (pt > min_pt) & (abs(eta) < max_abs_eta)


def sort_by_pt(objects):
    return objects[ak.argsort(objects.pt, axis=1, ascending=False, stable=True)]


def min_delta_r(objects, others):
    """Per object, the smallest deltaR to any of others (inf if none)."""
    pairs = ak.cartesian([objects, others], axis=1, nested=True)
    dr = pairs["0"].delta_r(pairs["1"])
    return ak.fill_none(ak.min(dr, axis=2), np.inf)


def isr_score(ak8, ak4_nocuts, cfg: ObjectConfig):
    """ISR-vs-W discriminant of each AK8 jet (legacy isrScore)."""
    signal = (
        ak8.scoutGlobalParT_prob_Xud
        + ak8.scoutGlobalParT_prob_Xcs
        + ak8.scoutGlobalParT_prob_Xqq
        + ak8.scoutGlobalParT_prob_Xbb
        + ak8.scoutGlobalParT_prob_Xcc
        + ak8.scoutGlobalParT_prob_Xss
    )
    total = signal + ak8.scoutGlobalParT_prob_QCD
    tag = ak.where(total != 0, signal / ak.where(total != 0, total, 1.0), 0.0)
    mass_term = np.minimum(abs(ak8.msoftdrop - W_MASS), abs(ak8.msoftdrop - Z_MASS))

    pairs = ak.cartesian([ak8, ak4_nocuts], axis=1, nested=True)
    near = pairs["0"].delta_r(pairs["1"]) < cfg.isr_ratio_dr
    ratio = ak.where(pairs["0"].pt > 0, pairs["1"].pt / pairs["0"].pt, 0.0)
    max_ratio = ak.fill_none(ak.max(ratio[near], axis=2), cfg.isr_ratio_default)

    return (
        cfg.isr_mass_coef * mass_term
        + cfg.isr_tag_coef * tag
        + cfg.isr_ratio_coef * max_ratio
        + cfg.isr_offset
    )


def value_at(array, index, default):
    """array[:, index] with a default where the event has too few objects."""
    padded = ak.pad_none(array, index + 1, axis=1)
    return ak.fill_none(padded[:, index], default)


def ak4_jet_id(jets, cfg: ObjectConfig):
    return (
        kinematic_mask(jets, cfg.ak4_min_pt, cfg.ak4_max_abs_eta)
        & (jets.nConstituents > 1)
        & (jets.neHEF < 0.99)
        & (jets.neEmEF < 0.90)
        & (jets.chHadMultiplicity > 0)
        & (jets.muEF < 0.80)
    )


def electron_veto_id(ele, cfg: ObjectConfig):
    barrel = abs(ele.eta) < 1.479
    eb = (
        (ele.sigmaIetaIeta < 0.0126)
        & (abs(ele.dEtaIn) < 0.00463)
        & (abs(ele.dPhiIn) < 0.148)
        & (ele.hOverE < 0.10)
        & (abs(ele.ooEMOop) < 0.209)
        & (ele.missingHits <= 2)
    )
    ee = (
        (ele.sigmaIetaIeta < 0.0457)
        & (abs(ele.dEtaIn) < 0.00814)
        & (abs(ele.dPhiIn) < 0.19)
        & (ele.hOverE < 0.10)
        & (abs(ele.ooEMOop) < 0.132)
        & (ele.missingHits <= 3)
    )
    iso = ele.trackIso + ele.ecalIso + ele.hcalIso
    rel_iso = ak.where(ele.pt != 0, iso / ak.where(ele.pt != 0, ele.pt, 1.0), 99.0)
    return ak.where(barrel, eb, ee) & (rel_iso < cfg.ele_max_rel_iso)


def muon_veto_id(mu, cfg: ObjectConfig):
    rel_iso = ak.where(mu.pt != 0, mu.trackIso / ak.where(mu.pt != 0, mu.pt, 1.0), 99.0)
    return (
        (mu.nValidStandAloneMuonHits > 0)
        & (mu.normchi2 < 3)
        & (mu.nTrackerLayersWithMeasurement > 5)
        & (mu.nValidPixelHits > 0)
        & (mu.nRecoMuonMatchedStations >= 2)
        # mu.type would be the awkward array type, not the field.
        & ((mu["type"] == 2) | (mu["type"] == 6) | (mu["type"] == 10) | (mu["type"] == 14))
        & (rel_iso < cfg.mu_max_rel_iso)
    )


def _lepton_charge(charge, is_electron, cfg: ObjectConfig):
    """Charge as legacy pairs it (see ObjectConfig.legacy_electron_charge)."""
    charge = ak.values_astype(charge, np.int64)
    if cfg.legacy_electron_charge:
        return ak.where(is_electron & (charge == INT32_SENTINEL), INT32_MIN, charge)
    return ak.where(is_electron & (charge == INT32_SENTINEL), 0, charge)


def _opposite_sign(q1, q2, cfg: ObjectConfig):
    if cfg.legacy_electron_charge:
        # int32 product with wrap-around, as in the legacy C++.
        product = ak.values_astype(q1 * q2, np.int64)
        wrapped = ((product + 2**31) % 2**32) - 2**31
        return wrapped < 0
    return (q1 * q2) < 0


def leptons(electrons, muons, clean_against, cfg: ObjectConfig):
    """Veto leptons (pT-sorted, massless) cleaned against the given AK8 jets."""
    ele = electrons[
        kinematic_mask(electrons, cfg.ele_min_pt, cfg.ele_max_abs_eta)
        & (min_delta_r(electrons, clean_against) > cfg.clean_dr)
    ]
    mu = muons[
        kinematic_mask(muons, cfg.mu_min_pt, cfg.mu_max_abs_eta)
        & (min_delta_r(muons, clean_against) > cfg.clean_dr)
    ]
    ele = ele[electron_veto_id(ele, cfg)]
    mu = mu[muon_veto_id(mu, cfg)]

    def record(objs, is_electron):
        return ak.zip(
            {
                "pt": objs.pt,
                "eta": objs.eta,
                "phi": objs.phi,
                "mass": ak.zeros_like(objs.pt),
                "charge": _lepton_charge(objs.charge, is_electron, cfg),
                "is_electron": ak.full_like(objs.pt, is_electron, dtype=bool),
            },
            with_name="PtEtaPhiMCandidate",
            behavior=objs.behavior,
        )

    merged = ak.concatenate([record(ele, True), record(mu, False)], axis=1)
    return sort_by_pt(merged)


def best_sfos_pair(leps, cfg: ObjectConfig):
    """(found, pair p4) of the SFOS pair closest to mZ among the first leptons."""
    first = leps[:, : cfg.sfos_max_leptons]
    pairs = ak.combinations(first, 2, fields=["l1", "l2"])
    sfos = (pairs.l1.is_electron == pairs.l2.is_electron) & _opposite_sign(
        pairs.l1.charge, pairs.l2.charge, cfg
    )
    pairs = pairs[sfos]
    p4 = pairs.l1 + pairs.l2
    best = ak.argmin(abs(p4.mass - Z_MASS), axis=1, keepdims=True)
    chosen = ak.firsts(p4[best])
    return ~ak.is_none(chosen), chosen


def build_objects(ak4_in, ak8_in, electrons, muons, cfg: ObjectConfig):
    """All analysis objects from corrected AK4/AK8 jets and raw leptons."""
    ak8 = sort_by_pt(ak8_in)
    ak8 = ak8[kinematic_mask(ak8, cfg.ak8_min_pt, cfg.ak8_max_abs_eta)]
    ak4_nocuts = sort_by_pt(ak4_in)
    score = isr_score(ak8, ak4_nocuts, cfg)
    ak8 = ak.with_field(ak8, score, "isrScore")
    n_ak8 = ak.num(ak8)

    # ak8_sel3: the three leading AK8 jets.
    third_pt = value_at(ak8.pt, 2, 0.0)
    in_top3 = ak8.pt >= third_pt
    ak8_sel3 = ak8[in_top3]

    # ak8_sel2: two jets of the leading three, dropping the one with the lowest score.
    min_score3 = np.minimum(
        np.minimum(value_at(score, 0, -1e9), value_at(score, 1, -1e9)), value_at(score, 2, -1e9)
    )
    ak8_sel2 = ak8[in_top3 & ((n_ak8 == 2) | (score > min_score3))]

    good_ak4 = ak4_nocuts[ak4_jet_id(ak4_nocuts, cfg)]

    def cleaned(against, dr):
        return good_ak4[min_delta_r(good_ak4, against) > dr]

    return {
        "ak8": ak8,
        "ak8_sel2": ak8_sel2,
        "ak8_sel3": ak8_sel3,
        "ak4_nocuts": ak4_nocuts,
        "ak4_all": good_ak4,
        "ak4": cleaned(ak8, cfg.clean_dr),
        "ak4_dR0p4": cleaned(ak8, cfg.clean_dr_loose),
        "ak4_c2": cleaned(ak8_sel2, cfg.clean_dr),
        "ak4_dR0p4_c2": cleaned(ak8_sel2, cfg.clean_dr_loose),
        "ak4_c3": cleaned(ak8_sel3, cfg.clean_dr),
        "ak4_dR0p4_c3": cleaned(ak8_sel3, cfg.clean_dr_loose),
        "lep_veto": leptons(electrons, muons, ak8, cfg),
        "lep_veto_c2": leptons(electrons, muons, ak8_sel2, cfg),
        "lep_veto_c3": leptons(electrons, muons, ak8_sel3, cfg),
        "min_score3": min_score3,
    }


CATEGORIES = (
    "boosted3",
    "boosted2_resolved",
    "boosted2_unresolved",
    "boosted2_leptonic_1lep",
    "boosted2_leptonic_2lep",
)


def categories(obj, cfg: ObjectConfig):
    """Boolean mask per category (legacy tree_selection)."""
    n_ak8 = ak.num(obj["ak8"])
    n_lep2 = ak.num(obj["lep_veto_c2"])
    n_lep3 = ak.num(obj["lep_veto_c3"])
    three_like = (n_ak8 >= 3) & (obj["min_score3"] > cfg.boosted3_min_score)
    two_like = (n_ak8 == 2) | ((n_ak8 >= 3) & (obj["min_score3"] <= cfg.boosted3_min_score))
    n_resolved = ak.num(obj["ak4_c2"])
    has_sfos, _ = best_sfos_pair(obj["lep_veto_c2"], cfg)
    boosted3_veto = three_like & (n_lep3 == 0)
    return {
        "boosted3": boosted3_veto,
        "boosted2_resolved": two_like & (n_resolved >= 2) & (n_lep2 == 0),
        "boosted2_unresolved": two_like & (n_resolved < 2) & (n_lep2 == 0),
        "boosted2_leptonic_1lep": (n_ak8 >= 2) & (n_lep2 == 1) & ~boosted3_veto,
        "boosted2_leptonic_2lep": (n_ak8 >= 2) & has_sfos & ~boosted3_veto,
    }
