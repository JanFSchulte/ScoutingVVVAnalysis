"""Per-category event variables: the classifier inputs and analysis-cut variables.

Port of the output formulas of the legacy selections/convert/branch.json
(output.trees[*].scalars), restricted to the branches the k-fold BDTs read
(selections/BDT/study_boosted3/kfold_finedecor_jec_powheg*/<tree>/fold0/branch.json),
their selection thresholds, and the mass-decorrelation variables.

Legacy evaluates every formula on TLorentzVectors built with SetPtEtaPhiM in
double precision (leptons massless), so kinematics here go through the same
Cartesian components. Defaults for missing objects follow convert_branch.C:
def = -99 (aggregations, slots), kMissingDistance = -999 (pair and closest
distances), -1 (event shapes with fewer than two objects).
"""

import awkward as ak
import numpy as np

from vvv.config import W_MASS, Z_MASS, ObjectConfig
from vvv.objects import best_sfos_pair

DEF = -99.0
MISSING_DISTANCE = -999.0
WZ_MASS_SIGMA = 12.7
PAIR_DR_SIGMA = 0.60

AK8 = "ScoutingFatPFJetRecluster"
AK4 = "ScoutingPFJetRecluster2"
PART = ("Xbb", "Xcc", "Xcs", "Xqq", "Xss", "Xud")


# --- TLorentzVector-like helpers on jagged or flat arrays ------------------


def _p4(objects):
    """Cartesian components (px, py, pz, E) as legacy builds them."""
    pt = abs(ak.values_astype(objects.pt, np.float64))
    eta = ak.values_astype(objects.eta, np.float64)
    phi = ak.values_astype(objects.phi, np.float64)
    mass = ak.values_astype(objects.mass, np.float64)
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    p2 = px * px + py * py + pz * pz
    energy = ak.where(mass >= 0, np.sqrt(p2 + mass * mass), np.sqrt(np.maximum(p2 - mass * mass, 0.0)))
    return px, py, pz, energy


def _pt(px, py):
    return np.sqrt(px * px + py * py)


def _eta(px, py, pz):
    mag = np.sqrt(px * px + py * py + pz * pz)
    cos_theta = ak.where(mag == 0, 1.0, pz / ak.where(mag == 0, 1.0, mag))
    inside = cos_theta * cos_theta < 1.0
    safe = ak.where(inside, cos_theta, 0.0)
    eta = -0.5 * np.log((1.0 - safe) / (1.0 + safe))
    return ak.where(inside, eta, ak.where(pz == 0, 0.0, ak.where(pz > 0, 1e10, -1e10)))


def _phi(px, py):
    return ak.where((px == 0) & (py == 0), 0.0, np.arctan2(py, px))


def _mass(px, py, pz, energy):
    mm = energy * energy - (px * px + py * py + pz * pz)
    return np.copysign(np.sqrt(abs(mm)), mm)  # TLorentzVector::M: -sqrt(-m2) if m2 < 0


def _wrap(dphi):
    """TVector2::Phi_mpi_pi."""
    return (dphi + np.pi) % (2 * np.pi) - np.pi


class Kin:
    """Per-object (or per-event) kinematics in legacy convention."""

    def __init__(self, px, py, pz, energy):
        self.px, self.py, self.pz, self.energy = px, py, pz, energy

    @classmethod
    def of(cls, objects):
        return cls(*_p4(objects))

    def __add__(self, other):
        return Kin(self.px + other.px, self.py + other.py, self.pz + other.pz, self.energy + other.energy)

    @property
    def pt(self):
        return _pt(self.px, self.py)

    @property
    def eta(self):
        return _eta(self.px, self.py, self.pz)

    @property
    def phi(self):
        return _phi(self.px, self.py)

    @property
    def mass(self):
        return _mass(self.px, self.py, self.pz, self.energy)

    def delta_phi(self, other):
        return _wrap(self.phi - other.phi)

    def delta_r(self, other):
        deta = self.eta - other.eta
        dphi = self.delta_phi(other)
        return np.sqrt(deta * deta + dphi * dphi)

    def __getitem__(self, where):
        return Kin(self.px[where], self.py[where], self.pz[where], self.energy[where])


def _at(objects, index):
    """objects[:, index] (None where the event has too few objects)."""
    return ak.pad_none(objects, index + 1, axis=1)[:, index]


def _slot(values, index, default=DEF):
    return ak.fill_none(_at(values, index), default)


def _safe_div(num, den, fallback):
    return ak.where(den == 0, fallback, num / ak.where(den == 0, 1.0, den))


def _max_value(values, default):
    return ak.fill_none(ak.max(values, axis=1), default)


def _nth_max(values, rank, default):
    ordered = ak.sort(values, axis=1, ascending=False)
    return ak.where(ak.num(values) >= rank, ak.fill_none(_at(ordered, rank - 1), default), default)


def _pair_metric(objects, limit, delta_phi, take_min):
    kin = Kin.of(objects[:, :limit] if limit else objects)
    i, j = ak.unzip(ak.argcombinations(kin.px, 2))
    a, b = kin[i], kin[j]
    values = abs(a.delta_phi(b)) if delta_phi else a.delta_r(b)
    best = ak.min(values, axis=1) if take_min else ak.max(values, axis=1)
    return ak.fill_none(best, MISSING_DISTANCE)


def _closest_delta_r(objects, refs):
    """closest_deltaR(collection, refs...): refs are per-event Kin (never missing)."""
    kin = Kin.of(objects)
    best = None
    for ref in refs:
        ref_b = Kin(*(ak.broadcast_arrays(c, kin.px)[0] for c in (ref.px, ref.py, ref.pz, ref.energy)))
        dr = ak.min(kin.delta_r(ref_b), axis=1)
        best = dr if best is None else np.minimum(best, dr)
    return ak.fill_none(best, MISSING_DISTANCE)


def _event_shapes(kins):
    """(sphericity, aplanarity, planarity) of single objects; -1 if < 2 with |p| > 0."""
    sxx = syy = szz = sxy = sxz = syz = norm = 0.0
    n = 0
    for k in kins:
        px, py, pz = (ak.fill_none(c, 0.0) for c in (k.px, k.py, k.pz))
        p2 = px * px + py * py + pz * pz
        use = p2 > 0
        sxx = sxx + ak.where(use, px * px, 0.0)
        syy = syy + ak.where(use, py * py, 0.0)
        szz = szz + ak.where(use, pz * pz, 0.0)
        sxy = sxy + ak.where(use, px * py, 0.0)
        sxz = sxz + ak.where(use, px * pz, 0.0)
        syz = syz + ak.where(use, py * pz, 0.0)
        norm = norm + ak.where(use, p2, 0.0)
        n = n + ak.values_astype(use, np.int64)
    valid = (n >= 2) & (norm > 0)
    norm = ak.where(valid, norm, 1.0)
    l1, l2, l3 = symmetric_eigenvalues3(
        *(ak.to_numpy(x / norm).astype(np.float64) for x in (sxx, syy, szz, sxy, sxz, syz))
    )
    l3 = np.maximum(l3, 0.0)
    valid = ak.to_numpy(valid)
    return (
        np.where(valid, 1.5 * (l2 + l3), -1.0),
        np.where(valid, 1.5 * l3, -1.0),
        np.where(valid, l2 - l3, -1.0),
    )


def symmetric_eigenvalues3(a, b, c, d, e, f):
    """Descending eigenvalues of [[a, d, e], [d, b, f], [e, f, c]] (legacy Smith 1961)."""
    p1 = d * d + e * e + f * f
    q = (a + b + c) / 3.0
    p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b11, b22, b33 = (a - q) / p, (b - q) / p, (c - q) / p
        b12, b13, b23 = d / p, e / p, f / p
    det = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13) + b13 * (b12 * b23 - b22 * b13)
    r = np.clip(det / 2.0, -1.0, 1.0)
    phi = np.arccos(r) / 3.0
    l1 = q + 2.0 * p * np.cos(phi)
    l3 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    l2 = 3.0 * q - l1 - l3
    diagonal = p1 <= 1e-18
    srt = np.sort(np.stack([a, b, c], axis=-1), axis=-1)
    l1 = np.where(diagonal, srt[..., 2], l1)
    l2 = np.where(diagonal, srt[..., 1], l2)
    l3 = np.where(diagonal, srt[..., 0], l3)
    return l1, l2, l3


def best_wz_dijet(jets, limit=4):
    """(found, i, j) of the pair minimising the legacy combined W/Z-mass + dR chi2."""
    kin = Kin.of(jets[:, :limit])
    i, j = ak.unzip(ak.argcombinations(kin.px, 2))
    a, b = kin[i], kin[j]
    m = (a + b).mass
    mass_term = np.minimum(abs(m - W_MASS), abs(m - Z_MASS)) / WZ_MASS_SIGMA
    dr_term = a.delta_r(b) / PAIR_DR_SIGMA
    chi2 = mass_term * mass_term + dr_term * dr_term
    best = ak.argmin(chi2, axis=1, keepdims=True)
    return ~ak.is_none(ak.firsts(best)), ak.firsts(i[best]), ak.firsts(j[best])


def _ratio(ak8, signal):
    num = sum(ak8[f"scoutGlobalParT_prob_{p}"] for p in signal)
    return _safe_div(num, num + ak8.scoutGlobalParT_prob_QCD, 0.0)


def _fill_kin(kin):
    """Missing objects become the zero vector, as in legacy (TLorentzVector())."""
    return Kin(*(ak.fill_none(c, 0.0) for c in (kin.px, kin.py, kin.pz, kin.energy)))


def _kin_at(kin, index):
    return _fill_kin(Kin(*(_at(c, index) for c in (kin.px, kin.py, kin.pz, kin.energy))))


def _kin_pick(kin, index):
    """Per-event object kin[:, index] for a per-event (optional) index."""
    where = ak.singletons(index)
    return _fill_kin(Kin(*(ak.firsts(c[where]) for c in (kin.px, kin.py, kin.pz, kin.energy))))


def category_variables(obj, category, cfg: ObjectConfig):
    """Legacy output branches of one category's tree, one value per event.

    Computed for every event; only events in the category carry legacy
    meaning (the caller masks).
    """
    boosted3 = category == "boosted3"
    suffix = "c3" if boosted3 else "c2"
    n_slots = 3 if boosted3 else 2
    fat = obj["ak8_sel3"] if boosted3 else obj["ak8_sel2"]
    jets = obj[f"ak4_{suffix}"]
    jets_loose = obj[f"ak4_dR0p4_{suffix}"]
    leps = obj[f"lep_veto_{suffix}"]
    met_pt = ak.values_astype(obj["met_pt"], np.float64)
    met_phi = ak.values_astype(obj["met_phi"], np.float64)

    out = {}
    fat_kin = Kin.of(fat)
    fat_slots = [_kin_at(fat_kin, i) for i in range(n_slots)]
    fat_sum = fat_slots[0]
    for k in fat_slots[1:]:
        fat_sum = fat_sum + k
    fat_pts = [k.pt for k in fat_slots]

    # Slot branches of the selected AK8 jets.
    for i in range(n_slots):
        n = i + 1
        out[f"{AK8}_pt_{n}"] = _slot(fat.pt, i)
        out[f"{AK8}_eta_{n}"] = _slot(fat.eta, i)
        out[f"{AK8}_msoftdrop_{n}"] = _slot(fat.msoftdrop, i)
        out[f"{AK8}_WvsQCD_{n}"] = _slot(_ratio(fat, ("Xud", "Xcs")), i)
        out[f"{AK8}_VvsQCD_{n}"] = _slot(_ratio(fat, ("Xud", "Xcs", "Xqq", "Xbb", "Xcc", "Xss")), i)
        out[f"{AK8}_ZvsQCD_{n}"] = _slot(_ratio(fat, ("Xbb", "Xcc", "Xss", "Xqq")), i)
        for p in PART:
            out[f"{AK8}_prob_{p}_{n}"] = _slot(fat[f"scoutGlobalParT_prob_{p}"], i)

    out[f"{AK8}_selected_pt_sum"] = ak.sum(Kin.of(fat).pt, axis=1)
    msd = [ak.values_astype(_slot(fat.msoftdrop, i), np.float64) for i in range(n_slots)]
    out[f"{AK8}_sphere_mass"] = np.sqrt(sum((m - 85.0) ** 2 for m in msd))
    fat_mass = fat_sum.mass
    fat_pt = fat_sum.pt
    out[f"{AK8}_mass"] = fat_mass
    if boosted3:
        out[f"{AK8}_pt"] = fat_pt
    for i in range(n_slots):
        out[f"{AK8}_mass_fraction_{i + 1}"] = _safe_div(_slot(fat.mass, i), fat_mass, 0.0)
    if boosted3:
        for i in range(3):
            out[f"{AK8}_pt_fraction_{i + 1}"] = _safe_div(fat_pts[i], fat_pt, 0.0)
        out[f"{AK8}_pt_balance"] = _safe_div(fat_pt, sum(fat_pts), 0.0)
        out[f"{AK8}_deltaR_min"] = _pair_metric(fat, 3, False, True)
        out[f"{AK8}_deltaR_max"] = _pair_metric(fat, 3, False, False)
        out[f"{AK8}_deltaPhi_min"] = _pair_metric(fat, 3, True, True)
        out[f"{AK8}_deltaPhi_max"] = _pair_metric(fat, 3, True, False)
    else:
        out[f"{AK8}_deltaR"] = fat_slots[0].delta_r(fat_slots[1])
        out[f"{AK8}_deltaEta"] = abs(fat_slots[0].eta - fat_slots[1].eta)
        out[f"{AK4}_deltaR_min_{AK8}"] = _closest_delta_r(jets, fat_slots)
        out[f"{AK4}_deltaR_min"] = _pair_metric(jets, None, False, True)
        for i in range(4):
            out[f"{AK4}_pt_{i + 1}"] = _slot(jets.pt, i)
            out[f"{AK4}_mass_{i + 1}"] = _slot(jets.mass, i)

    out[f"SelectedLepton_deltaR_min_{AK8}"] = _closest_delta_r(leps, fat_slots)

    probs = (
        jets_loose.scoutUParT_probb + jets_loose.scoutUParT_probbb + jets_loose.scoutUParT_probleptonicB
    )
    total = probs + jets_loose.scoutUParT_probc + jets_loose.scoutUParT_probg + jets_loose.scoutUParT_probuds
    out[f"{AK4}_scoutUParT_bvsall_max_dR0p4"] = _max_value(_safe_div(probs, total, DEF), DEF)
    out[f"{AK4}_scoutUParT_probb_second_max_dR0p4"] = _nth_max(jets_loose.scoutUParT_probb, 2, DEF)
    out[f"n{AK4}_selected"] = ak.num(jets)

    jet_kin = Kin.of(jets)
    ht = ak.sum(jet_kin.pt, axis=1)
    out[f"{AK4}_selected_HT"] = ht
    out["ScoutingMET_pt"] = met_pt
    out["ScoutingMET_over_HT"] = _safe_div(met_pt, out[f"{AK8}_selected_pt_sum"] + ht, -1.0)
    out["ScoutingMET_ak4_cosdphi_max"] = _max_value(np.cos(jet_kin.phi - met_phi), DEF)
    out["ScoutingMET_ak8_cosdphi_max"] = _max_value(np.cos(Kin.of(fat).phi - met_phi), DEF)
    out[f"n{AK4}_btag_loose"] = ak.sum(jets.scoutUParT_probb > 0.2, axis=1)
    out[f"n{AK4}_btag_medium"] = ak.sum(jets.scoutUParT_probb > 0.5, axis=1)

    # Resolved W/Z -> jj candidate among the first four resolved AK4 jets.
    resolved = obj[f"ak4_{suffix}"]  # ak4_resolved_c2 has the ak4_c2 selection
    found, bi, bj = best_wz_dijet(resolved, 4)
    res_kin = Kin.of(resolved)
    ji, jj = _kin_pick(res_kin, bi), _kin_pick(res_kin, bj)
    dijet = ji + jj
    if not boosted3:
        out[f"{AK4}_dijet_combined_mass"] = ak.where(found, dijet.mass, 0.0)
        out[f"{AK4}_dijet_combined_pt"] = ak.where(found, dijet.pt, 0.0)

    shape_objects = list(fat_slots)
    if category == "boosted2_resolved":
        r0, r1 = _kin_at(res_kin, 0), _kin_at(res_kin, 1)
        shape_objects += [r0, r1]
        fjj = fat_sum + r0 + r1
        out[f"{AK8}{AK4}_mass"] = fjj.mass
        out[f"{AK8}{AK4}_pt"] = fjj.pt
        out[f"{AK8}{AK4}_pt_fraction_1"] = _safe_div(fat_pts[0], fjj.pt, 0.0)
        out[f"{AK8}{AK4}_pt_fraction_2"] = _safe_div(fat_pts[1], fjj.pt, 0.0)
        out[f"{AK8}{AK4}_pt_balance"] = _safe_div(fjj.pt, fat_pts[0] + fat_pts[1] + r0.pt + r1.pt, 0.0)
        dr_fat = [dijet.delta_r(k) for k in fat_slots]
        out[f"{AK4}_dijet_combined_deltaR_ak8_min"] = np.minimum(dr_fat[0], dr_fat[1])
        out[f"{AK4}_dijet_combined_deltaR_ak8_max"] = np.maximum(dr_fat[0], dr_fat[1])
        dr_jj = ji.delta_r(jj)
        out[f"{AK4}_dijet_combined_deltaR"] = dr_jj
        m = dijet.mass
        out[f"{AK4}_dijet_combined_chi2"] = (
            np.minimum(abs(m - W_MASS), abs(m - Z_MASS)) / WZ_MASS_SIGMA
        ) ** 2 + (dr_jj / PAIR_DR_SIGMA) ** 2
    sph, apl, pla = _event_shapes(shape_objects)
    out["ScoutingEvent_sphericity"] = sph
    out["ScoutingEvent_aplanarity"] = apl
    out["ScoutingEvent_planarity"] = pla

    # Leptons (merged electron + muon veto collection, massless p4).
    lep_kin = Kin.of(leps)
    if not boosted3:
        out["SelectedLepton_pt_1"] = _slot(lep_kin.pt, 0)
        out["SelectedLepton_eta_1"] = _slot(lep_kin.eta, 0)
        # first_valid(electron field, muon field): the merged record carries the source's value.
        out["SelectedLepton_ecalIso_1"] = _slot(leps.ecalIso, 0)
        out["SelectedLepton_hcalIso_1"] = _slot(leps.hcalIso, 0)
    out["nVetoMuon"] = ak.sum(~leps.is_electron, axis=1)
    out["nVetoElectron"] = ak.sum(leps.is_electron, axis=1)

    if category == "boosted2_leptonic_1lep":
        lep_pt = ak.fill_none(_at(lep_kin.pt, 0), 0.0)
        lep_phi = ak.fill_none(_at(lep_kin.phi, 0), 0.0)
        out["LeptonicW_mT"] = np.sqrt(2.0 * lep_pt * met_pt * (1.0 - np.cos(lep_phi - met_phi)))
        out["LeptonicW_pt"] = np.sqrt(
            (lep_pt * np.cos(lep_phi) + met_pt * np.cos(met_phi)) ** 2
            + (lep_pt * np.sin(lep_phi) + met_pt * np.sin(met_phi)) ** 2
        )
        lep0 = _kin_at(lep_kin, 0)
        dr = [lep0.delta_r(k) for k in fat_slots]
        out["SelectedLepton_deltaR_ak8_1"] = dr[0]
        out["SelectedLepton_deltaR_ak8_2"] = dr[1]
        out["SelectedLepton_deltaR_ak8_max"] = np.maximum(dr[0], dr[1])

    if category == "boosted2_leptonic_2lep":
        found_z, _ = best_sfos_pair(leps, cfg)
        z = _sfos_kin(leps, cfg)
        out["DileptonZ_mass"] = ak.where(found_z, z.mass, 0.0)
        out["DileptonZ_pt"] = ak.where(found_z, z.pt, 0.0)
        dr = [z.delta_r(k) for k in fat_slots]
        out["DileptonZ_deltaR_ak8_min"] = np.minimum(dr[0], dr[1])
        out["DileptonZ_deltaR_ak8_max"] = np.maximum(dr[0], dr[1])

    return {k: _to_numpy(v) for k, v in out.items()}


def _sfos_kin(leps, cfg: ObjectConfig):
    """Legacy-convention p4 of the SFOS pair closest to mZ (zero vector if none)."""
    from vvv.objects import _opposite_sign

    first = leps[:, : cfg.sfos_max_leptons]
    i, j = ak.unzip(ak.argcombinations(first, 2))
    l1, l2 = first[i], first[j]
    sfos = (l1.is_electron == l2.is_electron) & _opposite_sign(l1.charge, l2.charge, cfg)
    kin = Kin.of(first)
    pair = kin[i] + kin[j]
    m = ak.where(sfos, abs(pair.mass - Z_MASS), np.inf)
    best = ak.argmin(m, axis=1, keepdims=True)
    return _fill_kin(Kin(*(ak.firsts(c[best]) for c in (pair.px, pair.py, pair.pz, pair.energy))))


def _to_numpy(values):
    array = ak.to_numpy(ak.fill_none(values, DEF)) if isinstance(values, ak.Array) else np.asarray(values)
    if array.dtype.kind == "b":
        return array.astype(np.int32)
    if array.dtype.kind in "iu":
        return array.astype(np.int32)
    return array.astype(np.float32)


def all_variables(obj, categories, cfg: ObjectConfig):
    return {cat: category_variables(obj, cat, cfg) for cat in categories}


def passthrough(events, is_mc):
    """Event identifiers and (MC) generator inputs copied unchanged."""
    out = {
        "run": ak.to_numpy(events.run),
        "luminosityBlock": ak.to_numpy(events.luminosityBlock),
        "event": ak.to_numpy(events.event),
    }
    if is_mc:
        out["genWeight"] = ak.to_numpy(events.genWeight).astype(np.float32)
        out["Pileup_nTrueInt"] = ak.to_numpy(events.Pileup.nTrueInt).astype(np.float32)
    return out
