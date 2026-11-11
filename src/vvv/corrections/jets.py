"""Jet energy corrections, rho recovery, JER, JMS/JMR and Type-1 MET.

Started as a port of JetPtCorrector in the legacy convert_branch.C. Nominal chain:

- AK4 (ScoutingPFJetRecluster2) are stored with the production
  Winter24HLT_V1 L1L2L3Res AK4PFHLT JEC; raw = stored * (1 - rawFactor).
- The event rho is not stored: it is recovered by inverting the AK4 JEC.
- AK8 (ScoutingFatPFJetRecluster) are stored raw and get the AK8PFHLT JEC.
- The identified (prompt) leptons are subtracted from the raw jets they lie in before
  JEC and JER (subtract_leptons); the JEC of a subtracted AK4 jet is re-evaluated.
- MC jets are smeared with the hybrid method (generator-matched scaling, stochastic
  otherwise; JERSmear) and the offline PUPPI JER of the Summer24 MC, AK4PFPuppi for AK4
  and AK8PFPuppi for AK8. AK8 jets have no stored generator jets and are smeared
  stochastically.
- The AK8 soft-drop mass takes every pT factor except L1FastJet, and for MC
  the JMS scale and JMR smearing; it is not lepton-subtracted (no subjets are stored).
- MET is Type-1 corrected with the (lepton-subtracted) AK4 jets.

All work is done on flat numpy arrays over every jet of the chunk; per-event
quantities are broadcast with the jet -> event index.
"""

import json
from dataclasses import dataclass

import awkward as ak
import correctionlib
import numpy as np

from vvv.config import JetCorrectionConfig, RhoRecoveryConfig


def _flat(array):
    return ak.to_numpy(ak.flatten(array)).astype(np.float64)


def _event_index(counts):
    return np.repeat(np.arange(len(counts)), counts)


# --- deterministic per-jet Gaussian (legacy jetGaussian) -------------------


def _splitmix64(value):
    with np.errstate(over="ignore"):
        value = value + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def _float_bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)


def jet_gaussian(event_id, pt, eta, phi):
    """Standard-normal number fixed by the event and the stored jet kinematics."""
    h = _splitmix64(np.asarray(event_id, dtype=np.uint64))
    h = _splitmix64(h ^ _float_bits(pt))
    h = _splitmix64(h ^ ((_float_bits(eta) << np.uint64(32)) | _float_bits(phi)))
    h2 = _splitmix64(h)
    inv = 1.0 / 9007199254740992.0
    u1 = ((h >> np.uint64(11)).astype(np.float64) + 0.5) * inv
    u2 = ((h2 >> np.uint64(11)).astype(np.float64) + 0.5) * inv
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


# --- rho recovery ----------------------------------------------------------


class RhoRecovery:
    """Event rho from the AK4 production JEC: stored/raw = JEC(A, eta, phi, raw pT, rho)."""

    def __init__(self, jec_ak4, cfg: RhoRecoveryConfig):
        self.jec = jec_ak4
        self.cfg = cfg

    def _jec(self, area, eta, phi, raw_pt, rho):
        return self.jec.evaluate(area, eta, phi, raw_pt, rho)

    def _extended(self, area, eta, phi, raw_pt, rho):
        # Continued beyond the L1FastJet clamp [0, rho_max] by point reflection.
        rmax = self.cfg.rho_max
        rho = np.asarray(rho, dtype=np.float64)
        above = rho > rmax
        below = rho < 0.0
        reflected = np.where(above, 2 * rmax - rho, np.where(below, -rho, rho))
        value = self._jec(area, eta, phi, raw_pt, reflected)
        outside = np.flatnonzero(above | below)
        if len(outside):
            o = outside
            edge = np.where(above[o], rmax, 0.0)
            value[o] = 2.0 * self._jec(area[o], eta[o], phi[o], raw_pt[o], edge) - value[o]
        return value

    def _carries(self, area, eta, phi, raw_pt, stored):
        low = self._jec(area, eta, phi, raw_pt, np.zeros_like(raw_pt))
        high = self._jec(area, eta, phi, raw_pt, np.full_like(raw_pt, self.cfg.rho_max))
        return low - high > self.cfg.min_sensitivity * stored

    def _solve(self, area, eta, phi, raw_pt, stored):
        """Illinois regula falsi on [-margin, rho_max + margin]; returns (rho, ok).

        Same per-jet iteration as legacy solveRho; only unconverged jets are evaluated.
        """
        c = self.cfg
        n = len(raw_pt)
        low = np.full(n, -c.bracket_margin)
        high = np.full(n, c.rho_max + c.bracket_margin)
        f_low = self._extended(area, eta, phi, raw_pt, low) - stored
        f_high = self._extended(area, eta, phi, raw_pt, high) - stored
        ok = (f_low >= 0) & (f_high <= 0) & (f_low > f_high)
        rho = np.zeros(n)
        side = np.zeros(n, dtype=np.int8)
        act = np.flatnonzero(ok)
        for _ in range(c.solver_iterations):
            if not len(act):
                break
            lo, hi, flo, fhi = low[act], high[act], f_low[act], f_high[act]
            guess = (lo * fhi - hi * flo) / (fhi - flo)
            rho[act] = guess
            f = self._extended(area[act], eta[act], phi[act], raw_pt[act], guess) - stored[act]
            step = ~(np.abs(f) <= 1e-10)
            pos = step & (f > 0)
            neg = step & ~(f > 0)
            sd = side[act]
            f_high[act] = np.where(pos & (sd == 1), 0.5 * fhi, fhi)
            f_low[act] = np.where(neg & (sd == -1), 0.5 * flo, flo)
            low[act] = np.where(pos, guess, lo)
            f_low[act] = np.where(pos, f, f_low[act])
            high[act] = np.where(neg, guess, hi)
            f_high[act] = np.where(neg, f, f_high[act])
            side[act] = np.where(pos, 1, np.where(neg, -1, sd))
            act = act[step]
        return np.clip(rho, 0.0, c.rho_max), ok

    def _agrees(self, area, eta, phi, raw_pt, stored, carries, rho):
        tol = self.cfg.tolerance
        upper = self._extended(area, eta, phi, raw_pt, rho - tol)
        lower = self._extended(area, eta, phi, raw_pt, rho + tol)
        return ~carries | ((stored >= lower) & (stored <= upper))

    def __call__(self, area, eta, phi, raw_pt, stored, event_index, n_events, has_jets):
        """Flat AK4 inputs -> (rho per event, ok per event, n_consensus)."""
        usable = (raw_pt > 0) & (stored > self.cfg.min_factor)
        idx = np.flatnonzero(usable)
        a, e, p, r, s = area[idx], eta[idx], phi[idx], raw_pt[idx], stored[idx]
        evt = event_index[idx]

        carries = self._carries(a, e, p, r, s)
        sol = np.zeros_like(r)
        solved = np.zeros(r.shape, dtype=bool)
        if carries.any():
            ci = np.flatnonzero(carries)
            sol_c, ok_c = self._solve(a[ci], e[ci], p[ci], r[ci], s[ci])
            sol[ci] = sol_c
            solved[ci] = ok_c
        candidate = carries & solved

        # First candidate jet of each event (jets are in stored order).
        rho = np.zeros(n_events)
        ok = ~has_jets  # events without jets have rho = 0
        cand_evt, first = np.unique(evt[candidate], return_index=True)
        first_rho = np.full(n_events, np.nan)
        first_rho[cand_evt] = sol[np.flatnonzero(candidate)[first]]

        jet_rho = first_rho[evt]
        has_first = ~np.isnan(jet_rho)
        agree = np.ones_like(r, dtype=bool)
        if has_first.any():
            hi = np.flatnonzero(has_first)
            agree[hi] = self._agrees(a[hi], e[hi], p[hi], r[hi], s[hi], carries[hi], jet_rho[hi])
        agree &= has_first
        all_agree = np.ones(n_events, dtype=bool)
        np.logical_and.at(all_agree, evt, agree)
        good = ~np.isnan(first_rho) & all_agree
        rho[good] = first_rho[good]
        ok |= good

        # Rare events: consensus over neighbouring JEC bins.
        fallback = np.flatnonzero(has_jets & ~good)
        if len(fallback):
            in_fallback = np.isin(evt, fallback)
            fi = np.flatnonzero(in_fallback)
            values = self._consensus(a[fi], e[fi], p[fi], r[fi], s[fi], evt[fi])
            for ev, value in values.items():
                if value is not None:
                    rho[ev] = value
                    ok[ev] = True
        return rho, ok, len(fallback)

    def _shifted_solutions(self, a, e, p, r, s, shifts):
        """Per (jet, shift): whether the shifted JEC bin counts and its rho solution."""
        n, m = len(r), len(shifts)
        de = np.array([x for x, _ in shifts])
        dp = np.array([y for _, y in shifts])
        A, R, S = (np.repeat(v, m) for v in (a, r, s))
        E = (e[:, None] + de[None, :]).ravel()
        P = (p[:, None] + dp[None, :]).ravel()
        own = np.tile((de == 0.0) & (dp == 0.0), n)
        probe = np.full(n * m, self.cfg.bin_probe_rho)
        own_bin = np.repeat(self._jec(a, e, p, r, np.full(n, self.cfg.bin_probe_rho)), m)
        same_bin = ~own & (self._jec(A, E, P, R, probe) == own_bin)
        carries = ~same_bin & self._carries(A, E, P, R, S)
        sol = np.zeros(n * m)
        solved = np.zeros(n * m, dtype=bool)
        ci = np.flatnonzero(carries)
        if len(ci):
            sol[ci], solved[ci] = self._solve(A[ci], E[ci], P[ci], R[ci], S[ci])
        return carries.reshape(n, m), sol.reshape(n, m), solved.reshape(n, m)

    def _consensus(self, a, e, p, r, s, evt):
        """Legacy consensusRho for every fallback event: {event: rho or None}."""
        d = self.cfg.bin_edge_probe
        axis = [(0.0, 0.0), (d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)]
        diagonal = [(x, y) for x in (d, -d) for y in (d, -d)]
        results = {int(ev): None for ev in np.unique(evt)}
        for shifts in (axis, axis + diagonal):
            pending = np.array([ev for ev, v in results.items() if v is None])
            if not len(pending):
                break
            sel = np.isin(evt, pending)
            carries, sol, solved = self._shifted_solutions(a[sel], e[sel], p[sel], r[sel], s[sel], shifts)
            events = evt[sel]
            for ev in pending:
                rows = np.flatnonzero(events == ev)
                results[int(ev)] = self._median_of_solutions(carries[rows], sol[rows], solved[rows])
        return results

    def _median_of_solutions(self, carries, sol, solved):
        solutions = []
        for jet_carries, jet_sol, jet_ok in zip(carries, sol, solved):
            if not jet_carries.any():
                continue
            values = jet_sol[jet_carries & jet_ok]
            if not len(values):
                return None
            solutions.append(values)
        if not solutions:
            return None
        for anchors in solutions:
            for anchor in anchors:
                chosen = np.array([js[np.argmin(np.abs(js - anchor))] for js in solutions])
                median = float(np.median(chosen))
                if np.all(np.abs(chosen - median) <= self.cfg.tolerance):
                    return median
        return None


# --- prompt-lepton subtraction ---------------------------------------------


def _cartesian(pt, eta, phi, mass):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return px, py, pz, np.sqrt(px * px + py * py + pz * pz + mass * mass)


@dataclass
class SubtractedJets:
    """Raw jets after the lepton subtraction (flat arrays over every jet of the chunk)."""

    pt: np.ndarray
    eta: np.ndarray
    phi: np.ndarray
    mass: np.ndarray
    energy: np.ndarray  # of the remainder
    stored_energy: np.ndarray  # of the raw jet before the subtraction
    subtracted: np.ndarray  # a lepton was subtracted
    muon_energy: np.ndarray  # energy of the subtracted muons


def subtract_leptons(jet_pt, jet_eta, jet_phi, jet_mass, jet_counts, has_muon, leptons, radius, max_abs_eta):
    """Raw jets minus the prompt leptons inside them.

    JERC FAQ ("jets containing prompt leptons"): subtract the prompt-lepton four-momentum from
    the raw jet before JEC and JER; here the lepton is not added back afterwards. Each identified
    lepton (objects.identified_leptons, massless) is subtracted from the closest jet within
    deltaR < radius; a muon only from a jet that contains a muon candidate (has_muon). Scouting
    jets contain no electron candidates: the HLT particle flow takes an electron as a charged
    hadron and photons, which stay in the jet around it, so any jet within radius qualifies. A jet
    whose remainder has no positive energy or pT, or points beyond |eta| = max_abs_eta (the range of
    the JME payloads; only remainders of a few GeV get there), gets pt = 0, which fails every
    selection.

    jet_*: flat raw kinematics of all jets, jet_counts: jets per event, has_muon: flat bool;
    leptons: jagged (pt, eta, phi, is_electron). Jets without a subtracted lepton keep their
    stored values exactly.
    """
    px, py, pz, energy = _cartesian(jet_pt, jet_eta, jet_phi, jet_mass)
    stored_energy = energy.copy()
    muon_energy = np.zeros(len(jet_pt))
    subtracted = np.zeros(len(jet_pt), dtype=bool)
    lep_counts = ak.to_numpy(ak.num(leptons))
    if len(jet_pt) and lep_counts.sum():
        lep_event = np.repeat(np.arange(len(lep_counts)), lep_counts)
        l_pt, l_eta, l_phi = (_flat(leptons[f]) for f in ("pt", "eta", "phi"))
        l_electron = ak.to_numpy(ak.flatten(leptons.is_electron)).astype(bool)
        jet_start = np.cumsum(jet_counts) - jet_counts
        # Every (lepton, jet) pair of the same event.
        n_pairs = jet_counts[lep_event]
        pair_lep = np.repeat(np.arange(len(l_pt)), n_pairs)
        within = np.arange(n_pairs.sum()) - np.repeat(np.cumsum(n_pairs) - n_pairs, n_pairs)
        pair_jet = np.repeat(jet_start[lep_event], n_pairs) + within
        dphi = (l_phi[pair_lep] - jet_phi[pair_jet] + np.pi) % (2 * np.pi) - np.pi
        dr = np.hypot(l_eta[pair_lep] - jet_eta[pair_jet], dphi)
        allowed = (dr < radius) & (l_electron[pair_lep] | has_muon[pair_jet])
        if allowed.any():
            cand_lep, cand_jet, cand_dr = pair_lep[allowed], pair_jet[allowed], dr[allowed]
            order = np.lexsort((cand_dr, cand_lep))  # by lepton, then by deltaR
            first = np.unique(cand_lep[order], return_index=True)[1]
            lep_i, jet_i = cand_lep[order][first], cand_jet[order][first]
            lx, ly, lz, le = _cartesian(l_pt[lep_i], l_eta[lep_i], l_phi[lep_i], 0.0)
            for component, lepton_part in ((px, lx), (py, ly), (pz, lz), (energy, le)):
                np.subtract.at(component, jet_i, lepton_part)
            is_muon = ~l_electron[lep_i]
            np.add.at(muon_energy, jet_i[is_muon], le[is_muon])
            subtracted[jet_i] = True

    pt = np.hypot(px, py)
    positive = subtracted & (energy > 0) & (pt > 0)
    new_eta = np.where(positive, np.arcsinh(pz / np.where(positive, pt, 1.0)), jet_eta)
    alive = positive & (np.abs(new_eta) < max_abs_eta)
    new_phi = np.where(alive, np.arctan2(py, px), jet_phi)
    new_mass = np.sqrt(np.maximum(energy * energy - (px * px + py * py + pz * pz), 0.0))
    return SubtractedJets(
        pt=np.where(subtracted, np.where(alive, pt, 0.0), jet_pt),
        eta=np.where(subtracted, new_eta, jet_eta),
        phi=np.where(subtracted, new_phi, jet_phi),
        mass=np.where(subtracted, np.where(alive, new_mass, 0.0), jet_mass),
        energy=np.where(subtracted, np.where(alive, energy, 0.0), stored_energy),
        stored_energy=stored_energy,
        subtracted=subtracted,
        muon_energy=muon_energy,
    )


def matched_gen_jets(gen_index, gen_jets):
    """Flat (pt, eta, phi) of the generator jet each jet points to (pt = -1 for no valid index)."""
    counts = ak.to_numpy(ak.num(gen_index))
    index = ak.to_numpy(ak.flatten(gen_index)).astype(np.int64)
    n_gen = ak.to_numpy(ak.num(gen_jets))
    event = np.repeat(np.arange(len(counts)), counts)
    valid = (index >= 0) & (index < n_gen[event])
    position = (np.cumsum(n_gen) - n_gen)[event] + np.where(valid, index, 0)
    out = []
    for field, default in (("pt", -1.0), ("eta", 0.0), ("phi", 0.0)):
        values = _flat(gen_jets[field])
        out.append(np.where(valid, values[np.minimum(position, max(len(values) - 1, 0))] if len(values) else default, default))
    return out


def _evaluate(correction, **inputs):
    """correction.evaluate with the arguments in the order of its declared inputs."""
    return correction.evaluate(*(inputs[i.name] for i in correction.inputs))


# --- corrections -----------------------------------------------------------


@dataclass
class CorrectedJets:
    """Configuration-independent per-jet factors of one chunk (flat arrays).

    The raw kinematics (ak4_raw_*, ak8_raw_*) are those after the lepton subtraction.
    """

    is_mc: bool
    ak4_counts: np.ndarray
    ak8_counts: np.ndarray
    ak4_raw_pt: np.ndarray
    ak4_raw_mass: np.ndarray
    ak4_eta: np.ndarray
    ak4_phi: np.ndarray
    ak4_jec: np.ndarray
    ak4_l1: np.ndarray
    ak4_raw_pt_nomu: np.ndarray
    ak4_type1: np.ndarray
    ak4_jer: np.ndarray  # (n_jets, 3): nom, up, down
    ak4_jer_matched: np.ndarray  # scaled with a matched generator jet
    ak8_raw_pt: np.ndarray
    ak8_raw_mass: np.ndarray
    ak8_eta: np.ndarray
    ak8_phi: np.ndarray
    ak8_raw_msd: np.ndarray
    ak8_jec: np.ndarray
    ak8_msd_factor: np.ndarray
    ak8_jer: np.ndarray
    ak8_jmr_random: np.ndarray
    raw_met_px: np.ndarray
    raw_met_py: np.ndarray
    rho: np.ndarray
    rho_ok: np.ndarray
    n_rho_consensus: int


class JetCorrector:
    def __init__(self, cfg: JetCorrectionConfig, rho_cfg: RhoRecoveryConfig):
        self.cfg = cfg
        ak4 = correctionlib.CorrectionSet.from_file(str(cfg.jec_ak4_file))
        ak8 = correctionlib.CorrectionSet.from_file(str(cfg.jec_ak8_file))
        self.jec_ak4 = ak4.compound[cfg.jec_ak4]
        self.jec_ak4_l1 = ak4[cfg.jec_ak4_l1]
        self.jec_ak8 = ak8.compound[cfg.jec_ak8]
        self.jec_ak8_l1 = ak8[cfg.jec_ak8_l1]
        self.jer_ak4 = self._jer_payload(cfg.jer_ak4_file, cfg.jer_ak4_collection)
        self.jer_ak8 = self._jer_payload(cfg.jer_ak8_file, cfg.jer_ak8_collection)
        self.jer_smear = correctionlib.CorrectionSet.from_file(str(cfg.jer_smear_file))["JERSmear"]
        self.rho_recovery = RhoRecovery(self.jec_ak4, rho_cfg)

        results = json.loads(cfg.jms_jmr_file.read_text())
        self.jms, self.jms_err = results["JMS"], results["JMS_err"]
        self.jmr, self.jmr_err = results["JMR"], results["JMR_err"]
        mc_peak = results["pass"]["mc"]
        self.jmr_rel_resolution = mc_peak["sigma"] / mc_peak["mu"]

    def _jer_payload(self, path, collection):
        cs = correctionlib.CorrectionSet.from_file(str(path))
        tag = self.cfg.jer_tag
        return tuple(cs[f"{tag}_{part}_{collection}"] for part in ("PtResolution", "ScaleFactor", "SFUncertainty"))

    def _jer_factors(self, payload, radius, pt, eta, phi, rho, seed, gen_pt, gen_eta, gen_phi):
        """(n, 3) hybrid JER factors (nom, up, down) and the generator-matched flag.

        JERC: scaling with the generator jet if dR < R/2 and |pT - pT_gen| < 3 sigma_JER pT
        (pT after the nominal JEC), stochastic smearing otherwise; the same random draw for the
        nominal and the up/down scale factors (JERSmear, hashprng of JetPt, JetEta, Rho, EventID).
        """
        resolution_c, sf_c, unc_c = payload
        resolution = _evaluate(resolution_c, JetEta=eta, JetPt=pt, Rho=rho)
        dr = np.hypot(eta - gen_eta, (phi - gen_phi + np.pi) % (2 * np.pi) - np.pi)
        matched = (gen_pt >= 0) & (dr < radius / 2) & (np.abs(pt - gen_pt) < self.cfg.jer_match_max_sigma * resolution * pt)
        gen = np.where(matched, gen_pt, -1.0)
        sf = _evaluate(sf_c, JetEta=eta, JetPt=pt)
        unc = _evaluate(unc_c, JetEta=eta, JetPt=pt)
        out = np.ones((len(pt), 3))
        for k, scale in enumerate((sf, sf * (1.0 + unc), sf * (1.0 - unc))):
            out[:, k] = _evaluate(self.jer_smear, JetPt=pt, JetEta=eta, GenPt=gen, Rho=rho, EventID=seed,
                                  JER=resolution, JERSF=scale)
        return out, matched

    def prepare(self, ak4, ak8, met, event, is_mc, leptons, rho=None, gen_jets=None):
        """Evaluate everything the jet configurations share.

        ak4/ak8 are the (truncated, raw-pT-filtered) input collections in stored order; met the
        ScoutingMET record; event the event number; leptons the identified leptons subtracted from
        the jets (objects.identified_leptons); gen_jets the GenJet collection (MC) the AK4
        genJetIdx points into. The productions store no AK8 generator jets, so every AK8 jet is
        smeared stochastically.
        """
        n_events = len(event)
        c4 = ak.to_numpy(ak.num(ak4))
        c8 = ak.to_numpy(ak.num(ak8))
        e4, e8 = _event_index(c4), _event_index(c8)

        # AK4: stored with the production JEC, raw = stored * (1 - rawFactor).
        raw_scale = 1.0 - _flat(ak4.rawFactor)
        stored = 1.0 / raw_scale
        a4_eta, a4_phi, a4_area = _flat(ak4.eta), _flat(ak4.phi), _flat(ak4.area)
        a4_raw = _flat(ak4.pt) * raw_scale
        if rho is None:
            # Recovered from the production JEC of the stored (unsubtracted) jets.
            has_jets = (c4 + c8) > 0
            rho, rho_ok, n_cons = self.rho_recovery(a4_area, a4_eta, a4_phi, a4_raw, stored, e4, n_events, has_jets)
        else:
            rho = ak.to_numpy(rho).astype(np.float64)
            rho_ok, n_cons = np.ones(n_events, dtype=bool), 0
        rho4, rho8 = rho[e4], rho[e8]

        sub4 = subtract_leptons(a4_raw, a4_eta, a4_phi, _flat(ak4.mass) * raw_scale, c4,
                                ak.to_numpy(ak.flatten(ak4.muMultiplicity)) > 0, leptons, self.cfg.ak4_radius,
                                self.cfg.jme_max_abs_eta)
        good4 = sub4.pt > 0
        # The production JEC for unchanged jets; re-evaluated on the remainder of a subtracted jet.
        jec4 = np.where(good4, stored, 1.0)
        redo = good4 & sub4.subtracted
        if redo.any():
            jec4[redo] = self.jec_ak4.evaluate(a4_area[redo], sub4.eta[redo], sub4.phi[redo], sub4.pt[redo], rho4[redo])
        l1_4 = np.ones(len(sub4.pt))
        if good4.any():
            l1_4[good4] = self.jec_ak4_l1.evaluate(a4_area[good4], sub4.eta[good4], sub4.phi[good4], sub4.pt[good4], rho4[good4])
        # Type-1: muon-subtracted raw pT of the remainder (the muon energy left after the lepton
        # subtraction); the EM-fraction requirement on the stored fractions.
        mu_ef = _flat(ak4.muEF)
        remaining_mu = np.maximum(mu_ef * sub4.stored_energy - sub4.muon_energy, 0.0)
        mu_fraction = np.where(sub4.subtracted,
                               np.minimum(remaining_mu / np.where(sub4.energy > 0, sub4.energy, 1.0), 1.0), mu_ef)
        nomu = sub4.pt * (1.0 - mu_fraction)
        emf = _flat(ak4.chEmEF) + _flat(ak4.neEmEF)
        type1 = good4 & (nomu * jec4 > self.cfg.type1_min_pt) & (emf < self.cfg.type1_max_em_fraction)

        # AK8: stored raw, JEC from raw.
        a8_eta, a8_phi, a8_area = _flat(ak8.eta), _flat(ak8.phi), _flat(ak8.area)
        sub8 = subtract_leptons(_flat(ak8.pt), a8_eta, a8_phi, _flat(ak8.mass), c8,
                                ak.to_numpy(ak.flatten(ak8.nMuons)) > 0, leptons, self.cfg.ak8_radius,
                                self.cfg.jme_max_abs_eta)
        a8_raw_msd = _flat(ak8.msoftdrop)  # no soft-drop subjets are stored: not lepton-subtracted
        good8 = sub8.pt > 0
        jec8 = np.ones(len(sub8.pt))
        l1_8 = np.ones(len(sub8.pt))
        if good8.any():
            jec8[good8] = self.jec_ak8.evaluate(a8_area[good8], sub8.eta[good8], sub8.phi[good8], sub8.pt[good8], rho8[good8])
            l1_8[good8] = self.jec_ak8_l1.evaluate(a8_area[good8], sub8.eta[good8], sub8.phi[good8], sub8.pt[good8], rho8[good8])

        jer4 = np.ones((len(sub4.pt), 3))
        matched4 = np.zeros(len(sub4.pt), dtype=bool)
        jer8 = np.ones((len(sub8.pt), 3))
        jmr_random = np.zeros(len(sub8.pt))
        if is_mc:
            event_np = ak.to_numpy(event).astype(np.uint64)
            # JERSmear's EventID input is an int (entropy of its hash only).
            seed = (event_np & np.uint64(0x7FFFFFFF)).astype(np.int64)
            if good4.any():
                gen_pt, gen_eta, gen_phi = matched_gen_jets(ak4.genJetIdx, gen_jets)
                g = good4
                jer4[g], matched4[g] = self._jer_factors(
                    self.jer_ak4, self.cfg.ak4_radius, (sub4.pt * jec4)[g], sub4.eta[g], sub4.phi[g], rho4[g],
                    seed[e4][g], gen_pt[g], gen_eta[g], gen_phi[g])
            if good8.any():
                g = good8
                n = int(g.sum())
                jer8[g], _ = self._jer_factors(
                    self.jer_ak8, self.cfg.ak8_radius, (sub8.pt * jec8)[g], sub8.eta[g], sub8.phi[g], rho8[g],
                    seed[e8][g], np.full(n, -1.0), np.zeros(n), np.zeros(n))
            jmr_random = np.where(
                good8,
                jet_gaussian(event_np[e8], ak.flatten(ak8.pt), ak.flatten(ak8.eta), ak.flatten(ak8.phi)),
                0.0,
            )

        met_pt = ak.to_numpy(met.pt).astype(np.float64)
        met_phi = ak.to_numpy(met.phi).astype(np.float64)
        return CorrectedJets(
            is_mc=is_mc,
            ak4_counts=c4,
            ak8_counts=c8,
            ak4_raw_pt=sub4.pt,
            ak4_raw_mass=sub4.mass,
            ak4_eta=sub4.eta,
            ak4_phi=sub4.phi,
            ak4_jec=jec4,
            ak4_l1=l1_4,
            ak4_raw_pt_nomu=nomu,
            ak4_type1=type1,
            ak4_jer=jer4,
            ak4_jer_matched=matched4,
            ak8_raw_pt=sub8.pt,
            ak8_raw_mass=sub8.mass,
            ak8_eta=sub8.eta,
            ak8_phi=sub8.phi,
            ak8_raw_msd=a8_raw_msd,
            ak8_jec=jec8,
            ak8_msd_factor=np.where(good8, jec8 / l1_8, 1.0),
            ak8_jer=jer8,
            ak8_jmr_random=jmr_random,
            raw_met_px=met_pt * np.cos(met_phi),
            raw_met_py=met_pt * np.sin(met_phi),
            rho=rho,
            rho_ok=rho_ok,
            n_rho_consensus=n_cons,
        )

    def configuration(self, name):
        """(jer index, jes factor, jms, jmr) of one jet configuration."""
        jer_index, jes, jms, jmr = 0, 1.0, self.jms, self.jmr
        if name == "jes_up":
            jes = 1.0 + self.cfg.jes_shift
        elif name == "jes_down":
            jes = 1.0 - self.cfg.jes_shift
        elif name == "jer_up":
            jer_index = 1
        elif name == "jer_down":
            jer_index = 2
        elif name == "jms_up":
            jms = self.jms + self.jms_err
        elif name == "jms_down":
            jms = self.jms - self.jms_err
        elif name == "jmr_up":
            jmr = self.jmr + self.jmr_err
        elif name == "jmr_down":
            jmr = self.jmr - self.jmr_err
        elif name != "nominal":
            raise ValueError(f"unknown jet configuration {name!r}")
        return jer_index, jes, jms, jmr

    def apply(self, state: CorrectedJets, name="nominal"):
        """Corrected jet kinematics and MET of one configuration.

        Returns a dict of jagged float32 arrays (ak4_pt, ak4_mass, ak4_eta, ak4_phi, ak8_pt,
        ak8_mass, ak8_eta, ak8_phi, ak8_msoftdrop; the axes of lepton-subtracted jets move) and flat
        met_pt, met_phi.
        """
        jer_index, jes, jms, jmr = self.configuration(name)
        if state.is_mc:
            var4 = state.ak4_jer[:, jer_index] * jes
            var8 = state.ak8_jer[:, jer_index] * jes
        else:
            var4 = np.ones_like(state.ak4_raw_pt)
            var8 = np.ones_like(state.ak8_raw_pt)

        factor4 = state.ak4_jec * var4
        shift = np.where(state.ak4_type1, state.ak4_raw_pt_nomu * (factor4 - state.ak4_l1), 0.0)
        n_events = len(state.rho)
        e4 = _event_index(state.ak4_counts)
        type1_px = np.bincount(e4, weights=shift * np.cos(state.ak4_phi), minlength=n_events)
        type1_py = np.bincount(e4, weights=shift * np.sin(state.ak4_phi), minlength=n_events)
        met_px = state.raw_met_px - type1_px
        met_py = state.raw_met_py - type1_py

        msd_scale = state.ak8_msd_factor * var8
        if state.is_mc and self.cfg.apply_jms_jmr:
            width = np.sqrt(max(jmr * jmr - 1.0, 0.0)) * self.jmr_rel_resolution
            # Same association as legacy (scale *= jms * (1 + r * width)) for bit-identical floats.
            msd_scale = msd_scale * (jms * (1.0 + state.ak8_jmr_random * width))

        def jag(values, counts):
            return ak.unflatten(values.astype(np.float32), counts)

        return {
            "ak4_pt": jag(state.ak4_raw_pt * factor4, state.ak4_counts),
            "ak4_mass": jag(state.ak4_raw_mass * factor4, state.ak4_counts),
            "ak4_eta": jag(state.ak4_eta, state.ak4_counts),
            "ak4_phi": jag(state.ak4_phi, state.ak4_counts),
            "ak8_pt": jag(state.ak8_raw_pt * state.ak8_jec * var8, state.ak8_counts),
            "ak8_mass": jag(state.ak8_raw_mass * state.ak8_jec * var8, state.ak8_counts),
            "ak8_eta": jag(state.ak8_eta, state.ak8_counts),
            "ak8_phi": jag(state.ak8_phi, state.ak8_counts),
            "ak8_msoftdrop": jag(state.ak8_raw_msd * msd_scale, state.ak8_counts),
            "met_pt": np.hypot(met_px, met_py).astype(np.float32),
            "met_phi": np.arctan2(met_py, met_px).astype(np.float32),
        }
