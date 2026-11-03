"""Jet energy corrections, rho recovery, JER, JMS/JMR and Type-1 MET.

Port of JetPtCorrector in the legacy convert_branch.C. Nominal chain:

- AK4 (ScoutingPFJetRecluster2) are stored with the production
  Winter24HLT_V1 L1L2L3Res AK4PFHLT JEC; raw = stored * (1 - rawFactor).
- The event rho is not stored: it is recovered by inverting the AK4 JEC.
- AK8 (ScoutingFatPFJetRecluster) are stored raw and get the AK8PFHLT JEC.
- MC jets are smeared with the AK4PFPuppi JER proxy (JERSmear, GenPt = -1).
- The AK8 soft-drop mass takes every pT factor except L1FastJet, and for MC
  the JMS scale and JMR smearing.
- MET is Type-1 corrected with the AK4 jets.

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
        inner = np.clip(rho, 0.0, rmax)
        reflected = np.where(above, 2 * rmax - rho, np.where(below, -rho, inner))
        value = self._jec(area, eta, phi, raw_pt, reflected)
        edge = np.where(above, rmax, 0.0)
        edge_value = self._jec(area, eta, phi, raw_pt, edge)
        return np.where(above | below, 2.0 * edge_value - value, value)

    def _carries(self, area, eta, phi, raw_pt, stored):
        low = self._jec(area, eta, phi, raw_pt, np.zeros_like(raw_pt))
        high = self._jec(area, eta, phi, raw_pt, np.full_like(raw_pt, self.cfg.rho_max))
        return low - high > self.cfg.min_sensitivity * stored

    def _solve(self, area, eta, phi, raw_pt, stored):
        """Illinois regula falsi on [-margin, rho_max + margin]; returns (rho, ok)."""
        c = self.cfg

        def residual(r):
            return self._extended(area, eta, phi, raw_pt, r) - stored

        low = np.full_like(raw_pt, -c.bracket_margin)
        high = np.full_like(raw_pt, c.rho_max + c.bracket_margin)
        f_low = residual(low)
        f_high = residual(high)
        ok = (f_low >= 0) & (f_high <= 0) & (f_low > f_high)
        rho = np.zeros_like(raw_pt)
        side = np.zeros(raw_pt.shape, dtype=np.int8)
        active = ok.copy()
        for _ in range(c.solver_iterations):
            if not active.any():
                break
            with np.errstate(invalid="ignore", divide="ignore"):
                guess = (low * f_high - high * f_low) / (f_high - f_low)
            rho = np.where(active, guess, rho)
            f = residual(np.where(active, rho, 0.0))
            done = active & (np.abs(f) <= 1e-10)
            step = active & ~done
            pos = step & (f > 0)
            neg = step & ~(f > 0)
            f_high = np.where(pos & (side == 1), 0.5 * f_high, f_high)
            low = np.where(pos, rho, low)
            f_low = np.where(pos, f, f_low)
            f_low = np.where(neg & (side == -1), 0.5 * f_low, f_low)
            high = np.where(neg, rho, high)
            f_high = np.where(neg, f, f_high)
            side = np.where(pos, 1, np.where(neg, -1, side)).astype(np.int8)
            active = step
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

        # Rare events: per-event consensus over neighbouring JEC bins.
        fallback = np.flatnonzero(has_jets & ~good)
        for ev in fallback:
            sel = evt == ev
            value = self._consensus(a[sel], e[sel], p[sel], r[sel], s[sel])
            if value is not None:
                rho[ev] = value
                ok[ev] = True
        return rho, ok, len(fallback)

    def _consensus(self, a, e, p, r, s):
        c = self.cfg
        d = c.bin_edge_probe
        axis = [(0.0, 0.0), (d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)]
        diagonal = [(de, dp) for de in (d, -d) for dp in (d, -d)]
        for shifts in (axis, axis + diagonal):
            value = self._from_shifted_bins(a, e, p, r, s, shifts)
            if value is not None:
                return value
        return None

    def _from_shifted_bins(self, a, e, p, r, s, shifts):
        probe = np.full(1, self.cfg.bin_probe_rho)
        solutions = []
        for i in range(len(r)):
            one = lambda x: np.array([x], dtype=np.float64)  # noqa: E731
            own_bin = self._jec(one(a[i]), one(e[i]), one(p[i]), one(r[i]), probe)[0]
            jet_solutions = []
            carries = False
            for de, dp in shifts:
                eta, phi = one(e[i] + de), one(p[i] + dp)
                own = de == 0.0 and dp == 0.0
                if not own and self._jec(one(a[i]), eta, phi, one(r[i]), probe)[0] == own_bin:
                    continue
                if not self._carries(one(a[i]), eta, phi, one(r[i]), one(s[i]))[0]:
                    continue
                carries = True
                value, ok = self._solve(one(a[i]), eta, phi, one(r[i]), one(s[i]))
                if ok[0]:
                    jet_solutions.append(value[0])
            if not carries:
                continue
            if not jet_solutions:
                return None
            solutions.append(jet_solutions)
        if not solutions:
            return None
        for anchor_solutions in solutions:
            for anchor in anchor_solutions:
                chosen = [min(js, key=lambda v: abs(v - anchor)) for js in solutions]
                median = float(np.median(chosen))
                if all(abs(v - median) <= self.cfg.tolerance for v in chosen):
                    return median
        return None


# --- corrections -----------------------------------------------------------


@dataclass
class CorrectedJets:
    """Configuration-independent per-jet factors of one chunk (flat arrays)."""

    is_mc: bool
    ak4_counts: np.ndarray
    ak8_counts: np.ndarray
    ak4_raw_pt: np.ndarray
    ak4_raw_mass: np.ndarray
    ak4_phi: np.ndarray
    ak4_jec: np.ndarray
    ak4_l1: np.ndarray
    ak4_raw_pt_nomu: np.ndarray
    ak4_type1: np.ndarray
    ak4_jer: np.ndarray  # (n_jets, 3): nom, up, down
    ak8_raw_pt: np.ndarray
    ak8_raw_mass: np.ndarray
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
        jer = correctionlib.CorrectionSet.from_file(str(cfg.jer_file))
        self.jer_resolution = jer[cfg.jer_resolution]
        self.jer_sf = jer[cfg.jer_scale_factor]
        self.jer_smear = correctionlib.CorrectionSet.from_file(str(cfg.jer_smear_file))["JERSmear"]
        self.rho_recovery = RhoRecovery(self.jec_ak4, rho_cfg)

        results = json.loads(cfg.jms_jmr_file.read_text())
        self.jms, self.jms_err = results["JMS"], results["JMS_err"]
        self.jmr, self.jmr_err = results["JMR"], results["JMR_err"]
        mc_peak = results["pass"]["mc"]
        self.jmr_rel_resolution = mc_peak["sigma"] / mc_peak["mu"]

    def _jer_factors(self, pt, eta, rho, seed):
        jer_rho = np.minimum(rho, np.nextafter(self.cfg.jer_rho_upper_edge, 0.0))
        resolution = self.jer_resolution.evaluate(eta, pt, jer_rho)
        out = np.ones((len(pt), 3))
        for k, syst in enumerate(("nom", "up", "down")):
            sf = self.jer_sf.evaluate(eta, pt, syst)
            out[:, k] = self.jer_smear.evaluate(pt, eta, np.full_like(pt, -1.0), rho, seed, resolution, sf)
        return out

    def prepare(self, ak4, ak8, met, event, is_mc, rho=None):
        """Evaluate everything the jet configurations share.

        ak4/ak8 are the (truncated, raw-pT-filtered) input collections in stored
        order; met the ScoutingMET record; event the event number.
        """
        n_events = len(event)
        c4 = ak.to_numpy(ak.num(ak4))
        c8 = ak.to_numpy(ak.num(ak8))
        e4, e8 = _event_index(c4), _event_index(c8)

        raw_scale = 1.0 - _flat(ak4.rawFactor)
        pt4 = _flat(ak4.pt)
        a4_raw = pt4 * raw_scale
        a4_raw_mass = _flat(ak4.mass) * raw_scale
        a4_eta, a4_phi, a4_area = _flat(ak4.eta), _flat(ak4.phi), _flat(ak4.area)
        stored = 1.0 / raw_scale

        if rho is None:
            has_jets = (c4 + c8) > 0
            rho, rho_ok, n_cons = self.rho_recovery(
                a4_area, a4_eta, a4_phi, a4_raw, stored, e4, n_events, has_jets
            )
        else:
            rho = ak.to_numpy(rho).astype(np.float64)
            rho_ok, n_cons = np.ones(n_events, dtype=bool), 0
        rho4, rho8 = rho[e4], rho[e8]

        good4 = a4_raw > 0
        l1_4 = np.where(good4, self.jec_ak4_l1.evaluate(a4_area, a4_eta, a4_phi, a4_raw, rho4), 1.0)
        jec4 = np.where(good4, stored, 1.0)
        nomu = a4_raw * (1.0 - _flat(ak4.muEF))
        emf = _flat(ak4.chEmEF) + _flat(ak4.neEmEF)
        type1 = good4 & (nomu * jec4 > self.cfg.type1_min_pt) & (emf < self.cfg.type1_max_em_fraction)

        a8_raw, a8_raw_mass, a8_raw_msd = _flat(ak8.pt), _flat(ak8.mass), _flat(ak8.msoftdrop)
        a8_eta, a8_phi, a8_area = _flat(ak8.eta), _flat(ak8.phi), _flat(ak8.area)
        good8 = a8_raw > 0
        jec8 = np.where(good8, self.jec_ak8.evaluate(a8_area, a8_eta, a8_phi, a8_raw, rho8), 1.0)
        l1_8 = np.where(good8, self.jec_ak8_l1.evaluate(a8_area, a8_eta, a8_phi, a8_raw, rho8), 1.0)

        jer4 = np.ones((len(a4_raw), 3))
        jer8 = np.ones((len(a8_raw), 3))
        jmr_random = np.zeros(len(a8_raw))
        if is_mc:
            event_np = ak.to_numpy(event).astype(np.uint64)
            # JERSmear's EventID input is an int (entropy of its hash only).
            seed = (event_np & np.uint64(0x7FFFFFFF)).astype(np.int64)
            if good4.any():
                jer4[good4] = self._jer_factors(
                    (a4_raw * jec4)[good4], a4_eta[good4], rho4[good4], seed[e4][good4]
                )
            if good8.any():
                jer8[good8] = self._jer_factors(
                    (a8_raw * jec8)[good8], a8_eta[good8], rho8[good8], seed[e8][good8]
                )
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
            ak4_raw_pt=a4_raw,
            ak4_raw_mass=a4_raw_mass,
            ak4_phi=a4_phi,
            ak4_jec=jec4,
            ak4_l1=l1_4,
            ak4_raw_pt_nomu=nomu,
            ak4_type1=type1,
            ak4_jer=jer4,
            ak8_raw_pt=a8_raw,
            ak8_raw_mass=a8_raw_mass,
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

        Returns a dict of jagged float32 arrays (ak4_pt, ak4_mass, ak8_pt,
        ak8_mass, ak8_msoftdrop) and flat met_pt, met_phi.
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
            "ak8_pt": jag(state.ak8_raw_pt * state.ak8_jec * var8, state.ak8_counts),
            "ak8_mass": jag(state.ak8_raw_mass * state.ak8_jec * var8, state.ak8_counts),
            "ak8_msoftdrop": jag(state.ak8_raw_msd * msd_scale, state.ak8_counts),
            "met_pt": np.hypot(met_px, met_py).astype(np.float32),
            "met_phi": np.arctan2(met_py, met_px).astype(np.float32),
        }
