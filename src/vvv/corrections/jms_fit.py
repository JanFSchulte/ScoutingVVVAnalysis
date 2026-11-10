"""JMS/JMR tag-and-probe fit of the hadronic-W peak in the ttbar CR.

Port of the legacy selections/jms_jmr/jms_jmr_fit.py. The probe is the
leading AK8 jet opposite the muon (processors/ttbar_cr.py). Events passing the
fit selection are split by the probe W tag (pass: tag_branch > tag_wp, fail:
the rest), and the probe soft-drop mass in the pass region is fitted, for data
and MC, with a Gaussian on an exponential in the fit window:

    f(m) = A exp(-(m - mu)^2 / 2 sigma^2) + B exp(-C (m - m0)),  m0 = window low edge

    JMS = mu_data / mu_MC,  JMR = sigma_data / sigma_MC

The fit (starting values, bounds, chi2 with bin errors, absolute_sigma) and the
results JSON are those of legacy; the JSON is what corrections/jets.py reads. The peak
position's starting value and bounds default to legacy's (80, [65, 100] GeV) and can be set
with mu_start / mu_bounds in the config: the JEC applied to the groomed mass moves the W
peak to ~98 GeV, outside legacy's window.
"""

import numpy as np
from scipy.optimize import curve_fit

from vvv.ml.inference import threshold_mask


def background(m, B, C, m0):
    return B * np.exp(-C * (m - m0))


def model(m, A, mu, sigma, B, C, m0):
    return A * np.exp(-0.5 * ((m - mu) / sigma) ** 2) + background(m, B, C, m0)


def edges(cfg):
    lo, hi = cfg["fit_window"]
    return np.linspace(lo, hi, int(cfg["n_bins"]) + 1)


def fit_peak(centers, counts, errors, window, label, mu_start=80.0, mu_bounds=(65.0, 100.0)):
    """Legacy _fit_peak: m0 is fixed (bounds of +-1e-6) at the window low edge; empty bins get error 1."""
    m0 = window[0]
    peak = max(counts.max(), 1.0)
    p0 = [peak, mu_start, 9.0, max(counts[0], 1.0), 0.01, m0]
    lo = [0.0, mu_bounds[0], 3.0, 0.0, -0.05, m0 - 1e-6]
    hi = [10 * peak + 10, mu_bounds[1], 25.0, 1e9, 0.2, m0 + 1e-6]
    sig = np.where(errors > 0, errors, 1.0)
    popt, pcov = curve_fit(model, centers, counts, p0=p0, sigma=sig, absolute_sigma=True, bounds=(lo, hi), maxfev=300000)
    perr = np.sqrt(np.diag(pcov))
    chi2 = float(np.sum(((counts - model(centers, *popt)) / sig) ** 2))
    return {
        "mu": float(popt[1]),
        "mu_err": float(perr[1]),
        "sigma": float(abs(popt[2])),
        "sigma_err": float(perr[2]),
        "B": float(popt[3]),
        "C": float(popt[4]),
        "m0": m0,
        "chi2": chi2,
        "ndf": max(len(centers) - 5, 1),
        "popt": [float(x) for x in popt],
        "label": label,
    }


class PassFail:
    """Pass/fail histograms (sum w, sum w^2) of the probe mass in the fit window."""

    def __init__(self, n_bins):
        self.sumw = {"pass": np.zeros(n_bins), "fail": np.zeros(n_bins)}
        self.sumw2 = {"pass": np.zeros(n_bins), "fail": np.zeros(n_bins)}

    def add_events(self, columns, weight, cfg):
        """Events of a CR skim (or legacy tree): fit selection, tag split, np.histogram binning as legacy."""
        keep = threshold_mask(columns, cfg["selection"])
        mass = np.asarray(columns[cfg["mass_branch"]])[keep]
        passed = np.asarray(columns[cfg["tag_branch"]])[keep] > float(cfg["tag_wp"])
        w = np.asarray(weight, dtype=np.float64)[keep]
        bins = edges(cfg)
        for tag, sel in (("pass", passed), ("fail", ~passed)):
            self.sumw[tag] += np.histogram(mass[sel], bins=bins, weights=w[sel])[0]
            self.sumw2[tag] += np.histogram(mass[sel], bins=bins, weights=w[sel] ** 2)[0]

    def add_hist(self, h):
        """A (tag, mass) slice of the processor's fit histogram (or of a control histogram, already
        cut to the fit window)."""
        for tag in ("pass", "fail"):
            view = h[{"tag": tag}].view(flow=False)
            self.sumw[tag] += view.value
            self.sumw2[tag] += view.variance

    def counts(self, tag):
        return self.sumw[tag], np.sqrt(self.sumw2[tag])


def measure(data: PassFail, mc: PassFail, cfg, lumi_fb):
    """Fit both pass regions; the results record in the legacy jms_jmr_results.json format."""
    bins = edges(cfg)
    centers = 0.5 * (bins[:-1] + bins[1:])
    window = tuple(cfg["fit_window"])
    peak = {"mu_start": float(cfg.get("mu_start", 80.0)), "mu_bounds": tuple(cfg.get("mu_bounds", (65.0, 100.0)))}
    data_fit = fit_peak(centers, *data.counts("pass"), window, "Data (Pass)", **peak)
    mc_fit = fit_peak(centers, *mc.counts("pass"), window, "MC (Pass)", **peak)
    md, sd, md_e, sd_e = data_fit["mu"], data_fit["sigma"], data_fit["mu_err"], data_fit["sigma_err"]
    mm, sm, mm_e, sm_e = mc_fit["mu"], mc_fit["sigma"], mc_fit["mu_err"], mc_fit["sigma_err"]
    jms = md / mm
    jmr = sd / sm
    return {
        "lumi_fb": lumi_fb,
        "tag_branch": cfg["tag_branch"],
        "tag_wp": float(cfg["tag_wp"]),
        "fit_window": list(window),
        "n_bins": int(cfg["n_bins"]),
        "w_mass_ref": cfg["w_mass_ref"],
        "pass": {"data": data_fit, "mc": mc_fit},
        "yields": {
            "data_pass": float(data.sumw["pass"].sum()),
            "mc_pass": float(mc.sumw["pass"].sum()),
            "data_fail": float(data.sumw["fail"].sum()),
            "mc_fail": float(mc.sumw["fail"].sum()),
        },
        "JMS": jms,
        "JMS_err": jms * np.hypot(md_e / md, mm_e / mm),
        "JMR": jmr,
        "JMR_err": jmr * np.hypot(sd_e / sd, sm_e / sm),
        "selection": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg["selection"].items()},
    }
