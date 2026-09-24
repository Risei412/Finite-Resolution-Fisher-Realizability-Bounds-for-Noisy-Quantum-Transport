#!/usr/bin/env python3
"""N-vs-significance driver for the finite-resolution witness (order 3, K=3).

Turns the finite_sample.py machinery into a one-sided claim: given N events,
how confidently does the data alone certify that no order-3 classical renewal
process, matched on (m1, m2, m3), can reach the observed fine/coarse Fisher
ratio.

Two honest limitations, stated once here rather than buried in a caveat list:

  1. I_coarse is NOT estimated from data. The variational chi2 estimator
     recovers at most ~72% of the theoretical I_coarse and gets WORSE as the
     probe delta shrinks (the coarse detector damps the oscillation that
     carries the theta-signal), and underestimating I_coarse is the unsafe
     direction for a one-sided certificate (it sits in the denominator). Until
     a validated finite-sample estimator for I_coarse exists, this driver uses
     its theoretical (quadrature) value, i.e. assumes I_coarse is known from
     independent calibration of the detector response.

  2. The classical bound rho_relaxed(3, K=3) is evaluated by resolving weights
     at a FIXED pole layout (the best Coxian(3) optimum found by
     ph_classes.rho_direct, rho ~= 28.05) rather than re-running the expensive
     multistart pole search at every point of the moment confidence region.
     This is a fast local proxy for the worst-case bound, not a re-certified
     supremum at each point; it is adequate for a first N-vs-significance
     estimate but should be replaced by the full search before submission.

Writes results/finite_sample.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import finite_sample as fs  # noqa: E402
import ph_classes as pc  # noqa: E402
import transport_native_perturbation_gate as frozen  # noqa: E402

X0 = np.array([1.0, 0.0, 0.0])
SURVIVAL = np.array([1.0, 1.0, 0.0])
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "finite_sample.json"

# best Coxian(3) pole layout located for (m1, m2, m3) at gamma_phi = 0
FIXED_LAMS = np.array([0.058169813441852736, 4.7943110386821175, 0.04922016250902127])
FIXED_RATES = FIXED_LAMS.astype(complex)


def rho_at_fixed_poles(m1, m2, m3, tau_fine, tau_coarse):
    """rho for the exponential-sum density with FIXED_RATES matching (m1,m2,m3).

    Re-solves only the 3 weights (a linear system), so this is cheap enough to
    grid-search a confidence region; see the module docstring for what this
    approximates.
    """
    coeffs, resid = pc.coeffs_from_moments(FIXED_RATES, np.array([m1, m2, m3]))
    if resid > 1e-6:
        return np.nan
    if pc.density_minimum(FIXED_RATES, coeffs) < -1e-10:
        return np.nan
    rho = pc.rho_of_modes(FIXED_RATES, coeffs, tau_fine, tau_coarse, grid=(16, 8, 6))
    return rho if rho is not None else np.nan


def one_run(gamma_phi, n_events, delta, rng, n_boot_moments=200, n_boot_chi2=30, alpha=0.1):
    a, _ = frozen.quantum_generator(frozen.THETA, gamma_phi)
    a_delta, _ = frozen.quantum_generator(frozen.THETA + delta, gamma_phi)
    fine_theory, _, _ = frozen.quantum_fisher(frozen.THETA, gamma_phi, frozen.TAU_FINE)
    coarse_theory, _, _ = frozen.quantum_fisher(frozen.THETA, gamma_phi, frozen.TAU_COARSE)

    t_true = fs.sample_true_intervals(a, X0, SURVIVAL, n_events, rng)
    t_fine = fs.observed_intervals(t_true, frozen.TAU_FINE, rng)
    t_true_delta = fs.sample_true_intervals(a_delta, X0, SURVIVAL, n_events, rng)
    t_fine_delta = fs.observed_intervals(t_true_delta, frozen.TAU_FINE, rng)

    # I_fine: data-driven lower bound via the variational chi2 surrogate.
    #
    # A K=42 basis (n_decay=6, n_harmonics=3) recovers ~100% of the
    # theoretical value on average once N is large, but is numerically
    # unstable at small-to-moderate N -- it swung to -144x the true value at
    # N=5000 in testing, an overfitting failure the sample-splitting bias
    # correction does not fully absorb. A K=20 basis (n_decay=4,
    # n_harmonics=2) is stable across the whole N ladder tested here,
    # settling at ~53-57% of the theoretical value with small variance. Since
    # I_fine is the numerator and this estimator is a legitimate lower bound,
    # trading efficiency for stability is the conservative and correct choice
    # for a one-sided certificate.
    lam, _ = fs.survival_modes(a, X0, SURVIVAL)
    omega = np.abs(np.imag(lam)).max()
    basis = fs.default_basis(frozen.TAU_FINE, 60.0, omega, n_decay=4, n_harmonics=2)
    surrogate = fs.fisher_surrogate(t_fine, t_fine_delta, delta, basis, rng, n_boot=n_boot_chi2)
    i_fine_hat = surrogate["fisher_surrogate"]
    i_fine_lo = i_fine_hat - 1.645 * surrogate.get("bootstrap_std", 0.0)  # one-sided 95%
    i_fine_lo = max(i_fine_lo, 0.0)  # a Fisher information cannot be negative;
    # a negative surrogate signals estimator breakdown, not a real value

    # moments of the UNDETECTED process, deconvolved from the observed fine record.
    # The deconvolution mixes components (m_k_true depends on m_1_obs..m_k_obs
    # jointly), so the CI is built by deconvolving each bootstrap replicate and
    # then taking percentiles -- not by deconvolving the already-percentiled
    # observed-moment endpoints, which would not track the joint uncertainty.
    m_obs = fs.moment_estimates(t_fine, 3)
    m_hat = fs.deconvolve_moments(m_obs, frozen.TAU_FINE, 3)
    m_lo, m_hi, _ = fs.bootstrap_moment_ci(
        t_fine, 3, n_boot_moments, rng, alpha=alpha,
        transform=lambda raw: fs.deconvolve_moments(raw, frozen.TAU_FINE, 3))

    def rho_fn(m1, m2, m3):
        return rho_at_fixed_poles(m1, m2, m3, frozen.TAU_FINE, frozen.TAU_COARSE)

    rho_point = rho_fn(*m_hat)
    rho_worst = fs.certified_bound(rho_fn, m_hat, m_lo, m_hi, n_grid=5)

    # rho_worst can be NaN if no point in or near the CI was feasible for the
    # fixed pole layout; that is a failure to certify, not a bound of zero
    if rho_worst != rho_worst:
        b_n_certified = np.nan
        violation_certified = np.nan
    else:
        b_n_certified = rho_worst * coarse_theory  # I_coarse held at its theoretical value
        violation_certified = i_fine_lo / b_n_certified if b_n_certified > 0 else np.nan

    return {
        "gamma_phi": gamma_phi, "n_events": n_events, "delta": delta,
        "theory": {"I_fine": fine_theory, "I_coarse": coarse_theory},
        "data_driven": {
            "I_fine_hat": i_fine_hat,
            "I_fine_lower_95": i_fine_lo,
            "I_fine_bootstrap_std": surrogate.get("bootstrap_std"),
            "m_hat": m_hat.tolist(), "m_ci_lo": m_lo.tolist(), "m_ci_hi": m_hi.tolist(),
            "rho_point_estimate": rho_point,
            "rho_worst_case": rho_worst,
        },
        "B_n_certified": b_n_certified,
        "violation_certified_lower_95": violation_certified,
        "certifies_at_95": bool(violation_certified > 1.0) if violation_certified == violation_certified else False,
    }


def main():
    rng = np.random.default_rng(2026)
    delta = 0.05  # plateau region confirmed in finite_sample validation
    ladder = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
    gamma_values = [0.0]

    rows = []
    for gamma_phi in gamma_values:
        for n_events in ladder:
            row = one_run(gamma_phi, n_events, delta, rng)
            rows.append(row)
            print(json.dumps({k: row[k] for k in
                              ("gamma_phi", "n_events", "violation_certified_lower_95",
                               "certifies_at_95")}), flush=True)

    payload = {
        "study": "FINITE_SAMPLE_CERTIFIED_VIOLATION",
        "order_and_moments": {"n": 3, "K": 3},
        "fixed": {"theta": frozen.THETA, "Gamma_drain": frozen.GAMMA_DRAIN,
                  "tau_fine": frozen.TAU_FINE, "tau_coarse": frozen.TAU_COARSE},
        "limitations": {
            "I_coarse": "held at its theoretical (quadrature) value; the variational "
                       "chi2 estimator recovers at most ~72% of it and gets worse as "
                       "delta shrinks, which is the unsafe direction for a one-sided "
                       "certificate. Not yet finite-sample validated.",
            "rho_worst_case": "evaluated at a FIXED pole layout (the best located "
                              "Coxian(3) optimum) with weights re-solved per grid point, "
                              "not a re-certified supremum at each point in the moment CI.",
        },
        "rows": rows,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
