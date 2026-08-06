#!/usr/bin/env python3
"""Finite-sample statistics for the finite-resolution Fisher witness.

The witness compares theoretical Fisher information at two detector
bandwidths. This module makes it testable from data:

  sample_true_intervals   draws the undetected waiting time T_true from its
                          exact survival function.
  observed_intervals      adds the detector jitter: because the detector
                          state in the augmented generator resets after every
                          click, T_obs = T_true + Exp(tau) exactly, per
                          interval, independent of the record's history. No
                          detector state needs to be simulated, and no
                          quantum-jump / Gillespie process can be written for
                          the dimer generator `a` itself -- it has negative
                          off-diagonal entries (it is a Bloch-type generator,
                          not a Markov one) and does not describe a jump chain.
  degrade_to_coarse       builds the tau_coarse record from the SAME draws
                          used for tau_fine, via the exact coupling
                          T_coarse = T_fine + Bernoulli(1-alpha)*Exp(tau_coarse),
                          alpha = tau_fine/tau_coarse. This is not a modeling
                          choice: matching Laplace transforms shows it is the
                          unique coupling with the right marginals, and using
                          it collapses the variance of ratio estimators
                          because the two records share their randomness
                          instead of being drawn independently.
  chi2_variational        a two-point surrogate for the Fisher ratio,
                          estimable from finite samples. Restricting the
                          variational family to a finite basis makes the
                          estimator a LOWER bound on the true chi-squared
                          divergence; sample splitting removes the plug-in's
                          upward bias so the reported number stays a lower
                          bound in expectation.
  certified_violation     turns finite samples into a one-sided claim by
                          taking the classical bound at the worst point in a
                          confidence region for the moments, not at the point
                          estimate.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig as scipy_eig

# --------------------------------------------------------------------------
# exact survival function of the undetected dimer, as an exponential sum
# --------------------------------------------------------------------------


def survival_modes(a, x0, survival):
    """Poles and coefficients of S(t) = Re[sum_k coeffs_k exp(lam_k t)]."""
    lam, vecs = scipy_eig(a)
    coeffs = (survival @ vecs) * np.linalg.solve(vecs, x0)
    return lam, coeffs


def survival_eval(t, lam, coeffs):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.real(coeffs @ np.exp(np.outer(lam, t)))


def sample_true_intervals(a, x0, survival, n, rng, bisect_iters=64, tail_decades=40.0):
    """Inverse-CDF sampling of T_true from its exact survival function.

    S is smooth and monotonically decreasing from 1 to 0, so a fixed number of
    vectorized bisection steps gives every sample to machine precision cheaply;
    no per-sample root finder is needed.
    """
    lam, coeffs = survival_modes(a, x0, survival)
    slow_re = np.min(-np.real(lam))
    t_hi = tail_decades / slow_re

    u = rng.uniform(0.0, 1.0, size=n)
    lo = np.zeros(n)
    hi = np.full(n, t_hi)
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        below = survival_eval(mid, lam, coeffs) > u  # S(mid) > u -> root is further right
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# detector jitter and the exact fine/coarse coupling
# --------------------------------------------------------------------------


def observed_intervals(t_true, tau, rng):
    """T_obs = T_true + Exp(tau), the detector's own exponential decay time."""
    return t_true + rng.exponential(tau, size=len(t_true))


def degrade_to_coarse(t_fine_obs, tau_fine, tau_coarse, rng):
    """Couples a coarse-detector record to an already-drawn fine record.

    Matching Laplace transforms of the two detector kernels gives
    1/(1+s tau_c) = 1/(1+s tau_f) * [alpha + (1-alpha)/(1+s tau_c)],
    alpha = tau_fine/tau_coarse, i.e. Exp(tau_coarse) =d
    Exp(tau_fine) + Bernoulli(1-alpha)*Exp(tau_coarse) (a fresh, independent
    draw on the right). Adding that increment to the already-drawn fine record
    reuses its randomness instead of drawing an independent coarse sample.
    """
    if tau_coarse < tau_fine:
        raise ValueError("degrade_to_coarse requires tau_coarse >= tau_fine")
    alpha = tau_fine / tau_coarse
    n = len(t_fine_obs)
    extra = (rng.random(n) < (1.0 - alpha)) * rng.exponential(tau_coarse, size=n)
    return t_fine_obs + extra


# --------------------------------------------------------------------------
# variational chi-squared, restricted to a finite basis
# --------------------------------------------------------------------------


def default_basis_rates(t_scale_lo, t_scale_hi, n_modes=8):
    """Log-spaced real decay rates spanning the record's fast and slow scales.

    Adequate only when the parameter's sensitivity lives in the envelope of
    the density. Here it does not: theta drives Rabi oscillation, so most of
    I_fine is carried by phase sensitivity at the oscillation frequency, and a
    pure decay-rate basis misses it almost entirely (empirically recovers well
    under 1% of the theoretical Fisher information). Use default_basis with an
    omega argument for the oscillatory family this problem actually needs.
    """
    return np.geomspace(1.0 / t_scale_hi, 1.0 / t_scale_lo, n_modes)


def default_basis(t_scale_lo, t_scale_hi, omega, n_decay=6, n_harmonics=3):
    """Damped-oscillation design: exp(-r t){1, cos(k*omega*t), sin(k*omega*t)}.

    omega should be the record's intrinsic oscillation frequency (e.g. the
    imaginary part of the dominant complex pole). Returns a callable
    design(samples) -> (n, K) matrix rather than a rate vector, since the
    basis functions are no longer pure exponentials.
    """
    rates = np.geomspace(1.0 / t_scale_hi, 1.0 / t_scale_lo, n_decay)

    def design(samples):
        samples = np.asarray(samples, dtype=float)
        env = np.exp(-np.outer(samples, rates))  # (n, n_decay)
        cols = [env]
        for k in range(1, n_harmonics + 1):
            phase = k * omega * samples
            cols.append(env * np.cos(phase)[:, None])
            cols.append(env * np.sin(phase)[:, None])
        return np.concatenate(cols, axis=1)

    return design


def _design(samples, rates):
    return np.exp(-np.outer(samples, rates))  # (n, K)


def chi2_variational(samples_p, samples_q, basis, rng, split=True, ridge=1e-10):
    """Lower bound on chi2(P || Q), restricted to a finite basis.

    chi2(P||Q) = sup_g {2 E_P[g] - E_Q[g^2]} - 1; restricting g to a finite
    basis turns the sup into a linear least-squares solve, g = A^-1 b with
    A = E_Q[phi phi^T], b = E_P[phi]. The plug-in estimate of that sup is
    biased upward by ~K/N; fitting g on half the data and evaluating the
    variational functional on the other half removes that bias (in
    expectation, the reported value stays a lower bound on the true
    divergence, which is the correct direction for a violation claim on the
    numerator I_fine).

    `basis` is a callable samples -> (n, K) design matrix (see default_basis);
    passing a plain rate vector is accepted for the pure-decay family.
    """
    design = basis if callable(basis) else (lambda s: _design(s, basis))
    if split:
        i = rng.permutation(len(samples_p))
        j = rng.permutation(len(samples_q))
        p_fit, p_eval = samples_p[i[: len(i) // 2]], samples_p[i[len(i) // 2 :]]
        q_fit, q_eval = samples_q[j[: len(j) // 2]], samples_q[j[len(j) // 2 :]]
    else:
        p_fit = p_eval = samples_p
        q_fit = q_eval = samples_q

    phi_q_fit = design(q_fit)
    phi_p_fit = design(p_fit)
    k = phi_q_fit.shape[1]
    a_mat = phi_q_fit.T @ phi_q_fit / len(q_fit) + ridge * np.eye(k)
    b_vec = phi_p_fit.mean(axis=0)
    g_coef = np.linalg.solve(a_mat, b_vec)

    g_p = design(p_eval) @ g_coef
    g_q = design(q_eval) @ g_coef
    return float(2.0 * g_p.mean() - (g_q**2).mean() - 1.0)


def fisher_surrogate(sample_theta, sample_theta_plus_delta, delta, basis, rng, n_boot=0):
    """chi2(theta+delta || theta) / delta^2, a finite-sample surrogate for the Fisher ratio's terms."""
    val = chi2_variational(sample_theta_plus_delta, sample_theta, basis, rng)
    out = {"chi2": val, "delta": delta, "fisher_surrogate": val / delta**2}
    if n_boot:
        boots = []
        n = len(sample_theta)
        for _ in range(n_boot):
            idx_p = rng.integers(0, len(sample_theta_plus_delta), len(sample_theta_plus_delta))
            idx_q = rng.integers(0, n, n)
            boots.append(chi2_variational(sample_theta_plus_delta[idx_p], sample_theta[idx_q],
                                          basis, rng, split=False) / delta**2)
        out["bootstrap_std"] = float(np.std(boots))
    return out


# --------------------------------------------------------------------------
# moment estimation and a worst-case-over-confidence-region bound
# --------------------------------------------------------------------------


def moment_estimates(samples, kmax):
    return np.array([np.mean(samples**k) for k in range(1, kmax + 1)])


def deconvolve_moments(m_obs, tau, kmax):
    """Recovers raw moments of T_true from raw moments of T_true + Exp(tau).

    The comparison classes in ph_classes.py are built from the moments of the
    UNDETECTED process (m1, m2, m3 of the record before the tau_fine jitter is
    added), but only T_obs = T_true + Exp(tau) is observable. Because the
    jitter's distribution is known exactly, its contribution can be subtracted
    off. Binomial expansion of E[(T+J)^k] for independent T, J gives a
    triangular system in the true moments (E[J^n] = n! tau^n for J ~ Exp(tau)):

        m_k_obs = m_k_true + sum_{i=0}^{k-1} C(k,i) m_i_true (k-i)! tau^(k-i)

    with m_0_true = 1, solved here by forward substitution for k = 1..kmax
    rather than by a hand-derived closed form, so it generalizes to any kmax.
    """
    from math import comb, factorial

    m_obs = np.asarray(m_obs, dtype=float)
    m_true = np.zeros(kmax + 1)
    m_true[0] = 1.0
    for k in range(1, kmax + 1):
        tail = sum(comb(k, i) * m_true[i] * factorial(k - i) * tau ** (k - i)
                  for i in range(k))
        m_true[k] = m_obs[k - 1] - tail
    return m_true[1:]


def bootstrap_moment_ci(samples, kmax, n_boot, rng, alpha=0.05):
    """Percentile bootstrap CI for (m1, ..., m_kmax)."""
    n = len(samples)
    boots = np.empty((n_boot, kmax))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = moment_estimates(samples[idx], kmax)
    lo = np.percentile(boots, 100 * alpha / 2, axis=0)
    hi = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)
    return lo, hi, boots


def certified_bound(rho_fn, m_hat, m_lo, m_hi, n_grid=15):
    """Worst-case rho over an axis-aligned box CI, by grid search.

    A one-sided certificate needs the SUPREMUM of the classical bound over the
    confidence region, not its value at the point estimate: using the point
    estimate would silently drop coverage whenever the true moments sit
    anywhere else in the region. The grid is coarse by design -- rho_relaxed is
    expensive per point -- and is adequate because rho is smooth and, in the
    two-moment case checked here, monotone in cv^2, so a boundary grid already
    brackets the interior optimum.
    """
    grids = [np.linspace(lo, hi, n_grid) for lo, hi in zip(m_lo, m_hi)]
    best = -np.inf
    if len(grids) == 2:
        for m1 in grids[0]:
            for m2 in grids[1]:
                val = rho_fn(m1, m2)
                if val == val and val > best:
                    best = val
    else:
        mesh = np.meshgrid(*grids, indexing="ij")
        for idx in np.ndindex(mesh[0].shape):
            point = [g[idx] for g in mesh]
            val = rho_fn(*point)
            if val == val and val > best:
                best = val
    return best
