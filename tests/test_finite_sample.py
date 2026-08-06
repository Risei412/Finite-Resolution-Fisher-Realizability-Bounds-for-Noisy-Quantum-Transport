#!/usr/bin/env python3
"""Tests for the finite-sample statistics module."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "numerics"))

import finite_sample as fs  # noqa: E402
import transport_native_perturbation_gate as frozen  # noqa: E402

X0 = np.array([1.0, 0.0, 0.0])
SURVIVAL = np.array([1.0, 1.0, 0.0])


@pytest.fixture(scope="module")
def dimer():
    a, _ = frozen.quantum_generator(frozen.THETA, 0.0)
    return a


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------


def test_survival_modes_reproduce_the_analytic_moments(dimer):
    lam, coeffs = fs.survival_modes(dimer, X0, SURVIVAL)
    assert abs(fs.survival_eval(0.0, lam, coeffs)[0] - 1.0) < 1e-10
    m1_analytic, _ = frozen.quantum_moments(frozen.THETA, 0.0)
    # m1 = integral of S(t) dt = sum coeffs_k * (-1/lam_k)
    m1_from_modes = np.real(np.sum(coeffs / (-lam)))
    assert abs(m1_from_modes - m1_analytic) < 1e-8


def test_sampler_matches_the_analytic_survival_function(dimer):
    rng = np.random.default_rng(0)
    t_true = fs.sample_true_intervals(dimer, X0, SURVIVAL, 50_000, rng)
    lam, coeffs = fs.survival_modes(dimer, X0, SURVIVAL)

    def cdf(t):
        return 1.0 - fs.survival_eval(t, lam, coeffs)

    stat, pval = stats.kstest(t_true, cdf)
    assert pval > 0.01
    m1_analytic, _ = frozen.quantum_moments(frozen.THETA, 0.0)
    assert abs(t_true.mean() - m1_analytic) < 5 * t_true.std() / np.sqrt(len(t_true))


def test_observed_intervals_add_the_detector_jitter(dimer):
    rng = np.random.default_rng(1)
    t_true = fs.sample_true_intervals(dimer, X0, SURVIVAL, 30_000, rng)
    t_obs = fs.observed_intervals(t_true, frozen.TAU_FINE, rng)
    assert np.all(t_obs >= t_true)
    m1_analytic, _ = frozen.quantum_moments(frozen.THETA, 0.0)
    expected = m1_analytic + frozen.TAU_FINE
    assert abs(t_obs.mean() - expected) < 5 * t_obs.std() / np.sqrt(len(t_obs))


# --------------------------------------------------------------------------
# the fine/coarse coupling
# --------------------------------------------------------------------------


def test_coupling_reproduces_the_coarse_marginal(dimer):
    """degrade_to_coarse must have the same distribution as an independent draw.

    This is the exact-coupling identity derived from matching Laplace
    transforms of the two detector kernels; a KS test against an independently
    sampled coarse record is the direct check that the identity was coded
    correctly.
    """
    rng = np.random.default_rng(2)
    t_true = fs.sample_true_intervals(dimer, X0, SURVIVAL, 80_000, rng)
    t_fine = fs.observed_intervals(t_true, frozen.TAU_FINE, rng)
    t_coarse_coupled = fs.degrade_to_coarse(t_fine, frozen.TAU_FINE, frozen.TAU_COARSE, rng)

    t_true2 = fs.sample_true_intervals(dimer, X0, SURVIVAL, 80_000, rng)
    t_coarse_indep = fs.observed_intervals(t_true2, frozen.TAU_COARSE, rng)

    stat, pval = stats.ks_2samp(t_coarse_coupled, t_coarse_indep)
    assert pval > 0.01


def test_coupling_shares_randomness_with_the_fine_record(dimer):
    """The whole point of the coupling: high correlation, not independence."""
    rng = np.random.default_rng(3)
    t_true = fs.sample_true_intervals(dimer, X0, SURVIVAL, 30_000, rng)
    t_fine = fs.observed_intervals(t_true, frozen.TAU_FINE, rng)
    t_coarse = fs.degrade_to_coarse(t_fine, frozen.TAU_FINE, frozen.TAU_COARSE, rng)
    assert np.corrcoef(t_fine, t_coarse)[0, 1] > 0.5


def test_coupling_rejects_coarse_faster_than_fine():
    with pytest.raises(ValueError):
        fs.degrade_to_coarse(np.array([1.0, 2.0]), 8.0, 0.2, np.random.default_rng(0))


# --------------------------------------------------------------------------
# variational chi-squared
# --------------------------------------------------------------------------


def test_chi2_variational_is_near_zero_for_identical_distributions(dimer):
    rng = np.random.default_rng(4)
    t_true = fs.sample_true_intervals(dimer, X0, SURVIVAL, 40_000, rng)
    t_obs = fs.observed_intervals(t_true, frozen.TAU_FINE, rng)
    half = len(t_obs) // 2
    lam, _ = fs.survival_modes(dimer, X0, SURVIVAL)
    omega = np.abs(np.imag(lam)).max()
    basis = fs.default_basis(frozen.TAU_FINE, 60.0, omega, n_decay=4, n_harmonics=2)
    val = fs.chi2_variational(t_obs[:half], t_obs[half:], basis, rng)
    assert abs(val) < 0.05  # should fluctuate around 0, not show a spurious divergence


def test_fine_fisher_surrogate_reaches_a_plateau_near_the_theoretical_value(dimer):
    """The headline validation: chi2/delta^2 recovers I_fine in its plateau.

    theta drives Rabi oscillation, so most of I_fine is phase sensitivity at
    the oscillation frequency omega; a basis without that oscillatory
    component recovers under 1% of the theoretical value (checked during
    development), which is why default_basis includes harmonics of omega.
    """
    rng = np.random.default_rng(5)
    theta0 = frozen.THETA
    delta = 0.05
    n = 250_000

    a0, _ = frozen.quantum_generator(theta0, 0.0)
    a1, _ = frozen.quantum_generator(theta0 + delta, 0.0)
    lam, _ = fs.survival_modes(a0, X0, SURVIVAL)
    omega = np.abs(np.imag(lam)).max()
    basis = fs.default_basis(frozen.TAU_FINE, 60.0, omega, n_decay=6, n_harmonics=3)

    t0 = fs.observed_intervals(fs.sample_true_intervals(a0, X0, SURVIVAL, n, rng),
                               frozen.TAU_FINE, rng)
    t1 = fs.observed_intervals(fs.sample_true_intervals(a1, X0, SURVIVAL, n, rng),
                               frozen.TAU_FINE, rng)
    out = fs.fisher_surrogate(t0, t1, delta, basis, rng)

    fine_theory, _, _ = frozen.quantum_fisher(theta0, 0.0, frozen.TAU_FINE)
    assert 0.7 < out["fisher_surrogate"] / fine_theory < 1.3


# --------------------------------------------------------------------------
# moments and the worst-case bound
# --------------------------------------------------------------------------


def test_moment_estimates_match_theory(dimer):
    rng = np.random.default_rng(6)
    t_true = fs.sample_true_intervals(dimer, X0, SURVIVAL, 200_000, rng)
    m1_analytic, m2_analytic = frozen.quantum_moments(frozen.THETA, 0.0)
    m1_hat, m2_hat = fs.moment_estimates(t_true, 2)
    assert abs(m1_hat - m1_analytic) / m1_analytic < 0.02
    assert abs(m2_hat - m2_analytic) / m2_analytic < 0.05


def test_bootstrap_ci_covers_the_true_moments(dimer):
    rng = np.random.default_rng(7)
    t_true = fs.sample_true_intervals(dimer, X0, SURVIVAL, 20_000, rng)
    m1_analytic, m2_analytic = frozen.quantum_moments(frozen.THETA, 0.0)
    lo, hi, _ = fs.bootstrap_moment_ci(t_true, 2, n_boot=300, rng=rng, alpha=0.05)
    assert lo[0] < m1_analytic < hi[0]
    assert lo[1] < m2_analytic < hi[1]


def test_deconvolve_moments_recovers_true_moments_from_jittered_observations(dimer):
    """The comparison classes use moments of T_true; only T_obs is observable.

    Because the tau jitter is exactly Exp(tau), its contribution can be
    subtracted exactly (not just asymptotically); this checks that recovery
    with finite N.
    """
    import ph_classes as pc

    m_true_analytic = pc.quantum_moments_k(dimer, X0, SURVIVAL, 3)
    rng = np.random.default_rng(9)
    t_true = fs.sample_true_intervals(dimer, X0, SURVIVAL, 400_000, rng)
    t_obs = fs.observed_intervals(t_true, frozen.TAU_FINE, rng)
    m_obs = fs.moment_estimates(t_obs, 3)
    m_deconv = fs.deconvolve_moments(m_obs, frozen.TAU_FINE, 3)
    assert np.all(np.abs(m_deconv - m_true_analytic) / np.abs(m_true_analytic) < 0.02)


def test_deconvolve_moments_is_identity_at_tau_zero(dimer):
    rng = np.random.default_rng(10)
    t_true = fs.sample_true_intervals(dimer, X0, SURVIVAL, 10_000, rng)
    m_obs = fs.moment_estimates(t_true, 3)
    m_deconv = fs.deconvolve_moments(m_obs, 0.0, 3)
    assert np.allclose(m_deconv, m_obs)


def test_certified_bound_is_at_least_the_point_estimate():
    """The worst-case-over-CI bound cannot be smaller than rho at the center."""
    def rho_fn(m1, m2):
        return m1 + 10.0 * m2  # monotone increasing toy surrogate

    m_hat = [20.0, 800.0]
    lo = [19.0, 750.0]
    hi = [21.0, 850.0]
    point = rho_fn(*m_hat)
    worst = fs.certified_bound(rho_fn, m_hat, lo, hi, n_grid=5)
    assert worst >= point
    assert abs(worst - rho_fn(21.0, 850.0)) < 1e-9  # monotone -> corner is the max
