#!/usr/bin/env python3
"""Tests for the order-n classical comparison classes."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "numerics"))

import ph_classes as pc  # noqa: E402
import transport_native_perturbation_gate as frozen  # noqa: E402

TAU_FINE, TAU_COARSE = frozen.TAU_FINE, frozen.TAU_COARSE
GATE = json.loads((ROOT / "results" / "transport_native_dephasing_gate.json").read_text())
ROW0 = GATE["rows"][0]
X0 = np.array([1.0, 0.0, 0.0])
SURVIVAL = np.array([1.0, 1.0, 0.0])


def quantum_moments(gamma_phi, kmax):
    a, _ = frozen.quantum_generator(frozen.THETA, gamma_phi)
    return pc.quantum_moments_k(a, X0, SURVIVAL, kmax)


# --------------------------------------------------------------------------


@pytest.mark.parametrize("row", GATE["rows"])
def test_quantum_moments_agree_with_the_frozen_module(row):
    """The k-moment generalization must reproduce the published m1, m2."""
    m = quantum_moments(row["gamma_phi"], 2)
    assert abs(m[0] - row["m1"]) < 1e-10 * abs(row["m1"])
    assert abs(m[1] - row["m2"]) < 1e-10 * abs(row["m2"])


def test_moment_matrix_round_trip():
    """moment_values returns (mass, m_1, ..., m_K) -- verified by quadrature."""
    import fisher_kernel as fk

    rates = np.array([0.05, 0.7, 4.0], dtype=complex)
    coeffs = np.array([0.04, -0.5, 2.0], dtype=complex)
    got = pc.moment_values(rates, coeffs, 3)
    xs, ws = fk.adaptive_quadrature(rates, [1.0], order=24, per_decay=24)
    w = np.real(coeffs @ np.exp(-np.outer(rates, xs)))
    brute = [ws @ w] + [ws @ (xs**k * w) for k in (1, 2, 3)]
    assert np.allclose(got, brute, rtol=1e-6)


# --------------------------------------------------------------------------
# positivity by root isolation
# --------------------------------------------------------------------------


def test_density_minimum_is_nonnegative_for_a_hypoexponential():
    rates = np.array([1.0, 2.0], dtype=complex)
    coeffs = np.array([2.0, -2.0], dtype=complex)  # 2(e^-t - e^-2t) >= 0
    assert pc.density_minimum(rates, coeffs) >= -1e-14


def test_density_minimum_finds_a_dip_below_zero():
    """A sum that is negative only near the origin must be rejected."""
    rates = np.array([1.0, 2.0], dtype=complex)
    coeffs = np.array([1.0, -1.1], dtype=complex)  # w(0) = -0.1
    got = pc.density_minimum(rates, coeffs)
    t = np.geomspace(1e-12, 200.0, 2_000_00)
    brute = np.real(coeffs @ np.exp(-np.outer(rates, t))).min()
    assert got < 0
    assert got <= brute + 1e-12


def test_density_minimum_matches_a_brute_force_scan():
    rates = np.array([0.05, 0.9, 3.0], dtype=complex)
    coeffs = np.array([0.06, 0.8, -1.4], dtype=complex)
    t = np.geomspace(1e-10, 2000.0, 2_000_000)
    brute = np.real(coeffs @ np.exp(-np.outer(rates, t))).min()
    assert pc.density_minimum(rates, coeffs) <= brute + 1e-10


# --------------------------------------------------------------------------
# O'Cinneide cone
# --------------------------------------------------------------------------


def test_cone_forbids_oscillation_at_order_two():
    assert pc.in_ocinneide_cone(np.array([1.0, 2.0], dtype=complex), 2)
    assert not pc.in_ocinneide_cone(np.array([1 + 0.01j, 1 - 0.01j]), 2)


def test_cone_cap_matches_cot_pi_over_n():
    for n in (3, 4, 6, 8):
        cap = 1.0 / np.tan(np.pi / n)
        inside = np.array([1 + 0.99 * cap * 1j, 1 - 0.99 * cap * 1j])
        outside = np.array([1 + 1.01 * cap * 1j, 1 - 1.01 * cap * 1j])
        assert pc.in_ocinneide_cone(inside, n)
        assert not pc.in_ocinneide_cone(outside, n)


def test_feedback_erlang_saturates_the_cone():
    """The cyclic chain sits exactly on the cone boundary, as it should."""
    n, mu, q = 6, 1.0, 1e-9
    t_mat = np.zeros((n, n))
    for i in range(n - 1):
        t_mat[i, i] = -mu
        t_mat[i, i + 1] = mu
    t_mat[n - 1, n - 1] = -mu
    t_mat[n - 1, 0] = (1 - q) * mu
    rates = -np.linalg.eigvals(t_mat)
    ratio = np.max(np.abs(np.imag(rates)) / np.real(rates))
    assert abs(ratio - 1.0 / np.tan(np.pi / n)) < 1e-4


# --------------------------------------------------------------------------
# weights from moments
# --------------------------------------------------------------------------


def test_weight_system_is_square_at_k_equals_n_minus_one():
    """n weights against 1+K constraints: exactly determined when K = n-1."""
    m = quantum_moments(0.0, 2)
    rates = np.array([0.05, 0.6, 5.0], dtype=complex)
    coeffs, resid = pc.coeffs_from_moments(rates, m)
    assert resid < 1e-12
    assert np.allclose(pc.moment_values(rates, coeffs, 2)[1:], m, rtol=1e-10)


def test_weight_system_is_over_determined_once_k_reaches_n():
    """Raising K past n-1 is what shrinks the relaxation to a variety."""
    m = quantum_moments(0.0, 3)
    rates = np.array([0.05, 0.6, 5.0], dtype=complex)
    _, resid = pc.coeffs_from_moments(rates, m)
    assert resid > 1e-8  # a generic pole triple misses, by the code's own tolerance

    z, projected = pc.project_to_variety(np.log(np.real(rates)), 3, 0, m)
    assert projected < 1e-8  # ... but the projection reaches the variety


# --------------------------------------------------------------------------
# the ratio itself
# --------------------------------------------------------------------------


def test_matching_four_moments_over_determines_order_three_further():
    """K=4 leaves 5 constraints on 3 weights -- codimension 2, not 1.

    Empirically no order-3 pole set with a nonnegative density survives at
    K=4, which would falsify order 3 from the moments alone and leave the
    Fisher ratio doing no work.  K=3 is the setting where the class is still
    populated and the ratio is what does the falsifying.
    """
    m = quantum_moments(0.0, 4)
    rates = np.array([0.05, 0.6, 5.0], dtype=complex)
    amat = pc.moment_matrix(rates, 4)
    assert amat.shape == (5, 3)
    _, resid = pc.coeffs_from_moments(rates, m)
    assert resid > 1e-8


def test_relaxed_maximizer_at_order_three_stays_under_the_quantum_ratio():
    """Regression on the Phase 1a result: rho at (n=3, K=3) ~= 28.2.

    Locks the maximizer located by the search rather than re-running it.  The
    bound only has to fall below rho_quantum = 156.86 to certify order 3; it
    does not have to be tight.

    The relaxed search is known to under-converge here: a direct Coxian(3)
    search reaches 28.2876, which exceeds this maximizer's 28.208 and so proves
    the true upper bound is at least that.  The margin to 156.86 is a factor of
    5.5, so the verdict does not turn on it -- but no figure at this K is a
    proof, only a lower estimate of the bound.
    """
    m = quantum_moments(0.0, 3)
    rates = np.array([0.04568371, 0.05148707, 4.97113451], dtype=complex)
    coeffs, resid = pc.coeffs_from_moments(rates, m)
    assert resid < 1e-7
    rho = pc.rho_of_modes(rates, coeffs, TAU_FINE, TAU_COARSE, grid=(20, 16, 12))
    assert rho is not None
    assert 27.0 < rho < 29.5
    assert rho < ROW0["rho_quantum"]


def test_rho_of_modes_reproduces_the_ph2_bound():
    m1, cv2 = ROW0["m1"], ROW0["cv2"]
    root = np.sqrt(2.0 * cv2 - 1.0)
    x, y = 0.5 * m1 * (1.0 + root), 0.5 * m1 * (1.0 - root)
    amp = 1.0 / (x - y)
    rho = pc.rho_of_modes(np.array([1 / x, 1 / y], dtype=complex),
                          np.array([amp, -amp], dtype=complex),
                          TAU_FINE, TAU_COARSE, grid=(20, 16, 12))
    assert abs(rho - ROW0["rho_PH2_endpoint"]) / ROW0["rho_PH2_endpoint"] < 1e-6
