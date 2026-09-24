#!/usr/bin/env python3
"""Tests for the finite-resolution Fisher kernel.

Run with:  python3 -m pytest tests/ -q
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "numerics"))

import fisher_kernel as fk  # noqa: E402
import transport_native_perturbation_gate as frozen  # noqa: E402

TAU_FINE, TAU_COARSE = frozen.TAU_FINE, frozen.TAU_COARSE
GATE = json.loads((ROOT / "results" / "transport_native_dephasing_gate.json").read_text())
ROW0 = GATE["rows"][0]


REF_ARG_MAX = 1.0  # the series reference is only trustworthy for |d*s| <~ 1


def reference_conv(rate, t, tau, which):
    """40-term series reference, entire in d, valid for any complex rate.

    conv0 = e^-s sum_{m>=0} s^(m+1) d^m /(m+1)!
    conv1 = e^-s tau sum_{m>=1} m s^(m+1) d^(m-1) /(m+1)!

    The series is only used as a reference where |d*s| is small.  For larger
    arguments its own terms alternate and cancel, so it stops being a reference
    at all -- callers must restrict `t` via `reference_domain`.
    """
    from math import factorial

    d, s = 1.0 - rate * tau, t / tau
    acc = np.zeros_like(np.asarray(s, dtype=complex))
    for m in range(0, 40):
        if which == 0:
            acc = acc + s ** (m + 1) * d**m / factorial(m + 1)
        elif m >= 1:
            acc = acc + m * s ** (m + 1) * d ** (m - 1) / factorial(m + 1)
    return np.exp(-s) * acc * (tau if which == 1 else 1.0)


def reference_domain(rate, t, tau):
    """Mask of times where the series reference is meaningful."""
    return np.abs((1.0 - rate * tau) * t / tau) < REF_ARG_MAX


# --------------------------------------------------------------------------
# convolved modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.2, 8.0])
@pytest.mark.parametrize("scale", [0.0, 1e-7, 1e-3, 1e-1])
def test_conv_matches_series_through_the_removable_singularity(tau, scale):
    """conv0/conv1 stay accurate where 1 - rate*tau -> 0.

    This is where the frozen module's finite-difference fallback for conv1
    loses up to six orders of magnitude; the series switch must not.
    """
    t = np.array([1e-3, 0.05, 1.0, 5.0, 20.0])
    rate = (1.0 / tau) * (1.0 + scale)
    keep = reference_domain(rate, t, tau)
    assert keep.any()
    for which, fn in ((0, fk.conv0), (1, fk.conv1)):
        ref = np.real(reference_conv(rate, t[keep], tau, which))
        got = fn(rate, t[keep], tau)
        assert np.max(np.abs(got - ref) / np.abs(ref)) < 1e-10


def test_conv_supports_complex_rates():
    t = np.array([1e-3, 0.1, 1.0, 10.0])
    r = 0.3 + 1.7j
    keep = reference_domain(r, t, 0.2)
    assert keep.any()
    for which, fn in ((0, fk.conv0), (1, fk.conv1)):
        ref = reference_conv(r, t[keep], 0.2, which)
        assert np.max(np.abs(fn(r, t[keep], 0.2) - ref) / np.abs(ref)) < 1e-10


@pytest.mark.parametrize("rate", [0.05, 0.5, 3.0, 0.2 + 0.9j])
def test_mode_integrals(rate):
    """conv0 integrates to 1/r and conv1 to 1/r^2."""
    xs, ws = fk.adaptive_quadrature([rate], [0.2, 8.0], order=20)
    for tau in (0.2, 8.0):
        assert abs(ws @ fk.conv0(rate, xs, tau) - 1.0 / rate) < 1e-10
        assert abs(ws @ fk.conv1(rate, xs, tau) - 1.0 / rate**2) < 1e-9


# --------------------------------------------------------------------------
# tangent space
# --------------------------------------------------------------------------


def test_tangent_basis_is_mass_preserving():
    rates = np.array([0.05, 0.4 + 1.1j, 0.4 - 1.1j])
    xs, ws = fk.adaptive_quadrature(rates, [TAU_FINE], order=20)
    basis, mass = fk.real_mode_basis(rates, xs, TAU_FINE)
    tangents = basis @ fk.mass_preserving_null(mass)
    assert np.max(np.abs(ws @ tangents)) < 1e-9


# --------------------------------------------------------------------------
# the anchor: PH2
# --------------------------------------------------------------------------


def ph2_endpoint_modes(m1, cv2):
    root = np.sqrt(2.0 * cv2 - 1.0)
    x, y = 0.5 * m1 * (1.0 + root), 0.5 * m1 * (1.0 - root)
    amp = 1.0 / (x - y)
    return np.array([1.0 / x, 1.0 / y]), np.array([amp, -amp])


@pytest.mark.parametrize("grid", [(16, 8, 6), (20, 16, 12), (24, 24, 18)])
def test_ph2_endpoint_regression(grid):
    """The kernel reproduces the published PH2 bound, on every grid."""
    rates, coeffs = ph2_endpoint_modes(ROW0["m1"], ROW0["cv2"])
    xs, ws = fk.adaptive_quadrature(rates, [TAU_FINE, TAU_COARSE],
                                    order=grid[0], per_period=grid[1], per_decay=grid[2])
    rho, rank = fk.rho_from_modes(rates, coeffs, TAU_FINE, TAU_COARSE, xs, ws)
    assert rank == 3
    assert abs(rho - ROW0["rho_PH2_endpoint"]) / ROW0["rho_PH2_endpoint"] < 1e-6


def test_ph2_endpoint_maximizes_the_moment_matched_family():
    """The T2 endpoint-maximizer proposition, as a regression test.

    Matching (m1, m2) inside APH2 leaves exactly one free parameter p, with
    p*s^2 - p*m1*s + (m1^2 - m2/2) = 0 for s = 1/lambda_1.  rho is maximal at
    p = 1, the pure hypoexponential the frozen module assumes.
    """
    m1, cv2 = ROW0["m1"], ROW0["cv2"]
    m2 = m1**2 * (1.0 + cv2)
    k = m1**2 - m2 / 2.0
    endpoint = None
    interior = []
    for p in np.linspace(2.0 * (1.0 - cv2) + 1e-6, 1.0, 40):
        disc = p * p * m1 * m1 - 4.0 * p * k
        if disc < 0:
            continue
        for sign in (+1.0, -1.0):
            s = (p * m1 + sign * np.sqrt(disc)) / (2.0 * p)
            r = m1 - p * s
            if s <= 0 or r <= 0 or abs(s - r) < 1e-9:
                continue
            lam1, lam2 = 1.0 / s, 1.0 / r
            a = p * lam1 * lam2 / (lam2 - lam1)
            rates, coeffs = np.array([lam1, lam2]), np.array([a, lam2 * (1 - p) - a])
            xs, ws = fk.adaptive_quadrature(rates, [TAU_FINE, TAU_COARSE], order=16)
            rho, _ = fk.rho_from_modes(rates, coeffs, TAU_FINE, TAU_COARSE, xs, ws)
            if p == 1.0:
                endpoint = rho
            else:
                interior.append(rho)
    assert endpoint is not None and interior
    assert max(interior) <= endpoint + 1e-6


# --------------------------------------------------------------------------
# what the kernel was built to settle
# --------------------------------------------------------------------------


def test_data_processing_floor():
    """Coarse is a degradation of fine, so the ratio can never drop below 1."""
    rates, coeffs = ph2_endpoint_modes(ROW0["m1"], ROW0["cv2"])
    xs, ws = fk.adaptive_quadrature(rates, [TAU_FINE, TAU_COARSE], order=16)
    rho, _ = fk.rho_from_modes(rates, coeffs, TAU_FINE, TAU_COARSE, xs, ws)
    assert rho >= 1.0


def test_ph3_supremum_leaves_the_witness_alive():
    """Locks in the critical-order finding at n=3.

    A genuine Coxian(3) search reaches rho_3 ~= 52.4, well under the quantum
    ratio 156.86, so the order-3 classical class does not explain the record.
    The number is a multistart lower bound on the true supremum; the assertion
    is deliberately one-sided and loose.
    """
    lam = np.array([10.6187721, 0.05102169, 0.3568565])  # located by multistart
    ps = np.array([0.8930472978918961, 0.9999997373541855])
    t_mat = np.diag(-lam)
    for i in range(2):
        t_mat[i, i + 1] = ps[i] * lam[i]
    alpha = np.array([1.0, 0.0, 0.0])
    inv = np.linalg.inv(-t_mat)
    m1 = alpha @ inv @ np.ones(3)
    assert abs(m1 - ROW0["m1"]) < 1e-4  # lam/ps are rounded to the digits above

    vals, vecs = np.linalg.eig(t_mat)
    coeffs = (alpha @ vecs) * np.linalg.solve(vecs, -t_mat @ np.ones(3))
    rates = -vals
    xs, ws = fk.adaptive_quadrature(rates, [TAU_FINE, TAU_COARSE], order=20,
                                    per_period=16, per_decay=12)
    rho, rank = fk.rho_from_modes(rates, coeffs, TAU_FINE, TAU_COARSE, xs, ws)
    assert rank == 5
    assert 52.0 < rho < 53.0
    assert rho < ROW0["rho_quantum"]
