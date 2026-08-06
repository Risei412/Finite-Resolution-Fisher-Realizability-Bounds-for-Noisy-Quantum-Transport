#!/usr/bin/env python3
"""Smoke tests for the N-vs-significance driver."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "numerics"))

import finite_stats_driver as driver  # noqa: E402


def test_rho_at_fixed_poles_reproduces_the_locked_optimum():
    """The fixed pole layout must actually match the analytic (m1,m2,m3) target."""
    import ph_classes as pc
    import transport_native_perturbation_gate as frozen

    a, _ = frozen.quantum_generator(frozen.THETA, 0.0)
    m = pc.quantum_moments_k(a, driver.X0, driver.SURVIVAL, 3)
    rho = driver.rho_at_fixed_poles(*m, frozen.TAU_FINE, frozen.TAU_COARSE)
    assert rho == rho  # not NaN
    assert 27.0 < rho < 29.0


def test_one_run_produces_a_well_formed_row():
    rng = np.random.default_rng(123)
    row = driver.one_run(0.0, 30_000, 0.05, rng, n_boot_moments=40, n_boot_chi2=10)
    assert row["n_events"] == 30_000
    assert row["data_driven"]["I_fine_hat"] == row["data_driven"]["I_fine_hat"]  # not NaN
    assert row["data_driven"]["rho_worst_case"] > 0
    assert row["B_n_certified"] > 0
    assert isinstance(row["certifies_at_95"], bool)


def test_small_n_does_not_falsely_certify():
    """At N too small for the chi2 basis, the estimator must not silently pass."""
    rng = np.random.default_rng(456)
    row = driver.one_run(0.0, 2000, 0.05, rng, n_boot_moments=40, n_boot_chi2=10)
    assert row["data_driven"]["I_fine_lower_95"] >= 0.0  # clipped, never negative
    # a breakdown at this N must not be reported as a confident certificate
    if row["data_driven"]["I_fine_hat"] < 50.0:
        assert not row["certifies_at_95"]
