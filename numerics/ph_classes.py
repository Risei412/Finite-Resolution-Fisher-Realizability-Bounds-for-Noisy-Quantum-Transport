#!/usr/bin/env python3
"""Order-n classical comparison classes for the finite-resolution witness.

Two objects are computed for each (order n, number of matched moments K):

  rho_direct(n, K)   supremum over genuine Coxian(n) renewal processes.
                     A multistart search, so a LOWER bound on the true
                     supremum.  Cannot support a violation claim on its own.

  rho_relaxed(n, K)  supremum over densities that are n-term exponential sums
                     with the same matched moments and a nonnegative density.
                     Every phase-type density of order n is such a sum, so this
                     is an UPPER bound on the classical supremum -- the object a
                     violation claim actually needs.

The witness is claimed when I_fine exceeds rho_relaxed * I_coarse.  With K=2
the relaxation is far too loose to do that (it reaches ~203 at n=3 against a
quantum ratio of 156.9), which is why K is raised here.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.linalg import inv
from scipy.optimize import brentq, least_squares, minimize

import fisher_kernel as fk

# --------------------------------------------------------------------------
# quantum side: exact moments of the waiting-time density
# --------------------------------------------------------------------------


def quantum_moments_k(a, x0, survival, kmax):
    """m_k = k! * survival @ (-a)^-k @ x0 for k = 1..kmax."""
    resolvent = inv(-np.asarray(a, dtype=float))
    out, acc, fact = [], np.asarray(x0, dtype=float), 1.0
    for k in range(1, kmax + 1):
        acc = resolvent @ acc
        fact *= k
        out.append(float(fact * (np.asarray(survival, dtype=float) @ acc)))
    return np.asarray(out)


# --------------------------------------------------------------------------
# exponential-sum densities:  w(t) = sum_k c_k exp(-r_k t)
# --------------------------------------------------------------------------


def moment_matrix(rates, kmax):
    """Rows of the linear map c -> (mass, m_1, ..., m_kmax)."""
    rates = np.asarray(rates, dtype=complex)
    fact = 1.0
    rows = [1.0 / rates]
    for k in range(1, kmax + 1):
        fact *= k
        rows.append(fact / rates ** (k + 1))
    return np.vstack(rows)


def moment_values(rates, coeffs, kmax):
    """Returns (mass, m_1, ..., m_kmax) -- the mass row comes first."""
    return np.real(moment_matrix(rates, kmax) @ np.asarray(coeffs, dtype=complex))


def density_minimum(rates, coeffs, n_scan=4000):
    """min of w(t) over (0, inf), located by isolating every real root.

    A real exponential sum with n terms has at most n-1 real zeros, so a
    log-spaced prescan followed by Brent on each sign change finds all of them.
    Sampling the density on a grid instead lets narrow dips slip through; that
    failure mode produces enormous spurious ratios, because the Fisher integrand
    carries 1/w.
    """
    rates = np.asarray(rates, dtype=complex)
    coeffs = np.asarray(coeffs, dtype=complex)

    def w_vec(t):
        return np.real(coeffs @ np.exp(-np.outer(rates, np.atleast_1d(t))))

    def w(t):
        return float(w_vec(t)[0])

    re = np.real(rates)
    lo, hi = 1e-9 / re.max(), 60.0 / re.min()
    ts = np.concatenate([[0.0], np.geomspace(lo, hi, n_scan)])
    vals = np.real(coeffs @ np.exp(-np.outer(rates, ts)))

    candidates = [vals.min()]
    sign_change = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    for i in sign_change:
        try:
            root = brentq(lambda z: w(z), ts[i], ts[i + 1], xtol=1e-14)
        except ValueError:
            continue
        # the density dips below zero on one side of any interior root
        for side in (-1.0, 1.0):
            probe = root + side * 1e-6 * max(root, 1e-9)
            if probe > 0:
                candidates.append(w(probe))
    return float(min(candidates))


# --------------------------------------------------------------------------
# pole parameterisations
# --------------------------------------------------------------------------


def pole_layouts(n):
    """Every split of n poles into real ones and conjugate pairs."""
    return [(n - 2 * j, j) for j in range(n // 2 + 1)]


# Log-coordinate box for the pole search.  Rates outside it are physically
# irrelevant here -- m1 is ~20, so no pole can be slower than ~1/m1 without the
# weights diverging -- and letting the projection wander outside overflows
# 1/rate**k in the moment matrix and hands LAPACK a matrix full of inf.
LOG_RATE_LO, LOG_RATE_HI = -12.0, 12.0


def build_rates(z, n_real, n_pair):
    """Map unconstrained coordinates to a conjugation-closed rate set."""
    z = np.clip(np.asarray(z, dtype=float), LOG_RATE_LO, LOG_RATE_HI)
    reals = np.exp(z[:n_real])
    rest = z[n_real:]
    rates = list(reals.astype(complex))
    for j in range(n_pair):
        re, im = np.exp(rest[2 * j]), np.exp(rest[2 * j + 1])
        rates += [re + 1j * im, re - 1j * im]
    return np.asarray(rates, dtype=complex)


def coeffs_from_moments(rates, targets):
    """Least-squares weights matching (mass, m_1..m_K); returns (c, residual).

    With n weights and 1+K constraints the system is square at K = n-1,
    under-determined below and over-determined above.  Raising K past n-1 is
    exactly the mechanism that shrinks the relaxation: the pole set must then
    lie on a variety for any weight vector to reproduce the moments at all.
    """
    rates = np.asarray(rates, dtype=complex)
    kmax = len(targets)
    amat = moment_matrix(rates, kmax)
    rhs = np.concatenate([[1.0], np.asarray(targets, dtype=float)])
    # solve in the real embedding so conjugate pairs get real weights
    big = np.vstack([np.real(amat), np.imag(amat)])
    rhs_big = np.concatenate([rhs, np.zeros_like(rhs)])
    sol, *_ = np.linalg.lstsq(big, rhs_big, rcond=None)
    resid = float(np.linalg.norm(big @ sol - rhs_big) / max(np.linalg.norm(rhs_big), 1e-30))
    return sol.astype(complex), resid


# --------------------------------------------------------------------------
# the ratio for one exponential-sum density
# --------------------------------------------------------------------------


def rho_of_modes(rates, coeffs, tau_fine, tau_coarse, grid=(16, 8, 6)):
    rates = np.asarray(rates, dtype=complex)
    if np.any(np.real(rates) <= 0):
        return None
    xs, ws = fk.adaptive_quadrature(rates, [tau_fine, tau_coarse],
                                    order=grid[0], per_period=grid[1], per_decay=grid[2])
    dens = fk.mode_density(coeffs, rates, xs, tau_fine)
    if dens.min() < -1e-14 * max(dens.max(), 1e-300):
        return None
    rho, rank = fk.rho_from_modes(rates, coeffs, tau_fine, tau_coarse, xs, ws)
    if rho != rho or rank != 2 * len(rates) - 1:
        return None
    return rho


# --------------------------------------------------------------------------
# the relaxed (upper-bound) supremum
# --------------------------------------------------------------------------


def in_ocinneide_cone(rates, n):
    """Necessary condition for an order-n phase-type density.

    Every pole of a PH_n density satisfies |Im| / |Re| <= cot(pi/n), the cone
    saturated by the cyclic feedback-Erlang chain.  Imposing it keeps the
    relaxation an upper bound on PH_n while excluding wildly oscillatory pole
    sets that no classical Markov chain of that order can produce.
    """
    rates = np.asarray(rates, dtype=complex)
    if n < 2:
        return bool(np.all(np.abs(np.imag(rates)) < 1e-12))
    cap = 1.0 / np.tan(np.pi / n)
    return bool(np.all(np.abs(np.imag(rates)) <= cap * np.real(rates) + 1e-12))


def relaxed_objective(z, n_real, n_pair, targets, tau_fine, tau_coarse,
                      grid, resid_tol, pos_tol):
    rates = build_rates(z, n_real, n_pair)
    if np.min(np.abs(rates[:, None] - rates[None, :] + np.eye(len(rates)))) < 1e-8:
        return None
    if not in_ocinneide_cone(rates, n_real + 2 * n_pair):
        return None
    coeffs, resid = coeffs_from_moments(rates, targets)
    if resid > resid_tol:
        return None
    if density_minimum(rates, coeffs) < -pos_tol:
        return None
    return rho_of_modes(rates, coeffs, tau_fine, tau_coarse, grid)


def moment_residual_vector(z, n_real, n_pair, targets):
    """Componentwise mismatch of the over-determined weight system."""
    rates = build_rates(z, n_real, n_pair)
    if np.any(np.real(rates) <= 0):
        return np.full(2 * (1 + len(targets)), 1e3)
    amat = moment_matrix(rates, len(targets))
    rhs = np.concatenate([[1.0], np.asarray(targets, dtype=float)])
    big = np.vstack([np.real(amat), np.imag(amat)])
    rhs_big = np.concatenate([rhs, np.zeros_like(rhs)])
    sol, *_ = np.linalg.lstsq(big, rhs_big, rcond=None)
    return (big @ sol - rhs_big) / np.maximum(np.abs(rhs_big), 1.0)


def moment_residual_at(z, n_real, n_pair, targets):
    rates = build_rates(z, n_real, n_pair)
    return coeffs_from_moments(rates, targets)[1]


def project_to_variety(z, n_real, n_pair, targets):
    """Pull a starting point onto the set where the moments are reproducible.

    With n weights and 1+K constraints the weight system is over-determined
    once K > n-1, so feasible pole sets form a positive-codimension variety and
    random sampling lands on it with probability zero.  Raising K is precisely
    what tightens the relaxation, so the search has to project first.
    """
    z = np.clip(np.asarray(z, dtype=float), LOG_RATE_LO + 1e-6, LOG_RATE_HI - 1e-6)
    try:
        res = least_squares(moment_residual_vector, z, method="trf",
                            args=(n_real, n_pair, targets),
                            bounds=(LOG_RATE_LO, LOG_RATE_HI),
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=600)
    except Exception:
        return z, np.inf
    return res.x, moment_residual_at(res.x, n_real, n_pair, targets)


def rho_relaxed(n, targets, tau_fine, tau_coarse, seed=0, n_random=4000,
                n_refine=25, grid=(16, 8, 6), resid_tol=1e-8, pos_tol=1e-12,
                log_lo=-3.2, log_hi=3.2, penalty=1e6):
    """Upper-bound supremum over n-term exponential sums matching `targets`.

    Searched over every real/complex pole layout.  The result is the best value
    found; the search is a maximisation of an upper-bound *functional*, so an
    incomplete search under-reports and must be labelled accordingly.
    """
    rng = np.random.default_rng(seed)
    over_determined = (1 + len(targets)) > n
    best = {"rho": -np.inf, "z": None, "layout": None}

    for n_real, n_pair in pole_layouts(n):
        dim = n_real + 2 * n_pair

        def fun(z):
            val = relaxed_objective(z, n_real, n_pair, targets, tau_fine,
                                    tau_coarse, grid, resid_tol, pos_tol)
            if val is not None:
                return -val
            # off the variety the objective is undefined; steer back to it
            return 1e6 + penalty * moment_residual_at(z, n_real, n_pair, targets)

        starts = []
        for _ in range(n_random):
            z = rng.uniform(log_lo, log_hi, dim)
            if over_determined:
                z, resid = project_to_variety(z, n_real, n_pair, targets)
                if resid > resid_tol or not np.all(np.isfinite(z)):
                    continue
            starts.append(z)

        for z in starts:
            val = fun(z)
            if -val > best["rho"]:
                best = {"rho": -val, "z": z, "layout": (n_real, n_pair)}

        if best["layout"] != (n_real, n_pair):
            continue
        for k in range(n_refine):
            z0 = best["z"] + (0.0 if k == 0 else rng.normal(0, 0.4, dim))
            if over_determined:
                z0, resid = project_to_variety(z0, n_real, n_pair, targets)
                if resid > resid_tol:
                    continue
            res = minimize(fun, z0, method="Nelder-Mead",
                           options={"maxiter": 1500, "xatol": 1e-10, "fatol": 1e-12})
            if -res.fun > best["rho"]:
                best = {"rho": -res.fun, "z": res.x, "layout": (n_real, n_pair)}

    if best["z"] is None or best["rho"] < 0:
        return None
    rates = build_rates(best["z"], *best["layout"])
    coeffs, resid = coeffs_from_moments(rates, targets)
    return {
        "rho": best["rho"],
        "rates": rates,
        "coeffs": coeffs,
        "layout": best["layout"],
        "moment_residual": resid,
        "density_minimum": density_minimum(rates, coeffs),
        "certification": "multistart-maximised upper-bound functional",
    }


# --------------------------------------------------------------------------
# the direct (lower-bound) supremum over genuine Coxian models
# --------------------------------------------------------------------------


def coxian_generator(lams, ps):
    t_mat = np.diag(-np.asarray(lams, dtype=float))
    for i in range(len(lams) - 1):
        t_mat[i, i + 1] = ps[i] * lams[i]
    return t_mat


def coxian_modes(lams, ps):
    t_mat = coxian_generator(lams, ps)
    n = len(lams)
    alpha = np.zeros(n)
    alpha[0] = 1.0
    vals, vecs = np.linalg.eig(t_mat)
    if np.linalg.cond(vecs) > 1e10:
        return None
    coeffs = (alpha @ vecs) * np.linalg.solve(vecs, -t_mat @ np.ones(n))
    return -vals, coeffs


def unpack_coxian(z, n):
    lams = np.exp(z[:n])
    ps = 1.0 / (1.0 + np.exp(-np.clip(z[n:], -40.0, 40.0)))
    return lams, ps


def rho_direct(n, targets, tau_fine, tau_coarse, seed=0, n_random=3000,
               n_refine=25, grid=(16, 8, 6), penalty=1e4):
    """Lower bound on the classical supremum, over genuine Coxian(n) models.

    Nonnegativity of the density is automatic here (the rates are a real
    Markov chain), so only the moment constraints need enforcing; they are
    imposed by penalty, which keeps the search unconstrained and robust.
    """
    rng = np.random.default_rng(seed)
    kmax = len(targets)
    tgt = np.concatenate([[1.0], np.asarray(targets, dtype=float)])

    def fun(z):
        lams, ps = unpack_coxian(z, n)
        if np.min(np.diff(np.sort(lams))) < 1e-8:
            return 1e6
        modes = coxian_modes(lams, ps)
        if modes is None:
            return 1e6
        rates, coeffs = modes
        got = moment_values(rates, coeffs, kmax)  # (mass, m_1..m_K)
        miss = np.sum(((got - tgt) / np.maximum(np.abs(tgt), 1e-12)) ** 2)
        rho = rho_of_modes(rates, coeffs, tau_fine, tau_coarse, grid)
        if rho is None:
            return 1e6
        return -rho + penalty * miss

    best = {"val": np.inf, "z": None}
    for _ in range(n_random):
        z = np.concatenate([rng.uniform(-3.2, 3.2, n), rng.uniform(-4.0, 6.0, n - 1)])
        val = fun(z)
        if val < best["val"]:
            best = {"val": val, "z": z}
    for k in range(n_refine):
        z0 = best["z"] + (0.0 if k == 0 else rng.normal(0, 0.4, 2 * n - 1))
        res = minimize(fun, z0, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-12})
        if res.fun < best["val"]:
            best = {"val": res.fun, "z": res.x}

    lams, ps = unpack_coxian(best["z"], n)
    rates, coeffs = coxian_modes(lams, ps)
    got = moment_values(rates, coeffs, kmax)[1:]
    return {
        "rho": rho_of_modes(rates, coeffs, tau_fine, tau_coarse, grid),
        "lams": lams,
        "ps": ps,
        "moment_error": float(np.max(np.abs((got - targets) / np.abs(targets)))),
        "certification": "multistart lower bound",
    }
