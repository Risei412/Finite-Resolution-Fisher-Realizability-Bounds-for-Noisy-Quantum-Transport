#!/usr/bin/env python3
"""Numerically stable kernel for finite-resolution Fisher ratios.

The witness compares the Fisher information carried by a renewal record at two
detector bandwidths.  For a density built from exponential modes the tangent
space is spanned by the detector-convolved modes and their pole derivatives,
and the ratio

    rho = max_v  (v F_fine v) / (v F_coarse v)

is a generalized eigenvalue problem.  Forming the Gram matrices explicitly
squares their condition number, and F_coarse is intrinsically ill-conditioned:
the coarse detector suppresses high-frequency tangents by 1/(1+omega^2 tau^2),
so its small eigenvalues are physical, not noise.  Everything here is therefore
written to avoid ever forming a Gram matrix, and to avoid the cancellation-prone
expressions that the two-pole implementation could get away with.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import qr, solve_triangular, svdvals

# --------------------------------------------------------------------------
# quadrature (identical panel layout to the PH2 implementation)
# --------------------------------------------------------------------------


def panel_quadrature(edges, order=16):
    x0, w0 = leggauss(order)
    edges = np.asarray(edges, dtype=float)
    left, right = edges[:-1], edges[1:]
    half, mid = 0.5 * (right - left), 0.5 * (right + left)
    xs = (half[:, None] * x0[None, :] + mid[:, None]).ravel()
    ws = (half[:, None] * w0[None, :]).ravel()
    return xs, ws


def quadrature(order=16):
    # Panels remain narrower than the coherent period; wide logarithmic tail
    # panels bias the Fisher integral even when they integrate the mass well.
    edges = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2]
    edges += list(np.arange(1.0, 201.0, 1.0))
    edges += list(np.arange(205.0, 505.0, 5.0))
    edges += list(np.arange(525.0, 1501.0, 25.0))
    return panel_quadrature(edges, order)


MAX_PANELS = 20000


def adaptive_quadrature(rates, taus, order=16, per_period=8, per_decay=6,
                        tail_decades=45.0, fast_decays=30.0, detector_decays=12.0,
                        max_panels=MAX_PANELS):
    """Panel layout driven by the poles actually present.

    The fixed layout above was tuned for two well-separated real poles.  With
    more modes the Fisher integrand b_i b_j / p is set by the *fastest* pole
    while the density in the denominator decays at the *slowest*, so the ratio
    is sharply peaked near the origin on a scale the fixed panels never see.

    Each feature imposes a panel width only over the interval where it is still
    alive: a detector of constant tau over [0, 12 tau], the fastest pole and the
    fastest oscillation over [0, 30/max Re], and the slowest pole everywhere.
    Because every constraint's range is proportional to its own width, the panel
    count is bounded by the resolution settings alone and cannot blow up when
    the poles happen to span many decades -- a naive single-zone layout reaches
    4e5 nodes on ordinary draws from the pole search.
    """
    rates = np.asarray(rates, dtype=complex)
    re, im = np.real(rates), np.abs(np.imag(rates))
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    if not np.all(np.isfinite(rates)) or np.any(re <= 0):
        raise ValueError("all rates must be finite with positive real part")

    # the convolved modes carry the detector poles 1/tau as well as the density's
    fast_re = re.max()
    slow_re = min(re.min(), (1.0 / taus).min())
    t_max = tail_decades / slow_re
    if not np.isfinite(t_max):
        # a rate underflowed to zero: the integration range is unbounded and the
        # panel walk below would never terminate
        raise ValueError("integration range is not finite; a rate underflowed")

    # (range, width) pairs; each is active on [0, range)
    limits = [(detector_decays * tau, tau / per_decay) for tau in taus]
    limits.append((fast_decays / fast_re, 1.0 / (per_decay * fast_re)))
    if im.max() > 0:
        limits.append((fast_decays / fast_re, 2.0 * np.pi / (per_period * im.max())))
    h_slow = 1.0 / (per_decay * slow_re)

    edges, t = [0.0], 0.0
    while t < t_max:
        h = min([h_slow] + [w for r, w in limits if t < r])
        if not (h > 0.0):
            raise ValueError("panel width underflowed")
        t = min(t + h, t_max)
        edges.append(t)
        if len(edges) > max_panels:
            raise ValueError(f"panel count exceeded {max_panels}")
    return panel_quadrature(np.asarray(edges), order)


# --------------------------------------------------------------------------
# entire functions phi0, phi1 -- the whole point is that these are smooth at 0
# --------------------------------------------------------------------------

_SERIES_CUT = 1e-2


def phi0_series(z):
    """expm1(z)/z near z=0."""
    return 1.0 + z / 2.0 + z**2 / 6.0 + z**3 / 24.0 + z**4 / 120.0


def phi1_series(z):
    """(z e^z - expm1(z))/z^2 near z=0."""
    return 0.5 + z / 3.0 + z**2 / 8.0 + z**3 / 30.0 + z**4 / 144.0


# --------------------------------------------------------------------------
# detector-convolved exponential modes
# --------------------------------------------------------------------------
#
#   conv0(r) = filter * exp(-r t)          (integral 1/r)
#   conv1(r) = filter * t exp(-r t)        (integral 1/r^2, equals -d/dr conv0)
#
# with filter(t) = exp(-t/tau)/tau.  Both are valid for complex r.
#
# In terms of z = (1 - r tau) t / tau these are exp(-t/tau)(t/tau) phi0(z) and
# exp(-t/tau)(t^2/tau) phi1(z), which exposes the r -> 1/tau singularity as
# removable.  That grouping cannot be evaluated directly: z reaches ~5e3 on the
# fine detector, so exp(z) overflows before exp(-t/tau) can damp it.  Instead
# the two exponentials are kept separate -- exp(-t/tau) exp(z) = exp(-r t) --
# and the series is used only where the difference actually cancels.  The
# branch is keyed on |z|, which is exactly where the cancellation lives; the
# large-|z| form divides by z^2 rather than d^2, so a small d never amplifies.


def _branch(z, small_val, big_val):
    return np.where(np.abs(z) < _SERIES_CUT, small_val, big_val)


def conv0(rate, t, tau):
    t = np.asarray(t, dtype=float)
    z = (1.0 - rate * tau) * t / tau
    zsafe = np.where(np.abs(z) < _SERIES_CUT, 1.0, z)
    decay, filt = np.exp(-rate * t), np.exp(-t / tau)
    return _branch(z,
                   filt * (t / tau) * phi0_series(np.where(np.abs(z) < _SERIES_CUT, z, 0.0)),
                   (t / tau) * (decay - filt) / zsafe)


def conv1(rate, t, tau):
    t = np.asarray(t, dtype=float)
    z = (1.0 - rate * tau) * t / tau
    zsafe = np.where(np.abs(z) < _SERIES_CUT, 1.0, z)
    decay, filt = np.exp(-rate * t), np.exp(-t / tau)
    return _branch(z,
                   filt * (t**2 / tau) * phi1_series(np.where(np.abs(z) < _SERIES_CUT, z, 0.0)),
                   (t**2 / tau) * ((zsafe - 1.0) * decay + filt) / zsafe**2)


def mode_masses(rates):
    """Integrals of conv0 and conv1 for each rate: 1/r and 1/r^2."""
    rates = np.asarray(rates)
    return np.concatenate([1.0 / rates, 1.0 / rates**2])


# --------------------------------------------------------------------------
# real tangent basis
# --------------------------------------------------------------------------


def real_mode_basis(rates, t, tau):
    """Real basis for span{conv0(r_k), conv1(r_k)} and its mass functional.

    `rates` must be closed under conjugation.  A conjugate pair {r, conj(r)}
    contributes the four real functions Re/Im conv0(r) and Re/Im conv1(r),
    which span the same real subspace as the two complex pairs.

    Returns (basis, mass) with basis of shape (len(t), 2*len(rates)).
    """
    rates = np.asarray(rates, dtype=complex)
    cols, mass = [], []
    used = np.zeros(len(rates), dtype=bool)
    for k, r in enumerate(rates):
        if used[k]:
            continue
        used[k] = True
        if abs(r.imag) < 1e-14 * max(1.0, abs(r.real)):
            rr = r.real
            cols += [np.real(conv0(rr, t, tau)), np.real(conv1(rr, t, tau))]
            mass += [1.0 / rr, 1.0 / rr**2]
            continue
        # consume the partner so the pair is emitted exactly once
        partner = np.argmin(np.abs(rates - np.conj(r)) + np.where(used, 1e30, 0.0))
        used[partner] = True
        c0, c1 = conv0(r, t, tau), conv1(r, t, tau)
        cols += [np.real(c0), np.imag(c0), np.real(c1), np.imag(c1)]
        mass += [np.real(1.0 / r), np.imag(1.0 / r),
                 np.real(1.0 / r**2), np.imag(1.0 / r**2)]
    return np.column_stack(cols), np.asarray(mass, dtype=float)


def mass_preserving_null(mass):
    """Orthonormal basis of {c : mass . c = 0}, via a single Householder QR.

    Using a full QR of the mass vector keeps the projection orthogonal, so the
    subsequent basis combination cannot amplify roundoff the way an explicit
    difference construction does.
    """
    q, _ = qr(np.asarray(mass, dtype=float)[:, None], mode="full")
    return q[:, 1:]


def mode_density(coeffs, rates, t, tau):
    """Detector-convolved density  sum_k c_k conv0(r_k), taken real."""
    rates = np.asarray(rates, dtype=complex)
    coeffs = np.asarray(coeffs, dtype=complex)
    acc = np.zeros(len(t), dtype=complex)
    for c, r in zip(coeffs, rates):
        acc += c * conv0(r, t, tau)
    return np.real(acc)


# --------------------------------------------------------------------------
# the generalized ratio, without ever forming a Gram matrix
# --------------------------------------------------------------------------


def fisher_ratio(basis_fine, dens_fine, basis_coarse, dens_coarse, ws,
                 floor=0.0, rank_tol=1e-10):
    """max_v |A_fine v|^2 / |A_coarse v|^2 for A_tau = sqrt(w/p) * basis.

    Solved as a generalized singular value problem: a column-pivoted QR of the
    coarse design matrix supplies the whitening, and the ratio is the largest
    singular value of A_fine R^-1.  Working on the design matrices rather than
    their Grams halves the condition-number exponent, and the pivoted QR gives
    a rank decision rather than an eigenvalue cutoff.

    `floor` drops quadrature nodes where the density has fallen below
    floor*max(density); it is a convergence knob and must be reported.
    """
    def design(basis, dens):
        keep = dens > max(floor * dens.max(), 1e-300)
        scale = np.sqrt(ws[keep] / dens[keep])
        return basis[keep] * scale[:, None]

    a_f = design(basis_fine, dens_fine)
    a_c = design(basis_coarse, dens_coarse)

    q, r, piv = qr(a_c, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if diag[0] <= 0.0:
        return np.nan, 0
    rank = int(np.sum(diag > rank_tol * diag[0]))
    r = r[:rank, :rank]
    # X = A_fine[:, piv[:rank]] @ inv(R)  <=>  R^T X^T = A_fine[:, piv[:rank]]^T
    xt = solve_triangular(r, a_f[:, piv[:rank]].T, trans="T", lower=False)
    return float(svdvals(xt)[0] ** 2), rank


def rho_from_modes(rates, coeffs, tau_fine, tau_coarse, xs, ws,
                   floor=0.0, rank_tol=1e-10):
    """Finite-resolution Fisher ratio for the density sum_k c_k exp(-r_k t).

    The tangent space is the mass-preserving subspace of
    span{conv0(r_k), conv1(r_k)}, of dimension 2n-1.
    """
    b_f, mass = real_mode_basis(rates, xs, tau_fine)
    b_c, _ = real_mode_basis(rates, xs, tau_coarse)
    null = mass_preserving_null(mass)
    d_f = mode_density(coeffs, rates, xs, tau_fine)
    d_c = mode_density(coeffs, rates, xs, tau_coarse)
    return fisher_ratio(b_f @ null, d_f, b_c @ null, d_c, ws,
                        floor=floor, rank_tol=rank_tol)
