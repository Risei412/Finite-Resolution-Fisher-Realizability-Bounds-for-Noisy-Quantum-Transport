#!/usr/bin/env python3
"""Transport-native dephasing gate for the finite-resolution PH2 witness.

Adds unobserved local pure dephasing to the coherent two-node transport dimer,
recomputes its summary S=(m1,m2,I_coarse), and evaluates the summary-matched
order-2 phase-type bound at the T2 hypoexponential maximizer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm, expm_frechet, eigh

THETA = 1.0
GAMMA_DRAIN = 0.1
TAU_FINE = 0.2
TAU_COARSE = 8.0
DEPHASING_RATES = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def quadrature(order=16):
    # Panels remain narrower than the coherent period; wide logarithmic tail
    # panels bias the Fisher integral even when they integrate the mass well.
    edges = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2]
    edges += list(np.arange(1.0, 201.0, 1.0))
    edges += list(np.arange(205.0, 505.0, 5.0))
    edges += list(np.arange(525.0, 1501.0, 25.0))
    x0, w0 = leggauss(order)
    xs, ws = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        xs.extend(0.5 * (right-left) * x0 + 0.5 * (right+left))
        ws.extend(0.5 * (right-left) * w0)
    return np.asarray(xs), np.asarray(ws)


def quantum_generator(theta, gamma_phi, tau=None):
    # Column state (rho_LL, rho_RR, Im rho_LR [, detector population]).
    decay = GAMMA_DRAIN / 2.0 + gamma_phi
    a = np.array([[0.0, 0.0, -theta],
                  [0.0, -GAMMA_DRAIN, theta],
                  [theta/2.0, -theta/2.0, -decay]])
    da = np.array([[0.0, 0.0, -1.0],
                   [0.0, 0.0, 1.0],
                   [0.5, -0.5, 0.0]])
    if tau is None:
        return a, da
    g = np.zeros((4, 4)); dg = np.zeros((4, 4))
    g[:3, :3] = a; dg[:3, :3] = da
    g[3, 1] = GAMMA_DRAIN; g[3, 3] = -1.0/tau
    return g, dg


def quantum_moments(theta, gamma_phi):
    a, _ = quantum_generator(theta, gamma_phi)
    x0 = np.array([1.0, 0.0, 0.0])
    survival = np.array([1.0, 1.0, 0.0])
    inv = np.linalg.inv(-a)
    return float(survival @ inv @ x0), float(2.0 * survival @ inv @ inv @ x0)


def quantum_fisher(theta, gamma_phi, tau, order=16):
    g, dg = quantum_generator(theta, gamma_phi, tau)
    x0 = np.array([1.0, 0.0, 0.0, 0.0])
    read = np.array([0.0, 0.0, 0.0, 1.0/tau])
    xs, ws = quadrature(order)
    total = mass = tangent_mass = 0.0
    for t, wt in zip(xs, ws):
        e = expm(g*t)
        p = float(read @ e @ x0)
        dp = float(read @ expm_frechet(g*t, dg*t, compute_expm=False) @ x0)
        mass += wt*p; tangent_mass += wt*dp
        if p > 1e-300:
            total += wt*dp*dp/p
    return total, mass, tangent_mass


def conv0(rate, t, tau):
    d = 1.0-rate*tau
    if abs(d) < 1e-8:
        return (t/tau)*np.exp(-t/tau)
    return (np.exp(-rate*t)-np.exp(-t/tau))/d


def conv1(rate, t, tau):
    d = 1.0-rate*tau
    if abs(d) < 1e-6:
        # Stable centered derivative of conv0: C[t exp(-rate t)] = -d_rate C0.
        h = 1e-5/max(1.0, abs(rate))
        return -(conv0(rate+h, t, tau)-conv0(rate-h, t, tau))/(2*h)
    n = np.exp(-rate*t)-np.exp(-t/tau)
    return (t*np.exp(-rate*t)*d-tau*n)/(d*d)


def ph2_endpoint_ratio(m1, m2, order=16):
    cv2 = m2/m1**2 - 1.0
    if not (0.5 <= cv2 < 1.0):
        return math.nan, cv2
    root = math.sqrt(2.0*cv2-1.0)
    x, y = 0.5*m1*(1.0+root), 0.5*m1*(1.0-root)
    mu, nu = 1.0/x, 1.0/y
    amp = 1.0/(x-y)
    xs, ws = quadrature(order)

    def gram(tau):
        c_mu = conv0(mu, xs, tau); c_nu = conv0(nu, xs, tau)
        density = amp*(c_mu-c_nu)
        basis = np.vstack([
            c_mu-(x/y)*c_nu,
            conv1(mu, xs, tau)-(x*x/y)*c_nu,
            conv1(nu, xs, tau)-y*c_nu,
        ]).T
        keep = density > 1e-300
        return np.einsum('n,ni,nj->ij', ws[keep]/density[keep], basis[keep], basis[keep])

    ff, fc = gram(TAU_FINE), gram(TAU_COARSE)
    vals, vecs = eigh(fc)
    keep = vals > 1e-11*vals[-1]
    whitening = vecs[:, keep]/np.sqrt(vals[keep])[None, :]
    reduced = whitening.T@ff@whitening
    return float(np.linalg.eigvalsh((reduced+reduced.T)/2.0)[-1]), cv2


def evaluate(gamma_phi, order=16):
    m1, m2 = quantum_moments(THETA, gamma_phi)
    fine, mf, df = quantum_fisher(THETA, gamma_phi, TAU_FINE, order)
    coarse, mc, dc = quantum_fisher(THETA, gamma_phi, TAU_COARSE, order)
    rho_bound, cv2 = ph2_endpoint_ratio(m1, m2, order)
    bound = rho_bound*coarse
    return {
        'gamma_phi': gamma_phi, 'm1': m1, 'm2': m2, 'cv2': cv2,
        'I_fine': fine, 'I_coarse': coarse,
        'rho_quantum': fine/coarse, 'rho_PH2_endpoint': rho_bound,
        'B2': bound, 'violation': fine/bound,
        'fine_mass': mf, 'coarse_mass': mc,
        'max_abs_tangent_mass': max(abs(df), abs(dc)),
    }


def main():
    rows = [evaluate(g) for g in DEPHASING_RATES]
    # Independent quadrature check at the unperturbed point and last passing point.
    passing = [r for r in rows if r['violation'] > 1.0]
    check_rates = sorted(set([0.0, passing[-1]['gamma_phi']] if passing else [0.0]))
    convergence = []
    for g in check_rates:
        a, b = evaluate(g, 12), evaluate(g, 20)
        convergence.append({'gamma_phi': g,
            'max_relative_change_12_to_20': max(abs(a[k]-b[k])/max(abs(b[k]),1e-15)
                for k in ('I_fine','I_coarse','rho_PH2_endpoint','violation'))})
    out = {
        'gate': 'TRANSPORT_NATIVE_DEPHASING_PERTURBATION',
        'model': 'coherent two-node dimer + unobserved local pure dephasing',
        'convention': 'gamma_phi is the off-diagonal coherence-decay rate',
        'fixed': {'theta': THETA, 'Gamma_drain': GAMMA_DRAIN,
                  'tau_fine': TAU_FINE, 'tau_coarse': TAU_COARSE},
        'rows': rows, 'convergence': convergence,
        'pass_definition': 'summary-matched I_fine > B2=(PH2 endpoint ratio)*I_coarse',
        'verdict': 'PASS_OPEN_DEPHASING_SET' if any(r['gamma_phi']>0 and r['violation']>1 for r in rows) else 'KILL',
        'repository_decision': 'CREATE' if sum(r['gamma_phi']>0 and r['violation']>=2 for r in rows)>=2 else 'HOLD',
        'caveat': 'Uses the T2 endpoint-maximizer proposition; its global monotonicity remains computer-certified rather than analytic.'
    }
    path = Path(__file__).resolve().parents[1]/'results'/'transport_native_dephasing_gate.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
