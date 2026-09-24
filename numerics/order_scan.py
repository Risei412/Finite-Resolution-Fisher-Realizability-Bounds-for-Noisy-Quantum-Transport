#!/usr/bin/env python3
"""Critical-order scan for the finite-resolution witness.

For each classical order n and each number of matched moments K, reports

    rho_relaxed(n, K)   upper bound on the classical supremum
    rho_direct(n, K)    lower bound (multistart over genuine Coxian models)

against the quantum ratio.  The witness is certified at order n only when
rho_relaxed < rho_quantum; rho_direct is reported alongside to show how loose
the relaxation is, and must never be mistaken for the bound.

Writes results/order_scan.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ph_classes as pc  # noqa: E402
import transport_native_perturbation_gate as frozen  # noqa: E402

X0 = np.array([1.0, 0.0, 0.0])
SURVIVAL = np.array([1.0, 1.0, 0.0])
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "order_scan.json"


def quantum_summary(gamma_phi, kmax):
    a, _ = frozen.quantum_generator(frozen.THETA, gamma_phi)
    moments = pc.quantum_moments_k(a, X0, SURVIVAL, kmax)
    fine, _, _ = frozen.quantum_fisher(frozen.THETA, gamma_phi, frozen.TAU_FINE)
    coarse, _, _ = frozen.quantum_fisher(frozen.THETA, gamma_phi, frozen.TAU_COARSE)
    return {"gamma_phi": gamma_phi, "moments": moments.tolist(),
            "I_fine": fine, "I_coarse": coarse, "rho_quantum": fine / coarse}


def monotone_envelope(raw):
    """PH_n is nested at FIXED K, so rho_n cannot decrease in n.

    A drop means the search at that order failed to converge, which would
    understate the classical bound and overstate the violation.  The envelope
    is applied but the raw value is always kept alongside it.

    The nesting does not hold across K, and the envelope must never be taken
    over it: raising K adds a constraint on the same n weights, so a class that
    was non-empty at K can be empty at K+1.  Order-2 matching (m1, m2) is a
    one-parameter family, while order-2 matching (m1, m2, m3) is four
    constraints on two weights and generically empty.  rho at (n=3, K=3) coming
    out below rho at (n=2, K=2) is therefore expected, not a convergence bug.
    """
    out, running = [], -np.inf
    for value in raw:
        if value is None:
            out.append(None)
            continue
        running = max(running, value)
        out.append(running)
    return out


def scan(gamma_phi, orders, k_values, budget, seed):
    kmax = max(k_values)
    quantum = quantum_summary(gamma_phi, kmax)
    rho_q = quantum["rho_quantum"]
    rows = []
    for k in k_values:
        targets = np.asarray(quantum["moments"][:k])
        raw_relaxed = []
        for n in orders:
            t0 = time.time()
            relaxed = pc.rho_relaxed(n, targets, frozen.TAU_FINE, frozen.TAU_COARSE,
                                     seed=seed, n_random=budget,
                                     n_refine=max(4, budget // 100))
            direct = pc.rho_direct(n, targets, frozen.TAU_FINE, frozen.TAU_COARSE,
                                   seed=seed, n_random=budget,
                                   n_refine=max(4, budget // 100))
            row = {
                "K": k, "n": n,
                "rho_relaxed_raw": None if relaxed is None else relaxed["rho"],
                "rho_direct": None if direct is None else direct["rho"],
                "relaxed_layout": None if relaxed is None else list(relaxed["layout"]),
                "relaxed_density_minimum": None if relaxed is None
                else relaxed["density_minimum"],
                "direct_moment_error": None if direct is None else direct["moment_error"],
                "class_empty": relaxed is None,
                "seconds": round(time.time() - t0, 1),
            }
            raw_relaxed.append(row["rho_relaxed_raw"])
            rows.append(row)
            print(json.dumps(row), flush=True)

        env = monotone_envelope(raw_relaxed)
        for row, enveloped in zip([r for r in rows if r["K"] == k], env):
            row["rho_relaxed"] = enveloped
            row["envelope_lifted"] = (enveloped is not None
                                      and row["rho_relaxed_raw"] is not None
                                      and enveloped > row["rho_relaxed_raw"] + 1e-9)
            row["certifies_witness"] = (enveloped is not None and enveloped < rho_q)
            row["violation"] = None if not enveloped else rho_q / enveloped
    return quantum, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma-phi", type=float, default=0.0)
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    quantum, rows = scan(args.gamma_phi, args.orders, args.k, args.budget, args.seed)
    certified = [r["n"] for r in rows if r["certifies_witness"]]
    payload = {
        "scan": "CRITICAL_ORDER",
        "fixed": {"theta": frozen.THETA, "Gamma_drain": frozen.GAMMA_DRAIN,
                  "tau_fine": frozen.TAU_FINE, "tau_coarse": frozen.TAU_COARSE},
        "quantum": quantum,
        "rows": rows,
        "certification": {
            "rho_relaxed": "upper bound on the classical supremum; the search "
                           "maximises it, so an incomplete search under-reports",
            "rho_direct": "multistart lower bound; NOT usable as a bound",
        },
        "max_certified_order": max(certified) if certified else None,
        "settings": vars(args),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"max_certified_order": payload["max_certified_order"]}, indent=2))


if __name__ == "__main__":
    main()
