# Transport-native dephasing gate

Reproducible numerical package for the finite-resolution Fisher
realizability witness in a noisy two-node quantum-transport dimer.

## Result

At fixed `theta=1`, `Gamma_drain=0.1`, `tau_fine=0.2`, and
`tau_coarse=8`, adding unobserved local pure dephasing preserves a
summary-matched order-2 phase-type (PH2) violation on a nonzero interval.

| `gamma_phi` | `I_fine / B2` |
|---:|---:|
| 0 | 5.2403 |
| 0.01 | 3.8103 |
| 0.03 | 3.2490 |
| 0.10 | 2.3724 |
| 0.30 | 0.9962 |

The conclusion is `PASS_OPEN_DEPHASING_SET`, supporting the decision to
create a dedicated repository for finite-resolution Fisher realizability
bounds in noisy quantum transport.

## Contents

- `numerics/transport_native_perturbation_gate.py` — calculation and result generation.
- `results/transport_native_dephasing_gate.json` — machine-readable output.
- `results/TRANSPORT_NATIVE_DEPHASING_GATE.md` — concise interpretation.
- `requirements.txt` — Python dependencies.

## Reproduce

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python numerics/transport_native_perturbation_gate.py
```

The script overwrites `results/transport_native_dephasing_gate.json` with
the recalculated values.

## Method

For every dephasing rate the program recomputes
`S = (m1, m2, I_coarse)` and compares the quantum fine-resolution Fisher
information with the corresponding PH2 endpoint bound:

`B2 = rho_PH2_endpoint(m1, m2) * I_coarse`.

Thus the reported violation is not a comparison to a bound frozen at the
unperturbed point.  Detector convolution is evaluated through an augmented
linear generator, Fisher tangents through Frechet derivatives of matrix
exponentials, and integration through panelized Gauss-Legendre quadrature.

## Scope and caveat

The PH2 bound uses the existing T2 endpoint-maximizer proposition. Its global
monotonicity over the complete moment-matched PH2 interval remains
computer-certified rather than analytically proved. This package does not yet
address reverse transport, nonrenewal records, or arbitrary classical order.
