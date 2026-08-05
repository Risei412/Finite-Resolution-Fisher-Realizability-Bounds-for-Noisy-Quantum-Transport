# Transport-native dephasing gate

## Decision

**PASS / CREATE dedicated repository.**

The coherent two-node transport dimer was extended by unobserved local pure
dephasing. This breaks the exact resonance-fluorescence pullback while
preserving the finite-resolution PH2 violation on a nonzero interval.

Fixed parameters: `theta=1`, `Gamma_drain=0.1`, `tau_fine=0.2`,
`tau_coarse=8`. Here `gamma_phi` denotes the off-diagonal coherence-decay
rate. At every point, moments and coarse Fisher information were recomputed,
and the PH2 bound was reevaluated at that point's summary.

| gamma_phi | I_fine | I_coarse | PH2 bound | violation |
|---:|---:|---:|---:|---:|
| 0 | 645.215 | 4.11324 | 123.125 | 5.2403 |
| 0.001 | 559.639 | 3.88904 | 116.251 | 4.8141 |
| 0.003 | 457.361 | 3.48922 | 104.009 | 4.3973 |
| 0.01 | 277.650 | 2.46822 | 72.8681 | 3.8103 |
| 0.03 | 105.861 | 1.13339 | 32.5822 | 3.2490 |
| 0.1 | 15.2598 | 0.243301 | 6.43222 | 2.3724 |
| 0.3 | 1.28191 | 0.0585901 | 1.28683 | 0.9962 |
| 1.0 | 0.193964 | 0.0700998 | 1.05829 | 0.1833 |

The pass is not a frozen-bound comparison. For each `gamma_phi`, the script
uses its new `(m1,m2,I_coarse)` and evaluates

`B2 = rho_PH2_endpoint(m1,m2) * I_coarse`.

The unperturbed point reproduces the existing QF-0/T2 values. Quadrature-order
variation (12 to 20 Gauss nodes per narrow panel) changes the reported ratios
by at most `5.6e-5` at `gamma_phi=0` and `4.4e-7` at `gamma_phi=0.1`.

## Scope

This establishes a transport-native witness family, not merely a basis rename:
the unobserved dephasing Liouvillian has no counterpart in the original
self-reset pure-state waiting model. It does not yet prove robustness to
finite reverse bias, nonrenewal records, or arbitrary classical order.

The PH2 value uses T2's endpoint-maximizer proposition. Its monotonicity over
the complete moment-matched PH2 interval remains computer-certified rather
than analytically proved.

