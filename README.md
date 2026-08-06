# Finite-resolution Fisher realizability bounds for noisy quantum transport

Reproducible numerical package for a finite-resolution Fisher realizability
witness in a noisy two-node quantum-transport dimer: a detector-bandwidth
comparison that certifies a lower bound on the classical (phase-type) order
needed to reproduce a coherently-driven transport record, and a finite-sample
statistic that makes the certificate testable from data.

## What the witness measures

A coherent two-site transport dimer (tunneling `theta`, drain rate
`Gamma_drain`, optional local pure dephasing `gamma_phi`) produces a
renewal record of drain-detection waiting times. Passing that record
through a detector modeled as a first-order lowpass of time constant `tau`
gives a Fisher information about `theta` at any chosen bandwidth. Comparing
the Fisher information at a fine bandwidth (`tau_fine`) to a coarse one
(`tau_coarse`) gives a ratio

`rho_quantum = I_fine / I_coarse`.

Because the coarse record is a data-processing degradation of the fine one,
`rho_quantum >= 1` always. The question the witness answers is how large a
classical (memoryless-state, i.e. phase-type) renewal process would have to
be — how many internal states, order `n` — to reproduce the same ratio,
given only a matched summary of the record's moments. If the classical
supremum over order-`n` models falls below `rho_quantum`, no order-`n`
classical process can explain the record: order `n` is falsified, and the
classical memory needed is at least `n+1`.

## Results

### 1. The original PH2 gate

At fixed `theta=1`, `Gamma_drain=0.1`, `tau_fine=0.2`, and `tau_coarse=8`,
adding unobserved local pure dephasing preserves a summary-matched order-2
phase-type (PH2) violation on a nonzero interval:

| `gamma_phi` | `I_fine / B2` |
|---:|---:|
| 0 | 5.2403 |
| 0.01 | 3.8103 |
| 0.03 | 3.2490 |
| 0.10 | 2.3724 |
| 0.30 | 0.9962 |

`B2 = rho_PH2_endpoint(m1, m2) * I_coarse`, recomputed at every `gamma_phi`
from that point's own summary. The T2 endpoint-maximizer proposition this
uses — that the moment-matched order-2 phase-type ratio is maximal at the
pure hypoexponential endpoint — has since been checked directly: the
moment-matched set is exactly one-parameter, and a full scan of it confirms
the endpoint is the maximum (`tests/test_fisher_kernel.py`,
`tests/test_ph_classes.py`).

### 2. Which classical order is falsified

Matching more moments tightens the classical comparison class. `n` weights
against `1 + K` linear constraints (unit mass plus `K` moments) is square at
`K = n-1`, leaving one degree of freedom the classical supremum can exploit;
at `K = n` the system is over-determined by one, which collapses the
feasible classical models onto a lower-dimensional set and sharply tightens
the bound.

| matched moments | order falsified | violation | certification |
|---|---:|---:|---|
| `(m1)` | 2 | — | exhaustive (one-parameter family, scanned in full) |
| `(m1, m2, m3)` | 3 | 5.5 | lower estimate of an upper-bound search |
| `(m1, m2, m3, m4)` | 4 | 4.7 | lower estimate of an upper-bound search |

Full detail, including the consistency check between the relaxed
(upper-bound) and direct (lower-bound) searches at each order, is in
`results/order_scan.json`.

**What this does not claim.** The O'Cinneide cone bound gives the classical
order needed to reproduce the record's waiting-time density *exactly*:
`~63` at `gamma_phi=0`, scaling as `~2*pi*theta/(Gamma+2*gamma_phi)`. A
witness that only matches a handful of moments and one Fisher ratio
certifies far less than exact reproduction requires — the order it reaches
here is single-digit, set by how many moments are matched, not by the
density's true complexity. This package does not claim the `~63` figure as
a result; `results/order_scan.json` records why not.

### 3. Making it testable from finite data

`numerics/finite_sample.py` and `numerics/finite_stats_driver.py` turn the
order-3, `K=3` certificate into a one-sided statistical claim from simulated
event data:

- **Sampler.** The detector state in the augmented generator resets after
  every detection, so the observed interval is exactly
  `T_obs = T_true + Exp(tau)` per event, independently — no detector-state
  simulation or quantum-jump process is needed (the dimer generator has
  negative off-diagonal entries and is not a Markov generator, so no jump
  chain exists for it).
- **Fine/coarse coupling.** The coarse record is built from the *same* fine
  draws via the exact identity `Exp(tau_coarse) =d Exp(tau_fine) +
  Bernoulli(1 - tau_fine/tau_coarse)*Exp(tau_coarse)`, derived by matching
  the two detector kernels' Laplace transforms.
- **Fisher surrogate.** A two-point variational chi-squared estimator, with
  sample splitting to remove plug-in bias, recovers the theoretical `I_fine`
  in a delta-ladder plateau once its basis includes damped harmonics of the
  record's natural oscillation frequency — a pure decay-rate basis misses
  essentially all of it, since `theta` mostly drives phase sensitivity, not
  envelope sensitivity.
- **One-sided bound.** The classical bound is evaluated at the worst point
  of a bootstrap confidence region for the (deconvolved) moments, not at
  the point estimate, so the reported violation is a genuine lower
  confidence bound rather than a plug-in number.

| events `N` | certified violation (95% CL) | certifies? |
|---:|---:|:---:|
| 1,000 | 0.0 | no |
| 3,000 | 2.89 | yes |
| 10,000 | 2.41 | yes |
| 30,000 | 2.76 | yes |
| 100,000 | 3.43 | yes |
| 300,000 | 3.22 | yes |

Full detail, including two explicitly unresolved simplifications (`I_coarse`
held at its theoretical value rather than estimated, and the classical
bound evaluated at a fixed pole layout rather than re-searched at every
point of the confidence region), is in `results/finite_sample.json`.

## Contents

- `numerics/transport_native_perturbation_gate.py` — the original PH2 gate;
  frozen as a regression anchor, never edited.
- `numerics/fisher_kernel.py` — numerically stable detector-convolution and
  generalized-eigenvalue kernel shared by everything below.
- `numerics/ph_classes.py` — order-`n` classical comparison classes: the
  relaxed (upper-bound) and direct (lower-bound, genuine Coxian) searches.
- `numerics/order_scan.py` — critical-order scan driver.
- `numerics/finite_sample.py` — sampler, fine/coarse coupling, variational
  chi-squared, moment deconvolution and bootstrap, worst-case bound.
- `numerics/finite_stats_driver.py` — the N-vs-significance driver.
- `results/transport_native_dephasing_gate.json` — PH2 gate output.
- `results/order_scan.json` — critical-order scan output.
- `results/finite_sample.json` — finite-sample certificate output.
- `results/TRANSPORT_NATIVE_DEPHASING_GATE.md` — concise interpretation of
  the PH2 gate.
- `tests/` — regression, analytic-limit, and consistency tests (`pytest`).
- `requirements.txt` — Python dependencies.

## Reproduce

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python numerics/transport_native_perturbation_gate.py   # regenerates the PH2 gate JSON
python numerics/order_scan.py                            # critical-order scan (slow: multistart search)
python numerics/finite_stats_driver.py                   # N-vs-significance driver
```

`transport_native_perturbation_gate.py` overwrites
`results/transport_native_dephasing_gate.json` with the recalculated
values; its output is bit-for-bit stable and covered by a regression test.

## Method notes

- Detector convolution is evaluated through entire functions (a series
  switch keyed on the argument, not the naive `1/(1-rate*tau)` form), which
  removes a numerical instability in the original PH2 code: its finite-
  difference fallback near `rate = 1/tau` carried up to `7e6` relative error,
  harmless only because the PH2 endpoint's poles happen to avoid that
  region.
- Quadrature panel width is set from the poles actually present rather
  than a fixed layout tuned for two well-separated real poles; the fixed
  layout under-resolves the fast modes of higher-order densities and was
  the source of an early, since-corrected, false conclusion that the
  witness died at order 4.
- Positivity of a candidate classical density is decided by isolating every
  real root of the exponential sum (a sum of `n` terms has at most `n-1`
  real zeros), not by sampling on a grid — a sampled check was found to
  miss a narrow negative dip and report a spurious ratio of `2.5e10`.
- The classical comparison always maximizes a functional that is an upper
  bound on the true supremum; an incomplete search under-reports it. Every
  such figure in `results/*.json` is labeled with its certification level,
  and a genuine lower-bound search (direct Coxian multistart) is used
  alongside it as a consistency check wherever feasible.

## Scope and caveats

- The order-2 result is exhaustively verified; results at order 3 and above
  come from multistart searches and are lower estimates of the true
  classical bound, not proofs. `results/order_scan.json` records where the
  two search directions (relaxed upper bound vs. direct lower bound)
  disagree and by how much.
- The finite-sample certificate at order 3 currently assumes `I_coarse` is
  known independently (e.g. from detector calibration) rather than
  estimated from the same data, and approximates the worst-case classical
  bound at a fixed pole layout rather than a full per-point re-search.
  Both are recorded as open items in `results/finite_sample.json`.
- This package does not yet address reverse transport, nonrenewal records,
  or arbitrary classical order beyond what is explicitly scanned.
