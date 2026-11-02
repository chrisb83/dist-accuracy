# trapezoid: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `trapezoid_gen`, lines 9952-10060
- **Implemented:** pdf, cdf, ppf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** none (triang is mathematically the case c = d but has its own code); helpers: `_lazyselect`; special functions: none
- **Reference:** `refs/trapezoid_ref.py`, 400 digits; self-checks: FAIL on the pdf-integral check only, which is a quadrature artifact (see below); all other checks pass and the reference agrees with SciPy to <= 1.33 ulp at every bulk point
- **Parameter sets:** (0.2, 0.8), (1e-200, 0.5), (1e-20, 0.9), (0.5, 0.5), (0, 1), (0.1, 1 - 1e-12), (0.3, 0.3 + 1e-9)
- **Worst finding:** loss in cdf and ppf

Note on the self-check FAIL: the harness integrates the pdf with `mp.quad`
over `[x1, x2]` without breakpoints, and those intervals contain a kink of the
piecewise-linear pdf (at c or d). Tanh-sinh quadrature over a kink converges
only to about 1e-8 (relative error 1.7e-8 for (0.2, 0.8), 2.1e-6 for
(1e-200, 0.5)). Splitting the same integrals at c and d gives relative errors
of 0 and 2e-31, so the reference pdf and cdf agree. The ppf/cdf round-trip
checks pass (<= 2e-31).

## Summary

pdf is accurate everywhere (max 1.14 ulp), and cdf and ppf are accurate in
the bulk and the right tail (max 1.33 ulp). The only problems are in the left
branch for a tiny argument: cdf forms `x**2` before dividing by `c`, and ppf
forms `q * c` before the square root. Both intermediates underflow (to
subnormal or 0) even though the result is a normal double. This affects small
c only (x < c ~ 1e-154 or less for cdf; q*c below 2.2e-308 for ppf), and in
that regime the result is 0.0 or loses up to 2e11 ulp. The inherited sf, logsf and isf lose all accuracy in the tails,
and closed forms exist for them.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | x < c, x < 1.5e-154, x**2/(c(d-c+1)) normal (needs small c) | `x**2` underflows before division by c (pattern 4) | 0.0 for 6.7e-301; 5.0e10 ulp | loss |
| F2 | ppf | q < c/(d-c+1), q*c < 2.2e-308 (needs small c) | `q * c` underflows before sqrt (pattern 4) | 0.0 for 1.2e-250; 2.1e11 ulp | loss |

### F1: cdf left branch underflows in x**2

- **Code:** `_cdf`, line 10008: `x**2 / c / (d-c+1)`
- **Mechanism:** pattern 4. `x**2` is subnormal for x < 1.49e-154 and 0 for
  x < ~2.2e-162. The following division by a tiny `c` would bring the result back into
  the normal range, but the bits are already lost. Total loss if `x**2`
  rounds to 0, and partial loss (fewer than 53 significant bits) if it is
  subnormal.
- **Regime:** x < c and x < 1.49e-154, with x**2/(c(d-c+1)) >= 2.2e-308; this
  needs c below roughly x**2/2.2e-308, i.e. tiny c (any c < ~1e-154 hits it
  for every x in (0, c) below the threshold). With c = 1e-200, every x in
  (0, c) where the cdf is normal returns 0.0.
- **Reproducer:** `python accuracy.py refs/trapezoid_ref.py --at cdf 1e-160 1e-20 0.9`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-250, 1e-200, 0.5)` | 0.0 | 6.666666666666668e-301 | inf | 2 | loss |
  | `cdf(5e-201, 1e-200, 0.5)` | 0.0 | 1.6666666666666668e-201 | inf | 2 | loss |
  | `cdf(9.99999999e-201, 1e-200, 0.5)` | 0.0 | 6.666666653333333e-201 | inf | 2 | loss |
  | `cdf(1e-170, 1e-100, 0.5)` | 0.0 | 6.666666666666666e-241 | inf | 2 | loss |
  | `cdf(1e-160, 1e-20, 0.9)` | 5.263099300961491e-301 | 5.263157894736842e-301 | 5.0e+10 | 2 | severe |
  | `cdf(1e-155, 1e-20, 0.9)` | 5.2631578947368264e-291 | 5.263157894736843e-291 | 14 | 2 | minor |

  (8 loss points for (1e-200, 0.5); cond (shapes) is 1.3-1.5, so the errors are not
  explained by sensitivity to the shapes.)
- **Fix direction:** `x * (x / (c * (d-c+1)))` (or `(x/c) * x / (d-c+1)`), so no
  intermediate is smaller than the result.

### F2: ppf left branch underflows in q * c

- **Code:** `_ppf`, line 10017: `np.sqrt(q * c * (1 + d - c))`
- **Mechanism:** pattern 4. The product `q * c` underflows to a subnormal or 0
  although its square root (the result) is around 1e-150 to 1e-250. `np.select`
  evaluates the left branch for all q, but only the left-branch points are wrong.
- **Regime:** q < qc = c/(d-c+1) and q*c < 2.2e-308. Total loss when
  q*c < 4.9e-324. This needs small c: for c = 0.2 and q = 3e-308 the error is
  only 0.76 ulp because sqrt halves the relative error of a mildly subnormal product.
  With c = 1e-200, every q in the left branch returns 0.0.
- **Reproducer:** `python accuracy.py refs/trapezoid_ref.py --at ppf 1e-300 1e-200 0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 1e-200, 0.5)` | 0.0 | 1.224744871391589e-250 | inf | 0.5 | loss |
  | `ppf(6.66666666e-201, 1e-200, 0.5)` | 0.0 | 9.999999995e-201 | inf | 0.5 | loss |
  | `ppf(1e-250, 1e-100, 0.5)` | 0.0 | 1.2247448713915891e-175 | inf | 0.5 | loss |
  | `ppf(1e-300, 1e-20, 0.9)` | 1.3784688875362527e-160 | 1.3784048752090221e-160 | 2.1e+11 | 0.5 | severe |

  (12 loss points for (1e-200, 0.5).)
- **Fix direction:** `np.sqrt(q) * np.sqrt(c * (1 + d - c))` (or
  `sqrt(q*(1+d-c)) * sqrt(c)`).

## RuntimeWarnings

- cdf: `underflow encountered in square` / `underflow encountered in divide`
  in 6 of 7 parameter sets, from `x**2` in line 10008 at tiny x. With moderate c
  the true cdf is then subnormal or near-subnormal and the value is right; e.g.
  `cdf(1e-155, 0.2, 0.8)` = 3.12499999999997e-310 (true 3.125e-310, subnormal,
  not measured). With small c the warnings coincide with F1.
- ppf: `underflow encountered in multiply` (2 parameter sets) with F2. It also
  occurs at accurate points, e.g. `ppf(3e-308, 0.2, 0.8)` (0.76 ulp).
  Because `np.select` evaluates all three branches for every q, the warning can
  come from a branch that is not selected.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, all 7 parameter sets, both sides of x = c and x = d, c = 0, d = 1 - 1e-12, c = d, d - c = 1e-9, x down to 1e-300 | 1.14 ulp |
| cdf | bulk and right tail (x up to 1 - 1e-15, d close to 1), both sides of c and d, all parameter sets; left tail for moderate c | 1.33 ulp (outside F1) |
| ppf | bulk and q up to 1 - 1e-15 (right branch `1 - sqrt((1-q)...)` is within conditioning), both sides of qc and qd, q down to 1e-300 for moderate c | 0.992 ulp (outside F2) |

## Suspected, not confirmed

none. All hypotheses in `scratch/trapezoid/notes.md` were either confirmed
(F1, F2) or covered and found fine (branch thresholds, d near 1, c = 0, c = d,
ppf right branch near q = 1).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | x > d: `(1-x)**2 / ((d-c+1)*(1-d))`; c <= x <= d: `(1 + d - 2x)/(d-c+1)`; x < c: `1 - x**2/(c(d-c+1))` | right tail: `sf(1-1e-10, 0.2, 0.8)` = 0.0 vs 3.1250005171273408e-20 (loss); total loss once sf < ~1.1e-16 |
| logsf | log(sf) | `log` of the closed-form sf above (`2*log1p(-x) - log((d-c+1)*(1-d))` for x > d); `log1p(-cdf)` for x < c | right tail: `logsf(1-1e-10, 0.2, 0.8)` = -inf vs -44.912267411211814 |
| logcdf | log(cdf) | x < c: `2*log(x) - log(c) - log(d-c+1)`; x > d: `log1p(-sf)` | left tail where cdf underflows or is subnormal: `logcdf(1e-160, 0.2, 0.8)` = -735.6878066077855 vs -735.68779547490625 (6.8e7 ulp; cdf = x**2/... is subnormal); -inf once cdf = 0 (x < ~1e-162) |
| logpdf | log(pdf) | `log(u) + log(x) - log(c)`, `log(u) + log1p(-x) - log1p(-d)` | only where pdf is subnormal/0 (x < ~1e-308 * c); `logpdf(1e-310, 0.2, 0.8)` is still 0.11 ulp. Low priority |
| isf | ppf(1 - q) | x > d branch: `1 - sqrt(q*(d-c+1)*(1-d))` for q < 1 - qd; middle: `(c + (1-q)(d-c+1))/2`; left: `sqrt((1-q)*c*(d-c+1))` | small q: `1 - q` rounds away q. `isf(1e-20, 0.2, 0.8)` = 1.0 vs 0.99999999994343146 (2.5e5 ulp) |

## Shared code

none. `_lazyselect` is used correctly. F1 and F2 are local expressions.

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e-250, 1e-200, 0.5)` | 0.0 | 6.6666666666666675e-301 | verified | loss; reference = `mp.quad` of the docstring pdf (rescaled to [0,1], normalised integrand), 100 vs 200 digits agree to 30 digits |
| F1 | `cdf(5e-201, 1e-200, 0.5)` | 0.0 | 1.6666666666666666e-201 | verified | loss |
| F1 | `cdf(9.99999999e-201, 1e-200, 0.5)` | 0.0 | 6.6666666533333339e-201 | verified | loss |
| F1 | `cdf(1e-170, 1e-100, 0.5)` | 0.0 | 6.6666666666666663e-241 | verified | loss |
| F1 | `cdf(1e-160, 1e-20, 0.9)` | 5.263099300961491e-301 | 5.2631578947368422e-301 | verified | 5.01e10 ulp (severe), as reported |
| F1 | `cdf(1e-155, 1e-20, 0.9)` | 5.2631578947368264e-291 | 5.2631578947368425e-291 | verified | 13.8 ulp (minor), as reported. Mechanism matches line 10008 `x**2 / c / (d-c+1)` |
| F2 | `ppf(1e-300, 1e-200, 0.5)` | 0.0 | 1.2247448713915891e-250 | verified | loss; reference = bisection on the quadrature cdf, cdf(x_ref)/q - 1 = 0 at 100 digits |
| F2 | `ppf(6.66666666e-201, 1e-200, 0.5)` | 0.0 | 9.9999999950000001e-201 | verified | loss |
| F2 | `ppf(1e-250, 1e-100, 0.5)` | 0.0 | 1.2247448713915891e-175 | verified | loss |
| F2 | `ppf(1e-300, 1e-20, 0.9)` | 1.3784688875362527e-160 | 1.3784048752090222e-160 | verified | 2.09e11 ulp (severe), as reported. Mechanism matches line 10017 `np.sqrt(q * c * (1 + d - c))` |

Verified by dist-accuracy-verifier: 2 of 2 verified.
