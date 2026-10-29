# gompertz: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `gompertz_gen`, lines 4242-4287
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** none; helpers: none; special functions: sc.expm1, sc.log1p (and `sc._ufuncs._scaled_exp1` in `_entropy`, not reviewed)
- **Reference:** `refs/gompertz_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (0.9474371307510525), (1), (1 + 1e-8), (0.001), (1e-300), (1e-307), (1e-310), (1000), (1e6)
- **Worst finding:** loss in logpdf, pdf, cdf, sf, ppf, isf

## Summary

For ordinary shapes (c from 1e-3 to 1e6) cdf, ppf and isf are accurate to
about 2 ulp, and pdf and sf stay within their conditioning. There are three
real problems. (1) logpdf loses all its digits at small x when c is 1
(more generally when log(c) is close to 0), because `x - c*expm1(x)` cancels.
For c = 1 and x < ~1e-16 it returns 0.0. (2) `expm1(x)` overflows for
x > 709.78 even when `c*expm1(x)` is a moderate number. logpdf then returns
-inf for any c < 1, and for c below ~4e-306, pdf, cdf and sf return 0 or 1
even though the true values are ordinary numbers. (3) In ppf and isf the
quotient by c overflows for c below ~1e-306 (ppf even for every q once
c < 5.6e-309), so they return inf where the true quantile is about 20 to 720.
Separately, for tiny c, pdf and logpdf pick up a few hundred ulp from
rounding log(c).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf | small x, c = 1 or log(c) small | `log(c) + x - c*expm1(x)` cancels (pattern 5) | 0.0 for -5e-41 | loss |
| F2 | logpdf, pdf, cdf, sf | x > 709.78; logpdf any c < 1, others c < ~4e-306 | `expm1(x)` overflows before multiplying by c (pattern 3) | -inf, 0.0, 1.0 | loss |
| F3 | ppf, isf | c < ~1e-306 (ppf any q once c < 5.6e-309) | `-1.0/c`, `-log1p(-q)/c`, `-log(p)/c` overflow (pattern 3/8) | inf for 23.0 | loss |
| F4 | pdf, logpdf | tiny c (|log c| ~ 700) | rounding of log(c) amplified by exp / cancelled against x (pattern 5) | 306 ulp | minor |

### F1: logpdf cancels at small x when log(c) is close to 0

- **Code:** `_logpdf`, line 4272: `np.log(c) + x - c * sc.expm1(x)`.
- **Mechanism:** pattern 5. For small x, `x - c*expm1(x) = (1 - c)*x - c*(x**2/2 + ...)`.
  When c = 1, `log(c)` is exactly 0 and the remaining two terms are both about x
  but differ only by x**2/2. The result keeps only about x**2/(2x*eps) of its
  digits, and once x**2/2 < eps*x/2 (x < ~2.2e-16) `expm1(x)` rounds to x and the
  result is exactly 0. For c close to 1 the cancellation is weaker but still
  present: there the result is about log(c) - x**2/2 while the terms are of size x.
- **Regime:** x much smaller than 1 with |log c| much smaller than x. With
  c = 1 the error is about 2/x ulp, and the result is 0.0 for x < ~2e-16. All
  points 1e-150 <= x <= 1e-20 return 0.0.
- **Reproducer:** `python accuracy.py refs/gompertz_ref.py --at logpdf 1e-20 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-20, 1)` | 0.0 | -5e-41 | inf | 2 | loss |
  | `logpdf(1e-150, 1)` | 0.0 | -5e-301 | inf | 2 | loss |
  | `logpdf(1e-12, 1)` | -5.000242179397697e-25 | -5.000000000001666e-25 | 2.2e+11 | 2 | severe |
  | `logpdf(1e-08, 1)` | -4.999999918875795e-17 | -5.0000000166666667e-17 | 8.8e+07 | 2 | severe |
  | `logpdf(1e-05, 1)` | -5.0000166667394535e-11 | -5.000016666708334e-11 | 2.8e+04 | 2 | moderate |
  | `logpdf(0.001, 1.00000001)` | -4.901767134541084e-07 | -4.90176713454049e-07 | 546 | 2.04 | minor |

  (cond (shapes) is 2e40 at `logpdf(1e-20, 1)`: logpdf is extremely
  sensitive to c here, but c is an exact user input. pdf = exp(logpdf) is not
  affected, because an absolute error of eps*x in logpdf is harmless after exp.)
- **Fix direction:** compute `log(c) + (1 - c)*x - c*(expm1(x) - x)` with
  `expm1(x) - x` taken from a series (x**2/2 + x**3/6 + ...) for |x| < ~0.5.

### F2: expm1(x) overflows for x > 709.78 although c*expm1(x) is finite

- **Code:** `_logpdf` line 4272 (`c * sc.expm1(x)`; `_pdf` line 4269 is
  `exp(_logpdf)`), `_cdf` line 4275 `-sc.expm1(-c * sc.expm1(x))`,
  `_sf` line 4281 `np.exp(-c * sc.expm1(x))`.
- **Mechanism:** pattern 3. `sc.expm1(x)` is inf for x > 709.78, so
  `c*expm1(x)` is inf even when c is small enough that the product c*e^x is a
  normal number. logpdf becomes -inf and pdf becomes 0. For x > 709.78 the cdf
  is computed as exactly 1 and sf as exactly 0.
- **Regime:** x > 709.78 and c*e^x < 1.8e308. For logpdf that means every c < 1
  just past 709.78 (true logpdf = -c*e^x, finite). For pdf, cdf and sf the
  true value is noticeably different from 0/1 only when c*e^709.78 is
  below ~745 (sf, pdf) or ~37 (cdf), that is c < ~4e-306.
- **Reproducer:** `python accuracy.py refs/gompertz_ref.py --at sf 710 1e-307`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(710, 0.001)` | -inf | -2.233994766161711e+305 | inf | 710 | loss |
  | `logpdf(709.8, 0.947437)` | -inf | -1.7329006154802423e+308 | inf | 710 | loss |
  | `logpdf(710, 1e-300)` | -inf | -223399457.39169902 | inf | 710 | loss |
  | `logpdf(710, 1e-307)` | -inf | -19.233571210789133 | inf | 788 | loss |
  | `pdf(710, 1e-307)` | 0.0 | 4.435740752960085e-09 | inf | 1.52e+04 | loss |
  | `sf(710, 1e-307)` | 0.0 | 1.9855645233141061e-10 | inf | 1.59e+04 | loss |
  | `sf(713.8013788281542, 1e-310)` | 0.0 | 0.3678794411714455 | inf | 714 | loss |
  | `cdf(710, 1e-310)` | 1.0 | 0.02209225891072085 | 2.0e+17 | 702 | loss |
  | `cdf(709.79, 1e-307)` | 1.0 | 0.9999999863347487 | 6.2e+07 | 0.000176 | severe |

- **Fix direction:** for large x form the product in log space:
  `c*expm1(x) = exp(x + log(c)) - c` (or `exp(x + log(c))` when x > ~40).

### F3: ppf and isf overflow in the division by c for tiny c

- **Code:** `_ppf` line 4278: `sc.log1p(-1.0 / c * sc.log1p(-q))`;
  `_isf` line 4284: `sc.log1p(-np.log(p)/c)`.
- **Mechanism:** pattern 3 / 8. In ppf, `-1.0/c` is formed first and is
  inf for every c < 5.6e-309, so ppf is inf for every q. For larger tiny c,
  the product `(-1/c)*log1p(-q)` overflows once -log1p(-q) > 1.8e308*c. In isf,
  `-log(p)/c` overflows once -log(p) > 1.8e308*c. The true quantile
  log1p(t) is about log(t) and stays around 700.
- **Regime:** c < ~3.8e-307 for isf (-log p <= 690 at p = 1e-300) and ppf
  q near 1 (-log1p(-q) <= 37). ppf fails at every q once c < 5.6e-309.
  Measured: c = 1e-307 (isf for p < ~1e-13, ppf for 1 - q < ~6e-9) and
  c = 1e-310 (all of ppf; isf for every p measured in the low tail and bulk).
- **Reproducer:** `python accuracy.py refs/gompertz_ref.py --at ppf 1e-300 1e-310`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 1e-310)` | inf | 23.02585093004046 | inf | 0.0434 | loss |
  | `ppf(0.9999999943765867, 1e-307)` | inf | 709.8378691944046 | inf | 1.32e+04 | loss |
  | `isf(3.4807005884285624e-14, 1e-307)` | inf | 710.3272544872947 | inf | 4.54e-05 | loss |
  | `isf(1e-300, 1e-310)` | inf | 720.3391937480583 | inf | 2.01e-06 | loss |

- **Fix direction:** with `t = -log1p(-q)/c` (resp. `-log(p)/c`) computed as
  `exp(log(-log1p(-q)) - log(c))`, use `log1p(t)` for moderate t and
  `log(-log1p(-q)) - log(c) + log1p(1/t)` when t would overflow; at least
  write `-sc.log1p(-q)/c` instead of `-1.0/c * ...`.

### F4: rounding of log(c) costs a few hundred ulp for tiny c

- **Code:** `_pdf` line 4269 `np.exp(self._logpdf(x, c))`; `_logpdf` line 4272.
- **Mechanism:** For c = 1e-300, `log(c)` = -690.8 carries a rounding error
  of up to half an ulp of 690 (about 5.7e-14 absolute). pdf = exp(logpdf)
  turns that absolute error into a relative error of 100 to 300 ulp
  (pattern 3/4 family: a round trip through log space with a large log term).
  In logpdf the same error shows when x is close to -log(c) and the log(c)
  and x terms cancel (pattern 5), e.g. logpdf = -1 at x = 690.78.
- **Regime:** c below about 1e-100 (error grows like |log c|*eps); pdf at
  every x where pdf is normal; logpdf only near x = -log(c).
- **Reproducer:** `python accuracy.py refs/gompertz_ref.py --at pdf 1e-10 1e-300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-10, 1e-300)` | 1.000000000100068e-300 | 1.0000000001e-300 | 306 | 1e-10 | minor |
  | `pdf(1e-05, 1e-307)` | 1.0000100000499518e-307 | 1.00001000005e-307 | 244 | 1e-05 | minor |
  | `logpdf(690.7755278982137, 1e-300)` | -0.9999999999999762 | -1.0 | 107 | 1.64e-11 | minor |

- **Fix direction:** `pdf = c * exp(x - c*expm1(x))` so that log(c) never
  forms (and it also covers F2 when combined with the large-x rewrite).

## RuntimeWarnings

- `overflow encountered in multiply` in logpdf/cdf/sf/pdf: from
  `c * sc.expm1(x)` with large c or large x. For moderate and large c the
  results (sf = 0, cdf = 1, logpdf = -inf when the true value is
  below -1.8e308) are correct, so the warning is spurious there. In the F2
  regime it comes with wrong results.
- `underflow encountered in exp` (pdf, sf): the true values underflow, so
  these are harmless.
- `underflow encountered in multiply` (pdf, logpdf, cdf, sf) for tiny c
  (c*expm1(x) is subnormal): the results are accurate except for F4.
- `overflow encountered in divide` / `in multiply` in ppf and isf: these
  always come with the wrong inf results of F3.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid, all parameter sets except c <= 1e-307 (F2) | 1.98 ulp |
| sf | full grid, c >= 1e-300 (right tail up to underflow) | 385 ulp, within conditioning (cond up to 9.3e3) |
| pdf | full grid, c >= 1e-3 | 506 ulp, within conditioning (cond up to 9.3e3) |
| logpdf | c = 1000, 1e6 everywhere; other c away from F1/F2 regimes | 1.57 ulp |
| ppf | q in [1e-300, 1 - 3e-16], c >= 1e-300 | 0.99 ulp |
| isf | q in [1e-300, 1 - 3e-16], c >= 1e-300 (p near 1 included: `log(p)` is accurate for the double p) | 1.15 ulp |

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | log(sf) | closed form `-c * sc.expm1(x)` (with the large-x log-space form of F2) | sf underflows once c*expm1(x) > ~745: `logsf(10, 1)` gives -inf, true -22025.465794806718 |
| logcdf | log(cdf) | `log(-expm1(-u))` with u = c*expm1(x); for small u use `log(c) + log(expm1(x)) + log1p(-u/2 + ...)` (u itself underflows) | cdf subnormal or zero (c*x < ~2.2e-308): `logcdf(1e-300, 1e-30)` gives -inf, true -759.8530806880351 |

## Shared code

none (only standard `sc.expm1`, `sc.log1p` used correctly; no helpers)

## Existing GitHub issues

- #4031 (closed): improve accuracy of scipy.stats.gompertz distribution for small x - context for F1 (the small-x fix used expm1/log1p, PR #4407; the logpdf cancellation at c near 1 remains)
- #17480 (closed, PR): ENH: stats: Implement _sf and _isf for halfnorm, gibrat, gompertz - origin of the current sf/isf (F2, F3)
- No open issue found for any finding (GitHub search API; gh not installed).

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `logpdf(1e-20, 1)` | 0.0 | -4.9999999999999995e-41 | verified | ref: log(c) + (1-c)x - c*(e^x-1-x) with series for e^x-1-x; 100/200 digits agree |
| F1 | `logpdf(1e-150, 1)` | 0.0 | -5.0000000000000001e-301 | verified | same route |
| F1 | `logpdf(1e-12, 1)` | -5.000242179397697e-25 | -5.0000000000016665e-25 | verified | 2.18e11 ulp |
| F1 | `logpdf(1e-08, 1)` | -4.999999918875795e-17 | -5.0000000166666669e-17 | verified | 8.81e7 ulp; mechanism matches line 4272 |
| F2 | `logpdf(710, 0.001)` | -inf | -2.2339947661617111e+305 | verified | ref: log(c) + x - c*exp(x) + c |
| F2 | `logpdf(709.8, 0.9474371307510525)` | -inf | -1.7329006154802422e+308 | verified | |
| F2 | `logpdf(710, 1e-300)` | -inf | -223399457.39169901 | verified | |
| F2 | `logpdf(710, 1e-307)` | -inf | -19.233571210789133 | verified | |
| F2 | `pdf(710, 1e-307)` | 0.0 | 4.4357407529600853e-09 | verified | |
| F2 | `sf(710, 1e-307)` | 0.0 | 1.9855645233141061e-10 | verified | ref: mp.quad of pdf from x to tail; matches closed form to 1e-100 |
| F2 | `sf(713.8013788281542, 1e-310)` | 0.0 | 0.36787944117144548 | verified | quad route |
| F2 | `cdf(710, 1e-310)` | 1.0 | 0.02209225891072085 | verified | ref: mp.quad of pdf on [0, x]; 1.99e17 ulp |
| F2 | `cdf(709.79, 1e-307)` | 1.0 | 0.99999998633474874 | verified | 6.15e7 ulp (severe); mechanism matches lines 4272/4275/4281 |
| F3 | `ppf(1e-300, 1e-310)` | inf | 23.02585093004046 | verified | cdf(x_ref) = q to 7e-101; -1.0/c = -inf at line 4278 |
| F3 | `ppf(0.9999999943765867, 1e-307)` | inf | 709.83786919440453 | verified | cdf(x_ref) = q exactly at 100 digits |
| F3 | `isf(3.4807005884285624e-14, 1e-307)` | inf | 710.32725448729469 | verified | sf(x_ref) = p to 2e-98; -log(p)/c overflows at line 4284 |
| F3 | `isf(1e-300, 1e-310)` | inf | 720.33919374805832 | verified | sf(x_ref) = p to 2e-96 |

Verified by dist-accuracy-verifier: 3 of 3 verified.
