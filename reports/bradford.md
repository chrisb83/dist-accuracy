# bradford: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `bradford_gen`, lines 1087-1139
- **Implemented:** pdf, cdf, ppf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf) / log1p(-sf) (switch at the median), sf = 1 - cdf, logsf = log1p(-cdf) / log(sf) (switch at the median), isf = ppf(1 - q)
- **Related:** none; helpers: none; special functions: sc.log1p, sc.expm1
- **Reference:** `refs/bradford_ref.py`, 400 digits; self-checks: PASS (two NOTEs on the inherited logpdf, see Opportunities)
- **Parameter sets:** (0.29891359763170633), (1e-300), (1e-10), (0.001), (1), (100), (1e10), (1e300)
- **Worst finding:** loss in cdf, ppf

## Summary

pdf is accurate everywhere (max 1.35 ulp), and cdf and ppf are accurate in the
bulk and for moderate and large c. cdf and ppf fail only when c is tiny and the
argument is small. Then the intermediate `c*x` (cdf) or `q*log1p(c)` (ppf)
becomes subnormal or underflows to 0, while the true result (about x or q) is
still a normal double. With c = 1e-300 both methods return 0.0 for arguments
up to about 1e-24 and lose 7 to 10 digits up to about 1e-16. The errors become
minor (14 to 211 ulp) at c = 1e-10, x = 1e-300. The inherited logpdf loses
every digit for small c (pdf rounds to 1).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | c*x < 2.2e-308 with cdf normal (tiny c, small x) | subnormal/underflowing c*x (pattern 4) | 0.0 for 1e-300 | loss |
| F2 | ppf | q*log1p(c) < 2.2e-308 with ppf normal (tiny c, small q) | subnormal/underflowing q*log1p(c) (pattern 4) | 0.0 for 1e-300 | loss |

### F1: cdf underflows in c*x for tiny c

- **Code:** `_cdf`, line 1117: `sc.log1p(c*x) / sc.log1p(c)`
- **Mechanism:** for small c the cdf is about x, but the code first forms
  `c*x`. When `c*x` drops below 2.2e-308 it is subnormal and keeps only
  log2(c*x / 4.9e-324) significant bits. The later division by
  `log1p(c)` (about c) brings the value back to a normal size, but the lost
  bits are not restored. Below about 2.5e-324, `c*x` is 0 and the cdf is 0.0
  (pattern 4).
- **Regime:** c*x < 2.2e-308 while x (about the cdf) is a normal double. This
  needs c < 1, in practice tiny c. With c = 1e-300: loss for x < ~5e-24,
  severe for x up to ~1e-15, moderate up to ~1e-12, and 14 ulp at x = 1e-10.
  With c = 1e-10, x = 1e-300: 14.2 ulp (minor). Measured counts: 28 loss,
  3 severe, 2 moderate, 2 minor (all are this mechanism).
- **Reproducer:** `python accuracy.py refs/bradford_ref.py --at cdf 1e-16 1e-300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-300, 1e-300)` | 0.0 | 1e-300 | inf | 1 | loss |
  | `cdf(1e-30, 1e-300)` | 0.0 | 1e-30 | inf | 1 | loss |
  | `cdf(1e-20, 1e-300)` | 9.99988867182683e-21 | 1e-20 | 5.0e+10 | 1 | severe |
  | `cdf(1e-16, 1e-300)` | 9.999999836597144e-17 | 1e-16 | 7.4e+07 | 1 | severe |
  | `cdf(1e-15, 1e-300)` | 9.999999984816838e-16 | 1e-15 | 6.8e+06 | 1 | severe |
  | `cdf(1e-14, 1e-300)` | 9.999999999638807e-15 | 1e-14 | 1.6e+05 | 1 | moderate |
  | `cdf(1e-12, 1e-300)` | 9.999999999984654e-13 | 1e-12 | 6.91e+03 | 1 | moderate |
  | `cdf(1e-300, 1e-10)` | 1.0000000000499969e-300 | 1.00000000005e-300 | 14.2 | 1 | minor |

  (cond (shapes) is 5e-301 here: the shape does not excuse the error.)
- **Fix direction:** where `c*x` is tiny (e.g. < 1e-300 or < eps), use
  `x * (c / log1p(c))`. Since log1p(c*x) = c*x to double precision there,
  this is exact to rounding.

### F2: ppf underflows in q*log1p(c) for tiny c

- **Code:** `_ppf`, line 1120: `sc.expm1(q * sc.log1p(c)) / c`
- **Mechanism:** the same as F1, in the inverse direction. `q*log1p(c)` (about
  q*c) is subnormal or 0. `expm1` returns it unchanged, and the division by c
  scales it back to a normal size (about q) with the lost bits missing
  (pattern 4).
- **Regime:** q*log1p(c) < 2.2e-308 while q is normal. With c = 1e-300: 0.0
  for q < ~1e-24, severe up to ~1e-14, moderate to ~1e-12, and 208 ulp at
  q = 1e-10. With c = 1e-10, q = 1e-300: 211 ulp (minor). Measured counts:
  25 loss, 3 severe, 2 moderate, 2 minor (all are this mechanism).
- **Reproducer:** `python accuracy.py refs/bradford_ref.py --at ppf 1e-16 1e-300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 1e-300)` | 0.0 | 1e-300 | inf | 1 | loss |
  | `ppf(1.2115276586286942e-26, 1e-300)` | 0.0 | 1.2115276586286942e-26 | inf | 1 | loss |
  | `ppf(1e-20, 1e-300)` | 9.99988867182683e-21 | 1e-20 | 5.0e+10 | 1 | severe |
  | `ppf(1e-16, 1e-300)` | 9.999999342531498e-17 | 1e-16 | 3.0e+08 | 1 | severe |
  | `ppf(1e-14, 1e-300)` | 9.999999994698151e-15 | 1e-14 | 2.4e+06 | 1 | severe |
  | `ppf(1e-12, 1e-300)` | 1.000000000003406e-12 | 1e-12 | 1.53e+04 | 1 | moderate |
  | `ppf(1e-300, 1e-10)` | 9.99999999950047e-301 | 9.9999999995e-301 | 211 | 1 | minor |

- **Fix direction:** where `q*log1p(c)` is tiny, use
  `q * (log1p(c) / c)`. In general, use
  `expm1(q*L)/(q*L) * q * (L/c)` with L = log1p(c), so that no tiny
  intermediate forms.

## RuntimeWarnings

- pdf: "underflow encountered in multiply" in 4 parameter sets, from `c*x`
  at tiny x. The result is accurate (c*x is negligible next to 1.0), so the
  warning is spurious.
- cdf: "underflow encountered in multiply/divide" in 7 parameter sets, and
  ppf in 8. They come with the F1/F2 errors when c is tiny. For moderate c
  they occur only where the result itself is subnormal (not measured), for
  example at x = 1e-308.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid on [0, 1] incl. x = 1e-308 and x = 1 - 1e-16, all parameter sets (c from 1e-300 to 1e300) | 1.35 ulp |
| cdf | bulk and high tail, all parameter sets; low tail for c >= 1e-3 | 1.49 ulp |
| ppf | bulk and q near 1 (1 - 1e-16), all parameter sets; small q for c >= 1e-3 | 321 ulp at c = 1e300, within conditioning (cond 691: expm1 of q*log1p(c) ~ 690) |

## Suspected, not confirmed

none. The small-c limit of `c/log1p(c)` in the pdf (pattern 9) was tested at
c = 1e-300 and 1e-10 and is fine. Overflow of `c*x` or `expm1` for large c
cannot happen, because x <= 1 and q*log1p(c) <= 709.8. Tested up to c = 1e300.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `-log1p(c*x) - log1p(log1pmx(c)/c)` (`sc._ufuncs._log1pmx`). For c < ~1e-150, log1pmx(c) underflows, so use the series `c/2 - 5*c**2/24` for log(c/log1p(c)) | small c: pdf = 1 + O(c) rounds, so log(pdf) loses digits in proportion to 1/c. `logpdf(0.1, 1e-10)` = 4.000000330881484e-11 vs 3.9999999997966665e-11 (3.7e8 ulp, severe). `logpdf(0.1, 1e-300)` = 0.0 vs 4e-301 (loss). These are the self-check NOTEs |
| logcdf | log(cdf) below the median, log1p(-sf) above | `log(log1p(c*x)) - log(log1p(c))` adds nothing. Only the F1 fix of cdf is needed | inherits F1 (tiny c, small x); otherwise fine |
| sf | 1 - cdf | `log1p(c*(1 - x)/(1 + c*x)) / log1p(c)` | near x = 1, 1 - cdf cancels, but sf is just as ill-conditioned there (cond (arg) ~ 1/(1 - x)). `sf(1 - 1e-12, 1)` has 5.7e11 ulp at cond 1.0e12 (ok). The closed form is cheaper than the generic path but gives little accuracy gain |
| logsf | log1p(-cdf) below the median, log(sf) above | `log(log1p(c*(1 - x)/(1 + c*x))) - log(log1p(c))` | near x = 1. Within conditioning, as for sf: `logsf(1 - 1e-12, 1)` has 2.0e10 ulp at cond 3.6e10 (ok) |
| isf | ppf(1 - q) | `expm1((1 - q)*L)/c` without rounding 1 - q: `(c - (1 + c)*(-expm1(-q*L)))/c` with L = log1p(c) | small q (result near 1). Rounding 1 - q costs an error of order eps in an O(1) result, so there is no measurable loss: `isf(1e-20, 1)` = 1.0 vs 0.99999999999999999998614 (0 ulp) |

## Shared code

none (sc.log1p and sc.expm1 are used correctly and are accurate; the errors
come from the subnormal intermediates formed before the calls).

## Existing GitHub issues

- #6216 (closed): ENH: improve accuracy of ppf cdf roundtrip for bradford - fixed in PR #6239 (SciPy 0.18) by switching to the current log1p/expm1 formulas; relates to F1/F2 only as background. The tiny-c underflow of `c*x` / `q*log1p(c)` is not covered.

(Searched via the GitHub search API for "bradford" in scipy/scipy issues; `gh` was not available.)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e-300, 1e-300)` | 0.0 | 1.0000000000000000251e-300 | verified | loss. Ref = x * mp.quad(pdf(x*u), u in [0,1]), 100 and 200 dps agree to 30 digits |
| F1 | `cdf(1e-20, 1e-300)` | 9.99988867182683e-21 | 9.9999999999999994515e-21 | verified | 5.01e10 ulp, severe |
| F1 | `cdf(1e-16, 1e-300)` | 9.999999836597144e-17 | 9.999999999999999791e-17 | verified | 7.36e7 ulp, severe |
| F1 | `cdf(1e-12, 1e-300)` | 9.999999999984654e-13 | 9.9999999999999997989e-13 | verified | 6.91e3 ulp, moderate |
| F1 | `cdf(1e-300, 1e-10)` | 1.0000000000499969e-300 | 1.0000000000500000251e-300 | verified | 14.1 ulp (report 14.2), minor. Mechanism matches line 1117 (`log1p(c*x)`, subnormal c*x) |
| F2 | `ppf(1e-300, 1e-300)` | 0.0 | 1.0000000000000000251e-300 | verified | loss. Ref = findroot of the quadrature cdf, checked cdf(x_ref)/q - 1 = 0 |
| F2 | `ppf(1.2115276586286942e-26, 1e-300)` | 0.0 | 1.2115276586286941615e-26 | verified | loss |
| F2 | `ppf(1e-16, 1e-300)` | 9.999999342531498e-17 | 9.999999999999999791e-17 | verified | 2.96e8 ulp, severe |
| F2 | `ppf(1e-12, 1e-300)` | 1.000000000003406e-12 | 9.9999999999999997989e-13 | verified | 1.53e4 ulp, moderate |
| F2 | `ppf(1e-300, 1e-10)` | 9.99999999950047e-301 | 9.9999999995000002506e-301 | verified | 211 ulp, minor. Mechanism matches line 1120 (`q * log1p(c)` subnormal before expm1) |

Verified by dist-accuracy-verifier: 2 of 2 verified.
