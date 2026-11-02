# triang: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `triang_gen`, lines 10066-10136
- **Implemented:** pdf, cdf, ppf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** trapezoid (triang is its c = d case; trapezoid has its own code, reviewed separately); helpers: `_lazyselect`; special functions: none
- **Reference:** `refs/triang_ref.py`, 400 digits; self-checks: FAIL (one item, a harness quadrature artifact: "integral of pdf != cdf difference on [0.125638, 0.351097] (0.15785)". The interval contains the kink of the pdf at x = c, and `mp.quad` does not split there. With the integral split at c, it matches the cdf difference to 30 digits: 0.400000477963428037576193182775 both. All other checks pass, including bulk agreement with SciPy.)
- **Parameter sets:** (0.15785029824528218), (0.5), (0), (1e-10), (1e-200), (0.999), (1 - 1e-6), (1 - 1e-8), (1 - 1e-10), (1)
- **Worst finding:** loss in ppf and cdf; severe in cdf

## Summary

pdf is exact to about 1 ulp everywhere. cdf and ppf are accurate for moderate c,
but each has a closed form that is evaluated in a cancelling or underflowing order.
The right-branch ppf `1 - sqrt((1-c)*(1-q))` cancels when the result is small.
This affects c = 0 or tiny c with q small: the result is 0 for q < 1.1e-16, and the
error reaches 1e8 to 1e12 ulp up to q ~ 1e-8. For tiny c, `sqrt(c*q)` underflows to 0.
The right-branch cdf `(x*x - 2*x + c)/(c - 1)` cancels when c is close to 1,
with errors up to 1.35e7 ulp around 1 - c ~ 1e-8. `x*x/c` underflows to 0 for
x < c < 1.5e-154.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf | q >= c, small c (incl. c = 0), small q | 1 - sqrt((1-c)(1-q)) cancels (patterns 1, 8) | 0.0 for 5e-21; 7.0e12 ulp | loss |
| F2 | ppf | q < c, c*q < 2.2e-308 (c < 1.5e-154) | c*q underflows (pattern 4) | 0.0 for 1e-225 | loss |
| F3 | cdf | x < c < 1.5e-154, x*x < 2.2e-308 | x*x underflows (pattern 4) | 0.0 for 8.1e-201 | loss |
| F4 | cdf | x >= c, c close to 1 | x*x - 2*x + c cancels to c - 1 + (1-x)**2 (pattern 5) | 1.35e7 ulp | severe |

### F1: ppf right branch cancels for small c and small q

- **Code:** `_ppf`, line 10127: `np.where(q < c, np.sqrt(c * q), 1-np.sqrt((1-c) * (1-q)))`
- **Mechanism:** for q >= c with c and q small, `sqrt((1-c)*(1-q))` is
  1 - (c + q)/2 + ..., so subtracting it from 1 leaves an absolute error of
  ~eps in a result of size ~(c + q)/2. The relative error is about eps/ppf (patterns 1 and 8).
  When (c + q)/2 is below eps/2, `1 - q` and `1 - c` round to 1 and the result is exactly 0.
  The branch switch is at q = c, so for small c this branch also covers the start
  of the right segment (e.g. q = c itself).
- **Regime:** c < ~1e-3 (including c = 0, where it is the only branch), c <= q < ~1e-2.
  Loss for q < 1.1e-16 (c = 0 or c tiny). Severe for q up to ~1e-8, moderate up to ~1e-4.
  Measured: 28 loss points for c = 0 and 20 for c = 1e-200. Max q at loss is 1e-16.
- **Reproducer:** `python accuracy.py refs/triang_ref.py --at ppf 1e-20 0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-20, 0)` | 0.0 | 5e-21 | inf | 1 | loss |
  | `ppf(1e-20, 1e-200)` | 0.0 | 5e-21 | inf | 1 | loss |
  | `ppf(3.4807005884285624e-14, 0)` | 1.7430501486614958e-14 | 1.7403502942142963e-14 | 6.99e+12 | 1 | severe |
  | `ppf(1e-12, 0)` | 5.000444502911705e-13 | 5.00000000000125e-13 | 4.0e+11 | 1 | severe |
  | `ppf(1e-10, 1e-10)` | 1.000000082740371e-10 | 1e-10 | 3.7e+08 | 0.5 | severe |
  | `ppf(1e-08, 1e-10)` | 5.050000084771966e-09 | 5.05000001225125e-09 | 6.47e+07 | 0.99 | severe |
  | `ppf(1e-06, 1e-10)` | 5.000501249741163e-07 | 5.000501249750637e-07 | 8.53e+03 | 1 | moderate |
  | `ppf(0.01, 1e-10)` | 0.005012562943129462 | 0.005012562943129417 | 39.7 | 1 | minor |

- **Fix direction:** `(c + q - c*q) / (1 + sqrt((1-c)*(1-q)))` or
  `-expm1(0.5*(log1p(-c) + log1p(-q)))`.

### F2: ppf left branch underflows in c*q

- **Code:** `_ppf`, line 10127: `np.sqrt(c * q)`
- **Mechanism:** the product c*q underflows (to 0 or to a subnormal) before the square root
  takes it back into the normal range (pattern 4).
- **Regime:** q < c and c*q < 2.2e-308, which requires c < 1.5e-154. Measured: 11 loss points for c = 1e-200.
- **Reproducer:** `python accuracy.py refs/triang_ref.py --at ppf 1e-250 1e-200`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-250, 1e-200)` | 0.0 | 1e-225 | inf | 0.5 | loss |
  | `ppf(1e-300, 1e-200)` | 0.0 | 1e-250 | inf | 0.5 | loss |
  | `ppf(5e-201, 1e-200)` | 0.0 | 7.071067811865475e-201 | inf | 0.5 | loss |

- **Fix direction:** `np.sqrt(c) * np.sqrt(q)`.

### F3: cdf left branch underflows in x*x

- **Code:** `_cdf`, line 10120: `x * x / c`
- **Mechanism:** x*x underflows for x < 1.5e-154, but the division by a tiny c would bring
  the result back into the normal range (pattern 4).
- **Regime:** x < c < 1.5e-154 with x**2/c > 2.2e-308. This is only reachable for tiny c.
  For c >= 1.5e-154 the x < c branch has x*x subnormal only where x*x/c is (nearly) subnormal too.
  Measured: 8 loss points for c = 1e-200.
- **Reproducer:** `python accuracy.py refs/triang_ref.py --at cdf 9e-201 1e-200`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(9e-201, 1e-200)` | 0.0 | 8.100000000000001e-201 | inf | 2 | loss |
  | `cdf(1e-250, 1e-200)` | 0.0 | 1.0000000000000002e-300 | inf | 2 | loss |

- **Fix direction:** `x * (x / c)`.

### F4: cdf right branch cancels when c is close to 1

- **Code:** `_cdf`, line 10121: `(x*x - 2*x + c) / (c-1)`
- **Mechanism:** the numerator is (c - 1) + (1 - x)**2, built from terms of size 1 to 2.
  `x*x - 2*x` rounds to -1 + O(eps), so the numerator has an absolute error of
  ~min(eps, (1-x)**2). If (1-x)**2 < eps/2, the numerator becomes exactly c - 1 and the cdf exactly 1.0.
  Dividing by c - 1 makes this a relative error of up to ~eps/(1-c) in the cdf
  (pattern 5). Since x >= c, (1-x)**2 <= (1-c)**2, so the error is bounded by about
  min(1/(1-c), (1-c)/eps) ulp. That peaks near 1 - c ~ 1e-8.
  For moderate c the factor 1/(1-c) is small and the error stays at a few ulp.
- **Regime:** c close to 1, x in [c, 1]. Measured: c = 0.999 gives minor errors (227 ulp max, 8 points).
  c = 1 - 1e-6 gives up to 1.0e5 ulp. c = 1 - 1e-8 gives up to 1.35e7 ulp. c = 1 - 1e-10 gives up to 4.5e5 ulp.
  In the last two cases SciPy returns exactly 1.0.
- **Reproducer:** `python accuracy.py refs/triang_ref.py --at cdf 0.999999995 0.99999999`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(0.9999999909999999, 0.99999999)` | 0.9999999888977699 | 0.9999999918999999 | 1.35e+07 | 1.8 | severe |
  | `cdf(0.999999995, 0.99999999)` | 1.0 | 0.9999999975 | 1.1e+07 | 1 | severe |
  | `cdf(0.99999999, 0.99999999)` | 0.9999999888977699 | 0.99999999 | 4.96e+06 | 2 | severe |
  | `cdf(0.9999999999, 0.9999999999)` | 1.0 | 0.9999999999 | 4.5e+05 | 2 | moderate |
  | `cdf(0.9999991, 0.999999)` | 0.9999991899812812 | 0.99999919 | 8.43e+04 | 1.8 | moderate |
  | `cdf(0.9999, 0.999)` | 0.9999899999999498 | 0.99999 | 227 | 0.2 | minor |

- **Fix direction:** `1 - (1 - x)**2 / (1 - c)` (1 - x and 1 - c are exact for x, c >= 0.5).

## RuntimeWarnings

Both cdf and ppf emit "underflow encountered in multiply" (and cdf also
"underflow encountered in divide") for tiny arguments. In the F2/F3 regimes
the warning comes with a wrong 0. Elsewhere it comes with an accurate result where the
true value is itself subnormal or zero, so it is spurious there. Example:
`cdf(1e-160, 1e-10)` returns 9.999888671827e-311 (true 1e-310, subnormal, not
measured) with both warnings. `_lazyselect` evaluates only the selected
branch, so no warnings come from unused branches.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, all parameter sets incl. c = 0, c = 1, c = 1e-200, c close to 1, both sides of x = c | 0.909 ulp |
| cdf | x < c branch (c >= 1e-10), c = 0 branch `2x - x*x`, c = 1 branch `x*x`, right branch for c <= 0.5 incl. tiny c just above x = c | 1.04 ulp |
| ppf | q < c branch (c >= 1e-10), q near 1 (all c), c >= 0.158 | 1.32 ulp |

## Suspected, not confirmed

none. All hypotheses in `scratch/triang/notes.md` were either confirmed (F1-F4)
or measured as fine: pdf `2*(1-x)/(1-c)` and the cdf right branch for small c.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | `(1-x)**2/(1-c)` for x >= c, `1 - x*(x/c)` for x < c | right tail: `sf(0.999999999, 0.5)` gives 0.0, true 1.9999998868722757e-18 (loss) |
| logsf | log(sf) | `2*log1p(-x) - log1p(-c)` for x >= c, `log1p(-x*(x/c))` for x < c | same as sf: `logsf(0.999999999, 0.5)` gives -inf, true -40.753384549896744 (loss) |
| logcdf | log(cdf) | `2*log(x) - log(c)` for x < c, `log1p(-(1-x)**2/(1-c))` for x >= c | x < ~1.5e-154 (x*x underflows): `logcdf(1e-200, 0.5)` gives -inf, true -920.3408900170583 (loss) |
| isf | ppf(1 - q) | `1 - sqrt((1-c)*q)` for q <= 1-c, `sqrt(c*(1-q))` otherwise (with F1/F2 fixes) | small q (isf near 1): `isf(1e-20, 0.5)` gives 1.0, true 0.9999999999292893 (3.2e5 ulp, moderate; cond 3.5e-11). Also inherits F1/F2 via ppf |
| logpdf | log(pdf) | `log(2*x/c)` etc. no gain | none: pdf only underflows for subnormal x |

## Shared code

none. `_lazyselect` is used correctly; no special functions. trapezoid
(c = d generalization) has its own implementation with similar closed forms. It
may contain analogous expressions; see its report.

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(1e-20, 0)` | 0.0 | 4.9999999999999997e-21 | verified | loss; also `ppf(1e-20, 1e-200)` = 0.0. Reference: Newton on cdf = mp.quad of the pdf split at c, start from `-expm1((log1p(-c)+log1p(-q))/2)`; residual of cdf(x_ref) - q is 0 |
| F1 | `ppf(3.4807005884285624e-14, 0)` | 1.7430501486614958e-14 | 1.7403502942142963e-14 | verified | 6.99e12 ulp; `ppf(1e-10, 1e-10)` 3.73e8 ulp, `ppf(1e-08, 1e-10)` 6.47e7 ulp, `ppf(1e-06, 1e-10)` 8.53e3 ulp, all as reported. Mechanism matches line 10127 (right branch `1-np.sqrt((1-c)*(1-q))`) |
| F2 | `ppf(1e-250, 1e-200)` | 0.0 | 9.9999999999998173e-226 | verified | loss; also `ppf(1e-300, 1e-200)` and `ppf(5e-201, 1e-200)` give 0.0 (ref 9.99999999999982e-251, 7.0710678118653458e-201). The report's "1e-225" and "7.071067811865475e-201" are rounded; the exact double inputs give the values here. Mechanism matches `np.sqrt(c * q)`, line 10127 |
| F3 | `cdf(9e-201, 1e-200)` | 0.0 | 8.1000000000002972e-201 | verified | loss; `cdf(1e-250, 1e-200)` also gives 0.0 (ref 1.0000000000000367e-300). Mechanism matches `x * x / c`, line 10120 |
| F4 | `cdf(0.9999999909999999, 0.99999999)` | 0.9999999888977699 | 0.9999999918999999 | verified | 1.35e7 ulp, severe; `cdf(0.999999995, 0.99999999)` = 1.0 (1.13e7 ulp), `cdf(0.9999999999, 0.9999999999)` = 1.0 (4.5e5 ulp), `cdf(0.9999991, 0.999999)` 8.43e4 ulp, all as reported. Mechanism matches line 10121 |

Verified by dist-accuracy-verifier: 4 of 4 verified.
