# skewcauchy: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `skewcauchy_gen`, lines 9654-9718
- **Implemented:** pdf, cdf, ppf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** none (a = 0 is `cauchy`, but no code is shared); helpers: none; special functions: np.arctan, np.tan
- **Reference:** `refs/skewcauchy_ref.py`, 400 digits; self-checks: FAIL on the pdf-integral check for a = +-0.5 only. This is a quadrature artifact, not a reference error: the pdf's second derivative jumps at x = 0, and `mp.quad` over an interval containing 0 misses by 1e-8 relative. Split at 0, the integral matches the cdf difference to all 30 digits (`quad(pdf, [-0.688191, 0.866025])` = 0.399999897516..., `quad(pdf, [-0.688191, 0, 0.866025])` = 0.399999907976580746173935648107 = cdf difference). Every other self-check passes. pdf and cdf agree with SciPy in the bulk to 1.4 ulp, so the parametrization is right.
- **Parameter sets:** (0.5), (0), (-0.5), (0.9), (-0.9), (0.99999999), (-0.99999999), (0.999999999999999), (-0.999999999999999)
- **Worst finding:** loss in cdf (left tail) and ppf (low tail, and at the median)

## Summary

pdf is accurate everywhere (max 1.73 ulp). cdf is accurate for x > 0 (max 0.5 ulp), but its x <= 0 branch computes `(1-a)/2 + (1-a)/pi*arctan(x/(1-a))`. The two terms cancel in the left tail, so precision loss starts at |x| of about 10 and the result is 0.0 for |x| > ~1e16. When a is near 1, `(1 - a)` is small and the cancellation covers the whole negative half-line: for a = 0.999999999999999, `cdf(-10)` returns 0.0 (true value 3.2e-32). ppf has the matching problem: `q - (1-a)/2` absorbs small q, so the left-tail quantile saturates at about -8e15 for every q < ~1e-16. Two smaller problems: ppf returns 0.0 for some q next to the median, because `(1-a)/2` is rounded; and ppf near q = 1 has errors that slightly exceed the conditioning. Both come from roundings in the tan argument.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | x < 0 with \|x\| >> 1 - a; all x < 0 when a is near 1 | `(1-a)/2 + (1-a)/pi*arctan(x/(1-a))` cancels (pattern 1/5) | 0.0 for 7.96e-302; 0.0 for 3.18e-32 at x = -10, a = 0.999999999999999 | loss |
| F2 | ppf | q << (1-a)/2 | `q - (1-a)/2` absorbs q, then tan near its pole (pattern 8) | -8165619676597685.0 for -7.96e298 | loss |
| F3 | ppf | q next to the median q0 = (1-a)/2, a where 1 - a is inexact | rounded `(1-a)/2` in the `q - (1-a)/2` difference (pattern 7/8) | 0.0 for -1.74e-16 | loss |
| F4 | ppf | q -> 1, a near 1 | rounding of the tan argument near pi/2 (pattern 8) | 2.3e11 ulp at cond 2.3e11 | severe |

### F1: cdf cancels in the left tail

- **Code:** `_cdf`, line 9699: `(1 - a) / 2 + (1 - a) / np.pi * np.arctan(x / (1 - a))`
- **Mechanism:** for x/(1-a) -> -inf, `arctan` -> -pi/2, so the second term tends to -(1-a)/2 and cancels against the first (pattern 1/5). The true value is about (1-a)^2/(pi |x|). The relative error is about eps * (1-a)^2 / (2 cdf) ~ eps * |x| / (1-a). The result is 0.0 (or a multiple of ulp((1-a)/2)) once |x|/(1-a) > ~1e16. `cauchy` avoids this with `arctan2(1, -x)`.
- **Regime:** x < 0. Loses ~|x|/(1-a) ulp: moderate at |x| ~ 1e3 (1-a), severe from ~1e6 (1-a), loss from ~1e16 (1-a). For a near 1 this covers all of x < 0: for a = 0.99999999, 4.6e7 ulp already at x = -1. For a = 0.999999999999999 the result is 0.0 for x <= -6 and has no correct digits for -5 <= x <= -0.5. Max error in the region is inf; 334 loss, 83 severe, 5 moderate and 7 minor points in the low tail (the minor ones at x ~ -7 to -10 for a = 0.5).
- **Reproducer:** `python accuracy.py refs/skewcauchy_ref.py --at cdf -1e17 0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-1e300, 0.5)` | 0.0 | 7.957747154594766e-302 | inf | 1 | loss |
  | `cdf(-1e17, 0.5)` | 0.0 | 7.957747154594767e-19 | inf | 1 | loss |
  | `cdf(-1e8, 0.5)` | 7.957746961384515e-10 | 7.957747154594766e-10 | 1.1e8 | 1 | severe |
  | `cdf(-1000, 0.9)` | 3.1830988512296376e-06 | 3.183098851227576e-06 | 2.92e3 | 1 | moderate |
  | `cdf(-10, 0.999999999999999)` | 0.0 | 3.178012534596114e-32 | inf | 1 | loss |
  | `cdf(-1, 0.999999999999999)` | 2.9582283945787943e-31 | 3.178012534596114e-31 | 3.1e14 | 1 | severe |
  | `cdf(-1, 0.99999999)` | 3.1830988614559274e-17 | 3.183098893826518e-17 | 4.6e7 | 1 | severe |

  (cond (shapes) is 2e8 and 2e15 for a = 0.99999999 and 0.999999999999999: the cdf is very sensitive to a there, but a is an exact input.)
- **Fix direction:** for x < 0 use `(1 - a)/pi * arctan2(1 - a, -x)`, which equals (1-a)/pi * arctan((1-a)/|x|) and has no cancellation. Likewise, sf for x > 0 is `(1 + a)/pi * arctan2(1 + a, x)`.

### F2: ppf loses small q

- **Code:** `_ppf`, line 9705: `np.tan(np.pi / (1 - a) * (x - (1 - a) / 2)) * (1 - a)`
- **Mechanism:** the argument is -pi/2 + pi q/(1-a). Forming `q - (1-a)/2` keeps q only to an absolute accuracy of eps*(1-a)/2 (pattern 8). The relative error of the result is about (1-a)/(2q) ulp. For q < ~eps*(1-a)/4 the difference rounds to -(1-a)/2, and tan(-pi/2 in floating point) gives the constant -1.633e16 * (1-a). For a = 0.5 that is -8165619676597685.0 for every such q.
- **Regime:** q < (1-a)/2. Error ~ (1-a)/(2q) ulp: minor at q = 0.01 (10-53 ulp), severe for q < ~1e-7, loss (saturation) for q < ~1e-16 * (1-a). 205 loss, 73 severe and 3 moderate points in the low tail. Every parameter set is affected; the worst error is 4.5e15 ulp in all of them.
- **Reproducer:** `python accuracy.py refs/skewcauchy_ref.py --at ppf 1e-300 0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 0.5)` | -8165619676597685.0 | -7.957747154594767e+298 | 4.5e15 | 1 | loss |
  | `ppf(1e-20, 0.0)` | -1.633123935319537e+16 | -3.1830988618379067e+19 | 4.5e15 | 1 | severe |
  | `ppf(1e-12, 0.5)` | -79576558263.44081 | -79577471545.94766 | 5.2e10 | 1 | severe |
  | `ppf(1e-08, 0.5)` | -7957747.149121365 | -7957747.154594756 | 3.1e6 | 1 | severe |

  (`ppf(1e-20, 0)` is reported as severe, not loss: the relative error is 0.9995, just below 1.)
- **Fix direction:** for q < (1-a)/2 use `-(1 - a) / tan(pi * q / (1 - a))` (no subtraction; for tiny q this is -(1-a)^2/(pi q)).

### F3: ppf returns 0 next to the median

- **Code:** `_ppf`, lines 9703-9706: `i = x < self._cdf(0, a)` and `x - (1 - a) / 2` in both branches.
- **Mechanism:** the median q0 = (1-a)/2 is rounded when 1 - a is inexact, which happens for a < 0.5 (outside Sterbenz's range). For q within a rounding error of q0, the computed difference is 0 or has the wrong size and sign, while the true ppf is a tiny nonzero number (patterns 7/8). The tool rates it loss (0.0 where the true value is a normal double). For context, cond (arg) is 1.7e16 and cond (shapes) 8e15: the absolute error is about 1.7e-16, the size of the rounding of q0 itself.
- **Regime:** |q - (1-a)/2| ~ eps*(1-a)/2, a in (-1, 0.5) with 1 - a not exactly representable; found for a = -0.9, -0.99999999, -0.999999999999999. Not seen for a = 0.5 or 0.9, where (1-a)/2 is exact.
- **Reproducer:** `python accuracy.py refs/skewcauchy_ref.py --at ppf 0.95 -0.9`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.95, -0.9)` | 0.0 | -1.743934249004316e-16 | inf | 1.7e16 | loss |
  | `ppf(0.999999995, -0.99999999)` | 0.0 | 1.7439342490043161e-16 | inf | 1.8e16 | loss |
  | `ppf(0.9999999999999996, -0.999999999999999)` | 0.0 | 1.7618604646065624e-16 | inf | 1.8e16 | loss |

- **Fix direction:** form the difference as `(q - 0.5) + a/2` (q - 0.5 is exact for q in [0.25, 1], a/2 is exact), and use that sign for the branch test.

### F4: ppf near q = 1 slightly beyond conditioning for a near 1

- **Code:** `_ppf`, line 9706: `np.tan(np.pi / (1 + a) * (x - (1 - a) / 2)) * (1 + a)`
- **Mechanism:** near q = 1 the tan argument is close to pi/2. Three roundings each perturb the argument by about eps*pi/4 absolute, which tan amplifies by 1/(distance to the pole): `x - (1-a)/2` (inexact when (1-a)/2 is not a multiple of ulp(q), e.g. a = 0.999999999999999 where (1-a)/2 = 5.6e-16), `np.pi/(1+a)` and the product. Each one alone costs about the conditioning (cond ~ 1/(1-q)), so the total can exceed it slightly (pattern 8). For a = 0.5 the subtraction is exact and the same q stays within conditioning (3.7e10 ulp, ok).
- **Regime:** q -> 1 with (1-a)/2 inexact relative to ulp(q). Seen for a = 0.999999999999999 only: 1 severe and 1 moderate point, both about 1.0-1.07 times cond.
- **Reproducer:** `python accuracy.py refs/skewcauchy_ref.py --at ppf 0.9999999999956061 0.999999999999999`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999956061, 0.999999999999999)` | 289757740049.0411 | 289772400092.1067 | 2.3e11 | 2.3e11 | severe |
  | `ppf(0.9991516571017559, 0.999999999999999)` | 1500.853951529505 | 1500.8539515299285 | 1.27e3 | 1.18e3 | moderate |

- **Fix direction:** for q > (1-a)/2 near 1 use `(1 + a) / tan(pi * (1 - q) / (1 + a))`. Here 1 - q is exact for q >= 0.5, so only the conditioning error is left.

## RuntimeWarnings

- pdf: `overflow encountered in square/multiply/divide` and `underflow encountered in square/divide` from `x**2` and `x**2/(1+-a)**2` for |x| beyond ~1.3e154 (or smaller when 1 +- a is tiny), and for tiny |x|. The true pdf is subnormal where it returns 0.0, and the results are accurate elsewhere (max 1.73 ulp), so these warnings are spurious.
- cdf: `underflow encountered in arctan` (tiny x/(1 +- a)) and `overflow encountered in divide` (x/(1-a) with huge |x| and 1 - a ~ 1e-15). `np.where` evaluates both branches on the whole array, so the warnings also appear on accurate x > 0 results. Where they coincide with inaccuracy, the cause is F1, not the overflow: `arctan(-inf) = -pi/2` gives the right limit.
- ppf: none recorded.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid (+-1e-300 ... +-1.4e154, defaults to 1e300), both sides of x = 0, all 9 parameter sets including 1 +- a = 1e-15 | 1.73 ulp |
| cdf | x > 0, full grid, all parameter sets including a near -1 (1 + a = 1e-15) | 0.5 ulp |
| cdf | bulk x <= 0 for moderate a | ok except the F1 onset (minor, 9-71 ulp at x = -1 ... -10 for a = 0.9 and 0.5) |
| ppf | q in (q0, 1 - 1e-4] for all a; q near 1 for a in {0, +-0.5, +-0.9, a near -1} | within conditioning |

## Suspected, not confirmed

none. Every hypothesis in `scratch/skewcauchy/notes.md` was either confirmed (F1-F4) or tested and found fine. The `x**2` overflow in pdf only hits where the true pdf is subnormal, because 1 +- a <= 2. `1 +- a` is exact for a near +-1.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | x > 0: `(1 + a)/pi * arctan2(1 + a, x)`; x <= 0: `1 - cdf` (cdf <= (1-a)/2, no cancellation) | right tail: `sf(1e20, 0.5)` = 0.0, true 7.1619724391352901e-21 (loss); total loss once (1+a)^2/(pi x) < eps/2 |
| logsf | log(sf) | log of the arctan2 form above (`log1p(-cdf)` for x <= 0) | inherits the sf loss: `logsf(1e20, 0.5)` = -inf, true -46.385501529513985 |
| isf | ppf(1 - q) | q < (1+a)/2: `(1 + a) / tan(pi * q / (1 + a))`; otherwise the mirrored left-branch form with 1 - q | small q: `isf(1e-20, 0.5)` = 2.4496859029793056e16, true 7.1619724391352905e19 (4.5e15 ulp); also inherits F2 through the ppf |
| logcdf | log(cdf) | x < 0: `log((1 - a)/pi) + log(arctan2(1 - a, -x))` | inherits F1: `logcdf(-1e20, 0.5)` = -inf, true -48.582726106850204 |
| logpdf | log(pdf) | `-log(pi) - log1p((x/(1 +- a))**2)`, and for \|x/(1 +- a)\| > 1: `-log(pi) - 2*log(\|x\|/(1 +- a)) - log1p(((1 +- a)/x)**2)` | pdf underflows for \|x\| > ~1e154: `logpdf(1e200, 0.5)` = -inf, true -921.36783686725134 |

## Shared code

none (no helpers or scipy.special functions; only np.arctan/np.tan, used correctly for their own arguments)

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-1e17, 0.5)` | 0.0 | 7.9577471545947668e-19 | verified | reference = quad of the pdf after t = -1/u substitution, cross-checked with (1-a)/pi*acot(-x/(1-a)); 100 vs 200 digits agree to 1e-101. Other rows also match: `cdf(-1e300, 0.5)` 0.0 vs 7.957747154594766e-302, `cdf(-1e8, 0.5)` 1.09e8 ulp, `cdf(-1000, 0.9)` 2.92e3 ulp, `cdf(-10, 0.999999999999999)` 0.0 vs 3.1780125345961141e-32, `cdf(-1, 0.999999999999999)` 3.11e14 ulp, `cdf(-1, 0.99999999)` 4.58e7 ulp. Mechanism matches line 9699 |
| F2 | `ppf(1e-300, 0.5)` | -8165619676597685.0 | -7.9577471545947666e+298 | verified | reference = -(1-a)/tan(pi q/(1-a)), checked with cdf(x_ref) = q via the substituted quad (rel. 1e-201). Also `ppf(1e-20, 0)` 4.5e15 ulp, `ppf(1e-12, 0.5)` 5.17e10 ulp, `ppf(1e-8, 0.5)` 3.1e6 ulp, same as the report. Mechanism matches line 9705 |
| F3 | `ppf(0.95, -0.9)` | 0.0 | -1.7439342490043159e-16 | verified | in double, 0.95 - (1-a)/2 = 0.0, so the right branch is taken and tan(0) = 0. The exact q0 for a = double(-0.9) is 0.95000000000000001110 > q, so the true value is negative. Also verified: `ppf(0.999999995, -0.99999999)` -> 0.0 vs 1.7439342490043161e-16, `ppf(0.9999999999999996, -0.999999999999999)` -> 0.0 vs 1.7618604646065625e-16. Mechanism matches lines 9703-9706 |
| F4 | `ppf(0.9999999999956061, 0.999999999999999)` | 289757740049.0411 | 289772400092.10667 | verified | reference = (1+a)*cot(pi(1-q)/(1+a)), checked with cdf(x_ref) = q. 2.28e11 ulp, about 1.0x cond, so it is borderline. Second row: 1.27e3 ulp, same as the report. Mechanism plausible (line 9706), but the text's "(1-a)/2 = 5.6e-16" should read 5.0e-16 (4.996e-16) |

Verified by dist-accuracy-verifier: 4 of 4 verified.
