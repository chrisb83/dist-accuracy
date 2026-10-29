# foldcauchy: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `foldcauchy_gen`, lines 2494-2539
- **Implemented:** pdf, cdf, sf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf), ppf = numerical inversion of cdf, isf = ppf(1 - q)
- **Related:** none (does not call into `cauchy`; `_rvs` only uses `cauchy.rvs`); helpers: none; special functions: none (`np.arctan`, `np.arctan2`)
- **Reference:** `refs/foldcauchy_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (4.716467345583189), (0), (0.001), (1), (1000), (1e6), (1e100)
- **Worst finding:** loss in cdf

## Summary

pdf and sf are accurate everywhere measured (max 1.5 and 1.0 ulp). cdf is
written as `(arctan(x - c) + arctan(x + c))/pi`, which for any c > 0 and
x small compared with max(1, c) is the difference of two numbers close to
+-arctan(c): the result loses about arctan(c)*(1 + c**2)/(2x) ulp and is
exactly 0.0 once it falls below roughly eps*arctan(c). For c >= ~1e8 the cdf
is 0.0 wherever both arctans round to +-pi/2, i.e. for all x with
c - x >~ 1e16 (for c = 1e100 still at x = c/2). Of 584 measured cdf points, 189
are losses and 27 severe. The exact identity
`arctan(x - c) + arctan(x + c) = arctan2(2x, 1 + (c - x)(c + x))` removes
the problem (candidate measured at max 1.2 ulp on the same grid).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | c > 0, x small relative to max(1, c) (cdf << arctan(c)/pi) | arctan(x-c) + arctan(x+c) cancels (pattern 5) | 0.0 for 3.18e-21 (c = 1); 0.0 for 6.4e-201 (c = 1e100) | loss |

### F1: cdf cancels between arctan(x - c) and arctan(x + c)

- **Code:** `_cdf`, line 2529: `1.0/np.pi*(np.arctan(x-c) + np.arctan(x+c))`
- **Mechanism:** for x << max(1, c), `arctan(x - c) ~ -arctan(c)` and
  `arctan(x + c) ~ +arctan(c)`; their sum is about
  `2x/(1 + c**2)` (cdf ~ 2x/(pi(1 + c**2))). Each arctan carries a rounding
  error of ~ulp(arctan(c)), so the relative error of the sum is about
  `arctan(c)*(1 + c**2)/(2x)` ulp (pattern 5). Once the true sum is below
  ~ulp(arctan(c)) both arctans round to the same magnitude and the result is
  exactly 0.0. For c >= ~1e8, `arctan(+-c)` already rounds to +-pi/2 and
  `x - c`, `x + c` lose x entirely, so the cdf is 0.0 for all x with
  c - x >~ 1e16 (for c = 1e100 that includes x = 5e99).
- **Regime:** any c > 0 (c = 0 is fine: the two terms are equal and add).
  Moderate/severe for 2x/(1 + c**2) between ~1e-16*arctan(c) and ~1e-3;
  total loss below. Examples: c = 1: severe for x = 1e-10, loss for
  x <= 1e-20; c = 0.001: severe at x = 1e-10, loss for x <= 1e-20;
  c = 1000: moderate (1e4 to 3e5 ulp) for x in [0.5, 10], severe at x <= 0.1;
  c = 1e6: severe (up to 4e11 ulp) for x <= 10, moderate at x = 5e5;
  c = 1e100: loss for every measured x <= 0.99999c. The bulk is also
  affected mildly: `cdf(0.1, 4.716...)` 27.1 ulp (minor).
- **Reproducer:** `python accuracy.py refs/foldcauchy_ref.py --at cdf 1e-10 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-20, 1)` | 0.0 | 3.1830988618379067e-21 | inf | 1 | loss |
  | `cdf(1e-300, 4.716467345583189)` | 0.0 | 2.738730967954739e-302 | inf | 1 | loss |
  | `cdf(1, 1e100)` | 0.0 | 6.3661977236758134e-201 | inf | 1 | loss |
  | `cdf(5e99, 1e100)` | 0.0 | 4.244131815783876e-101 | inf | 1.67 | loss |
  | `cdf(10, 1e6)` | 6.3661982504173755e-12 | 6.366197724306067e-12 | 3.7e8 | 1 | severe |
  | `cdf(1e-10, 1)` | 3.1830991252086877e-11 | 3.183098861837907e-11 | 3.7e8 | 1 | severe |
  | `cdf(1e-10, 0.001)` | 6.366191355074218e-11 | 6.366191357484457e-11 | 1.7e6 | 1 | severe |
  | `cdf(1e-5, 4.716467345583189)` | 2.738730967791498e-07 | 2.738730967965845e-07 | 2.9e5 | 1 | moderate |
  | `cdf(1, 1000)` | 6.366197723858878e-07 | 6.366197723667326e-07 | 1.4e5 | 1 | moderate |
  | `cdf(1e-5, 1)` | 3.1830988618940997e-06 | 3.1830988618909584e-06 | 4.4e3 | 1 | moderate |
  | `cdf(0.1, 4.716467345583189)` | 0.002739842008360179 | 0.0027398420083601625 | 27.1 | 1 | minor |

  Totals over the grid (584 points): 189 loss, 27 severe, 23 moderate,
  3 minor; all in the low tail except the one bulk minor point.
- **Fix direction:** `np.arctan2(2*x, 1 + (c - x)*(c + x))/np.pi` (exact
  identity, the sum is arg((1 + i(x-c))(1 + i(x+c))) and lies in [0, pi)).
  Measured as a candidate (`scratch/foldcauchy/cand.py`): max 1.2 ulp over
  all 584 cdf points, no non-ok point. Writing the denominator as
  `1 + (c - x)(c + x)` rather than `1 + c**2 - x**2` keeps it accurate near
  x = sqrt(1 + c**2); the product overflows for c or x >~ 1e154, which needs a
  scaled form (or the old formula for x > c) there.

## RuntimeWarnings

- pdf: "overflow encountered in square" for |x +- c| > ~1.3e154 (e.g.
  `pdf(1e200, 1)`), where the true pdf (~2/(pi x**2)) is below the double
  range anyway, so the returned 0.0 is correct; "underflow encountered in
  square" for tiny x with c = 0 or tiny c, where the result is accurate
  (`pdf(1e-200, 4.716...)` 0.57 ulp). Both spurious.
- cdf: "underflow encountered in arctan" for c = 0 and tiny x
  (`cdf(1e-300, 0)` returns 6.366197723675814e-301, 0 ulp). Spurious.

## Checked and fine

| Method | Regime checked | Max error |
|---|---|---|
| pdf | full grid x in [1e-300, 1e300], all parameter sets incl. c = 0, 1e-3, 1e6, 1e100, x = c(1 +- 1e-6), sqrt(1 + c**2) | 1.5 ulp |
| sf | full grid, all parameter sets; `arctan2(1, x -+ c)` terms are both positive, no cancellation in either tail | 1.0 ulp |
| cdf | c = 0 (all x); high tail and x >= ~c for all c | 0.98 ulp (c = 0), 0.5 ulp (high tail) |

Note: `sf(1e-300, 1e6)` returns 1.0000000000000002 (1 ulp above 1, within
conditioning; the two arctan2 terms are pi - 1e-6 and 1e-6 and round up).

## Suspected, not confirmed

- Overflow of `(x - c)**2` / `(x + c)**2` in pdf (pattern 3): only happens
  where the true pdf is subnormal or zero, so no measurable error. Not a finding.
- For c >~ 1e154 the pdf terms near x = c are fine (x - c exact), and no
  error was seen at c = 1e100; larger c was not separately measured.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `log(1/(1+(x-c)**2) + 1/(1+(x+c)**2)) - log(pi)`, for large x as `-2*log(x) + log(2/pi) + log1p(...)` (or `logaddexp` of `-log1p((x-+c)**2)` with `(x-+c)**2` computed in log form) | pdf underflows for x >~ 1.7e154: `logpdf(1e200, 1)` = -inf, true -921.4856199029077 |
| logcdf | log(cdf) | `log(arctan2(2x, 1 + (c-x)(c+x))/pi)`; for cdf below the double range `log(2x) - log1p(...)`-type asymptotic `log(2x/(pi(1+c**2)))` | inherits F1: `logcdf(1e-20, 1)` = -inf, true -47.196431745730314; after fixing F1, only where cdf underflows (large c) |
| logsf | log(sf) | not needed: sf ~ 2/(pi x) stays normal for all finite x | none found: `logsf(1e300, 1)` = -691.2271106035032, rel. error 6.7e-17 |
| ppf | numerical inversion of cdf | closed form: with `s = cot(pi q)`, `x = (1 + c**2)/(s + sqrt(s**2 + 1 + c**2))` for s > 0 and `x = -s + sqrt(s**2 + 1 + c**2)` for s <= 0 (from `2x/(1 + c**2 - x**2) = tan(pi q)`) | inherits F1 for small q: `ppf(1e-20, 1)` = 0.0, true 3.141592653589793e-20 (closed form gives 3.141592653589793e-20); slow everywhere |
| isf | ppf(1 - q) | same closed form with `s = -cot(pi q)` (i.e. `x = cot(pi q) + sqrt(cot(pi q)**2 + 1 + c**2)` for q < 1/2) | small q: `isf(1e-20, 1)` = 1e16, true 6.3661977236758138e19 (1 - q rounds to 1) |

## Shared code

none

## Existing GitHub issues

None found about F1 (cdf cancellation) or the inherited ppf/isf. For context:

- #17694 (closed PR): ENH: stats: Implement _sf and _isf for halfcauchy; _sf for foldcauchy - added the current accurate `_sf`; it did not touch `_cdf` or add `_ppf`/`_isf`
- #17832 (open): Tracking issue: custom methods for continuous distributions - general tracker where a foldcauchy `_ppf`/`_isf` would fit

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e-20, 1)` | 0.0 | 3.1830988618379065e-21 | verified | loss; ref = atan2(2x, 1+c^2-x^2)/pi, matches mp.quad of the pdf to 9e-202 |
| F1 | `cdf(1e-300, 4.716467345583189)` | 0.0 | 2.7387309679547388e-302 | verified | loss; quad agrees to 4e-14 (quad limit at tiny interval), atan2 stable 100 vs 200 digits |
| F1 | `cdf(1, 1e100)` | 0.0 | 6.3661977236758132e-201 | verified | loss; quad agrees to 4e-14 |
| F1 | `cdf(5e99, 1e100)` | 0.0 | 4.2441318157838756e-101 | verified | loss; quad agrees to 2e-191 at 200 digits |
| F1 | `cdf(10, 1e6)` | 6.3661982504173755e-12 | 6.366197724306067e-12 | verified | 3.7e8 ulp, severe |
| F1 | `cdf(1e-10, 1)` | 3.1830991252086877e-11 | 3.1830988618379068e-11 | verified | 3.7e8 ulp, severe |
| F1 | `cdf(1e-10, 0.001)` | 6.366191355074218e-11 | 6.3661913574844562e-11 | verified | 1.7e6 ulp, severe |
| F1 | `cdf(1, 1000)` | 6.366197723858878e-07 | 6.3661977236673252e-07 | verified | 1.4e5 ulp, moderate |
| F1 | `cdf(0.1, 4.716467345583189)` | 0.002739842008360179 | 0.0027398420083601627 | verified | 26.9 ulp, minor |

All verifier references: `scratch/foldcauchy/verify/verify_f1.py`, computed two ways (mp.quad of the docstring pdf; atan2 identity), atan2 route agrees between 100 and 200 digits to better than 1e-100. Mechanism checked against source: `_cdf` line 2529 is `1.0/np.pi*(np.arctan(x-c) + np.arctan(x+c))`, the cancellation described is correct.

Verified by dist-accuracy-verifier: 1 of 1 verified.
