# irwinhall: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `irwinhall_gen`, lines 9402-9527
- **Implemented:** pdf, cdf, sf (all via `scipy.interpolate.BSpline`: `BSpline.basis_element(arange(n+1))` and its `antiderivative()`)
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf), ppf = numerical inversion of cdf, isf = ppf(1 - q)
- **Related:** uniform (n = 1), triang (n = 2 is `triang(0.5, scale=2)`), Bates (= irwinhall with scale 1/n); helpers: `_isintegral`, `_cardbspl`; special functions: sc.comb, sc.stirling2 (only in `_munp`, not in pdf/cdf/sf); `scipy.interpolate.BSpline` (de Boor evaluation, `splantider`)
- **Reference:** `refs/irwinhall_ref.py`, 400 digits (+ 0.75 n + 20 guard digits per call for the alternating sum); self-checks: FAIL x2 on the pdf-integral check for n = 3, shown to be an mp.quad artifact (see Summary); all other checks pass
- **Parameter sets:** (1), (2), (3), (10), (30), (100), (300); separate run (1000), (2000)
- **Worst finding:** none beyond conditioning (minor 10 ulp in cdf/sf for n = 2000, cdf returns values slightly above 1)

## Summary

SciPy does not evaluate the alternating sum `sum (-1)^k C(n,k) (x-k)^(n-1)`.
Instead, pdf, cdf and sf evaluate a cardinal B-spline (pdf) and its
antiderivative (cdf; sf = cdf(n - x) by symmetry) with de Boor's recursion.
This is a convex-combination scheme with no cancellation. So for n up to 300,
all three methods are within 8.8 ulp everywhere on the grid: both tails down
to the underflow threshold, both sides of every integer knot, and the
endpoints. For n = 1000 and 2000 the worst errors are 18 to 45 ulp in the far
left tail. These stay within the argument conditioning (cond ~ n there).
Outside the tails, the only points beyond conditioning are 4 minor points
(8.5 to 10 ulp) for n = 2000 where cdf (or sf) is ~1 and comes out as
1.0000000000000022, which is above 1. No loss, severe or moderate results were
found. The only real weakness is in the inherited methods. log(pdf),
log(cdf) and log(sf) give -inf as soon as the B-spline value underflows, and
isf = ppf(1 - q) loses the small-q tail completely (2.0e13 ulp at
`isf(1e-20, 10)`).

The reference self-check reported FAIL on "integral of pdf != cdf difference"
for n = 3 on [0.843, 1.5] and [1.5, 2.157]. These intervals contain the knots
x = 1 and x = 2, where the pdf is only C^1, so mp.quad without breakpoints
has a relative error of 1.6e-9. With the knot as a breakpoint the ratio is
exactly 1 (`scratch/irwinhall/selfcheck_quad.txt`). The reference is an exact
alternating sum with guard digits, and it agrees with SciPy to a few ulp at
every bulk point. For n = 2 the SciPy cdf also matches `triang(0.5, scale=2)`
exactly at the points checked. (Side observation for the triang report:
`triang.sf(2 - 1e-10, 0.5, scale=2)` returns 0.0, while `irwinhall.sf`
gives 5.00000083e-21, correct for the rounded argument.)

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| - | - | no finding above minor | - | - | - |

No loss, severe or moderate points. The 4 minor points (n = 2000) are listed
under "Checked and fine" because they have no specific faulty expression
(de Boor rounding accumulates over 2000 recursion levels).

## RuntimeWarnings

`underflow encountered in vpdf / vcdf / vsf (vectorized)` appears for
n >= 2 (pdf: n = 300; cdf/sf: n = 2, 100, 300, 1000, 2000) whenever some
de Boor intermediate underflows. This includes results that are accurate and
normal, so in those cases the warning is spurious:

| call | scipy | reference | ulp | severity |
|---|---|---|---|---|
| `pdf(11.082398014297766, 300)` | 2.171738281047912e-300 | 2.1717382810479123e-300 | 0.688 | ok |
| `cdf(11.082398014297766, 300)` | 8.022689337933942e-302 | 8.022689337933942e-302 | 0 | ok |
| `sf(288.91760198570223, 300)` | 8.022689337937021e-302 | 8.022689337937029e-302 | 4.07 | ok |
| `sf(1e-300, 300)` | 1.0 | 1.0 | 0 | ok |

In other cases (`cdf(1e-200, 2)`, `cdf(0.5, 2000)`) the warning comes with a
true value below the normal range, and returning 0 is correct.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | n = 1, 2, 3, 10, 30, 100, 300: full default grid (relative grid from both endpoints down to 1e-300), integer knots +-1e-9, half-integers, left/right tail points where log pdf = -30 ... -690 | 8.14 ulp (n = 300, cond 269) |
| cdf | same | 8.8 ulp (n = 300, cond 270) |
| sf | same (sf = cdf(n - x); n - x is exact for x >= n/2 and harmless for small x where sf ~ 1) | 7.83 ulp (n = 100) |
| pdf | n = 1000, 2000 | 43.5 ulp, left tail, cond 1.52e3 (within conditioning) |
| cdf | n = 1000, 2000 | 45.4 ulp, left tail, cond 1.52e3 (within conditioning); minor: 10 ulp at `cdf(1995.000000001, 2000)` = 1.0000000000000022 and 8.5 ulp at `cdf(1994.5, 2000)` = 0.9999999999999981 (reference 1.0 to double precision). A cdf slightly above 1 is wrong in principle, but the error is within 1e-14 |
| sf | n = 1000, 2000 | minor: 10 ulp at `sf(4.999999999, 2000)` = 1.0000000000000022, 8.5 ulp at `sf(5.5, 2000)` (mirror of the cdf points) |

Hypotheses from `scratch/irwinhall/notes.md` that did not show up:
- H1 (error growth with degree): it grows slowly, about 4.4 (n=30), 7.5 (100), 8 (300), 18 (1000) and 45 ulp (2000), and stays within the argument conditioning in the tails.
- H2 (subnormal de Boor intermediates in the left tail): results are accurate down to ~1e-302 (n = 300). Only the spurious warnings appear.
- H3 (sf = cdf(n - x) rounding): no error. x is near 0 only where sf ~ 1.
- H4 (knot/interval selection at integer x, x = 0, x = n): all within a few ulp.
- H5 (n = 1, 2 exactness): 0 ulp for n = 1 and at most 1 ulp for n = 2.

## Suspected, not confirmed

- Very large n (>= 1e4): not measured (runtime). The reference needs ~n terms
  at ~0.75 n extra digits, and SciPy builds an O(n) spline and runs an O(n^2)
  de Boor recursion per point. The trend (about n/45 ulp in the left tail,
  cdf exceeding 1 by a few ulp near the right end) suggests slow,
  conditioning-sized growth rather than a break. Separately, SciPy rebuilds
  the spline and its antiderivative for every element through
  `np.vectorize`. That costs performance, not accuracy.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | for x <= 1: `(n-1)*log(x) - gammaln(n)`; in general a log-scaled de Boor, or `xlogy(n-1, x) - gammaln(n)` plus a log1p of the alternating corrections for small x/n | pdf underflows: `logpdf(1e-3, 100)` = -inf, true -1043.0019779888069 (loss) |
| logcdf | log(cdf) | for x <= 1: `n*log(x) - gammaln(n+1)`; same idea beyond | cdf underflows: `logcdf(1e-3, 100)` = -inf, true -1054.5149034537772 (loss) |
| logsf | log(sf) | logcdf(n - x) by symmetry | sf underflows: `logsf(99.999, 100)` = -inf, true -1054.5149034532997 (loss) |
| ppf | numerical inversion of cdf | for q <= 1/n!: closed form `x = exp((log(q) + gammaln(n+1))/n)`; otherwise inversion is fine but slow | accurate at the one point tried: `ppf(1e-20, 10)` = 0.04528728688116763, 2.07 ulp (ok) |
| isf | ppf(1 - q) | `n - ppf(q)` by symmetry (with the closed form above for q <= 1/n!) | small q: 1 - q rounds; `isf(1e-20, 10)` = 10.0, true 9.954712713118832, 2.0e13 ulp (severe) |

## Shared code

none found. The pdf/cdf/sf go through `scipy.interpolate.BSpline` (de Boor),
which is accurate here. sc.comb and sc.stirling2 are used only in `_munp`
(moments), which is outside the scope of this review.

## Existing GitHub issues

No findings. For context: PR #20481 (merged, closes #14806): ENH: stats: Add the Irwin-Hall distribution - the review discussed tail cancellation in symmetric distributions, which relates to the inherited isf = ppf(1 - q) opportunity.

(web search; gh unavailable)

## Verification

pending
