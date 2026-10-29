# fatiguelife: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `fatiguelife_gen`, lines 2414-2488
- **Implemented:** pdf (= exp(logpdf)), logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** none; helpers: `_norm_cdf` (= `sc.ndtr`), `_norm_sf` (= `ndtr(-x)`), `_norm_ppf` (= `sc.ndtri`); special functions: sc.ndtr, sc.ndtri
- **Reference:** `refs/fatiguelife_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (29), (1), (0.1), (0.001), (1e-155), (1000), (1e6), (1e100)
- **Worst finding:** loss in ppf, isf, logpdf, pdf, sf; severe in cdf

## Summary

cdf and sf are accurate (within conditioning) over the whole grid except at
x = 1 +- 1 ulp with tiny c, where `sqrt(x) - 1/sqrt(x)` rounds to 0. ppf and
isf lose accuracy in the lower half of the distribution (ppf for q < 0.5,
isf for q > 0.5) because `tmp + sqrt(tmp**2 + 4)` cancels for large
`|tmp| = c*|ndtri(q)|`. The error grows like tmp**2: 500 ulp for c = 29,
q = 0.01; 3.2e11 ulp for c = 1e6; and the result is 0.0 once |tmp| > ~1e8.
logpdf returns -inf (and pdf returns nan) for x > 1.34e154 because
`(x-1)**2` overflows. It also returns nan at x = 1 when c**2 underflows
(c < ~1.5e-162). pdf = exp(logpdf) loses up to ~600 ulp when large log terms
cancel (large c).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf, isf | lower half (ppf q < 0.5, isf q > 0.5), large c*abs(ndtri(q)) | `tmp + sqrt(tmp**2 + 4)` cancels (pattern 8) | 0.0 for 1.6e-199; 3.2e11 ulp | loss |
| F2 | logpdf, pdf | x > 1.34e154, any c (pdf: c > ~1e75) | `(x-1)**2` overflows (pattern 3) | logpdf -inf, pdf nan | loss |
| F3 | logpdf, pdf | x == 1, c < ~1.5e-162 | `c**2` underflows to 0, 0/0 (pattern 4) | nan for 390.5 | loss |
| F4 | cdf, sf | x = 1 + 2**-52, c < ~1e-16 | `sqrt(x) - 1/sqrt(x)` rounds to 0 (pattern 5) | cdf 0.5 for 1.0; sf 0.5 for 1.6e-109 | loss |
| F5 | pdf | large c, tiny or huge x | exp of a sum of large cancelling log terms (pattern 5) | 575 ulp | minor |

### F1: ppf and isf cancel in tmp + sqrt(tmp**2 + 4) for negative tmp

- **Code:** `_ppf`, lines 2466-2467: `tmp = c * _norm_ppf(q)`;
  `0.25 * (tmp + np.sqrt(tmp**2 + 4))**2`; `_isf`, lines 2473-2474, same
  formula with `tmp = -c * _norm_ppf(q)`.
- **Mechanism:** for tmp < 0 the two terms have opposite signs and nearly
  equal size. The sum is 2/|tmp| + ..., so the relative error is about
  tmp**2 ulp (doubled by the square). Once tmp**2 + 4 rounds to tmp**2
  (|tmp| > ~1e8), the sum is exactly 0 and ppf returns 0.0. This is pattern 8.
- **Regime:** ppf with q < 0.5, isf with q > 0.5. The error is roughly
  c**2 * ndtri(q)**2 ulp: about 150 ulp for c = 1 at q = 1e-300; 500 ulp
  for c = 29 at q = 0.01; 5e5 ulp for c = 29 at q = 1e-300; 8e5 ulp for c = 1000 at
  q = 0.1. Total loss (0.0) for c*abs(ndtri(q)) > ~1e8, while the true
  quantile (~1/tmp**2) is a normal double up to |tmp| ~ 1e154. The upper
  half (ppf q > 0.5, isf q < 0.5) is accurate (max 3.4 ulp).
- **Reproducer:** `python accuracy.py refs/fatiguelife_ref.py --at ppf 0.1 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.4, 1e100)` | 0.0 | 1.5580023717212127e-199 | inf | 8.17 | loss |
  | `ppf(0.1, 1e9)` | 0.0 | 6.088745603777447e-19 | inf | 0.889 | loss |
  | `isf(0.9, 1e100)` | 0.0 | 6.0887456037774456e-201 | inf | 8 | loss |
  | `ppf(0.1, 1e8)` | 5.551115123125783e-17 | 6.088745603777447e-17 | 4.0e14 | 0.889 | severe |
  | `ppf(0.1, 1e6)` | 6.08917877600347e-13 | 6.088745603770033e-13 | 3.2e11 | 0.889 | severe |
  | `isf(0.9, 1e6)` | 6.08917877600347e-13 | 6.088745603770031e-13 | 3.2e11 | 8 | severe |
  | `ppf(0.1, 1000)` | 6.088738190317177e-07 | 6.088738189224128e-07 | 8.1e5 | 0.889 | moderate |
  | `ppf(1e-300, 29)` | 8.663530392144682e-07 | 8.663530391108032e-07 | 5.4e5 | 0.00146 | moderate |
  | `ppf(0.01, 29)` | 0.00021961596649374999 | 0.0002196159664937736 | 484 | 0.322 | minor |
  | `isf(0.99, 29)` | 0.00021961596649374999 | 0.00021961596649377368 | 486 | 31.9 | minor |
  | `ppf(1e-300, 1)` | 0.0007275443702921276 | 0.0007275443702921522 | 152 | 0.00145 | minor |

  Over the full grid: ppf has 37 loss, 64 severe, 34 moderate and 35 minor points,
  and isf has 24 loss, 29 severe, 10 moderate and 6 minor points.
- **Fix direction:** for tmp < 0 use the conjugate form
  `4 / (np.sqrt(tmp**2 + 4) - tmp)**2` (equivalently `1/ppf(1-q)`, since the
  distribution is invariant under x -> 1/x). Use `np.hypot(tmp, 2)` to avoid
  overflow of tmp**2.

### F2: logpdf overflows in (x-1)**2 for x > 1.34e154

- **Code:** `_logpdf`, line 2459: `(x-1)**2 / (2.0*x*c**2)`; `_pdf`, line 2456: `np.exp(self._logpdf(x, c))`.
- **Mechanism:** `(x-1)**2` is inf for x > 1.34e154, so the term is inf and
  logpdf is -inf. If `2*x*c**2` also overflows (large c), the result is
  inf/inf = nan. The true term is about x/(2c**2), which is finite. This is
  pattern 3.
- **Regime:** logpdf for x > 1.34e154 and every c (the true value
  -x/(2c**2) is finite for every c > ~1e-77 at x = 1e154). pdf is affected only
  where its true value is still a normal double, which needs
  x/(2c**2) < ~708, i.e. c > ~3e75. There pdf returns nan (or 0).
- **Reproducer:** `python accuracy.py refs/fatiguelife_ref.py --at logpdf 1.4e154 29`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.4e154, 29)` | -inf | -8.323424494649227e+150 | inf | 1 | loss |
  | `logpdf(1e300, 1e100)` | nan | -5e+99 | inf | 1 | loss |
  | `pdf(1e160, 1e100)` | nan | 1.9947114020071635e-181 | inf | 0.5 | loss |
  | `pdf(1.4e154, 1e100)` | nan | 1.6858388283618386e-178 | inf | 0.5 | loss |

  This gives 108 loss points in the logpdf high tail (all parameter sets),
  plus 8 pdf loss points for c = 1e100.
- **Fix direction:** write the term as `z**2 / 2` with
  `z = (x - 1) / (c * sqrt(x))` (or `((x-1)/c) * ((x-1)/(c*x)) / 2`). This
  never forms (x-1)**2 or c**2 and also fixes F3.

### F3: logpdf and pdf return nan at x = 1 when c**2 underflows

- **Code:** `_logpdf`, line 2459: `(x-1)**2 / (2.0*x*c**2)`.
- **Mechanism:** for c < ~1.5e-162, `c**2` is 0, so at x == 1 the term is
  0/0 = nan. For 1.5e-162 < c < 1.5e-154, `c**2` is subnormal. The
  measurements with c = 1e-155 stay within conditioning (68 ulp, cond 2e8),
  because the term is huge and ill-conditioned for every x != 1. This is
  pattern 4.
- **Regime:** x == 1 exactly, c < ~1.5e-162. For x != 1, -inf is the correct
  rounding, because the true logpdf is below -1.8e308.
- **Reproducer:** `python accuracy.py refs/fatiguelife_ref.py --at logpdf 1 1e-170`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.0, 1e-170)` | nan | 390.5205272757831 | inf | 0.00256 | loss |
  | `pdf(1.0, 1e-170)` | nan | 3.989422804014327e+169 | inf | 1 | loss |

- **Fix direction:** the same as F2: `z = (x - 1)/(c*sqrt(x))`, term `z**2/2`.

### F4: cdf and sf at x = 1 + 1 ulp with tiny c

- **Code:** `_cdf`, line 2463 and `_sf`, line 2470:
  `1.0 / c * (np.sqrt(x) - 1.0/np.sqrt(x))`.
- **Mechanism:** `np.sqrt(1 + 2**-52)` rounds to exactly 1.0, so
  `sqrt(x) - 1/sqrt(x)` is 0 and cdf = sf = 0.5. The true z is
  2.2e-16/c. More generally, `sqrt(x) - 1/sqrt(x)` near x = 1 has a
  relative error of about eps/|x - 1| (pattern 5). For the next double below 1
  (0.9999999999999999), sqrt does not round to 1, but z comes out three times
  too large (-3.3e-16/c instead of -1.1e-16/c). The measurement there falls
  within the (huge) conditioning: cdf(0.9999999999999999, 1e-17) = 1.5e-243
  vs 6.1e-29, rated ok.
- **Regime:** x = 1 + 2**-52 and c < ~1e-16, so that the true z is far from 0.
  For larger c, z is close to 0 and the error is within conditioning (e.g.
  c = 0.001: 798 ulp, cond 798).
- **Reproducer:** `python accuracy.py refs/fatiguelife_ref.py --at sf 1.0000000000000002 1e-17`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1.0000000000000002, 1e-17)` | 0.5 | 1.5550263204185765e-109 | 1.4e124 | 2.2e18 | loss |
  | `cdf(1.0000000000000002, 1e-155)` | 0.5 | 1.0 | 2.3e15 | 0 | severe |
  | `cdf(1.0000000000000002, 1e-17)` | 0.5 | 1.0 | 2.3e15 | 3.46e-91 | severe |

  Severity per method: sf loss, cdf severe. The finding carries the worse
  of the two (loss). For sf the value is extremely ill-conditioned with respect to x
  (cond (arg) 2.2e18). Still, returning 0.5 for a value of 1.6e-109 is
  wrong for the exact input.
- **Fix direction:** `z = (x - 1) / (c * np.sqrt(x))`. Here `x - 1` is exact
  near 1 (Sterbenz), so z keeps full relative accuracy.

### F5: pdf = exp(logpdf) amplifies the absolute error of large log terms

- **Code:** `_pdf`, line 2456: `np.exp(self._logpdf(x, c))`, with
  `_logpdf`, lines 2459-2460 summing `log(x+1)`, `-log(2*c)`, `-1.5*log(x)`.
- **Mechanism:** logpdf is a sum of terms of size up to ~1.5*|log x| +
  |log(2c)| (several hundred). These terms cancel to a smaller result. The
  absolute rounding error (~eps times the largest term) becomes a relative
  error of the same size in exp. This is pattern 5 in log space.
- **Regime:** large c (where the exponential term is negligible and the pdf
  is a power law), with |log x| large. For moderate c the same effect is
  hidden by the conditioning, which is ~|log pdf| in the tails (max
  396 ulp, cond 451).
- **Reproducer:** `python accuracy.py refs/fatiguelife_ref.py --at pdf 1e-180 1e100`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-180, 1e100)` | 1.9947114020069084e+169 | 1.9947114020071632e+169 | 575 | 1.5 | minor |
  | `pdf(1e150, 1e100)` | 1.9947114020070195e-176 | 1.9947114020071632e-176 | 325 | 0.5 | minor |
  | `pdf(1e-160, 1e100)` | 1.9947114020070265e+139 | 1.9947114020071634e+139 | 309 | 1.5 | minor |

  This gives 70 minor pdf points in total: 61 for c = 1e100, the rest for
  c = 1e6, 1000 and 1e-155.
- **Fix direction:** evaluate pdf directly as
  `_norm_pdf(z) * (x + 1) / (2*c*x*sqrt(x))`, with z from F4. Split the
  powers (e.g. `x**-0.5 * x**-1`) to avoid overflow for tiny x.

## RuntimeWarnings

- logpdf/pdf: "overflow encountered in square" from `(x-1)**2` (F2), plus
  "overflow encountered in multiply" and "invalid value encountered in divide"
  for large c (F2, nan results). With small c there are also "underflow
  encountered in square" and "divide by zero encountered in divide" from
  `c**2` (F3). The divide-by-zero warning also appears for x != 1, where
  -inf is the correct result. "underflow encountered in exp" in pdf where the
  true pdf underflows (correct). The harness evaluates whole arrays, so the
  warnings are recorded per parameter set. Some are spurious with respect to
  individual accurate points: e.g. logpdf for c = 29 at x = 1e-300 is exact,
  but the batch warns about overflow from the large-x points.
- ppf/isf: "underflow encountered in square" for c = 1e-155 (`tmp**2`
  underflows, harmless). Results there are exact (0 ulp), so the warning is
  spurious.
- cdf/sf: none.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid, all parameter sets except F4 point | 4.4e3 ulp at (0.001), x = 0.97, within cond 3.0e4; one minor point 74.7 ulp (cond 50.5) at (1e100), x = 1e-202 |
| sf | full grid, all parameter sets except F4 point | 798 ulp within cond 798; 4 minor points up to 672 ulp (cond 470) at (0.1), x = 9.5: z is formed in several roundings and its relative error is amplified by ~z**2 (cond (shapes) 762), slightly more than the x-conditioning accounts for |
| logpdf | low tail and bulk, all parameter sets except F2/F3 | 68.3 ulp at (1e-155), x = 0.99999999, cond 2e8 |
| pdf | all parameter sets except c = 1e100 (F2, F5) | 396 ulp at (1e6), within cond 451 |
| ppf | upper half q > 0.5, all parameter sets; small c (0.1, 0.001, 1e-155) everywhere | 3 ulp in the upper tail; 5.4 ulp for c = 0.1 |
| isf | q < 0.5, all parameter sets; small c everywhere | 3.4 ulp |

## Suspected, not confirmed

- logpdf with subnormal `c**2` (1.5e-162 < c < 1.5e-154) away from x = 1:
  the term carries up to ~1e-3 relative error, but the logpdf there is so
  sensitive to x (cond ~2e8 at c = 1e-155) that the measured error (68 ulp)
  stays within conditioning. A visible case would need |x - 1| near 1e-160,
  which is not representable next to 1.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | `sc.log_ndtr(z)`, z = (x - 1)/(c*sqrt(x)) | cdf underflows for z < ~-38 (e.g. c = 1, x < ~7e-4). `logcdf(1e-4, 1)` = -inf, true -5004.524158709197 (`log_ndtr` gives -5004.524158709199) |
| logsf | log(sf) | `sc.log_ndtr(-z)` | sf underflows for z > ~38 (e.g. c = 1, x > ~1500). `logsf(1e4, 1)` = -inf, true -5004.524158709197 |

## Shared code

none. The helpers `_norm_cdf`/`_norm_sf`/`_norm_ppf` (`sc.ndtr`, `sc.ndtri`)
were accurate for the arguments they received. All findings come from the
expressions in `fatiguelife_gen` itself.

## Existing GitHub issues

None found about F1-F5. For context:

- #13333 (closed PR): ENH: stats: Some more _sf and _isf implementations - added fatiguelife `_sf`/`_isf` and their upper-tail tests; the isf it added shares the cancelling formula of F1 (for q > 0.5)
- #17832 (open): Tracking issue: custom methods for continuous distributions - general tracker; logcdf/logsf via `log_ndtr` (Opportunities) would fit there

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(0.4, 1e100)` | 0.0 | 1.5580023717212127e-199 | verified | ref by conjugate form 4/(sqrt(t**2+4)-t)**2 with t = c*z from a log-space root of Phi(z) = q; cdf(x_ref) = q to 1e-200 |
| F1 | `ppf(0.1, 1e9)` | 0.0 | 6.088745603777447e-19 | verified | loss |
| F1 | `isf(0.9, 1e100)` | 0.0 | 6.0887456037774456e-201 | verified | loss |
| F1 | `ppf(0.1, 1e8)` | 5.551115123125783e-17 | 6.088745603777447e-17 | verified | 4.0e14 ulp |
| F1 | `ppf(0.1, 1e6)` | 6.08917877600347e-13 | 6.088745603770033e-13 | verified | 3.2e11 ulp |
| F1 | `isf(0.9, 1e6)` | 6.08917877600347e-13 | 6.088745603770031e-13 | verified | 3.2e11 ulp |
| F1 | `ppf(0.1, 1000)` | 6.088738190317177e-07 | 6.088738189224128e-07 | verified | 8.1e5 ulp; mechanism matches lines 2466-2467 / 2473-2474 |
| F1 | `ppf(1e-300, 29)` | 8.663530392144682e-07 | 8.663530391108032e-07 | verified | 5.4e5 ulp |
| F1 | `ppf(0.01, 29)` | 0.00021961596649374999 | 0.0002196159664937736 | verified | 484 ulp |
| F2 | `logpdf(1.4e154, 29)` | -inf | -8.323424494649227e+150 | verified | ref from the docstring pdf; `(x-1)**2` overflows at line 2459 |
| F2 | `logpdf(1e300, 1e100)` | nan | -5e+99 | verified | inf/inf |
| F2 | `pdf(1e160, 1e100)` | nan | 1.9947114020071635e-181 | verified | |
| F2 | `pdf(1.4e154, 1e100)` | nan | 1.6858388283618386e-178 | verified | |
| F3 | `logpdf(1.0, 1e-170)` | nan | 390.5205272757831 | verified | `c**2` underflows to 0, 0/0 at line 2459 |
| F3 | `pdf(1.0, 1e-170)` | nan | 3.989422804014327e+169 | verified | |
| F4 | `sf(1.0000000000000002, 1e-17)` | 0.5 | 1.5550263204185765e-109 | verified | ref by mp.quad of the pdf over [x, inf) (200 digits, fine breakpoints) and by erfc(z/sqrt2)/2 with z = (x-1)/(c*sqrt(x)); they agree to 1e-120. The 100-digit quad with coarse breakpoints differed by 7.6e-6 (quadrature error); the erfc route agrees to 1e-99 between 100 and 200 digits |
| F4 | `cdf(1.0000000000000002, 1e-155)` | 0.5 | 1.0 | verified | 2.3e15 ulp, severe |
| F4 | `cdf(1.0000000000000002, 1e-17)` | 0.5 | 1.0 | verified | 2.3e15 ulp, severe; mechanism matches lines 2463 / 2470 |

Verified by dist-accuracy-verifier: 4 of 4 verified.
