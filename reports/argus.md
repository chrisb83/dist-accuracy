# argus: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `argus_gen`, lines 11888-12087 (helper `_argus_phi`, lines 11874-11885)
- **Implemented:** pdf, logpdf, cdf, sf
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf), ppf = numerical inversion of cdf, isf = ppf(1 - q)
- **Related:** none; helpers: `_argus_phi` (also used by `_stats`), `_norm_pdf_logC`; special functions: sc.gammainc (via `_argus_phi`), sc.ive (`_stats` only)
- **Reference:** `refs/argus_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (1), (0.001), (1e-50), (1e-100), (1e-110), (1e-200), (0.1), (10), (100), (1e4), (1e155)
- **Worst finding:** loss in cdf, sf, pdf, logpdf

## Summary

sf, pdf and logpdf are accurate for moderate chi (sf within 28 ulp, pdf/logpdf
within conditioning near x = 1). The cdf is computed as `1 - sf` and loses
every digit for small x (any chi, x below ~1e-8) and for large chi everywhere
except close to x = 1 (chi = 10: x <= 0.45; chi = 100: x <= 0.99). For tiny chi
the normalizing constant `_argus_phi(chi) = gammainc(1.5, chi**2/2)/2`
underflows (chi < ~1e-102.5), so pdf/logpdf return inf and sf/cdf return nan;
before that, `3*log(chi) - log(_argus_phi(chi))` cancels and logpdf loses up
to 1.6e4 ulp. sf returns 0 near x = 1 for small chi, and logpdf returns -inf
for chi > 1.34e154 because `chi**2` overflows.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | small x (any chi); large chi away from x = 1 | `1 - sf` cancels (pattern 1) | 0.0 for 1.2e-20; 0.0 for 4.8e-211 | loss |
| F2 | pdf, logpdf, sf, cdf | chi < ~1e-102.5 | `_argus_phi(chi)` underflows to 0 (patterns 4, 10) | inf, nan | loss |
| F3 | sf (cdf) | small chi, chi**2 (1 - x**2)/2 < ~1e-205 | `_argus_phi(chi*sqrt(y))` underflows in the numerator (pattern 4) | 0.0 for 2.8e-9 | loss |
| F4 | logpdf | chi > 1.34e154 | `chi**2` overflows (pattern 3) | -inf for -9.95e307 | loss |
| F5 | logpdf | small chi, near the zero of logpdf | `3*log(chi) - log(_argus_phi(chi))` cancels (pattern 5) | 1.61e4 ulp | moderate |
| F6 | sf | small chi (chi <= 1e-3) | sc.gammainc(1.5, z) inaccurate for tiny z (pattern 10, shared) | 383 ulp | minor |
| F7 | pdf | large -logpdf (tiny x, or chi**2 (1-x**2)/2 large) | `exp(logpdf)` turns absolute error of logpdf into relative error | 672 ulp | minor |

### F1: cdf = 1 - sf cancels for small x and for large chi

- **Code:** `_cdf`, line 11943: `return 1.0 - self._sf(x, chi)`
- **Mechanism:** pattern 1. The cdf is small in two regimes and `1 - sf`
  keeps only the digits of sf beyond the leading ones: (a) for small x,
  cdf ~ C(chi) x**2 (about 1.22 x**2 for chi = 1), so the result is wrong for
  x < ~1e-4 and exactly 0 (or a multiple of eps) once x**2 < eps; (b) for large
  chi the mass concentrates at 1 - x ~ 1/chi**2 and cdf ~ exp(-chi**2 x**2/2)
  is tiny over most of the support. In the bulk the cancellation also amplifies
  the relative error of sf by sf/cdf (with F6, 2.5e4 ulp for chi = 1e-100).
- **Regime:** x < ~1e-3 for every chi (loss for x <= 1e-8 or 1e-10); for
  chi = 10 loss for x <= 0.45 and severe up to x = 0.8; chi = 100 loss at
  x = 0.95 and 0.99; chi = 1e4 loss at x = 0.999999. 144-162 loss points in the low tail
  over all parameter sets.
- **Reproducer:** `python accuracy.py refs/argus_ref.py --at cdf 1e-10 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-10, 1)` | 0.0 | 1.2174747521859012e-20 | inf | 2 | loss |
  | `cdf(1e-08, 1)` | 1.1102230246251565e-16 | 1.2174747521859012e-16 | 4.0e+14 | 2 | severe |
  | `cdf(1e-05, 1)` | 1.2174716790269713e-10 | 1.2174747521859014e-10 | 1.1e+10 | 2 | severe |
  | `cdf(0.001, 1)` | 1.2174747520132811e-06 | 1.2174747521857997e-06 | 6.4e+05 | 2 | moderate |
  | `cdf(1e-05, 0.001)` | 1.4999890218803102e-10 | 1.4999996999625175e-10 | 3.2e+10 | 2 | severe |
  | `cdf(0.5, 10)` | 3.3306690738754696e-16 | 3.623303813850349e-16 | 3.6e+14 | 24.7 | severe |
  | `cdf(0.95, 100)` | 0.0 | 4.767900552974467e-211 | inf | 9.02e+03 | loss |
  | `cdf(0.999999, 10000)` | 0.0 | 4.218750983153304e-43 | inf | 1.0e+08 | loss |
  | `cdf(0.1, 1e-100)` | 0.014962437264362527 | 0.014962437264446246 | 2.52e+04 | 1.99 | moderate |

- **Fix direction:** compute the numerator directly: with a = chi**2/2,
  b = a(1 - x**2), cdf = (gammaincc(1.5, b) - gammaincc(1.5, a)) / gammainc(1.5, a)
  for large chi, and a series in chi**2 x**2/2 (leading term C x**2/2,
  C = chi**3 exp(-a) / (sqrt(2 pi) Psi(chi))) for small x.

### F2: normalizing constant underflows for tiny chi

- **Code:** `_argus_phi`, line 11885: `sc.gammainc(1.5, chi**2/2) / 2`, used at
  line 11936 (`- np.log(_argus_phi(chi))`) and line 11946 (`/ _argus_phi(chi)`).
- **Mechanism:** _argus_phi(chi) ~ chi**3 / (3 sqrt(2 pi)) underflows: it is
  1.33e-307 at chi = 1e-102 and 0.0 at chi = 1e-103 (true 1.33e-310; scipy's
  gammainc flushes the subnormal result to 0). For chi < 1.5e-162 already
  `chi**2` underflows to 0. Then logpdf = A + ... with A = +inf, pdf = inf,
  sf = 0/0 = nan and cdf = 1 - nan = nan, at every x. The final results are
  ordinary numbers (in the limit chi -> 0 the distribution has pdf 3x sqrt(1-x**2)).
- **Regime:** chi < ~1e-102.5 (between 1e-103 and 1e-102), every x.
- **Reproducer:** `python accuracy.py refs/argus_ref.py --at logpdf 0.5 1e-103`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.5, 1e-103)` | inf | 0.26162407188227393 | inf | 2.55 | loss |
  | `pdf(0.5, 1e-103)` | inf | 1.299038105676658 | inf | 0.667 | loss |
  | `sf(0.5, 1e-103)` | nan | 0.649519052838329 | inf | 1 | loss |
  | `cdf(0.5, 1e-103)` | nan | 0.350480947161671 | inf | 1.85 | loss |
  | `logpdf(1e-300, 1e-110)` | inf | -689.6769156095456 | inf | 0.00145 | loss |
  | `pdf(1e-300, 1e-110)` | inf | 3e-300 | inf | 1 | loss |

- **Fix direction:** use gammainc(1.5, z) = z**1.5 exp(-z) hyp1f1(1, 2.5, z) / Gamma(2.5):
  A = log(3) + chi**2/2 - log(hyp1f1(1, 2.5, chi**2/2)) (no chi**3 formed), and
  sf = y**1.5 exp(chi**2 x**2/2) hyp1f1(1, 2.5, b) / hyp1f1(1, 2.5, a) for small chi.

### F3: sf underflows near x = 1 for small chi

- **Code:** `_sf`, line 11946: `_argus_phi(chi * np.sqrt((1 - x)*(1 + x))) / _argus_phi(chi)`
- **Mechanism:** pattern 4. The numerator gammainc(1.5, b), b = chi**2 (1 - x**2)/2,
  behaves like b**1.5 and underflows (to 0, or subnormal) once b < ~1e-205,
  although the ratio is ~ (1 - x**2)**1.5, a normal number. sf returns 0 and
  the cdf rounds to 1.0.
- **Regime:** chi**2 (1 - x**2)/2 < ~1e-205 with chi > ~1e-102.5 (below that F2);
  e.g. chi = 1e-100: x >= 1 - 1e-8 (loss), x = 0.999999 already loss.
- **Reproducer:** `python accuracy.py refs/argus_ref.py --at sf 0.999999 1e-100`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(0.999999, 1e-100)` | 0.0 | 2.8284250035481114e-09 | inf | 1.5e+06 | loss |
  | `sf(0.99999999, 1e-100)` | 0.0 | 2.8284271248512346e-12 | inf | 1.5e+08 | loss |
  | `sf(0.9999999999999999, 1e-100)` | 0.0 | 3.3087224502121103e-24 | inf | 1.4e+16 | loss |
  | `cdf(0.999999, 1e-100)` | 1.0 | 0.999999997171575 | 1.3e+07 | 0.00424 | severe |

- **Fix direction:** the hyp1f1 form of F2,
  sf = y**1.5 exp(chi**2 x**2/2) hyp1f1(1, 2.5, b)/hyp1f1(1, 2.5, a), y = (1-x)(1+x).

### F4: logpdf returns -inf for chi > 1.34e154

- **Code:** `_logpdf`, line 11937: `... - chi**2 * y / 2`
- **Mechanism:** pattern 3. `chi**2` overflows to inf for chi > 1.34e154,
  so the last term is -inf (or nan at y = 0) although chi**2 y/2 is finite
  for y up to ~3.6e308/chi**2.
- **Regime:** chi > 1.34e154, x where chi**2 (1 - x**2)/2 < 1.8e308. Edge case
  (pdf underflows there anyway; only logpdf is affected).
- **Reproducer:** `python accuracy.py refs/argus_ref.py --at logpdf 0.99 1e155`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.99, 1e155)` | -inf | -9.950000000000008e+307 | inf | 98.5 | loss |
  | `logpdf(0.9999999999999999, 1e155)` | -inf | -1.1102230246251566e+294 | inf | 9.0e+15 | loss |

- **Fix direction:** `-chi * (chi * y / 2)` (or `-0.5 * (chi*sqrt(y))**2` evaluated in an order that stays finite).

### F5: logpdf cancels in 3*log(chi) - log(_argus_phi(chi)) for small chi

- **Code:** `_logpdf`, line 11936: `A = 3*np.log(chi) - _norm_pdf_logC - np.log(_argus_phi(chi))`
- **Mechanism:** pattern 5. For small chi, `3*log(chi)` and `log(_argus_phi(chi))`
  are both about 3 log(chi) (-690 for chi = 1e-100) while A tends to log(3) = 1.0986.
  The absolute error of A is ~|3 log chi| eps plus the error of
  sc.gammainc for tiny arguments (F6): A has 20 ulp at chi = 1e-3,
  267 ulp at chi = 1e-100 (measured with a scratch probe). Where logpdf itself
  is small (it crosses 0 near x = 0.35) this absolute error becomes a large relative
  error.
- **Regime:** chi << 1, largest near the zero of logpdf (x ~ 0.35, cond (arg) 52)
  and growing with |log chi|; 1.02e3 ulp at x = 0.5 for chi = 1e-102.
- **Reproducer:** `python accuracy.py refs/argus_ref.py --at logpdf 0.35 1e-100`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.35, 1e-100)` | -0.016548997651685438 | -0.01654899765162614 | 1.61e+04 | 52 | moderate |
  | `logpdf(0.35, 0.001)` | -0.0165491364016393 | -0.01654913640163471 | 1.25e+03 | 52 | moderate |
  | `logpdf(0.35, 1e-50)` | -0.016548997651628594 | -0.01654899765162614 | 668 | 52 | minor |
  | `logpdf(0.5, 1e-102)` | 0.26162407188221476 | 0.26162407188227393 | 1.02e+03 | 2.55 | moderate |

- **Fix direction:** A = log(3) + chi**2/2 - log(hyp1f1(1, 2.5, chi**2/2))
  (exact identity; for small chi, A = log(3) - log1p(series)).

### F6: sf inherits sc.gammainc's error for tiny arguments

- **Code:** `_argus_phi`, line 11885: `sc.gammainc(1.5, chi**2/2) / 2`; `_sf` line 11946.
- **Mechanism:** pattern 10. sc.gammainc(1.5, z) itself has relative errors
  growing as z -> 0 (scratch probe against mpmath: 1.8 ulp at z = 0.1,
  13.7 ulp at 1e-20, 127 ulp at 1e-50, 112 ulp at 5e-101, 430 ulp at 1e-200).
  sf is a ratio of two such values.
- **Regime:** chi <= 1e-3, bulk of x (cond (arg) < 1). 97 ulp at chi = 1e-3,
  277 at 1e-50, 383 at 1e-100.
- **Reproducer:** `python accuracy.py refs/argus_ref.py --at sf 0.1 1e-100`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(0.1, 1e-100)` | 0.9850375627356375 | 0.9850375627355538 | 383 | 0.0303 | minor |
  | `sf(0.3, 1e-50)` | 0.868084673289472 | 0.8680846732894205 | 267 | 0.297 | minor |

- **Fix direction:** the hyp1f1 form of F2/F3 avoids gammainc for small chi;
  otherwise fix gammainc's small-z path (see Shared code).

### F7: pdf = exp(logpdf) amplifies the absolute error of logpdf

- **Code:** `_pdf`, line 11940: `return np.exp(self._logpdf(x, chi))`
- **Mechanism:** an absolute error delta in logpdf becomes a relative error
  delta in pdf. For tiny x (log(x) ~ -645 at x = 1e-280) or large chi**2 y/2,
  |logpdf| is hundreds, so pdf carries ~|logpdf| ulp although its cond (arg)
  is 1.
- **Regime:** |logpdf| >~ 50: x < ~1e-20 for all chi, and the left part of the
  support for chi >= 10 (191 minor points in the low tail over all sets).
- **Reproducer:** `python accuracy.py refs/argus_ref.py --at pdf 1e-280 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-280, 0.001)` | 2.9999993999995865e-280 | 2.999999400000034e-280 | 672 | 1 | minor |
  | `pdf(1e-290, 1)` | 2.4349495043720213e-290 | 2.4349495043718023e-290 | 405 | 1 | minor |

- **Fix direction:** compute the pdf as a product,
  `exp(A) * x * sqrt((1-x)*(1+x)) * exp(-chi**2*y/2)`, so only the
  genuinely exponential factor goes through exp.

## RuntimeWarnings

The harness evaluates each parameter set as one array call, so warnings are
per set, not per point.
- `underflow encountered in multiply` in logpdf and pdf for every chi: `x*x`
  underflows for tiny x (lines 11935, 11937). The results there are accurate
  (log1p(-0) = 0 is right), so the warning is spurious.
- `underflow encountered in exp` in pdf for chi >= 10: true result subnormal or
  0; spurious.
- `overflow encountered in square` for chi = 1e155: in cdf, sf and pdf the
  results are correct (spurious warning); in logpdf it accompanies F4.
- `invalid value encountered in add/divide` and `underflow encountered in square`
  for chi = 1e-110 and 1e-200: accompany the losses of F2.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| sf | full grid, chi in {0.1, 1, 10, 100, 1e4, 1e155}, including x -> 1 (1 - 1e-16) | 28 ulp (minor points, no clear mechanism beyond gammainc rounding) |
| pdf | x -> 1 (`1 - x*x` rounding, hypothesis H4), large chi | within conditioning (5.0e6 ulp at cond 9.9e7 for chi = 1e4) |
| logpdf | x -> 1, chi in {1, 10, 100, 1e4} full grid | within conditioning (1.8e5 ulp at cond 9.8e6) |
| logpdf | chi = 0.1, near the zero crossing | 245 ulp (minor, same mechanism as F5 with smaller A error) |
| cdf | x near 1, chi <= 1 (cdf near 1) | within conditioning |
| cdf, sf | chi = 1e155 | 0 ulp |

## Suspected, not confirmed

none. (H4, `y = 1.0 - x*x` and `log1p(-x*x)` near x = 1, was tested at
x = 1 - 1e-16 ... 1 - 1e-2 for all sets and stays within conditioning.)

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | log of the direct numerator of F1: log(gammaincc(1.5, b) - gammaincc(1.5, a)) - log(gammainc(1.5, a)), via log1mexp of log-gammaincc difference; small-x series 2 log(x) + log(C/2) | all regimes of F1; `logcdf(1e-10, 1)` = -inf, reference -45.85492302154564 (loss) |
| logsf | log(sf) | 1.5 log1p(-x**2) + chi**2 x**2/2 + log(hyp1f1(1, 2.5, b)/hyp1f1(1, 2.5, a)) | regimes of F2/F3; `logsf(1 - 1.1e-16, 1e-100)` = -inf, reference -54.06548008367574 (loss) |
| ppf | numerical inversion of cdf | closed form via gamma inverses: x = sqrt(1 - 2 z/chi**2) with z = gammainccinv(1.5, gammaincc(1.5, a) + q gammainc(1.5, a)) (for small q compute 1 - 2z/chi**2 as 2(a - z)/chi**2) | small q (inherits F1); `ppf(1e-30, 1)` = 6.5e-14, true ~9.06e-16 (leading-order sqrt(q/1.2175), computed by hand, not by the harness) |
| isf | ppf(1 - q) | x = sqrt(1 - 2 gammaincinv(1.5, q gammainc(1.5, a))/chi**2) | small q, where 1 - q rounds; also slow (numerical inversion) |

## Shared code

- `sc.gammainc(1.5, z)` loses accuracy for tiny z: 127 ulp at z = 1e-50,
  430 ulp at z = 1e-200 (scratch probe against mpmath, `scratch/argus/probe1.py`),
  and returns 0 where the true value is subnormal. This is a scipy.special
  issue (F6). It may also affect other distributions that call gammainc
  with small arguments (e.g. gamma, chi2, chi cdfs in their left tail), but
  only a = 1.5 was checked here.
- `_argus_phi` (F2) is also used by `argus._stats` (mean and variance),
  which therefore inherits the tiny-chi underflow.

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e-10, 1)` | 0.0 | 1.2174747521859012e-20 | verified | reference by mp.quad of the unnormalized pdf in u = 1 - x**2, normalized by its own integral (`scratch/argus/verify/verify.py`); also confirmed `cdf(1e-08, 1)` 3.97e14 ulp, `cdf(1e-05, 0.001)` 3.21e10 ulp, `cdf(0.5, 10)` 3.64e14 ulp; line 11943 `1.0 - self._sf(x, chi)` matches |
| F1 | `cdf(0.95, 100)` | 0.0 | 4.767900552974467e-211 | verified | also `cdf(0.999999, 1e4)` = 0.0 vs 4.218750983153304e-43 |
| F2 | `logpdf(0.5, 1e-103)` | inf | 0.26162407188227393 | verified | `pdf` inf vs 1.299038105676658, `sf` nan vs 0.649519052838329, `cdf` nan vs 0.350480947161671, `pdf(1e-300, 1e-110)` inf vs 3e-300; confirmed sc.gammainc(1.5, 5e-207) = 0.0 and 2.66e-307 at chi = 1e-102 (line 11885) |
| F3 | `sf(0.999999, 1e-100)` | 0.0 | 2.8284250035481114e-09 | verified | also `sf(0.9999999999999999, 1e-100)` 0.0 vs 3.3087224502121103e-24, `cdf(0.999999, 1e-100)` 1.0 vs 0.999999997171575 (1.27e7 ulp); numerator gammainc(1.5, 1.0e-206) = 0.0 (line 11946) |
| F4 | `logpdf(0.99, 1e155)` | -inf | -9.950000000000008e+307 | verified | also `logpdf(1 - 1.1e-16, 1e155)` -inf vs -1.1102230246251566e+294; float64(1e155)**2 = inf (line 11937) |
| F5 | `logpdf(0.35, 1e-100)` | -0.016548997651685438 | -0.01654899765162614 | verified | 1.61e4 ulp; `logpdf(0.35, 0.001)` 1.25e3 ulp, `logpdf(0.5, 1e-102)` 1.02e3 ulp; verifier probe of A (line 11936) vs log(3) + a - log(hyp1f1(1, 2.5, a)): 245 ulp at chi = 1e-100, 18 ulp at 1e-3 (report: 267, 20; same order) |

All references evaluated at 100 and 200 digits; agreement better than 1e-98 relative for every call.

Verified by dist-accuracy-verifier: 5 of 5 verified.
