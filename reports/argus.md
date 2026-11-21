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

## Fix

- **Branch:** `argus` in scipy, based on main da43ab1957 (not committed)
- **Check script:** `dist-accuracy/fixes/argus.py`
- **Fixed:** F1 (where b = chi**2 (1 - x**2)/2 >= 1), F2, F3, F4, F5, F6, F7 (tiny x)
- **Not fixed:**
  - F1 for small x (x below ~0.1-0.3, depending on chi): this needs a series (e.g. a mixture of
    Beta(k+1, 3/2) cdfs); decided with the user to keep the change to closed forms.
    `cdf(1e-10, 1)` is still 0.
  - F7 for large chi: the remaining error comes from rounding chi**2 (1 - x**2)/2 inside exp,
    i.e. conditioning with respect to chi; fixing it would need extended precision.
  - `sc.gammainc(1.5, z)` itself (shared code) is unchanged. The small-chi sf and the normalizing
    constant no longer call it (F6 fixed there), but `_argus_phi` still does: the large-chi sf near
    x = 1 (numerator `_argus_phi(chi*sqrt(1 - x**2))` with a small argument, up to ~28 ulp) and
    `_stats` keep its error. A hyp1f1 branch in `_argus_phi` (<= 2 ulp) was tried and dropped to
    keep the change small; this belongs in `scipy.special`.
  - Opportunities: ppf/isf (closed form via gammaincinv/gammainccinv): left for a follow-up PR.
    logcdf: skipped, `log(cdf)` is accurate where the cdf does not underflow. logsf: needs no
    custom method any more, since the new sf does not underflow (sf >= ~3e-24 for x < 1).
  - `argus._stats` fails for chi < ~1e-102 (0/0, since Psi(chi) ~ chi**3 underflows; outside this
    review).

### Pull request summary

**ENH: stats.argus: improve accuracy of pdf, logpdf, cdf and sf for small and large chi**

`argus` lost accuracy in several regimes: for chi < ~1e-102 the normalizing constant
`_argus_phi(chi) ~ chi**3` underflowed, so pdf/logpdf returned inf and cdf/sf nan; for small chi,
logpdf lost up to 1.6e4 ulp through the cancellation of `3*log(chi)` and `log(_argus_phi(chi))`, and
sf returned 0 close to x = 1; the cdf was computed as `1 - sf` and returned 0 (or lost all digits)
for large chi, e.g. `cdf(0.95, 100)` = 0 instead of 4.8e-211; and logpdf returned -inf for
chi > 1.3e154 because `chi**2` overflowed.

The small-chi fixes rest on the Kummer representation of the regularized lower incomplete gamma
function. DLMF 8.5.1 gives `gamma(s, z) = z**s exp(-z) M(1, 1 + s, z) / s` for the
non-regularized function; dividing by `Gamma(s)` (`gammainc` is `P = gamma/Gamma`) gives

    P(s, z) = z**s exp(-z) M(1, s + 1, z) / Gamma(s + 1),

with M = `hyp1f1`. For s = 3/2 and a = chi**2/2, this makes `P(a) = gammainc(1.5, a)` a product
of `a**1.5 ~ chi**3` and a factor that is smooth and close to 1 (`exp(-a) hyp1f1(1, 2.5, a)`). So
the `chi**3` can be cancelled analytically in the normalizing constant
(`K = 3 exp(a) / hyp1f1(1, 2.5, a)`) and in the ratio `sf = P(b)/P(a)`, instead of forming
quantities that underflow and then dividing them. The series of `hyp1f1(1, 2.5, z)` has only
positive terms, so it is accurate for small arguments, where `gammainc(1.5, z)` loses up to
~100 ulp.

Changes in `argus_gen`:

- new `_log_norm_const(chi)`: log of the normalizing constant. For chi <= sqrt(2) it uses
  `log(3) + a - log(hyp1f1(1, 2.5, a))`, a = chi**2/2, which follows from the Kummer
  representation of the incomplete gamma function and does not form chi**3. For larger chi the
  previous expression is kept.
- `_logpdf`: uses `_log_norm_const`; `chi**2*y/2` is evaluated as `(0.5*chi)*(chi*y)` to avoid overflow.
- `_logpdf`, `_pdf`: `1 - x**2` is computed as `(1 - x)*(1 + x)` (as in `_sf`) and `log1p(-x**2)`
  as `log((1 - x)*(1 + x))`; rounding `x*x` near x = 1 cost up to 3.5e6 ulp in logpdf and 2.3e6
  ulp in pdf (e.g. x = 1 - 1e-8), now <= ~170 and ~13 ulp.
- `_pdf`: `x*exp(_log_norm_const(chi) + 0.5*log(y) - a*y)` instead of `exp(_logpdf)`, so
  log(x) no longer passes through exp (relative error ~|logpdf| ulp for tiny x before). log(y)
  stays inside exp so that x = 1 gives exp(-inf) = 0 when the normalizing constant overflows
  (chi > ~1e102.5); `x*sqrt(y)*exp(...)` gave 0*inf = nan there. Cost: <= ~|0.5 log y| ulp
  (<= ~20 ulp near x = 1).
- `_sf`: for chi <= sqrt(2), `y**1.5 * exp(a*x**2) * hyp1f1(1, 2.5, a*y) / hyp1f1(1, 2.5, a)`
  with y = (1 - x)(1 + x); unchanged for larger chi.
- `_cdf`: for b = a*(1 - x**2) >= 1, `(gammaincc(1.5, b) - gammaincc(1.5, a)) / gammainc(1.5, a)`;
  otherwise `1 - sf` as before.
- spurious overflow warnings for chi > 1.3e154 are suppressed (the results were and are correct).

Tests (in `TestArgus`, reference values from mpmath):

- new rows in the existing `test_pdf_small_chi`, `test_sf_small_chi`, `test_cdf_small_chi`
  (chi = 1e-103, 1e-200 and both sides of the switch at chi = sqrt(2)),
  `test_sf_near_1` (chi = 1e-100 and chi = 10, x close to 1);
- tighter tolerances: `test_pdf_small_chi` 1e-13 -> 5e-15, `test_sf_small_chi` 1e-14 -> 5e-15,
  `test_cdf_small_chi` 1e-12 -> 1e-13 (still `1 - sf` there);
- `test_argus_phi_small_chi`: four reference values recomputed (they were 1-2 ulp off), one
  rtol of 1e-14 for all rows (was 1e-14, and 1e-15 for two rows; gammainc reaches 4e-15 at
  chi = 1e-3);
- new `test_cdf_large_chi` (cdf = 1 - sf cancelled), `test_logpdf` (small, tiny and huge chi,
  x close to 1), `test_pdf_tiny_x`, `test_pdf_near_1` and `test_cdf_near_1_small_chi`;
- new `test_huge_chi` (chi = 1e155, x in {0.5, 0.99, 1}: pdf, cdf, sf are 0 or 1 without
  RuntimeWarnings, since the mass lies within 1 - x ~ 1/chi**2 of 1, beyond double resolution)
  and `test_large_chi_at_mode` (chi = 1e7 at the double nearest to the mode, ~1 - 1/(2 chi**2), the largest-chi
  regime where the bulk is still resolved; passes on main too).

34 of the parametrized tests fail on main.

Not addressed: the cdf for small x (still `1 - sf`), and the generic ppf/isf (follow-up).

### Derivations

Notation as in the code: `a = chi**2/2`, `y = 1 - x**2 = (1 - x)(1 + x)`, `b = a*y`,
`P(t) = gammainc(1.5, t)`, `Q(t) = gammaincc(1.5, t) = 1 - P(t)`, `M(z) = hyp1f1(1, 2.5, z)`.
The docstring pdf is `K x sqrt(y) exp(-b)` with `K = chi**3/(sqrt(2 pi) Psi(chi))`.

**Pass 1: algebra**

1. **Psi via P** (existing, docstring of `_argus_phi`): `Psi(chi) = P(a)/2`.
2. **Kummer form of P.** DLMF 8.5.1 is stated for the non-regularized lower incomplete gamma
   function: `gamma(s, z) = z**s exp(-z) M(1, 1 + s, z) / s`. `gammainc` is the regularized
   `P(s, z) = gamma(s, z) / Gamma(s)` (DLMF 8.2.4). Dividing 8.5.1 by `Gamma(s)` and using
   `s Gamma(s) = Gamma(s + 1)` turns the factor `1/s` into `1/Gamma(s + 1)`:
   `P(s, z) = z**s exp(-z) M(1, s + 1, z) / Gamma(s + 1)`.
   Notation pitfalls on the DLMF page: the bold **M** there is Olver's scaled function
   `M(a, b, z) / Gamma(b)`, in which the identity reads `P(s, z) = z**s exp(-z) Mbold(1, 1 + s, z)`
   (Mbold = bold M) without a gamma factor; `sc.hyp1f1` is the unscaled M, hence the `1/Gamma(s + 1)` here. The
   second form in 8.5.1, `gamma(s, z) = z**s M(s, 1 + s, -z) / s` (Kummer's transformation
   `M(a, b, z) = exp(z) M(b - a, b, -z)`), is equivalent but has alternating terms for z > 0;
   the form with `M(1, s + 1, z)` has only positive series terms, so it has no cancellation.
   For s = 3/2:
   `P(z) = z**1.5 exp(-z) M(z) / Gamma(2.5)`, with `Gamma(2.5) = 3 sqrt(pi)/4`.
   Numerical check: `mp.gammainc(1.5, 0, z, regularized=True) - z**1.5*mp.exp(-z)*mp.hyp1f1(1, 2.5, z)/mp.gamma(2.5)`
   is ~1e-32 at 30 digits for z = 0.7.
3. **Normalizing constant** (`_log_norm_const`, small branch). Insert 1 and 2 with
   `a**1.5 = chi**3 / 2**1.5`:
   `K = 2 chi**3 / (sqrt(2 pi) P(a)) = 2 chi**3 * 2**1.5 * (3 sqrt(pi)/4) / (sqrt(2 pi) chi**3 exp(-a) M(a)) = 3 exp(a) / M(a)`.
   So `log K = log(3) + a - log(M(a))`. The factor chi**3 cancels exactly. Limit chi -> 0:
   K -> 3, pdf -> 3 x sqrt(1 - x**2), the chi = 0 limit already used in `test_rvs_small_chi`.
   The large branch `3 log(chi) - log(sqrt(2 pi)) - log(P(a)/2)` is `log K` written directly
   (`_norm_pdf_logC = log(sqrt(2 pi))`).
4. **logpdf / pdf.** `logpdf = log K + log(x) + 0.5 log(y) - b` and
   `pdf = x sqrt(y) exp(log K - b)`; `b = a*y = (0.5*chi)*(chi*y)` is only a reordering.
5. **sf, small branch.** `sf = P(b)/P(a)` (existing). With 2:
   `P(b)/P(a) = (b/a)**1.5 exp(a - b) M(b)/M(a) = y**1.5 exp(a x**2) M(b)/M(a)`,
   since `b/a = y` and `a - b = a x**2`.
6. **cdf.** `cdf = 1 - P(b)/P(a) = (P(a) - P(b))/P(a) = (Q(b) - Q(a))/P(a)` (using `P = 1 - Q`).
7. **Switch-overs.** Both branches of `_log_norm_const` and `_sf` are the same function (3 and 5), so
   the switch at chi = sqrt(2) only changes rounding. Same for the cdf at b = 1 (6).

**Pass 2: numerics**

Error of the cdf (item 6). Let `D = P(a) - P(b) = Q(b) - Q(a)` (> 0). `1 - sf` rounds sf with
relative error ~eps, i.e. absolute error ~eps P(b)/P(a), so the relative error of the cdf is
~eps P(b)/D. The gamma difference has absolute error ~eps Q(b) (Q(a) <= Q(b)), i.e. relative
error ~eps Q(b)/D. The second form is better iff Q(b) < P(b), i.e. b > median of Gamma(3/2) =
1.18299. Switching at b = 1 costs at most a factor Q(1)/P(1) = 0.5724/0.4276 = 1.34 against
the optimal choice. For small x, b -> a and D -> 0 in both forms: this is the unfixed part of F1.

| Where | Trick | Avoids | Switch-over |
|---|---|---|---|
| `_log_norm_const` | `log(3) + a - log(M(a))` | underflow of `_argus_phi(chi) ~ chi**3` (F2), cancellation of `3 log(chi)` and `log(_argus_phi)` (F5) | chi <= sqrt(2) (a <= 1); M(a) grows like exp(a), and for a > 1 the old form has no cancellation |
| `_logpdf`, `_pdf` | `(0.5*chi)*(chi*y)` | overflow of `chi**2` for chi > 1.3e154 (F4) | - |
| `_pdf` | `x*exp(log K + 0.5 log(y) - b)` | relative error ~\|logpdf\| eps from `exp(logpdf)` for tiny x (F7); `0*inf` at x = 1 for chi > ~1e102.5 | - |
| `_sf` | `y**1.5 exp(a x**2) M(b)/M(a)` | underflow of `P(b) ~ b**1.5` near x = 1 (F3), of `P(a)` (F2), and the error of `gammainc(1.5, z)` for tiny z (F6) | chi <= sqrt(2) |
| `_sf`, `_cdf`, `_logpdf`, `_pdf` | `y = (1 - x)(1 + x)`, `log(y)` | rounding of `x*x` in `1 - x*x` and `log1p(-x*x)` near x = 1 (up to 3.5e6 ulp in logpdf); for small x, `log(y)` is as accurate as `log1p(-x*x)` within logpdf (<= 158 ulp, at a zero crossing of logpdf) | - |
| `_cdf` | `(Q(b) - Q(a))/P(a)` | cancellation in `1 - sf` when the cdf is small (F1) | b >= 1 (see above) |
| all | `np.errstate(over='ignore')` | spurious warnings when chi**2 overflows; `gammainc(1.5, inf) = 1` and `gammaincc(1.5, inf) = 0` give the correct limits | - |

`hyp1f1` (Boost-based in scipy) was checked against mpmath for z in [0, 1]: `_log_norm_const` is within
1.2 ulp and the new sf within 2.9 ulp for chi in [1e-300, sqrt(2)], x up to 1 - 2**-53.

### Verification

- `python fixes/argus.py <site-packages>`: 0 of 74 cases (one per x value of the new test
  rows, plus all `test_argus_phi_small_chi` rows) outside the test tolerance, 0 stale values.
- `_argus_phi` is unchanged apart from the overflow warning: plain gammainc has up to 4.1e-15
  relative error on the `test_argus_phi_small_chi` rows (chi = 1e-3), hence rtol 1e-14 there;
  `sf(1 - 2**-53, 10)` has 6.3e-15 (that row was dropped from `test_sf_near_1`).
- Existing tests with the new code: largest errors pdf small chi 3.6e-16, sf small chi 4.5e-16,
  cdf small chi 3.0e-14 (x = 0.1), basis for the tightened tolerances. The references use mpmath's incomplete gamma integrals over [0, b], [b, a] and [0, a]
  directly (no cancellation), with Psi via P(a) checked against Phi - chi phi - 1/2.
- New and tightened tests on main (main's source, new tests): 34 fail, at least one per fixed
  finding. Passing on main: the switch-over rows, `cdf(0.8, 3)`, `sf(0.999999, 10)` (8.5 ulp on
  main, inside 5e-15) and the rows of the existing tests where main was already accurate enough.
- `pixi run test`: `TestArgus` 101 passed; `test_distributions.py` 2589 passed;
  `test_continuous_basic.py -k argus` 13 passed. `pixi run lint`: all checks passed.
- Sweep against `refs/argus_ref.py` (chi from 1e-300 to 1e155, 21 x values up to 1 - 2**-53):
  pdf, logpdf and sf within 8 + cond ulp everywhere (largest ratio 3.3 for pdf at chi = 10,
  the unfixed F7 part); cdf 0.1-40 ulp where b >= 1 or x >= ~0.3 (`cdf(0.5, 10)`: 16 ulp, was
  3.6e14; `cdf(0.95, 100)`: 94 ulp at cond 9e3, was 0); for x <= 0.1 and moderate chi the cdf
  still has up to ~5e4 ulp (unfixed part of F1).
- Switch-overs: cdf monotone on 20001 points across b = 1 for chi in {1.4143, 1.5, 2, 3, 10,
  100}, |cdf + sf - 1| <= 1.3e-15; relative steps across chi = sqrt(2) at x = 0.5: logpdf
  1.7e-14 (absolute 1.3e-15, i.e. a few ulp of log K ~ 1.1), sf 1.3e-15, cdf 3.9e-15.
- No RuntimeWarnings for chi = 1e155 or 1e-300 (checked with `-W error`).
- Scratch scripts: `scratch/argus/fix/` (sweep.py, focus.py, switch.py, prototypes).
