# geninvgauss: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `geninvgauss_gen`, lines 5180-5493; logpdf kernel in the compiled Cython module `scipy.stats._stats` (`geninvgauss_logpdf`, `_geninvgauss_pdf`; .pyx not shipped)
- **Implemented:** pdf, logpdf, cdf
- **Inherited (not measured):** logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), ppf = numerical inversion of cdf, isf = ppf(1 - q)
- **Related:** invgauss is the special case p = -1/2 (separate implementation, no shared code); helpers: `_check_shape` (rvs only), `_stats.geninvgauss_logpdf`, `_stats._geninvgauss_pdf` (LowLevelCallable for quad); special functions: `sc.kve` (logpdf), `integrate.quad` (cdf); `sc.kve` also in `_munp`
- **Reference:** `refs/geninvgauss_ref.py`, 100 digits (cdf by mp.quad in t = log x, so 400 digits was too slow); self-checks: PASS
- **Parameter sets:** (2.3, 1.5), (-0.5, 1), (0, 0.001), (3, 1e-06), (0.5, 10000), (-1.5, 1e+06), (100, 0.01), (1000, 1000), (0.5, 2e+09); cdf grid run on (2.3, 1.5), (-0.5, 1), (0, 0.001), (0.5, 10000) only, other sets with `--at`
- **Worst finding:** loss in pdf, logpdf, cdf

## Summary

logpdf and pdf are accurate (a few ulp) for moderate shapes, but return nan
whenever `kve(p, b)` overflows (large |p| relative to b, e.g. p = 100,
b = 0.01) and whenever b >= 2**30 - 0.5, where `scipy.special.kve` itself
returns nan. For large b they lose up to ~b*eps absolute in logpdf near x = 1
(5e5 ulp at b = 1e6) because `b` and `b*(x + 1/x)/2` cancel; large p and b
cost ~1e3 ulp. The cdf is a plain `integrate.quad(pdf, 0, x)` with default
tolerances: it returns 0.0 instead of 1.0 for x far beyond the bulk (already
at x = 1e5 for the example parameters), has relative errors up to ~1 for small
cdf values (absolute tolerance 1.5e-8), and is only good to ~1e-11 (up to
4.8e5 ulp) in the bulk.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | x far right of the bulk (x >~ 1e5 for (2.3, 1.5)), all shapes | quad on [0, x] never samples the peak | 0.0 for 1.0 | loss |
| F2 | logpdf, pdf, cdf | log K_p(b) > 709.8 (large \|p\|, small b), any x | `kve(p, b)` overflows, replaced by nan (pattern 3) | nan | loss |
| F3 | logpdf, pdf, cdf | b >= 2**30 - 0.5 = 1073741823.5, any p, x | `scipy.special.kve` returns nan (pattern 10, shared) | nan | loss |
| F4 | cdf | left tail, cdf << 1.5e-8 | quad absolute tolerance epsabs = 1.49e-8 (pattern 10) | 1.39e-27 for 4.65e-24 | severe |
| F5 | cdf | bulk, all shapes | quad relative tolerance epsrel = 1.49e-8 | 4.8e5 ulp | moderate |
| F6 | logpdf, pdf | large b, x near 1 | `b` and `b*(x + 1/x)/2` cancel (pattern 5) | 7.5e5 ulp (pdf) | moderate |
| F7 | logpdf, pdf | large p and b, near the mode | terms of size ~p, b cancel (pattern 5) | 1.57e3 ulp (pdf) | moderate |

### F1: cdf returns 0 for x far beyond the bulk

- **Code:** `_cdf`, line 5253: `integrate.quad(llc, _a, x)[0]` with `_a = 0`.
- **Mechanism:** QUADPACK (qags) starts with a 21-point Gauss-Kronrod rule on
  [0, x]. When x is many orders of magnitude larger than the region where the
  pdf is non-negligible, every node falls where the pdf underflows to 0, the
  error estimate is 0 and quad returns 0 (or a tiny partial value) although
  the true cdf is 1. Not one of the listed patterns: an integration interval
  that is not adapted to the shape of the integrand.
- **Regime:** right tail of every parameter set. For (2.3, 1.5): correct at
  x = 1e4, 3.5e-31 at x = 1e5, 0.0 from 1e6 on. For (0.5, 1e4): 0.0 at
  x = 1e10. In the grid run 125 of 154 right-tail points are losses, across
  all four parameter sets.
- **Reproducer:** `python accuracy.py refs/geninvgauss_ref.py --at cdf 1e6 2.3 1.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e6, 2.3, 1.5)` | 0.0 | 1.0 | inf | 0 | loss |
  | `cdf(1e5, 2.3, 1.5)` | 3.5133086665398156e-31 | 1.0 | 4.5e+15 | 0 | loss |
  | `cdf(1e10, 0.5, 1e4)` | 0.0 | 1.0 | inf | 0 | loss |
  | `cdf(10000000000.0, 2.3, 1.5)` | 0.0 | 1.0 | inf | 0 | loss |

- **Fix direction:** split the integral at the mode (`self._mode`) and
  integrate only the smaller tail (`cdf = 1 - integral(x, inf)` right of the
  mode), preferably in t = log x where the integrand
  `exp(p t - b cosh t)` is well shaped; or use `points=` with the mode.

### F2: kve(p, b) overflow gives nan for large |p|

- **Code:** `_stats.geninvgauss_logpdf` (called at line 5230), confirmed
  bit-for-bit equal to
  `-log(2) - log(kve(p, b)) + b + (p - 1)*log(x) - b*(x + 1/x)/2`, with nan
  returned when `kve(p, b)` is inf; the warning is issued at line 5236.
  pdf = exp(logpdf) (line 5243) and cdf (quad of the same kernel) inherit it.
- **Mechanism:** pattern 3. `kve(p, b) = K_p(b) e^b ~ Gamma(|p|)/2 (2/b)^|p|`
  overflows although only `log K_p(b)` is needed and the density is finite and
  normal.
- **Regime:** log(K_p(b) e^b) > 709.8, roughly
  `lgamma(|p|) + |p| log(2/b) > 710`: e.g. |p| = 100, b = 0.01; |p| >= 170,
  b = 1; any |p| around 1e6. Both signs of p (K_{-p} = K_p). The whole
  distribution (every x) is lost.
- **Reproducer:** `python accuracy.py refs/geninvgauss_ref.py --at pdf 19800.00005050505 100 0.01`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(19800.00005050505, 100, 0.01)` | nan | 0.00020030735965664704 | inf | 7.72e-15 | loss |
  | `logpdf(19800.00005050505, 100, 0.01)` | nan | -8.515657572799032 | inf | 9.06e-16 | loss |
  | `logpdf(50, -100, 0.01)` | nan | -1284.3303653200965 | inf | 0.0788 | loss |
  | `logpdf(1e-300, 100, 0.01)` | nan | -5e+297 | inf | 1 | loss |
  | `cdf(19800, 100, 0.01)` | nan | 0.4733043202829278 | inf | 8.38 | loss |

- **Fix direction:** compute `log K_p(b)` directly for large |p|/b (uniform
  asymptotic expansion in the order, or `log(kve)` with scaling of the
  recurrence); a `log_kv` helper would also serve `_munp` (line 5481).

### F3: scipy.special.kve returns nan for b >= 2**30 - 0.5

- **Code:** as F2; the nan comes from `sc.kve(p, b)` itself.
- **Mechanism:** pattern 10 (special function outside its working range).
  AMOS `zbesk` refuses arguments with |z| > ~2**30 (integer-range limit,
  ierr = 4) and SciPy maps that to nan, although
  `kve(p, b) ~ sqrt(pi/(2b))` is trivial to evaluate there. Bisection gives
  the threshold b = 1073741823.5 (`kve(0.5, 1073741823.5)` = 3.96e-5 is
  returned, the next double gives nan; same for p = 3). geninvgauss only checks
  `isinf`, so the nan passes through; the warning text then wrongly says
  "Infinite values encountered".
- **Regime:** b >= 1073741823.5, any p, any x (pdf, logpdf, cdf all nan).
- **Reproducer:** `python accuracy.py refs/geninvgauss_ref.py --at logpdf 1.0 0.5 2e9`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.0, 0.5, 2e9)` | nan | 9.789267975548505 | inf | 0.0511 | loss |
  | `pdf(1.0, 0.5, 2e9)` | nan | 17841.241161527712 | inf | 0.5 | loss |
  | `logpdf(1.0, -0.5, 1.2e9)` | nan | 9.53385516366551 | inf | 0.157 | loss |
  | `cdf(1.0, 0.5, 2e9)` | nan | 0.49999553968971017 | inf | 3.57e+04 | loss |

- **Fix direction:** for large b use the asymptotic expansion of
  `K_p(b) e^b sqrt(2b/pi)` (Hankel series) in geninvgauss, and report the
  range limit of `kve` to scipy.special.

### F4: small cdf values have no relative accuracy

- **Code:** `_cdf`, line 5253: `integrate.quad(llc, _a, x)[0]` (default
  `epsabs = 1.49e-8`, `epsrel = 1.49e-8`).
- **Mechanism:** pattern 10. QUADPACK stops when the error estimate is below
  `max(epsabs, epsrel*|I|)`; for a tiny integral the absolute tolerance
  1.49e-8 is met by the first crude estimate, so the relative error of a cdf
  value far below 1.5e-8 is uncontrolled. For a sharp peak (large b) the left
  tail value can be wrong by orders of magnitude.
- **Regime:** left tail where cdf << 1.5e-8, all shapes; worst for large b.
  The grid run found 4 severe and 1 moderate point in the left tail
  (max 4.0e13 ulp).
- **Reproducer:** `python accuracy.py refs/geninvgauss_ref.py --at cdf 0.99 -1.5 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(0.99, -1.5, 1e6)` | 1.3889192775665914e-27 | 4.6458987285926996e-24 | 4.5e+15 | 1.01e+04 | severe |
  | `cdf(0.0026758953019778327, -0.5, 1)` | 7.922434653945866e-83 | 7.92242352363866e-83 | 6.3e+09 | 187 | severe |
  | `cdf(0.005243131982553904, 2.3, 1.5)` | 1.8403127136498982e-70 | 1.840312656511383e-70 | 1.4e+08 | 146 | severe |

- **Fix direction:** `epsabs=0` and `epsrel` near machine precision, or a
  series/asymptotic expansion for the tail (the left tail of GIG(p, b) is the
  right tail of GIG(-p, b) at 1/x).

### F5: cdf is only good to ~1e-11 in the bulk

- **Code:** `_cdf`, line 5253, default quad tolerances (`epsrel = 1.49e-8`).
- **Mechanism:** quad is asked for 1.5e-8 relative accuracy only; it usually
  achieves ~1e-12 but nothing more is guaranteed.
- **Regime:** bulk of every parameter set: 67 moderate and 48 minor of 132
  bulk points; max 4.8e5 ulp, median 1.22e3 ulp.
- **Reproducer:** `python accuracy.py refs/geninvgauss_ref.py --at cdf 3.2849433241279202 2.3 1.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(3.2849433241279202, 2.3, 1.5)` | 0.5463437912552782 | 0.5463437912509308 | 3.58e+04 | 1.21 | moderate |
  | `cdf(0.6758413057007908, -0.5, 1)` | 0.5000000000000006 | 0.5000000000036882 | 3.32e+04 | 0.898 | moderate |
  | `cdf(0.99, 0.5, 1e4)` | 0.15623560730249983 | 0.1562356073024124 | 2.52e+03 | 153 | moderate |

- **Fix direction:** tighter tolerances (`epsabs=0, epsrel=1e-14`) together
  with the F1 interval splitting.

### F6: cancellation of b and b*(x + 1/x)/2 for large b

- **Code:** `_stats.geninvgauss_logpdf` (line 5230):
  `c + (p - 1)*log(x) - b*(x + 1/x)/2` with `c = -log(2) - log(kve(p, b)) + b`.
- **Mechanism:** pattern 5. The `+b` from the exponential scaling of `kve`
  and `-b*(x + 1/x)/2` are both ~b while their sum is
  `-b*(x - 1)**2/(2x)`, of order 1 near the peak. The absolute error is about
  b*eps, i.e. ~1e6/|logpdf| ulp in logpdf; pdf = exp(logpdf) turns the
  absolute error into the same relative error (line 5243).
- **Regime:** large b, x within a few multiples of 1/sqrt(b) of 1 (the
  bulk). About b*eps/|logpdf| ulp: ~1e4 at b = 1e4, ~5e5 at b = 1e6.
- **Reproducer:** `python accuracy.py refs/geninvgauss_ref.py --at logpdf 0.9970020029952514 -1.5 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.9970020029952514, -1.5, 1e6)` | 1.488815484684892 | 1.488815484517634 | 5.1e+05 | 2.02e+03 | moderate |
  | `pdf(0.9970020029952514, -1.5, 1e6)` | 4.431842822970418 | 4.4318428222291555 | 7.5e+05 | 3e+03 | moderate |
  | `logpdf(0.99999999, -1.5, 1e6)` | 5.988815770833753 | 5.988815770727965 | 7.96e+04 | 0.416 | moderate |
  | `pdf(0.99999999, -1.5, 1e6)` | 398.94189145535483 | 398.9418914131514 | 4.8e+05 | 2.49 | moderate |
  | `logpdf(1.030403012495567, 0.5, 10000)` | -0.8140911072387098 | -0.8140911072399876 | 7.07e+03 | 369 | moderate |
  | `pdf(0.9899505012374998, 0.5, 10000)` | 24.075471990722576 | 24.07547199064141 | 1.52e+04 | 101 | moderate |

- **Fix direction:** combine analytically: `-b*(x - 1)**2/(2*x)` instead of
  `b - b*(x + 1/x)/2`.

### F7: cancellation of large terms for large p and b

- **Code:** as F6.
- **Mechanism:** pattern 5. For p = b = 1000 near the mode (x ~ 2.4) the
  terms `(p - 1)*log(x)` (~900), `b*(x + 1/x)/2` (~1450), `b` and
  `log(kve(p, b))` (~450) are each of order 1e3 and cancel to a logpdf of
  order 1; each carries ~1/2 ulp of its own size.
- **Regime:** large p and b (both ~1e3 and above), bulk; error ~ (|p| + b)/|logpdf| ulp.
- **Reproducer:** `python accuracy.py refs/geninvgauss_ref.py --at logpdf 2.5 1000 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(2.5, 1000, 1000)` | 0.9226571520748621 | 0.9226571520746513 | 1.03e+03 | 55.3 | moderate |
  | `pdf(2.612847811156222, 1000, 1000)` | 0.06345200175470661 | 0.06345200175468448 | 1.57e+03 | 116 | moderate |
  | `pdf(2.4775193139094682, 1000, 1000)` | 3.759591686548061 | 3.7595916865469863 | 1.29e+03 | 37.9 | moderate |

  (cond (shapes) is 76.6 and 163 here: the error is ~10x larger than the
  sensitivity to the shapes.)
- **Fix direction:** the F6 rewrite removes the `b` part; the remaining
  `(p - 1)*log(x) - log K_p(b)` cancellation needs a log-K expansion around
  the mode (hard; low priority).

## RuntimeWarnings

- "Infinite values encountered in scipy.special.kve(p, b). Values replaced by
  NaN ..." for F2 (correct) and F3, where the message is wrong: `kve` returned
  nan, not inf.
- "underflow encountered in logpdf_single (vectorized)" and "underflow
  encountered in exp" in pdf/logpdf tails, and "underflow encountered in
  _cdf_single (vectorized)" in nearly every cdf call, including accurate ones
  (e.g. `cdf(30, 0, 0.001)`, 2 ulp; `logpdf(2.5, 1000, 1000)`); these are
  spurious where the result is accurate (underflow of the integrand / of
  intermediate terms inside the Cython kernel).
- `IntegrationWarning: roundoff error is detected` from quad in some F1/F4 cases;
  "invalid value encountered in _cdf_single" for F2/F3 (nan integrand).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | full grid, (2.3, 1.5), (-0.5, 1), (0, 0.001), (3, 1e-06), including x = 1e-300 and 1e300 | 24.8 ulp, all within conditioning |
| pdf | full grid, same sets | 185 ulp (high tail, cond 685), within conditioning |
| logpdf, pdf | tails (x -> 0, x -> inf) for (0.5, 1e4), (-1.5, 1e6), (1000, 1000) | the few moderate tail points are the F6/F7 mechanism near the peak edges; deep tails a few ulp |
| cdf | x near the bulk for tiny b, e.g. (0, 0.001) at x = 30 | 2 ulp |

## Suspected, not confirmed

- Accuracy of `kve` for very large orders (|p| ~ 1e4 with b ~ 1e4) could not
  be checked: `mp.besselk(1e4, 1e4)` failed to converge in mpmath, so no
  parameter set with |p| > 1e3 and finite `kve` was measured.
- cdf for the parameter sets (3, 1e-06), (-1.5, 1e6), (1000, 1000) was not
  run on the grid (each cdf condition number needs several 200-digit
  quadratures); single points with `--at` show the same F1/F4 behaviour
  (e.g. `cdf(0.99, -1.5, 1e6)`).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | integrate the pdf from x to inf, or use the reflection 1/X ~ GIG(-p, b): `sf(x, p, b) = cdf(1/x, -p, b)` | right tail: `sf(60, 2.3, 1.5)` = -8.659739592076221e-15 vs 5.018792951908354e-18 (wrong sign, loss); also all F1 points give sf = 1 |
| logsf | log(sf) | log of the direct tail integral, `logcdf(1/x, -p, b)` | wherever sf <= 0 or loses digits (right tail) |
| logcdf | log(cdf) | log of the direct left-tail integral (equivalently `logsf(1/x, -p, b)`), with an asymptotic tail expansion where it underflows | left tail (F4 errors carry over) and -inf where cdf underflows |
| ppf | numerical inversion of cdf | invert the improved cdf/sf, starting from the mode | fails outright where cdf is broken: `ppf` with q in [1e-12, 0.999] for (0.5, 1e4) raised `RuntimeError: Failed to converge after 100 iterations` (brentq), `ppf(0.5, 0.5, 1e4)` raised `ValueError` (nan) |
| isf | ppf(1 - q) | `isf(q, p, b) = 1/ppf(q, -p, b)` via the reflection | q < eps (1 - q rounds to 1) |

For p = -1/2 (and p = 1/2 by reflection) the cdf has the invgauss closed
form in terms of `ndtr`, which could be used for cdf/sf/logcdf/logsf there.

## Shared code

- `scipy.special.kve` (and `kv`) return nan for real argument
  b >= 1073741823.5 (AMOS `zbesk` ierr = 4), although the value is an easy
  normal double (`kve(0.5, 2e9)` should be 2.80e-5). Affects every user of
  `kv`/`kve` with large arguments (in scipy.stats also `geninvgauss._munp`,
  and any distribution normalised by a K Bessel function).
- `_stats.geninvgauss_logpdf` / `_geninvgauss_pdf` (Cython) are used only by
  geninvgauss (logpdf, pdf, cdf integrand); F2, F6, F7 live there.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e6, 2.3, 1.5)` | 0.0 | 1.0 | verified (loss) | ref = 1 - quad(pdf, x, inf) in x; also `cdf(1e5, 2.3, 1.5)` = 3.51e-31 and `cdf(1e10, 0.5, 1e4)` = 0.0 vs 1.0 |
| F2 | `pdf(19800.00005050505, 100, 0.01)` | nan | 0.00020030735965664705 | verified (loss) | K_p(b) from the integral representation int exp(-b cosh t) cosh(pt) dt (checked against an upward recurrence from K_0, K_1); `logpdf(50, -100, 0.01)` nan vs -1284.3303653200966; `cdf(19800, 100, 0.01)` nan vs 0.47330432028292778; `sc.kve(100, 0.01)` = inf confirmed |
| F3 | `logpdf(1.0, 0.5, 2e9)` | nan | 9.7892679755485055 | verified (loss) | closed form K_{1/2}; cdf by the reflection to the invgauss closed form gives 0.49999553968971018. The threshold is confirmed (`kve(0.5, 1073741823.5)` finite, next double nan), but the value there is 3.82e-5, not 3.96e-5 as stated. The ierr = 4 cause was not checked |
| F4 | `cdf(0.99, -1.5, 1e6)` | 1.3889192775665914e-27 | 4.6458987285926995e-24 | verified (severe) | x-space quad split at the mode, with K_{3/2} in closed form; `cdf(0.0026758953019778327, -0.5, 1)` 6.33e9 ulp (ref from the invgauss ndtr closed form), `cdf(0.005243131982553904, 2.3, 1.5)` 1.4e8 ulp |
| F5 | `cdf(3.2849433241279202, 2.3, 1.5)` | 0.5463437912552782 | 0.54634379125093085 | verified (moderate) | `cdf(0.6758413057007908, -0.5, 1)` 3.32e4 ulp against the invgauss closed form |
| F6 | `logpdf(0.9970020029952514, -1.5, 1e6)` | 1.488815484684892 | 1.4888154845176339 | verified (moderate) | closed form K_{3/2}; pdf 7.53e5 ulp, `pdf(0.99999999, -1.5, 1e6)` 4.76e5 ulp; `_stats.geninvgauss_logpdf` bit-for-bit equals the stated formula. The line citations are slightly off: the kernel is called at 5234 (not 5230) and the warning is at 5238 (not 5236) |

All references were computed at 100 and 200 digits and agreed to 30 digits (scripts in `scratch/geninvgauss/verify/`).

Verified by dist-accuracy-verifier: 6 of 6 verified.
