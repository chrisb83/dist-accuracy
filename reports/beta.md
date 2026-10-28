# beta: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `beta_gen`, lines 708-967
- **Implemented:** pdf (757-762, Boost `ibeta_derivative` via `scu._beta_pdf`), logpdf (764-767), cdf (769-770, `sc.betainc`), sf (772-773, `sc.betaincc`), ppf (778-779, Boost `ibeta_inv` via `scu._beta_ppf`), isf (775-776, `sc.betainccinv`)
- **Inherited (not measured):** logcdf, logsf (rv_continuous default in 1.18.1: `log(cdf)` / `log1p(-sf)` split at the median, resp. `log(sf)` / `log1p(-cdf)`)
- **Related:** `betaprime` (calls `beta._cdf/_sf/_ppf/_isf`), `rdist` (calls `beta._logpdf/_cdf/_sf/_ppf` with a = b = c/2); helpers: none used by the measured methods (`_get_fixed_fit_value`, `_kurtosis`, `_lazyselect`, `_remove_optimizer_parameters`, `_skew` are fit/stats only); special functions: `sc.betainc`, `sc.betaincc`, `sc.betainccinv`, `sc.betaln`, `sc.xlog1py`, `sc.xlogy`, `scu._beta_pdf`, `scu._beta_ppf`
- **Reference:** `refs/beta_ref.py`, 120 digits (continued fraction for I_x(a, b), quantiles by inversion on the side where the small quantity lives); self-checks: PASS. Supplementary quantile run: `scratch/beta/beta_quantile_ref.py` (same references, dense q grid), self-checks PASS
- **Parameter sets:** (2.30985, 0.62688), (0.5, 5), (1, 3), (0.001, 0.001), (20, 50), (1000, 1000), (1e5, 3e5), (1e6, 2); quantile run: (0.5, 2), (0.5, 3), (2, 10), (5, 2), (2, 1e6), (20, 50)
- **Worst finding:** loss in ppf, isf and pdf

## Summary

cdf and sf (Boost `ibeta` through `sc.betainc`/`sc.betaincc`) are accurate
everywhere measured, including a = 1e6 and the far tails: the only flagged
points sit just above the conditioning for very large shapes. The quantile
functions are the weak point. `scu._beta_ppf` (Boost `ibeta_inv`) stops its
root finder after too many iterations for small q, and returns values off by
up to 1e119 ulp, values stuck near 1e-52, 0.5 or nan. This happens for
ordinary shapes such as (0.5, 2) with q = 1e-9 and (2, 10) with q < 1e-101.
`sc.betainccinv` (isf) returns nan for b = 2 and q <~ 1e-186, where the answer
is 1.0. pdf raises `OverflowError` instead of returning a value for
x <~ 1e-308 (all subnormal x, and up to 1.7e-305 for large a), even where the
true density is 1e155. logpdf loses up to 2.4e6 ulp for large shapes. It
subtracts `sc.betaln(a, b)`, which has a large absolute error, from terms of
size ~a.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf | small q; scattered bands, e.g. (0.5, 2) q in [1e-14, 1e-9], (0.5, 3) q in [1e-16, 1e-8], (2, 10) q <= 1e-101, (2, 1e6) q <= 1e-91, (5, 2) q <= 1e-151 | Boost `ibeta_inv` root finder exceeds its iteration cap and returns the last iterate or nan (pattern 10) | 0.5 for 2.8e-33; nan for 7.0e-41; 6.6e-53 for 1.4e-156 | loss |
| F2 | pdf | x <~ 1e-308 for every shape set measured except b < 1; up to x < 1.7e-305 for a = 1e3 | Boost `ibeta_derivative` raises an overflow error when `1/(x*(1-x))` overflows; SciPy turns it into a Python `OverflowError` (pattern 3) | exception where the true pdf is 1.23e155 | loss |
| F3 | isf | b = 2 (also a = 5, b = 2), q <~ 1e-186 | `sc.betainccinv` (Boost `ibeta_inv`, complementary) returns nan where the root is 1 - tiny, i.e. 1.0 as a double (pattern 10) | nan for 1.0 | loss |
| F4 | logpdf | large shapes: (1e5, 3e5) near the mean, (1e6, 2) near x = 1 | `xlogy + xlog1py - betaln` cancel terms of size ~a to an O(1) result (pattern 5); `sc.betaln` itself is up to 4e4 ulp off for a large, b small | 2.4e6 ulp | severe |
| F5 | isf | (20, 50), q <= 1e-295 (targets near the underflow limit) | Boost `ibeta_inv` iteration ends early when the target probability is near DBL_MIN (mechanism unclear in detail) | 8.5e6 ulp | severe |
| F6 | ppf, isf | a small, b huge in the bulk: (2, 1e6), ppf q in [0.64, 0.99999], isf q in [1e-6, 0.36] | Boost `ibeta_inv` converges only to ~2e-11 relative (mechanism unclear, inside Boost) | 9.82e4 ulp | moderate |
| F7 | ppf | (1000, 1000), q <= 2e-238; (0.001, 0.001), q = 0.43 | Boost `ibeta_inv` loses a few thousand ulp in the far tail of large symmetric shapes (mechanism unclear) | 2.82e3 ulp | moderate |
| F8 | pdf, cdf, sf | large shapes in the tails: (1e5, 3e5) at mean +- 10 sd, (1000, 1000) at mean - 10 sd | Boost `ibeta_power_terms` evaluates x**a (1-x)**b / B(a, b) with an error just above the conditioning (pattern 5/6) | 4.64e3 ulp (cond (arg) 3.7e3) | moderate |
| F9 | pdf | far left tail, x <= 1e-140 with a >= 1 | Boost forms x**a through exp(a log x); error ~ 0.5 * a * abs(log x) ulp (pattern 6) | 756 ulp | minor |

### F1: ppf returns garbage or nan for small q (Boost ibeta_inv does not converge)

- **Code:** `_ppf`, line 779: `scu._beta_ppf(q, a, b)` (Boost `ibeta_inv`).
- **Mechanism:** Boost's Halley iteration in `ibeta_inv` hits its iteration
  limit and SciPy returns the last iterate (and emits
  `RuntimeWarning: ... Root finding evaluation exceeded %1% iterations, giving up now`)
  or nan (pattern 10). The result is often a value that does not depend on q
  at all: 6.588487656280434e-53 for every q <= 1e-150 at (2, 1e6), and
  8.196e-53 at (2, 10). The failing q ranges are bands that depend on the
  shapes in an irregular way. For (0.5, 2) only q in [1e-14, 1e-9] fails,
  while q = 1e-15 and q = 1e-8 are correct. For (0.5, 3) the band is
  [1e-16, 1e-8]. For (2, b) with b >= 10 every q <~ 1e-100 fails, and for
  (5, 2) every q <~ 1e-151 returns nan. (1, b), (1.5, b), (10, b) and
  (0.5, 5) showed no failures in a scan of q in [1e-300, 1e-3]
  (`scratch/beta` scan, checked with `sc.betainc(ppf(q))`).
  `sc.betaincinv` returns correct values at (0.5, 2) and (0.5, 3) (e.g.
  4.444444444444448e-19 at q = 1e-9) but also returns nan at (5, 2, 1e-200)
  and (2, 1e6, 1e-300).
- **Regime:** q below ~1e-8 for many shape pairs with a in [0.5, 5] and
  b >= 2. Supplementary run: 598 loss, 20 severe points in the low tail.
- **Reproducer:** `python accuracy.py refs/beta_ref.py --at ppf 1e-9 0.5 2` (and the other rows)

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-16, 0.5, 3)` | 0.5 | 2.8444444444444444e-33 | 7.9e+47 | 2 | loss |
  | `ppf(1e-200, 5, 2)` | nan | 6.988271187715792e-41 | inf | 0.2 | loss |
  | `ppf(1e-300, 2, 1e6)` | 6.588487656280434e-53 | 1.4142128552668443e-156 | 2.1e+119 | 0.5 | loss |
  | `ppf(1e-120, 2, 10)` | 8.196050044267321e-53 | 1.3483997249264841e-61 | 2.7e+24 | 0.5 | loss |
  | `ppf(1e-9, 0.5, 2)` | 2.4678339236759816e-24 | 4.444444444444445e-19 | 4.5e+15 | 2 | severe |

  (The last row is a relative error of ~1.0 but the tool classifies it as
  severe; reported as given.)
- **Fix direction:** for small q use the leading-order inversion
  `x0 = (q*a*B(a,b))**(1/a)` (in log space) and refine with Newton on
  `log(betainc)`, or fall back to `sc.betaincinv` / a bracketed solver when
  Boost signals non-convergence. The issue should also go to Boost.

### F2: pdf raises OverflowError for tiny x

- **Code:** `_pdf`, line 762: `scu._beta_pdf(x, a, b)` inside
  `np.errstate(over='ignore')`. The errstate does not help because the error
  comes from Boost as a C++ exception.
- **Mechanism:** Boost's `ibeta_derivative` computes `1/(x*(1-x))` as a
  prefactor and raises an overflow error when this is inf (pattern 3). SciPy
  maps it to a Python `OverflowError`, so `beta.pdf` aborts for the whole
  array. The prefactor is later multiplied by `x**a`, so the true pdf is
  finite and normal for a < 1 (x**(a-1)/B is ~1e155 at a = 0.5,
  x = 1e-310) and for 1 <= a < ~2 (a = 1.5, x = 1e-308: 1.35e-153).
  The threshold grows with a: the last raising x is 5.5e-309 for a <= 0.5,
  9.7e-309 for a = 1, 2.2e-308 for a = 1.5, 5.1e-308 for a = 3, 1.7e-306
  for a = 20 and 1.7e-305 for a = 1e3. For large a the true value is 0
  (underflow), so raising is still wrong, but it is not an accuracy loss.
  No exception for b < 1 (e.g. (2.31, 0.627)), and none at the right end
  because `1 - x >= 1.1e-16`. `logpdf` is accurate at the same points
  (e.g. 357.10808460842316 vs 357.10808460842315).
- **Regime:** x below ~1e-308 (subnormal) for all a measured with b > 1;
  up to 1.7e-305 for large a.
- **Reproducer:** `accuracy.py` cannot measure these points: the exception
  aborts the run (the first two full runs crashed there), so
  `refs/beta_ref.py` excludes them from `x_points`. Measured with
  `python scratch/beta/pdf_overflow.py` against the same reference:

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-310, 0.5, 5)` | OverflowError | 1.2304687500000019e+155 | n/a (exception) | 0.5 | loss |
  | `pdf(5e-324, 0.5, 0.5)` | OverflowError | 1.4320487378917307e+161 | n/a (exception) | 0.5 | loss |
  | `pdf(1e-310, 0.001, 0.001)` | OverflowError | 2.4488981192367078e+306 | n/a (exception) | 1.0 | loss |
  | `pdf(1e-308, 1.5, 5)` | OverflowError | 1.3535156249999999e-153 | n/a (exception) | 0.5 | loss |

  (Severity assigned by the scale's definition "0, inf, nan ... where the
  true value is a normal double". The tool could not grade an exception.
  cond values are the analytic abs(a - 1) for x -> 0.)
- **Fix direction:** catch the error / use the `user_overflow_error` policy
  returning inf, and for x < tiny evaluate `exp(_logpdf)` (logpdf is accurate there).

### F3: isf returns nan instead of 1.0 for tiny q

- **Code:** `_isf`, line 776: `sc.betainccinv(a, b, x)`.
- **Mechanism:** for b = 2 the root of `betaincc(a, 2, x) = q` is
  `1 - x ~ sqrt(2q/(a(a+1)))`. For q <~ 1e-186 this is below 1e-90, so the
  correct double is 1.0, but Boost's complementary `ibeta_inv` returns nan
  (pattern 10). The same happens for (5, 2) with q <= 1e-186 (124 points in
  the supplementary run) and (1e6, 2) with q <= 1.3e-188. Between ~1e-180
  and ~1e-30 it correctly returns 1.0. No failure for b = 1, b = 0.5 or
  b >= 5 in the scan.
- **Regime:** b = 2 (any a tested: 1e3, 1e5, 1e6, 5), q <~ 1e-185.
- **Reproducer:** `python accuracy.py refs/beta_ref.py --at isf 1e-200 1e6 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1e-300, 1e6, 2)` | nan | 1.0 | inf | 0 | loss |
  | `isf(1e-200, 1e6, 2)` | nan | 1.0 | inf | 7.07e-101 | loss |
  | `isf(1e-200, 5, 2)` | nan | 1.0 | inf | 1.29e-101 | loss |

- **Fix direction:** return 1.0 when the leading-order estimate
  `(q*b*B(a,b))**(1/b) < eps/2`, and otherwise compute `1 - ppf_mirror`
  with Boost's `ibeta_inv(b, a, q)` (y = 1 - x) and return `1 - y`.

### F4: logpdf loses up to 2.4e6 ulp for large shapes

- **Code:** `_logpdf`, lines 765-766:
  `lPx = sc.xlog1py(b - 1.0, -x) + sc.xlogy(a - 1.0, x); lPx -= sc.betaln(a, b)`.
- **Mechanism:** two things add up.
  (1) Pattern 5: near the mode the three terms are of size
  `a*abs(log x) + b*abs(log(1-x))` (1.4e5 at (1e5, 3e5), x = 0.25) and
  cancel to an O(1) result (6.37). Even with a correctly rounded `betaln`
  this costs ~1e4 ulp: 842 ulp at x = 0.2486 and 9.9e3 ulp at x = 0.25,
  computed with an exact betaln in scratch.
  (2) `sc.betaln` itself has a large absolute error. At (1e5, 3e5) it is
  3.25e-10 absolute (6.6 ulp of a value of -2.25e5). At (1e6, 2) it is
  2.41e-10 absolute, which is 3.97e4 ulp of the value -27.63. cephes `lbeta`
  computes `lgam(a+b) - lgam(a) - lgam(b)` when it does not take its
  asymptotic branch (a > 1e6*b and a > 1e6), so the `lgam` terms of size
  1.3e7 cancel. Near x = 1 for (1e6, 2) the logpdf is ~0.45, so the
  absolute betaln error alone gives 2.4e6 ulp. With an exact betaln the
  same point is 32 ulp.
- **Regime:** a + b >~ 1e5 near the bulk (moderate: 15 points at
  (1e5, 3e5)); a >~ 1e3 with b of order 1 wherever abs(logpdf) is small
  (at (1e6, 2): 11 moderate and 1 severe point for x in
  [0.9993, 0.999999]). At (1000, 1000) the error is 984 ulp, flagged minor.
- **Reproducer:** `python accuracy.py refs/beta_ref.py --at logpdf 0.9999838579178736 1e6 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.9999838579178736, 1e6, 2)` | 0.45474494783051966 | 0.4547449475892312 | 2.4e+06 | 2.1e+06 | severe |
  | `logpdf(0.24863069531786686, 1e5, 3e5)` | 4.366439835168421 | 4.3664398348443205 | 3.3e+05 | 167 | moderate |
  | `logpdf(0.25, 1e5, 3e5)` | 6.367658694187412 | 6.367658693848445 | 2.4e+05 | 0.105 | moderate |

  (The severe row is only just above its conditioning, 2.1e6. With an
  exact betaln the error there is 32 ulp, so the error is real, not an
  effect of the conditioning.)
- **Fix direction:** for large shapes, compute logpdf with Stirling-corrected
  forms in which the large terms cancel analytically (as Boost's
  `ibeta_power_terms` does for the pdf, which is 1.7 ulp at (1e6, 1e6),
  x = 0.5), or use `log(_beta_pdf)` wherever the pdf is a normal double. Fix
  `sc.betaln` for a >> b by using `lgam(b) + log-gamma-ratio` asymptotics
  for a > ~1e3*b, not only for a > 1e6*b.

### F5: isf erratic for targets near the underflow limit

- **Code:** `_isf`, line 776: `sc.betainccinv(a, b, x)`.
- **Mechanism:** at (20, 50) the isf is correct (< 1 ulp) for
  q >= 1e-290. Below that the error jumps irregularly: 254 ulp at 1e-302,
  8.5e6 ulp at 1e-300, 4.65e4 ulp at 1e-298, 4.0e6 ulp at 1e-307. In
  terms of 1 - x (the small quantity, 4.6e-7) this is 0.4 % relative error.
  The non-monotone pattern points to the Boost iteration stopping on an
  absolute residual once the target probability is within a few decades of
  DBL_MIN. The `ppf` mirror `scu._beta_ppf(1e-300, 50, 20)` gives the same
  wrong 1 - x (4.660850756901678e-07 vs 4.6419165115068557e-07). The
  details are inside Boost; mechanism not pinned down further.
- **Regime:** q <~ 1e-295; observed at (20, 50); (3, 40) and (1e3, 1e3) at
  q = 1e-300 are fine or minor (553 ulp).
- **Reproducer:** `python accuracy.py refs/beta_ref.py --at isf 1e-300 20 50`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1e-300, 20, 50)` | 0.9999995339149244 | 0.9999995358083489 | 8.5e+06 | 9.28e-09 | severe |
  | `isf(1e-299, 20, 50)` | 0.9999995138293982 | 0.9999995139316663 | 4.6e+05 | 9.72e-09 | moderate |

- **Fix direction:** solve in log space (`log(betaincc) = log(q)`), or
  rescale the target, when q < ~1e-280.

### F6: ppf/isf only ~2e-11 accurate for a small, b huge

- **Code:** `_ppf` line 779 (`scu._beta_ppf`), `_isf` line 776 (`sc.betainccinv`).
- **Mechanism:** for (2, 1e6) both quantiles have a relative error of about
  2.2e-11 in the bulk (9.8e4 ulp), with small conditioning (0.5-1.7). Both
  entry points give the same wrong value, so the error is in the shared
  Boost `ibeta_inv` (most likely a loose termination tolerance of its
  Halley iteration in the large-b branch); mechanism unclear.
- **Regime:** (2, 1e6): ppf q in [0.64, 0.99999] (6 points), isf q in
  [1e-6, 0.36] (7 points). Not seen at (1e6, 2), (1e5, 3e5) or (1, 3).
- **Reproducer:** `python accuracy.py refs/beta_ref.py --at ppf 0.78 2 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.78, 2, 1e6)` | 2.866464025139682e-06 | 2.8664640250771984e-06 | 9.82e+04 | 1.67 | moderate |
  | `isf(0.22, 2, 1e6)` | 2.8664640251396815e-06 | 2.8664640250771984e-06 | 9.82e+04 | 0.471 | moderate |

- **Fix direction:** one Newton step on `betainc(a, b, x) - q`, using the
  accurate pdf, after the Boost call.

### F7: ppf loses ~3e3 ulp in the far tail of large symmetric shapes and for tiny shapes

- **Code:** `_ppf`, line 779.
- **Mechanism:** Boost `ibeta_inv`. At (1000, 1000) the error grows steadily
  as q decreases (18.7 ulp at q = 9e-139, 2.82e3 ulp at q = 2.4e-263) while
  cond (arg) is ~1e-3, i.e. the result should be nearly insensitive to q.
  At (0.001, 0.001), q = 0.43 the error is 1.42e3 ulp against a
  conditioning of 1e3, just above it. Mechanism unclear (inside Boost).
- **Regime:** (1000, 1000), q <~ 1e-230 (3 moderate, 9 minor points);
  (0.001, 0.001) near the median.
- **Reproducer:** `python accuracy.py refs/beta_ref.py --at ppf 2.3713737056616554e-263 1000 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(2.3713737056616554e-263, 1000, 1000)` | 0.16406676825030245 | 0.16406676825019972 | 2.82e+03 | 0.00124 | moderate |
  | `ppf(0.42999999999999994, 0.001, 0.001)` | 3.1458491990111627e-66 | 3.1458491990101685e-66 | 1.42e+03 | 1e+03 | moderate |

- **Fix direction:** a final Newton step in log space,
  `x *= exp(-(log F(x) - log q) * F(x) / (x f(x)))`.

### F8: pdf, cdf and sf just beyond conditioning for very large shapes

- **Code:** lines 762, 770, 773 (Boost `ibeta_derivative`, `ibeta`, `ibetac`).
- **Mechanism:** all three use Boost's `ibeta_power_terms` for the
  prefactor `x**a (1-x)**b / B(a, b)`. Its error grows like the
  exponent `a*log(x/x0) + b*log((1-x)/(1-x0))` (patterns 5/6). This is the
  same size as the condition number with respect to x, so the error stays
  just above the conditioning (4.6e3 ulp vs cond 3.7e3).
- **Regime:** (1e5, 3e5) at mean +- 10 sd; (1000, 1000) at mean - 10 sd. In
  the bulk of the same sets everything is within conditioning, and (1e6, 2)
  is fine.
- **Reproducer:** `python accuracy.py refs/beta_ref.py --at sf 0.25684652341066566 1e5 3e5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(0.25684652341066566, 1e5, 3e5)` | 1.3804596324611288e-23 | 1.3804596324597056e-23 | 4.64e+03 | 3.72e+03 | moderate |
  | `pdf(0.25684652341066566, 1e5, 3e5)` | 2.0003971639383982e-19 | 2.0003971639363507e-19 | 4.61e+03 | 3.69e+03 | moderate |
  | `cdf(0.24315347658933434, 1e5, 3e5)` | 4.086085837969341e-24 | 4.086085837965273e-24 | 4.48e+03 | 3.65e+03 | moderate |
  | `pdf(0.27644908299505205, 1000, 1000)` | 6.2158397912694175e-96 | 6.215839791271284e-96 | 1.35e+03 | 617 | moderate |

- **Fix direction:** none needed in SciPy. This is close to the inherent
  limit for these shapes (cond (shapes) is 5.5e3 as well).

### F9: pdf far left tail, error proportional to a*abs(log x)

- **Code:** `_pdf`, line 762 (Boost `ibeta_derivative` -> `ibeta_power_terms`).
- **Mechanism:** for extreme x Boost evaluates `x**a` as `exp(a*log(x))`. The
  rounding of the exponent (about a*abs(log x)*eps absolute) becomes a
  relative error of the same size (pattern 6). Even at a = 1, where the
  pdf is exactly b = 3, the result is off by 257 ulp at x = 1e-250
  (abs(log x) = 575).
- **Regime:** x <~ 1e-100 with a >= 1; 41 minor points in the low tail.
- **Reproducer:** `python accuracy.py refs/beta_ref.py --at pdf 1e-200 2.3098496451481823 0.6268795430096368`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-200, 2.30985, 0.62688)` | 1.205107710272275e-262 | 1.2051077102720727e-262 | 756 | 1.31 | minor |
  | `pdf(1e-250, 1, 3)` | 2.9999999999998286 | 3.0 | 257 | 0 | minor |

- **Fix direction:** in Boost, use `pow(x, a)` directly when it does not
  under/overflow. In SciPy, `exp(_logpdf)` is accurate for x << 1 when the
  shapes are moderate (logpdf: <= 10 ulp at (1, 3)).

## RuntimeWarnings

- ppf: `RuntimeWarning: Error in function boost::math::ibeta<double>(%1%, %1%, %1%): Root finding evaluation exceeded %1% iterations, giving up now.`
  in 4 of the 6 supplementary parameter sets. It always comes with a wrong
  result (F1), so the warning is a real signal. Some F1 failures (the nan
  returns at (5, 2)) give no warning.
- isf: the nan results of F3 and the wrong results of F5 give no warning.
- pdf: F2 is an exception (`OverflowError`), not a warning. No other warnings
  were recorded for pdf, logpdf, cdf or sf on the grids.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid incl. tails 1e-300 and 1 - 1e-16, all 8 sets | <= 4.5 ulp for small/moderate shapes; 33.5 ulp at (20, 50) (minor, cond 20); large shapes in F8 |
| cdf, sf | large shapes near the mean (analogue of the gamma F1 series-cap problem): (1e5, 3e5), (1e6, 2), and probe (1e6, 1e6) at x = 0.4999 | within conditioning (139 ulp at cond ~1.4e3 for (1e6, 1e6)); Boost's asymptotic branches have no iteration cap problem here |
| sf | full grid, all 8 sets | <= 8.8 ulp for small/moderate shapes; 25 ulp (20, 50); 695 ulp at (1000, 1000) within cond 4.7e3 |
| pdf | (0.5, 5), (0.001, 0.001) incl. x down to 2.2e-308 and pdf ~1e306; (1e6, 2) | <= 6.1 ulp; 121 ulp at (1e6, 2) within cond 1e6 |
| logpdf | small/moderate shapes, both tails (xlogy/xlog1py near x = 0 and 1) | <= 10.3 ulp at (0.5, 5), (1, 3); 264 ulp at (2.31, 0.627) within cond 264; 465 ulp at (0.001, 0.001) within cond 3.6e5; 170 ulp at (20, 50) (minor, cond 159) |
| ppf | q in [1e-300, 1 - 3e-16] for (2.31, 0.627), (0.5, 5), (1, 3), (1e5, 3e5), (1e6, 2) | <= 5.8 ulp; 65.6 ulp at (20, 50) for q -> 1 (cond up to 2.5e10, within) |
| isf | q in [1e-300, 1 - 3e-16] for (2.31, 0.627), (0.5, 5), (1, 3), (1e5, 3e5); (0.5, 2), (0.5, 3), (2, 10) in the dense grid | <= 4 ulp; 1.85e3 ulp at (0.001, 0.001) within cond 2.45e3 |

## Suspected, not confirmed

- `pdf` at x = 0 or 1 exactly (inf / 1/B / 0) is not on the harness
  grids; not probed further.
- The quantile references at (0.001, 0.001) for q >~ 0.3 are 1 - 1e-222 or
  closer to 1. They are 1.0 as doubles, and scipy agrees. Those points are
  measured, but `1 - x` itself cannot be resolved, so a hypothetical
  error in the complement would not show.
- F1 bands were scanned only for a in {0.5, 1, 1.5, 2, 3, 5, 10} and
  b in {2, 10, 100, 1e3, 1e6} (consistency check `betainc(ppf(q)) == q`,
  which cannot detect results whose cdf underflows); other shape pairs may
  have further bands.
- Shapes beyond 1e6 (e.g. a = b = 1e8) were only spot-checked: pdf
  (0.25, 1e8, 3e8) is 0.13 ulp, logpdf 5.8e8 ulp (same mechanism as F4,
  `sc.betaln(1e8, 3e8)` is 26 ulp / 1.3e-6 absolute off). The harness was
  not run there because the continued-fraction reference is slow near the
  mean.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | `log(cdf)` below the median, `log1p(-sf)` above | `a*log(x) + b*log1p(-x) - betaln(a, b) - log(a) + log(CF)`, where CF is the continued fraction / hypergeometric factor `2F1(a+b, 1; a+1; x)` (log of Boost's `ibeta` prefix plus the series) | cdf underflows in the left tail: `logcdf(1e-200, 3, 2)` returns `-inf`, true -1380.1647614353075 (loss) |
| logsf | `log(sf)` above the median, `log1p(-cdf)` below | the mirror formula with `(b, a, 1 - x)` (1 - x is exact for x >= 0.5) | sf underflows in the right tail: `logsf(1 - 1e-12, 3, 40)` returns `-inf`, true -1098.4836350113086; `logsf(0.9, 1000, 1000)` returns `-inf`, true -1026.1478995158183 (loss) |

## Shared code

- `scu._beta_ppf` / `sc.betainccinv` / `sc.betaincinv` (Boost `ibeta_inv`):
  F1, F3, F5, F6 and F7 are defects of the inverse incomplete beta function,
  not of `beta_gen`, which only forwards its arguments. Affected:
  `betaprime._ppf`/`_isf` (call `beta._ppf` / `beta._isf`), `rdist._ppf`
  (`beta._ppf(q, c/2, c/2)`), `jf_skew_t._ppf` (`beta.ppf`), and any direct user of
  `sc.betaincinv`/`sc.betainccinv`. `sc.betaincinv` avoids the (0.5, 2) and
  (0.5, 3) bands of F1 but has the same nan at (5, 2, 1e-200) and
  (2, 1e6, 1e-300).
- `scu._beta_pdf` (Boost `ibeta_derivative`): F2 (OverflowError for tiny x)
  and F9; any distribution calling `beta._pdf` is affected.
- `sc.betaln` (cephes `lbeta`): large absolute error when a >> b but
  a < 1e6*b. Relative error 3.97e4 ulp at (1e6, 2), 6.87e5 ulp at
  (1e6, 0.5), 5.7e3 ulp at (1e5, 2), 300 ulp at (1e3, 2); fine once
  a > 1e6*b (asymptotic branch: 0.05 ulp at (1e7, 2)). It is used in the
  logpdf of `beta` (F4), `betaprime` (line 1038), `f` (lines 2600, 2650),
  `rdist` (via `beta._logpdf`) and in `beta._entropy`.

## Existing GitHub issues

(web search; `gh` was not available)

- #12796 (closed, fixed by PR #17697 for 1.12.0): "BUG: nonmonotonicity in
  betaincinv, btdtri, stats.beta.ppf" - an earlier failure of the same
  inverse for small a and b ~ 1e5 near q -> 1; relates to F1/F6 (same
  function, different regime; the current failures are new)
- boostorg/math #1511 (closed upstream in Boost PR #1512, Oct 2026):
  "`special.betaincinv` / `stats.beta.ppf` wrong when `min(a, b) == 1000`
  exactly" - a bug in Boost `ibeta_inv`'s initial guess for a = 1000 and
  b >~ 2e7, reproduced with SciPy 1.17.1/1.18.1; same routine as F1, F3, F5,
  F6, F7. F7's (1000, 1000) case has `min(a, b) == 1000` exactly, so it may
  be related; not checked whether the Boost fix changes it
- F2 (pdf `OverflowError`), F4 (`betaln` for a >> b), F8, F9: none found.
  Excluded as unrelated: #14722 and #16517 (overflow RuntimeWarnings in
  `_beta_ppf` on Apple M1, duplicates of #14901), #2385 (pdf nan for large
  shapes, fixed in 0.13.0), #10713 (pdf inf at x = 1)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(1e-16, 0.5, 3)` | 0.5 | 2.8444444444444443e-33 | verified | 7.9e47 ulp; ref by log-space root of the 2F1 form of I_x, cdf(ref) = q checked with `mp.betainc`; Boost iteration-limit RuntimeWarning emitted |
| F1 | `ppf(1e-200, 5, 2)` | nan | 6.9882711877157924e-41 | verified | no warning, as the report says |
| F1 | `ppf(1e-300, 2, 1e6)` | 6.588487656280434e-53 | 1.4142128552668442e-156 | verified | 2.1e119 ulp; warning emitted |
| F1 | `ppf(1e-120, 2, 10)` | 8.196050044267321e-53 | 1.3483997249264842e-61 | verified | 2.7e24 ulp |
| F1 | `ppf(1e-9, 0.5, 2)` | 2.4678339236759816e-24 | 4.444444444444445e-19 | verified | 4.5e15 ulp; `sc.betaincinv` gives 4.444444444444448e-19, as stated |
| F2 | `pdf(1e-310, 0.5, 5)` | OverflowError | 1.2304687500000019e+155 | verified | raw `scu._beta_pdf` raises "ibeta_derivative ... Overflow Error"; the overflow check is inside Boost, and its exact condition was not checked against the Boost source |
| F2 | `pdf(5e-324, 0.5, 0.5)` | OverflowError | 1.4320487378917307e+161 | verified | |
| F2 | `pdf(1e-310, 0.001, 0.001)` | OverflowError | 2.4488981192367093e+306 | verified | report's reference 2.4488981192367078e+306 is ~5 ulp off (mine checked with `mp.beta` at 60 digits); does not affect the finding |
| F2 | `pdf(1e-308, 1.5, 5)` | OverflowError | 1.3535156249999999e-153 | verified | |
| F3 | `isf(1e-300, 1e6, 2)` | nan | 1.0 (1 - x = 1.414e-156) | verified | `sc.betainccinv` returns nan directly |
| F3 | `isf(1e-200, 1e6, 2)` | nan | 1.0 (1 - x = 1.414e-106) | verified | |
| F3 | `isf(1e-200, 5, 2)` | nan | 1.0 (1 - x = 2.582e-101) | verified | |
| F4 | `logpdf(0.9999838579178736, 1e6, 2)` | 0.45474494783051966 | 0.45474494758923115 | verified | 2.39e6 ulp; `sc.betaln(1e6, 2)` abs error 2.41e-10 confirmed |
| F4 | `logpdf(0.24863069531786686, 1e5, 3e5)` | 4.366439835168421 | 4.3664398348443207 | verified | 3.34e5 ulp; betaln abs error 3.25e-10 confirmed |
| F4 | `logpdf(0.25, 1e5, 3e5)` | 6.367658694187412 | 6.3676586938484448 | verified | 2.4e5 ulp |
| F5 | `isf(1e-300, 20, 50)` | 0.9999995339149244 | 0.99999953580834885 | verified | 8.53e6 ulp; 1 - x = 4.6419165115068557e-7; `scu._beta_ppf(1e-300, 50, 20)` mirror gives the same wrong value |
| F5 | `isf(1e-299, 20, 50)` | 0.9999995138293982 | 0.99999951393166635 | verified | 4.61e5 ulp |
| F6 | `ppf(0.78, 2, 1e6)` | 2.866464025139682e-06 | 2.8664640250771985e-06 | verified | 9.82e4 ulp; ref from closed form sf = (1-x)^b (1 + b x) for a = 2, checked by `mp.quad` of the pdf |
| F6 | `isf(0.22, 2, 1e6)` | 2.8664640251396815e-06 | 2.8664640250771983e-06 | verified | 9.82e4 ulp |
| F8 | `sf(0.25684652341066566, 1e5, 3e5)` | 1.3804596324611288e-23 | 1.3804596324597055e-23 | verified | 4.64e3 ulp; ref by `mp.quad` of the pdf over the tail |
| F8 | `pdf(0.25684652341066566, 1e5, 3e5)` | 2.0003971639383982e-19 | 2.0003971639363507e-19 | verified | 4.61e3 ulp |
| F8 | `cdf(0.24315347658933434, 1e5, 3e5)` | 4.086085837969341e-24 | 4.0860858379652728e-24 | verified | 4.48e3 ulp |
| F8 | `pdf(0.27644908299505205, 1000, 1000)` | 6.2158397912694175e-96 | 6.2158397912712837e-96 | verified | 1.35e3 ulp |

All references were computed at 100 and 200 digits and agree to better than 1e-94 relative. The quantile references were obtained by solving log I = log q in log x, using the hypergeometric form of I_x or the a = 2 closed form, and checked with `mp.betainc` or `mp.quad`. Scripts: `scratch/beta/verify/verify.py`, `verify2.py`, `mech.py`. The cited code lines (762, 765-766, 776, 779) match SciPy 1.18.1. The Boost-internal mechanisms (F1, F2, F3, F5, F6, F8) agree with the observed behaviour but were not traced in the Boost source.

Verified by dist-accuracy-verifier: 7 of 7 verified.
