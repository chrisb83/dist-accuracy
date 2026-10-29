# genextreme: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `genextreme_gen`, lines 3355-3525
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, ppf, isf (all via the helper `_loglogcdf`, lines 3406-3411)
- **Inherited (not measured):** logsf = log(sf)
- **Related:** none (gumbel_r is the c = 0 case but is a separate class); helpers: `_loglogcdf` (class-internal), `_skew` (fit only); special functions: sc.log1p, sc.expm1 (sc.gamma, sc.gammaln, sc.comb only in `_stats`/`_munp`, not reviewed)
- **Reference:** `refs/genextreme_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (-0.1), (0), (0.5), (2), (1.00000001), (1e-12), (-1e-12), (1000), (-1000); plus single points at c = 0.7, 0.99999999, 1.000000000001
- **Worst finding:** loss in logpdf, pdf, cdf, logcdf, sf, ppf, isf

## Summary

The log1p/expm1-based implementation is accurate in the bulk and in both
tails for moderate shapes, including c near 0 (c = +-1e-12 within
conditioning). Complete failures occur in four places: `c*x` overflows for
|c*x| > 1.8e308 (all x-methods return -inf, 0 or 1 where the true values are
ordinary numbers for |c| >~ 100, and logpdf/logcdf also for c = 2);
`expm1(-c*y)` overflows in ppf/isf in a window where `expm1(...)/c` is still
finite (|c| > 1); `c*x` rounding to exactly 1 just inside the upper endpoint
gives -inf/0; and ppf/isf return 0 at the q nearest to exp(-1) /
1 - exp(-1). For c near 1, logpdf cancels near the upper endpoint (severe,
up to 1e9+ ulp). Wherever an exponent of size L is computed and then
exponentiated, results lose up to ~L ulp (minor, <= 487 ulp).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf, logcdf, cdf, sf (pdf where normal) | \|c*x\| > 1.8e308 on the unbounded side; values normal for \|c\| >~ 100 (logpdf/logcdf also for c = 2) | `c*x` overflows to -inf (pattern 3) | -inf, 0.0, 1.0 for O(1) values | loss |
| F2 | ppf, isf | \|c\| > 1, -c*y in (709.78, 709.78 + log\|c\|), y = -log(-log q) | `expm1(-c*y)` overflows before `/c` (pattern 3, 8) | +-inf for +-1.7e308 | loss |
| F3 | logpdf, pdf, sf (logcdf) | c > 0, x the double just below 1/c where fl(c*x) == 1 but c*x < 1 exactly | `cx == 1` -> fill -inf, `log1p(-1)` (pattern 7) | -inf / 0.0 for -16.2 / 3.9e-24 | loss |
| F4 | ppf, isf | q = fl(exp(-1)) (ppf), q = fl(1 - exp(-1)) (isf), any c | `-log(q)` rounds to 1.0, `log(1.0) = 0` (pattern 2) | 0.0 for 3.4e-17 | loss |
| F5 | logpdf | c near 1 (\|1 - c\| <= ~1e-8), 1 - c*x small (x near the upper endpoint 1/c) | `lpex2 - lex2` = L/c - L cancels (pattern 5) | 5.8e10 ulp | severe |
| F6 | pdf, logpdf, logcdf, sf, isf | results computed as exp(E) or with E = log-size term of magnitude 100...700 | rounding of E amplified by \|E\| (pattern 6-like, exp of a computed log) | 487 ulp | minor |

### F1: c*x overflows for |c*x| > 1.8e308

- **Code:** `_logpdf`, line 3421: `cx = ... operator.mul (c, x)` and line 3429 `~((cx == 1) | (cx == -np.inf))` (fill -inf); `_loglogcdf`, line 3410: `sc.log1p(-c*x)/c`, used by `_logcdf`, `_cdf`, `_sf` (lines 3436-3443).
- **Mechanism:** pattern 3. On the unbounded side of the support (x -> -inf for
  c > 0, x -> +inf for c < 0) the product `c*x` overflows once
  |x| > 1.8e308/|c|. In `_logpdf` the overflow is caught by the
  `cx == -inf` test and replaced by -inf; in `_loglogcdf`,
  `log1p(inf)/c` is +inf (c > 0) or -inf (c < 0), so logcdf is -inf or -0,
  cdf 0 or 1, and sf 1 or 0. The true quantities only involve
  log(1 - c*x) ~ 710, divided by c: for |c| = 1000, `log(-log F)` is about
  0.71 and cdf is an ordinary O(1) number. For c = 2 the cdf underflows but
  logcdf and logpdf (~ -1.9e154) are still normal doubles.
- **Regime:** |c*x| > 1.8e308, i.e. |x| > 1.8e308/|c|; cdf/sf/pdf values are
  normal only for |c| >~ 100 (1.8e308**(1/c) moderate), log values for
  |c| >~ 1.
- **Reproducer:** `python accuracy.py refs/genextreme_ref.py --at cdf -1e308 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-1e308, 1000)` | 0.0 | 0.12919341754512428 | inf | 0.00205 | loss |
  | `sf(-1e308, 1000)` | 1.0 | 0.8708065824548757 | 6.7e+14 | 0.000304 | severe |
  | `logcdf(-1e308, 1000)` | -inf | -2.046444636724674 | inf | 0.001 | loss |
  | `logpdf(-1e308, 1000)` | -inf | -717.4343045939518 | inf | 0.0014 | loss |
  | `cdf(1e308, -1000)` | 1.0 | 0.6134525509770794 | 2.8e+15 | 0.000489 | severe |
  | `sf(1e308, -1000)` | 0.0 | 0.38654744902292065 | inf | 0.000775 | loss |
  | `logcdf(1e308, -1000)` | -0.0 | -0.4886523593428336 | inf | 0.001 | loss |
  | `logpdf(-1.79e308, 2)` | -inf | -1.89208879284245e+154 | inf | 0.5 | loss |

  Every point with |x| > 1.8e308/|c| on the parameter sets (2), (1000), (-1000) is affected (3-4 points per method and set in the run). pdf is not flagged because the true pdf is subnormal there (logpdf ~ -711 ... -717).
- **Fix direction:** compute `log1p(-c*x)` as `log(|c|) + log(|x|) + log1p(-1/(c*x))` (or `log(-c) + log(-x)`) when |c*x| is large, and avoid forming `c*x` in the `cx == -inf` test.

### F2: expm1 overflow in ppf and isf

- **Code:** `_ppf`, line 3449 and `_isf`, line 3456: `-sc.expm1(-c * x) / c`.
- **Mechanism:** pattern 3/8. `expm1(-c*y)` overflows for -c*y > 709.78 while
  `expm1(-c*y)/c` stays below 1.8e308 up to -c*y = 709.78 + log|c|. The
  result is -inf (c > 0, q small-ish) or +inf (c < 0, q large-ish) instead
  of a finite value up to 1.7e308.
- **Regime:** |c| > 1 and -c*y in (709.78, 709.78 + log|c|) with
  y = -log(-log q) (ppf) or -log(-log1p(-q)) (isf). For c = 1000 this is
  q in about (0.1290, 0.1306) for ppf and (0.8691, 0.8710) for isf; the
  window widens in y as log|c|/|c| but narrows in q.
- **Reproducer:** `python accuracy.py refs/genextreme_ref.py --at ppf 0.1290580551878314 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.1290580551878314, 1000)` | -inf | -1.6688296665906455e+308 | inf | 488 | loss |
  | `ppf(0.612, -1000)` | inf | 7.909840797398866e+305 | inf | 2.04e+03 | loss |
  | `isf(0.8691479746591456, 1000)` | -inf | -1.9246985495371606e+305 | inf | 3.27e+03 | loss |
  | `isf(0.3863939526808092, -1000)` | inf | 1.6688296665904717e+308 | inf | 1.29e+03 | loss |

  All 6 window points per method on (1000) and (-1000) are losses.
- **Fix direction:** for -c*y > 709 use `-sign(c)*exp(-c*y - log|c|)` (expm1 + 1 is negligible there), i.e. evaluate in log space.

### F3: c*x rounding to 1 just inside the upper endpoint

- **Code:** `_logpdf`, line 3429: `~((cx == 1) | ...)` -> fill -inf; `_loglogcdf`, line 3410: `log1p(-c*x)` = `log1p(-1)` = -inf, so `_sf` gives 0.
- **Mechanism:** pattern 7. For c > 0 the double x = fl(1/c) (or a
  neighbour) can lie strictly inside the support (exact c*x < 1) while
  fl(c*x) == 1. The code then treats x as the endpoint: logpdf -inf, pdf 0,
  sf 0, logcdf -0. The true values are normal (1 - c*x = 4.1e-17 for
  c = 0.7, x = 1.4285714285714286).
- **Regime:** c > 0, x within an ulp of 1/c with exact c*x < 1 but rounding to 1.
  The problem is extremely ill-conditioned (cond (arg) ~1e15-1e16: one ulp
  of x changes 1 - c*x by order 100%), so this is a loss in the tool's
  sense but a 1-ulp-of-x effect. The next double below (x = 1.4285714285714284)
  is evaluated without a loss and within conditioning.
- **Reproducer:** `python accuracy.py refs/genextreme_ref.py --at sf 1.4285714285714286 0.7`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1.4285714285714286, 0.7)` | 0.0 | 3.9198103378150115e-24 | inf | 3.5e+16 | loss |
  | `logpdf(1.4285714285714286, 0.7)` | -inf | -16.168799688730708 | inf | 6.4e+14 | loss |
  | `pdf(1.4285714285714286, 0.7)` | 0.0 | 9.505599587478995e-08 | inf | 1.0e+16 | loss |
  | `logpdf(1e12, 1e-12)` | -inf | -38445147782285.01 | inf | 1.3e+15 | loss |

- **Fix direction:** compute 1 - c*x with an FMA-style exact residual (`-fma(c, x, -1)`) or work with x - 1/c in double-double; not worth much given the conditioning.

### F4: ppf and isf return 0 at the q nearest exp(-1)

- **Code:** `_ppf`, line 3446: `x = -np.log(-np.log(q))`; `_isf`, line 3453: `x = -np.log(-sc.log1p(-q))`.
- **Mechanism:** pattern 2. At q = fl(exp(-1)) = 0.36787944117144233 the exact
  -log(q) is 1 - 3.4e-17, which rounds to 1.0, and `log(1.0)` is 0. The
  outer `log` has no `log1p` form for an argument near 1. Same for isf at
  q = fl(1 - exp(-1)). Both branches (c == 0 fill and c != 0) are affected
  because y is computed before the branch.
- **Regime:** q within a few ulp of exp(-1) (ppf) or 1 - exp(-1) (isf), any c;
  relative error ~ eps/|y| for |y| < ~1e-14. cond (arg) is 3e16 (ppf) and
  5.1e16 (isf): the absolute error 3.4e-17 is what one ulp of q causes.
- **Reproducer:** `python accuracy.py refs/genextreme_ref.py --at ppf 0.36787944117144233 0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.36787944117144233, 0)` | 0.0 | 3.3784855259134224e-17 | inf | 3.0e+16 | loss |
  | `ppf(0.36787944117144233, -0.1)` | 0.0 | 3.3784855259134224e-17 | inf | 3.0e+16 | loss |
  | `isf(0.6321205588285577, 0)` | 0.0 | 3.3784855259134224e-17 | inf | 5.1e+16 | loss |

  The point is a loss for all 9 parameter sets in the run.
- **Fix direction:** near q0 = exp(-1), compute `log(-log q) = log1p(-(log(q) + 1))` with `log(q) + 1` from `log1p((q - q0_hi - q0_lo)/q0)` (q0 split into two doubles).

### F5: logpdf cancels for c near 1 near the upper endpoint

- **Code:** `_logpdf`, line 3431: `-pex2 + lpex2 - lex2` with `lpex2 = log1p(-c*x)/c` (line 3410) and `lex2 = log1p(-c*x)` (line 3423).
- **Mechanism:** pattern 5. With L = log1p(-c*x), `lpex2 - lex2` = L/c - L =
  L*(1 - c)/c. For c close to 1 the two terms of size |L| cancel to a
  result |1 - c| times smaller; the rounding of L/c (eps*|L|/2) remains.
  Near the upper endpoint L is large negative while `pex2` = (1 - c*x)**(1/c)
  is small, so the cancelled term dominates logpdf and its relative error is
  about eps*|L|/|logpdf|.
- **Regime:** |1 - c| small (1e-8 ... 1e-12 tested) and 1 - c*x small enough
  that (1 - c*x)**(1/c) <~ |L*(1 - c)|; errors up to about |L|/|1 - c| ulp
  of the cancelled term. Part of each measured error is conditioning; the
  tool flags the excess.
- **Reproducer:** `python accuracy.py refs/genextreme_ref.py --at logpdf 1.0 0.99999999`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.0, 0.99999999)` | -1.9420680885673391e-07 | -1.942068083651187e-07 | 1.1e+07 | 0.949 | severe |
  | `logpdf(0.9999999800000002, 1.00000001)` | 1.7420680364921282e-07 | 1.7420680257946992e-07 | 2.8e+07 | 1.1e+07 | severe |
  | `logpdf(0.9999999998989999, 1.000000000001)` | -7.697309456489165e-11 | -7.697211041771563e-11 | 5.8e+10 | 1.3e+10 | severe |
  | `logpdf(0.9999999989989999, 1.000000000001)` | -9.792735511382489e-10 | -9.792748635641848e-10 | 6.0e+09 | 1.0e+09 | severe |
  | `logpdf(0.9999989999989999, 1.000000000001)` | -9.999861827481027e-07 | -9.999861833048077e-07 | 2.5e+06 | 1.0e+06 | severe |

  Only the second row lies on the main run's grid (parameter set
  (1.00000001)); the others are targeted `--at` points. (cond (shapes) is 1.2e8 there: logpdf is very sensitive to c, but c is
  an exact user input.) pdf = exp(logpdf) is not affected materially because
  logpdf is tiny there (absolute error ~4e-15).
- **Fix direction:** compute the difference directly as `L * ((1 - c)/c)` (1 - c is exact for c in [0.5, 2]).

### F6: exponentiation of a computed large exponent (minor)

- **Code:** `_pdf`, line 3417: `np.exp(self._logpdf(x, c))`; `_logcdf`, line 3437 / `_sf`, line 3443: `-exp(_loglogcdf)` with `_loglogcdf` = `log1p(-c*x)/c`; `_isf`, line 3456: `expm1(-c * x)` with x = -log(-log1p(-q)).
- **Mechanism:** the exponent E (logpdf ~ -700, log(-log F) ~ +-690,
  -c*y ~ 69) carries a rounding error of ~|E|*eps; exp turns it into a
  relative error of ~|E| ulp, while the conditioning with respect to x is
  O(1) (it is the cond (shapes) that is ~|E|, since 1/c enters the
  exponent).
- **Regime:** |E| between ~100 and ~709: pdf for |c| = 1000 at |x| = 1e30 ...
  1e300; logcdf/sf for c = -0.1 at x = 1e20 ... 1e30; logpdf/logcdf for
  c = 0.5, 1.00000001, 2 at x = -1e10 ... -1.8e308; isf for c = -0.1 at
  q = 1e-300 ... 1e-76.
- **Reproducer:** `python accuracy.py refs/genextreme_ref.py --at sf 1e30 -0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1e30, -0.1)` | 9.999999999999282e-291 | 1.0000000000000364e-290 | 487 | 10 | minor |
  | `logcdf(1e30, -0.1)` | -9.999999999999282e-291 | -1.0000000000000364e-290 | 487 | 10 | minor |
  | `pdf(1e300, -1000)` | 3.025767338206693e-304 | 3.025767338206373e-304 | 476 | 1 | minor |
  | `pdf(-3.5, -0.1)` | 6.294549288196871e-31 | 6.29454928819744e-31 | 407 | 394 | minor |
  | `logpdf(-1e150, 0.5)` | -2.4999999999999314e+299 | -2.4999999999999998e+299 | 123 | 2 | minor |
  | `isf(1e-300, -0.1)` | 9.999999999999976e+30 | 1.0000000000000038e+31 | 27.9 | 0.1 | minor |

  Maximum per method in the run: pdf 476, logcdf 487, sf 487, logpdf 457 (c = 1.00000001, x -> -inf), isf 28.3 ulp.
- **Fix direction:** limited: use `pow(1 - c*x, 1/c)` where 1/c is exact, or split the exponent (double-double) where it matters; probably not worth it.

## RuntimeWarnings

- `overflow encountered in multiply` (all x-methods): emitted at F1 points; the results there are wrong (F1).
- `overflow encountered in exp` (pdf, logpdf, cdf, logcdf, sf; all parameter sets): `pex2 = exp(logpex2)` overflows where -log F > 1.8e308, e.g. c = 0 at x = -30 or -1e300. The returned values (-inf logpdf/logcdf, 0 pdf/cdf, 1 sf) are correct (true values outside the double range); spurious warning.
- `underflow encountered in exp`: from `exp(logpex2)` and `exp(logpdf)` in the far right tail where the result truly underflows; values accurate.
- ppf/isf emit no warning in the F2 overflow window (silent inf).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid, all parameter sets except F1 points; c = +-1e-12 (near gumbel), c = 0 branch, endpoint 1/c | 1.72e3 ulp (within cond 3.2e3, c = -0.1 low tail); high tail 0.5 ulp |
| sf | low tail; high tail apart from F6 | 0 ulp (low tail); within conditioning elsewhere |
| logcdf | high tail near endpoints | 5.0e13 ulp, within cond 1.0e15 |
| logpdf | c = 0, c = -1e-12 everywhere; c = -0.1 low tail | 0.58 / 3.57 ulp; 5.3e14 within cond 9.9e15 |
| pdf | c = +-1000 near the endpoint, c = 1.00000001 | 1.9e14 ulp within cond 9.4e14; 5.0e5 within cond 1.0e7 |
| ppf | low tail (q down to 1e-300), high tail (1 - q down to 3e-16), all sets | 4.27 ulp (low), 1.41 ulp (high) |
| isf | high tail; bulk apart from F2/F4 | 2.84 ulp |
| all | small shapes c = +-1e-12 (pattern 9: log1p/expm1 keep accuracy; no c -> 0 problem) | within conditioning (max 572 ulp, cond 708) |

## Suspected, not confirmed

- Subnormal `c*x` for |c| < ~1e-300 (pattern 4): `log1p(-c*x)/c` loses relative
  accuracy when c*x is subnormal. By estimate the effect is absorbed in all
  methods (ll ~ -x enters only as exp(ll) ~ 1 + ll for small x) except ppf/isf
  near y = 0, where it is below the conditioning. Not measured (no shape
  below 1e-12 in PARAMS).
- `_stats`, `_munp`, `_entropy` (gamma-based moments, thresholds 1e-7, eps**0.29) are outside the scope of this review.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | log(sf), sf = -expm1(-t), t = exp(`_loglogcdf`) | `ll + log(-expm1(-t)/t)` with ratio 1 when t underflows (or `_log1mexp(-t)` for t not tiny), ll = `_loglogcdf(x, c)` | right tail once t < 2.2e-308: c = 0 for x > ~708, c < 0 for large x (c = -0.1: x > ~1e31), c > 0 for 1 - c*x < 2.2e-308**c. Demonstration: `logsf(800, 0)` = -inf, true -800.0; `logsf(1e40, -0.1)` = -inf, true -898.0081862676778 |

## Shared code

none (sc.log1p and sc.expm1 are used correctly; all problems are in genextreme's own expressions, with the `_loglogcdf` helper used only by this class).

## Existing GitHub issues

- #1296 (closed): scipy.stats.distributions.genextreme.pdf return nan for x=1 and c=1, which is wrong (Trac #769) - relates to F3 (upper-endpoint handling, older code)
- #6138 (closed, PR): ENH: stats: Improve behavior of genextreme.sf and genextreme.isf - background for F2/F3 (introduced current sf/isf forms)
- #14949 (closed): BUG: genextreme.ppf differs from analytical ppf - sign-convention mix-up, not an accuracy bug; no overlap with F2/F4
- No open issue found for any finding (searched via GitHub search API, gh not installed).

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-1e308, 1000)` | 0.0 | 0.12919341754512429 | verified | closed form exp(-(1-cx)^(1/c)) and mp.quad of the pdf in u = 1 - cx agree to 35 digits; also sf 1.0 vs 0.8708065824548757, logcdf/logpdf -inf vs -2.0464/-717.43, (1e308, -1000) cdf/sf/logcdf and logpdf(-1.79e308, 2) = -inf vs -1.892e154 all reproduced; mechanism (c*x overflow, lines 3410, 3422/3429) matches |
| F2 | `ppf(0.1290580551878314, 1000)` | -inf | -1.6688296665906456e308 | verified | ref from closed form (1 - (-log q)^c)/c, cdf(ref) == q; ppf(0.612, -1000), isf(0.8691..., 1000), isf(0.3863..., -1000) also inf vs finite; expm1 overflow at lines 3449/3456 matches |
| F3 | `sf(1.4285714285714286, 0.7)` | 0.0 | 3.9198103378150117e-24 | verified | quad of pdf over [x, 1/c] agrees; logpdf -inf vs -16.1688, pdf 0 vs 9.5056e-8, logpdf(1e12, 1e-12) -inf vs -3.8445e13 reproduced; `cx == 1` fill (line 3429) and log1p(-1) (line 3410) match; cond ~1e16 as the report says |
| F4 | `ppf(0.36787944117144233, 0)` | 0.0 | 3.3784855259134226e-17 | verified | also c = -0.1 and isf(0.6321205588285577, 0); -log(-log q) at line 3446/3453 matches; cond ~3e16 as stated |
| F5 | `logpdf(0.9999999998989999, 1.000000000001)` | -7.697309456489165e-11 | -7.6972110417715636e-11 | verified | 5.8e10 ulp; other four rows reproduced (1.1e7, 2.8e7, 6.0e9, 2.5e6 ulp); ref from -(1-cx)^(1/c) + (1/c - 1)log(1-cx); cancellation lpex2 - lex2 at line 3431 matches |

Verified by dist-accuracy-verifier: 5 of 5 verified.
