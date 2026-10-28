# kappa4: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `kappa4_gen`, lines 7258-7526
- **Implemented:** pdf (7397, `exp(_logpdf)`), logpdf (7402-7438), cdf (7440, `exp(_logcdf)`), logcdf (7443-7476), ppf (7478-7501); also `_get_support` (7356-7390)
- **Inherited (not measured):** sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** limiting cases of `logistic` (h = -1, k = 0), `gumbel_r` (h = k = 0), `genextreme` (h = 0), `genpareto` (h = 1), `expon`, `uniform`; helpers: `_lazyselect`; special functions: `sc.log1p`, `sc.xlog1py`
- **Reference:** `refs/kappa4_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (0, 0), (0, 0.1), (0.1, 0), (-1, 0), (-0.1, 0.1), (2, 0.5), (1, -0.5), (-2, -0.5), (0.5, 1e-8), (1e-8, 0.5), (-1000, 0.1), (-1e6, 0), (-1e6, -0.01)
- **Worst finding:** loss in cdf, logcdf, pdf, logpdf and ppf — including `cdf = 0.0` where the true value is 0.999 (bulk, cond 7e-4) and `ppf = nan` for every q when k = 0 and h < 0

## Summary

All five implemented methods are accurate in the bulk for "nice" shapes
(|h|, |k| of order 1): the Gumbel branch (h = k = 0) and the h < 0, k < 0
branch are essentially exact, and the generalized extreme value branch
(h = 0) is good to a few hundred ulp even at x = -1e30. Everything else fails
in one of five ways. (1) The intermediate `y = (1 - k x)**(1/k)` (resp.
`exp(-x)` for k = 0) is formed explicitly and overflows for h < 0 in the left
tail, so cdf/pdf return 0 and logcdf/logpdf return -inf in the **middle** of
the distribution (cdf 0.0 where the truth is 0.9993, cond 7e-4). (2) The ppf
branch for k = 0, h < 0 takes `np.log(h)` of a negative shape and returns nan
for *every* q, i.e. the quantile function of the logistic limit is entirely
broken. (3)-(4) The ppf composes two cancelling differences, `(1 - q**h)/h`
and `(1 - y**k)/k`, which lose all information about the small deviations:
small |h| or |k| gives 1e7-1e22 ulp, q -> 1 returns the support endpoint or
inf, and q -> 0 with h = 1 returns 0. (5) `log1p(-h*(1 - k x)**(1/k))` cancels
at the lower support endpoint for h > 0, so the whole lower tail of
cdf/logcdf collapses (cdf(1e-300, 1, -0.5) = 0.0 with cond 1). In addition,
`(1 - k x)**(1/k)` amplifies the rounding of `1 - k x` by 1/|k|, which gives
1e7 ulp in cdf/logcdf/pdf/logpdf at k = 1e-8 (and nan next to the lower
support endpoint, where the amplified y pushes h y past 1); the same
cancellation makes
the lower support endpoint returned by `_get_support` wrong by 1.6e-9
relative. The three inherited methods (sf, logsf, isf) collapse in the right
tail as usual for `1 - cdf` / `ppf(1 - q)`.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, logcdf, pdf, logpdf | h < 0 and `log\|h\| + log y > 709.78` in the left tail (e.g. x < -696 for h = -1e6, k = 0) | the product `-h*y` (and for small \|h\| the power `(1.0 - k*x)**(1.0/k)` / `np.exp(-x)` itself) overflows to inf, then `log1p(inf) = inf` (pattern 3) | cdf 0.0 where true = 0.9993 (cond 7e-4); logcdf -inf where true = -1e300 | loss |
| F2 | ppf | k = 0 and h < 0 (includes the logistic limit h = -1), all q | `-sc.log1p(-(q**h)) + np.log(h)`: `q**h > 1` and `h < 0`, so both logs are of negative numbers (pattern 7) | nan for all 133 measured q (cond 2.7) | loss |
| F3 | ppf | h != 0 and (q -> 1, or \|h\| <~ 1e-4, or h = 1 with q < 1e-8) | `(1.0 - q**h)/h` cancels; `q**h` rounds to 1 for \|h log q\| < eps (patterns 1, 8, 9) | inf / support endpoint returned; 2.2e13 ulp at h = 1e-8; 0.0 where true = 1e-300 (cond 1) | loss |
| F4 | ppf | k != 0 and \|k\| <~ 1e-4, or y**k -> 1 (q near exp(-1) for h = 0) | `1.0/k*(1.0 - y**k)` cancels; relerr ~ eps/(2\|k log y\|) (patterns 1, 9) | 1.2e22 ulp (1e8 returned instead of 36.74) at k = 1e-8 | loss |
| F5 | ppf | \|h\| >~ 1e3 and q < exp(-709.8/\|h\|) | `q**h` overflows to inf, then `(1 - inf)/h` and `inf**k` give -inf or the support endpoint (pattern 3) | -inf where true = -5.01e100 (cond 100); 1.1e12 ulp (cond 2.4) | loss |
| F6 | cdf, logcdf, pdf, logpdf | h > 0, x within ~1e-8 of the lower support endpoint a (relative) | `sc.log1p(-h*(1.0 - k*x)**(1.0/k))` with `h*y -> 1`: the small quantity 1 - h y is lost, then log1p(-1) = -inf (pattern 1) | cdf 0.0 where true = 1e-300 (cond 1); 3.7e8 ulp at x = 1e-10 (cond 1) | loss |
| F7 | cdf, logcdf, pdf, logpdf | \|k\| <~ 1e-4, all x | `(1.0 - k*x)**(1.0/k)`: the eps/2 rounding of `1 - k*x` is raised to the power 1/k, relerr ~ eps/(2\|k\|) (pattern 6) | nan (loss) at the lower endpoint, where the amplified y pushes `h*y` past 1; otherwise 3.7e7 ulp (severe) at k = 1e-8, cond 1.4 | loss |
| F8 | pdf, logcdf, logpdf | far tails, \|log y\| >~ 50, any shapes | `1.0/k` (and `1.0/h`) is a rounded quotient, its absolute error times log(1 - k x) is the relative error of y; plus `pdf = exp(logpdf)` turning the absolute error of logpdf into a relative error (patterns 11, 6) | 788 ulp (pdf), 175 ulp (logcdf) | minor |

### F1: the intermediate `(1 - k x)**(1/k)` (or `exp(-x)`) overflows, so cdf, pdf and their logs collapse in the bulk

- **Code:** `_logcdf.f0`, line 7453: `return (1.0/h)*sc.log1p(-h*(1.0 - k*x)**(1.0/k))`;
  `_logcdf.f2`, line 7465: `return (1.0/h)*sc.log1p(-h*np.exp(-x))`;
  the same subexpressions in `_logpdf.f0` (7414), `f1` (7421) and `f2` (7427).
- **Mechanism:** `y = (1 - k x)**(1/k)` is formed as a double. For h < 0 the
  support extends to -inf (k >= 0) or to 1/k (k < 0) and y grows without
  bound there, while the cdf stays O(1) because
  `log F = log1p(-h y)/h ~ (log|h| + log y)/h` is *divided* by the large |h|.
  As soon as y passes 1.8e308 the code gets `log1p(inf) = inf`, so
  logcdf = -inf, cdf = 0, logpdf = -inf, pdf = 0 — with no warning that the
  true answer is of order 1. For h = -1 (the logistic limit) the cdf itself
  underflows at the same time, but logcdf/logpdf are still normal doubles and
  are returned as -inf.
- **Regime:** h < 0 and `log|h| + ly > 709.78`, where `ly = log y` is `-x`
  (k = 0) or `sc.log1p(-k*x)/k` (k != 0). The overflow that actually triggers
  is the *product* `-h*y` (RuntimeWarning "overflow encountered in multiply"),
  so for |h| > 1 the failure starts before y itself overflows; for |h| < 1 the
  power/exp overflow (`ly > 709.78`) binds instead. Concretely:
  k = 0: x < -(709.78 - log|h|), i.e. x < -696 for h = -1e6 — not only
  x < -709.8 (`cdf(-700, -1e6, 0)` is already 0.0, see the first row below);
  k > 0: x < -(exp(k (709.78 - log|h|)) - 1)/k (x < -3.4e31 for
  (h, k) = (-1000, 0.1));
  k < 0: x - 1/k < exp(k (709.78 - log|h|))/|k|, which is representable only
  for |k| <~ 0.05 (for k = -0.5 the nearest double above 1/k still gives only
  y ~ 2e31, so the regime is unreachable there — not measured).
  The returned value has no correct digits as long as the true
  |log F| = |log1p(-h y)/h| stays below ~709, i.e. while
  |log|h| + ly| <~ 709.8 |h|.
- **Reproducer:** `python accuracy.py refs/kappa4_ref.py --at cdf -709.0 -1000000.0 0.0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-700.0, -1e6, 0.0)` | 0.0 | 0.9992864391951257 | inf | 7.0e-4 | loss |
  | `cdf(-709.0, -1e6, 0.0)` | 0.0 | 0.9992774456576439 | inf | 7.1e-4 | loss |
  | `logcdf(-709.0, -1e6, 0.0)` | -inf | -7.228155105579643e-4 | inf | 0.98 | loss |
  | `cdf(-6.702276267921237e+31, -1000.0, 0.1)` | 0.0 | 0.48835740432588315 | inf | 0.01 | loss |
  | `logcdf(-6.702276267921237e+31, -1000.0, 0.1)` | -inf | -0.7167077552789821 | inf | 0.014 | loss |
  | `cdf(-99.94613906403504, -1e6, -0.01)` | 0.0 | 0.9992338261506324 | inf | 0.19 | loss |
  | `pdf(-99.94613906403504, -1e6, -0.01)` | 0.0 | 0.0018552106610265507 | inf | 1.9e3 | loss |
  | `logcdf(-1e300, -1.0, 0.0)` | -inf | -1e300 | inf | 1.0 | loss |
  | `logcdf(-1e300, -0.1, 0.1)` | -inf | -68824.26842959202 | inf | 1.5e-3 | loss |
  | `pdf(-1e300, -1000.0, 0.1)` | 0.0 | 1.016248692870696e-305 | inf | 1.0 | loss |

  Affected counts in the run (parameter sets (-1e6, 0), (-1000, 0.1), (-1, 0),
  (-0.1, 0.1), (-1e6, -0.01)): 46 cdf, 46 pdf, 144 logcdf and 144 logpdf loss
  points, of which 35 per method are in the *bulk* region (1e-3 < F < 1 - 1e-3).
- **Fix direction:** never form y; work with `ly = sc.log1p(-k*x)/k` (resp.
  `-x`) and use `logcdf = sc.log1p(-h*exp(ly))/h` only while
  `ly + log|h| < 0`, otherwise `logcdf = (log(-h) + ly)/h + sc.log1p(exp(-ly)/(-h))/h`
  (the reference module does exactly this).

### F2: `ppf` returns nan for every q when k = 0 and h < 0

- **Code:** `_ppf.f2`, line 7493: `return -sc.log1p(-(q**h)) + np.log(h)`
  (branch `np.logical_and(h != 0, k == 0)`, so h < 0 lands here).
- **Mechanism:** the branch rewrites `ppf = -log((1 - q**h)/h)` as
  `-log1p(-q**h) + log(h)`, which is only valid for h > 0. For h < 0,
  `q**h = q**-|h| > 1`, so `log1p(-(q**h))` is the log of a negative number,
  and `np.log(h)` is the log of a negative shape; both give nan
  (RuntimeWarning "invalid value encountered in log"). The quotient
  `(1 - q**h)/h` itself is perfectly positive — only the split into two logs
  is invalid. The logistic limit (h = -1, k = 0) is in this branch, so
  `kappa4.ppf(q, -1, 0)` is nan for every q while `kappa4.cdf` is correct.
- **Regime:** k == 0 exactly and h < 0; all q in (0, 1). 133 of 134 measured
  points are nan (the exception is q where the true answer is 0.0 and is not
  measured).
- **Reproducer:** `python accuracy.py refs/kappa4_ref.py --at ppf 0.36 -1.0 0.0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.36, -1.0, 0.0)` | nan | -0.5753641449035619 | inf | 2.7 | loss |
  | `ppf(1e-300, -1.0, 0.0)` | nan | -690.7755278982137 | inf | 1.4e-3 | loss |
  | `ppf(0.36, -1e6, 0.0)` | nan | -1021637.4320214235 | inf | 0.98 | loss |

- **Fix direction:** use the sign-safe form
  `y = -sc.expm1(h*np.log(q))/h; ppf = -np.log(y)`, which is correct for both
  signs of h and also removes the small-|h| cancellation of F3.

### F3: `(1 - q**h)/h` in `_ppf` loses the small quantity

- **Code:** `_ppf.f0`, line 7485: `return 1.0/k*(1.0 - ((1.0 - (q**h))/h)**k)`;
  `_ppf.f2`, line 7493: `-sc.log1p(-(q**h)) + np.log(h)`.
- **Mechanism:** pattern 1/8/9 in three flavours.
  (a) q -> 1: `q**h` rounds to exactly 1 when |h log q| < eps, so
  `1 - q**h` is 0 and the formula returns the support endpoint (f0) or
  `log1p(-1) = -inf`, i.e. ppf = +inf (f2), instead of a finite quantile.
  (b) |h| small: `1 - q**h = -expm1(h log q) ~ -h log q` is a difference of
  two nearby numbers, so its relative error is ~eps/(2|h log q|) — 2e-8 at
  h = 1e-8 — and that error passes straight into x.
  (c) h = 1 (generalized Pareto limit) with q -> 0: y = 1 - q, whose deviation
  from 1 is all that matters for x = (1 - y**k)/k, but `1.0 - q` rounds to 1.0
  for q < eps, so ppf returns 0 where the true quantile is q itself.
- **Regime:** h != 0 and any of 1 - q <~ 1e-15/|h|, |h| <~ 1e-4,
  q <~ 1e-8 with h = 1 (more generally: h > 0 and q**h below ~1e-8).
- **Reproducer:** `python accuracy.py refs/kappa4_ref.py --at ppf 0.9999999999999997 0.1 0.0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999999997, 0.1, 0.0)` | inf | 35.63818828100899 | inf | 8.4e13 | loss |
  | `ppf(0.9999999999999997, -0.1, 0.1)` | 10.0 (= 1/k) | 9.716695711644448 | 1.3e14 | 8.8e12 | severe |
  | `ppf(0.3, 1e-08, 0.5)` | -0.19451388624928168 | -0.1945138842834269 | 4.6e7 | 4.7 | severe |
  | `ppf(0.3678794374926479, 1e-08, 0.5)` | -5.024759275329416e-09 | -4.9999999919574115e-09 | 2.2e13 | 2.0e8 | severe |
  | `ppf(1e-300, 1.0, -0.5)` | 0.0 | 1e-300 | inf | 1.0 | loss |
  | `ppf(1e-08, 1.0, -0.5)` | 9.99999993922529e-09 | 1.0000000075000001e-08 | 6.1e7 | 1.0 | severe |

  Note the large cond (q) for the q -> 1 rows (8e12-8e13): the quantile is
  genuinely ill-conditioned there, but returning inf or the endpoint is a
  total loss, and the h = 1e-8 and h = 1 rows have cond 1-5.
- **Fix direction:** `y = -sc.expm1(h*np.log(q))/h` for the value and
  `log_y = np.log(-sc.expm1(h*np.log(q))) - np.log(abs(h))` (via `log1p` of
  the small part when `q**h` is tiny) for the subsequent power, so that the
  deviation of y from its limit is never rounded away.

### F4: `1.0/k*(1.0 - y**k)` in `_ppf` cancels for small |k|

- **Code:** `_ppf.f0`, line 7485 (`1.0/k*(1.0 - (...)**k)`) and `_ppf.f1`,
  line 7488 (`1.0/k*(1.0 - (-np.log(q))**k)`).
- **Mechanism:** `y**k -> 1` as k -> 0, so `1 - y**k ~ -k log y` is computed
  as a difference of two numbers near 1: relative error ~eps/(2|k log y|)
  (pattern 9). At k = 1e-8 that is 1e-8 relative, i.e. ~5e7 ulp, over the
  whole q range; and when |k log y| drops below eps the difference is exactly
  0 and the formula returns the support endpoint 1/k. The same cancellation
  occurs for any k when y -> 1, i.e. for h = 0 at q near exp(-1).
- **Regime:** |k| <~ 1e-4 (all q), or |k log y| <~ 1e-8 (q near exp(-1) for
  h = 0, q near the value where the quantile crosses 0).
- **Reproducer:** `python accuracy.py refs/kappa4_ref.py --at ppf 0.9999999999999999 0.5 1e-08`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999999999, 0.5, 1e-08)` | 100000000.0 (= 1/k) | 36.736793821715345 | 1.2e22 | 2.5e14 | loss |
  | `ppf(0.01, 0.5, 1e-08)` | -0.5877866637860052 | -0.5877866666295848 | 2.2e7 | 0.094 | severe |
  | `ppf(1e-300, 0.5, 1e-08)` | -0.6931471840943004 | -0.6931471829622103 | 7.4e6 | 7e-151 | severe |
  | `ppf(0.3678794374926479, 0.0, 0.1)` | -1.000000082740371e-08 | -1.0000000036540745e-08 | 3.6e8 | 1.0e8 | severe |

  51 of the 67 measured q values are severe for (h, k) = (0.5, 1e-8).
- **Fix direction:** `x = -sc.expm1(k*log_y)/k` with `log_y` as in F3 (and the
  k -> 0 limit `x = -log_y` recovered automatically by `expm1`).

### F5: `q**h` overflows in `_ppf` for large |h|

- **Code:** `_ppf.f0`, line 7485: `1.0/k*(1.0 - ((1.0 - (q**h))/h)**k)`.
- **Mechanism:** for h < 0, `q**h = exp(h log q)` overflows as soon as
  |h log q| > 709.8, which for h = -1000 already happens at q < 0.49. Then
  `(1 - inf)/h = inf`, `inf**k` is inf (k > 0) or 0 (k < 0), and the result is
  -inf or the support endpoint 1/k, although the true quantile is a perfectly
  normal double (-5e100 resp. -99.976). Pattern 3.
- **Regime:** h < 0 with q < exp(-709.8/|h|) (any k != 0); 11 of 67 q values
  for (h, k) = (-1000, 0.1).
- **Reproducer:** `python accuracy.py refs/kappa4_ref.py --at ppf 0.1 -1000.0 0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.1, -1000.0, 0.1)` | -inf | -5.011872336272759e+100 | inf | 100 | loss |
  | `ppf(0.9991516571017559, -1e6, -0.01)` | -100.0 (= 1/k) | -99.9763337230032 | 1.1e12 | 2.4 | severe |

- **Fix direction:** stay in log space: `log_y = log(-expm1(h*log q)) - log|h|`
  (asymptotically `h*log(q) - log|h|`), then
  `x = -sc.expm1(k*log_y)/k`, which never forms `q**h`.

### F6: `log1p(-h*y)` cancels at the lower support endpoint (h > 0)

- **Code:** `_logcdf.f0`, line 7453 and `_logcdf.f2`, line 7465
  (`sc.log1p(-h*...)`); `_logpdf.f0`/`f2`, lines 7414/7427 (`sc.xlog1py(1.0/h - 1.0, -h*...)`).
- **Mechanism:** for h > 0 the support starts where h y = 1, and the cdf is
  `exp(log1p(-h y)/h)`. The product `h*y` is formed first, so the small
  quantity 1 - h y is computed as a difference of doubles near 1: all its
  digits are lost when 1 - h y <~ 1e-8 and it becomes exactly 0 when
  1 - h y <~ eps, so logcdf = -inf and cdf = 0. (A nan appears instead when
  the computed `h*y` lands just *above* 1; the only measured instance of
  that, at (h, k) = (0.5, 1e-8), is caused by F7's 1/k amplification of y and
  is reported there, not here.) The h = 1, k = -0.5 case is the generalized Pareto limit,
  where the lower endpoint is 0 and cdf(x) ~ x: the entire lower tail
  x < 1e-16 returns 0 although the conditioning is 1.
- **Regime:** h > 0 and (x - a)/|a| <~ 1e-8 where a is the lower support
  endpoint ((1 - h**-k)/k, or log h for k = 0); 30 loss + 3 severe points for
  (1, -0.5), plus the endpoint point of (0.1, 0).
- **Reproducer:** `python accuracy.py refs/kappa4_ref.py --at cdf 1e-300 1.0 -0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-300, 1.0, -0.5)` | 0.0 | 1e-300 | inf | 1.0 | loss |
  | `logcdf(1e-300, 1.0, -0.5)` | -inf | -690.7755278982137 | inf | 1.4e-3 | loss |
  | `cdf(1e-10, 1.0, -0.5)` | 1.0000000827403705e-10 | 9.999999999250001e-11 | 3.7e8 | 1.0 | severe |
  | `cdf(1e-08, 1.0, -0.5)` | 9.999999828202971e-09 | 9.999999925e-09 | 4.4e7 | 1.0 | severe |
  | `cdf(-2.3025850929940455, 0.1, 0.0)` | 0.0 | 2.2014201533621374e-158 | inf | 1.3e17 | loss |

- **Fix direction:** compute `log(h*y) = log(h) + sc.log1p(-k*x)/k` and then
  `log1p(-h*y) = np.log(-sc.expm1(log_hy))`, which keeps full relative
  accuracy of 1 - h y right up to the endpoint.

### F7: `(1 - k x)**(1/k)` amplifies the rounding of `1 - k x` by 1/|k|

The systematic error over the whole x range is severe (3.7e7 ulp at
k = 1e-8); the two points next to the lower support endpoint are a loss
(nan), which is why the finding is labelled loss.

- **Code:** `_logcdf.f0`, line 7453 and `_logpdf.f0`/`f1`, lines 7414/7421:
  `(1.0 - k*x)**(1.0/k)`.
- **Mechanism:** pattern 6. `fl(1 - k x)` carries a representation error of
  up to eps/2 relative; raising it to the power 1/k multiplies that relative
  error by 1/|k|, so y (and hence cdf, logcdf, pdf, logpdf) has relative
  error ~eps/(2|k|) — about 1e-8 at k = 1e-8 — for *all* x, including the
  centre of the distribution. Near the lower support endpoint (h > 0) the
  amplified error is enough to push the product `h*y` past 1 although the
  exact product is below it (at (h, k) = (0.5, 1e-8),
  x = -0.6931471740943004: `h*y` = 1.00000000113209 where the exact value is
  0.99999999113209), so `sc.log1p`/`sc.xlog1py` of a number below -1 returns
  nan; these are the two nan points flagged for pdf, logpdf, cdf and logcdf
  at this parameter set. The same cancellation of `1 - h**-k` makes the lower
  support endpoint returned by `_get_support.f0` (line 7365, `(1.0 - np.float_power(h, -k))/k`)
  wrong: for (h, k) = (0.5, 1e-8) SciPy returns a = -0.69314718409430043 where
  the true endpoint is -0.69314718296221038 (relative error 1.6e-9), which is
  why pdf/cdf are also asked at a few points outside the true support
  there.
- **Regime:** |k| <~ 1e-4, all x; the error is ~0.5/|k| ulp (2.2e2 ulp at
  k = 1e-4, 1e7 ulp at k = 1e-8). Not visible for k = 0 (separate branch) or
  |k| >~ 1e-3.
- **Reproducer:** `python accuracy.py refs/kappa4_ref.py --at cdf 1.0 0.5 1e-08`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1.0, 0.5, 1e-08)` | 0.6659543826472618 | 0.6659543811387698 | 1.0e7 | 0.45 | severe |
  | `logcdf(-0.5, 0.5, 1e-08)` | -3.478644861017411 | -3.4786448895419264 | 3.7e7 | 1.4 | severe |
  | `logcdf(2.0, 0.5, 1e-08)` | -0.14013183757021674 | -0.14013183741740748 | 4.9e6 | 2.1 | severe |
  | `cdf(-0.5, 0.5, 1e-08)` | 0.030849187656566718 | 0.030849186776608608 | 1.3e8 | 4.7 | severe |
  | `pdf(-0.6931471740943004, 0.5, 1e-08)` | nan | 1.773581953159361e-08 | inf | 7.8e7 | loss |
  | `cdf(-0.6931471824645996, 0.5, 1e-08)` | nan | 2.476164800229633e-19 | inf | 2.8e9 | loss |

- **Fix direction:** `np.exp(sc.log1p(-k*x)/k)`, i.e. keep `log y` and
  exponentiate once (relative error then ~|log y| eps instead of eps/|k|);
  the same rewrite fixes `_get_support` via `-sc.expm1(-k*np.log(h))/k`.

### F8: rounded `1.0/k` exponent and `pdf = exp(logpdf)` in the far tails

- **Code:** `_logpdf`/`_logcdf`, lines 7413-7465 (`**(1.0/k)`, `1.0/h`), and
  `_pdf`, line 7400 (`np.exp(self._logpdf(...))`).
- **Mechanism:** patterns 11 and 6. `1.0/k` is a rounded double: for k = 0.1
  it is 10.0 while 1/fl(0.1) = 9.99999999999999944, an absolute difference of
  5.6e-16. The relative error of `(1 - k x)**(1.0/k)` is that difference times
  log(1 - k x), i.e. |log y| * eps/2 ulp — 169 ulp at x = -1e30, k = 0.1,
  exactly what is measured. Separately, `pdf = exp(logpdf)` converts the
  absolute error of logpdf (~|logpdf| eps) into a relative error, giving
  ~|logpdf|/2 ulp in the far tails (same as gamma F2 in `reports/gamma.md`).
- **Regime:** any shapes, |log y| or |logpdf| large (deep tails); max measured
  788 ulp (pdf) and 175 ulp (logcdf/logpdf).
- **Reproducer:** `python accuracy.py refs/kappa4_ref.py --at logcdf -1e30 0.0 0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logcdf(-1e30, 0.0, 0.1)` | -1.000000000000001e+290 | -9.999999999999637e+289 | 168 | 10 | minor |
  | `pdf(-1e30, -1000.0, 0.1)` | 5.093308710572845e-33 | 5.093308710571954e-33 | 788 | 1.0 | minor |
  | `pdf(10.0, 0.5, 1e-08)` | 4.539888102662621e-05 | 4.5398881026631595e-05 | 534 | 10 | minor |

- **Fix direction:** nothing cheap for `exp(logpdf)`; for the exponent, use
  `sc.log1p(-k*x)/k` (one rounding of the quotient instead of an amplified
  rounding of `1.0/k`).

## RuntimeWarnings

`_lazyselect` evaluates each branch only on the elements that select it, so
the warnings are not the usual "both branches of `np.where`" noise — most of
them accompany genuinely wrong values:

- `overflow encountered in power` / `overflow encountered in multiply` /
  `overflow encountered in exp`: F1 and F5 (y or `q**h` overflowing). Of the
  points that emit them, 298 + 116 + 160 are loss cases; the rest are points
  where the true value is 0 or 1 to double precision (e.g. `pdf(-1e300, 0, 0)`),
  so the warning is harmless there.
- `invalid value encountered in log`: F2 (`np.log(h)` with h < 0) — 133 of 134
  such points are nan losses.
- `underflow encountered in power` / `underflow encountered in exp` /
  `underflow encountered in multiply`: mostly harmless (the result is an
  accurate tail value), e.g. `ppf(1e-300, 2, 0.5) = 0.5857864376269049`
  (accurate, cond 1) emits "underflow encountered in power"; but the same
  warning also appears on the F4/F6 losses, so it is not a reliable signal.
- `divide by zero encountered in power` + `invalid value encountered in add`:
  emitted mostly at or next to the support endpoints (e.g.
  `pdf(-2.0, -2.0, -0.5)`, 206 accurate / 44 minor / 14 loss points), where
  `(1 - k x)**(1/k - 1)` is 0**negative; the returned nan matches the
  (undefined) limit and is not counted as an accuracy finding.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf, logpdf, cdf, logcdf | (0, 0) Gumbel branch f3, full grid x in [-1e300, 1e300] | 56 ulp (cond 2.4e3), logpdf/logcdf exact |
| pdf, logpdf, cdf, logcdf | (-2, -0.5) branch f0 with h < 0, k < 0, full grid | 155 ulp (pdf, far tail, F8), logcdf 0.83 ulp |
| pdf, logpdf, cdf, logcdf | (0, 0.1) and (0, 0.1)-like genextreme branch f1, bulk and both tails | 175 ulp (F8); bulk < 10 ulp |
| cdf, logcdf | (1e-8, 0.5): the h -> 0 limit is handled correctly by `log1p(-h*y)/h` | 13.4 ulp (cond 60) |
| pdf, logpdf, cdf, logcdf | high tail (x -> upper endpoint 1/k) for (2, 0.5), (0, 0.1), (0.1, 0) | 2.6e15 ulp but cond (x) 7.4e16: within conditioning |
| ppf | (2, 0.5) and (-2, -0.5), full q grid including q = 1e-300 and 1 - 1e-15 | 475 ulp (cond 6.7e3) and 1.9e7 ulp (cond 8.9e7): within conditioning |
| ppf | (0, 0) branch f3, full q grid | only q = fl(exp(-1)) fails (0.0 vs 3.4e-17) with cond 3.0e16: conditioning, not a code defect |

## Suspected, not confirmed

- **F1 for k < 0 with |k| >~ 0.05:** the same overflow of `(1 - k x)**(1/k)`
  must occur for x within exp(709.8 k)/|k| of the lower endpoint 1/k, but for
  k = -0.5 the nearest double above 1/k still gives only y ~ 2e31, so the
  regime is unreachable in double precision and could not be measured. It was
  measured for k = -0.01 (parameter set (-1e6, -0.01)).
- **`_logpdf`/`_logcdf` for very large |h| combined with h y near 1:** not
  reachable, since h > 0 is required for h y -> 1 and the overflow regime of
  F1 needs h < 0.
- **`_stats`/`_mom1_sc` (lines 7503-7524):** `(-1.0/h*k).astype(int)` looks
  like a transcription of "-1/(h k)" and truncates rather than rounds, so the
  number of existing moments may be misreported; moments are outside the
  scope of this review and were not measured.
- 11 pdf/cdf and 5 logpdf/logcdf grid points (exactly at a support endpoint,
  e.g. x = 1/k, or below the inaccurate lower endpoint of (0.5, 1e-8)) have no
  mpmath reference (the exact-arithmetic value is 0**negative or complex) and
  were skipped.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | `1 - cdf` | `sf = -expm1(logcdf)` with the accurate `_logcdf`; for h != 0 this is `-expm1(log1p(-h*y)/h)` | whenever cdf > 1 - eps, i.e. the whole right tail: `sf(40, 0, 0) = 0.0`, true 4.248354255291589e-18 (cond 40) |
| logsf | `log(sf)` = `log(1 - cdf)` | `sc.log1p(-exp(logcdf))` -> `_log1mexp(logcdf)` | same cliff: `logsf(40, 0, 0) = -inf`, true -40.0 (cond 1) |
| isf | `ppf(1 - q)` | invert the sf directly: `y = -expm1(log1p(-q**... ))`; concretely `y = -expm1(h*log1p(-q))/h` then `x = -expm1(k*log y)/k`, so small q keeps full relative accuracy | q < eps (1 - q rounds to 1): `isf(1e-300, 0, 0) = inf`, true 690.7755278982137 (cond 1.4e-3); already broken for *all* q when k = 0, h < 0 because it calls the nan branch F2 |

A custom `_sf`/`_logsf` is cheap here because `1 - F = -expm1(logcdf)` and
`_logcdf` is already a closed form; a custom `_isf` is the same expression
tree as `_ppf` with `log1p(-q)` in place of `log(q)`.

## Shared code

- No `scipy.special` function is used outside its accurate range: `sc.log1p`
  and `sc.xlog1py` are called correctly, and the losses are all in the
  arguments handed to them.
- `_lazyselect` (shared by several distributions) behaves correctly; it is the
  *content* of the branches that is wrong. Worth noting for the fixing step:
  because `_lazyselect` dispatches on `h == 0` / `k == 0` exactly, the h = 0
  and k = 0 branches do not help at h = 1e-8 or k = 1e-8, which is exactly
  where F3, F4 and F7 do their damage (pattern 9).
- The mechanism of F8 (a rounded reciprocal exponent, `1.0/k`, amplified by
  `log(1 - k x)`) is the same as fisk F5/F6 in `reports/fisk.md`, and the
  `pdf = exp(logpdf)` amplification is the same as gamma F2 in
  `reports/gamma.md`; both are generic to this code style rather than
  specific to kappa4.
- kappa4 is the four-parameter generalization of `logistic`, `gumbel_r`,
  `genextreme` and `genpareto`. Those classes have their own (correct)
  implementations, so the bugs here do not propagate; conversely, their
  formulas are a ready source of fixes for the limiting branches — e.g.
  `genpareto`'s `_cdf` avoids the F6 cancellation that kappa4 suffers at
  h = 1.

## Existing GitHub issues
- #17832 (open): Tracking issue: custom methods for continuous distributions -
  relates to the inherited `sf`, `logsf` and `isf`, which all collapse at their
  demonstration points (`sf(40, 0, 0) = 0.0` vs 4.25e-18, `logsf = -inf` vs
  -40, `isf(1e-300, 0, 0) = inf` vs 690.78); the issue names `sf = 1 - _cdf` as
  a "risk of catastrophic cancellations in the right tail" and tracks which
  distributions still lack their own implementations.
- #11465 (closed, implemented in PR #21050): "Numerical stability of
  distributions: Add log_p as argument to all distributions?" - same scope for
  the inherited `logsf`, which takes `log` of an already rounded `sf`.
- F1-F8 and the `_get_support` defect: none found. Nothing on SciPy's tracker
  reports the branch failures measured here - in particular F2 (`ppf` returning
  nan for every q when k = 0 and h < 0, the logistic limit included) is a total
  failure of a whole parameter branch and appears to be unreported.
- Two kappa4 issues were checked and excluded as unrelated to accuracy:
  #7416 (closed, "scipy.stats.kappa4(h,k) raise a ValueError for positive
  integer values of h and k", fixed by PR #7420 - an `_argcheck` dtype problem,
  and its discussion contains nothing about nan or accuracy), and PR #15250
  ("ENH: Make `stats.kappa4` work with array"), which introduced the
  `_lazyselect` structure but is about array support, not precision.
- Also checked and excluded: #15195 (closed, the tracking issue for
  distributions that warn or return nan at support endpoints) does **not**
  list kappa4, although F6 and the `_get_support` defect are exactly that class
  of problem; and #23358 (closed, "ppf of 0 returns a value outside the
  distribution support") concerns `randint` only, not kappa4, despite F5
  returning a support endpoint for small q.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-709.0, -1e6, 0.0)` | 0.0 | 0.99927744565764392 | verified | independent ref (log-space, exact truncation bound) matches the report to all 17 digits; cond_x = 7.09e-4 confirmed. Also `logcdf(-709,-1e6,0)` -inf vs -7.2281551055796427e-4, `cdf(-6.702276267921237e31,-1000,0.1)` 0.0 vs 0.48835740432588314, `cdf(-99.94613906403504,-1e6,-0.01)` 0.0 vs 0.99923382615063237, `logcdf(-1e300,-1,0)` -inf vs -1e300, `pdf(-1e300,-1000,0.1)` 0.0 vs 1.0162486928706959e-305 - all reproduced |
| F1 (regime) | `cdf(-700.0, -1e6, 0.0)` | 0.0 | 0.99928644 | verified, regime understated | mechanism confirmed (at x = -709, h = -1e6 the overflow is in `-h*y`, not in `exp(-x)`); the failure therefore starts at `log|h| + ly > 709.78`, i.e. already at x = -695.6 for h = -1e6, not at the x < -709.78 given in the finding's regime line |
| F2 | `ppf(0.36, -1.0, 0.0)` | nan | -0.57536414490356191 | verified | nan for 17/17 q on an independent grid (q = 1e-300 ... 1 - 1e-15) and for h = -1, -0.5, -1e-3, -1e6; `logistic.ppf(0.36) = -0.5753641449035619` and `kappa4.cdf(-0.5753641449035619, -1, 0) = 0.36` confirm the reference. Stepwise: `q**h = 2.7777777777777777`, `log1p(-2.7778) = nan`, `np.log(-1.0) = nan`, while the unsplit `-log((1-q**h)/h)` gives -0.5753641449035618. Whole branch dead, as claimed |
| F3 | `ppf(0.9999999999999997, 0.1, 0.0)` | inf | 35.638188281008992 | verified | `q**h` rounds to 1.0, `log1p(-1) = -inf`. Other rows reproduced: `(…, -0.1, 0.1)` 10.0 vs 9.7166957116444475 (1.31e14 ulp), `ppf(0.3,1e-8,0.5)` 4.55e7 ulp (cond 4.69), `ppf(0.3678794374926479,1e-8,0.5)` 2.23e13 ulp, `ppf(1e-300,1,-0.5)` 0.0 vs 1e-300, `ppf(1e-8,1,-0.5)` 6.11e7 ulp |
| F4 | `ppf(0.9999999999999999, 0.5, 1e-08)` | 100000000.0 | 36.736793821715347 | verified | 1.23e22 ulp (report 1.2e22); stepwise `q**h -> 1.0`, `1-q**h = 0.0`, result = 1/k. `ppf(0.01,0.5,1e-8)` 2.18e7 ulp (`y**k = 1.0000000058778666`, 8 digits lost), `ppf(1e-300,0.5,1e-8)` 7.36e6, `ppf(0.3678794374926479,0,0.1)` 3.56e8 - all as reported |
| F5 | `ppf(0.1, -1000.0, 0.1)` | -inf | -5.0118723362727586e+100 | verified | `q**h = inf` (exact 1e1000), `(1-inf)/h = inf`, `inf**0.1 = inf` -> -inf; cond_q = 100 confirmed. `ppf(0.9991516571017559,-1e6,-0.01)` -100.0 vs -99.976333723003189, 1.07e12 ulp (`q**h = exp(848) = inf`, `inf**-0.01 = 0`) |
| F6 | `cdf(1e-300, 1.0, -0.5)` | 0.0 | 1e-300 | verified | `1 - h*y` is exactly 0.0 in doubles -> `log1p(-1) = -inf`; cond_x = 1.0 confirmed. `cdf(1e-10,1,-0.5)` 3.73e8 ulp (double `1-h*y = 1.000000082740371e-10` vs exact 9.999999999e-11), `cdf(1e-8,1,-0.5)` 4.36e7, `cdf(-2.3025850929940455,0.1,0)` 0.0 vs 2.2014201533621373e-158. For `pdf(-0.6931471740943004,0.5,1e-8)` = nan the reference is 1.7735819531593608e-8 as reported, but the point is *inside* the true support: the nan arises because the F7 error makes `h*y = 1.00000000113209 > 1`, i.e. F7 feeding F6, not pure cancellation |
| F7 | `cdf(1.0, 0.5, 1e-08)` | 0.6659543826472618 | 0.66595438113876986 | verified | 1.02e7 ulp, cond_x = 0.451. Mechanism isolated: `fl(1 - k*x)` is off by -5.02e-17 relative, `(…)**(1/k)` turns that into -5.02e-9 in y, and re-running SciPy's own expression with an exact y reproduces the reference bit for bit (0.6659543811387698). `1.0/k` is exact (1e8) here, so the loss is the 1/\|k\| amplification (F7), not the rounded exponent (F8). `logcdf(-0.5,0.5,1e-8)` 3.69e7 ulp (cond 1.35), `logcdf(2,0.5,1e-8)` 4.91e6, `cdf(-0.5,0.5,1e-8)` 1.28e8 - as reported |
| `_get_support` defect (outside F1-F8, stated in F7) | `kappa4._get_support(np.array([0.5]), np.array([1e-8]))[0]` | -0.6931471840943004 | -0.69314718296221038 | verified | relative error 1.63e-9 = 7.36e6 ulp, exactly as hand-measured. The error is signed *downwards*, so SciPy's declared support extends 1.13e-9 below the true endpoint and pdf/cdf are indeed evaluated where the density is undefined (`pdf = nan`, `cdf = 0.0` at the returned endpoint). It also propagates to `ppf`: `kappa4.ppf(1e-300, 0.5, 1e-8)` returns exactly the wrong endpoint. Worse for smaller k: h = 3, k = 1e-10 gives 2.15e-8 relative (9.7e7 ulp); harmless for \|k\| >= 0.5 (0.74 ulp at (2, 0.5)) |

Method: reference rebuilt independently from the docstring pdf as
`log F = log1p(-h*exp(ly))/h` with `ly = log1p(-k x)/k`, and
`log f = (1-k)*ly + (1-h)*log F` (a different arrangement from both the
implementation and `refs/kappa4_ref.py`); overflow handled by
`log1p(exp(L)) = L + log1p(exp(-L))` with the term dropped only for L > 1e4,
where it is below 1e-4342 (no saturating exp needed, so x = -1e300 is exact).
Self-checks: pdf agrees with `mp.diff(cdf)` to 60 digits, cdf agrees with
`mp.quad(pdf)` from the support endpoint to 27+ digits, and `cdf(ppf(q)) = q`
to 1e-49 at all reproducer points; every value stable between 100 and 200
digits (one point, `ppf(1e-300, 1, -0.5)`, needed `log1p(-exp(A))` instead of
`log(|expm1(A)|)` to stay stable - fixed before use). Cited code lines read in
the installed 1.18.1 file: `_logcdf.f0` (7453), `_logcdf.f2` (7465),
`_get_support.f0` (7365) and `_pdf` (7400) are exact; the `_logpdf` branches
are at 7413/7421/7427 (cited 7420/7427/7433) and `_ppf.f0`/`f1` at 7485/7488
(cited 7487/7490) - the quoted expressions are right, only the line numbers
drift by 2-6 lines. Scripts: `scratch/kappa4/verify/vref.py`, `compare.py`,
`mech.py`, `extra.py`, `selfcheck*.py`.

Verified by dist-accuracy-verifier: 7 of 7 verified.
