# weibull_max: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `weibull_max_gen`, lines 3006-3074
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, ppf
- **Inherited (not measured):** logsf = log(sf), isf = ppf(1 - q)
- **Related:** weibull_min (mirror image, x -> -x; separate code but same formulas); helpers: none; special functions: sc.expm1, sc.xlogy (sc.gamma only in `_munp`)
- **Reference:** `refs/weibull_max_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (2.8687961709100187), (0.001), (0.1), (0.5), (1), (30), (1000), (1e6), (1e8)
- **Worst finding:** minor in pdf, cdf, ppf (at most 235 ulp); no loss, severe or moderate case

## Summary

All six implemented methods are accurate over the full grid (|x| from 1e-300
to 1e300, and t = (-x)**c from 1e-300 to 750) for every parameter set.
logpdf, logcdf and sf stay within 4 ulp (except where large c makes them
ill-conditioned). The only errors beyond conditioning are minor and
appear for c < 1. In pdf and cdf, the rounding error of `pow(-x, c)` is
amplified by t through `exp(-t)` (up to 235 ulp). In pdf, the rounded shape
`c-1` is raised to the power log(-x) (up to 86 ulp for |x| -> 0). In ppf,
the rounding error of `log(q)` is amplified by 1/c (up to 19 ulp). Large errors
for large c (up to 1e9 ulp at c = 1e8) are all within the argument
conditioning c*t. The inherited isf and logsf fail completely in their tails
(see Opportunities).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, pdf | c < 1, t = (-x)**c large (x -> -inf) | rounding of pow(-x, c) amplified by t in exp(-t) (pattern 6) | 235 ulp | minor |
| F2 | pdf | c < 1, x -> 0 (and x -> -inf) | rounded exponent c-1 in pow(-x, c-1) (pattern 6/11) | 86 ulp | minor |
| F3 | ppf | c < 1, small q | rounding of log(q) amplified by 1/c (pattern 8) | 19.1 ulp | minor |

### F1: exp(-pow(-x, c)) amplifies the rounding of t by t

- **Code:** `_cdf`, line 3054: `np.exp(-pow(-x, c))`; `_pdf`, line 3048: `c*pow(-x, c-1)*np.exp(-pow(-x, c))`.
- **Mechanism:** t = pow(-x, c) carries a relative rounding error of up to
  0.5 ulp. exp(-t) turns this into a relative error of t*0.5 ulp. The
  conditioning with respect to x is only c*t, so for c < 1 the error exceeds
  the conditioning by a factor of about 1/c (pattern 6 with the "power" being
  the exponential). For c >= 1 the same error is within conditioning.
- **Regime:** c < 1 with t of order 10 or more, up to t ~ 708 where the cdf
  stops being a normal double. The error is bounded by about t/2 ulp
  (≤ ~370 ulp).
- **Reproducer:** `python accuracy.py refs/weibull_max_ref.py --at cdf -2.8247524899999964e+28 0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-2.8247524899999964e+28, 0.1)` | 9.85967654375865e-305 | 9.859676543758136e-305 | 235 | 70 | minor |
  | `cdf(-6.046617599999976e+27, 0.1)` | 2.6503965530043108e-261 | 2.6503965530043848e-261 | 126 | 60 | minor |
  | `pdf(-6.046617599999976e+27, 0.1)` | 2.6299627940794373e-287 | 2.6299627940795155e-287 | 134 | 60.9 | minor |

  (cond (shapes) is 4.59e3 and 3.83e3 here: the results are much more
  sensitive to c than to x.)
- **Fix direction:** only an extra-precision t helps (e.g. t as a
  double-double `exp(c*log(-x))`). Otherwise this is the inherent limit of
  forming t in double precision.

### F2: pdf uses the rounded exponent c-1

- **Code:** `_pdf`, line 3048: `pow(-x, c-1)`.
- **Mechanism:** `c-1` is rounded (for c = 0.1 the double -0.9 differs from
  the exact 0.1 - 1 by 2.8e-17). Raising to this exponent gives a relative
  error of |log(-x)| * 2.8e-17, i.e. about 0.12 ulp per unit of |log(-x)|.
  The x-conditioning is |c-1 - c*t| ≈ 0.9, so the error is beyond
  conditioning. Computing `c*t/(-x)*exp(-t)` with t = pow(-x, c) gives
  9.999999999999964e+268, within 1 ulp, at the first reproducer point.
- **Regime:** c not exactly representable relative to c-1 (any c < 1 that is
  not a power of 2 fraction, e.g. 0.1); |log(-x)| large, i.e. x -> 0 (error
  grows linearly to 86 ulp at x = -1e-300) and x -> -inf (where it adds to F1).
- **Reproducer:** `python accuracy.py refs/weibull_max_ref.py --at pdf -1e-300 0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(-1e-300, 0.1)` | 1.0000000000000153e+269 | 9.999999999999962e+268 | 85.8 | 0.9 | minor |
  | `pdf(-1e-250, 0.1)` | 1.0000000000000127e+224 | 9.999999999999969e+223 | 71.3 | 0.9 | minor |
  | `pdf(-1e-210, 0.1)` | 1.0000000000000107e+188 | 9.999999999999974e+187 | 59.9 | 0.9 | minor |

  (cond (shapes) is 68.1 at x = -1e-300, comparable to the error.)
- **Fix direction:** `c * t / (-x) * exp(-t)` with `t = pow(-x, c)` where t
  neither underflows nor overflows; keep the current form otherwise.

### F3: ppf amplifies the rounding of log(q) by 1/c

- **Code:** `_ppf`, line 3063: `-pow(-np.log(q), 1.0/c)`.
- **Mechanism:** `np.log(q)` is correct to about 0.5 ulp. Raising it to the
  power 1/c multiplies that relative error by 1/c, giving about 1/(2c) ulp,
  while the q-conditioning is 1/(c*|log q|). For |log q| > 1 (q < 0.37) the
  error exceeds the conditioning by a factor of about |log q|.
- **Regime:** c < 1, small q. For c = 0.1 the error is 13 to 19 ulp for
  q < 1e-160. For much smaller c the result overflows before the error
  grows large (c = 1e-3: errors up to 336 ulp but within cond 660).
- **Reproducer:** `python accuracy.py refs/weibull_max_ref.py --at ppf 6.8129206905797616e-251 0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(6.8129206905797616e-251, 0.1)` | -4.022085566321483e+27 | -4.022085566321466e+27 | 19.1 | 0.0174 | minor |
  | `ppf(1e-300, 0.1)` | -2.4738276207484507e+28 | -2.4738276207484424e+28 | 15.2 | 0.0145 | minor |

  (cond (shapes) about 64 here.)
- **Fix direction:** needs `log(q)` in extended precision (double-double);
  there is no cheap reformulation. The error is small in absolute terms.

## RuntimeWarnings

- **pdf, line 3048:** for large c and -x > 1, `pow(-x, c-1)` overflows to
  inf while `exp(-pow(-x, c))` underflows to 0, so the product is
  `inf*0 = nan` ("overflow encountered in power", "invalid value encountered
  in multiply"). The true value is far below the double range, so the
  harness does not measure it, but SciPy returns **nan instead of 0**, e.g.
  `pdf(-2.1, 1000)`, `pdf(-1e30, 30)` and `pdf(-1e300, 30)` all return nan
  (at x = -2, c = 1000 the true value is 1.8e-433991 and SciPy returns 0.0).
  This is a wrong special value, not an accuracy loss on a normal
  double. Computing pdf as `exp(logpdf)` or as `c*t/(-x)*exp(-t)` avoids it.
- **pdf, cdf, logpdf, logcdf, sf, ppf:** "overflow/underflow encountered in
  power" and "underflow encountered in exp/multiply" in the far tails
  (t > 1.8e308 or t < 1e-308). The returned values (0, 1, -inf, or tiny
  values) are correct in these cases, so the warnings are spurious.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | full grid incl. t = 700...750, x -> 0, all c; zero crossing of logpdf | 3.66 ulp beyond cond 0; larger values (67 ulp at c = 1e3, 4.6e7 at c = 1e8) are within cond |
| logcdf | full grid, all parameter sets | 1.49 ulp |
| sf | full grid incl. x -> 0 (sf small via -expm1), all c | 2.24 ulp |
| cdf | c >= 1, all x | within cond (e.g. 331 ulp at c = 1e6, cond 7.1e8) |
| pdf | large c (1e3...1e8) where exp(-t) is subnormal near t ~ 708...730 and c*t lifts the result into the normal range (H1, pattern 4) | 1.0e9 ulp at c = 1e8, within cond 7.3e10 |
| ppf | q near 1, bulk, c >= 0.5 | 2.84 ulp; c = 1e-3 up to 336 ulp, within cond 660 |

## Suspected, not confirmed

- logpdf (line 3051) cancellation of `log(c) + (c-1)*log(-x) - t` near its zero
  crossing (pattern 5): the full grid and c = 1e-3...1e8 show no error above
  3.66 ulp beyond conditioning. Not confirmed.
- pdf subnormal `exp(-t)` (pattern 4) for huge c: errors exist but stay
  within the argument conditioning c*t for every c tested (up to 1e8). For
  c of order 1e16 the grid of representable x near -1 is too coarse to place
  a point where the result is normal.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | log(sf) = log(-expm1(-t)) | `log(-expm1(-t))` for t not tiny; `c*log(-x) + log1p(-t/2)`-type series (or `xlogy(c, -x) + log(-expm1(-t)/t)`) when t underflows | t = (-x)**c < 2.2e-308, i.e. x -> 0: `logsf(-1e-200, 2)` = -inf, reference -921.0748199285109 (loss) |
| isf | ppf(1 - q) | `-pow(-np.log1p(-q), 1.0/c)` | small q (1 - q rounds): `isf(1e-20, 2)` = 0.0, reference -1e-10 (loss); digits are lost in proportion to 1/q for any q < ~1e-3 |

## Shared code

none (only `np.exp`, `pow`, `np.log`, `sc.expm1`, `sc.xlogy`, all used
correctly. weibull_min uses the same formulas mirrored, so F1-F3 and the pdf
nan probably have counterparts there.)

## Existing GitHub issues

none found (web search; `gh` not installed)

## Verification

pending
