# exponweib: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `exponweib_gen`, lines 2283-2352
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf = generic (log(cdf) below the median, log1p(-sf) above), logsf = generic (log(sf) above the median, log1p(-cdf) below)
- **Related:** none (weibull_min is the a = 1 special case but has its own code); helpers: `_pow1pm1` (lines 2264-2280); special functions: sc.expm1, sc.log1p, sc.xlogy (sc.xlog1py via `_pow1pm1`)
- **Reference:** `refs/exponweib_ref.py`, 400 digits; self-checks: FAIL only in the harness's pdf-integral check for (0.01, 1) (see note below); all other checks PASS
- **Parameter sets:** (2.8923945291034436, 1.9505288745913174), (1, 1), (0.01, 1), (0.3, 50), (1e6, 1), (1000, 0.5), (5, 1000), (2, 0.01)
- **Worst finding:** loss in sf, isf, ppf, pdf, logpdf, cdf

Self-check note: the only remaining FAIL is "integral of pdf != cdf difference"
on [1e-100, 7.9e-31] and [7.9e-31, 2.7e-5] for (a, c) = (0.01, 1). The pdf
there behaves like a*x**(a-1) = 0.01*x**-0.99 (integrable singularity across
70 decades), and the harness integrates with `mp.quad` in the linear variable.
Integrating the same reference pdf in log(x) gives 0.400000000600733168221021147825
and 0.400000015782362689733765109156, equal to the cdf differences to all 30
digits checked, so the reference is correct and the FAIL is a quadrature
artifact. The WARN/NOTE lines of the self-checks (sf, logcdf, logsf at
(0.01, 1)) are the SciPy problem F1.

## Summary

For moderate shapes (a, c of order 1) all six implemented methods are within
a few ulp in the bulk and in most of the tails. Three formulas break down
when `x**c` is small, and in that regime the results are completely wrong.
sf computes `1 - exp(-x**c)` as `log1p(-exp(-x**c))`, and isf computes
`-log(1 - (1-p)**(1/a))` with a plain `log`. Both cancel, so for small a
they return 1.0 and 0.0 where the true values are of order 0.5 (F1, F2). pdf,
logpdf and cdf return 0, inf or -inf once `x**c` underflows (F5). ppf
cancels in `1 - q**(1/a)` for large a or q near 1, and returns inf there (F3).
It returns 0 once `q**(1/a)` underflows for small a (F4). Large a costs up to
about a/4 ulp in pdf, logpdf and cdf (F6), and sf gets a subnormal
`exp(-x**c)` intermediate (F7).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | sf | a < 1 and small x**c (error ~ a*(x**c)**(a-1) ulp) | 1 - exp(-x**c) cancels inside log1p (pattern 1/2) | 1.0 for 0.4988 | loss |
| F2 | isf | (1-p)**(1/a) < ~1e-3, i.e. x**c small (small a, or p near 1) | -log(1 - u) with u small, plain log (pattern 2) | 0.0 for 7.9e-31 | loss |
| F3 | ppf | large a (any q), or q near 1 | 1 - q**(1/a) cancels in log1p (pattern 8) | inf for 37.58; 9.2e5 ulp in the bulk | loss |
| F4 | ppf | small a, small q: q**(1/a) < 2.2e-308 | q**(1/a) underflows (pattern 3/8) | 0.0 for 2.2e-7 | loss |
| F5 | pdf, logpdf, cdf | c*log(x) < -708 (x**c underflows) | x**c underflows to 0, xlogy(a-1, 0) = -+inf (pattern 4) | -inf, +inf, 0.0 | loss |
| F6 | pdf, logpdf, cdf | large a, x**c > ~1 | log/power of -expm1(-x**c) close to 1, error times a (pattern 2/6) | 2.5e5 ulp | moderate |
| F7 | sf | large a, x**c in [708, 745] | exp(-x**c) subnormal, then lifted by a (pattern 4) | 1.28e4 ulp | moderate |
| F8 | pdf | |logpdf| of several hundred | pdf = exp(logpdf) amplifies the absolute error of logpdf | 448 ulp | minor |

### F1: sf returns 1 when x**c is small (small a)

- **Code:** `_sf`, line 2349: `-_pow1pm1(-np.exp(-x**c), a)` = `-expm1(xlog1py(a, -exp(-x**c)))` (helper line 2280).
- **Mechanism:** `log1p(-exp(-t))` with t = x**c is log(1 - exp(-t)).
  `exp(-t)` is rounded to a double close to 1, and the subtraction inside
  `log1p` keeps only the digits of `exp(-t)` beyond its leading 1s
  (pattern 1/2). The relative error of `1 - exp(-t)` is about eps/t, which
  becomes an absolute error a*eps/t in `a*log(...)` and an error of about
  cdf*a*eps/t in sf. For t < eps/2, `exp(-t)` is exactly 1.0, `log1p(-1)` is
  -inf and sf = 1.0. The error is visible when cdf = (1 - exp(-t))**a ~ t**a
  is not negligible against 1. For small a that holds for every small x, so
  sf is wrong across the whole lower half of the distribution.
- **Regime:** relative error about a*t**(a-1)*eps (absolute). sf is exactly 1.0
  for x**c < 1.1e-16. For a = 0.01 this covers all x < 1e-16, where the true
  sf runs from 0.32 to 0.999. For a = 0.3, c = 50 it starts at x = 0.46
  (true sf 1 - 8e-6).
- **Reproducer:** `python accuracy.py refs/exponweib_ref.py --at sf 1e-30 0.01 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1e-30, 0.01, 1)` | 1.0 | 0.4988127663727277 | 4.5e+15 | 0.01 | loss |
  | `sf(9.99999999999999e-18, 0.01, 1)` | 1.0 | 0.32391702460801824 | 9.4e+15 | 0.0209 | loss |
  | `sf(1e-12, 0.01, 1)` | 0.2414225927830619 | 0.24142242497082003 | 3.1e+09 | 0.0314 | severe |
  | `sf(0.457088189614875, 0.3, 50)` | 1.0 | 0.9999920567176528 | 3.6e+10 | 0.000119 | severe |
  | `sf(1e-300, 0.01, 1)` | 1.0 | 0.999 | 4.5e+12 | 1e-05 | severe |

- **Fix direction:** compute log(1 - exp(-t)) as `log(-expm1(-t))` for t < ln 2
  (and `log1p(-exp(-t))` above), i.e. `sf = -expm1(a * log1mexp(-x**c))`.
  The inherited logsf and logcdf (x > median) also show this error (see Opportunities).

### F2: isf returns 0 when (1-p)**(1/a) is small

- **Code:** `_isf`, line 2352: `(-np.log(-_pow1pm1(-p, 1/a)))**(1/c)`.
- **Mechanism:** `u = 1 + _pow1pm1(-p, 1/a) = (1-p)**(1/a)` is computed
  accurately as `expm1(log1p(-p)/a)`, but its negation `1 - u` is formed
  inside, and then `-log(1 - u)` is evaluated with a plain `log` (pattern 2).
  When u is small, `1 - u` rounds and `-log` (about u) has relative error eps/u.
  For u < eps/2, `1 - u` is 1.0 and the result is exactly 0. Since 1 - p = cdf =
  (1 - exp(-x**c))**a, u = 1 - exp(-x**c) ~ x**c, so u is small exactly in
  the left part of the distribution. That covers most p for small a, and p near 1 for any a.
- **Regime:** relative error about eps/(x**c). The result is 0 for x**c < 1.1e-16,
  which for a = 0.01, c = 1 means every p > 0.31. For a = 0.3, c = 50 it means
  every p > 1 - 7e-6.
- **Reproducer:** `python accuracy.py refs/exponweib_ref.py --at isf 0.5 0.01 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.5, 0.01, 1)` | 0.0 | 7.888609052210129e-31 | inf | 100 | loss |
  | `isf(0.9, 0.01, 1)` | 0.0 | 9.999999999999826e-101 | inf | 900 | loss |
  | `isf(0.999999938945977, 0.3, 50)` | 0.0 | 0.3304057670155899 | inf | 1.1e+06 | loss |
  | `isf(0.29, 0.01, 1)` | 1.3322676295501888e-15 | 1.3360874108179045e-15 | 1.3e+13 | 40.8 | severe |
  | `isf(0.9999218629262349, 0.3, 50)` | 0.5323589515299477 | 0.5323416865561277 | 1.5e+11 | 853 | severe |

- **Fix direction:** `isf = (-log1mexp(log1p(-p)/a))**(1/c)` with
  `log1mexp(z) = log(-expm1(z))` for z > -ln 2 and `log1p(-exp(z))` below,
  i.e. `-log1p(-u)` instead of `-log(1 - u)` when u is small.

### F3: ppf cancels in 1 - q**(1/a) for large a and q near 1

- **Code:** `_ppf`, line 2346: `(-sc.log1p(-q**(1.0/a)))**np.asarray(1.0/c)`.
- **Mechanism:** `q**(1/a)` is rounded to a double close to 1 when a is large
  or q is close to 1. `log1p(-q**(1/a))` then sees only the digits of
  `q**(1/a)` beyond its leading 1s (pattern 8), so the relative error of
  1 - q**(1/a) is about eps/(1 - q**(1/a)) ~ eps*a/|log q|. When
  `q**(1/a)` rounds to 1.0, `log1p(-1) = -inf` and ppf = inf. The rounding of
  the `pow` is extra error on top of the input rounding of q, so the error
  exceeds the conditioning by roughly a factor a.
- **Regime:** large a at every q (a = 1e6: 1e3 to 9e5 ulp across the bulk, and
  14 to 370 ulp in the low tail down to q = 1e-300). Any a when 1 - q is small:
  ppf = inf once 1 - q < ~a*eps/2 (a = 1e6: q > 1 - 5e-11; a = 1000: q > 1 - 4e-14).
- **Reproducer:** `python accuracy.py refs/exponweib_ref.py --at ppf 0.99 1e6 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999522942, 1e6, 1)` | inf | 37.58147786671787 | inf | 5.6e+08 | loss |
  | `ppf(0.9999999999999627, 1000, 0.5)` | inf | 1430.9155716763564 | inf | 1.4e+12 | loss |
  | `ppf(0.9999999999995953, 1000, 0.5)` | 1249.658289201242 | 1256.2377370573925 | 2.4e+13 | 1.4e+11 | severe |
  | `ppf(0.9999999994820525, 1e6, 1)` | 35.127362657243 | 35.1966578598828 | 8.9e+12 | 5.5e+07 | severe |
  | `ppf(0.9999999999956061, 5, 1000)` | 1.0033291357275511 | 1.0033291339010582 | 8.2e+06 | 8.2e+06 | severe |
  | `ppf(0.99, 1e6, 1)` | 18.41565979147438 | 18.41565978976602 | 4.2e+05 | 5.4 | moderate |
  | `ppf(0.5, 1e6, 1)` | 14.182023825183233 | 14.182023825119508 | 2.02e+04 | 0.102 | moderate |
  | `ppf(0.9999999999999997, 2, 0.01)` | 3.2369281880503524e+156 | 1.0669306428797774e+156 | 9.2e+15 | 8.3e+15 | loss |

  (The last row is at the edge of the conditioning: the error is a single
  rounding of `q**(1/2)`, amplified by 1/c = 100.)
- **Fix direction:** `-log1p(-q**(1/a)) = -log(-expm1(log(q)/a))`
  (a `log1mexp` of log(q)/a), choosing `log1p(-exp(...))` when log(q)/a < -ln 2.

### F4: ppf returns 0 when q**(1/a) underflows

- **Code:** `_ppf`, line 2346: `q**(1.0/a)`.
- **Mechanism:** for small a, `q**(1/a)` underflows to 0 (pattern 3/8) while
  ppf = (-log1p(-q**(1/a)))**(1/c) ~ q**(1/(a*c)) is normal when c is large.
- **Regime:** log(q)/a < -745, with q**(1/(a*c)) normal. For a = 0.3, c = 50
  that is q < ~1e-97.
- **Reproducer:** `python accuracy.py refs/exponweib_ref.py --at ppf 1e-100 0.3 50`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-100, 0.3, 50)` | 0.0 | 2.1544346900318824e-07 | inf | 0.0667 | loss |
  | `ppf(1e-300, 0.3, 50)` | 0.0 | 9.999999999999983e-21 | inf | 0.0667 | loss |

- **Fix direction:** for small u = log(q)/a use log space: ppf = exp(log(-log1mexp(u))/c),
  with -log1mexp(u) ~ exp(u) evaluated as `exp(u/c)` when exp(u) underflows.

### F5: pdf, logpdf and cdf fail when x**c underflows

- **Code:** `_logpdf`, lines 2335-2338 (`negxc = -x**c`,
  `exm1c = -sc.expm1(negxc)`, `sc.xlogy(a - 1.0, exm1c)`); `_pdf`, line 2332
  (`exp(_logpdf)`); `_cdf`, lines 2342-2343 (`exm1c**a`).
- **Mechanism:** for c*log(x) < -745, `x**c` underflows to 0 (pattern 4), so
  `exm1c = 0`. `xlogy(a - 1, 0)` is -inf for a > 1, which gives logpdf = -inf
  and pdf = 0. For a < 1 it is +inf, which gives logpdf = +inf and pdf = inf.
  The true logpdf is about log(a*c) + (a*c - 1)*log(x), finite and moderate.
  The cdf becomes 0 where the true value x**(a*c) is still normal (needs
  a*c*|log x| < 708, i.e. small a).
- **Regime:** x < exp(-745/c): x < 1e-166 for c = 1.95, x < 1e-6.5 for c = 50,
  x < 0.47 for c = 1000. logpdf is affected for every a. pdf and cdf are affected
  where x**(a*c) is normal (here a = 0.3, c = 50).
- **Reproducer:** `python accuracy.py refs/exponweib_ref.py --at logpdf 1e-200 2.8923945291034436 1.9505288745913174`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-200, 2.8923945291034436, 1.9505288745913174)` | -inf | -2135.851220501392 | inf | 0.00217 | loss |
  | `logpdf(1e-100, 0.3, 50)` | inf | -3220.9110799905616 | inf | 0.00435 | loss |
  | `logpdf(1e-300, 5, 1000)` | -inf | -3453178.3467699788 | inf | 0.00145 | loss |
  | `pdf(1e-20, 0.3, 50)` | inf | 1.5000000000000372e-279 | inf | 14 | loss |
  | `cdf(1e-20, 0.3, 50)` | 0.0 | 1.0000000000000247e-300 | inf | 15 | loss |

- **Fix direction:** for small t = x**c write log(1 - exp(-t)) as
  `c*log(x) + log(-expm1(-t)/t)`, where the second term is about -t/2 and can
  be evaluated without forming t when it underflows. Then `xlogy(a-1, ...)` becomes
  `(a-1)*c*log(x) + ...`, and the cdf is `exp(a*c*log(x) + ...)`.

### F6: pdf, logpdf and cdf lose about a/4 ulp for large a

- **Code:** `_logpdf`, line 2337: `sc.xlogy(a - 1.0, exm1c)`; `_cdf`, line 2343: `exm1c**a`.
- **Mechanism:** for t = x**c > ln 2, `exm1c = -expm1(-t) = 1 - exp(-t)` is
  close to 1. Its rounding error (up to 1/2 ulp, i.e. 1.1e-16 absolute)
  becomes an absolute error of up to (a-1)*1.1e-16 in `(a-1)*log(exm1c)`
  (pattern 2: log of a value near 1 instead of `log1p(-exp(-t))`), and the
  same factor a appears in `exm1c**a` (pattern 6). pdf = exp(logpdf) and cdf
  have relative error ~a*eps/2. logpdf has that absolute error divided by
  |logpdf|.
- **Regime:** large a with x**c > ln 2, i.e. the bulk of the distribution for
  a = 1e6 (x ~ 7 ... 20); a = 1000 gives up to ~700 ulp (minor).
- **Reproducer:** `python accuracy.py refs/exponweib_ref.py --at cdf 9.5 1e6 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(9.5, 1e6, 1)` | 3.097751863680524e-33 | 3.097751863509832e-33 | 2.5e+05 | 711 | moderate |
  | `pdf(9.5, 1e6, 1)` | 2.3188975290685954e-31 | 2.3188975289408176e-31 | 2.5e+05 | 702 | moderate |
  | `logpdf(9.5, 1e6, 1)` | -70.5390460130696 | -70.5390460131247 | 3.52e+03 | 9.95 | moderate |
  | `cdf(7.5, 1e6, 1)` | 5.395847539166772e-241 | 5.395847538892483e-241 | 2.3e+05 | 4.15e+03 | moderate |
  | `cdf(1.5, 1000, 0.5)` | 8.08407351093009e-152 | 8.084073510931375e-152 | 716 | 255 | minor |

- **Fix direction:** use `log1p(-exp(-t))` for t > ln 2 (a `log1mexp(-t)`
  helper), and `cdf = exp(a*log1mexp(-t))`. The error is then bounded by about
  min(a, 745)*eps relative where the cdf is tiny.

### F7: sf with subnormal exp(-x**c) for large a

- **Code:** `_sf`, line 2349: `np.exp(-x**c)` inside `_pow1pm1`.
- **Mechanism:** for x**c in [708, 745], `exp(-x**c)` is subnormal and has
  fewer than 53 significant bits (pattern 4). sf ~ a*exp(-x**c) is normal
  again for large a.
- **Regime:** large a (a*exp(-x**c) > 2.2e-308) and x**c > 708; a = 1e6 gives
  708 < x**c < ~722.
- **Reproducer:** `python accuracy.py refs/exponweib_ref.py --at sf 720.0000000000001 1e6 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(720.0000000000001, 1e6, 1)` | 2.03223080241836e-307 | 2.0322308024240622e-307 | 1.28e+04 | 720 | moderate |

- **Fix direction:** in the far right tail use log space,
  sf = -expm1(a*log1p(-exp(-t))) ~ exp(log(a) - t + ...) without forming exp(-t) alone.

### F8: pdf = exp(logpdf) amplifies the logpdf rounding error

- **Code:** `_pdf`, line 2332: `np.exp(self._logpdf(x, a, c))`.
- **Mechanism:** logpdf is a sum of terms of size up to |logpdf| (several
  hundred), with absolute rounding error ~|logpdf|*eps. `exp` turns that into
  a relative error of about |logpdf| ulp in pdf, also where the pdf itself
  is well conditioned.
- **Regime:** |logpdf| of 100 to 700: x near 0 with c != 1 or a < 1 (pdf
  huge), or deep tails with pdf near 1e-300.
- **Reproducer:** `python accuracy.py refs/exponweib_ref.py --at pdf 1e-280 0.01 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-280, 0.01, 1)` | 1.5848931924612712e+275 | 1.5848931924611135e+275 | 448 | 0.99 | minor |
  | `pdf(1e-60, 2.8923945291034436, 1.9505288745913174)` | 1.7760990075263464e-278 | 1.7760990075262002e-278 | 371 | 4.64 | minor |

- **Fix direction:** evaluate the pdf as a product of factors (e.g.
  `a*c*exm1c**(a-1)*exp(-t)*x**(c-1)`, with a log-space fallback only where the
  factors overflow or underflow).

## RuntimeWarnings

- `underflow encountered in power` (x**c, q**(1/a)) in pdf, logpdf, cdf, sf,
  ppf, isf: x**c in logpdf/cdf (F5) and q**(1/a) in ppf (F4) come with wrong
  results. In sf and isf the same warning is spurious: when x**c underflows,
  the sf result 1.0 is the correctly rounded value. It is not spurious where
  it coincides with F1.
- `overflow encountered in power` (x**c for large x) in pdf, logpdf, cdf, sf:
  spurious; the results (pdf 0, cdf 1, sf 0, logpdf -inf where the true logpdf
  is below -1.8e308) are correct or outside the double range.
- `underflow encountered in exp` / `expm1` in pdf, sf, isf: accompanies F7 in
  sf. Elsewhere (sf, pdf far right tail, true value subnormal or 0) the
  results are correct.
- `overflow encountered in exp` in pdf: only where the true pdf exceeds
  1.8e308 (x near 0 with a*c < 1); not an accuracy problem.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf, logpdf, cdf, sf, ppf, isf | (1, 1) (exponential), full grid | 0.89 ulp |
| cdf | high tail (x**c large), all parameter sets | 1 ulp |
| logpdf | bulk and high tail, all sets except (1e6, 1) (F6) | 17.4 ulp (bulk), 4.89 ulp (high tail) |
| sf | high tail outside F7, moderate shapes | 220 ulp at cond 1.38e3 (within conditioning); 141 ulp (1000, 0.5) within cond 354 |
| ppf | low tail q down to 1e-300 for a >= 1 and moderate c | within conditioning except F3/F4; up to 370 ulp minor for a = 1e6 (F3 mechanism) |
| isf | small p (right tail) down to 1e-300, all sets | 95.1 ulp for c = 0.01 (minor): the 1/2-ulp rounding of `-log(...)` raised to 1/c = 100, intrinsic to the final power; within conditioning otherwise |
| pdf, cdf | (5, 1000) bulk and tails (huge cond 7e5) | 185 ulp, within conditioning |
| logpdf | x**c subnormal (x**c in 1e-323 ... 1e-308) | no flagged point |

## Suspected, not confirmed

none

## Opportunities: inherited methods

The generic `_logcdf` and `_logsf` of `rv_continuous` in 1.18.1 switch at the
median: `log(cdf)` / `log1p(-sf)` and `log(sf)` / `log1p(-cdf)`.

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) for x < median, log1p(-sf) above | `a * log1mexp(-x**c)`, with `log1mexp(-t) = log(-expm1(-t))` for t < ln 2, `log1p(-exp(-t))` above, and `c*log(x) + log(-expm1(-t)/t)` when x**c underflows | cdf underflows for small x**c and large a: `logcdf(0.01, 1000, 1)` = -inf, true -4610.166019324897. Above the median it inherits F1: `logcdf(7.888609052210129e-31, 0.01, 1)` = -inf, true -0.6931471805599453 |
| logsf | log(sf) for x > median, log1p(-cdf) below | `log1mexp(a * log1mexp(-x**c))` (i.e. `log(-expm1(logcdf))`, switching to `log1p(-exp(logcdf))` when logcdf < -ln 2); for large x**c this is about log(a) - x**c | sf underflows for x**c > ~745: `logsf(800, 2, 1)` = -inf, true -799.3068528194401. Above the median it inherits F1: `logsf(7.888609052210129e-31, 0.01, 1)` = 0.0, true -0.6931471805599453 |

## Shared code

`_pow1pm1` (lines 2264-2280) is accurate for what it computes
((1+x)**y - 1 via `expm1(xlog1py(y, x))`). The F1 error comes from its
argument, `-exp(-x**c)`, which `exponweib._sf` rounds before the call (`1 + x`
cancels inside `xlog1py` when x is near -1). Other users of `_pow1pm1` are
affected only if they also pass arguments near -1. No `scipy.special` function
was found to be inaccurate.

## Existing GitHub issues

- #3508 (closed): exponweib.logpdf fails for valid parameters - historical logpdf failure; possibly the same regime as F5 (x**c underflow -> xlogy(a-1, 0)), which still occurs in 1.18.1
- No issue found for F1-F4 (sf/isf/ppf cancellation and underflow), F6 or F7.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `sf(1e-30, 0.01, 1)`; `sf(1e-12, 0.01, 1)`; `sf(0.457088189614875, 0.3, 50)` | 1.0; 0.2414225927830619; 1.0 | 0.49881276637272772; 0.24142242497082003; 0.99999205671765276 (mp.quad of the pdf in log(x**c) from x up, 100 vs 200 digits agree to 1e-101) | verified | 4.5e15, 3.1e9, 3.6e10 ulp as reported. Mechanism matches line 2349 (exp(-x**c) rounds to a double near 1 before log1p). |
| F2 | `isf(0.5, 0.01, 1)`; `isf(0.9, 0.01, 1)`; `isf(0.29, 0.01, 1)`; `isf(0.999999938945977, 0.3, 50)` | 0.0; 0.0; 1.3322676295501888e-15; 0.0 | 7.8886090522101294e-31; 9.9999999999998259e-101; 1.3360874108179045e-15; 0.33040576701558991 (log1p form, checked by 1 - cdf(x_ref) = p via quadrature) | verified | Line 2352 matches. Small wording point: `1 - u = -_pow1pm1(...)` is computed accurately by expm1. The loss is the plain `log` of that double near 1. |
| F3 | `ppf(0.9999999999522942, 1e6, 1)`; `ppf(0.9999999999999627, 1000, 0.5)`; `ppf(0.99, 1e6, 1)`; `ppf(0.5, 1e6, 1)`; `ppf(0.9999999999999997, 2, 0.01)` | inf; inf; 18.41565979147438; 14.182023825183233; 3.2369281880503524e+156 | 37.581477866717871; 1430.9155716763564; 18.415659789766021; 14.182023825119509; 1.0669306428797774e+156 (checked by sf(x_ref) = 1 - q via quadrature) | verified | Line 2346 matches. The last row (9.2e15 ulp at cond 8.3e15) is about a 1-ulp change in q. Rating it "loss" goes by the scale but is within the conditioning. |
| F4 | `ppf(1e-100, 0.3, 50)`; `ppf(1e-300, 0.3, 50)` | 0.0; 0.0 | 2.1544346900318825e-07; 9.999999999999983e-21 (checked by cdf(x_ref) = q) | verified | Line 2346: `q**(1/a)` underflows. |
| F5 | `logpdf(1e-200, 2.8923945291034436, 1.9505288745913174)`; `logpdf(1e-100, 0.3, 50)`; `pdf(1e-20, 0.3, 50)`; `cdf(1e-20, 0.3, 50)` | -inf; inf; inf; 0.0 | -2135.8512205013921; -3220.9110799905616; 1.5000000000000371e-279; 1.0000000000000247e-300 (cdf from the closed form and exp(a*c*log x) together; the verifier's w = s**a quadrature gave 1.00000000000006e-300, a quadrature inaccuracy that does not affect the verdict) | verified | Lines 2335-2338 and 2342-2343 match (x**c underflows, xlogy(a-1, 0) = -/+inf). |
| F6 | `cdf(9.5, 1e6, 1)`; `pdf(9.5, 1e6, 1)`; `logpdf(9.5, 1e6, 1)` | 3.097751863680524e-33; 2.3188975290685954e-31; -70.5390460130696 | 3.0977518635098316e-33 (exp(a*log1p(-exp(-t)))); 2.3188975289408174e-31; -70.539046013124697 | verified | 2.5e5, 2.5e5, 3.5e3 ulp (moderate). Lines 2337 and 2343 match. |
| F7 | `sf(720.0000000000001, 1e6, 1)` | 2.03223080241836e-307 | 2.032230802424062115e-307 (binomial series sum (-1)^(k+1) C(a,k) e^(-k t) and -expm1(a*log1p(-e^-t)) agree at 100 and 200 digits) | verified | 1.28e4 ulp (moderate). Line 2349 matches (subnormal exp(-x**c)). |

Verified by dist-accuracy-verifier: 7 of 7 verified.
