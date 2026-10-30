# invweibull: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `invweibull_gen`, lines 5659-5723
- **Implemented:** pdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf)
- **Related:** no shared code. Mathematically this is the Frechet / type II extreme value distribution (genextreme with c < 0, and 1/X for X ~ weibull_min). Helpers: none. Special functions: np.power, np.exp, np.expm1, np.log, np.log1p (sc.gamma only in `_munp`)
- **Reference:** `refs/invweibull_ref.py`, 400 digits; self-checks: PASS (run on the first three parameter sets; see "Suspected, not confirmed" for c <~ 0.01)
- **Parameter sets:** (10.58), (0.1), (1), (0.001), (0.003), (0.005), (0.009), (0.01), (0.02), (0.5), (3), (1000), (1e6), (1e10)
- **Worst finding:** loss in pdf

## Summary

sf is accurate everywhere (at most 1.73 ulp). cdf, ppf and isf have only
minor errors beyond conditioning (at most 254 ulp). These come from rounding
of `x**-c`, `-1/c` and `log(q)`, which `exp` and `pow` then amplify. The pdf fails
for small c (roughly c <= 1) in the left tail. There `exp(-x**-c)` is
subnormal or 0 while the factor `x**(-c-1)` is large enough to make the true
pdf a normal double. The errors range from moderate through severe to 0.0.
For c in about [0.002, 0.01], `x**(-c-1)` also overflows near the smallest
normal x. The pdf then returns inf or nan where the true value is finite
(between 1e-210 and 7.6e291). The rounded exponent `-c-1` costs the pdf up
to ~350 ulp for small c. The inherited logpdf, logcdf and logsf fail
completely in the tails, although simple closed forms exist.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf | c <~ 1, t = x**-c > 708 while the pdf is normal | exp(-t) is subnormal or 0, and a large x**(-c-1) scales it back up (pattern 4) | 0.0 for 5.1e-134 | loss |
| F2 | pdf | c in ~[0.002, 0.01], x near the smallest normal double | pow(x, -c-1) overflows (pattern 3) | inf for 7.6e291; nan for 6.6e-203 | loss |
| F3 | pdf | small c, abs(log x) large | rounded exponent -c-1 (pattern 6, rounded shape-derived exponent) | 352 ulp | minor |
| F4 | cdf (and pdf) | c < 1, t = x**-c large | rounding of pow(x, -c) amplified by t in exp(-t) (pattern 6) | 251 ulp | minor |
| F5 | ppf, isf | small q (ppf) or small p (isf); largest for small c | rounded exponent -1/c and rounded log(q), amplified (patterns 6 and 8) | 224 ulp | minor |

### F1: pdf underflows in exp(-x**-c) although the pdf is normal

- **Code:** `_pdf`, lines 5696-5699:
  `xc1 = np.power(x, -c - 1.0)`; `xc2 = np.exp(-np.power(x, -c))`; `return c * xc1 * xc2`.
- **Mechanism:** with t = x**-c the pdf is `c * t**(1 + 1/c) * exp(-t)`.
  For small c, the factor `x**(-c-1) = t**(1 + 1/c)` is huge. For example, c = 0.01 and
  t = 1000 give 1e303. `exp(-t)` is subnormal for 708 < t < 745, with only a
  few significant bits, and it is 0 for t > 745. The true product is still a
  normal double (pattern 4: an underflowed factor multiplied by a large one).
- **Regime:** t > 708 and log(c) + (1 + 1/c) log(t) - t > -708.
  - 708 < t < 745: moderate or severe errors for every c <= 1 where the
    product is normal. For c = 1 this is near x = 1.4e-3.
  - t > 745: the result is exactly 0.0. A normal true value remains here
    only for small c:
    - c = 0.1: t up to ~780.
    - c = 0.02: x from about 1e-150 to 2.5e-144.
    - c = 0.01: t up to ~1400, i.e. x from 6e-288 down to the smallest
      normal double.
- **Reproducer:** `python accuracy.py refs/invweibull_ref.py --at pdf 1e-300 0.01`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-300, 0.01)` | 0.0 | 5.075958897548729e-134 | inf | 8.99 | loss |
  | `pdf(1e-150, 0.02)` | 0.0 | 1.0151917795097457e-283 | inf | 19 | loss |
  | `pdf(1.8986165949114554e-29, 0.1)` | 0.0 | 1.1074649299056353e-293 | inf | 73.4 | loss |
  | `pdf(2.030873172573601e-29, 0.1)` | 1.5302151578492189e-291 | 1.5262733060324326e-291 | 1.2e+13 | 72.9 | severe |
  | `pdf(2.671007058389899e-29, 0.1)` | 5.47810674309209e-283 | 5.478106743107113e-283 | 1.24e+04 | 70.9 | moderate |
  | `pdf(1.9290123456790115e-06, 0.5)` | 3.7926304127052425e-305 | 3.792630412715748e-305 | 1.25e+04 | 359 | moderate |
  | `pdf(0.0013888888888888885, 1)` | 1.0535084479736784e-307 | 1.0535084479765463e-307 | 1.16e+04 | 718 | moderate |

- **Fix direction:** evaluate in log space,
  `exp(log(c) - (c+1)*log(x) - t)` or `exp(log(c) + (1 + 1/c)*log(t) - t)`,
  at least for t > ~700. The same expression gives the missing logpdf.

### F2: pow(x, -c-1) overflows near the smallest normal x

- **Code:** `_pdf`, line 5696: `xc1 = np.power(x, -c - 1.0)`.
- **Mechanism:** `x**(-c-1) = t/x`, and 1/x is already 4.5e307 near
  x = 2.2e-308. Once t = x**-c exceeds about 4, the power overflows to inf
  (pattern 3), while `exp(-t)` is still moderate, so the pdf is
  `c * inf * exp(-t) = inf`. For c >= ~0.0093, `exp(-t)` also underflows and
  the product is `inf * 0 = nan`. In both cases the true pdf is a normal
  double.
- **Regime:** c in roughly [0.002, 0.01] and x below ~1e-306. For c = 0.009
  this holds up to at least x = 1e-306. The true pdf there ranges from 1e-210 to 1e302.
- **Reproducer:** `python accuracy.py refs/invweibull_ref.py --at pdf 2.3e-308 0.005`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(2.3e-308, 0.003)` | inf | 2.52161e+302 | inf | 0.978 | loss |
  | `pdf(2.3e-308, 0.005)` | inf | 7.575742930594174e+291 | inf | 0.832 | loss |
  | `pdf(1e-306, 0.009)` | inf | 1.6855907780068774e+60 | inf | 4.1 | loss |
  | `pdf(1e-307, 0.01)` | nan | 6.583776632124189e-203 | inf | 10.7 | loss |

  Subnormal x fail the same way, e.g. `pdf(2.964e-323, 0.009)` = nan where
  the true value is 1.66e-24.
- **Fix direction:** the log-space form of F1. Alternatively, keep the factors
  in range: `c * t * (exp(-t) / x)`.

### F3: the rounded exponent -c-1 costs up to ~350 ulp in pdf

- **Code:** `_pdf`, line 5696: `np.power(x, -c - 1.0)`.
- **Mechanism:** rounding `-c - 1.0` to a double gives an absolute error of
  up to (1+c)*eps/2. The relative error of the result is that error times
  |log x|, and |log x| reaches ~708. For small c this is much more than a
  1-ulp change of c itself, which would cost only c*|log x| (cond (shapes)).
  The argument condition number is ~1, so the error is beyond conditioning.
- **Regime:** small c (about c <= 0.1) and large |log x| (x near 1e-308 or
  1e300), wherever the pdf is normal.
- **Reproducer:** `python accuracy.py refs/invweibull_ref.py --at pdf 2.3e-308 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(2.3e-308, 0.001)` | 1.1587876422272208e+304 | 1.1587876422273114e+304 | 352 | 0.999 | minor |
  | `pdf(1e+300, 0.005)` | 1.5319211316323152e-304 | 1.5319211316322023e-304 | 332 | 1 | minor |

  cond (shapes) is 0.27 and 2.34, so the error is far larger than a 1-ulp change of c would cause.
- **Fix direction:** form `x**(-c-1)` as `x**-c / x` (t is computed anyway),
  or use the log-space form of F1 with `-c*log(x) - log(x)`.

### F4: rounding of x**-c amplified by exp(-t) (cdf, pdf)

- **Code:** `_cdf`, lines 5702-5703: `np.exp(-np.power(x, -c))`; the same
  expression is in `_pdf`, line 5698.
- **Mechanism:** `pow(x, -c)` has a relative error of ~0.5 ulp, and
  `exp(-t)` turns it into a relative error of about t/2 ulp. The argument
  condition number is only c*t, so for c < 1 the error exceeds conditioning
  by about 1/c. weibull_max has the same effect (its F1).
- **Regime:** c < 1 and large t (up to ~745 for a normal cdf), i.e. the far
  left tail of the cdf.
- **Reproducer:** `python accuracy.py refs/invweibull_ref.py --at cdf 3.54013317464144e-29 0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(3.54013317464144e-29, 0.1)` | 9.85967654375865e-305 | 9.859676543758101e-305 | 251 | 70 | minor |
  | `cdf(1.653817168792027e-28, 0.1)` | 2.6503965530043108e-261 | 2.6503965530044e-261 | 152 | 60 | minor |

  For c = 0.009 to 0.02 the maximum is 254 ulp near t = 700; cond (shapes) there is ~4.6e3.
- **Fix direction:** would need t in extended precision (a double-double
  `pow`). Otherwise accept: the error is bounded by about t/2 <= ~370 ulp.

### F5: ppf and isf amplify the rounding of -1/c and log(q)

- **Code:** `_ppf`, line 5709: `np.power(-np.log(q), -1.0/c)`;
  `_isf`, line 5712: `(-np.log1p(-p))**(-1/c)`.
- **Mechanism:** the result is `exp(y)` with `y = -log(L)/c` and L = -log(q).
  Rounding the exponent `-1/c` adds a relative error of about |y|*eps/2, and
  rounding L adds about (1/c)*eps/2. Near the ends of the double range this
  reaches several hundred ulp. The argument condition number,
  1/(c*|log q|), is much smaller for small q. The error stays below
  cond (shapes), i.e. it is equivalent to a ~1 ulp change of c.
- **Regime:** ppf at small q and isf at small p, where the results are
  near 1e+-300. Largest for small c, but present for any c (isf with c = 3
  at p = 1e-300).
- **Reproducer:** `python accuracy.py refs/invweibull_ref.py --at ppf 0.01 0.003`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.01, 0.003)` | 8.281427652865034e-222 | 8.281427652864623e-222 | 224 | 72.4 | minor |
  | `ppf(5.623413251903491e-226, 0.009)` | 2.221326539925151e-302 | 2.2213265399250686e-302 | 167 | 0.214 | minor |
  | `isf(1.2115276586286942e-26, 0.1)` | 1.4677992676207888e+259 | 1.4677992676207401e+259 | 149 | 10 | minor |
  | `isf(1e-300, 3)` | 9.999999999999872e+99 | 1e+100 | 57.7 | 0.333 | minor |

  cond (shapes) for these rows: 509, 695, 597, 230.
- **Fix direction:** `exp(-log(L)/c)` removes the rounding of 1/c. The rest
  is inherent to computing exp(y) with large |y| in double precision. Low
  priority.

## RuntimeWarnings

- pdf: `overflow encountered in power`, `underflow encountered in exp` and
  `invalid value encountered in multiply` appear in the F1 and F2 regimes,
  where the result is wrong. `underflow encountered in power` appears in
  the right tail where the pdf truly underflows (e.g. `pdf(1e40, 10.58)` =
  0.0, which is accurate).
- cdf: `underflow encountered in exp` where the cdf truly underflows (e.g.
  `cdf(1e-3, 10.58)` = 0.0, accurate); `overflow encountered in power` for tiny x.
- sf: `overflow encountered in power` for tiny x, where the result 1.0 is
  correct (`sf(2.3e-308, 10.58)`). `underflow encountered in power` for huge
  x, where sf truly underflows (`sf(1e300, 10.58)` = 0.0). These warnings
  are spurious; the values are accurate.
- ppf/isf: overflow or underflow in power only where the true quantile lies
  outside the double range. The values are accurate.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| sf | full grid incl. t = x**-c from 1e-300 to 1400, all 14 parameter sets | 1.73 ulp |
| cdf | full grid, all parameter sets (beyond F4) | 11.1 ulp in the bulk. Errors up to 1.6e5 ulp at c = 1e10 are within the argument conditioning c*t |
| pdf | large c (1e3, 1e6, 1e10), incl. x where pow(x, -c-1) is subnormal ((c+1) log x in [700, 745]); the subnormal-intermediate hypothesis | 8.0e8 ulp at c = 1e10 (cond 1.0e10) and 4.88e4 ulp at c = 1e6 (cond 7.2e8): within conditioning |
| pdf | c = 3 and 10.58, all x | 216 ulp, within conditioning (cond 7.5e3) |
| ppf | q near 1 (1 - q down to 3e-16), all c | 90.1 ulp, within conditioning (cond 3.0e16) |
| isf | p near 1, all c; large c | 134 ulp, within conditioning |
| ppf, isf | c = 0.001 | 336 / 420 ulp, within conditioning (660 / 1740) |

## Suspected, not confirmed

- Self-check artifact: the harness check "integral of pdf != cdf difference"
  FAILs when a set with c <= 0.005 is among the first three. For c = 0.005
  the bulk interval is [3.6e-73, 6.8e31], and plain `mp.quad` misses most of
  the mass (relative error -0.64). With the substitution x = exp(u) on 20
  subintervals, the integral matches the cdf difference to 1e-30
  (`scratch/invweibull/notes.md`). The reference is therefore correct, and
  PARAMS are ordered so that the self-checks run on 10.58, 0.1 and 1.
- Out of scope, but observed: the harness does not measure moments, and `_munp`
  returns `sc.gamma(1 - n/c)` without checking n < c. As a result
  `invweibull.moment(2, 1.5)` = -4.062353818279203 and
  `invweibull.var(1.5)` = -11.239065491021297, although both moments are
  infinite. This is a correctness bug, not a rounding issue.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `log(c) - (c+1)*log(x) - exp(-c*log(x))` | pdf underflows for t = x**-c > ~745: `logpdf(1e-5, 1)` = -inf, true -99976.97414907005 |
| logcdf | log(cdf) | `-x**-c` (closed form) | t > ~745: `logcdf(1e-5, 1)` = -inf, true -99999.99999999999 |
| logsf | log(sf) | `log(-expm1(-t))` for moderate t; for small t, `-c*log(x) + log(-expm1(-t)/t)`, which avoids forming t when it underflows | sf underflows when x**-c < 1e-308: `logsf(1e200, 2)` = -inf, true -921.07481992851093 |

## Shared code

none

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(1e-300, 0.01)` | 0.0 | 5.0759588975487288e-134 | verified | loss |
| F1 | `pdf(1e-150, 0.02)` | 0.0 | 1.0151917795097457e-283 | verified | loss |
| F1 | `pdf(1.8986165949114554e-29, 0.1)` | 0.0 | 1.1074649299056353e-293 | verified | loss |
| F1 | `pdf(2.030873172573601e-29, 0.1)` | 1.5302151578492189e-291 | 1.5262733060324327e-291 | verified | 1.16e13 ulp, severe; t = 740, so exp(-t) is subnormal as stated |
| F2 | `pdf(2.3e-308, 0.003)` | inf | 2.5216113200461059e+302 | verified | loss |
| F2 | `pdf(2.3e-308, 0.005)` | inf | 7.5757429305941733e+291 | verified | loss |
| F2 | `pdf(1e-306, 0.009)` | inf | 1.6855907780068773e+60 | verified | loss |
| F2 | `pdf(1e-307, 0.01)` | nan | 6.5837766321241884e-203 | verified | loss; t = 1.2e3, so exp(-t) = 0 and inf*0 = nan as stated |
| F2 | `pdf(2.964e-323, 0.009)` | nan | 1.6569025020115094e-24 | verified | subnormal x, as stated |

References: two independent routes, c*t**(1+1/c)*exp(-t) with t = x**-c and `mp.diff` of the cdf exp(-exp(-c*u)) in u = log x. They agree with each other and between 100 and 200 digits to 30 digits (`scratch/invweibull/verify/verify_pdf.py`). The mechanisms match `_pdf`, lines 5696-5699. Minor findings F3-F5 were not re-checked.

Verified by dist-accuracy-verifier: 2 of 2 verified.
