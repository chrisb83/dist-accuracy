# halfgennorm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `halfgennorm_gen`, lines 11623-11697
- **Implemented:** pdf (11672-11676, = exp(logpdf)), logpdf (11678-11679), cdf (11681-11682), sf (11687-11688), ppf (11684-11685), isf (11690-11691)
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** `gennorm` (same formulas on |x|, reports/gennorm.md), `gengamma(a=1/beta, c=beta)` (reports/gengamma.md F1/F2 are the same underflow), `expon` (beta = 1), `halfnorm` (beta = 2, scale 1/sqrt(2)); helpers: none; special functions: `sc.gammaln`, `sc.gammainc`, `sc.gammaincc`, `sc.gammaincinv`, `sc.gammainccinv` (`sc.poch`, `special.gamma` only in `_munp`/`_entropy`, not measured)
- **Reference:** `refs/halfgennorm_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (0.6748054997000371), (0.01), (0.2), (1), (2), (20), (1000), (1e6), (1e300)
- **Worst finding:** loss in cdf, sf, ppf, isf, logpdf

## Summary

All methods are accurate in the bulk for moderate beta (0.2 ... 20).
Everything that forms y = `x**beta` breaks once y underflows, i.e. for
x < exp(-745/beta), although cdf ~ x/Gamma(1 + 1/beta) is a normal double
there: cdf returns 0 and sf returns 1. For beta >~ 1.05 this is the far
left tail (`cdf(1e-300, 2)` = 0 vs 1.1e-300). For large beta it is most of
the support (`cdf(0.4, 1000)` = 0 vs 0.40, `sf(0.99, 1e6)` = 1 vs 0.01).
ppf and isf fail in the same way, because `gammaincinv`/`gammainccinv`
return the underflowed y before `**(1/beta)` (`ppf(0.3, 1000)` = 0 vs
0.2998, `ppf(0.9, 1e300)` = 0 vs 0.9). logpdf loses every digit at huge
beta, because `log(beta) - gammaln(1/beta)` cancels to ~0.577/beta
(`logpdf(0.5, 1e300)` = 0 vs 5.8e-301; 4.7e6 ulp at beta = 1e6). Smaller
(minor) errors come from the `sc.gammainc`/`gammaincc` exp prefactor (up
to 646 ulp in the cdf left tail) and from `exp` of a large logpdf at small
beta (232 ulp).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | x < exp(-708/beta) (y = x**beta subnormal or 0) with cdf normal: left tail for beta >~ 1.05; most of (0, 1) for beta >~ 100 | `x**beta` underflows before `gammainc`/`gammaincc` (patterns 4, 10) | cdf 0.0 for 0.40; sf 1.0 for 0.010 (4.5e17 ulp) | loss |
| F2 | ppf, isf | result x < exp(-708/beta): ppf for q <~ exp(-708/beta) (q < 1.1e-162 at beta = 2, q < 0.475 at beta = 1e3, all q at beta = 1e300); isf for 1 - q in that range (beta >~ 20) | `gammaincinv`/`gammainccinv` return y = x**beta, which underflows, before `**(1/beta)` (patterns 4, 8) | 0.0 for 0.2998; 4.8e11 ulp in the subnormal band | loss |
| F3 | logpdf | large beta, x**beta small (x < 1) | `log(beta) - gammaln(1/beta)` cancels to -gammaln(1 + 1/beta) ~ 0.577/beta (pattern 5) | 0.0 for 5.8e-301 (beta = 1e300); 4.7e6 ulp (beta = 1e6) | loss |
| F4 | cdf, sf, ppf | small y = x**beta (cdf/ppf left tail) or large shape a = 1/beta | `sc.gammainc`/`gammaincc` exp-prefactor rounding (reports/gamma.md F6; pattern 10), passed through `gammaincinv` for ppf | cdf 646 ulp, sf 361 ulp, ppf 151 ulp | minor |
| F5 | pdf | small beta (logpdf ~ -360 at beta = 0.01) | `exp(logpdf)` turns the absolute rounding of `gammaln(1/beta)` into relative error (pattern 3/5) | 232 ulp | minor |

### F1: cdf and sf return 0 and 1 once x**beta underflows

- **Code:** `_cdf`, line 11682: `sc.gammainc(1.0/beta, x**beta)`; `_sf`, line 11688: `sc.gammaincc(1.0/beta, x**beta)`.
- **Mechanism:** patterns 4 and 10. For beta*log(x) < -745, `x**beta` is
  0.0, so `gammainc(a, 0) = 0` and `gammaincc(a, 0) = 1`. The true value is
  P(a, y) ~ y**a / Gamma(1 + a) = x / Gamma(1 + 1/beta). The power
  y**a = x undoes the underflow, so the cdf is a normal double whenever
  x is. In the subnormal band (-745 < beta*log x < -708) y has only a few
  significant bits. Its relative error is divided by beta in y**a, so the
  band is severe for moderate beta (2.5e10 ulp at beta = 2) and minor at
  beta = 1e3 (314 ulp).
- **Regime:** x < exp(-708/beta) wherever the cdf is a normal double.
  - beta = 2: x < 1.6e-154, total loss below 1e-162.
  - beta = 20: x < 4e-16.
  - beta = 1e3: x < 0.49.
  - beta = 1e6: x < 0.99926.
  - beta = 1e300: all of (0, 1).
  - For x >= 2.2e-308 the regime exists only for beta >~ 1.05. For
    beta < 1 the cdf left tail is fine (beta = 0.6748, 0.2, 0.01 show only
    F4).
  - sf is affected only where the cdf is not negligible against 1, i.e.
    for large beta: the sf is returned as exactly 1.0 there. Not all of
    these cases are loss by the tool's definition; `sf(0.4, 1000)` is
    3.0e15 ulp, severe.
- **Reproducer:** `python accuracy.py refs/halfgennorm_ref.py --at cdf 0.4 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-300, 2)` | 0.0 | 1.1283791670955126e-300 | inf | 1 | loss |
  | `cdf(1e-300, 20)` | 0.0 | 1.0272168652716764e-300 | inf | 1 | loss |
  | `cdf(0.4, 1000)` | 0.0 | 0.4002306238979976 | inf | 1 | loss |
  | `cdf(0.99, 1e6)` | 0.0 | 0.9900005714428589 | inf | 1 | loss |
  | `cdf(0.5, 1e300)` | 0.0 | 0.5 | inf | 1 | loss |
  | `sf(0.99, 1e6)` | 1.0 | 0.009999428557141076 | 4.5e+17 | 99 | loss |
  | `sf(0.5, 1e300)` | 1.0 | 0.5 | 4.5e+15 | 1 | loss |
  | `sf(0.4, 1000)` | 1.0 | 0.5997693761020024 | 3.0e+15 | 0.667 | severe |
  | `sf(1e-08, 1000)` | 1.0 | 0.9999999899942345 | 4.5e+07 | 1e-08 | severe |
  | `cdf(1e-160, 2)` | 1.128372886058438e-160 | 1.1283791670955126e-160 | 2.5e+10 | 1 | severe |
  | `cdf(1.3998449603822993e-16, 20)` | 1.43794437275433e-16 | 1.4379443520702595e-16 | 6.5e+07 | 1 | severe |
  | `cdf(0.48186080160065803, 1000)` | 0.4821386232802579 | 0.4821386231415514 | 1.3e+06 | 1 | severe |

  On the grid (beta <= 1e3): cdf has 81 loss, 4 severe and 2 moderate
  points; sf has 7 severe and 1 moderate point (beta = 1e3). This is the
  same defect as gennorm F2 and gengamma F1.
- **Fix direction:** when beta*log(x) < -700, use the series
  P(a, y) = exp(a*beta*log(x) - gammaln(1 + a)) * (1 - a*y/(a + 1) + ...)
  with log y = beta*log x, so y is never formed, and use sf = 1 - cdf
  there. Equivalently, use a log-space lower incomplete gamma.

### F2: ppf and isf return 0 once x**beta underflows

- **Code:** `_ppf`, line 11685: `sc.gammaincinv(1.0/beta, x)**(1.0/beta)`; `_isf`, line 11691: `sc.gammainccinv(1.0/beta, x)**(1.0/beta)`.
- **Mechanism:** patterns 4 and 8. The inverse returns
  y = x**beta ~ (q*Gamma(1 + a))**beta, which underflows to 0 (or to a
  subnormal) whenever the quantile x < exp(-745/beta) (or < exp(-708/beta)).
  `0**(1/beta)` is 0. A subnormal y keeps only a few bits, and
  `**(1/beta)` carries that relative error divided by beta. The special
  functions are correct here, because the true y is not a normal double;
  the formula needs y. For moderate beta, `isf` is not affected
  (1 - q >= 1.1e-16 keeps x above the band for beta <~ 20).
- **Regime:** ppf for q <~ exp(-708/beta)/Gamma(1 + 1/beta):
  - beta = 2: q < 1.1e-162 (loss), up to ~1.8e-154 (severe band).
  - beta = 20: q < 6e-17 (loss).
  - beta = 1e3: q < 0.475 (loss), q up to 0.49 (severe).
  - beta = 1e6: q < 0.99926.
  - beta = 1e300: every q.

  isf has the mirror image (1 - q in the same ranges), and so is
  reachable only for beta >~ 20.
- **Reproducer:** `python accuracy.py refs/halfgennorm_ref.py --at ppf 0.3 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 2)` | 0.0 | 8.86226925452758e-301 | inf | 1 | loss |
  | `ppf(1e-17, 20)` | 0.0 | 9.735042655627758e-18 | inf | 1 | loss |
  | `ppf(0.3, 1000)` | 0.0 | 0.2998271317453786 | inf | 1 | loss |
  | `isf(0.7, 1000)` | 0.0 | 0.29982713174537867 | inf | 2.33 | loss |
  | `ppf(0.7, 1e6)` | 0.0 | 0.6999995959497268 | inf | 1 | loss |
  | `isf(0.3, 1e6)` | 0.0 | 0.6999995959497269 | inf | 0.429 | loss |
  | `ppf(0.9, 1e300)` | 0.0 | 0.9 | inf | 1 | loss |
  | `ppf(1e-160, 2)` | 8.863206964116216e-161 | 8.86226925452758e-161 | 4.8e+11 | 1 | severe |
  | `ppf(0.48, 1000)` | 0.4797234079228047 | 0.47972341079260583 | 2.7e+07 | 1 | severe |

  Outside the band: `ppf(0.9, 1000)` 1.11 ulp, `isf(1e-10, 1000)` 0 ulp.
- **Fix direction:** for small y the leading term is
  y ~ (q*Gamma(1 + a))**(1/a), so x = y**a ~ q*Gamma(1 + 1/beta)
  (times a series correction in y). More generally, invert in
  log y = beta*log x, so y is never formed. As beta -> inf the quantile
  tends to q (uniform on (0, 1)).

### F3: logpdf cancels log(beta) against gammaln(1/beta) for large beta

- **Code:** `_logpdf`, line 11679: `np.log(beta) - sc.gammaln(1.0/beta) - x**beta`.
- **Mechanism:** pattern 5. log(beta) - gammaln(1/beta) = -gammaln(1 + 1/beta)
  ~ 0.5772/beta tends to 0 as beta grows, while both terms are
  ~log(beta). The rounding of either term (~log(beta)*eps absolute) is
  then ~log(beta)*beta/0.577 ulp of the result where x**beta is small.
  At beta = 1e300 both terms round to the same double and the result is
  0.0. Unlike gennorm F3 (where log 2 remains), the half-line constant
  vanishes, so the loss is total. pdf = exp(logpdf) only sees the absolute
  error and stays accurate (`pdf(0.5, 1e6)` 2 ulp, `pdf(0.5, 1e300)` 0 ulp).
- **Regime:** beta >~ 100 and x < 1 (x**beta << 0.577/beta).
  - beta = 100: 241 ulp.
  - beta = 1e3: 481-520 ulp (minor).
  - beta = 1e6: 4.7e6 ulp (severe).
  - beta = 1e300: 0.0 (loss).
- **Reproducer:** `python accuracy.py refs/halfgennorm_ref.py --at logpdf 0.5 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.5, 1e300)` | 0.0 | 5.772156649015328e-301 | inf | 0 | loss |
  | `logpdf(0.5, 1e6)` | 5.772148430338575e-07 | 5.772148424349001e-07 | 4.7e+06 | 0 | severe |
  | `logpdf(0.99, 1000)` | 0.0005332223508727733 | 0.0005332223508727117 | 520 | 81 | minor |
  | `logpdf(0.5, 1000)` | 0.0005763935982834312 | 0.0005763935982833696 | 481 | 1.62e-295 | minor |
  | `logpdf(0.5, 100)` | 0.00569030794606995 | 0.005690307946069646 | 241 | 1.39e-26 | minor |

  (cond (shapes) is 1 at `logpdf(0.5, 1e6)`.)
- **Fix direction:** `-scipy.special._ufuncs._lgam1p(1.0/beta) - x**beta`.
  `_lgam1p(1e-6)` gives 5.772148424349001e-07 exactly, so this is the
  correctly rounded constant.

### F4: cdf, sf and ppf inherit the incomplete-gamma prefactor error

- **Code:** lines 11682, 11688, 11685 (`sc.gammainc`, `sc.gammaincc`, `sc.gammaincinv`).
- **Mechanism:** pattern 10, shared code. cephes `igam`/`igamc` multiply
  the series by `exp(a*log(y) - y - lgam(a))`. The absolute rounding of
  that exponent becomes relative error. It is large when |a*log y| is
  large: in the left tail, a*log y = log x ~ -690 at x = 1e-300; for
  small beta (a = 100), the terms are ~350. This is reports/gamma.md F6.
  For beta = 2 the related known `sc.gammaincc(0.5, x)` 42-118 ulp error
  for x in [0.5, 1] (special-functions.md) shows up directly:
  `sf(1, 2)` = `gammaincc(0.5, 1)` is 117 ulp off. ppf in the far left
  tail inverts the same prefactor.
- **Regime:** cdf left tail for beta <~ 1 (no underflow there, so F1
  does not apply); cdf/sf for small beta at any x; sf for beta = 2 near
  x = 1.
- **Reproducer:** `python accuracy.py refs/halfgennorm_ref.py --at cdf 1e-240 0.6748054997000371`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-240, 0.674805)` | 7.618221207894872e-241 | 7.618221207895965e-241 | 646 | 1 | minor |
  | `cdf(1e-240, 0.2)` | 8.333333333332517e-243 | 8.333333333333337e-243 | 443 | 1 | minor |
  | `cdf(1e-280, 1)` | 9.999999999999464e-281 | 1e-280 | 241 | 1 | minor |
  | `sf(1e+220, 0.01)` | 2.590827660224434e-07 | 2.5908276602242265e-07 | 361 | 0.61 | minor |
  | `sf(1.0, 2)` | 0.15729920705028105 | 0.15729920705028513 | 117 | 2.64 | minor |
  | `ppf(1e-300, 0.674805)` | 1.3126423776767059e-300 | 1.3126423776767498e-300 | 151 | 1 | minor |

- **Fix direction:** in `scipy.special` (accurate prefactor, e.g. Boost's
  `regularised_gamma_prefix` / `log1pmx` form). For the left tail here,
  the leading term `exp(log(x) - gammaln(1 + a))` gives x/Gamma(1 + a)
  directly.

### F5: pdf exponentiates a large logpdf for small beta

- **Code:** `_pdf`, line 11676: `np.exp(self._logpdf(x, beta))`.
- **Mechanism:** for beta = 0.01, `-gammaln(100)` = -359. Its absolute
  rounding error (~|logpdf|*eps/2) becomes the relative error of `exp`.
  This is the same mechanism as gennorm F4.
- **Regime:** small beta, error ~|logpdf| ulp. On the grid it reaches
  232 ulp at beta = 0.01 and stays below 60 ulp beyond conditioning at
  beta = 0.2 and 0.6748.
- **Reproducer:** `python accuracy.py refs/halfgennorm_ref.py --at pdf 1e230 0.01`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e+230, 0.01)` | 2.381524267205928e-245 | 2.3815242672060507e-245 | 232 | 2 | minor |
  | `pdf(1, 0.01)` | 3.9418660600505593e-159 | 3.941866060050517e-159 | 48.5 | 0.01 | minor |

  (cond (shapes) is 596 and 461 at these points.)
- **Fix direction:** inherent to `exp` of a large argument unless the
  constant is carried in extended precision; low priority.

## RuntimeWarnings

- `underflow encountered in power` from `x**beta`:
  - In cdf/sf, it coincides with F1.
  - In logpdf/pdf for large beta it is spurious: `exp(-0)` is the right
    limit, and `pdf(0.5, 1e6)` is 2 ulp off while warning. logpdf is
    wrong there because of F3, not because of the underflow.
- `overflow encountered in power`: for huge x, x**beta = inf, which gives
  logpdf = -inf, cdf = 1 and sf = 0. All are correct, because the true
  values are below double range, so the warning is harmless.
- `underflow encountered in exp` in pdf where the true pdf is subnormal or
  0. Harmless.
- ppf/isf emit no warning in the F2 regime: the 0 comes silently out of
  `gammaincinv`.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | grid, beta = 0.01 ... 20, all x | 76.5 ulp at beta = 20 (cond 2.7, minor; F3 mechanism at smaller scale); 8.8 ulp at beta = 2 (cond 8.2) |
| logpdf | near its zero (x**beta = -gammaln(1 + 1/beta)), beta = 2 | `logpdf(0.34757, 2)` 1.79e4 ulp vs cond (arg) 1.07e4, flagged moderate by the tool. The rounding of the constant `log(2) - gammaln(0.5)` (0.12) relative to a result of 2.3e-5 accounts for it; the F3 fix removes the excess |
| pdf | grid, beta = 0.2 ... 1e3 | 87.6 ulp (beta = 1e3, cond 1.3e5) |
| cdf | right tail, all beta | 0.5 ulp |
| sf | right tail, beta = 0.6748, 0.2, 1, 2, 20 | 107 ulp at cond 90 (beta = 0.6748, x = 1424); otherwise within conditioning |
| ppf | beta = 0.6748, 0.2, 2, 1e3 bulk and right tail (`ppf(0.5, 0.6748)` 4.8 ulp, `ppf(0.999, 0.6748)` 0 ulp, `ppf(0.9, 1000)` 1.1 ulp) | 8.3 ulp |
| ppf, isf | beta = 0.01 (`ppf(0.9, 0.01)` 75.3 ulp at cond 49; `ppf(0.5, 0.01)` 41.7 ulp at cond 12.6; `isf(1e-100, 0.01)` 37.5 ulp at cond 0.26) | 75.3 ulp (minor; the 1/beta = 100 amplification of the `gammaincinv` rounding, pattern 6, reports/gamma.md F5) |
| isf | far tail (`isf(1e-300, 0.6748)` 1.5 ulp, `isf(1e-300, 2)` 0.6 ulp, `isf(1e-10, 1000)` 0 ulp, `isf(1 - 2.2e-16, 20)` 6.3e3 ulp within cond 4.5e15) | within conditioning |

The x-method grid (pdf, logpdf, cdf, sf; beta = 0.6748 ... 1e3) ran
completely (`scratch/halfgennorm/run4_x.txt`). The ppf/isf grid was
started (`scratch/halfgennorm/run2.txt`). Its condition-number phase
(854 points) did not finish within the time limit, so ppf/isf grid maxima
are not measured (time limit). The ppf/isf rows above come from
single-point `--at` evaluations. At beta = 1e6 and 1e300 the x-method grid
did complete (`scratch/halfgennorm/run3.txt`). It confirms F1 (cdf: 82
loss points) and F3 (logpdf: 41 loss, 44 severe, median 4.7e6 ulp); pdf
stays within 86.9 ulp (within conditioning). ppf/isf at these two values
were measured with `--at` points only (time limit: the quantile
references fall back to slow bisection when SciPy's seed is 0).

## Suspected, not confirmed

- The `sc.gammainc` unconverged-series defect (INDEX item 7, reports/gamma.md
  F1) needs a = 1/beta >~ 2e5 and y = x**beta near a. That would require
  x ~ a**a, which is inf, so it cannot be reached.
- isf for small beta (pattern 6, 1/beta amplification of `gammainccinv`):
  seen as 37.5 ulp at beta = 0.01. No larger error found; not mapped on a
  grid (time limit).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | left tail: `log(x) - gammaln(1 + 1/beta) + log1p(series)` (log of the leading term x/Gamma(1+a)); otherwise a log-space lower incomplete gamma; also inherits F1 | cdf underflows for small beta: `logcdf(1e-200, 0.01)` = -inf, true -824.26629513966596; and F1 makes it -inf wherever x**beta underflows |
| logsf | log(sf) | `(a - 1)*beta*log(x) - x**beta - gammaln(a) + log(continued fraction)` (log-space upper incomplete gamma, a = 1/beta) | sf underflows for x**beta >~ 745: `logsf(30, 2)` = -inf, true -903.97411711064388 (`logsf(1000, 0.6748)` is still fine: -103.41097493945489 vs -103.41097493945487) |

## Shared code

- `scipy.special.gammainc` / `gammaincc` exp-prefactor rounding (reports/gamma.md
  F6) gives F4 (minor, up to 646 ulp). The known `sc.gammaincc(0.5, x)`
  42-118 ulp error (special-functions.md) appears as `sf(1, 2)` 117 ulp.
- F1/F2 are halfgennorm's own formulation, not special-function errors:
  it forms the underflowing y = x**beta. The same defect exists in
  `gennorm` (F1/F2) and `gengamma` (F1/F2), which use the same `x**c` /
  `gammaincinv(...)**(1/c)` construction. A shared log-space
  incomplete-gamma helper would fix all three.
- F3: `gennorm` has the same constant (`log(0.5*beta) - gammaln(1/beta)`,
  its F3). There the loss is bounded by log 2; here it is total.

## Existing GitHub issues

- #17832 (open): Tracking issue: custom methods for continuous distributions - relates to the inherited logcdf/logsf (Opportunities)
- F1-F5: none found

(web search; gh unavailable)

## Verification

Verifier route: cdf = x*exp(-y)*1F1(1; 1 + a; y)/Gamma(1 + a) (Kummer form, y = x**beta, a = 1/beta), cross-checked against `mp.quad` of the docstring pdf (agreement 1e-51 or better for beta <= 1e3); sf = 1 - cdf; ppf/isf by root finding/bisection on that cdf with cdf(x_ref) == q checked; logpdf directly from the docstring. All values agree to 30 digits at 100 and 200 digits, except `logpdf(0.5, 1e300)`, which needs > 300 digits (computed at 400/600 digits and by the series -lgamma(1 + a) = euler*a - sum zeta(k)(-a)**k/k). Scripts and output: `scratch/halfgennorm/verify/`. Mechanisms checked against lines 11679, 11682, 11685, 11688, 11691: `gammainc(1e-3, 0.4**1000)` = 0, `gammaincinv(1e-3, 0.3)` = 0, `log(1e6) - gammaln(1e-6)` = 5.772148430338575e-07 (the SciPy value).

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e-300, 2)` | 0.0 | 1.1283791670955126e-300 | verified | loss |
| F1 | `cdf(1e-300, 20)` | 0.0 | 1.0272168652716764e-300 | verified | loss |
| F1 | `cdf(0.4, 1000)` | 0.0 | 0.4002306238979976 | verified | loss; quad agrees |
| F1 | `cdf(0.99, 1e6)` | 0.0 | 0.9900005714428589 | verified | loss |
| F1 | `cdf(0.5, 1e300)` | 0.0 | 0.5 | verified | loss |
| F1 | `sf(0.99, 1e6)` | 1.0 | 0.009999428557141076 | verified | 4.46e17 ulp |
| F1 | `sf(0.5, 1e300)` | 1.0 | 0.5 | verified | 4.5e15 ulp |
| F1 | `sf(0.4, 1000)` | 1.0 | 0.5997693761020024 | verified | 3.0e15 ulp, severe |
| F1 | `sf(1e-08, 1000)` | 1.0 | 0.9999999899942345 | verified | 4.5e7 ulp, severe |
| F1 | `cdf(1e-160, 2)` | 1.128372886058438e-160 | 1.1283791670955126e-160 | verified | 2.5e10 ulp |
| F1 | `cdf(1.3998449603822993e-16, 20)` | 1.43794437275433e-16 | 1.4379443520702595e-16 | verified | 6.5e7 ulp |
| F1 | `cdf(0.48186080160065803, 1000)` | 0.4821386232802579 | 0.4821386231415514 | verified | 1.3e6 ulp |
| F2 | `ppf(1e-300, 2)` | 0.0 | 8.86226925452758e-301 | verified | loss |
| F2 | `ppf(1e-17, 20)` | 0.0 | 9.735042655627758e-18 | verified | loss |
| F2 | `ppf(0.3, 1000)` | 0.0 | 0.2998271317453786 | verified | loss |
| F2 | `isf(0.7, 1000)` | 0.0 | 0.29982713174537867 | verified | loss |
| F2 | `ppf(0.7, 1e6)` | 0.0 | 0.6999995959497268 | verified | loss |
| F2 | `isf(0.3, 1e6)` | 0.0 | 0.6999995959497269 | verified | loss |
| F2 | `ppf(0.9, 1e300)` | 0.0 | 0.9 | verified | loss (bisection) |
| F2 | `ppf(1e-160, 2)` | 8.863206964116216e-161 | 8.86226925452758e-161 | verified | 4.8e11 ulp |
| F2 | `ppf(0.48, 1000)` | 0.4797234079228047 | 0.47972341079260583 | verified | 2.7e7 ulp |
| F3 | `logpdf(0.5, 1e300)` | 0.0 | 5.7721566490153283e-301 | verified | loss; needs > 300 digits |
| F3 | `logpdf(0.5, 1e6)` | 5.772148430338575e-07 | 5.772148424349001e-07 | verified | 4.67e6 ulp |
| F3 | `logpdf(0.99, 1000)` | 0.0005332223508727733 | 0.0005332223508727117 | verified | 520 ulp |
| F3 | `logpdf(0.5, 1000)` | 0.0005763935982834312 | 0.0005763935982833696 | verified | 481 ulp |
| F3 | `logpdf(0.5, 100)` | 0.00569030794606995 | 0.005690307946069646 | verified | 241 ulp |

No moderate findings in the report (F4, F5 are minor; not verified).

Verified by dist-accuracy-verifier: 3 of 3 verified.
