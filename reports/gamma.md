# gamma: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `gamma_gen`, lines 3581-3788
- **Implemented:** pdf (3628-3630), logpdf (3632-3633), cdf (3635-3636), sf (3638-3639), ppf (3641-3642), isf (3644-3645)
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** `erlang` (subclasses `gamma_gen`, same methods), `dgamma`; helpers: none used by the measured methods (`_digammainv`, `_get_fixed_fit_value`, `_remove_optimizer_parameters`, `_skew` are fit-only); special functions: `sc.xlogy`, `sc.gammaln`, `sc.gammainc`, `sc.gammaincc`, `sc.gammaincinv`, `sc.gammainccinv`
- **Reference:** `refs/gamma_ref.py`, 400 digits; self-checks: PASS. Auxiliary reference for a = 1e8 (where mpmath's own `gammainc` does not converge): `scratch/gamma/gamma_a1e8_ref.py`, 100 digits, self-checks PASS
- **Parameter sets:** (0.001), (0.1), (1), (1.9932305483800778), (10.5), (1000), (1e6); plus (1e8) with the auxiliary reference
- **Worst finding:** severe in cdf (relative error 0.375, i.e. 1.7e15 ulp, at a = 1e8, x = a - 4.6*sqrt(a)), propagating to sf, ppf and isf; severe in pdf (6.3e6 ulp)

## Summary

In the bulk and for small/moderate shapes gamma is good: cdf and sf are
within a few ulp, ppf and isf within ~100 ulp of their conditioning. Two
mechanisms break it. (1) `sc.gammainc`, used directly by `_cdf`, returns an
**unconverged** value for large shapes: cephes' Kummer series is cut off
after 2000 terms, so for a >~ 2e5 and `4.5*sqrt(a) < a - x <~ 15*sqrt(a)`
the cdf loses digits progressively - 4.3e-6 relative at a = 1e6 and 0.375
relative (only one correct digit) at a = 1e8 - and `sf`, `ppf` and `isf`
inherit it. (2) `_logpdf` adds and subtracts terms of size `a*log(a)`
(`xlogy(a-1, x)`, `x`, `gammaln(a)`) to produce an O(1) result, and `_pdf`
exponentiates it, so the pdf's relative error is about
`max(|(a-1)log x|, x, gammaln(a))` ulp: 500 ulp in the far tails for
moderate a, 5.5e3 ulp at a = 1e3 and 6.3e6 ulp at a = 1e6. In addition
`sc.gammainc`/`sc.gammaincc` lose `|a log x - x - gammaln(a)|` ulp wherever
their `exp(a*log(x) - x - lgam(a))` prefactor branch is used (cdf 1.6e3 ulp
at a = 1e3, x = 500; sf 2.0e3 ulp at a = 1e3, x = 2000), and
`sc.gammaincinv`/`sc.gammainccinv` lose ~1/a ulp-amplification for a << 1
(3.6e3 ulp at a = 1e-3). Nothing returns 0/inf/nan in the measured range
(the inherited logcdf/logsf do - see Opportunities).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | a >~ 2e5 and 4.5*sqrt(a) < a - x <~ 15*sqrt(a) | `sc.gammainc` returns the 2000-term partial sum of a series that needs 1e4-3e5 terms (pattern 10) | cdf 1.7e15 ulp (rel. 0.375) at a = 1e8; sf 3.6e9 ulp | severe |
| F2 | pdf | all a; worst for large a near the mode | `exp(_logpdf)` amplifies the absolute error of logpdf; error ~ max(\|(a-1)log x\|, x, gammaln(a)) ulp (patterns 5, 3) | 6.3e6 ulp at a = 1e6, x = 1.01e6 | severe |
| F3 | isf, ppf | a >~ 2e5, q in the left tail (isf: q -> 1) | `sc.gammainccinv`/`sc.gammaincinv` invert the inaccurate `gammainc` of F1 | isf 5.4e6 ulp, ppf 1.0e6 ulp at a = 1e6 | severe |
| F4 | logpdf | a >~ 1e3, x within a few sqrt(a) of a | cancellation of `xlogy(a-1,x)`, `x`, `gammaln(a)` (~a log a) into an O(1) result (pattern 5) | 4.6e5 ulp at a = 1e6, x = 1.003e6 | moderate |
| F5 | ppf, isf | a << 1 (a = 1e-3), bulk q | the result is `exp(exponent/a)` with \|exponent/a\| ~ 700, inside `gammaincinv` (patterns 3, 8) | 3.6e3 ulp at a = 1e-3, q = 0.5 | moderate |
| F6 | cdf, sf | \|a log x - x - gammaln(a)\| large and \|x - a\| > 0.4a (e.g. a = 1e3 with x = 500 or 2000; x < 1e-100) | `exp(a*log(x) - x - lgam(a))` prefactor inside `igam`/`igamc` rounds its exponent (pattern 5) | sf 2.0e3 ulp, cdf 1.6e3 ulp | moderate |

### F1: cdf (and sf) return an unconverged series for large a below the mean

- **Code:** `_cdf`, line 3636: `return sc.gammainc(a, x)` (and `_sf`, line 3639,
  `sc.gammaincc(a, x)`, which for x < a is computed as `1 - igam`). The defect
  is inside `scipy.special.gammainc` (cephes `igam`): for x < a outside the
  asymptotic window it evaluates the Kummer series
  `P(a,x) = x**a e**-x/Gamma(a+1) * sum_n prod_i x/(a+i)` in `igam_series`,
  whose loop is capped at `MAXITER = 2000` iterations; when the cap is hit the
  partial sum is returned as the result.
- **Mechanism:** the uniform asymptotic (Temme) branch is used only for
  `|x - a| < 4.5*sqrt(a)`; just outside it the series is used and needs
  about 1e4 (a = 1e6) to 3e5 (a = 1e9) terms, so the returned value is short
  by the tail of the series, a relative error of roughly
  `exp(-2000*|log(x/a)| - 2000**2/(2a))`. Measured agreement with that model
  (`scratch/gamma/probe4.py`): scipy's value equals the 2000-term partial sum
  to 1e-14 relative at a = 1e6 for x = a - 5..12*sqrt(a), while the converged
  series needs 2433-4374 terms there. The error grows with a:
  relative error at x = a - 5*sqrt(a) is 3e-15 (a = 1e5), 8e-12 (a = 3e5),
  4.3e-6 (a = 1e6), 3.1e-2 (a = 1e7), 3.5e-1 (a = 1e8), 7.2e-1 (a = 1e9)
  (`scratch/gamma/probe5.py`, series reference validated against
  `mp.gammainc` at a = 1e6 to 20 digits).
- **Regime:** a >~ 2e5 (at a = 1e5 the series still converges inside 2000
  terms) and `4.5*sqrt(a) < a - x`; the error fades once
  `a - x >~ 15*sqrt(a)` because the series then converges quickly. Only the
  lower tail is affected; `gammaincc` for x > a (continued fraction) is
  accurate to <= 355 ulp at a = 1e6 up to x = a + 30*sqrt(a). The
  `4.5*sqrt(a)` edge is visible in the a = 1e8 grid: `x = a - 4*sqrt(a)` is
  accurate (asymptotic branch, 12 ulp) while `x = a - 4.6*sqrt(a)` is off by
  37.5%.
- **Reproducer:** `python accuracy.py refs/gamma_ref.py --at cdf 995000 1e6`
  and `python accuracy.py scratch/gamma/gamma_a1e8_ref.py` (a = 1e8 grid)

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(995000, 1e6)` | 2.7495684457992286e-07 | 2.749580359270071e-07 | 2.0e+10 | 5.19e+03 | severe |
  | `cdf(992000, 1e6)` | 5.240122755715965e-16 | 5.24012281543083e-16 | 5.1e+07 | 8.12e+03 | severe |
  | `cdf(990000, 1e6)` | 5.4466446918738522e-24 | 5.4466446930108087e-24 | 9.4e+05 | 1.01e+04 | moderate |
  | `sf(995000, 1e6)` | 0.9999997250431554 | 0.9999997250419641 | 5.37e+03 | 0.00143 | moderate |
  | `cdf(99954000, 1e8)` | 1.3154316493365769e-06 | 2.105648671238922e-06 | 1.7e+15 | 4.8e+04 | severe |
  | `cdf(99900000, 1e8)` | 6.413019934564911e-24 | 7.369931066896994e-24 | 5.8e+14 | 1.0e+05 | severe |
  | `sf(99954000, 1e8)` | 0.9999986845683506 | 0.9999978943513288 | 3.6e+09 | 0.101 | severe |

- **Fix direction:** in `scipy.special`, extend the uniform asymptotic
  expansion to the whole large-a window (boost uses it for
  `|x - a| < ~20*sqrt(a)`) or raise/remove `MAXITER` in `igam_series`
  (and raise an sf_error instead of returning an unconverged sum). Nothing
  can be done inside `gamma_gen` short of not calling `gammainc`.

### F2: pdf exponentiates a logpdf that carries an absolute error of ~a*log(a)*eps

- **Code:** `_pdf`, lines 3628-3630: `return np.exp(self._logpdf(x, a))` with
  `_logpdf = sc.xlogy(a-1.0, x) - x - sc.gammaln(a)`.
- **Mechanism:** patterns 5 and 3. `exp` turns an *absolute* error of its
  argument into a *relative* error of the result, and the three terms of
  `_logpdf` are each rounded to their own ulp, so the pdf's relative error is
  about `max(|(a-1) log x|, x, gammaln(a)) * eps`, i.e. an ulp count equal to
  that magnitude. For a = 1e6, x = 1e6 the terms are 1.38e7, 1e6 and 1.28e7
  while the result is -7.83, so one rounding of 1.4e7 (ulp 3.7e-9) gives
  ~1e-9 relative error in the pdf (3.1e6 ulp measured). The same effect gives
  ~500 ulp in the far tails of moderate shapes (|logpdf| ~ 650 there).
- **Regime:** everywhere; the ulp count tracks the largest term. Measured:
  513 ulp at a = 1.993, x = 1e-280; 430 ulp at a = 1e-3, x = 1e-280;
  5.5e3 ulp at a = 1e3; 3.1e6-6.3e6 ulp at a = 1e6 near the mode.
- **Reproducer:** `python accuracy.py refs/gamma_ref.py --at pdf 1000000 1e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1e6, 1e6)` | 0.0003989422468833316 | 0.00039894224715624404 | 3.1e+06 | 1 | severe |
  | `pdf(1010000, 1e6)` | 1.0606015720532774e-25 | 1.0606015705798405e-25 | 6.3e+06 | 1e+04 | severe |
  | `pdf(1003000, 1e6)` | 4.4584490244843422e-06 | 4.4584490189059984e-06 | 5.6e+06 | 3e+03 | severe |
  | `pdf(999, 1000)` | 0.012620922336039673 | 0.012620922336024371 | 5.5e+03 | 0 | moderate |
  | `pdf(1e-280, 1.9932305483800778)` | 7.8828449702354946e-279 | 7.8828449702363918e-279 | 513 | 0.993 | minor |

- **Fix direction:** build the pdf from the Lanczos-scaled prefactor that
  `scipy.special`'s `igam_fac` (boost's `gamma_p_derivative`) already uses:
  `pdf = igam_fac(a, x)/x`, i.e.
  `logpdf = a*log1pmx((x-a)/a) + 0.5*log(a/(2*pi)) - log(x) - log(lanczos_sum_expg_scaled(a))`,
  where no term exceeds O(log a) except the genuinely large
  `a*log1pmx`. Note that recomputing the Stirling correction as
  `gammaln(a) - ((a-0.5)log a - a + 0.5 log 2pi)` is *not* enough: that
  subtraction re-introduces the cancellation and still leaves 2.2e6 ulp at
  a = 1e6 (measured).

### F3: isf and ppf inherit F1 through gammainccinv / gammaincinv

- **Code:** `_ppf`, line 3642: `sc.gammaincinv(a, q)`; `_isf`, line 3645:
  `sc.gammainccinv(a, q)`.
- **Mechanism:** both inverse functions Newton-iterate on `igam`/`igamc`, so
  they converge to the root of the *inaccurate* cdf of F1. The check is
  quantitative: at a = 1e6, q = 1e-8 the root is x = 994398 = a - 5.6*sqrt(a),
  inside the F1 window where `gammainc` is off by 1.3e-6 relative; dividing
  that by `dlogP/dlogx = 1/cond(arg) = 5.8e3` predicts 2.2e-10 relative error
  in x, i.e. 1.0e6 ulp - exactly what is measured.
- **Regime:** a >~ 2e5 with the quantile falling in `a - 15*sqrt(a) < x <
  a - 4.5*sqrt(a)`: at a = 1e6 that is q in ~[1e-51, 2e-6] for ppf, and
  `1 - q` in the same range for isf.
- **Reproducer:** `python accuracy.py refs/gamma_ref.py --at isf 0.9999993371296838 1e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `isf(0.9999993371296838, 1e6)` | 995171.60511821054 | 995171.60391565238 | 5.4e+06 | 301 | severe |
  | `isf(0.999999938945977, 1e6)` | 994718.68141643691 | 994718.6809620338 | 2.1e+06 | 3e+03 | severe |
  | `ppf(1e-08, 1e6)` | 994398.1600147608 | 994398.15978612122 | 1.0e+06 | 0.000173 | severe |
  | `ppf(3.4807005884285624e-14, 1e6)` | 992529.75300748763 | 992529.75300321169 | 1.9e+04 | 0.000132 | moderate |

- **Fix direction:** none inside `gamma_gen`; fixing F1 fixes this.

### F4: logpdf cancels three terms of size a*log(a)

- **Code:** `_logpdf`, line 3633:
  `sc.xlogy(a-1.0, x) - x - sc.gammaln(a)`.
- **Mechanism:** pattern 5. Near the mode the result is O(1) (about
  `-0.5*log(2*pi*a)`) while the terms are of size `a*log a`; each is rounded
  to ~`a*log(a)*eps`, so the relative error of logpdf is about
  `a*log(a)*eps/|logpdf|`. At a = 1e6 that is 1.4e7*2.2e-16/8 = 3.9e-10
  (3.9e5 ulp measured).
- **Regime:** a >~ 1e3 and x within a few `sqrt(a)` of a (where |logpdf| is
  small); 1.3e3 ulp at a = 1e3, 4.6e5 ulp at a = 1e6. Far out in the tails
  |logpdf| grows and the relative error drops back (<= 1.6e3 ulp at
  x = 1.1e6). A second, mild instance is a = 1e-3 at x ~ a, where logpdf
  crosses zero (1.0e3 ulp, cond (arg) 368).
- **Reproducer:** `python accuracy.py refs/gamma_ref.py --at logpdf 1000000 1e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e6, 1e6)` | -7.826693896204233 | -7.82669389552014312 | 3.9e+05 | 0.128 | moderate |
  | `logpdf(1003000, 1e6)` | -12.320709604769945 | -12.320709606021129 | 4.6e+05 | 244 | moderate |
  | `logpdf(999, 1000)` | -4.3723993392750344 | -4.3723993392762468 | 1.3e+03 | 0 | moderate |

- **Fix direction:** same Lanczos/`log1pmx` form as F2 (that form gives
  69 ulp instead of 1.3e3 at a = 1e3, x = 999, measured).

### F5: ppf and isf for a << 1 amplify the rounding of a large exponent

- **Code:** `_ppf`, line 3642 / `_isf`, line 3645 (inside
  `sc.gammaincinv`/`sc.gammainccinv`).
- **Mechanism:** patterns 3 and 8. For a << 1 the quantile is
  `x = (q*Gamma(a+1))**(1/a) = exp((log q + gammaln(a+1))/a)`, so the exponent
  has magnitude `|log x|` (694 at a = 1e-3, q = 0.5) and each half-ulp of it
  costs ~`|log x|` ulp of x, on top of the inherent `cond (arg) = 1/a = 1e3`.
  Measured errors are ~3.6x the conditioning, consistent with the sum of the
  two effects.
- **Regime:** a <~ 1e-2 with q in the bulk (the result is then a tiny but
  normal double, 1e-300 ... 1e-70); at a = 0.1 the effect is down to 21 ulp.
- **Reproducer:** `python accuracy.py refs/gamma_ref.py --at ppf 0.5000000000000001 0.001`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(0.5000000000000001, 1e-3)` | 5.2442064082749762e-302 | 5.2442064082791425e-302 | 3.6e+03 | 1e+03 | moderate |
  | `isf(0.36, 1e-3)` | 8.5045125183234004e-195 | 8.5045125183274596e-195 | 2.2e+03 | 562 | moderate |
  | `ppf(0.85, 1e-3)` | 1.4743519045953742e-71 | 1.474351904596337e-71 | 2.9e+03 | 1e+03 | moderate |

- **Fix direction:** in `scipy.special`'s `igami`, do the final Newton step on
  `log P` instead of `P` for a < 1 (the relative accuracy of x is then limited
  only by `1/a`, the unavoidable conditioning).

### F6: cdf/sf lose the rounding of the exp-prefactor exponent

- **Code:** `_cdf`, line 3636 / `_sf`, line 3639; the loss is in
  `igam_fac(a, x)` inside `scipy.special`, which for `|a - x| > 0.4*a`
  computes the prefactor as `exp(a*log(x) - x - lgam(a))`.
- **Mechanism:** pattern 5/3 - the same amplification as F2, now inside the
  special function: the exponent is a cancelling sum of terms of size
  `a*log x` (6.2e3 for a = 1e3, x = 500), so the result carries
  `~a*log(x)*eps` relative error. The far left tail of moderate shapes is the
  same effect with `|a log x|` ~ 600.
- **Regime:** `|x - a| > 0.4a` and `max(|a log x|, x, gammaln(a)) >> 1`:
  a >~ 1e2 with x far from a (1.6e3-2.0e3 ulp at a = 1e3), and x < 1e-50 for
  any a (150-414 ulp, minor).
- **Reproducer:** `python accuracy.py refs/gamma_ref.py --at sf 2000 1000`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `sf(2000, 1000)` | 6.8473494596177583e-136 | 6.847349459614753e-136 | 2.0e+03 | 1e+03 | moderate |
  | `cdf(500, 1000)` | 3.2982727970682507e-86 | 3.2982727970670999e-86 | 1.6e+03 | 501 | moderate |
  | `cdf(1e-130, 1.9932305483800778)` | 3.8168729838076914e-260 | 3.8168729838080423e-260 | 414 | 1.99 | minor |
  | `cdf(1e-280, 1)` | 9.9999999999994638e-281 | 1e-280 | 241 | 1 | minor |

- **Fix direction:** in `scipy.special`, use the Lanczos prefactor branch of
  `igam_fac` for all `a > ~20` instead of only for `|a - x| <= 0.4a`.

## RuntimeWarnings

The only warning seen is `underflow encountered in exp` from `_pdf`
(`np.exp(self._logpdf(x, a))`), raised for every parameter set. It comes from
grid points where the pdf genuinely underflows (e.g. `pdf(1e300, 2)`), where
the true value is 0 or subnormal, so it is not an accuracy problem; because
the harness evaluates a whole grid in one call the warning is attributed to
all 686 points of those calls, including accurate ones. No warnings in
cdf, sf, logpdf, ppf or isf anywhere on the grids.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | bulk (all 8 parameter sets), high tail, low tail for a <= 1e3 | 13.2 ulp in the bulk; 124 ulp in the low tail of a = 10.5 (F6, minor) |
| sf | whole low tail (x < a) for a <= 1e3 and the high tail of a = 1e6 (up to a + 30*sqrt(a)) | 0 flagged points in the low tail; 23.3 ulp in the bulk; 355 ulp at a = 1e6, x = a + 30 sqrt(a) (probe) |
| pdf, logpdf | a = 1 (xlogy(0, x) is exact), a = 0.1 and 1.993 in the bulk | <= 7.1 ulp for logpdf; pdf as in F2 |
| ppf, isf | q in [1e-300, 1 - 1e-15.5] for a = 1, 1.993, 10.5, 1e3 | 125 ulp (ppf, a = 1, q ~ 1e-126), 17.5 ulp (isf) - minor, within a factor of ~100 of the conditioning |
| ppf, isf | q -> 1 for all a <= 1e3 (ill-conditioned region, cond (arg) up to 3e15) | all within conditioning |
| cdf, sf | a = 1e-3 and 0.1 over the full grid (cdf ~ x**a/Gamma(a+1) regime) | 76 ulp (cdf), 23 ulp (sf) |

## Suspected, not confirmed

- `sc.gammaincc` (sf) for x > a with very large a: the Temme window is
  symmetric, so an `igamc`-side analogue of F1 might exist. Probed at
  a = 1e6 for x up to a + 30*sqrt(a) (<= 355 ulp) and at a = 1e8 for
  x = a + 10*sqrt(a) (29 ulp): no loss found - the upper tail uses a
  continued fraction that converges quickly, so this looks genuinely safe.
- a >= 1e10 could not be measured: the Kummer-series reference needs >1e6
  terms and mpmath's `gammainc` does not converge there. Extrapolating F1's
  model, the cdf relative error exceeds 0.9 (a "loss" by the harness scale)
  for a >~ 1e10; this is stated as a trend, not a measurement.
- `x = 0` (pdf = inf for a < 1, 0 for a > 1, 1 for a = 1) is not covered by
  the harness grids (the references are not normal doubles except for a = 1,
  where scipy is exact).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | `log(sf)` = `log(sc.gammaincc(a, x))` | `logsf = -x + a*log(x) - gammaln(a+1) + log(continued-fraction sum)`, i.e. a log-space `igamc` (what `sc.chndtr`-style code or boost's `gamma_q` in log space would give) | sf underflows for `x - (a-1)log x + gammaln(a) > 745`: `logsf(2000, 1.9932)` returns `-inf`, true -1992.447207771643 (loss); `logsf(1e5, 1e3)` returns `-inf`, true -94403.79784357095 |
| logcdf | `log(cdf)` = `log(sc.gammainc(a, x))` | `logcdf = a*log(x) - x - gammaln(a+1) + log1p(series)` (log-space `igam`; for x << a the series sum is ~1) | cdf underflows in the left tail: `logcdf(1000, 1e6)` returns `-inf`, true -5908763.104675533 (loss); also inherits F1 wherever the cdf itself is wrong |

Both would also avoid the `log(1 - ...)`-style loss of the generic default
near the other end (logcdf for x >> a is `log1p(-sf)`, which the default
cannot express better than 1 - sf).

## Shared code

All of F1, F3, F5 and F6 are defects of `scipy.special`, not of
`gamma_gen`, which only forwards its arguments:

- `sc.gammainc` / `sc.gammaincc` (cephes `igam`/`igamc`): F1 (2000-iteration
  cap in `igam_series`, severe for a >~ 2e5) and F6 (`exp(a log x - x -
  lgam a)` prefactor). Every distribution that maps onto the regularized
  incomplete gamma function in that regime is affected the same way:
  `erlang` (a subclass of `gamma_gen`, so literally the same code),
  `dgamma`, `invgamma`, `gengamma`, `chi`/`chi2` (`chdtr`/`chdtrc` are the
  same cephes code: df >~ 4e5), and the Poisson / negative-binomial cdfs
  (`pdtr`, which is `igamc`).
- `sc.gammaincinv` / `sc.gammainccinv` (cephes `igami`): F3 (inherits F1) and
  F5 (`exp` of a `1/a`-scaled exponent for a << 1). Same list of dependents
  (`erlang.ppf`, `chi2.ppf`, ...).
- `sc.gammaln`, `sc.xlogy` themselves are accurate; F2/F4 are caused by
  *combining* them in `gamma_gen._logpdf`, so they must be fixed in
  `gamma_gen` (or by exposing `igam_fac`).

## Existing GitHub issues
- #17832 (open): Tracking issue: custom methods for continuous distributions -
  relates to the inherited `logcdf`/`logsf` noted above; names the generic
  `log(cdf)` / `sf = 1 - _cdf` patterns as a precision risk and tracks which
  distributions still lack their own `_logcdf`/`_logsf`
- #11465 (closed, implemented in PR #21050): "Numerical stability of
  distributions: Add log_p as argument to all distributions?" - same scope:
  the generic path "computes log(cdf) which loses precision"
- #26471 (open, filed 2026-10-21 from issue-reports/gammainc-large-a.md): BUG: inaccuracy in special.gammainc for certain large parameters - relates to F1 and F3
- F2 and F6: none found. Before #26471 was filed, the `sc.gammainc` convergence
  failure of F1/F3 (cephes `igam_series` returning its 2000-term partial sum
  outside the Temme window) does not appear to be reported on SciPy's tracker.
  Searches over scipy/scipy found only unrelated incomplete-gamma bugs
  (#10856, limits at infinite `a`, fixed in PR #10865; PR #6307 and PR #8066,
  earlier accuracy work on gammainc/gammaincc and their inverses). The same
  *class* of unconverged-iterate bug has been filed against downstream
  reimplementations (e.g. jax-ml/jax #41201, about JAX's own continued
  fraction - there SciPy is cited as the correct reference), which is further
  reason to report F1 upstream rather than assume it is known.
- #6111 ("scipy.stats.gamma is wrong", closed) was checked and excluded: it is
  about the loc/scale argument order, not accuracy.

## Verification

Independent references built from scratch under `scratch/gamma/verify/`
(`quadref.py`, `crosscheck.py`, `verify.py`, `verify_ppf2.py`, `mech.py`,
`boundary.py`/`boundary2.py`), by two routes that do not reuse
`refs/gamma_ref.py` or `scratch/gamma/gamma_a1e8_ref.py`: (a) high-precision
tanh-sinh/Gauss-Legendre quadrature of the pdf after the substitution
`t = x - sqrt(a)*v` (no incomplete-gamma series anywhere), and (b) the Kummer
series summed to full convergence. The two agree to 42 digits at every F1
point, and at a = 1e6 both agree with `mp.gammainc(a, 0, x, regularized=True)`
to all printed digits. Quantiles were obtained by Newton on `log P` with the
quadrature cdf (residual `P(a, x_ref)/q - 1 ~ 1e-58`); dps 100 vs 200
(quadrature) and 60 vs 110 (inversion) agree to > 60 digits everywhere.

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(995000, 1e6)` | 2.7495684457992286e-07 | 2.749580359270070753827908e-07 | verified (severe) | 1.95e10 ulp, rel 4.33e-6; cond (arg) 5.19e3 |
| F1 | `cdf(992000, 1e6)` | 5.2401227557159648e-16 | 5.240122815430830664030316e-16 | verified (severe) | 5.13e7 ulp, rel 1.14e-8 |
| F1 | `cdf(990000, 1e6)` | 5.4466446918738522e-24 | 5.446644693010808670771526e-24 | verified (moderate) | 9.4e5 ulp |
| F1 | `sf(995000, 1e6)` | 0.99999972504315537 | 0.999999725041964073 | verified (moderate) | 5.37e3 ulp |
| F1 | `cdf(99954000, 1e8)` | 1.3154316493365769e-06 | 2.105648671238922060053592e-06 | verified (severe) | 1.69e15 ulp, rel 0.375 - both routes agree to 42 digits, no mpmath `gammainc` used |
| F1 | `cdf(99900000, 1e8)` | 6.4130199345649108e-24 | 7.369931066896993946965663e-24 | verified (severe) | 5.85e14 ulp, rel 0.130 |
| F1 | `sf(99954000, 1e8)` | 0.99999868456835062 | 0.999997894351328761 | verified (severe) | 3.56e9 ulp |
| F1 | headline `gammainc(a, a - 5 sqrt(a))` | see note | see note | verified | rel err 6.7e-15 (a=1e5), 8.2e-12 (3e5), 4.33e-6 (1e6), 3.09e-2 (1e7), 0.347 (1e8), 0.719 (1e9) - matches the report's 4.3e-6 / 3.5e-1 / 7.2e-1 |
| F1 | mechanism (MAXITER = 2000) | - | - | verified | SciPy's value equals the **exact 2000-term** partial sum of `igam_series` to 1.6e-15 ... 4.8e-15 relative at (1e6, 995000), (1e6, 990000), (1e8, 99954000), (1e9, a-5 sqrt(a)), while convergence to double precision needs 2807 / 4374 / 43075 / 126562 terms. Branch edge confirmed at a = 1e8: 11.5 ulp at `x = a - 4.49 sqrt(a)`, rel 0.382 at `a - 4.51 sqrt(a)`. |
| F2 | `pdf(1e6, 1e6)` | 0.0003989422468833316 | 0.00039894224715624402970 | verified (severe) | 3.08e6 ulp |
| F2 | `pdf(1010000, 1e6)` | 1.0606015720532774e-25 | 1.060601570579840493e-25 | verified (severe) | 6.26e6 ulp |
| F2 | `pdf(1003000, 1e6)` | 4.4584490244843422e-06 | 4.458449018905998265e-06 | verified (severe) | 5.63e6 ulp |
| F2 | `pdf(999, 1000)` | 0.012620922336039673 | 0.0126209223360243718 | verified (moderate) | 5.46e3 ulp |
| F2 | mechanism | - | - | verified | `pdf` rel err == `logpdf` abs err exactly (6.84e-10 at a = 1e6, x = 1e6) == rel err of `exp(logpdf_scipy)`, and is bounded by `max(xlogy, x, gammaln)*eps = 3.07e-9`; terms 1.38e7 / 1e6 / 1.28e7 for an O(1) result, as stated |
| F3 | `isf(0.9999993371296838, 1e6)` | 995171.60511821054 | 995171.603915652369062256 | verified (severe) | 5.44e6 ulp; cond (arg) 301 |
| F3 | `isf(0.999999938945977, 1e6)` | 994718.68141643691 | 994718.680962033762089 | verified (severe) | 2.06e6 ulp |
| F3 | `ppf(1e-08, 1e6)` | 994398.1600147608 | 994398.159786121271213 | verified (severe) | 1.04e6 ulp |
| F3 | `ppf(3.4807005884285624e-14, 1e6)` | 992529.75300748763 | 992529.753003211727015 | verified (moderate) | 1.94e4 ulp |
| F3 | mechanism | - | - | verified | `P(a, x_scipy)/q - 1 = 1.33e-6` at the `ppf(1e-8)` root (i.e. exactly F1's error there) and `dlogP/dlogx = 5770`, giving 2.3e-10 relative in x = 1.04e6 ulp, as claimed |

Reference agreement: my values match `refs/gamma_ref.py` / the auxiliary
a = 1e8 reference to every printed digit, so there is no reference
disagreement; the analyst's a = 1e8 series reference is confirmed by a route
that uses no series at all.

Two regime remarks (not corrections to any number):
- The upper edge of F1's window, "`a - x <~ 15*sqrt(a)`", holds at a = 1e6
  (15.8 ulp at `a - 15 sqrt(a)`, 234 ulp at `a - 30 sqrt(a)`) but not at
  a = 1e8, where the relative error is still 4.8e-2 at `a - 15 sqrt(a)`,
  1.8e-2 at `a - 20 sqrt(a)` and 2.4e-3 at `a - 30 sqrt(a)` (the series needs
  23937 terms even there). The affected window therefore widens with a; the
  regime column understates it for a >= 1e7.
- The cited C source could not be read (this install ships only compiled
  `_ufuncs`), so the `MAXITER = 2000` / `igam_series` attribution was
  confirmed behaviourally by the exact partial-sum fingerprint above, which is
  conclusive. The Python-side citations (`_cdf` line 3636 `sc.gammainc(a, x)`,
  `_pdf` 3628-3630 `np.exp(self._logpdf(x, a))`, `_logpdf` 3633, `_ppf` 3642,
  `_isf` 3645) all match the installed `_continuous_distns.py`.

Verified by dist-accuracy-verifier: 3 of 3 verified.
