# genlogistic: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `genlogistic_gen`, lines 3080-3162
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, ppf, isf
- **Inherited (not measured):** logsf = log(sf)
- **Related:** logistic is the c = 1 special case (separate class, reviewed in `reports/logistic.md`); helpers: none; special functions: sc.expm1, sc.log1p, sc.powm1 (sc.psi, sc.zeta only in `_stats`)
- **Reference:** `refs/genlogistic_ref.py`, 400 digits; self-checks: PASS for the reference. The harness printed FAIL for sf(1e-10), plus a WARN and NOTEs for c = 0.001 and 1e-10. All of them are SciPy errors: for those shapes the bulk of the distribution lies at x < -709.78, where SciPy's cdf/logcdf/sf break (F2, F3). I checked this independently: cdf(-23025850929.940456, 1e-10) = 0.099999999999999996 from the closed form, and `mp.quad` of the pdf gives the same value.
- **Parameter sets:** (0.41192440799679475), (0.001), (1e-10), (1e-20), (1), (50), (1e6), (1e15)
- **Worst finding:** loss in pdf, logpdf, cdf, logcdf, sf, ppf, isf

## Summary

For c around 1, every method is accurate in the bulk and the right tail. Below x = -709.78, `exp(-x)` overflows. In that region cdf returns 0 and logcdf returns -inf at every c. sf returns 1.0, which is wrong whenever c is small. For small c the bulk itself lies in this region, so the failures are not limited to a far tail. logpdf/pdf compute the left-branch slope -c as `-(c-1) - 1`. This loses about log10(1/c) digits and gives exactly 0 once c < 1.1e-16. ppf and isf return +inf whenever `q**(-1/c)` overflows. That is the wrong end of the support, because `sc.powm1` returns 0.0 instead of inf on overflow (shared code). isf is `ppf(1 - q)` and loses all accuracy for q < 1e-8 (inf for q < 1.1e-16). cdf raises a rounded `1 + exp(-x)` to the power -c, which costs about c/2 ulp: 4.8e5 ulp at c = 1e6 and 12 % relative error at c = 1e15.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logcdf, cdf | x < -709.78, any c | exp(-x) overflows (pattern 3) | logcdf -inf for -292.5; cdf 0.0 for 0.49 | loss |
| F2 | sf | x < -709.78, small c | -expm1(logcdf) with logcdf = -inf (pattern 3, via F1) | 1.0 for 7.1e-18 | loss |
| F3 | logpdf, pdf | x < 0, small c | `-(c-1) - 1` rounds the slope -c (pattern 5/9) | logpdf -46.05 for -1e280; pdf 1e-20 for 3.7e-64 | loss |
| F4 | ppf, isf | -log(q)/c > 709.78 (small q or small c) | `sc.powm1` returns 0 instead of inf on overflow (pattern 3/10) | +inf for -1676.9 | loss |
| F5 | isf | q < ~1e-4 | `ppf(1 - q)` rounds q away (pattern 8) | inf for 39.14 (q = 1e-17) | loss |
| F6 | cdf | large c | (1 + exp(-x))**(-c) amplifies rounding (pattern 6) | 4.7e14 ulp (c = 1e15) | severe |
| F7 | ppf, isf | q = 0.5 - 2^-54, c = 1 (zero crossing) | powm1 1-ulp error at the root (cond 9e15) | 0.0 for -2.2e-16 | loss |
| F8 | pdf, logpdf | large logpdf magnitude from log(c); x near log(c) for huge c | exp of a log with absolute error; log(c) - x cancels (pattern 5) | 196 ulp (pdf), 107 ulp (logpdf) | minor |

### F1: logcdf and cdf overflow in the left tail

- **Code:** `_cdf`, line 3129: `(1+np.exp(-x))**(-c)`; `_logcdf`, line 3133: `-c * np.log1p(np.exp(-x))`.
- **Mechanism:** for x < -709.78, `exp(-x)` is inf, so cdf = inf**(-c) = 0 and logcdf = -c*log1p(inf) = -inf (pattern 3). The true values are logcdf = c*x - c*log1p(exp(x)) ≈ c*x, which is always finite, and cdf ≈ exp(c*x), which is normal as long as c*|x| < 708.
- **Regime:** logcdf: every c for x < -709.78. cdf: x < -709.78 with c*|x| < 708, i.e. c < 1. For c <= ~1e-3 this covers the whole bulk; for c = 1e-10 the median is at x = -6.9e9.
- **Reproducer:** `python accuracy.py refs/genlogistic_ref.py --at cdf -710 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-710, 0.001)` | 0.0 | 0.4916441974609651 | inf | 0.71 | loss |
  | `cdf(-1000, 0.41192440799679475)` | 0.0 | 1.269119875918382e-179 | inf | 412 | loss |
  | `cdf(-1e10, 1e-10)` | 0.0 | 0.36787944117144233 | inf | 1 | loss |
  | `logcdf(-710, 1)` | -inf | -710.0 | inf | 1 | loss |
  | `logcdf(-1000, 0.001)` | -inf | -1.0 | inf | 1 | loss |
  | `logcdf(-1000, 0.41192440799679475)` | -inf | -411.92440799679474 | inf | 1 | loss |
  | `logcdf(-1e300, 0.41192440799679475)` | -inf | -4.119244079967948e+299 | inf | 1 | loss |

- **Fix direction:** `logcdf = where(x < 0, c*x - c*log1p(exp(x)), -c*log1p(exp(-x)))`, `cdf = exp(logcdf)`.

### F2: sf returns 1.0 in the left tail for small c

- **Code:** `_sf`, line 3139: `-sc.expm1(self._logcdf(x, c))`.
- **Mechanism:** the formula itself is the right one, but it receives logcdf = -inf from F1, so sf = -expm1(-inf) = 1.0 exactly. With small c the true sf is far from 1: sf ≈ c*|x| when c*|x| is small.
- **Regime:** x < -709.78 whenever the true sf differs from 1 by more than eps/2, i.e. c*|x| < ~37. Total loss (relative error >= 1) when c*|x| < 0.5.
- **Reproducer:** `python accuracy.py refs/genlogistic_ref.py --at sf -710 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(-710, 1e-20)` | 1.0 | 7.099999999999999e-18 | 6.3e+32 | 1 | loss |
  | `sf(-1000, 1e-10)` | 1.0 | 9.999999500000018e-08 | 4.5e+22 | 1 | loss |
  | `sf(-1e7, 1e-10)` | 1.0 | 0.0009995001666250085 | 4.5e+18 | 1 | loss |
  | `sf(-710, 0.001)` | 1.0 | 0.5083558025390349 | 4.4e+15 | 0.687 | severe |
  | `sf(-1000, 0.001)` | 1.0 | 0.6321205588285577 | 2.6e+15 | 0.582 | severe |

- **Fix direction:** fixed by the logcdf rewrite in F1 (`sf = -expm1(logcdf)` with the left-branch logcdf).

### F3: logpdf/pdf lose the slope -c for small c

- **Code:** `_logpdf`, lines 3124-3126: `mult = -(c - 1) * (x < 0) - 1` and `np.log(c) + mult*absx - (c+1) * sc.log1p(np.exp(-absx))`; `_pdf`, line 3118: `np.exp(self._logpdf(x, c))`.
- **Mechanism:** for x < 0, `mult` should be -c. It is computed as `-(c - 1) - 1`. `c - 1` is rounded to the precision of a number near 1, with an absolute error of up to eps/2, and then 1 is subtracted again (pattern 5, with a small shape as in pattern 9). The relative error of `mult` is therefore about eps/(2c), i.e. ~1/(2c) ulp. For c < 1.1e-16, `c - 1` rounds to -1.0 and `mult` is exactly 0, so the c*x term vanishes. The error matters wherever c*|x| is a significant part of logpdf, which for small c includes the bulk. pdf = exp(logpdf) turns the absolute error of logpdf into a relative error.
- **Regime:** x < 0, c << 1. The absolute error of logpdf is up to |x|*eps/2. That is up to ~1/(2c) ulp relative where c*|x| dominates logpdf, and up to ~|x|/2 ulp in pdf. Measured: c = 1e-3 gives 2.6e3 ulp (pdf, x = -7e5). c = 1e-10 gives 3.7e8 to 3.7e10 ulp. For c = 1e-20 the result is a constant log(c), independent of x.
- **Reproducer:** `python accuracy.py refs/genlogistic_ref.py --at logpdf -1e300 1e-20`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(-1e300, 1e-20)` | -46.051701859880914 | -1e+280 | 4.5e+15 | 1 | loss |
  | `logpdf(-1e10, 1e-10)` | -24.02585101268083 | -24.025850929940457 | 1.6e+07 | 0.0416 | severe |
  | `pdf(-1e22, 1e-20)` | 9.999999999999992e-21 | 3.720075976020856e-64 | 1.2e+59 | 100 | loss |
  | `pdf(-1e20, 1e-20)` | 9.999999999999992e-21 | 3.678794411714423e-21 | 7.7e+15 | 1 | loss |
  | `pdf(-1e12, 1e-10)` | 3.720045196101518e-54 | 3.7200759760208224e-54 | 3.7e+10 | 100 | severe |
  | `pdf(-1e10, 1e-10)` | 3.67879410732962e-11 | 3.678794411714423e-11 | 3.7e+08 | 1 | severe |
  | `pdf(-700000, 0.001)` | 9.859676543754449e-308 | 9.859676543759627e-308 | 2.62e+03 | 700 | moderate |

- **Fix direction:** use the slope directly: `np.where(x < 0, c*x, -x)` instead of `mult*absx`.

### F4: ppf and isf return +inf when q**(-1/c) overflows (sc.powm1 bug)

- **Code:** `_ppf`, line 3136: `-np.log(sc.powm1(q, -1.0/c))`; `_isf`, line 3142: `self._ppf(1 - q, c)`.
- **Mechanism:** when -log(q)/c > 709.78, the true value of `q**(-1/c) - 1` overflows, so ppf should be computed in log space (pattern 3/8). The intermediate overflow would give -log(inf) = -inf, which is already a total loss. `sc.powm1(x, y)` with x < 1, y < 0 makes it worse: it returns **0.0** instead of inf on overflow. For example `powm1(0.5, -1024) = 0.0`, `powm1(1e-300, -2) = 0.0`, `powm1(0.5, -1e20) = 0.0`, while `powm1(2, 1024) = inf` is correct. So ppf returns -log(0) = **+inf**, the upper end of the support, for a quantile that lies deep in the lower tail. isf inherits this through `ppf(1 - q)`.
- **Regime:** -log(q) > 709.78*c: q < ~1e-127 for c = 0.41, q < 0.49 for c = 1e-3, every q for c <= 1e-10 (ppf and isf are inf over the whole of (0, 1)).
- **Reproducer:** `python accuracy.py refs/genlogistic_ref.py --at ppf 1e-300 0.41192440799679475`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 0.41192440799679475)` | inf | -1676.9473099627269 | inf | 0.00145 | loss |
  | `ppf(1e-05, 0.01)` | inf | -1151.2925464970228 | inf | 0.0869 | loss |
  | `ppf(0.5, 1e-20)` | inf | -6.931471805599453e+19 | inf | 1.44 | loss |
  | `isf(0.5, 1e-10)` | inf | -6931471805.599453 | inf | 1.44 | loss |

- **Fix direction:** with `z = -log(q)/c` (isf: `z = -log1p(-q)/c`), `ppf = -log(expm1(z))`, and for z > ~700 `ppf = -z - log1p(-exp(-z))`. Separately, fix `sc.powm1` to return inf on overflow (see Shared code).

### F5: isf loses accuracy for small q (ppf(1 - q))

- **Code:** `_isf`, line 3142: `self._ppf(1 - q, c)`.
- **Mechanism:** `1 - q` keeps only the digits of q above eps (pattern 8). Then `powm1(1 - q, -1/c) ≈ q/c` inherits a relative error of about eps/q, and isf ≈ log(c/q) inherits that error in absolute terms. For q < 1.1e-16, `1 - q` is exactly 1, powm1 gives 0 and isf = +inf.
- **Regime:** all c. The error is about 1/q ulp-equivalents: about 200 ulp at q = 1e-4, 1e6 ulp at q = 1e-8, 1e13 ulp at q = 1e-16, and loss (inf) at q < 1.1e-16.
- **Reproducer:** `python accuracy.py refs/genlogistic_ref.py --at isf 1e-17 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1e-17, 1)` | inf | 39.14394658089878 | inf | 0.0255 | loss |
  | `isf(1e-300, 50)` | inf | 694.6875509036419 | inf | 0.00144 | loss |
  | `isf(1e-16, 1)` | 36.7368005696771 | 36.841361487904734 | 1.3e+13 | 0.0271 | severe |
  | `isf(1e-12, 1)` | 27.63104323789236 | 27.631021115927549 | 3.61e+09 | 0.0362 | severe |
  | `isf(1e-08, 1)` | 18.420680728927607 | 18.420680733952366 | 1.2e+06 | 0.0543 | severe |
  | `isf(1e-08, 1e-10)` | -100.0000010024759 | -100.0000005 | 2.26e+07 | 1 | severe |
  | `isf(0.0001, 0.001)` | 2.2521159171305265 | 2.2521159171304106 | 232 | 0.467 | minor |

- **Fix direction:** `isf = -log(expm1(-log1p(-q)/c))`, with the log-space form of F4 for large arguments.

### F6: cdf amplifies the rounding of 1 + exp(-x) by c

- **Code:** `_cdf`, line 3129: `(1+np.exp(-x))**(-c)`.
- **Mechanism:** the up to 1/2-ulp rounding error of `1 + exp(-x)` is raised to the power -c, giving a relative error of up to c/2 ulp (pattern 6). logcdf (`-c*log1p(exp(-x))`) does not have this problem.
- **Regime:** large c, in the bulk and both tails (wherever exp(-x) is not negligible against eps): about 20 ulp at c = 50, 4.8e5 ulp at c = 1e6, 4.7e14 ulp (12 % relative error) at c = 1e15.
- **Reproducer:** `python accuracy.py refs/genlogistic_ref.py --at cdf 13.815510557964274 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(34.538776394910684, 1e15)` | 0.32948546950694785 | 0.36787944117144206 | 4.7e+14 | 34.5 | severe |
  | `cdf(30, 1e15)` | 2.5223982597296125e-41 | 2.2927664021665373e-41 | 4.5e+14 | 2.81e+03 | severe |
  | `cdf(8, 1e6)` | 2.1620366726983237e-146 | 2.1620366724681254e-146 | 4.8e+05 | 2.68e+03 | moderate |
  | `cdf(13.815510557964274, 1e6)` | 0.3678796251413505 | 0.3678796251110861 | 3.7e+05 | 13.8 | moderate |
  | `cdf(8, 50)` | 0.9833695203429025 | 0.98336952034289715 | 24.4 | 0.134 | minor |

- **Fix direction:** `cdf = exp(-c*log1p(exp(-x)))`, i.e. `exp(logcdf)` with the F1 logcdf.

### F7: ppf/isf return 0 at the zero crossing of the quantile

- **Code:** `_ppf`, line 3136: `-np.log(sc.powm1(q, -1.0/c))`.
- **Mechanism:** the quantile is 0 at q = 2**(-1/c). Next to that point the result is tiny and the problem is extremely ill-conditioned (cond (arg) 9.0e15). `powm1(0.49999999999999994, -1.0)` returns 1.0 (true 1.0000000000000004, 1 ulp low), so -log gives exactly 0. For isf, `1 - q` rounds 0.5 + 2^-54 to 0.5. The tool classifies 0.0 for a nonzero value as loss. The absolute error is only 2.2e-16, though, and it is within the conditioning, so this is of little practical importance.
- **Regime:** q within a few ulp of 2**(-1/c).
- **Reproducer:** `python accuracy.py refs/genlogistic_ref.py --at ppf 0.49999999999999994 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.49999999999999994, 1)` | 0.0 | -2.220446049250313e-16 | inf | 9.0e+15 | loss |
  | `isf(0.49999999999999994, 1)` | 0.0 | 2.220446049250313e-16 | inf | 9.0e+15 | loss |

- **Fix direction:** the forms in F4/F5 (`-log(expm1(-log(q)/c))`) give the right sign and magnitude here. For c = 1, `-sc.logit`-style evaluation is exact.

### F8: pdf/logpdf: small excess errors from large log terms

- **Code:** `_logpdf`, line 3126 (`np.log(c) + mult*absx - ...`); `_pdf`, line 3118 (`np.exp(self._logpdf(x, c))`).
- **Mechanism:** (a) pdf = exp(logpdf). The absolute error of logpdf, about |logpdf| * eps/2, becomes a relative error of about |logpdf|/2 ulp in pdf. This exceeds the conditioning when logpdf is large because of the shape (log(c), (c+1)*log 2) rather than x. (b) Near the mode x = log(c) with huge c, `log(c) - x` cancels: both terms are ~690 for c = 1e300 and the result is -1 (pattern 5).
- **Regime:** small or large c with x where cond (arg) is small; huge c near the mode.
- **Reproducer:** `python accuracy.py refs/genlogistic_ref.py --at pdf 0 50`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0, 50)` | 2.2204460492503235e-14 | 2.220446049250313e-14 | 21.1 | 0 | minor |
  | `pdf(-1, 1e-300)` | 7.310585786300341e-301 | 7.310585786300049e-301 | 180 | 0.269 | minor |
  | `pdf(690.7755278982137, 1e300)` | 0.36787944117143356 | 0.36787944117144233 | 107 | 1.64e-11 | minor |
  | `logpdf(690.7755278982137, 1e300)` | -1.0000000000000238 | -1.0 | 107 | 1.64e-11 | minor |

- **Fix direction:** largely inherent to the log/exp formulation. For huge c, `-(x - log(c))` could be formed as `-x + log(c)` with a compensated log(c), but this is low priority.

## RuntimeWarnings

- `overflow encountered in exp` (cdf, logcdf, sf; x < -709.78): coincides with the wrong results of F1/F2.
- `divide by zero encountered in log` (ppf, isf): coincides with the wrong +inf results of F4/F5.
- `underflow encountered in exp` / `underflow encountered in multiply` (pdf, logpdf; `exp(-absx)` for large |x|): spurious. The results in the right tail and moderate left tail are accurate (logpdf max 6 ulp outside F3).
- `underflow encountered in power` (cdf, right/left tail where the result underflows): spurious, the result is correctly 0 or subnormal.
- `overflow encountered in multiply` (logpdf with c = 1e15 at x = -1e300): spurious. `mult*absx` = -inf, and the true value -1e315 is not representable.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | full grid, c >= 1e-3 (outside F3), incl. c = 1e15 | 6 ulp |
| pdf | full grid, c = 0.41, 1, 50, 1e6, 1e15 (outside F3) | 260 ulp, within cond (700) |
| cdf | x > -709.78, c <= 1 | 0.88 ulp |
| logcdf | x > -709.78, all parameter sets | 1.04 ulp |
| sf | x > -709.78, all parameter sets (right tail to 1e300) | 1.54 ulp |
| ppf | -log(q)/c < 709 (outside F4), q in [1e-300, 1 - 3e-16] | 8.04 ulp (c = 50), apart from F7 |
| isf | q >= 0.5 (outside F4/F5) | 7.24 ulp |

## Suspected, not confirmed

- logpdf cancellation `log(c) - x` near the mode for large c: measured at c = 1e15 (6 ulp, ok) and c = 1e300 (107 ulp, minor, F8). It does not grow beyond ~log(c) ulp, so no larger error is expected.
- c = 5e-324 (the domain edge): not in PARAMS. By F3, `mult` is 0 and logpdf ≈ log(c) for all x < 0. By F4, ppf/isf are +inf for all q. Not measured separately.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | log(sf) | `log(-expm1(logcdf))` = `log1mexp(logcdf)`, with right-tail form `log(c) + log(log1p(exp(-x))) ≈ log(c) - x` for large x (and the F1 left-branch logcdf) | right tail: sf underflows for x > ~745 + log(c), e.g. `logsf(800, 1)` = -inf (true ≈ -800); left tail for small c via F2: `logsf(-23025850929.940456, 1e-10)` = -0.0 (true -0.1053605156578263) |

## Shared code

- **`scipy.special.powm1`** returns 0.0 instead of +inf when x < 1, y < 0 and x**y overflows: `powm1(0.5, -1024) = 0.0`, `powm1(1e-300, -2) = 0.0`, `powm1(0.5, -1e20) = 0.0` (and `powm1(0.5, -1023) = 8.98846567431158e+307` is fine). The x > 1, y > 0 direction is correct (`powm1(2, 1024) = inf`). This turns the F4 overflow into a wrong-signed infinity. Any other distribution that calls `sc.powm1` with base < 1 and a negative exponent is affected.

## Existing GitHub issues

none found (web search only; `gh` unavailable). Related merged PRs on
`sc.powm1` overflow handling: #17855, #17872 (the case in Shared code,
x < 1 with y < 0 returning 0.0, still occurs in 1.18.1).

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-710, 0.001)` | 0.0 | 0.4916441974609651 | verified | All 7 table rows reproduced (cdf 0.0, logcdf -inf). Reference is the closed form exp(c*(x - log1p(e^x))), cross-checked by `mp.quad` of the pdf (agrees to 1e-30 or better at 200 digits). Mechanism matches lines 3129/3133. |
| F2 | `sf(-710, 1e-20)` | 1.0 | 7.0999999999999996e-18 | verified | All 5 rows reproduced; the reference `mp.quad` of the pdf over (x, inf) agrees with 1 - cdf to 1e-66. Mechanism (line 3139 gets -inf from logcdf) matches. |
| F3 | `logpdf(-1e300, 1e-20)` | -46.051701859880914 | -1.0e+280 | verified | All 6 rows reproduced, ulp identical. Reference uses log(c) + c*x - (c+1)*log1p(e^x). The value `-(1e-20 - 1) - 1` is exactly 0.0 in float64, which matches line 3124. |
| F4 | `ppf(1e-300, 0.41192440799679475)` | inf | -1676.9473099627268 | verified | All 4 rows reproduced. Reference is -log(expm1(-log(q)/c)), with residual cdf(x)/q - 1 < 1e-98. Confirmed `sc.powm1(0.5, -1024) = 0.0` and `powm1(2, 1024) = inf` (line 3136). |
| F5 | `isf(1e-17, 1)` | inf | 39.143946580898777 | verified | All 7 rows reproduced (232 ulp to 1.3e13 ulp, plus inf). isf(1e-300, 50): the reference 694.68755090364185 needs log1p(-q), because 1 - q rounds at 100 digits. It was cross-checked by `mp.quad` of the scaled tail integral. Mechanism matches line 3142. |
| F6 | `cdf(34.538776394910684, 1e15)` | 0.32948546950694785 | 0.36787944117144207 | verified | All 4 rows reproduced (4.7e14, 4.5e14, 4.8e5, 3.7e5 ulp). `1 + exp(-34.54)` rounds to 1 + 1.11e-15 (true 1 + 1e-15), and raising it to -1e15 gives e^-1.11 = 0.329. This matches line 3129. |
| F7 | `ppf(0.49999999999999994, 1)` | 0.0 | -2.2204460492503131e-16 | verified | isf row also reproduced. Confirmed `powm1(0.49999999999999994, -1.0) = 1.0`. The error is within the conditioning (cond 9e15), as the report states. |

All references were computed at 100 and 200 digits and agree to at least 30 digits; scripts are in `scratch/genlogistic/verify/`. No reference disagreements: every value matches the analyst's to the 16 to 17 digits printed.

Verified by dist-accuracy-verifier: 7 of 7 verified.
