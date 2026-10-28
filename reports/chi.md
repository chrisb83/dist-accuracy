# chi: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `chi_gen`, lines 1524-1602
- **Implemented:** pdf (1561-1565), logpdf (1567-1569), cdf (1571-1572), sf (1574-1575), ppf (1577-1578), isf (1580-1581)
- **Inherited (not measured):** logcdf, logsf (generic defaults: in SciPy 1.18.1 they switch at the median to `log1p(-sf)` / `log1p(-cdf)`, otherwise `log(cdf)` / `log(sf)`)
- **Related:** `chi2` and `gamma` (same incomplete-gamma code with a = df/2, y = x**2/2); helpers: none; special functions: `sc.gammaln`, `sc.xlogy`, `sc.gammainc`, `sc.gammaincc`, `sc.gammaincinv`, `sc.gammainccinv` (`sc.poch`, `sc.digamma`, `special.gamma` only in `_stats`/`_entropy`, not measured)
- **Reference:** `refs/chi_ref.py`, 400 digits; self-checks: PASS. Auxiliary grid for df = 2e8 (a = 1e8, where `mp.gammainc` does not converge; lower function from the fully converged Kummer series): `scratch/chi/chi_df2e8_ref.py`, 100 digits, run with `--no-checks` (the pdf-integral self-check did not finish in 25 min at a = 1e8; the same reference code passed all self-checks at df = 2e6)
- **Parameter sets:** (78), (1), (3), (0.001), (0.1), (1000), (2e6); plus (2e8) with the auxiliary reference (pdf, logpdf, cdf, sf only)
- **Worst finding:** loss in cdf, sf, ppf, isf (x**2 / quantile underflow for small df) and logpdf (x**2 overflow); severe in cdf/sf (relative error 0.375 at df = 2e8), pdf, logpdf, ppf, isf

## Summary

For moderate df and ordinary arguments every implemented method is good
(cdf/sf/ppf/isf within a few tens of ulp). Four mechanisms break it.
(1) All methods square x in double precision: `x**2` underflows for
x < ~1.5e-154, so for df <~ 2 cdf returns 0 (true 8e-201 at df = 1) and sf
returns 1.0 (true 0.37 at df = 0.001), and `gammaincinv` returns an
underflowed y = x**2/2 although the chi quantile sqrt(2y) is a normal double,
so ppf/isf return 0 over most of the distribution for tiny df
(`ppf(0.57, 0.001)` = 0, true 7.9e-245); `x**2` also overflows in logpdf for
1.34e154 < x < 1.9e154 (-inf for a finite result). (2) For df >~ 4e5 the
cephes `gammainc` returns an unconverged 2000-term series (shared defect,
reports/gamma.md F1): cdf off by 4.3e10 ulp at df = 2e6 and by 37.5 % at
df = 2e8, with sf, ppf and isf following. (3) `_logpdf` cancels terms of size
~(df/2) log df into an O(1) result, and `_pdf = exp(_logpdf)` turns that
absolute error into a relative one: 1.7e3 ulp at df = 1000, 1e7 ulp at
df = 2e6, 1e9 ulp at df = 2e8. (4) Inherited `gammaincinv` amplification for
tiny df (moderate) and the igam exp-prefactor (minor).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | df <~ 2 (sf: df <~ 0.05), x < ~1.5e-154 | `x**2` underflows to 0 / subnormal before `gammainc` (patterns 4, 10) | cdf 0.0 for 8.0e-201; sf 1.0 for 0.369 | loss |
| F2 | ppf, isf | df <~ 2 and quantile < ~1.5e-154 (tiny df: most of q in (0, 1)) | `gammaincinv` result y = x**2/2 underflows / is subnormal; `sqrt(2y)` would be normal (patterns 4, 8) | 0.0 for 7.9e-245; 6.3e10 ulp | loss |
| F3 | logpdf | any df, 1.34e154 < x < 1.9e154 | `.5*x**2` overflows although the result is a finite double (pattern 3) | -inf for -1.125e308 | loss |
| F4 | cdf, sf | df >~ 4e5, x**2/2 in a - (4.5 ... >15)*sqrt(a), a = df/2 | `sc.gammainc` returns an unconverged 2000-term series (shared, gamma F1; pattern 10) | cdf 1.7e15 ulp (rel. 0.375) at df = 2e8; 4.3e10 ulp at df = 2e6 | severe |
| F5 | pdf | large df near the mode (also tiny df far left tail) | `exp(_logpdf)` turns the absolute error of logpdf into relative error (patterns 3, 5) | 1.1e9 ulp at df = 2e8; 1.1e7 at df = 2e6 | severe |
| F6 | logpdf | df >~ 1e3, x within a few units of sqrt(df) | cancellation of `.5*log(2)*df`, `gammaln(.5*df)`, `xlogy(df-1, x)`, `.5*x**2` (pattern 5) | 8.0e8 ulp at df = 2e8; 8.8e6 at df = 2e6 | severe |
| F7 | ppf, isf | df >~ 4e5, quantile in the F4 window | `gammaincinv`/`gammainccinv` invert the inaccurate `gammainc` of F4 (shared, gamma F3) | isf 2.7e6 ulp, ppf 5.2e5 ulp at df = 2e6 | severe |
| F8 | ppf, isf | df << 1 (0.001), bulk q where the quantile is still normal | `gammaincinv` for a << 1 rounds an exponent scaled by 1/a (shared, gamma F5) | 2.98e3 ulp | moderate |
| F9 | cdf | df ~ 1-3, x < 1e-10 (and above the F1 range) | `exp(a*log(y) - y - lgam(a))` prefactor in cephes `igam` (shared, gamma F6) | 187 ulp | minor |

### F1: cdf and sf square x and lose it to underflow for small df

- **Code:** `_cdf`, line 1572: `sc.gammainc(.5*df, .5*x**2)`; `_sf`, line 1575: `sc.gammaincc(.5*df, .5*x**2)`.
- **Mechanism:** patterns 4 and 10. `x**2` is 0 for x < ~1.5e-162 and
  subnormal (fewer than 53 significant bits) for x < ~1.5e-154. Then
  `gammainc(a, 0) = 0` and `gammaincc(a, 0) = 1`, although
  `cdf ~ x**df / (2**(df/2) Gamma(df/2+1))` is still a normal double for
  df <~ 2 (for df = 0.001 the cdf is 0.5-0.69 there). In the subnormal range
  the relative error of `x**2` (up to 2**-(bits left)) is multiplied by
  a = df/2 in `y**a`, giving 1e5-1e10 ulp.
- **Regime:** x < ~1.5e-154 and df small enough that cdf(x) is a normal
  double, i.e. df <~ 2 (df = 3: cdf(1.5e-154) ~ 1e-462, not measurable).
  sf is visibly wrong only when cdf(x) > eps, i.e. df <~ 0.05; for df = 0.001
  sf returns 1.0 instead of 0.31-0.50 on the whole range x < 1e-162.
- **Reproducer:** `python accuracy.py refs/chi_ref.py --at cdf 1e-200 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-200, 1)` | 0.0 | 7.978845608028653e-201 | inf | 1 | loss |
  | `cdf(1e-200, 0.001)` | 0.0 | 0.6309206419232786 | inf | 0.001 | loss |
  | `cdf(1e-300, 0.1)` | 0.0 | 9.92226087850207e-31 | inf | 0.1 | loss |
  | `cdf(1e-160, 1)` | 7.978801194389789e-161 | 7.978845608028654e-161 | 2.5e+10 | 1 | severe |
  | `cdf(1e-160, 0.1)` | 9.922255355336987e-17 | 9.922260878502088e-17 | 2.5e+09 | 0.1 | severe |
  | `sf(1e-200, 0.001)` | 1.0 | 0.3690793580767215 | 7.7e+15 | 0.00171 | loss |
  | `sf(1e-160, 0.001)` | 0.3082092764852745 | 0.30820927263446324 | 5.6e+07 | 0.00224 | severe |
  | `sf(1e-158, 0.001)` | 0.305016111743252 | 0.3050161117204055 | 3.4e+05 | 0.00228 | moderate |

  (cond (shapes) for df = 1 is 369-691: the cdf is sensitive to df here, but
  df is an exact user input.)
- **Fix direction:** for x < ~1e-150 use the leading term in log space,
  `logcdf = df*log(x) - (df/2)*log(2) - gammaln(df/2 + 1)` (the series
  correction is O(x**2) < 1e-300), `cdf = exp(logcdf)`, `sf = -expm1(logcdf)`;
  or avoid forming `x**2` alone (e.g. pass `(x*sqrt(.5))**2` only where it is
  normal).

### F2: ppf and isf compute the quantile of x**2/2 and underflow

- **Code:** `_ppf`, line 1578: `np.sqrt(2*sc.gammaincinv(.5*df, q))`;
  `_isf`, line 1581: `np.sqrt(2*sc.gammainccinv(.5*df, q))`.
- **Mechanism:** patterns 4 and 8. The gamma quantile y = x**2/2 is the
  square of the result, so it leaves the normal range when x < ~1.5e-154:
  y underflows to 0 (x < ~1e-162) or is subnormal, and `sqrt(2y)` returns 0
  or a value with only a few correct digits, although x itself is a normal
  double. For tiny df the quantile is tiny over most of (0, 1):
  ppf(q) ~ sqrt(2) * (q*Gamma(df/2+1))**(1/df), which is below 1.5e-154 for
  q < ~0.7 at df = 0.001.
- **Regime:** df <~ 2 and ppf(q) < ~1.5e-154: df = 1, q < ~1e-162;
  df = 0.1, q < ~1e-15; df = 0.001, q < ~0.7 for ppf and q > ~0.3 for isf
  (the whole lower part of the distribution).
- **Reproducer:** `python accuracy.py refs/chi_ref.py --at ppf 1e-200 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-200, 1)` | 0.0 | 1.2533141373155002e-200 | inf | 1 | loss |
  | `ppf(0.57, 0.001)` | 0.0 | 7.945462051175789e-245 | inf | 1e+03 | loss |
  | `ppf(0.5000000000000001, 0.001)` | 0.0 | 9.891629048214158e-302 | inf | 1e+03 | loss |
  | `ppf(1e-17, 0.1)` | 0.0 | 1.0811690022643788e-170 | inf | 10 | loss |
  | `ppf(1e-16, 0.1)` | 1.0811842202235423e-160 | 1.0811690022643765e-160 | 6.3e+10 | 10 | severe |
  | `isf(0.36, 0.001)` | 0.0 | 1.6041222735699583e-194 | inf | 562 | loss |
  | `isf(0.42999999999999994, 0.001)` | 0.0 | 7.945462051177337e-245 | inf | 754 | loss |

- **Fix direction:** where `gammaincinv` returns y < ~1e-300, use the
  log-space leading term `x = sqrt(2)*exp((log(q) + gammaln(df/2 + 1))/df)`
  (isf: `log1p(-q)` in place of `log(q)`), optionally polished by one Newton
  step on logcdf.

### F3: logpdf overflows in x**2 for 1.34e154 < x < 1.9e154

- **Code:** `_logpdf`, line 1569: `l + sc.xlogy(df - 1., x) - .5*x**2`.
- **Mechanism:** pattern 3. `x**2` overflows for x > 1.34e154, so
  `-.5*x**2 = -inf`, but the true logpdf ~ -x**2/2 stays above -1.8e308
  until x ~ 1.9e154.
- **Regime:** all df, 1.34e154 < x < 1.9e154 (narrow, but total loss).
- **Reproducer:** `python accuracy.py refs/chi_ref.py --at logpdf 1.5e154 78`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.35e154, 78)` | -inf | -9.1125e+307 | inf | 2 | loss |
  | `logpdf(1.5e154, 78)` | -inf | -1.1250000000000002e+308 | inf | 2 | loss |
  | `logpdf(1.8e154, 2e6)` | -inf | -1.62e+308 | inf | 2 | loss |
  | `logpdf(1.34e154, 78)` | -8.978e+307 | -8.978e+307 | 0 | 2 | ok |

- **Fix direction:** `- (.5*x)*x` (or `-.5*x*x` evaluated as `x*(-.5*x)`),
  which overflows only when the result does.

### F4: cdf and sf inherit the unconverged gammainc series for df >~ 4e5

- **Code:** `_cdf`, line 1572 (`sc.gammainc`) and `_sf`, line 1575
  (`sc.gammaincc`, computed as `1 - igam` for y < a). The defect is in
  `scipy.special` and is analysed in reports/gamma.md F1 and
  issue-reports/gammainc-large-a.md: cephes `igam_series` stops after 2000
  terms and returns the partial sum just outside the Temme window
  `|y - a| < 4.5*sqrt(a)`.
- **Mechanism:** pattern 10 (shared). chi calls `gammainc(a, y)` with
  a = df/2, y = x**2/2, so it hits exactly that window; nothing chi-specific.
  In chi variables the window is
  `x**2 < df - 4.5*sqrt(2*df)`, i.e. about 3.2 to >10 standard deviations
  (sd ~ 0.71) below the mode sqrt(df).
- **Regime:** df >~ 4e5 (a >~ 2e5), left tail between ~ -3 and ~ -10 sd
  (the window widens with df); the upper tail (continued fraction) is fine.
- **Reproducer:** `python accuracy.py refs/chi_ref.py --at cdf 1410.6735979665884 2e6`;
  df = 2e8: `python accuracy.py scratch/chi/chi_df2e8_ref.py --no-checks`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1410.9571219565817, 2e6)` | 2.045146368700871e-06 | 2.0451657660633962e-06 | 4.27e+10 | 9.6e+03 | severe |
  | `cdf(1410.6735979665884, 2e6)` | 2.7495684457992286e-07 | 2.7495803592698485e-07 | 2.0e+10 | 1.04e+04 | severe |
  | `cdf(1409.2551223962253, 2e6)` | 1.1410046126992977e-12 | 1.1410047078304667e-12 | 3.75e+08 | 1.43e+04 | severe |
  | `cdf(1407.1247279470288, 2e6)` | 5.44664469186744e-24 | 5.446644693002451e-24 | 9.38e+05 | 2.02e+04 | moderate |
  | `sf(1410.9571219565817, 2e6)` | 0.9999979548536313 | 0.9999979548342339 | 8.74e+04 | 0.0196 | moderate |
  | `cdf(14138.882558391946, 2e8)` | 1.3154316493273995e-06 | 2.1056486712300553e-06 | 1.7e+15 | 9.6e+04 | severe |
  | `cdf(14135.062787267696, 2e8)` | 6.413019934564911e-24 | 7.369931066919224e-24 | 5.8e+14 | 2.0e+05 | severe |
  | `cdf(14120.906486483083, 2e8)` | 1.9861345556856892e-198 | 1.990882289706629e-198 | 1.1e+13 | 6.0e+05 | severe |
  | `sf(14138.882558391946, 2e8)` | 0.9999986845683507 | 0.9999978943513288 | 3.6e+09 | 0.202 | severe |

  The df = 2e8 values equal the gamma report's a = 1e8 values
  (`cdf(99954000, 1e8)` = 1.3154316493365769e-06 there), confirming the
  identical code path.
- **Fix direction:** in `scipy.special` (see reports/gamma.md F1); nothing
  in `chi_gen` short of not calling `gammainc`.

### F5: pdf exponentiates a logpdf with a large absolute error

- **Code:** `_pdf`, line 1565: `np.exp(self._logpdf(x, df))`.
- **Mechanism:** patterns 3 and 5. `exp` converts an absolute error of its
  argument into a relative error of the result. The terms of `_logpdf`
  (F6) are each rounded at their own magnitude, so the pdf's relative error
  is about `max(.5*log(2)*df, gammaln(df/2), |(df-1)*log(x)|, x**2/2)` ulp:
  ~1e7 at df = 2e6 (terms ~1.3e7), ~1e9 at df = 2e8. The same effect
  appears, minor, in the far left tail of small df where
  `|(df-1)*log(x)|` ~ 600 (659 ulp at df = 0.1, x = 1e-280).
- **Regime:** all x for large df (worst near the mode, where the pdf is
  largest); 1.7e3 ulp at df = 1000, 124 ulp at df = 78.
- **Reproducer:** `python accuracy.py refs/chi_ref.py --at pdf 1414.213562373095 2e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(14140.014144264496, 2e8)` | 0.006262878572390661 | 0.006262880074963077 | 1.1e+09 | 6e+04 | severe |
  | `pdf(14142.13562373095, 2e8)` | 0.5641895256490528 | 0.5641895830775984 | 4.6e+08 | 1 | severe |
  | `pdf(1419.1546779685434, 2e6)` | 1.4424161903720707e-11 | 1.442416186909687e-11 | 1.1e+07 | 1.4e+04 | severe |
  | `pdf(1414.213562373095, 2e6)` | 0.5641895371431063 | 0.5641895365319596 | 4.9e+06 | 1 | severe |
  | `pdf(31.622776601683793, 1000)` | 0.5640955597983511 | 0.564095559798565 | 1.71e+03 | 1 | moderate |
  | `pdf(1e-280, 0.1)` | 9.922260878503525e+250 | 9.922260878502073e+250 | 659 | 0.9 | minor |

- **Fix direction:** chi pdf = `2*igam_fac(a, y)/x` with a = df/2,
  y = x**2/2, i.e. the Lanczos/`log1pmx` form recommended in
  reports/gamma.md F2 (`a*log1pmx((y-a)/a)` plus O(log a) terms); subtracting
  a Stirling approximation from `gammaln` is not enough (gamma F2).

### F6: logpdf cancels terms of size (df/2) log df near the mode

- **Code:** `_logpdf`, lines 1568-1569:
  `l = np.log(2) - .5*np.log(2)*df - sc.gammaln(.5*df)`;
  `l + sc.xlogy(df - 1., x) - .5*x**2`.
- **Mechanism:** pattern 5. Near x = sqrt(df) the result is about
  -0.57 while `xlogy(df-1, x)` ~ (df/2) log df (1.45e7 at df = 2e6,
  1.9e9 at df = 2e8) and `gammaln(df/2)` is of the same size; each term
  carries half an ulp of its own magnitude, so the relative error is
  ~ (df/2) log(df) * eps / |logpdf|.
- **Regime:** df >~ 1e3 and |logpdf| small (x within a few sd of
  sqrt(df)); 2.98e3 ulp at df = 1000, 8.8e6 at df = 2e6, 8.0e8 at df = 2e8.
  At df = 78 the effect is 116 ulp (minor).
- **Reproducer:** `python accuracy.py refs/chi_ref.py --at logpdf 1414.213562373095 2e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(14142.13562373095, 2e8)` | -0.5723650455474854 | -0.5723649437580334 | 8.0e+08 | 1.75 | severe |
  | `logpdf(1414.2132088196602, 2e6)` | -0.5723649001447484 | -0.5723649012580125 | 8.8e+06 | 4.66e-10 | severe |
  | `logpdf(1414.213562373095, 2e6)` | -0.572365025174804 | -0.5723650262580334 | 8.5e+06 | 1.75 | severe |
  | `logpdf(1414.9204924659196, 2e6)` | -1.072531690937467 | -1.0725316928915611 | 8.2e+06 | 1.87e+03 | severe |
  | `logpdf(31.622776601683793, 1000)` | -0.5725316095695234 | -0.5725316095691445 | 2.98e+03 | 1.75 | moderate |
  | `logpdf(8.831760866327848, 78)` | -0.5745016482424674 | -0.5745016482424526 | 116 | 1.74 | minor |

- **Fix direction:** same as F5 (log-space Lanczos prefactor:
  `log(2) - log(x) + log(igam_fac(a, y))`, with `a*log1pmx((y-a)/a)` as the
  only large term).

### F7: ppf and isf inherit F4 through gammaincinv / gammainccinv

- **Code:** `_ppf`, line 1578; `_isf`, line 1581.
- **Mechanism:** shared (reports/gamma.md F3): the inverse functions
  Newton-iterate on the inaccurate `igam` of F4 and converge to the wrong
  root; the error in y is then halved (relative) by the square root.
- **Regime:** df >~ 4e5 with the quantile in the F4 window: at df = 2e6
  ppf for q in ~[1e-51, 2e-6], isf for 1 - q in the same range.
- **Reproducer:** `python accuracy.py refs/chi_ref.py --at isf 0.9999993371296838 2e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.9999993371296838, 2e6)` | 1410.7952403649585 | 1410.7952395125612 | 2.7e+06 | 150 | severe |
  | `isf(0.999999938945977, 2e6)` | 1410.474162412369 | 1410.4741620902055 | 1.0e+06 | 1.5e+03 | severe |
  | `ppf(1e-08, 2e6)` | 1410.2469003793349 | 1410.2469002172074 | 5.2e+05 | 8.67e-05 | moderate |
  | `ppf(3.4807005884285624e-14, 2e6)` | 1408.9213980967766 | 1408.9213980937416 | 9.7e+03 | 6.58e-05 | moderate |

- **Fix direction:** none inside `chi_gen`; fixing gamma F1 fixes it.

### F8: ppf/isf for tiny df amplify the gammaincinv exponent rounding

- **Code:** `_ppf`, line 1578 / `_isf`, line 1581 (inside `sc.gammaincinv`).
- **Mechanism:** shared (reports/gamma.md F5): for a << 1, cephes `igami`
  returns y = exp(exponent/a) with |log y| in the hundreds, and the rounding
  of the exponent is multiplied by |log y|; the square root halves it.
- **Regime:** df <~ 0.01, q where the quantile is still normal
  (above the F2 range): q ~ 0.75-0.9 at df = 0.001.
- **Reproducer:** `python accuracy.py refs/chi_ref.py --at ppf 0.9 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9, 0.001)` | 1.852562738253096e-46 | 1.8525627382543217e-46 | 2.98e+03 | 1e+03 | moderate |
  | `ppf(0.8499999999999999, 0.001)` | 2.7809245081893525e-71 | 2.780924508190853e-71 | 2.43e+03 | 1e+03 | moderate |
  | `ppf(0.7799999999999999, 0.001)` | 1.3178505911521412e-108 | 1.317850591152817e-108 | 2.31e+03 | 1e+03 | moderate |

- **Fix direction:** in `scipy.special` `igami` (final Newton step on log P),
  see gamma F5; the log-space formula of F2 also covers this regime.

### F9: cdf for small x loses ~|a log y| ulp in the igam prefactor

- **Code:** `_cdf`, line 1572; the loss is in `igam_fac` inside
  `sc.gammainc` (`exp(a*log(y) - y - lgam(a))`).
- **Mechanism:** shared (reports/gamma.md F6): the exponent is a sum of
  terms of size |a log y| (~230 for df = 1, x = 1e-100), so its rounding
  becomes a relative error of that many ulp.
- **Regime:** small df (1-3), x from ~1e-10 down to the F1 range.
- **Reproducer:** `python accuracy.py refs/chi_ref.py --at cdf 1e-100 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-90, 3)` | 2.659615202676328e-271 | 2.6596152026762176e-271 | 187 | 3 | minor |
  | `cdf(1e-120, 1)` | 7.978845608028435e-121 | 7.978845608028654e-121 | 123 | 1 | minor |
  | `cdf(1e-100, 1)` | 7.978845608028499e-101 | 7.978845608028654e-101 | 87.4 | 1 | minor |

- **Fix direction:** the log-space leading-term formula of F1 for small x;
  in general the `scipy.special` fix of gamma F6.

## RuntimeWarnings

- `overflow encountered in square` (logpdf, pdf, cdf, sf) for
  x > 1.34e154: wrong result in logpdf for x < 1.9e154 (F3); elsewhere the
  true value is 0, -huge (not a double) or 1 and the warning is harmless.
- `underflow encountered in square` / `in multiply` (all x-methods) for
  x < 1.5e-154: accompanies the wrong results of F1 in cdf/sf; in logpdf
  and pdf the underflowed `x**2/2` is negligible and the result is accurate
  (spurious warning).
- `underflow encountered in exp` (pdf): genuine underflow of the pdf.
- ppf/isf emit no warning in F2: the underflow happens silently inside
  `gammaincinv`.
- The harness evaluates whole grids per call, so the warnings are attributed
  to every point of a parameter set, including accurate ones.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | bulk and high tail for all parameter sets; low tail for df = 78, 1000 | 81.3 ulp (df = 78, cond 39); high tail max 3.5 ulp |
| sf | whole grid for df = 1, 3, 0.1, 78, 1000 (left tail outside F1) | 76 ulp (df = 78, cond 190); 954 ulp at df = 1000 within cond 1.35e3; minor 41.8 ulp at `sf(1, 1)` (cond 1.53, no clear mechanism) |
| sf | df = 2e8 upper tail, x up to sqrt(df) + 10 sd | 1.83e4 ulp at `sf(14149.204924659194, 2e8)`; the harness did not compute cond there, but x*pdf/sf = 2.0e5, so this is within conditioning (rounding of x**2) |
| ppf, isf | q in [1e-300, 1 - 1e-16] for df = 78, 1000, 3; isf for df = 1 | ppf 7.7 ulp (df = 78), 88 ulp (df = 3, q ~ 1e-288, minor, no clear mechanism; probably the igami analogue of F9), 194 ulp (df = 0.1, q = 3.5e-14, minor, edge of F8); isf 45 ulp (df = 1, q = 0.15, minor) |
| logpdf | low and high tails for all df (outside F3) | 0.5 ulp median; F6 only near the mode |
| pdf | df = 1 (xlogy(0, x) exact) | 3.07 ulp |

## Suspected, not confirmed

- df >= ~2e10 (a >= 1e10): the reference series becomes too long to
  evaluate; by the gamma F1 extrapolation the cdf error would exceed 0.9
  relative (loss). Not measured.
- ppf/isf at df = 2e8 were not measured (each inversion needs dozens of
  Kummer-series cdf evaluations with ~1e5 terms; the run did not finish in
  25 min). By F7 they are expected to be worse than at df = 2e6.
- x = 0 exactly (pdf = inf for df < 1, sqrt(2/pi) for df = 1, 0 for df > 1)
  is not on the harness grids.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | `log(sf)` below the median, `log1p(-cdf)` above it | log-space upper incomplete gamma: `logsf = a*log(y) - y - gammaln(a) + log(continued fraction)` with a = df/2, y = x**2/2 (asymptotically `-x**2/2 + (df-2)*log(x) - (df/2-1)*log(2) - gammaln(df/2) + log1p(...)`) | sf underflows once x**2/2 >~ 745: `logsf(60, 78)` returns -inf, true -1618.1162798350102 (loss) |
| logcdf | `log(cdf)` below the median, `log1p(-sf)` above it | `logcdf = df*log(x) - (df/2)*log(2) - gammaln(df/2 + 1) + log(Kummer series)` (series ~1 for y << a) | cdf underflows in the left tail: `logcdf(1e-10, 78)` returns -inf, true -1929.680872837837; and, through F1, `logcdf(1e-200, 1)` returns -inf, true -460.74280995145386 (loss) |

## Shared code

- `sc.gammainc` / `sc.gammaincc` (cephes `igam`/`igamc`): F4 is reports/gamma.md
  F1 (unconverged 2000-term `igam_series` for a >~ 2e5 just outside the
  Temme window; see issue-reports/gammainc-large-a.md) reached through
  a = df/2, y = x**2/2, so chi is affected for df >~ 4e5; F9 is gamma F6
  (exp-prefactor rounding). The df = 2e8 values match gamma's a = 1e8 values
  digit for digit.
- `sc.gammaincinv` / `sc.gammainccinv` (cephes `igami`): F7 is gamma F3,
  F8 is gamma F5.
- `chi2` and `gamma` share F4/F7 directly; `chi`-specific are F1, F2, F3
  (the double-precision `x**2`) and F5/F6 (the chi `_logpdf`, same
  cancellation pattern as gamma F2/F4).

## Existing GitHub issues

- #26471 (open): BUG: inaccuracy in special.gammainc for certain large parameters - relates to F4 and F7 (the same cephes `igam_series` cap reached via a = df/2, y = x**2/2)
- F1, F2, F3, F5, F6, F8, F9: none found (GitHub search API for "stats.chi" issues and web search; `gh` was not available)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e-200, 1)`; `cdf(1e-160, 1)`; `sf(1e-200, 0.001)` | 0.0; 7.978801194389789e-161; 1.0 | 7.9788456080286534e-201; 7.9788456080286535e-161; 0.36907935807672149 | verified | Reference `y**a/Gamma(a+1) * 1F1(a; a+1; -y)` (a = df/2, y = x**2/2), not `mp.gammainc`; all 7 report calls reproduced (2.5e10, 2.5e9, 5.6e7 ulp for the non-zero ones). `1e-200**2` = 0.0 and `1e-160**2` = 1e-320 (subnormal) in double, matching the mechanism at lines 1572/1575 |
| F2 | `ppf(0.57, 0.001)`; `ppf(1e-16, 0.1)`; `isf(0.36, 0.001)` | 0.0; 1.0811842202235423e-160; 0.0 | 7.9454620511757885e-245; 1.0811690022643765e-160; 1.6041222735699584e-194 | verified | Root of my 1F1 cdf in log x, with cdf(x_ref) = q checked to 1e-40; all 7 report calls reproduced (6.3e10 ulp for the non-zero one). `gammaincinv(0.0005, 0.57)` = 0.0 and `gammaincinv(0.05, 1e-16)` = 5.8e-321 (subnormal), as described (line 1578/1581) |
| F3 | `logpdf(1.35e154, 78)`; `logpdf(1.8e154, 2e6)` | -inf; -inf | -9.1125000000000004e+307; -1.6200000000000001e+308 | verified | Reference via chi2 relation `log(2x) + logpdf_chi2(x**2)`; `1.35e154**2` = inf in double (line 1569) |
| F4 | `cdf(1410.9571219565817, 2e6)`; `cdf(14138.882558391946, 2e8)`; `cdf(14120.906486483083, 2e8)`; `sf(14138.882558391946, 2e8)` | 2.045146368700871e-06; 1.3154316493273995e-06; 1.9861345556856892e-198; 0.9999986845683507 | 2.045165766063396e-06; 2.1056486712300555e-06; 1.9908822897066292e-198; 0.99999789435132877 | verified | cdf by `mp.quad` of the docstring pdf (for the 1.99e-198 point: quadrature in u = x - t with the pdf ratio `exp((df-1)log1p(-u/x) + xu - u**2/2)`, agreeing with the converged lower series to 35 digits); 4.3e10, 1.7e15, 1.1e13, 3.6e9 ulp, as reported. Mechanism is in `sc.gammainc` (shared), not re-examined in the C source |
| F5 | `pdf(14140.014144264496, 2e8)`; `pdf(1414.213562373095, 2e6)`; `pdf(31.622776601683793, 1000)` | 0.006262878572390661; 0.5641895371431063; 0.5640955597983511 | 0.0062628800749630771; 0.5641895365319596; 0.56409555979856491 | verified | 1.1e9, 4.9e6, 1.7e3 ulp, as reported; line 1565 is `exp(self._logpdf)` |
| F6 | `logpdf(14142.13562373095, 2e8)`; `logpdf(1414.2132088196602, 2e6)`; `logpdf(31.622776601683793, 1000)` | -0.5723650455474854; -0.5723649001447484; -0.5725316095695234 | -0.57236494375803339; -0.57236490125801259; -0.57253160956914453 | verified | 8.0e8, 8.8e6, 3.0e3 ulp, as reported; cancelling terms at lines 1568-1569 as described |
| F7 | `isf(0.9999993371296838, 2e6)`; `isf(0.999999938945977, 2e6)`; `ppf(1e-08, 2e6)` | 1410.7952403649585; 1410.474162412369; 1410.2469003793349 | 1410.7952395125611; 1410.4741620902056; 1410.2469002172075 | verified | Newton on my quadrature cdf, cdf(x_ref) = q checked to 1e-40; 2.7e6, 1.0e6, 5.2e5 ulp, as reported |
| F8 | `ppf(0.9, 0.001)`; `ppf(0.7799999999999999, 0.001)` | 1.852562738253096e-46; 1.3178505911521412e-108 | 1.8525627382543215e-46; 1.3178505911528171e-108 | verified | 3.0e3 and 2.3e3 ulp, as reported; mechanism is inside `sc.gammaincinv` (shared, gamma F5), not re-examined |

All verifier references were computed at 100 and 200 digits and agree to at least 30 digits (scripts: `scratch/chi/verify/verify.py`, `case_2e8_far.py`, `case_2e8_far_quad.py`).

Verified by dist-accuracy-verifier: 8 of 8 verified.
