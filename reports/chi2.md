# chi2: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `chi2_gen`, lines 1608-1693
- **Implemented:** pdf (1648-1650), logpdf (1652-1653), cdf (1655-1656), sf (1658-1659), isf (1661-1662), ppf (1664-1665)
- **Inherited (not measured):** logcdf, logsf (generic `rv_continuous` versions: log(cdf) / log(sf) on one side of the median, `log1p(-sf)` / `log1p(-cdf)` on the other)
- **Related:** `ncx2` (docstring cross-reference, separate code); chi2(df) is gamma(a = df/2, scale = 2) and its cdf/sf/ppf/isf are **bitwise identical** to `gamma` at (a = df/2, x/2) (checked on 2600 points per df, `scratch/chi2/probe_equiv.py`); helpers: none; special functions: `sc.xlogy`, `sc.gammaln`, `sc.chdtr`, `sc.chdtrc`, `sc.chdtri`, `sc.gammaincinv` (`sc.psi` only in `_entropy`)
- **Reference:** `refs/chi2_ref.py`, 400 digits; self-checks: PASS. Auxiliary reference for df = 2e7 and 2e8 (where mpmath's `gammainc` does not converge): `scratch/chi2/chi2_df2e8_ref.py` (Kummer series summed to convergence, 100 digits, cdf/sf/pdf/logpdf only), self-checks PASS
- **Parameter sets:** (55), (0.002), (1), (2), (3.9864610967601556), (2000), (2e6); plus (2e7), (2e8) with the auxiliary reference
- **Worst finding:** severe in cdf (relative error 0.375, 1.7e15 ulp, at df = 2e8), propagating to sf, ppf and isf; severe in pdf (1.5e9 ulp at df = 2e8) and logpdf (1.0e8 ulp at df = 2e8)

## Summary

For small and moderate df (up to ~2e3) chi2 is accurate: cdf, sf, ppf and
isf stay within a few hundred ulp (mostly within a few ulp of the
conditioning), and logpdf is within 11 ulp for df <= 55. Because `chdtr`,
`chdtrc`, `chdtri` and `gammaincinv` are the cephes incomplete-gamma code,
chi2 inherits every `scipy.special` defect documented in `reports/gamma.md`
with a = df/2: the 2000-term cap of `igam_series` makes cdf wrong for
df >~ 5e5 just below `df - 4.5*sqrt(2 df)` (4.3e-6 relative at df = 2e6,
0.375 relative at df = 2e8), and sf, ppf and isf inherit that. This confirms
the prediction in `issue-reports/gammainc-large-a.md` (onset between
df = 4e5 and 6e5). Independently, `_logpdf` sums four terms of size
`(df/2) log df` into an O(log df) result, and `_pdf = exp(_logpdf)`
turns that absolute error into relative error: 5e3 ulp at df = 2e3, 6.3e6 ulp
at df = 2e6, 1.5e9 ulp at df = 2e8. Smaller effects are amplification by 1/a
in `gammaincinv`/`igamci` for tiny df (3.6e3 ulp at df = 2e-3) and rounding
of the exponent in the `igam`/`igamc` prefactor (3.4e3 ulp at df = 2e3).
Nothing returns 0/inf/nan in the measured range. The inherited logcdf and
logsf return -inf in the tails.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | df >~ 5e5 and 4.5*sqrt(2 df) < df - x <~ 15..30*sqrt(2 df) | `sc.chdtr` = cephes `igam(df/2, x/2)` returns the 2000-term partial sum of an unconverged series (pattern 10, shared with gamma F1) | cdf 1.7e15 ulp (rel. 0.375) at df = 2e8; sf 3.6e9 ulp | severe |
| F2 | pdf | all df; grows with df, worst for large df near the mean | `exp(_logpdf)` turns the absolute error of the cancelling logpdf sum into relative error (patterns 5, 3) | 1.5e9 ulp at df = 2e8; 6.3e6 ulp at df = 2e6 | severe |
| F3 | logpdf | df >~ 2e3, x within a few sqrt(2 df) of df | `xlogy(df/2-1,x) - x/2 - gammaln(df/2) - log(2)*df/2`: terms ~(df/2)log df cancel to O(log df) (pattern 5) | 1.0e8 ulp at df = 2e8; 4.0e5 ulp at df = 2e6 | severe |
| F4 | ppf, isf | df >~ 5e5, quantile in the F1 window (ppf: small q; isf: q -> 1) | `gammaincinv`/`chdtri` invert the inaccurate `igam` of F1 (pattern 10) | 6.5e6 ulp at df = 2e6 (both) | severe |
| F5 | ppf, isf | df << 1 (df = 2e-3), bulk q | inside `igami`/`igamci` the result is `exp(./a)` with a = df/2, exponent ~ 700 (patterns 3, 8; shared with gamma F5) | 3.58e3 ulp | moderate |
| F6 | cdf, sf | `|x/2 - df/2| > 0.4 df/2` with a large exponent: df = 2e3 at x = 1e3 or 4e3; tiny x for any df | `exp(a log y - y - lgam(a))` prefactor in `igam`/`igamc` (pattern 5; shared with gamma F6) | sf 3.35e3 ulp, cdf 2.87e3 ulp at df = 2e3 | moderate |

### F1: cdf (and sf) return an unconverged series for large df below the mean

- **Code:** `_cdf`, line 1656: `return sc.chdtr(df, x)`; `_sf`, line 1659:
  `return sc.chdtrc(df, x)`. `chdtr(df, x)` is `igam(df/2, x/2)` (bitwise
  equal to `sc.gammainc(df/2, x/2)` at every tested point), so this is gamma's
  F1 at a = df/2, y = x/2; see `reports/gamma.md` F1 and
  `issue-reports/gammainc-large-a.md` for the full mechanism.
- **Mechanism:** pattern 10. For a = df/2 > 200, `igam` uses the uniform
  asymptotic expansion only for `|y - a| < 4.5*sqrt(a)`, i.e.
  `|x - df| < 4.5*sqrt(2 df)`. Just below that window it evaluates the Kummer
  series in `igam_series`, whose loop is capped at 2000 iterations; for large a
  the series needs 1e4 to 1e5 terms there, and the partial sum is returned
  silently. sf is computed as `1 - igam` for x < df and inherits the error
  (smaller in ulp because sf is close to 1).
- **Regime:** df >~ 5e5 (onset measured in `scratch/chi2/probe_onset.py` at
  x = df - 5*sqrt(2 df): 27 ulp at df = 4e5, 3.7e4 ulp at df = 6e5, 3.7e7 ulp at
  df = 1e6, 2.0e10 ulp at df = 2e6), for x between `df - 4.5*sqrt(2 df)` and
  about `df - 15*sqrt(2 df)` (df = 2e6) or beyond `df - 30*sqrt(2 df)`
  (df = 2e8: still 1.1e13 ulp at x = df - 30*sqrt(2 df)). The window widens
  with df. Only the lower side is affected: the upper side (continued fraction)
  is within 34 ulp up to `df + 10*sqrt(2 df)` at df = 2e8.
  **This confirms the prediction in `issue-reports/gammainc-large-a.md`**
  that `chdtr`/`chdtrc` are affected for df >~ 4e5. The actual onset is
  between df = 4e5 and 6e5.
- **Reproducer:** `python accuracy.py refs/chi2_ref.py --at cdf 1990800 2e6`;
  `python accuracy.py scratch/chi2/chi2_df2e8_ref.py --at cdf 199908000 2e8`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(199908000, 2e8)` | 1.3154316493365769e-06 | 2.105648671238922e-06 | 1.7e+15 | 4.8e+04 | severe |
  | `cdf(199900000, 2e8)` | 1.8630998932585132e-07 | 2.854642139958626e-07 | 1.6e+15 | 5.19e+04 | severe |
  | `cdf(19970907.045526452, 2e7)` | 2.008512445946794e-06 | 2.0909909828073737e-06 | 1.8e+14 | 1.52e+04 | severe |
  | `cdf(1990800, 2e6)` | 2.045146368700871e-06 | 2.0451657660631527e-06 | 4.3e+10 | 4.8e+03 | severe |
  | `cdf(1990000, 2e6)` | 2.7495684457992286e-07 | 2.749580359270071e-07 | 2.0e+10 | 5.19e+03 | severe |
  | `cdf(1986000, 2e6)` | 1.1410046127002487e-12 | 1.1410047078311216e-12 | 3.8e+08 | 7.14e+03 | severe |
  | `cdf(1980000, 2e6)` | 5.446644691873852e-24 | 5.446644693010809e-24 | 9.4e+05 | 1.01e+04 | moderate |
  | `sf(199908000, 2e8)` | 0.9999986845683506 | 0.9999978943513288 | 3.6e+09 | 0.101 | severe |
  | `sf(19970907.045526452, 2e7)` | 0.9999979914875541 | 0.9999979090090172 | 3.7e+08 | 0.0317 | severe |
  | `sf(1990800, 2e6)` | 0.9999979548536313 | 0.9999979548342339 | 8.74e+04 | 0.00982 | moderate |

- **Fix direction:** in `scipy.special` (cephes `igam`): widen the uniform
  asymptotic window for large a, or remove the iteration cap of `igam_series`.
  This fixes gamma, chi2 and the other dependents together. Nothing can be done
  in `chi2_gen` short of not calling `chdtr`.

### F2: pdf exponentiates a logpdf with absolute error ~(df/2) log(df) eps

- **Code:** `_pdf`, line 1650: `return np.exp(self._logpdf(x, df))`.
- **Mechanism:** patterns 5 and 3. `exp` turns the *absolute* error of
  `_logpdf` (F3) into a *relative* error of the pdf. Each of the four terms of
  `_logpdf` is rounded to its own ulp. At df = 2e8, x = 2e8 the terms are
  `xlogy(1e8-1, 2e8) = 1.91e9`, `gammaln(1e8) = 1.74e9`, `x/2 = 1e8` and
  `log(2)*1e8 = 6.9e7`, while the result is -10.8. One rounding of 1.9e9 is
  2.4e-7 absolute, which is 1e9 ulp of relative error in the pdf. In the far
  tails of moderate df the same effect gives a few hundred ulp
  (|logpdf| ~ 700).
- **Regime:** all df; the ulp count is about the largest term:
  217 ulp (df = 55), 442-578 ulp (df <= 4, far tails), 4.9e3 ulp (df = 2e3),
  6.3e6 ulp (df = 2e6), 1.4e8 ulp (df = 2e7), 1.5e9 ulp (df = 2e8). df = 2 is
  exact (7.4 ulp) because `xlogy(0, x) = 0`.
- **Reproducer:** `python accuracy.py refs/chi2_ref.py --at pdf 1980000 2e6`;
  `python accuracy.py scratch/chi2/chi2_df2e8_ref.py --at pdf 200000000 2e8`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(199400000, 2e8)` | 2.9986192045501486e-201 | 2.998620177615371e-201 | 1.5e+09 | 3.0e+05 | severe |
  | `pdf(200000000, 2e8)` | 1.9947109067549423e-05 | 1.994711400344904e-05 | 1.1e+09 | 1 | severe |
  | `pdf(20000000, 2e7)` | 6.3078310499459e-05 | 6.30783125248514e-05 | 1.4e+08 | 1 | severe |
  | `pdf(1980000, 2e6)` | 2.7775470374884574e-26 | 2.777547033599348e-26 | 6.3e+06 | 1e+04 | severe |
  | `pdf(1994000, 2e6)` | 2.202633507476837e-06 | 2.202633504959279e-06 | 5.1e+06 | 3e+03 | severe |
  | `pdf(1999998, 2e6)` | 0.0001994712231928538 | 0.00019947122331374198 | 2.7e+06 | 0 | severe |
  | `pdf(1980, 2000)` | 0.006058262693563576 | 0.006058262693557183 | 4.75e+03 | 9 | moderate |
  | `pdf(1367.5444679663242, 2000)` | 1.63067568886359e-30 | 1.6306756888618296e-30 | 4.86e+03 | 315 | moderate |

- **Fix direction:** same as gamma F2: build the density from the
  Lanczos-scaled prefactor (`igam_fac(a, y)/y` with a = df/2, y = x/2, times
  1/2), so the only large term is the genuine `a*log1pmx((y-a)/a)`.

### F3: logpdf cancels four terms of size (df/2) log(df)

- **Code:** `_logpdf`, line 1653:
  `sc.xlogy(df/2.-1, x) - x/2. - sc.gammaln(df/2.) - (np.log(2)*df)/2.`
- **Mechanism:** pattern 5. Near the mean the result is
  `~ -0.5*log(4*pi*df)` (about -8.5 at df = 2e6) while the terms are
  1.45e7, 1e6, 1.28e7 and 6.9e5. Their rounding errors (~1e-9 absolute) give
  ~1e-10 relative error (4e5 ulp). The extra `log(2)*df/2` term and the
  unscaled `xlogy(., x)` (rather than `xlogy(., x/2)`) make the terms slightly
  larger than in gamma's `_logpdf`, but the mechanism is the same as gamma F4.
- **Regime:** df >~ 2e3 and x within a few `sqrt(2 df)` of df: 931 ulp
  (df = 2e3, minor), 4.0e5 ulp (df = 2e6), 1.5e7 ulp (df = 2e7), 1.0e8 ulp
  (df = 2e8). Further out |logpdf| grows and the relative error falls
  (834 ulp at df = 2e6, x = 1.8e6).
- **Reproducer:** `python accuracy.py refs/chi2_ref.py --at logpdf 1994000 2e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(200000000, 2e8)` | -10.822426334023476 | -10.822426086574135 | 1.0e+08 | 0.0924 | severe |
  | `logpdf(20000000, 2e7)` | -9.671133579686284 | -9.671133547577112 | 1.5e+07 | 0.103 | severe |
  | `logpdf(1994000, 2e6)` | -13.025856864638627 | -13.025856865781604 | 4.0e+05 | 230 | moderate |
  | `logpdf(1999998, 2e6)` | -8.519840576685965 | -8.519840576079922 | 3.2e+05 | 0 | moderate |
  | `logpdf(1980, 2000)` | -5.1063322042331265 | -5.106332204234182 | 931 | 1.76 | minor |

- **Fix direction:** same as gamma F4: `a*log1pmx((y-a)/a) + 0.5*log(a/(2*pi))
  - log(x) - log(lanczos_sum_expg_scaled(a))` with a = df/2, y = x/2 (plus
  the constant change of variable from gamma to chi2), so that no O(a log a) terms cancel.

### F4: ppf and isf inherit F1 through gammaincinv / chdtri

- **Code:** `_ppf`, line 1665: `2*sc.gammaincinv(df/2, p)`; `_isf`, line 1662:
  `sc.chdtri(df, p)` (= `2*gammainccinv(df/2, p)`, bitwise).
- **Mechanism:** both are Newton iterations on cephes `igam`/`igamc` and
  converge to the root of the inaccurate cdf of F1 (gamma F3). At df = 2e6,
  q = 1e-6 the quantile is x = 1990507.5 = df - 4.66*sqrt(2 df), right at the
  start of the F1 window.
- **Regime:** df >~ 5e5 with the quantile in the F1 window. At df = 2e6 that
  is ppf for q in about [1e-17, 2e-6] and isf for 1 - q in the same range.
- **Reproducer:** `python accuracy.py refs/chi2_ref.py --at ppf 1e-6 2e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-06, 2e6)` | 1990507.5468255973 | 1990507.5439550001 | 6.5e+06 | 0.000202 | severe |
  | `ppf(1e-08, 2e6)` | 1988796.3200295216 | 1988796.3195722424 | 1.0e+06 | 0.000173 | severe |
  | `ppf(1e-16, 2e6)` | 1983600.2096096554 | 1983600.2096078314 | 4.14e+03 | 0.00012 | moderate |
  | `isf(0.999999, 2e6)` | 1990507.546825609 | 1990507.5439550118 | 6.5e+06 | 202 | severe |
  | `isf(0.9999993371296838, 2e6)` | 1990343.210236421 | 1990343.2078313048 | 5.4e+06 | 301 | severe |
  | `isf(0.9999999994820525, 2e6)` | 1987816.6026229316 | 1987816.6024624356 | 3.6e+05 | 3.1e+05 | moderate |

- **Fix direction:** none in `chi2_gen`. Fixing F1 in `igam` fixes this.

### F5: ppf and isf for tiny df amplify the rounding of a large exponent

- **Code:** `_ppf`, line 1665 / `_isf`, line 1662 (inside `igami`/`igamci`).
- **Mechanism:** patterns 3 and 8, gamma F5 with a = df/2 = 1e-3: the quantile
  is `2*exp((log q + gammaln(a+1))/a)`, and the exponent (~ -694) is computed
  in double precision, so each half-ulp of it becomes ~700 ulp of x, on top of
  the inherent cond (arg) = 1/a = 1e3.
- **Regime:** df <~ 2e-2 with q in the bulk (results 1e-301 ... 1e-70).
- **Reproducer:** `python accuracy.py refs/chi2_ref.py --at ppf 0.5000000000000001 0.002`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(0.5000000000000001, 0.002)` | 1.0488412816549952e-301 | 1.0488412816558285e-301 | 3.58e+03 | 1e+03 | moderate |
  | `ppf(0.8499999999999999, 0.002)` | 2.9487038091907485e-71 | 2.948703809192674e-71 | 2.94e+03 | 1e+03 | moderate |
  | `isf(0.4999999999999999, 0.002)` | 1.0488412816549952e-301 | 1.0488412816558285e-301 | 3.58e+03 | 1e+03 | moderate |
  | `isf(0.36, 0.002)` | 1.70090250366468e-194 | 1.700902503665492e-194 | 2.15e+03 | 562 | moderate |

- **Fix direction:** in `scipy.special` `igami`/`igamci`, do the final Newton
  step on `log P` (`log Q`) for a < 1 (see gamma F5).

### F6: cdf/sf lose the rounding of the igam_fac exponent

- **Code:** `_cdf`, line 1656 / `_sf`, line 1659; the loss is in cephes
  `igam_fac(a, y)`, which for `|a - y| > 0.4a` computes
  `exp(a*log(y) - y - lgam(a))`.
- **Mechanism:** pattern 5 inside the special function (gamma F6). The exponent
  is a cancelling sum of terms of size `a*|log y|`, and its absolute rounding
  error becomes relative error of the result.
- **Regime:** a = df/2 >~ 100 with x far from df (moderate at df = 2e3), and
  x < ~1e-50 for any df (minor, 100-300 ulp, e.g. `cdf(1e-250, 2)` 251 ulp
  where the exact value is x/2).
- **Reproducer:** `python accuracy.py refs/chi2_ref.py --at sf 3897.3665961010274 2000`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `sf(3897.3665961010274, 2000)` | 7.180551658663522e-125 | 7.18055165865818e-125 | 3.35e+03 | 951 | moderate |
  | `sf(4000, 2000)` | 6.847349459617758e-136 | 6.847349459614753e-136 | 1.98e+03 | 1e+03 | moderate |
  | `cdf(1051.3167019494863, 2000)` | 1.3530223125744093e-75 | 1.3530223125735475e-75 | 2.87e+03 | 475 | moderate |
  | `cdf(1000, 2000)` | 3.298272797068251e-86 | 3.2982727970671e-86 | 1.57e+03 | 501 | moderate |
  | `cdf(1e-130, 3.98646)` | 9.5870617486045e-261 | 9.587061748605144e-261 | 303 | 1.99 | minor |
  | `cdf(1e-250, 2)` | 5.0000000000002786e-251 | 5e-251 | 251 | 1 | minor |

- **Fix direction:** in `scipy.special`, use the Lanczos prefactor branch of
  `igam_fac` for all a > ~20 (gamma F6).

## RuntimeWarnings

The only warning is `underflow encountered in exp` from `_pdf`
(`np.exp(self._logpdf(x, df))`), for all 7 parameter sets. It comes from grid
points where the pdf truly underflows (true value 0 or subnormal), so it is
not an accuracy problem. The harness evaluates a whole grid per call, so
the warning is attributed to every point of that call. No warnings from
logpdf, cdf, sf, ppf or isf.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | full grid for df = 2e-3, 1, 2, 3.986, 55 | 11.1 ulp |
| pdf | df = 2 (xlogy(0, x) exact) | 7.43 ulp |
| cdf | bulk of all sets; high tail (x > df) of all sets | 21.4 ulp bulk, 3 ulp high tail |
| cdf | low tail for df <= 55 | 303 ulp (F6 minor, tiny x) |
| sf | bulk of all sets; low tail (x < df) for df <= 2e3; high tail of df = 2e6 up to df + 30*sqrt(2 df) and of df = 2e7, 2e8 up to df + 10*sqrt(2 df) | 117 ulp (df = 1 near x = 2, i.e. `igamc(0.5, y ~ 1)`, minor, no specific mechanism found); 34.4 ulp at df = 2e7 |
| ppf | q in [1e-300, 1 - 2e-16] for df = 1, 2, 3.986, 55, 2e3 | 273 ulp (df = 3.986, q ~ 1e-288, minor); df = 55 and 2e3 <= 14.4 ulp |
| isf | q in [1e-300, 1 - 2e-16] for df = 1, 2, 3.986, 55, 2e3 | 90.7 ulp (df = 1, minor); otherwise within conditioning |
| ppf, isf | q -> 1 / q -> 0 ill-conditioned ends, all sets | within conditioning |

## Suspected, not confirmed

- df >= ~2e10 could not be measured (the Kummer-series reference needs >1e6
  terms). Following F1's trend the cdf error should reach relative error ~1
  ("loss") there. This is an extrapolation, not a measurement.
- ppf/isf at df = 2e7 and 2e8 were not measured (inverting the series
  reference is too slow). By F4's mechanism they should be much worse than at
  df = 2e6, because the cdf they invert is off by up to 37%.
- x = 0 is not on the harness grid (pdf is inf for df < 2, 1/2 for df = 2,
  0 for df > 2). Not measured.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | `log(chdtrc(df, x))` above the median, `log1p(-chdtr)` below | log-space `igamc`: `logsf = a*log(y) - y - gammaln(a) + log(continued fraction)` with a = df/2, y = x/2 (or `-y + (a-1)log y - gammaln(a) + log1p(...)` asymptotically) | sf underflows for x >~ 1500 (moderate df): `logsf(4000, 2)` returns -inf, true -2000.0 (loss) |
| logcdf | `log(chdtr(df, x))` below the median, `log1p(-chdtrc)` above | log-space `igam`: `a*log(y) - y - gammaln(a+1) + log(series sum)` | cdf underflows in the left tail: `logcdf(1e-300, 55)` returns -inf, true -19081.607741499824; `logcdf(200, 2000)` returns -inf, true -1406.8527552662533 (loss) |

Both share their fix with gamma's logcdf/logsf (`reports/gamma.md`,
Opportunities). A single log-space incomplete-gamma routine would serve
gamma, chi2 and erlang.

## Shared code

chi2's `_cdf`, `_sf`, `_ppf` and `_isf` only forward to the cephes
incomplete-gamma code. `sc.chdtr`, `sc.chdtrc` and `sc.chdtri` are bitwise
equal to `sc.gammainc(df/2, x/2)`, `sc.gammaincc(df/2, x/2)` and
`2*sc.gammainccinv(df/2, q)` (`scratch/chi2/probe_equiv.py`, 7 df values,
2600 points each). So F1, F4, F5 and F6 are the `scipy.special` defects
already documented in `reports/gamma.md` (F1, F3, F5, F6) and in
`issue-reports/gammainc-large-a.md`:

- `igam` 2000-iteration cap (gamma F1 -> chi2 F1, F4). The issue report's
  prediction that `chdtr`/`chdtrc` fail for df >~ 4e5 is **confirmed**. The
  onset is between df = 4e5 (27 ulp) and 6e5 (3.7e4 ulp) at
  x = df - 5*sqrt(2 df). At df = 2e8 only one digit is correct (1.7e15 ulp),
  the same number as gamma at a = 1e8, as the bitwise equivalence implies.
  The issue report could add `chdtr`/`chdtrc`/`chdtri` and `scipy.stats.chi2`
  as affected public entry points.
- `igami`/`igamci` small-a amplification (gamma F5 -> chi2 F5).
- `igam_fac` exponent prefactor (gamma F6 -> chi2 F6).

F2/F3 (logpdf/pdf cancellation) are in `chi2_gen` itself, but they are the
same construction as `gamma_gen._logpdf`, so a shared helper (e.g. exposing
`igam_fac`) would fix both.

## Existing GitHub issues

- #26471 (open): BUG: inaccuracy in special.gammainc for certain large parameters - relates to F1 and F4 (the same cephes `igam_series` cap; the issue names `chdtr`/`chdtrc` for df >~ 4e5, which this report confirms)
- #4837 (closed): ppf near 0 looks accurate for expon but not for chi2 or for gamma - background for F5 (small-df ppf)
- #1568 (closed, 2009): chi2.pdf numerically unstable for large values of df - the fix that moved pdf to log space; F2/F3 are the residual cancellation of that log-space form

Excluded: #24683 (open) and #20972 (closed), `chdtr`/`chdtrc` edge cases at inf/nan/negative arguments; #1527 (closed, nan for 0 < df < 1, old). Searched via the GitHub search API for "chi2" (titles) and "chdtr"; `gh` was not available.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1990800, 2e6)` | 2.045146368700871e-06 | 2.045165766063152545665e-06 | verified (4.27e10 ulp, rel. 9.5e-6) | Reference (`scratch/chi2/verify/`): mp.quad of the docstring pdf, 100 and 200 digits agree to >= 40 digits. The 2000-term partial Kummer sum gives 2.04514624e-6, matching SciPy, so the mechanism (iteration cap in `igam_series`) is supported. Summary text says "4.3e-6 relative at df = 2e6"; the measured relative error at this point is 9.5e-6 (4.3e10 ulp), so that sentence should be corrected. |
| F1 | `cdf(19970907.045526452, 2e7)` | 2.008512445946794e-06 | 2.090990982807373698631e-06 | verified (1.78e14 ulp, rel. 0.039) | mp.quad of the pdf |
| F1 | `cdf(199908000, 2e8)` | 1.3154316493365769e-06 | 2.105648671238922060054e-06 | verified (1.69e15 ulp, rel. 0.375) | mp.quad of the pdf; 2000-term partial sum 1.31504e-6, close to SciPy's value |
| F1 | `sf(199908000, 2e8)` | 0.9999986845683506 | 0.9999978943513287610779 | verified (3.56e9 ulp) | 1 - quad cdf |
| F2 | `pdf(200000000, 2e8)` | 1.9947109067549423e-05 | 1.994711400344903888720e-05 | verified (1.11e9 ulp) | Docstring pdf with mp.loggamma. Mechanism confirmed: ulp of `xlogy` term (1.9e9) is 2.4e-7 absolute, equal to the relative error 2.47e-7 |
| F2 | `pdf(199400000, 2e8)` | 2.9986192045501486e-201 | 2.998620177615371110063e-201 | verified (1.46e9 ulp) | |
| F2 | `pdf(1980000, 2e6)` | 2.7775470374884574e-26 | 2.777547033599348073663e-26 | verified (6.31e6 ulp) | |
| F3 | `logpdf(200000000, 2e8)` | -10.822426334023476 | -10.82242608657413412060 | verified (1.03e8 ulp) | Terms 1.91e9, 1e8, 1.74e9, 6.9e7 as stated (line 1653) |
| F3 | `logpdf(1994000, 2e6)` | -13.025856864638627 | -13.02585686578160296566 | verified (3.95e5 ulp) | moderate at this point, as the report's table says |
| F4 | `ppf(1e-6, 2e6)` | 1990507.5468255973 | 1990507.543955000198227 | verified (6.49e6 ulp) | Newton on the quad cdf; cdf(x_ref)/q - 1 = 1e-198 at 200 digits |
| F4 | `isf(0.999999, 2e6)` | 1990507.546825609 | 1990507.543955011783799 | verified (6.49e6 ulp) | Newton on 1 - quad cdf; check 7e-199 |
| F5 | `ppf(0.5000000000000001, 0.002)` | 1.0488412816549952e-301 | 1.048841281655828573654e-301 | verified (3.58e3 ulp) | Alternating series P(a,y) = y^a/Gamma(a) sum (-y)^k/(k!(a+k)), Newton on log P. The explanation is incomplete: `2*exp((log q + gammaln(a+1))/a)` evaluated in double gives 1.048841281655856e-301 (about 117 ulp), and one half-ulp of the exponent -693.7 is about 256 ulp. Most of the 3.6e3 ulp must come from elsewhere in `igami` (Newton or `igam` at tiny a). Mechanism should be marked partly unclear. |
| F5 | `isf(0.36, 0.002)` | 1.70090250366468e-194 | 1.700902503665492037196e-194 | verified (2.15e3 ulp) | same series and Newton |
| F6 | `sf(3897.3665961010274, 2000)` | 7.180551658663522e-125 | 7.180551658658180066187e-125 | verified (3.35e3 ulp) | mp.quad of the pdf from x to inf. Mechanism consistent: the ulp of a*log(y) ~ 7.6e3 is 9e-13, about 4e3 ulp relative. The cephes C source was not inspected. |
| F6 | `cdf(1051.3167019494863, 2000)` | 1.3530223125744093e-75 | 1.353022312573547504477e-75 | verified (2.87e3 ulp) | mp.quad of the pdf from 0 to x |

Verified by dist-accuracy-verifier: 6 of 6 verified.
