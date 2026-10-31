# ncx2: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `ncx2_gen`, lines 7827-7917; helper `_ncx2_log_pdf`, lines 7811-7824
- **Implemented:** pdf (7878-7881, `scu._ncx2_pdf`), logpdf (7874-7876, `_ncx2_log_pdf` with `sc.ive`), cdf (7883-7886, `sc.chndtr`), ppf (7888-7891, `sc.chndtrix`), sf (7893-7896, `scu._ncx2_sf`), isf (7898-7901, `scu._ncx2_isf`). Everything except logpdf wraps Boost `non_central_chi_squared_distribution`. Every method dispatches `nc == 0` to `chi2` (`xpx.apply_where(nc != 0, ...)`).
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** chi2 (handles nc == 0 for every method); helpers: `_ncx2_log_pdf`; special functions: `sc.chndtr`, `sc.chndtrix`, `scu._ncx2_pdf`, `scu._ncx2_sf`, `scu._ncx2_isf`, `sc.ive`, `sc.xlogy`
- **Reference:** `refs/ncx2_ref.py`, 100 digits. pdf from the docstring formula (`mp.besseli`). cdf and sf are Poisson-weighted regularized incomplete gamma sums with positive terms only: one `mp.gammainc` call, then the two-term recurrence in its stable direction, with a rigorous Poisson-tail truncation bound. ppf/isf invert the cdf or sf, whichever side has target <= 1/2. Self-checks: PASS (main run). The second run (`--params "21,1000;3,10000;21,1e-100" --methods pdf,logpdf,cdf`) used `--no-checks`.
- **Parameter sets:** (21, 1.0560465975116415), (0.5, 1), (2, 1e-10), (5, 60), (10, 300), (1000, 1); second run (pdf, logpdf, cdf): (21, 1000), (3, 1e4), (21, 1e-100). Single points (`--at`) also at (21, 60), (2.5, 1e-300), (21, 1e-300), (1, 342).
- **Worst finding:** loss in pdf, logpdf, cdf, sf (including a Python `OverflowError`), ppf

## Summary

In the bulk, pdf, cdf, sf, ppf and isf agree with the reference to <= 20 ulp. logpdf is accurate (<= 11 ulp) where it is finite. The failures are all in the tails, except that logpdf also fails in the bulk:

- **Boost starts its series at the Poisson mode.** For moderate and large nc, Boost starts the Poisson-weighted series at the Poisson mode j = nc/2 and returns 0 when that starting term underflows. Wide tails come out exactly 0 although the true values are normal doubles (F1):
  - pdf, for nc > 50, in both tails;
  - sf, in the right tail;
  - cdf, in the left tail for nc >= 200.
- **Quantiles inherit F1.** ppf returns a constant for small q, and isf saturates for small q (F6).
- **sf raises a Python `OverflowError`** for x < 3e-8 once df/2 + nc/2 >~ 171 (F2). This aborts the whole array call.
- **logpdf (SciPy's own code) loses everything in three regimes:**
  - `sc.ive` underflows: small nc*x, or large df, including the entire bulk of (1000, 1) (F3);
  - `sc.ive` returns nan for sqrt(nc x) > 2^30 (F4);
  - `x/nc` overflows to inf (F5).
- **Subnormal band in the cdf left tail.** For nc < 200, a band of left-tail x where a Boost intermediate is subnormal costs up to 5e10 ulp in cdf, and up to 9e7 ulp in ppf (F7).
- **Quantiles near q -> 1.** ppf/isf exceed the (already huge) conditioning by a factor 2-4 (F8).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf, cdf, sf | pdf: nc > 50, both tails; sf: right tail (x > df + nc); cdf: left tail for nc >= 200. Exact x limits depend on (df, nc), see below | Boost starts the Poisson series at the mode k = nc/2 and returns 0 if the starting term underflows (pattern 4, 10); subnormal starting term just inside the boundary | 0.0 for 1.24e-299 (pdf), 5.0e-140 (cdf), 9.0e-217 (sf); 6.0e12 ulp at the edge | loss |
| F2 | sf | df/2 + round(nc/2) >~ 171, x < 2^-25 = 2.98e-8 | Boost `tgamma` overflow inside the incomplete-gamma call at the Poisson mode raises; SciPy turns it into `OverflowError` (pattern 3) | exception where the true sf is 1.0 | loss |
| F3 | logpdf | df > 2 and `ive(df/2-1, sqrt(nc x))` < ~1e-308: small nc*x (tiny x or tiny nc), or large df (the whole bulk of (1000, 1)) | `sc.ive` underflows to 0, `apply_where` fills -inf (pattern 4) | -inf for -4.72 (bulk) | loss |
| F4 | logpdf | sqrt(nc x) > 2^30 = 1.07e9, i.e. x > 1.15e18/nc | `sc.ive` (AMOS) returns nan for z > 2^30, filled with -inf (pattern 10) | -inf for -5.0e19 | loss |
| F5 | logpdf | x/nc > 1.8e308 with ive still finite (tiny nc), df != 2 | `xlogy(df2/2, x/nc)` with x/nc = inf gives +inf (pattern 3) | +inf for -5.0e9 | loss |
| F6 | ppf, isf | ppf: q below the smallest nonzero Boost cdf (nc >= 200); isf: q below the smallest nonzero Boost sf (F1 regimes) | root finder on a cdf/sf that is 0 (inherits F1) | ppf 0.978 for 5.57e-27 (7.9e41 ulp); isf 2424.4 for 2970.9 | loss |
| F7 | cdf, ppf | nc < 200 (Ding path), left-tail band where (x/2)^(df/2+1) is subnormal | Boost `gamma_p_derivative(df/2+1, x/2)` with a subnormal prefix (pattern 4, inferred) | cdf 5.0e10 ulp; ppf 9.0e7 ulp | severe |
| F8 | ppf, isf | ppf q -> 1, isf q -> 1 (the far tails reached through 1 - q) | Boost inverts the cdf (ppf) or sf (isf) at a target close to 1, so the few-eps absolute error of that function adds to the q rounding (pattern 8) | 1.2e12 ulp at cond 2.8e11; 3.9e12 at cond 1.9e12 | severe |
| F9 | pdf | large nc (3, 1e4), right of the mean | mechanism unclear (Boost series from the Poisson mode) | 1.6e3 ulp at cond 550 | moderate |

### F1: pdf, cdf and sf return 0 (or lose digits) in the tails because Boost starts the series at the Poisson mode

- **Code:** `_pdf`, line 7880 (`scu._ncx2_pdf`); `_cdf`, line 7885 (`sc.chndtr`); `_sf`, line 7895 (`scu._ncx2_sf`); all Boost `non_central_chi_squared`.
- **Mechanism:** Boost sums the Poisson mixture by starting at the Poisson mode k = nc/2 and recursing in both directions.
  - The pdf uses this series for nc > 50. For nc <= 50 it uses `exp(r)*I_{df/2-1}(sqrt(nc x))`, unless |r| >~ 177.
  - The sf uses it whenever x > df + nc.
  - The cdf uses it only for nc >= 200 and x <= df + nc. For nc < 200 the cdf uses Ding's series from j = 0, which does not have this problem.

  The starting term is `gamma_p_derivative(df/2 + k, x/2)` times the mode Poisson weight for the pdf, and `gamma_p`/`gamma_q(df/2 + k, x/2)` for the cdf/sf. When it underflows to 0, Boost returns 0, although the true sum is dominated by terms far from the mode. Those are j near 0 in the left tail and j near sqrt(nc x)/2 in the right tail, and they are normal doubles (pattern 4, and pattern 10: the series is started where the summand is not largest). Just inside the boundary the starting term is subnormal, and the result keeps only a few digits (pdf/cdf(1, 10, 300): 4.5e12 / 6.0e12 ulp; pdf(3797.37, 21, 1000): 2.9e7 ulp). The mechanism is inferred from the output and from the Boost algorithm description; the Boost source was not inspected.
- **Regime:** the boundary is where the starting term crosses ~1e-308. Direct calls (x on [1e-3, 1e5], 100 points per decade; the (5, 60) left limit is estimated from the starting term and is consistent with the harness grid, where the last loss is at x = 1e-10) give:
  - **pdf:** 0 for x <~ 4e-9 at (5, 60) and x <= 1e-7 at (21, 60). The left-tail limit is x <~ 0.95 at (10, 300), x <~ 97 at (21, 1000) and x <~ 5600 at (3, 1e4). In the right tail the pdf is 0 from x ~ 1790 at (5, 60), 1830 at (21, 60), 2470 at (10, 300), 3900 at (21, 1000) and 16700 at (3, 1e4).
  - **sf:** the same right-tail limits.
  - **cdf:** the same left-tail limits for nc >= 200.

  In the harness runs this gives:
  - 21 pdf losses at (5, 60), 8 at (10, 300), 24 at (21, 1000) and 21 at (3, 1e4);
  - cdf losses at (10, 300), (21, 1000) and (3, 1e4);
  - sf losses at (5, 60) and (10, 300).

  Small nc (21, 1.056), (0.5, 1), (2, 1e-10) and (1000, 1) are not affected.
- **Reproducer:** `python accuracy.py refs/ncx2_ref.py --at pdf 1e-7 21 60`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-190, 5, 60)` | 0.0 | 1.2443838154419746e-299 | inf | 1.5 | loss |
  | `pdf(1e-7, 21, 60)` | 0.0 | 1.803075954270799e-89 | inf | 9.5 | loss |
  | `pdf(1832, 21, 60)` | 0.0 | 1.5047833182034422e-262 | inf | 746 | loss |
  | `pdf(50, 21, 1000)` | 0.0 | 9.209951839615363e-140 | inf | 91.4 | loss |
  | `cdf(50, 21, 1000)` | 0.0 | 5.001685851652358e-140 | inf | 92.1 | loss |
  | `cdf(0.5, 10, 300)` | 0.0 | 5.472943698791299e-69 | inf | 8.67 | loss |
  | `sf(1913.709601853868, 5, 60)` | 0.0 | 1.3318078535658802e-282 | inf | 787 | loss |
  | `sf(4000, 21, 1000)` | 0.0 | 9.035208488926061e-217 | inf | 996 | loss |
  | `sf(16700, 3, 10000)` | 0.0 | 5.4488657801652125e-188 | inf | 1.89e+03 | loss |
  | `cdf(1, 10, 300)` | 4.4822716878748713e-66 | 4.476337671126287e-66 | 6.0e+12 | 10.8 | severe |
  | `pdf(1, 10, 300)` | 4.843723354183076e-65 | 4.8388493982327275e-65 | 4.5e+12 | 10.1 | severe |
  | `pdf(3797.366596101028, 21, 1000)` | 9.225779484074824e-196 | 9.225779544198666e-196 | 2.9e+07 | 920 | severe |

- **Fix direction:** upstream in Boost, start the recursion at the index of the largest summand, not at the Poisson mode. That index is roughly sqrt(nc x)/2 in the right tail and near 0 in the left tail. Alternatively, carry the starting term in log space and rescale. In SciPy, an interim option is to use Ding's series (from j = 0) for the left tail at any nc, and to apply a log-space scaling.

### F2: sf raises OverflowError for tiny x when df/2 + nc/2 >~ 171

- **Code:** `_sf`, line 7895: `scu._ncx2_sf(x, df, nc)` inside `np.errstate(over='ignore')`.
- **Mechanism:** for x <= df + nc and nc >= 200, Boost computes the sf as the complement of its mode-started P series. At y = x/2 < 2^-26, the incomplete gamma at a = df/2 + round(nc/2) takes a small-argument path that forms `tgamma(a + 1)`, inferred from the message "boost::math::tgamma<double>(%1%): Result of tgamma is too large to represent". This overflows for a + 1 > 171.6, and SciPy's Boost policy turns the overflow into a Python `OverflowError` (pattern 3). `np.errstate(over='ignore')` does not catch it, because it is not a numpy floating-point error. Because the call is vectorized, one bad element aborts the whole array, so the harness aborted on (3, 1e4). That set and (21, 1000) were therefore run without sf, and sf was measured with `--at` only. The cdf at the same points returns 0.0 without raising (correct: the true cdf underflows).
- **Regime:** df/2 + round(nc/2) >= ~170.7 and x < 2.98e-8. The x threshold is 2.980232238769531e-08 = 2^-25 for both (21, 1000) and (1, 400). The exception occurs at (1, 342) and not at (1, 340), and at (1000, 200) and (300, 200), but not at (10, 300) or (21, 300).
- **Reproducer:** `python accuracy.py refs/ncx2_ref.py --at sf 1e-10 21 1000` raises.

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1e-10, 21, 1000)` | `OverflowError` | 1.0 | - | - | loss |
  | `sf(1e-10, 1, 342)` | `OverflowError` | 1.0 | - | - | loss |

- **Fix direction:** in SciPy, return 1.0 (or `1 - cdf`) when x < df + nc and the cdf is below eps. Upstream, use a log-space prefix (`lgamma`) in the small-x incomplete-gamma path, or have the Boost policy return inf instead of raising.

### F3: logpdf returns -inf where `sc.ive` underflows, including the bulk for large df

- **Code:** `_ncx2_log_pdf`, line 7818: `corr = sc.ive(df2, xs*ns) / 2.0`. Lines 7820-7824: `apply_where(corr > 0, ..., fill_value=-np.inf)`.
- **Mechanism:** for z = sqrt(nc x) small compared with the order v = df/2 - 1, `ive(v, z) ~ (z/2)**v / Gamma(v+1)`. This underflows to 0 while its log is an ordinary number. The helper then returns -inf by design, so the exponentially scaled Bessel function removes only the large-z overflow and does not help here (pattern 4). For df <= 2 (v <= 0), `ive` does not underflow at small z.
- **Regime:** df > 2 and `(z/2)**(df/2-1) / Gamma(df/2) <~ 1e-308`.
  - **Large df:** at (1000, 1), every grid point except x = 1e10 is -inf, including the bulk (91 loss points). At x = 1000 the order is 499 and z = 31.6, so `ive` ~ e^-1233.
  - **Tiny x:** at (21, 1.056) logpdf is -inf for x <= ~1e-70 (24 loss points), at (10, 300) for x <= 1e-160, and at (21, 1000) for x <= 1e-70.
  - **Tiny nc:** at (21, 1e-100), logpdf is -inf for x <= 1e30 and for x >= 1e120 (the latter is F4). That covers the whole bulk, while pdf is fine (77 ulp max).
- **Reproducer:** `python accuracy.py refs/ncx2_ref.py --at logpdf 1000 1000 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1000, 1000, 1)` | -inf | -4.7198055998919735 | inf | 0.106 | loss |
  | `logpdf(10, 21, 1e-100)` | -inf | -4.344112231839755 | inf | 1.04 | loss |
  | `logpdf(1e-100, 21, 1.0560465975116415)` | -inf | -2209.2025322583822 | inf | 0.0043 | loss |
  | `logpdf(1e-300, 21, 1000)` | -inf | -7083.586185648313 | inf | 0.00134 | loss |

- **Fix direction:** compute log I_v(z) directly, not log(ive) + z. For z^2 < v + 1, use the log of the power series: `v*log(z/2) - gammaln(v+1) + log1p(...)`. For large v, use the uniform (Debye) asymptotic expansion in log form. Where the pdf itself is normal, `log(scu._ncx2_pdf)` would also do.

### F4: logpdf returns -inf for sqrt(nc x) > 2^30 because `sc.ive` returns nan

- **Code:** `_ncx2_log_pdf`, line 7818 (`sc.ive(df2, xs*ns)`) and line 7820 (`corr > 0` is False for nan, so the fill value -inf is used).
- **Mechanism:** `sc.ive(v, z)` returns nan for every order once z > 1073741823.5 = 2^30 - 0.5, with no warning. Just below the threshold it is accurate: `ive(0, 1.07e9)` = 1.2196021380098267e-05, 0.84 ulp. This looks like AMOS's "complete loss of significance" (ierr = 4) limit for large arguments. The scaled function itself is perfectly representable there (~1/sqrt(2 pi z)), so this is a `scipy.special` limitation (pattern 10). Because the fill value hides the nan, logpdf returns -inf, although the true value is about -(sqrt(x) - sqrt(nc))**2/2 and can be any normal double down to -1.8e308.
- **Regime:** x > 1.15e18/nc, any df. Measured: 29 loss points at (0.5, 1), (5, 60), (10, 300) and (21, 1.056) for x >= 1e20; 28 at (2, 1e-10) for x >= 1e30; (21, 1e-100) for x >= 1e120.
- **Reproducer:** `python accuracy.py refs/ncx2_ref.py --at logpdf 1e20 21 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e20, 21, 1)` | -inf | -4.999999999e+19 | inf | 1 | loss |
  | `logpdf(1e30, 2, 1e-10)` | -inf | -5e+29 | inf | 1 | loss |

- **Fix direction:** for z > ~1e8, use the asymptotic `log(ive(v, z)) = -0.5*log(2*pi*z) + log1p(-(4v^2-1)/(8z) + ...)`. Upstream, extend `ive` beyond 2^30 (the scaled function needs no range limit).

### F5: logpdf returns +inf when x/nc overflows

- **Code:** `_ncx2_log_pdf`, line 7817: `res = sc.xlogy(df2/2.0, x/nc) - 0.5*(xs - ns)**2`.
- **Mechanism:** for tiny nc, `x/nc` overflows to inf (RuntimeWarning "overflow encountered in divide"), and `xlogy(df2/2, inf)` = +inf for df != 2. That +inf absorbs the finite -0.5*(xs - ns)**2. When sqrt(nc x) is small enough that `ive` is finite and not underflowed (otherwise F3 or F4 give -inf first), the result is +inf instead of a large negative number (pattern 3). For df = 2, `xlogy(0, inf)` = 0, so only the warning remains.
- **Regime:** x > 1.8e308*nc (needs nc <~ 1e-290 for x <= 1e20, or x near 1e300 with nc <~ 1e-8), df != 2, and `ive(df/2-1, sqrt(nc x))` > 0.
- **Reproducer:** `python accuracy.py refs/ncx2_ref.py --at logpdf 1e10 2.5 1e-300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e10, 2.5, 1e-300)` | inf | -4999999995.0117 | inf | 1 | loss |
  | `logpdf(1e300, 21, 1e-300)` | inf | -5e+299 | inf | 1 | loss |

- **Fix direction:** `xlogy(df2/2, x) - xlogy(df2/2, nc)` (or `(df2/2)*(log(x) - log(nc))`).

### F6: ppf returns a constant and isf saturates in the tails (inherits F1)

- **Code:** `_ppf`, line 7890 (`sc.chndtrix`); `_isf`, line 7900 (`scu._ncx2_isf`). Both are Boost root finders on the Boost cdf/sf.
- **Mechanism:** where the Boost cdf (or sf) is 0 instead of the true tiny value (F1), the root finder cannot go further into the tail. ppf returns the left edge of the region where the cdf is nonzero, independent of q. isf returns the right edge where the sf becomes 0. The direct probe `ncx2.cdf(ppf(q))` gives values like 4.25e-66 or 0 instead of q.
- **Regime:**
  - **ppf:** q below the smallest nonzero Boost cdf value, for nc >= 200. At (10, 300) this is q <~ 1e-66, where ppf is 0.97807 for every q down to 1e-300. Direct probes give ppf(q, 21, 1000) = 97.50287 for q <= 1e-200 and ppf(q, 3, 1e4) = 5511.05 for q <= 1e-200.
  - **isf:** q below the smallest nonzero Boost sf value. At (5, 60), q <~ 1e-251 gives 1758.87. At (10, 300), q <~ 5.6e-226 gives 2424.38. At (21, 1000), q = 1e-300 gives 3851.3 (direct probe).
- **Reproducer:** `python accuracy.py refs/ncx2_ref.py --at ppf 1e-100 10 300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-100, 10, 300)` | 0.9780664303673066 | 5.568011423095474e-07 | 7.9e+21 | 0.2 | loss |
  | `ppf(1e-200, 10, 300)` | 0.9780664303673063 | 5.568018915428153e-27 | 7.9e+41 | 0.2 | loss |
  | `isf(1e-300, 10, 300)` | 2424.37535139129 | 2970.9401734041535 | 8.3e+14 | 0.000988 | severe |
  | `isf(1e-300, 5, 60)` | 1758.866323029844 | 2014.9037987675667 | 5.7e+14 | 0.0012 | severe |
  | `isf(5.623413251903491e-226, 10, 300)` | 2424.3753513912907 | 2454.0256089330032 | 5.4e+13 | 0.00126 | severe |

  (The isf values are 13-18 % off. The tool rates them severe rather than loss because the relative error is below 1.)
- **Fix direction:** fix F1. Until then, return nan or warn when the cdf/sf at the bracket end is 0.

### F7: cdf (and ppf) lose up to 10 digits in a left-tail band for nc < 200

- **Code:** `_cdf`, line 7885 (`sc.chndtr`, Boost Ding's method for nc < 200, x <= df + nc).
- **Mechanism:** Ding's series starts with `tk = gamma_p_derivative(df/2 + 1, x/2)`. In Boost this is the regularized prefix `(x/2)**a e^{-x/2}/Gamma(a)` divided by x/2, with a = df/2 + 1. In a band of x the prefix is subnormal before the division, so only a few significant bits survive (pattern 4). Below the band the prefix underflows to 0. There Boost apparently switches to a log-space path, and the error drops back to ~100 ulp. The band edges match this prediction:
  - (2, 1e-10): a = 2, prefix y^2. Predicted band x in [4.4e-162, 3e-154]; measured [1e-161, 1e-154].
  - (5, 60): predicted [6e-93, 2e-88]; measured [10^-91.5, 10^-88].
  - (0.5, 1): predicted [~1e-259, 2e-246]; measured [10^-259, 10^-247].
  - (21, 1.056): predicted [7e-28, 1.6e-26]; measured `cdf(1e-26)` 4.2e3 ulp.

  The mechanism is inferred; the Boost source was not inspected. ppf inherits the error when the root falls into the band. The pdf is not affected (for nc <= 50 it uses the Bessel form).
- **Regime:** nc < 200, x such that `(x/2)**(df/2+1)/Gamma(df/2+1)` is subnormal. The cdf value there is ~e^{-nc/2} times the chi2 cdf, which is a normal number.
- **Reproducer:** `python accuracy.py refs/ncx2_ref.py --at cdf 1e-160 2 1e-10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-160, 2, 1e-10)` | 4.999944335663418e-161 | 4.99999999975e-161 | 5.0e+10 | 1 | severe |
  | `cdf(1e-158, 2, 1e-10)` | 5.000000659147041e-159 | 4.999999999750001e-159 | 5.9e+08 | 1 | severe |
  | `cdf(1e-90, 5, 60)` | 4.97754349164977e-240 | 4.977535261767898e-240 | 7.4e+09 | 2.5 | severe |
  | `cdf(1e-250, 0.5, 1)` | 1.7794024180878059e-63 | 1.779402418117842e-63 | 7.6e+04 | 0.25 | moderate |
  | `cdf(1e-26, 21, 1.0560465975116415)` | 3.422484850510512e-284 | 3.422484850513703e-284 | 4.2e+03 | 10.5 | moderate |
  | `ppf(5.1089697745070394e-64, 0.5, 1)` | 6.795738147512616e-253 | 6.795738011756869e-253 | 9.0e+07 | 4 | severe |
  | `ppf(1.9573417814877456e-238, 5, 60)` | 4.343711049196645e-90 | 4.343711058954297e-90 | 1.0e+07 | 0.4 | severe |
  | `ppf(2.872984833353539e-288, 21, 1.0560465975116415)` | 4.090804663422376e-27 | 4.09080466710396e-27 | 4.1e+06 | 0.0952 | severe |

- **Fix direction:** upstream in Boost, compute the `regularised_gamma_prefix` in log space when `a*log(x)` is below log(DBL_MIN) + a margin, not only when it underflows completely. Alternatively, start Ding's series with `exp(a*log(y) - y - lgamma(a+1))`.

### F8: ppf and isf near q = 1 exceed the conditioning by a factor 2-4

- **Code:** `_ppf`, line 7890 (`sc.chndtrix`); `_isf`, line 7900 (`scu._ncx2_isf`).
- **Mechanism:** for ppf with q close to 1, Boost solves cdf(x) = q, and for isf with q close to 1 it solves sf(x) = q. It does not solve the complementary equation for the small quantity 1 - q, which is exact in double for q > 1/2. The Boost cdf near 1 has an absolute error of a few eps (it is formed as 1 - Q). Its effect on x adds to the rounding of q (pattern 8). Most of the error is the conditioning q/(x f(x)), which is huge here. The excess is a factor 2-4 at (10, 300) and (1000, 1). Direct scan of all q points: the factor is up to 8.5 at q = 0.9999999999999966, (10, 300).
- **Regime:** ppf for q >~ 1 - 1e-6, isf for q >~ 1 - 1e-6, all parameter sets with flagged points ((10, 300), (5, 60), (21, 1.056), (1000, 1)).
- **Reproducer:** `python accuracy.py refs/ncx2_ref.py --at ppf 0.9999999999999627 10 300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999999627, 10, 300)` | 625.5876162789734 | 625.7516930967009 | 1.2e+12 | 2.8e+11 | severe |
  | `ppf(0.9999999943765867, 10, 300)` | 540.7832874895618 | 540.7832882408493 | 6.3e+06 | 2.6e+06 | severe |
  | `isf(0.9999999999999966, 1000, 1)` | 691.7229901020223 | 691.1214607045797 | 3.9e+12 | 1.9e+12 | severe |
  | `isf(0.9999999999522942, 10, 300)` | 124.72060706263865 | 124.72058893461298 | 6.5e+08 | 5.6e+08 | severe |

- **Fix direction:** `ppf(q) = isf(1 - q)` and `isf(q) = ppf(1 - q)` for q > 1/2 (1 - q is exact there).

### F9: pdf at very large nc loses ~1e3 ulp to the right of the mean

- **Code:** `_pdf`, line 7880 (`scu._ncx2_pdf`, Boost mode-started series for nc > 50).
- **Mechanism:** unclear. The error grows with nc ((21, 1000): 172 ulp at x = 1500, cond 133; (3, 1e4): 1.6e3 ulp at cond 550). This is consistent with accumulated rounding in the Poisson-weight and gamma-density recurrences over several hundred terms. It was not checked against the Boost source.
- **Regime:** nc >~ 1e3, x a few standard deviations above df + nc.
- **Reproducer:** `python accuracy.py refs/ncx2_ref.py --at pdf 12100 3 10000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(12100, 3, 10000)` | 3.8472993133518433e-25 | 3.84729931335321e-25 | 1.6e+03 | 550 | moderate |
  | `pdf(14400, 3, 1e4)` | 2.760474181079113e-90 | 2.7604741810798814e-90 | 1.25e+03 | 1.2e+03 | moderate |

- **Fix direction:** none specific; a log-space evaluation around the dominant term would bound the error.

## RuntimeWarnings

- logpdf: "overflow encountered in divide" from `x/nc` (line 7817) when x/nc > 1.8e308. It appeared at (2, 1e-10) for x >= ~2e298 and at (21, 1e-100) for large x. For df = 2 the overflow is harmless (`xlogy(0, inf)` = 0), but those points are -inf anyway because of F4. For df != 2 the result is +inf (F5).
- The Boost-backed methods run inside `np.errstate(over='ignore')` (gh-17432), so Boost overflow is silent. Boost's `tgamma` overflow, however, is raised as a Python `OverflowError` (F2), not a warning.
- No "Series did not converge" warnings or nan results from Boost appeared. Sweep over 12 parameter sets (including (1e4, 1e4), (1e5, 10), (5, 1e6), (1e-3, 5)) with x in [1e-300, 1e300] and q in [1e-300, 1 - 1e-15].

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | nc <= 50 (Bessel path): full grid at (21, 1.056), (0.5, 1), (2, 1e-10), (1000, 1), (21, 1e-100) | 94 ulp (cond 279) at (1000, 1); 87 ulp minor at (0.5, 1), x = 1e-200 (cond 0.75); 77 ulp at (21, 1.056), x = 1e-30 |
| pdf | bulk of every set, including nc = 60, 300, 1000, 1e4 | 12.3 ulp |
| logpdf | wherever finite (outside F3-F5) | 11.3 ulp (minor, (0.5, 1), x = 7) |
| cdf | bulk and right tail of every set; Ding left tail outside the F7 band | bulk 19.9 ulp; left tail 499 ulp at (0.5, 1), x = 1e-280, and 251 ulp at (2, 1e-10), x = 1e-250 (minor, no clear mechanism) |
| sf | (21, 1.056), (0.5, 1), (2, 1e-10): full grid; (1000, 1) | 10 ulp; 170 ulp at cond 252 for (1000, 1) |
| ppf, isf | bulk q in [1e-10, 0.999] for (21, 1.056), (5, 60): spot checks; full q grids from the cached references (`scratch/ncx2/quantile_scan.py`) | <= 2.1 ulp in the bulk; small-q ppf at (2, 1e-10) <= 184 ulp (minor); `ppf(1e-150, 2, 1e-10)` 0 ulp |
| all | tiny nc (1e-10, 1e-100, 1e-300) vs chi2: continuity at nc -> 0 | pdf/cdf/sf/ppf/isf agree with chi2 to <= 26 ulp at nc = 1e-300 |

## Suspected, not confirmed

- **logpdf cancellation (pattern 5).** `xlogy(df2/2, x/nc)` and `log(ive)` are both large and of opposite sign for large df with small nc. Wherever logpdf is finite the error was <= 11 ulp, for example 0.8 ulp at (300, 1), x = 1000. The regime where cancellation would matter is mostly covered by F3 (ive underflow).
- **Boost series non-convergence** (the ncf F7 analogue) was not observed. Huge nc (5, 1e6) and df (1e5, 10) gave no warnings or nan. Their accuracy was not measured: the reference would need ~1e6 terms.
- **Very large nc (>= 1e5).** F9 suggests that the error grows with nc. Not measured beyond nc = 1e4.
- **F2 mechanism.** That the exception comes from `tgamma(a + 1)` in Boost's small-x incomplete-gamma path is inferred from the message and the a + 1 > 171.6 threshold. The Boost source was not inspected.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | Log-space Poisson sum: logsumexp over j of `log w_j + log P(df/2 + j, x/2)`, with a log-space incomplete-gamma prefix. For small x the leading term `-nc/2 + (df/2)*log(x/2) - x/2 - gammaln(df/2 + 1)` plus a log1p correction | The cdf underflows (left tail). `logcdf(1e-100, 21, 1.0560465975116415)` = -inf vs -2441.8124168149507 (loss). Also everywhere F1/F7 affect the cdf |
| logsf | log(sf) | Log-space sum around the dominant index j* ~ sqrt(nc x)/2 (terms `log w_j + log Q(df/2 + j, x/2)`), or the Marcum-Q asymptotic `log Q_M(a, b)` with a = sqrt(nc), b = sqrt(x) | The sf underflows (right tail). `logsf(2000, 21, 1.0560465975116415)` = -inf vs -922.5160821001564 (loss). Also everywhere F1 makes the sf 0 |

## Shared code

- **chi2 (nc == 0 exactly).** Every method calls the `chi2` method for nc == 0, so `reports/chi2.md` applies unchanged at nc = 0. That covers its F1/F4: the cephes `igam` 2000-iteration cap (INDEX item 7), which gives cdf/sf/ppf/isf errors up to 1.7e15 ulp for df >~ 5e5. It also covers F2/F3 (pdf/logpdf cancellation for large df) and F5/F6. These were not re-measured here. For nc > 0, Boost is used instead, so the cephes issue does not carry over to nc = 1e-300 (agreement with chi2 <= 26 ulp at moderate df).
- **`sc.ive` returns nan for z > 2^30 (F4).** This is a `scipy.special` (AMOS wrapper) limitation for every order, without a warning, although `ive` is finite and smooth there. Other users in `scipy.stats`: `argus._stats` (`sc.ive(1, chi**2/4)`, only for chi >~ 6.5e4; not measured). `i0e`/`i1e` (cephes, used by rice and vonmises) are fine at 2e9. `sc.kve(0, 2e9)` also returns nan.
- **Boost `non_central_chi_squared` (F1, F2, F6, F7, F8, F9)** is used only by ncx2 in `scipy.stats`. Comparison with the Boost `non_central_beta` problems in `reports/ncf.md`:
  - The ncf F1 analogue (sign error at nc = 0) cannot occur, because SciPy dispatches nc = 0 to chi2.
  - The `OverflowError` exception is analogous to ncf F5 (Boost overflow policy raising through `np.errstate`).
  - The series non-convergence of ncf F7 was not seen.
  - The mode-started series that returns 0 when its starting term underflows (F1) is specific to the chi-squared family. It is the most damaging problem here.
- **Boost incomplete-gamma internals (F2, F7).** Both are inferred to be in Boost's `gamma_p`/`gamma_p_derivative` (subnormal prefix, `tgamma` overflow in the small-x path). SciPy's own `gammainc`/`gammaincc` are cephes and are not affected. Other Boost-gamma consumers in SciPy (none in `scipy.stats` besides ncx2) were not checked.
- INDEX item 8 (Boost incomplete-beta family) does not apply: ncx2 does not use `ibeta`.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(1e-7, 21, 60)` | 0.0 | 1.8030759542707998e-89 | verified | Poisson mixture of chi2 densities in log space; matches the Bessel form to 1e-199 |
| F1 | `cdf(50, 21, 1000)` | 0.0 | 5.0016858516523576e-140 | verified | Direct Poisson sum (one `gammainc` per term), agrees with `mp.quad` of the pdf at 200 digits. At 100 digits quad was not converged |
| F1 | `sf(4000, 21, 1000)` | 0.0 | 9.0352084889260618e-217 | verified | Direct Poisson sum of upper `gammainc`. `mp.quad` from 4000 to inf was off by 2e-11 to 1.5e-8 relative, depending on subdivision (not converged), so it was not used |
| F2 | `sf(1e-10, 21, 1000)` | `OverflowError` (Boost tgamma) | 1.0 | verified | 1 - quad of the pdf; the exception message matches the report |
| F3 | `logpdf(1000, 1000, 1)` | -inf | -4.7198055998919732 | verified | `sc.ive(499, 31.6)` = 0.0 confirmed (line 7818) |
| F4 | `logpdf(1e20, 21, 1)` | -inf | -4.999999999e+19 | verified | `sc.ive(9.5, 1e10)` = nan, `ive(9.5, 1.07e9)` finite (line 7818) |
| F5 | `logpdf(1e10, 2.5, 1e-300)` | inf | -4999999995.0116994 | verified | `x/nc` overflows to inf at line 7817, and `xlogy(0.125, inf)` = inf |
| F6 | `ppf(1e-100, 10, 300)` | 0.9780664303673066 | 5.5680114230954736e-07 | verified | Root of log cdf (Poisson sum) = log q; 8.0e21 ulp |
| F6 | `isf(1e-300, 10, 300)` | 2424.37535139129 | 2970.9401734041533 | verified | Root of log sf (Poisson sum of upper `gammainc`); 8.4e14 ulp |
| F7 | `cdf(1e-160, 2, 1e-10)` | 4.999944335663418e-161 | 4.99999999975e-161 | verified | Poisson sum (quad on [0, x] agrees at 200 digits); 5.1e10 ulp |
| F8 | `ppf(0.9999999999999627, 10, 300)` | 625.5876162789734 | 625.75169309670095 | verified | Root of sf = 1 - q; 1.19e12 ulp, cond q/(x f) = 2.80e11 recomputed (excess factor ~4) |
| F9 | `pdf(12100, 3, 10000)` | 3.8472993133518433e-25 | 3.8472993133532097e-25 | verified | Poisson mixture, matches the Bessel form to 1e-198; 1.6e3 ulp |

All references agree between 100 and 200 digits to at least 30 digits. The scripts are in `scratch/ncx2/verify/`. The cited lines (7817, 7818, 7820, 7880-7900) match SciPy 1.18.1. The Boost-internal mechanisms of F1, F2, F7 and F8 were not checked against the Boost source, as the report says.

Verified by dist-accuracy-verifier: 9 of 9 verified.
