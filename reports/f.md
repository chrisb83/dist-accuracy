# f: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `f_gen`, lines 2545-2652
- **Implemented:** pdf (2590-2594, `exp(_logpdf)`), logpdf (2596-2601), cdf (2603-2604, `sc.fdtr`), sf (2606-2607, `sc.fdtrc`), ppf (2609-2610, `sc.fdtri`). `fdtr`/`fdtrc`/`fdtri` wrap Boost `fisher_f_distribution`: cdf is `ibeta(n/2, m/2, nx/(m+nx))` for n*x <= m, otherwise `ibetac(m/2, n/2, m/(m+nx))` (the reverse for the complement); the quantile is `m*y/(n*(1-y))` with `y` from `ibeta_inv`
- **Inherited (not measured):** logcdf, logsf (rv_continuous default in 1.18.1: `log(cdf)` / `log1p(-sf)` split at the median, resp. `log(sf)` / `log1p(-cdf)`), isf = `ppf(1 - q)`
- **Related:** none (no other distribution calls `f_gen`); helpers: none; special functions: `sc.betaln`, `sc.fdtr`, `sc.fdtrc`, `sc.fdtri`, `sc.xlogy` (`sc.psi` only in `_entropy`)
- **Reference:** `refs/f_ref.py`, 120 digits (continued fraction for I_y(n/2, m/2) on its convergent side, the same approach as `refs/betaprime_ref.py`; quantiles by inverting F or S, whichever is the small side). Self-checks: PASS except one harness artefact. The "integral of pdf != cdf difference" check FAILs for (0.01, 3) on [1.6e-198, 1.0e-58] and [1.0e-58, 1.1e-7]: the harness integrates in linear x across the x**-0.995 singularity. The reference cdf there agrees with `mp.betainc` to 30 digits, and the pdf integral in log x agrees with the cdf difference to 25 digits (`scratch/f/check_selfcheck.py`, output in `check_selfcheck.txt`). All other checks pass, including bulk agreement with SciPy and precision stability.
- **Parameter sets:** (29, 18), (1, 1), (0.01, 3), (2, 0.01), (5, 1e4), (1000, 1000), (1e6, 2); plus (1e5, 3e5) for pdf, logpdf, cdf and sf only (quantile references take 5-30 s per point there), output in `scratch/f/run_big.txt`
- **Worst finding:** loss in logpdf, pdf, sf and ppf

## Summary

cdf, sf and ppf (Boost `fisher_f`) are accurate in the bulk and in most tails, but each one fails in a specific regime:

- sf returns 0 and cdf returns 1.0 once `n*x` overflows (small dfd).
- ppf returns nan for small q with moderate dfn and dfd >= 30 (Boost `ibeta_inv`, the same failure as beta F1). ppf also returns 2.2e-304 instead of ~1e-306 when the beta-scale root is subnormal.
- At dfn = dfd = 1, cdf and sf lose up to 1.4e7 ulp because `sc.betaincc(0.5, 0.5, x)` itself is inaccurate.

logpdf is a sum of large logarithms, and that causes three kinds of failure:

- It returns `-inf` once `m + n*x` overflows; pdf then returns 0 where the true value is normal.
- It loses all relative accuracy where logpdf is near 0 (dfn = 2 with x << dfd).
- For large shapes it loses up to 6.2e6 ulp, and pdf = exp(logpdf) up to 6.7e7 ulp.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf, pdf | n*x > 1.8e308 (x = 1e308 for n >= 2; x = 1e303 for n = 1e6) | `m + n*x` overflows, `log(inf)` (pattern 3) | logpdf -inf for -706.5; pdf 0.0 for 1.49e-307 | loss |
| F2 | sf, cdf | n*x > 1.8e308 with dfd small enough that sf is still normal (dfd <~ 2) | Boost `fisher_f` forms `df1*x` and `df2/(df2 + df1*x)`; the product overflows (pattern 3) | sf 0.0 for 0.0281; cdf 1.0 for 0.9719 | loss |
| F3 | ppf | small q with dfd >= ~30 and dfn ~ 5-10: (5, 1e4) q <= 1e-170, (10, 100) q <= 1e-164 | Boost `ibeta_inv` returns nan (pattern 10; shared with `sc.betaincinv`, beta F1) | nan for 6.5e-69 | loss |
| F4 | ppf | dfd/dfn large, beta root n*x/m below DBL_MIN while x is normal: (1, 1e4) q ~ 1e-153 | `ibeta_inv` clamps its root to DBL_MIN, then x = m*y/(n*(1-y)) (pattern 4/10) | 2.2e-304 for 1.57e-306 | loss |
| F5 | logpdf | logpdf near 0: dfn = 2 with x << dfd (e.g. (2, 0.01), x <= 5e-13); also near the zero crossing for other shapes | `log(m + n*x)` drops n*x (pattern 2) and the O(1)-O(10) terms cancel to a tiny result (pattern 5) | -1.1e-15 for -2.0e-118 | loss |
| F6 | logpdf, pdf | large dfn and/or dfd: (1e6, 2) all x, (1e5, 3e5), (5, 1e4), (1000, 1000) near the mode | terms like `n/2*log(n)` and `(n+m)/2*log(m+n*x)` of size ~1e7-1e8 cancel (pattern 5) | pdf 6.7e7 ulp, logpdf 6.2e6 ulp | severe |
| F7 | sf, cdf | dfn = dfd = 1: sf for x <~ 1e-10, cdf for x >~ 1e10 | `sc.betaincc(0.5, 0.5, z)` (Boost arcsine special case) is inaccurate for small z (pattern 10, shared) | 1.4e7 ulp | severe |
| F8 | cdf | (1e5, 3e5) near mean - 10 sd | Boost `ibeta` power terms, error just above conditioning (pattern 5/6, as beta F8) | 3.07e3 ulp (cond (arg) 1.93e3) | moderate |
| F9 | pdf | far tails, all shapes, \|logpdf\| of a few hundred | pdf = exp(logpdf): absolute rounding of logpdf becomes relative error (pattern 5) | 753 ulp | minor |

### F1: logpdf returns -inf (pdf returns 0) when n*x overflows

- **Code:** `_logpdf`, line 2600: `((n+m)/2) * np.log(m + n*x)`; `_pdf` = `np.exp(self._logpdf(...))`, line 2594.
- **Mechanism:** `n*x` overflows to inf for n*x > 1.8e308, so `log(m + n*x)` = inf and logpdf = -inf (pattern 3). The true logpdf is about -(m/2 + 1)*log(x) + const, which is finite. With small dfd the pdf decays only like x**(-1 - m/2), so it is still a normal double there, and pdf = exp(-inf) = 0 is a loss too. A RuntimeWarning "overflow encountered in multiply" is emitted.
- **Regime:** x > 1.8e308/dfn. logpdf fails at x = 1e308 for every measured set with dfn >= 2. pdf fails only where it is still normal there, which needs small dfd (e.g. (1e6, 0.01) at x = 1e303).
- **Reproducer:** `python accuracy.py refs/f_ref.py --at logpdf 1e303 1e6 0.01`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e303, 1e6, 0.01)` | -inf | -706.4936429848717 | inf | 0.00142 | loss |
  | `pdf(1e303, 1e6, 0.01)` | 0.0 | 1.4917957032662561e-307 | inf | 1 | loss |
  | `logpdf(1e308, 29, 18)` | -inf | -7080.688277709889 | inf | 0.00141 | loss |
  | `logpdf(1e308, 1e5, 3e5)` | -inf | -106102877.27320181 | inf | 0.00141 | loss |

- **Fix direction:** for n*x > m use `log(m + n*x) = log(n) + log(x) + log1p(m/(n*x))` (or the betaprime form `(a-1)*log(t) - (a+b)*log1p(t)` with `t = (n/m)*x` evaluated as `log(n/m) + log(x)`).

### F2: sf returns 0 and cdf returns 1 when n*x overflows inside Boost

- **Code:** `_sf`, line 2607: `sc.fdtrc(dfn, dfd, x)`; `_cdf`, line 2604: `sc.fdtr(dfn, dfd, x)` (Boost `fisher_f_distribution`).
- **Mechanism:** Boost computes `v1x = df1 * x` and then `df2 / (df2 + v1x)`. For df1*x > 1.8e308, `v1x` is inf, the beta argument is 0, so the complement is `ibeta(df2/2, df1/2, 0) = 0` and the cdf is `ibetac(..., 0) = 1` (pattern 3). The true sf decays like x**(-m/2), so for dfd <~ 2 it is still a normal double far beyond the overflow point.
- **Regime:** x > 1.8e308/dfn and dfd small enough that sf(x) > 2.2e-308 (roughly dfd*log(x)/2 < 708). Measured at (2, 0.01), (1e6, 1) and (1e5, 0.5).
- **Reproducer:** `python accuracy.py refs/f_ref.py --at sf 1e308 2 0.01`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1e308, 2, 0.01)` | 0.0 | 0.02808632067071855 | inf | 0.005 | loss |
  | `sf(1e303, 1e6, 1)` | 0.0 | 2.523131891237108e-152 | inf | 0.5 | loss |
  | `sf(1e304, 1e5, 0.5)` | 0.0 | 7.801230394443054e-77 | inf | 0.25 | loss |
  | `cdf(1e308, 2, 0.01)` | 1.0 | 0.9719136793292814 | 1.3e+14 | 0.000144 | severe |

  (The cdf row has relative error 0.029; the tool classifies it as severe and it is reported as given.)
- **Fix direction:** in `_sf`/`_cdf`, call `sc.betainc(m/2, n/2, yc)` / `sc.betaincc` directly with `yc = m/(m + n*x)` formed without the product, e.g. `r = (m/n)/x; yc = r/(1 + r)`. This also belongs upstream in Boost `fisher_f`.

### F3: ppf returns nan for small q (Boost ibeta_inv)

- **Code:** `_ppf`, line 2610: `sc.fdtri(dfn, dfd, q)` (Boost `fisher_f` quantile = `ibeta_inv(df1/2, df2/2, p)`).
- **Mechanism:** Boost `ibeta_inv` returns nan for these small targets (pattern 10). `sc.betaincinv(2.5, 5000, q)` returns nan at exactly the same q (scan in this review: q = 1e-160 fine, q = 1e-170 nan for both). This is the failure reported for beta (beta F1, `ibeta_inv` non-convergence).
- **Regime:** small q, with the onset depending on the shapes. The failure starts at q ~ 1e-174...1e-176 for dfn = 5 and dfd in {30, 50, 100, 1e4}, and at q ~ 1e-164 for (10, 100). None was found for (5, 18), (29, 1e3), (1, 1e4), (2.5, 1e4) or (3, 1e4) over q in [1e-308, 1e-10]. 11 loss points in the main run, all (5, 1e4), q <= 3.8e-176.
- **Reproducer:** `python accuracy.py refs/f_ref.py --at ppf 1e-170 5 1e4`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-170, 5, 1e4)` | nan | 6.465845740872791e-69 | inf | 0.4 | loss |
  | `ppf(1e-300, 5, 1e4)` | nan | 6.465845740872792e-121 | inf | 0.4 | loss |
  | `ppf(1e-180, 10, 100)` | nan | 5.011798692755652e-37 | inf | 0.2 | loss |

- **Fix direction:** for small q use the leading-order inversion of `I_y(a, b) ~ y**a/(a*B(a, b))` in log space and refine with Newton on `fdtr`, or fix `ibeta_inv` upstream (same fix as beta F1).

### F4: ppf clamps to 2.2e-304 when the beta-scale root is subnormal

- **Code:** `_ppf`, line 2610: `sc.fdtri(dfn, dfd, q)`; Boost returns `df2*y/(df1*(1-y))` with `y = ibeta_inv(df1/2, df2/2, q)`.
- **Mechanism:** the beta variable is `y = n*x/(m + n*x)`. For (1, 1e4) and q = 1e-153 the root is y = 1.57e-310, which is subnormal, although x = 1.57e-306 is normal. `ibeta_inv` returns DBL_MIN = 2.225e-308 instead (`sc.betaincinv(0.5, 5000, 1e-153)` gives 2.225073858507201e-308), and multiplying by m/n = 1e4 gives 2.2e-304, too large by a factor of 142 (pattern 4/10). For q = 1e-154 the true x is subnormal (not measured), but SciPy still returns 2.2e-304.
- **Regime:** dfd/dfn large and q small enough that `n*x/m < 2.2e-308` while `x > 2.2e-308`, i.e. x in [2.2e-308, 2.2e-308*m/n].
- **Reproducer:** `python accuracy.py refs/f_ref.py --at ppf 1e-153 1 1e4`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-153, 1, 1e4)` | 2.225073858507201e-304 | 1.5708748685746336e-306 | 6.3e+17 | 2 | loss |

- **Fix direction:** invert on a scaled variable, or compute small quantiles from the log-space leading-order formula `x = (m/n) * (q*a*B(a,b))**(1/a)` with Newton refinement.

### F5: logpdf loses all relative accuracy where logpdf is close to 0

- **Code:** `_logpdf`, lines 2599-2600: `m/2*log(m) + n/2*log(n) + xlogy(n/2-1, x) - (((n+m)/2)*log(m + n*x) + betaln(n/2, m/2))`.
- **Mechanism:** for dfn = 2 the `xlogy` term is 0 and the exact logpdf is `-(1 + m/2)*log1p(2x/m)`, which goes to 0 as x -> 0. The code computes this as the sum of terms of size up to ~5 (`betaln(1, 0.005)` = 5.3, `log(m + 2x)` = -4.6, ...). The sum cancels to the tiny result, leaving rounding residue of order eps (pattern 5). In addition, `m + n*x` rounds n*x away once n*x/m < eps, so the x-dependence is lost entirely (pattern 2). The result is a constant -1.1e-15 for every x <= 5e-19. The same effect gives a moderate error near the zero crossing of logpdf for any shapes (e.g. (29, 18) at x = 0.8805). The pdf is not affected (exp of a value near 0).
- **Regime:** |logpdf| much smaller than the largest term. Total loss for dfn = 2 and x < ~1e-16*dfd/2; severe up to x ~ 1e-10 at (2, 0.01). 13 loss and 2 severe points at (2, 0.01).
- **Reproducer:** `python accuracy.py refs/f_ref.py --at logpdf 1e-20 2 0.01`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-20, 2, 0.01)` | -1.1102230246251565e-15 | -2.0099999999999997e-18 | 2.5e+18 | 1 | loss |
  | `logpdf(1e-120, 2, 0.01)` | -1.1102230246251565e-15 | -2.006002611891947e-118 | 2.5e+118 | 1 | loss |
  | `logpdf(1e-10, 2, 0.01)` | -2.0100000108769223e-08 | -2.0099999799000003e-08 | 6.9e+07 | 1 | severe |
  | `logpdf(0.8805, 29, 18)` | 0.008665888074546046 | 0.008665888074533274 | 6.64e+03 | 32.7 | moderate |

  (cond (shapes) is 2.0e19 at the first row: logpdf there is extremely sensitive to the shapes, but they are exact user inputs.)
- **Fix direction:** combine analytically: `logpdf = log(n/m) + (n/2-1)*log(t) - (n+m)/2*log1p(t) - betaln(n/2, m/2)` with `t = n*x/m`. The constant terms then cancel exactly, and for n = 2 (`betaln(1, b) = -log(b)`) the result is `-(1 + m/2)*log1p(t)` with full accuracy.

### F6: logpdf and pdf lose up to 1e7-1e8 ulp for large shapes

- **Code:** `_logpdf`, lines 2599-2600 (as in F5); `_pdf`, line 2594.
- **Mechanism:** with large dfn or dfd, `m/2*log(m)`, `n/2*log(n)`, `xlogy(n/2-1, x)`, `(n+m)/2*log(m+n*x)` and `betaln` are each of size ~n*log(n) or m*log(m), i.e. 1e6-1e8. Their sum is O(1) near the mode, or a few hundred in the tails. The absolute rounding error of ~eps*1e7 remains (pattern 5): it is a relative error in logpdf, and through `exp` the same absolute error becomes a relative error in pdf. For (1e6, 2), the terms `n/2*log(n*x)` reach ~1.4e8 at x = 1e120, which gives 6.7e7 ulp in pdf at every x. cond (arg) is small (1-2) at the worst points, so this is not conditioning.
- **Regime:** max(dfn, dfd) >~ 1e3. Measured: (1e6, 2) 33 severe pdf points and 12 severe logpdf points; (1e5, 3e5) every bulk pdf point severe; (5, 1e4) and (1000, 1000) moderate (up to 7.7e4 ulp in pdf).
- **Reproducer:** `python accuracy.py refs/f_ref.py --at pdf 1e120 1e6 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e120, 1e6, 2)` | 1.000000014894359e-240 | 1.0000000000000001e-240 | 6.7e+07 | 2 | severe |
  | `pdf(4, 1e6, 2)` | 0.048675027582335335 | 0.04867502764663862 | 5.9e+06 | 1.75 | severe |
  | `logpdf(0.5, 1e6, 2)` | -0.6137056397274137 | -0.6137056388827761 | 6.2e+06 | 6.52e-06 | severe |
  | `pdf(0.999999999999, 1e5, 3e5)` | 77.25470101770703 | 77.25470091693481 | 5.9e+06 | 1 | severe |
  | `logpdf(1.0103814730063743, 1e5, 3e5)` | 2.3333394057117403 | 2.3333394049646015 | 1.4e+06 | 167 | severe |
  | `pdf(1e-80, 5, 1e4)` | 7.436638328769993e-120 | 7.4366383288971e-120 | 7.7e+04 | 1.5 | moderate |
  | `pdf(8, 1000, 1000)` | 1.6949669337434153e-202 | 1.6949669337489265e-202 | 1.46e+04 | 390 | moderate |

- **Fix direction:** use the betaprime/beta form with `t = n*x/m` (removes the `n/2*log(n)` vs `n/2*log(n*x)` cancellation that dominates for (1e6, 2)). Near the mode of large balanced shapes, use a deviance (bd0) / Stirling-difference formulation, or the pdf via Boost `ibeta_derivative(n/2, m/2, y) * dy/dx`, as `scu._beta_pdf` does for beta.

### F7: cdf and sf at dfn = dfd = 1 lose up to 1.4e7 ulp (sc.betaincc(0.5, 0.5, z))

- **Code:** `_sf`, line 2607, and `_cdf`, line 2604. For (1, 1), Boost evaluates `ibetac(0.5, 0.5, z)` with z = n*x/(m + n*x) small (sf at small x) or z = m/(m + n*x) small (cdf at large x).
- **Mechanism:** the beta argument z is passed accurately. The error is inside `ibetac` for a = b = 0.5: `sc.betaincc(0.5, 0.5, 1e-16)` returns 0.9999999905136262 (true 0.9999999936338023, 1.4e7 ulp). `sc.betaincc(0.5, 0.5, 1e-20)` returns 1.0 (true 1 - 6.4e-11). Other shapes are fine: (0.25, 0.25), (0.5, 1) and (1, 0.5) at z = 1e-16 or 1e-17 are within 1 ulp. This points to the a = b = 0.5 (arcsine) special case in Boost `ibeta`, which works with the rounded y = 1 - z. The returned deficit 1 - sf = 9.49e-9 at z = 1e-16 equals (2/pi)*sqrt(2.2e-16), so the effective argument has been rounded to a multiple of eps (pattern 1/10). The exact line in Boost was not inspected (mechanism inferred).
- **Regime:** dfn = dfd = 1 only. sf for x <~ 1e-10, cdf for x >~ 1e10. Errors are 1.2e3 ulp at x = 1e10 (or 1e-10), 1.3e5 at 1e12, 4.9e6 at 1e15 and 1.4e7 at 1e16; beyond 1e16 the result is exactly 1.0 (2.9e5 ulp at 1e20).
- **Reproducer:** `python accuracy.py refs/f_ref.py --at sf 1e-16 1 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1e-16, 1, 1)` | 0.9999999905136262 | 0.9999999936338023 | 1.4e+07 | 3.18e-09 | severe |
  | `cdf(1e20, 1, 1)` | 1.0 | 0.999999999936338 | 2.9e+05 | 3.18e-11 | moderate |
  | `cdf(1e10, 1, 1)` | 0.9999936338020129 | 0.9999936338022766 | 1.19e+03 | 3.18e-06 | moderate |

- **Fix direction:** for (1, 1) the closed forms are `cdf = (2/pi)*atan(sqrt(x))` and `sf = (2/pi)*atan(1/sqrt(x))`. Fix the Boost a = b = 0.5 complement branch (use `asin(sqrt(z))` for the small side instead of going through `1 - z`).

### F8: cdf moderate error for large shapes in the far left tail (Boost ibeta)

- **Code:** `_cdf`, line 2604: `sc.fdtr` -> Boost `ibeta(n/2, m/2, y)`.
- **Mechanism:** the prefactor `y**a * (1-y)**b / B(a, b)` with a, b ~ 1e5 is evaluated inside Boost with an error just above the conditioning (pattern 5/6). This is the same behaviour as beta F8.
- **Regime:** (1e5, 3e5) near mean - 10 sd (cdf ~ 1e-23); (5, 1e4) and (1000, 1000) stay below 310 ulp (minor).
- **Reproducer:** `python accuracy.py refs/f_ref.py --at cdf 0.9496708976337849 1e5 3e5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(0.9496708976337849, 1e5, 3e5)` | 1.1887121952866418e-23 | 1.1887121952858322e-23 | 3.07e+03 | 1.93e+03 | moderate |

- **Fix direction:** upstream (Boost `ibeta_power_terms`); not specific to `f`.

### F9: pdf error grows with |logpdf| in the tails

- **Code:** `_pdf`, line 2594: `np.exp(self._logpdf(x, dfn, dfd))`.
- **Mechanism:** logpdf in the tails is a sum of terms like `(n/2-1)*log(x)` of size L (a few hundred). Its absolute rounding error of ~eps*L becomes a relative error of ~L ulp after `exp` (pattern 5). The harness's cond (arg) for pdf (|x f'/f|, 1-10 here) does not include this, so the points are minor rather than ok.
- **Regime:** all shapes, x in the far tails (x <~ 1e-100 or x >~ 1e20); error ~ |logpdf| ulp, up to 753 ulp. 144 minor pdf points across (29, 18), (1, 1), (0.01, 3) and (2, 0.01).
- **Reproducer:** `python accuracy.py refs/f_ref.py --at pdf 1e30 29 18`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e30, 29, 18)` | 7.873229545335347e-296 | 7.873229545334031e-296 | 753 | 10 | minor |
  | `pdf(1e-280, 0.01, 3)` | 1.9404980115120766e+276 | 1.9404980115118596e+276 | 504 | 0.995 | minor |

- **Fix direction:** compute pdf in the tails as a product of powers with the dominant power formed directly, e.g. `x**(n/2-1) * ...` in the left tail, or accept as inherent to the log-space design.

## RuntimeWarnings

- logpdf, pdf: "overflow encountered in multiply" from `n*x` at x near 1e308 (F1). The harness records warnings per call over the whole grid, so they are also attached to accurate points of the same parameter set. The genuine source is only the n*x overflow, and the affected results are wrong (F1).
- pdf: "underflow encountered in exp" where the true pdf is below the double range (not measured) or subnormal; the results are correct (0 or subnormal).
- cdf, sf, ppf: no warnings, including F2, F3 and F4 (nan and wrong values are returned silently).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full x grid, all parameter sets; both Boost branches (n*x vs m); tails down to 1e-300 | 303 ulp minor at (5, 1e4), x <= 1e-40 (cond (shapes) 228-689); otherwise within conditioning except F2, F7, F8 |
| sf | full x grid, all parameter sets including (1e5, 3e5) | 386 ulp at (1000, 1000), x = 5.5 (cond (arg) 347, minor); otherwise within conditioning except F2, F7 |
| ppf | q in [1e-300, 1 - 3e-16], all parameter sets except (1e5, 3e5) | 831 ulp at (1000, 1000), q = 2.2e-101 (cond 0.0033) and 643 ulp at (0.01, 3), q = 0.1 (cond 200): minor, mechanism inside Boost `ibeta_inv` unclear; high tail <= 21 ulp. Apart from F3/F4, q near 1 and the bulk are fine |
| logpdf | moderate shapes ((1, 1), (0.01, 3)) over the full grid | 1.1 ulp |
| pdf | small shapes, bulk | within conditioning (apart from F9 in the tails) |

## Suspected, not confirmed

- ppf for (1e5, 3e5) was not measured (reference too slow, 5-30 s per point). Beta F6/F7 (Boost `ibeta_inv` converging only to ~2e-11, or a few thousand ulp in far tails of large symmetric shapes) may apply here too.
- F3 nan bands were mapped only on a coarse grid (q = 1e-10 ... 1e-308 in steps of 1e2) and for 9 shape pairs. Beta F1 also found bands where `ibeta_inv` returns non-nan garbage for small dfn (a = n/2 = 0.25-1). Here only (1, 1e4) and (0.01, 3) cover a <= 0.5, and neither showed that.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | `log(cdf)` below the median, `log1p(-sf)` above | log-space incomplete beta: `log I_y(a, b) = a*log(y) + b*log1p(-y) - log(a) - betaln(a, b) + log(2F1(a+b, 1; a+1; y))` with a = n/2, b = m/2, y = n*x/(m+n*x), for small y | cdf underflows for x <~ (m/n)*1e-308**(2/n): `logcdf(1e-30, 29, 18)` = -inf, true -981.8171396959421 |
| logsf | `log(sf)` above the median, `log1p(-cdf)` below | same series in the complement with (b, a) and yc = m/(m+n*x) | sf underflows for x >~ (m/n)*1e308**(2/m): `logsf(1e100, 29, 18)` = -inf, true -2063.2499995602053 |
| isf | `ppf(1 - q)` | `yc = sc.betaincinv(m/2, n/2, q)`, `x = (m/n)*(1 - yc)/yc` (or Boost's complement `ibetac_inv`) | q < 1.1e-16 gives ppf(1) = inf, and q < 1e-8 loses digits: `isf(1e-20, 29, 18)` = inf, true 455.85181280537855 (the betaincinv formula gives 455.85181280537864) |

## Shared code

- `sc.betaincc(0.5, 0.5, z)` (Boost `ibetac`, a = b = 0.5 arcsine branch) is inaccurate for small z: 1.4e7 ulp at z = 1e-16, 1.0 instead of 1 - 6.4e-11 at z = 1e-20 (F7). This affects `sc.fdtr`/`sc.fdtrc` at (1, 1), `beta.sf` at (0.5, 0.5) and `betaprime` at (0.5, 0.5).
- Boost `ibeta_inv` (`sc.fdtri`, `sc.betaincinv`): nan for small q (F3, same as beta F1) and a DBL_MIN clamp for subnormal roots (F4).
- Boost `fisher_f` cdf (`sc.fdtr`/`sc.fdtrc`): overflow of `df1*x` (F2). This is inside the scipy.special functions `fdtr`/`fdtrc`, which are public.
- Boost `ibeta` power terms for large shapes (F8), as in beta F8.

## Existing GitHub issues

(web search; `gh` was not available)

- boostorg/math #1511 (closed upstream in Boost PR #1512): `ibeta_inv` failures behind `special.betaincinv` / `stats.beta.ppf`; same routine as `sc.fdtri` - relates to F3 (and cited for beta F1); F3 reproduces in 1.18.1
- F1, F2, F4-F9: none found specific to `stats.f`, `fdtr`/`fdtrc`/`fdtri` or `betaincc(0.5, 0.5, z)`.

## Verification

Verifier reference: `scratch/f/verify/verify.py` (logpdf in beta-prime form with `log1p(t)`, and for dfn = 2 the identity `-(1 + m/2)*log1p(2x/m)`; cdf/sf by `mp.betainc` on the small side, with cdf = 1 - sf for n*x > m; ppf by secant in log x on log cdf, then checked against cdf(x_ref)/q - 1 < 1e-200; (1, 1) also checked against `(2/pi)*atan`). F8 is computed separately by `mp.quad` of the docstring pdf (`f8_quad.py`). All values agree between 100 and 200 digits to better than 1e-83. Outputs are in `scratch/f/verify/out_*.txt`.

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `logpdf(1e303, 1e6, 0.01)`; `pdf(1e303, 1e6, 0.01)`; `logpdf(1e308, 29, 18)`; `logpdf(1e308, 1e5, 3e5)` | -inf; 0.0; -inf; -inf | -706.49364298487169; 1.4917957032662562e-307; -7080.6882777098887; -106102877.27320181 | verified | mechanism matches line 2600 (`np.log(m + n*x)` with n*x = inf) |
| F2 | `sf(1e308, 2, 0.01)`; `sf(1e303, 1e6, 1)`; `sf(1e304, 1e5, 0.5)`; `cdf(1e308, 2, 0.01)` | 0.0; 0.0; 0.0; 1.0 | 0.028086320670718552; 2.5231318912371084e-152; 7.8012303944430542e-77; 0.97191367932928145 | verified | Boost source not inspected; consistent with the stated overflow: `sc.betainc(0.005, 1, yc)` with `yc` formed as `r/(1+r)`, `r = (m/n)/x`, gives 0.02808632067071856 |
| F3 | `ppf(1e-170, 5, 1e4)`; `ppf(1e-300, 5, 1e4)`; `ppf(1e-180, 10, 100)` | nan; nan; nan | 6.4658457408727912e-69; 6.4658457408727913e-121; 5.0117986927556518e-37 | verified | `sc.betaincinv(2.5, 5000, q)` is fine at 1e-160 and nan at 1e-170, as stated |
| F4 | `ppf(1e-153, 1, 1e4)` | 2.225073858507201e-304 | 1.5708748685746337e-306 | verified | `sc.betaincinv(0.5, 5000, 1e-153)` = 2.225073858507201e-308 (DBL_MIN clamp) confirmed |
| F5 | `logpdf(1e-20, 2, 0.01)`; `logpdf(1e-120, 2, 0.01)`; `logpdf(1e-10, 2, 0.01)` | -1.1102230246251565e-15; same; -2.0100000108769223e-08 | -2.0099999999999998e-18; -2.0099999999999999e-118; -2.0099999799000003e-08 | verified | 2.5e18, 2.5e118 and 6.9e7 ulp; mechanism matches lines 2599-2600 |
| F6 | `pdf(1e120, 1e6, 2)`; `pdf(4, 1e6, 2)`; `logpdf(0.5, 1e6, 2)`; `pdf(0.999999999999, 1e5, 3e5)`; `logpdf(1.0103814730063743, 1e5, 3e5)` | 1.000000014894359e-240; 0.048675027582335335; -0.6137056397274137; 77.25470101770703; 2.3333394057117403 | 1.0e-240; 0.048675027646638621; -0.61370563888277604; 77.254700916934811; 2.3333394049646015 | verified | 6.7e7, 6.0e6, 6.2e6, 5.9e6, 1.4e6 ulp, as reported |
| F7 | `sf(1e-16, 1, 1)`; `cdf(1e20, 1, 1)`; `cdf(1e10, 1, 1)` | 0.9999999905136262; 1.0; 0.9999936338020129 | 0.99999999363380228; 0.99999999993633802; 0.99999363380227654 | verified | the reference matches `(2/pi)*atan` closed form; `sc.betaincc(0.5, 0.5, 1e-16)` returns the same wrong value, while `betaincc(0.5, 1, 1e-16)` is exact. The Boost internals were not inspected; "inferred" is appropriate |
| F8 | `cdf(0.9496708976337849, 1e5, 3e5)` | 1.1887121952866418e-23 | 1.188712195285832156e-23 (by `mp.quad` of the pdf) | verified | 3.07e3 ulp, matches; mechanism (inside Boost `ibeta`) not checked at source level |

Verified by dist-accuracy-verifier: 8 of 8 verified.
