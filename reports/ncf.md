# ncf: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `ncf_gen`, lines 7923-8016
- **Implemented:** pdf (7984-7985, `scu._ncf_pdf`), cdf (7987-7988, `sc.ncfdtr`), ppf (7990-7992, `sc.ncfdtri`), sf (7994-7995, `scu._ncf_sf`), isf (7997-7999, `scu._ncf_isf`). All five wrap Boost `non_central_f_distribution`, which maps x to the non-central beta variable `y = t/(1+t)`, `t = x*(n/2)/(m/2)`, and calls Boost `non_central_beta_distribution` (Poisson-weighted incomplete-beta series started at the Poisson mode).
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf)
- **Related:** none in scipy.stats; shares Boost `non_central_beta` with nothing else in scipy.stats (ncx2 uses the separate Boost `non_central_chi_squared` family). Special functions: `sc.ncfdtr`, `sc.ncfdtri`, `scu._ncf_pdf`, `scu._ncf_sf`, `scu._ncf_isf` (moments: `scu._ncf_mean/_variance/_skewness/_kurtosis_excess`, not reviewed)
- **Reference:** `refs/ncf_ref.py`, 100 digits (Poisson-weighted incomplete-beta sums with positive terms only, one continued fraction plus the stable direction of the a-recurrence, rigorous Poisson-tail truncation bound); self-checks: PASS in the main run. In the second run they FAIL only for sf at nc = 0, which is SciPy bug F1 (reference = central F sf), and WARN for cdf at (1, 1, 1e-20), which is F3
- **Parameter sets:** (27, 27, 0.41578441799226107), (2, 3, 10), (1, 1, 1), (5, 0.5, 2), (0.5, 10, 200), (100, 1000, 50), (10, 20, 1e-8); second run (pdf, cdf, sf): (27, 27, 0), (1, 1, 1e-20), (10, 20, 1e4)
- **Worst finding:** loss in sf, pdf, cdf, ppf, isf

## Summary

cdf is accurate (<= 89 ulp, mostly within conditioning) except at dfn + dfd = 2. The other methods fail completely in at least one regime:

- **sf at nc = 0 returns -cdf.** This is a negative "probability" at every x, for every dfn and dfd (F1).
- **dfn + dfd = 2.**
  - cdf, sf and ppf return nan in the far left tail (F2).
- **Overflow of t = n*x/m.** sf returns 0 and cdf returns 1 for x > 1.8e308*m/n (F8).
  - When nc is also tiny (<~1e-12), cdf returns 1.0 and sf returns 0.0 in the bulk (F3).
- **Right tail, t = n*x/m large.**
  - pdf loses about t ulp and returns 0 for t > ~1e16 (F4).
  - ppf and isf lose about t ulp because Boost forms `r/(1 - r)` from a non-central beta quantile r close to 1 (F5, F6).
  - ppf and isf clamp at (2^53 - 1)*m/n.
  - isf raises a Python `OverflowError` for quantiles as small as 2e16.
- **sf far right tail, dfd = 1000.** The Boost series stops converging, so sf returns values that are far off or exactly 0 (F7).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | sf | nc = 0 exactly, every x, every dfn, dfd | Boost complement for l == 0 negates the cdf instead of returning 1 - cdf (sign flip without the -1 offset) | -0.49999999999999944 for 0.5 | loss |
| F2 | cdf, sf, ppf | dfn + dfd = 2, x < ~2e-205 (1, 1), ~3e-246 (0.5, 1.5); ppf for q <~ 1e-100 | Boost non-central beta series recurrence has divisor a + b + i - 2 = 0 (inferred) | nan for 3.9e-106 | loss |
| F3 | cdf, sf | dfn + dfd = 2 and nc <~ 1e-12 (cdf), nc <~ 1e-200 (sf), bulk x | same series, tiny Poisson weights (inferred) | cdf 1.0 for 0.805; sf 0.0 for 0.195 | loss |
| F4 | pdf | t = n*x/m >~ 1e3, total loss for t >= ~1e16 | Boost passes only y = t/(1+t) to the non-central beta pdf, which forms 1 - y (pattern 1/10) | 0.0 for 1.69e-224; 7.5e10 ulp | loss |
| F5 | isf | small q with t = n*isf/m >~ 1e3 | `(r/(1-r))*(m/n)` with r the beta-scale complement quantile near 1 (pattern 8); r == 1 raises `OverflowError` | OverflowError for 2.0e16; 9.0e14 for 4.5e19 | loss |
| F6 | ppf | q near 1 with t = n*ppf/m >~ 1e3 (heavy right tails, small dfd) | same `r/(1-r)` (pattern 8); clamp at r = 1 - 2^-53 | 9.0e14 for 2.3e53; 4.0e8 ulp at q = 0.99 | loss |
| F7 | sf (and isf) | dfd ~ 1e3, nc ~ 50, sf <~ 1e-270 | Boost `non_central_beta` complement series does not converge ("Series did not converge") | 0.0 for 3.6e-295; 2.8e14 ulp | loss |
| F8 | sf, cdf | t = n*x/m > 1.8e308 (x > ~1.8e308*m/n) | Boost forms t = x*alpha/beta, which overflows (pattern 3, as f F2) | sf 0.0 for 0.0281; cdf 1.0 for 0.9719 | loss |

### F1: sf returns -cdf when nc = 0

- **Code:** `_sf`, line 7995: `scu._ncf_sf(x, dfn, dfd, nc)` (Boost `cdf(complement(non_central_f_distribution))`).
- **Mechanism:** for nc = 0 the value returned is exactly `-ncf.cdf(x)`. For example, `ncf.sf(10, 2, 3, 0)` = -0.952892492270786 and `ncf.cdf(10, 2, 3, 0)` = 0.952892492270786. Boost's `non_central_beta_cdf` computes the complement for l > 0 by accumulating a series that starts at -1 and negating the result at the end (`if (invert) result = -result`). The l == 0 shortcut returns the central `ibeta` without that offset, so the final negation yields -I_y. This is inferred from the output; the Boost source was not inspected (it is not available in the binary install). nc = 0 is inside the documented domain (`nc >= 0`, `_argcheck`). The docstring says the distribution then "becomes equivalent to the Fisher distribution". isf, cdf, ppf and pdf at nc = 0 are correct.
- **Regime:** every x > 0, every dfn and dfd, nc == 0 exactly. nc = 1e-300 already gives the right sign.
- **Reproducer:** `python accuracy.py refs/ncf_ref.py --at sf 1 27 27 0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1, 27, 27, 0)` | -0.49999999999999944 | 0.5 | inf | 2.05 | loss |

  | `sf(2, 27, 27, 0)` | -0.9613881140565564 | 0.038611885943443627 | inf | - | loss |
  | `sf(1e-100, 27, 27, 0)` | -0.0 | 1.0 | inf | 0 | loss |
  | `sf(1e4, 27, 27, 0)` | -1.0 | 1.0184180089510741e-47 | inf | - | loss |

  In the second harness run, (27, 27, 0) has every sf point flagged as loss. The self-check reports "FAIL: sf(27, 27, 0): reference disagrees with scipy at every bulk point", with x = 0.606, scipy -0.1, ref 0.9. This FAIL is the SciPy bug, not a reference error. At x = 1 the reference gives 0.5, the exact value for dfn = dfd, and `stats.f.sf` agrees with it. The same run also gives "NOTE: logsf(27, 27, 0): scipy returns nan in the bulk" (log of a negative number).

- **Fix direction:** in `_sf`, use `sc.fdtrc(dfn, dfd, x)` where nc == 0. Upstream: apply the `-1` offset (or call `ibetac`) in Boost's l == 0 branch.

### F2: cdf, sf and ppf return nan in the left tail when dfn + dfd = 2

- **Code:** `_cdf`, line 7988: `sc.ncfdtr(dfn, dfd, nc, x)`; `_sf`, line 7995; `_ppf`, line 7992: `sc.ncfdtri` (which inverts the same cdf).
- **Mechanism:** the nan appears only when a + b = (dfn + dfd)/2 = 1 exactly. (1, 1), (0.5, 1.5), (0.4, 1.6), (0.1, 1.9) and (1.5, 0.5) all fail. (1, 1.0000001), (0.5, 0.5), (2, 2) and (4, 2) never fail on x in [1e-307, 1e-10]. It starts where the incomplete-beta term `ibeta(a+1, b, y)` used by the series reaches the underflow range. For (1, 1) the threshold x = 2e-205 matches `betainc(1.5, 0.5, 2e-205)` = 3.8e-308. Boost's series recurrence for the incomplete-beta increments, `xterm *= (a + i - 1) / (x * (a + b + i - 2))`, divides by a + b + i - 2. At i = 1 that divisor is exactly 0 when a + b = 1, so a 0/0 (underflowed `xterm` times 1/0) gives nan. The mechanism is inferred from the a + b = 1 dependence (Boost source not inspected). The central F (`sc.fdtr`) and nc = 0 are not affected.
- **Regime:** dfn + dfd = 2, nc > 0. The x threshold is ~2e-205 for (1, 1, 1) and (1, 1, 5), ~1.2e-199 for (1, 1, 50), 3e-246 for (0.5, 1.5) and 2.5e-176 for (1.5, 0.5). ppf returns nan once the root falls below the threshold: q <~ 1e-101 for (1, 1, 1), q <= 1e-90 for (0.5, 1.5, 1).
- **Reproducer:** `python accuracy.py refs/ncf_ref.py --at cdf 1e-210 1 1 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-210, 1, 1, 1)` | nan | 3.8612941052021565e-106 | inf | 0.5 | loss |
  | `cdf(1e-300, 1, 1, 1)` | nan | 3.861294105202156e-151 | inf | 0.5 | loss |
  | `sf(1e-300, 1, 1, 1)` | nan | 1.0 | inf | 0 | loss |
  | `ppf(1e-110, 1, 1, 1)` | nan | 6.707091574390156e-220 | inf | 2 | loss |
  | `ppf(3.162277660168379e-151, 1, 1, 1)` | nan | 6.707091574390154e-301 | inf | 2 | loss |

- **Fix direction:** upstream in Boost `non_central_beta_p`/`_q`: guard the recurrence when a + b + i - 2 == 0, or compute `xterm` from `ibeta_derivative` directly. In SciPy, fall back to the series with log-scaled terms for x below the threshold.

### F3: cdf = 1 and sf = 0 in the bulk when dfn + dfd = 2 and nc is tiny

- **Code:** `_cdf`, line 7988 (`sc.ncfdtr`); `_sf`, line 7995 (`scu._ncf_sf`).
- **Mechanism:** with a + b = 1 and a tiny Poisson mean, Boost's series returns a saturated value. `cdf(10, 1, 1, nc)` is 1.0 for every nc in [1e-300, 1e-16] and correct for nc >= 1e-12. `sf(10, 1, 1, nc)` is 0.0 for nc <= 1e-200 and emits "Series did not converge". The dependence on a + b = 1 points to the same recurrence divisor as F2 (inferred, Boost source not inspected). dfn + dfd != 2 is not affected: (2, 3) and (10, 20) agree with the central F to <= 5 ulp for nc down to 1e-300.
- **Regime:** dfn + dfd = 2, nc <~ 1e-12 (cdf) or <~ 1e-200 (sf), all x in the bulk.
- **Reproducer:** `python accuracy.py refs/ncf_ref.py --at cdf 10 1 1 1e-20`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(10, 1, 1, 1e-20)` | 1.0 | 0.8050177709578633 | 1.1e+15 | 0.114 | severe |
  | `cdf(10, 0.5, 1.5, 1e-20)` | 1.0 | 0.8915361525121008 | 5.5e+14 | 0.0787 | severe |
  | `sf(10, 1, 1, 1e-300)` | 0.0 | 0.19498222904213663 | inf | 0.469 | loss |

  | `cdf(1.5, 1, 1, 1e-20)` | 1.0 | 0.564094216848975 | 3.5e+15 | 0.276 | severe |
  | `cdf(3, 1, 1, 1e-20)` | 1.0 | 0.6666666666666666 | 2.3e+15 | 0.207 | severe |

  In the second harness run, (1, 1, 1e-20) has 19 severe cdf points in the bulk, all returning 1.0. The cdf left tail returns nan from x = 1e-200 on (11 points, F2).

- **Fix direction:** below a small nc threshold (where the Poisson weights beyond j = 0, 1 are below eps), return `fdtr`/`fdtrc` plus the first-order term in nc. Upstream, the same guard as F2.

### F4: pdf loses about t ulp in the right tail and returns 0 for t > ~1e16

- **Code:** `_pdf`, line 7985: `scu._ncf_pdf(x, dfn, dfd, nc)`. Boost `non_central_f` pdf computes `t = x*alpha/beta`, then `pdf(non_central_beta(alpha, beta, l), t/(1+t)) * (alpha/beta)/((1+t)*(1+t))`.
- **Mechanism:** the non-central beta pdf receives only y = t/(1+t). Internally it forms 1 - y, whose relative error is about t*eps/2 because y is rounded near 1 (pattern 1, pattern 10). That error is then raised to the power b - 1 = m/2 - 1 inside the beta density. The measured error grows in proportion to t. For (2, 3, 10) it is 367 ulp at t = 1e4, 1.1e5 at 1e6, 8.8e6 at 1e8, 1.9e8 at 1e10, 5.0e10 at 1e12 and 1.8e12 at 1e14. Once t > 2/eps, y rounds to 1, 1 - y = 0 and the pdf is exactly 0, although the true pdf decays only like x**(-m/2-1). The f distribution does not have this problem, because `f.pdf` does not go through Boost.
- **Regime:** t = n*x/m above ~1e3 (moderate), above ~1e6 (severe), and >= ~1e16 (loss, returns 0.0). This covers all parameter sets. For heavy right tails (small dfd), it reaches the bulk of the distribution: (5, 0.5, 2) has its 90 % quantile at x = 4490.
- **Reproducer:** `python accuracy.py refs/ncf_ref.py --at pdf 1e16 27 27 0.41578441799226107`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e16, 27, 27, 0.415784)` | 0.0 | 1.6919091043186993e-224 | inf | 14.5 | loss |
  | `pdf(1.5e16, 2, 3, 10)` | 0.0 | 1.2291593373667152e-39 | inf | 2.5 | loss |
  | `pdf(1e16, 1, 1, 1)` | 0.0 | 4.654184080593727e-25 | inf | 1.5 | loss |
  | `pdf(1e11, 5, 0.5, 2)` | 3.6392032671657544e-15 | 3.639142887924416e-15 | 7.47e+10 | 1.25 | severe |
  | `pdf(1e12, 27, 27, 0.415784)` | 1.6914413145532934e-166 | 1.6919091042723285e-166 | 1.25e+12 | 14.5 | severe |
  | `pdf(2e13, 0.5, 10, 200)` | 2.5552430996580702e-65 | 2.55546921764377e-65 | 3.98e+11 | 6 | severe |
  | `pdf(1e8, 27, 27, 0.415784)` | 1.6919087235226091e-108 | 1.691908640565599e-108 | 2.2e+08 | 14.5 | severe |
  | `pdf(1e7, 5, 0.5, 2)` | 3.639142738023679e-10 | 3.639142748729662e-10 | 1.3e+07 | 1.25 | severe |
  | `pdf(1e4, 27, 27, 0.415784)` | 1.6872781571981787e-50 | 1.6872781571946739e-50 | 9.35e+03 | 14.5 | moderate |
  | `pdf(1e4, 1, 1, 1)` | 4.6533499080649294e-07 | 4.6533499080653158e-07 | 374 | 1.5 | minor |

- **Fix direction:** work with yc = 1/(1+t) (formed as `(m/n)/x / (1 + (m/n)/x)` for large x) and evaluate the non-central beta density from both y and yc, e.g. as the Poisson series of `ibeta_derivative` computed with the explicit pair (x, y) that Boost's `ibeta` internals already accept. Alternatively, compute it in log space from `(m/2)*log(yc)`.

### F5: isf raises OverflowError, clamps at (2^53-1)*m/n, and loses about t ulp for small q

- **Code:** `_isf`, lines 7997-7999: `with np.errstate(over='ignore'): return scu._ncf_isf(x, dfn, dfd, nc)`. Boost computes `r = quantile(complement(non_central_beta(alpha, beta, l), q))` and returns `(r/(1-r))*(m/n)`, raising an overflow error when r == 1.
- **Mechanism:** r is a beta-scale quantile close to 1. 1 - r carries an absolute error of about eps/2, so the result has a relative error of about t*eps (pattern 8, and pattern 1 inside the complement quantile). The measured errors track t = n*isf/m. For (27, 27, 0.416), t = 1.7e4 gives 1.6e4 ulp and t = 8.6e7 gives 8.1e7 ulp. For (10, 20, 1e-8), t = 2e10 gives 1.8e10 ulp. When the root search ends at the largest double below 1, r = 1 - 2^-53, and the result is clamped at (2^53 - 1)*m/n, e.g. 9007199254740991 for (1, 1) and 9.007e14 for (5, 0.5). When r rounds to 1, Boost raises its overflow error. SciPy turns this into a Python `OverflowError` exception. The `np.errstate(over='ignore')` around the call does not suppress it, because the error comes from the Boost policy, not from numpy. The exception is raised for results as small as 2e16, far below the double range. Because the call is vectorized, one bad element aborts the whole array call. This also made the harness abort, so isf was measured point by point with `--at` (11 q values per parameter set, from 1e-300 to 1 - 1e-10), not on the full grid.
- **Regime:** small q with t = n*isf/m >~ 1e3 (moderate), >~ 1e6 (severe), clamp or exception once t >~ 1e15. Heavy right tails (small dfd, or small dfn with large nc) reach this already at q = 1e-5 ... 0.1.
- **Reproducer:** `python accuracy.py refs/ncf_ref.py --at isf 1e-213 27 27 0.41578441799226107` (raises)

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1e-213, 27, 27, 0.415784)` | OverflowError | 2.011679359431735e+16 | - | 0.0741 | loss |
  | `isf(1e-100, 2, 3, 10)` | OverflowError | 3.7082019858568276e+67 | - | 0.667 | loss |
  | `isf(1e-10, 1, 1, 1)` | 9007199254740991.0 | 8.664571782420834e+19 | 4.5e+15 | 2 | severe |
  | `isf(1e-5, 5, 0.5, 2)` | 900719925474099.1 | 4.4898972028395586e+19 | 4.5e+15 | 4 | severe |
  | `isf(1e-20, 2, 3, 10)` | 175464920546900.94 | 172119489303332.62 | 8.8e+13 | 0.667 | severe |
  | `isf(1e-100, 27, 27, 0.415784)` | 85740969.07246557 | 85740967.53915018 | 8.1e+07 | 0.0741 | severe |
  | `isf(1e-50, 27, 27, 0.415784)` | 16960.40749091675 | 16960.40749097634 | 1.58e+04 | 0.0741 | moderate |
  | `isf(0.3, 5, 0.5, 0)` | 38.97369385211911 | 38.973693852116426 | 310 | 4.02 | minor |

  Note: two of the clamped values are rated severe instead of loss. The returned number is finite, and its relative error is just below 1 (1 - 1e-4 and 1 - 2e-5).
- **Fix direction:** return x from the complement quantile yc = 1 - r directly: `x = (m/n)*(1 - yc)/yc`. This needs Boost to solve for yc (as `ibetac_inv` does for the central case), not for r. Return inf instead of raising when the result really overflows (`sc.errstate`/policy).

### F6: ppf loses about t ulp and clamps at (2^53-1)*m/n for q near 1

- **Code:** `_ppf`, lines 7990-7992: `sc.ncfdtri(dfn, dfd, nc, q)`. Boost returns `(r/(1-r))*(m/n)` with r the non-central beta quantile.
- **Mechanism:** same as F5 (pattern 8). r is close to 1, so 1 - r has about t*eps relative error. The root on the beta scale also cannot exceed 1 - 2^-53, so the result saturates at (2^53 - 1)*m/n. ppf does not raise; it returns the clamp or inf. The error is well above the conditioning of ppf in q (cond 396 for (5, 0.5, 2) at q = 0.99, error 4.0e8 ulp).
- **Regime:** q near 1 with t = n*ppf/m >~ 1e3. For heavy right tails this is already in the bulk: (5, 0.5, 2) is moderate from q = 0.78 and severe at q = 0.99. (1, 1, 1) is moderate at q = 0.99. Clamping occurs once the true t exceeds ~9e15.
- **Reproducer:** `python accuracy.py refs/ncf_ref.py --at ppf 0.99 5 0.5 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999943765867, 5, 0.5, 2)` | 900719925474099.1 | 4.489897079322275e+32 | 4.5e+15 | 7.1e+08 | loss |
  | `ppf(0.9999999999999627, 5, 0.5, 2)` | 900719925474099.1 | 2.3186669359908273e+53 | 4.5e+15 | 1.1e+14 | loss |
  | `ppf(0.9999999943765867, 1, 1, 1)` | 9007199254740991.0 | 2.7399781405590068e+16 | 3.0e+15 | 3.6e+08 | severe |
  | `ppf(0.99, 5, 0.5, 2)` | 44898975.72423336 | 44898971.722370245 | 4.0e+08 | 396 | severe |
  | `ppf(0.99078944682310521, 5, 0.5, 2)` | 62386926.562600389 | 62386919.029416844 | 5.44e+08 | 430 | severe |
  | `ppf(0.999999938945977, 1, 1, 1)` | 225179981368523.8 | 232444103018695.75 | 1.4e+14 | 3.3e+07 | severe |
  | `ppf(0.9, 5, 0.5, 2)` | 4489.5911743064225 | 4489.5911742631806 | 4.34e+04 | 36 | moderate |
  | `ppf(0.99, 1, 1, 1)` | 8663.376729588388 | 8663.37672957264 | 8.19e+03 | 198 | moderate |
  | `ppf(0.78, 5, 0.5, 2)` | 191.36015825039033 | 191.36015825032874 | 1.45e+03 | 14.2 | moderate |

  Points with q within ~1e-13 of 1 for the other sets ((0.5, 10, 200), (2, 3, 10), (100, 1000, 50)) are also flagged severe, but only 1 to 4 times beyond their large cond (arg) (e.g. 1.1e15 ulp vs cond 6.0e14). They are consistent with the same mechanism but add little.
- **Fix direction:** as F5, solve for yc = 1 - r with the complement target 1 - q (`ncfdtri` could call the complement quantile for q > 1/2) and form `x = (m/n)*(1 - yc)/yc`.

### F7: sf fails in the far right tail for large dfd (series does not converge)

- **Code:** `_sf`, line 7995: `scu._ncf_sf` (Boost `non_central_beta` complement series, `non_central_beta_q`).
- **Mechanism:** for (100, 1000, 50), sf is accurate down to ~1e-241 (73 ulp at x = 40). Then Boost reports "Series did not converge, closest value was ..." and returns the partial sum. It is 1.7e6 ulp off at sf ~ 1e-270, 2.8e14 ulp off at x = 50, and exactly 0.0 from x ~ 51.6 on, while the true sf is still ~1e-279 ... 1e-295. The series starts at the Poisson mode and sums outward. Its stopping test is relative to a partial sum that is near the bottom of the double range, and the terms there are probably subnormal or underflowing, so the test is never met before the iteration limit. The exact loop condition was not inspected (mechanism partly unclear). (10, 1000, 50) shows the same pattern: sf drops from 3.5e-251 to 0 at x = 369. (100, 1000, 0.5), (100, 1000, 500), (100, 100, 50) and (1000, 1000, 10) did not show it on x in [0.3, 1000]. isf inherits the problem. Its root search on the broken sf ends at x = 50.313 for every q <= 1e-280.
- **Regime:** dfd ~ 1e3 with moderate nc (~50), sf below ~1e-250. These sf grid points between x = 10 and 100 for (100, 1000, 50) are not on the default grid and were measured with `--at`.
- **Reproducer:** `python accuracy.py refs/ncf_ref.py --at sf 55 100 1000 50`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(55, 100, 1000, 50)` | 0.0 | 3.6417615681222598e-295 | inf | 401 | loss |
  | `sf(50, 100, 1000, 50)` | 9.301408502146496e-279 | 9.907009232749784e-279 | 2.8e+14 | 393 | severe |
  | `sf(47.697176666471336, 100, 1000, 50)` | 9.999999994808302e-271 | 9.999999998636507e-271 | 1.7e+06 | 389 | severe |
  | `isf(1e-300, 100, 1000, 50)` | 50.31307734023954 | 56.77863961512092 | 5.1e+14 | 0.00248 | severe |
  | `isf(1e-280, 100, 1000, 50)` | 50.31307734023946 | 50.587441887891124 | 2.4e+13 | 0.00254 | severe |

- **Fix direction:** upstream in Boost (scaled or log-space terms, stopping test robust to underflow). In SciPy, the far tail could use a log-space Poisson mixture of `betaincc` terms.

### F8: sf returns 0 and cdf returns 1 when t = n*x/m overflows

- **Code:** `_sf`, line 7995 (`scu._ncf_sf`), and `_cdf`, line 7988 (`sc.ncfdtr`). Boost `non_central_f` forms `t = x*alpha/beta` before mapping it to the beta scale.
- **Mechanism:** for n*x/m > 1.8e308, t is inf, so the beta-scale complement yc = 1/(1+t) is 0. sf then returns 0 and cdf returns 1 (pattern 3), as in f F2 (Boost `fisher_f`). The true sf decays only like x**(-m/2), so for small dfd it is still a normal double, or even far from 0.
- **Regime:** x > 1.8e308*m/n. (5, 0.5, 2) gives 0 from x ~ 1.8e307 on, where the true sf is ~1e-77. (1, 0.01, 1) gives 0 from x ~ 1.8e306 on, where the true sf is ~0.03. This regime is not in the grid run. The `x_points` entries meant for it were written as `3e308/n`, and 3e308 is inf in floating point, so the points were dropped. `refs/ncf_ref.py` now uses t = 1e308 and 1e309 instead, but the grid was not rerun. The points above were measured with `--at`.
- **Reproducer:** `python accuracy.py refs/ncf_ref.py --at sf 1e308 1 0.01 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1e308, 1, 0.01, 1)` | 0.0 | 0.02810981871784956 | inf | 0.005 | loss |
  | `cdf(1e308, 1, 0.01, 1)` | 1.0 | 0.9718901812821504 | 1.3e+14 | 0.000145 | severe |
  | `sf(8e307, 5, 0.5, 2)` | 0.0 | 8.65538922969564e-78 | inf | 0.25 | loss |

- **Fix direction:** form yc = r/(1+r) with r = (m/n)/x without the product (as proposed for f F2). This belongs upstream in Boost `non_central_f`.

## RuntimeWarnings

- sf: "Error in function cdf(non_central_beta_distribution<double>, %1%): Series did not converge, closest value was %1%". It appears for (1, 1, 1) in the far-left-tail call (with the nan results of F2, attached to the whole vectorized call, so 75 accurate points carry it too) and for (1, 1, 1e-300) at x = 10 (F3). It also appears for (100, 1000, 50) in the far right tail (F7, inaccurate results).
- isf: the same warning in the F7 regime. Where r rounds to 1, isf raises `OverflowError` instead of warning (F5), despite the `np.errstate(over='ignore')` in `_isf`.
- No warnings were recorded for pdf, cdf or ppf on the grids.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid, all 7 parameter sets, x from 1e-300 to 1.7e308 incl. n*x/m = 1 and x*n/2 near overflow | 89.3 ulp minor at (0.5, 10, 200), x = 9 (cond 22.8; cdf ~1e-26, large nc, no clear mechanism); 35 ulp at (100, 1000, 50), x = 1.5. Otherwise within conditioning, apart from F2/F3 |
| sf | full grid, all parameter sets | 405 ulp minor at (100, 1000, 50), x = 6.5 (cond 134); 198 ulp at (0.5, 10, 200), x = 1e60 (cond 5). No clear mechanism, so reported here; apart from F1, F2, F3, F7 |
| pdf | left tail, x down to 1e-300 | 641 ulp minor at (2, 3, 10), x = 1e-290 (true value e^-5 = 0.006737946999085467, cond 0); up to 595 ulp for (5, 0.5, 2) and 546 ulp for (0.5, 10, 200). The errors vary irregularly with x. They probably come from the Boost series starting at the Poisson mode k ~ nc/2 and recursing down to j = 0 through power terms y**(a+k) of magnitude exp(-k*|log y|) (pattern 5), but this is not demonstrated, so it is not a finding |
| pdf | bulk, (100, 1000, 50) everywhere | <= 289 ulp (cond up to 101); bulk of the other sets <= 17 ulp |
| ppf | low tail, q down to 1e-300 | 8.55 ulp (100, 1000, 50), apart from F2 |
| ppf | bulk q in [0.01, 0.99] for light-tailed sets | <= 153 ulp ((0.5, 10, 200) at q = 0.99, cond 25), the F6 mechanism at moderate t |
| isf | q in [1e-5, 1 - 1e-10], 11 points per set via `--at` | within conditioning where t < ~100 |

Large nc, (10, 20, 1e4), pdf/cdf/sf (second run): cdf <= 31 ulp and sf <= 213 ulp (minor, cond 10, right tail x >= 1e10, no clear mechanism). pdf shows only the F4 right-tail problem: 6.36e3 ulp at x = 2e4 (t = 1e4), 3.4e9 ulp at x = 1e10, and 0.0 from x = 2e16 on. The far left tail for this set is far below the double range, so it is not measured.

pdf at nc = 0, (27, 27, 0): same F4 behaviour (0.0 at x = 1e16, true 1.378e-224). The cdf is accurate (<= 10.1 ulp).

## Suspected, not confirmed

- Left tail with large nc (Poisson mode far from j = 0): Boost starts the series at the mode, whose term underflows for tiny x. Tested with (0.5, 10, 200) (Poisson mode 100) down to x = 1e-300: no loss found (pdf <= 546 ulp minor, cdf <= 8.75 ulp at x = 1e-10). For (10, 20, 1e4) the left-tail values are below the double range, so this could not be tested there.
- ppf for small q with large shapes (Boost root finding, cf. f F3/F4 `ibeta_inv` nan): ncfdtri uses its own root finder on the non-central beta cdf. No nan was seen apart from F2.
- f F2-like overflow of `x*alpha` in the Boost cdf/sf for x > 1.8e308/alpha: it should affect only dfd <~ 2 with very small sf. It was not reached with a normal true value on the grid.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | log of the Poisson mixture with the terms in log space: `logsumexp_j [log w_j + log f_j(x)]` with `log f_j = log(n/m) + (a+j-1) log t - (a+j+b) log1p(t) - betaln(a+j, b)`. Only a few terms matter in either tail (the j = 0 term dominates as x -> 0, a fixed range around the Poisson mode otherwise) | wherever pdf underflows or is 0: left tail for n > 2, right tail (also F4 for t > 1e16). `logpdf(1e-300, 27, 27, 0.415784)` = -inf, true -8616.160441488155 |
| logcdf | log(cdf) | same log-space Poisson mixture of `log(betainc(a+j, b, y))`. For the left tail the j = 0 term with `betainc` in log form gives `a*log(y) - log(a) - betaln(a, b) + ...` | left tail once cdf underflows: `logcdf(1e-200, 27, 27, 0.415784)` = -inf, true -6201.048783529851 |
| logsf | log(sf) | log-space mixture of `betaincc` terms. The right tail is dominated by `(b)*log(yc) - log(b) - betaln(a+j, b)` terms | right tail once sf underflows (x**(-m/2)): `logsf(1e250, 2, 3, 10)` = -inf, true -860.3522966474511 |

## Shared code

- Boost `non_central_beta` (behind `sc.ncfdtr`, `sc.ncfdtri`, `scu._ncf_pdf`, `scu._ncf_sf`, `scu._ncf_isf`) is responsible for F1, F2, F3, F7. Boost `non_central_f` (the `r/(1-r)` quantile mapping, the y-only pdf call and the `t = x*alpha/beta` product) is responsible for F4, F5, F6 and F8. F8 is the same overflow as f F2 in Boost `fisher_f`. `sc.ncfdtr` and `sc.ncfdtri` are public `scipy.special` functions, so F2, F3 (cdf) and F6 affect `scipy.special` users directly. F1 in `scu._ncf_sf` is reachable only through `stats.ncf.sf`/`logsf`.
- No other scipy.stats distribution uses `non_central_beta`. ncx2 uses Boost `non_central_chi_squared` (separate review).
- The isf `OverflowError` (F5) is a SciPy-side handling issue: the Boost overflow policy is mapped to a Python exception, so `np.errstate` cannot suppress it. Any other `scu` wrapper with a Boost quantile that raises overflow may behave the same way.

## Existing GitHub issues

- #14901 (closed): BUG: stats: distribution methods emit unnecessary warnings from Boost - later comments report `ncf.cdf` overflow and `ncf.sf` divide-by-zero warnings (upstream Boost fixes); about warnings, not the wrong values in F1-F8. No issue found for any finding (web search; `gh` CLI unavailable).

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `sf(1, 27, 27, 0)` | -0.49999999999999944 | 0.5 | verified | Poisson/`mp.betainc` mixture (complement form for t >= 1), 100 = 200 digits. `sf(10, 2, 3, 0)` == `-cdf(10, 2, 3, 0)` exactly. Boost mechanism inferred (source not available), consistent with the output |
| F2 | `cdf(1e-210, 1, 1, 1)`; `ppf(1e-110, 1, 1, 1)` | nan; nan | 3.8612941052021564e-106 (`mp.quad` of the 1F1-form pdf on [0, x] agrees to 14 digits); 6.7070915743901561e-220 (Newton on own cdf, residual 4e-135) | verified | (1, 1.0000001, 1) and (2, 2, 1) give finite correct values at the same x, which supports the a + b = 1 dependence. Divisor mechanism inferred, not checkable in source |
| F3 | `cdf(10, 1, 1, 1e-20)`; `sf(10, 1, 1, 1e-300)` | 1.0; 0.0 | 0.80501777095786335 (also `mp.quad` of pdf); 0.19498222904213665 | verified | 1.1e15 ulp (severe) and 4.5e15 ulp (loss), as reported |
| F4 | `pdf(1e16, 27, 27, 0.41578441799226107)`; `pdf(1e8, ...)` | 0.0; 1.6919087235226091e-108 | 1.6919091043186993e-224; 1.6919086405655989e-108 | verified | own pdf from the closed form exp(-l/2) f_F 1F1((n+m)/2; n/2; z). 2.2e8 ulp at t = 1e8. pdf(8e15) is nonzero and pdf(1e16) is 0, consistent with y = t/(1+t) rounding to 1 at t ~ 2^53 |
| F5 | `isf(1e-213, 27, 27, 0.41578441799226107)`; `isf(1e-10, 1, 1, 1)` | OverflowError; 9007199254740991.0 | 2.0116793594317353e16; 8.6645717824208337e19 | verified | references from Newton steps on own sf (relative residual <= 5e-123, 100 = 200 digits). The clamp equals exactly 2^53 - 1 = r/(1-r) for r = 1 - 2^-53, consistent with the stated mechanism |
| F6 | `ppf(0.99, 5, 0.5, 2)`; `ppf(0.9999999943765867, 5, 0.5, 2)` | 44898975.72423336; 900719925474099.1 | 44898971.722370243; 4.4898970793222752e32 | verified | 4.0e8 ulp (severe) and 4.5e15 ulp (clamped, loss). Solved on the sf with target 1 - q |
| F7 | `sf(55, 100, 1000, 50)`; `sf(50, 100, 1000, 50)` | 0.0; 9.301408502146496e-279 | 3.6417615681222596e-295; 9.9070092327497835e-279 | verified | 4.5e15 and 2.75e14 ulp. Mechanism (series non-convergence) is partly unclear in the report itself and could not be checked in source |
| F8 | `sf(1e308, 1, 0.01, 1)`; `cdf(1e308, 1, 0.01, 1)` | 0.0; 1.0 | 0.028109818717849561 (`mp.quad` with u = 1/x, u = s^200 agrees: 0.028109818717849559); 0.97189018128215044 | verified | sf(1.7e306, 1, 0.01, 1) is still nonzero and sf(1.9e306) is 0, consistent with the n*x/m overflow threshold 1.8e308*m/n. `sf(8e307, 5, 0.5, 2)` = 0 vs 8.6553892296956399e-78 also confirmed |

All cited code lines (7985, 7988, 7992, 7995, 7999) match the source. The Boost-internal mechanisms cannot be checked against source in the binary install. The black-box checks above are consistent with all of them. Verifier scripts: `scratch/ncf/verify/` (vref.py, check1-4.py, mech.py).

Verified by dist-accuracy-verifier: 8 of 8 verified.
