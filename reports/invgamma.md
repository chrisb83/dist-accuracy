# invgamma: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `invgamma_gen`, lines 4930-5017
- **Implemented:** pdf (4965-4967), logpdf (4969-4970), cdf (4972-4973), sf (4978-4979), ppf (4975-4976), isf (4981-4982)
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** none calling into it (`gengamma` with c = -1 is the same distribution but has its own code); helpers: none; special functions: `sc.gammaln`, `sc.gammainc`, `sc.gammaincc`, `sc.gammaincinv`, `sc.gammainccinv` (`sc.psi`, `special.gamma` only in `_stats`/`_entropy`)
- **Reference:** `refs/invgamma_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (4.066899613699307), (0.001), (0.1), (1), (10.5), (1000), (1e6); single points at a = 1e8 with `--at` (pdf/logpdf only)
- **Worst finding:** severe in sf (2.0e10 ulp, relative error 4.3e-6, a = 1e6), from `sc.gammainc` (shared with gamma F1); severe in pdf (7.5e6 ulp at a = 1e6, 1.9e8 ulp at a = 1e8) and logpdf (1.0e6 ulp at a = 1e6, 7.2e6 ulp at a = 1e8) from `invgamma_gen` itself; severe in ppf (5.4e6 ulp) and isf (1.0e6 ulp) at a = 1e6 via `sc.gammainccinv`/`sc.gammaincinv` (shared with gamma F3)

## Summary

invgamma maps every method onto the regularized incomplete gamma functions
with y = 1/x (`cdf = gammaincc(a, 1/x)`, `sf = gammainc(a, 1/x)`,
`ppf = 1/gammainccinv(a, q)`, `isf = 1/gammaincinv(a, q)`), so for small and
moderate shapes cdf and sf are within a few ulp in the bulk and within a few
hundred ulp in the far tails. Two groups of problems remain. (1) Its own
`_logpdf` sums three terms of size `a*log(a)` into an O(log a) result, and
`_pdf = exp(_logpdf)` turns that absolute error into relative error: for
a = 1e3 a few thousand ulp, for a = 1e6 up to 7.5e6 ulp (pdf) and 1.0e6 ulp
(logpdf), for a = 1e8 1.9e8 ulp. (2) The `scipy.special` incomplete-gamma
defects already documented in `reports/gamma.md` show up one-to-one with
y = 1/x: the unconverged `igam` series for a >~ 2e5 (sf 2.0e10 ulp,
cdf 5.4e3 ulp at a = 1e6), and the `igam_fac` prefactor rounding
(1.6e3-1.9e3 ulp at a = 1e3). ppf and isf are `1/gammainccinv` and `1/gammaincinv`, i.e. exactly `1/gamma.isf` and `1/gamma.ppf`, and inherit gamma's F3 (large a: ppf 5.4e6 ulp, isf 1.0e6 ulp at a = 1e6) and F5 (a = 1e-3: up to 3.6e3 ulp); elsewhere they are within ~270 ulp.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | sf, cdf | a >~ 2e5 and 4.5*sqrt(a) < a - 1/x <~ 15*sqrt(a) (x just above 1/a) | `sc.gammainc` returns the 2000-term partial sum of its series (gamma F1; pattern 10, shared) | sf 2.0e10 ulp (rel. 4.3e-6); cdf 5.4e3 ulp | severe |
| F2 | pdf | large a, x near the mode 1/(a+1) and anywhere in the bulk | `exp(_logpdf)` amplifies the absolute error of the cancelling logpdf sum (patterns 5, 3) | 7.5e6 ulp at a = 1e6; 1.9e8 ulp at a = 1e8 | severe |
| F3 | logpdf | large a, x within a few sqrt(a)/a of 1/a | `-(a+1)*log(x) - gammaln(a) - 1/x`: three terms ~a log a cancel to O(log a) (pattern 5) | 1.0e6 ulp at a = 1e6; 7.2e6 ulp at a = 1e8 | severe |
| F4 | ppf, isf | a >~ 2e5, quantile with 1/x in the F1 window (ppf: q -> 1, isf: small q) | `sc.gammainccinv`/`sc.gammaincinv` invert the inaccurate `igam` of F1 (gamma F3; shared) | ppf 5.4e6 ulp, isf 1.0e6 ulp at a = 1e6 | severe |
| F5 | ppf, isf | a << 1 (a = 1e-3), bulk q | `igami` result is `exp(exponent/a)` with \|exponent/a\| ~ 700 (gamma F5; patterns 3, 8; shared) | 3.58e3 ulp at a = 1e-3 | moderate |
| F6 | cdf, sf | a ~ 1e3 with 1/x far from a (\|1/x - a\| > 0.4a) | `exp(a log y - y - lgam a)` prefactor inside `igam`/`igamc` (gamma F6; pattern 5, shared) | cdf 1.9e3 ulp, sf 1.6e3 ulp | moderate |

### F1: sf (and cdf) inherit the unconverged `gammainc` series for large a

- **Code:** `_sf`, line 4979: `sc.gammainc(a, 1.0 / x)`; `_cdf`, line 4973:
  `sc.gammaincc(a, 1.0 / x)`, which for y = 1/x < a is evaluated as
  `1 - igam(a, y)`.
- **Mechanism:** pattern 10, located in `scipy.special` (see
  `reports/gamma.md`, F1): for a > ~2e5 and y just outside the uniform
  asymptotic window `|y - a| < 4.5*sqrt(a)`, cephes `igam_series` stops after
  2000 terms and returns the partial sum. With y = 1/x this is the *right*
  tail of invgamma, so `sf = P(a, y)` is directly wrong (relative error
  4.3e-6 at a = 1e6, y = a - 5 sqrt(a), the same value gamma reports), and
  `cdf = 1 - P` carries the same absolute error (5.4e3 ulp, small only because
  the cdf is close to 1 there).
- **Regime:** a >~ 2e5 and `a - 15*sqrt(a) <~ 1/x < a - 4.5*sqrt(a)`; at
  a = 1e6 the measured error falls from 2.0e10 ulp (y = a - 5 sqrt a) through
  5.1e7 (a - 8 sqrt a), 9.4e5 (a - 10 sqrt a) to 1.7e4 ulp (a - 12 sqrt a).
  Grows rapidly with a (gamma: 0.375 relative at a = 1e8).
- **Reproducer:** `python accuracy.py refs/invgamma_ref.py --at sf 1.0050251256281407e-06 1e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `sf(1.0050251256281407e-06, 1e6)` | 2.7495684457992286e-07 | 2.7495803592705044e-07 | 2.0e+10 | 5.19e+03 | severe |
  | `sf(1.0080645161290323e-06, 1e6)` | 5.240122755715965e-16 | 5.24012281542908e-16 | 5.1e+07 | 8.12e+03 | severe |
  | `sf(1.01010101010101e-06, 1e6)` | 5.446644691873852e-24 | 5.446644693011318e-24 | 9.4e+05 | 1.01e+04 | moderate |
  | `sf(1.0121457489878542e-06, 1e6)` | 9.935030455799691e-34 | 9.935030455837919e-34 | 1.73e+04 | 1.21e+04 | moderate |
  | `cdf(1.0050251256281407e-06, 1e6)` | 0.9999997250431554 | 0.9999997250419641 | 5.37e+03 | 0.00143 | moderate |

- **Fix direction:** none inside `invgamma_gen`; fix `igam_series` in
  `scipy.special` (gamma F1).

### F2: pdf = exp(logpdf) for large a

- **Code:** `_pdf`, line 4967: `np.exp(self._logpdf(x, a))`.
- **Mechanism:** patterns 5 and 3. The pdf inherits the *absolute* error of
  `_logpdf` (F3) as a *relative* error. At a = 1e6 the logpdf terms are
  ~1.4e7, so an absolute error of a few 1e7*eps (~3e-9) becomes ~1e7 ulp of
  the pdf, independent of the size of logpdf itself. Example: at
  x = 1.005e-6 logpdf is off by 1.0e6 ulp of 7.26, i.e. 1.6e-9 absolute,
  which is exactly the 7.5e6-ulp relative error of the pdf.
- **Regime:** error ~ `max((a+1)|log x|, gammaln(a), 1/x)` ulp, so
  ~a log a ulp across the whole bulk of large shapes: 1.5e3-4.5e3 ulp at
  a = 1e3, 1.4e6-7.5e6 at a = 1e6, 1.9e8 at a = 1e8. For small shapes the
  far tails cost a few hundred ulp (|logpdf| ~ 700, minor; see
  "Checked and fine").
- **Reproducer:** `python accuracy.py refs/invgamma_ref.py --at pdf 1e-06 1e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1.0050251256281407e-06, 1e6)` | 1418.6927613171017 | 1418.6927589524414 | 7.5e+06 | 5e+03 | severe |
  | `pdf(1.001001001001001e-06, 1e6)` | 241648110.10019457 | 241648110.37350464 | 5.1e+06 | 1e+03 | severe |
  | `pdf(1e-06, 1e6)` | 398942246.8834673 | 398942247.15624404 | 3.1e+06 | 1 | severe |
  | `pdf(1e-08, 1e8)` | 398942297041.93365 | 398942280068.9808 | 1.9e+08 | 1 | severe |
  | `pdf(0.0009693465699682844, 1000)` | 7974.794931843156 | 7974.794931835262 | 4.46e+03 | 30.6 | moderate |
  | `pdf(0.0005, 1000)` | 1.3722006825340535e-129 | 1.3722006825331258e-129 | 3.04e+03 | 999 | moderate |

- **Fix direction:** same as gamma F2: with y = 1/x,
  `pdf = y * [y**a * exp(-y) / Gamma(a)]`, where the bracket is the
  Lanczos-scaled prefactor `igam_fac(a, y)` that cephes already computes
  without forming the a log a terms (not exposed in `scipy.special`).

### F3: logpdf cancels three terms of size a*log(a)

- **Code:** `_logpdf`, line 4970: `-(a+1) * np.log(x) - sc.gammaln(a) - 1.0/x`.
- **Mechanism:** pattern 5. Near the mode x ~ 1/a, `-(a+1)*log(x)` ~
  a log a (1.38e7 at a = 1e6), `gammaln(a)` ~ a log a - a (1.28e7) and
  `1/x` ~ a (1e6) cancel to `0.5*log(a) + O(1)` (19.8). Each term carries
  half an ulp of its own size, and `log(x)` is additionally multiplied by
  a + 1, so the result has an absolute error of ~`a*log(a)*eps`
  (~3e-9 at a = 1e6) - 1e5 to 1e6 ulp relative to a result of size 7-50.
  Close to the zero crossings of logpdf the relative error is unbounded.
- **Regime:** a >~ 1e3 for x within a few sqrt(a)/a of 1/a (the bulk):
  up to 633 ulp at a = 1e3 (minor), 6e4-1.0e6 ulp at a = 1e6, 7.2e6 at a = 1e8.
  Away from the bulk logpdf is large and the relative error is small.
- **Reproducer:** `python accuracy.py refs/invgamma_ref.py --at logpdf 1.0050251256281407e-06 1e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.0050251256281407e-06, 1e6)` | 7.257491135969758 | 7.2574911343029695 | 1.0e+06 | 689 | severe |
  | `logpdf(1.001001001001001e-06, 1e6)` | 19.30299313541036 | 19.302993136541385 | 2.6e+05 | 51.9 | moderate |
  | `logpdf(1e-06, 1e6)` | 19.804327219724655 | 19.804327220408407 | 1.6e+05 | 0.0505 | moderate |
  | `logpdf(1e-08, 1e8)` | 26.712082624435425 | 26.71208258189054 | 7.2e+06 | 0.0374 | severe |
  | `logpdf(0.0011111111111111111, 1000)` | 3.9767348784544083 | 3.9767348784538497 | 633 | 25.4 | minor |

- **Fix direction:** combine the large terms analytically, e.g. with
  y = 1/x: `logpdf = log(y) + [a*log(y) - y - gammaln(a)]` where the bracket
  is `log(igam_fac(a, y))` computed with the Lanczos/`log1pmx` form
  (`a*log1pmx((y - a)/a)` style) as cephes does for |y - a| < 0.4a.

### F4: ppf and isf inherit F1 through gammainccinv / gammaincinv

- **Code:** `_ppf`, line 4976: `1.0 / sc.gammainccinv(a, q)`; `_isf`,
  line 4982: `1.0 / sc.gammaincinv(a, q)`.
- **Mechanism:** pattern 10, in `scipy.special` (gamma F3): the inverse
  functions Newton-iterate on `igam`/`igamc` and converge to the root of the
  inaccurate function of F1. invgamma `ppf(q, a) = 1/gamma.isf(q, a)`, and the
  values are bit-identical to gamma's: `ppf(0.9999993371296838, 1e6)` is
  1/(gamma's `isf` at the same q, 5.4e6 ulp there too), and
  `isf(1e-8, 1e6)` is 1/(gamma `ppf(1e-8, 1e6)`, 1.0e6 ulp). The final
  `1.0 / ...` adds only half an ulp.
- **Regime:** a >~ 2e5 with `1/x` in `a - 15*sqrt(a) < 1/x < a - 4.5*sqrt(a)`:
  at a = 1e6, ppf for 1 - q in ~[1e-16, 1e-6] and isf for q in the same range
  (q = 1e-17 still 2.3e3 ulp).
- **Reproducer:** `python accuracy.py refs/invgamma_ref.py --at ppf 0.9999993371296838 1e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999993371296838, 1e6)` | 1.0048518213913628e-06 | 1.0048518226056187e-06 | 5.4e+06 | 301 | severe |
  | `ppf(0.999999938945977, 1e6)` | 1.0053093589999162e-06 | 1.0053093594591574e-06 | 2.1e+06 | 3e+03 | severe |
  | `isf(1e-08, 1e6)` | 1.0056333973759122e-06 | 1.005633397607135e-06 | 1.0e+06 | 0.000173 | severe |
  | `ppf(0.9999999943765867, 1e6)` | 1.00573291200352e-06 | 1.0057329121908786e-06 | 8.4e+05 | 3.03e+04 | moderate |
  | `isf(3.4807005884285624e-14, 1e6)` | 1.007526471594304e-06 | 1.0075264715986444e-06 | 1.94e+04 | 0.000132 | moderate |
  | `isf(1e-16, 1e6)` | 1.0082676893816078e-06 | 1.0082676893825348e-06 | 4.14e+03 | 0.00012 | moderate |

- **Fix direction:** none inside `invgamma_gen`; fixing F1 in `igam_series`
  fixes this.

### F5: ppf and isf for a << 1 amplify the rounding of a large exponent

- **Code:** `_ppf`, line 4976 / `_isf`, line 4982 (inside
  `sc.gammainccinv` / `sc.gammaincinv`).
- **Mechanism:** patterns 3 and 8, in `scipy.special` (gamma F5). For a << 1
  `y = gammaincinv(a, q) ~ exp((log q + gammaln(a+1))/a)`, an exponent of
  magnitude |log y| ~ 700 at a = 1e-3; each half-ulp of it costs ~700 ulp of y,
  on top of the inherent `cond (arg) = 1/a = 1e3`. `1/y` passes the relative
  error through unchanged. Values equal 1/(gamma's ppf/isf) at the same q.
- **Regime:** a <~ 1e-2 with q in the bulk such that the result is a huge but
  finite double (1e46 ... 1e301 at a = 1e-3); at a = 0.1 the effect is down
  to ~20-270 ulp (minor).
- **Reproducer:** `python accuracy.py refs/invgamma_ref.py --at ppf 0.4999999999999999 1e-3`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(0.4999999999999999, 1e-3)` | 1.9068662103422793e+301 | 1.9068662103407642e+301 | 3.58e+03 | 1e+03 | moderate |
  | `isf(0.8499999999999999, 1e-3)` | 6.782641219393569e+70 | 6.782641219389141e+70 | 2.94e+03 | 1e+03 | moderate |
  | `ppf(0.36, 1e-3)` | 1.1758463496237435e+194 | 1.1758463496231822e+194 | 2.15e+03 | 562 | moderate |
  | `isf(0.71, 1e-3)` | 9.816933295974443e+148 | 9.816933295971626e+148 | 1.29e+03 | 1e+03 | moderate |

  (The neighbouring point `ppf(0.5, 1e-3)` has 2.58e3 ulp but the tool's
  numerical cond (arg) there came out as 1.2e250 - an artefact of
  differentiating the inverted reference - so it is rated ok; the true
  conditioning is ~1/a = 1e3 as at the adjacent points.)
- **Fix direction:** in `scipy.special` `igami`, do the final Newton step on
  `log P` / `log Q` for a < 1 (gamma F5).

### F6: cdf/sf lose the rounding of the igam prefactor exponent

- **Code:** `_cdf`, line 4973 / `_sf`, line 4979; the loss is in
  `igam_fac(a, y)` inside `scipy.special` (gamma F6), which for
  `|y - a| > 0.4a` computes `exp(a*log(y) - y - lgam(a))`.
- **Mechanism:** pattern 5 inside the special function: the exponent is a
  cancelling sum of terms of size `a log y` (6.2e3-7.6e3 for a = 1e3), so
  the result carries ~`a*log(y)*eps` relative error. The values are
  bit-identical to gamma's: `cdf(0.0005, 1000) = Q(1000, 2000)` = gamma
  `sf(2000, 1000)`, `sf(0.002, 1000) = P(1000, 500)` = gamma `cdf(500, 1000)`.
- **Regime:** a >~ 1e2 with 1/x far from a; for small shapes the same effect
  in the far right tail (1/x < 1e-50) costs 100-400 ulp (minor, see
  "Checked and fine").
- **Reproducer:** `python accuracy.py refs/invgamma_ref.py --at cdf 0.0005 1000`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(0.0005, 1000)` | 6.847349459617758e-136 | 6.847349459614896e-136 | 1.88e+03 | 1e+03 | moderate |
  | `sf(0.002, 1000)` | 3.298272797068251e-86 | 3.2982727970670656e-86 | 1.62e+03 | 501 | moderate |

- **Fix direction:** in `scipy.special`, use the Lanczos prefactor branch of
  `igam_fac` for all a > ~20 (gamma F6).

## RuntimeWarnings

`pdf` emits "underflow encountered in exp" in all seven parameter sets. It
occurs where the true pdf is subnormal or below the double range (far tails),
so the returned 0 / subnormal value is correct; the warning is spurious in
the sense that nothing is wrong with the result. 

`ppf` and `isf` emit "divide by zero encountered in divide" (a = 1e-3 and 0.1) when `gammainccinv`/`gammaincinv` return 0, e.g. `ppf(0.9, 1e-3)`, `isf(1e-17, 1e-3)`, `isf(1e-300, 0.1)`. There the true quantile (~1e1000 for `ppf(0.9, 1e-3)`) exceeds the double range, so `inf` is the correct overflow and no accuracy is lost.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | full grid, a <= 10.5 | 67.7 ulp at a = 10.5 (cond 51.7, so minor: small-scale version of F3); 10.7 ulp for a = 4.07 |
| pdf | full grid, a <= 10.5 | 785 ulp (a = 1e-3), 555 (a = 0.1), 471 (a = 4.07): far tails where \|logpdf\| ~ 700 and exp amplifies its rounding; minor, mechanism of F2 at small scale |
| cdf | full grid, a <= 10.5 | 606 ulp at a = 4.07 (cond 707, within conditioning); 11.2 ulp minor at a = 1e-3, x = 0.5 and 10.6 at a = 0.1 (no clear mechanism) |
| sf | full grid, a <= 10.5 | 401 ulp (a = 4.07, x = 1e70), 241 ulp (a = 1, x = 1e280): far right tail, the igam prefactor effect of F6 at small scale (minor); bulk <= 76 ulp |
| sf | left tail (x -> 0), all a | 2 ulp |
| ppf | full grid, a in {4.07, 1, 10.5, 1e3}; low tail all a | 12.8 ulp (a = 1); 7.5 ulp (a = 1e3) |
| ppf | a = 0.1 | 205 ulp at q = 1 - 1e-13 (cond 1e14, within conditioning); 21.6 ulp minor at q = 0.36 |
| isf | full grid, a in {4.07, 1, 0.1, 10.5, 1e3} | 267 ulp (a = 0.1, q = 1.2e-26), 129 ulp (a = 4.07, q = 2.9e-288): 49 minor points in the left tail of q where the result is huge; same igami mechanism as F5 at small scale |
| isf | high tail (q -> 1), all a | 5 ulp |

## Suspected, not confirmed

- `1.0/x` overflow for x < 5.6e-309 and `1/gammainc*inv(...)` overflow/underflow:
  only reachable where the true result is outside the normal double range
  (subnormal x or results > 1.8e308); not a finding.
- a = 1e8 for cdf/sf/ppf/isf was not measured (mpmath's `gammainc` does not converge reliably there; see gamma's auxiliary reference). By the identity with gamma, gamma's measured 0.375 relative error of `gammainc` at a = 1e8 applies to `invgamma.sf` at x = 1/(a - 4.6 sqrt(a)).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) = log(gammaincc(a, 1/x)) | for small x (y = 1/x > a): `log Q(a, y) = a log y - y - gammaln(a) + log(CF(a, y))`, i.e. the continued fraction of `igamc` with its prefactor kept in log space; `log1p(-gammainc(a, y))` for y < a | cdf underflows once y >~ 700: `logcdf(1e-3, 4.0669)` returns -inf, true -980.68796390274998 |
| logsf | log(sf) = log(gammainc(a, 1/x)) | for large x (y small) `log P(a, y) = a log y - gammaln(a+1) + log(series)` with the Kummer series summed in log space | sf underflows once y**a/Gamma(a+1) < 1e-308: `logsf(1e80, 4.0669)` returns -inf, true -752.42989939204427 |

(Both demonstration values computed with mpmath at 60 digits; SciPy's
results from `stats.invgamma.logsf/logcdf` directly.)

## Shared code

F1, F4, F5 and F6 are defects of `scipy.special`, not of
`invgamma_gen`, which only forwards `1/x` (or q) to them; they are the
findings of `reports/gamma.md` seen through y = 1/x:

- `sc.gammainc` / `sc.gammaincc` (cephes `igam`/`igamc`): gamma F1
  (2000-iteration cap in `igam_series`, severe for a >~ 2e5) appears here as
  F1 (invgamma's right tail, sf severe, cdf moderate); gamma F6 (`igam_fac`
  prefactor) appears as F6.
- `sc.gammaincinv` / `sc.gammainccinv` (cephes `igami`): gamma F3 appears as F4 (ppf/isf, severe for a >~ 2e5), gamma F5 as F5 (a << 1). invgamma's ppf/isf are exactly 1/(gamma's isf/ppf).
- F2/F3 are caused by combining `sc.gammaln` and `np.log` in
  `invgamma_gen._logpdf`; `gengamma` (c = -1 is invgamma) and `gamma` (its
  F2/F4) have the same structure.

## Existing GitHub issues

- #26471 (open): BUG: inaccuracy in special.gammainc for certain large parameters - relates to F1 and F4 (same `gammainc` series non-convergence as gamma F1/F3)
- #17832 (open): Tracking issue: custom methods for continuous distributions - relates to the inherited `log(cdf)`/`log(sf)` under Opportunities
- #6245 (closed, fixed by PR #6258 in 2016): ENH: improve accuracy for ppf-cdf and sf-isf roundtrips for invgamma - earlier fix of cdf/ppf and sf/isf at a = 1 (small and large x). It is a different regime from F1, F4 and F5.
- F2, F3, F5 and F6: none found

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `sf(1.0050251256281407e-06, 1e6)` | 2.7495684457992286e-07 | 2.7495803592705046e-07 | verified | 1.95e10 ulp; reference from an explicit Kummer series P(a,y) = y^a e^-y/Gamma(a+1) * sum y^k/(a+1)_k (100/200 dps agree to 1e-94); SciPy equals `sc.gammainc(1e6, 1/x)` bit for bit; line 4979 matches |
| F1 | `sf(1.0080645161290323e-06, 1e6)` | 5.240122755715965e-16 | 5.2401228154290798e-16 | verified | 5.13e7 ulp |
| F1 | `cdf(1.0050251256281407e-06, 1e6)` | 0.9999997250431554 | 0.99999972504196407 | verified | 5.37e3 ulp (moderate as stated); 1 - series |
| F2 | `pdf(1.0050251256281407e-06, 1e6)` | 1418.6927613171017 | 1418.6927589524414 | verified | 7.51e6 ulp; reference as the product x^(-a-1) * exp(-1/x) * rgamma(a), not in log space; line 4967 matches |
| F2 | `pdf(1e-08, 1e8)` | 398942297041.93365 | 398942280068.98077 | verified | 1.92e8 ulp; `pdf(1e-06, 1e6)` also verified (3.08e6 ulp) |
| F3 | `logpdf(1.0050251256281407e-06, 1e6)` | 7.257491135969758 | 7.2574911343029699 | verified | 1.03e6 ulp; reference is the log of the product form; line 4970 matches |
| F3 | `logpdf(1e-08, 1e8)` | 26.712082624435425 | 26.712082581890542 | verified | 7.17e6 ulp |
| F4 | `ppf(0.9999993371296838, 1e6)` | 1.0048518213913628e-06 | 1.0048518226056185e-06 | verified | 5.44e6 ulp; root of the series P(a, 1/x) = 1 - q, and `cdf(x_ref)` equals q to 1e-200; SciPy equals 1/`gamma.isf` bit for bit; lines 4976/4982 match |
| F4 | `ppf(0.999999938945977, 1e6)` | 1.0053093589999162e-06 | 1.0053093594591573e-06 | verified | 2.06e6 ulp |
| F4 | `isf(1e-08, 1e6)` | 1.0056333973759122e-06 | 1.0056333976071351e-06 | verified | 1.04e6 ulp; sf(x_ref)/q - 1 = 3e-195; equals 1/`gamma.ppf` |
| F5 | `ppf(0.4999999999999999, 1e-3)` | 1.9068662103422793e+301 | 1.9068662103407643e+301 | verified | 3.58e3 ulp (moderate); cdf(x_ref) - q = 0; isf(0.8499999999999999, 1e-3) also verified (2.94e3 ulp) |
| F6 | `cdf(0.0005, 1000)` | 6.847349459617758e-136 | 6.847349459614896e-136 | verified | 1.88e3 ulp (moderate); 1 - series at 300/400 dps and mp.quad at 200 dps agree. Note that y = 1/0.0005 must be taken from the double (with exactly 2000 the value is ...4753e-136); `sf(0.002, 1000)` also verified (1.62e3 ulp) |

Verified by dist-accuracy-verifier: 6 of 6 verified.
