# alpha: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `alpha_gen`, lines 509-560
- **Implemented:** pdf, logpdf, cdf, ppf
- **Inherited (not measured):** logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** none; helpers: `_norm_cdf` (sc.ndtr), `_norm_pdf`, `_norm_logpdf`, `_norm_ppf` (sc.ndtri); special functions: sc.ndtr, sc.ndtri (via the helpers)
- **Reference:** `refs/alpha_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (3.570477051665046), (0.001), (0.5), (2), (40), (1e6), (1e154)
- **Worst finding:** loss in logpdf

## Summary

pdf, cdf and ppf are accurate to within their conditioning over the full
grids for every parameter set from a = 1e-3 to a = 1e6; the largest errors
(9.1e6 ulp in pdf, 2.6e6 ulp in cdf at a = 1e6, 2.6e15 ulp in ppf at q -> 1)
are all below the condition number with respect to the argument. logpdf
returns -inf in a narrow left-tail window x in ~[5.3e-155, 7.5e-155] (any a),
where `_norm_logpdf` squares an argument of size ~1/x that overflows although
-z**2/2 is still finite. For astronomically large a (1e154) the cancellation
in `a - 1.0/x` near the mode makes cdf and logpdf meaningless (cdf 0.5 for
1.0, logpdf +708 for -4.9e273). The inherited sf, logsf, logcdf and isf fail
completely in their tails; isf can even return negative values.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf | any a, x in ~[5.3e-155, 7.5e-155] | z**2 overflows in `_norm_logpdf` (pattern 3) | -inf for -1.78e308 | loss |
| F2 | logpdf, cdf | huge a (1e154), x near 1/a | `a - 1.0/x` cancels; rounding of 1/x dominates (pattern 5) | logpdf 708.28 for -4.86e273 (loss); cdf 0.5 for 1.0 (severe) | loss |

### F1: logpdf overflows to -inf where the true value is finite

- **Code:** `_logpdf`, line 551: `-2*np.log(x) + _norm_logpdf(a-1.0/x) - np.log(_norm_cdf(a))`,
  with `_norm_logpdf` (line 367): `-x**2 / 2.0 - _norm_pdf_logC`.
- **Mechanism:** for small x, z = a - 1/x is about -1/x. `z**2` overflows to
  inf once |z| > 1.34e154, but the needed quantity -z**2/2 stays finite until
  |z| ~ 1.9e154. In between, logpdf returns -inf (pattern 3).
- **Regime:** 5.3e-155 <~ x <~ 7.5e-155, for every a that is small compared
  with 1/x (all measured parameter sets up to 1e6; 30 loss points, 5 per set).
- **Reproducer:** `python accuracy.py refs/alpha_ref.py --at logpdf 6e-155 3.570477051665046`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(5.3e-155, 3.57048)` | -inf | -1.77999288002848e+308 | inf | 2 | loss |
  | `logpdf(6e-155, 3.57048)` | -inf | -1.388888888888889e+308 | inf | 2 | loss |
  | `logpdf(7.4e-155, 3.57048)` | -inf | -9.130752373995618e+307 | inf | 2 | loss |
  | `logpdf(5.3e-155, 0.001)` | -inf | -1.77999288002848e+308 | inf | 2 | loss |
  | `logpdf(7e-155, 0.001)` | -inf | -1.020408163265306e+308 | inf | 2 | loss |
  | `logpdf(5.5e-155, 0.5)` | -inf | -1.652892561983471e+308 | inf | 2 | loss |

- **Fix direction:** evaluate as `-(0.5*z)*z` (or `-0.5*(z*z)` scaled, e.g.
  `-2*(z/2)**2`), so no intermediate exceeds the final value; note
  `_norm_logpdf` is shared with `norm` and other distributions.

### F2: `a - 1.0/x` cancels for huge a near the mode

- **Code:** lines 548, 551, 554: every method forms `a-1.0/x`.
- **Mechanism:** near the mode x ~ 1/a, z = a - 1/x is a small difference of
  two numbers of size a. `1.0/x` carries a rounding error of up to
  0.5 ulp(1/x) ~ a*1.1e-16 in absolute terms, so the computed z has an
  absolute error ~ a*1e-16 (pattern 5). For x = 1e-154, a = 1e154 the
  rounded 1/x equals a exactly, giving z = 0, while the exact z for the
  double x is 9.86e136. cdf then returns ndtr(0)/ndtr(a) = 0.5 instead of 1,
  and logpdf returns the value at the mode (+708) instead of -4.9e273.
- **Regime:** x near 1/a with a*1.1e-16 comparable to or larger than |z|,
  i.e. a >~ 1e16. For moderate a (1e6) the same rounding produces up to
  9.1e6 ulp in pdf and 2.6e6 ulp in cdf, but the tool rates those within
  conditioning (cond (arg) 3.8e7 and 3.0e7). Also at a = 1e18, x = 1e-18
  logpdf is -8110.03 vs -2477.19 (1.0e16 ulp, rated ok, cond (arg) 2.9e16).
  Only at a = 1e154 does the tool flag it.
- **Reproducer:** `python accuracy.py refs/alpha_ref.py --at cdf 1e-154 1e154`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-154, 1e154)` | 0.5 | 1.0 | 2.3e+15 | 0 | severe |
  | `logpdf(1e-154, 1e154)` | 708.2772701089614 | -4.85727729701530e+273 | inf | 2.0e+17 | loss |

  Context: the cdf's cond (arg) is 0 only because phi(z) underflows at the
  exact z; a 1-ulp change of x moves z by ~2e138, so the result is
  extremely sensitive to x. logpdf's cond (arg) is 2.0e17. Both cases are
  practically irrelevant, but the error is avoidable because z is exactly
  determined by the double x.
- **Fix direction:** compute z with a compensated reciprocal:
  `r = 1/x; e = fma(-r, x, 1)/x; z = (a - r) - e`, or `z = (a*x - 1)/x` with
  `a*x - 1` evaluated by fma.

## RuntimeWarnings

- pdf: `overflow encountered in divide`, `divide by zero encountered in divide`,
  `invalid value encountered in multiply`, plus underflow warnings in
  `exp`/`square`/`multiply`, in the left tail x < ~7.5e-155. There
  `1.0/(x**2)` overflows to inf (x**2 subnormal or 0) while
  `_norm_pdf(a-1.0/x)` is 0, and the product inf*0 returns **nan**, for
  example `alpha.pdf(1e-155, 3.57)`, `pdf(1e-200, 3.57)`, `pdf(1e-320, 3.57)`
  all give nan. The true pdf underflows to 0 there, so the harness does not
  measure these points, but nan instead of 0 is wrong output (pattern 3);
  `xpx.apply_where`/`np.where` guarding on `x` or computing
  `exp(-z**2/2 - 2*log(x))` would give 0.
- logpdf: `overflow encountered in square` in and below the F1 window; the
  -inf below x ~ 5.3e-155 is correct (true value < -1.8e308), within the
  window it is the F1 loss.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full x grid 1e-300...1e300 plus z = a - 1/x in [-38, 3], a = 1e-3 ... 1e6 and 1e154 | 9.1e6 ulp at a = 1e6 (cond 3.8e7, F2 mechanism, ok); otherwise <= 309 ulp, all within conditioning |
| logpdf | bulk and right tail, all sets | 3.2e4 ulp in bulk, within conditioning (near zero crossings of logpdf); 0 points beyond conditioning outside F1/F2 |
| cdf | full grid, all sets | 2.6e6 ulp at a = 1e6 (cond 3.0e7, ok); otherwise <= 470 ulp, within conditioning |
| ppf | q in [1e-300, 1 - 2.2e-16], a = 1e-3 ... 1e6 | low tail 1.47 ulp, bulk 25.9 ulp; high tail up to 2.6e15 ulp but within conditioning (cond up to 4.5e15; `a - ndtri(q*ndtr(a))` cancels as q -> 1 in proportion to 1/(1-q)) |

ppf at a = 1e154 could not be measured: the reference inversion is
unavailable at that shape (`--at ppf 0.5 1e154` gives "reference
unavailable").

## Suspected, not confirmed

- pdf, line 548, pattern 4: `1.0/(x**2)` with x**2 subnormal
  (7.5e-155 < x < 1.5e-154) while the pdf ~ 0.4/x**2 would be normal. This
  needs z = a - 1/x = O(1) at such x, i.e. a ~ 1e154, where ulp(a) ~ 2e138:
  a double x near 1/a has |z| either 0 (only for powers of two, where x**2
  is exact) or >~ 1e136 (pdf underflows). The regime is not reachable with
  representable inputs, so nothing could be demonstrated.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | `(ndtr(a) - ndtr(a - h))/ndtr(a)` with h = 1/x, the difference computed as `ndtr(h - a) - ndtr(-a)` when a - h > 0 and, for small h, as `phi(a) * integral_0^h exp(a*s - s**2/2) ds` (series `phi(a)*h*(1 + a*h/2 + (a**2 - 1)*h**2/6 + ...)`) | right tail: sf ~ phi(a)/(ndtr(a)*x) < 1.1e-16; `sf(1e17, 3.570477051665046)` = 0.0, true 6.803429561002819e-21 |
| logsf | log(sf) | log of the sf formula above (log of the series for small h: `log(phi(a)) + log(h) + log1p(...) - log_ndtr(a)`) | same regime: `logsf(1e17, 3.570477051665046)` = -inf, true -46.43686012062691 |
| logcdf | log(cdf) | `log_ndtr(a - 1/x) - log_ndtr(a)` | left tail, a - 1/x < ~-38 (cdf underflows): `logcdf(0.01, 3.570477051665046)` = -inf, true -4654.814128296165 |
| isf | ppf(1 - q) | solve `ndtr(a) - ndtr(a - h) = q*ndtr(a)` for h with the accurate difference above (small q: h ~ q*ndtr(a)/phi(a)), isf = 1/h; for larger q `1/(a - ndtri((1 - q)*ndtr(a)))` is fine | q < ~1e-16: 1 - q rounds to 1 and `_ppf(1.0)` = 1/(a - ndtri(ndtr(a))) is the reciprocal of a rounding error with random sign: `isf(1e-20, 3.570477051665046)` = -15970211444576.227 (negative!), true 6.80342956100282e+16 |

## Shared code

- F1 is located in the shared helper `_norm_logpdf` (line 367,
  `-x**2 / 2.0`), used by `norm` and other distributions. Any caller passing
  arguments with 1.34e154 < |x| < 1.9e154 gets -inf instead of a finite
  value; `_norm_logpdf(` appears 8 times in `_continuous_distns.py`.
  Example: `norm.logpdf(1.5e154)` = -inf, true -1.125e308 (computed by hand
  as -x**2/2 - 0.919; not run through the harness).
- `_norm_pdf` (`exp(-x**2/2.0)`) shares the overflow of `x**2` but the true
  value underflows there, so it is harmless.

## Existing GitHub issues

none found (web search of scipy/scipy issues for "alpha" accuracy and for the `_norm_logpdf` overflow; `gh` was not available)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `logpdf(5.3e-155, 3.570477051665046)` | -inf | -1.77999288002848e+308 | verified | ref: log of docstring pdf, Phi(a) by mp.quad of the normal density; 100/200 digits agree |
| F1 | `logpdf(6e-155, 3.570477051665046)` | -inf | -1.388888888888889e+308 | verified | |
| F1 | `logpdf(7.4e-155, 3.570477051665046)` | -inf | -9.130752373995617e+307 | verified | last digit differs from report (...618), immaterial |
| F1 | `logpdf(5.3e-155, 0.001)` | -inf | -1.77999288002848e+308 | verified | |
| F1 | `logpdf(7e-155, 0.001)` | -inf | -1.020408163265306e+308 | verified | |
| F1 | `logpdf(5.5e-155, 0.5)` | -inf | -1.652892561983471e+308 | verified | mechanism matches line 367 (`-x**2 / 2.0`, x**2 overflows) |
| F2 | `cdf(1e-154, 1e154)` | 0.5 | 1.0 | verified | ref via complementary route 1 - erfc(z/sqrt2)/2; confirmed `1.0/1e-154 == 1e154`, exact z = 9.85624e+136 |
| F2 | `logpdf(1e-154, 1e154)` | 708.2772701089614 | -4.8572772970153e+273 | verified | error is 4.5e15 ulp (relative error 1, wrong sign), not "inf" as in the report table; severity loss stands. Mechanism matches lines 548/551/554 |

Verified by dist-accuracy-verifier: 2 of 2 verified.
