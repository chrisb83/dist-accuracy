# exponpow: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `exponpow_gen`, lines 2358-2408
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf = log(cdf) (median split: log1p(-sf) above the median), logsf = log(sf) (median split: log1p(-cdf) below the median)
- **Related:** none; helpers: none; special functions: sc.expm1, sc.log1p, sc.xlogy
- **Reference:** `refs/exponpow_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (2.697119160358469), (1), (1 + 1e-8), (0.5), (0.001), (0.1), (10), (1000)
- **Worst finding:** loss in logpdf

## Summary

cdf and sf are accurate everywhere they were measured (cdf at most 1.83 ulp). sf errors
stay within conditioning. ppf and isf are close to their conditioning everywhere. One
ppf point (b = 0.001) is just past the tool's bound (1.21e3 ulp against cond 1.12e3). logpdf
loses every digit for b = 1 and small x. It also has large relative errors (up to 2.6e10 ulp)
for b close to 1 at small x. The cause is that `1 + xb - exp(xb)` is summed with absolute
rounding error of about eps while the true logpdf is much smaller than 1. The other
problems are small. pdf = exp(logpdf) turns the absolute error of a large |logpdf| into up
to 736 ulp of relative error. Rounding of `1.0/b` costs ppf up to 149 ulp. Separately, pdf and
logpdf return nan (not 0 / -inf) once `x**b` overflows.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf | b = 1 or b near 1, small x (\|logpdf\| << 1) | 1 + log(b) + ... + xb - exp(xb) cancels (patterns 2, 5) | 0.0 for -5.0e-21; 2.6e10 ulp | loss |
| F2 | ppf, isf | small b (1/b = 1000), bulk q | rounding of the inner log1p terms raised to the power 1/b (pattern 6) | 1.21e3 ulp (cond 1.12e3) | moderate |
| F3 | pdf | large \|logpdf\| (x tiny with b != 1, e.g. b = 0.001, x = 1e-302) | exp(logpdf) turns absolute error of logpdf into relative error (pattern 3/4, log round trip) | 736 ulp (cond 1) | minor |
| F4 | ppf | small q, b with inexact 1/b (b = 0.1, 1 + 1e-8, 10, 2.697) | rounding of `1.0/b` multiplied by \|log(ppf)\| (pattern 11) | 149 ulp (cond 10) | minor |

### F1: logpdf cancels when b is close to 1 and x is small

- **Code:** `_logpdf`, line 2395: `f = 1 + np.log(b) + sc.xlogy(b - 1.0, x) + xb - np.exp(xb)`
- **Mechanism:** the exact logpdf is `log(b) + (b-1)*log(x) - (exp(xb) - 1 - xb)`, and the last
  term is about xb**2/2 for small xb. The code forms it as `1 + ... + xb - exp(xb)`. Both `1`
  and `exp(xb) ~ 1` are of order 1, so the sum has an absolute rounding error of about eps
  (pattern 2: exp(x) - 1 without expm1; pattern 5: cancellation of O(1) terms). When the true
  logpdf is much smaller than 1, the relative error is about eps/|logpdf|. For b = 1 the log
  terms vanish and logpdf = -(expm1(x) - x) ~ -x**2/2: for x < ~1e-8 the computed sum is 0.0.
  With b close to 1, `1 + log(b)` already rounds away the low digits of log(b) ~ b - 1.
- **Regime:** |log(b) + (b-1)*log(x)| << 1 and xb small, i.e. b = 1 (any x < ~0.01) or
  |b - 1| small (1e-12 ... 1e-4) with small x. Total loss for b = 1, x <= 1e-8. cond (arg) is
  at most 2 here, so the error is far beyond conditioning. cond (shapes) is large (up to
  1e8 for b = 1 + 1e-8 and 4e21 for b = 1, x = 1e-10), but b is an exact user input. pdf is not
  affected, because exp(logpdf) ~ 1 and an absolute error of eps is a relative error of eps.
- **Reproducer:** `python accuracy.py refs/exponpow_ref.py --at logpdf 1e-10 1.0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-10, 1)` | 0.0 | -5.000000000166667e-21 | inf | 2 | loss |
  | `logpdf(1e-150, 1)` | 0.0 | -5e-301 | inf | 2 | loss |
  | `logpdf(1e-05, 1)` | -5.000000413701855e-11 | -5.000016666708334e-11 | 1.5e+10 | 2 | severe |
  | `logpdf(0.001, 1)` | -5.001667084947314e-07 | -5.00166708341668e-07 | 1.4e+06 | 2 | severe |
  | `logpdf(1e-10, 1.000000000001)` | -2.202793503158773e-11 | -2.2027809045915186e-11 | 2.6e+10 | 0.0454 | severe |
  | `logpdf(1.0000001151292594e-05, 1.00000001)` | -1.0517925286279706e-07 | -1.0517925307615685e-07 | 9.1e+06 | 0.0941 | severe |
  | `logpdf(1e-10, 1.00000001)` | -2.2025850798801372e-07 | -2.2025850801079488e-07 | 4.7e+05 | 0.0454 | moderate |
  | `logpdf(1e-10, 1.0001)` | -0.002202590092660439 | -0.0022025900926604947 | 114 | 0.0454 | minor |
  | `logpdf(1e-10, 0.9999)` | 0.002202580092660389 | 0.002202580092660445 | 114 | 0.0454 | minor |

  (15 loss, 4 severe, 33 moderate points in the grid run, all at b = 1 and b = 1 + 1e-8.
  logpdf(0.1, 1) is 1.51 ulp: at larger xb the true value is large enough.)

- **Fix direction:** `sc.log1p(b - 1) + sc.xlogy(b - 1, x) - (sc.expm1(xb) - xb)`, with a
  series (`xb**2/2 + xb**3/6 + ...`) for `expm1(xb) - xb` when xb is small.

### F2: ppf/isf amplify inner rounding by the power 1/b for small b

- **Code:** `_ppf`, line 2408: `pow(sc.log1p(-sc.log1p(-q)), 1.0/b)`; `_isf`, line 2405:
  `(sc.log1p(-np.log(x)))**(1./b)`.
- **Mechanism:** the relative rounding error of the inner value y (two rounded log
  functions, about 1-2 ulp) is multiplied by 1/b = 1000 in `y**(1/b)` (pattern 6). The
  conditioning with respect to q has the same factor 1/b, so the error is only slightly
  larger than cond (arg). The tool rates it moderate because 1.21e3 ulp > 8 + 1.12e3.
- **Regime:** small b (b = 0.001 measured), q in the bulk. isf at the same b is at 393 ulp
  (cond 324, minor) and 782 ulp (cond 886, ok).
- **Reproducer:** `python accuracy.py refs/exponpow_ref.py --at ppf 0.49999999999999994 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.49999999999999994, 0.001)` | 2.963541238665232e-279 | 2.9635412386644346e-279 | 1.21e+03 | 1.12e+03 | moderate |
  | `isf(0.15, 0.001)` | 6.701239263629116e+26 | 6.701239263629701e+26 | 393 | 324 | minor |

- **Fix direction:** none cheap. The error is within about 1.1x of the conditioning in q.
  Only a more accurate inner value (extended precision for y) would remove the excess.

### F3: pdf loses digits through exp(logpdf) when |logpdf| is large

- **Code:** `_pdf`, line 2391: `np.exp(self._logpdf(x, b))`
- **Mechanism:** logpdf ~ (b-1)*log(x) has magnitude up to ~700 at tiny x. Its absolute
  rounding error of ~|logpdf|*eps becomes a relative error of the same size in exp
  (log-space round trip). cond (arg) of the pdf is only |b - 1| there.
- **Regime:** x tiny with b != 1 (pdf far from 1: b = 0.001, 0.1 with pdf ~ 1e+297, and b = 10
  or 2.697 with pdf ~ 1e-270). 133 minor points, all low tail and bulk, at most 736 ulp.
- **Reproducer:** `python accuracy.py refs/exponpow_ref.py --at pdf 9.332636185032189e-302 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(9.332636185032189e-302, 0.001)` | 4.6171803756169104e+297 | 4.617180375616156e+297 | 736 | 0.999 | minor |
  | `pdf(1e-30, 10)` | 1.0000000000000686e-269 | 1.0000000000000008e-269 | 305 | 9 | minor |

- **Fix direction:** for small xb, evaluate the product `b * x**(b-1) * exp(-(expm1(xb) - xb))`
  directly, so the large factor `x**(b-1)` is rounded only once.

### F4: ppf multiplies the rounding of 1.0/b by |log(ppf)|

- **Code:** `_ppf`, line 2408: `pow(..., 1.0/b)`
- **Mechanism:** `1.0/b` is rounded (for b = 0.1 it becomes exactly 10.0 instead of
  9.99999999999999944...). An exponent error δ changes the result by a relative
  |log(result)|*δ. For ppf ~ 1e-260 that is about 600 * 5.6e-17 = 150 ulp, which matches the
  measurement.
- **Regime:** small q (result far from 1), any b whose reciprocal is inexact: b = 0.1 (149 ulp),
  b = 1 + 1e-8 (34 ulp), b = 10 (17 ulp), b = 2.697 (10.6 ulp). 39 minor points.
- **Reproducer:** `python accuracy.py refs/exponpow_ref.py --at ppf 1.2115276586286942e-26 0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1.2115276586286942e-26, 0.1)` | 6.812920690585558e-260 | 6.812920690585783e-260 | 149 | 10 | minor |
  | `ppf(1e-300, 1.00000001)` | 1.000006907779019e-300 | 1.0000069077790266e-300 | 34.3 | 1 | minor |

- **Fix direction:** low priority. `exp(log(y)/b)` has an error of the same order, so only
  extended precision for log(y)/b removes it. cond (shapes) (597 and 691 here) is larger than
  the error.

## RuntimeWarnings

- pdf, logpdf (lines 2394-2395): `overflow encountered in power`, `overflow encountered in exp`,
  `invalid value encountered in subtract` once `x**b` overflows. `xb - np.exp(xb)` is then
  `inf - inf = nan`. **pdf returns nan where the true value is 0** (e.g.
  `pdf(1e200, 2.697119160358469)` = nan; `pdf(1e100, ...)` = 0.0 correctly), and logpdf returns
  nan instead of -inf. The true values are not normal doubles, so the tool does not measure
  them, but nan for an in-support x is a real defect. Fix direction: `np.where(xb == inf, -inf, ...)` in logpdf.
- logpdf: `overflow encountered in exp` for 709.8 < xb < overflow of x**b. The result is -inf,
  which is correct because the true value is below -1.8e308.
- cdf, sf, ppf, isf: `overflow/underflow encountered in power` in the extreme tails. The
  results are correct (cdf = 1, sf = 0, isf = inf where the true value overflows, ppf = 0
  where the true value underflows). The underflow warnings in pdf/sf come from exp of very
  negative arguments, where the true values also underflow.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid, all parameter sets (x**b from 1e-300 to overflow) | 1.83 ulp |
| sf | full grid, all parameter sets | 1.32e3 ulp at cond 1.15e4 (ok). One minor point: `sf(9765625.0, 0.1)` 326 ulp, cond 74.2. The rounding of x**b (0.5 ulp) is amplified by xb*exp(xb), while cond (arg) is b*xb*exp(xb), i.e. 1/b smaller. No double-precision formula avoids it |
| logpdf | right tail, bulk, b != 1 | 241 ulp at cond 7.1e5 (ok). Minor at b = 0.1, xb ~ 708: `logpdf(3.2096845506516386e+28, 0.1)` 157 ulp, cond 70.9. Same x**b rounding mechanism as sf, beyond cond by 1/b |
| pdf | right tail | 1.23e3 ulp at cond 1.14e4 (ok). Minor at b = 0.1, x = 9765625: 334 ulp, cond 74.6 (x**b rounding as for sf) |
| isf | q from 1e-300 to 1 - 3e-16, all parameter sets | 782 ulp at cond 886 (ok); otherwise <= 89 ulp within conditioning. One minor point (F2) |
| ppf | q near 1 (log1p(-q) of an exact q) | 5.97 ulp |

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | log(sf) above the median | `-sc.expm1(x**b)` (exact closed form, logsf = 1 - exp(x**b)) | sf underflows once exp(x**b) > ~709: `logsf(3, 2.697119160358469)` = -inf, true -255226234.00273789 |
| logcdf | log(cdf) below the median | `log(-expm1(-expm1(xb)))`, and `b*log(x)` when xb underflows (cdf = xb*(1 + O(xb**2))) | x**b underflows (b*log(x) < -745): `logcdf(1e-200, 2)` = -inf, true -921.03403719761827 |

## Shared code

none (only correct uses of sc.expm1, sc.log1p, sc.xlogy; no helpers or related distributions).

## Existing GitHub issues

- #3982 (closed): scipy.stats.exponpow.logpdf(0,1) returns nan. The correct value is 0.0. - historical logpdf at b = 1, x -> 0; relates to F1 (same regime, the cancellation remains for small x > 0)
- No issue found for F1's cancellation at small x > 0, F2 (ppf), or the nan from pdf/logpdf once x**b overflows.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `logpdf(1e-10, 1)` | 0.0 | -5.000000000166667e-21 | verified (loss) | reference via log(b) + (b-1)log(x) - sum_{k>=2} xb^k/k!, 100/200 digits agree; cross-checked by log of the pdf product at 400 digits. Mechanism matches line 2395 (`1 + ... + xb - np.exp(xb)`) |
| F1 | `logpdf(1e-150, 1)` | 0.0 | -5.0000000000000001e-301 | verified (loss) | |
| F1 | `logpdf(1e-05, 1)` | -5.000000413701855e-11 | -5.0000166667083342e-11 | verified (severe) | 1.46e10 ulp |
| F1 | `logpdf(0.001, 1)` | -5.001667084947314e-07 | -5.0016670834166808e-07 | verified (severe) | 1.38e6 ulp |
| F1 | `logpdf(1e-10, 1.000000000001)` | -2.202793503158773e-11 | -2.2027809045915185e-11 | verified (severe) | 2.58e10 ulp |
| F1 | `logpdf(1.0000001151292594e-05, 1.00000001)` | -1.0517925286279706e-07 | -1.0517925307615685e-07 | verified (severe) | 9.14e6 ulp |
| F1 | `logpdf(1e-10, 1.00000001)` | -2.2025850798801372e-07 | -2.2025850801079487e-07 | verified (moderate) | 4.66e5 ulp |
| F2 | `ppf(0.49999999999999994, 0.001)` | 2.963541238665232e-279 | 2.9635412386644346e-279 | verified (moderate) | 1.21e3 ulp; closed-form inverse checked by cdf as mp.quad of the pdf (substitution t = x**b), cdf(x_ref) = q; cond (q) recomputed as 1.12e3. Mechanism matches line 2408 |
| F2 | `isf(0.15, 0.001)` | 6.701239263629116e+26 | 6.701239263629702e+26 | verified (minor) | 394 ulp, quad check sf(x_ref) = q. Mechanism matches line 2405 |

Verified by dist-accuracy-verifier: 2 of 2 verified.
