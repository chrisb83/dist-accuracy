# gumbel_l: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `gumbel_l_gen`, lines 4433-4508
- **Implemented:** pdf, logpdf, cdf, sf, logsf, ppf, isf
- **Inherited (not measured):** logcdf = log(cdf)
- **Related:** mirror of gumbel_r (separate class, no shared code; see `reports/gumbel_r.md`), gumbel_r is genextreme with c = 0 (`reports/genextreme.md`); helpers: none; special functions: sc.expm1, sc.log1p
- **Reference:** `refs/gumbel_l_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in ppf and isf at the single point where the quantile is 0 (cond (arg) ~ 3e16 to 5e16)

## Summary

All seven implemented methods are accurate over the full default grid
(x up to +-1e300, q down to 1e-300 and up to 1 - 3e-16). pdf and sf lose
up to 315 and 186 ulp in the right tail (x of 4 to 6.5), which is within
their conditioning (cond (arg) ~ 4.6e3 there). The only flagged results are
`ppf(0.6321205588285577)` and `isf(0.36787944117144233)`, the doubles
nearest 1 - 1/e and 1/e, where the true quantile is -3.4e-17 and SciPy
returns exactly 0.0. The tool rates this a loss because 0 is returned, but
the problem is ill-conditioned there (cond (arg) 5.1e16 and 3.0e16), and at
the neighbouring doubles SciPy's errors of ~1e14 ulp are within
conditioning. The inherited logcdf = log(cdf) breaks down in the left tail
(x < -708), where the closed form is easy.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf, isf | q = 1 - 1/e (ppf), q = 1/e (isf), i.e. quantile near 0 | log of an inner value that rounds to exactly 1 (patterns 2, 8) | 0.0 for -3.4e-17 | loss |

### F1: ppf and isf return 0 where the quantile is a tiny nonzero number

- **Code:** `_ppf`, line 4476: `np.log(-sc.log1p(-q))`; `_isf`, line 4485:
  `np.log(-np.log(x))`.
- **Mechanism:** near the quantile 0 the inner value `y = -log1p(-q)`
  (or `-log(q)`) is 1 + d with |d| ~ 1e-16. It is rounded to a double before
  the outer `log`, so the absolute error in `y` (up to 1.1e-16) becomes the
  absolute error of the result, while the result itself is only of size d.
  At the double nearest 1 - 1/e, `y` rounds to exactly 1.0 and `log(1.0) = 0`.
  This is the conditioning of the problem (log(y) near y = 1, patterns 2
  and 8): cond (arg) = q / ((1 - q) |log(1 - q)| |ppf|) diverges as ppf -> 0.
  The outer `log` cannot be replaced by `log1p` without computing
  `y - 1` from q in higher than double precision.
- **Regime:** only within a few ulp of q = 1 - 1/e (ppf) or q = 1/e (isf).
  One ulp away, SciPy gives 2.2e-16 against 2.7e-16 (7.7e14 ulp, rated ok
  because cond (arg) is 6.4e15).
- **Reproducer:** `python accuracy.py refs/gumbel_l_ref.py --at ppf 0.6321205588285577`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.6321205588285577)` | 0.0 | -3.3784855259134224e-17 | inf | 5.1e+16 | loss |
  | `isf(0.36787944117144233)` | 0.0 | -3.3784855259134224e-17 | inf | 3.0e+16 | loss |
  | `ppf(0.6321205588285578)` | 2.2204460492503128e-16 | 2.68005052078406e-16 | 7.7e+14 | 6.4e+15 | ok |
  | `isf(0.3678794411714424)` | -2.2204460492503136e-16 | -1.8467980892790434e-16 | 9.1e+14 | 5.4e+15 | ok |

  (The tool rates the 0.0 results as loss. The relative error of 1 is the
  same size as what the conditioning predicts, so in practice this is a
  limit of double precision, not a defect of the formula.)
- **Fix direction:** none needed in practice. A fully accurate result
  would need `-log1p(-q) - 1` (or `-log(q) - 1`) in double-double near
  q = 1 - 1/e (1/e), followed by `log1p`.

## RuntimeWarnings

The harness recorded `overflow encountered in exp` and
`underflow encountered in exp` for pdf, logpdf, cdf, sf and logsf.
Overflow comes from `np.exp(x)` for x > 709.78 (the true values of
logpdf/logsf are then below -1.8e308 and pdf/sf are far below the double
range, so -inf and 0 are correct). Underflow comes from `np.exp(x)` for
x < -745 (cdf true value subnormal or smaller) and from the outer `exp` in
pdf/sf. In every case seen the returned value is correct or the true value
is outside the normal double range: the warnings are spurious as far as
accuracy goes. The inherited logcdf gives `underflow encountered in exp` on
inaccurate results (see Opportunities).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, x from -1e300 to 1e300, the underflow edge x ~ 6.5-6.6 | 315 ulp in the high tail (cond (arg) 4.57e3, ok); low tail 2.97 ulp, bulk 1.58 ulp |
| logpdf | full grid, including x near 0 (mode, value -1) and x up to 709.8 | 0.739 ulp |
| cdf | full grid, left tail down to x = -745 (cdf ~ e^x), right tail | 1.58 ulp |
| sf | full grid, right tail up to the underflow edge | 186 ulp in the high tail (cond (arg) 4.63e3, ok); bulk 0.68 ulp |
| logsf | full grid, x up to 709.8 | 0.5 ulp |
| ppf | q from 1e-300 to 1 - 3e-16, q near 1 - 1/e +- 1e-10 | 0.712 ulp except F1 |
| isf | q from 1e-300 to 1 - 3e-16, q near 1/e +- 1e-10 | 0.735 ulp except F1 |

The pdf error comes from `exp(logpdf)` (line 4467): the absolute rounding
error of `x - exp(x)` (~ e^x eps) is turned into a relative error by the
outer `exp`, which is exactly the conditioning x (e^x - 1) of the pdf;
sf = `exp(-exp(x))` behaves the same way. Both are within conditioning.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | `log(-expm1(-exp(x)))`, and for x < about -5 use `x + log1p(-t/2 + t**2/6 ...)` with t = e^x, or simply `x - t/2` once t < 1e-8; equivalently `scipy.special._ufuncs._log1mexp(-exp(x))` with an `x`-based branch where exp(x) underflows | cdf = -expm1(-e^x) ~ e^x becomes subnormal for x < -708.4 and 0 for x < -745: `logcdf(-740)` = -740.009255122538 vs -740.0 (5.6e10 ulp, severe); `logcdf(-750)` = -inf vs -750.0 (loss) |

## Shared code

none (the class uses no helpers; `sc.expm1` and `sc.log1p` are called
correctly and showed no inaccuracy). gumbel_r is a separate class with the
mirrored formulas (it implements logcdf but inherits logsf, the mirror of
this logcdf gap), so the same F1 point (quantile near 0) and the mirrored
opportunity should appear there.

## Existing GitHub issues

none found (searched the web for gumbel_l accuracy/ppf/logcdf issues; `gh` unavailable). The mirrored expression is in gumbel_r F1 and genextreme F4.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(0.6321205588285577)` | 0.0 | -3.3784855259134225844e-17 | verified | Reference by Newton on cdf = mp.quad of pdf (100 and 200 digits agree to 86 digits; matches the closed form to 1e-186). Relative error 1 (4.5e15 ulp); cond (arg) ~ 5.1e16, so it is consistent with conditioning, as the report says. `-log1p(-q)` is exactly 1.0 (line 4476), so the mechanism is confirmed |
| F1 | `isf(0.36787944117144233)` | 0.0 | -3.3784855259134225844e-17 | verified | Same point (q_isf + q_ppf == 1 exactly in doubles). `-log(q)` is exactly 1.0 (line 4485) |
| F1 (context) | `ppf(0.6321205588285578)` | 2.2204460492503128e-16 | 2.6800505207840598667e-16 | verified | 7.72e14 ulp, matches the report (ok, within cond) |
| F1 (context) | `isf(0.3678794411714424)` | -2.2204460492503136e-16 | -1.846798089279043321e-16 | verified | 9.11e14 ulp, matches the report (ok, within cond) |

Verified by dist-accuracy-verifier: 1 of 1 verified.
