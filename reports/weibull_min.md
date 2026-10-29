# weibull_min: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `weibull_min_gen`, lines 2731-2889
- **Implemented:** pdf, logpdf, cdf, sf, logsf, ppf, isf
- **Inherited (not measured):** logcdf (`rv_continuous._logcdf`, line 2035 of `_distn_infrastructure.py`: log(cdf) below the median, log1p(-sf) above)
- **Related:** dweibull (separate class, not checked here; it has the same `x**(c-1)` / `exp(-x**c)` building blocks); helpers: `_check_fit_input_parameters` (fit only); special functions: sc.expm1, sc.log1p, sc.xlogy (sc.gamma only in `_munp`/fit)
- **Reference:** `refs/weibull_min_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (1.7866166930421596), (1), (2), (3), (0.1), (0.01), (1e6), (1e10), (1e14)
- **Worst finding:** minor in sf (235 ulp); nothing moderate, severe or loss

## Summary

All seven implemented methods are short closed forms and stay within a few
hundred ulp everywhere the result is a normal double. cdf, logpdf and logsf
are within conditioning on the whole grid. Three small error sources go
beyond conditioning, all minor: (1) for c < 1, `exp(-x**c)` amplifies the
rounding of `x**c` by a factor of `x**c`, while cond (arg) is only `c*x**c` (sf, pdf, right tail).
(2) Rounding of `c - 1` for c < 0.5 is amplified by `|log x|` in the pdf left tail.
(3) Rounding of `1.0/c`, and in isf of `-log(q)`, is amplified by
`|log result|` in ppf and isf when the quantile is far from 1. For large c (1e6 to 1e14), pdf
intermediates become subnormal and the error grows to 2e12 ulp, but this stays
inside the (huge) conditioning with respect to x. pdf returns nan instead of 0
when `x**(c-1)` overflows (true value 0, so the tool does not count it).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | sf, pdf | c < 1, right tail with x**c large (up to ~745) | rounding of `pow(x, c)` amplified by `exp` (pattern 6) | 235 ulp | minor |
| F2 | pdf | c < 0.5 (c - 1 inexact), small x | rounding of `c - 1` amplified by `|log x|` (pattern 6) | 173 ulp | minor |
| F3 | ppf, isf | 1/c inexact, quantile far from 1 (small q); isf also small c | rounding of `1.0/c` (and of `-log(q)`) amplified by the power (pattern 6) | 149 ulp | minor |

### F1: sf and pdf amplify the rounding of x**c for c < 1

- **Code:** `_logsf`, line 2796: `-pow(x, c)`; `_sf`, line 2793: `np.exp(self._logsf(x, c))`; `_pdf`, line 2781: `np.exp(-pow(x, c))`.
- **Mechanism:** `t = pow(x, c)` carries up to 0.5 ulp relative error, i.e. an
  absolute error of about `t * 1.1e-16`; `exp(-t)` turns this into a relative
  error of about `t/2` ulp. The condition number of `sf` with respect to x is
  `c*t`, so the computed error is larger than the conditioning by about `1/c`. For
  c >= 1 the error stays within conditioning (e.g. 333 ulp at c = 1.79 with
  cond 1.26e3, rated ok). logsf itself is fine (0 ulp): the error enters only
  through `exp`.
- **Regime:** c < 1 and `x**c` between roughly 20 and 745 (sf or pdf still normal);
  worst near `x**c = 700`.
- **Reproducer:** `python accuracy.py refs/weibull_min_ref.py --at sf 2.8247524899999964e+28 0.1`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `sf(2.8247524899999964e+28, 0.1)` | 9.85967654375865e-305 | 9.859676543758136e-305 | 235 | 70 | minor |
  | `sf(1.0076872625154164e+285, 0.01)` | 3.307553003638408e-308 | 3.3075530036383466e-308 | 124 | 7.08 | minor |
  | `pdf(1e+190, 0.01)` | 2.5279185915692202e-225 | 2.5279185915692025e-225 | 31.5 | 1.78 | minor |

  (cond (shapes) is 4.59e3, 4.65e3 and 342: the result is very sensitive to c,
  but c is an exact input.)
- **Fix direction:** the error comes from rounding `t` itself, so only extra
  precision in `t = exp(c*log(x))` helps (e.g. a double-double `c*log(x)`).
  It is minor, so a fix is optional.

### F2: pdf loses the rounding of c - 1 for c < 0.5 and small x

- **Code:** `_pdf`, line 2781: `c*pow(x, c-1)*np.exp(-pow(x, c))`.
- **Mechanism:** for c < 0.5, `c - 1` lies in [-1, -0.5) and is not exact
  (absolute error up to 2^-54 = 5.6e-17). `pow(x, c-1) = exp((c-1)*log x)`,
  so the relative error is `|log x| * 5.6e-17`, up to 3.9e-14 = 176 ulp at
  x = 1e-300. cond (arg) of pdf is `|c - 1 - c*x**c|`, about 1, so this is
  beyond conditioning. (For 0.5 <= c <= 2, `c - 1` is exact by Sterbenz, so
  the example parameter and c = 1, 2 are not affected.) logpdf is not
  affected in a measurable way: the same absolute error is small compared with
  `|logpdf|`.
- **Regime:** c < 0.5, x << 1; error is about `|log x| * |fl(c-1) - (c-1)| / 2.2e-16` ulp.
- **Reproducer:** `python accuracy.py refs/weibull_min_ref.py --at pdf 1e-300 0.3`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-300, 0.3)` | 2.9999999999999075e+209 | 3.000000000000023e+209 | 173 | 0.7 | minor |
  | `pdf(1e-300, 0.1)` | 1.0000000000000153e+269 | 9.999999999999962e+268 | 85.8 | 0.9 | minor |
  | `pdf(1e-290, 0.01)` | 1.25734151581434e+285 | 1.2573415158143474e+285 | 26.5 | 0.99 | minor |

- **Fix direction:** form `x**(c-1)` as `pow(x, c) / x`, reusing `t = pow(x, c)`
  (`c * t / x * exp(-t)`); keep `pow(x, c-1)` where `t/x` would overflow
  (large c, x > 1).

### F3: ppf and isf amplify the rounding of 1/c

- **Code:** `_ppf`, line 2790: `pow(-sc.log1p(-q), 1.0/c)`; `_isf`, line 2799: `(-np.log(q))**(1/c)`.
- **Mechanism:** `1.0/c` is rounded (relative error up to 0.5 ulp) unless c is
  a power of 2. `L**(1/c) = exp(log(L)/c)` turns this into a relative error of
  `|log result| * err(1/c)`, up to about 0.5*|log result| ulp (at most about 370 ulp
  for a normal result). cond (arg) for small q is `1/c` (ppf) or
  `1/(c*|log q|)` (isf), which is far smaller. In isf there is a second term:
  the 0.5 ulp rounding of `-log(q)` is amplified by `1/c` (up to 50 ulp at
  c = 0.01), while cond is `1/(c*|log q|)`.
- **Regime:** ppf with small q (result << 1) and 1/c inexact, e.g. c = 3,
  1.79, 0.1; isf with small q and small c (result >> 1). Not seen for
  c = 1, 2 (1/c exact) or for large c (the result is close to 1, so log(result) is small).
- **Reproducer:** `python accuracy.py refs/weibull_min_ref.py --at ppf 1.2115276586286942e-26 0.1`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(1.2115276586286942e-26, 0.1)` | 6.812920690585558e-260 | 6.812920690585783e-260 | 149 | 10 | minor |
  | `ppf(1e-300, 1.7866166930421596)` | 1.2157816935951894e-168 | 1.2157816935951687e-168 | 76.7 | 0.56 | minor |
  | `ppf(1e-300, 3.0)` | 1.0000000000000128e-100 | 1e-100 | 57.7 | 0.333 | minor |
  | `isf(6.8129206905797616e-251, 0.01)` | 1.1079318718998692e+276 | 1.1079318718998473e+276 | 89 | 0.174 | minor |

- **Fix direction:** use `np.sqrt`/`np.cbrt` for c = 2, 3; in general,
  improving on this needs `log(L)/c` in extended precision (double-double).
  It is minor, so a fix is optional.

## RuntimeWarnings

- **pdf, large c (c >= 3 in the grid, also 1e6 ... 1e14):** "overflow encountered in power" and
  "invalid value encountered in multiply". Where `(c-1)*log(x) > 709.8`,
  `pow(x, c-1)` is inf and `exp(-pow(x, c))` is 0, so line 2781 returns
  **nan** although the true pdf is 0 (it underflows), e.g. `pdf(2, 1100)` and
  `pdf(1.5, 2000)` give nan. The tool rates it ok because the true value is not a
  normal double, but nan here is wrong (it should be 0). Computing the pdf as
  `c * t / x * exp(-t)` (see F2), or zeroing the result where `exp(-t)` underflows, avoids it.
- **pdf, c < 1 at x = 0:** "divide by zero encountered in power" from
  `pow(0, c-1)`; the returned inf is the correct limit.
- **cdf, logsf, logpdf, sf:** "overflow/underflow encountered in power" for
  very large/small x (`pow(x, c)` is inf or 0); the results (1, -inf, 0, ...)
  are correct limits, so these warnings are spurious.
- **pdf, sf:** "underflow encountered in exp/multiply" in the far right tail
  where the result really underflows; the results are accurate.
- **ppf, isf (c = 0.1, 0.01):** "underflow encountered in power" where the
  quantile itself underflows (correct 0).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | full grid, all 9 parameter sets incl. c = 1e14 | 5.85 ulp |
| cdf | full grid, all parameter sets | 197 ulp in bulk, 9.4e5 ulp at c = 1e14; all within conditioning (cond (arg) up to 1e14) |
| logsf | full grid, all parameter sets | 0 ulp for c <= 1e6; 9.4e5 ulp at c = 1e14 within cond 1e14 |
| sf | c >= 1, all regions | 333 ulp, within cond 1.26e3 (minor cases for c < 1 are F1) |
| pdf | large c (1e6, 1e10, 1e14), x near 1 where `pow(x, c-1)` (x < 1) or `exp(-x**c)` (x > 1) is subnormal (pattern 4) | 1.51e4 / 4.2e8 / 2.0e12 ulp, all within conditioning (cond (arg) 7.2e8 / 7.3e12 / 1.0e14). E.g. `pdf(0.9992802584183207, 1e6)` 5.03e3 ulp, cond 1.0e6, ok. Errors up to 1e15 ulp occur only where the pdf itself is subnormal (not measured). |
| pdf | c = 1.79, 2, 3 bulk and tails | 224 ulp, within cond 1.26e3 |
| ppf | q near 1 (1 - q down to 3e-16), all c | 60.4 ulp, within conditioning |
| isf | q near 1, all c; small q for c >= 1 | 89.9 ulp (c = 0.1, cond 3e16); 4.3 ulp for c = 1.79 |
| ppf, isf | c = 1e6 ... 1e14 | 0 ulp |

## Suspected, not confirmed

none. Every hypothesis in `scratch/weibull_min/notes.md` was tested. The
subnormal-intermediate cases in the pdf (pattern 4, large c) exist but stay within
conditioning with respect to x. The overflow of `pow(x, c-1)` for x < 1 cannot
happen for normal x and c > 0.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | `rv_continuous._logcdf`: `log(cdf)` for x < median, `log1p(-sf)` otherwise (calls `_ppf(0.5)` each time) | `sc._ufuncs._log1mexp(pow(x, c))` = log(-expm1(-x**c)), one line, no median evaluation | none found: `logcdf(30, 1.7866166930421596)` = -6.887175742643963e-190 vs reference -6.88717574264409e-190, 83.4 ulp, cond 778, ok. A custom method would only make it faster and simpler. |

## Shared code

none. Everything is inline in `weibull_min_gen`; no helper or `scipy.special`
function is at fault. `dweibull` uses the same expression style and may
share F1/F2 and the pdf nan (not checked here).

## Existing GitHub issues

none found (web search; `gh` not installed)

## Verification

pending
