# dweibull: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `dweibull_gen`, lines 1975-2044
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf (sf and isf call `weibull_min._sf` / `weibull_min._isf`)
- **Inherited (not measured):** logcdf, logsf (`rv_continuous._logcdf/_logsf`, `_distn_infrastructure.py` lines 2035-2049: log(cdf) / log(sf) on the tail side of the median, log1p(-sf) / log1p(-cdf) on the other)
- **Related:** weibull_min (`_sf`, `_isf` called directly); helpers: none; special functions: sc.xlogy (sc.gamma only in `_munp`)
- **Reference:** `refs/dweibull_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (2.0685080649914673), (1), (2), (3), (0.3), (0.1), (0.01), (1e6), (1e10)
- **Worst finding:** minor in cdf/sf (235 ulp); nothing moderate, severe or loss in the implemented methods

## Summary

All six implemented methods are short closed forms built from the same
pieces as weibull_min and are within a few hundred ulp wherever the result is
a normal double. logpdf is within 5.5 ulp everywhere; cdf, sf, ppf and isf are
within conditioning for c >= 1 (including c = 1e6 and 1e10). The three
minor error sources are the ones found in weibull_min: `exp(-|x|**c)` amplifies the
rounding of `|x|**c` for c < 1 (cdf left tail, sf right tail, pdf), the
inexact `c - 1` costs up to 173 ulp in the pdf for c < 0.5 and tiny |x|, and
ppf/isf lose up to 92 ulp from the rounding of `1/c` and `-log(2q)` for small c.
pdf returns nan instead of 0 once `|x|**(c-1)` overflows, already at
|x| > 1e288 for the example c = 2.07 (true value 0, so not an accuracy count).
The inherited logcdf/logsf return -inf in the tails, which a one-line closed
form would fix.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf, pdf | c < 1, outer tails with \|x\|**c between ~20 and 745 | rounding of `\|x\|**c` amplified by `exp` (pattern 6) | 235 ulp | minor |
| F2 | pdf | c < 0.5 (c - 1 inexact), tiny \|x\| | rounding of `c - 1` amplified by `\|log x\|` (pattern 6) | 173 ulp | minor |
| F3 | ppf, isf | small c, outer tails (q -> 0 or 1) | rounding of `1/c` and `-log(2q)` amplified by the power (pattern 6) | 91.7 ulp | minor |

### F1: cdf, sf and pdf amplify the rounding of |x|**c for c < 1

- **Code:** `_cdf`, line 2016: `0.5 * np.exp(-abs(x)**c)` (x < 0 branch);
  `_sf`, line 2025: `0.5 * stats.weibull_min._sf(np.abs(x), c)` = `0.5*exp(-pow(|x|, c))` (weibull_min line 2793/2796);
  `_pdf`, line 2008: `np.exp(-ax**c)`.
- **Mechanism:** same as weibull_min F1. `t = |x|**c` carries up to 0.5 ulp
  relative error; `exp(-t)` turns this into about `t/2` ulp. cond (arg) of the
  tail probability is `c*t`, so the error exceeds conditioning by about `1/c`.
  For c >= 1 the same error stays within conditioning (104 ulp at c = 2.07, cond 1.45e3).
  cdf and sf give identical results by symmetry (cdf(-x) = sf(x) in the code).
- **Regime:** c < 1 and `|x|**c` roughly 20 ... 745; cdf for x < 0, sf for x > 0, pdf both signs.
- **Reproducer:** `python accuracy.py refs/dweibull_ref.py --at sf 2.8247524899999964e+28 0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(2.8247524899999964e+28, 0.1)` | 4.929838271879325e-305 | 4.929838271879068e-305 | 235 | 70 | minor |
  | `cdf(-2.8247524899999964e+28, 0.1)` | 4.929838271879325e-305 | 4.929838271879068e-305 | 235 | 70 | minor |
  | `sf(1e+260, 0.01)` | 6.356566969249866e-174 | 6.356566969250027e-174 | 114 | 3.98 | minor |
  | `pdf(1.000000000000022e+200, 0.01)` | 1.860037988010332e-244 | 1.8600379880103188e-244 | 32.3 | 1.99 | minor |

  (cond (shapes) 4.59e3, 2.38e3 for the first rows: very sensitive to c, but c is exact.)
- **Fix direction:** only extra precision in `t = exp(c*log|x|)` helps
  (double-double); minor, fix optional.

### F2: pdf loses the rounding of c - 1 for c < 0.5 and tiny |x|

- **Code:** `_pdf`, line 2008: `c / 2.0 * ax**(c-1.0) * np.exp(-ax**c)`.
- **Mechanism:** same as weibull_min F2. For c < 0.5, `c - 1.0` is inexact
  (absolute error up to 2^-54); `pow(ax, c-1) = exp((c-1)*log ax)` gets a relative
  error of `|log ax| * 5.6e-17`, up to ~176 ulp at |x| = 1e-300, while cond (arg) is ~1.
  For 0.5 <= c <= 2, `c - 1` is exact (Sterbenz) and nothing is seen.
- **Regime:** c < 0.5, |x| << 1, both signs.
- **Reproducer:** `python accuracy.py refs/dweibull_ref.py --at pdf 1e-300 0.3`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-300, 0.3)` | 1.4999999999999538e+209 | 1.5000000000000114e+209 | 173 | 0.7 | minor |
  | `pdf(-1e-290, 0.3)` | 1.4999999999999553e+202 | 1.500000000000011e+202 | 167 | 0.7 | minor |

  (85.8 ulp max for c = 0.1, 27 ulp for c = 0.01.)
- **Fix direction:** `c/2 * t / ax * exp(-t)` with `t = ax**c` (keep `pow(ax, c-1)` where `t/ax` would overflow).

### F3: ppf and isf amplify the rounding of 1/c and -log(2q) for small c

- **Code:** `_ppf`, line 2021: `np.power(-np.log(fac), 1.0 / c)`;
  `_isf`, line 2030: `stats.weibull_min._isf(double_q, c)` = `(-np.log(2q))**(1/c)` (weibull_min line 2799).
- **Mechanism:** same as weibull_min F3. `L**(1/c)` turns a relative error
  `e` in `L = -log(2q)` into `e/c`, and an error in `1/c` into
  `|log result| * err(1/c)`. For c = 0.01 the 0.5 ulp rounding of `L` becomes
  up to 50 ulp, plus the `1/c` term; cond (arg) is `1/(c*|log 2q|)`, far
  smaller. Unlike weibull_min, small q gives a huge |result| here, so c = 3 or
  2.07 are fine (3.2 ulp); near the median (tiny results) the ppf/isf
  conditioning is huge and the errors are within it.
- **Regime:** c <= 0.1, q < ~1e-13 (ppf) / q > 1 - 1e-13 mirror; isf identical with q <-> 1 - q.
- **Reproducer:** `python accuracy.py refs/dweibull_ref.py --at ppf 1.9573417814877456e-238 0.01`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1.9573417814877456e-238, 0.01)` | -5.901689657405238e+273 | -5.901689657405117e+273 | 91.7 | 0.183 | minor |
  | `isf(1.9573417814877456e-238, 0.01)` | 5.901689657405238e+273 | 5.901689657405117e+273 | 91.7 | 0.183 | minor |
  | `ppf(1e-300, 0.01)` | -7.764149833303492e+283 | -7.76414983330335e+283 | 82.4 | 0.145 | minor |

  (cond (shapes) 630 and 654; c = 0.1 max 18.7 ulp.)
- **Fix direction:** extended-precision `log(L)/c`; minor, fix optional.

## RuntimeWarnings

- **pdf, any c > 1:** "overflow encountered in power" and "invalid value
  encountered in multiply". Where `(c-1)*log|x| > 709.8`, `ax**(c-1)` is inf
  and `exp(-ax**c)` is 0, so line 2008 returns **nan** although the true pdf is
  0 (it underflows). Seen in the run for c = 2.07 (|x| >= 1e290), c = 3
  (|x| >= 1e160), c = 1e6 (|x| >= 2), c = 1e10 (|x| >= 1.000001); 192 grid points in total.
  Examples: `pdf(1e290, 2.0685080649914673)` = nan, `pdf(1.5, 2000)` = nan, `pdf(1.000001, 1e10)` = nan.
  The tool rates these ok (true value 0 is not a normal double), but nan is wrong;
  same defect as weibull_min (RuntimeWarnings there). The F2 fix form or zeroing where
  `exp(-t)` underflows avoids it.
- **pdf, c < 1 at x = 0:** "divide by zero encountered in power"; the returned inf is the correct limit.
- **cdf, sf, logpdf:** "overflow/underflow encountered in power", "underflow encountered in exp/multiply"
  for very large/small |x| or large c; results (0, 0.5, 1, -inf) are correct limits; spurious.
- **ppf, isf (c = 0.01):** "underflow encountered in power" where the quantile itself underflows near the median (correct 0).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | full grid (both signs, \|x\| 1e-300 ... 1e300), all 9 parameter sets incl. c = 1e10 | 5.49 ulp |
| cdf, sf | inner side (cdf x > 0, sf x < 0: `1 - 0.5*exp(..)`, no cancellation since the result is >= 0.5) | 0 ulp in the high-tail region, 46.2 ulp bulk |
| cdf, sf | c >= 1 outer tails; c = 1e6, 1e10 near \|x\| = 1 | 210 ulp, within cond 7.0e8 |
| pdf | c = 2, 2.07, 3 bulk and tails; c = 1e6, 1e10 near \|x\| = 1 (subnormal intermediates, pattern 4) | 225 ulp (cond 1.42e3); 1.51e4 / 4.2e8 ulp within cond 7.2e8 / 7.3e12 |
| ppf, isf | q around the median (0.5 +- 1e-15 ... 1e-2; `1 - q` exact by Sterbenz, `2q` exact), q near 1, c >= 0.3 everywhere, c = 1e6/1e10 | 94 ulp (c = 0.1 near median, cond 9e16); 4.76 ulp for c >= 0.3 |

## Suspected, not confirmed

none. All hypotheses in `scratch/dweibull/notes.md` were tested; the median
branch of ppf/isf (H6) and the logpdf combination (H7) are fine. Overflow of
`ax**(c-1)` for |x| < 1 needs subnormal x (not measured).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | `log(cdf)` for x < 0 (median), `log1p(-sf)` for x >= 0; evaluates `_ppf(0.5)` each call | x < 0: `-log(2) - \|x\|**c`; x >= 0: `log1p(-0.5*exp(-x**c))` | x < 0 with \|x\|**c > ~745: cdf underflows, -inf. `logcdf(-30, 2)` = -inf vs reference -900.6931471805599 (loss) |
| logsf | `log(sf)` for x > 0, `log1p(-cdf)` otherwise | x > 0: `-log(2) - x**c`; x <= 0: `log1p(-0.5*exp(-\|x\|**c))` | x > 0 with x**c > ~745: `logsf(30, 2)` = -inf vs reference -900.6931471805599 (loss) |

## Shared code

`_sf` and `_isf` call `weibull_min._sf` and `weibull_min._isf`; F1 (sf part)
and F3 (isf part) originate there and are the same mechanisms as
weibull_min F1 and F3 (`reports/weibull_min.md`). `_pdf`, `_cdf` and `_ppf` are
independent copies of the same expressions, so F1 (pdf, cdf), F2 and the pdf
nan (inf*0) repeat weibull_min F1, F2 and its RuntimeWarnings finding; a fix
should be applied to both classes. No `scipy.special` function is at fault.

## Existing GitHub issues

none found (web search; `gh` not installed). Related: merged PR #18232 introduced the current `_sf`/`_isf` (delegating to weibull_min) - relates to F1, F3.

## Verification

pending
