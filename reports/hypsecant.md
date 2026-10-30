# hypsecant: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `hypsecant_gen`, lines 4822-4865
- **Implemented:** pdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logpdf = log(pdf); logcdf and logsf = generic `rv_continuous` versions that switch at the median (`_distn_infrastructure.py` lines 2035-2050: logcdf = log(cdf) for x < median, log1p(-sf) otherwise; logsf mirrored)
- **Related:** none; helpers: none; special functions: none (numpy `cosh`, `exp`, `arctan`, `tan`, `log` only)
- **Reference:** `refs/hypsecant_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** none beyond conditioning

## Summary

All five implemented methods are accurate over the full grid (x in +-1e-300 ... +-745, q in 1e-308 ... 1 - 2**-53). pdf, cdf and sf stay within 1.6 ulp; the only zeros returned (pdf for |x| > 710.5, cdf/sf in the far tails) are where the true value is already subnormal. ppf and isf show large ulp errors (up to 4.1e15 ulp near q = 0.5 and 6.0e13 ulp near q = 1), but every point is within the conditioning with respect to q, so no finding is reported. The inherited log methods return -inf for |x| > ~708 where the true value is about -|x| - 0.45; closed forms exist (see Opportunities).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|

none

## RuntimeWarnings

- pdf: `overflow encountered in cosh` / `underflow encountered in divide` for |x| > ~708; returned values (subnormal or 0) are correct, where the true pdf is below the normal range.
- cdf, sf: `overflow encountered in exp` (e.g. `cdf(745)`, arctan(inf) = pi/2 gives the correct 1.0) and `underflow encountered in exp/arctan/multiply` in the far tail (e.g. `cdf(-740)` = 2.67e-322, true value subnormal). Spurious: results are accurate or the true value is subnormal.
- ppf, isf: `underflow encountered in tan` at the smallest q; results accurate (0 ulp in the low tail).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid incl. |x| = 700 ... 745 (cosh overflow edge at 710.5, true pdf subnormal there) | 0.988 ulp |
| cdf | full grid; left tail down to exp underflow (x = -745), right tail to x = 745 | 1.55 ulp |
| sf | full grid (mirror of cdf) | 1.55 ulp |
| ppf | q in 1e-308 ... 1 - 2**-53, q = 0.5 +- 2**-53, +-1e-10, +-1e-5 | 4.1e15 ulp at cond 9.0e15 (all within conditioning); low tail 0 ulp |
| isf | same q grid | 4.1e15 ulp at cond 9.0e15 (all within conditioning); low tail 0 ulp |

Context for ppf/isf (all `ok` by the tool):

| call | scipy | ulp | cond (arg) | severity |
|---|---|---|---|---|
| `ppf(0.5000000001)` | 3.1415936513962265e-10 | 1.1e+09 | 5.0e+09 | ok |
| `ppf(0.50001)` | 3.1415926540965394e-05 | 6.12e+03 | 5e+04 | ok |
| `ppf(0.9999999999)` | 22.574267517065337 | 1.2e+08 | 4.4e+08 | ok |
| `ppf(0.9999999999999999)` | 35.800106650945416 | 6.0e+13 | 2.5e+14 | ok |

Near the median, `log(tan(pi*q/2))` takes the log of a number close to 1
(pattern 2), and near q = 1 the argument `pi*q/2` approaches the pole of tan
(and `np.pi/2` is not exactly pi/2). Both losses stay below the conditioning
with respect to q, so they are not findings. Because `q - 0.5` is exact in
floating point, the inverse-Gudermannian form `asinh(tan(pi*(q - 0.5)))`
(equivalently `2*atanh(tan(pi*(q - 0.5)/2))`) would give near-full relative
accuracy around the median, and `isf(q) = -ppf(q)` could use
`-log(tan(pi*q/2))` for q < 0.5 and `asinh(tan(pi*(0.5 - q)))` otherwise; this is an optional
improvement, not a defect.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `log(2/pi) - |x| - log1p(exp(-2|x|))` | pdf underflows for |x| > ~708: `logpdf(800)` = -inf, true -800.45158270528945 |
| logcdf | log(cdf) for x < 0, log1p(-sf) for x >= 0 | for x < 0: `log(2/pi) + log(arctan(exp(x)))` with `log(arctan(t)) = log(t) + log(arctan(t)/t)` (i.e. `x + log1p(...)`; for x < ~-20, `x - exp(2x)/3`) | cdf underflows for x < ~-708: `logcdf(-800)` = -inf, true -800.45158270528945; right half (log1p(-sf)) is fine |
| logsf | log(sf) for x > 0, log1p(-cdf) otherwise | mirror of logcdf: logsf(x) = logcdf(-x) | sf underflows for x > ~708: `logsf(800)` = -inf, true -800.45158270528945 |

## Shared code

none

## Existing GitHub issues

not checked

## Verification

pending
