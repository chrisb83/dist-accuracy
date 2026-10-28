# arcsine: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `arcsine_gen`, lines 612-654
- **Implemented:** pdf, cdf, ppf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = median split log(cdf) / log1p(-sf), sf = 1 - cdf, logsf = median split log(sf) / log1p(-cdf), isf = ppf(1 - q)
- **Related:** beta(1/2, 1/2) is the same distribution, but no code is shared; helpers: none; special functions: none (numpy `sqrt`, `arcsin`, `sin`)
- **Reference:** `refs/arcsine_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** none beyond conditioning

## Summary

pdf, cdf and ppf are accurate on the whole grid: x from 5e-324 to a few ulp
below 1, and q from 1e-300 to 1 - 1e-16, with extra points at both support ends
and around 0.5. pdf and ppf stay within 1.12 ulp. cdf is within 4.59 ulp
except near x = 1. There it reaches 1.3e7 ulp at x = 1 - 1.1e-16, but that is
within conditioning (cond 3.0e7). The tool flagged no points. The only real
weakness is in the inherited `sf = 1 - cdf` and `logsf` near x = 1. These are
formally within conditioning with respect to x, but a closed form
`2/pi*arcsin(sqrt(1 - x))` would make them accurate to about 1 ulp for exact
inputs (see Opportunities).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|

No points beyond conditioning in pdf, cdf or ppf.

## RuntimeWarnings

ppf emits `underflow encountered in power` (from `**2.0`, line 644) for
q below about 1e-154. At those q the true result is subnormal (e.g.
`ppf(1e-155)` = 2.4674e-310, `ppf(1e-160)` = 2.4674e-320), and SciPy returns
the correctly rounded subnormal. The warning is expected, not spurious, and
there is no accuracy problem. pdf at x = 0 returns inf with `divide` warnings
suppressed (`np.errstate`, line 637), which is correct.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid; x = 5e-324 ... 1e-3 (incl. subnormal x); 1 - k ulp (k = 1 ... 1e8) and 1 - 1e-15 ... 1 - 1e-3; around 0.5 (where `1 - x` stops being exact) | 1.12 ulp |
| cdf | low tail and bulk, same points as pdf | 4.59 ulp (bulk), 0.99 ulp (low tail) |
| cdf | high tail x -> 1 | 1.3e7 ulp at x = 1 - 1.1e-16 (cond 3.0e7), 1.23e3 ulp at x = 1 - 1e-8 (cond 3.18e3). All within conditioning: `sqrt(x)` rounds, and `arcsin` near 1 magnifies that rounding by 1/sqrt(1 - x). For exact inputs, `1 - 2/pi*arcsin(sqrt(1 - x))` for x > 0.5 would give 0.11 ulp at x = 1 - 1.1e-16 (not a finding) |
| ppf | q from 1e-300 to 1 - 1e-16; down to the subnormal threshold (q ~ 9.5e-155); q = 0.5 ± 1e-16 ... 1e-2; q -> 1 (sin is flat near pi/2) | 1.09 ulp |

Reproducers for the high-tail cdf points:

| call | scipy | reference | ulp | cond | severity |
|---|---|---|---|---|---|
| `cdf(0.9999999999999999)` | 0.9999999905136262 | 0.9999999932921207 | 1.3e+07 | 3.0e+07 | ok |
| `cdf(0.99999999)` | 0.9999363380222235 | 0.9999363380224972 | 1.23e+03 | 3.18e+03 | ok |

## Suspected, not confirmed

none. I tested pattern 1 (`1 - x` in the pdf for x < 0.5), pattern 4
(subnormal `x*(1-x)` in the pdf, and the `sin(...)**2` underflow in the ppf)
and pattern 8 (ppf near q = 1) with targeted points. All were within
conditioning or exact.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | `2/pi*arcsin(sqrt(1 - x))` (1 - x is exact for x >= 0.5); or `2/pi*arccos(sqrt(x))` for x < 0.5 | x -> 1. `sf(0.9999999999999999)` = 9.486373797606973e-09 vs 6.707879276254073e-09 (1.9e15 ulp, cond 4.5e15: within conditioning). `sf(0.99999999)`: 1.9e7 ulp (cond 5.0e7). The candidate gives 0.40 and 0.18 ulp at these points |
| logsf | log(sf) above the median, log1p(-cdf) below | `log(2/pi*arcsin(sqrt(1 - x)))` above the median | x -> 1, same cause as sf. `logsf(0.99999999)` = -9.661923068787093 vs -9.66192307308659 (2.0e6 ulp, cond 5.2e6: within conditioning). The candidate gives 0.22 ulp |
| logpdf | log(pdf) | none needed (`-log(pi) - 0.5*log(x) - 0.5*log1p(-x)` is equivalent) | the pdf never under- or overflows inside (0, 1), so log(pdf) is accurate |
| logcdf | log(cdf) below the median, log1p(-sf) above | none needed | above the median cdf >= 0.5, so log1p(-(1 - cdf)) is accurate |
| isf | ppf(1 - q) | optional: `cos(pi/2*q)**2` for q < 0.5 | none in practice. For small q the result is about 1 - (pi q/2)^2, so rounding of 1 - q is invisible (`isf(1e-20)` = 1.0, exact). For q near 1, 1 - q is exact (`isf(0.99999999)`: 0 ulp) |

## Shared code

none

## Existing GitHub issues

none found (web search of scipy/scipy issues for "arcsine"; `gh` was not available)

## Verification

pending
