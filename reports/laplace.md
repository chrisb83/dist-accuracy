# laplace: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `laplace_gen`, lines 6083-6152
- **Implemented:** pdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf) / log1p(-sf) split at the median, logsf = log(sf) / log1p(-cdf) split at the median
- **Related:** none; helpers: `_check_fit_input_parameters` (fit only, not accuracy-relevant); special functions: none (np.exp, np.log)
- **Reference:** `refs/laplace_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** none beyond conditioning

## Summary

All five implemented methods are correctly rounded or within 1 ulp on the
whole grid (x from +-1e-300 to +-1000 including both sides of the x = 0
branch and the exp underflow/overflow limits; q from 1e-308 to 1 - 2.2e-16
including both neighbours of q = 0.5). The cdf branches on x > 0 so that
`1 - 0.5*exp(-x)` is only used where the result is in [0.5, 1], `sf` uses
symmetry, and `ppf` uses `1 - q` only for q > 0.5 where it is exact
(Sterbenz). The only weaknesses are in the inherited log methods, which
return -inf in the far tails (|x| > ~708) where a closed form is trivial.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|

No findings: no point beyond conditioning in any implemented method.

## RuntimeWarnings

- `pdf`, `cdf`, `sf`: "underflow encountered in exp" (and "in multiply")
  when |x| is above ~708. For `pdf` and for `cdf` at x < -708 (`sf` at
  x > 708) the true result is itself subnormal or below the double range,
  so the warning is legitimate. For `cdf` at large positive x (and `sf` at
  large negative x) the warning comes from `exp(-x)` in
  `1.0 - 0.5*np.exp(-x)` (line 6115) and the result 1.0 is correct
  (`cdf(1000)` = 1.0, 0 ulp): spurious warning. The unused `np.where`
  branch `0.5*np.exp(x)` overflows there as well but that is silenced by
  `np.errstate(over='ignore')`.
- `ppf`, `isf`: none on (0, 1).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | x in +-[1e-300, 1000], incl. 0, ln 2, 708-745 | 0.5 ulp |
| cdf | both sides of x = 0 (down to 1e-300), x up to +-1000, exp underflow limit | 1 ulp |
| sf | same grid (symmetry, `_cdf(-x)`) | 1 ulp |
| ppf | q in [1e-308, 1 - 2.2e-16], both neighbours of q = 0.5, 0.5 +- 1e-16 ... 0.25 | 0 ulp |
| isf | same grid (`-_ppf(q)`) | 0 ulp |

Hypotheses checked (from `scratch/laplace/notes.md`): `1 - 0.5*exp(-x)`
(pattern 1) never cancels relative to the result because the branch is
only taken for x > 0; `-log(2*(1-q))` (pattern 8) is exact in `1 - q` for
q > 0.5; `log(2q)` near q = 0.5 acts on an exactly representable argument.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `-abs(x) - log(2)` | pdf underflows for \|x\| > ~708.4 (subnormal, precision lost) and is 0 for \|x\| > ~745: `logpdf(-800)` = -inf, true -800.6931471805599 (loss) |
| logcdf | log(cdf) for x < 0, log1p(-sf) above | `x - log(2)` for x <= 0, `log1p(-0.5*exp(-x))` for x > 0 | x < ~-708: `logcdf(-800)` = -inf, true -800.6931471805599 (loss); the right side is already fine (`logcdf(40)` 0 ulp) |
| logsf | log(sf) for x > 0, log1p(-cdf) below | `-x - log(2)` for x >= 0, `log1p(-0.5*exp(x))` for x < 0 | x > ~708: `logsf(800)` = -inf, true -800.6931471805599 (loss) |

## Shared code

none

## Existing GitHub issues

not checked

## Verification

pending
