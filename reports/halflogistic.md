# halflogistic: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `halflogistic_gen`, lines 4607-4730
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf, logsf (generic `rv_continuous` versions, `_distn_infrastructure.py` lines 2035-2049: these switch at the median, so logcdf = log(cdf) below the median and log1p(-sf) above it; logsf = log(sf) above the median and log1p(-cdf) below it)
- **Related:** none (genhalflogistic and logistic have their own code and do not call into this class); helpers: `_check_fit_input_parameters` (fit only); special functions: sc.expit, sc.log1p, sc.logit (sc.gamma, sc.zeta only in `_munp`)
- **Reference:** `refs/halflogistic_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** none beyond conditioning

## Summary

All six implemented methods are accurate over the full default grids
(x from 1e-300 to 1e300, q from 1e-300 to 1 - 3e-16) and at the extra points
near the underflow edges (x = 700 ... 745) and on both sides of the isf branch
at q = 0.5. The largest error is 271 ulp in pdf near x = 709. That is within
conditioning (cond (arg) ~ x), because pdf = exp(logpdf) scales the
rounding error of logpdf by |logpdf| ~ x. Everything else is at most 1 ulp.
The only accuracy problem is in the inherited logsf. Its `log(sf)` branch
returns -inf for x > ~745, where the true value is about log(2) - x.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| - | - | - | none beyond conditioning | - | - |

No implemented method has a point beyond conditioning (the tool's flagged-points list is empty).

## RuntimeWarnings

- **pdf, logpdf:** "underflow encountered in exp" for x > ~708. `np.exp(-x)` in
  `_logpdf` (line 4642) underflows. The term it feeds, `2*log1p(exp(-x))`, is
  then below 1 ulp of the result, so logpdf stays exact (`logpdf(800)` =
  -799.3068528194401, matching the reference). For pdf the final `np.exp`
  also underflows once the true pdf is subnormal. Where the pdf is still a
  normal double (`pdf(709.8)` = 1.0934696684707856e-308 vs reference
  1.093469668470846e-308) the error is within conditioning (cond 710). The
  warning is spurious.
- **cdf:** "underflow encountered in divide" for subnormal x (e.g. 1e-310).
  `x/2.0` (line 4645) is subnormal, and so is the true cdf, so this point is
  not measured. The result (5e-311) is correct to subnormal precision.
  The warning is spurious.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full x grid 1e-300 ... 1e300, x = 700 ... 745 near the underflow edge | 271 ulp (x ~ 709, cond 709, ok); bulk 1.44 ulp |
| logpdf | full x grid, large x where exp(-x) underflows | 0.72 ulp |
| cdf | full x grid, small x (tanh(x/2) ~ x/2), saturation x ~ 37-40 | 0.55 ulp |
| sf | full x grid up to the underflow at x ~ 745 | 1 ulp |
| ppf | q in [1e-300, 1 - 3e-16] | 0.96 ulp |
| isf | q in [1e-300, 1 - 3e-16], both sides of the q = 0.5 branch (0.5 - 2**-53, 0.5, 0.5 + 2**-52) | 0.98 ulp |

Hypotheses checked: no cancellation in `_logpdf` (all terms after log(2)
have the same sign). The isf branch at q = 0.5 (pattern 7) is fine: for
q >= 0.5, `1 - q` is exact (Sterbenz), and for q < 0.5, `-logit(q/2)` has no
cancellation. ppf near q = 1 is only conditioning-limited.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | log(sf) for x > median (log(3) = 1.0986), log1p(-cdf) below | `log(2) + sc.log_expit(-x)`, i.e. `log(2) - x - log1p(exp(-x))` | sf = 2*expit(-x) underflows for x > ~745, giving -inf; e.g. `logsf(800)` = -inf vs reference -799.3068528194401 (loss) |
| logcdf | log(cdf) for x < median, log1p(-sf) above | optional: `log(tanh(x/2))` below the median, `log1p(-2*expit(-x))` above (same thing written out, which avoids the median ppf call) | none found: `logcdf(40)` = -8.496708510583178e-18, 0 ulp |

## Shared code

none

## Existing GitHub issues

not checked

## Verification

pending
