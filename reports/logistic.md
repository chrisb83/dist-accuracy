# logistic: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `logistic_gen`, lines 6514-6627
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf
- **Inherited (not measured):** none
- **Related:** none sharing code (the docstring notes logistic is genlogistic with c = 1, but the classes share no code); helpers: `_check_fit_input_parameters` (fit only); special functions: sc.expit, sc.log_expit, sc.logit, sc.log1p
- **Reference:** `refs/logistic_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** none beyond conditioning

## Summary

Every method is a thin wrapper around a `scipy.special` logistic function
(`expit`, `log_expit`, `logit`) or the symmetric log-space form
`-|x| - 2*log1p(exp(-|x|))`. This makes it accurate from 1e-300 to 1.7e308 in |x|, on both
signs, and for q from 1e-308 to 1 - 2.2e-16. No point is beyond conditioning.
The largest error is 5.42 ulp in pdf at |x| = 20, well within its conditioning
(cond about |x|). The only side effect is a spurious "underflow encountered in
exp" warning in pdf/logpdf for |x| > ~708 under `np.errstate(under='warn')`.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| - | - | - | none | - | - |

No loss, severe, moderate or minor points (harness: "Points beyond
conditioning: none").

## RuntimeWarnings

- **pdf, logpdf:** `underflow encountered in exp` from `np.exp(y)` with
  `y = -|x|` in `_logpdf`, line 6550, once |x| > ~708 (exp(y) is subnormal or 0).
  This shows up only when underflow warnings are enabled (the harness uses
  `np.errstate(all="warn")`; NumPy ignores underflow by default, and the public
  API with default settings emits nothing). The returned values are accurate:
  `logpdf(800)` = -800.0 (0 ulp), `pdf(744)` = 1e-323 (subnormal reference
  7.67e-324, not measured). The warning is spurious. Since
  `log1p(exp(y))` is just `exp(y)` there, it is harmless.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | x = 0, +-1e-300 ... +-1.7e308, incl. +-36.7 (exp(-\|x\|) < eps) and +-708 ... +-746 (underflow edge) | 5.42 ulp at x = +-20 (cond ~ 20); 1.77 ulp in bulk; `pdf(-700)` 0 ulp (cond 700) |
| logpdf | same grid, up to \|x\| = 1.7e308 | 0.72 ulp |
| cdf | same grid; low tail to underflow, high tail to 1.7e308 | 1 ulp |
| logcdf | same grid; x -> -1.7e308 (logcdf = x), x = 36.7 ... 40 (logcdf ~ -exp(-x)) | 0.96 ulp (`logcdf(40)` 0 ulp) |
| sf | mirror of cdf | 1 ulp |
| logsf | mirror of logcdf; x up to 1.7e308 | 0.96 ulp (`logsf(-40)` 0 ulp) |
| ppf | q = 0.5 +- 1e-16 ... 0.25, both sides of logit's internal 0.3/0.65 switch (0.29, 0.31, 0.64, 0.66), q = 1e-308 ... 1e-3, 1 - q = 2.2e-16 ... 1e-4 | 1.12 ulp at q = 0.29; `ppf(0.5000001)` 0.6 ulp |
| isf | same q grid | 1.12 ulp at q = 0.29; `isf(0.9999999999)` 0 ulp |

ppf/isf at q = 0.5 (reference exactly 0) is returned as 0.0 and not counted
in ulp.

## Suspected, not confirmed

none. Hypotheses tested and refuted (see `scratch/logistic/notes.md`):
- cancellation in `logit(q)` near q = 1/2 (pattern 2/8): `sc.logit` uses a
  log1p form there; 0.6 ulp at q = 0.5 + 1e-7.
- error amplification in `pdf = exp(logpdf)` for large |x|: logpdf is
  accurate to < 1 ulp, so pdf stays within its conditioning (5.4 ulp max).
- `log_expit` near its internal branch and in both tails: < 1 ulp.

## Opportunities: inherited methods

none (all methods are implemented)

## Shared code

none. `sc.expit`, `sc.log_expit` and `sc.logit`, which other distributions
also use, were accurate to about 1 ulp over the full double range here.

## Existing GitHub issues

not checked

## Verification

pending
