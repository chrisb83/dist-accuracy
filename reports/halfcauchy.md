# halfcauchy: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `halfcauchy_gen`, lines 4514-4601
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** foldcauchy (same distribution at c = 0, separate code), cauchy (separate code); helpers: `_check_fit_input_parameters` (fit only); special functions: `sc.log1p` (plus `np.arctan`, `np.arctan2`, `np.tan`)
- **Reference:** `refs/halfcauchy_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in logpdf

## Summary

pdf, cdf and sf are accurate to below 1 ulp on the full grid
x in [1e-300, 1.7e308]; sf uses `arctan2(1, x)`, so there is no `1 - cdf`
cancellation in the right tail. ppf and isf are within conditioning
everywhere (up to 2.8e15 ulp next to q = 1, where cond is 9.0e15).
The one problem is logpdf: `log(2/pi) - log1p(x*x)` overflows in `x*x` for
x > ~1.34e154 and returns -inf, while the true value (about
`log(2/pi) - 2 log x`, between -710 and -1420) is an ordinary double. All 19
measured logpdf points with x >= 1.4e154 are losses.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf | x > ~1.34e154 | `x*x` overflows to inf, `log1p(inf)` = inf (pattern 3) | -inf for -710.32 (x = 1.4e154) ... -1419.9 (x = 1.7e308) | loss |

### F1: logpdf returns -inf for x > ~1.34e154

- **Code:** `_logpdf`, line 4542: `np.log(2.0/np.pi) - sc.log1p(x*x)`
- **Mechanism:** `x*x` exceeds 1.8e308 for x > ~1.34e154 and becomes inf,
  so `log1p(x*x)` is inf and the result is -inf (pattern 3). The true logpdf
  is `log(2/pi) - 2 log x - log1p(x**-2)`, finite for every finite x. The
  pdf itself becomes subnormal near x ~ 5.4e153, so only the log form can
  represent these values and it is exactly the method meant for this regime.
- **Regime:** all x > ~1.34e154 (checked: 1.3e154 gives -710.1725198763904,
  0.721 ulp; 1.35e154 already -inf). 19 of 19 grid points above the boundary
  are losses.
- **Reproducer:** `python accuracy.py refs/halfcauchy_ref.py --at logpdf 1.4e154`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.4e154)` | -inf | -710.3207358206979 | inf | 0.00282 | loss |
  | `logpdf(1e155)` | -inf | -714.2529615334436 | inf | 0.0028 | loss |
  | `logpdf(1e200)` | -inf | -921.4856199029077 | inf | 0.00217 | loss |
  | `logpdf(1e250)` | -inf | -1151.7441292023123 | inf | 0.00174 | loss |
  | `logpdf(1.7e308)` | -inf | -1419.9052564917459 | inf | 0.00141 | loss |

- **Fix direction:** for x > 1 use `log(2/pi) - 2*log(x) - log1p((1/x)**2)`
  (as `cauchy._logpdf` already does with its branch at |x| = 1). A quick
  check of this form gives -710.3207358206979, -921.4856199029077 and
  -1419.9052564917458 at x = 1.4e154, 1e200, 1.7e308.

## RuntimeWarnings

- logpdf: "overflow encountered in multiply" from `x*x` for x > ~1.34e154;
  goes with F1 (the result is wrong). "underflow encountered in multiply"
  for tiny x (x*x underflows); the result is accurate (max 0.554 ulp in the
  low tail), spurious.
- pdf: "overflow encountered in multiply" / "underflow encountered in
  divide" for x > ~1.34e154, where the true pdf is subnormal (below
  2.2e-308 from x ~ 5.4e153), so the 0.0 result is not measured and not a
  finding; "underflow encountered in multiply" for tiny x, spurious.
- cdf: "underflow encountered in arctan" for tiny x (e.g. 1e-300); result
  accurate (max 0.978 ulp), spurious.
- sf: "underflow encountered in multiply" where `2/pi*arctan2(1, x)` is
  subnormal (x >~ 1e307); true value subnormal there, not measured.
- ppf/isf: "underflow encountered in multiply / tan" for tiny q; results
  accurate (max 0.951 / 0.962 ulp), spurious.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | x = 0, 1e-300 ... 1.7e308, both sides of x*x underflow and overflow (1e-154, 1.3e154, 1.4e154), x near 1 | 0.908 ulp |
| logpdf | x = 0 ... 1.3e154 (low tail, bulk, high tail below the overflow) | 0.874 ulp |
| cdf | full grid incl. tiny x (1e-300) and x -> inf | 0.978 ulp |
| sf | full grid; `arctan2(1, x)` keeps full relative accuracy in the right tail (e.g. x = 1e300) | 0.985 ulp |
| ppf | q from 1e-308 to 1 - 1.1e-16 | 0.951 ulp in the low tail; bulk up to 48.2 ulp and next to q = 1 up to 1.7e15 ulp, all within conditioning (cond ~ 1/(1 - q), 9.0e15 at the last double below 1) - ok |
| isf | same q grid | 0.962 ulp in the low tail (e.g. `isf(1e-300)` 0.526 ulp); near q = 1 up to 2.8e15 ulp with cond 9.0e15 - ok |

Note on ppf near q = 1: `np.pi/2` is below pi/2 by ~6e-17, comparable to
the distance of the angle `pi*q/2` from the pole for q = 1 - 1.1e-16, so the
computed tan can be off by tens of percent there; this stays below the
condition number with respect to q (the tool rates all such points ok).
`isf` uses `1/tan(pi*p/2)` directly and therefore keeps full accuracy for
small p, unlike an `isf = ppf(1 - p)` default (compare foldcauchy).

## Suspected, not confirmed

- pdf overflow of `x*x` (pattern 3): happens only for x > ~1.34e154, where
  the true pdf is already subnormal; no measurable error. Not a finding.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | `log(arctan(x)) + log(2/pi)`; not needed in practice | none found: cdf is normal for x >= ~3.5e-308 and log(cdf) stays accurate near cdf = 1; `logcdf(1e-300)` = -691.2271106035032, true -691.22711060350316 |
| logsf | log(sf) | `log(arctan2(1, x)) + log(2/pi)`; not needed in practice | none found: sf ~ 2/(pi x) is normal up to x ~ 2.9e307; `logsf(1e300)` = -691.2271106035032, true -691.22711060350316 |

## Shared code

none (the logpdf overflow is in `halfcauchy_gen._logpdf` itself; `sc.log1p`
is used correctly and is accurate)

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `logpdf(1.4e154)` | -inf | -710.3207358206979 | verified | reference via u = 1/x: log(2/pi) + 2 log u - log1p(u^2), and independently log(-d/dx sf) by mp.diff; 100 vs 200 digits agree to 1e-101 |
| F1 | `logpdf(1.35e154)` | -inf | -710.2480005323562 | verified | confirms boundary; `logpdf(1.3e154)` = -710.1725198763904 is 0.728 ulp |
| F1 | `logpdf(1e200)` | -inf | -921.4856199029077 | verified | |
| F1 | `logpdf(1e250)` | -inf | -1151.7441292023123 | verified | |
| F1 | `logpdf(1.7e308)` | -inf | -1419.9052564917458 | verified | report's table shows ...459, the verifier's rounded reference is ...458 (1 ulp apart; irrelevant to the loss); mechanism matches line 4542 `np.log(2.0/np.pi) - sc.log1p(x*x)` |

Verified by dist-accuracy-verifier: 1 of 1 verified.
