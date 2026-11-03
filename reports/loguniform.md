# loguniform: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `reciprocal_gen`, lines 9225-9328 (instances at 9331-9332)
- **Also covers:** reciprocal (same class, undocumented alias)
- **Implemented:** pdf, logpdf, cdf, ppf (also `_munp` and `_entropy`; these are outside the harness, see Shared code)
- **Inherited (not measured):** logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** none (reciprocal is the same class); helpers: `_log_diff` (shared with truncnorm, used only in `_munp`); special functions: none (`sc.logsumexp` via `_log_diff`)
- **Reference:** `refs/loguniform_ref.py`, 400 digits; self-checks: PASS, apart from one artifact. For (1e-300, 1e300) the harness reports "integral of pdf != cdf difference" on [1e-240, 1] and [1, 1e240]. This is `mp.quad` failing on 1/x over 240 decades in a single interval: it returns 0.0905 where the cdf difference is 0.4. The same quad subdivided per 10 decades gives 0.39999999999999999999, so the reference is correct. All other checks pass.
- **Parameter sets:** (0.01, 1.25), (1, 10), (1e-300, 1e300), (1e-300, 2e-300), (1e10, 1.000001e10), (1e300, 1.0000000001e300), (1e10, 10000000000.000002) [b = nextafter(a)], (5e-324, 1e308), (1, 1.000000000001)
- **Worst finding:** loss in pdf, logpdf (log(b) - log(a) cancels to 0) and cdf (log(x) - log(a) cancels to 0 near x = a)

## Summary

With log(a) = 0 or moderate shapes (a = 0.01, b = 1.25; a = 1, b = 10) all four implemented methods are accurate (at most 3 ulp). Every method builds the normalizer log(b/a) as `np.log(b) - np.log(a)`. That subtraction loses about |log a| / log(b/a) ulp. pdf and logpdf return inf when b is within about half an ulp of log(a) of a, with errors of 2e12 ulp (pdf) and 3e9 ulp (logpdf) before that. cdf is severe in the bulk for a close to b, and is already moderate (1.1e3 ulp) for (1e-300, 2e-300). cdf also forms `log(x) - log(a)`, so it returns 0 a few ulp above a. Before that it loses about |log a| times the conditioning: up to 7.7e10 ulp for a = 1e-300. Because pdf = exp(logpdf) and ppf = exp(log a + ...), both lose up to about |log x| ulp (at most ~750) at extreme x. ppf can also return values slightly outside [a, b].

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf, logpdf, cdf | log(b/a) small relative to \|log a\| (b/a - 1 << \|log a\| * 1e-13 for severe), any x | `np.log(b) - np.log(a)` cancels (pattern 5) | pdf/logpdf inf for (1e10, nextafter(1e10)); pdf 2.0e12 ulp, logpdf 3.0e9 ulp, cdf 9.3e11 ulp for (1e300, 1.0000000001e300) | loss |
| F2 | cdf | x within a relative ~eps*\|log a\| of a; errors beyond conditioning up to x/a - 1 ~ 1e-2, \|log a\| not small | `np.log(x) - np.log(a)` cancels (pattern 5) | 0.0 for 3.6e-17 (a = 0.01), 0.0 for 1.5e-6 (a = 1e300); 7.7e10 ulp at cond 7.2e8 | loss |
| F3 | pdf | \|log pdf\| large: x or log(b/a) far from 1 | pdf = exp(logpdf) amplifies the absolute rounding of logpdf (pattern 3/5, exp of a rounded log) | 522 ulp | minor |
| F4 | ppf | \|log x\| large (a or b far from 1) | exp(log(a) + q*L) amplifies the absolute rounding of the exponent; results can leave [a, b] | 412 ulp | minor |

### F1: log(b) - log(a) cancels when b is close to a

- **Code:** `_logpdf`, line 9299: `-np.log(x) - np.log(np.log(b) - np.log(a))`; `_pdf` (9296) is `exp(_logpdf)`; `_cdf`, line 9302: `(np.log(x)-np.log(a)) / (np.log(b) - np.log(a))`. The same expression is in `_ppf` (9305), `_munp` (9310) and `_entropy` (9315).
- **Mechanism:** `np.log(a)` and `np.log(b)` each carry an absolute rounding error of up to eps/2 * |log a|. Their difference L = log(b/a) is much smaller, so its relative error is about eps * |log a| / L (pattern 5). If log(b) and log(a) round to the same double, L = 0: logpdf = -log(0) = +inf, pdf = inf (with "divide by zero encountered in log"). In cdf the denominator carries the same relative error. At (1e-300, 2e-300) that is about 690/0.69 = 1e3 ulp in the bulk, where the condition number is 1-5. For x close to b, the error can also push cdf to exactly 1.0 while the true value is 1 - 5.7e-10. ppf is not affected in relative terms, because q*L is small next to log(a) (see F4).
- **Regime:** |log a| / log(b/a) > ~1e3 gives moderate, > ~1e6 severe. Loss when b/a - 1 < ~ulp(log a)/2, e.g. b = nextafter(a) for any a with |log a| > ~1. The shapes are exact user inputs, so the large cond (shapes) values below (1e16 for b = nextafter(a)) are context only. The logpdf of the doubles a, b is well defined (13.17 for (1e10, nextafter(1e10))).
- **Reproducer:** `python accuracy.py refs/loguniform_ref.py --at logpdf 1e10 1e10 10000000000.000002`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e10, 1e10, 10000000000.000002)` | inf | 13.16979643063896 | inf | 0.0759 | loss |
  | `pdf(1e10, 1e10, 10000000000.000002)` | inf | 524288.0 | inf | 1 | loss |
  | `logpdf(1.00000000005e300, 1e300, 1.0000000001e300)` | -667.7501210416583 | -667.7496765080046 | 3.0e+09 | 0.0015 | severe |
  | `pdf(1.00000000005e300, 1e300, 1.0000000001e300)` | 9.995560252009368e-291 | 1.000000460268713e-290 | 2.0e+12 | 1 | severe |
  | `cdf(1.000000000099907e300, 1e300, 1.0000000001e300)` | 0.9988636363636364 | 0.9990706140044807 | 9.3e+11 | 1.0e+10 | severe |
  | `cdf(10000000500.0, 1e10, 1.000001e10)` | 0.050000022026835816 | 0.05000002374999525 | 1.6e+08 | 2.0e+07 | severe |
  | `cdf(10000009999.999994, 1e10, 1.000001e10)` | 1.0 | 0.9999999994277957 | 2.6e+06 | 1e+06 | severe |
  | `pdf(10000005500.0, 1e10, 1.000001e10)` | 9.999999499049488e-05 | 9.9999994999994417e-05 | 4.3e+05 | 1 | moderate |
  | `logpdf(10000005500.0, 1e10, 1.000001e10)` | -9.210340422071235 | -9.21034042197624 | 4.65e+04 | 0.109 | moderate |
  | `cdf(1.25e-300, 1e-300, 2e-300)` | 0.321928094887284 | 0.3219280948873623 | 1.09e+03 | 4.48 | moderate |

  Counts from the run, all parameter sets with b close to a: pdf 2 loss, 30 severe, 34 moderate; logpdf 2 loss, 30 severe, 34 moderate. cdf bulk/high tail: 27 severe for (1e10, 1.000001e10) and (1e300, 1.0000000001e300); 23 minor and 1 moderate in the bulk for (1e-300, 2e-300), up to 1.09e3 ulp.
  The control set (1, 1.000000000001), where log a = 0, is accurate: logpdf 0.58 ulp, cdf 0.83 ulp.
- **Fix direction:** compute L = log(b/a) without cancelling, e.g. `np.log1p((b - a)/a)` (b - a is exact for b <= 2a) or `np.log(b/a)` when b/a is far from 1. Use `log1p((x - a)/a)` for the cdf numerator (F2).

### F2: cdf cancels in log(x) - log(a) near the lower endpoint

- **Code:** `_cdf`, line 9302: `(np.log(x)-np.log(a)) / (np.log(b) - np.log(a))`.
- **Mechanism:** for x = a(1 + d) with small d, log(x) - log(a) = log1p(d) ~ d, but both logs carry absolute error ~eps*|log a|/2. The numerator therefore has relative error ~eps*|log a|/d. The cdf condition number with respect to x is 1/log(x/a) ~ 1/d, so the error is about |log a| times the conditioning. Once d is below about half an ulp of log(a) (~eps*|log a|), the two logs round to the same double and cdf = 0.0 exactly, although the true cdf (e.g. 1.5e-6 for a = 1e300 with b/a - 1 = 1e-10) is a normal double. Pattern 5.
- **Regime:** x just above a, with |log a| not small (a = 1 is exact). cdf returns 0 for x a few ulp above a. Beyond conditioning by a factor of ~|log a| (up to ~745) for x/a - 1 up to ~1e-2. The tool rates the exact zeros as loss, even though cond (arg) there is ~6e15. The 1.5e-6 case shows the zero is not only a far-tail effect: there F1 makes the true cdf large.
- **Reproducer:** `python accuracy.py refs/loguniform_ref.py --at cdf 0.010000000000000002 0.01 1.25`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(0.010000000000000002, 0.01, 1.25)` | 0.0 | 3.592814324750239e-17 | inf | 5.8e+15 | loss |
  | `cdf(1.0000000000000002e300, 1e300, 1.0000000001e300)` | 0.0 | 1.4870175929794924e-06 | inf | 6.7e+15 | loss |
  | `cdf(1.0000000013815461e-300, 1e-300, 1e300)` | 9.999792958767399e-13 | 9.999963993891832e-13 | 7.7e+10 | 7.2e+08 | severe |
  | `cdf(1.0000000001e-300, 1e-300, 2e-300)` | 1.4416956911890104e-10 | 1.4426957092861923e-10 | 3.1e+12 | 1.0e+10 | severe |
  | `cdf(0.010000000482831391, 0.01, 1.25)` | 1.0000000023848244e-08 | 1.0000000120157093e-08 | 4.3e+07 | 2.1e+07 | severe |
  | `cdf(1.013911385736686e-300, 1e-300, 1e300)` | 9.999999999987616e-06 | 1.0000000000004722e-05 | 7.7e+03 | 72.4 | moderate |

  Counts in the low tail over all sets: 16 loss, 12 severe, 3 moderate (part of the severe ones for close a, b also include F1).
- **Fix direction:** numerator `np.log1p((x - a)/a)` (x - a is exact for x <= 2a), or `np.log(x/a)` away from a.

### F3: pdf = exp(logpdf) loses up to |log pdf| ulp

- **Code:** `_pdf`, line 9296: `np.exp(self._logpdf(x, a, b))`.
- **Mechanism:** logpdf is accurate in absolute terms (eps*|logpdf|). exp turns that absolute error into a relative error of the same size, i.e. about |logpdf| ulp. With x near 1e+-300, |logpdf| is ~690, but the pdf itself (1/(x L)) has cond 1. In the control set (1, 1.000000000001), logpdf is 27.6, which gives 14.8 ulp.
- **Regime:** |log x + log L| large; up to ~750 ulp at the ends of the double range.
- **Reproducer:** `python accuracy.py refs/loguniform_ref.py --at pdf 1.9999999999e-300 1e-300 2e-300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1.9999999999e-300, 1e-300, 2e-300)` | 7.213475204806326e+299 | 7.2134752048054908e+299 | 522 | 1 | minor |
  | `pdf(1e280, 1e-300, 1e300)` | 7.238241365053638e-284 | 7.2382413650541969e-284 | 348 | 1 | minor |
  | `pdf(1.00000000000035, 1, 1.000000000001)` | 999911107320.4233 | 999911107320.4201 | 14.8 | 1 | minor |

  Counts: 58 minor for (1e-300, 1e300), 38 for (1e-300, 2e-300), 28 for (5e-324, 1e308), 16 for (1, 1.000000000001).
- **Fix direction:** `1 / (x * L)` directly (with L from the F1 fix). Overflow can only happen where the true pdf overflows.

### F4: ppf loses up to |log x| ulp and can leave the support

- **Code:** `_ppf`, line 9305: `np.exp(np.log(a) + q*(np.log(b) - np.log(a)))`.
- **Mechanism:** same as F3. The exponent log(x) has absolute rounding error ~eps*|log x| (log(a) itself is rounded), and exp makes that a relative error of ~|log x| ulp. The condition number with respect to q is only q*L. Since exp(log(a)) != a for about half of all doubles a, ppf(q -> 0) can fall below a and ppf(q -> 1) above b, e.g. `ppf(1e-300, 1e300, 1.0000000001e300)` = 9.999999999999763e+299 < a. The `q*(log b - log a)` term also carries F1's relative error, but that is small next to log(a) here.
- **Regime:** |log a| or |log b| large; up to ~750 ulp.
- **Reproducer:** `python accuracy.py refs/loguniform_ref.py --at ppf 0.07999999999999999 5e-324 1e308`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.07999999999999999, 5e-324, 1e308)` | 1.578635570655002e-273 | 1.5786355706548577e-273 | 412 | 116 | minor |
  | `ppf(0.9, 1e-300, 2e-300)` | 1.8660659830734817e-300 | 1.866065983073615e-300 | 322 | 0.624 | minor |
  | `ppf(0.01, 1e-300, 1e300)` | 1.00000000000008e-294 | 1.0000000000000004e-294 | 359 | 13.8 | minor |
  | `ppf(0.07999999999999999, 1e300, 1.0000000001e300)` | 1.0000000000079343e+300 | 1.000000000008e+300 | 296 | 8e-12 | minor |

  Counts: 144 minor points in total (57 for (1e-300, 2e-300), 55 for (1e300, 1.0000000001e300)); none moderate.
- **Fix direction:** `a * np.exp(q*L)` for q <= 0.5 and `b * np.exp(-(1 - q)*L)` otherwise (L from the F1 fix), then clip to [a, b].

## RuntimeWarnings

- pdf, logpdf: "divide by zero encountered in log" when log(b) == log(a) (F1). These coincide with the inf results (loss).
- pdf, (5e-324, 1e308): "overflow encountered in exp" and "underflow encountered in exp". They come from x near 5e-324 (true pdf > 1.8e308, not measured) and from the top of the support (true pdf subnormal). The harness evaluates each parameter set as one array call, so the warnings also attach to points that are accurate; they are spurious there.
- ppf: "underflow encountered in multiply" for (1e300, 1.0000000001e300) and (1, 1.000000000001), from q*L with q ~ 1e-300 and small L. The underflow is harmless; the results are accurate or minor (F4). "underflow encountered in exp" for (5e-324, 1e308) comes from subnormal results at tiny q.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | (0.01, 1.25), (1, 10): full grid including endpoints | 2.3 ulp |
| logpdf | all sets with b/a not close to 1, including (1e-300, 1e300), (5e-324, 1e308) | 2.8 ulp |
| cdf | (1, 10) and (1, 1.000000000001) (log a = 0 exact): full grid | 0.91 ulp |
| cdf | high tail for b/a not close to 1 | ~200 ulp for (1e-300, 2e-300) (F1), otherwise ok |
| ppf | (0.01, 1.25), (1, 10), (1, 1.000000000001), (1e10, nextafter(1e10)): q in [1e-300, 1 - 3e-16] | 2.7 ulp |
| all | extreme a = 5e-324 (subnormal), b = 1e308 | cdf 357 ulp minor at x = 1e-323 (F2 mechanism, cond 1.44); pdf/ppf F3/F4 |

## Suspected, not confirmed

- `_munp` (mean, var, moments) and `_entropy` are not measured by `accuracy.py`. The checks below are by hand against mpmath (formulas (b - a)/log(b/a) and (log a + log b)/2 + log(log(b/a))):
  - `loguniform.mean(1e10, 10000000000.000002)` = inf (true 10000000000.000001); `entropy` = -inf (true -13.169796430638961). Cause: L = 0 (F1) in `t1 = 1/(log b - log a)` (line 9310) and in line 9315.
  - `loguniform.mean(0.5, 0.5000000000000001)` = 0.24999999999999897 (true 0.50000000000000006). Here L is accurate (log a = -0.69). The error comes from `_log_diff` (see Shared code).
  - `loguniform.mean(1e300, 1.0000000001e300)` = 1.0000000001000632e+300, which is above b (true 1.00000000005e300; relative error 5e-11).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | `log1p((b - x)/x) / L` (or `(log b - log x)/L` away from b) | x near b: `sf(1.2499999999999998, 0.01, 1.25)` = 0.0, true 3.679e-17 (loss); total once 1 - x/b < ~eps*L |
| logsf | log(1 - cdf) | `log(log1p((b - x)/x)) - log(L)` | same regime: `logsf(1.2499999999999998, 0.01, 1.25)` = -inf, true -37.84 |
| logcdf | log(cdf) | `log(log1p((x - a)/a)) - log(L)` | inherits F2 near a (log(0) = -inf a few ulp above a); fine elsewhere (`logcdf(0.5, 0.01, 1.25)` 1.2 ulp) |
| isf | ppf(1 - q) | `b * exp(-q*L)` | 1 - q rounds (relative error up to L*eps/2 in x) plus F4: `isf(1e-10, 1e-300, 1e300)` 537 ulp at cond 1.4e-7 |

The class comment (lines 9326-9328) says sf was deliberately left undefined (scipy PR 18614). The reason was not checked.

## Shared code

- `_log_diff` (line 10215-10216, `sc.logsumexp([log_p, log_q + np.pi*1j])`, shared with truncnorm) is called only in `reciprocal_gen._munp` (line 9311) as `np.real(np.exp(_log_diff(n*np.log(b), n*np.log(a))))`. None of the measured methods (pdf, logpdf, cdf, ppf) and none of the inherited ones reach it. Only `moment`, `mean`, `var` and `stats` do.
  - The p == q case is truncnorm F1 / INDEX item 20: `_log_diff(5.0, 5.0)` = -31.64 + 1.57j instead of -inf, because exp(1j*pi) = -1 + 1.22e-16j. This is reachable whenever n*log(b) == n*log(a) (b = nextafter(a)). In loguniform it is masked by t1 = 1/0 = inf, so the mean is inf.
  - The case where p - q is small but nonzero also fails. `_log_diff(p, q)` effectively computes `log(1 - exp(q - p))` without expm1, plus the 1.22e-16j residue (pattern 2). Its relative error in exp space is ~eps/(p - q):

    | call | scipy (real part) | reference | rel. error |
    |---|---|---|---|
    | `_log_diff(1e-10, 0)` | -23.025850847099342 | -23.025850929890457 | 1.6e7 ulp |
    | `_log_diff(1e-13, 0)` | -29.9344050552732 | -29.933606208922544 | 1.2e11 ulp |
    | `_log_diff(log(nextafter(0.5,1)), log(0.5))` | -37.03 | -36.74 | Re(exp) off by a factor 2 |

    The last row is what makes `mean(0.5, 0.5000000000000001)` = 0.25. For truncnorm this matters where the two log-cdfs are close (narrow intervals); see reports/truncnorm.md F1. These numbers come from a direct script at 50 digits, not from `accuracy.py`.
- No `scipy.special` problems found.

## Existing GitHub issues

- PR #18614 (state not confirmed): don't override sf for loguniform/reciprocal distribution - about the inherited sf (see Opportunities), which uses the same log(b) - log(a) normalization as F1
- #17832 (open): Tracking issue: custom methods for continuous distributions - relates to the inherited sf/logsf/isf opportunities
- F1-F4: none found

(web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `logpdf(1e10, 1e10, 10000000000.000002)` | inf | 13.169796430638961 | verified | loss; L from `log1p((b-a)/a)` |
| F1 | `pdf(1e10, 1e10, 10000000000.000002)` | inf | 524288.00000000005 | verified | loss |
| F1 | `logpdf(1.00000000005e300, 1e300, 1.0000000001e300)` | -667.7501210416583 | -667.74967650800464 | verified | 3.0e9 ulp |
| F1 | `pdf(1.00000000005e300, 1e300, 1.0000000001e300)` | 9.995560252009368e-291 | 1.000000460268713e-290 | verified | 2.0e12 ulp |
| F1 | `cdf(1.000000000099907e300, 1e300, 1.0000000001e300)` | 0.9988636363636364 | 0.99907061400448071 | verified | 9.3e11 ulp; cdf as ratio of quad integrals of 1/t |
| F1 | `cdf(10000000500.0, 1e10, 1.000001e10)` | 0.050000022026835816 | 0.05000002374999525 | verified | 1.55e8 ulp |
| F1 | `cdf(10000009999.999994, 1e10, 1.000001e10)` | 1.0 | 0.9999999994277957 | verified | 2.6e6 ulp |
| F1 | `pdf(10000005500.0, 1e10, 1.000001e10)` | 9.999999499049488e-05 | 9.9999994999994417e-5 | verified | 4.3e5 ulp |
| F1 | `logpdf(10000005500.0, 1e10, 1.000001e10)` | -9.210340422071235 | -9.2103404219762398 | verified | 4.65e4 ulp |
| F1 | `cdf(1.25e-300, 1e-300, 2e-300)` | 0.321928094887284 | 0.3219280948873623 | verified | 1.1e3 ulp |
| F2 | `cdf(0.010000000000000002, 0.01, 1.25)` | 0.0 | 3.5928143247502389e-17 | verified | loss (exact 0) |
| F2 | `cdf(1.0000000000000002e300, 1e300, 1.0000000001e300)` | 0.0 | 1.4870175929794924e-6 | verified | loss (exact 0) |
| F2 | `cdf(1.0000000013815461e-300, 1e-300, 1e300)` | 9.999792958767399e-13 | 9.9999639938918314e-13 | verified | 7.7e10 ulp |
| F2 | `cdf(1.0000000001e-300, 1e-300, 2e-300)` | 1.4416956911890104e-10 | 1.4426957092861922e-10 | verified | 3.1e12 ulp |
| F2 | `cdf(0.010000000482831391, 0.01, 1.25)` | 1.0000000023848244e-08 | 1.0000000120157094e-8 | verified | 4.3e7 ulp |
| F2 | `cdf(1.013911385736686e-300, 1e-300, 1e300)` | 9.999999999987616e-06 | 1.0000000000004722e-5 | verified | 7.7e3 ulp |

References: pdf/logpdf = 1/(x log1p((b-a)/a)); cdf = mp.quad of 1/t on geometric subdivisions, ratio of [a, x] to [a, b]. Each was computed at 100 and 200 digits, and the two agree to at least 30 digits. Script: `scratch/loguniform/verify/verify.py`. The mechanism lines (9296, 9299, 9302) match the SciPy 1.18.1 source. There are no moderate findings; F3 and F4 are minor and were not checked.

Verified by dist-accuracy-verifier: 2 of 2 verified.
