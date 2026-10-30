# levy: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `levy_gen`, lines 6306-6404
- **Implemented:** pdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logpdf = log(pdf); logcdf and logsf use the rv_continuous median split (log(cdf) / log1p(-sf) for logcdf, log(sf) / log1p(-cdf) for logsf, `_distn_infrastructure.py` lines 2035-2049)
- **Related:** levy_l (mirror image, separate class); levy_stable (alpha = 1/2, beta = 1, no shared code); helpers: `_norm_isf` (= -`_norm_ppf`, i.e. `sc.ndtri`); special functions: sc.erf, sc.erfc, sc.erfinv, sc.ndtri (via `_norm_isf`)
- **Reference:** `refs/levy_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** none beyond conditioning

## Summary

All five implemented methods are accurate on the default grid and on the
targeted points (the subnormal window of `exp(-1/(2x))`, subnormal `0.5/x`
for x > 2.2e307, subnormal `q/2` in ppf, and `erfinv(q)**2` near underflow
in isf). The largest errors are in the low tail of pdf (253 ulp) and cdf
(335 ulp). Both are within conditioning (cond about 700 there), and come from
the unavoidable rounding of `1/(2x)` inside `exp`/`erfc`. The only problems are
in the inherited logpdf and logcdf, which return -inf for x < ~7e-4 where
closed forms are finite (see Opportunities).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|

No findings: the tool flagged no points beyond conditioning.

## RuntimeWarnings

- pdf: `underflow encountered in exp` (and divide/multiply under- and overflow)
  for small x (about x < 7.06e-4) and huge x (`2*np.pi*x` overflows for
  x > 2.86e307). The results are accurate (e.g. `pdf(7.0e-4)` =
  1.3271462650773607e-306, 157 ulp, cond 713, ok) or the true value is
  subnormal. The warnings are spurious.
- cdf, sf, ppf: `underflow encountered in divide`, from `0.5/x` for huge x
  or `q/2` for q near the smallest normal. The results are accurate (`sf(1.7e308)`
  0.609 ulp).
- isf: `underflow encountered in square` / `divide by zero` / `overflow` for
  tiny q, where the true isf overflows or is near the overflow limit.
  `isf(7e-155)` = 1.2992240252399614e+308, 1.38 ulp. The warnings are spurious.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid plus x in [6e-4, 8e-4] (exp(-1/(2x)) subnormal for 1/(2x) > 708) and x up to 1.7e308 | 253 ulp (low tail, cond 693; within conditioning) |
| cdf | full grid plus erfc underflow region x ~ 7e-4 | 335 ulp (low tail, cond 705; within conditioning); bulk 6.24 ulp |
| sf | full grid plus x in [1e305, 1.7e308] (0.5/x subnormal for x > 2.25e307) | 1.71 ulp |
| ppf | default q grid plus q in [2.3e-308, 1e-306] (q/2 subnormal), q up to 1 - 2.2e-16 | 3.01 ulp |
| isf | default q grid plus q in [6e-155, 1e-153] (erfinv(q)**2 near subnormal), q up to 1 - 2.2e-16 | 5.08 ulp |

Hypotheses from code reading tested on purpose:
- pdf, line 6386: `np.exp(-1/(2*x))` is subnormal while the pdf can still be
  normal (pattern 4). The prefactor `x**-1.5` is only ~5e4, so the
  window where the pdf is normal and the exponential subnormal is narrow
  (x ~ 7.0e-4 ... 7.06e-4). The subnormal there still has more than 40 bits, and
  the error stays below the conditioning (157-205 ulp, cond ~710).
- sf, line 6393 and ppf, line 6397: subnormal `0.5/x` and `q/2` keep almost
  all their bits near the normal limit. Measured at 1-3 ulp.
- isf, line 6401: `sc.erfinv(p)**2` is subnormal only where the isf already
  overflows. Measured at 1.38 ulp at the overflow edge.
- ppf as q -> 1 (`_norm_isf(q/2)` near 0, `1/val**2`): ppf is accurate
  (1.65 ulp max in the high tail).

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `-0.5/x - 1.5*log(x) - 0.5*log(2*pi)` | pdf underflows for x < ~6.7e-4 (exp(-1/(2x)) = 0) and for x > ~1e205 (x**-1.5 underflows; 2*pi*x overflows for x > 2.86e307); default returns -inf. `logpdf(5e-4)`: -inf vs -989.51758484389153 (closed form gives -989.5175848438915) |
| logcdf | log(cdf) below the median | `log(2) + sc.log_ndtr(-1/sqrt(x))` (cdf = 2*ndtr(-1/sqrt(x))) | erfc underflows for x < ~7e-4; default returns -inf. `logcdf(5e-4)`: -inf vs -1004.0267419589519 (log_ndtr form gives -1004.0267419589522) |
| logsf | log(sf) above the median, log1p(-cdf) below | no change needed; `log1p(-erfc(z))` is what the generic split already uses | none found: `logsf(0.01)` = -1.523970604832094e-23 vs -1.5239706048321068e-23 |

## Shared code

none

## Existing GitHub issues

not checked

## Verification

pending
