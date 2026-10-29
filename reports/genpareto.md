# genpareto: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `genpareto_gen`, lines 3168-3282
- **Implemented:** pdf, logpdf, cdf, sf, logsf, ppf, isf
- **Inherited (not measured):** logcdf (generic `rv_continuous._logcdf`: `log(cdf)` below the median, `log1p(-sf)` above)
- **Related:** none; helpers: none; special functions: sc.inv_boxcox1p (cdf), sc.inv_boxcox (sf), sc.boxcox1p (ppf), sc.boxcox (isf), sc.xlog1py (logpdf), sc.log1p (logsf)
- **Reference:** `refs/genpareto_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (0.1), (0), (1), (-0.5), (-2), (1e-08), (-1e-08), (1e-300), (-1e-160), (1000), (1e+10); extra check run with (1e-20), (-1e-20)
- **Worst finding:** loss in cdf, logpdf, logsf

## Summary

ppf and the exact c = 0 branch are accurate everywhere measured. sf, pdf and isf
are accurate except for up to ~500 ulp (minor) where the result is exp of a log
of size several hundred. cdf breaks down for |c| below about 1e-139:
`scipy.special.inv_boxcox1p` has a shortcut that returns its argument when
|lmbda*x| < 1e-154. As a result `cdf(x, c) = x`, which gives values above 1
(e.g. `cdf(2, 1e-300) = 2.0`). This is a shared `scipy.special` problem.
logpdf and logsf compute `c*x` directly. That product underflows for tiny |c|
(giving `-0.0` for a normal result) and overflows for large c with x near
1e300 (giving `-inf`). They also return `-inf` at the upper support endpoint
-1/c when that endpoint is not exactly representable.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | \|c*x\| < 1e-154 and x > ~4e-16 (i.e. \|c\| < ~2.5e-139) | `inv_boxcox1p` shortcut returns x (patterns 9, 10) | 2.0 for 0.865; 5e145 for 1.0 | loss |
| F2 | logpdf, logsf | \|c*x\| < 2.2e-308 (tiny \|c\|, small x) | `c*x` subnormal / zero (pattern 4) | -0.0 for -1e-30 | loss |
| F3 | logpdf, logsf | c*x > 1.8e308 (c > 1, x near 1e300) | `c*x` overflows (pattern 3) | -inf for -7.1e-8 | loss |
| F4 | logpdf, logsf | c < 0, x = -1/c rounded (upper endpoint) | `c*x` rounds to -1, `log1p(-1) = -inf` (pattern 5) | -inf for -3.99e161 | loss |
| F5 | pdf, sf, isf | result far in the tail, \|log result\| ~ 100-700 | exp/expm1 of a rounded large argument | 519 ulp | minor |

### F1: cdf returns x for tiny |c| (scipy.special.inv_boxcox1p shortcut)

- **Code:** `_cdf`, line 3227: `-sc.inv_boxcox1p(-x, -c)`
- **Mechanism:** `sc.inv_boxcox1p(y, lmbda)` returns `y` unchanged when
  `|lmbda*y| < 1e-154`, i.e. it uses the first-order approximation
  `(1 + lmbda*y)**(1/lmbda) - 1 ~ y`. That is valid only when `y` itself is
  small, not when `lmbda*y` is small. The correct limit for lmbda -> 0 is
  `expm1(y)`. So for tiny |c| the cdf becomes `x` instead of `1 - exp(-x)`
  (patterns 9 and 10). The relative error is about x/2 for small x. For
  x > 1 the result is above 1 (loss). The fault is in `scipy.special`; the
  distribution calls it correctly. `sc.inv_boxcox` (used by sf) has no such
  shortcut and is accurate in the same regime.
- **Regime:** |c*x| < 1e-154 and x large enough that x/2 > eps, i.e. any
  |c| < ~2.5e-139; for |c| = 1e-152 only x < 0.01 is affected. Both signs of
  c. Measured: 37 loss, 9 severe, 2 moderate points for c = 1e-300 and
  c = -1e-160; c = +-1e-20 and c = +-1e-8 are fine (<= 0.8 ulp).
- **Reproducer:** `python accuracy.py refs/genpareto_ref.py --at cdf 2 1e-300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(2.0, 1e-300)` | 2.0 | 0.8646647167633873 | 5.9e+15 | 0.313 | loss |
  | `cdf(5e+145, 1e-300)` | 5e+145 | 1.0 | 2.3e+161 | 0 | loss |
  | `cdf(1.0, 1e-160)` | 1.0 | 0.6321205588285577 | 2.6e+15 | 0.582 | severe |
  | `cdf(0.001, 1e-152)` | 0.001 | 0.0009995001666250082 | 2.3e+12 | 1 | severe |
  | `cdf(1e-10, 1e-300)` | 1e-10 | 9.999999999500001e-11 | 2.3e+05 | 1 | moderate |

  The inherited logcdf uses `log(cdf)` below the median and inherits this:
  `logcdf(0.5, 1e-200)` returns -0.6931471805599453 (= log 0.5); the true value
  is log(0.3935) = -0.9327.
- **Fix direction:** in `inv_boxcox1p`, replace the shortcut by
  `expm1(x)` (or a series in lmbda) when |lmbda| is tiny. In genpareto,
  `-expm1(-log1p(c*x)/c)` (with the same small-c care as F2).

### F2: logpdf and logsf lose c*x to underflow for tiny |c|

- **Code:** `_logpdf`, line 3223: `-sc.xlog1py(c + 1., c*x) / c`;
  `_logsf`, line 3234: `-sc.log1p(c*x) / c`
- **Mechanism:** for tiny |c| the product `c*x` is subnormal (fewer than 53
  significant bits) when |c*x| < 2.2e-308, and rounds to 0 when
  |c*x| < 2.5e-324. Dividing by c afterwards cannot restore the lost bits, so
  the result is `-0.0` where the true value is about -x (pattern 4, in the
  c -> 0 limit of pattern 9).
- **Regime:** |c*x| < 2.2e-308: severe with subnormal `c*x`, loss once it is 0.
  For c = 1e-300 that is every x < 2.2e-8 (loss below x = 2.5e-24). For
  c = +-1e-20 it is x < ~2.5e-288. The pdf (`exp(logpdf)` ~ 1) and sf are
  not affected.
- **Reproducer:** `python accuracy.py refs/genpareto_ref.py --at logpdf 1e-30 1e-300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-30, 1e-300)` | -0.0 | -1e-30 | inf | 1 | loss |
  | `logpdf(2.4703282292062327e-164, -1e-160)` | -0.0 | -2.4703282292062327e-164 | inf | 1 | loss |
  | `logsf(1e-20, 1e-300)` | -9.99988867182683e-21 | -1e-20 | 5.0e+10 | 1 | severe |
  | `logpdf(4.89124989382834e-24, 1e-300)` | -4.940656458412465e-24 | -4.89124989382834e-24 | 4.5e+13 | 1 | severe |

- **Fix direction:** when |c*x| is tiny, use the series
  `log1p(c*x)/c = x*(1 - c*x/2 + ...)` (logsf = -x, logpdf = -(1 + c)*x to
  first order), or rescale: `log1p(c*x)/c` evaluated as `x * log1p(t)/t`
  with `t = c*x` and `log1p(t)/t -> 1`.

### F3: logpdf and logsf overflow in c*x for large c

- **Code:** `_logpdf`, line 3223 and `_logsf`, line 3234 (same expressions as F2).
- **Mechanism:** for c*x > 1.8e308 the product overflows to inf, so
  `log1p(inf) = inf` and the result is `-inf`. The true value is finite:
  `log1p(c*x) = log(c) + log(x) + log1p(1/(c*x))` (pattern 3).
- **Regime:** c > 1 and x > 1.8e308/c. For c = 1e10 that is x > 1.8e298; for
  c = 1000, x > 1.8e305. The true logsf there is small (-7e-8 for c = 1e10), so
  the error is total. pdf and sf are not affected: there the true pdf is
  subnormal, and `sc.inv_boxcox` handles the sf correctly.
- **Reproducer:** `python accuracy.py refs/genpareto_ref.py --at logsf 1e300 1e10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logsf(1e+300, 1e10)` | -inf | -7.138013788281542e-08 | inf | 0.0014 | loss |
  | `logpdf(1e+300, 1e10)` | -inf | -713.8013788995343 | inf | 0.0014 | loss |
  | `logpdf(1e+308, 1000)` | -inf | -716.8200678850693 | inf | 0.0014 | loss |

- **Fix direction:** for c*x > 1 use `log(c) + log(x) + log1p(1/(c*x))`
  (or `log(x) + log(c + 1/x)`).

### F4: logpdf and logsf return -inf at an inexact upper endpoint (c < 0)

- **Code:** `_logpdf`, line 3223 and `_logsf`, line 3234.
- **Mechanism:** for c < 0 the support ends at b = -1/c. When -1/c is not
  exactly representable, `x = b` (the rounded endpoint) is inside the support
  and `1 + c*x` is about 1e-16 in exact arithmetic. The rounded product
  `c*x` is exactly -1, so `log1p(-1) = -inf` (pattern 5: the 1 + c*x
  cancellation is decided entirely by the rounding of the product). cond (arg)
  is 5.2e15 here, but -inf counts as loss under the severity rule. In practice
  this matters only for code that evaluates exactly at `support()[1]`.
- **Regime:** c < 0 with -1/c inexact, x within ~1 ulp of -1/c. Seen for
  c = -1e-160 and c = -1e-20.
- **Reproducer:** `python accuracy.py refs/genpareto_ref.py --at logpdf 1e160 -1e-160`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e+160, -1e-160)` | -inf | -3.9870661983575e+161 | inf | 5.2e+15 | loss |
  | `logsf(1e+160, -1e-160)` | -inf | -3.9870661983575e+161 | inf | 5.2e+15 | loss |

- **Fix direction:** compute `1 + c*x` with an error-free product (fma or a
  TwoProduct split) before taking the log; low priority.

### F5: pdf, sf and isf lose up to ~500 ulp far in the tail

- **Code:** `_pdf`, line 3219: `np.exp(self._logpdf(x, c))`; `_sf`, line 3230:
  `sc.inv_boxcox(-x, -c)` (= `exp(log1p(c*x)/(-c))`); `_isf`, line 3241:
  `-sc.boxcox(q, -c)` (= `expm1(-c*log(q))/c`).
- **Mechanism:** each method exponentiates a log quantity L with |L| up to
  ~700. The 1/2-ulp rounding error of L (absolute about |L|*1.1e-16) becomes a
  relative error of the same size after exp, i.e. up to ~|L|/2 ulp. The
  exponent is a rounded large argument raised to a power, the exp analogue of
  pattern 6. cond (arg) is only 1-11 here, so the error is beyond
  conditioning. cond (shapes) is 300-700 at these points: the values are very
  sensitive to c.
- **Regime:** pdf and sf with values below ~1e-60 (c >= 0.1); isf for q < ~1e-60
  with c ~ 1. Measured: 75 minor pdf points, 27 sf, 34 isf; max 519, 487, 173 ulp.
- **Reproducer:** `python accuracy.py refs/genpareto_ref.py --at pdf 1e240 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e+240, 1000)` | 5.714786366719329e-244 | 5.7147863667186704e-244 | 519 | 1 | minor |
  | `sf(1e+30, 0.1)` | 9.999999999999282e-291 | 1.0000000000000364e-290 | 487 | 10 | minor |
  | `isf(1e-300, 1)` | 9.999999999999763e+299 | 9.999999999999999e+299 | 106 | 1 | minor |

- **Fix direction:** for c*x >> 1 use the power form directly, e.g.
  `sf = (c*x)**(-1/c) * exp(-log1p(1/(c*x))/c)` and `pdf = sf/(1 + c*x)`.
  That bounds the error by about 1/c ulp. Low priority.

## RuntimeWarnings

- logpdf, logsf, pdf: `underflow encountered in multiply` for tiny |c| or tiny x
  (the product `c*x` underflows). This coincides with F2 where the result is
  wrong. For x near 0 with moderate c it is spurious: the result is accurate.
- logpdf, logsf, pdf: `overflow encountered in multiply` for c = 1000 and
  c = 1e10 (F3; the values are wrong there).
- logpdf, pdf: `underflow encountered in divide` for c in {0.1, 1e-8, -1e-8,
  1000, 1e10} at tiny x (`xlog1py(...)/c` gives a subnormal). The results
  there are accurate, so these warnings are spurious.
- pdf: `underflow encountered in exp` where the true pdf is subnormal (expected).
- Warnings are recorded per batched call, so they are attributed per parameter
  set rather than per point.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| ppf | full q grid, all parameter sets incl. c = 0, 1e-300, -1e-160, 1000, 1e10; expm1 overflow window c*(-log1p(-q)) in (709.8, 709.8 + log c) | 441 ulp, all within conditioning (cond 727 at c = 1e10) |
| isf | full q grid, all sets except F5; expm1 overflow window for c = 1000 and 1e6 (`isf(0.49, 1000)` 86.5 ulp, `isf(0.99928..., 1e6)` 292 ulp, both ok) | 345 ulp within conditioning (cond 1e3) |
| cdf | all sets with \|c\| >= 1e-20, incl. c = 0 and the near-endpoint region for c < 0 | 1.92 ulp |
| sf | left tail and bulk all sets; tiny \|c\| (no shortcut in `inv_boxcox`); c*x overflow for c = 1e10; near endpoint for c < 0 | 7.08 ulp median high tail; 27.7 ulp at c = -0.5 with cond 2e15 (ok); beyond that F5 |
| logpdf, logsf | all sets outside F2-F4; c = +-1e-8 (c != 0 branch near the limit), c = 0 (fill branch) | 1.25 ulp; c = -1e-8 near endpoint 3.0e12 ulp with cond 2.8e13 (ok) |
| pdf | c < 0 incl. c = -2 (pdf infinite at endpoint), tiny \|c\| | 14.2 ulp at c = -0.5 (cond 1e15); beyond that F5 |
| ppf, isf | small-lambda threshold \|lmbda\| < 1e-19 in `boxcox`/`boxcox1p` (c = 1e-300, -1e-160) | 0.73 ulp |

## Suspected, not confirmed

none. All hypotheses in `scratch/genpareto/notes.md` were tested. H4 (expm1
overflow in ppf/isf for large c) did not show up, because `sc.boxcox` and
`sc.boxcox1p` stay finite in that window. H7 (endpoint cancellation for
c < 0) stays within conditioning except at the endpoint itself (F4).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | `log(cdf)` for x < median, `log1p(-sf)` above | `log(-expm1(logsf))` = `log1mexp(logsf)` with logsf = `-log1p(c*x)/c` (fixed per F2/F3); c = 0: `log(-expm1(-x))` | Right tail is fine (`logcdf(1e20, 1)` = -9.999999999999992e-21, true -1e-20). Below the median it inherits F1: `logcdf(0.5, 1e-200)` = -0.693 vs true -0.933 |

## Shared code

- **`scipy.special.inv_boxcox1p`** (F1): returns `x` when `|lmbda*x| < 1e-154`.
  The correct small-lambda limit is `expm1(x)`, so the result is wrong whenever
  lmbda is tiny and x is not. Any caller that passes |lmbda| below ~1e-139 is
  affected. In scipy.stats, genpareto's `_cdf` is the known user. It is worth
  checking other distributions (and `scipy.stats.boxcox` inverses) for the same
  call. `inv_boxcox`, `boxcox` and `boxcox1p` did not show this problem
  (boxcox/boxcox1p switch to `log` only for |lmbda| < 1e-19, which is
  accurate).

## Existing GitHub issues

- #5048 (closed): BUG: Upper tail accurracy lost in genpareto.logsf - relates to F2, F3, F4 (introduced the current `-log1p(c*x)/c` form of `_logsf`; the tiny/large-c and endpoint cases were not addressed)
- No issue found for the `inv_boxcox1p` small-lambda shortcut (F1); searched "genpareto" and "inv_boxcox1p".

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(2.0, 1e-300)` | 2.0 | 0.86466471676338731 | verified (loss) | reference by `mp.quad` of the docstring pdf; 100/200 digits agree. `inv_boxcox1p(-2, -1e-150)` is correct and `inv_boxcox1p(-2, -1e-300)` returns -2, so the shortcut sits at \|lmbda*x\| ~ 1e-154, as stated (line 3227) |
| F1 | `cdf(5e+145, 1e-300)` | 5e+145 | 1.0 | verified (loss) | |
| F1 | `cdf(1.0, 1e-160)` | 1.0 | 0.63212055882855768 | verified (severe) | |
| F1 | `cdf(0.001, 1e-152)` | 0.001 | 0.00099950016662500835 | verified (severe) | |
| F1 | `cdf(1e-10, 1e-300)` | 1e-10 | 9.9999999995000004e-11 | verified (moderate) | logcdf note also reproduced: -0.6931 vs true -0.93275212956718857 |
| F2 | `logpdf(1e-30, 1e-300)` | -0.0 | -1.0000000000000001e-30 | verified (loss) | `1e-300*1e-30` = 0.0 in double (line 3223) |
| F2 | `logpdf(2.4703282292062327e-164, -1e-160)` | -0.0 | -2.4703282292062327e-164 | verified (loss) | |
| F2 | `logsf(1e-20, 1e-300)` | -9.99988867182683e-21 | -9.9999999999999995e-21 | verified (severe) | 5.0e10 ulp, subnormal product (line 3234) |
| F2 | `logpdf(4.89124989382834e-24, 1e-300)` | -4.940656458412465e-24 | -4.89124989382834e-24 | verified (severe) | 4.5e13 ulp |
| F3 | `logsf(1e+300, 1e10)` | -inf | -7.1380137882815416e-08 | verified (loss) | reference by `-(log c + log x + log1p(1/(c x)))/c`; `1e10*1e300` = inf in double |
| F3 | `logpdf(1e+300, 1e10)` | -inf | -713.8013788995343 | verified (loss) | |
| F3 | `logpdf(1e+308, 1000)` | -inf | -716.82006788506936 | verified (loss) | |
| F4 | `logpdf(1e+160, -1e-160)` | -inf | -3.9870661983575002e+161 | verified (loss) | reference by both `log1p` and `log(power(1+c*x, -1-1/c))` with the exact doubles; `-1e-160*1e160` = -1.0 in double. Practical relevance low (argument exactly at `support()[1]`, cond 5.2e15) |
| F4 | `logsf(1e+160, -1e-160)` | -inf | -3.9870661983575002e+161 | verified (loss) | |

No severe-only or moderate findings to sample beyond these (F5 is minor and was not re-checked). Scripts: `scratch/genpareto/verify/verify.py`.

Verified by dist-accuracy-verifier: 4 of 4 verified.
