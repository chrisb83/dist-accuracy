# maxwell: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `maxwell_gen`, lines 7125-7187
- **Implemented:** pdf (7158-7160), logpdf (7162-7165), cdf (7167-7168), sf (7173-7174), ppf (7170-7171), isf (7176-7177)
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** `chi` with df = 3 (same formulas, but maxwell has its own code: `x*x` instead of `x**2`, no `exp(logpdf)`); `gamma`/`chi2` (same incomplete-gamma calls at a = 1.5); helpers: none; special functions: `sc.gammainc`, `sc.gammaincc`, `sc.gammaincinv`, `sc.gammainccinv`
- **Reference:** `refs/maxwell_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** minor in cdf and ppf (shared `scipy.special` igam prefactor); no moderate, severe or loss

## Summary

All six implemented methods are accurate for every argument where the true
result is a normal double. The pdf, logpdf, sf and isf stay within
conditioning everywhere (max 1.9 ulp for logpdf, up to 329 ulp for pdf and
277 ulp for sf in the far right tail, where cond is about 1.4e3). The cdf far
left tail (x <= 1e-10) loses up to 200 ulp, and ppf for q < 1e-38 up to 108 ulp.
Both errors come from the `exp(a*log(y) - y - lgam(a))` prefactor inside
cephes `igam`/`igami` at a = 1.5 (shared code, reports/gamma.md F6,
reports/chi.md F9). Unlike `chi`, maxwell has no `x**2` overflow in logpdf
(`-0.5*x*x` is evaluated as `(-0.5*x)*x`). It also has no logpdf
cancellation, because there are no shapes. The inherited logsf and logcdf
return -inf once sf/cdf underflow, and closed forms exist for both.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | x <~ 1e-10 down to ~1e-103 (where cdf ~ 0.266 x**3 leaves the normal range) | rounding of the igam prefactor exponent `1.5*log(y) - y - lgam(1.5)` (shared, gamma F6; pattern 10/5) | 200 ulp | minor |
| F2 | ppf | q <~ 1e-38 down to 1e-300 | `gammaincinv(1.5, q)` inherits the igam error of F1 (shared); `sqrt` halves it | 108 ulp | minor |

### F1: cdf far left tail loses ~|a log y| ulp in the igam prefactor

- **Code:** `_cdf`, line 7168: `sc.gammainc(1.5, x*x/2.0)`
- **Mechanism:** shared `scipy.special` defect (reports/gamma.md F6,
  reports/chi.md F9; patterns 5 and 10). For small y, cephes `igam` computes
  the prefactor as `exp(a*log(y) - y - lgam(a))`. The exponent is about
  1.5*|log y| (about 700 at y = 5e-205), and its rounding error becomes a
  relative error of up to that many ulp in the result. The argument
  `x*x/2.0` is not the cause. Called directly at the exact double y,
  `sc.gammainc(1.5, y)` has the same errors: 201, 186, 165, 122, 27.5 and
  19 ulp at x = 1e-102, 1e-90, 1e-70, 1e-40, 1e-10, 1e-3
  (`scratch/maxwell/special_check.py`).
- **Regime:** x from ~1e-3 down to ~1e-103, the last x where the cdf is
  still a normal double. The error grows roughly with |log x|. 11 points
  are minor; cond (arg) is 3 everywhere.
- **Reproducer:** `python accuracy.py refs/maxwell_ref.py --at cdf 1e-102`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-102)` | 2.659615202676336e-307 | 2.6596152026762173e-307 | 200 | 3 | minor |
  | `cdf(1e-90)` | 2.659615202676328e-271 | 2.6596152026762176e-271 | 187 | 3 | minor |
  | `cdf(1e-70)` | 2.659615202676315e-211 | 2.659615202676218e-211 | 164 | 3 | minor |
  | `cdf(1e-40)` | 2.659615202676145e-121 | 2.6596152026762173e-121 | 122 | 3 | minor |
  | `cdf(1e-10)` | 2.659615202676202e-31 | 2.659615202676218e-31 | 27.4 | 3 | minor |

- **Fix direction:** for small x use the series
  `cdf = sqrt(2/pi) * x**3/3 * (1 - 3x**2/10 + ...)` (or
  `erf(x/sqrt2) - sqrt(2/pi)*x*exp(-x**2/2)` above the cancellation range);
  the general fix belongs in `igam_fac`.

### F2: ppf for small q inherits the igam error through gammaincinv

- **Code:** `_ppf`, line 7171: `np.sqrt(2*sc.gammaincinv(1.5, q))`
- **Mechanism:** shared. `gammaincinv(1.5, q)` refines its root with the
  inaccurate `igam` of F1, so y = x**2/2 carries an error of the same kind.
  At the exact reference y the error is 161, 215, 62.5, 62.9 and 4.2 ulp for
  q = 1e-300, 1e-290, 1e-200, 1e-100, 1e-17
  (`scratch/maxwell/special_check2.py`). `sqrt` halves it, which gives the
  ppf errors. The ppf condition number is 1/3, so nothing is excused by
  conditioning. Same as the 88 ulp at df = 3 in reports/chi.md
  ("Checked and fine").
- **Regime:** q <~ 1e-38 (13.6 ulp at q = 4.2e-39) down to 1e-300. 22 points
  are minor. isf (via `gammainccinv`) is fine, with at most 8.2 ulp.
- **Reproducer:** `python accuracy.py refs/maxwell_ref.py --at ppf 1e-290`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-290)` | 3.350120263580887e-97 | 3.350120263580807e-97 | 108 | 0.333 | minor |
  | `ppf(2.872984833353539e-288)` | 2.210572903061828e-96 | 2.2105729030617847e-96 | 88.1 | 0.333 | minor |
  | `ppf(1e-300)` | 1.554988080669685e-100 | 1.5549880806696572e-100 | 80.9 | 0.333 | minor |
  | `ppf(1e-100)` | 7.217615311637198e-34 | 7.2176153116372485e-34 | 31.5 | 0.333 | minor |

- **Fix direction:** for small q invert the leading series directly:
  `x = (3*q/sqrt(2/pi))**(1/3)` with a series correction (or a Newton step on
  an accurate small-x cdf); the general fix belongs in `igam`/`igami`.

## RuntimeWarnings

- **pdf** (line 7160): `overflow encountered in multiply` for x > ~1.34e154
  (`x*x`). For x > ~1.5e154, `_SQRT_2_OVER_PI*x*x` is inf and
  `exp(-x*x/2)` is 0, so `inf*0` gives `invalid value` and the pdf
  returns **nan**: `maxwell.pdf(1.6e154)` and `maxwell.pdf(inf)` are nan,
  although the true value underflows to 0. This is not counted as an
  accuracy finding because the true value is not a normal double, but the
  return value is wrong (it should be 0). Between 1.34e154 and 1.5e154 it
  correctly returns 0. `underflow encountered in exp/multiply` for
  x >~ 37.6 is expected, and the results there are accurate (see below).
- **logpdf** (line 7165): `overflow encountered in multiply` only once
  `(-0.5*x)*x` truly overflows (x > ~1.9e154), where -inf is correct.
  `logpdf(inf)` returns nan (`2*log(inf) - inf = inf - inf`), but the true
  value is -inf. Same caveat as for the pdf.
- **cdf, sf** (lines 7168, 7174): `overflow`/`underflow encountered in
  multiply` from `x*x/2.0` for x > 1.34e154 and x < ~1.5e-154. The results
  (1.0/0.0 and 0.0/1.0) are correct to double precision there: the true cdf
  at x < 1.5e-154 is ~1e-462. These warnings are spurious.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | x in 1e-300 ... 1e300, incl. the window 37.6 < x < 37.83 where `exp(-x*x/2)` is subnormal but the pdf is normal, and the `x*x` underflow edge ~1.5e-154 | 329 ulp in the subnormal-exp window (pattern 4), within cond (arg) ~1.43e3; 1.16 ulp in the bulk |
| logpdf | full grid, incl. 1.3e154 ... 1.9e154 (chi F3 regime: no overflow here because `(-0.5*x)*x` is evaluated left to right) | 1.88 ulp |
| cdf | bulk and right tail | 4.05 ulp |
| sf | full grid, right tail to underflow (~38.6) | 277 ulp at x ~ 38 (cond 1.37e3); 9.56 ulp in the bulk |
| ppf | bulk, q -> 1 (1 - 1e-16 ... 1 - 1e-9) | 3.35 ulp |
| isf | q in [1e-300, 1 - 1e-16] | 8.17 ulp (cond up to 3e15 near q -> 1) |

The `x*x/2` underflow (chi F1/F2) cannot cause a measurable error at df = 3:
the cdf and the ppf quantile are far from the underflow range wherever the
cdf is a normal double. For q down to 1e-300 the quantile is about 1e-100.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | log(sf) (generic; `log1p(-cdf)` above the median) | `-x**2/2 + log(sc.erfcx(x/sqrt2) + sqrt(2/pi)*x)` (from sf = erfc(x/sqrt2) + sqrt(2/pi) x exp(-x**2/2)); all terms positive | sf underflows for x >~ 38.6: `logsf(40)` returns -inf, true -796.5362874834134 (loss); the closed form gives -796.5362874834134 |
| logcdf | log(cdf) | `log(sc.gammainc(1.5, x*x/2))` while the cdf is normal; for small x `log(sqrt(2/pi)/3) + 3*log(x) + log1p(-3*x**2/10 + ...)` | cdf underflows for x <~ 1e-103: `logcdf(1e-110)` returns -inf, true -761.1774843293479 (loss); the leading term alone gives -761.177484329348 |

## Shared code

F1 and F2 are the `scipy.special` igam prefactor issue (`igam_fac`:
`exp(a*log(y) - y - lgam(a))`), already documented as reports/gamma.md F6
and reports/chi.md F9. Here it is confirmed at a = 1.5 directly on
`sc.gammainc` (up to 201 ulp) and `sc.gammaincinv` (up to 215 ulp in y).
Nothing new is specific to maxwell.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

pending
