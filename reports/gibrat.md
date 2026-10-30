# gibrat: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `gibrat_gen`, lines 7061-7119 (module helper `_lognorm_logpdf`, lines 6852-6857)
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf (logpdf via the module helper `_lognorm_logpdf(x, 1.0)`, shared with lognorm; pdf = `exp(logpdf)`)
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** lognorm (gibrat = lognorm with s = 1; shares `_lognorm_logpdf`); helpers: `_lognorm_logpdf`, `_norm_cdf`, `_norm_sf`, `_norm_ppf`, `_norm_isf`; special functions: `sc.ndtr`, `sc.ndtri` (via the helpers)
- **Reference:** `refs/gibrat_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in logpdf

## Summary

cdf, sf, ppf and isf are written correctly (`ndtr(log x)`, `ndtr(-log x)`,
`exp(±ndtri(q))`, with no `1 - F`) and are accurate in the bulk and the
right/left tails where the result is near 1. The shared helper
`_lognorm_logpdf` forms `x*sqrt(2 pi)` before the log. That product overflows
for x > 7.2e307, so logpdf returns `-inf` (lognorm F1). It also loses bits
when x is subnormal, so logpdf at x = 5e-324 is off by 2.9e9 ulp. The second
problem is new: the lognorm report did not cover it. In the far tails
(|log x| from 8 to 38) cdf and sf are off by up to 347 ulp. Most of that error
is the known far-tail error of `sc.ndtr` itself, and only a small part comes
from rounding log(x). pdf (173 ulp) and ppf/isf (32 ulp) lose a little more
accuracy because `exp` magnifies small absolute errors. The inherited logcdf
and logsf return `-inf` once ndtr underflows (x < ~1e-17 or x > ~1e17);
`sc.log_ndtr` would fix both.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf | x > 7.2e307 | `x*sqrt(2 pi)` overflows inside the log (pattern 3; shared, lognorm F1) | -inf for -252566.7 | loss |
| F2 | logpdf | subnormal x (x < ~1e-313) | `x*sqrt(2 pi)` rounded to a subnormal loses bits before the log (pattern 4; shared helper) | 2.9e9 ulp | severe |
| F3 | cdf, sf | far tails, \|log x\| ~ 8 ... 38 | `sc.ndtr` far-tail error (exp(-z**2) rounding inside erfc), plus a small part from the rounding of log x | 347 ulp | minor |
| F4 | pdf | large \|log x\| (x < ~1e-9 or x > ~1e8) | exp(logpdf) turns the absolute rounding error of the large logpdf terms into a relative error (shared, lognorm F5) | 173 ulp | minor |
| F5 | ppf, isf | small q (both methods), q < ~1e-70 | exp magnifies the ~1 ulp relative error of ndtri by \|ndtri(q)\| | 32.4 ulp | minor |

### F1: logpdf returns -inf when x*sqrt(2 pi) overflows

- **Code:** `_logpdf`, line 7096 -> `_lognorm_logpdf`, lines 6855-6856: `- np.log(s * x * np.sqrt(2 * np.pi))` with s = 1.0
- **Mechanism:** the product is formed before the log. For x > 1.8e308/sqrt(2 pi) = 7.2e307
  it is inf and logpdf is `-inf`, although the true value is about -2.5e5
  (pattern 3). pdf is not affected because the true pdf there is far below
  the normal range. This is lognorm F1 at s = 1; see `reports/lognorm.md`.
- **Regime:** x > 7.2e307. `logpdf(7e307)` is still exact (0 ulp in the run); `logpdf(8e307)` = -inf.
- **Reproducer:** `python accuracy.py refs/gibrat_ref.py --at logpdf 1.7e308`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(8e+307)` | -inf | -252031.2955157827 | inf | 0.00282 | loss |
  | `logpdf(1.7e+308)` | -inf | -252566.73727865994 | inf | 0.00281 | loss |

- **Fix direction:** `-log(x)**2/2 - log(x) - _norm_pdf_logC` (that is,
  `_norm_logpdf(log x) - log x`). This fixes F1 and F2 together.

### F2: logpdf is severely wrong for subnormal x

- **Code:** `_lognorm_logpdf`, lines 6855-6856: `- np.log(s * x * np.sqrt(2 * np.pi))`
- **Mechanism:** for subnormal x, the product `x*sqrt(2 pi)` is rounded to a
  subnormal, which keeps only a few significant bits (pattern 4). Its relative
  error turns into an absolute error of the log term, and so of logpdf.
  Measured product errors: 0.197 relative at x = 5e-324 (log term off by
  0.180), 2.6e-3 at 1e-322 and 8.2e-5 at 1e-320. The quadratic term
  `-log(x)**2/2` (~-2.7e5) uses `log(x)` of the exact subnormal input and is
  accurate, so the whole error comes from the product. The result itself is a
  normal number of size ~2.7e5, and cond (arg) is only 0.0027. The lognorm
  report observed accurate results with a subnormal `s*x`, but only for
  s = 1e-20. There the quadratic term (~1e305) hides the error. At s = 1 it
  does not.
- **Regime:** x below the normal range (x < 2.2e-308). The error is severe
  for x <= ~1e-320 and drops to ok around x = 1e-315 (7 ulp). The same holds
  for lognorm with moderate s, where `s*x*sqrt(2 pi)` is subnormal.
- **Reproducer:** `python accuracy.py refs/gibrat_ref.py --at logpdf 5e-324`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(5e-324)` | -276352.1688815229 | -276351.9892077675 | 2.9e+09 | 0.00269 | severe |
  | `logpdf(1e-322)` | -274129.32634898537 | -274129.3289967867 | 4.4e+07 | 0.0027 | severe |
  | `logpdf(1e-320)` | -270721.28307521885 | -270721.28315714485 | 1.4e+06 | 0.00272 | severe |
  | `logpdf(1e-315)` | -262316.0246414489 | -262316.02464144933 | 7 | 0.00276 | ok |

- **Fix direction:** as F1: compute `log(x) + log(s) + log(sqrt(2 pi))` as a sum, never the product.

### F3: cdf and sf far tails, about 350 ulp from sc.ndtr

- **Code:** `_cdf`, line 7099: `_norm_cdf(np.log(x))`; `_sf`, line 7105: `_norm_sf(np.log(x))` = `sc.ndtr(-np.log(x))`.
- **Mechanism:** the measured error is mostly the far-tail error of
  `sc.ndtr` itself. At exact arguments: `ndtr(-30)` is 256 ulp off,
  `ndtr(-23.025850929940457)` (= the double log(1e-10)) 302 ulp, `ndtr(-20)` 128 ulp,
  `ndtr(-8)` 32 ulp. Of this, the rounding of y = z/sqrt 2 accounts for part
  (150 / 250 / 67 / 20 ulp), and erfc's own error at the rounded y for the rest
  (106 / 53 / 61 / 12 ulp). This is the ndtr behaviour documented under
  "Shared code" in `reports/exponnorm.md`. The rounding of `log(x)` adds at
  most z**2 * |dz| ~ 45 ulp at x = 1e-10 and 3 ulp at x = 9.36e-14 (where log
  x = -30 almost exactly). The total is 347 ulp at x = 1e-10. Relative to ndtr's
  own conditioning (~z**2) this is acceptable for ndtr. For gibrat, however,
  cond (arg) is only ~|z| (23.1), so the tool rates it beyond conditioning
  (minor). Correction to lognorm F6: that report attributes this error to the
  rounding of z = log(x)/s with ndtr accurate for its input. At s = 1 most of
  it comes from ndtr. sf in the right tail mirrors cdf in the left tail exactly.
- **Regime:** cdf for x < ~3e-4 (log x < -8), sf for x > ~3e3, down to the
  point where the result leaves the normal range (|log x| ~ 37.5).
- **Reproducer:** `python accuracy.py refs/gibrat_ref.py --at cdf 1e-10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-10)` | 1.2841756306434319e-117 | 1.2841756306435309e-117 | 347 | 23.1 | minor |
  | `sf(1e+10)` | 1.2841756306434319e-117 | 1.2841756306435298e-117 | 343 | 23.1 | minor |
  | `cdf(9.357622968840175e-14)` | 4.906713927147908e-198 | 4.90671392714819e-198 | 259 | 30 | minor |
  | `cdf(0.00033546262790251185)` | 6.22096057427174e-16 | 6.220960574271786e-16 | 32.8 | 8.12 | minor |

- **Fix direction:** in `sc.ndtr` (shared): an accurately rounded far tail,
  e.g. `0.5*erfcx(-z/sqrt2)*exp(-z**2/2)` with z**2/2 split exactly
  (Dekker product). Low priority.

### F4: pdf = exp(logpdf) magnifies the rounding of large log terms

- **Code:** `_pdf`, line 7093: `np.exp(self._logpdf(x))`; terms from line 6855-6856.
- **Mechanism:** this is lognorm F5 at s = 1. logpdf = `-L**2/2 - log(x sqrt(2 pi))`
  with L = log x. At x = 5.18e-17 (L = -37.5) the quadratic term is -703. Its
  absolute rounding error (a few eps*703) becomes a relative error of the pdf
  after `exp`, so the pdf is off by about 1e2 ulp with cond (arg) 36.5.
- **Regime:** |log x| > ~15, both tails, while the pdf is a normal double (|log x| < ~37.6).
- **Reproducer:** `python accuracy.py refs/gibrat_ref.py --at pdf 5.175555005801869e-17`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(5.175555005801869e-17)` | 3.3392239679544857e-290 | 3.339223967954614e-290 | 173 | 36.5 | minor |
  | `pdf(2.104223637677623e-17)` | 1.2055425770182242e-304 | 1.2055425770181884e-304 | 134 | 37.4 | minor |
  | `pdf(10686474581524.463)` | 1.378982491968232e-209 | 1.3789824919682012e-209 | 101 | 31 | minor |

- **Fix direction:** only partly fixable in double precision: needs log(x)
  and L**2/2 in extended precision (double-double), see lognorm F5.

### F5: ppf/isf, exp magnifies the rounding of ndtri

- **Code:** `_ppf`, line 7102: `np.exp(_norm_ppf(q))`; `_isf`, line 7108: `np.exp(_norm_isf(p))` = `exp(-ndtri(p))`.
- **Mechanism:** `exp(t)` turns an absolute error of t into the same relative
  error of the result. ndtri is accurate to ~0.8-0.9 ulp relative
  (`ndtri(2.87e-288)`: 0.90 ulp, t = -36.3), so the quantile is off by about
  0.9 * 36.3 = 32.5 ulp (measured 32.4). cond (arg) is ~1/|t| (0.03), so this is
  beyond conditioning. This is lognorm F7 at s = 1, without the product rounding.
- **Regime:** small q (ppf: lower quantiles, isf: upper quantiles), |ndtri(q)| > ~15 (q < ~1e-70).
- **Reproducer:** `python accuracy.py refs/gibrat_ref.py --at ppf 2.872984833353539e-288`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(2.872984833353539e-288)` | 1.7793804885043971e-16 | 1.7793804885043843e-16 | 32.4 | 0.0276 | minor |
  | `isf(2.872984833353539e-288)` | 5619933490675279.0 | 5619933490675319.0 | 32.1 | 0.0276 | minor |
  | `ppf(5.6234132519034905e-226)` | 1.1798057723857771e-14 | 1.1798057723857708e-14 | 24.1 | 0.0312 | minor |

- **Fix direction:** needs ndtri with a correction term (double-double
  result); low priority.

## RuntimeWarnings

- logpdf: `overflow encountered in multiply` with F1 (x > 7.2e307).
- logpdf: `underflow encountered in multiply` for every subnormal x (F2).
  At x = 1e-310 the warning is spurious: the result is exact (0 ulp).
  At x = 1e-315 the result is still ok (7 ulp).
- pdf: `underflow encountered in exp` where the true pdf is below the normal
  range (correct 0/subnormal results), and `overflow encountered in multiply`
  through logpdf for x > 7.2e307 (the true pdf is 0 there, so the result is correct).
- cdf, sf, ppf, isf: no warnings.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | bulk, x near 1 (1 ± 2**-52, 1 ± 1e-12), mode exp(-1), low tail down to 1e-300 (normal x), high tail up to 7e307 | 1.28 ulp low tail, 1.09 ulp bulk, 0 ulp high tail (excluding F1) |
| pdf | bulk | 2.8 ulp |
| cdf | bulk, right tail (cdf -> 1) | 0.788 ulp bulk, 0 ulp high tail |
| sf | bulk, left tail (sf -> 1) | 3.91 ulp bulk, 0 ulp low tail |
| ppf | bulk, q -> 1 down to 1 - 2**-53 | 1.41 ulp bulk, 6.36 ulp high tail (within conditioning) |
| isf | bulk, q -> 1 | 1.56 ulp bulk, 6.29 ulp high tail (within conditioning) |
| logpdf | cancellation near zeros of logpdf (lognorm F4) | not applicable: logpdf has no zeros for s = 1 (max of logpdf is -0.419) |

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) (rv_continuous; uses log1p(-sf) above the median) | `sc.log_ndtr(np.log(x))` (as lognorm `_logcdf` does) | ndtr(log x) underflows for log x < ~-38.5, i.e. x < ~1e-17: `logcdf(1e-17)` = -inf (true -770.7111128608465), `logcdf(1e-20)` = -inf (true -1065.1287963223679; `log_ndtr` gives -1065.128796322368). The right tail is already fine: `logcdf(1e4)` 48.4 ulp (minor, from ndtr's tail error) |
| logsf | log(sf) | `sc.log_ndtr(-np.log(x))` | sf underflows for x > ~1e17: `logsf(1e20)` = -inf (true -1065.1287963223679) |

## Shared code

- `_lognorm_logpdf(x, s)` (lines 6852-6857) is shared with **lognorm**:
  - F1 is lognorm F1 at s = 1.
  - F2 is new. It also applies to lognorm whenever `s*x*sqrt(2 pi)` is
    subnormal and the quadratic term does not hide the error, e.g. s ~ 1 and
    subnormal x. The lognorm report only checked s = 1e-20, where the error is
    hidden.
  - F4 (pdf) is lognorm F5 at s = 1.
  - lognorm F2, F3 and F4 do not occur at s = 1.
  The fix suggested for lognorm (`_norm_logpdf(log(x)/s) - log(x) - log(s)`) fixes F1 and F2 here too.
- `sc.ndtr` far tail (via `_norm_cdf`/`_norm_sf`, used by norm, lognorm and many
  others): 128-302 ulp at exact arguments in [-30, -20]. This is the main part
  of F3. It is already documented in `reports/exponnorm.md` (Shared code). It
  also corrects lognorm F6's attribution: at s = 1 the error comes mostly from
  ndtr, not from the rounding of log(x)/s.
- `sc.ndtri`: accurate (~0.9 ulp); F5 is only exp magnification.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `logpdf(8e+307)` | -inf | -252031.29551578269133 | verified | reference = mp.log(npdf(log x)/x), 100 vs 200 digits agree to 30 digits; `x*sqrt(2 pi)` overflows to inf (lines 6855-6856), mechanism confirmed |
| F1 | `logpdf(1.7e+308)` | -inf | -252566.73727865993465 | verified | same |
| F2 | `logpdf(5e-324)` | -276352.1688815229 | -276351.98920776747667 | verified | 2.93e9 ulp; product `x*sqrt(2 pi)` relative error 0.197 (subnormal rounding), mechanism confirmed |
| F2 | `logpdf(1e-322)` | -274129.32634898537 | -274129.32899678670108 | verified | 4.35e7 ulp; product rel. error 2.64e-3 |
| F2 | `logpdf(1e-320)` | -270721.28307521885 | -270721.28315714487534 | verified | 1.36e6 ulp; product rel. error 8.19e-5 |
| F2 | `logpdf(1e-315)` | -262316.0246414489 | -262316.02464144933972 | verified | 7 ulp (ok), boundary as reported |

No moderate findings to sample (F3-F5 are minor). Scripts: `scratch/gibrat/verify/verify_logpdf.py`.

Verified by dist-accuracy-verifier: 2 of 2 verified.
