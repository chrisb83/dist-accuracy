# lognorm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `lognorm_gen`, lines 6860-7055 (module helper `_lognorm_logpdf`, lines 6852-6857)
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf (logpdf via the module helper `_lognorm_logpdf`; pdf = `exp(logpdf)`)
- **Inherited (not measured):** none
- **Related:** gibrat (calls `_lognorm_logpdf(x, 1.0)`); helpers: `_lognorm_logpdf`, `_norm_cdf`, `_norm_sf`, `_norm_logcdf`, `_norm_logsf`, `_norm_ppf`, `_norm_isf`; special functions: `sc.ndtr`, `sc.log_ndtr`, `sc.ndtri` (via the helpers)
- **Reference:** `refs/lognorm_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (0.9536822696057533), (1), (0.001), (10), (1000), (1e10), (1e-150), (1e-158), (1e-200)
- **Worst finding:** loss in logpdf and pdf

## Summary

cdf, sf, logcdf, logsf, ppf and isf are written correctly: they use the
complementary and log forms of `ndtr`/`ndtri`. Their remaining errors are
up to 754 ulp in the cdf/sf tails and 260 ulp in ppf/isf. In the cdf/sf
tails, about half of the error comes from the far-tail error of `sc.ndtr`
itself and about half from rounding `log(x)/s`, which the tail magnifies.
In ppf/isf it comes from rounding `s*ndtri(q)`. All the serious problems are
in the shared helper `_lognorm_logpdf`, and so also in pdf.
`log(s*x*sqrt(2*pi))` forms the product `s*x` first. When that product
overflows, logpdf returns `-inf` (for example at x > 7.2e307 when s = 1).
When it underflows to 0, logpdf returns `+inf`, with the wrong sign
(s < ~1e-16). When it is subnormal, it keeps only a few bits, and logpdf
and pdf at subnormal x are off by up to 2.8e10 and 1.2e13 ulp (F8).
`log(x)**2 / (2*s**2)` squares s first. For s < 1.57e-162,
`s**2` is 0 and `0/0` at x = 1 gives nan. For 1.6e-162 < s < 1.5e-154,
`s**2` is subnormal and the error reaches 5e10 ulp. Close to the zeros of
logpdf, two large terms cancel and the result is 0.0 or off by 2.2e8 ulp.
pdf = `exp(logpdf)` turns the absolute rounding error of logpdf's large terms
into a relative error, up to 1.12e3 ulp.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf | s*x*sqrt(2 pi) > 1.8e308, i.e. x > 7.2e307/s (any s) | product inside log overflows (pattern 3) | -inf for -252566.7 | loss |
| F2 | logpdf | s*x*sqrt(2 pi) < ~2.5e-324 (needs s < ~1e-16) | product inside log underflows to 0, log(0) = -inf, sign flips (pattern 4) | +inf for -2.5e39 | loss |
| F3 | logpdf, pdf | s < 1.5e-154 (s**2 subnormal), s < 1.57e-162 (s**2 = 0) | `2*s**2` underflows (pattern 4); 0/0 at x = 1 | nan for 3.99e199 (pdf); 5.0e10 ulp | loss |
| F4 | logpdf | x near the zeros of logpdf (measured for s = 10) | cancellation of -log(x)**2/(2 s**2) and -log(s x sqrt(2 pi)) (pattern 5) | 0.0 for 3.36e-15; 2.2e8 ulp | loss |
| F5 | pdf | large \|log x\| (logpdf terms of size several hundred) | exp(logpdf) turns absolute error of the large terms into relative error | 1.12e3 ulp | moderate |
| F6 | cdf, sf, logcdf, logsf | tails, \|z\| = \|log(x)/s\| up to ~38 | far-tail error of `sc.ndtr` itself (shared), plus rounding of z = log(x)/s magnified ~z**2; similar size | 754 ulp | minor |
| F7 | ppf, isf | far tails, \|s*ndtri(q)\| up to ~700 | rounding of s*ndtri(q) magnified by \|log result\| in exp | 260 ulp | minor |
| F8 | logpdf, pdf | subnormal x with s*x*sqrt(2 pi) subnormal (x < ~1e-315; s from ~0.2 up to ~1e3) | product inside log rounded to a subnormal with few bits (pattern 4) | 2.8e10 ulp (logpdf), 1.2e13 ulp (pdf) | severe |

### F1: logpdf returns -inf when s*x overflows

- **Code:** `_lognorm_logpdf`, lines 6855-6856: `- np.log(s * x * np.sqrt(2 * np.pi))`
- **Mechanism:** the product `s*x*sqrt(2 pi)` is formed before the log.
  Once it passes 1.8e308 it is inf, and logpdf is `-inf` even though the
  true value is an ordinary number such as -710 or -252566 (pattern 3).
  pdf is not affected in this regime: there pdf < 1/(s x sqrt(2 pi)) is
  already below 2.2e-308.
- **Regime:** x > 7.2e307/s. For s = 1 (and so for gibrat) that is
  x > 7.2e307; for s = 1e3, x > 7.2e304; for s = 1e10, x > 7.2e297.
  Boundary checked: `logpdf(7e307, 1)` is accurate (0.52 ulp), `logpdf(8e307, 1)` = -inf.
- **Reproducer:** `python accuracy.py refs/lognorm_ref.py --at logpdf 1.7e308 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.7e308, 1)` | -inf | -252566.73727865994 | inf | 0.00281 | loss |
  | `logpdf(1.7e308, 0.9536822696057533)` | -inf | -277624.68079659634 | inf | 0.00281 | loss |
  | `logpdf(1e307, 10)` | -inf | -3208.6081222477633 | inf | 0.00251 | loss |
  | `logpdf(1e305, 1000)` | -inf | -710.3617517112343 | inf | 0.00141 | loss |
  | `logpdf(1e300, 1e10)` | -inf | -714.7203173613589 | inf | 0.0014 | loss |

- **Fix direction:** split the log: `-log(x) - log(s) - _norm_pdf_logC`, or
  `_norm_logpdf(log(x)/s) - log(x) - log(s)` (fixes F1-F3 together).

### F2: logpdf returns +inf when s*x underflows

- **Code:** `_lognorm_logpdf`, lines 6855-6856: `- np.log(s * x * np.sqrt(2 * np.pi))`
- **Mechanism:** for small s and small x, the product `s*x*sqrt(2 pi)` rounds
  to 0. Then `log(0) = -inf`, and subtracting it makes logpdf `+inf`,
  although the true logpdf is a very large negative number
  (pattern 4, wrong sign). When the product is subnormal but not 0, the
  lost bits stay below the rounding of the much larger quadratic term
  (`logpdf(1e-300, 1e-20)`: 0.598 ulp).
- **Regime:** s*x*sqrt(2 pi) < ~2.5e-324. With x a normal double this needs
  s < ~1e-16, for example s = 1e-17 with x <= ~1e-308, or s = 1e-150 with
  x <= ~1e-175. Measured for every x from 1e-300 to 1e-180 at s = 1e-150.
- **Reproducer:** `python accuracy.py refs/lognorm_ref.py --at logpdf 1e-308 1e-17`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-308, 1e-17)` | inf | -2.5147963117621134e+39 | inf | 0.00282 | loss |
  | `logpdf(1e-300, 1e-150)` | inf | -2.3858541497152793e+305 | inf | 0.0029 | loss |
  | `logpdf(1e-180, 1e-150)` | inf | -8.589074938975005e+304 | inf | 0.00483 | loss |

- **Fix direction:** same as F1: compute `log(x) + log(s)` as a sum, not `log(s*x)`.

### F3: s**2 underflows for tiny s (nan at x = 1, severe near x = 1)

- **Code:** `_lognorm_logpdf`, line 6855: `-np.log(x)**2 / (2 * s**2)`; `_pdf` line 6914: `np.exp(self._logpdf(x, s))`
- **Mechanism:** `s**2` is formed separately (pattern 4). For s < 1.5e-154
  it is subnormal and keeps only a few significant bits: 24 bits at
  s = 1e-158 and 11 bits at s = 1e-160. That relative error carries
  straight into the quadratic term. For s < 1.57e-162, `s**2` is 0. At x = 1
  the code then computes `0/0 = nan`, although the true logpdf is
  `-log(s sqrt(2 pi))` (459.6 for s = 1e-200) and the true pdf
  `1/(s sqrt(2 pi))` = 3.99e199 is finite. For x != 1 the quadratic term is
  `-inf`, which is also the correctly rounded result there, so x = 1 is the
  only lost point.
- **Regime:** s < 1.57e-162: nan at x = 1 (pdf and logpdf).
  1.6e-162 < s < 1.5e-154: relative error up to 2**-(bits of s**2) in the
  quadratic term for x close to 1. The tool rates these points severe,
  beyond the cond (arg) of about 2/|log x|. At x = 1 exactly the term is 0
  and the result is exact (`logpdf(1.0, 1e-160)`: 0 ulp).
- **Reproducer:** `python accuracy.py refs/lognorm_ref.py --at pdf 1.0 1e-200`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1.0, 1e-200)` | nan | 3.989422804014327e+199 | inf | 1 | loss |
  | `logpdf(1.0, 1e-200)` | nan | 459.59808006560445 | inf | 0.00218 | loss |
  | `logpdf(1.000001, 1e-160)` | -5.000050663832534e+307 | -4.999994999181918e+307 | 5.0e+10 | 2.0e+06 | severe |
  | `logpdf(1.0000001, 1e-158)` | -4.999999587540137e+301 | -4.999999505838716e+301 | 7.4e+07 | 2.0e+07 | severe |

- **Fix direction:** square the ratio, `-(log(x)/s)**2 / 2` (as `_norm_logpdf(log(x)/s)` does), so no `s**2` is formed.

### F4: logpdf cancels near its zeros

- **Code:** `_lognorm_logpdf`, lines 6855-6856: `-np.log(x)**2 / (2 * s**2) - np.log(s * x * np.sqrt(2 * np.pi))`
- **Mechanism:** logpdf is zero at the roots
  L = s**2 (-1 ± sqrt(1 - 2 log(s sqrt(2 pi))/s**2)) of L = log x. For s = 10
  the lower root is L = -196.7. There both terms are about 193.45 with
  opposite signs (pattern 5). Each carries an absolute rounding error of
  about eps*193, and `log(x)` itself is rounded to eps*|L| absolute, so the
  difference has an absolute error near 4e-14. Results smaller than that
  lose all their digits. cond (arg) is large here (2.9e14), but the error
  goes beyond it: the conditioning allows an absolute error of
  ~|dlogpdf/dlog x| * eps = 0.97 eps, while the rounding of the large terms
  produces ~|L| eps.
- **Regime:** x close to a zero of logpdf, where |logpdf| is comparable to
  eps*|log x| times a few hundred or less.
  Zeros exist when 2 log(s sqrt(2 pi)) <= s**2, i.e. for every s outside
  0.439 < s < 1.704. Measured beyond conditioning for s = 10. For
  s = 1e-3 the error (3.0e13 ulp) stayed within cond (5.7e16). gibrat
  (s = 1) has no zeros. cond (shapes) is 1.1e17 and 4.0e8 at the two points
  below (context only).
- **Reproducer:** `python accuracy.py refs/lognorm_ref.py --at logpdf 3.6600466269602515e-86 10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(3.660042966917285e-86, 10)` | 0.0 | 3.362223475099197e-15 | inf | 2.9e+14 | loss |
  | `logpdf(3.6600466269602515e-86, 10)` | 9.672478995526035e-07 | 9.67247945737884e-07 | 2.2e+08 | 1.0e+06 | severe |

- **Fix direction:** hard in double precision. Near a root, evaluate
  logpdf as `(L - L1)*(L - L2)/(-2 s**2)`, with `L - L1` computed from
  `log(x/x1)` plus a correction term for a double-double root
  `L1 = log(x1) + d`. Low priority: the absolute error stays at a few
  eps*|log x|.

### F5: pdf = exp(logpdf) magnifies the rounding of large log terms

- **Code:** `_pdf`, lines 6912-6914: `np.exp(self._logpdf(x, s))`; terms from line 6855-6856.
- **Mechanism:** logpdf is the sum of `-L**2/(2 s**2)`, `-L` and
  `-log(s sqrt(2 pi))`, with L = log(x) rounded. When these terms are large
  (|L| in the hundreds), each has an absolute error of eps times its size.
  `exp` turns that absolute error into a relative error of the pdf. For
  s = 10, x = 1e-210 the terms are -1169 and +483.5 (sum -688). The derivative
  with respect to log x is only 3.84, but the rounding of L alone contributes
  3.84*|L| eps, so the result is about 1e3 ulp off with cond (arg) 3.84. This is
  the lognormal form of the "normal density in a transformed variable"
  pattern seen in johnsonsb F2.
- **Regime:** |log x| large compared to 1 for any s with pdf still normal.
  Largest for moderate s (s = 10: 1.12e3 ulp; s = 1e3: 380 ulp; s = 1: 173 ulp;
  s = 1e10: 243 ulp). 152 points are beyond conditioning: one moderate,
  the rest minor.
- **Reproducer:** `python accuracy.py refs/lognorm_ref.py --at pdf 1e-210 10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-210, 10)` | 7.60146080706588e-300 | 7.601460807063997e-300 | 1.12e+03 | 3.84 | moderate |
  | `pdf(1e-170, 10)` | 7.539874691203295e-165 | 7.53987469120417e-165 | 523 | 2.91 | minor |
  | `pdf(1e+270, 1000)` | 3.288371224542732e-274 | 3.2883712245430097e-274 | 380 | 1 | minor |
  | `pdf(1e-250, 1e10)` | 3.989422804014104e+239 | 3.98942280401432e+239 | 243 | 1 | minor |

  (cond (shapes) is 2.34e3 at the first point, context only.)
- **Fix direction:** only partly fixable in double precision. A
  double-double `log(x)` (or `log1p` for x near 1) plus combining
  `-L**2/(2 s**2) - L` as `-L*(L/(2 s**2) + 1)` reduces the rounding to a
  few ulp times cond.

### F6: cdf/sf tails, sc.ndtr far-tail error plus rounding of z = log(x)/s

- **Code:** `_cdf` line 6920 `_norm_cdf(np.log(x) / s)`; `_sf` line 6929; `_logcdf` line 6923; `_logsf` line 6932.
- **Mechanism (rechecked):** two sources of similar size add up.
  (a) `sc.ndtr` is itself inaccurate in the far tail. At exact double
  arguments it is off by up to 52 / 245 / 704 / 1026 ulp for |z| in
  [5, 10) / [10, 20) / [20, 30) / [30, 37.5), with a median of about
  0.3 z**2 ulp (800 sampled z). This is the erfc behaviour documented in
  `reports/exponnorm.md` (Shared code) and gibrat F3.
  (b) `z = log(x)/s` is rounded (measured relative error 0.003 to 0.42 eps at
  the flagged points). In the tail, `ndtr(-|z|) ~ phi(z)/|z|` has relative
  sensitivity z**2 to z, so this adds z**2 * |dz/z| ulp.
  The two parts were separated by evaluating `sc.ndtr`/`sc.log_ndtr` at the
  double z that SciPy passes, against mpmath at that double z (part a) and at
  the exact z (part b) (`scratch/lognorm/recheck/f6_split.txt`, `f6_all.txt`):

  | point | z (double) | total | ndtr at double z | argument rounding |
  |---|---|---|---|---|
  | `cdf(1e-160, 10)` | -36.841 | 754 | 178 | 577 |
  | `cdf(1e-110, 10)` | -25.328 | 563 | 391 | 172 |
  | `cdf(1e-10, 1)` | -23.026 | 347 | 302 | 45 |
  | `cdf(9.357622968840175e-14, 1)` | -30.000 | 259 | 256 | 3 |
  | `cdf(1e-120, 10)` | -27.631 | 361 | 479 | 118 (opposite sign) |
  | `cdf(0.9631944177208218, 0.001)` | -37.500 | 595 | 208 | 388 |

  Of the 27 flagged cdf points, the ndtr part is the larger one at 20. The
  argument rounding dominates only where |dz/z| is unusually large, as at the
  754 ulp worst point (dz/z = 0.42 eps). At s = 1 (gibrat) the argument part
  is small. sf, logcdf (right tail) and logsf (left tail) give the same
  numbers: `log_ndtr(z)` for z > 0 inherits the error of `ndtr(-z)` exactly
  (178 ulp at z = 36.84, 256 ulp at z = 30). The left tail of `log_ndtr` is
  accurate (Checked and fine). The conditioning with respect to x is only
  `phi(z)/(s Q(z)) ~ |z|/s`, so both parts are beyond conditioning for s >~ 1.
  For the ndtr part alone (cond ~z**2 with respect to z) the error is
  within ndtr's own conditioning.
- **Regime:** |z| between ~5 and ~38 (cdf/sf normal). Worst for s >= 1
  (s = 10: 754 ulp at x = 1e-160, cond 3.69). For s = 1e-3, 616 ulp stays
  within cond 3.75e4.
- **Reproducer:** `python accuracy.py refs/lognorm_ref.py --at cdf 1e-160 10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-160, 10)` | 2.0107732810175033e-297 | 2.01077328101784e-297 | 754 | 3.69 | minor |
  | `sf(1e+160, 10)` | 2.0107732810175033e-297 | 2.01077328101784e-297 | 754 | 3.69 | minor |
  | `logcdf(1e+160, 10)` | -2.0107732810175033e-297 | -2.01077328101784e-297 | 754 | 3.69 | minor |
  | `logsf(1e-160, 10)` | -2.0107732810175033e-297 | -2.01077328101784e-297 | 754 | 3.69 | minor |
  | `cdf(1e-10, 1)` | 1.2841756306434319e-117 | 1.2841756306435309e-117 | 347 | 23.1 | minor |

- **Fix direction:** both parts have to be fixed. A more accurate z alone
  would leave the ndtr part (up to ~1e3 ulp at exact arguments). (a) An
  accurately rounded far tail in `sc.ndtr` (shared, see Shared code), e.g.
  `0.5*erfcx(-z/sqrt2)*exp(-z**2/2)` with z**2/2 split exactly. (b) Then `log(x)/s` to
  better than double precision (double-double log, or the low part dz
  applied as a correction `Q(z) * exp(-z*dz)`). Low priority.

### F7: ppf/isf magnify the rounding of s*ndtri(q) in exp

- **Code:** `_ppf` line 6926 `np.exp(s * _norm_ppf(q))`; `_isf` line 6935 `np.exp(s * _norm_isf(q))`.
- **Mechanism:** `exp(t)` turns the absolute error of t = s*ndtri(q) (product
  rounding plus the ~1 ulp error of ndtri, both relative to |t|) into a
  relative error of about |t| ulp in the result. cond (arg) is small here
  (~s/|z|), so this is beyond conditioning.
- **Regime:** |s*ndtri(q)| large, i.e. small q with s >= 1 (s = 10, q < 1e-20:
  up to 260 ulp; s = 1: up to 32 ulp). isf mirrors ppf exactly.
- **Reproducer:** `python accuracy.py refs/lognorm_ref.py --at ppf 2.872984833353539e-288 10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(2.872984833353539e-288, 10)` | 3.1819125588079338e-158 | 3.18191255880775e-158 | 260 | 0.276 | minor |
  | `isf(2.872984833353539e-288, 10)` | 3.1427639242689886e+157 | 3.1427639242691703e+157 | 260 | 0.276 | minor |
  | `ppf(2.872984833353539e-288, 1)` | 1.7793804885043971e-16 | 1.7793804885043843e-16 | 32.4 | 0.0276 | minor |

- **Fix direction:** an error-free product (fma) gives `t = p + e` and
  `exp(p)*(1 + e)`, which removes the product rounding. The ndtri part needs
  a more accurate ndtri. Low priority.

### F8: logpdf and pdf are severely wrong for subnormal x

- **Code:** `_lognorm_logpdf`, lines 6855-6856: `- np.log(s * x * np.sqrt(2 * np.pi))`; `_pdf` line 6914 `np.exp(self._logpdf(x, s))`
- **Mechanism:** for subnormal x, the product `s*x*sqrt(2 pi)` is often also
  subnormal. It is then rounded to a multiple of 4.9e-324 and keeps only a
  few significant bits (pattern 4). Its relative error becomes an absolute
  error of the log term, and so of logpdf. Measured product errors: -2.6e-3
  relative at (x, s) = (5e-324, 30), log term off by -0.00265, which is
  exactly the logpdf error (432.23868 vs 432.23604); 0.197 at (5e-324, 1);
  -3.1e-6 at (1e-320, 10). The quadratic term uses `log(x)` of the exact
  input and is accurate. This is the same mechanism as gibrat F2 (s = 1). The
  error is most visible where |logpdf| is small compared to the absolute
  error, that is for moderate s (10 ... 1e3), where logpdf is a few hundred to
  ~2000. pdf = `exp(logpdf)` turns the absolute error into the same relative
  error. pdf is a normal double at subnormal x only for s of about 20 to 1e3,
  and there it is off by up to 1.2e13 ulp. This is a separate finding, not
  folded into F1/F2. F1/F2 are the endpoints where the product becomes
  inf or 0 (`-inf`/`+inf` results, loss). F8 covers the products that stay
  finite and nonzero but are inaccurate. The fix is the same. With subnormal x
  the F2 endpoint is reached already for s < ~0.2: `logpdf(5e-324, 0.001)` =
  +inf (true -2.77e11).
- **Regime:** x subnormal and `s*x*sqrt(2 pi)` < 2.2e-308. Severe for
  x <= ~1e-320 at every s measured from 0.954 to 1e3. Moderate/minor around
  x = 1e-315 for s = 10 ... 100, ok at x = 1e-310. Measured
  for s in {1e-3, 0.954, 1, 10, 30, 100, 1e3, 1e10} and x in {5e-324, 1e-322,
  1e-320, 1e-315, 1e-310} (`scratch/lognorm/recheck/subnormal.txt`). Small s:
  the quadratic term hides the error (s = 1e-3: 241 ulp at 1e-320, minor;
  s = 1e-20 earlier: 0.598 ulp). Huge s: the product has more bits
  (s = 1e10: 77.5 ulp at 5e-324, minor). For normal x the product is subnormal
  only for s < 1, where the quadratic term hides the error.
- **Reproducer:** `python accuracy.py refs/lognorm_ref.py --at logpdf 5e-324 30`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(5e-324, 30)` | 432.2386834287832 | 432.2360356274526 | 2.8e+10 | 0.0004 | severe |
  | `logpdf(5e-324, 1)` | -276352.1688815229 | -276351.9892077675 | 2.9e+09 | 0.00269 | severe |
  | `logpdf(1e-320, 10)` | -1980.9661942499051 | -1980.9661973302514 | 7.0e+06 | 0.00321 | severe |
  | `logpdf(1e-320, 1000)` | 728.7290898139753 | 728.7290898873276 | 4.5e+05 | 0.00137 | moderate |
  | `logpdf(1e-315, 30)` | 428.72703503840194 | 428.7270350384306 | 301 | 0.000453 | minor |
  | `pdf(5e-324, 30)` | 5.2344984826200874e+187 | 5.220656903535348e+187 | 1.2e+13 | 0.173 | severe |
  | `pdf(1e-322, 100)` | 4.6630657363103e+307 | 4.662827107773954e+307 | 2.3e+11 | 0.926 | severe |
  | `pdf(1e-315, 100)` | 1.5038681921946094e+301 | 1.503868192198207e+301 | 1.08e+04 | 0.927 | moderate |

- **Fix direction:** as F1/F2: `_norm_logpdf(log(x)/s) - log(x) - log(s)`,
  so no product of x with anything is formed.

## RuntimeWarnings

- logpdf/pdf: `underflow encountered in square` (s**2) for s < 1.5e-154 on
  every point, including accurate results (`logpdf(1.0, 1e-160)`: 0 ulp). This
  part is spurious; the inaccurate points of F3 also warn. `invalid value
  encountered in divide` accompanies the F3 nan at x = 1.
- logpdf: `overflow encountered in multiply` with F1, `divide by zero
  encountered in log` / `underflow encountered in multiply` with F2. The
  underflow warning also appears where `s*x` is subnormal but the result is
  accurate (`logpdf(1e-300, 1e-20)`: 0.598 ulp). It accompanies F8 as well
  (`underflow encountered in multiply` wherever the product is subnormal).
- pdf: `underflow encountered in exp` where the true pdf is below the normal
  range (correct 0/subnormal results).
- ppf/isf: `overflow/underflow encountered in exp` for s = 1e3 and 1e10 where
  the true quantile is outside the double range (e.g. `ppf(1e-300, 1000)` true
  4.5e-16090, returns 0.0, ok). These warnings are expected.
- cdf, sf, logcdf, logsf: no warnings.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf, sf | bulk, both tails, x near 1, all parameter sets | 2.2 / 4.42 ulp in the bulk; tails as F6 |
| logcdf, logsf | the far-tail side (log_ndtr of very negative z), all parameter sets | 2.65 / 2.86 ulp |
| logcdf, logsf | tiny s (1e-150 ... 1e-200), \|z\| up to 1e152 | reference skipped by the harness (OverflowError, 64/105 points). Checked by hand against -z**2/2 - log\|z\| - log(2 pi)/2: `logcdf(0.5, 1e-150)` = `logsf(2, 1e-150)` = -2.402265069591007e+299, asymptotic -2.4022650695910071e+299 |
| ppf, isf | q in [1e-300, 1 - 2**-53], all parameter sets | bulk max 429 ulp (s = 1e3, within cond 847); s <= 1e-3: < 1 ulp |
| logpdf | bulk and tails away from F1-F4, all parameter sets | 0.55 ulp median in the low tail; subnormal s*x with normal x (`logpdf(1e-300, 1e-20)`) 0.598 ulp, because the quadratic term hides the product error (subnormal x: F8) |
| logpdf, pdf | s = 1e-150 (s**2 normal), x near 1 | pdf 36.9 ulp max, logpdf ok |

## Suspected, not confirmed

none

## Opportunities: inherited methods

none (all eight methods are implemented)

## Shared code

- `_lognorm_logpdf(x, s)` (lines 6852-6857) is shared with **gibrat**, which
  calls `_lognorm_logpdf(x, 1.0)` (line 7096). For s = 1:
  - F1 applies: `logpdf(x)` = -inf for x > 7.2e307 (measured:
    `lognorm.logpdf(1.7e308, 1)` = -inf vs -252566.73727865994).
  - F5 applies to pdf = exp(logpdf) (173 ulp worst at s = 1).
  - F8 applies to logpdf at subnormal x (`logpdf(5e-324, 1)` 2.9e9 ulp). This
    is gibrat F2, found first in the gibrat review.
  - F2, F3 and F4 do not apply: they need s < 1e-16 (or s < ~0.2 with
    subnormal x), s < 1.5e-154, and zeros of logpdf, which do not exist for
    0.439 < s < 1.704.
  The fix for F1-F3 and F8 (`_norm_logpdf(log(x)/s) - log(x) - log(s)`) fixes gibrat too.
- `_norm_cdf`, `_norm_sf`, `_norm_logcdf`, `_norm_logsf`, `_norm_ppf`,
  `_norm_isf` (lines 370-391) are thin wrappers around `sc.ndtr`,
  `sc.log_ndtr`, `sc.ndtri`, shared with norm and many others.
  - **`sc.ndtr` far tail:** inaccurate at exact arguments. Up to 52 / 245 /
    704 / 1026 ulp for |z| in [5, 10) / [10, 20) / [20, 30) / [30, 37.5),
    median ~0.3 z**2 ulp (`scratch/lognorm/recheck/ndtr_sweep.txt`). This is
    about half of F6 and the larger part at 20 of 27 flagged points (the
    earlier version of this report attributed F6 to argument rounding only).
    Same finding as `reports/exponnorm.md` (Shared code) and gibrat F3. It is
    within ndtr's own conditioning (~z**2), so it matters to callers whose
    conditioning is smaller (lognorm: ~|z|/s).
  - **`sc.log_ndtr(z)` for z > 0** (used by lognorm logcdf right tail and,
    via `_norm_logsf(z) = log_ndtr(-z)`, logsf left tail) returns
    a value whose error equals that of `ndtr(-z)` (178 ulp at z = 36.84, 256 ulp at z = 30).
    For z < 0 it is accurate.
  - `sc.ndtri`: no error traced to it beyond ~1 ulp. F7 comes from
    rounding of `s*ndtri(q)` and exp magnification.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `logpdf(1.7e308, 1)` | -inf | -252566.73727865993 | verified | also 0.954, 10, 1e3, 1e10 points -inf; boundary confirmed (7e307: 0.32 ulp, 8e307: -inf); mechanism matches line `np.log(s * x * np.sqrt(2 * np.pi))` |
| F2 | `logpdf(1e-308, 1e-17)` | inf | -2.5147963117621134e+39 | verified | also `logpdf(1e-300, 1e-150)`, `logpdf(1e-180, 1e-150)` = +inf; mechanism matches |
| F3 | `pdf(1.0, 1e-200)` | nan | 3.9894228040143269e+199 | verified | `logpdf(1.0, 1e-200)` nan vs 459.59808006560446; `logpdf(1.000001, 1e-160)` 5.01e10 ulp, `logpdf(1.0000001, 1e-158)` 7.36e7 ulp; 1.57e-162**2 = 0 confirmed; mechanism matches `2 * s**2` |
| F4 | `logpdf(3.660042966917285e-86, 10)` | 0.0 | 3.3622234750991971e-15 | verified | `logpdf(3.6600466269602515e-86, 10)` 2.15e8 ulp; mechanism (cancellation of the two terms) matches |
| F5 | `pdf(1e-210, 10)` | 7.60146080706588e-300 | 7.6014608070639962e-300 | verified | 1.12e3 ulp; `pdf(1e-170, 10)` 523 ulp; mechanism matches `_pdf = exp(_logpdf)` |
| F8 | `logpdf(5e-324, 30)` | 432.2386834287832 | 432.2360356274526 | verified | 2.76e10 ulp; also `logpdf(5e-324, 1)` 2.93e9, `logpdf(1e-320, 10)` 7.0e6, `logpdf(1e-320, 1000)` 4.53e5, `logpdf(1e-315, 30)` 301 ulp; `pdf(5e-324, 30)` 1.19e13, `pdf(1e-322, 100)` 2.3e11, `pdf(1e-315, 100)` 1.08e4 ulp; all 8 report rows match. Mechanism matches lines 6856 and 6914: float64 product `s*x*sqrt(2 pi)` relative error -2.64e-3 at (5e-324, 30), 0.197 at (5e-324, 1), -3.08e-6 at (1e-320, 10) |

Verifier references (`scratch/lognorm/verify/verify.py`, 100 and 200 digits, agree to 30 digits): logpdf as `log(mp.npdf(log(x)/s)) - log(x) - log(s)`; pdf as `d/dL mp.ncdf(L/s) / x` via `mp.diff` (for `pdf(1.0, 1e-200)` the numerical derivative was unstable, so `exp` of the logpdf route was used and checked against `1/(s sqrt(2 pi))`).

F8 verifier reference (`scratch/lognorm/verify/verify_f8.py`, 100 and 200 digits, agree to 30 digits): pdf as `mp.diff` of `mp.ncdf(log(x)/s)` with respect to log(x), divided by x, checked against `mp.npdf(log(x)/s)/(s x)`; logpdf as its log.

Verified by dist-accuracy-verifier: 6 of 6 verified.
