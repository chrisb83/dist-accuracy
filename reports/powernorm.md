# powernorm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `powernorm_gen`, lines 8980-9031
- **Implemented:** pdf, logpdf, cdf, sf, logsf, ppf, isf
- **Inherited (not measured):** logcdf (generic `rv_continuous` median switch: `log(cdf)` below the median, `log1p(-sf)` above)
- **Related:** powerlognorm (same construction in log x, see `reports/powerlognorm.md`); helpers: `_norm_pdf`, `_norm_logpdf`, `_norm_cdf`, `_norm_logcdf`, `_norm_ppf` (lines 362-379); special functions: `sc.ndtr`, `sc.log_ndtr`, `sc.ndtri` (via the helpers), `sc.expm1`
- **Reference:** `refs/powernorm_ref.py`, 400 digits for pdf/logpdf/cdf/sf/logsf, 100 digits for ppf/isf (root finding; 400 digits ran past the 20 min limit in the condition-number phase); self-checks: PASS
- **Parameter sets:** (4.445365225459078), (0.001), (1e-08), (0.5), (1), (50), (10000), (1e+06); plus single `--at` points with c = 1e10 and 1e20
- **Worst finding:** loss in ppf, isf, pdf, logpdf, logsf, cdf

## Summary

All methods are built on `logsf = c*log_ndtr(-x)`, which is accurate, and cdf,
sf and logsf are fine for ordinary shapes. The quantile functions are the weak
part. `ppf` evaluates `1 - q` and returns `-inf` for every q < 1.1e-16 (any c).
Both `ppf` and `isf` raise a probability to the power `1/c`. That power
underflows for small c and returns `+inf` (c = 1e-8: `isf` is inf for every q,
and `ppf` for every q > ~1e-5). For large c the power lands next to 1 and
`ndtri` loses up to all digits there. `pdf` returns nan/inf in the right tail
for c < 1 (x > ~37.5), because `ndtr(-x)**(c-1)` forms 0**negative. `logpdf`
loses up to ~1/c ulp to cancellation for small c (2e8 ulp at c = 1e-8). `logpdf`
and `logsf` also overflow for |x| > 1.34e154. `pdf` for large c in the left
tail loses ~c/4 ulp (pattern 6). Huge c (>= 1e10) makes cdf/logsf
underflow to 0 in the far left tail.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf | q < ~1e-5, any c; total for q < 1.1e-16 | `1.0 - q` rounds (pattern 8) | -inf for -8.67; 8.8e11 ulp | loss |
| F2 | ppf, isf | small c: `log1p(-q)/c` or `log(q)/c` < -745 | `pow(1-q, 1/c)` / `exp(log(q)/c)` underflow, `ndtri(0) = -inf` (pattern 4/8) | inf for 67.8 (ppf), inf for 52.5 (isf) | loss |
| F3 | isf, ppf | large c (argument of ndtri close to 1), and q -> 1 in isf for c > 1 | rounding of p ~ 1 amplified by `ndtri` near 1; p == 1.0 gives -inf (pattern 8) | -inf for -8.31; 4.8e12 ulp | loss |
| F4 | pdf | c < 1, x > ~37.5 | `ndtr(-x)` underflows, `0**(c-1) = inf`, `phi(x) = 0` gives nan (pattern 3/4) | nan for 9.95e-6 | loss |
| F5 | logpdf, logsf | \|x\| > 1.34e154 (logpdf, any c); x > ~1.9e154 with c < 1 (logsf, and logpdf nan) | `x**2` overflows in `_norm_logpdf`; `log_ndtr(-x) = -inf` before `* c` (pattern 3) | logpdf -inf / nan for -5e306; logsf -inf | loss |
| F6 | cdf, logsf, pdf | huge c (>= ~1e10), x < ~-37.5 where c*Phi(x) is normal | `log_ndtr(-x)` = -Phi(x) underflows before `* c`; `phi(x)` subnormal in pdf (pattern 4) | cdf 0.0 for 2.9e-306; pdf 7.9e12 ulp | loss |
| F7 | logpdf | small c, x >> 1 | `-x**2/2` and `(c-1)*log_ndtr(-x)` cancel (pattern 5) | 2.0e8 ulp (c = 1e-8) | severe |
| F8 | pdf | large c, left tail x in about [-9, -3] | `ndtr(-x)**(c-1)` with `ndtr(-x)` rounded near 1 (pattern 6) | 2.5e5 ulp (c = 1e6) | moderate |
| F9 | pdf | small c, right tail 5 < x < 37.5 (just before F4) | far-tail error of `sc.ndtr(-x)` passed through `**(c-1)` (shared) | 508 ulp | minor |
| F10 | sf | large c (measured 50), x near 0 | `exp(c*log_ndtr(-x))` turns the rounding of a large logsf into relative error | 50.2 ulp | minor |
| F11 | logpdf | near a zero of logpdf (e.g. c = 50, x = -2) | `log(c)`, `_norm_logpdf`, `(c-1)*log_ndtr(-x)` cancel (pattern 5) | 20.4 ulp | minor |

### F1: ppf rounds 1 - q

- **Code:** `_ppf`, line 9022: `-_norm_ppf(pow(1.0 - q, 1.0 / c))`
- **Mechanism:** `1.0 - q` has an absolute rounding error of up to 1.1e-16, i.e.
  a relative error of ~1.1e-16/q in the small quantity q (pattern 8). For q < 1.1e-16
  it is exactly 1.0, `pow(1, 1/c) = 1`, `ndtri(1) = inf`, and the result is
  `-inf`. The conditioning with respect to q is tiny here (~1/x**2, about 0.015), so
  the whole error is the formula's. The error is independent of c, and every parameter set fails.
- **Regime:** q < ~1e-5 (moderate at 1e-5, severe below ~1e-8), loss for q < 1.1e-16.
  For c = 1e-8 and c = 1e6 the error compounds with F2/F3.
- **Reproducer:** `python accuracy.py refs/powernorm_ref.py --dps 100 --at ppf 1e-17 4.445365225459078`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-17, 4.445365225459078)` | -inf | -8.665405997867559 | inf | 0.0131 | loss |
  | `ppf(1e-15, 4.445365225459078)` | -8.125890664701908 | -8.124312079829519 | 8.8e11 | 0.0149 | severe |
  | `ppf(2e-16, 1e-08)` | -5.472356947851237 | -5.4908517538703485 | 1.52e13 | 0.0322 | severe |
  | `ppf(1e-08, 1e-08)` | 0.3374749735500885 | 0.33747496864506094 | 6.5e7 | 2.89 | severe |
  | `ppf(1e-08, 4.445365225459078)` | -5.8647142577911735 | -5.864714261009843 | 2.47e6 | 0.0283 | severe |
  | `ppf(1e-05, 0.001)` | -2.3282198720578178 | -2.3282198720561187 | 3.29e3 | 0.16 | moderate |

- **Fix direction:** use `log1p(-q)` and stay in log space: `-sc.ndtri_exp(np.log1p(-q)/c)` (in a
  plain check this gives `ppf(1e-20, 4.445...)` to 0.85 ulp and `ppf(1e-15, 4.445...)` to 0.98 ulp).

### F2: the power 1/c underflows for small c

- **Code:** `_ppf`, line 9022: `pow(1.0 - q, 1.0 / c)`; `_isf`, line 9031: `np.exp(np.log(q) / c)`
- **Mechanism:** the argument passed to `ndtri` is Phi(-x) itself, the normal tail probability at
  the answer. For small c it drops below the double range although x is a
  modest number: `isf(1e-300, 0.5)` needs Phi(-x) = 1e-600, and `ppf(0.9, 1e-3)` needs
  0.1**1000. The power/exp underflows to 0 (or to a subnormal with few bits),
  `ndtri(0) = -inf`, and the result is `+inf` (patterns 4 and 8).
- **Regime:** `log(q)/c < -745` (isf) or `log1p(-q)/c < -745` (ppf). Examples: c = 0.5: isf for
  q < ~1e-162; c = 1e-3: isf for q < ~0.47, ppf for q > ~0.53; c = 1e-8: isf at every
  measured q, ppf for q >= 1e-5. Results are subnormal-limited (inaccurate) just before that.
- **Reproducer:** `python accuracy.py refs/powernorm_ref.py --dps 100 --at isf 1e-300 0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1e-300, 0.5)` | inf | 52.47230638850346 | inf | 0.000726 | loss |
  | `isf(1e-20, 0.001)` | inf | 303.4635650253095 | inf | 0.0109 | loss |
  | `isf(1e-300, 1e-08)` | inf | 371692.21884800494 | inf | 0.000724 | loss |
  | `ppf(0.9, 0.001)` | inf | 67.78568559660262 | inf | 1.96 | loss |
  | `ppf(0.01, 1e-08)` | inf | 1417.7626102954132 | inf | 0.503 | loss |

- **Fix direction:** `ndtri_exp` takes the log directly: `isf = -sc.ndtri_exp(np.log(q)/c)`,
  `ppf = -sc.ndtri_exp(np.log1p(-q)/c)` (in a plain check: `isf(1e-300, 0.5)` exact, `ppf(0.9, 1e-3)` exact;
  see "Suspected" for `ndtri_exp` at very negative arguments).

### F3: the argument of ndtri is close to 1 (large c, and isf for q near 1)

- **Code:** lines 9022 and 9031 as in F2.
- **Mechanism:** for large c, `(1-q)**(1/c)` and `q**(1/c)` are 1 - O(|log q|/c). The
  double next to 1 carries an absolute error of ~1.1e-16, i.e. a relative error of
  1.1e-16 / (1 - p) in the small tail probability 1 - p that `ndtri` actually
  resolves (pattern 8). When p rounds to 1.0, `ndtri(1) = inf` and isf returns `-inf`.
  In isf the same happens for q -> 1 with any c > 1 (`log(q) ~ -(1-q)`, divided by c).
- **Regime:** c = 1e6: every q (moderate in the bulk, severe for q > 0.98 in isf and q < 1e-5 in ppf),
  loss for isf q > 1 - 5e-11. c = 1e4: up to 811 ulp in the bulk, severe for q > 1 - 7e-6 (isf),
  loss for q > 1 - 4e-13. c = 50: severe from q = 1 - 6.6e-7 (isf), loss at 1 - 3.3e-16. Beyond the
  (large) conditioning at q -> 1 by 1e4 to 1e5 times.
- **Reproducer:** `python accuracy.py refs/powernorm_ref.py --dps 100 --at isf 0.9999999999522942 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.9999999999522942, 1e6)` | -inf | -8.310360612195343 | inf | 3.0e8 | loss |
  | `isf(0.9999999994820525, 1e6)` | -8.014015948775546 | -8.022529521751723 | 4.8e12 | 3.0e7 | severe |
  | `isf(0.9999993371296838, 50)` | -5.563018206747586 | -5.563018207995352 | 1.0e6 | 4.73e4 | severe |
  | `isf(0.99999999999999967, 50)` | -inf | -8.540859813221227 | inf | 4.06e13 | loss |
  | `ppf(0.01, 1e6)` | -5.611132555482641 | -5.6111325551870515 | 2.4e5 | 0.031 | moderate |
  | `ppf(0.5, 1e6)` = `isf(0.5, 1e6)` | -4.826965122556483 | -4.826965122543787 | 1.18e4 | 0.0595 | moderate |
  | `ppf(1e-08, 1e6)` | -7.650730905155642 | -7.650628092292365 | 6.05e10 | 0.0168 | severe |

- **Fix direction:** same as F2: `ndtri_exp(y)` is accurate for y close to 0, where it works with
  `-expm1(y)` internally (plain check: `isf(0.9999999994820525, 1e6)` and `isf(0.99999999999999967, 1e6)` exact, 0 ulp).

### F4: pdf is nan/inf in the right tail for c < 1

- **Code:** `_pdf`, line 9013: `c*_norm_pdf(x) * (_norm_cdf(-x)**(c-1.0))`
- **Mechanism:** `ndtr(-x)` becomes subnormal for x > ~37.5 and 0 for x > ~38.5. With
  c - 1 < 0 the power overflows to inf (`0**negative = inf`, divide-by-zero warning). `exp(-x**2/2)`
  underflows to 0 for x > 38.6, so the product is inf (37.5 < x < 38.6) or nan (x > 38.6). The true
  pdf ~ c*phi(x)**c * x**(1-c) is an ordinary number for small c (pattern 3/4).
- **Regime:** 0 < c < 1, x > ~37.5, as long as the true pdf is normal (for c = 1e-8 up to x ~ 4e5;
  the self-check NOTE at the bulk point `pdf(4590.43, 1e-8)` is this finding).
- **Reproducer:** `python accuracy.py refs/powernorm_ref.py --at pdf 1000 1e-8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1000.0, 1e-08)` | nan | 9.950133963265067e-06 | inf | 0.99 | loss |
  | `pdf(40.0, 0.001)` | nan | 0.017901688509460535 | inf | 0.602 | loss |
  | `pdf(37.6, 0.001)` | inf | 0.018472506450969425 | inf | 0.416 | loss |
  | `pdf(38.0, 0.5)` | inf | 3.2296762361052196e-157 | inf | 722 | loss |
  | `pdf(100000.0, 1e-08)` | nan | 1.9287496083772477e-25 | inf | 99 | loss |

- **Fix direction:** `pdf = exp(logpdf)` with the logpdf from F7 (`log Q = -x**2/2 + log(erfcx(x/sqrt 2)/2)`).

### F5: x**2 and log_ndtr(-x) overflow for |x| > 1.34e154

- **Code:** `_logpdf`, line 9016: `np.log(c) + _norm_logpdf(x) + (c-1)*_norm_logcdf(-x)` with
  `_norm_logpdf` (line 367) `-x**2 / 2.0 - _norm_pdf_logC`; `_logsf`, line 9028: `c * _norm_logcdf(-x)`.
- **Mechanism:** `x**2` overflows for |x| > 1.34e154 although `-x**2/2` is finite up to 1.9e154
  (INDEX item 2, shared helper). For x > 0 the true logpdf is ~ -c*x**2/2, finite up to
  x ~ 1.9e154/sqrt(c) for c < 1. In that regime `log_ndtr(-x)` (correctly) returns -inf for
  x > ~1.9e154, so `-inf + (c-1)*(-inf)` = nan in logpdf, and `c * (-inf)` = -inf in logsf (pattern 3).
- **Regime:** logpdf: |x| in (1.34e154, 1.9e154) for every c (x < 0 side), and x in
  (1.34e154, 1.9e154/sqrt(c)) for c < 1. logsf: x in (1.9e154, 1.9e154/sqrt(c)) for c < 1.
- **Reproducer:** `python accuracy.py refs/powernorm_ref.py --at logpdf 1e155 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e+155, 0.001)` | nan | -5e+306 | inf | 2 | loss |
  | `logsf(1e+155, 0.001)` | -inf | -5e+306 | inf | 2 | loss |
  | `logpdf(1.4e+154, 0.5)` | -inf | -4.9e+307 | inf | 2 | loss |
  | `logpdf(-1.4e+154, 4.445365225459078)` | -inf | -9.8e+307 | inf | 2 | loss |

- **Fix direction:** `-(0.5*x)*x` in the helper; in logpdf/logsf for x > 0 use
  `log Q(x) = -(0.5*x)*x + log(erfcx(x/sqrt 2)/2)` and multiply `c` into `x` first: `-(0.5*c*x)*x`.

### F6: huge c, far left tail: log_ndtr(-x) underflows before it is multiplied by c

- **Code:** `_logsf` line 9028 `c * _norm_logcdf(-x)`; `_cdf` line 9019 `-sc.expm1(self._logsf(x, c))`;
  `_pdf` line 9013 (`_norm_pdf(x)` subnormal).
- **Mechanism:** for x < -37.5, `log_ndtr(-x) = log1p(-Phi(x)) ~ -Phi(x)` is subnormal or 0. A
  huge c lifts `c*Phi(x)` back into the normal range, but the bits are already gone: cdf and logsf
  return 0. In pdf `exp(-x**2/2)` is subnormal (~1.5e-322 at x = -38.5, a few bits) and the factor c makes the
  result normal (pattern 4). Same as powerlognorm F6.
- **Regime:** c >~ 1e10 (so that c*Phi(x) > 2.2e-308 where Phi(x) < 2.2e-308), x < ~-37.5. Not in
  PARAMS; measured with `--at` only.
- **Reproducer:** `python accuracy.py refs/powernorm_ref.py --at cdf -38.0 1e10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-38.0, 1e10)` | 0.0 | 2.8854283600687844e-306 | inf | 1.44e3 | loss |
  | `cdf(-38.5, 1e20)` | 0.0 | 1.4081824631705175e-304 | inf | 1.48e3 | loss |
  | `logsf(-38.5, 1e20)` | -0.0 | -1.4081824631705175e-304 | inf | 1.48e3 | loss |
  | `pdf(-38.5, 1e20)` | 5.434722104253712e-303 | 5.4251551813365905e-303 | 7.9e12 | 1.48e3 | severe |

- **Fix direction:** for x < -37 use `cdf = exp(log(c) + log_ndtr(x))` (leading term; the relative correction
  is ~(c-1)Phi(x)/2) and `pdf = exp(logpdf)`.

### F7: logpdf cancels for small c in the right tail

- **Code:** `_logpdf`, line 9016.
- **Mechanism:** for x >> 1, `_norm_logpdf(x) ~ -x**2/2` and
  `(c-1)*log_ndtr(-x) ~ +(1-c)*x**2/2`; the sum is ~ -c*x**2/2, so the relative error is about
  1/c times the rounding of the large terms (pattern 5). For c = 1e-3 the loss is ~1e3 ulp, for c = 1e-8 ~1e8.
- **Regime:** c < 1 (noticeable for c < ~1e-2), x > ~5; grows with 1/c, not with x.
- **Reproducer:** `python accuracy.py refs/powernorm_ref.py --at logpdf 1e5 1e-8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(100000.0, 1e-08)` | -56.90775489807129 | -56.90775540320078 | 4.0e7 | 1.74 | severe |
  | `logpdf(1e+140, 1e-08)` | -4.99999978169636e+271 | -5e+271 | 1.97e8 | 2 | severe |
  | `logpdf(10000000000.0, 1e-08)` | -499999997952.0 | -499999999995.39484 | 1.8e7 | 2 | severe |
  | `logpdf(1e+100, 0.001)` | -4.999999999998013e+196 | -5.000000000000001e+196 | 1.79e3 | 2 | moderate |
  | `logpdf(1000.0, 1e-08)` | -11.517924543237314 | -11.517924543239676 | 924 | 0.086 | minor |

- **Fix direction:** for x > 0 write `log_ndtr(-x) = -x**2/2 + log(erfcx(x/sqrt 2)/2)`; then
  `logpdf = log(c) - c*x**2/2 - log(sqrt(2 pi)) + (c-1)*log(erfcx(x/sqrt 2)/2)` has no cancellation
  (plain check: `logpdf(1e5, 1e-8)` 0 ulp).

### F8: pdf left tail, large c: rounding of ndtr(-x) near 1 raised to c-1

- **Code:** `_pdf`, line 9013: `_norm_cdf(-x)**(c-1.0)`
- **Mechanism:** for x < 0, `ndtr(-x) = 1 - Phi(x)` is rounded with an absolute error of up to
  1.1e-16. Raising it to the power c - 1 multiplies that relative error by c - 1 (pattern 6). The error is
  ~c/4 ulp, and it does not change much with x until `ndtr(-x)` rounds to exactly 1.0 (then only the
  neglected factor (1-Phi(x))**(c-1) is missing, which is ~1 for c*Phi(x) < eps). logpdf is not affected
  (it uses `log_ndtr`).
- **Regime:** c >~ 1e3, x in about [-9, -3] (where c*Phi(x) is not negligible). Also visible at c = 50
  near x = 0 (12.8 ulp).
- **Reproducer:** `python accuracy.py refs/powernorm_ref.py --at pdf -6 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(-6.0, 1e6)` | 0.006069891420544109 | 0.0060698914208803 | 2.5e5 | 36 | moderate |
  | `pdf(-5.0, 1e6)` | 1.1161895513352105 | 1.1161895513847035 | 2.0e5 | 17.6 | moderate |
  | `pdf(-6.0, 1e4)` | 6.0758229122003986e-05 | 6.075822912203763e-05 | 2.49e3 | 36 | moderate |
  | `pdf(-1e-10, 50)` | 3.54331925543816e-14 | 3.54331925543817e-14 | 12.8 | 3.91e-09 | minor |

- **Fix direction:** `exp((c-1)*log_ndtr(-x))`, or simply `pdf = exp(logpdf)`.

### F9: pdf for small c before the F4 cliff: ndtr's own far-tail error

- **Code:** `_pdf`, line 9013.
- **Mechanism:** for c ~ 0, the pdf is `c*phi(x)/ndtr(-x)` up to a factor `ndtr(-x)**c`. In exact
  arithmetic phi and Q nearly cancel (cond 0.4), but the relative error of `sc.ndtr(-x)` passes
  through unchanged. Measured `sc.ndtr(-x)` alone: 128 / 256 / 512 ulp at x = 20 / 30 / 37, the
  same as the pdf errors below (`exp(-x**2/2)` is within 0.2 ulp there). This is the shared
  `sc.ndtr` far-tail error (lognorm F6, exponnorm, gibrat F3).
- **Regime:** c < 1 (measured 1e-3, 1e-8), 5 < x < 37.5.
- **Reproducer:** `python accuracy.py refs/powernorm_ref.py --at pdf 37 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(37.0, 0.001)` | 0.01858986348252721 | 0.018589863482525115 | 508 | 0.371 | minor |
  | `pdf(30.0, 0.001)` | 0.019067478341734445 | 0.019067478341733366 | 255 | 0.0968 | minor |
  | `pdf(37.0, 1e-08)` | 3.7026732559738796e-07 | 3.702673255973587e-07 | 356 | 0.999 | minor |

- **Fix direction:** the F7 logpdf, which uses `erfcx` (accurate) instead of the ratio phi/ndtr.

### F10: sf = exp(logsf) magnifies the rounding of a large logsf

- **Code:** `_sf`, line 9025: `np.exp(self._logsf(x, c))`
- **Mechanism:** `c*log_ndtr(-x)` has a relative rounding error of ~1 ulp; `exp` turns it into
  a relative error of |logsf| ulp in sf (|logsf| = 34.7 at x = 0, c = 50). Near x = 0 the
  conditioning with respect to x is ~0, so this is beyond conditioning, though below `cond (shapes)` (34.7).
- **Regime:** large c with x near 0, more generally wherever |logsf| is large but the cond in x is
  small. Bounded by ~708 ulp (sf underflows beyond).
- **Reproducer:** `python accuracy.py refs/powernorm_ref.py --at sf -1e-10 50`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(-1e-10, 50)` | 8.881784232434541e-16 | 8.881784232434445e-16 | 49 | 3.99e-09 | minor |
  | `sf(0.3, 50)` | 1.282929384082448e-21 | 1.2829293840824338e-21 | 50.2 | 15 | minor |

- **Fix direction:** low priority; `ndtr(-x)**c` has a similar error (~c/2 ulp from rounding of ndtr). It would
  need a more accurate logsf (e.g. compensated product), not just a different formula.

### F11: logpdf near its zeros

- **Code:** `_logpdf`, line 9016.
- **Mechanism:** the three terms `log(c)` (3.9), `_norm_logpdf(x)` (-2.9) and
  `(c-1)*log_ndtr(-x)` (-1.13) cancel to -0.135 (pattern 5). The relative error grows without bound as
  logpdf approaches a zero (the conditioning grows too).
- **Regime:** logpdf near 0; any c > ~2.5 has such zeros.
- **Reproducer:** `python accuracy.py refs/powernorm_ref.py --at logpdf -2 50`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(-2.0, 50)` | -0.134548084895737 | -0.1345480848957376 | 20.4 | 10.5 | minor |

- **Fix direction:** inherent to the log form; acceptable.

## RuntimeWarnings

- pdf: `divide by zero encountered in power` / `invalid value encountered in multiply` at the F4
  points (inaccurate result: inf/nan). `underflow encountered in exp/power/multiply` in the far
  tails, where the true pdf is mostly not normal either (accurate), and at F6 (inaccurate).
  `overflow encountered in square` for |x| > 1.34e154.
- logpdf: `overflow encountered in square` (F5, inaccurate where the true value is finite, harmless
  otherwise), `invalid value encountered in add` at the F5 nan points.
- cdf, logsf, sf: `overflow encountered in multiply` in 4 parameter sets at |x| >= ~1e154, where
  the true result is 0, 1 or not representable (the returned value is right), except logsf at F5.
- ppf: `underflow encountered in power` (F2, inf result). isf: `underflow encountered in exp` (F2, inf result).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid, all 8 parameter sets (left tail via `-expm1(logsf)`, right tail -> 1) | 257 ulp, within cond (901): `log_ndtr(x > 0)` carries `ndtr(-x)`'s far-tail error (shared, see lognorm) |
| logsf | full grid, all 8 parameter sets, except F5 | 257 ulp, within cond (same `log_ndtr` error) |
| sf | full grid, all parameter sets, except F10 | 739 ulp in the right tail, within cond (1.37e3) |
| pdf | bulk and tails for c = 4.445, 0.5, 1, 50 apart from F4/F8 | 403 ulp (c = 50), within cond |
| logpdf | left tail all c; right tail c >= 0.5 apart from F5 | within cond |
| ppf | q in [1e-3, 1 - 1e-15] for 0.5 <= c <= 50 | within cond (F1 below q ~1e-5) |
| isf | small q, c >= 1 (c = 1 exact: 1.06 ulp) | within cond apart from F3 for large c |

## Suspected, not confirmed

- `sc.ndtri_exp`, the proposed fix for F1-F3, gave 2.5e3 ulp at `y = log(1e-300)/1e-3 = -6.9e5`
  (`isf(1e-300, 1e-3)` candidate 1175.3872039518037 vs reference 1175.3872039511482, plain
  Python check, not via the harness; the argument rounding accounts for only ~0.5 ulp). If confirmed,
  that is a `scipy.special` issue for very negative arguments, and it limits the fix for tiny c.
- pdf for c slightly above 1 in the right tail, where `ndtr(-x)` is subnormal (x in 37.5-38.6):
  the subnormal bits are scaled by c - 1 only, so the error is expected to be small. Not measured (no c in (1, 1.1)).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | median switch: `log(cdf)` below the median, `log1p(-sf)` above | x < median: `log(-expm1(logsf))`, and where logsf underflows (x < ~-37) `log(c) + log_ndtr(x)` (relative correction ~(c-1)Phi(x)/2) | left tail once cdf ~ c*Phi(x) underflows: `logcdf(-40, 4.445365225459078)` = -inf, reference -803.1165799827057 (loss, cond 1.99) |

## Shared code

- `_norm_logpdf` (line 366): `-x**2 / 2.0` overflows for |x| > 1.34e154 (F5); INDEX item 2,
  also powerlognorm F4 and lognorm.
- `sc.ndtr` far tail (|z| in ~5-37.5): its relative error (128 / 256 / 512 ulp at z = 20 / 30 / 37) shows up
  directly in pdf for small c (F9). `sc.log_ndtr(z > 0)` carries the same error into cdf/logsf in the left
  tail, there within conditioning. Same issue as lognorm F6, exponnorm, gibrat F3, powerlognorm F8.
- powerlognorm uses the same formulas with z = log(x)/s; its findings F1, F2, F3, F4, F5, F6 and F7 are the
  same mechanisms as powernorm F1, F2, F3, F5, F5, F6 and F7, and one fix pattern serves both.
- Possible `sc.ndtri_exp` inaccuracy at very negative arguments (see "Suspected").

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(1e-17, 4.445365225459078)` | -inf | -8.6654059978675587 | verified | Reference by Newton (`mp.findroot`) on `c*log(erfc(x/sqrt2)/2) = log1p(-q)`, then checked that cdf(x_ref) = q (rel. 1e-185). All 6 table rows match: 8.75e11, 1.52e13, 6.55e7, 2.47e6, 3.29e3 ulp. Mechanism matches line 9022 |
| F2 | `isf(1e-300, 0.5)` | inf | 52.472306388503462 | verified | All 5 rows give inf against finite references (303.46, 371692.2, 67.79, 1417.76). Underflow of `pow`/`exp` at lines 9022/9031 confirmed |
| F3 | `isf(0.9999999999522942, 1e6)` | -inf | -8.3103606121953429 | verified | Other rows also match: 4.78e12, 1.01e6, inf, 2.37e5, 1.18e4, 6.05e10 ulp |
| F4 | `pdf(1000.0, 1e-08)` | nan | 9.9501339632650659e-6 | verified | Reference `exp(logpdf)` cross-checked with `-d/dx sf` (`mp.diff`) to 1e-196. All 5 rows nan/inf. Mechanism matches line 9013 |
| F5 | `logpdf(1e155, 0.001)` | nan | -5.0000000000000002e+306 | verified | For \|x\| > 1e6 the reference uses the asymptotic Mills-ratio series. logsf(1e155, 0.001) = -inf; logpdf(+-1.4e154) = -inf, as reported. Confirmed that `x**2` overflows and `log_ndtr(-1e155) = -inf` (line 367, line 9028) |
| F6 | `cdf(-38.0, 1e10)` | 0.0 | 2.8854283600687843e-306 | verified | Cross-checked with the binomial series `c*Phi - c(c-1)Phi**2/2` (20 digits) and an `mp.quad` of the pdf (agrees to 6e-9, the quad tolerance). cdf(-38.5, 1e20) and logsf give 0; pdf(-38.5, 1e20) is 7.94e12 ulp off (severe, as stated) |
| F7 | `logpdf(1e5, 1e-08)` | -56.90775489807129 | -56.907755403200778 | verified | 4.0e7 ulp; also 1.97e8 (1e140), 1.84e7 (1e10), 1.79e3 (c = 1e-3). Cancellation at line 9016 |
| F8 | `pdf(-6.0, 1e6)` | 0.006069891420544109 | 0.0060698914208803003 | verified | 2.49e5 ulp; pdf(-5, 1e6) 2.0e5, pdf(-6, 1e4) 2.49e3. The reference was checked against `mp.diff` of sf. Mechanism matches line 9013 |

Verified by dist-accuracy-verifier: 8 of 8 verified.
