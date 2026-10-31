# t: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `t_gen`, lines 8022-8131
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf (cdf/sf via `sc.stdtr`, ppf/isf via `sc.stdtrit`, both Boost `students_t`; the df = inf branch of pdf/logpdf via `norm._pdf`/`norm._logpdf`)
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** norm (df = inf branch, shared helper `_norm_logpdf`); cauchy is t with df = 1 but has its own implementation; helpers: `_lazyselect` (only in `_stats`); special functions: sc.stdtr, sc.stdtrit, sc.poch (pdf/logpdf); sc.beta, sc.digamma (entropy only, not measured)
- **Reference:** `refs/t_ref.py`, 400 digits; self-checks: PASS except the harness pdf-integral check for df = 0.001, which fails only because `mp.quad` in x cannot resolve the interval [-2.1e152, 0] (bulk points taken from SciPy's capped ppf, see F4). Integrating in log|x| instead (`scratch/t/integral_check.py`) gives 0.14937504325598016183 for both the integral and the cdf difference (relative difference 2e-31), so the reference is consistent
- **Parameter sets:** (2.7433514990818093), (1), (0.001), (0.1), (30), (1e6), (1e15), (1e100), (inf); extra `--at` points at df = 2.5, 2.9, 3
- **Worst finding:** loss in pdf, logpdf, cdf, sf, ppf, isf

## Summary

In the bulk, for typical df, all six methods are accurate: cdf/sf within
conditioning (max 41 ulp at cond 101), logpdf ≤ 29 ulp. The failures are
in the extremes, and most are inside the Boost `students_t` code behind
`sc.stdtr`/`sc.stdtrit`. `stdtr` returns exactly 0/1 for every
|x| > 1.34e154 (`t*t` overflows). For df ≲ 2 this is wrong: for example
cdf(-1e300, 0.001) = 0 where the truth is 0.25. `stdtrit` caps |x| at
sqrt(df/DBL_MIN) for small df. For generic df it returns a quantile off by
a factor of about 2 once |x| ≳ 1e50, and +inf (wrong sign) for even
smaller q. For df == 1 exactly, `stdtr` near x = 0 is off by up to 1.4e7
ulp. In SciPy's own code, logpdf overflows in `x * x/df` (line 8070) and
returns -inf for |x| > 1.34e154 at every df (and earlier for small df).
pdf = exp(logpdf) does the same where the pdf is still normal, and
amplifies logpdf's rounding to up to 554 ulp.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | \|x\| > 1.34e154, df ≲ 2 (where the tail value is normal) | `t*t` overflows inside Boost `students_t` cdf (`sc.stdtr`) (pattern 3) | 0.0 for 0.2496; 1.0 for 0.6508 | loss |
| F2 | logpdf, pdf | \|x\| > 1.34e154 (any df); \|x\| > sqrt(1.8e308·df) for small df | `x * x/df` overflows, `log1p(inf)` (pattern 3), line 8070 | -inf for -1327.76; pdf 0.0 for 2.5e-304 | loss |
| F3 | logpdf | df = inf, 1.34e154 < \|x\| < 1.9e154 | `-x**2/2` in the shared helper `_norm_logpdf` overflows (pattern 3) | -inf for -9.8e307 | loss |
| F4 | ppf, isf | small df: true \|x\| > sqrt(df/DBL_MIN) (df = 0.001: 2.1e152) | `sc.stdtrit` caps the root (z = df/(df+x²) cannot go below DBL_MIN) (patterns 4, 10) | -2.1e152 for -5.9e234 | loss |
| F5 | ppf, isf | generic df (2.74, 3, 5), \|x\| ≳ 1e50 (q ≲ 1e-150 at df = 2.74) | inside Boost `students_t` quantile; mechanism unclear | +inf for -1.78e112; factor 2.15 off | loss |
| F6 | cdf, sf | df == 1 exactly, \|x\| ≲ 1e-2 | Boost df = 1 path uses z = df/(df+x²) rounded to a double (patterns 1, 10) | 1.4e7 ulp | severe |
| F7 | ppf, isf | 2 < df < 3, q ≈ 0.78-0.80 (\|x\| ≈ 0.9) | inside Boost `students_t` quantile; mechanism unclear | 2.14e3 ulp | moderate |
| F8 | ppf, isf | df = 0.001, q in [0.36, 0.43] | Boost `ibeta_inv` accuracy on a tiny z; mechanism unclear | 1.85e3 ulp (cond 1e3) | moderate |
| F9 | pdf | \|logpdf\| large (far tails), any df | pdf = exp(logpdf) amplifies the absolute error of logpdf (line 8063) | 554 ulp | minor |
| F10 | logpdf, pdf | x near 0, df = 30 ... 1e100 | normalizer `log(poch(df/2, 1/2)) - 0.5*(log df + log pi)` cancels (pattern 5) | 28.8 ulp | minor |

### F1: stdtr returns 0 / 1 for |x| > 1.34e154

- **Code:** `_cdf`, line 8078: `sc.stdtr(df, x)`; `_sf`, line 8081: `sc.stdtr(df, -x)`.
- **Mechanism:** Boost `students_t` cdf forms `x2 = t*t` and then
  `z = df/(df + x2)`. For |t| > sqrt(DBL_MAX) = 1.34e154, `x2 = inf`,
  `z = 0`, and `ibeta(df/2, 1/2, 0) = 0`. The result is exactly 0 in the near
  tail and exactly 1 in the far tail, with no warning. The cliff sits exactly
  between x = 1.3e154 (correct) and 1.4e154 (0) for every df (pattern 3, in
  `scipy.special`).
- **Regime:** |x| > 1.34e154. The true tail value is ~ C·|x|^-df, so it is a
  normal double there only for df ≲ 2. For larger df the 0 is correct
  (subnormal truth). For df ≲ 0.1 the "1.0" side is also measurably wrong.
- **Reproducer:** `python accuracy.py refs/t_ref.py --at cdf -1e300 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-1e+300, 0.001)` | 0.0 | 0.2495566532717524 | inf | 0.001 | loss |
  | `cdf(1.4e+154, 0.001)` | 1.0 | 0.6508411738353554 | 2.4e+15 | 0.000536 | severe |
  | `sf(-1.4e+154, 0.001)` | 1.0 | 0.6508411738353554 | 2.4e+15 | 0.000536 | severe |
  | `cdf(-1e+200, 0.1)` | 0.0 | 4.173803137173207e-21 | inf | 0.1 | loss |
  | `cdf(-1.4e+154, 1.0)` | 0.0 | 2.2736420441699336e-155 | inf | 1 | loss |
  | `sf(1.351287904186225e+154, 1.0)` | 0.0 | 2.3556037554815815e-155 | inf | 1 | loss |

  Harness totals: 33 loss points in each of cdf and sf (df = 0.001, 0.1, 1),
  plus 16 severe (the "1.0" side at df = 0.001).
- **Fix direction:** for |x| > sqrt(df)·1e100 (say), use the
  complement-free tail: `z = (df/x)/x` or `0.5*betainc(df/2, 0.5, exp(log(df) - 2*log|x|))`
  in SciPy, or fix Boost to form `df/(df + t*t)` as `(df/t)/t/(1 + df/t/t)`.

### F2: logpdf / pdf overflow in `x * x/df`

- **Code:** `_logpdf`, line 8070: `- (df + 1)/2*np.log1p(x * x/df)`; `_pdf`, line 8063: `np.exp(self._logpdf(x, df))`.
- **Mechanism:** `x * x` overflows for |x| > 1.34e154, and for small df the
  quotient `x*x/df` overflows earlier, at |x| > sqrt(1.8e308·df)
  (df = 0.001: 4.2e152). `log1p(inf) = inf`, so logpdf = -inf and pdf = 0
  (pattern 3). logpdf is finite at every df (e.g. -21362 at x = 1e300,
  df = 30). pdf is still a normal double for df ≲ 0.1.
- **Regime:** logpdf: all df, |x| > 1.34e154, or |x| > sqrt(1.8e308·df) if that is
  smaller. pdf: same x, measured loss for df = 0.001 and 0.1.
- **Reproducer:** `python accuracy.py refs/t_ref.py --at logpdf 1.4e154 2.7433514990818093`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.4e+154, 2.7433514990818093)` | -inf | -1327.7643120279508 | inf | 0.00282 | loss |
  | `logpdf(-1e+300, 30.0)` | -inf | -21362.25007575424 | inf | 0.00145 | loss |
  | `pdf(-1e+300, 0.001)` | 0.0 | 2.4955665327175236e-304 | inf | 1 | loss |
  | `pdf(4.273147551863848e+152, 0.001)` | 0.0 | 8.199559386250564e-157 | inf | 1 | loss |

  Harness totals: logpdf 232 loss points (all eight finite df), pdf 66.
- **Fix direction:** for large |x| use `log1p(x*x/df) = 2*log|x| - log(df) + log1p(df/x**2)`
  (switch at e.g. |x| > 1e100·sqrt(df)), or `2*log(|x|/sqrt(df)) + log1p((sqrt(df)/x)**2)`.

### F3: df = inf branch overflows in `_norm_logpdf`

- **Code:** `_logpdf`, line 8073: `norm._logpdf(x)` -> `_norm_logpdf`, line 367: `-x**2 / 2.0 - _norm_pdf_logC`.
- **Mechanism:** `x**2` overflows for |x| > 1.34e154 although `-x**2/2` is
  finite up to |x| = 1.9e154 (pattern 3). This is INDEX item 2 (shared
  helper, also behind `alpha` F1 and `norm.logpdf`).
- **Regime:** df = inf, 1.34e154 < |x| < 1.9e154. Beyond that the true value is below -1.8e308, so -inf is correct.
- **Reproducer:** `python accuracy.py refs/t_ref.py --at logpdf 1.4e154 inf`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.4e+154, inf)` | -inf | -9.8e+307 | inf | 2 | loss |

- **Fix direction:** `-(x/2)*x` or `-0.5*x*x` in `_norm_logpdf` (shared fix).

### F4: stdtrit caps the quantile at sqrt(df/DBL_MIN) for small df

- **Code:** `_ppf`, line 8084: `sc.stdtrit(df, q)`; `_isf`, line 8087: `-sc.stdtrit(df, q)`.
- **Mechanism:** the quantile is obtained from z = df/(df + x²) via
  incomplete-beta inversion, and x = sqrt(df/z - df). z cannot go below
  DBL_MIN = 2.2e-308, so |x| never exceeds sqrt(df/2.2e-308). That is
  2.12e152 for df = 0.001, 2.12e153 for df = 0.1, 4.74e153 for df = 0.5.
  Every q whose quantile lies beyond returns this cap (patterns 4 and 10).
  For df = 0.5, `stdtr(df, stdtrit(df, 1e-100))/1e-100 = 4.7e22`.
- **Regime:** small df and every q whose true |x| exceeds sqrt(df/DBL_MIN).
  At df = 0.001 that is q ≲ 0.35 (and q ≳ 0.65); measurable down to where
  the true quantile passes 1.8e308 (q = 0.29 measured). At df = 0.1 it is
  q ≲ 1e-20 (measured at 1.2e-26 and 1e-20).
- **Reproducer:** `python accuracy.py refs/t_ref.py --at ppf 0.29 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.29, 0.001)` | -2.119960574434296e+152 | -5.9041163510103366e+234 | 4.5e+15 | 1e+03 | loss |
  | `ppf(1e-20, 0.1)` | -2.1199605744342963e+153 | -1.6044257056665295e+196 | 4.5e+15 | 10 | loss |
  | `isf(0.29, 0.001)` | 2.119960574434296e+152 | 5.9041163510103366e+234 | 4.5e+15 | 1e+03 | loss |

  Where the true quantile exceeds 1.8e308 (e.g. `ppf(1e-300, 0.001)`), SciPy
  also returns the finite cap -2.1e152 instead of -inf. That is not measurable
  (non-normal truth), but it is the same defect.
- **Fix direction:** invert in log space for tiny z (`log z = log df - 2 log|x|`),
  e.g. use the tail asymptote `|x| ≈ (df·B(df/2,1/2)·q·…)^(-1/df)·sqrt(df)` and refine with the cdf in log form.

### F5: stdtrit wrong by a factor ~2, then +inf, for very small q at generic df

- **Code:** `_ppf`, line 8084; `_isf`, line 8087.
- **Mechanism:** unclear (inside Boost `students_t` quantile, source not
  shipped with SciPy). Observations: integer df = 1, 2, 4 (closed forms) are
  fine. For df = 2.74, 3, 5, once |x| ≳ 1e50 the returned x satisfies
  `stdtr(df, x)/q ≈ 8` (8.0 at df = 3, 8.1 at df = 2.74) at every such q.
  That is consistent with an unrefined tail approximation that is off by a
  constant factor. For still smaller q the result is +inf. ppf returns
  **+inf for a lower-tail q** (wrong sign), and isf returns -inf. No
  RuntimeWarning is raised.
- **Regime:** df = 2.74: q ≲ 3e-151 (factor 2.15 off), q ≲ 2e-238 (+inf);
  df = 3: q ≲ 1e-200 / 1e-250; df = 5: q = 1e-300 gives inf. Starts where the
  true |x| passes ~1e50 (z = df/(df+x²) ≲ 1e-100).
- **Reproducer:** `python accuracy.py refs/t_ref.py --at ppf 1e-308 2.7433514990818093`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-308, 2.7433514990818093)` | inf | -1.78267036120809e+112 | inf | 0.365 | loss |
  | `isf(1e-308, 2.7433514990818093)` | -inf | 1.78267036120809e+112 | inf | 0.365 | loss |
  | `ppf(5.6234132519034905e-226, 2.7433514990818093)` | -5.686309995983878e+81 | -1.2224659190312126e+82 | 2.4e+15 | 0.365 | severe |
  | `ppf(3.162277660168379e-151, 2.7433514990818093)` | -3.218606662422051e+54 | -6.911042369297216e+54 | 2.4e+15 | 0.365 | severe |

  Harness totals at df = 2.74: 9 loss and 8 severe points in each of ppf and isf.
- **Fix direction:** in SciPy, for tiny q use the tail asymptote in log space
  plus Newton steps on `log stdtr`; report the defect upstream (Boost Math).

### F6: stdtr for df == 1 exactly loses up to 1.4e7 ulp near x = 0

- **Code:** `_cdf`, line 8078; `_sf`, line 8081.
- **Mechanism:** only df == 1.0 exactly is affected; df = 1 ± 1e-7 and
  1 - 1e-15 are at < 1 ulp, so this is a df = 1 special path in
  `stdtr`. The returned deficit 0.5 - F equals asin(sqrt(1 - z))/pi with
  z = 1/(1 + x²) **rounded to a double** before 1 - z is formed. At
  x = 1e-8, 1 - z becomes 2.22e-16 instead of 1e-16, so the deficit is
  4.74e-9 instead of 3.18e-9: ratio 1.49 = sqrt(2.22e-16/1e-16), exactly as
  observed. That is an arcsine formula applied to the rounded z instead of the
  complementary argument x²/(1 + x²) (patterns 1 and 10). Below
  |x| ≈ 1e-8, z rounds to 1 and the result is exactly 0.5. Inferred from
  the error pattern; the Boost source is not shipped with SciPy.
- **Regime:** df == 1, |x| ≲ 1e-2 (11 ulp at 1e-2, 210 at 1e-4, 1.3e5 at 1e-6, 1.4e7 at 1e-8, 2.9e5 at 1e-10).
- **Reproducer:** `python accuracy.py refs/t_ref.py --at cdf -1e-8 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-1e-08, 1.0)` | 0.4999999952568131 | 0.49999999681690116 | 1.4e+07 | 6.37e-09 | severe |
  | `sf(1e-08, 1.0)` | 0.4999999952568131 | 0.49999999681690116 | 1.4e+07 | 6.37e-09 | severe |
  | `cdf(-1e-10, 1.0)` | 0.5 | 0.499999999968169 | 2.9e+05 | 6.37e-11 | moderate |
  | `cdf(1e-06, 1.0)` | 0.5000003183240348 | 0.5000003183098862 | 1.3e+05 | 6.37e-07 | moderate |

- **Fix direction:** in `t._cdf`, route df == 1 to `0.5 + arctan(x)/pi`
  (cauchy's accurate form), or fix the Boost df = 1 path to use x²/(1+x²).

### F7: stdtrit loses up to 2.1e3 ulp for 2 < df < 3 near q = 0.79

- **Code:** `_ppf`, line 8084; `_isf`, line 8087.
- **Mechanism:** unclear (inside Boost `students_t` quantile). A bulk scan
  (`scratch/t/ppf_scan2.py`, errors from `(F(x) - q)/f(x)` at 50 digits)
  gives ≤ 9 ulp for df = 0.3 ... 1.9 and for df ≥ 4.5. It gives 272 ulp at
  df = 2.2, 786 at 2.5, 747 at 2.74 and 2137 at 2.9, always around
  q = 0.78-0.80, but only 2.6 ulp at df = 3.0. This pattern points to an
  initial approximation for non-integer df in (2, 3) whose final
  refinement stops early.
- **Regime:** 2 < df < 3, q ≈ 0.77-0.80 and the mirror q ≈ 0.20-0.23.
- **Reproducer:** `python accuracy.py refs/t_ref.py --at ppf 0.795 2.9`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.795, 2.9)` | 0.9601867773648516 | 0.9601867773653079 | 2.14e+03 | 3.87 | moderate |
  | `isf(0.205, 2.9)` | 0.9601867773648515 | 0.9601867773653078 | 2.14e+03 | 0.998 | moderate |
  | `ppf(0.785, 2.5)` | 0.9373919663112203 | 0.9373919663113839 | 786 | 3.92 | minor |
  | `ppf(0.785, 2.7433514990818093)` | 0.9225624044867646 | 0.9225624044869172 | 745 | 3.87 | minor |
  | `ppf(0.795, 3.0)` | 0.9550723237857264 | 0.9550723237857269 | 2.62 | 3.85 | ok |

  (The harness grid at the example df = 2.74 hit q = 0.78 and 0.22: 431 ulp, minor.)
- **Fix direction:** one Newton step on `stdtr` after `stdtrit` in `t._ppf`
  (cheap, and stdtr is accurate here); report upstream.

### F8: stdtrit at df = 0.001 slightly beyond conditioning in the bulk

- **Code:** `_ppf`, line 8084; `_isf`, line 8087.
- **Mechanism:** for df = 0.001 the bulk quantiles are already huge
  (5e63 at q = 0.43) and z = df/(df+x²) ~ 4e-131. The quantile is
  conditioned at cond = 1/df = 1000. The result is off by 1.85×
  that, i.e. the inverse incomplete beta gives z only to ~3700 ulp. Mechanism
  in the Boost inversion not determined (INDEX item 8 documents
  `ibeta_inv` stopping early in other regimes).
- **Regime:** df = 0.001, q in [0.36, 0.43] (and mirrored), between the F4 cap and q ≈ 0.5.
- **Reproducer:** `python accuracy.py refs/t_ref.py --at ppf 0.42999999999999994 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.42999999999999994, 0.001)` | -5.019924978887992e+63 | -5.019924978885931e+63 | 1.85e+03 | 1e+03 | moderate |

- **Fix direction:** Newton refinement on the log cdf; upstream.

### F9: pdf = exp(logpdf) amplifies logpdf rounding in the tails

- **Code:** `_pdf`, line 8063: `np.exp(self._logpdf(x, df))`.
- **Mechanism:** an absolute error δ in logpdf becomes a relative error δ
  in pdf. logpdf ≈ -660 carries ~1e-13 absolute error, i.e. ~500 ulp in
  pdf. The conditioning of pdf with respect to x is only ~df+1, so this is
  beyond it.
- **Regime:** far tails where |logpdf| is in the hundreds, any df.
- **Reproducer:** `python accuracy.py refs/t_ref.py --at pdf -1e10 30`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(-10000000000.0, 30.0)` | 3.1093603957690025e-288 | 3.10936039576862e-288 | 554 | 31 | minor |
  | `pdf(-1e+150, 0.001)` | 3.525081422567804e-154 | 3.5250814225675316e-154 | 348 | 1 | minor |

  Harness: 268 minor pdf points across the parameter sets.
- **Fix direction:** compute pdf directly as
  `C(df) * (1 + x*x/df)**(-(df+1)/2)` with `C` from `poch` (the power is
  accurate to ~min(df, 745) ulp), or `exp` of a logpdf whose large term is
  split off exactly.

### F10: logpdf normalizer is a difference of logarithms

- **Code:** `_logpdf`, lines 8068-8069: `np.log(sc.poch(0.5 * df, 0.5)) - 0.5 * (np.log(df) + np.log(np.pi))`.
- **Mechanism:** `log(poch(df/2, 1/2)) ≈ 0.5*log(df/2)` and
  `0.5*log(df)` cancel to `-0.5*log(2*pi)` for large df (pattern 5). The loss
  grows like log(df): 11 ulp at df = 30, 29 ulp at df = 1e100. pdf inherits it.
- **Regime:** large df, |x| small (where logpdf ≈ the normalizer).
- **Reproducer:** `python accuracy.py refs/t_ref.py --at logpdf -1e-8 1e100`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(-1e-08, 1e+100)` | -0.9189385332046669 | -0.9189385332046728 | 28.8 | 1.09e-16 | minor |
  | `logpdf(-0.005477225575051661, 30.0)` | -0.9272858253710979 | -0.9272858253710956 | 11.3 | 3.34e-05 | minor |
  | `pdf(-1e-08, 1e+100)` | 0.39894228040143503 | 0.39894228040143265 | 26.9 | 1e-16 | minor |

- **Fix direction:** `log(poch(df/2, 1/2)/sqrt(df/2)) - 0.5*log(2*pi)`, the
  ratio being 1 + O(1/df) (or a `log1p` series for large df).

## RuntimeWarnings

With NumPy's default error state, warnings appear only together with
inaccurate results. These are "overflow encountered in multiply" and
"overflow encountered in divide" from `x * x/df` (F2), and "overflow
encountered in square" from `_norm_logpdf` (F3; also on
`logpdf(1e300, inf)`, where -inf is correct). With all floating-point
warnings enabled, the harness additionally records underflow warnings
(`x*x` underflow for tiny |x|, `exp` underflow in pdf). These are on
accurate results, i.e. spurious. `sc.stdtr`/`sc.stdtrit` emit no warning
in any failure regime (F1, F4-F8 are silent).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf, sf | full x grid, all 9 parameter sets, outside F1/F6 (incl. df = 1e6, 1e15, 1e100 and df = inf -> normal) | 40.8 ulp at cond 101 (df = 1e100, x = ±10), within conditioning |
| logpdf | full grid outside F2/F3/F10; large df (1e6 ... 1e100) right tail: `(df+1)/2*log1p(x*x/df)` shows no pattern-6 amplification | ≤ 29 ulp (F10) |
| pdf | bulk, all parameter sets | 1.85 ulp median; max as in F9/F10 |
| ppf, isf | full q grid for df = 1, 30, 1e6, 1e15, 1e100, inf; df = 2.74 outside F5/F7; df = 0.001/0.1 outside F4/F8 | 17.1 ulp in the tails; near q = 0.5 within conditioning; minor 431 ulp at df = 2.74, q = 0.78 belongs to F7 |

## Suspected, not confirmed

- `_entropy` (not a measured method) uses `sc.beta(df/2, 0.5)` for df < 100;
  INDEX item 13 reports `sc.beta` losing accuracy for large/unbalanced
  shapes. With df < 100 the shapes stay below (50, 0.5), so the effect is
  probably small. Not measured.
- The bracketing in F5 (factor ≈ 8 in the cdf, then inf) suggests a fixed
  prefactor error in an asymptotic tail formula. Without the Boost source
  this could not be pinned to a line.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) (public method switches to log1p(-sf) above the median) | for x < 0: `-log(2) + a*log(z) + 0.5*log1p(-z) - log(a) - betaln(a, 0.5) + log(hyp2f1(a+0.5, 1; a+1; z))` with a = df/2, z = df/(df+x²) formed as `(df/x)/x/(1+df/x/x)` for large \|x\|, i.e. the incomplete-beta prefix in log space | cdf underflows below 2.2e-308: `logcdf(-1e120, 2.7433514990818093)` = -inf vs -758.1446451379171 (loss), cond 0.0036 |
| logsf | log(sf) | logsf(x) = logcdf(-x) with the formula above | `logsf(1e120, 2.7433514990818093)` = -inf vs -758.1446451379171 (loss) |

## Shared code

- **`sc.stdtr` (Boost `students_t` cdf), F1 and F6:** returns exact 0/1 for
  |t| > 1.34e154 at every df. It is wrong near t = 0 for df == 1 exactly.
  `stdtr` is also used by the t-tests in `scipy/stats/_stats_py.py` and
  `_mstats_basic.py` (p-values for tiny df with huge t, and df = 1 with
  tiny t).
- **`sc.stdtrit` (Boost `students_t` quantile), F4, F5, F7, F8:** caps at
  sqrt(df/DBL_MIN) for small df. It is wrong by a factor ~2, then +inf with
  the wrong sign, for |x| ≳ 1e50 at generic df. It loses up to 2.1e3 ulp
  for 2 < df < 3. Likely related to INDEX item 8 (Boost incomplete-beta
  inversion), but F5/F7 also appear where the root is moderate, so the
  quantile driver itself is suspect.
- **`_norm_logpdf` (F3):** INDEX item 2, same overflow as `alpha` F1 and `norm.logpdf`.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable). Related outside SciPy: Boost Trac #4504 (fixed in Boost 1.44, 2010), wrong `students_t` quantiles for 2 < df < 3 - the same df band as F7, possibly a remnant (not checked).

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-1e300, 0.001)` | 0.0 | 0.24955665327175237 | verified | Reference: mp.quad of the docstring pdf in s = log\|x\| (100 and 200 digits agree). Also `cdf(1.4e154, 0.001)` = 1.0 vs 0.65084117383535541 and `cdf(-1e200, 0.1)` = 0 vs 4.1738031371732071e-21. df = 1 points checked with atan(1/a)/pi: 2.2736420441699334e-155 and 2.3556037554815813e-155, SciPy 0. The cliff is confirmed: `stdtr(0.001, -1.3e154)` = 0.349, `stdtr(0.001, -1.4e154)` = 0. The mechanism is in Boost and the source was not read |
| F2 | `logpdf(1.4e154, 2.7433514990818093)` | -inf | -1327.7643120279509 | verified | Reference from loggamma. Also `logpdf(-1e300, 30)` = -inf vs -21362.25007575424, `pdf(-1e300, 0.001)` = 0 vs 2.4955665327175237e-304, and `pdf(4.273147551863848e152, 0.001)` = 0 vs 8.1995593862505642e-157. Mechanism confirmed: line 8070 emits "overflow encountered in multiply/divide" |
| F3 | `logpdf(1.4e154, inf)` | -inf | -9.8e307 | verified | Line 367 `-x**2 / 2.0` emits "overflow encountered in square". Mechanism confirmed |
| F4 | `ppf(0.29, 0.001)` | -2.119960574434296e152 | -5.904116351010336751e234 | verified | Quantile from findroot on the log betainc cdf. It is checked with an independent identity (z^a sqrt(1-z) 2F1(a+1/2, 1; a+1; z)/(a B(a,1/2))/2): the relative residual is 1e-201. Also `ppf(1e-20, 0.1)`: -2.12e153 vs -1.6044257056665295e196. `isf(0.29, 0.001)` mirrors it. The cap equals sqrt(df/DBL_MIN) = 2.1199605744342957e152 exactly |
| F5 | `ppf(1e-308, 2.7433514990818093)` | inf | -1.7826703612080898e112 | verified | Same check as F4, hyp2f1 residual 9e-202. Also `ppf(5.6234e-226, ...)` = -5.686e81 vs -1.2224659190312126e82 and `ppf(3.1623e-151, ...)` = -3.2186e54 vs -6.9110423692972171e54. `stdtr(df, stdtrit(df, q))/q` = 8.14, as stated. The mechanism is unclear (Boost) |
| F6 | `cdf(-1e-8, 1.0)` | 0.4999999952568131 | 0.49999999681690114 | verified (mechanism detail differs) | Reference 1/2 + atan(x)/pi. Also `cdf(1e-6, 1)` 1.27e5 ulp and `cdf(-1e-10, 1)` 2.87e5 ulp, as reported. `stdtr(1+1e-7, -1e-8)` is accurate. The implied argument of asin, w = sin(pi·deficit)², equals 1 - fl(1/(1+x²)) at x = 1e-2 ... 1e-6. At x = 1e-8, fl(1/(1+x²)) = 1 exactly in double, so 1 - z = 0, but the implied w is 2.22e-16 (one eps). So "z = 1/(1+x²) rounded to a double" does not exactly explain the 1e-8 point; the true intermediate differs slightly (inferred, Boost source not read) |
| F7 | `ppf(0.795, 2.9)` | 0.9601867773648516 | 0.96018677736530785 | verified | 2.14e3 ulp. `isf(0.205, 2.9)` also gives 2.14e3 ulp (ref 0.96018677736530772). The mechanism is unclear (Boost) |
| F8 | `ppf(0.42999999999999994, 0.001)` | -5.019924978887992e63 | -5.0199249788859313e63 | verified | 1.85e3 ulp at cond 1e3, as reported. The mechanism is unclear (Boost) |

Verified by dist-accuracy-verifier: 8 of 8 verified.
