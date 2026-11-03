# burr: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `burr_gen`, lines 1145-1264
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf
- **Inherited (not measured):** none
- **Related:** fisk (forwards every method to `burr_gen` with d = 1, so it inherits the findings below); burr12 and mielke are related families with their own code; helpers: none; special functions: sc.expm1, sc.log1p, sc.xlog1py, sc.xlogy (sc.beta only in `_stats`/`_munp`, not reviewed)
- **Reference:** `refs/burr_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (10.5, 4.3), (2, 3), (0.5, 0.001), (100, 0.001), (2, 0.5), (3, 1e4), (100, 1e6), (0.01, 10), (1, 1)
- **Worst finding:** loss in pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf

## Summary

For moderate shapes all eight methods are accurate in the bulk, to a few ulp. Every method fails completely in at least one tail:
- `x**-c` (and the power `(1 + x**-c)**(d+1)`) overflows for small x. This hits pdf, logpdf, cdf and logcdf, and through the cdf also sf and logsf.
- `1 - cdf` cancels in the right tail (sf, logsf).
- `q**(-1/d) - 1` cancels for q near 1 or large d, and overflows for small q or small d (ppf).
- `expm1` overflows for small d (isf).
- logpdf loses every digit when c*d = 1.

Large d costs up to ~1e4 to 4e5 ulp in pdf, cdf and ppf, even in the bulk. Subnormal intermediates cost up to 2.6e6 ulp in pdf and logcdf. All eight findings in the skill's worked example (`burr-example.md`) reproduce with identical numbers. This review adds more overflow modes to F2, a logcdf case for F7, and a minor finding F9 for rounded shape exponents.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | sf, logsf | right tail; partial loss ~1/sf, total once d*x**-c < ~eps | `log1p(-cdf)` with cdf near/rounded to 1 (pattern 1) | sf 0.0, logsf -inf; 8.8e9 ulp before the cliff | loss |
| F2 | pdf, logpdf, cdf, logcdf, sf, logsf | left tail: c*log(x) < -709.8, or (c+1)*log(x) < -709.8, or c*(d+1)*log(x) < -709.8 (pdf) | `x**-c`, `x**(-c-1)` or `(1+x**-c)**(d+1)` overflows (pattern 3); no small-x branch besides x == 0 (pattern 7) | nan, inf, 0.0, -inf; sf 1.0 for 0.99 | loss |
| F3 | logpdf | x < 1, c*d close to 1 | `xlogy(-c-1, x) - xlog1py(d+1, x**-c)` cancel (pattern 5) | 0.0 for -1.5e-20; 8.0e10 ulp | loss |
| F4 | ppf | q near 1 (any shapes); whole range for large d | `q**(-1/d) - 1` cancels (pattern 8) | inf for 9.49e7; 1.2e12 ulp | loss |
| F5 | ppf | -log(q)/d > 709.8 (small q or small d) | `q**(-1/d)` overflows (pattern 8) | 0.0 for 1e-300 | loss |
| F6 | isf | -log1p(-q)/d > 709.8 (small d) | `expm1` overflows (pattern 8) | 0.0 for 2.16e-4 | loss |
| F7 | pdf, logcdf | large c, x where `x**(-c-1)` / `x**-c` is subnormal, large d | subnormal intermediate (pattern 4) | 2.6e6 ulp | severe |
| F8 | pdf, cdf, logsf | large d, bulk | rounding of `1 + x**-c` raised to the power d (pattern 6) | 3.7e5 ulp; cdf 1.0 for 1 - 2.5e-12 | moderate |
| F9 | pdf, ppf, isf | \|log x\| or \|log q\| large; shapes with inexact c+1, d+1, 1/c, 1/d | rounded exponent amplified by the log of the base (patterns 6, 11) | 213 ulp | minor |

### F1: sf and logsf cancel in 1 - cdf

- **Code:** `_logsf`, line 1228: `np.log1p(- (1 + x**(-c))**(-d))`; `_sf`, line 1225: `np.exp(self._logsf(x, c, d))`.
- **Mechanism:** for large x the cdf is 1 - d*x**-c + ..., so `log1p(-cdf)` keeps only the digits of the cdf beyond its leading 1s. The relative error is about eps/sf. Once d*x**-c < eps/2 the cdf rounds to 1 and the result is `log1p(-1) = -inf` (sf = exp(-inf) = 0). This is pattern 1.
- **Regime:** right tail for all shapes. Partial loss ~0.5/sf ulp, total loss once d*x**-c < 1.1e-16. With (0.01, 10) the sf decays so slowly that it stays above 1e-2 over the whole double range. There the loss is only minor (351 ulp at x = 1e290), from the same cancellation.
- **Reproducer:** `python accuracy.py refs/burr_ref.py --at sf 1e8 2 3`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `sf(1e8, 2, 3)` | 0.0 | 2.9999999999999994e-16 | inf | 2 | loss |
  | `logsf(1e8, 2, 3)` | -inf | -35.74274919923662 | inf | 0.056 | loss |
  | `sf(10, 10.5, 4.3)` | 1.35977673565435e-10 | 1.3597793937584532e-10 | 8.8e9 | 10.5 | severe |
  | `logsf(10, 10.5, 4.3)` | -22.718530408629075 | -22.718528453821765 | 3.88e8 | 0.462 | severe |
  | `sf(1e10, 1, 1)` | 1.0000000827403705e-10 | 9.999999999e-11 | 3.7e8 | 1 | severe |
  | `sf(1.5, 100, 1e6)` | 0.0 | 2.4596544265768042e-12 | inf | 100 | loss |

- **Fix direction:** `logsf = log1mexp(logcdf)` with logcdf = `-d*log1p(x**-c)`, and `sf = -expm1(logcdf)`.

### F2: overflow of x**-c and related powers in the left tail

- **Code:** `_pdf` lines 1203-1205: `c_ * d_ * (x_ ** (-c_ - 1.0)) / ((1 + x_ ** (-c_)) ** (d_ + 1.0))`; `_logpdf` lines 1213-1215: `sc.xlog1py(d_+1, x_**(-c_))`; `_cdf` line 1219, `_logcdf` line 1222, `_logsf` line 1228: `x**(-c)`.
- **Mechanism (pattern 3):** three intermediates overflow while the final result is still a normal double.
  - `x**-c` overflows for c*log(x) < -709.8. Then logpdf and logcdf return `-inf` (via `log1p(inf)`), cdf returns `inf**-d = 0` and the pdf returns `inf/inf = nan`. Through the cdf, sf returns exactly 1.0 and logsf returns -0.0. Both are wrong when d is small, because the true cdf (~x**(c*d)) is then still far from 0.
  - Before that point, the pdf denominator `(1 + x**-c)**(d+1)` overflows once c*(d+1)*|log x| > 709.8. With a finite numerator the pdf then returns 0.0. For (2, 3) this starts at x ~ 1e-39, and the true pdf (~6*x**5) is normal down to x ~ 1e-61.
  - The numerator `x**(-c-1)` overflows for (c+1)*|log x| > 709.8 → inf (e.g. c = 0.5, d = 0.001).
  - The pdf/logpdf code has a small-x form (`x**(c*d-1)/(1+x**c)`), but it is used only at x == 0 exactly (pattern 7). The same `apply_where` also returns nan (with "invalid value") where the true pdf underflows to 0, e.g. `pdf(1e-300, 10.5, 4.3)`. The tool does not measure that case.
- **Regime:** x < exp(-709.8/c) for all methods except ppf/isf, e.g. x < 1e-154 for c = 2. For the pdf also x < exp(-709.8/(c*(d+1))) and x < exp(-709.8/(c+1)). For c = 100 the threshold is x < 8.4e-4, i.e. inside the bulk for small d.
- **Reproducer:** `python accuracy.py refs/burr_ref.py --at pdf 1e-160 2 0.6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-200, 2, 3)` | -inf | -2300.7933335248176 | inf | 0.00217 | loss |
  | `pdf(1e-160, 2, 0.6)` | nan | 1.2000000000000196e-32 | inf | 0.2 | loss |
  | `pdf(1e-60, 2, 3)` | 0.0 | 5.999999999999999e-300 | inf | 5 | loss |
  | `pdf(1e-300, 0.5, 0.001)` | inf | 3.5397289219206894e+296 | inf | 1 | loss |
  | `logcdf(1e-200, 2, 3)` | -inf | -2763.1021115928547 | inf | 0.00217 | loss |
  | `cdf(1e-300, 100, 0.001)` | 0.0 | 9.999999999999985e-31 | inf | 0.1 | loss |
  | `cdf(0.00053969845016714859, 100, 0.001)` | 0.0 | 0.47121067121984411 | inf | 0.1 | loss |
  | `sf(1e-70, 100, 0.001)` | 1.0 | 0.9999999 | 4.5e8 | 1e-08 | severe |
  | `sf(1e-20, 100, 0.001)` | 1.0 | 0.99 | 4.55e13 | 0.00101 | severe |
  | `logsf(1e-300, 2, 0.5)` | -0.0 | -1e-300 | inf | 1 | loss |

- **Fix direction:** for x < 1 use `c*d*x**(c*d-1) * (1 + x**c)**(-d-1)` (pdf), `(c*d - 1)*log(x) - (d+1)*log1p(x**c)` (logpdf) and `-d*(log1p(x**c) - c*log(x))` (logcdf).

### F3: logpdf cancels when c*d is close to 1

- **Code:** `_logpdf`, lines 1213-1215: `np.log(c_) + np.log(d_) + sc.xlogy(-c_ - 1, x_) - sc.xlog1py(d_+1, x_**(-c_))`.
- **Mechanism:** for small x the last two terms are each about (c+1)*|log x|. Their difference is `(c*d - 1)*log(x) - (d+1)*log1p(x**c)`, and for c*d = 1 the log(x) parts cancel exactly. With c = 2, d = 0.5, x = 1e-10 both terms round to the same double and the result is 0.0 (pattern 5).
- **Regime:** x < 1 with |c*d - 1| small. The loss is about (c+1)/|c*d - 1| ulp relative to the log(x) part, and total when c*d = 1 exactly. This also covers fisk with c = 1.
- **Reproducer:** `python accuracy.py refs/burr_ref.py --at logpdf 1e-10 2 0.5`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-10, 2, 0.5)` | 0.0 | -1.5e-20 | inf | 2 | loss |
  | `logpdf(1e-100, 2, 0.5)` | 0.0 | -1.5e-200 | inf | 2 | loss |
  | `logpdf(1e-10, 1, 1)` | -1.999964638343954e-10 | -1.9999999999000001e-10 | 8.0e10 | 1 | severe |
  | `logpdf(1e-10, 2, 0.51)` | -0.44071439130263457 | -0.4407143913026298 | 48.8 | 0.0454 | minor |

  cond (shapes) is 2.9e21 and 2.2e11 for the first and third rows. logpdf is extremely sensitive to c and d here, but c and d are exact user inputs, so this does not excuse the error.
- **Fix direction:** for x < 1 use the combined form `(c*d - 1)*log(x) - (d+1)*log1p(x**c)` (the x == 0 branch already has it).

### F4: ppf cancels in q**(-1/d) - 1

- **Code:** `_ppf`, line 1231: `(q**(-1.0/d) - 1)**(-1.0/c)`.
- **Mechanism:** `q**(-1/d)` is close to 1 when (-log q)/d is small. That happens for q near 1 with any shapes, and over the whole q range when d is large. Subtracting 1 then leaves only the trailing digits, and the outer power `-1/c` passes the relative error on. When `q**(-1/d)` rounds to exactly 1, the result is `0**(-1/c) = inf`. This is pattern 8.
- **Regime:** q > ~1 - d*eps gives inf. Partial loss is about d/(-log q) ulp/c, which makes it moderate in the bulk for d = 1e4 to 1e6.
- **Reproducer:** `python accuracy.py refs/burr_ref.py --at ppf 0.99999999 3 1e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999999997, 2, 3)` | inf | 94906265.62425154 | inf | 1.5e15 | loss |
  | `ppf(0.99999999, 3, 1e6)` | 46428.26132663437 | 46415.88818102501 | 1.2e12 | 3.3e7 | severe |
  | `ppf(0.99999999948205254, 3, 1e4)` | 26837.04026853048 | 26826.958040708603 | 1.69e12 | 6.44e8 | severe |
  | `ppf(0.5, 3, 1e6)` | 112.99471457967724 | 112.99471458007619 | 1.59e4 | 0.481 | moderate |
  | `ppf(0.98999999999999999, 0.01, 10)` | 5.755945288453716e+299 | 5.7559452885097316e+299 | 4.38e4 | 9.95e3 | moderate |

- **Fix direction:** `expm1(-log(q)/d)**(-1/c)`.

### F5: ppf overflows in q**(-1/d)

- **Code:** `_ppf`, line 1231.
- **Mechanism:** for -log(q)/d > 709.8, `q**(-1/d)` is inf, and `inf**(-1/c) = 0`. The true quantile `~q**(1/(c*d))` is often still normal (pattern 8).
- **Regime:** q < exp(-709.8*d). For d = 0.5 that is q < 1e-154. For d = 0.001 it is q < 0.49, so for c = 100 every ppf below the median is lost.
- **Reproducer:** `python accuracy.py refs/burr_ref.py --at ppf 1e-300 2 0.5`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 2, 0.5)` | 0.0 | 1e-300 | inf | 1 | loss |
  | `ppf(0.42999999999999994, 100, 0.001)` | 0.0 | 0.00021611482313284221 | inf | 10 | loss |
  | `ppf(0.01, 100, 0.001)` | 0.0 | 1.0000000000000011e-20 | inf | 10 | loss |

- **Fix direction:** in log space, `exp(-log(expm1(z))/c)` with `z = -log(q)/d`, and `log(expm1(z)) = z + log1mexp(-z)` for large z.

### F6: isf overflows in expm1 for small d

- **Code:** `_isf`, lines 1234-1235: `_q = sc.xlog1py(-1.0 / d, -q)`; `sc.expm1(_q) ** (-1.0 / c)`.
- **Mechanism:** `_q = -log1p(-q)/d` exceeds 709.8 when d is small, so `expm1` returns inf and `inf**(-1/c) = 0` (pattern 8). This is the same overflow as F5, on the other side.
- **Regime:** -log1p(-q)/d > 709.8. For d = 0.001 that means q > 0.51.
- **Reproducer:** `python accuracy.py refs/burr_ref.py --at isf 0.57 100 0.001`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `isf(0.57, 100, 0.001)` | 0.0 | 0.00021611482313284278 | inf | 13.3 | loss |
  | `isf(0.99, 100, 0.001)` | 0.0 | 1.0000000000000099e-20 | inf | 990 | loss |
  | `isf(0.99999999999999967, 100, 0.001)` | 0.0 | 1.68002152189882e-155 | inf | 3e16 | loss |

- **Fix direction:** same log-space form as F5, with `z = -log1p(-q)/d`.

### F7: subnormal intermediates in pdf and logcdf

- **Code:** `_pdf` line 1204: `c_ * d_ * (x_ ** (-c_ - 1.0))`; `_logcdf` line 1222: `sc.log1p(x**(-c)) * (-d)`.
- **Mechanism:** for large c and x > 1, `x**(-c-1)` or `x**-c` falls below 2.2e-308 and keeps only part of its 53 bits. The factor `c*d` (pdf) or `d` (logcdf) then lifts the result back into the normal range (pattern 4).
- **Regime:** (c+1)*log(x) in (708, 745) for the pdf, and c*log(x) in (708, 745) for logcdf, with d large enough that the result is normal.
- **Reproducer:** `python accuracy.py refs/burr_ref.py --at pdf 1300 100 1e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1300, 100, 1e6)` | 3.102568774501476e-307 | 3.102568772686277e-307 | 2.6e6 | 101 | severe |
  | `pdf(1275.1256477602437, 100, 1e6)` | 2.183478396996865e-306 | 2.1834783969658329e-306 | 6.41e4 | 101 | moderate |
  | `logcdf(1400, 100, 1e8)` | -2.4389136925787873e-307 | -2.4389136939891708e-307 | 2.6e6 | 100 | severe |
  | `logcdf(1300, 100, 1e6)` | -4.033339404490285e-306 | -4.0333394044921606e-306 | 2.1e3 | 100 | moderate |

- **Fix direction:** evaluate in log space: `c*d*exp(-(c+1)*log(x) - (d+1)*log1p(x**-c))` in the pdf; in logcdf, `-d*x**-c` computed as `-exp(log(d) - c*log(x))` when x**-c is tiny.

### F8: large d amplifies the rounding of 1 + x**-c

- **Code:** `_pdf` line 1205: `(1 + x_ ** (-c_)) ** (d_ + 1.0)`; `_cdf` line 1219: `(1 + x**(-c))**(-d)`; `_logsf` line 1228 (through the cdf).
- **Mechanism:** the half-ulp rounding of `1 + u` (u = x**-c) is multiplied by d in `(1 + u)**d` (pattern 6). When u < eps/2, `1 + u` is exactly 1, and the cdf returns 1.0 although the true value is 1 - d*u.
- **Regime:** d >~ 1e3, anywhere u is not tiny compared with 1/d.
- **Reproducer:** `python accuracy.py refs/burr_ref.py --at pdf 100 3 1e6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(100, 3, 1e6)` | 0.011036377717862799 | 0.01103637771695487 | 3.7e5 | 1 | moderate |
  | `cdf(1e3, 3, 1e4)` | 0.9999900000491775 | 0.9999900000500048 | 3.73e3 | 3e-05 | moderate |
  | `cdf(1.5, 100, 1e6)` | 1.0 | 0.9999999999975403 | 1.11e4 | 2.46e-10 | moderate |
  | `cdf(7, 3, 1e4)` | 2.2739175440012255e-13 | 2.2739175440034565e-13 | 4.42e3 | 87.2 | moderate |
  | `logsf(8.5, 3, 1e4)` | -8.589957716735585e-08 | -8.5899577167422053e-08 | 3.47e3 | 48.8 | moderate |

- **Fix direction:** `exp(-d*log1p(x**-c))` for the cdf, and the matching log-space form for the pdf.

### F9: rounded shape exponents amplified by large |log| of the base

- **Code:** `_pdf` lines 1204-1205: exponents `-c_ - 1.0` and `d_ + 1.0`; `_ppf` line 1231: `-1.0/d`, `-1.0/c`; `_isf` lines 1234-1235: `-1.0/d`, `-1.0/c`.
- **Mechanism:** the exponents are rounded sums or quotients of the shapes. An absolute exponent error δ becomes a relative result error δ*|log(base)|, and that can reach several hundred ulp (patterns 6, 11). Examples:
  - `d + 1.0` with d = 0.001 against `log(1 + x**-0.5) = 230` at x = 1e-200;
  - `-1.0/d` = -0.1 against `log q = -59.7`, then raised to `-1/c = -100`.
- **Regime:** |log x| or |log q| in the hundreds, with shapes whose c+1, d+1, 1/c or 1/d are inexact. Fisk F5/F6 report the same mechanism.
- **Reproducer:** `python accuracy.py refs/burr_ref.py --at ppf 1.2115276586286942e-26 0.01 10`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(1.2115276586286942e-26, 0.01, 10)` | 8.803983902472657e-260 | 8.803983902473073e-260 | 213 | 10 | minor |
  | `pdf(1e-200, 0.5, 0.001)` | 3.9716411736215084e+196 | 3.971641173621408e+196 | 114 | 1 | minor |
  | `isf(1e-300, 3, 1e4)` | 2.1544346900318555e+101 | 2.1544346900318838e+101 | 59.1 | 0.333 | minor |
  | `pdf(1e-300, 0.01, 10)` | 9.890657149980097e+268 | 9.890657149980036e+268 | 28 | 0.9 | minor |
  | `ppf(1e-300, 2, 3)` | 1.0000000000000064e-50 | 1e-50 | 28.9 | 0.167 | minor |

  cond (shapes) is 1.19e3, 1.5, 234, 136 and 230 for these rows, so most errors are below the shape sensitivity. The shapes are exact inputs, though, and the error comes from the code rounding them.
- **Fix direction:** avoid the rounded combinations: `x**(-c)/x` instead of `x**(-c-1)`, and log-space forms such as `exp(-(d+1)*log1p(u))` and `exp(log(expm1(z))*(-1/c))` (the outer `-1/c` remains).

## RuntimeWarnings

The harness evaluates each parameter set in one vectorized call, so its warning lists are aggregated per call. The following come from single-point `--at` checks.
- Overflow warnings ("overflow encountered in power", "invalid value encountered in divide") accompany the F2, F5 and F6 losses. "divide by zero encountered in log1p" accompanies F1. "divide by zero encountered in power" accompanies the ppf inf in F4. "underflow encountered in power" accompanies F7.
- Spurious warnings on accurate results:
  - `cdf(1e300, 2, 3)` = 1.0 (exact) warns "underflow encountered in power".
  - `sf(1e-300, 10.5, 4.3)` = 1.0 (exact) warns "overflow encountered in power".
  - `logcdf(1259, 100, 1e6)` is within conditioning (19 ulp) but warns "underflow".
  - `pdf(0, 0.5, 1)` returns the correct inf with "divide by zero encountered in power".
- `pdf(1e-300, 10.5, 4.3)` returns nan with "invalid value encountered in divide" where the true value underflows to 0. The tool does not count this as an accuracy finding (the reference is below the normal range), but nan is the wrong answer (see F2).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | bulk and right tail, moderate shapes ((10.5, 4.3), (2, 3), (1, 1)) | 3.49 ulp |
| logpdf | right tail, all parameter sets; bulk for c*d far from 1 | 2 ulp |
| cdf | bulk and right tail for d <~ 10 | 6.7 ulp |
| logcdf | bulk and right tail outside F7 | 136 ulp max (where x**-c is near-subnormal, F7); otherwise ~1 ulp |
| isf | all q for all parameter sets except (100, 0.001) (F6) | 59.1 ulp (F9); within conditioning elsewhere |
| ppf | low tail and bulk for moderate shapes | 28.9 ulp (F9) |
| pdf, logpdf, cdf, sf, logsf at x = 0 | not measurable by the reference (0**negative); checked by hand: `pdf(0, 2, 0.5)` = 1, `pdf(0, 2, 3)` = 0, `pdf(0, 0.5, 1)` = inf, `cdf(0, 2, 3)` = 0, `sf(0, 2, 3)` = 1 | exact |

## Suspected, not confirmed

none. Every hypothesis in `scratch/burr/notes.md` was confirmed. Small c (0.01) causes only F1/F9-type minor errors.

## Opportunities: inherited methods

none (all methods are implemented)

## Shared code

No helper or `scipy.special` function was found inaccurate. All problems are in `burr_gen` itself. `fisk_gen` forwards every method to `burr_gen` with d = 1. This review confirms the fisk report's findings in `burr_gen`:
- fisk F1 = F1 here;
- fisk F2/F3 (pdf, logpdf, logcdf overflow) = F2 here;
- fisk F4 (logpdf at c = 1) = F3 here: `logpdf(1e-10, 1, 1)` is 8.0e10 ulp, measured with parameter set (1, 1);
- fisk F5/F6 (rounded exponents) = F9 here.

burr12 and mielke have the same failure types in their own code (`x**c` overflow, `(1+u)**d` rounding, `expm1` overflow in ppf/isf), so a common fix pattern would apply.

## Existing GitHub issues

- #17832 (open): Tracking issue: custom methods for continuous distributions - relates to F1; names the `sf = 1 - _cdf` cancellation risk and lists burr as lacking `_logsf`/`_logcdf` overrides
- #7491 (closed, PR #10381): Several scipy.stats distributions (fisk, burr, burr12, f) return incorrect pdf at left endpoint - relates to F2; fixed `pdf(0)` exactly, not the overflow for small x > 0
- F3-F9: none found

(web search; gh unavailable; same issues as reports/fisk.md)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `sf(1e8, 2, 3)` | 0.0 | 2.9999999999999994e-16 | verified | Reference is sf = -expm1(d*(c*log x - log1p(x**c))), confirmed by mp.quad of the pdf in t = log x. All 6 rows reproduce (sf(10, 10.5, 4.3) 8.8e9 ulp, logsf 3.88e8, sf(1e10, 1, 1) 3.73e8, sf(1.5, 100, 1e6) = 0). Mechanism matches lines 1225/1228. |
| F2 | `pdf(1e-160, 2, 0.6)` | nan | 1.2000000000000196e-32 | verified | Reference is the pdf in x**c form. All 10 rows reproduce. The cdf rows (1e-300 and 5.397e-4, c = 100, d = 0.001) are confirmed by mp.quad. logsf(1e-300, 2, 0.5) is -0.0 vs -1e-300. Mechanism matches lines 1204-1205, 1215, 1219, 1222 and 1228. |
| F3 | `logpdf(1e-10, 2, 0.5)` | 0.0 | -1.5000000000000001e-20 | verified | logpdf(1e-10, 1, 1) is 7.96e10 ulp; (2, 0.51) is 48.6 ulp. Mechanism matches line 1215. |
| F4 | `ppf(0.99999999, 3, 1e6)` | 46428.26132663437 | 46415.888181025009 | verified | Reference is expm1(-log(q)/d)**(-1/c), with cdf(x_ref) == q checked. ppf(1-3e-16, 2, 3) = inf. The other rows match (1.69e12, 1.59e4, 4.38e4 ulp). Mechanism matches line 1231. |
| F5 | `ppf(1e-300, 2, 0.5)` | 0.0 | 1.0e-300 | verified | ppf(0.43, 100, 0.001) and ppf(0.01, 100, 0.001) are both 0.0 vs 2.16e-4 and 1.0e-20. Mechanism matches line 1231. |
| F6 | `isf(0.57, 100, 0.001)` | 0.0 | 2.1611482313284277e-4 | verified | Reference is expm1(-log1p(-q)/d)**(-1/c), with sf(x_ref) == q checked. All 3 rows return 0.0. Mechanism matches lines 1234-1235. |
| F7 | `pdf(1300, 100, 1e6)` | 3.102568774501476e-307 | 3.1025687726862772e-307 | verified | 2.63e6 ulp. logcdf(1400, 100, 1e8) is 2.6e6 ulp and logcdf(1300, 100, 1e6) 2.09e3 ulp. The logcdf references need the log1p(x**-c) form; the x**c form cancels even at 200 digits. Mechanism matches lines 1204 and 1222. |
| F8 | `pdf(100, 3, 1e6)` | 0.011036377717862799 | 0.011036377716954871 | verified | 3.7e5 ulp. cdf(1.5, 100, 1e6) = 1.0 vs 1 - 2.46e-12. cdf(7, 3, 1e4) is 4.42e3 ulp (confirmed by mp.quad), and logsf(8.5, 3, 1e4) 3.47e3 ulp. Mechanism matches lines 1205, 1219 and 1228. |

Verified by dist-accuracy-verifier: 8 of 8 verified.
