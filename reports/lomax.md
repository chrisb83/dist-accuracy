# lomax: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `lomax_gen`, lines 8372-8426
- **Implemented:** pdf, logpdf, cdf, sf, logsf, ppf, isf
- **Inherited (not measured):** logcdf = generic median switch (`log(cdf)` below the median, `log1p(-sf)` above)
- **Related:** pareto (lomax is pareto with loc = -1; `_stats` calls `pareto.stats`), genpareto (lomax(c) = genpareto(1/c) scaled by 1/c); no shared methods for pdf/cdf/ppf; helpers: none; special functions: sc.expm1, sc.log1p
- **Reference:** `refs/lomax_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (1.8771398388773268), (1), (1e-17), (0.001), (1000), (1e+06)
- **Worst finding:** loss in isf, pdf and logpdf

## Summary

cdf, logsf and ppf use `log1p`/`expm1` and are accurate everywhere
(at most 1.7 ulp, ppf within conditioning). isf is written as
`q**(-1/c) - 1`. ppf already uses `expm1`, but isf does not, so it cancels
whenever `q**(-1/c)` is close to 1. For large c this happens over the whole
range: for c = 1e6 it costs 1e4-1e6 ulp in the bulk. For q near 1 it
returns 0.0 (c = 1000: `isf(1 - 1e-14) = 0.0` vs 1e-17). The pdf
`c/(1+x)**(c+1)` overflows in the denominator for large c, where the pdf is
still normal (loss). It also amplifies the rounding of `1.0 + x` by c + 1
(5e5 ulp for c = 1e6 at tiny x). For small c, the rounding of the exponent
`c + 1.0` costs up to 384 ulp. sf `exp(-c*log1p(x))` loses up to 405 ulp in
the far tail (minor). logpdf returns 0.0 or the wrong sign at its zero
crossing. This is a loss by the tool's rule but is within the conditioning
with respect to x.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | isf | q**(-1/c) close to 1: large c (whole range), q near 1 (1 - q < ~c*1.1e-16 gives 0) | `q**(-1/c) - 1` cancels (pattern 8) | 0.0 for 1e-17; 1.2e8 ulp; 3.2e6 ulp at q = 0.99 | loss |
| F2 | pdf | large c, (c+1)*log1p(x) > 709.78 while the pdf is still normal | `(1.0+x)**(c+1.0)` overflows (pattern 3) | 0.0 for 4.5e-306 | loss |
| F3 | logpdf | zero crossing x = c**(1/(c+1)) - 1 | `log(c) - (c+1)*log1p(x)` cancels (pattern 5); within cond (arg) | 0.0 for -2.4e-17; wrong sign | loss |
| F4 | pdf | large c (1e6), small x and bulk | rounding of `1.0 + x` raised to c + 1 (patterns 2, 6) | 5.0e5 ulp | moderate |
| F5 | pdf | small c (1e-3, 1e-17), huge x | rounding of exponent `c + 1.0` amplified by log(1+x) (pattern 6) | 384 ulp | minor |
| F6 | sf | far right tail, c of order 1 | rounding of `-c*log1p(x)` amplified by `exp` (pattern 6) | 405 ulp | minor |
| F7 | isf | small q, moderate c | rounding of `-1.0/c` amplified by log(q) (pattern 6) | 32.1 ulp | minor |

### F1: isf cancels in q**(-1/c) - 1

- **Code:** `_isf`, line 8419: `return q**(-1.0 / c) - 1`
- **Mechanism:** `q**(-1/c) = exp(-log(q)/c)` is close to 1 when
  `-log(q)/c` is small, and subtracting 1 keeps only its digits after the
  leading 1. The relative error is about eps/(2*isf). Once
  `-log(q)/c < eps/2`, the power rounds to 1.0 and the result is 0.0
  (pattern 8). Note that `_ppf` (line 8416) already uses the `expm1` form;
  `_isf` does not.
- **Regime:** isf(q, c) small compared with 1. For c = 1e6 this is the whole
  q range: 1e4-1e5 ulp at q = 1e-10 ... 0.5, and 1e6-1e8 ulp for q > 0.85.
  For c = 1000 it starts at q > ~0.9. The result is 0.0 when 1 - q < ~c*1.1e-16
  (c = 1e6: q >= 1 - 1e-10). For c of order 1 the error stays within the
  conditioning (cond ~ 1/(1 - q) near q = 1).
- **Reproducer:** `python accuracy.py refs/lomax_ref.py --at isf 0.99 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.99999999999999, 1000)` | 0.0 | 9.99200722162646e-18 | inf | 1.0e+14 | loss |
  | `isf(0.9999999999, 1e6)` | 0.0 | 1.000000082790371e-16 | inf | 1.0e+10 | loss |
  | `isf(0.999999, 1000)` | 1.0000005268295808e-09 | 1.0000005005290896e-09 | 1.2e+08 | 1.0e+06 | severe |
  | `isf(0.99, 1e6)` | 1.005033589684956e-08 | 1.0050335904006076e-08 | 3.2e+06 | 99.5 | severe |
  | `isf(0.5, 1e6)` | 6.931474207938493e-07 | 6.931474207865077e-07 | 4.77e+04 | 1.44 | moderate |

- **Fix direction:** `sc.expm1(-np.log(q)/c)`, which mirrors `_ppf`.

### F2: pdf denominator overflows for large c

- **Code:** `_pdf`, line 8401: `return c*1.0/(1.0+x)**(c+1.0)`
- **Mechanism:** `(1+x)**(c+1)` overflows to inf once
  `(c+1)*log1p(x) > 709.78`, and `c/inf = 0`. The true pdf
  `c*exp(-(c+1)*log1p(x))` is still normal up to
  `(c+1)*log1p(x) ~ 708 + log(c)` (pattern 3).
- **Regime:** large c. The window where the true pdf is normal but SciPy returns 0
  is `709.78 < (c+1)*log1p(x) < 708.4 + log(c)`. Its width is log(c): 6.9 for
  c = 1000 and 13.8 for c = 1e6.
- **Reproducer:** `python accuracy.py refs/lomax_ref.py --at pdf 1.032549079066965 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1.032549079066965, 1000)` | 0.0 | 4.476286225675463e-306 | inf | 509 | loss |
  | `pdf(0.0007102513991588548, 1e6)` | 0.0 | 4.4762862256751083e-303 | inf | 710 | loss |

- **Fix direction:** evaluate in log space, `c*np.exp(-(c+1)*sc.log1p(x))`,
  or better `np.exp(np.log(c) - sc.log1p(x) - c*sc.log1p(x))` (this also fixes F4 and F5).

### F3: logpdf loses everything at its zero crossing

- **Code:** `_logpdf`, line 8404: `return np.log(c) - (c+1)*sc.log1p(x)`
- **Mechanism:** at x0 = c**(1/(c+1)) - 1 the two terms are equal (each about
  log(c)), so their difference is only accurate to about 1 ulp of log(c) in
  absolute terms (pattern 5). Near x0 the true logpdf is ~1e-17, so the result
  is 0.0 or has the wrong sign.
- **Regime:** |x - x0| within a few ulp of x0. x0 lies in the support only for
  c > 1, because the pdf at 0 equals c.
- **Reproducer:** `python accuracy.py refs/lomax_ref.py --at logpdf 0.24468229012549414 1.8771398388773268`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.24468229012549414, 1.8771398388773268)` | 0.0 | -2.3987396170820582e-17 | inf | 2.4e+16 | loss |
  | `logpdf(0.006924720186917837, 1000)` | -8.881784197001252e-16 | 2.6323736313358985e-17 | inf | 2.6e+17 | loss |

  Context: cond (arg) is 2.4e16 and 2.6e17, so a 1-ulp change of x alone
  moves the true value by more than its size. The tool rates it loss because
  of the 0 and the wrong sign. The absolute error, about 1e-15, is
  what double arithmetic gives. The (1e6) points near x0 measure 2.9e16 ulp
  and are rated within conditioning.
- **Fix direction:** no cheap fix; it would take extended precision for
  `log(c) - (c+1)*log1p(x)` near x0. Low practical priority.

### F4: pdf amplifies the rounding of 1 + x by c + 1

- **Code:** `_pdf`, line 8401: `(1.0+x)**(c+1.0)`
- **Mechanism:** `1.0 + x` carries a rounding error of up to eps/2 relative.
  Raising to the power c + 1 multiplies it by c + 1 (patterns 2 and 6).
  For tiny x, `1.0 + x` loses x entirely, e.g. `pdf(1e-16, 1e6) = 1e6` exactly.
- **Regime:** large c. For c = 1e6 the error is 1e5-5e5 ulp for every
  x <= 7e-4 (the range where the pdf is normal), including x < 1e-9 where
  cond (arg) < 1e-3.
- **Reproducer:** `python accuracy.py refs/lomax_ref.py --at pdf 1e-15 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-15, 1e6)` | 999999.9988897758 | 999999.998999999 | 5.0e+05 | 1e-09 | moderate |
  | `pdf(1e-12, 1e6)` | 999998.9999105994 | 999998.9999995 | 4.0e+05 | 1e-06 | moderate |
  | `pdf(0.0007002443566872014, 1e6)` | 9.859676543263245e-299 | 9.859676543759977e-299 | 2.3e+05 | 700 | moderate |

- **Fix direction:** `exp(log(c) - (c+1)*log1p(x))`, using `log1p` as logpdf already does.

### F5: pdf rounds the exponent c + 1 for small c

- **Code:** `_pdf`, line 8401: `(c+1.0)`
- **Mechanism:** `c + 1.0` is rounded (for c < 1.1e-16 it is exactly 1.0).
  The relative error of the exponent, up to eps/2, is multiplied by
  `(c+1)*log(1+x)`, which is up to ~700 for huge x (pattern 6).
- **Regime:** c < 1 and large x. Measured 384 ulp for c = 1e-3 and 34 ulp for
  c = 1e-17 (there the dropped `c*log(x)` term itself is 7e-15).
- **Reproducer:** `python accuracy.py refs/lomax_ref.py --at pdf 1e300 1e-3`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(5.040050609012411e+303, 0.001)` | 9.859676543759344e-308 | 9.859676543758585e-308 | 384 | 1 | minor |
  | `pdf(1e+300, 0.001)` | 5.011872336273104e-304 | 5.011872336272723e-304 | 343 | 1 | minor |

- **Fix direction:** keep c and 1 apart: `c/(1+x) * exp(-c*log1p(x))` or
  `exp(log(c) - log1p(x) - c*log1p(x))`.

### F6: sf exponentiates a rounded product

- **Code:** `_sf`, line 8410: `return np.exp(-c*sc.log1p(x))`
- **Mechanism:** the relative error of `c*log1p(x)`, about 1 ulp, becomes an absolute
  error in the exponent, and `exp` turns that into a relative error multiplied by
  `c*log1p(x)` (up to ~745). cond (arg) is only `c*x/(1+x) ~ c` (pattern 6).
- **Regime:** far right tail with c of order 1 (cond 1-2). For large c the
  same error is within conditioning. Max 405 ulp, median of flagged points
  ~40 ulp.
- **Reproducer:** `python accuracy.py refs/lomax_ref.py --at sf 1e140 1.8771398388773268`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1e+140, 1.8771398388773268)` | 1.5864360025280503e-263 | 1.586436002528193e-263 | 405 | 1.88 | minor |
  | `sf(1e+280, 1)` | 9.999999999999464e-281 | 1e-280 | 241 | 1 | minor |

- **Fix direction:** for large x use `np.power(1+x, -c)`. The rounding of `1 + x` (eps/2)
  is then amplified only by c, which stays within conditioning. Keep the
  `exp(-c*log1p(x))` form for small x.

### F7: isf rounds the exponent -1/c

- **Code:** `_isf`, line 8419: `q**(-1.0 / c)`
- **Mechanism:** the rounding of `-1.0/c` (eps/2 relative) is multiplied by
  `|log(q)/c|`, which is up to 368 for c = 1.88 and q = 1e-300 (pattern 6;
  same as pareto F4).
- **Regime:** q -> 0 with c of order 1. 9-32 ulp, growing linearly with |log q|.
- **Reproducer:** `python accuracy.py refs/lomax_ref.py --at isf 1e-300 1.8771398388773268`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1e-300, 1.8771398388773268)` | 6.570653561229815e+159 | 6.570653561229862e+159 | 32.1 | 0.533 | minor |

  Context: cond (shapes) is 368. The `expm1(-log(q)/c)` form recommended in F1
  has an error bound of the same size, so F1's fix does not worsen this.
- **Fix direction:** none cheap; it is inherent to forming 1/c or log(q)/c in double.

## RuntimeWarnings

- pdf: `overflow encountered in power` in the F2 regime, where the result is
  wrong. `underflow encountered in divide` where the true pdf is subnormal
  (e.g. `pdf(1e300, 1e-17) = 1e-317`), which is accurate and not measured.
- cdf/logsf: `underflow encountered in multiply` for c = 1e-17 and tiny x,
  where `c*log1p(x)` is subnormal (`cdf(1e-300, 1e-17) = 1.0000007e-317`;
  the true value is subnormal). Spurious.
- sf: `underflow encountered in exp/multiply` where the sf underflows; correct.
- isf: `overflow encountered in power` for c = 1e-3, 1e-17 and small q, where
  the true isf overflows (e.g. `isf(1e-300, 1e-3) = inf`); correct.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid (x down to 1e-300, up to 1e300), all six parameter sets, overflow edges | 1.7 ulp |
| logsf | full grid, all parameter sets | 0.99 ulp |
| ppf | q in [1e-300, 1 - 1e-16], all parameter sets including c = 1e-17 and 1e6 | 257 ulp at cond 1e3 (c = 1e-3), within conditioning everywhere |
| logpdf | everything except the zero crossing (F3) | 1.21 ulp |
| pdf | c = 1, 1.877 | 1.6 ulp |
| sf | c = 1e-17, 1e-3 | 0.93 ulp |
| isf | c = 1e-17, 1e-3 | within conditioning (cond up to 1e17) |

## Suspected, not confirmed

none. All hypotheses in `scratch/lomax/notes.md` were either confirmed (F1-F7)
or covered by the grid and found accurate (cdf, logsf, ppf).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | median switch: `log(cdf)` below the median, `log1p(-sf)` above | with `L = c*log1p(x)`: `log(-expm1(-L))`, and for `L` < ~1e-300 use `log(c) + log(log1p(x))` | only when `c*log1p(x)` underflows (tiny c and tiny x): `logcdf(1e-30, 1e-300) = -inf` vs -759.85 (loss). Elsewhere the default is accurate, e.g. `logcdf(1e-10, 1e-300)` 0.72 ulp and `logcdf(1e20, 1)` 3.39 ulp |

## Shared code

none. lomax uses no helpers, and `sc.log1p`/`sc.expm1` behave correctly.
The isf pattern F1 (`q**(-1/c) - 1` instead of `expm1`) belongs with
INDEX cross-cutting items 3 and 4 (`1 - q` and rounded exponents). It mirrors
pareto F4 (exponent rounding) and the genpareto quantile forms.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `isf(0.99999999999999, 1000)` | 0.0 | 9.9920072216264588e-18 | verified | reference by Newton on sf(x) = q, sf from `mp.quad` of the docstring pdf; 100/200 digits agree |
| F1 | `isf(0.9999999999, 1e6)` | 0.0 | 1.0000000827903711e-16 | verified | |
| F1 | `isf(0.999999, 1000)` | 1.0000005268295808e-09 | 1.0000005005290895e-09 | verified | 1.18e8 ulp |
| F1 | `isf(0.99, 1e6)` | 1.005033589684956e-08 | 1.0050335904006076e-08 | verified | 3.21e6 ulp |
| F1 | `isf(0.5, 1e6)` | 6.931474207938493e-07 | 6.9314742078650777e-07 | verified | 4.77e4 ulp; mechanism matches line `q**(-1.0 / c) - 1` |
| F2 | `pdf(1.032549079066965, 1000)` | 0.0 | 4.476286225675463e-306 | verified | reference `c/(1+x)**(c+1)` in mpmath; overflow of the power matches `_pdf` |
| F2 | `pdf(0.0007102513991588548, 1e6)` | 0.0 | 4.4762862256751082e-303 | verified | |
| F3 | `logpdf(0.24468229012549414, 1.8771398388773268)` | 0.0 | -2.3987396170820581e-17 | verified | reference `log(c/(1+x)**(c+1))`; within cond (arg) 2.4e16 as the report states |
| F3 | `logpdf(0.006924720186917837, 1000)` | -8.881784197001252e-16 | 2.6323736313358985e-17 | verified | wrong sign; within cond (arg) |
| F4 | `pdf(1e-15, 1e6)` | 999999.9988897758 | 999999.998999999 | verified | 4.96e5 ulp; mechanism (rounding of `1.0+x` raised to c+1) matches `_pdf` |
| F4 | `pdf(1e-12, 1e6)` | 999998.9999105994 | 999998.9999995 | verified | 4.0e5 ulp |
| F4 | `pdf(0.0007002443566872014, 1e6)` | 9.859676543263245e-299 | 9.8596765437599763e-299 | verified | 2.27e5 ulp |

Verified by dist-accuracy-verifier: 4 of 4 verified.
