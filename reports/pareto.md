# pareto: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `pareto_gen`, lines 8211-8366
- **Implemented:** pdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf)
- **Related:** none; helpers: `_check_fit_input_parameters` (fit only); special functions: none (plain `**`, `pow`, `np.power`)
- **Reference:** `refs/pareto_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (2.621716532144454), (1), (0.001), (1e-08), (1e-17), (1000), (1e+06)
- **Worst finding:** loss in cdf and ppf

## Summary

sf and isf are direct powers and accurate; pdf is accurate except for a
minor exponent-rounding error for b < 1. The cdf, written as `1 - x**(-b)`,
cancels whenever `x**(-b)` is close to 1, i.e. b*log(x) small. For b of
order 1 this only happens next to x = 1, where the cdf is ill-conditioned
anyway, but for small b (below about 0.1) the loss goes beyond conditioning
over the whole support: with b = 1e-17 the cdf is 0.0 for x below about 257,
1.11e-16 (still loss/severe) from there up to about 6e4, and has only one
or two correct digits up to 1e308. ppf forms
`1 - q` before raising it to `-1/b`, so the rounding of `1 - q` is
amplified by 1/b: for b = 1e-8 and small q the error is up to 2.4e7 ulp, and
for b = 1e-17 the result is completely wrong (66318.9 instead of 22026.5).
The fixes are standard (`-expm1(-b*log(x))`, `exp(-log1p(-q)/b)`).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | b*log(x) small with b below ~0.1 (any x for b = 1e-17; x near 1 for b = 1e-3) | 1 - x**(-b) cancels (patterns 1, 2) | 0.0 for 6.9e-18; 1.6e14 ulp | loss |
| F2 | ppf | small q (below ~1e-6) and small b | rounding of 1 - q raised to -1/b (pattern 8) | 66318.9 for 22026.5 | loss |
| F3 | pdf | b < 1, large x | rounding of the exponent -b-1, amplified by log(x) (pattern 6) | 385 ulp | minor |
| F4 | isf | q small (large result) | rounding of the exponent -1/b, amplified by log(q) (pattern 6) | 73.4 ulp | minor |

### F1: cdf cancels in 1 - x**(-b) for small b

- **Code:** `_cdf`, line 8241: `return 1 - x**(-b)`
- **Mechanism:** `x**(-b) = exp(-b*log(x))` is close to 1 when b*log(x) is
  small, and the subtraction keeps only its digits beyond the leading 1s
  (pattern 1/2: this is `1 - exp(-y)`). The relative error is about
  eps/(b*log(x)). Once b*log(x) < 2**-54 = 5.55e-17 (half the spacing 2**-53 of doubles just below 1) the power rounds to 1.0 and the cdf
  is 0.0. Near x = 1 the cdf's condition number with respect to x is about
  1/log(x), so the error relative to conditioning is about 1/b: harmless
  for b of order 1 (e.g. `cdf(1.01, 0.5)` is 19.7 ulp with cond 100, ok),
  but beyond conditioning for small b. For b = 1e-17 the cancellation happens
  at every representable x, because b*log(x) <= 7.1e-15 on the whole
  support.
- **Regime:** b below about 0.1. The cdf is exactly 0.0 for
  b*log(x) < 2**-54 = 5.55e-17, i.e. x < exp(5.55e-17/b) (for b = 1e-17,
  x < ~257; for b = 1e-3, x = 1 + 2.2e-16 and 1 + 1e-15). Just above that,
  `x**(-b)` rounds to 1 - 2**-53 and the cdf is 1.11e-16 (for b = 1e-17 at
  every x from ~258 to ~6e4), still loss/severe. Severe up to about 1e6 ulp
  at b*log(x) ~ 1e-10, moderate/minor beyond.
- **Reproducer:** `python accuracy.py refs/pareto_ref.py --at cdf 2.0 1e-17`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(2.0, 1e-17)` | 0.0 | 6.9314718055994535e-18 | inf | 1.44 | loss |
  | `cdf(1.0000000000000002, 0.001)` | 0.0 | 2.2204460492503126e-19 | inf | 4.5e+15 | loss |
  | `cdf(1e+20, 1e-17)` | 4.440892098500626e-16 | 4.605170185988091e-16 | 1.6e+14 | 0.0217 | severe |
  | `cdf(2.0, 1e-08)` | 6.931471729920702e-09 | 6.931471781576803e-09 | 3.4e+07 | 1.44 | severe |
  | `cdf(1.0001, 0.001)` | 9.999499528401401e-08 | 9.999499533379744e-08 | 2.2e+06 | 1e+04 | severe |
  | `cdf(10000000001.0, 1e-08)` | 2.3025848283619865e-07 | 2.3025848279091606e-07 | 8.9e+05 | 0.0434 | moderate |
  | `cdf(1.01, 0.001)` | 9.950281348758416e-06 | 9.950281348790243e-06 | 1.44e+04 | 100 | moderate |
  | `cdf(1.5, 0.001)` | 0.00040538291823999373 | 0.0004053829182399677 | 289 | 2.47 | minor |

  Counts over the grid: b = 1e-17: 28 loss, 32 severe (every point);
  b = 1e-8: 4 loss, 23 severe, 33 moderate (every point); b = 1e-3: 2 loss,
  5 severe, 1 moderate, 19 minor. No flags for b >= 1.
- **Fix direction:** `-expm1(-b*log(x))` (or `-expm1(-b*log1p(x - 1))`
  for x near 1; x - 1 is exact there).

### F2: ppf rounds 1 - q before raising it to -1/b

- **Code:** `_ppf`, line 8244: `return pow(1-q, -1.0/b)`
- **Mechanism:** `1 - q` carries an absolute rounding error up to 2**-54 (eps/4) for 1 - q in [0.5, 1)
  (and is exactly 1.0 for q < 2**-54 = 5.55e-17, half the spacing 2**-53 of doubles just below 1). The power `(1-q)**(-1/b)`
  multiplies this relative error by 1/b (pattern 8, with pattern 6
  amplification). The condition number with respect to q is only
  q/((1-q)*b), so for small q and small b nearly all of the error is
  beyond conditioning. For q < 2**-54 = 5.55e-17 the result is exactly 1.0, although
  the true quantile is `exp(q/b)` (2.718... for q = b = 1e-17).
- **Regime:** q small (roughly q < 1e-6, the error is up to about (eps/4)/b = 2**-54/b
  relative, independent of q) and b small: up to ~250 ulp for b = 1e-3,
  ~2.4e7 ulp for b = 1e-8, relative error >= 1 for b = 1e-17 and q >= 1e-16.
  `1 - q` rounds to exactly 1.0, so the result is exactly 1.0, for
  q < 2**-54 = 5.55e-17 (checked: q = 5.5e-17 gives 1.0, q = 5.56e-17 does not).
- **Reproducer:** `python accuracy.py refs/pareto_ref.py --at ppf 1e-16 1e-17`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(5e-16, 1e-17)` | 1.28288185880303e+24 | 5.184705528587139e+21 | 1.1e+18 | 50 | loss |
  | `ppf(1e-16, 1e-17)` | 66318.90284033472 | 22026.465794806707 | 9.1e+15 | 10 | loss |
  | `ppf(1e-17, 1e-17)` | 1.0 | 2.718281828459045 | 2.8e+15 | 1 | severe |
  | `ppf(3.4807005884285624e-14, 1e-08)` | 1.0000034861063738 | 1.000003480706646 | 2.4e+07 | 3.48e-06 | severe |
  | `ppf(1e-12, 1e-08)` | 1.0001000027877733 | 1.0001000050001667 | 1.0e+07 | 0.0001 | severe |
  | `ppf(1e-17, 1e-08)` | 1.0 | 1.000000001 | 4.5e+06 | 1e-09 | severe |
  | `ppf(5e-16, 0.001)` | 1.0000000000005551 | 1.0000000000005 | 248 | 5e-13 | minor |

  Counts: b = 1e-17: 2 loss, 2 severe; b = 1e-8: 7 severe; b = 1e-3: 10 minor
  (including bulk points q = 0.08, 0.1 at 128-129 ulp, cond 87-111, where
  the rounding of 1 - q is amplified by 1/b = 1000).
- **Fix direction:** `exp(-log1p(-q)/b)`; keeps the relative accuracy
  of the small quantity log1p(-q).

### F3: pdf loses up to ~log(x)/2 ulp from rounding of -b-1

- **Code:** `_pdf`, line 8238: `return b * x**(-b-1)`
- **Mechanism:** the exponent `-b-1` is rounded to a double near -1 with an
  absolute error up to 1.1e-16 whenever b has bits below 2**-52 relative
  to 1 (in particular for every b < 1 that is not a dyadic fraction).
  `x**e` turns an exponent error de into a relative error de*log(x), up to
  ~700*1.1e-16 = 350 ulp for x near 1e304 (pattern 6: rounding error raised
  to a large power). For b < 1.1e-16, `-b-1` is exactly -1 and the factor
  `x**(-b)` is lost entirely (31 ulp at x = 1e290 for b = 1e-17). The cond
  with respect to x is 1 and cond (shapes) is 0.3-1, so the error is
  beyond both.
- **Regime:** b < 1 (non-dyadic), large x; error ~ log(x) * ulp(1 + b)/2 / eps.
  b = 0.001: 31 minor points, max 385 ulp; b = 1e-8: 28 minor points, max
  183 ulp; b = 1e-17: 20 minor points, max 33 ulp.
- **Reproducer:** `python accuracy.py refs/pareto_ref.py --at pdf 5.040050609012411e+303 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(5.040050609012411e+303, 0.001)` | 9.859676543759346e-308 | 9.859676543758585e-308 | 385 | 1 | minor |
  | `pdf(1e+300, 0.001)` | 5.011872336273104e-304 | 5.011872336272723e-304 | 343 | 1 | minor |
  | `pdf(1e+290, 1e-08)` | 9.999933225255653e-299 | 9.999933225255248e-299 | 183 | 1 | minor |

- **Fix direction:** `b * x**(-b) / x` (the exponent -b is exact); keep
  `x**(-b-1)` where `x**(-b)` would underflow before the division.

### F4: isf loses up to ~|log(result)|/2 ulp from rounding of -1/b

- **Code:** `_isf`, line 8250: `return np.power(q, -1.0 / b)`
- **Mechanism:** `-1.0/b` is rounded (relative error up to eps/2); the
  power turns this into a relative error of up to (eps/2)*|log(q)/b| =
  (eps/2)*log(result) (pattern 6). The same mechanism applies to `_ppf`
  (line 8244) for large results. cond (arg) is 1/b = 0.38, but cond
  (shapes) is 263 at the worst point, i.e. the error is below the effect
  of a 1-ulp change in b (context only, does not change the severity).
- **Regime:** small q, b of order 1 (results up to 1e308): b = 2.62:
  22 minor points, max 73.4 ulp at q = 1e-300. For b <= 1e-3 the results
  overflow (not measured); for b >= 1000 the results are small and the
  error stays below 1 ulp.
- **Reproducer:** `python accuracy.py refs/pareto_ref.py --at isf 1e-300 2.621716532144454`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1e-300, 2.621716532144454)` | 2.684398732766657e+114 | 2.6843987327666134e+114 | 73.4 | 0.381 | minor |

- **Fix direction:** split 1/b into a double-double `r + r_lo`
  (`r_lo = fma(-r, b, 1)/b`) and multiply `q**(-r)` by
  `exp(-r_lo*log(q))`; probably not worth it given cond (shapes).

## RuntimeWarnings

- pdf: `underflow encountered in power` / `in multiply` for large x or large
  b where the true pdf is subnormal or 0; the results are correct (spurious
  in the sense of accuracy, the true value is below the normal range).
- cdf, sf: `underflow encountered in power` where `x**(-b)` underflows;
  cdf = 1.0 and sf = 0/subnormal there are correct.
- ppf, isf: `overflow encountered in power` for small b, where the true
  quantile exceeds 1.8e308; inf is the correct rounded answer. Reported for
  every point of those parameter sets because the warnings are per array call.
- No warnings coincide with the inaccurate results of F1-F4.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| sf | full grid (x from 1 to 1e308, near 1 down to 1 + 2.2e-16), all parameter sets | 1 ulp |
| isf | q in [1e-300, 1 - 3e-16], all parameter sets | 73.4 ulp (F4); q near 1 up to 2.56e3 ulp, within conditioning (cond 1e17) |
| pdf | b >= 1, full grid; subnormal-intermediate regime ((b+1)*log(x) in 708-745) for b = 1000, 1e6, 1e15 | 111 ulp (b = 1000) and 3.1e4 ulp (b = 1e6), both within conditioning (cond = b + 1) |
| cdf | b >= 1 (2.62, 1, 1000, 1e6), full grid incl. x = 1 + 2.2e-16 | within conditioning (no flagged points) |
| ppf | b >= 1, q in [1e-300, 1 - 3e-16] | 3.93 ulp |

## Suspected, not confirmed

- pdf, pattern 4 (subnormal `x**(-b-1)` lifted by a large b): occurs for
  (b+1)*log(x) in (708, 708 + log(b)), but the pdf's condition number with
  respect to x is b + 1 there, which always exceeds the lost precision
  (`pdf(1.00072027, 1e6)`: 3.13e4 ulp, cond 1e6, ok). Not a finding.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `log(b) - (b+1)*log(x)` (or `log(b) - b*log(x) - log(x)` to avoid rounding of b+1, cf. F3) | pdf underflows for (b+1)*log(x) > ~745 + log(b): `logpdf(1e200, 2.6217)` = -inf, true -1666.8982703249953 |
| logcdf | log(cdf) | `log(-expm1(-b*log(x)))`, i.e. `log1mexp(-b*log(x))` | inherits F1: `logcdf(2.0, 1e-17)` = -inf, true -39.51045950148044; also loses digits near x = 1 for all b |
| logsf | log(sf) | `-b*log(x)` | sf underflows for b*log(x) > ~745: `logsf(1e200, 2.6217)` = -inf, true -1207.3450809943729 |

## Shared code

none (no helpers or special functions are used by the distribution methods;
`_check_fit_input_parameters` is only used by `fit`).

## Existing GitHub issues

none found (searched scipy/scipy issues with "pareto" in the title; #24867 (pdf docs) and #12545 (fit warning) are unrelated to F1-F4)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(2.0, 1e-17)` | 0.0 | 6.9314718055994536e-18 | verified (loss) | ref by mp.quad of b*exp(-b*t) over [0, log x]; 100/200 digits agree. Mechanism matches line 8241 `1 - x**(-b)`. Threshold correction: x**(-b) rounds to 1.0 for b*log(x) < 2**-54 = 5.55e-17 (not 1.1e-16), so for b = 1e-17 cdf is 0.0 only for x < ~260; up to ~6e4 it returns 1.11e-16 (still loss/severe) |
| F1 | `cdf(1.0000000000000002, 0.001)` | 0.0 | 2.2204460492503129e-19 | verified (loss) | |
| F1 | `cdf(1e+20, 1e-17)` | 4.440892098500626e-16 | 4.6051701859880906e-16 | verified (severe) | 1.61e14 ulp |
| F1 | `cdf(2.0, 1e-08)` | 6.931471729920702e-09 | 6.9314717815768026e-09 | verified (severe) | 3.36e7 ulp |
| F1 | `cdf(1.0001, 0.001)` | 9.999499528401401e-08 | 9.9994995333797446e-08 | verified (severe) | 2.24e6 ulp |
| F2 | `ppf(5e-16, 1e-17)` | 1.28288185880303e+24 | 5.1847055285871389e+21 | verified (loss) | ref exp(-log1p(-q)/b), checked by quad cdf(x_ref) == q to 30 digits; mechanism matches line 8244 `pow(1-q, -1.0/b)` |
| F2 | `ppf(1e-16, 1e-17)` | 66318.90284033472 | 22026.465794806707 | verified (loss) | 9.06e15 ulp |
| F2 | `ppf(1e-17, 1e-17)` | 1.0 | 2.7182818284590452 | verified (severe) | threshold correction: 1 - q is exactly 1.0 for q < ~5.55e-17 (not 1.1e-16); q = 6e-17 already gives 66318.9 |
| F2 | `ppf(3.4807005884285624e-14, 1e-08)` | 1.0000034861063738 | 1.0000034807066461 | verified (severe) | 2.43e7 ulp |
| F2 | `ppf(1e-12, 1e-08)` | 1.0001000027877733 | 1.0001000050001667 | verified (severe) | 9.96e6 ulp |
| F2 | `ppf(1e-17, 1e-08)` | 1.0 | 1.000000001 | verified (severe) | 4.5e6 ulp |

Verified by dist-accuracy-verifier: 2 of 2 verified.
