# wald: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `wald_gen`, lines 11386-11444
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf (all eight are one-line wrappers `invgauss._<method>(x, 1.0)`, lines 11415-11438; the arithmetic is in `invgauss_gen`, lines 5074-5121)
- **Inherited (not measured):** none
- **Related:** invgauss (base class `invgauss_gen`; wald = invgauss with mu = 1, see `reports/invgauss.md`); helpers: `_norm_logcdf`, `_norm_logsf` (via invgauss); special functions: `sc.log_ndtr`, `scu._invgauss_ppf`, `scu._invgauss_isf` (Boost, via invgauss)
- **Reference:** `refs/wald_ref.py`, 400 digits; self-checks: PASS. Run without `--plot` (crashes on this machine); output in `scratch/wald/run1.txt`, points in `scratch/wald/run1.json`.
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in isf and logpdf

## Summary

wald's wrappers add no error of their own: mu = 1.0 is exact, so `x/mu`
and `2/mu` are exact, and on a grid of 2001 x values and 700 q values every
method returns exactly the same bits as `invgauss.<method>(x, 1.0)`. Every
problem found is an invgauss problem at mu = 1. pdf, cdf, ppf and the bulk
of all methods are accurate. isf returns Boost's search bound (e.g.
4.4e34 for 443.8) for q below about 1e-68, with severe errors from about
1e-65. logpdf returns -inf for x > 1.34e154 (overflow of `(x - 1)**2`).
sf and logcdf lose up to 3.8e5 ulp in the right tail (x = 1e3) because
`log1p(-exp(b - a))` cancels.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | isf | q < ~1e-65 (severe), q < ~1e-68 (loss) | Boost `_invgauss_isf` does not converge and returns its search bound (invgauss F2) | 4.8e233 for 1361.4 | loss |
| F2 | logpdf | x > 1.34e154 | `(x/mu - 1)**2` overflows (pattern 3; invgauss F6) | -inf for -7e153 | loss |
| F3 | sf, logcdf, logsf | right tail x >~ 5, growing like x ulp | `log1p(±exp(b - a))` cancels when exp(b - a) -> 1 (patterns 1, 5; invgauss F1) | 3.8e5 ulp | moderate |
| F4 | cdf, logsf | left tail x <~ 2e-3 | exp of a log of size ~500 with an extra absolute error in b - a | 660 ulp (cond 500) | minor |

### F1: isf returns Boost's search bound in the far right tail

- **Code:** `wald_gen._isf`, line 11429 -> `invgauss_gen._isf`, line 5116:
  `isf = scu._invgauss_isf(x, mu, 1)`.
- **Mechanism:** Boost's `quantile(complement(inverse_gaussian))` does not
  converge. It emits "Unable to locate solution in a reasonable time ...
  Current best guess is %1%" and returns a finite value, so the nan fallback
  at line 5119 never runs. This is not a pattern from patterns.md; it is a
  convergence failure inside the special function.
- **Regime:** q in about [1e-68, 1e-65]: severe (1.9e12 ulp at 1e-65,
  1.4e15 ulp at 1e-67). q <= ~1e-68: loss, with results that are not even
  monotone in q (1e-68 -> 970.3, 1e-69 -> 9419.0, 1e-70 -> 91476.6). Where
  Boost does converge (q from 5e-64 to 1e-20) errors still reach 154 ulp
  (minor). ppf is not affected: for q > 0.5 it calls `_invgauss_isf(1 - q)`
  with 1 - q >= 1.1e-16.
- **Reproducer:** `python accuracy.py refs/wald_ref.py --at isf 1e-100`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1e-300)` | 4.849712021959747e+233 | 1361.4454371385305 | 1.6e+246 | 0.00147 | loss |
  | `isf(1e-100)` | 4.43905398835726e+34 | 443.76389884560626 | 4.5e+47 | 0.00448 | loss |
  | `isf(1e-70)` | 91476.55228318354 | 306.7100384323816 | 1.3e+18 | 0.00646 | loss |
  | `isf(1e-67)` | 201.95731938668308 | 293.03036473289706 | 1.4e+15 | 0.00676 | severe |
  | `isf(1e-65)` | 283.7925324223683 | 283.9140878835025 | 1.9e+12 | 0.00697 | severe |
  | `isf(5.1089697745070394e-64)` | 276.1296464332499 | 276.12964643324045 | 154 | 0.00717 | minor |

- **Fix direction:** treat Boost non-convergence as nan and invert logsf
  (after F3 is fixed), starting from x ~ 2(-log q) in the right tail.

### F2: logpdf overflows in (x - 1)**2

- **Code:** `wald_gen._logpdf`, line 11432 -> `invgauss_gen._logpdf`,
  line 5079: `- (x/mu - 1)**2/(2*x)`.
- **Mechanism:** `(x - 1)**2` overflows for x > 1.34e154. The result,
  about -x/2, is finite (pattern 3).
- **Regime:** x > 1.34e154 (all x from 1.35e154 up to 1.8e308).
- **Reproducer:** `python accuracy.py refs/wald_ref.py --at logpdf 1.4e154`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.4e+154)` | -inf | -7e+153 | inf | 1 | loss |
  | `logpdf(1e+200)` | -inf | -5e+199 | inf | 1 | loss |
  | `logpdf(1.3e+154)` | -6.5e+153 | -6.5e+153 | 0 | 1 | ok |

- **Fix direction:** `-((x/mu - 1)/sqrt(x))**2/2` or `-(x/(2 mu**2) - 1/mu + 1/(2x))`.

### F3: sf, logcdf, logsf cancel in the right tail

- **Code:** `invgauss_gen._logsf`, lines 5092-5095:
  `a = _norm_logsf(fac*(x/mu - 1)); b = 2/mu + _norm_logcdf(-fac*(x/mu + 1)); return a + np.log1p(-np.exp(b - a))`;
  `_sf = exp(_logsf)` (line 5098); `_logcdf`, lines 5086-5089:
  `a + np.log1p(np.exp(b - a))`. wald reaches these through lines 11423,
  11435 and 11438.
- **Mechanism:** sf = Phi(-z1) - e**2 Phi(-z2) with z1 = (x - 1)/sqrt(x) and
  z2 = (x + 1)/sqrt(x). For large x, exp(b - a) ~ (x - 1)/(x + 1) -> 1, and
  `log1p(-exp(b - a))` keeps only the absolute error eps*|a| of a and b,
  where |a| ~ x/2. Relative to 1 - exp(b - a) ~ 2/x this is an error of
  about x/4 to x ulp in sf, while cond is about x/2 (patterns 1 and 5).
  logcdf ~ -sf has the same cancellation in `log1p(exp(b - a))`. In logsf
  the same absolute error is divided by |logsf| ~ x/2, so it stays minor.
- **Regime:** onset around x ~ 5 (13-40 ulp minor at x = 6...9.5), moderate
  from x ~ 100. Measured up to x = 1410 (sf underflows near x ~ 1416).
- **Reproducer:** `python accuracy.py refs/wald_ref.py --at sf 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logcdf(1000.0)` | -4.869434437102362e-222 | -4.8694344366891734e-222 | 3.8e+05 | 501 | moderate |
  | `sf(1000.0)` | 4.869434437047494e-222 | 4.8694344366891734e-222 | 3.3e+05 | 501 | moderate |
  | `sf(300.0)` | 2.960519417136735e-69 | 2.960519417132538e-69 | 6.38e+03 | 151 | moderate |
  | `sf(100.0)` | 4.043703566769238e-25 | 4.043703566764897e-25 | 4.83e+03 | 51.5 | moderate |
  | `logsf(1000.0)` | -509.5909128463506 | -509.5909128464242 | 651 | 0.984 | minor |
  | `sf(9.5)` | 0.0004801858666489938 | 0.00048018586664899806 | 40.2 | 6.01 | minor |

- **Fix direction:** evaluate sf = exp(-z1**2/2) (erfcx(z1/sqrt2) - erfcx(z2/sqrt2))/2
  (2/mu - z2**2/2 = -z1**2/2 exactly), and use logcdf = log1p(-sf) where sf is small.

### F4: cdf and logsf slightly beyond conditioning in the left tail

- **Code:** `invgauss_gen._cdf`, line 5101: `np.exp(self._logcdf(x, mu))`
  with `_logcdf` lines 5086-5089; `_logsf` lines 5092-5095.
- **Mechanism:** in the left tail logcdf ~ -500. It is formed as
  `a + log1p(exp(b - a))`, where a and b both have size ~500 and carry
  absolute errors of eps*500. exp amplifies these errors to about 500 ulp,
  which matches the conditioning. The extra error comes from rounding in
  `b - a`, which is about as large as the terms. logsf = -cdf there is
  computed as `a + log1p(-exp(b - a))` with the same exp(b - a) of size
  ~1e-219, so it has the same error.
- **Regime:** x <~ 2e-3 (cdf normal down to x ~ 1.4e-3).
- **Reproducer:** `python accuracy.py refs/wald_ref.py --at cdf 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(0.001)` | 4.8791443010843687e-219 | 4.879144301085083e-219 | 660 | 500 | minor |
  | `logsf(0.001)` | -4.879144301084398e-219 | -4.879144301085083e-219 | 632 | 500 | minor |
  | `cdf(0.0014)` | 6.35649036920343e-157 | 6.356490369204056e-157 | 444 | 358 | minor |

- **Fix direction:** a direct left-tail form, e.g. cdf = exp(-z1**2/2)
  (erfcx(-z1/sqrt2) + erfcx(z2/sqrt2))/2 with -z1 > 0. This also removes
  F3's cancellation.

## RuntimeWarnings

- pdf: for x < ~1e-103, `x**3.0` underflows to 0. `1/sqrt(0) = inf` is then
  multiplied by `exp(-(x-1)**2/(2x)) = 0`, which gives nan with "divide by
  zero" and "invalid value in multiply" warnings. The true pdf underflows
  there (reference 0), so this is not an accuracy finding, but returning nan
  instead of 0 is wrong. Measured at x = 1e-110 ... 1e-300. "overflow in power"
  at x > 5.6e102 is harmless for wald: the true pdf has underflowed long before.
- logpdf: "overflow encountered in square" at x > 1.34e154 (F2).
- cdf/logcdf/sf/logsf: "underflow encountered in exp" on accurate tail
  results (spurious); "divide by zero encountered in log1p" where sf
  underflows (x >~ 1416, true sf subnormal or 0).
- isf: Boost "Unable to locate solution in a reasonable time ... Current
  best guess is %1%" (unformatted message) for every F1 point.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid (x from 1.4e-3 to 1416 where pdf is normal), x**3 overflow threshold | 123 ulp (cond 332), within conditioning; bulk 1.73 ulp |
| logpdf | full grid below x = 1.34e154 | 0.97 ulp |
| cdf | bulk and right tail (left tail is F4) | 1.5 ulp in the bulk |
| logcdf | left tail and bulk (right tail is F3) | 2.63 ulp in the left tail, 25.6 ulp (minor, F3 onset) in the bulk |
| sf, logsf | left tail / bulk | sf 0 ulp in the left tail; logsf 5.04 ulp in the bulk |
| ppf | full q grid including 1e-300 and 1 - 3e-16 | 21 ulp, within conditioning (cond up to 1.1e8 near q -> 1) |
| isf | bulk and q -> 1 | 2.52 ulp |
| wrappers | `wald.<m>(x)` vs `invgauss.<m>(x, 1.0)` for all eight methods, 2001 x in [1e-4, 1e4], 700 q | bit-identical |

## Suspected, not confirmed

- pdf `x**3.0` overflow (invgauss F7) cannot cause a finding for wald,
  because pdf(x) underflows for x >~ 1500, long before 5.6e102.
- invgauss F4, F5, F8, F9, F10 (Boost nan fallback, small-mu `2/mu`
  problems) need mu << 1 and are not reachable with mu = 1.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| none | - | - | all eight methods are implemented (via invgauss) |

## Shared code

All four findings are in `invgauss_gen` and apply to invgauss at mu = 1
(see `reports/invgauss.md`): F1 = invgauss F2 (Boost `scu._invgauss_isf`,
a `scipy.special` Boost ufunc), F2 = invgauss F6, F3 = invgauss F1, F4 = the
minor left-tail cdf error that invgauss lists under "Checked and fine". wald's
own wrappers add no error. Fixing invgauss fixes wald.

## Existing GitHub issues

No wald-specific issues found. The relevant invgauss items (see reports/invgauss.md):
- PR #16790 (merged): ENH: stats.invgauss: use Boost implementation of ppf/isf - introduced the Boost isf behind F1 (invgauss F2)
- PR #13665 (closed, superseded by #16790): ENH: Implement ppf and isf for invgauss - reported far-tail isf errors, same tail as F1
- F2, F3, F4: none found

(web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `wald.isf(1e-300)` | 4.849712021959747e+233 | 1361.4454371385304 | verified | ref by findroot on log of erfc-form sf; checked with quad sf(x_ref)/q - 1 = 4e-99; Boost warning emitted, line 5116 matches |
| F1 | `wald.isf(1e-100)` | 4.43905398835726e+34 | 443.76389884560625 | verified | 4.5e47 ulp, loss |
| F1 | `wald.isf(1e-70)` | 91476.55228318354 | 306.7100384323816 | verified | 1.3e18 ulp, loss |
| F1 | `wald.isf(1e-67)` | 201.95731938668308 | 293.03036473289705 | verified | 1.4e15 ulp, severe |
| F1 | `wald.isf(1e-65)` | 283.7925324223683 | 283.91408788350248 | verified | 1.9e12 ulp, severe |
| F2 | `wald.logpdf(1.4e154)` | -inf | -7.0e+153 | verified | direct log of docstring pdf; `(x/mu - 1)**2` at line 5079 overflows for x > sqrt(1.8e308) = 1.34e154 |
| F2 | `wald.logpdf(1e200)` | -inf | -5.0e+199 | verified | loss |
| F3 | `wald.sf(1000.0)` | 4.869434437047494e-222 | 4.8694344366891734e-222 | verified | 3.3e5 ulp; ref by mp.quad of pdf (scaled integrand), agrees with erfc form to 2e-96; cancellation in `log1p(-exp(b - a))`, line 5095 |
| F3 | `wald.logcdf(1000.0)` | -4.869434437102362e-222 | -4.8694344366891734e-222 | verified | 3.8e5 ulp; ref log1p(-sf_quad) |
| F3 | `wald.sf(300.0)` | 2.960519417136735e-69 | 2.9605194171325379e-69 | verified | 6.4e3 ulp |
| F3 | `wald.sf(100.0)` | 4.043703566769238e-25 | 4.0437035667648971e-25 | verified | 4.8e3 ulp |

Verified by dist-accuracy-verifier: 3 of 3 verified.
