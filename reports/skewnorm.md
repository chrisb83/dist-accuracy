# skewnorm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `skewnorm_gen`, lines 9724-9946
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf (cdf/ppf/isf via Boost `skew_normal` ufuncs `scu._skewnorm_cdf/_ppf/_isf`; cdf falls back to the generic `rv_continuous._cdf` (quad) for small values; sf = `_cdf(-x, -a)`)
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** none; helpers: `_norm_pdf`, `_norm_logpdf`, `_norm_cdf` (sc.ndtr), `_norm_logcdf` (sc.log_ndtr), generic `rv_continuous._cdf` (`integrate.quad`); special functions: scu._skewnorm_cdf, scu._skewnorm_ppf, scu._skewnorm_isf, sc.ndtr, sc.log_ndtr
- **Reference:** `refs/skewnorm_ref.py`, 100 digits (cdf from a cancellation-free Owen's T integral, quantiles by Newton on log F / log S); self-checks: PASS
- **Parameter sets:** (4), (1), (10), (100), (1e4), (1e-8), (-4)
- **Worst finding:** loss in cdf, sf, ppf, isf and logpdf

## Summary

pdf is accurate everywhere (within the conditioning of `exp(-x**2/2)` and
`ndtr(a*x)`), and logpdf is accurate except where `_norm_logpdf` squares x
into overflow. The tail of the cdf on the "short" side (left tail for a > 0,
and by symmetry the right tail of sf for a < 0) is badly broken. Boost's
`Phi(x) - 2 T(x, a)` cancels there (up to 3.9e4 ulp while cdf >= 1e-6). Below
1e-6 SciPy replaces the result with the generic `integrate.quad` cdf, whose
absolute tolerance of 1.5e-8 gives no relative accuracy: errors run from 1e7
to 4.4e15 ulp, and the result is 0.0 for large a (including `cdf(0, 1e6)`).
`ppf` (a > 0) and `isf` (a < 0) call Boost's quantile directly, so SciPy's
fallback does not reach them. They lose digits from q ~ 1e-6 on (1.4e4 ulp),
and from q ~ 1e-26 on they return values off by a factor of 2 to 7000
(e.g. `ppf(1e-300, 1e4)` = -22.8 vs -0.00367).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | a > 0, cdf < 1e-6 (sf: a < 0, sf < 1e-6); also tiny a and x = 0 for a >~ 3e5 | generic `integrate.quad` fallback with absolute tolerance 1.5e-8 (patterns 7, 10) | 0.0 for 5.96e-29; 4.4e15 ulp | loss |
| F2 | ppf, isf | ppf: a > 0, q <~ 1e-5; isf: a < 0, q <~ 1e-5 | Boost quantile iterates on Boost's cancelling cdf, SciPy's fallback not used (patterns 8, 10) | -22.8 for -0.00367 (4.4e19 ulp) | loss |
| F3 | logpdf | 1.34e154 < abs(x) < 1.9e154, any a | `x**2` overflows in `_norm_logpdf` (pattern 3; INDEX item 2) | -inf for -1.125e308 | loss |
| F4 | cdf, sf | a > 0, 1e-6 <= cdf << Phi(x) (sf mirrored, a < 0) | Boost `Phi(x) - 2 T(x, a)` cancels (pattern 5) | 3.87e4 ulp | moderate |

### F1: the small-cdf fallback to generic quad is inaccurate or returns 0

- **Code:** `_cdf`, lines 9777-9779:
  `i_small_cdf = (cdf < 1e-6) & (a > 0)`;
  `cdf[i_small_cdf] = super()._cdf(x[i_small_cdf], a[i_small_cdf])`;
  `_sf` (line 9788) is `self._cdf(-x, -a)`, so sf for a < 0 takes the same path.
  `super()._cdf` is `rv_continuous._cdf_single` =
  `integrate.quad(self._pdf, -inf, x)[0]`.
- **Mechanism:** this fallback is the fix for scipy/scipy#7746 (Boost's
  left tail when a > 0). `quad` runs with its default absolute tolerance
  `epsabs = 1.49e-8`. Every result below 1e-6 is therefore accepted after
  very few refinements, so its relative accuracy is arbitrary: QUADPACK
  returns 3.40e-203 with an error estimate of 6.7e-203 at
  `cdf(-0.3, 100)`. When a is large the integrand is concentrated in a
  window of width ~1/(a^2 |x|) next to x, which the transformed infinite
  interval does not resolve. quad then sees only zeros and returns 0.0
  (patterns 10 and 7). The threshold also applies to every a > 0, including
  tiny a, where Boost has nothing to cancel and is accurate.
  `scu._skewnorm_cdf(-6, 0, 1, 1e-8)` = 9.8658759655917e-10 (0.3 ulp),
  which the fallback replaces with a value 6.2e7 ulp off. For a >~ 3.2e5,
  cdf(0) = 1/2 - atan(a)/pi < 1e-6, so even x = 0 takes the fallback and
  returns 0. (Boost alone gives 3.1830988617143063e-07 there, 2.5e4 ulp off,
  see F4.) Fixing #7746 removed Boost's cancellation (F4) below 1e-6, but the
  replacement is often worse.
- **Regime:** a > 0 and true cdf < 1e-6: x <~ -4.75 for a = 1e-8, x <~ -3.1
  for a = 1, x <~ -1 for a = 4, x <~ -0.033 for a = 100, x <~ -1.7e-4 for
  a = 1e4; for a >~ 3.2e5 also x = 0. Loss (0.0) for a = 1e4 at every
  measured x <= -2e-4. The same holds for sf with a < 0 and x > 0. The
  run measured 6 loss, 38 severe, 8 moderate and 2 minor points on this
  path (cdf and sf together).
- **Reproducer:** `python accuracy.py refs/skewnorm_ref.py --at cdf -0.3 100`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-0.001, 10000)` | 0.0 | 5.963833185488469e-29 | inf | 102 | loss |
  | `cdf(-0.0002, 10000)` | 0.0 | 6.774600339577657e-07 | inf | 5.36 | loss |
  | `cdf(0.0, 1e6)` | 0.0 | 3.1830988618368455e-07 | inf | 0 | loss |
  | `cdf(-0.3, 100)` | 3.3966870168751285e-203 | 1.2446926326895438e-201 | 4.4e+15 | 902 | severe |
  | `cdf(-3.0, 10)` | 1.4482520126257573e-202 | 1.4322262922090077e-202 | 5.0e+13 | 911 | severe |
  | `cdf(-9.0, 4)` | 5.613436559644227e-304 | 5.612314673715925e-304 | 9.0e+11 | 1.38e+03 | severe |
  | `cdf(-3.2, 1)` | 4.721585353748802e-07 | 4.721585457232438e-07 | 9.9e+07 | 22.2 | severe |
  | `cdf(-6.0, 1e-08)` | 9.865876102401416e-10 | 9.86587596559167e-10 | 6.2e+07 | 37 | severe |
  | `cdf(-1.25, 4)` | 4.6124825440398446e-09 | 4.6124825762980074e-09 | 3.1e+07 | 28.4 | severe |
  | `sf(9.0, -4)` | 5.613436559644227e-304 | 5.612314673715925e-304 | 9.0e+11 | 1.38e+03 | severe |
  | `sf(1.25, -4)` | 4.6124825440398446e-09 | 4.6124825762980074e-09 | 3.1e+07 | 28.4 | severe |

- **Fix direction:** use the cancellation-free Owen's T identity on the
  small side, `F(x; a) = (1/pi) * int_a^inf exp(-x^2 (1+t^2)/2)/(1+t^2) dt`
  for x <= 0, a >= 0 (what the reference module uses, after the substitution
  t = tan(theta), so the integral runs over a finite interval). A dedicated
  quadrature with relative tolerance (`epsabs=0`) and the scale 1/(a^2|x|)
  also works. At minimum, drop the fallback for small a, where Boost is
  accurate.

### F2: ppf (a > 0) and isf (a < 0) fail in the small tail

- **Code:** `_ppf`, line 9783: `scu._skewnorm_ppf(x, 0.0, 1.0, a)`;
  `_isf`, line 9791: `scu._skewnorm_isf(x, 0.0, 1.0, a)`.
- **Mechanism:** Boost's skew-normal quantile refines an initial guess with
  Newton/Halley iterations on Boost's own cdf, `Phi(x) - 2 T(x, a)`. For
  a > 0 in the left tail that cdf has absolute error ~eps*Phi(x) (F4), and
  F1's fallback is not applied inside the C++ code. Once q is comparable to
  that error the iteration converges to wherever Boost's cdf crosses q:
  `scipy.cdf(scipy.ppf(1e-20, 4), 4)` = 5.6e-79 instead of 1e-20. Boost
  computes isf through the mirror image (`isf(q, a) = -ppf(q, -a)`, exactly
  equal in the measurements), so isf fails for a < 0 in the same way
  (patterns 8 and 10).
- **Regime:** a > 0 for ppf, a < 0 for isf, small q: about 1e4 ulp at
  q = 1e-6, 1e5-1e7 ulp at q = 1e-8 ... 1e-10, 1e10 ulp at q = 3.5e-14.
  From q ~ 1e-26 (a >= 4) or q ~ 1e-39 (a = 1) the result is a different
  number (loss). For large a the error is worst: `ppf(q, 1e4)` mostly returns values between -7
  and -35 instead of about -0.001 to -0.004. a = 1e-8 and the opposite tails
  are fine (2.8 ulp). Measured: ppf 79 loss + 41 severe; isf (a = -4)
  19 loss + 5 severe.
- **Reproducer:** `python accuracy.py refs/skewnorm_ref.py --at ppf 1e-10 4`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 10000)` | -22.796552107417966 | -0.003669367846261095 | 2.8e+19 | 0.000742 | loss |
  | `ppf(1.2115276586286942e-26, 100)` | -1.1843301000538753 | -0.09929745269413771 | 4.9e+16 | 0.00995 | loss |
  | `ppf(1e-300, 4)` | -36.24724793959885 | -8.951013837577287 | 1.4e+16 | 0.000733 | loss |
  | `ppf(1e-20, 4)` | -4.4958822473233875 | -2.1454006401974937 | 4.9e+15 | 0.0125 | loss |
  | `isf(1e-300, -4)` | 36.24724793959885 | 8.951013837577287 | 1.4e+16 | 0.000733 | loss |
  | `ppf(1e-50, 1)` | -11.727552545869624 | -10.420452200803098 | 5.6e+14 | 0.00456 | severe |
  | `ppf(3.4807005884285624e-14, 4)` | -1.6980337036275437 | -1.698004398579527 | 7.8e+10 | 0.0196 | severe |
  | `ppf(1e-10, 4)` | -1.4097249855125822 | -1.4097249651707064 | 6.5e+07 | 0.0281 | severe |
  | `isf(1e-10, -4)` | 1.4097249855125822 | 1.4097249651707064 | 6.5e+07 | 0.0281 | severe |
  | `ppf(1e-08, 4)` | -1.21549428516743 | -1.2154942850448993 | 4.5e+05 | 0.0371 | moderate |
  | `ppf(1e-06, 4)` | -0.9892283250886824 | -0.9892283250855115 | 1.44e+04 | 0.0544 | moderate |

- **Fix direction:** invert an accurate tail cdf (the F1 fix) in log space,
  e.g. Newton on `log F(x) - log q`, which converges monotonically because
  the skew-normal cdf is log-concave. A starting point for a > 0 is the
  asymptote `log F ~ -(1+a^2) x^2/2 - log(pi a (1+a^2) x^2)`.

### F3: logpdf overflows in `x**2` for 1.34e154 < |x| < 1.9e154

- **Code:** `_logpdf`, line 9769: `np.log(2)+_norm_logpdf(x)+_norm_logcdf(a*x)`;
  `_norm_logpdf(x)` = `-x**2 / 2.0 - _norm_pdf_logC` (line 367).
- **Mechanism:** `x**2` overflows to inf before the division by 2, although
  `-x**2/2` is still a normal double (pattern 3). This is the shared
  `_norm_logpdf` issue (INDEX item 2). The `log_ndtr(a*x)` term adds
  `-(a x)^2/2` in the left tail for a > 0, so there the true value already
  exceeds the double range. The measured losses are at x = +1.5e154 for
  every a > 0 and at x = -1.5e154 for a <= 1e-8 or a < 0.
- **Regime:** 1.34e154 < |x| < 1.9e154 (where the result is still finite).
- **Reproducer:** `python accuracy.py refs/skewnorm_ref.py --at logpdf 1.5e154 4`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.5e+154, 4)` | -inf | -1.1250000000000002e+308 | inf | 2 | loss |
  | `logpdf(-1.5e+154, -4)` | -inf | -1.1250000000000002e+308 | inf | 2 | loss |
  | `logpdf(-1.5e+154, 1e-08)` | -inf | -1.1250000000000002e+308 | inf | 2 | loss |

- **Fix direction:** `-(x/2)*x` or `-0.5*x*x` in `_norm_logpdf`
  (shared helper).

### F4: Boost's cdf cancels between 1e-6 and Phi(x)

- **Code:** `_cdf`, line 9774: `scu._skewnorm_cdf(x, 0.0, 1.0, a)`, kept
  wherever the result is >= 1e-6 (line 9778).
- **Mechanism:** Boost evaluates `Phi(x) - 2*owens_t(x, a)`. For a > 0 and
  x <= 0 the cdf is much smaller than Phi(x), so the subtraction loses about
  `Phi(x)/F(x)` ulp times the relative error of `owens_t` (pattern 5). The
  1e-6 switch (F1) bounds the cancellation factor at `Phi(x)/1e-6`, which is
  up to 5e5. Near x = 0 with large a, `F(0) = 1/2 - atan(a)/pi` is small
  while `Phi(0) = 1/2`, so even the origin loses 952 ulp at a = 1e4 and
  2.5e4 ulp at a = 1e5.
- **Regime:** a > 0, x < ~0, 1e-6 <= F(x) << Phi(x); sf mirrored (a < 0,
  x > 0). Measured on this path: 10 moderate and 139 minor points (cdf), 1
  moderate and 2 minor (sf).
- **Reproducer:** `python accuracy.py refs/skewnorm_ref.py --at cdf -0.033 100`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-0.033, 100)` | 1.0148196133430432e-06 | 1.0148196133343312e-06 | 3.87e+04 | 12.5 | moderate |
  | `cdf(-0.0001, 10000)` | 6.6476126903136645e-06 | 6.647612690359757e-06 | 3.12e+04 | 1.9 | moderate |
  | `cdf(0.0, 1e5)` | 3.1830988617143063e-06 | 3.1830988617318035e-06 | 2.48e+04 | 0 | moderate |
  | `cdf(-3.0, 1)` | 1.8222246958010929e-06 | 1.8222246957988037e-06 | 5.66e+03 | 19.7 | moderate |
  | `cdf(-0.75, 4)` | 5.458936393709646e-05 | 5.458936393712078e-05 | 2.01e+03 | 11.2 | moderate |
  | `sf(0.75, -4)` | 5.458936393709646e-05 | 5.458936393712078e-05 | 2.01e+03 | 11.2 | moderate |
  | `cdf(1e-300, 10000)` | 3.18309885122825e-05 | 3.183098851227577e-05 | 952 | 0 | minor |

- **Fix direction:** same as F1: for x <= 0, a > 0 use the positive integral
  `(1/pi) int_a^inf ...` (or Owen's identity
  `T(h, a) = (Phi(h) + Phi(ah))/2 - Phi(h) Phi(ah) - T(ah, 1/a)` arranged so
  that no subtraction of nearly equal terms remains). At x = 0 use
  `atan2(1, a)/pi` exactly.

## RuntimeWarnings

- pdf: `overflow encountered in square` (from `_norm_pdf`, `exp(-x**2/2)`)
  for |x| > 1.34e154. The result 0.0 is correct, because the true pdf is
  far below the double range. `underflow encountered in exp/multiply/divide`
  in the far tails: spurious, results correct.
- logpdf: `overflow encountered in square` for |x| > 1.34e154. The result is
  wrong only in 1.34e154-1.9e154 (F3); beyond that the true value is below
  -1.8e308 anyway.
- cdf/sf: `underflow encountered in exp/divide` comes from the quad fallback
  evaluating the pdf deep in the tail. It accompanies the inaccurate F1
  results and also the correct 0.0 for subnormal true values.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, all 7 parameter sets, including ndtr tail range of a*x | 514 ulp, within conditioning (cond 1.09e3; ndtr / exp(-x^2/2) argument sensitivity) |
| logpdf | full grid except F3 window | 2.6 ulp |
| cdf | a = -4 (large side of the tail), right tail for all a | 190 ulp, within conditioning (cond 901) |
| sf | a > 0 everywhere; a = -4 left side | 530 ulp at sf(30, 1e-8), within conditioning (cond 901) |
| ppf | bulk and q -> 1 for all a; all q for a = -4 and a = 1e-8 | bulk 24 ulp; q -> 1 up to 1.9e11 ulp, within conditioning in q; `ppf(0.01, 1e4)` 22.4 ulp (minor, no clear mechanism, Boost iteration tolerance) |
| isf | bulk and q -> 1 for all a; all q for a > 0 | bulk 28 ulp; q -> 1 within conditioning in q |
| all | a = 1e-8 (near-normal, Owen's T with tiny a) | <= 2.8 ulp for ppf/isf; cdf/sf only F1 |

## Suspected, not confirmed

- a == 0 branch in pdf/logpdf (`apply_where(a == 0, ...)`) is an exact
  equality test only; a = 1e-8 is accurate, so no gap is expected; a = 0 itself
  was not in PARAMS.
- Boost's `owens_t` for very large a (> 1e6) and moderate h was not
  measured separately; the 1e4 and 1e5 points show only the F4 cancellation.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | small side (x < 0, a > 0): `-(1+a^2) x^2/2 - log(pi) + log(int_a^inf exp(-x^2 (t^2 - a^2)/2)/(1+t^2) dt)` (the integral is O(1/(a x^2)) and representable); a < 0: `log_ndtr(x) + log1p(2 T(x, abs(a))/Phi(x))` with the ratio bounded by 1 | once cdf underflows: `logcdf(-40, -4)` = -inf vs -803.91529483319384; `logcdf(-12, 4)` = -inf vs -1234.3353000900688 |
| logsf | log(sf) | `logsf(x, a) = logcdf(-x, -a)` with the logcdf above | `logsf(40, -4)` = -inf vs -13612.742109076372; `logsf(12, -4)` = -inf vs -1234.3353000900688 |

## Shared code

- `_norm_logpdf` (`-x**2 / 2.0`) overflows for 1.34e154 < |x| < 1.9e154
  (F3); the same helper is behind INDEX item 2 (alpha, lognorm, ...).
- Generic `rv_continuous._cdf` (`integrate.quad(self._pdf, a, x)` with
  the default `epsabs = 1.49e-8`) has no relative accuracy for results
  below ~1e-8 (F1). Every distribution that falls back to it, or calls
  `super()._cdf`, inherits this. Here it is used deliberately as the
  "accurate" replacement.
- Boost `skew_normal` ufuncs `scu._skewnorm_cdf/_ppf/_isf` (only used by
  skewnorm): cdf cancels for a > 0 on the left side (F4) and the quantiles
  inherit it without SciPy's fallback (F2).
- `sc.ndtr`/`sc.log_ndtr` tail error (INDEX item 6) shows up in pdf and in
  sf for a > 0 (up to 514 / 530 ulp) but stays within conditioning.

## Existing GitHub issues

- #7746 (closed, PR #8501, 1.1.0): False CDF values for skew normal distribution - the fix introduced the `integrate.quad` tail fallback that F1 is about; the thread already noted that `Phi - 2T` cancels (F4) and that a full fix would avoid the subtraction
- PR #20643 (merged): Boost update fixing `skewnorm.ppf`; its thread reports Boost `_skewnorm_cdf` far-tail errors that the quantile inherits - relates to F2, F4
- boostorg/math #1120: upstream discussion of skew-normal cdf instability (links scipy/scipy#20124, not checked) - relates to F2, F4
(web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-0.001, 1e4)` | 0.0 | 5.9638331854884692e-29 | verified | reference by direct `mp.quad` of the docstring pdf with breakpoints at the 1/((1+a^2)\|x\|) scale; 100 vs 200 digits agree to 1e-92 |
| F1 | `cdf(0.0, 1e6)` | 0.0 | 3.1830988618368457e-07 | verified | equals atan(1/a)/pi |
| F1 | `cdf(-0.3, 100)` | 3.3966870168751285e-203 | 1.2446926326827539e-201 | verified | 4.4e15 ulp |
| F1 | `cdf(-6.0, 1e-8)` | 9.865876102401416e-10 | 9.8658759655916695e-10 | verified | 6.2e7 ulp; Boost alone gives 9.8658759655917e-10, so the fallback at line 9778-9779 is the cause |
| F1 | `sf(9.0, -4)` | 5.613436559644227e-304 | 5.6123146736860773e-304 | verified | 9.0e11 ulp; same path through `_sf` = `_cdf(-x, -a)` |
| F2 | `ppf(1e-300, 1e4)` | -22.796552107417966 | -0.0036693678462610802 | verified | Newton on log F with the verifier's quad cdf, from the asymptotic start; cdf(ref)/q - 1 = 3e-99; 2.8e19 ulp (the Findings table says 4.4e19, the reproducer table 2.8e19) |
| F2 | `ppf(1e-20, 4)` | -4.4958822473233875 | -2.1454006401974938 | verified | Boost's own cdf at SciPy's result is 9.74e-21 (about q), the true cdf is 5.6e-79: this confirms the stated mechanism |
| F2 | `isf(1e-300, -4)` | 36.24724793959885 | 8.9510138375772522 | verified | `isf(q, a) == -ppf(q, -a)` bit-for-bit in SciPy |
| F2 | `ppf(1e-10, 4)` | -1.4097249855125822 | -1.4097249651707065 | verified | 6.5e7 ulp |
| F3 | `logpdf(1.5e154, 4)` | -inf | -1.1250000000000002e+308 | verified | `-x**2 / 2.0` at line 367 overflows |
| F3 | `logpdf(-1.5e154, 1e-8)` | -inf | -1.1250000000000003e+308 | verified | the verifier's value is 1 ulp from the report's (rounding of the log Phi term, about 1e292), not material |
| F4 | `cdf(-0.033, 100)` | 1.0148196133430432e-06 | 1.0148196133343312e-06 | verified | 3.87e4 ulp |
| F4 | `cdf(0.0, 1e5)` | 3.1830988617143063e-06 | 3.1830988617318034e-06 | verified | 2.48e4 ulp |

Verified by dist-accuracy-verifier: 4 of 4 verified.
