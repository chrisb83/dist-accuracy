# foldnorm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `foldnorm_gen`, lines 2666-2725
- **Implemented:** pdf, cdf, sf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf), ppf = numerical inversion of cdf, isf = generic (via ppf)
- **Related:** none sharing code (halfnorm is the c = 0 case but has its own class); helpers: `_norm_pdf` (line 362), `_norm_sf` (line 382, = `sc.ndtr(-x)`); special functions: sc.erf, sc.ndtr
- **Reference:** `refs/foldnorm_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (1.952125337355587), (0), (0.001), (1), (5), (30), (1000); extra pdf run with (37), (20)
- **Worst finding:** loss in cdf

## Summary

`sf` is accurate everywhere (max 546 ulp, within conditioning) and `pdf` is
accurate apart from a minor error of up to ~c**2/2 ulp for x << c caused by
rounding `x + c` and `x - c` before squaring. `cdf` is badly broken left of
the mode: it is computed as the sum of two O(1) `erf` values of opposite sign,
so it cancels (a) for x small compared with 1/c (any c > 0; 0.0 for
x < ~1e-15 at the example c), and (b) wherever x - c < about -5 (moderate
errors from x - c = -5, 0.0 once Phi(x - c) < eps, i.e. x - c < ~-8.3). 155
of 588 cdf points are losses. c = 0 is unaffected.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | c > 0, x small (x*c, x << 1, roughly cdf << erf(c/sqrt2)) | erf(x-c) + erf(x+c) cancel (pattern 5) | 0.0 for 1.19e-21 | loss |
| F2 | cdf | x - c < about -5 (left of the mode, any c >~ 5) | 1 + erf(x-c) cancels, erf instead of erfc/ndtr (patterns 1, 10) | 0.0 for 3.3e-185 | loss |
| F3 | pdf | c large (>~ 15), x << c | rounding of x + c and x - c amplified by the square in exp (pattern 5) | 275 ulp (c=30), 225 ulp (c=37) | minor |

### F1: cdf cancels for small x

- **Code:** `_cdf`, line 2703: `0.5 * (sc.erf((x - c)/sqrt_two) + sc.erf((x + c)/sqrt_two))`
- **Mechanism:** for small x the two terms are -erf((c - x)/sqrt2) and
  +erf((c + x)/sqrt2), each of size erf(c/sqrt2), while their sum is
  about x*pdf(0) = 2*x*phi(c). The relative error is about
  erf(c/sqrt2) / cdf ulp; once x*phi(c) drops below eps*erf(c/sqrt2) both
  terms round to the same magnitude and the result is 0.0 (pattern 5).
  For tiny c the terms are ~c and the result ~x, so the loss is about c/x ulp.
  c = 0 is exact (both terms equal erf(x/sqrt2)).
- **Regime:** any c > 0, x such that cdf(x) << erf(c/sqrt2). Severe below
  x ~ 1e-6 for c ~ 1...5, total loss for x < ~1e-15 (c ~ 2) and for
  x < ~c*1e-16 when c is tiny.
- **Reproducer:** `python accuracy.py refs/foldnorm_ref.py --at cdf 1e-12 1.952125337355587`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-20, 1.952125337355587)` | 0.0 | 1.1869619671695304e-21 | inf | 1 | loss |
  | `cdf(1e-12, 1.952125337355587)` | 1.187383524836605e-13 | 1.1869619671695306e-13 | 1.6e+12 | 1 | severe |
  | `cdf(1e-10, 5)` | 2.7755575615628914e-16 | 2.9734390294685953e-16 | 3.0e+14 | 1 | severe |
  | `cdf(1e-17, 0.001)` | 7.914675859144182e-18 | 7.978841618606847e-18 | 3.6e+13 | 1 | severe |
  | `cdf(1e-10, 0.001)` | 7.978841610849219e-11 | 7.978841618606847e-11 | 4.4e+06 | 1 | severe |
  | `cdf(1e-06, 1)` | 4.839414490254157e-07 | 4.839414490382867e-07 | 1.2e+05 | 1 | moderate |
  | `cdf(0.5, 5)` | 3.378683562260676e-06 | 3.378683562264173e-06 | 4.66e+03 | 2.38 | moderate |
  | `cdf(0.001, 1)` | 0.00048394144903818326 | 0.00048394144903827867 | 888 | 1 | minor |

  Losses (0.0) at all x from 1e-300 up to roughly 1e-16 for c = 0.001, 1,
  1.95, 5 (29-30 points each).
- **Fix direction:** for small x use a form without the O(1) terms, e.g.
  cdf = 2*phi(c) * integral_0^x exp(-t**2/2) cosh(c t) dt evaluated by a
  series in x (or Owen's-T / `erf` difference identities with
  `erf(a) - erf(b)` computed via a dedicated difference routine).

### F2: cdf loses everything left of the mode for large c

- **Code:** `_cdf`, line 2703 (same expression).
- **Mechanism:** when x - c < about -5, erf((x - c)/sqrt2) = -1 + 2*Phi(x - c)
  is close to -1 and erf((x + c)/sqrt2) is 1 (or 1 - tiny); the sum keeps only
  the digits of 2*Phi(x - c) beyond the leading 1, i.e. the relative error is
  about eps/Phi(x - c) (pattern 1). Once Phi(x - c) < eps/2 (x - c < ~-8.3)
  the result is 0.0, although cdf = Phi(x - c) - Phi(-x - c) is a normal
  double down to x - c ~ -37.5. `erf` is used where the complementary
  `erfc`/`ndtr` is needed (pattern 10).
- **Regime:** x - c < ~-5 (needs c > ~5). Moderate at x - c = -5, loss for
  x - c < ~-8.3. For c = 30 this is x in [0, ~21.7]; for c = 1000 every x in
  [~962, ~992].
- **Reproducer:** `python accuracy.py refs/foldnorm_ref.py --at cdf 1 30`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(1.0, 30.0)` | 0.0 | 3.28978526670438e-185 | inf | 29 | loss |
  | `cdf(965.0, 1000.0)` | 0.0 | 1.1249107064724062e-268 | inf | 3.38e+04 | loss |
  | `cdf(25.0, 30.0)` | 2.8665157186802404e-07 | 2.866515718791939e-07 | 1.8e+05 | 130 | moderate |
  | `cdf(995.0, 1000.0)` | 2.8665157186802404e-07 | 2.866515718791939e-07 | 1.8e+05 | 5.16e+03 | moderate |

  37 loss points for c = 30 (overlapping with F1 at tiny x), 1 for c = 1000.
- **Fix direction:** `ndtr(x - c) - ndtr(-x - c)` (the sf-style form, both
  terms in the lower tail, no cancellation except in the F1 regime), with
  `0.5*(erf+erf)` kept only where both erf arguments are small.

### F3: pdf loses up to ~c**2 ulp for x << c

- **Code:** `_pdf`, line 2699: `_norm_pdf(x + c) + _norm_pdf(x-c)`, with
  `_norm_pdf(z) = np.exp(-z**2/2.0) / _norm_pdf_C` (line 363).
- **Mechanism:** `x + c` and `x - c` are rounded (absolute error up to
  eps*c/2) before squaring; the exponent z**2/2 ~ c**2/2 then carries an
  absolute error ~c**2*eps/2, i.e. up to ~c**2/2 ulp in the result. The
  derivative with respect to x is small here (cond (arg) ~ x*c), so the
  error is beyond conditioning; it looks like a 1-ulp perturbation of c
  (cond (shapes) 897 at c = 30). Pattern 5 (information lost when a small
  argument is added to a large shape before a large cancellation inside
  exp(-(x**2 + c**2)/2 + c x)).
- **Regime:** c >~ 15 and x < c - few, while the pdf stays normal
  (c < ~37.6 at x ~ 0). Bounded by ~c**2/2 <= ~700 ulp.
- **Reproducer:** `python accuracy.py refs/foldnorm_ref.py --at pdf 0.1 30`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(0.1, 30)` | 2.952435094968932e-195 | 2.9524350949687516e-195 | 275 | 2.98 | minor |
  | `pdf(0.1, 37)` | 8.537303017835156e-297 | 8.53730301783473e-297 | 225 | 3.69 | minor |
  | `pdf(1e-06, 37)` | 4.240013105949572e-298 | 4.24001310594938e-298 | 204 | 1.37e-09 | minor |
  | `pdf(0.1, 20)` | 4.133458580406912e-87 | 4.13345858040678e-87 | 144 | 1.92 | minor |

- **Fix direction:** for x < c use
  `exp(-(x - c)**2/2) * (1 + exp(-2*c*x)) / sqrt(2*pi)` with
  `(x - c)**2` replaced by `x*x - 2*c*x + c*c` evaluated so that the c**2/2
  term is exact relative to the others (or `exp(-c*c/2)*exp(-x*x/2)*cosh(c*x)`
  split as in the docstring formula when no overflow is possible).

## RuntimeWarnings

pdf: "overflow encountered in square" (x**2 for x > ~1.3e154; result 0.0 is
correct since the true value underflows), "underflow encountered in exp /
divide / square" in the right tail and for large c. These accompany correct
results (0 where the true value is below the normal range); spurious. No
warnings recorded for cdf or sf.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| sf | full grid, all parameter sets incl. far right tail and x ~ c | 546 ulp, within conditioning (cond up to 3.8e4); bulk 9.32 ulp |
| pdf | full grid, all parameter sets; right tail; c = 0, 1e-3, 1e3 | 408 ulp within conditioning (cond 1.44e3); beyond-conditioning errors only in F3 |
| cdf | c = 0 (all x); right tail and x >= c for all c | 1.02 ulp (c = 0); 0.5 ulp high tail |

## Suspected, not confirmed

none. (The plotting option `--plot` of `accuracy.py` terminated the process
silently on this machine; the run was done without the plot. This does not
affect the measurements.)

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `_norm_logpdf(x - c) + log1p(exp(-2*c*x))` (x, c >= 0) | pdf underflows: x - c > ~38.6, or x << c with c > ~37.6. `logpdf(50, 1)` = -inf, true -1201.4189385332047 |
| logcdf | log(cdf) | `log_ndtr(x - c) + log1mexp(log_ndtr(-x - c) - log_ndtr(x - c))`, plus a small-x series (F1) | inherits F1/F2 and cdf underflow for x - c < ~-37.5. `logcdf(1, 30)` = -inf, true -424.78741990973015 |
| logsf | log(sf) | `logaddexp(log_ndtr(c - x), log_ndtr(-x - c))` | sf underflows for x - c > ~37.5. `logsf(50, 1)` = -inf, true -1205.3111748916654 |
| ppf | root finding on cdf | no closed form; root finding on an accurate cdf (F1/F2 fixed), start at `c + ndtri(q)` for large c, `q/(2 phi(c))`-type start for small q | inherits F1/F2: `ppf(1e-20, 1.952125337355587)` = 0.0, true 8.424869773920675e-20 |
| isf | generic (via ppf of 1 - q) | root finding on the (accurate) sf, start at `c + ndtri(1 - q)`-type values | q < eps: `isf(1e-20, 1)` = 10.0, true 10.26234008990587 |

## Shared code

none. `_norm_pdf`, `_norm_sf`, `sc.erf` and `sc.ndtr` are accurate for their
own results; the problems come from how `foldnorm` combines them.

## Existing GitHub issues

None found about F1-F3 (searched via web search; `gh` not available). For context:
- PR #18207 (merged, 1.11.0): introduced the current erf-sum `cdf` formula on line 2703 to improve the left tail; that formula is the source of F1 and F2.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e-20, 1.952125337355587)` | 0.0 | 1.1869619671695304e-21 | verified | loss; ref = mp.quad of docstring pdf on [0, x], 100/200 digits agree |
| F1 | `cdf(1e-12, 1.952125337355587)` | 1.187383524836605e-13 | 1.1869619671695306e-13 | verified | 1.6e12 ulp |
| F1 | `cdf(1e-10, 5)` | 2.7755575615628914e-16 | 2.9734390294685953e-16 | verified | 3.0e14 ulp |
| F1 | `cdf(1e-17, 0.001)` | 7.914675859144182e-18 | 7.978841618606847e-18 | verified | 3.6e13 ulp |
| F1 | `cdf(1e-06, 1)` | 4.839414490254157e-07 | 4.839414490382867e-07 | verified | 1.2e5 ulp (moderate, as reported) |
| F2 | `cdf(1, 30)` | 0.0 | 3.2897852667043802e-185 | verified | loss; ref = erfc(-(x-c)/sqrt2)/2 - erfc((x+c)/sqrt2)/2 (stable 100/200 digits); quad of pdf agrees at 200 digits |
| F2 | `cdf(965, 1000)` | 0.0 | 1.1249107064724062e-268 | verified | loss; erfc route stable to 35 digits; quad of pdf with dense splitting agrees to 8 digits (quad error), enough for the verdict |
| F2 | `cdf(25, 30)` | 2.8665157186802404e-07 | 2.866515718791939e-07 | verified | 1.75e5 ulp (moderate) |
| F2 | `cdf(995, 1000)` | 2.8665157186802404e-07 | 2.866515718791939e-07 | verified | 1.75e5 ulp (moderate) |

Mechanisms checked against source: both findings cite line 2703, `0.5 * (sc.erf((x - c)/sqrt_two) + sc.erf((x + c)/sqrt_two))`, which matches SciPy 1.18.1; the cancellation explanations (F1: two O(erf(c/sqrt2)) terms of opposite sign; F2: erf((x-c)/sqrt2) near -1 instead of using ndtr/erfc) are consistent with the code. F3 (minor) was not selected for verification.

Verified by dist-accuracy-verifier: 2 of 2 verified.
