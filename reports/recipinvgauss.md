# recipinvgauss: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `recipinvgauss_gen`, lines 9534-9585
- **Implemented:** pdf (= `exp(_logpdf)`), logpdf, cdf, sf
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf), ppf = numerical inversion of cdf, isf = ppf(1 - q)
- **Related:** recipinvgauss is 1/X for X ~ invgauss (cdf(x, mu) = invgauss.sf(1/x, mu)), but it shares no code with `invgauss_gen`: invgauss works in log space with `log_ndtr`, recipinvgauss in linear space with `ndtr`. Helpers: `_norm_cdf` (= `sc.ndtr`); special functions: `sc.ndtr`, `np.exp`, `np.log`
- **Reference:** `refs/recipinvgauss_ref.py`, 400 digits (the cdf adds `log10((1/mu + x)/(2x)) + 20` digits to cover its own cancellation); self-checks: PASS
- **Parameter sets:** (0.6300426780936912), (1), (0.1), (0.01), (0.003), (0.001), (1000), (1e8), (1e160), (1e-200)
- **Worst finding:** loss in cdf, sf, logpdf, pdf

## Summary

For moderate mu (about 0.2 to 1e150) and away from the left tail, pdf, logpdf,
cdf and sf are accurate (bulk <= 22 ulp; tails within conditioning, apart from
minor ndtr-related excess of up to 520 ulp). The cdf and sf are written as
`Phi(.) -/+ exp(2/mu)*Phi(-z2)` in linear space. As a result, (F1) both
return nan for every x once `exp(2/mu)` overflows (mu < 2.818e-3). (F3) For
mu up to about 0.1, `ndtr(-z2)` underflows while the product is still normal, so
the left-tail cdf and right-tail sf are wrong by factors up to 8. (F2) For
all mu, the left-tail cdf is a difference of nearly equal terms, with up to
9.2e12 ulp of error and 0.0 for large mu. logpdf (and through it pdf) overflows in `mu**2`
for mu > 1.34e154 (nan everywhere), in `(1 - mu*x)**2` for mu*x > 1.34e154
(all mu), and underflows in `mu**2` for mu < 1.5e-154.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | mu < 2.818e-3, every x | `np.exp(2.0/mu)` overflows, `inf*Phi` = nan (pattern 3) | nan for 0.4937 | loss |
| F2 | cdf | left tail, cdf << Phi(-z1): x << 1/mu (all mu) and x << 1 (large mu) | `Phi(-z1) - exp(2/mu)*Phi(-z2)` cancels (patterns 1, 5), amplifies ndtr tail error | 0.0 for 8.0e-21; 9.2e12 ulp | loss |
| F3 | cdf, sf | mu <~ 0.1: cdf left tail, sf right tail, where ndtr(-z2) < 2.2e-308 but exp(2/mu)*ndtr(-z2) is normal | `ndtr(-z2)` underflows or is subnormal (pattern 4) | cdf 3.1e17 ulp (8x too large); sf 2.2e15 ulp | loss |
| F4 | logpdf, pdf | mu > 1.34e154 | `mu**2.0` overflows (pattern 3): nan, or the shape term is dropped | nan for -1.419; 5.0e8 ulp | loss |
| F5 | logpdf | mu*x > 1.34e154, any mu | `(1 - mu*x)**2.0` overflows (pattern 3) | -inf for -7e153 | loss |
| F6 | logpdf | mu < 1.49e-154 | `mu**2.0` subnormal or 0 (pattern 4) | -inf for -2.5e161; 5.0e10 ulp | loss |
| F7 | sf (cdf) | right tail x ~ 100 ... 1400, moderate mu | rounding in `isqx*trm1` and `sc.ndtr` far-tail error, amplified by ~z1**2 | 520 ulp (cond 496) | minor |
| F8 | pdf | large mu (>~1e50), small x where pdf ~ 1/sqrt(2 pi x) is huge | `exp(logpdf)` multiplies logpdf's rounding by abs(logpdf) | 127 ulp (cond 0.5) | minor |

### F1: cdf and sf are nan for mu < 2.818e-3

- **Code:** `_cdf` line 9576: `_norm_cdf(-isqx*trm1) - np.exp(2.0/mu)*_norm_cdf(-isqx*trm2)`; `_sf` line 9582: `_norm_cdf(isqx*trm1) + np.exp(2.0/mu)*_norm_cdf(-isqx*trm2)`.
- **Mechanism:** `np.exp(2.0/mu)` overflows to inf once 2/mu > 709.78. The
  product with `ndtr(-z2)` is inf*0 = nan (or inf*tiny = inf, giving
  `-inf`/`inf`). The true product `exp(2/mu)*Phi(-z2)` is always below
  `phi(z1)/z2` and is finite (pattern 3). This threshold is far more
  restrictive than invgauss F8 (mu < 1.1e-308), which adds `2/mu` in log space.
- **Regime:** every x for mu < 2/709.78 = 2.818e-3. Boundary checked:
  `cdf(355, 0.00282)` = 0.4976757100878011 (20.6 ulp, ok), `cdf(355, 0.00281)` = nan.
- **Reproducer:** `python accuracy.py refs/recipinvgauss_ref.py --at cdf 1000 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1000, 0.001)` | nan | 0.4936937444715336 | inf | 25.6 | loss |
  | `cdf(500, 0.001)` | nan | 3.1656625234045584e-111 | inf | 752 | loss |
  | `sf(1000, 0.001)` | nan | 0.5063062555284664 | inf | 24.9 | loss |
  | `sf(1e-10, 0.001)` | nan | 1.0 | inf | 0 | loss |
  | `cdf(1e+300, 1e-200)` | nan | 1.0 | inf | 0 | loss |

- **Fix direction:** use the exact identity `exp(2/mu)*phi(z2) = phi(z1)` to write
  `exp(2/mu)*Phi(-z2) = phi(z1)*sqrt(pi/2)*erfcx(z2/sqrt(2))`, so the large factor never forms (this also fixes F3).

### F2: cdf cancels in the left tail

- **Code:** `_cdf` line 9576: `_norm_cdf(-isqx*trm1) - np.exp(2.0/mu)*_norm_cdf(-isqx*trm2)`.
- **Mechanism:** with z1 = (1/mu - x)/sqrt(x) and z2 = (1/mu + x)/sqrt(x),
  the two terms agree to leading order whenever the cdf is small compared with Phi(-z1)
  (patterns 1, 5). Two sub-regimes:
  (a) x << 1/mu: both terms are about `phi(z1)/z1` and `phi(z1)/z2`, and the relative
  difference is (z2 - z1)/z2 = 2x/(1/mu + x). So the cancellation factor is about
  1/(2 mu x). It multiplies the relative errors of the two terms, which are already
  about z1**2 ulp (rounding in `isqx*trm1`, conditioning of ndtr, and the ndtr
  far-tail error from INDEX item 6). The total is roughly 1/(2 mu**3 x**2) ulp.
  (b) large mu and x << 1: `exp(2/mu)` is 1 and both Phi values are near 0.5;
  the true cdf is about `2*sqrt(x)*phi(0)` (the chi-square(1) limit). The result
  loses about 1/(0.8*sqrt(x)) ulp and is exactly 0.0 once `sqrt(x)` < ~1e-17.
- **Regime:** every mu, x well below the mode region. For mu = 1, the error grows from
  24 ulp at x = 0.1 to 3.9e5 ulp at x = 1e-3. For mu = 1e8, severe for x <= 1e-14.
  For mu = 1e160, the result is 0.0 for x <= 1e-40.
- **Reproducer:** `python accuracy.py refs/recipinvgauss_ref.py --at cdf 1e-14 1e8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-40, 1e160)` | 0.0 | 7.978845608028654e-21 | inf | 0.5 | loss |
  | `cdf(1e-30, 1e160)` | 8.326672684688674e-16 | 7.978845608028654e-16 | 1.96e14 | 0.5 | severe |
  | `cdf(1e-19, 1e8)` | 1.7885824676670868e-230 | 1.7922589638434179e-230 | 9.24e12 | 501 | severe |
  | `cdf(1e-14, 1e8)` | 7.01870669916893e-08 | 7.018706694281347e-08 | 3.1e6 | 0.566 | severe |
  | `cdf(1e-9, 1000)` | 1.7940521603663943e-225 | 1.7940521003997322e-225 | 1.51e8 | 501 | severe |
  | `cdf(0.001, 1)` | 4.8694344362720574e-222 | 4.8694344366892244e-222 | 3.9e5 | 501 | moderate |
  | `cdf(0.0025191850031604512, 0.6300426780936912)` | 1.3892949751283322e-221 | 1.3892949750949247e-221 | 1.08e5 | 501 | moderate |

- **Fix direction:** for z1 > 0, write `cdf = phi(z1)*sqrt(pi/2)*(erfcx(z1/sqrt(2)) - erfcx(z2/sqrt(2)))`.
  Evaluate the difference with a series in (z2 - z1) = 2*sqrt(x) when that is small relative to z1, and use `erf`-type forms in the large-mu, small-x limit.

### F3: ndtr(-z2) underflows while exp(2/mu)*ndtr(-z2) is normal

- **Code:** lines 9576 and 9582, factor `_norm_cdf(-isqx*trm2)`.
- **Mechanism:** z2**2 - z1**2 = 4/mu. For small mu, `ndtr(-z2)` drops below
  2.2e-308 (subnormal, then 0) while `phi(z1)` and the product with
  `exp(2/mu)` are still normal (pattern 4). In the cdf the subtrahend is
  lost or truncated. The cancellation of F2 then turns this into a result
  several times too large. In the sf the second (positive) term, a fixed
  fraction z1/z2 of the sum, is lost or truncated.
- **Regime:** mu <~ 0.1 (needs 2/mu >~ 20). cdf: left tail where z2 > ~37.5,
  e.g. x < ~7 for mu = 0.01, x < ~0.072 for mu = 0.1, x < ~170 for mu = 0.003.
  sf: right tail x >~ 1300 for mu = 0.01, x >~ 670 for mu = 0.003, x ~ 1400 for mu = 0.1.
  For mu = 1 both factors underflow together and nothing is lost.
- **Reproducer:** `python accuracy.py refs/recipinvgauss_ref.py --at cdf 6.5 0.01`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(0.07142857142857142, 0.1)` | 2.248754943762614e-302 | 3.1852146946495917e-304 | 3.13e17 | 701 | loss |
  | `cdf(6.5, 0.01)` | 9.583972602868364e-295 | 1.1684450909661516e-295 | 3.24e16 | 767 | loss |
  | `cdf(79.36507936507935, 0.003)` | 4.6794473474805656e-179 | 1.797594403238133e-179 | 7.22e15 | 662 | loss |
  | `cdf(166.66666666666666, 0.003)` | 1.9779307189572505e-38 | 1.3151519892414083e-38 | 2.27e15 | 252 | severe |
  | `sf(1400, 0.1)` | 2.248754943762614e-302 | 4.465657740578438e-302 | 2.24e15 | 700 | severe |
  | `sf(1400, 0.01)` | 8.54930556711888e-265 | 1.5960226041972563e-264 | 2.09e15 | 697 | severe |
  | `sf(666.6666666666666, 0.003)` | 1.9779307189572505e-38 | 2.6407094486730927e-38 | 1.13e15 | 251 | severe |

  Near the underflow edge the subnormal factor gives intermediate errors, e.g.
  `cdf(9, 0.01)`: 4.18e3 ulp (moderate) and `cdf(0.1, 0.1)`: 1.32e4 ulp (moderate).
  These are partly F2 amplification of the ndtr far-tail error.
- **Fix direction:** same as F1: `exp(2/mu)*Phi(-z2) = phi(z1)*sqrt(pi/2)*erfcx(z2/sqrt(2))`.

### F4: logpdf and pdf are nan (or drop a term) for mu > 1.34e154

- **Code:** `_logpdf` line 9568: `-(1 - mu*x)**2.0 / (2*x*mu**2.0) - 0.5*np.log(2*np.pi*x)`; `_pdf` line 9563: `np.exp(self._logpdf(x, mu))`.
- **Mechanism:** `mu**2.0` overflows to inf (pattern 3). If `(1 - mu*x)**2`
  also overflows (mu*x > 1.34e154), the quotient is inf/inf = nan. Otherwise it
  is finite/inf = 0, and the true shape term `-(1/mu - x)**2/(2x)` (about
  `-x/2 - 1/(2 mu**2 x)`) is dropped. That costs digits for moderate x and is a total loss at
  subnormal x, where `-1/(2 mu**2 x)` dominates.
- **Regime:** mu > 1.34e154: nan for x > 1.34e154/mu (for mu = 1e160, every
  x >= 1.4e-6); wrong for smaller x.
- **Reproducer:** `python accuracy.py refs/recipinvgauss_ref.py --at logpdf 1 1e160`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1, 1.5e154)` | nan | -1.4189385332046727 | inf | 0.705 | loss |
  | `pdf(1, 1e160)` | nan | 0.24197072451914334 | inf | 1 | loss |
  | `logpdf(5e-324, 1e160)` | 371.3241562260766 | -640.7101691090671 | inf | 1.58 | loss |
  | `pdf(1e-321, 1e160)` | 1.2629229649290634e+160 | 8.42452600134481e+157 | 6.7e17 | 4.51 | loss |
  | `logpdf(1.3e-6, 1e160)` | 5.857634613543719 | 5.8576339635437185 | 5.0e8 | 0.0854 | severe |
  | `logpdf(1e-10, 1e160)` | 10.593986931765556 | 10.593986931715555 | 2.13e4 | 0.0472 | moderate |

- **Fix direction:** divide before squaring: `-(1/mu - x)**2/(2*x)`, or `-((1 - mu*x)/mu)**2/(2*x)` (guarding 1/mu overflow for subnormal mu).

### F5: logpdf is -inf (or nan) for mu*x > 1.34e154

- **Code:** `_logpdf` line 9568, `(1 - mu*x)**2.0`.
- **Mechanism:** the square overflows to inf (pattern 3). The result is
  `-inf`, or nan when `2*x*mu**2` overflows too. The true value is about
  `-x/2`, a normal double. pdf is not affected because it underflows anyway.
- **Regime:** x > 1.34e154/mu for any mu (x > 1.4e154 for mu = 1;
  x > 1.4e146 for mu = 1e8; x > 1.3e254 for mu = 1e-100).
- **Reproducer:** `python accuracy.py refs/recipinvgauss_ref.py --at logpdf 1.4e154 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.4e154, 1)` | -inf | -7e+153 | inf | 1 | loss |
  | `logpdf(2.2220716924065444e+154, 0.6300426780936912)` | -inf | -1.1110358462032722e+154 | inf | 1 | loss |
  | `logpdf(1e+300, 1e8)` | nan | -5e+299 | inf | 1 | loss |
  | `logpdf(1e+300, 1e-100)` | -inf | -5e+299 | inf | 1 | loss |

- **Fix direction:** same as F4, `-(x - 1/mu)**2/(2*x)` = `-(x - 1/mu)*((x - 1/mu)/(2*x))`, which never forms a value above x.

### F6: logpdf loses accuracy, then returns -inf/nan, for mu < 1.49e-154

- **Code:** `_logpdf` line 9568, `2*x*mu**2.0`.
- **Mechanism:** `mu**2.0` is subnormal for 2.2e-162 < mu < 1.49e-154
  (fewer than 53 significant bits) and 0 below (pattern 4). The
  quotient then carries the relative error of the subnormal, or becomes
  `-1/0 = -inf`, or `0/0 = nan` at mu*x = 1.
- **Regime:** mu < 1.49e-154, every x. At mu = 1e-157 the error is 1.6e5 ulp, at
  1e-160 it is 5.0e10 ulp, and below 2.2e-162 the result is -inf/nan. pdf is affected
  the same way, but its normal values lie within ~sqrt(mu)/mu of x = 1/mu, where
  cond (arg) exceeds 1e100, so they could not be measured.
- **Reproducer:** `python accuracy.py refs/recipinvgauss_ref.py --at logpdf 2e160 1e-160`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(2e162, 1e-162)` | -inf | -2.4999999999999993e+161 | inf | 3 | loss |
  | `logpdf(2e200, 1e-200)` | -inf | -2.4999999999999995e+199 | inf | 3 | loss |
  | `logpdf(1e200, 1e-200)` | nan | -1.1600114189873572e+167 | inf | 4.15e16 | loss |
  | `logpdf(2e160, 1e-160)` | -2.500027832353145e+159 | -2.5e+159 | 5.0e10 | 3 | severe |
  | `logpdf(2e157, 1e-157)` | -2.5000000000902967e+156 | -2.4999999999999996e+156 | 1.6e5 | 3 | moderate |

- **Fix direction:** same as F4: `-(1/mu - x)**2/(2*x)` avoids `mu**2`.

### F7: sf right tail exceeds conditioning by up to ~2x (ndtr far tail)

- **Code:** `_sf` line 9582, `_norm_cdf(isqx*trm1)` with `isqx = 1/sqrt(x)`, `trm1 = 1/mu - x`.
- **Mechanism:** z1 = `isqx*trm1` carries a few ulp of rounding from three
  operations. Phi(z1) at z1 ~ -sqrt(x) amplifies that by z1**2 ~ x. `sc.ndtr`
  itself is up to several hundred ulp off for |z| in [10, 37] (INDEX item 6).
  The harness conditioning (~x/2) accounts only for the rounding of x.
- **Regime:** right tail x ~ 100 ... 1400, moderate mu.
- **Reproducer:** `python accuracy.py refs/recipinvgauss_ref.py --at sf 1000 0.01`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1000, 0.01)` | 3.2853718840066587e-178 | 3.2853718840062795e-178 | 520 | 496 | minor |
  | `sf(500, 1)` | 2.5812626818941298e-110 | 2.5812626818942887e-110 | 277 | 250 | minor |
  | `sf(100, 1000)` | 1.5254953301942316e-23 | 1.5254953301942598e-23 | 83.3 | 50.5 | minor |

- **Fix direction:** `log_ndtr`/`erfcx`-based evaluation, e.g. `phi(z1)*sqrt(pi/2)*erfcx(-z1/sqrt(2))` with z1**2/2 formed as `(1/mu - x)**2/(2x)`.

### F8: pdf = exp(logpdf) amplifies rounding for large mu, small x

- **Code:** `_pdf` line 9563: `np.exp(self._logpdf(x, mu))`.
- **Mechanism:** for large mu and small x, pdf ~ 1/sqrt(2 pi x) is huge and
  has cond (arg) 0.5. But logpdf ~ -0.5*log(2 pi x) is up to ~370 in magnitude,
  and its 0.5-ulp rounding becomes ~abs(logpdf)*0.5 ulp in the pdf.
- **Regime:** mu >~ 1e50 (so that 1/(2 mu**2 x) is negligible), x < ~1e-30.
- **Reproducer:** `python accuracy.py refs/recipinvgauss_ref.py --at pdf 1e-260 1e160`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-260, 1e160)` | 3.9894228040142144e+129 | 3.989422804014327e+129 | 127 | 0.5 | minor |
  | `pdf(1e-100, 1e60)` | 3.9894228040143154e+49 | 3.989422804014327e+49 | 12.9 | 0.5 | minor |

- **Fix direction:** compute the pdf directly as `exp(shape term)/sqrt(2 pi x)`.

## RuntimeWarnings

- pdf/logpdf: "overflow encountered in power" wherever F4/F5 occur (correct
  signal of a wrong result). "underflow encountered in power" / "divide by zero"
  for mu < 1.5e-154 (F6). "underflow encountered in exp" in pdf where the true
  pdf underflows (spurious but harmless).
- cdf/sf: "overflow encountered in exp" and "invalid value encountered in
  multiply" for mu < 2.818e-3 (F1, results are nan). No warning is raised for
  the silent underflow of `ndtr(-z2)` in F3 or the cancellation in F2.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | full grid, mu in [1e-3, 1e8] away from F5 | 1.87 ulp |
| pdf | full grid, mu in [1e-3, 1e8] | 757 ulp at cond 700 (`pdf(1400, 0.63)`, exp amplification as F8; 4.8 ulp in the bulk) |
| cdf | bulk and right tail (x >~ 1/mu), mu in [3e-3, 1e8] | 22 ulp (bulk), 0.5 ulp (right tail); 25.6 ulp at mu = 0.003 near x = 1/mu (no visible loss from `exp(2/mu)` rounding) |
| sf | left tail and bulk, all mu >= 3e-3 | 0.5 ulp (left tail), 21.5 ulp (bulk) |
| cdf, sf | mu = 2.82e-3, just above the F1 threshold | 20.6 ulp |

The hypothesis that the relative error of `exp(2/mu)` (~(2/mu) ulp, as in invgauss F10) would
show near x = 1/mu for small mu did not appear: at mu = 0.003 the cdf error there
is 25.6 ulp with cond 14.9 (minor), and below 2.818e-3 F1 takes over.

## Suspected, not confirmed

- pdf for mu < 1.49e-154 (F6 mechanism) cannot be measured: its normal values
  lie in a relative band of ~sqrt(mu) around x = 1/mu, where cond (arg) is above
  1e100.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | `-z1**2/2 + log(M(z1) - M(z2)) - 0.5*log(2 pi)`, with `M(z) = sqrt(pi/2)*erfcx(z/sqrt2)` and `z1**2/2 = (1/mu - x)**2/(2x)`; `log_ndtr(-z1) + log1p(-exp(2/mu + log_ndtr(-z2) - log_ndtr(-z1)))` for z1 <= 0 | cdf underflows for x < ~1/(1416 mu**2): `logcdf(1e-4, 1)` = -inf vs -5013.04 |
| logsf | log(sf) | `np.logaddexp(log_ndtr(z1), 2/mu + log_ndtr(-z2))` (both terms positive, no cancellation) | sf underflows for x >~ 1416: `logsf(2000, 1)` = -inf vs -1003.03 |
| ppf | brentq on cdf | `1/invgauss.isf(q, mu)` (Boost; INDEX item 14 caveats apply) | slow; inherits F1-F3 of the cdf. `ppf(1e-20, 1)` 33.9 ulp (minor) |
| isf | ppf(1 - q) | `1/invgauss.ppf(q, mu)` (Boost): `1/invgauss.ppf(1e-20, 1)` = 89.12891393431372 (~1 ulp) | q < ~1e-16: `isf(1e-20, 1)` = 100.0 vs 89.1289139343137 (5.5e14 ulp) |

## Shared code

- `sc.ndtr` (via `_norm_cdf`): its known far-tail error (INDEX item 6, up to
  ~1000 ulp for |z| in [20, 37.5]) feeds F7 directly and is amplified by the
  cancellation in F2 (e.g. 3.9e5 ulp at `cdf(0.001, 1)`). The function is used
  correctly here, and no new special-function defect was found.
- No helper or base class shares the faulty expressions. `invgauss`/`wald`
  compute the same quantity with different code (log space).

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1000, 0.001)` | nan | 0.49369374447153357 | verified | Also nan in `cdf(500, 0.001)` (ref 3.1656625234045582e-111), `sf(1000, 0.001)` (0.50630625552846643), `sf(1e-10, 0.001)` (1.0) and `cdf(1e300, 1e-200)` (1.0). Mechanism matches line 9576/9582 (`np.exp(2.0/mu)` overflows). |
| F2 | `cdf(1e-19, 1e8)` | 1.7885824676670868e-230 | 1.7922589638433096e-230 | verified | 9.24e12 ulp. Also `cdf(1e-40, 1e160)` = 0.0 (ref 7.9788456080286533e-21), `cdf(1e-30, 1e160)` 1.96e14 ulp, `cdf(1e-14, 1e8)` 3.14e6 ulp, `cdf(1e-9, 1000)` 1.51e8 ulp. Mechanism matches line 9576 (cancellation). The table's reference for `cdf(1e-19, 1e8)` (…434179e-230) is off by 6e-14 relative from the closed form at 400/600 digits and from `refs/recipinvgauss_ref.py` (…4330959e-230); this does not affect the ulp figure. |
| F3 | `cdf(0.07142857142857142, 0.1)` | 2.248754943762614e-302 | 3.1852146946495918e-304 | verified | 3.13e17 ulp. Also `cdf(6.5, 0.01)` 3.24e16, `cdf(79.365…, 0.003)` 7.22e15, `cdf(166.67, 0.003)` 2.27e15, `sf(1400, 0.1)` 2.24e15, `sf(1400, 0.01)` 2.09e15, `sf(666.67, 0.003)` 1.13e15 ulp. Mechanism matches (underflow of `_norm_cdf(-isqx*trm2)`). |
| F4 | `logpdf(1, 1.5e154)` | nan | -1.4189385332046727 | verified | Also `pdf(1, 1e160)` nan, `logpdf(5e-324, 1e160)` 371.32 vs -640.71, `pdf(1e-321, 1e160)` 6.71e17 ulp, `logpdf(1.3e-6, 1e160)` 5.0e8 ulp. Mechanism matches line 9568 (`mu**2.0` overflow). |
| F5 | `logpdf(1.4e154, 1)` | -inf | -6.9999999999999998e153 | verified | Also -inf/nan for the other three calls in the table (refs -1.1110358462032722e154, -5e299, -5e299). Mechanism matches (`(1 - mu*x)**2.0` overflow). |
| F6 | `logpdf(2e160, 1e-160)` | -2.500027832353145e159 | -2.5e159 | verified | 5.01e10 ulp. Also -inf for `logpdf(2e162, 1e-162)` and `logpdf(2e200, 1e-200)`, nan for `logpdf(1e200, 1e-200)` (ref -1.1600114189873572e167). Mechanism matches (`mu**2.0` subnormal/zero). |

Verifier references: cdf and sf by `mp.quad` of the docstring pdf at 100 and 200 digits, logpdf by the rewritten form `-(1/mu - x)**2/(2x) - log(2 pi x)/2` (`scratch/recipinvgauss/verify/verify.py`). For the cdf/sf values below 1e-100, quad was accurate only to about 5e-12 relative (and the 100- and 200-digit results disagreed for `cdf(500, 0.001)` and `cdf(79.365…, 0.003)`), so the closed erfc form at 400 and 600 digits was used for those values (`scratch/recipinvgauss/verify/closed.py`). It agrees with the analyst's reference to 20 digits.

Verified by dist-accuracy-verifier: 6 of 6 verified.
