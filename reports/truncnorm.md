# truncnorm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `truncnorm_gen`, lines 10259-10494 (helpers `_log_sum`, `_log_diff`, `_log_gauss_mass` at lines 10210-10256)
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf
- **Inherited (not measured):** none
- **Related:** none; helpers: `_log_diff`, `_log_sum`, `_log_gauss_mass`, `_norm_logpdf`, `_norm_logcdf` (= `sc.log_ndtr`), `_norm_cdf` (= `sc.ndtr`); special functions: `sc.ndtri_exp`, `sc.logsumexp`
- **Reference:** `refs/truncnorm_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (-1.09787, 2.73068), (0.1, 2), (0, inf), (-inf, 0), (-1, inf), (-0.001, 10), (1, 1.00000001), (-1e-06, 1e-06), (30, inf), (-50, -10)
- **Worst finding:** loss in cdf, logcdf, sf, logsf, ppf, isf

## Summary

For truncation intervals of ordinary width that do not end at or near 0,
all eight methods are accurate in the bulk and in the far tails (within
conditioning, including a = 30 and [-50, -10]). Everything goes through log
probability masses `_log_gauss_mass(lo, hi)`. These masses lose relative
accuracy when the interval [lo, hi] is short, and the code has no direct
formula for that case. The results:
- cdf/logcdf/sf/logsf are totally wrong next to an endpoint at 0. For example,
  `cdf(1e-17, 0, inf) = 0` and `sf(-1e-17, -inf, 0) = 1.2e-16` (true 8e-18).
- pdf/logpdf lose 1e5 to 1e7 ulp for narrow intervals.
- Close to any endpoint, cdf/sf errors exceed the (large) conditioning.

The quantile functions recover x from `ndtri_exp(log Phi(x))`, which only has
absolute accuracy, so ppf near a = 0 and isf near b = 0 return 0.0. `isf`
subtracts probabilities where `ppf` adds them. For b = inf it returns the same
value 8.2185 for every q < ~1e-17 (true 9.28 at q = 1e-20), and for b < 0
it loses accuracy as q -> 1.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, logcdf, sf, logsf | x next to an endpoint at or near 0 (a ~ 0: cdf/logcdf/logsf; b ~ 0: sf/logsf/logcdf) | mass of [a, x] / [x, b] cancels: `log1p(-Phi(a) - Phi(-x))` -> -inf; `_log_diff(p, p)` = p - 36.64 (patterns 1, 5) | `cdf(1e-20, 0, inf)` 0.0; `sf(-1e-300, -inf, 0)` 1.2e-16 for 8e-301 | loss |
| F2 | ppf, isf | quantile close to 0 with cond ~ 1: ppf for a ~ 0, isf for b ~ 0, narrow intervals around 0 | `ndtri_exp(log Phi(x))` gives x with absolute, not relative, accuracy (pattern 8) | `ppf(1e-20, 0, inf)` 0.0 for 1.25e-20 | loss |
| F3 | isf | small q with Phi(-b) << Phi(-a) (b = inf, large b); q -> 1 with b << 0 | isf subtracts `Phi(-a) - (1-q) Z` / `Phi(b) - q Z` in log space (patterns 1, 8) | `isf(1e-300, 30, inf)` 31.20 for 47.76, 1.6e15 ulp | severe |
| F4 | pdf, logpdf (and every method through Z) | narrow truncation interval b - a << 1 | `_log_gauss_mass(a, b)` cancels for short intervals (pattern 5) | `pdf(1.000000005, 1, 1.00000001)` 2.3e7 ulp | severe |
| F5 | cdf, logcdf, sf, logsf | x close to a (cdf) or b (sf), endpoint away from 0 | same short-interval cancellation as F4, error 2x to 100x the conditioning | `cdf(-0.0009999999999994458, -0.001, 10)` 1.8e14 ulp, cond 1.8e12 | severe |

### F1: cdf/sf next to an endpoint at 0 lose all digits

- **Code:** `_logcdf` line 10352 `_log_gauss_mass(a, x) - _log_gauss_mass(a, b)`,
  `_logsf` line 10363 `_log_gauss_mass(x, b) - ...`, and inside
  `_log_gauss_mass`: line 10246 (central case)
  `sc.log1p(-_norm_cdf(a) - _norm_cdf(-b))`, line 10230 (left case)
  `_log_diff(_norm_logcdf(b), _norm_logcdf(a))`, with `_log_diff` (line 10216)
  `sc.logsumexp([log_p, log_q+np.pi*1j], axis=0)`.
- **Mechanism:** the mass of the short interval [0, x] (a = 0 is the central
  case) is computed as `log1p(-(0.5 + Phi(-x)))`. That expression cancels
  (pattern 1). The relative error is about eps/(x*phi(0)), and for x < ~1e-16
  the sum rounds to 1, giving `log1p(-1) = -inf` and cdf = 0. In the
  mirrored case b = 0, [x, 0] is the left case and is computed as
  `log(exp(logPhi(0)) - exp(logPhi(x)))` by the complex logsumexp trick. When
  both logs round to the same double, the result should be -inf. Instead it is
  `p + log|1 + exp(i*pi)|` = p + log(1.2246e-16) = p - 36.64, because
  `exp(1j*np.pi)` is -1 + 1.22e-16j. So sf and logsf return a value that does
  not depend on x. logcdf for b = 0 and logsf for a = 0 inherit this through
  the `> -0.1` branches (lines 10355, 10366: `log1p(-exp(other))`).
- **Regime:** |x - endpoint| small with the endpoint at or near 0. The error
  is about 1/(|x - endpoint|*phi) ulp while the cond (arg) is about
  |x|/|x - endpoint| <= 1. The result is fully wrong once |x| < ~1e-16, and
  the error is above 1e6 ulp for |x| < ~1e-4. Tiny negative a (-0.001) still
  costs up to ~1e3 ulp in the central case.
- **Reproducer:** `python accuracy.py refs/truncnorm_ref.py --at cdf 1e-20 0 inf`
  (negative arguments need a leading space, e.g. `--at sf " -1e-17" " -inf" 0`)

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-20, 0, inf)` | 0.0 | 7.978845608028654e-21 | inf | 1 | loss |
  | `cdf(1e-12, 0, inf)` | 7.980283101005619e-13 | 7.978845608028653e-13 | 8.1e+11 | 1 | severe |
  | `cdf(1e-08, 0, inf)` | 7.978845584233329e-09 | 7.978845608028653e-09 | 1.3e+07 | 1 | severe |
  | `logcdf(1e-20, 0, inf)` | -inf | -46.27749321252564 | inf | 0.0216 | loss |
  | `logcdf(1e-12, 0, inf)` | -27.856632321773652 | -27.856812468573274 | 2.9e+10 | 0.0359 | severe |
  | `logsf(1e-20, 0, inf)` | -0.0 | -7.978845608028654e-21 | inf | 1 | loss |
  | `sf(-1e-300, -inf, 0)` | 1.224646799147351e-16 | 7.978845608028653e-301 | 6.9e+299 | 1 | loss |
  | `sf(-1e-17, -inf, 0)` | 1.224646799147351e-16 | 7.978845608028653e-18 | 6.5e+16 | 1 | loss |
  | `sf(-1e-12, -inf, 0)` | 7.979172971960659e-13 | 7.978845608028653e-13 | 1.8e+11 | 1 | severe |
  | `logsf(-1e-300, -inf, 0)` | -36.63870901270898 | -691.0013192508584 | 4.3e+15 | 0.00145 | severe |
  | `logsf(-1e-12, -inf, 0)` | -27.85677144043054 | -27.856812468573274 | 6.6e+09 | 0.0359 | severe |
  | `logcdf(-1e-20, -inf, 0)` | -1.224646799147351e-16 | -7.978845608028654e-21 | 6.9e+19 | 1 | loss |
  | `cdf(1e-08, -0.001, 10)` | 0.0007972562882902014 | 0.0007972562882903231 | 688 | 1e-05 | minor |

- **Fix direction:** for intervals inside about [-1, 1] or straddling 0, use
  `log(0.5*(erf(hi/sqrt2) - erf(lo/sqrt2)))`. That is a sum of positive terms
  for lo < 0 < hi and accurate near 0. Make `_log_diff(p, p)` return -inf
  (e.g. `p + log1mexp(q - p)` with `sc._ufuncs._log1mexp`).

### F2: ppf/isf lose relative accuracy for quantiles near 0

- **Code:** `_ppf` lines 10384-10392 and `_isf` lines 10413-10421:
  `sc.ndtri_exp(log_Phi_x)` / `-sc.ndtri_exp(log_Phi_x)`.
- **Mechanism:** the quantile is recovered from log Phi(x). Near x = 0,
  Phi(x) = 0.5 + x*phi(0) and its log is known only to absolute accuracy
  ~eps, so x gets an absolute error ~eps and a relative error ~eps/|x|
  (pattern 8). `ndtri_exp` itself is accurate for its input; the information
  is lost before the call. For a = 0 (ppf) or b = 0 (isf), x ~ q/f(0) with
  cond ~ 1, so the loss is all beyond conditioning, and for q < ~1e-16 the
  result is 0.0. Within the narrow interval (-1e-6, 1e-6) the same absolute
  error is large relative to the width (3e9 ulp, cond 5e3). In wide intervals
  that straddle 0 (example set, (-1, inf)) the crossing point is also
  ill-conditioned (cond ~ 1e11 to 1e16). There the errors are only 1.8x to 2x
  the cond, or 0.0 where cond is 2e16, which the tool still labels
  severe/loss.
- **Regime:** quantiles |x| << 1 where |x| is not small compared with
  |x - endpoint| (a ~ 0 for small q, b ~ 0 for small q in isf, narrow intervals
  around 0).
- **Reproducer:** `python accuracy.py refs/truncnorm_ref.py --at ppf 1e-20 0 inf`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-20, 0, inf)` | 0.0 | 1.2533141373155002e-20 | inf | 1 | loss |
  | `ppf(1e-12, 0, inf)` | 1.2532864118509303e-12 | 1.2533141373155002e-12 | 1.0e+11 | 1 | severe |
  | `ppf(1e-08, 0, inf)` | 1.25331412969852e-08 | 1.2533141373155004e-08 | 2.7e+07 | 1 | severe |
  | `isf(1e-20, -inf, 0)` | 0.0 | -1.2533141373155002e-20 | inf | 1 | loss |
  | `isf(1e-08, -inf, 0)` | -1.25331412969852e-08 | -1.2533141373155004e-08 | 2.7e+07 | 1 | severe |
  | `ppf(0.4999000000695737, -1e-06, 1e-06)` | -1.9999999383770713e-10 | -1.9999986085258434e-10 | 3.0e+09 | 5e+03 | severe |
  | `ppf(0.5000000000685737, -1e-06, 1e-06)` | 0.0 | 1.371474045441572e-16 | inf | 7.3e+09 | loss |
  | `ppf(0.4057132913274699, -1, inf)` | 0.0 | 3.9998307295497325e-17 | inf | 2.1e+16 | loss |

  (The last row is at the zero crossing with cond 2.1e16, so it is essentially
  within conditioning. It is listed because the tool labels it loss.)
- **Fix direction:** use `Phi(x) - 1/2 = erf(x/sqrt2)/2`, i.e.
  `x = sqrt2 * erfinv(erf(a/sqrt2) + 2*q*Z)`, when the target is in the central
  region (|x| < ~1). `sc.erfinv` is relatively accurate near 0.

### F3: isf subtracts probabilities (catastrophic for b = inf, small q)

- **Code:** `_isf` lines 10418-10421 (b >= 0):
  `_log_diff(_norm_logcdf(-a), np.log1p(-q) + _log_gauss_mass(a, b))`, i.e.
  Phi(-x) = Phi(-a) - (1-q)*Z; lines 10413-10416 (b < 0):
  `_log_diff(_norm_logcdf(b), np.log(q) + _log_gauss_mass(a, b))`, i.e.
  Phi(x) = Phi(b) - q*Z.
- **Mechanism:** for small q (b >= 0) the true Phi(-x) = Phi(-b) + q*Z is tiny
  compared with Phi(-a), but it is formed as a difference (patterns 1, 8).
  In log space the sum `log1p(-q) + mass` carries an absolute rounding error
  of eps*|mass|, so the relative error of Phi(-x) is about
  eps*max(|mass|, 1)*Phi(-a)/Phi(-x). For b = inf and q < ~1e-17,
  `log1p(-q) + mass == mass == log_ndtr(-a)` exactly, and `_log_diff(p, p)`
  returns p - 36.64 (see F1). The result is a constant that does not depend
  on q: 8.218 for a = -1, 8.281 for a = 0, 31.196 for a = 30. The mirror
  case b < 0, q -> 1 forms Phi(x) = Phi(b) - q*Z and loses about
  |log Phi(b)| times the conditioning (53x for b = -10). `_ppf` uses
  `_log_sum` (additions) in both branches and is accurate in the
  corresponding tails.
- **Regime:** b >= 0 and q small enough that Phi(-b) + q*Z << Phi(-a). This
  is total for b = inf, q < 1e-17, and for large b already above 1e6 ulp at
  q = 1e-10. Also b < 0, q -> 1, with |log Phi(b)| large.
- **Reproducer:** `python accuracy.py refs/truncnorm_ref.py --at isf 1e-300 30 inf`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1e-300, 30, inf)` | 31.196143833217686 | 47.755864424165964 | 1.6e+15 | 0.000438 | severe |
  | `isf(1e-300, -1, inf)` | 8.218496630017437 | 37.051755702175846 | 3.5e+15 | 0.000728 | severe |
  | `isf(1e-20, -1, inf)` | 8.218496630017437 | 9.280763163116815 | 5.2e+14 | 0.0115 | severe |
  | `isf(1e-10, -1, inf)` | 6.387818428028056 | 6.387818440684452 | 8.9e+06 | 0.0239 | severe |
  | `isf(1e-10, 30, inf)` | 30.757148888371677 | 30.757144852268773 | 5.9e+08 | 0.00106 | severe |
  | `isf(1e-10, -0.001, 10)` | 6.4668305032876585 | 6.466830515796068 | 8.7e+06 | 0.0234 | severe |
  | `isf(0.99999999999999, -50, -10)` | -12.832160971155496 | -12.805735946097663 | 9.3e+12 | 6.1e+11 | severe |
  | `isf(0.9999999999, -50, -10)` | -12.069856523764903 | -12.06985798558048 | 5.5e+08 | 6.8e+07 | severe |

  (The two (-50, -10) rows show an underflow RuntimeWarning in exp.)
- **Fix direction:** use the reflection `isf(q, a, b) = -ppf(q, -b, -a)`.
  Equivalently, b >= 0: `-ndtri_exp(_log_sum(log_ndtr(-b), log(q) + mass))`;
  b < 0: `ndtri_exp(_log_sum(log_ndtr(a), log1p(-q) + mass))`.

### F4: narrow truncation intervals: pdf/logpdf wrong in every digit beyond ~8

- **Code:** `_logpdf` line 10345 `_norm_logpdf(x) - _log_gauss_mass(a, b)`;
  `_log_gauss_mass` lines 10230/10233 (`_log_diff` of two `log_ndtr` values)
  and line 10246 (central case).
- **Mechanism:** for a short interval the mass is a difference of two nearly
  equal quantities (pattern 5).
  - Left/right case: log Phi(b) - log Phi(a) is about phi/Phi*(b - a), but each
    log carries an absolute error eps*|log Phi|. The mass therefore has a
    relative error of about eps*|log Phi|*Phi/(phi*(b - a)): about 2e7 ulp
    for (1, 1 + 1e-8).
  - Central case: `log1p(-Phi(a) - Phi(-b))` has a relative error of about
    eps/Z, which is 2.7e5 ulp for (-1e-6, 1e-6). This happens although
    cond (shapes) is only 1 there.

  Every method divides by Z, but cdf/sf/ppf/isf partly cancel the error
  (ratios of masses; quantiles measured to <= 3 ulp for (1, 1 + 1e-8)).
  pdf and logpdf carry it fully.
- **Regime:** b - a << 1/max(1, |a|, |b|). The error is about
  1/(b - a) ulp, multiplied by |log Phi| for intervals on one side of 0.
- **Reproducer:** `python accuracy.py refs/truncnorm_ref.py --at pdf 1.000000005 1 1.00000001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1.000000005, 1, 1.00000001)` | 100000001.12199335 | 100000000.60774711 | 2.3e+07 | 1 | severe |
  | `logpdf(1.000000005, 1, 1.00000001)` | 18.4206807551723 | 18.420680750029838 | 1.3e+06 | 0.0543 | severe |
  | `pdf(0, -1e-06, 1e-06)` | 500000.0000301196 | 500000.00000008335 | 2.7e+05 | 0 | moderate |
  | `logpdf(0, -1e-06, 1e-06)` | 13.122363377464568 | 13.122363377404495 | 2.06e+04 | 0 | moderate |

  (cond (shapes) is 2.0e8 for pdf and 1.1e7 for logpdf at (1, 1.00000001): the
  value is very sensitive to a and b, which are exact inputs. For
  (-1e-6, 1e-6) cond (shapes) is 1, so that error is entirely avoidable.)
- **Fix direction:** central case via `erf` differences (see F1). For
  same-sign short intervals, use a short-interval expansion around the
  midpoint m: `log(phi(m)*(b - a)) + log1p((m**2 - 1)*(b - a)**2/24 + ...)`.

### F5: cdf/sf close to an endpoint away from 0: errors beyond the (large) conditioning

- **Code:** same as F1/F4: `_log_gauss_mass(a, x)` (line 10352) and
  `_log_gauss_mass(x, b)` (line 10363) for x close to the endpoint.
- **Mechanism:** the mass of [a, x] has the short-interval relative error of
  F4, about eps*|log Phi(a)|*Phi(a)/(phi(a)*(x - a)). The cond (arg) is
  |x|/(x - a), so the excess factor is about |log Phi(a)|*Phi(a)/(phi(a)*|a|).
  That is ~3 for a = 0.1 and ~100 for a = -0.001 (and diverges as a -> 0,
  where it turns into F1). The tool labels these severe because cond is huge
  (1e9 to 1e15). In absolute terms the result has the accuracy of
  `(x - a)*f(a)` with x - a known only to a few digits.
- **Regime:** x - a (or b - x) << 1 with |a| (|b|) below about 1. For
  |a| >~ 1 the error stays within about 1.2x the conditioning (left/right
  case).
- **Reproducer:** `python accuracy.py refs/truncnorm_ref.py --at cdf 0.10000000000000009 0.1 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(0.10000000000000009, 0.1, 2)` | 1.2883401531861723e-16 | 7.556304213438576e-17 | 3.2e+15 | 1.2e+15 | severe |
  | `cdf(-0.0009999999999994458, -0.001, 10)` | 4.599311017960925e-16 | 4.4187006830077587e-16 | 1.8e+14 | 1.8e+12 | severe |
  | `cdf(-0.00099999999999, -0.001, 10)` | 7.870964460206773e-15 | 7.972504671293771e-15 | 5.7e+13 | 1.0e+11 | severe |
  | `cdf(0.10000000000100001, 0.1, 2)` | 9.072754132468106e-13 | 9.074869483532161e-13 | 1.0e+12 | 1.0e+11 | severe |
  | `sf(1.0000000089999999, 1, 1.00000001)` | 0.09999996150190162 | 0.10000000843178422 | 2.1e+09 | 1.0e+09 | severe |

- **Fix direction:** same as F4: an accurate short-interval mass, e.g.
  `phi(a)*(x - a)*(1 - a*(x - a)/2 + ...)` or erf differences near 0.

## RuntimeWarnings

- pdf/logpdf: "overflow encountered in square" from `x**2` in `_norm_logpdf`
  for |x| >~ 1.3e154 (b = inf or a = -inf sets). The true logpdf there is
  below -1.8e308 and the true pdf is 0, so the returned -inf / 0.0 is correct.
  "underflow encountered in exp/square" in the far tails also comes with
  accurate results. These warnings are spurious.
- cdf/logcdf/sf/logsf: "underflow encountered in exp" (e.g. from `exp` of
  very negative log masses in `logsumexp` and in the `> -0.1` branches) with
  accurate results in the far-tail sets (30, inf), (-50, -10). Spurious.
- isf/ppf: "underflow encountered in exp" for (-50, -10). It comes with the
  inaccurate q -> 1 isf values of F3 and with accurate ppf values.
- None of the loss cases of F1-F3 emits any warning (e.g. `cdf(1e-17, 0, inf)`
  returns 0.0 silently; `_log_diff(p, p)` returns a finite value).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, all sets except narrow intervals (F4) | 31 ulp at (-0.001, 10) (cond 100); 488 ulp for (30, inf) and 202 ulp for (-50, -10), within cond (x**2 ~ 900-1440) |
| logpdf | full grid, all sets except narrow intervals | 5 ulp; far tails 2.4e3 ulp at (30, inf), cond 6.1e3 |
| cdf, logcdf | bulk and tails incl. both sides of the `logcdf > -0.1` switch (example sets, (-1, inf), (-inf, 0), far tails) away from the endpoint regimes of F1/F5 | cdf (-inf, 0) 37 ulp (cond 73); (-50, -10) 752 ulp (cond 1.3e3) |
| sf, logsf | same, both sides of the `logsf > -0.1` switch | sf (30, inf) 526 ulp (cond 1.26e3); (0, inf), (-1, inf) <= 37 ulp (cond 73) |
| ppf | q in [1e-300, 1 - 3e-16], all sets; a > 0 and far tails | (0.1, 2) 3.75 ulp; (30, inf) 1.07; (-50, -10) 1.58; (1, 1.00000001) 2; (-inf, 0) 5e7 at q -> 1 within cond 1e8 |
| isf | q -> 1 for b >= 0, q -> 0 for finite moderate b | (0.1, 2) 5.62 ulp; example set 22 ulp (cond 79); (1, 1.00000001) 3 ulp |

Reference unavailable at 60 points each of sf/logsf and 30 each of
cdf/logcdf (mpmath OverflowError in `ncdf` at |x| >~ 1e150, b or a infinite).
There the true sf/cdf is far below the double range, and logsf/logcdf
(~ -x**2/2) is not a double either, so nothing measurable was skipped.

## Suspected, not confirmed

- Far tails beyond the tested sets (a >~ 38, where `log_ndtr(-a)` is about
  -700 or less): logcdf = mass(a, x) - mass(a, b) then cancels two terms of
  size a**2/2, with an absolute error of eps*a**2/2. For a = 30 the measured
  errors were within conditioning (cond ~ a**2 as well); not expected to
  exceed it, not tested for a >= 40.

## Opportunities: inherited methods

none (all methods are implemented)

## Shared code

- `_log_diff(log_p, log_q)` (line 10216, complex `logsumexp` with
  `log_q + pi*1j`) returns `log_p - 36.64` instead of -inf when
  `log_p == log_q`, because `exp(1j*pi)` is -1 + 1.22e-16j in double precision.
  Example: `_log_diff(-0.5, -0.5)` = -37.1387 + 1.5708j. It also gives no
  better than absolute accuracy for nearly equal arguments. Other users:
  `reciprocal._munp` (line 9311) and `_LogUniform` in
  `_new_distributions.py` (line 345). Their moments for `n*log(b)` close to
  `n*log(a)` would be affected (not measured).
- `_log_gauss_mass` (F1, F4, F5) is used only by truncnorm.
- `sc.ndtri_exp` is accurate for its input; F2 is a loss of information
  before the call, not a special-function problem.

## Existing GitHub issues

- #15110 (closed, fixed by PR #16586): BUG: truncnorm.cdf returns incorrect values at tail - a tail-interval cdf case, now fixed; F1/F4/F5 are the remaining short-interval regimes of the same `_log_gauss_mass`/`_log_diff` code
- PR #16586 (merged): MAINT: stats.truncnorm: improve CDF accuracy/speed - introduced the current log-space cdf machinery behind F1, F4, F5
- PR #17064 (state not confirmed): BUG: Fix numerical precision error of `truncnorm.logcdf` when cdf is nearly 1 - cancellation in `_log_gauss_mass`; related to F1 (logcdf) and F4
- F2, F3: none found

(web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e-20, 0, inf)` | 0.0 | 7.9788456080286531e-21 | verified | reference = mp.quad of pdf / mp.quad mass; 100 vs 200 digits agree to 1e-101; central case `log1p(-ndtr(0) - ndtr(-1e-20))` = -inf confirmed (line 10246) |
| F1 | `cdf(1e-12, 0, inf)` | 7.980283101005619e-13 | 7.9788456080286534e-13 | verified | 8.1e11 ulp, cond 1 |
| F1 | `sf(-1e-17, -inf, 0)` | 1.224646799147351e-16 | 7.9788456080286541e-18 | verified | `exp(1j*pi)` = -1 + 1.22e-16j and `_log_diff(-0.5, -0.5)` = -37.1387 + 1.5708j confirmed (line 10216) |
| F1 | `logsf(-1e-300, -inf, 0)` | -36.63870901270898 | -691.0013192508584 | verified | 4.3e15 ulp |
| F1 | `logcdf(-1e-20, -inf, 0)` | -1.224646799147351e-16 | -7.9788456080286531e-21 | verified | 6.9e19 ulp |
| F2 | `ppf(1e-20, 0, inf)` | 0.0 | 1.2533141373155002e-20 | verified | x_ref by bisection on erf/erfc cdf, cdf_quad(x_ref) = q to 1e-82 |
| F2 | `ppf(1e-12, 0, inf)` | 1.2532864118509303e-12 | 1.2533141373155002e-12 | verified | 1.0e11 ulp. For a = 0 the path is `ppf_right` (lines 10389-10392, `-ndtri_exp(log Phi(-x))` with `log1p(-q) + log(0.5)`), not `ndtri_exp(log Phi(x))`; same absolute-accuracy mechanism |
| F2 | `isf(1e-20, -inf, 0)` | 0.0 | -1.2533141373155002e-20 | verified | |
| F2 | `ppf(0.4999000000695737, -1e-06, 1e-06)` | -1.9999999383770713e-10 | -1.9999986085258433e-10 | verified | 3.0e9 ulp |
| F3 | `isf(1e-300, 30, inf)` | 31.196143833217686 | 47.755864424165965 | verified | `log1p(-q) + mass == log_ndtr(-30)` exactly, and `-ndtri_exp(log_ndtr(-30) + log(1.22e-16))` = 31.196143833217686; same value for q = 1e-20, 1e-25 |
| F3 | `isf(1e-20, -1, inf)` | 8.218496630017437 | 9.2807631631168148 | verified | 5.2e14 ulp |
| F3 | `isf(1e-10, 30, inf)` | 30.757148888371677 | 30.757144852268772 | verified | 5.9e8 ulp |
| F3 | `isf(0.99999999999999, -50, -10)` | -12.832160971155496 | -12.805735946097663 | verified | 9.3e12 ulp (b < 0 branch, line 10414) |
| F4 | `pdf(1.000000005, 1, 1.00000001)` | 100000001.12199335 | 100000000.6077471 | verified | 2.3e7 ulp; Z by mp.quad of the scaled tail integrand |
| F4 | `logpdf(1.000000005, 1, 1.00000001)` | 18.4206807551723 | 18.420680750029836 | verified | 1.3e6 ulp |
| F4 | `pdf(0, -1e-06, 1e-06)` (moderate) | 500000.0000301196 | 500000.00000008336 | verified | 2.7e5 ulp |
| F5 | `cdf(0.10000000000000009, 0.1, 2)` | 1.2883401531861723e-16 | 7.5563042134385762e-17 | verified | 3.2e15 ulp, cond 1.2e15 by verifier (x*f/F) |
| F5 | `cdf(-0.0009999999999994458, -0.001, 10)` | 4.599311017960925e-16 | 4.4187006830077587e-16 | verified | 1.8e14 ulp, cond 1.8e12 |
| F5 | `sf(1.0000000089999999, 1, 1.00000001)` | 0.09999996150190162 | 0.10000000843178422 | verified | 2.1e9 ulp, cond 1.0e9 |

Verifier scripts: `scratch/truncnorm/verify/verify.py` (references) and `scratch/truncnorm/verify/mech.py` (mechanism checks). Masses come from `mp.quad` of the normal pdf, using a scaled integrand phi(lo)*int exp(-lo*u - u^2/2) du for intervals with |endpoint| >= 1, because plain `mp.quad` lost about 2 digits on [30, inf). Quantiles come from bisection on an erf/erfc cdf and are checked against the quad cdf. All values agree between 100 and 200 digits to at least 80 digits. `refs/truncnorm_ref.py` was not consulted, because there were no disagreements.

Verified by dist-accuracy-verifier: 5 of 5 verified.
