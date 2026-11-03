# vonmises_line: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `vonmises_gen`, lines 11133-11379; cdf in the compiled `scipy/stats/_stats.pyx` (`von_mises_cdf`, `von_mises_cdf_series`, `von_mises_cdf_normalapprox`; the .pyx is not shipped in the wheel, behaviour confirmed by a Python reimplementation that reproduces SciPy's values)
- **Implemented:** pdf, logpdf, cdf
- **Inherited (not measured):** logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), ppf = numerical inversion of cdf, isf = ppf(1 - q)
- **Related:** `vonmises` is the same class `vonmises_gen` (support (-inf, inf) instead of [-pi, pi]). pdf, logpdf, cdf, sf, logcdf, logsf were bit-identical between the two at 80 points (8 kappa values x 10 points in (-pi, pi)), so every finding below applies to `vonmises` too. Differences: ppf/isf (generic inversion with different brackets; relative differences up to 3e-13), and at the endpoint x = -np.pi exactly, where `vonmises_line` returns 0 from the generic wrapper (x <= a) while `vonmises` returns `_cdf`. Helpers: `_check_fit_input_parameters` (fit only); special functions: sc.cosm1, sc.i0e, sc.i1e (entropy), sc.ndtr / i0e inside `von_mises_cdf_normalapprox`
- **Reference:** `refs/vonmises_line_ref.py`, 100 digits (cdf by mp.quad, which is slow at 400 digits); self-checks: PASS
- **Parameter sets:** (3.99390425810714), (0), (0.001), (0.5), (20), (49.9), (50), (200), (1000)
- **Worst finding:** loss in cdf

## Summary

pdf and logpdf are accurate everywhere (logpdf within 31 ulp and within
conditioning; pdf at most 229 ulp, minor beyond conditioning only near
x = +-pi for large kappa). The cdf is the problem. For kappa < 50 it uses a
series written as `0.5 + x/(2 pi) + V/pi`, which has an absolute error of a
few eps, so in the left half the relative error grows like eps/F; below
F ~ 1e-15 results are 0 or wrong by orders of magnitude (loss from kappa ~ 20
on). For kappa >= 50 it switches to a crude normal approximation
`ndtr(sqrt(2/pi)/i0e(kappa) * sin(x/2))` that is wrong by 1e-9 relative even
at the median-adjacent bulk, 1e-6 at moderate quantiles and up to 90% in the
tail; it does not improve with kappa, and the cdf is discontinuous at kappa = 50 (`cdf(-1, 49.9)` = 7.24e-12, `cdf(-1, 50)` = 6.76e-12; true values 7.20e-12 and 6.87e-12).
All inherited methods (sf, logsf, logcdf, ppf, isf) inherit these errors.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | kappa < 50, x < 0 with F(x) small (left tail); loss once F < ~1e-15 | series result `0.5 + x/(2pi) + V/pi` cancels (patterns 1, 5), then clipped to [0, 1] | 0.0 for 4.0e-17; 1.9e-15 for 7.2e-59 | loss |
| F2 | cdf | kappa >= 50, whole support | normal approximation used instead of the cdf (pattern 7); discontinuity at kappa = 50 | 4.1e15 ulp (rel. error 0.91); 1.3e10 ulp in the bulk | severe |
| F3 | pdf | large kappa, x near +-pi | rounding of `kappa*cosm1(x)` (~-2 kappa) amplified by exp (pattern 6) | 44.2 ulp | minor |

### F1: series branch of the cdf cancels in the left tail (kappa < 50)

- **Code:** `_cdf`, line 11262: `_stats.von_mises_cdf(kappa, x)`; in `_stats.pyx`, `von_mises_cdf_series` returns `0.5 + x / (2 * PI) + V / PI` (V = sum of p terms, p = int(29 + 0.5 kappa - 100/(kappa + 5))), and `von_mises_cdf` then clips: `temp[i] = 0 if temp[i] < 0 else 1 if temp[i] > 1 else temp[i]`.
- **Mechanism:** the three terms are O(1) and cancel to F(x) in the left half (x < 0) (patterns 1/5). The absolute error is a few eps, so the relative error is about eps/F(x). When F(x) is below the rounding error the result is either clipped to 0.0 or a positive value of size eps that has nothing to do with F. For kappa close to 50 the absolute error is larger (more terms, larger V): at kappa = 49.9 `cdf(-1.0)` is off by 4e-14 absolute.
- **Regime:** kappa < 50 and x < 0 with F(x) small: kappa = 3.99, x = -3 (F = 3.7e-5): 4.4e5 ulp; kappa = 10, x = -2 (F = 1e-7): 2.5e8 ulp; kappa = 20, x <= -2.24 (F < 1e-15): result 0.0; kappa = 49.9 everywhere x < -1.26. Grows with kappa because F at a fixed x drops like exp(-kappa(1 - cos x)). The right half (F near 1) is fine (absolute error is what matters there; 16.5 ulp at `cdf(3.0, 3.99)`). The points at x = -np.pi (cond 2.6e16, the true value is the mass of the 1.2e-16 sliver between -pi and the double -np.pi) are flagged as loss for every kappa; there the 0.0 comes from the generic wrapper (x <= a), and they carry no practical weight.
- **Reproducer:** `python accuracy.py refs/vonmises_line_ref.py --at cdf -2.5 20`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-2.5, 20)` | 0.0 | 3.9876789625197515e-17 | inf | 25.2 | loss |
  | `cdf(-3.0, 20)` | 0.0 | 1.142030627237744e-18 | inf | 24.2 | loss |
  | `cdf(-1.2566370614359172, 49.9)` | 0.0 | 6.241910203883385e-17 | inf | 60 | loss |
  | `cdf(-2.827433388230814, 49.9)` | 2.6867397195928788e-14 | 1.2153821284144993e-43 | 1.0e+45 | 34.2 | loss |
  | `cdf(-3.141592643589793, 20)` | 2.0120092672645288e-16 | 7.531123135032992e-26 | 1.2e+25 | 3.1e+08 | loss |
  | `cdf(-1.0, 49.9)` | 7.2405414996978834e-12 | 7.1993581870245835e-12 | 2.6e+13 | 42.6 | severe |
  | `cdf(-1.5, 20)` | 7.529816770102116e-10 | 7.529705407272298e-10 | 6.7e+10 | 30 | severe |
  | `cdf(-2.0, 10)` | 1.0518147133820044e-07 | 1.0518147714716647e-07 | 2.5e+08 | 16.8 | severe |
  | `cdf(-1.0, 20)` | 1.036747129884441e-05 | 1.0367471322565055e-05 | 1.0e+07 | 17.4 | severe |
  | `cdf(-3.0, 3.99390425810714)` | 3.7435583436405684e-05 | 3.743558343270957e-05 | 4.4e+05 | 21.8 | moderate |
  | `cdf(-2.0, 3.99390425810714)` | 0.0008785017404072948 | 0.0008785017404037874 | 1.8e+04 | 6.12 | moderate |

- **Fix direction:** for x < 0 compute F(x) = S(-x) from a series/continued fraction for the tail mass directly (e.g. integrate `exp(kappa*cosm1(t))/(2 pi i0e(kappa))` from -pi to x with a tail-adapted expansion in u = sin^2(x/2) or the incomplete-Bessel form), instead of subtracting O(1) terms; use the symmetry F(x) = 1 - F(-x) only for x > 0.

### F2: kappa >= 50 uses an uncorrected normal approximation

- **Code:** `von_mises_cdf` in `_stats.pyx`: `CK = 50`; for `kappa >= CK` it returns `von_mises_cdf_normalapprox(kappa, x)` = `ndtr(z)` with `z = sqrt(2/pi) / i0e(kappa) * sin(x/2)` (confirmed: SciPy's results equal this expression bit for bit at (50, -0.1), (200, -0.5), (1000, -1.0)).
- **Mechanism:** this is an approximation, not an evaluation of the cdf (pattern 7: the branch threshold only switches to a less accurate formula). Hill (1977), on which the code is based, adds correction terms to z; SciPy uses the leading term only. The error is O(1/kappa) relative in the bulk and does not vanish in the tails: at fixed z the relative error of the tail mass stays ~1e-3, and further out (|x| > ~1.5) it grows to order 1 because the Gaussian tail and the true exp(-kappa(1 - cos x)) tail differ in shape. The comment "These values should give 12 decimal digits" does not hold for the normal branch. The cdf is discontinuous in kappa: `cdf(-1.0, 49.9)` = 7.24e-12, `cdf(-1.0, 50)` = 6.76e-12 (true 7.20e-12 and 6.87e-12).
- **Regime:** every x for kappa >= 50; relative error ~1e-9 near x = 0, ~2e-6 at F ~ 0.1, ~1e-3 at F ~ 1e-12, up to 0.9 in the far left tail. Values near 1 (x > 0) are wrong by the same absolute amount (7.5e8 ulp at `cdf(0.05, 50)`).
- **Reproducer:** `python accuracy.py refs/vonmises_line_ref.py --at cdf -0.1 50`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-3.0, 200)` | 3.2968398261591996e-175 | 3.6198768164335782e-174 | 4.1e+15 | 66.2 | severe |
  | `cdf(-3.0, 50)` | 2.857552395811191e-45 | 1.7715594808454948e-44 | 3.8e+15 | 29.2 | severe |
  | `cdf(-1.0, 50)` | 6.75819958997843e-12 | 6.869235154586916e-12 | 7.3e+13 | 42.7 | severe |
  | `cdf(-1.0, 1000)` | 3.3457992891950757e-202 | 3.398713433729918e-202 | 7.0e+13 | 842 | severe |
  | `cdf(-0.5, 200)` | 1.3428790599657895e-12 | 1.3442228914941169e-12 | 4.5e+12 | 48.8 | severe |
  | `cdf(-0.1, 50)` | 0.2403959820672798 | 0.24039668356440694 | 1.3e+10 | 0.912 | severe |
  | `cdf(0.05, 50)` | 0.6378151549842074 | 0.637815049368301 | 7.5e+08 | 0.207 | severe |
  | `cdf(-0.01, 50)` | 0.47188510505348674 | 0.4718851059502849 | 8.6e+06 | 0.0595 | severe |

  Harness summary over all parameter sets: cdf low tail 47 loss / 79 severe / 10 moderate, bulk 20 severe / 19 moderate / 38 minor, high tail 4 severe / 3 moderate / 28 minor (F1 and F2 together); worst at kappa = 1000: 1.7e14 ulp. For kappa = 1e4 the true `cdf(-0.5)` is 1.8e-534 (not representable), so underflow there is not a finding.
- **Fix direction:** evaluate the cdf for large kappa properly, e.g. Hill's corrected normal approximation (with the higher-order terms in z) for the bulk plus an asymptotic tail expansion of the integral of exp(-2 kappa sin^2(t/2)), or a series in the incomplete gamma/erf form valid for all kappa; at minimum raise CK and use the series (which converges for any kappa, with p ~ 0.5 kappa terms).

### F3: pdf loses ~|kappa*cosm1(x)| x eps near x = +-pi for large kappa

- **Code:** `_pdf`, line 11255: `np.exp(kappa*sc.cosm1(x)) / (2*np.pi*sc.i0e(kappa))`
- **Mechanism:** the exponent kappa*cosm1(x) is about -2 kappa near +-pi and carries a relative rounding error of ~1 ulp; exp turns this into an absolute error of the exponent of ~2 kappa eps, i.e. up to ~2 kappa ulp in the result (pattern 6). The conditioning with respect to x is tiny there (sin x ~ 0), so the error counts as beyond conditioning; it is within cond (shapes) (399 at kappa = 200).
- **Regime:** kappa >~ 20, |x| close to pi. Max 229 ulp (kappa = 200) overall, of which the 7 points beyond conditioning reach 44.2 ulp.
- **Reproducer:** `python accuracy.py refs/vonmises_line_ref.py --at pdf 3.141592643589793 200`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(3.141592643589793, 200)` | 1.0798419278661284e-173 | 1.079841927866139e-173 | 44.2 | 6.28e-06 | minor |
  | `pdf(-3.141492653589793, 49.9)` | 1.277238805977777e-43 | 1.2772388059777707e-43 | 22.5 | 0.0157 | minor |
  | `pdf(-3.141592643589793, 50)` | 1.0467670669223442e-43 | 1.0467670669223466e-43 | 10.3 | 1.57e-06 | minor |

- **Fix direction:** low priority; would need an error-free product (fma / double-double) for kappa*cosm1(x) and `exp(hi) * (1 + lo)`.

## RuntimeWarnings

`underflow encountered in exp` in pdf for kappa = 1000 near x = +-pi, where
the true pdf (~exp(-2000)) is below the normal range; the returned 0 is
expected and not a finding. No warnings from the compiled cdf (its clipping to
[0, 1] silently hides negative series results, see F1).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, all parameter sets (kappa = 0 ... 1000) | 229 ulp (within cond 299); beyond conditioning only F3 (44.2 ulp) |
| logpdf | full grid, all parameter sets, including where logpdf ~ 0 (kappa = 50, cancellation of kappa*cosm1(x) and log terms; covered by conditioning) | 31.4 ulp (cond 52.5) |
| cdf | kappa < 50, x >= 0 and the left half where F is not small; kappa = 0 and 0.001 (series with k = 0 handled) | 16.5 ulp at `cdf(3.0, 3.99)` (minor; absolute error of the series near F = 1) |

## Suspected, not confirmed

- Overflow of exp(kappa) in a naive normal approximation for kappa > 709.8: not present (the code uses i0e); kappa = 1000 gives finite (but F2-inaccurate) values.
- `vonmises` only: the periodic reduction `x - round(x/(2 pi))*2 pi` in `von_mises_cdf` loses absolute accuracy for large |x| (outside [-pi, pi]); not applicable to `vonmises_line` and not measured.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | `cdf(-x)` (symmetric density on [-pi, pi]) | x > 0 with small sf: `sf(1.0, 20)` = 1.03674712987889e-05 vs 1.0367471322565055e-05 (2.3e-9 rel.), plus all of F1/F2 |
| logsf | log(1 - cdf) | `logcdf(-x)` | right tail: `logsf(2.0, 3.99390425810714)` = -7.037292669226658 vs -7.037292669230619 |
| logcdf | log(cdf) | log of an accurate cdf; for large kappa a log-space tail asymptotic (log pdf(x) - log(kappa sin\|x\|) + ...) where cdf underflows (kappa = 1e4, x = -0.5: true 1.8e-534) | left tail; currently also inherits F2: `logcdf(-3.0, 200)` = -401.7594268927412 vs -399.36336618216652 |
| ppf | numerical inversion (brentq) of cdf | Newton on an accurate cdf, started from the normal approximation; no closed form | small q: `ppf(1e-6, 3.99390425810714)` = -3.1377593539253286 vs -3.1377593539232235 (6.7e-13 rel.), inherits F1/F2 |
| isf | ppf(1 - q) | `-ppf(q)` by symmetry | small q: `isf(1e-6, 3.99390425810714)` = 3.1377593539271884 vs 3.1377593539232235 (1.3e-12 rel.) |

## Shared code

`_stats.von_mises_cdf` (compiled, `scipy/stats/_stats.pyx`) is used by both
`vonmises` and `vonmises_line`; F1 and F2 apply to both, and to everything
built on `vonmises.cdf` (e.g. goodness-of-fit tests with the von Mises null).
No scipy.special function was found inaccurate (`cosm1`, `i0e`, `ndtr` are
used correctly; the error in F2 is the approximation itself).

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-2.5, 20)` | 0.0 | 3.9876789625197513e-17 | verified | loss. Reference from mp.quad of the scaled pdf with 16 equal nodes; agrees with the Fourier series (x+pi)/(2pi) + sum I_j/I_0 sin(jx)/(j pi) at 450 digits to 1e-201. 100 and 200 digits agree to 1e-101 |
| F1 | `cdf(-1.2566370614359172, 49.9)` | 0.0 | 6.2419102038833848e-17 | verified | loss |
| F1 | `cdf(-2.827433388230814, 49.9)` | 2.6867397195928788e-14 | 1.2153821284144994e-43 | verified | loss (1.0e45 ulp) |
| F1 | `cdf(-1.0, 49.9)` | 7.2405414996978834e-12 | 7.1993581870245836e-12 | verified | severe (2.6e13 ulp) |
| F1 | `cdf(-2.0, 10)` | 1.0518147133820044e-07 | 1.0518147714716648e-07 | verified | severe (2.5e8 ulp) |
| F1 | `cdf(-3.0, 3.99390425810714)` | 3.7435583436405684e-05 | 3.7435583432709568e-05 | verified | moderate (4.4e5 ulp). Mechanism checked: a Python port of the backward-recursion series `0.5 + x/(2pi) + V/pi` with p = int(29 + 0.5k - 100/(k+5)), clipped to [0, 1], gives the same 0.0 results and agrees with SciPy elsewhere to a few eps absolute (`scratch/vonmises_line/verify/series_reimpl.py`) |
| F2 | `cdf(-0.1, 50)` | 0.2403959820672798 | 0.24039668356440693 | verified | severe (1.3e10 ulp). SciPy equals `ndtr(sqrt(2/pi)/i0e(k)*sin(x/2))` bit for bit here and at (200, -0.5), (1000, -1.0), (50, -3.0) |
| F2 | `cdf(-3.0, 50)` | 2.857552395811191e-45 | 1.7715594808454948e-44 | verified | severe (3.8e15 ulp) |
| F2 | `cdf(-3.0, 200)` | 3.2968398261591996e-175 | 3.6198768164335783e-174 | reference disagreement (severity unaffected) | Analyst's 3.619876824209961e-174 is off by 2.1e-9 relative. Mine comes from the u = sin^2(t/2) substitution with nodes clustered at the endpoint (100 = 200 digits), the 200-digit t-quad and the 450-digit Fourier series. SciPy rel. error 0.909 (4.1e15 ulp) either way |
| F2 | `cdf(-1.0, 1000)` | 3.3457992891950757e-202 | 3.3987134337299178e-202 | reference disagreement (severity unaffected) | Analyst's 3.3987151744337675e-202 is off by 5.1e-7 relative. Mine is confirmed by Watson's lemma (asymptotic series, last term 1e-160) and by quad in v = u - u0 with nodes at multiples of 1/kappa; the two agree to 25 digits at 100 and 200 digits. Cause: `_lower` in `refs/vonmises_line_ref.py` puts breakpoints at -2^j/sqrt(kappa), but in the tail the integrand is concentrated within ~1/(kappa sin\|x\|) of the endpoint x, which tanh-sinh does not fully resolve. Agreement between precisions does not detect this. My own first attempt, a t-quad with equal nodes, was wrong by 1e-3 here even though its 100- and 200-digit values agreed. SciPy rel. error 1.56e-2 (7.0e13 ulp) with either reference |
| F2 | `cdf(0.05, 50)` | 0.6378151549842074 | 0.63781504936830103 | verified | severe (7.5e8 ulp) |
| F2 | `cdf(-1.0, 50)` vs `cdf(-1.0, 49.9)` | 6.75819958997843e-12 / 7.2405414996978834e-12 | 6.8692351545869158e-12 / 7.1993581870245836e-12 | verified | the discontinuity at kappa = 50 is confirmed |

References for cdf(-1.0, 1000) and cdf(-3.0, 200) corrected after verification (tail of `_lower` in `refs/vonmises_line_ref.py` now uses the u = sin^2(t/2) substitution with breakpoints on the 1/(2 kappa) scale; new values agree with the verifier's; ulp figures and severities unchanged).

Verified by dist-accuracy-verifier: 2 of 2 verified.
