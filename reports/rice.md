# rice: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `rice_gen`, lines 9337-9397
- **Implemented:** pdf (9383-9390, `x * exp(-(x-b)**2/2) * i0e(x*b)`), cdf (9377-9378, `sc.chndtr(x**2, 2, b**2)`), ppf (9380-9381, `sqrt(sc.chndtrix(q, 2, b**2))`). cdf and ppf are ncx2 with df = 2 and nc = b**2 at x**2, i.e. Boost `non_central_chi_squared` through `sc.chndtr`/`sc.chndtrix`.
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** ncx2 (same Boost functions `sc.chndtr`/`sc.chndtrix`, reports/ncx2.md); rayleigh (b = 0, reports/rayleigh.md); helpers: none; special functions: `sc.chndtr`, `sc.chndtrix`, `sc.i0e` (`sc.gamma`, `sc.hyp1f1` only in `_munp`, not measured)
- **Reference:** `refs/rice_ref.py`, 100 digits. pdf from the docstring formula (`mp.besseli`); cdf/sf are the Poisson mixtures `sum_j w_j P(1+j, x**2/2)` / `Q(1+j, x**2/2)` with positive terms only (same algorithm as `refs/ncx2_ref.py`, df = 2); ppf/isf invert the cdf or the sf. Self-checks: PASS (run A; runs B and C, ppf only, used `--no-checks` after the PASS).
- **Parameter sets:** (0.7749725210111873), (1e-5), (0), (3), (15), (40); single points (`--at`) also at b = 14.14, 14.143, 30, 35, 37, 1e154, 2e154, 1e155.
- **Worst finding:** loss in cdf, ppf and pdf

## Summary

All three implemented methods are accurate in the bulk (pdf <= 2.8 ulp, cdf <= 15 ulp, ppf <= 7.6 ulp). b = 0 (Rayleigh) is accurate everywhere (pdf 0.96 ulp, cdf 1.1 ulp, ppf 0.95 ulp). The cdf and ppf get all of the Boost ncx2 defects with df = 2, nc = b**2 (reports/ncx2.md F1, F6, F7, F8):
- For b >= 14.143 (nc >= 200) the cdf returns 0 in the whole left tail.
- In the same regime ppf returns a constant, or nan for b = 40.
- For x ~ 3e-81 ... 1e-78, at any b < 14.143, the cdf loses up to every digit (relative error 1.09), and the ppf inherits a 1.7 % error there.
- For q -> 1 the ppf exceeds its (huge) conditioning by a factor 2-3.

The pdf returns 0 instead of ~0.4 when x*b overflows (x ~ b >= 1.34e154). It also loses up to ~b**2/2 ulp for small x and moderate b (minor).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | b >= 14.143 (nc = b**2 >= 200), left tail below a b-dependent edge (x < 0.36 at b = 15, x < 16.74 at b = 40) | Boost `chndtr` starts the Poisson series at the mode and returns 0 when the starting term underflows (ncx2 F1; patterns 4, 10) | 0.0 for 6.9e-56; 6.0e13 ulp at the edge | loss |
| F2 | ppf | b >= 14.143, q below the smallest nonzero Boost cdf | `chndtrix` root finder on a cdf that is 0 (ncx2 F6): constant result, or nan at b = 40 | 0.3589 for 3.8e-126; nan for 2.99 | loss |
| F3 | pdf | x ~ b, x*b > 1.8e308 (b >= ~1.34e154) | `x*b` overflows, `i0e(inf) = 0` (pattern 3) | 0.0 for 0.399 | loss |
| F4 | cdf | b < 14.143, x ~ 3e-81 ... 1e-78 (`(x**2/2)**2` subnormal) | Boost Ding series starts with a subnormal `gamma_p_derivative` prefix (ncx2 F7; pattern 4) | 4.9e15 ulp (relative error 1.09) | loss |
| F5 | ppf | b < 14.143, q such that the root is in the F4 band | `chndtrix` inverts the F4 cdf | 7.5e13 ulp (1.7 %) | severe |
| F6 | ppf | q -> 1, all b > 0 measured with flagged points (15, 40) | Boost solves cdf(x) = q near 1, adding the absolute error of the cdf to the rounding of q (ncx2 F8; pattern 8) | 3.4e13 ulp at cond 1.6e13 | severe |
| F7 | pdf | small x (cond ~1), moderate b (10 ... 37) | the exp argument `(x-b)**2/2 ~ b**2/2` carries an absolute rounding error of ~b**2/2 * eps (patterns 5, 6) | 605 ulp at cond 1 | minor |

### F1: cdf returns 0 in the left tail for b >= 14.143

- **Code:** `_cdf`, line 9378: `sc.chndtr(np.square(x), 2, np.square(b))`.
- **Mechanism:** this is ncx2 F1. For nc >= 200, Boost evaluates the cdf (x <= df + nc) as a Poisson series that starts at the mode k = nc/2 = b**2/2. If `gamma_p(1 + k, x**2/2)` underflows there, Boost returns 0, although the sum is dominated by j near 0 and is a normal number (~ x**2/2 * exp(-b**2/2) for small x). Just inside the edge the starting term is subnormal, and only a few digits survive (patterns 4, 10).
- **Regime:** b**2 >= 200, i.e. b >= 14.1422. Direct check: `cdf(1e-3, 14.14)` is fine (6.7 ulp), `cdf(1e-3, 14.143)` = 0. The cdf is 0 for x < 0.2227 at b = 14.143, x < 0.3597 at b = 15, x < 1.95 at b = 20, x < 16.74 at b = 40 and x < 74.4 at b = 100 (first nonzero values on a 4000-point grid). Harness: 19 losses at b = 15 (x from 1e-120 to 0.1), 15 at b = 40 (x from 3 to 10). Below these x the true cdf is subnormal.
- **Reproducer:** `python accuracy.py refs/rice_ref.py --at cdf 0.001 15`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-10, 15)` | 0.0 | 6.931716468205854e-70 | inf | 2 | loss |
  | `cdf(0.1, 15)` | 0.0 | 9.048338467218716e-52 | inf | 2.51 | loss |
  | `cdf(0.001, 14.143)` | 0.0 | 1.8374838521633284e-50 | inf | 2 | loss |
  | `cdf(10.0, 40)` | 0.0 | 2.4500393022163492e-198 | inf | 301 | loss |
  | `cdf(0.36, 15)` | 1.1317647912491813e-49 | 1.1168702313421318e-49 | 6.0e+13 | 5.88 | severe |
  | `cdf(16.75, 40)` | 4.613338446689296e-120 | 4.5976301201416325e-120 | 1.5e+13 | 391 | severe |
  | `cdf(0.4, 15)` | 2.1367270372386982e-49 | 2.1367270372463953e-49 | 1.62e+04 | 6.44 | moderate |
  | `cdf(17.0, 40)` | 1.5177311798105688e-117 | 1.5177311798336148e-117 | 6.84e+04 | 392 | moderate |

  (cond (shapes) is 225 at b = 15 and 932-1.5e3 at b = 40. This is context only.)
- **Fix direction:** upstream in Boost (see ncx2 F1). Locally, use the Poisson series from j = 0 (Ding) for small x at any b, or a log-space Marcum-Q evaluation.

### F2: ppf returns a constant or nan in the left tail for b >= 14.143

- **Code:** `_ppf`, line 9381: `np.sqrt(sc.chndtrix(q, 2, np.square(b)))`.
- **Mechanism:** this is ncx2 F6 (inherits F1). Below the edge of F1 the Boost cdf is 0, so the root finder cannot go further into the tail.
  - At b = 15, `chndtrix` returns the edge value for every q <= 1.5e-51, and ppf = 0.358906933350114 (the F1 edge).
  - At b = 40 it returns the edge 16.72465662984 for q in [2.6e-126, 1.3e-188]. For q <= 4.6e-201 it returns nan, without a warning (`sc.chndtrix(1e-201, 2, 1600)` = nan). The nan presumably comes from a Boost root-bracketing error mapped to nan, but this was not traced in the Boost source.
- **Regime:** b >= 14.143, q below the smallest nonzero Boost cdf value. 21 losses at b = 15. At b = 40: 9 losses (nan) and 6 severe points, where the constant is 4-56 % off.
- **Reproducer:** `python accuracy.py refs/rice_ref.py --at ppf 1e-60 15`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 15)` | 0.358906933350114 | 3.798215527547615e-126 | 4.3e+140 | 0.5 | loss |
  | `ppf(1e-60, 15)` | 0.3589069333501139 | 3.7982155267839134e-06 | 4.3e+20 | 0.5 | loss |
  | `ppf(1.4677992676221335e-51, 15)` | 0.3589069333501138 | 0.12038810130566534 | 8.9e+15 | 0.368 | loss |
  | `ppf(1e-300, 40)` | nan | 2.9880020605562105 | inf | 0.00899 | loss |
  | `ppf(4.641588833612982e-201, 40)` | nan | 9.79233838692414 | inf | 0.00337 | loss |
  | `ppf(1.333521432163324e-188, 40)` | 16.724656629842595 | 10.75468298973155 | 2.5e+15 | 0.00317 | severe |
  | `ppf(2.6101572156826508e-126, 40)` | 16.724656629842595 | 16.14130665758982 | 1.6e+14 | 0.00259 | severe |

- **Fix direction:** fix F1. Until then, return nan or warn when the cdf at the bracket end is 0. For small q an asymptotic start `x ~ sqrt(2 q exp(b**2/2))` is also available.

### F3: pdf returns 0 when x*b overflows

- **Code:** `_pdf`, line 9390: `x * np.exp(-(x-b)*(x-b)/2.0) * sc.i0e(x*b)`.
- **Mechanism:** pattern 3. The pdf is non-negligible only for |x - b| <~ 38. For x ~ b >= 1.34e154 the product `x*b` overflows to inf, and `sc.i0e(inf)` = 0. The true value of `i0e(x b)` is ~1/sqrt(2 pi x b), which `x` multiplies back to ~0.4. `sc.i0e(1e300)` = 3.99e-151 is still accurate, so only the overflow of the argument fails.
- **Regime:** x*b > 1.8e308 with |x - b| small, i.e. b (and x) >~ 1.34e154. Not on the harness grid (b <= 40); measured with `--at`.
- **Reproducer:** `python accuracy.py refs/rice_ref.py --at pdf 2e154 2e154`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(2e+154, 2e+154)` | 0.0 | 0.3989422804014327 | inf | 0 | loss |
  | `pdf(1e+155, 1e+155)` | 0.0 | 0.3989422804014327 | inf | 0 | loss |
  | `pdf(1e+154, 1e+154)` | 0.3989422804014327 | 0.3989422804014327 | 0 | 0 | ok |

- **Fix direction:** write `x * i0e(x*b)` as `sqrt(x/b) * (sqrt(x*b) * i0e(x*b))`, with the asymptotic `1/sqrt(2 pi)` for the bracket when `x*b` overflows. Alternatively, branch on `x*b > 1e300` to `sqrt(x/(2 pi b)) * exp(-(x-b)**2/2)`.

### F4: cdf loses every digit in a narrow left-tail band (x ~ 3e-81 ... 1e-78)

- **Code:** `_cdf`, line 9378 (`sc.chndtr`, Boost Ding series for nc < 200).
- **Mechanism:** this is ncx2 F7 with df = 2. Ding's series starts with `gamma_p_derivative(2, x**2/2)`. Its regularized prefix `(x**2/2)**2 e^{-x**2/2}` is subnormal for x**2/2 in ~[2e-162, 1.5e-154], i.e. x in ~[2e-81, 1.7e-77], so few significant bits survive (pattern 4). Below the band the prefix underflows to 0 and Boost apparently takes a log-space path (<= 180 ulp, see "Checked and fine"). The error does not depend on b: the same ulp values appear at b = 1e-5, 0.775 and 3. The relative error in a fine scan rises to 1.09 at x = 2.95e-81 and falls to 1e-14 at 2.9e-81, which is the lower band edge. The mechanism is inferred from the band position; the Boost source was not inspected.
- **Regime:** 0 < b < 14.143 (all Ding-path sets: 1e-5, 0.775, 3), x in ~[2.95e-81, 1e-78]. The cdf there is ~x**2/2 * exp(-b**2/2) ~ 1e-161, a normal double.
- **Reproducer:** `python accuracy.py refs/rice_ref.py --at cdf 2.95e-81 0.7749725210111873`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(2.95e-81, 0.7749725210111873)` | 6.727377442519947e-162 | 3.2225467270756576e-162 | 4.9e+15 | 2 | loss |
  | `cdf(3e-81, 3.0)` | 9.757464087670858e-164 | 4.999048442209037e-164 | 4.3e+15 | 2 | severe |
  | `cdf(3e-80, 1e-05)` | 4.50016948682855e-160 | 4.499999999775e-160 | 1.7e+11 | 2 | severe |
  | `cdf(1e-80, 0.7749725210111873)` | 3.7029713887407634e-161 | 3.703012613703714e-161 | 5.0e+10 | 2 | severe |
  | `cdf(1e-79, 0.7749725210111873)` | 3.7030131020548257e-159 | 3.703012613703714e-159 | 5.9e+08 | 2 | severe |
  | `cdf(1e-78, 0.7749725210111873)` | 3.703012613569964e-157 | 3.703012613703714e-157 | 1.6e+05 | 2 | moderate |

- **Fix direction:** upstream in Boost, compute the incomplete-gamma prefix in log space before it becomes subnormal (ncx2 F7). Locally, for tiny x use the leading term `-expm1(-x**2/2) * exp(-b**2/2) * (1 + ...)` of the Marcum-Q series.

### F5: ppf inherits the F4 band (1.7 % error)

- **Code:** `_ppf`, line 9381 (`sc.chndtrix` inverts the Boost cdf of F4).
- **Mechanism:** the root finder solves the inaccurate (and locally non-monotone) F4 cdf. The error in x is about half the cdf error, because the cdf is ~x**2. A direct scan gives ppf relative errors of -1.7 % for roots near 3e-81 and -0.9 % near 4.45e-81, falling to 1.8e-7 at 1e-79.
- **Regime:** b < 14.143, q ~ exp(-b**2/2) * [4e-162, 5e-157] (roots in the F4 band). Found at b = 3 on the harness grid; the same at b = 0.775 by direct probe.
- **Reproducer:** `python accuracy.py refs/rice_ref.py --at ppf 3.333e-162 0.7749725210111873`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(3.333e-162, 0.7749725210111873)` | 2.9498534537674374e-81 | 3.000129912878631e-81 | 7.5e+13 | 0.5 | severe |
  | `ppf(1.1006941712522576e-163, 3)` | 4.411059766872347e-81 | 4.451546511839082e-81 | 4.1e+13 | 0.5 | severe |

- **Fix direction:** fix F4. Alternatively, for tiny q use `x = sqrt(2 q exp(b**2/2))` with a series correction.

### F6: ppf near q = 1 exceeds the conditioning by a factor 2-3

- **Code:** `_ppf`, line 9381.
- **Mechanism:** this is ncx2 F8 (pattern 8). Boost solves cdf(x) = q for q close to 1, not sf(x) = 1 - q. The few-eps absolute error of its cdf near 1 adds to the rounding of q. Most of the error is the conditioning q/(x f(x)), which is huge here. The excess beyond it is a factor 2.1-3.4 at b = 15 and up to ~11 at b = 40 (`ppf(1 - 4.5e-11, 40)`: 7.6e8 ulp at cond 6.8e7).
- **Regime:** q >~ 1 - 1e-5. Points beyond conditioning: b = 15 (7 severe, 2 moderate, 1 minor) and b = 40 (8 severe, 3 moderate, 2 minor). At b = 0.775 and b = 3 the same points are large but within conditioning (e.g. 3.9e13 ulp at cond 4.0e13).
- **Reproducer:** `python accuracy.py refs/rice_ref.py --at ppf 0.9999999999999997 15`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999999997, 15)` | 22.930016816055357 | 23.103234311158456 | 3.4e+13 | 1.6e+13 | severe |
  | `ppf(0.9999999999999997, 40)` | 47.811524755353474 | 48.08795632026136 | 2.6e+13 | 7.6e+12 | severe |
  | `ppf(0.9999999999522942, 40)` | 46.48562540459734 | 46.48563320851418 | 7.6e+08 | 6.8e+07 | severe |
  | `ppf(0.999999938945977, 15)` | 20.318857226980555 | 20.318857228544154 | 3.5e+05 | 1.5e+05 | moderate |

- **Fix direction:** for q > 1/2 use `sqrt(scu._ncx2_isf(1 - q, 2, b**2))` (1 - q is exact there). This is also the natural `_isf`.

### F7: pdf loses up to ~b**2/2 ulp for small x

- **Code:** `_pdf`, line 9390: `np.exp(-(x-b)*(x-b)/2.0)`.
- **Mechanism:** for small x the exp argument is ~ -b**2/2, as large as 700 in magnitude. The rounding of `x - b` and of the square leaves an absolute error of ~b**2/2 * eps in the argument, and `exp` turns it into a relative error of the same size (patterns 5, 6). The pdf itself is ~x there (cond 1). The error grows with b: 43 ulp at b = 15, 142 at b = 30, 394 at b = 35, 605 at b = 37. cond (shapes) is about b**2 (224-1.4e3), so a 1-ulp change of b would cost as much; a double-precision exp argument cannot do much better without extended precision.
- **Regime:** x << b, b >~ 10, pdf still normal (b <~ 37.5).
- **Reproducer:** `python accuracy.py refs/rice_ref.py --at pdf 0.001 37`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.001, 37)` | 5.315884602030445e-301 | 5.315884602031159e-301 | 605 | 1 | minor |
  | `pdf(0.001, 35)` | 9.88012887759839e-270 | 9.880128877599255e-270 | 394 | 1 | minor |
  | `pdf(1e-10, 30)` | 3.69388306848714e-206 | 3.6938830684872564e-206 | 142 | 1 | minor |
  | `pdf(0.1, 15)` | 2.2715375214160123e-50 | 2.271537521416034e-50 | 43.3 | 1.88 | minor |

- **Fix direction:** evaluate `(x-b)**2/2` in double-double (two-product), or accept it as within the shape conditioning.

## RuntimeWarnings

- pdf: `overflow encountered in multiply` comes from `x*b` in F3 (inaccurate result). `underflow encountered in exp/multiply/divide` appears in the far tails, where the pdf is subnormal or 0 (accurate).
- cdf: `overflow encountered in square` for x > 1.34e154 is spurious: x**2 = inf gives cdf = 1.0, which is correct (high tail <= 7.5 ulp). `underflow encountered in square` for x < 1.5e-154 is also spurious in the measured sense: the true cdf there is subnormal.
- ppf at b = 40: the nan results of F2 are returned without any warning.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, all six sets (bulk 2.84 ulp; tails within conditioning, e.g. 514 ulp at cond 1.3e3 in the right tail of b = 1e-5) | 43.3 ulp minor (F7) |
| pdf | b = 0 (Rayleigh), x*b subnormal or 0 (tiny x), x = b = 1e154 | 0.96 ulp |
| cdf | bulk (14.9 ulp), right tail (7.5 ulp), b = 0 (1.1 ulp; `chndtr` with nc = 0 works), b = 14.14 just below the nc = 200 switch (6.7 ulp) | 14.9 ulp |
| cdf | left tail below the F4 band (x <= 1e-81), b < 14.143 | 179 ulp minor (b-independent: 158, 169, 179, 108, 56 ulp at x = 1e-150 ... 1e-90). Presumably Boost's log-space prefix, where `exp` of an argument of size ~1400 amplifies its rounding. Mechanism inferred, not in SciPy code |
| cdf | b = 15, x in 0.5 ... 2.5 (just right of the F1 edge) | 44 ulp minor (cond 8-32; Boost mode-start series, mechanism unclear) |
| ppf | bulk (7.6 ulp), b = 0 (0.95 ulp), q -> 1 at b = 0.775 and 3 (within conditioning) | 7.6 ulp |
| ppf | small q (roots below the F4 band) at b = 1e-5, 0.775, 3 | 93.5 ulp minor (inherits the minor cdf left-tail error) |

## Suspected, not confirmed

- The cdf's `np.square(x)` underflow for x < 1.5e-154 (pattern 4) only matters where the true cdf is itself subnormal (cdf ~ x**2/2), so it cannot produce a measurable finding. Likewise `np.square(b)` underflows to nc = 0 for b < 1.5e-154. The resulting nc = 0 is then a relative change in the cdf of order b**2/2, far below eps, so it is harmless.
- `_munp` uses `sc.gamma(n1) * sc.hyp1f1(n1, 1, b2) * exp(-b2)`, which overflows/underflows for large b (exp(-b**2/2) underflows at b > ~38.6 while hyp1f1 overflows) (pattern 3). Moments are outside the scope; not measured.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | `scu._ncx2_sf(x**2, 2, b**2)` (Boost complementary series), or Marcum Q_1(b, x) | right tail: `sf(9, 0.775)` = 2.220446049250313e-16 vs 3.4116684912600676e-16 (`_ncx2_sf` gives 3.411668491260068e-16); `sf(30, 0.775)` = 0.0 vs 2.9191362835714458e-187 (`_ncx2_sf`: 2.9191362835714463e-187). For nc > ~50 the Boost sf has its own far-tail loss (ncx2 F1) |
| isf | ppf(1 - q) | `sqrt(scu._ncx2_isf(q, 2, b**2))` | small q: `isf(1e-20, 0.775)` = inf vs 10.176070862873713 (`_ncx2_isf` gives 10.176070862873713) |
| logpdf | log(pdf) | `log(x) - (x-b)**2/2 + log(i0e(x*b))` (with the F3 overflow guard) | pdf underflows: `logpdf(40, 0.775)` = -inf vs -768.2443255707016 (the formula gives -768.2443255707016) |
| logcdf | log(cdf) | small x: `log(-expm1(-x**2/2)) - b**2/2 + log1p(series)`; otherwise log of the cdf with a log-space Poisson sum | `logcdf(1e-200, 0.775)` = -inf vs -922.0274755823394 |
| logsf | log(sf) | log-space Marcum Q; asymptotically `-(x-b)**2/2 + 0.5 log(x/b) + log(erfc-type correction)` | `logsf(40, 0.775)` = -inf vs -771.9139739994527 |

## Shared code

- `sc.chndtr` / `sc.chndtrix` (Boost `non_central_chi_squared`): F1, F2, F4, F5 and F6 are the ncx2 findings F1, F6, F7 and F8 of reports/ncx2.md, reached through df = 2, nc = b**2, x**2. Every distribution or user calling `chndtr`/`chndtrix` is affected. New relative to ncx2.md: `chndtrix` returns nan (not the edge constant) without a warning for very small q at nc = 1600 (`sc.chndtrix(1e-201, 2, 1600)` = nan; `sc.chndtrix(1e-190, 2, 1600)` = 279.714).
- `sc.i0e` itself is accurate; F3 is an argument overflow in rice's own code.
- Cross-cutting: INDEX item 1 (sf = 1 - cdf inherited, closed/complementary form available) applies; INDEX item 2 (x**2 overflow) appears here only as the spurious `np.square` warnings and as the `x*b` overflow of F3.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(0.1, 15)` | 0.0 | 9.0483384672187158e-52 | verified | Reference is mp.quad of the docstring pdf, normalized to O(1) because mp.quad uses an absolute tolerance. 100 and 200 digits agree to 1e-101. Also verified: `cdf(1e-10, 15)` = 0 (ref 6.9317e-70), `cdf(0.001, 14.143)` = 0 (ref 1.8375e-50), `cdf(10, 40)` = 0 (ref 2.4500393022163491e-198), `cdf(0.36, 15)` 6.01e13 ulp, `cdf(16.75, 40)` 1.54e13 ulp. `sc.chndtr(1e-6, 2, 225)` = 0. Line 9378 is correct. The Boost mode-start mechanism is inherited from ncx2 and was not traced here |
| F2 | `ppf(1e-300, 40)` | nan | 2.9880020605562105 | verified | Quantiles by Newton on log(cdf) of the quad cdf, with cdf(x_ref)/q - 1 < 1e-198. Also verified: `ppf(1e-300, 15)` = 0.358906933350114 (ref 3.7982155275476153e-126), `ppf(1e-60, 15)` (ref 3.7982155267839136e-6), `ppf(1.333521432163324e-188, 40)` = 16.7247 (ref 10.75468298973155, 2.5e15 ulp). `sc.chndtrix(1e-201, 2, 1600)` = nan. Line 9381 is correct |
| F3 | `pdf(2e154, 2e154)` | 0.0 | 0.39894228040143268 | verified | Also `pdf(1e155, 1e155)` = 0. `pdf(1e154, 1e154)` is accurate (0.28 ulp). `2e154*2e154` = inf and `sc.i0e(inf)` = 0.0, which matches the mechanism and line 9390 |
| F4 | `cdf(2.95e-81, 0.7749725210111873)` | 6.727377442519947e-162 | 3.2225467270756574e-162 | verified | 4.9e15 ulp. Also `cdf(3e-81, 3)` 4.29e15 ulp and `cdf(3e-80, 1e-5)` 1.7e11 ulp, matching the report. The mechanism (subnormal Boost prefix) is inferred inside Boost, as the report states. The error ratio is about the same at b = 0.775 and b = 3, which fits a b-independent cause |
| F5 | `ppf(3.333e-162, 0.7749725210111873)` | 2.9498534537674374e-81 | 3.0001299128786307e-81 | verified | 7.55e13 ulp. Also `ppf(1.1006941712522576e-163, 3)` = 4.411059766872347e-81 vs 4.4515465118390825e-81 (4.1e13 ulp) |
| F6 | `ppf(0.9999999999999997, 15)` | 22.930016816055357 | 23.103234311158455 | verified | 3.38e13 ulp at cond q/(x f) = 1.58e13. Newton on log(sf) of the quad sf. Also `ppf(0.9999999999999997, 40)` 2.59e13 ulp (cond 7.62e12) and `ppf(0.9999999999522942, 40)` 7.56e8 ulp (cond 6.81e7). Evaluated at SciPy's results, the true sf differs from 1 - q by 4.6, 12.4 and 11.1 eps (absolute), which fits the stated mechanism: a q -> 1 cdf solve with a few-eps absolute cdf error. That Boost solves cdf = q was not traced |

Verified by dist-accuracy-verifier: 6 of 6 verified.
