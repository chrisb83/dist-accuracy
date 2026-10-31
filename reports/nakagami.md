# nakagami: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `nakagami_gen`, lines 7707-7804
- **Implemented:** pdf (7745-7746), logpdf (7748-7752), cdf (7754-7755), sf (7760-7761), ppf (7757-7758), isf (7763-7764)
- **Inherited (not measured):** logcdf, logsf (generic defaults: `log(cdf)` / `log(sf)`, with the median switch to `log1p(-sf)` / `log1p(-cdf)`)
- **Related:** `chi` (nakagami(nu) = chi(2 nu) / sqrt(2 nu)), `gamma`, `chi2`, `maxwell` (same incomplete-gamma code with a = nu, y = nu*x**2); `norm` only in `_entropy` (not measured); helpers: none; special functions: `sc.xlogy`, `sc.gammaln`, `sc.gammainc`, `sc.gammaincc`, `sc.gammaincinv`, `sc.gammainccinv` (`sc.poch`, `sc.digamma` only in `_stats`/`_entropy`, not measured)
- **Reference:** `refs/nakagami_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (4.967379486666624), (0.5), (1.5), (0.001), (0.05), (1000), (1e6)
- **Worst finding:** loss in cdf, sf (severe, value 1.0 for 0.60), ppf, isf (nu*x**2 / quantile underflow for small nu) and logpdf (x**2 overflow for nu < 1); severe in cdf, sf, pdf, logpdf, ppf, isf for large nu

## Summary

For moderate nu (0.5 ... 1000) and ordinary arguments every implemented
method is within a few tens of ulp. The same four mechanisms as in
`reports/chi.md` break it, because nakagami is chi with a rescaled argument
and calls the same incomplete-gamma functions with a = nu, y = nu*x**2.
(1) `nu*x*x` underflows for x < sqrt(2.2e-308/nu) (~1e-154): for nu <~ 1
cdf returns 0 (true 0.36 at nu = 0.001) and sf returns 1.0 (true 0.60), and
ppf/isf return 0 because the gamma quantile y underflows although
x = sqrt(y/nu) is a normal double. (2) `x**2` in logpdf overflows for
x > 1.34e154, which loses finite results when nu < 1. (3) For nu = 1e6 the
cephes `gammainc` 2000-term series cap (gamma F1) gives 4.3e10 ulp in cdf and
2.7e6 ulp in isf. (4) `_logpdf` cancels terms of size nu*log(nu), so logpdf
(1.3e6 ulp) and `pdf = exp(logpdf)` (3.2e6 ulp) are poor for large nu.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | nu <~ 1 (sf: nu <~ 0.03), x < sqrt(2.2e-308/nu) | `nu*x*x` underflows to 0 / subnormal before `gammainc` (patterns 4, 10) | cdf 0.0 for 0.36; sf 1.0 for 0.60 | loss |
| F2 | ppf, isf | nu <~ 1 and gamma quantile y = nu*x**2 < 2.2e-308 (nu = 0.001: q < ~0.49 for ppf, q > ~0.51 for isf) | `gammaincinv`/`gammainccinv` return an underflowed / subnormal y; `sqrt(y/nu)` would be normal (patterns 4, 8) | 0.0 for 3.7e-268; 6.3e10 ulp | loss |
| F3 | logpdf | nu < 1, 1.34e154 < x < sqrt(1.8e308/nu) | `x**2` overflows although `-nu*x**2` is a finite double (pattern 3) | -inf for -1.6e308 | loss |
| F4 | cdf, sf | nu >~ 2e5, nu*x**2 in nu - (4.5 ... >15)*sqrt(nu) | `sc.gammainc` returns an unconverged 2000-term series (shared, gamma F1; pattern 10) | cdf 4.27e10 ulp at nu = 1e6 | severe |
| F5 | pdf | large nu (all x near the mode); minor in the far left tail for small nu | `exp(_logpdf)` turns the absolute error of logpdf into relative error (patterns 3, 5) | 3.2e6 ulp at nu = 1e6 | severe |
| F6 | logpdf | nu >~ 1e3, x within a few 1/sqrt(nu) of 1 | cancellation of `xlogy(nu, nu)`, `gammaln(nu)`, `nu*x**2` (~nu log nu) into an O(1) result (pattern 5) | 1.3e6 ulp at nu = 1e6 | severe |
| F7 | ppf, isf | nu >~ 2e5, quantile in the F4 window | `gammaincinv`/`gammainccinv` invert the inaccurate `gammainc` of F4 (shared, gamma F3) | isf 2.7e6 ulp, ppf 5.2e5 ulp at nu = 1e6 | severe |
| F8 | ppf, isf | nu << 1 (0.001), bulk q where the quantile is still normal | `gammaincinv` for a << 1 rounds an exponent scaled by 1/a (shared, gamma F5) | 1.79e3 ulp | moderate |
| F9 | cdf, sf | far from the mean: nu = 1000 at x ~ 0.73 or x >~ 1.2; small x for nu ~ 0.5-5 | `exp(a*log(y) - y - lgam(a))` prefactor in cephes `igam`/`igamc` (shared, gamma F6) | sf 3.25e3 ulp, cdf 2.93e3 ulp | moderate |

### F1: cdf and sf lose nu*x*x to underflow for small nu

- **Code:** `_cdf`, line 7755: `sc.gammainc(nu, nu*x*x)`; `_sf`, line 7761: `sc.gammaincc(nu, nu*x*x)`.
- **Mechanism:** patterns 4 and 10. `nu*x*x` is subnormal for
  x < sqrt(2.2e-308/nu) and 0 below sqrt(4.9e-324/nu). Then
  `gammainc(nu, 0) = 0` and `gammaincc(nu, 0) = 1`, although
  cdf ~ (nu*x**2)**nu / Gamma(nu + 1) is still a normal double for nu <~ 1
  (for nu = 0.001 it is 0.25-0.47 there). In the subnormal range the
  relative error of y (up to 2**-(bits left)) is multiplied by nu in `y**nu`
  only for the cdf; the sf = 1 - cdf error is that absolute error.
- **Regime:** x < sqrt(2.2e-308/nu) (2.1e-154 for nu = 0.5, 6.6e-154 for
  nu = 0.05, 4.7e-153 for nu = 0.001) and nu small enough that cdf(x) is a
  normal double (nu <~ 1). sf is visibly wrong when cdf(x) > eps, i.e.
  nu <~ 0.03; for nu = 0.001 sf returns 1.0 instead of 0.52-0.75 on all of
  x < ~1e-160.
- **Reproducer:** `python accuracy.py refs/nakagami_ref.py --at cdf 1e-220 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-220, 0.001)` | 0.0 | 0.3607865381521132 | inf | 0.002 | loss |
  | `cdf(1e-200, 0.5)` | 0.0 | 7.978845608028653e-201 | inf | 1 | loss |
  | `cdf(1e-200, 0.05)` | 0.0 | 8.843224316372743e-21 | inf | 0.1 | loss |
  | `cdf(1e-160, 0.05)` | 8.842344736582362e-17 | 8.843224316372748e-17 | 4.48e+11 | 0.1 | severe |
  | `cdf(1e-160, 0.5)` | 7.978801194389789e-161 | 7.978845608028654e-161 | 2.51e+10 | 1 | severe |
  | `cdf(1e-160, 0.001)` | 0.47560360649939976 | 0.4756092850994615 | 5.4e+10 | 0.002 | severe |
  | `cdf(2.097617696340303e-157, 0.5)` | 1.6736567744217583e-157 | 1.6736567743768008e-157 | 1.21e+05 | 1 | moderate |
  | `sf(1e-200, 0.001)` | 1.0 | 0.6044054274542265 | 2.9e+15 | 0.00131 | severe |
  | `sf(4.69041575982343e-161, 0.001)` | 1.0 | 0.5251103031972295 | 4.07e+15 | 0.00181 | severe |
  | `sf(1e-160, 0.001)` | 0.5243963935006002 | 0.5243907149005386 | 4.9e+10 | 0.00181 | severe |

  (`sf` returning 1.0 for 0.60 is rated severe, not loss, because the
  relative error 0.65 is below 1; cond (shapes) for nu = 0.5 is 370-690.)
- **Fix direction:** for small y use the leading term in log space,
  `logcdf = nu*log(nu*x**2)` formed as `nu*(log(nu) + 2*log(x)) - gammaln(nu + 1)`
  (the series correction is O(y) < 1e-300), `cdf = exp(logcdf)`,
  `sf = -expm1(logcdf)`.

### F2: ppf and isf compute the quantile of nu*x**2 and underflow

- **Code:** `_ppf`, line 7758: `np.sqrt(1.0/nu*sc.gammaincinv(nu, q))`;
  `_isf`, line 7764: `np.sqrt(1/nu * sc.gammainccinv(nu, p))`.
- **Mechanism:** patterns 4 and 8. The gamma quantile y = nu*x**2 is
  (roughly) the square of the result, so it leaves the normal range long
  before x does: y underflows to 0 or is subnormal, and `sqrt(y/nu)` returns
  0 or a value with few correct digits. The `1/nu` factor helps for nu < 1
  (ppf(0.5, 0.001) = 7.2e-150 comes from a normal y = 5.2e-302), but for
  ppf(q) < ~sqrt(2.2e-308/nu) the quantile is lost. For tiny nu the
  quantile is tiny over most of (0, 1):
  ppf(q) ~ (q*Gamma(nu+1))**(1/(2 nu)) / sqrt(nu).
- **Regime:** nu <~ 1 and y < 2.2e-308: nu = 0.5, q < ~1.7e-154;
  nu = 0.05, q < ~4e-16; nu = 0.001, q < ~0.49 for ppf and q > ~0.51 for isf
  (q**(1/nu) scaling: (2.2e-308)**0.001 = 0.49).
- **Reproducer:** `python accuracy.py refs/nakagami_ref.py --at ppf 0.29 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.29, 0.001)` | 0.0 | 3.7483200837547024e-268 | inf | 500 | loss |
  | `ppf(0.42999999999999994, 0.001)` | 0.0 | 1.2854805508886705e-182 | inf | 500 | loss |
  | `ppf(1e-17, 0.05)` | 0.0 | 3.418956582727181e-170 | inf | 10 | loss |
  | `ppf(1e-16, 0.05)` | 3.419004706139477e-160 | 3.4189565827271733e-160 | 6.34e+10 | 10 | severe |
  | `ppf(1e-300, 0.5)` | 0.0 | 1.2533141373155002e-300 | inf | 1 | loss |
  | `ppf(1.1006941712522576e-163, 0.5)` | 0.0 | 1.3795155656912228e-163 | inf | 1 | loss |
  | `isf(0.57, 0.001)` | 0.0 | 1.2854805508888364e-182 | inf | 663 | loss |
  | `isf(0.71, 0.001)` | 0.0 | 3.748320083755061e-268 | inf | 1.22e+03 | loss |

- **Fix direction:** where `gammaincinv` returns y < ~1e-300, use the
  log-space leading term `x = exp((log(q) + gammaln(nu + 1))/(2*nu)) / sqrt(nu)`
  (isf: `log1p(-q)` in place of `log(q)`), optionally polished by one Newton
  step on logcdf.

### F3: logpdf overflows in x**2 for nu < 1

- **Code:** `_logpdf`, line 7752: `sc.xlogy(2*nu - 1, x) - nu*x**2`.
- **Mechanism:** pattern 3. `x**2` overflows for x > 1.34e154, so
  `-nu*x**2 = -inf`, but the true logpdf ~ -nu*x**2 is a finite double up to
  x = sqrt(1.8e308/nu). For nu >= 1 the true value is already below -1.8e308
  there, so only nu < 1 is affected; the window grows as nu shrinks
  (nu = 0.001: up to 4.2e155). `_cdf`/`_sf` evaluate `(nu*x)*x` and do not
  overflow early.
- **Regime:** nu < 1, 1.34e154 < x < sqrt(1.8e308/nu).
- **Reproducer:** `python accuracy.py refs/nakagami_ref.py --at logpdf 4e155 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(4e155, 0.001)` | -inf | -1.6e+308 | inf | 2 | loss |
  | `logpdf(2.1e155, 0.001)` | -inf | -4.41e+307 | inf | 2 | loss |
  | `logpdf(1.35e154, 0.5)` | -inf | -9.1125e+307 | inf | 2 | loss |
  | `logpdf(1.8e154, 0.5)` | -inf | -1.62e+308 | inf | 2 | loss |
  | `logpdf(1.5e154, 0.05)` | -inf | -1.1250000000000003e+307 | inf | 2 | loss |
  | `logpdf(1.34e154, 0.5)` | -8.978e+307 | -8.978e+307 | 0 | 2 | ok |

- **Fix direction:** `- (nu*x)*x`, which overflows only when the result does.

### F4: cdf and sf inherit the unconverged gammainc series for nu >~ 2e5

- **Code:** `_cdf`, line 7755 (`sc.gammainc`) and `_sf`, line 7761
  (`sc.gammaincc`, computed as `1 - igam` below the mean). The defect is in
  `scipy.special`, analysed in reports/gamma.md F1: cephes `igam_series`
  stops after 2000 terms just outside the Temme window
  `|y - a| < 4.5*sqrt(a)`.
- **Mechanism:** pattern 10 (shared). nakagami calls `gammainc(a, y)` with
  a = nu, y = nu*x**2; nothing nakagami-specific. In nakagami variables the
  window is x**2 < 1 - 4.5/sqrt(nu), i.e. about 3.2 to >10 standard
  deviations (sd ~ 1/sqrt(4 nu)) below 1.
- **Regime:** nu >~ 2e5, left tail between ~ -3 and ~ -10 sd. The values at
  nu = 1e6 equal reports/gamma.md at a = 1e6 and reports/chi.md at
  df = 2e6 digit for digit (e.g. 2.7495684457992286e-07).
- **Reproducer:** `python accuracy.py refs/nakagami_ref.py --at cdf 0.9976973488989535 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(0.9976973488989535, 1e+06)` | 2.045146368700871e-06 | 2.0451657660636757e-06 | 4.27e+10 | 9.6e+03 | severe |
  | `cdf(0.9974968671630001, 1e+06)` | 2.7495684457992286e-07 | 2.7495803592688157e-07 | 1.95e+10 | 1.04e+04 | severe |
  | `cdf(0.9964938534682489, 1e+06)` | 1.1410046127002487e-12 | 1.141004707830788e-12 | 3.75e+08 | 1.43e+04 | severe |
  | `cdf(0.99498743710662, 1e+06)` | 5.446644691873852e-24 | 5.446644693011966e-24 | 9.41e+05 | 2.02e+04 | moderate |
  | `sf(0.9976973488989535, 1e+06)` | 0.9999979548536313 | 0.9999979548342339 | 8.74e+04 | 0.0196 | moderate |
  | `sf(0.9974968671630001, 1e+06)` | 0.9999997250431554 | 0.9999997250419641 | 5.37e+03 | 0.00285 | moderate |

- **Fix direction:** in `scipy.special` (reports/gamma.md F1); nothing in
  `nakagami_gen` short of not calling `gammainc`.

### F5: pdf exponentiates a logpdf with a large absolute error

- **Code:** `_pdf`, line 7746: `np.exp(self._logpdf(x, nu))`.
- **Mechanism:** patterns 3 and 5. `exp` turns the absolute error of its
  argument into relative error. The terms of `_logpdf` (F6) are each rounded
  at their own magnitude, so the pdf's relative error is about
  `max(nu*log(nu), gammaln(nu), nu*x**2, |(2nu-1)*log(x)|)` ulp:
  ~3e6 at nu = 1e6 (terms ~1.4e7), ~1.6e3 at nu = 1000. The same effect
  appears, minor, in the far left tail of small nu, where
  `|(2nu-1)*log(x)|` ~ 600 (490 ulp at nu = 0.001, x = 1e-280).
- **Regime:** large nu, all x where the pdf is a normal double (worst near
  the mode x ~ 1).
- **Reproducer:** `python accuracy.py refs/nakagami_ref.py --at pdf 1 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1.002297361066066, 1e+06)` | 0.020900189008252586 | 0.020900189023301523 | 3.2e+06 | 9.2e+03 | severe |
  | `pdf(0.999499874937461, 1e+06)` | 484.0220719792441 | 484.0220723169546 | 3.1e+06 | 2e+03 | severe |
  | `pdf(1.0, 1e+06)` | 797.8844938003174 | 797.8844943124881 | 2.9e+06 | 1 | severe |
  | `pdf(0.9848857801796105, 1e+06)` | 3.0010795004101555e-197 | 3.0010795022542024e-197 | 2.8e+06 | 6e+04 | severe |
  | `pdf(1.0673051190017824, 1000)` | 0.0033340459928936344 | 0.003334045992892469 | 1.57e+03 | 279 | moderate |
  | `pdf(1.0, 1000)` | 25.22922269744582 | 25.229222697443 | 504 | 1 | minor |
  | `pdf(1e-280, 0.001)` | 5.473691544292678e+276 | 5.473691544292083e+276 | 490 | 0.998 | minor |

- **Fix direction:** pdf = `2*x*nu*igam_fac(nu, nu*x**2)/(nu*x**2)`, i.e.
  the Lanczos/`log1pmx` form of reports/gamma.md F2 with
  `nu*log1pmx(x**2 - 1)` as the only large term.

### F6: logpdf cancels terms of size nu*log(nu) near x = 1

- **Code:** `_logpdf`, lines 7751-7752:
  `np.log(2) + sc.xlogy(nu, nu) - sc.gammaln(nu) + sc.xlogy(2*nu - 1, x) - nu*x**2`.
- **Mechanism:** pattern 5. At nu = 1e6, x = 1 the terms are
  1.38e7 (`xlogy(nu, nu)`), 1.28e7 (`gammaln(nu)`) and 1e6 (`nu*x**2`) for a
  result of 6.68; each carries half an ulp of its own size, so the
  relative error is ~ nu*log(nu)*eps / |logpdf|.
- **Regime:** nu >~ 1e3 and |logpdf| small (x within a few sd of 1);
  599 ulp at nu = 1000 (minor), 1.3e6 at nu = 1e6.
- **Reproducer:** `python accuracy.py refs/nakagami_ref.py --at logpdf 1 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.9984988733093293, 1e+06)` | 2.1744457981549203 | 2.174445798792505 | 1.3e+06 | 2.76e+03 | severe |
  | `logpdf(1.0021975853094038, 1e+06)` | -2.9719300388824195 | -2.9719300382513474 | 9.6e+05 | 2.96e+03 | moderate |
  | `logpdf(1.0, 1e+06)` | 6.6819638423621655 | 6.681963843004076 | 4.3e+05 | 0.15 | moderate |
  | `logpdf(0.9999997499999688, 1e+06)` | 6.681963967392221 | 6.681963968004097 | 4.1e+05 | 1.07e-11 | moderate |
  | `logpdf(1.0463595604786393, 1000)` | -1.05153462131207 | -1.05153462131193 | 599 | 181 | minor |

- **Fix direction:** same as F5: `log(2*nu*x) - log(y) + log(igam_fac(nu, y))`
  with y = nu*x**2, where `nu*log1pmx(x**2 - 1)` (plus O(log nu) terms) is
  the only large term.

### F7: ppf and isf inherit F4 through gammaincinv / gammainccinv

- **Code:** `_ppf`, line 7758; `_isf`, line 7764.
- **Mechanism:** shared (reports/gamma.md F3): the inverse functions
  Newton-iterate on the inaccurate `igam` of F4 and converge to the wrong
  root; the relative error in y is halved by the square root.
- **Regime:** nu >~ 2e5 with the quantile in the F4 window: at nu = 1e6,
  ppf for q in ~[1e-26, 1e-6] and isf for 1 - q in the same range.
- **Reproducer:** `python accuracy.py refs/nakagami_ref.py --at isf 0.9999993371296838 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.9999993371296838, 1e+06)` | 0.9975828813277674 | 0.9975828807250314 | 2.7e+06 | 150 | severe |
  | `isf(0.999999938945977, 1e+06)` | 0.9973558449302018 | 0.9973558447023979 | 1.0e+06 | 1.5e+03 | severe |
  | `isf(0.9999999943765867, 1e+06)` | 0.997145810266013 | 0.9971458101731336 | 4.2e+05 | 1.52e+04 | moderate |
  | `ppf(1e-08, 1e+06)` | 0.9971951464055372 | 0.9971951462908959 | 5.2e+05 | 8.67e-05 | moderate |
  | `ppf(3.4807005884285624e-14, 1e+06)` | 0.9962578747530619 | 0.996257874750916 | 9.7e+03 | 6.58e-05 | moderate |

- **Fix direction:** none inside `nakagami_gen`; fixing gamma F1 fixes it.

### F8: ppf/isf for tiny nu amplify the gammaincinv exponent rounding

- **Code:** `_ppf`, line 7758 / `_isf`, line 7764 (inside `sc.gammaincinv`).
- **Mechanism:** shared (reports/gamma.md F5): for a << 1, cephes `igami`
  returns y = exp(exponent/a) with |log y| in the hundreds, so the rounding
  of the exponent is multiplied by |log y|; the square root halves it.
- **Regime:** nu <~ 0.01, q where the quantile is still normal (above F2):
  q ~ 0.5-0.85 at nu = 0.001.
- **Reproducer:** `python accuracy.py refs/nakagami_ref.py --at ppf 0.8499999999999999 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.5000000000000001, 0.001)` | 7.241689311393424e-150 | 7.2416893113963e-150 | 1.79e+03 | 500 | moderate |
  | `ppf(0.8499999999999999, 0.001)` | 1.214228934178137e-34 | 1.2142289341785332e-34 | 1.47e+03 | 500 | moderate |
  | `isf(0.4999999999999999, 0.001)` | 7.241689311393424e-150 | 7.2416893113963e-150 | 1.79e+03 | 500 | moderate |
  | `isf(0.36, 0.001)` | 2.9162497352461777e-96 | 2.9162497352468737e-96 | 1.07e+03 | 281 | moderate |
  | `ppf(0.71, 0.001)` | 1.0092809589921698e-73 | 1.0092809589923147e-73 | 646 | 500 | minor |

  (`--at ppf 0.5 0.001` reports cond (arg) 1.5e221 at exactly q = 0.5, a
  numerical-derivative artefact of the harness; the neighbouring grid point
  0.5000000000000001 has cond 500.)
- **Fix direction:** in `scipy.special` `igami` (final Newton step on log P),
  see gamma F5; the log-space formula of F2 also covers this regime.

### F9: cdf and sf lose the rounding of the igam prefactor exponent

- **Code:** `_cdf`, line 7755; `_sf`, line 7761; the loss is in `igam_fac`
  inside `sc.gammainc`/`sc.gammaincc` (`exp(a*log(y) - y - lgam(a))`).
- **Mechanism:** shared (reports/gamma.md F6): the exponent is a sum of
  terms of size |a log y|, y, lgam(a); its rounding becomes a relative error
  of that many ulp.
- **Regime:** nu = 1000 with y far from a (x ~ 0.73, x >~ 1.2): moderate;
  small x for nu ~ 0.5-5 (x from 1e-10 down to the F1 range): minor,
  up to 262 ulp.
- **Reproducer:** `python accuracy.py refs/nakagami_ref.py --at sf 1.3959524698393257 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1.3959524698393257, 1000)` | 7.180551658663522e-125 | 7.180551658658347e-125 | 3.25e+03 | 1.9e+03 | moderate |
  | `sf(1.5, 1000)` | 2.078528343650372e-193 | 2.0785283436491197e-193 | 2.71e+03 | 2.5e+03 | moderate |
  | `sf(1.2142247110915083, 1000)` | 1.037886426558472e-39 | 1.037886426557919e-39 | 2.4e+03 | 955 | moderate |
  | `cdf(0.7250230003073993, 1000)` | 1.3530223125744093e-75 | 1.3530223125735276e-75 | 2.93e+03 | 951 | moderate |
  | `cdf(1e-20, 4.967379486666624)` | 5.1014823350959675e-198 | 5.1014823350962646e-198 | 262 | 9.93 | minor |
  | `cdf(1e-80, 1.5)` | 1.381976597885273e-240 | 1.3819765978853418e-240 | 224 | 3 | minor |
  | `cdf(1e-120, 0.5)` | 7.978845608028435e-121 | 7.978845608028654e-121 | 123 | 1 | minor |

  (The nu = 1000 errors exceed cond (arg) by only 1.3-2.4x; cond (shapes)
  there is 86-440.)
- **Fix direction:** in `scipy.special` (gamma F6); for small x the
  log-space leading-term formula of F1.

## RuntimeWarnings

- `overflow encountered in square` (logpdf, pdf) for x > 1.34e154: wrong
  result in logpdf for nu < 1 (F3); for nu >= 1 the true value is not a
  double (logpdf) or is 0 (pdf), so the warning is harmless there.
- `underflow encountered in multiply` (cdf, sf) for nu*x*x < 2.2e-308:
  accompanies the wrong results of F1; `underflow encountered in square`
  (logpdf, pdf) for x < 1.5e-154: there `nu*x**2` is negligible and the
  result is accurate (spurious warning).
- `overflow encountered in multiply` (cdf, sf) for huge x: the true value
  is 1 / 0 and the result is right (harmless).
- `underflow encountered in exp` (pdf): genuine underflow of the pdf.
- ppf/isf emit no warning in F2: the underflow happens silently inside
  `gammaincinv`/`gammainccinv`.
- The harness evaluates whole grids per call, so warnings are attributed to
  every point of a parameter set, including accurate ones.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | high tail, all parameter sets; bulk and low tail for nu = 4.97, 1.5, 1000 outside F9 | 4 ulp (high tail); low tail minor only (F9) |
| sf | low tail outside F1/F4 and bulk for nu = 4.97, 0.5, 1.5, 0.05 | 116 ulp (nu = 4.97, cond 797); minor 41.8 ulp at `sf(1.0, 0.5)` (cond 1.53), 17.3 ulp at nu = 0.05 (no clear mechanism; likely cephes `igamc` rounding, cf. chi `sf(1, 1)`) |
| ppf | high tail all sets; low tail for nu = 4.97, 1.5, 1000 | 88 ulp (nu = 1.5, q ~ 1e-288, minor, igami analogue of F9); 29.3 ulp (nu = 4.97); 4.02 (nu = 1000); 60 ulp (nu = 0.5, q = 1.5e-51, minor) |
| isf | low tail all sets; bulk nu = 0.5, 0.05 | 3.83 ulp (low tail); minor 45.2 ulp at `isf(0.15, 0.5)` (cond 0.368), 24 ulp at nu = 0.05 (no clear mechanism) |
| logpdf | all x for nu = 4.97, 1.5, 0.5, 0.05, 0.001 outside F3 | 36.2 ulp (nu = 4.97, cond 27.1); 10.3 ulp (nu = 1.5); median 0 |
| pdf | nu = 0.5 (`xlogy(0, x)` exact), all x | 7.07 ulp |

## Suspected, not confirmed

- Subnormal nu (< 5.6e-309): `1.0/nu` in `_ppf`/`_isf` overflows to inf.
  Not measurable: for such nu the cdf is ~1 for every normal x, so the true
  quantiles are not normal doubles. Not tested.
- nu >= ~1e7: by reports/gamma.md F1 the cdf error grows to 37 % at
  nu = 1e8 (loss beyond ~1e9). Not measured here (mpmath `gammainc` does
  not converge at a ~1e7; chi used a separate Kummer-series reference for
  df = 2e8, which maps to nu = 1e8 with identical cephes calls).
- x = 0 exactly (pdf = inf for nu < 0.5, sqrt(2/pi) for nu = 0.5, 0 for
  nu > 0.5) is not on the harness grids.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | `log(sf)` below the median, `log1p(-cdf)` above it | log-space upper incomplete gamma with a = nu, y = nu*x**2: `logsf = nu*log(y) - y - gammaln(nu) + log(continued fraction)`; asymptotically `(nu-1)*log(y) - y - gammaln(nu) + log1p(O((nu-1)/y))` | sf underflows once nu*x**2 >~ 745: `logsf(30, 4.967379486666624)` returns -inf, true -4440.4227279026017 (loss) |
| logcdf | `log(cdf)` below the median, `log1p(-sf)` above it | `logcdf = nu*log(y) - gammaln(nu + 1) + log(Kummer series 1F1(1; nu+1; y) * exp(-y))` (~ `nu*log(y) - gammaln(nu+1)` for y << nu), y = nu*x**2 formed as in F1 | cdf underflows in the left tail: `logcdf(1e-40, 4.967379486666624)` returns -inf, true -911.79487555110130 (loss); through F1 also for small nu |

## Shared code

- `sc.gammainc` / `sc.gammaincc` (cephes `igam`/`igamc`): F4 is
  reports/gamma.md F1 (unconverged 2000-term `igam_series` for a >~ 2e5,
  reports/INDEX.md cross-cutting gammainc item; issue #26471) reached through
  a = nu, y = nu*x**2; nakagami's nu = 1e6 values are identical to gamma's
  a = 1e6 and chi's df = 2e6 values. F9 is gamma F6 (exp-prefactor
  rounding, also maxwell F1, chi F9).
- `sc.gammaincinv` / `sc.gammainccinv` (cephes `igami`): F7 is gamma F3,
  F8 is gamma F5 (also chi F7/F8).
- nakagami-specific (but the same patterns as chi F1, F2, F3, F5, F6):
  the double-precision `nu*x*x` / `x**2` (F1, F2, F3) and the `_logpdf`
  cancellation (F5, F6, same as gamma F2/F4 and chi F5/F6).

## Existing GitHub issues

- #26471 (open): `gammainc` unconverged series for large a (filed 2026-10-21 from this review, see INDEX item 7) - relates to F4, F7
No nakagami-specific issue found (web search; `gh` CLI unavailable).

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e-220, 0.001)` | 0.0 | 0.3607865381521132 | verified | Kummer series `y**a e**-y/Gamma(a+1) 1F1(1; a+1; y)`; also `cdf(1e-200, 0.5)` = 7.9788456080286534e-201 (SciPy 0.0), `sf(1e-200, 0.001)` = 0.60440542745422652 (SciPy 1.0), `cdf(1e-160, 0.001)` 5.38e10 ulp; mechanism matches line 7755/7761 |
| F2 | `ppf(0.29, 0.001)` | 0.0 | 3.7483200837547026e-268 | verified | secant on own log cdf, cdf(x_ref)/q - 1 = 1.6e-61; also `ppf(1e-17, 0.05)`, `ppf(1e-300, 0.5)`, `isf(0.71, 0.001)` all 0.0 vs normal refs, `ppf(1e-16, 0.05)` 6.34e10 ulp; line 7758/7764 |
| F3 | `logpdf(4e155, 0.001)` | -inf | -1.6000000000000001e+308 | verified | logpdf via chi(2 nu) identity; also `logpdf(1.35e154, 0.5)` -inf vs -9.1125e+307; `x**2` at line 7752 |
| F4 | `cdf(0.9976973488989535, 1e6)` | 2.045146368700871e-06 | 2.0451657660636758e-06 | verified | Kummer series (own loop); 4.27e10 ulp; `cdf(0.9974968671630001, 1e6)` 1.95e10 ulp, `sf(0.9976973488989535, 1e6)` 8.74e4 ulp; shared cephes igam |
| F5 | `pdf(1.002297361066066, 1e6)` | 0.020900189008252586 | 0.020900189023301522 | verified | 3.24e6 ulp; `pdf(1.0, 1e6)` 2.89e6 ulp; `exp(_logpdf)` at line 7746 |
| F6 | `logpdf(0.9984988733093293, 1e6)` | 2.1744457981549203 | 2.1744457987925051 | verified | 1.32e6 ulp; `logpdf(1.0, 1e6)` 4.33e5 ulp; cancellation at lines 7751-7752 |
| F7 | `isf(0.9999993371296838, 1e6)` | 0.9975828813277674 | 0.99758288072503147 | verified | 2.72e6 ulp, cdf(x_ref)/(1-p) - 1 = 4.8e-55; `ppf(1e-8, 1e6)` 5.18e5 ulp (moderate, as reported) |
| F8 | `ppf(0.5000000000000001, 0.001)` | 7.241689311393424e-150 | 7.2416893113963007e-150 | verified | 1.79e3 ulp |
| F9 | `sf(1.3959524698393257, 1000)` | 7.180551658663522e-125 | 7.180551658658347e-125 | verified | Legendre continued fraction for Gamma(a, y) (mp.quad route was not stable at 100 digits); 3.25e3 ulp; `cdf(0.7250230003073993, 1000)` 2.93e3 ulp |

All verifier references agree between 100 and 200 digits to better than 1e-94 relative; scripts in `scratch/nakagami/verify/` (`verify.py`, `f9_cf.py`). `refs/nakagami_ref.py` was not needed.

Verified by dist-accuracy-verifier: 9 of 9 verified.
