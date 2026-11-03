# Easy fixes - where to start

SciPy 1.18.1. One- or two-line changes that fix verified findings at ordinary
inputs (q down to ~1e-8...1e-20, everyday shape values, the bulk), plus a few
trivial ones that are low priority but cost almost nothing. Each entry points
to the report with reproducers and reference values that can become tests.

Deliberately left out: fixes that need real effort for extreme regimes
(subnormal inputs or shapes, s ~ 1e-150, ...), see the end.

## 1. `1 +- q` rounding in ppf/isf - use `log1p`, `expm1`, `erfinv`

| Distribution | Today | Fix | Effect today | Report |
|---|---|---|---|---|
| `halfnorm` ppf | `ndtri((1+q)/2)` | `sqrt(2)*erfinv(q)` (and `-ndtri((1-q)/2)` for q > 0.5) | `ppf(q)` = 0.0 for every q <= 1e-16; inf at q = 1 - 2**-53 | halfnorm.md F1, F2 |
| `levy_l` ppf | `_norm_ppf((q+1)/2)` | `-0.5/erfinv(q)**2` | 5.5e7 ulp at q = 1e-8, -inf for q <= ~1.1e-16 | levy_l.md F2 |
| `pearson3` ppf, skew < 0 | `gammaincinv(alpha, 1-q)` | `gammainccinv(alpha, q)` | `ppf(1e-20, -2)` = -inf vs -45.05 (skew = -2 is the docstring example); up to 2.3e11 ulp for small q | pearson3.md F1 |
| `powernorm`, `powerlognorm` ppf | `pow(1.0 - q, 1.0/c)` | `exp(log1p(-q)/c)` | -inf / 0.0 for q < 1.1e-16; same expression in both classes | powernorm.md F1, powerlognorm.md F1 |
| `lomax` isf | `q**(-1/c) - 1` | `expm1(-log(q)/c)` (ppf already uses `expm1`) | 0.0 for q near 1; 4.8e4-1.2e8 ulp across the range at c = 1e6 | lomax.md F1 |
| `laplace_asymmetric` ppf/isf | `log(...)` near the branch point | `log1p(...)` | 3.1e7 ulp at kappa = 0.5, q = 0.200000008 (bulk) | laplace_asymmetric.md F3, F4 |
| `skewcauchy` ppf | `q - (1-a)/2` | rearrange so small q is not absorbed | a = 0.5: every q < ~1e-16 gives -8.2e15 (true -8e298 at q = 1e-300) | skewcauchy.md F2 |
| `burr` ppf (also fixes `fisk`) | `(q**(-1/d) - 1)**(-1/c)` | `expm1(-log(q)/d)**(-1/c)` | inf for q > ~1 - d*eps (`ppf(0.9999999999999997, 2, 3)` vs 9.49e7); 1.2e12 ulp at d = 1e6; 1.6e4 ulp at q = 0.5, d = 1e6 | burr.md F4 |
| `rdist` ppf (also fixes `semicircular` ppf) | `2*_beta_ppf(q, c/2, c/2) - 1` near q = 1/2 | `sign(q - 1/2)*sqrt(betaincinv(0.5, c/2, abs(2*q - 1)))` (X**2 ~ Beta(1/2, c/2)); keep the current form in the tails | 0.0 or the wrong sign next to q = 1/2 at every c; 3.95e3 ulp at q = 0.6, c = 1e6; the fix measured 0.004-3.1 ulp | rdist.md F1, semicircular.md F2 |
| `crystalball` ppf, q near 1 | `_norm_ppf` of a value near 1 | `_norm_isf((1 - p)/(N*_norm_pdf_C))` | severe near q = 1 | crystalball.md F7 |
| `triang` ppf, q >= c | `1 - sqrt((1-c)*(1-q))` | `(c + q - c*q) / (1 + sqrt((1-c)*(1-q)))` | small q with small c, including c = 0: 0.0 for true 5e-21; 7.0e12 ulp | triang.md F1 |

## 2. Tail built by subtraction - use `erf`, `-expm1`, `arctan2`

| Distribution | Today | Fix | Effect today | Report |
|---|---|---|---|---|
| `levy_l` cdf | `2*_norm_cdf(1/sqrt(abs(x))) - 1` | `erf(1/sqrt(2*abs(x)))` | 3.3e9 ulp at x = -1e20; 0.0 for x <= -1e32 | levy_l.md F1 |
| `skewcauchy` cdf | `(1-a)/2 + (1-a)/pi*arctan(x/(1-a))` | `arctan2` form as in `cauchy` | loses ~\|x\|/(1-a) ulp in the left tail; all of x < 0 is 0.0 for a near 1 | skewcauchy.md F1 |
| `laplace_asymmetric` cdf/sf | `1 - exp(...)` | `-expm1(...)` | 5e14 ulp at kappa = 1e-8; `cdf(1e-300, 1e-150)` = 0 | laplace_asymmetric.md F1, F2 |
| `burr` sf, logsf (also fixes `fisk`) | `log1p(-cdf)` | logcdf = `-d*log1p(x**-c)`, `sf = -expm1(logcdf)`, `logsf = log1mexp(logcdf)` | `sf(1e8, 2, 3)` = 0.0 vs 3e-16; `sf(10, 10.5, 4.3)` 8.8e9 ulp | burr.md F1 |
| `semicircular` cdf, x < 0 | `0.5 + (x*sqrt(1-x*x) + arcsin(x))/pi` | `sc.betainc(1.5, 1.5, (1+x)/2)` (= `rdist._cdf(x, 3)`) | `cdf(-0.9999)` 3.8e6 ulp; -1.1e-16 (negative) at 1 + x = 1e-12 | semicircular.md F1 |
| `wrapcauchy` pdf | `(1-c*c) / (... (1 + c*c - 2*c*cos(x)))` | `(1-c)*(1+c)` and `(1-c)**2 + 4*c*sin(x/2)**2` | `pdf(0, 0.999999)` 6.0e11 ulp; inf near x = 0 for c = 1 - 1e-10; 5e4 ulp everywhere at c = 1 - 1e-6 | wrapcauchy.md F1, F2 |
| `triang` cdf, x >= c | `(x*x - 2*x + c)/(c-1)` | `1 - (1-x)**2/(1-c)` | 1.35e7 ulp for c near 1 | triang.md F4 |
| `truncexpon` sf | difference of `exp(-b)` and `exp(-x)` | `exp(-x) * -expm1(x - b) / -expm1(-b)` | small b: error grows below b ~ 1e-3; `sf(5e-301, 1e-300)` = -0.0 vs 0.5 | truncexpon.md F1 |
| `truncexpon` logpdf | `log(-expm1(-b))` | `sc._ufuncs._log1mexp(-b)` | -0.0 for b >~ 37 (`logpdf(0, 50)`, true 1.9e-22) | truncexpon.md F3 |
| `truncweibull_min` normaliser, cdf/sf numerators | `exp(-a**c) - exp(-b**c)` (same for x) | `exp(-a**c) * -expm1(a**c - b**c)` | 1.0e11 ulp at a = 0, b**c = 1e-12; inf/nan once b**c < ~5.5e-17 | truncweibull_min.md F2-F4 |
| `truncpareto` normaliser, ppf/isf | `1 - c**-b` | `-expm1(-b*log(c))`; ppf `exp(-log1p(q*expm1(-b*log(c)))/b)` | up to 5.4e7 ulp (pdf) and 4.1e7 ulp (ppf) for tiny \|b\| or c near 1 | truncpareto.md F4, F5 |
| `crystalball` pdf/cdf/sf, x <= -beta | `(m/beta)**m` and `(B - x)**(-m)` formed separately | `exp(-beta**2/2 - m*log1p((-beta - x)*beta/m))` (with `1 - m` for the cdf) | `pdf(-2, 1, 150)` = nan (true 0.082); nan/inf once m >~ 144 at beta = 1 | crystalball.md F1-F3 |
| `crystalball` sf, x <= -beta | `1 - _cdf` | closed form with `-expm1((1-m)*log1p((-beta - x)*beta/m))` | 3.9e9 ulp at beta = 1e-10 | crystalball.md F4 |
| `halfgennorm` logpdf | `log(beta) - gammaln(1/beta)` | `-_lgam1p(1/beta) - x**beta` (also `gennorm` F3) | 4.7e6 ulp at beta = 1e6; 0.0 at beta = 1e300 | halfgennorm.md F3 |
| `loguniform` (and `reciprocal`) | `log(b) - log(a)`, `log(x) - log(a)` | `log1p((b - a)/a)`, `log1p((x - a)/a)` | cdf 7.7e10 ulp near x = a; pdf inf when b = nextafter(a) | loguniform.md F1, F2 |

## 3. Add a missing `_isf` where the inherited `ppf(1 - q)` / generic one is poor

| Distribution | Fix | Effect today | Report |
|---|---|---|---|
| `recipinvgauss` | `_isf = 1/invgauss.ppf(q, mu)` | `isf(1e-20, 1)` = 100.0 vs 89.13 | recipinvgauss.md, Opportunities |
| `powerlaw` | `_isf = exp(log1p(-q)/a)` | `isf(1e-13, 1e-12)` 1.4e11 ulp | powerlaw.md, Opportunities |
| `moyal` | `_isf = -2*log(erfinv(q)) - log(2)` | inf for q < 1.1e-16 | moyal.md, Opportunities |
| `rdist`, `semicircular` (separate class, same fix) | `_isf = -_ppf(q)` (symmetry) | `rdist.isf(1e-20, 1e6)` = 1.0 vs 0.00926 (4.8e17 ulp; -ppf gives 39.7 ulp); `semicircular.isf(1e-20)` = 1.0, while `-ppf(1e-20)` = 0.9999999999999347 | rdist.md, Opportunities |
| `crystalball` | q < 1 - pbeta: `_isf = _norm_isf(q/(N*_norm_pdf_C))`; else the power-law closed form with p = 1 - q | `isf(1e-20, 2, 3)` = inf vs 9.26; digits lost for all q < ~1e-3 | crystalball.md, Opportunities |

## 4. Slightly more work, but ordinary inputs

- `tukeylambda` ppf: `boxcox(q) - boxcox1p(-q)` cancels near q = 1/2.
  `ppf(0.49, 50)` is 3.2e13 ulp off, and `ppf(0.5000000001, 50)` = 0.0. This
  happens at every lam. Fix: `(q**lam - (1-q)**lam)/lam`, with a series in
  2q - 1 near 1/2. tukeylambda.md F5.
- `argus` cdf = `1 - sf` cancels for small x: `cdf(1e-5, 1)` is 1.1e10 ulp
  off, and `cdf(0.95, 100)` = 0.0. Fix: compute the numerator directly
  (difference of `gammaincc(1.5, ...)` for large chi, and a series in x**2
  for small x). argus.md F1.
- `kappa3`: `x**a` (cdf), `q**-a` (ppf) and `expm1(-a*log1p(-q))` (isf)
  overflow in the bulk: `cdf(3, 1000)` = 0.0 vs 1.0, `ppf(0.49, 1000)` =
  `isf(0.51, 1000)` = 0.0 vs 0.4934. Log-space forms (e.g.
  `exp(-log1p(a*x**-a)/a)`) fix it. Slightly more work than the rest, but
  a = 1000 is realistic. kappa3.md F1-F3 (F4: `q**-a - 1` cancels, fix with
  `expm1(-a*log q)`).

## 5. Trivial, low priority: `_norm_logpdf`

- `_norm_logpdf` (`_continuous_distns.py` line 367) computes `-x**2 / 2.0`;
  `x**2` overflows for 1.34e154 < |x| < 1.9e154 although the result is
  finite, so logpdf is -inf there. `-(x/2)*x` (or `-0.5*x*x` evaluated as
  `(-0.5*x)*x`) fixes it. One line; the helper is called 8 times and fixes
  `norm` F1, `alpha` F1, `t` F3 (df = inf branch), `skewnorm` F3,
  `powerlognorm` F4, `powernorm` F5, `truncnorm` (logpdf). The same overflow
  sits in other classes' own code: `halfnorm` F3 (`x*x`), `chi` F3, `t` F2
  (`x * x/df`), `invgauss` F6 (and `wald` F2), `halfcauchy` F1 (`log1p(x*x)`;
  fix: `log(2/pi) - 2*log(x) - log1p((1/x)**2)` for x > 1, as in `cauchy`),
  `argus` F4 (`chi**2`, fix `-chi*(chi*y/2)`), `crystalball` F6 (own copy,
  fix `-(0.5*x)*x`).
- Same kind of trivia at the other end: `x**2 / c` and `sqrt(q * c)`
  underflow for c < ~1.5e-154 in `trapezoid` F1/F2 and `triang` F2/F3. Fix:
  `x * (x / c)` and `sqrt(q) * sqrt(c)`.

## 6. Second step: closed-form `_logsf` / `_logcdf`

Many classes inherit `log(sf)` / `log(cdf)` and return -inf at moderate x,
e.g. `halfnorm.logsf(40)` (true -803.92, fix `log(2) + log_ndtr(-x)`),
`rayleigh.logcdf(1e-200)` (true -921.7, `log(-expm1(-r*r/2))`),
`maxwell.logsf(40)`, `moyal`, `gumbel_r`/`gumbel_l`, `laplace`, `levy`,
`gibrat` (`log_ndtr(+-log x)`), `crystalball.logsf(40, 2, 3)` (true -804.63,
`log(N*_norm_pdf_C) + _norm_logsf(x)` for x > -beta), `halfgennorm.logsf(30, 2)`
(true -903.97). These are opportunities rather than findings
(the review only scores methods a class implements), but each report gives
the closed form under "Opportunities", so they are easy to add.

## Left for later

- Extreme regimes that need real effort: subnormal x or shapes (`lognorm`
  F8 / `gibrat` F2, `loggamma` F1, ...), s ~ 1e-150 and similar.
- Upstream/Boost: the noncentral families `nct`, `ncf`, `ncx2`, `rice`
  (INDEX item 17), `t`'s `stdtr`/`stdtrit` (item 8).
- `sc.gammainc` at large a - filed as #26471 (INDEX item 7).
- `skewnorm` F1: the quad fallback in `_cdf` needs a design decision.
- `studentized_range`: needs a rework of the nquad integration.
- `tukeylambda` cdf/pdf: `sc.tklmbda` bisects to an absolute tolerance and
  substitutes the logistic cdf for 0 < |lam| < 1e-4 (#21370, INDEX item 21).
- `vonmises`/`vonmises_line` cdf: the series cancels in the left tail, and
  an uncorrected normal approximation is used for kappa >= 50 (relative
  error up to 0.91 in the tail; INDEX item 27).
- `truncnorm`: short intervals and endpoints near 0 (`_log_gauss_mass`,
  `_log_diff(p, p)`; INDEX items 19, 20).
- `gennorm` and `halfgennorm` for beta >~ 22: `ppf(0.6, 1000)` = 0.0,
  `cdf(0.4, 1000)` = 0.5 (gennorm), `cdf(0.4, 1000)` = 0 (halfgennorm). This
  needs log-space forms for small 1/beta (gennorm.md F1, F2;
  halfgennorm.md F1, F2).
- `rdist` at x = 1 - 2**-53: `(x+1)/2` rounds to 1.0. Fix by computing with
  1 - x, or with X**2 ~ Beta(1/2, c/2) (rdist.md F2). For large c,
  `betaln(c/2, c/2)` cancels in the logpdf (F3, same family as beta F4).
- `burr` left tail and small d (F2, F5, F6): log-space rewrites.
- Upstream/Boost as well: `wald` isf (= invgauss, item 14).
  `genhyperbolic`'s quad-based cdf turns nan into 1.0 (item 25). (The
  `semicircular` ppf is now in section 1, through the `rdist` fix.)

## Suggested first PR

`halfnorm` / `levy_l` erfinv ppf + `levy_l` erf cdf + `pearson3`
`gammainccinv` (+ `_norm_logpdf` if you want a free extra): all one-liners,
with reproducers and reference values in the reports.

A second PR in the same spirit: `burr` ppf `expm1` + sf/logsf via
`-expm1`/`log1mexp` (this fixes `fisk` too), `wrapcauchy` pdf, and the
`triang` ppf/cdf rewrites.
