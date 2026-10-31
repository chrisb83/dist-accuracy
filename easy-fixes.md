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

## 2. Tail built by subtraction - use `erf`, `-expm1`, `arctan2`

| Distribution | Today | Fix | Effect today | Report |
|---|---|---|---|---|
| `levy_l` cdf | `2*_norm_cdf(1/sqrt(abs(x))) - 1` | `erf(1/sqrt(2*abs(x)))` | 3.3e9 ulp at x = -1e20; 0.0 for x <= -1e32 | levy_l.md F1 |
| `skewcauchy` cdf | `(1-a)/2 + (1-a)/pi*arctan(x/(1-a))` | `arctan2` form as in `cauchy` | loses ~\|x\|/(1-a) ulp in the left tail; all of x < 0 is 0.0 for a near 1 | skewcauchy.md F1 |
| `laplace_asymmetric` cdf/sf | `1 - exp(...)` | `-expm1(...)` | 5e14 ulp at kappa = 1e-8; `cdf(1e-300, 1e-150)` = 0 | laplace_asymmetric.md F1, F2 |

## 3. Add a missing `_isf` where the inherited `ppf(1 - q)` / generic one is poor

| Distribution | Fix | Effect today | Report |
|---|---|---|---|
| `recipinvgauss` | `_isf = 1/invgauss.ppf(q, mu)` | `isf(1e-20, 1)` = 100.0 vs 89.13 | recipinvgauss.md, Opportunities |
| `powerlaw` | `_isf = exp(log1p(-q)/a)` | `isf(1e-13, 1e-12)` 1.4e11 ulp | powerlaw.md, Opportunities |
| `moyal` | `_isf = -2*log(erfinv(q)) - log(2)` | inf for q < 1.1e-16 | moyal.md, Opportunities |

## 4. Large but ordinary shapes

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
  `norm`, `alpha` F1, `t` F3 (df = inf branch), `skewnorm` F3,
  `powerlognorm` F4, `powernorm` F5. The same overflow sits in other classes'
  own code: `halfnorm` F3 (`x*x`), `chi` F3, `t` F2 (`x * x/df`), `invgauss` F6.

## 6. Second step: closed-form `_logsf` / `_logcdf`

Many classes inherit `log(sf)` / `log(cdf)` and return -inf at moderate x,
e.g. `halfnorm.logsf(40)` (true -803.92, fix `log(2) + log_ndtr(-x)`),
`rayleigh.logcdf(1e-200)` (true -921.7, `log(-expm1(-r*r/2))`),
`maxwell.logsf(40)`, `moyal`, `gumbel_r`/`gumbel_l`, `laplace`, `levy`,
`gibrat` (`log_ndtr(+-log x)`). These are opportunities rather than findings
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

## Suggested first PR

`halfnorm` / `levy_l` erfinv ppf + `levy_l` erf cdf + `pearson3`
`gammainccinv` (+ `_norm_logpdf` if you want a free extra): all one-liners,
with reproducers and reference values in the reports.
