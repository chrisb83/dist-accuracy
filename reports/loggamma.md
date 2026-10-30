# loggamma: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `loggamma_gen`, lines 6633-6747
- **Implemented:** pdf (6675-6677), logpdf (6679-6680), cdf (6682-6700), sf (6711-6716), ppf (6702-6709), isf (6718-6725)
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** `gamma` (loggamma is log of a gamma(c) variable; cdf/sf/ppf/isf are the gamma functions at z = exp(x)); `norm` only in `_entropy` (not measured); helpers: none; special functions: `sc.gammaln`, `sc.gammainc`, `sc.gammaincc`, `sc.gammaincinv`, `sc.gammainccinv` (`sc.digamma`, `sc.polygamma` only in `_stats`/`_entropy`)
- **Reference:** `refs/loggamma_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (0.4141193182605212), (1e-10), (1e-3), (1), (10.5), (1e3), (1e6); plus single points (`--at`) at c = 1e-14, 1e-6, 1e-300, 1e-310 (subnormal)
- **Worst finding:** loss in logpdf and cdf for subnormal c (and nan from ppf/isf); severe in cdf (4.4e10 ulp at c = 1e6), pdf/logpdf (8.4e6 / 4.3e6 ulp at c = 1e6) and sf (2.6e9 ulp at c = 1e-14, x < -708.4)

## Summary

For moderate shapes (1e-3 <= c <= 10.5) every implemented method is
accurate to within its conditioning (pdf/sf up to ~540 ulp only where cond is
~4e3). Four mechanisms fail. (1) For large c, `_pdf`/`_logpdf` add three terms
of size ~c*log(c) (`c*x`, `exp(x)`, `gammaln(c)`) to get an O(1) result:
4.3e6 ulp in logpdf and 8.4e6 ulp in pdf at c = 1e6, ~5e3 ulp at c = 1e3.
(2) `_cdf` calls `sc.gammainc(c, exp(x))`, which returns an unconverged
series for c >~ 2e5 (shared issue, gamma F1): cdf 4.4e10 ulp at c = 1e6, sf
9.4e4 ulp, and ppf/isf (via `gammaincinv`/`gammainccinv`) up to 3.9e5 ulp.
(3) The x < -708.4 branch of `_sf` uses `gammaln(c+1)`, where `c+1` rounds
away the low bits of a small c: 3.0e5 ulp at c = 1e-10, 2.6e9 ulp at
c = 1e-14. (4) For subnormal c (< 2.2e-308, in the declared domain),
`sc.gammaln(c)` is inf and `sc.gammainc(c, z)` returns 0: logpdf = -inf,
cdf = 0 instead of ~1, ppf/isf = nan.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf, cdf, ppf, isf | subnormal c (c < 2.2e-308), all x / q | `sc.gammaln(c)` = inf; `sc.gammainc(c, z)` = 0, `gammaincinv`/`gammainccinv` = nan (pattern 10, shared) | logpdf -inf for -714.8; cdf 0.0 for 1.0 | loss |
| F2 | cdf, sf | c >~ 2e5, x a few 1/sqrt(c) below log(c) | `sc.gammainc` returns an unconverged series (pattern 10, shared; gamma F1) | cdf 4.4e10 ulp, sf 9.4e4 ulp at c = 1e6 | severe |
| F3 | sf | small c (<~ 1e-6), x < _LOGXMIN = -708.4 | `gammaln(c+1)`: `c+1` rounds c, absolute error ~1e-17 in an exponent of size c*\|x\| (patterns 2, 10) | 2.6e9 ulp at c = 1e-14; 3.0e5 ulp at c = 1e-10 | severe |
| F4 | pdf, logpdf | large c (>~ 1e3), x near the mode log(c) | `c*x - exp(x) - gammaln(c)`: terms ~c*log(c) cancel to O(1) (pattern 5) | pdf 8.4e6 ulp, logpdf 4.3e6 ulp at c = 1e6 | severe |
| F5 | ppf, isf | c >~ 2e5, q in the left half of the distribution (isf: q -> 1) | `sc.gammaincinv`/`sc.gammainccinv` invert the inaccurate `gammainc` of F2 (shared; gamma F3) | isf 3.9e5 ulp, ppf 7.5e4 ulp at c = 1e6 | moderate |
| F6 | cdf | c tiny (1e-300) | `sc.gammainc(c, z)` returns values > 1 (shared) | 1.0000000000000127 for 1.0 (57 ulp) | minor |

### F1: subnormal c gives -inf / 0 / nan

- **Code:** `_logpdf`, line 6680: `c*x - np.exp(x) - sc.gammaln(c)`;
  `_pdf`, line 6677 (same exponent); `_cdf`, line 6700:
  `sc.gammainc(c, np.exp(x))`; `_ppf` line 6705 `sc.gammaincinv(c, q)`;
  `_isf` line 6721 `sc.gammainccinv(c, q)`.
- **Mechanism:** `_shape_info` declares c in (0, inf), so subnormal c is
  valid, but `sc.gammaln(c)` returns inf for c <~ 5.6e-309 (true value
  -log(c) ~ 709-744; INDEX cross-cutting issue from dgamma F5). logpdf
  becomes -inf. `sc.gammainc(1e-310, 0.37)` returns 0.0 (true 1 - 4e-311);
  this is consistent with igam's prefactor `exp(a*log(x) - x - lgam(a))`
  using the same infinite `lgam(a)`, but the inside of cephes was not traced.
  `sc.gammaincinv(1e-310, 0.5)` and `sc.gammainccinv(1e-310, 0.5)` return nan,
  so ppf and isf are nan. At c = 1e-308 (normal) all of these still work.
  (`sf` = `gammaincc` returns ~c*E1(z), subnormal, so it is not measurable;
  pdf is subnormal as well.)
- **Regime:** c < ~5.6e-309 (gammaln) / c subnormal (gammainc), every x or q.
- **Reproducer:** `python accuracy.py refs/loggamma_ref.py --at logpdf 0 1e-310`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.0, 1e-310)` | -inf | -714.8013788281542 | inf | 0 | loss |
  | `cdf(-1.0, 1e-310)` | 0.0 | 1.0 | inf | 6.92e-311 | loss |

  Not through the harness (scipy value vs closed-form limit, `scratch/loggamma/probe_lgam1p.py`):
  `ppf(0.99, 1e-310)` = nan, true log(0.99)/c = -1.005e308 (finite, normal);
  `isf(0.01, 1e-310)` = nan (same quantile). For q <~ 0.98 the true quantile
  is below -1.8e308, so -inf would be the correct double, but nan is returned.

- **Fix direction:** for c below ~1e-300 use the c -> 0 limits directly:
  `gammaln(c) = -log(c) - euler*c`, cdf = `exp(c*x - gammaln1p(c))` for all x
  where `exp(x)` is not large, and the existing log-form ppf/isf branches
  without calling the inverse functions; or fix `gammaln`/`igam` for
  subnormal a in `scipy.special`.

### F2: cdf and sf inherit the unconverged `gammainc` series for large c

- **Code:** `_cdf`, line 6700: `sc.gammainc(c, np.exp(x))`; `_sf`, line 6716:
  `sc.gammaincc(c, np.exp(x))`.
- **Mechanism:** identical to `gamma` F1 (and INDEX cross-cutting issue 7):
  for a >~ 2e5 and `4.5*sqrt(a) < a - z <~ 15*sqrt(a)` cephes `igam`
  returns the partial sum after 2000 series terms. With z = exp(x) this is
  x roughly between log(c) - 15/sqrt(c) and log(c) - 4.5/sqrt(c). `gammaincc`
  is `1 - igam` there, so sf carries the same absolute error (relatively
  smaller because sf ~ 1).
- **Regime:** c >~ 2e5, `log(c) - 15/sqrt(c) <~ x <~ log(c) - 4.5/sqrt(c)`.
- **Reproducer:** `python accuracy.py refs/loggamma_ref.py --at cdf 13.810910557964274 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(13.810910557964274, 1e6)` | 2.151900466483609e-06 | 2.1519213022256137e-06 | 4.4e+10 | 6.61e+04 | severe |
  | `cdf(13.809510557964273, 1e6)` | 1.0257710251325216e-09 | 1.0257716677332257e-09 | 2.8e+09 | 8.48e+04 | severe |
  | `cdf(13.805510557964274, 1e6)` | 9.04312377757342e-24 | 9.04312377965407e-24 | 1.0e+06 | 1.4e+05 | severe |
  | `sf(13.810910557964274, 1e6)` | 0.9999978480995335 | 0.9999978480786977 | 9.38e+04 | 0.142 | moderate |

- **Fix direction:** in `scipy.special` (raise the iteration cap or use the
  uniform asymptotic expansion over a wider window); nothing loggamma-specific.

### F3: sf for x < -708.4 loses the low bits of small c in `gammaln(c+1)`

- **Code:** `_sf`, line 6715: `-np.expm1(c*x - sc.gammaln(c+1))`.
- **Mechanism:** for small c the result is about `-(c*x) + gammaln(1+c)`
  ~ c*(|x| - 0.577), so it needs c*(...) to full relative accuracy. `c+1`
  is rounded to a multiple of 2.2e-16 (for c = 1e-10, `1 + c - 1 - c` =
  8.3e-18), so `sc.gammaln(c+1)` = -5.772157126783244e-11 instead of
  -5.7721566481928618e-11 (relative error 8e-8), an absolute error of 4.8e-18
  in an exponent of size 7e-8 (pattern 2: the information is lost in the
  addition `1 + c`). Verified: replacing `sc.gammaln(c+1)` by
  `scipy.special._ufuncs._lgam1p(c)` gives 7.078191778170444e-08, the
  reference to 1 ulp (`scratch/loggamma/probe_lgam1p.py`). The error is an absolute
  error ~|c + 1 - 1 - c|*0.577, so the relative error grows like 1/(c*|x|):
  2.6e9 ulp at c = 1e-14, 3.0e5 at c = 1e-10, 148 ulp at c = 1e-6 (x = -800).
  The `gammainc` branch (x > -708.4) is unaffected, so the error jumps at
  the threshold. The same expression in `_cdf` (line 6699) and `_ppf`
  (line 6708) has the same absolute error, but there it is negligible
  relative to the result (cdf ~ 1; ppf ~ log(q)/c).
- **Regime:** x < _LOGXMIN = -708.396, c <~ 1e-6 (beyond 1e3 ulp for c <~ 1e-7).
- **Reproducer:** `python accuracy.py refs/loggamma_ref.py --at sf -800 1e-14`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(-800, 1e-14)` | 7.994232456875913e-12 | 7.994227843319031e-12 | 2.6e+09 | 1 | severe |
  | `sf(-708.3964185322642, 1e-10)` | 7.078191777691853e-08 | 7.078191778170443e-08 | 3.0e+05 | 1 | moderate |
  | `sf(-30023.025850929935, 1e-10)` | 3.0022403567891256e-06 | 3.0022403567939114e-06 | 7.18e+03 | 1 | moderate |
  | `sf(-800, 1e-6)` | 0.00079910333189462 | 0.0007991033318945936 | 148 | 1 | minor |

- **Fix direction:** `-expm1(c*x - _lgam1p(c))` (cephes `lgam1p`, already in
  `scipy.special._ufuncs`), also in `_cdf` and `_ppf`/`_isf`.

### F4: pdf and logpdf cancel terms of size c*log(c) for large c

- **Code:** `_logpdf`, line 6680: `c*x - np.exp(x) - sc.gammaln(c)`;
  `_pdf`, line 6677: `np.exp(c*x - np.exp(x) - sc.gammaln(c))`.
- **Mechanism:** near the mode x ~ log(c), `c*x` ~ c*log(c) (1.38e7 at
  c = 1e6), `exp(x)` ~ c and `gammaln(c)` ~ c*log(c) - c; each carries a
  rounding error of ~c*log(c)*eps absolute, but the sum is O(log sqrt(c))
  (pattern 5). logpdf loses ~c*log(c)/|logpdf| ulp, pdf (= exp of it)
  ~c*log(c) ulp. The conditioning w.r.t. x is |x*(c - exp(x))|, which is
  0 at the mode and ~1e4-4e5 within a few 1/sqrt(c), so part of the error
  is excused there but not the bulk of it.
- **Regime:** c >~ 1e3, x within ~10/sqrt(c) of log(c) (and further out for
  pdf).
- **Reproducer:** `python accuracy.py refs/loggamma_ref.py --at pdf 13.845748741024885 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(13.845748741024885, 1e6)` | 1.0873305841015277e-198 | 1.0873305820765016e-198 | 8.4e+06 | 4.3e+05 | severe |
  | `pdf(13.815210557964274, 1e6)` | 381.3895004625327 | 381.3894997984101 | 7.8e+06 | 4.14e+03 | severe |
  | `logpdf(13.818510557964274, 1e6)` | 1.4843132868409157 | 1.4843132854192003 | 4.3e+06 | 2.8e+04 | severe |
  | `logpdf(13.815510557964274, 1e6)` | 5.988816661760211 | 5.988816662444131 | 5.1e+05 | 1.09e-09 | moderate |
  | `pdf(7.0, 1000)` | 0.15667500708680898 | 0.15667500708665902 | 4.31e+03 | 676 | moderate |
  | `logpdf(7.0, 1000)` | -1.8535816376388539 | -1.853581637639811 | 2.33e+03 | 365 | moderate |

  (cond (shapes) up to 3e4: the result is sensitive to c, but c is exact.)
- **Fix direction:** with t = x - log(c): `logpdf = c*(t - expm1(t)) +
  (c*log(c) - c - gammaln(c))`, where the bracket is
  `0.5*log(c/(2*pi)) - stirling_correction(c)` (no large terms); t itself
  is exact-ish only if log(c) is subtracted carefully, so a further
  refinement may be needed for x very close to log(c).

### F5: ppf and isf for large c inherit F2 through the inverse functions

- **Code:** `_ppf`, line 6705 `sc.gammaincinv(c, q)` then `np.log(g)`;
  `_isf`, line 6721 `sc.gammainccinv(c, q)`.
- **Mechanism:** the inverse functions solve `gammainc(c, g) = q` with the
  inaccurate `gammainc` of F2, so g is the quantile of the wrong cdf (gamma
  F3). `log(g)` divides the relative error of g by |log g| ~ 13.8, which is
  why the error is "only" moderate.
- **Regime:** c >~ 2e5; ppf for q from ~1e-14 to ~1e-6; isf for 1 - q in
  the same range.
- **Reproducer:** `python accuracy.py refs/loggamma_ref.py --at isf 0.9999993371296838 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.9999993371296838, 1e6)` | 13.810670468725416 | 13.810670467517022 | 3.9e+05 | 21.8 | moderate |
  | `isf(0.999999938945977, 1e6)` | 13.810215243919647 | 13.810215243462832 | 1.5e+05 | 217 | moderate |
  | `ppf(1e-08, 1e6)` | 13.809892968829743 | 13.809892968599815 | 7.5e+04 | 1.26e-05 | moderate |
  | `ppf(3.4807005884285624e-14, 1e6)` | 13.808012268935453 | 13.808012268931146 | 1.4e+03 | 9.53e-06 | moderate |

- **Fix direction:** fix `sc.gammainc` (F2); no loggamma-specific change.

### F6: `gammainc` returns values above 1 for tiny c

- **Code:** `_cdf`, line 6700: `sc.gammainc(c, np.exp(x))`.
- **Mechanism:** `sc.gammainc(1e-300, 0.37)` = 1.000000000000019 (true
  1 - 7.6e-301), so the cdf exceeds 1. At c = 1e-10 the cdf near 1 is
  11-18 ulp off over x in [-707, 0] (68 minor points). The error is inside
  `scipy.special.gammainc` for tiny a (mechanism inside cephes not traced).
- **Regime:** c <~ 1e-10, x > -708.4, cdf close to 1.
- **Reproducer:** `python accuracy.py refs/loggamma_ref.py --at cdf -1 1e-300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-1.0, 1e-300)` | 1.0000000000000127 | 1.0 | 57 | 6.92e-301 | minor |
  | `cdf(-1.0, 1e-10)` | 0.9999999999240624 | 0.9999999999240584 | 18 | 6.92e-11 | minor |

- **Fix direction:** for small c compute cdf as `1 - sf` with
  `sf = gammaincc(c, z)` (accurate, ~c*E1(z)), or fix igam for tiny a.

## RuntimeWarnings

All methods emit `overflow encountered in exp` for large x (`np.exp(x)` =
inf; pdf/sf return 0 there, correct since the true values are far below the
double range) and `underflow encountered in exp`/`multiply`/`expm1` in the
far tails where results are subnormal or 0. On the measured points these
warnings accompany correct results (spurious), except in F1 where
`gammaln` overflows silently.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, c in {1e-10, 1e-3, 0.414, 1, 10.5} | 554 ulp (within cond ~4.6e3); 79 minor points at c = 1e-10, max 19 ulp (exponent contains gammaln(c) ~ 23) |
| logpdf | full grid, c <= 10.5 | 44 ulp at c = 10.5 (cond 28; minor) |
| cdf | full grid incl. both sides of x = -708.396, c <= 1e3 | 1.6e3 ulp at c = 1e3 (within cond 3.6e3; gamma F6 prefactor) |
| sf | full grid, c in {1e-3, 0.414, 1, 10.5, 1e3}; right tail to sf ~ 1e-300 | 2.1e3 ulp at c = 1e3 (within cond 1.3e4); 63 minor points at c = 1e-3, max 12 ulp |
| ppf | q in [1e-300, 1 - 1e-16], c <= 1e3, both sides of the g < _XMIN branch (c = 1e-10, 1e-3) | 23 ulp (c = 0.414); c = 1e-10: 3.3e4 ulp within cond 1.5e6 |
| isf | same | 54 ulp (c = 1, q = 0.36, cond 46; minor, no clear mechanism) |

## Suspected, not confirmed

- The `x < _LOGXMIN` one-term-series branch in `_cdf`/`_sf` drops the
  relative correction ~c*exp(x)/(c+1) < 1e-307: harmless, checked at both
  sides of -708.396.
- Subnormal-c run over the full grid (`--params "1e-300;1e-310"`) was still
  computing ppf/isf references when this report was written; the F1/F6 points
  above were measured with `--at`.
- `_rvs`, `_stats`, `_entropy` are not measured.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | for x < _LOGXMIN: `c*x - gammaln1p(c)` (exact to O(exp(x))); otherwise `log(gammainc(c, z))`, falling back to `c*x - z - gammaln(c+1) + log(series)` (log of igam's prefactor times its series) where gammainc underflows | cdf underflows for c*x <~ -745; `logcdf(-2000, 0.414)` = -inf, true -827.880 |
| logsf | log(sf) | `log1p(-cdf)` for cdf < 1/2; for large z: `(c-1)*x - z - gammaln(c) + log(continued fraction of Q)` (log of igamc's prefactor) | sf underflows for exp(x) >~ 700 (x >~ 6.6 for c ~ 1); `logsf(7, 0.414)` = -inf, true -1101.497 |

## Shared code

- `sc.gammainc`/`sc.gammaincc` unconverged series for a >~ 2e5 (INDEX
  issue 7, gamma F1) -> loggamma F2, and through `gammaincinv`/`gammainccinv`
  F5.
- `sc.gammaln` returns inf for subnormal a (INDEX, dgamma F5) -> F1; new
  here: `sc.gammainc(a, z)` returns 0 and `sc.gammaincinv`/`sc.gammainccinv`
  return nan for subnormal a, and `sc.gammainc(1e-300, 0.37)` returns a value
  > 1 (F6).
- F3 (`gammaln(c+1)` instead of `lgam1p(c)`) is loggamma code, but the same
  idiom may occur in other distributions' small-shape branches.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `logpdf(0, 1e-310)`; `cdf(-1, 1e-310)`; `ppf(0.99, 1e-310)`, `isf(0.01, 1e-310)` | -inf; 0.0; nan; nan | -714.80137882815416; 1.0 (1 - upper `mp.gammainc`); -1.0050335853501441e308 (checked: series cdf(x_ref) - 0.99 = 0) | verified (loss) | `sc.gammaln(1e-310)` = inf, `sc.gammainc(1e-310, e^-1)` = 0, `gammaincinv`/`gammainccinv(1e-310, 0.5)` = nan confirmed; lines 6680/6700/6705/6721 match |
| F2 | `cdf(13.810910557964274, 1e6)`; `sf(...)` same x | 2.151900466483609e-06; 0.9999978480995335 | 2.1519213022256136e-06; 0.99999784807869777 (`mp.quad` of the docstring pdf) | verified (severe, 4.4e10 ulp; sf 9.4e4) | also verified x = 13.809510557964273 (1.0257716677332258e-09) and 13.805510557964274 (9.0431237796540698e-24); lines 6700/6716 match; cephes iteration cap not re-traced |
| F3 | `sf(-800, 1e-14)`; `sf(-708.3964185322642, 1e-10)` | 7.994232456875913e-12; 7.078191777691853e-08 | 7.9942278433190309e-12 (upper `mp.gammainc` and 1 - series agree); 7.0781917781704430e-08 | verified (severe, 2.6e9 ulp) | `sc.gammaln(1+1e-14)` = -5.7675e-15 vs true -5.7722e-15 confirms the `c+1` rounding; line 6715 matches |
| F4 | `pdf(13.845748741024885, 1e6)`; `logpdf(13.818510557964274, 1e6)` | 1.0873305841015277e-198; 1.4843132868409157 | 1.0873305820765017e-198; 1.4843132854192003 (`mp.gamma` / `mp.loggamma`) | verified (severe, 8.4e6 / 4.3e6 ulp) | also `pdf(13.815210557964274, 1e6)` ref 381.38949979841010; error size ~ eps*c*log(c) consistent with cancellation at lines 6677/6680 |
| F5 | `isf(0.9999993371296838, 1e6)`; `ppf(1e-08, 1e6)` | 13.810670468725416; 13.809892968829743 | 13.810670467517025; 13.809892968599816 (root of `mp.quad` cdf, residual < 1e-196) | verified (moderate, 3.9e5 / 7.5e4 ulp) | lines 6705/6721 match |

References in `scratch/loggamma/verify/verify.py` (100 and 200 digits agree to 30 digits for every value; x arguments taken as exact doubles). SciPy calls in `scratch/loggamma/verify/scipy_calls.py`.

Verified by dist-accuracy-verifier: 5 of 5 verified.
