# Potential issues in `scipy.special` (cephes, Boost, AMOS) - overview

SciPy 1.18.1, NumPy 2.5.4, Windows. Collected from the distribution reviews
in `reports/` (cross-cutting items in `reports/INDEX.md`). "Verified" means
the finding that exposed it was confirmed by an independent verifier; Boost
internals were not inspected (source not shipped), so Boost mechanisms are
inferred from outputs.

Left out: the `gammainc` unconverged series at large a (#26471, INDEX
item 7). The other incomplete-gamma problems below are separate defects.

Practical relevance: **high** = wrong in ordinary use (bulk, moderate tails,
ordinary parameters); **medium** = tails or large-but-plausible parameters;
**low** = extreme or subnormal regimes only.

## Cephes

| Function | Problem | Example | Relevance | Seen in | Status |
|---|---|---|---|---|---|
| `ndtr` (and `0.5*erfc`) | Not correctly rounded in the far left tail: up to 52 / 245 / 704 / 1026 ulp for \|z\| in [5,10) / [10,20) / [20,30) / [30,37.5), median ~0.3 z**2 ulp (one ulp of the internal exponent z**2/2). Within ndtr's own conditioning, but amplified wherever a tail is subtracted. | `ndtr(-30)` 256 ulp off at an exact argument | medium | exponnorm F2 (6.3e7 ulp after cancellation), lognorm F6, gibrat F3, powernorm, powerlognorm, skewnorm, recipinvgauss, t | verified; no issue found. Fix direction: `0.5*erfcx(-x/sqrt 2)*exp(-x**2/2)` with x**2/2 split exactly |
| `log_ndtr(z)`, z > 0 | Has exactly the error of `ndtr(-z)` (z < 0 is accurate, <= 1.8 ulp) | - | medium | lognorm F6 (logcdf right / logsf left tail) | verified |
| `igam_fac` (prefactor of `gammainc`/`gammaincc`) | For \|a - x\| > 0.4a computes `exp(a*log(x) - x - lgam(a))`; the cancelling exponent costs ~a*log(x)*eps | `gamma.sf(2000, 1000)` 2.0e3 ulp; `cdf(1e-280, 1)` 241 ulp | medium (minor-moderate size) | gamma F6, chi F9, chi2 F6, nakagami F9, maxwell F1 (a = 1.5, 200 ulp), dgamma F7 | verified. Fix: Lanczos prefactor branch for all a > ~20 |
| `igami` (`gammaincinv`/`gammainccinv`), a << 1 | The final Newton step on P amplifies the rounding of a large exponent; ~3.6x the conditioning | `gamma.ppf(0.5, 1e-3)` 3.6e3 ulp | medium | gamma F5, chi F8, chi2 F5, dgamma F6, nakagami F8, maxwell F2 | verified. Fix: last Newton step on log P for a < 1 |
| `gammaincc(0.5, x)` | 42-118 ulp off for x in [0.5, 1] | - | low (minor) | gengamma; would reach chi2 with df = 1 | measured, minor |
| `gammainc`/`gammaincinv`/`gammainccinv`, subnormal a | `gammainc` returns 0, the inverses nan; `gammainc(1e-300, 0.37)` = 1.000000000000019 > 1 | - | low | loggamma F1/F6 | verified; cephes path not traced |
| `gammaincc(1e-310, 1)` | Returns a negative value (-5.77e-311) | - | low | dgamma | suspected only |
| `gammaln(a)`, subnormal a | Returns inf for a <~ 5.6e-309 (true ~709-744) | - | low | dgamma F5, loggamma F1; every `gammaln(shape)` logpdf | verified; no issue found |
| `betaln` (`lbeta`) | Up to 6.9e5 ulp when a >> b but a < 1e6*b (4e4 ulp at (1e6, 2)); the asymptotic branch beyond 1e6*b is fine | `betaln(1000, 0.5)` 7.7e-13 absolute | medium | beta F4, betaprime F2/F3, f, rdist | verified |

## Boost.Math (via `scipy.special` ufuncs and `scu._*` wrappers)

| Function | Problem | Example | Relevance | Seen in | Status |
|---|---|---|---|---|---|
| `ibeta_inv` (`betaincinv`, `betainccinv`, `fdtri`, `scu._beta_ppf`) | Hits its iteration cap and returns the last iterate or nan (with a RuntimeWarning), in irregular bands of small q; also clamps subnormal roots to DBL_MIN | `beta.ppf(1e-9, 0.5, 2)`; `betaprime.ppf(q, 2, 10)` constant 8.196e-53 for q <= 7.5e-114; `f.ppf(1e-153, 1, 1e4)` = 2.2e-304 vs 1.57e-306 | high (q = 1e-9 is ordinary) | beta F1/F3/F5-F7, betaprime F1, f F3/F4, jf_skew_t F8/F12 | verified. Related: boostorg/math #1511 (another case, fixed upstream), SciPy #12796 (fixed 1.12) |
| `ibeta_derivative` (`scu._beta_pdf`) | Raises Python `OverflowError` for x <~ 1e-308 instead of returning a value | `beta.pdf` raises where the true pdf is 1.2e155 | low (but a crash) | beta F2 | verified |
| `ibetac` at a = b = 0.5 (`betaincc(0.5, 0.5, z)`) | Argument effectively rounded via `1 - z`: 1.4e7 ulp at z = 1e-16, exactly 1.0 at z = 1e-20 | - | medium | f F7 (dfn = dfd = 1); beta.sf, betaprime at (0.5, 0.5) by construction | verified; no issue found |
| `ibeta` near DBL_MIN / large shapes | `betainc` up to 8.7e7 ulp near DBL_MIN; prefactor error at large shapes | - | low / medium | jf_skew_t F10, beta F8, f F8 | verified |
| `fisher_f` (`fdtr`, `fdtrc`) | `n*x` overflows inside Boost: sf 0 where the true value is a normal double | `f.sf(1e308, 2, 0.01)` = 0 | low | f F2 | verified |
| `students_t` (`stdtr`, `stdtrit`; also used by the t-tests in `_stats_py`/`_mstats_basic`) | `stdtr` exactly 0/1 for \|x\| > 1.34e154 (wrong for df <~ 2), 1.4e7 ulp near 0 at df == 1; `stdtrit` caps at sqrt(df/DBL_MIN) for small df, ~2x off beyond 1e50 then +inf with the wrong sign, 2.1e3 ulp for 2 < df < 3 | `t.cdf(-1e300, 0.001)` = 0 vs 0.2496; `t.ppf(0.29, 0.001)` = -2.1e152 vs -5.9e234; `t.ppf(1e-308, 2.74)` = +inf | medium (df == 1 near 0 and 2 < df < 3 are ordinary) | t F1, F4-F8 | verified. The 2 < df < 3 band matches Boost Trac #4504 (fixed 2010), possibly a remnant |
| `non_central_f` (`ncfdtr`, `ncfdtri`, `scu._ncf_*`) | sf at nc = 0 returns -cdf (negative) for every x; dfn + dfd = 2: nan in the left tail, 1.0/0.0 in the bulk for tiny nc; isf raises `OverflowError` for results >= ~2e16; series non-convergence at dfd = 1000 (sf 0); ~t ulp right-tail loss | `ncf.sf(x, dfn, dfd, 0)` < 0 | high (nc = 0 and the bulk) | ncf F1-F8 | verified. #14901 (closed) covers only warnings |
| `non_central_chi_squared` (`chndtr`, `chndtrix`, `scu._ncx2_*`) | Poisson series starts at the mode and returns 0 once that term underflows (pdf/cdf/sf 0 in wide tails, quantiles constant); `chndtrix` silent nan; sf `OverflowError` (tgamma) for x < 2**-25 once df/2 + nc/2 >~ 171; 5e10 ulp in a left-tail band with a subnormal intermediate; 4.9e15 ulp for x in ~[3e-81, 1e-78] | `ncx2.sf(4000, 21, 1000)` = 0; `rice.cdf(0.1, 15)` = 0 vs 9e-52 | high (rice for every b >= 14.14) | ncx2 F1, F2, F6-F9; rice F1, F2, F4-F6 | verified |
| `non_central_t` (`nctdtr`, `nctdtrit`, `scu._nct_*`) | cdf left tail (nc > 0) too large by up to 1e113, nan, or negative sf clipped to 0; 0/1 for \|x\| > 1.34e154; nan at df == 1 for tiny \|x\|; pdf `OverflowError` (`ibeta_derivative`/`tgamma`) for \|x\| >~ 1.3e154 and df >= 200; pdf ~0.2*df ulp (4.5% at df = 1e15); pdf returns pdf(0) for \|x\| < 1.5e-8; quantiles saturate, nan at q = 1e-300 | `nct.ppf(1e-300, 1e6, 2)` = -8.76 vs -35.06 | high (left tail at moderate nc) | nct F1-F13 | verified. Related: #20693 (pdf, fixed 1.15), PR #21728 thread (nan at high nc) |
| `skew_normal` (`scu._skewnorm_cdf/ppf/isf`) | `Phi(x) - 2T(x, a)` cancels (3.9e4 ulp; 952 ulp at x = 0 for a = 1e4); the quantile iterates on that cancelling cdf | `skewnorm.ppf(1e-20, 4)` = -4.50 vs -2.145 | high (q = 1e-20, a = 4) | skewnorm F2, F4 (F1 is SciPy's quad fallback) | verified. Related: PR #20643, boostorg/math #1120 |
| inverse Gaussian quantile (`scu._invgauss_ppf/isf`) | isf returns its search bound in the far right tail at every mu; ppf returns 1.1e248 in the far left tail for mu in ~[0.01, 0.4]; nan everywhere for mu < ~3e-3, and the nan fallback can raise `OverflowError` | `invgauss.isf(1e-100, 1)` = 4.4e34 vs 443.8 (also `wald`) | medium | invgauss F2-F5, wald | verified; no issue found (PR #16790 introduced it) |
| `powm1` (backend not checked) | Returns 0.0 instead of +inf on overflow when base < 1 and exponent < 0 | `powm1(0.5, -1024)` = 0.0 | low-medium (wrong end of support) | genlogistic F4 (ppf = +inf instead of -1676.9) | verified; #17855/#17872 reworked overflow but leave this case |

## AMOS (Bessel functions)

| Function | Problem | Example | Relevance | Seen in | Status |
|---|---|---|---|---|---|
| `kv`, `kve` | nan for x >= 2**30 - 0.5 (AMOS range limit) | `kve(0.5, 1073741823.5)` = nan | low-medium | geninvgauss F3 (logpdf/pdf/cdf nan; warning says "Infinite values") | verified; no issue found |
| `ive` | nan without a warning for z > 2**30 at every order | `ive(9.5, 1e10)` = nan | low-medium | ncx2 F4; `argus._stats` also calls it (not measured) | verified |

`sc.k1e` (cephes) has no such limit and is accurate from 1e-300 to 1e300.

## SciPy's own `scipy.special` code

| Function | Problem | Example | Relevance | Seen in | Status |
|---|---|---|---|---|---|
| `inv_boxcox1p` | Returns x when \|lmbda*x\| < 1e-154 (the lmbda -> 0 limit is `expm1(x)`) | `genpareto.cdf(2, 1e-300)` = 2.0 vs 0.865 | low (tiny lmbda), but the result exceeds 1 | genpareto F1 | verified; no issue found (#25190 only renames it) |
| `hyp2f1` (backend not checked) | Gamma(a-c)/Gamma(c-a) terms cancel for c ~ a and large z (1.6e11 ulp); ~c ulp for large c (2e6 ulp at c ~ 100); inf/nan where finite | - | medium | gausshyper F3-F5 | verified; F5 matches #26089 (closed) |
| `beta` (backend not checked) | Up to 9.6e5 ulp for large or unbalanced shapes (1.09e6 at (2, 1e6)); underflows to 0 where `betaln` is finite | `sc.beta(1000, 5)` 2.6e3 ulp | medium | gausshyper F2/F7, jf_skew_t F11 | verified |
| `ndtri_exp` | 2.5e3 ulp at y = -6.9e5 (plain Python check, not the harness) | - | low | powernorm (proposed fix for its F1-F3) | suspected only |

## NumPy (not SciPy)

| Function | Problem | Example | Relevance | Seen in | Status |
|---|---|---|---|---|---|
| complex `np.sqrt` | Halves the imaginary part of the result once \|Re\| or \|Im\| of the argument reaches DBL_MAX/(1+sqrt 2) = 7.4463e307 | `np.sqrt(complex(-1e308, -1e154))` = `0.5-5e+153j` vs `0.5-1e+154j`; also `np.sqrt(1.5e308j)` | low | rel_breitwigner F2 | verified on this Windows build only; no NumPy issue found |

## Checked and accurate

`log1p`, `expm1`, `xlogy`, `xlog1py`, `erfcx`, `log_ndtr` (z < 0), `ndtri`,
`k1e`, `i0e`, `lambertw`, `boxcox`, `boxcox1p`, `inv_boxcox`, Boost
`_cauchy_ppf`/`_cauchy_isf` (<= 0.96 ulp), the `cosine` ufuncs, Boost
`landau_distribution` (0 findings), `cmath.sqrt`. In most reviews the
problem was the arguments passed in, not the special function.

## Suggested order for reporting

1. Boost `non_central_f` sf at nc = 0 (negative probability at a documented
   parameter value) and `non_central_t` left tail - wrong in ordinary use.
2. Boost `ibeta_inv` non-convergence at q ~ 1e-9 (several distributions and
   every direct `betaincinv` user).
3. `skew_normal` cdf/quantile cancellation and the `non_central_chi_squared`
   mode-start underflow (`rice` for b >= 14.14).
4. Cephes `igam_fac` and `igami` (small a) - clear fix directions, moderate
   sizes.
5. `students_t` at df == 1 and 2 < df < 3; inverse-Gaussian quantiles.
6. The rest (subnormal arguments, AMOS 2**30 limit, `powm1`,
   `inv_boxcox1p`, NumPy `sqrt`) as low-priority reports.
