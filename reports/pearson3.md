# pearson3: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `pearson3_gen`, lines 8432-8625
- **Implemented:** pdf (8523-8534), logpdf (8536-8549), cdf (8551-8573), sf (8575-8590), ppf (8606-8613); all share `_preprocess` (8478-8504). logpdf/cdf/sf call the public `gamma.logpdf`/`gamma.cdf`/`gamma.sf` (via `gamma_gen._logpdf`/`_cdf`/`_sf`, lines 3632-3639), ppf calls `sc.gammaincinv`.
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf), isf = ppf(1 - q)
- **Related:** `gamma` (pearson3 is X = (G - alpha)/beta with G ~ Gamma(alpha), alpha = 4/skew**2; see `reports/gamma.md`); helpers: `_norm_pdf`, `_norm_cdf`, `_norm_sf`, `_norm_ppf` (normal branch); special functions: `sc.gammainc`/`sc.gammaincc` (through gamma), `sc.gammaincinv`, `sc.gammaln`, `sc.xlogy`
- **Reference:** `refs/pearson3_ref.py`, 400 digits (pdf/logpdf/cdf/sf), 100 digits for the ppf grid and the skew = +-0.01 grid (`--dps 100`; the module computes the smaller of P/Q directly, so it is valid there); self-checks: PASS (both precisions). Auxiliary reference for |skew| <= 1e-3 (alpha >= 4e6, where mpmath's `gammainc` does not converge or is very slow): `scratch/pearson3/pearson3_tiny_ref.py`, 100 digits, tail integrals of the pdf by `mp.quad`, Newton ppf; used only with `--at` single points.
- **Parameter sets:** grid: (0.1), (-2), (1), (-0.1), (10), (-100); (0.01), (-0.01) for pdf/logpdf/cdf/sf (ppf there with `--at`); single points at skew = +-1e-3, 1.7e-5, 1.6e-5 (just above SciPy's normal-branch threshold) and +-1.59e-5 (just below it)
- **Worst finding:** loss in ppf (skew < 0, small q: -inf; |skew| < 1.6e-5: 0.0 for -2.65e-6) and logpdf (|skew| < 1.6e-5: -inf); severe in cdf/sf (relative error 0.92 at alpha = 1.56e10), pdf, logpdf, ppf

## Summary

For moderate skew (|skew| ~ 0.1 ... 100) pdf, logpdf, cdf and sf are accurate to
a few thousand ulp at worst and ppf for skew > 0 to a few hundred. Everything
degrades as |skew| shrinks, because pearson3 hands an O(1) problem to `gamma`
with alpha = 4/skew**2 and t = alpha + beta*x: the gamma.logpdf terms of size
alpha*log(alpha) cancel (pdf 2.7e5 ulp at skew 0.01, 9.8e10 ulp at 1.6e-5),
the rounding of x - zeta and of gammaincinv(alpha, q)/beta + zeta costs
~(2/|skew|)*eps absolute (cdf 1.7e5 ulp, ppf 1.5e9 ulp), and below |skew| ~ 4.5e-3
`sc.gammainc`'s unconverged series (INDEX item 7, gamma F1) makes the cdf
left tail (sf right tail for skew < 0) and ppf wrong in the leading digit
(relative error 0.92 at alpha = 1.56e10). Below |skew| = 1.6e-5 SciPy switches to
the plain standard normal, which is a relative error of ~|skew|*He3(x)/6
(2e10 ulp in the bulk, ppf(0.5) = 0.0 instead of -skew/6) and whose logpdf
is `log(_norm_pdf(x))` = -inf for |x| > 38.6. Independently of skew, ppf for
skew < 0 computes `gammaincinv(alpha, 1 - q)`, returning -inf for q < 1.1e-16, and
`zeta = -alpha/beta` with alpha = beta**2 rounded shifts the support edge (sf(-0.2, 10) = 0.79 instead of 1).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf | skew < 0, q < 1e-3 (total for q < 1.1e-16) | `q = 1 - q` then `gammaincinv(alpha, q)` (patterns 1, 8) | -inf for -45.05; 2.3e11 ulp | loss |
| F2 | logpdf | \|skew\| < 1.6e-5, \|x\| > 38.6 | `np.log(_norm_pdf(x))` underflows (pattern 4) | -inf for -800.75 | loss |
| F3 | pdf, logpdf, cdf, sf, ppf | \|skew\| < 1.6e-5, all x / q | standard normal used instead of the skewed density (patterns 7, 9) | ppf 0.0 for -2.65e-6; cdf 3.1e14 ulp | loss |
| F4 | cdf, sf, ppf | \|skew\| <~ 4.5e-3 (alpha >~ 2e5), gamma lower tail: skew > 0 cdf / skew < 0 sf for 4.5 <~ \|x\| <~ 15; ppf for the matching q | `sc.gammainc` returns an unconverged 2000-term series (pattern 10; INDEX 7, gamma F1/F3) | relative error 0.92 (4.1e15 ulp); ppf 1.4e14 ulp | severe |
| F5 | sf, cdf (and pdf) | x within ~2 ulp of the support edge zeta = -2/skew | `zeta = -alpha/beta` with `alpha = beta**2` rounded is not -2/skew (patterns 5, 7) | sf(-0.2, 10) = 0.79 for 1.0 | severe |
| F6 | pdf, logpdf | all skew; grows as alpha = 4/skew**2 grows | `gamma.logpdf(t, alpha)` cancels terms of size alpha*log(alpha) (pattern 5; gamma F2/F4) | pdf 9.8e10 ulp, logpdf 1.1e11 ulp at skew 1.6e-5; pdf 2.7e5 at 0.01 | severe |
| F7 | ppf | small \|skew\|, q near the median (result << 2/\|skew\|) | `gammaincinv(alpha, q)/beta + zeta` cancels two terms of size 2/\|skew\| (patterns 5, 8) | 1.5e9 ulp at skew 1.6e-5; 2.06e4 at 0.01; 390 at 0.1 | severe |
| F8 | cdf, sf | small \|skew\|, \|x\| << 2/\|skew\| | `x - zeta` rounds with absolute error (2/\|skew\|)*eps (pattern 5) | 1.7e5 ulp at skew 1.7e-5; 102 ulp at skew 0.01 | moderate |
| F9 | cdf, sf | moderate skew (+-0.1), far tails | exp-prefactor of `igam`/`igamc` rounds its exponent (pattern 5; gamma F6) | 2.67e3 ulp | moderate |

### F1: ppf for negative skew rounds q away in `1 - q`

- **Code:** `_ppf`, line 8611: `q[beta < 0] = 1 - q[beta < 0]`, then line 8612 `sc.gammaincinv(alpha, q)/beta + zeta`.
- **Mechanism:** for skew < 0 the lower tail of X is the upper tail of G, so
  ppf needs the gamma quantile at upper-tail probability q. Forming `1 - q`
  keeps only the digits of q above eps (pattern 1/8); `gammaincinv` then
  inverts a probability with relative error ~eps/q. For q < 1.1e-16 `1 - q`
  is exactly 1, `gammaincinv(alpha, 1) = inf`, and the result is -inf.
  This is INDEX item 3 (`1 - u` inside ppf).
- **Regime:** every skew < 0 with q < ~1e-3; total loss for q < 1.1e-16.
  Present in every negative parameter set of the grid ((-2), (-0.1), (-100), (-0.01)).
- **Reproducer:** `python accuracy.py refs/pearson3_ref.py --dps 100 --at ppf 1e-20 -2`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-20, -2)` | -inf | -45.051701859880914 | inf | 0.0222 | loss |
  | `ppf(1e-300, -2)` | -inf | -689.7755278982137 | inf | 0.00145 | loss |
  | `ppf(1e-300, -0.1)` | -inf | -62.6678421759545 | inf | 0.00105 | loss |
  | `ppf(1e-20, -100)` | -inf | -1732.7870700640706 | inf | 0.0281 | loss |
  | `ppf(3.4807005884285624e-14, -2)` | -29.987407583768846 | -29.988957709878154 | 2.3e+11 | 0.0333 | severe |
  | `ppf(1e-12, -0.1)` | -7.862162438912861 | -7.862158602503196 | 2.2e+09 | 0.0221 | severe |
  | `ppf(1e-10, -100)` | -629.8777629620296 | -629.8777668133503 | 2.8e+07 | 0.0739 | severe |
  | `ppf(1e-10, -0.01)` | -6.427265698282213 | -6.427265711259277 | 9.1e+06 | 0.0244 | severe |
  | `ppf(1e-05, -2)` | -10.51292546497478 | -10.512925464970229 | 1.95e+03 | 0.0951 | moderate |

- **Fix direction:** for skew < 0 use `sc.gammainccinv(alpha, q)/beta + zeta` instead of `gammaincinv(alpha, 1 - q)`.

### F2: logpdf of the normal branch underflows to -inf

- **Code:** `_logpdf`, line 8545: `ans[mask] = np.log(_norm_pdf(x[mask]))`.
- **Mechanism:** `_norm_pdf(x) = exp(-x**2/2)/sqrt(2*pi)` underflows to 0 for
  |x| > 38.6 (and is subnormal from |x| ~ 37.5), so its log is -inf
  (pattern 4) although the log-density is an ordinary number. Spurious
  `divide by zero encountered in log` warnings accompany it.
- **Regime:** |skew| < 1.6e-5 and |x| > ~37.5 (inaccurate) / 38.6 (-inf).
- **Reproducer:** `python accuracy.py scratch/pearson3/pearson3_tiny_ref.py --at logpdf 40 1.59e-5`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(40.0, 1.59e-5)` | -inf | -800.749696921971 | inf | 2 | loss |
  | `logpdf(100.0, 1.59e-5)` | -inf | -4998.27131227561 | inf | 2 | loss |

  (Even with `_norm_logpdf` the normal branch would still be off by F3:
  -722.9189385332162 vs -722.7738627260784 at x = 38.)
- **Fix direction:** `-x**2/2 - log(sqrt(2*pi))` (`_norm_logpdf`) - but see F3, the branch itself should go.

### F3: below |skew| = 1.6e-5 the skewed distribution is replaced by the standard normal

- **Code:** `_preprocess`, lines 8490/8496: `norm2pearson_transition = 0.000016`,
  `mask = np.absolute(skew) < norm2pearson_transition`; then lines 8545, 8555,
  8579, 8609 return `_norm_pdf`/`_norm_cdf`/`_norm_sf`/`_norm_ppf`.
- **Mechanism:** the normal is only the zeroth-order term of the Edgeworth
  expansion in skew: pdf = phi(x)*(1 + skew/6*He3(x) + O(skew**2)), so the
  relative error is about |skew*He3(x)|/6 - 5.3e-6 (2.4e10 ulp) at x = 1 and
  growing like |x|**3 into the tails; for ppf the median is -skew/6, not 0
  (patterns 7, 9). The comment says the cut was "found by brute force"; it
  sits where this error happens to be about as large as F6/F7 on the gamma
  side (compare pdf(0, 1.6e-5) in F6), i.e. both branches are poor there.
- **Regime:** |skew| < 1.6e-5, every x and q.
- **Reproducer:** `python accuracy.py scratch/pearson3/pearson3_tiny_ref.py --at ppf 0.5 1.59e-5`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(0.5, 1.59e-5)` | 0.0 | -2.649999999990075e-06 | inf | 4.7e+05 | loss |
  | `ppf(1e-10, 1.59e-5)` | -6.361340902404056 | -6.361236316133916 | 7.4e+10 | 0.0241 | severe |
  | `ppf(0.999, -1.59e-5)` | 3.090232306167813 | 3.0902096499120244 | 3.3e+10 | 96 | severe |
  | `cdf(-30.0, -1.59e-5)` | 4.906713927147908e-198 | 5.270585181789969e-198 | 3.1e+14 | 901 | severe |
  | `sf(30.0, 1.59e-5)` | 4.906713927147908e-198 | 5.270585181789969e-198 | 3.1e+14 | 901 | severe |
  | `cdf(-10.0, 1.59e-5)` | 7.61985302416047e-24 | 7.599691759915726e-24 | 1.2e+13 | 101 | severe |
  | `sf(10.0, 1.59e-5)` | 7.61985302416047e-24 | 7.640065359910357e-24 | 1.2e+13 | 101 | severe |
  | `cdf(-3.0, 1.59e-5)` | 0.0013498980316300933 | 0.0013498040779843445 | 3.1e+11 | 9.85 | severe |
  | `pdf(10.0, -1.59e-5)` | 7.694598626706409e-23 | 7.674843871548535e-23 | 1.2e+13 | 100 | severe |
  | `pdf(3.0, 1.59e-5)` | 0.004431848411938009 | 0.004432059811714105 | 2.1e+11 | 9 | severe |
  | `logpdf(10.0, 1.59e-5)` | -50.918938533204674 | -50.916368188046185 | 2.3e+11 | 1.96 | severe |
  | `cdf(1.0, 1.59e-5)` | 0.8413447460685429 | 0.8413447460710918 | 1.36e+04 | 0.288 | moderate |

  (cond (shapes) is at most 0.07 at these points: the skew matters at full
  weight, it is not a 1-ulp-of-skew effect.)
- **Fix direction:** drop the normal branch and evaluate the gamma form
  stably for small skew (log1pmx-based logpdf as in F6, a Temme-type uniform
  expansion of P/Q written in x rather than t for cdf/sf, and the
  Cornish-Fisher series or a Newton step for ppf); if a branch is kept,
  use the Edgeworth/Cornish-Fisher terms up to O(skew**2) so the error is
  O(skew**3) ~ 1e-15.

### F4: cdf, sf and ppf inherit the unconverged `sc.gammainc` series for small |skew|

- **Code:** `_cdf`, line 8562 `gamma.cdf(transx, alpha)` (skew > 0) and `_sf`,
  line 8588 `gamma.cdf(...)` (skew < 0) -> `gamma_gen._cdf`, line 3636
  `sc.gammainc(a, x)`; `_ppf`, line 8612 `sc.gammaincinv(alpha, q)`.
- **Mechanism:** identical to gamma F1/F3 and INDEX item 7: cephes `igam`
  uses the Temme asymptotic branch only for |t - alpha| < 4.5*sqrt(alpha) and
  otherwise a Kummer series capped at 2000 terms, which for alpha >~ 2e5 is
  returned unconverged. In pearson3 units (t - alpha)/sqrt(alpha) = sign(skew)*x,
  so the window is 4.5 < -sign(skew)*x < ~15 in the left tail of the gamma,
  i.e. the cdf left tail for skew > 0 and the sf right tail for skew < 0;
  `gammaincinv` inverts the same function, so ppf at the matching q (small q
  for skew > 0, q near 1 for skew < 0) is wrong as well. Not re-derived here,
  see `reports/gamma.md` F1.
- **Regime:** |skew| <~ 4.5e-3 (alpha >~ 2e5) down to the normal-branch threshold
  1.6e-5 (alpha = 1.56e10, where the error is largest); |x| between ~4.5 and ~15
  on the gamma lower-tail side. Inside |x| < 4.5 the same parameters are accurate
  (`cdf(-3.0, 1.6e-5)` 0.723 ulp, `cdf(1.0, 1.6e-5)` 0.594 ulp).
- **Reproducer:** `python accuracy.py scratch/pearson3/pearson3_tiny_ref.py --at cdf -5 1.6e-5`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(-5.0, 1.6e-5)` | 2.286372405384866e-08 | 2.86556434753912e-07 | 4.1e+15 | 25.9 | severe |
  | `cdf(-8.0, 1.6e-5)` | 7.580778291177065e-17 | 6.212478139376239e-16 | 4.0e+15 | 65 | severe |
  | `cdf(-12.0, 1.6e-5)` | 3.1117072979251298e-34 | 1.7683154750696884e-33 | 3.7e+15 | 145 | severe |
  | `sf(8.0, -1.6e-5)` | 7.580778291177065e-17 | 6.212478139376239e-16 | 4.0e+15 | 65 | severe |
  | `cdf(-6.0, 1e-3)` | 9.504772918692302e-10 | 9.516967803091505e-10 | 5.8e+12 | 37.1 | severe |
  | `cdf(-10.0, 1e-3)` | 6.446173433922907e-24 | 6.446330247393282e-24 | 1.1e+11 | 101 | severe |
  | `sf(10.0, -1e-3)` | 6.446173433922907e-24 | 6.446330247393282e-24 | 1.1e+11 | 101 | severe |
  | `ppf(1e-10, 1.6e-5)` | -6.170308521192055 | -6.361235658360998 | 1.4e+14 | 0.0241 | severe |
  | `ppf(1e-10, 1e-3)` | -6.354625999210839 | -6.354764605279946 | 9.8e+10 | 0.0241 | severe |
  | `ppf(0.9999999999, -1e-3)` | 6.3546259865292996 | 6.354764592600246 | 9.8e+10 | 2.4e+08 | severe |

  The skew = +-0.01 grid (alpha = 4e4) is below the onset: cdf/sf there are within
  428 ulp (`cdf(-4.6, 0.01)` 36.2 ulp).
- **Fix direction:** in `scipy.special` (cephes `igam`/`igamc`): extend the
  uniform asymptotic branch or raise the iteration cap / use a continued
  fraction (shared, see "Shared code").

### F5: the support edge is computed as -alpha/beta and is off by 2 ulp

- **Code:** `_preprocess`, lines 8499-8503: `beta = 2.0/skew`,
  `alpha = beta**2`, `zeta = loc - alpha/beta`, `transx = beta*(x - zeta)`.
- **Mechanism:** `alpha = beta**2` is rounded, so `alpha/beta` is not `beta`:
  for skew = 10, beta = 0.2, alpha = 0.04000000000000001 and
  zeta = -0.20000000000000004 instead of -0.2 (pattern 5). Doubles between the
  computed and the true edge get t > 0: at x = -0.2 (which as a double is
  1.1e-17 *outside* the true support) t = 5.55e-18 and, because alpha is small,
  P(0.04, 5.55e-18) = 0.21. So cdf = 0.21 and pdf = 3.0e14 where both are 0
  (not measured, true value 0), and sf = 0.79 where it is exactly 1. Just
  inside the edge the relative error of t is large as well, but there the
  cdf is ill-conditioned (cond = alpha*|x|/(x - zeta)), so the grid points at
  zeta*(1 +- 1e-12) are within conditioning.
- **Regime:** x within a few ulp of zeta = -2/skew, any skew for which
  `(2/skew)**2/(2/skew)` does not round back to `2/skew`; visible as a large
  error when alpha is small (large |skew|).
- **Reproducer:** `python accuracy.py refs/pearson3_ref.py --at sf -0.2 10`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `sf(-0.2, 10)` | 0.7914348179019166 | 1.0 | 9.4e+14 | 0 | severe |
  | `cdf(0.2, -10)` | 0.7914348179019166 | 1.0 | 9.4e+14 | 0 | severe |

- **Fix direction:** `zeta = -beta` (= -2/skew exactly as SciPy computes beta) and `t = alpha + beta*x` or `beta*(x + beta)`.

### F6: pdf and logpdf cancel terms of size alpha*log(alpha) inside gamma.logpdf

- **Code:** `_logpdf`, line 8548: `np.log(abs(beta)) + gamma.logpdf(transx, alpha)`
  -> `gamma_gen._logpdf`, line 3633 `sc.xlogy(a-1.0, x) - x - sc.gammaln(a)`;
  `_pdf`, line 8528 `np.exp(self._logpdf(x, skew))`.
- **Mechanism:** with t = alpha + beta*x ~ alpha, the three gamma terms are each
  ~alpha*log(alpha) (2.4e3 at skew 0.1, 4.2e5 at 0.01, 3.6e11 at 1.6e-5) while
  the result is O(1) (pattern 5, gamma F4); the absolute error ~alpha*log(alpha)*eps
  becomes the relative error of the pdf (gamma F2). The x - zeta rounding of F8
  adds to it. For |skew| just above 1.6e-5 the error (9.8e10 ulp at x = 0)
  exceeds that of the normal approximation it replaces (2.37e4 ulp at
  `pdf(0.0, 1.59e-5)`).
- **Regime:** every skew; relative error ~ 4*log(4/skew**2)/skew**2 * eps:
  ~2e3 ulp at |skew| = 0.1, ~2e5 ulp at 0.01, ~1e7 ulp at 1e-3, ~1e11 ulp at 1.6e-5.
- **Reproducer:** `python accuracy.py scratch/pearson3/pearson3_tiny_ref.py --at pdf 0 1.6e-5`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.0, 1.6e-5)` | -0.9189602805905626 | -0.9189385332100061 | 1.1e+11 | 0 | severe |
  | `pdf(0.0, 1.6e-5)` | 0.39893360454405197 | 0.39894228039930496 | 9.8e+10 | 0 | severe |
  | `logpdf(3.0, 1.6e-5)` | -5.418838210278063 | -5.418890534217986 | 4.3e+10 | 1.66 | severe |
  | `pdf(-5.0, -1e-3)` | 1.5141730764548838e-06 | 1.5141730866008598e-06 | 3.0e+07 | 24.9 | severe |
  | `logpdf(1.0, 1e-3)` | -1.4192718277174965 | -1.419271824888001 | 9.0e+06 | 0.705 | severe |
  | `pdf(0.0, 1e-3)` | 0.3989422713884718 | 0.3989422720901353 | 7.9e+06 | 0 | severe |
  | `pdf(-20.0, 0.01)` | 3.349437186079939e-94 | 3.34943718588102e-94 | 2.7e+05 | 444 | moderate |
  | `pdf(6.0, 0.01)` | 8.388452687083668e-09 | 8.388452686599789e-09 | 2.6e+05 | 35 | moderate |
  | `logpdf(-0.005, 0.01)` | -0.918928116408023 | -0.9189281164338381 | 1.3e+05 | 0 | moderate |
  | `pdf(-1.0, 0.1)` | 0.2502809162079795 | 0.2502809162078569 | 2.21e+03 | 1 | moderate |
  | `logpdf(-0.3, 0.1)` | -0.9494883527034328 | -0.9494883527038235 | 1.85e+03 | 0.0802 | moderate |

  On the grid all 190 pdf points of (+-0.01) are moderate (median 1.1e5-1.5e5 ulp).
- **Fix direction:** with u = x/beta (so t = alpha*(1 + u)) and
  log|beta| = log(alpha)/2: `logpdf = -log(2*pi)/2 - stirlerr(alpha) + alpha*log1pmx(u) - log1p(u)`,
  where `stirlerr(a) = gammaln(a) - (a - 1/2)log(a) + a - log(2*pi)/2` and
  `log1pmx(u) = log1p(u) - u` (`sc._ufuncs._log1pmx` exists); no large terms are formed.

### F7: ppf cancels gammaincinv(alpha, q)/beta against zeta

- **Code:** `_ppf`, line 8612: `sc.gammaincinv(alpha, q)/beta + zeta`.
- **Mechanism:** `gammaincinv(alpha, q)/beta` ~ 2/|skew| and `zeta` = -2/skew;
  their sum is O(1) (and only -skew/6 at the median), so the rounding of the two
  large terms, ~(2/|skew|)*eps absolute, becomes a relative error of
  (2/|skew|)/|ppf| ulp (patterns 5, 8). Near q = 0.5 the harness's cond (arg)
  absorbs part of it (ppf ~ -skew/6 is itself small), but not all.
- **Regime:** small |skew| (alpha large), q in the bulk; worst near the median.
  Grid: (0.1), (-0.1) at q = 0.5: 390-428 ulp (cond 75.2).
- **Reproducer:** `python accuracy.py scratch/pearson3/pearson3_tiny_ref.py --at ppf 0.5 1.6e-5`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(0.5, 1.6e-5)` | -2.6666675694286823e-06 | -2.666666666656553e-06 | 1.5e+09 | 4.7e+05 | severe |
  | `ppf(0.3, 1.7e-5)` | -0.5244025668653194 | -0.5244025668792529 | 1.2e+05 | 1.65 | moderate |
  | `ppf(0.7, -1.7e-5)` | 0.5244025668653194 | 0.5244025668792527 | 1.2e+05 | 3.84 | moderate |
  | `ppf(0.5, 1e-3)` | -0.0001666666641995107 | -0.00016666666419753065 | 5.35e+04 | 7.52e+03 | moderate |
  | `ppf(0.5, 0.01)` | -0.0016666641975007224 | -0.0016666641975083285 | 2.06e+04 | 752 | moderate |
  | `ppf(0.45, 1e-3)` | -0.1258253756288923 | -0.1258253756288485 | 1.57e+03 | 9.04 | moderate |
  | `ppf(0.5, -0.1)` | 0.016664195276781868 | 0.016664195276780425 | 390 | 75.2 | minor |
  | `ppf(0.3, 0.01)` | -0.5256064017136168 | -0.5256064017136407 | 205 | 1.64 | minor |

- **Fix direction:** return `(g - alpha)/beta` with `g - alpha` obtained
  without forming g (e.g. invert the stable cdf of F8 in x, or a Newton
  correction in x starting from the Cornish-Fisher quantile).

### F8: x - zeta rounds with an absolute error of (2/|skew|)*eps

- **Code:** `_preprocess`, line 8503: `transx = beta * (x[invmask] - zeta)`.
- **Mechanism:** |zeta| = 2/|skew|, so for |x| << 2/|skew| the difference
  rounds with absolute error up to |zeta|*eps/2, i.e. a shift of x by that
  amount; cdf/sf then change by |d log F/dx| * |zeta| * eps relative, which
  exceeds cond (arg) = |x d log F/dx| whenever |x| < |zeta| (pattern 5). It does not
  show when x + 2/skew is exactly representable (e.g. skew = 1.6e-5 or 1e-3,
  where zeta = -125000 / -2000 and x = -3.0, 0.1 subtract exactly at
  `cdf(-3.0, 1.6e-5)` 0.723 ulp, `cdf(0.1, 1e-3)` 0 ulp).
- **Regime:** small |skew|, |x| < 2/|skew|; ~ (2/|skew|)*|d log F/dx| ulp at most.
- **Reproducer:** `python accuracy.py scratch/pearson3/pearson3_tiny_ref.py --at cdf -3.3 1.7e-5`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(-3.3, 1.7e-5)` | 0.00048337587448659563 | 0.0004833758745046659 | 1.7e+05 | 11.8 | moderate |
  | `sf(3.3, 1.7e-5)` | 0.00048347241301976323 | 0.00048347241300990686 | 9.18e+04 | 11.8 | moderate |
  | `cdf(0.1, 1.7e-5)` | 0.5398289507309657 | 0.5398289507286944 | 1.89e+04 | 0.0735 | moderate |
  | `cdf(1e-10, 0.01)` | 0.5006649039328944 | 0.5006649039329057 | 102 | 7.97e-11 | minor |
  | `sf(1e-10, 0.01)` | 0.4993350960671055 | 0.4993350960670942 | 102 | 7.99e-11 | minor |

- **Fix direction:** form t as `alpha + beta*x` (one rounding of the O(sqrt(alpha))
  part) and, better, evaluate P/Q from the standardized x directly (Temme's
  eta from `log1pmx(x/beta)`), so no quantity of size 2/|skew| is rounded.

### F9: cdf/sf far tails lose the rounding of the igam/igamc prefactor

- **Code:** `_cdf`/`_sf`, lines 8562, 8571, 8584, 8588 -> `sc.gammainc`/`sc.gammaincc`.
- **Mechanism:** gamma F6: inside cephes `igam`/`igamc` the prefactor
  `exp(a*log(t) - t - lgam(a))` is computed from terms of size ~alpha*log(alpha)
  whose sum is the large negative log of the result; its rounding becomes the
  relative error (pattern 5). Not re-derived here.
- **Regime:** moderate skew (alpha = 400 at |skew| = 0.1), |x| >~ 9 in either tail.
- **Reproducer:** `python accuracy.py refs/pearson3_ref.py --at cdf -37 -0.1`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(-37.0, -0.1)` | 3.906111143528529e-142 | 3.9061111435262134e-142 | 2.67e+03 | 481 | moderate |
  | `sf(37.0, 0.1)` | 3.906111143528529e-142 | 3.9061111435262134e-142 | 2.67e+03 | 481 | moderate |
  | `cdf(-9.0, 0.1)` | 9.157449511317587e-28 | 9.157449511312678e-28 | 2.41e+03 | 148 | moderate |
  | `sf(9.0, 0.1)` | 1.0310505409529096e-15 | 1.0310505409525604e-15 | 1.53e+03 | 56.8 | moderate |

- **Fix direction:** in `scipy.special` (log-space prefactor with `log1pmx`, as in
  Boost's `regularised_gamma_prefix`); shared with gamma.

## RuntimeWarnings

- pdf: `underflow encountered in exp` in all parameter sets, where the true
  pdf is below the normal range; the results are correct (0 or subnormal) -
  spurious.
- logpdf, |skew| < 1.6e-5: `underflow encountered in exp` and
  `divide by zero encountered in log` for |x| >~ 38: the result is -inf and
  wrong (F2).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf, sf | full grid (0.1), (-2), (1), (-0.1), (10), (-100); (+-0.01) | within conditioning except F8/F9 rows; (-2) 16.8 ulp (cond 37), (1) 39.8 ulp, (10)/(-100) within cond (up to 3.5e10 ulp at the support edge with cond 4.0e10) |
| pdf, logpdf | (-2), (1) (alpha = 1, 4) | 0.68 / 1.6 ulp (logpdf), 61.5 ulp pdf within cond |
| pdf, logpdf | (10), (-100) (alpha = 0.04, 4e-4), near the support edge | 8.4e11 / 9.4e10 ulp, within cond (9.6e11 / 1.0e12) |
| ppf | (0.1), (1) full q grid, incl. q to 1e-300 and 1 - 1e-12 | 390 / 47.4 ulp (F7 at q = 0.5 for (0.1)); small q for skew > 0 at most 13.6 ulp |
| ppf | (10), (-100) single points (q = 1e-3, 0.5, 0.999, 1 - 1e-6) | 10 ulp (`ppf(1e-3, -100)`); gamma F5 (gammaincinv for alpha << 1) is diluted by + zeta and not visible |
| cdf, sf | \|skew\| = 1.6e-5, \|x\| <= 3 (Temme window of igam) | 0.72 ulp |

## Suspected, not confirmed

- **alpha >= 1e7 beyond the measured points:** the tiny-|skew| regime was
  measured only at the single points listed (skew = +-1e-3, 1.7e-5, 1.6e-5,
  +-1.59e-5), with the quadrature reference at 100 digits; no grid was run
  there (a previous grid attempt over this regime grew to 6.6 GB). In
  particular cdf/sf for |x| > 15 at alpha ~ 1e10 (beyond the unconverged-series
  window) and ppf for q < 1e-10 were not measured.
- **ppf for skew > 0 and q -> 1** uses `gammaincinv(alpha, q)` rather than
  `gammainccinv(alpha, 1 - q)`; the representation error of q is covered by cond (arg), so no excess error is expected or was seen (high tail 36.3 ulp).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| isf | ppf(1 - q) | skew > 0: `gammainccinv(alpha, q)/beta + zeta`; skew < 0: `gammaincinv(alpha, q)/beta + zeta` (with F7 caveat) | q < ~1e-3 loses eps/q; q < 1.1e-16 total: `isf(1e-20, 0.1)` = inf vs 10.722841915705135; `isf(1e-20, -0.1)` = 20.0 (the support edge) vs 7.9034544459197252 |
| logsf | log(sf) | log of Q(alpha, t) (skew > 0) / P(alpha, t) (skew < 0) in log space: log-prefactor `alpha*log1pmx(u) - ...` plus log of the series/continued fraction; `log_ndtr(-x)` style for the normal limit | sf underflows: `logsf(200, 0.1)` = -inf vs -3047.0596299934852 |
| logcdf | log(cdf) | same with P/Q swapped | cdf underflows near the support edge / left tail: `logcdf(-19.9, 0.1)` = -inf vs -1725.2368258113432 |

(Demonstration values from `scratch/pearson3/opp.py`, mpmath at 60 digits on the gamma form.)

## Shared code

- `scipy.special.gammainc` / `gammaincc` (cephes `igam`/`igamc`): unconverged
  2000-term series for large a (F4; INDEX item 7, gamma F1) and the rounded
  exp-prefactor (F9; gamma F6). `gammaincinv` inherits the first (F4 ppf;
  gamma F3). Affects gamma, erlang, chi2, chi, invgamma, dgamma, loggamma, pearson3, ...
- `gamma_gen._logpdf` (`xlogy(a-1, x) - x - gammaln(a)`): large-term
  cancellation (F6; INDEX item 5, gamma F2/F4), reached by pearson3 through
  the public `gamma.logpdf`.

## Existing GitHub issues

- #18625 (closed): BUG: scipy.stats.Pearson3 ppf function uses right tail probability instead of left - fixed in 1.9.2 (gh-17055) for skew < 0; the current `gammaincinv(alpha, 1-q)` branch behind F1 is presumably that fix, which corrects the tail but rounds `1 - q` (relates to F1)
No issue found for the other findings (web search; `gh` CLI unavailable).

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(1e-20, -2)` | -inf | -45.051701859880914 | verified | also `ppf(3.4807005884285624e-14, -2)` 2.33e11 ulp, `ppf(1e-12, -0.1)` 2.2e9 ulp; reference by Newton on a quadrature cdf; lines 8611/8612 match |
| F2 | `logpdf(40.0, 1.59e-5)` | -inf | -800.749696921971 | verified | `logpdf(100.0, 1.59e-5)` -inf vs -4998.2713122756104; line 8545 matches |
| F3 | `ppf(0.5, 1.59e-5)` | 0.0 | -2.6499999999900749e-06 | verified | `cdf(-3.0, 1.59e-5)` 3.13e11 ulp, `pdf(3.0, 1.59e-5)` 2.15e11 ulp match the report. `cdf(-30.0, -1.59e-5)`: verifier 5.2705851817899695e-198 vs analyst 5.270585181761525e-198 (5e-12 relative). The verifier's value is the one to trust: the analyst's `_tail_quad` integrates the unscaled pdf, and `mp.quad`'s absolute tolerance at 100 digits does not resolve values around 1e-198. The verifier's rescaled quadrature agrees between 100 and 200 digits to 6.5e-97. Severity unchanged (3.1e14 ulp) |
| F4 | `cdf(-5.0, 1.6e-5)` | 2.286372405384866e-08 | 2.86556434753912e-07 | verified | 4.14e15 ulp; `cdf(-8.0, 1.6e-5)`/`sf(8.0, -1.6e-5)` 3.95e15, `cdf(-6.0, 1e-3)` 5.77e12, `ppf(1e-10, 1.6e-5)` 1.35e14 ulp; the code path (8562/8588 -> `sc.gammainc`) matches; the cephes series cap was not re-derived |
| F5 | `sf(-0.2, 10)` | 0.7914348179019166 | 1.0 | verified | confirmed that alpha = 0.04000000000000001, zeta = -0.20000000000000004 and `gammainc(alpha, t)` = 0.2086 at x = -0.2; lines 8500-8503 match |
| F6 | `pdf(0.0, 1.6e-5)` | 0.39893360454405197 | 0.39894228039930499 | verified | 9.79e10 ulp; `logpdf(0.0, 1.6e-5)` 1.07e11, `pdf(0.0, 1e-3)` 7.92e6, `pdf(-20.0, 0.01)` 2.67e5 ulp; line 8548 matches |
| F7 | `ppf(0.5, 1.6e-5)` | -2.6666675694286823e-06 | -2.666666666656553e-06 | verified | 1.52e9 ulp; `ppf(0.5, 0.01)` 2.06e4 ulp; line 8612 matches |
| F8 | `cdf(-3.3, 1.7e-5)` | 0.00048337587448659563 | 0.0004833758745046659 | verified | 1.68e5 ulp; line 8503 matches |
| F9 | `cdf(-37.0, -0.1)` | 3.906111143528529e-142 | 3.9061111435262132e-142 | verified | 2.67e3 ulp; checked against `mp.gammainc(400, 1140, inf)` (complementary route) at 100/200/300 digits; `cdf(-9.0, 0.1)` 2.41e3 ulp |

The references are in `scratch/pearson3/verify/` (`vref.py`, `run.py`, `f3check.py`, `f9check.py`). The pdf is the docstring formula at the exact double skew, cdf/sf come from rescaled `mp.quad` of the pdf in x, and ppf is Newton on log cdf/log sf. All values agree between 100 and 200 digits to better than 1e-89.

Verified by dist-accuracy-verifier: 9 of 9 verified.
