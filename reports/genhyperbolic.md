# genhyperbolic: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `genhyperbolic_gen`, lines 4010-4236; logpdf/pdf kernels in the compiled Cython module `scipy.stats._stats` (`genhyperbolic_logpdf`, `genhyperbolic_pdf`, LowLevelCallable `_genhyperbolic_pdf`; .pyx not shipped, so their structure is inferred from the output, see F5)
- **Implemented:** pdf, logpdf (Cython kernel through `np.vectorize`), cdf, sf (`integrate.quad` of the Cython pdf in `_integrate_pdf`, lines 4133-4168)
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf), ppf = numerical inversion of cdf, isf = ppf(1 - q)
- **Related:** norminvgauss is the case p = -1/2 (separate implementation, no shared code); geninvgauss is the mixing distribution (used only by `_rvs`); helpers: `_integrate_pdf`; special functions: `sc.kve` (in the Cython kernel), `sc.kv` (mean for the quad split, `_stats`), `integrate.quad`
- **Reference:** `refs/genhyperbolic_ref.py`, 100 digits (mp.besselk with integer order takes up to 14 s at 400 digits). pdf from the docstring formula with `mp.besselk`; cdf/sf from the normal variance-mean mixture X = bV + sqrt(V) Z, V ~ GIG, integrated in log V by a converged trapezoidal rule (mp.quad split at the peak and the step for heavy tails); |b| = a uses the g -> 0 limit of the normalization. The reference was checked against `mp.quad` of the reference pdf (25 digits). Self-checks: PASS
- **Parameter sets:** (0.5, 1.5, -0.5), (1, 2, 0), (-1, 1, 1), (-1.5, 1, 0.9999999999), (1, 1e6, 0), (100, 0.01, 0), (0.5, 0.001, 0), (-2, 3, -2). pdf/logpdf on all 8 sets; cdf/sf grid on (0.5, 1.5, -0.5), (-1.5, 1, 0.9999999999), (1, 1e6, 0) only (quad-based SciPy cdf and quadrature reference are slow); the other sets with `--at` / single points. Custom x grid (REPLACE_GRID): -10..10, +-{1e-300, 1e-20, 1e-3, 0.3, 20, ..., 1e300}, both sides of a*sqrt(1+x^2) = 2**30 and of x**2 overflow.
- **Worst finding:** loss in pdf, logpdf, cdf, sf

## Summary

pdf and logpdf are accurate (a few ulp, up to ~250 ulp where `-a*s` and `b*x`
cancel at |x| ~ 100) for moderate shapes, and cdf/sf are good to ~150 ulp in
exponential tails and a few ulp in the bulk for those shapes. Everything
breaks down at the edges of the parameter space, almost always silently.
For |b| = a with p < 0, a combination `_argcheck` explicitly allows, pdf and
logpdf are nan for every x, and cdf and sf return 1.0 for every x (F1). The
Cython kernel calls `scipy.special.kve`, which returns nan for arguments
>= 2**30. logpdf is therefore nan for |x| >= ~1.07e9/a in every parameter set,
and for a >= 2**30 every method is nan or 1.0 (F2). `kve(p, g)` overflows
for large p and small g (p >= 81 at g = 0.01): pdf becomes 0 or nan and
cdf/sf 1.0 (F3). The cdf/sf clip `max(0, min(1, quad))` turns every nan from
the integrand into 1.0 and every negative quad result into 0 (F1-F4). In the
heavy right tail (b close to a) quad returns 1.0, 0.0 and 0.722 instead of
2.7e-7, 2.7e-10 and 1 (F4). For large a, cdf(-0.008, 1, 1e6, 0) = 1.0
(true 6.2e-16), because the split at the mean is disabled (`kv` underflows)
and quad samples beyond the kve limit (F4). The kernel adds the O(a) terms
of the normalizing constant and of the x-dependent part separately, so logpdf
has an absolute error ~0.3*a*eps: 9e5 ulp in pdf at a = 1e6, 1.8e8 ulp at
a = 1e9, and 2.5e7 ulp at x = 1e9 when b is close to a (F5). cdf/sf inherit
that error (3.5e5 ulp at a = 1e6, F6).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf, logpdf, cdf, sf | p < 0 and \|b\| = a (allowed by `_argcheck`), every x | g = 0: `log(0)` and `kve(p, 0) = inf` in the normalization; `mean` = nan; quad nan clipped to 1.0 (patterns 9, 10) | pdf nan for 0.368; cdf 1.0 for 7.6e-4; sf 1.0 for 0.051 | loss |
| F2 | logpdf (all shapes); pdf, cdf, sf for a >= 2**30 | a*sqrt(1+x^2) >= 2**30 (\|x\| >= ~2**30/a); every x when a >= 2**30 | `scipy.special.kve` returns nan for z >= 2**30 (INDEX item 16; pattern 10) | logpdf nan for -2.0e9 ... -2e300; pdf(0, 1, 1.1e9, 0) nan for 13231 | loss |
| F3 | pdf, logpdf, cdf, sf | log K_p(g) > 709.8: large \|p\|, small g (p >= 81 at g = 0.01), every x | `kve(p, g)` overflows to inf (pattern 3) | pdf 0.0 or nan for 3.1e-4; cdf/sf 1.0 for 0.5 | loss |
| F4 | cdf, sf | heavy right tail (b close to a), x >~ 1e4; g >~ 705, beyond ~8 sd from the peak | quad reaches the kve-nan region or extrapolates to a negative value; `max(0, min(1, .))` clip; split at the mean disabled because `kv(p, g)` underflows (pattern 10, INDEX item 15) | sf 1.0 for 2.7e-7, 0.0 for 2.7e-10; cdf 1.0 for 6.2e-16; cdf 0.722 for 1.0 | loss |
| F5 | pdf, logpdf | large a (bulk); b close to a with large x; mild at \|x\| ~ 100 | O(a) terms (`+g`, `-a*s`, `b*x`) rounded in separate partial sums before they cancel (pattern 5) | 1.8e8 ulp (pdf, a = 1e9); 2.5e7 ulp (pdf(1e9), b/a = 1 - 1e-10); 8.8e5 ulp (a = 1e6) | severe |
| F6 | cdf, sf | large a (bulk); heavy tails before F4 | integrates the pdf of F5; `epsrel = 1e-10` (pattern 10) | 3.5e5 ulp (a = 1e6); 4.7e5 ulp (sf(1e3), b/a = 1 - 1e-10) | moderate |

### F1: |b| = a with p < 0 gives nan pdf and cdf = sf = 1.0 everywhere

- **Code:** `_argcheck` (lines 4103-4105) accepts `abs(b) <= a` for p < 0. The Cython kernel (`_stats.genhyperbolic_logpdf`/`genhyperbolic_pdf`, called through `np.vectorize` in `_logpdf`/`_pdf`, lines 4115-4131) builds the normalization from `(a+b)*(a-b)` and `kve(p, sqrt((a+b)*(a-b)))`. `_integrate_pdf`, lines 4147-4168, computes `mean = b/d * sc.kv(p + 1, d) / sc.kv(p, d)` and returns `max(0.0, min(1.0, intgrl))`.
- **Mechanism:** with g = 0, the normalization combines `(p/2)*log(0)` = +inf (p < 0) with `-log(kve(p, 0))` = -inf, which gives nan. The finite limit `g^p/K_p(g) -> 1/(Gamma(-p) 2^(-p-1))` is never taken (pattern 9). In `_integrate_pdf`, `d = 0` makes `mean` = (1/0)*inf/inf = nan, so there is no split. The integrand is nan, so quad returns nan and the warning "Infinite values encountered in scipy.special.kve. Values replaced by NaN" fires. Python `min(1.0, nan)` then returns 1.0 (the first argument wins when the comparison is false). Result: cdf = sf = 1.0 for every x.
- **Regime:** p < 0, |b| = a (b = -a likewise), every x. The pdf grid on (-1, 1, 1) flagged all 69 points as loss. cdf is 1.0 at all 47 grid points where the reference is a normal double.
- **Reproducer:** `python accuracy.py refs/genhyperbolic_ref.py --at pdf 0 -1 1 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0, -1, 1, 1)` | nan | 0.36787944117144233 | inf | 0 | loss |
  | `pdf(-200, -1, 1, 1)` | nan | 2.3998643802752552e-179 | inf | 402 | loss |
  | `pdf(500, -1, 1, 1)` | nan | 2.0019889877201457e-06 | inf | 2 | loss |
  | `logpdf(0, -1, 1, 1)` | nan | -1.0 | inf | - | loss |
  | `cdf(0, -1, 1, 1)` | 1.0 | 0.18393972058572117 | 2.0e16 | - | loss |
  | `cdf(-2, -1, 1, 1)` | 1.0 | 0.0007635212590754344 | 5.9e18 | - | loss |
  | `cdf(-200, -1, 1, 1)` | 1.0 | 1.1939699146684509e-179 | 3.8e194 | - | loss |
  | `sf(10, -1, 1, 1)` | 1.0 | 0.05101293424214739 | 8.4e16 | - | loss |

  A cond of "-" means the harness could not differentiate here: perturbing b leaves the domain.
- **Fix direction:** for g == 0 and p < 0, use `log C = -lgamma(-p) - (-p-1) log 2 - log(2 pi)/2 - (p - 1/2) log a`. Take the split point from the inverse-gamma mixing law (mean b/(2(-p-1)) for p < -1, otherwise the mode). Keep nan in the clip (`np.clip`).

### F2: kve returns nan for arguments >= 2**30

- **Code:** Cython kernel: `log(kve(p - 1/2, a*sqrt(1 + x^2)))` and `log(kve(p, g))`.
- **Mechanism:** `scipy.special.kve(nu, z)` is nan for z >= 2**30 at every order (AMOS range limit, INDEX item 16). Measured: `kve(nu, 2**30 - 0.5)` is finite and `kve(nu, 2**30)` is nan for nu in {0, 0.5, 1, 1.5, -1.5, -2, 0.3, 99.5}. logpdf is therefore nan although its true value is about -(a - b sgn x)|x|. The pdf is affected where it is still a normal double, which needs a heavy tail (consequences for cdf/sf in F4) or a >= 2**30. In the latter case `kve(p, g)` is nan too and every method fails at every x: pdf/logpdf are nan, cdf/sf are 1.0 (F1 clip). x**2 also overflows for |x| > 1.34e154, but kve has failed long before that.
- **Regime:** logpdf for |x| >= ~2**30/a in every parameter set (all 8 sets: every grid point from 1.01*2**30/a to 1e300, both signs); all methods for a >= 2**30.
- **Reproducer:** `python accuracy.py refs/genhyperbolic_ref.py --at logpdf 1e9 0.5 1.5 -0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e9, 0.5, 1.5, -0.5)` | nan | -2000000009.7225168 | inf | 1 | loss |
  | `logpdf(542239620.8675, 1, 2, 0)` (1.01*2**30/a) | nan | -1084479240.4610758 | inf | 1 | loss |
  | `logpdf(1e+300, 1, 2, 0)` | nan | -2e+300 | inf | 1 | loss |
  | `logpdf(1e+20, 0.5, 0.001, 0)` | nan | -1.0000000000000003e+17 | inf | 1 | loss |
  | `logpdf(-1e+300, -2, 3, -2)` | nan | -1e+300 | inf | 1 | loss |
  | `pdf(0, 1, 1.1e9, 0)` | nan | 13231.418566492357 | inf | 0 | loss |
  | `logpdf(0, 1, 1.1e9, 0)` | nan | 9.490349474829786 | inf | 0 | loss |
  | `cdf(0, 1, 1.1e9, 0)` | 1.0 | 0.5 | 4.5e15 | 0 | loss |

- **Fix direction:** for large z use `log K_nu(z) = -z + log(sqrt(pi/(2z))) + log1p(asymptotic series)`, or fix `kve` (shared).

### F3: kve(p, g) overflows for large |p| and small g

- **Code:** Cython kernel: `log(kve(p, g))` and `log(kve(p - 1/2, a*s))`.
- **Mechanism:** K_p(g) ~ Gamma(|p|)/2 (2/g)^|p| overflows (pattern 3), although the log is moderate (logpdf(0, 100, 0.01, 0) = -8.17). If only `kve(p, g)` overflows, the normalization is -inf and pdf = 0.0 (p = 81). If `kve(p - 1/2, a*s)` overflows too, inf - inf = nan (p = 100, p = -100). cdf/sf get nan from quad and return 1.0 (F1 clip).
- **Regime:** `kve(p, g)` first overflows at p = 81 (g = 0.01), p = 150 (g = 1) and p = 241 (g = 10), for every x; negative p likewise.
- **Reproducer:** `python accuracy.py refs/genhyperbolic_ref.py --at pdf 0 100 0.01 0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0, 100, 0.01, 0)` | nan | 0.0002831581856131955 | inf | 0 | loss |
  | `pdf(0, 81, 0.01, 0)` | 0.0 | 0.0003148991533151041 | inf | 0 | loss |
  | `pdf(0, -100, 0.01, 0)` | nan | 5.634847908076101 | inf | - | loss |
  | `logpdf(0, 100, 0.01, 0)` | nan | -8.169504856716916 | inf | - | loss |
  | `cdf(3, 100, 0.01, 0)` | 1.0 | 0.500849473910032 | 4.5e15 | 0.0017 | severe |
  | `sf(0, 100, 0.01, 0)` | 1.0 | 0.5 | 4.5e15 | - | loss |

  `cdf(3, ...)` is rated severe rather than loss only because the relative error (0.997) is just below 1. The pdf and logpdf grids on (100, 0.01, 0) flagged all 69 points as loss.
- **Fix direction:** compute log K_nu directly where `kve` overflows: the small-z form `lgamma(|nu|) + (|nu|-1) log 2 - |nu| log z` with corrections, or the uniform large-order expansion. A shared `log_kv` helper would also fix geninvgauss F2.

### F4: cdf/sf from quad: nan clipped to 1.0, negative results clipped to 0, split at the mean disabled

- **Code:** `_integrate_pdf`, lines 4147-4168: `mean = b/d * sc.kv(p+1, d)/sc.kv(p, d)`; `if x0 < mean < x1` two quads, otherwise one `integrate.quad(llc, x0, x1, epsrel=1e-10, epsabs=0)`; `return max(0.0, min(1.0, intgrl))`.
- **Mechanism:**
  - (a) Heavy right tail (b close to a; density ~ x^(p-1) up to x ~ 1/(a-b)). `qagi` on [x, inf) subdivides toward large x and evaluates the pdf beyond the kve limit (F2). quad returns nan ("maximum number of subdivisions (50)"), which becomes 1.0, so sf(1e4) = 1.0 for 2.7e-7. Further out, the epsilon extrapolation goes negative: `quad(pdf, 1e6, inf)` = -3.99e-16 ("probably divergent"), clipped to 0.0. cdf(1e9) = 0.722 for 1 - 6.7e-15.
  - (b) For g >~ 705, `sc.kv(p, g)` underflows, so `mean` = 0/0 = nan and the split is never made. With a = 1e6 (sd 1e-3), `cdf(-0.008)` is one quad on (-inf, -0.008] whose nodes reach |x| > 1073 = 2**30/a. The integrand is nan there, so cdf = 1.0 (true 6.2e-16).
  - This is pattern 10, the same family as norminvgauss/geninvgauss (INDEX item 15), although here epsrel = 1e-10 and epsabs = 0 are set and the bulk is much better.
- **Regime:**
  - b/a = 1 - 1e-10, p = -1.5: sf is wrong from x ~ 1e4 on, and cdf near 1e9.
  - a = 1e6: cdf is wrong left of ~ -0.0075 and sf right of ~ +0.0075. cdf(-0.007) is still fine.
- **Reproducer:** `python accuracy.py refs/genhyperbolic_ref.py --at sf 10000 -1.5 1 0.9999999999`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1e4, -1.5, 1, 0.9999999999)` | 1.0 | 2.659826637246996e-07 | 1.7e22 | 1.5 | loss |
  | `cdf(-0.008, 1, 1e6, 0)` | 1.0 | 6.22434640851322e-16 | 7.2e30 | 65 | loss |
  | `sf(1e6, -1.5, 1, 0.9999999999)` | 0.0 | 2.6588289003285563e-10 | inf | - | loss |
  | `sf(1e12, -1.5, 1, 0.9999999999)` | 1.0 | 1.4482242707045098e-64 | 3.1e79 | - | loss |
  | `cdf(1e9, -1.5, 1, 0.9999999999)` | 0.7221111838918952 | 0.9999999999999933 | 1.25e15 | - | severe |
  | `sf(0.008, 1, 1e6, 0)` | 1.0 | 6.22434640851322e-16 | 7.2e30 | - | loss |

  The rows with cond "-" did not finish their cond computation before the time limit. For loss, cond does not matter. For `cdf(1e9)` the rating is severe for any cond below 1e6, and the cond is O(1) there. Not a measured point, but equally wrong: `cdf(-1e10, -1.5, 1, 0.9999999999)` = 1.0 where the true value is ~0.
- **Fix direction:** keep nan in the clip (or raise). Compute the split without underflow (`b/g * kve(p+1, g)/kve(p, g)`, or use the mode). For heavy tails, integrate in log x or use tail asymptotics. Fix F2.

### F5: O(a) terms cancel after separate rounding (pdf, logpdf)

- **Code:** Cython kernel (.pyx not shipped; structure inferred from measurements). The normalization contains `-(log(kve(p, g)) - g)` = `+g`; the x part contains `log(kve(p-1/2, a*s)) - a*s` and `+ b*x`. These are added as separately rounded partial sums.
- **Mechanism:** logpdf is O(1) in the bulk, but the partial sums are O(a), and each is rounded to ~a*eps absolute before they cancel (pattern 5). At x = 0, b = 0 the exact value is `sqrt(a/(2 pi)) kve(p-1/2, a)/kve(p, a)`, and `kve` is accurate to < 1 ulp there (checked for z = 1e3 to 1e6). The measured absolute logpdf error is 0.1 to 0.5 * a * eps for a = 10 to 1e9 and p in {-1, 0.5, 1}: 6.1e-11 at a = 1e6, 4.1e-8 at a = 1e9. In the pdf this becomes a relative error. With b close to a, `-a*s` and `b*x` are each ~1e9 at x = 1e9. At |x| ~ 100 with moderate shapes the cancellation costs ~250 ulp ((-2, 3, -2), x = -100: about -300 + 200).
- **Regime:** large a, anywhere in the bulk; b close to a with large |x| (~a*|x|*eps); mild at |x| ~ 100.
- **Reproducer:** `python accuracy.py refs/genhyperbolic_ref.py --at pdf 0 1 1e9 0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0, 1, 1e9, 0)` | 12615.663118693084 | 12615.662605369927 | 1.8e8 | 0 | severe |
  | `pdf(1e9, -1.5, 1, 0.9999999999)` | 1.1415123441397204e-23 | 1.1415123505324543e-23 | 2.5e7 | 2.6 | severe |
  | `pdf(1e6, -1.5, 1, 0.9999999999)` | 3.989029366866918e-16 | 3.9890293669491503e-16 | 9.28e4 | 2.5 | moderate |
  | `pdf(0.002, 1, 1e6, 0)` | 53.99105425897329 | 53.99105424837408 | 8.8e5 | 4 | moderate |
  | `pdf(-100, -2, 3, -2)` | 1.8752480285365606e-49 | 1.8752480285364554e-49 | 253 | 103 | minor |

  The median logpdf error on (1, 1e6, 0) over the grid is 4.6e4 ulp.
- **Fix direction:** combine the large terms analytically: `g - a*s = -(b^2 + a^2 x^2)/(g + a*s)`, then add `b*x`. For b close to a and large x, use `a*s - b*x = (a^2 s^2 - b^2 x^2)/(a*s + b*x)`.

### F6: cdf/sf inherit the pdf error and the quad tolerance

- **Code:** `_integrate_pdf` (`epsrel = 1e-10`, `epsabs = 0`) integrating the pdf of F5.
- **Mechanism:** the integrand is off by ~1e-10 relative at a = 1e6 (F5), so the cdf is too. In addition, `epsrel = 1e-10` allows ~4.5e5 ulp, which the heavy tail reaches (`sf(1e3)`: 1.04e-10 relative). Exponential tails with moderate shapes are far better (<= 157 ulp down to 1e-219).
- **Regime:** large a (bulk, every x); heavy tails before F4.
- **Reproducer:** `python accuracy.py refs/genhyperbolic_ref.py --at cdf -0.002 1 1e6 0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-0.002, 1, 1e6, 0)` | 0.02275022643412455 | 0.02275022643234614 | 3.5e5 | 4.75 | moderate |
  | `cdf(0.0005, 1, 1e6, 0)` | 0.6914623897911271 | 0.6914623897607929 | 2.0e5 | 0.255 | moderate |
  | `sf(1000, -1.5, 1, 0.9999999999)` | 8.417373345182504e-06 | 8.417373346058919e-06 | 4.7e5 | 1.5 | moderate |

- **Fix direction:** fix F5 first; integrate in a variable scaled to the peak width; only then tighten epsrel.

## RuntimeWarnings

- **pdf/logpdf:** "divide by zero", "invalid value", "overflow" and "underflow encountered in _pdf_single/_logpdf_single (vectorized)" in F1-F3, all on wrong results. "underflow" also appears where the pdf correctly underflows in far tails; it is spurious there.
- **cdf/sf, wrong results:** in F1-F4 the warnings are "Infinite values encountered in scipy.special.kve. Values replaced by NaN...", quad IntegrationWarnings and "scalar divide" warnings from `mean`. The kve message is misleading twice over: kve returns nan rather than inf beyond 2**30 (as in geninvgauss F3), and the announced NaN reaches the user as 1.0.
- **cdf/sf, otherwise accurate results:** "invalid value encountered in scalar divide" (`mean` = 0/0 for g >~ 705) appears even where the result is only moderately off, e.g. cdf(-0.002, 1, 1e6, 0). It is spurious there.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | (0.5, 1.5, -0.5), (1, 2, 0), (0.5, 0.001, 0), (-2, 3, -2): bulk and tails to 1e-219, \|x\| <= 500 | 253 ulp (F5, minor, cond 103); otherwise <= 213 ulp at x = -500 for the example (cond ~500), mostly within conditioning |
| logpdf | same sets, \|x\| below the kve limit | ~1 ulp (median 0.5-0.8) |
| cdf, sf | same four sets: bulk and exponential tails to ~1e-225 (ulp from cached references; no cond pass for (1, 2, 0), (0.5, 0.001, 0), (-2, 3, -2)) | 157 ulp (cdf(-500, 0.5, 1.5, -0.5), 6e-219); <= 6 ulp in the bulk |
| pdf, cdf | tiny a = 0.001 (very wide) | 12 ulp (pdf), 5 ulp (cdf) |
| pdf, logpdf | \|x\| from 1e-300 to 0.3 | no loss |

## Suspected, not confirmed

- a > 1.34e154: `(a+b)*(a-b)` overflows. Masked by F2 (everything already nan from a = 2**30); not measured separately.
- `sc.kv(p, d)` in `mean` overflows for large |p| and small g. This also disables the split, but is masked by F3.
- Large |p| just below the kve overflow: pdf(0, 80, 0.01, 0) is off by 410 ulp, possibly from cancellation of `(p/2) log g^2 - (p - 1/2) log a`. Not measured with cond (time limit).
- cdf/sf condition numbers for (1, 2, 0), (0.5, 0.001, 0) and (-2, 3, -2), and the cond of sf(1e6) and cdf(1e9) in the b ~ a set: not measured (time limit). The harness pass with cond was stopped.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | left-tail asymptotics `logcdf ~ logpdf(x) - log(a + b)` (exponential tails), or quad in log space | cdf underflows: `logcdf(-800, 0.5, 1.5, -0.5)` = -inf vs -802.7048545394568 |
| logsf | log(sf) | `logsf ~ logpdf(x) - log(a - b)` (b < a); power-law form for \|b\| = a | `logsf(600, 0.5, 1.5, -0.5)` = -inf vs -1203.2543005254747 |
| ppf | numerical inversion of cdf | no closed form; accurate where cdf is (ppf(1e-30) and ppf(0.3) exact for the example); inherits F1-F4 | wherever cdf fails |
| isf | ppf(1 - q) | root-find on the implemented `_sf` directly | small q: `isf(1e-30, 0.5, 1.5, -0.5)` = 100.0 vs 33.617041446294536 (8.9e15 ulp) |

## Shared code

- `scipy.special.kve` returns nan for z >= 2**30 at every order (INDEX item 16). The limit is exactly 2**30; 2**30 - 0.5 is finite. This causes F2 and part of F4, and affects geninvgauss F3.
- `kve` overflows for large order and small argument (F3), as in geninvgauss F2. A `log_kv` helper would fix both.
- quad as the cdf (INDEX item 15), with a new twist: `max(0.0, min(1.0, intgrl))` turns nan into 1.0 (Python `min` with nan first compares false), so silent wrong values replace the nan the warning announces. The same clip pattern is worth checking in other quad-based cdfs.

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(0, -1, 1, 1)` | nan | 0.36787944117144232 (= 1/e) | verified (loss) | Docstring pdf with the g -> 0 limit g^p/K_p(g) -> 2^(1+p)/Gamma(-p); pdf(-200) 2.3998643802752551e-179 and pdf(500) 2.0019889877201456e-06 are also nan in SciPy. Mechanism confirmed: `kve(-1, 0)` = inf, `log(0)` = -inf; `min(1.0, nan)` = 1.0 |
| F1 | `cdf(0, -1, 1, 1)`; `cdf(-200, ...)`; `sf(10, ...)` | 1.0; 1.0; 1.0 | 0.18393972058572116; 1.1939699146684509e-179; 0.051012934242147391 | verified (loss) | Reference: mp.quad of the pdf over geometric breakpoints, rescaled by pdf(x) (50 and 80 digits agree to 35 digits). cdf(-2) = 1.0 vs 7.6352125907543435e-4 too |
| F2 | `logpdf(1e9, 0.5, 1.5, -0.5)`; `logpdf(1e300, 1, 2, 0)`; `logpdf(1e20, 0.5, 0.001, 0)`; `logpdf(-1e300, -2, 3, -2)` | nan (all) | -2000000009.7225169; -2.0000000000000001e300; -1.0000000000000003e17; -1.0000000000000001e300 | verified (loss) | `kve(0.5, 2**30 - 0.5)` finite, `kve(0.5, 2**30)` nan confirmed. The second row of the report table has x = 542239620.8675, but 1.01*2**30/2 = 542239621.12. SciPy gives nan at both, and the reference -1084479240.4610758 is for 542239620.8675. This is only a label inconsistency |
| F2 | `pdf(0, 1, 1.1e9, 0)`; `logpdf(...)`; `cdf(...)` | nan; nan; 1.0 | 13231.418566492357; 9.4903494748297862; 0.5 (symmetry, b = 0) | verified (loss) | |
| F3 | `pdf(0, 100, 0.01, 0)`; `pdf(0, 81, ...)`; `pdf(0, -100, ...)`; `logpdf(0, 100, ...)` | nan; 0.0; nan; nan | 2.8315818561319546e-4; 3.1489915331510412e-4; 5.6348479080761014; -8.1695048567169151 | verified (loss) | `kve(81, 0.01)` = inf confirmed |
| F3 | `cdf(3, 100, 0.01, 0)`; `sf(0, 100, 0.01, 0)` | 1.0; 1.0 | 0.50084947391003200 (0.5 + mp.quad of pdf on [0, 3]); 0.5 (symmetry) | verified (severe; loss) | |
| F4 | `sf(1e4, -1.5, 1, 0.9999999999)`; `sf(1e12, ...)` | 1.0; 1.0 | 2.6598266372469961e-07; 1.4482242707045098e-64 | verified (loss) | 1.7e22 ulp. Direct `quad(pdf, 1e4, inf)` returns nan, which the clip turns into 1.0 |
| F4 | `sf(1e6, -1.5, 1, 0.9999999999)` | 0.0 | 2.6588289003285564e-10 | verified (loss) | Raw quad gives -3.989e-16, clipped to 0, as stated |
| F4 | `cdf(1e9, -1.5, 1, 0.9999999999)` | 0.7221111838918952 | 0.99999999999999329 (1 - sf, sf = 6.7053425385302875e-15) | verified (severe) | 1.25e15 ulp |
| F4 | `cdf(-0.008, 1, 1e6, 0)`; `sf(0.008, ...)` | 1.0; 1.0 | 6.2243464085132205e-16 (both) | verified (loss) | `kv(1, 1e6)` = 0 (no split); unsplit quad returns nan; pdf(-1100) is nan, pdf(-1000) = 0, which matches the 2**30/a limit |
| F5 | `pdf(0, 1, 1e9, 0)` | 12615.663118693084 | 12615.662605369927 | verified (severe, 1.83e8 ulp) | A float64 replica with separately rounded `-(log kve(p,g) - g)` and `log kve(.,a s) - a s` terms reproduces SciPy bit for bit. The combined form `sqrt(a/2pi) kve(0.5,a)/kve(1,a)` gives 12615.662605369927. This supports the inferred mechanism (.pyx not inspected) |
| F5 | `pdf(1e9, -1.5, 1, 0.9999999999)`; `pdf(1e6, ...)`; `pdf(0.002, 1, 1e6, 0)`; `pdf(-100, -2, 3, -2)` | 1.1415123441397204e-23; 3.989029366866918e-16; 53.99105425897329; 1.8752480285365606e-49 | 1.1415123505324543e-23; 3.9890293669491505e-16; 53.991054248374076; 1.8752480285364554e-49 | verified (severe 2.52e7; moderate 9.28e4; moderate 8.84e5; minor 253 ulp) | The replica matches SciPy exactly at (0.002, 1, 1e6, 0). At (1e9, b ~ a) it shows an error of the same size with a different term order |
| F6 | `cdf(-0.002, 1, 1e6, 0)`; `cdf(0.0005, ...)`; `sf(1000, -1.5, 1, 0.9999999999)` | 0.02275022643412455; 0.6914623897911271; 8.417373345182504e-06 | 0.022750226432346138; 0.69146238976079284; 8.4173733460589185e-06 | verified (moderate: 3.52e5, 1.98e5, 4.69e5 ulp) | |

Verified by dist-accuracy-verifier: 6 of 6 verified.

pdf/logpdf references: docstring formula with mp.besselk, 100 vs 200 digits (agree to at least 35 digits). cdf/sf: mp.quad of the pdf (different route from the analyst's GIG mixture), 50 vs 80 digits (agree to 35 digits). All agree with the analyst's references. Scripts: `scratch/genhyperbolic/verify/`.
