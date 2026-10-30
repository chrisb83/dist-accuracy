# norminvgauss: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `norminvgauss_gen`, lines 5516-5653; module-level helper `_norminvgauss_quadrature`, lines 5500-5513
- **Implemented:** pdf, cdf, sf, isf (cdf and sf by `integrate.quad` of `_pdf` through `_norminvgauss_quadrature`; isf by `optimize.brentq` on `_sf`)
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf), ppf = numerical inversion of cdf
- **Related:** none share code (`invgauss` is only used by `_rvs`); helpers: `_norminvgauss_quadrature`; special functions: `sc.k1e` (pdf), `integrate.quad` (cdf, sf), `optimize.brentq` with `self.xtol = 1e-14` (isf)
- **Reference:** `refs/norminvgauss_ref.py`, 100 digits. pdf uses its own K_1 series/asymptotic expansion (`mp.besselk(1, z)` takes 1-3 s per call). cdf/sf come from the normal variance-mean mixture X | V ~ N(bV, V), V ~ IG, integrated with the trapezoidal rule in log V at a fixed 60 digits, step halved from w/20 until two successive results agree to 1e-50 (after verification: a fixed w/20 step was off by up to 2.4e-4 relative for (10, 9.99999)). isf uses Newton in asinh(x). Self-checks: PASS
- **Parameter sets:** (1.25, 0.5), (1, 0), (1, 0.999), (2, -1.5), (1e4, 0), (0.001, 0), (10, 9.99999), (1e6, 5e5); extreme shapes (a = 1e7, 1e8, 1.3e154 to 1e160, 1e-170) only with `--at`
- **Worst finding:** loss in cdf, sf, isf and pdf

## Summary

pdf is accurate to a few ulp for moderate shapes. It loses up to ~a*|x| ulp
(1.6e8 ulp at most) when `b*x - a*sq + gamma` cancels: large a in the bulk, or
b close to a far in the right tail. It overflows to inf for a > 1.34e154. cdf
and sf are plain `integrate.quad` calls with default tolerances. When the
density is concentrated far from the finite endpoint of the integral (large a,
or strong skew with b close to a), quad misses the mass entirely, so sf = 0
where the true value is 1 and cdf is off by orders of magnitude. Even for
benign shapes, the absolute tolerance 1.49e-8 limits tail values to about 5
correct digits (5e10 ulp), and bulk values to about 1e-10 (5e5 ulp). isf
inherits all of this. It also returns roots that are off by hundreds where sf
jumps, loses relative accuracy near the median (absolute xtol = 1e-14), and
raises exceptions for a < ~2e-162 or a > 1.34e154.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | narrow or distant peak: a >~ 1e7 (bulk), a = 1e6 with b = a/2, b/a = 1 - 1e-6 | quad on a half-infinite interval misses the peak (pattern 10) | sf 0.0 for 1.0; cdf 9.6e-223 for 3.4e-201 | loss |
| F2 | isf | same shapes as F1 | brentq converges to a jump of the broken sf (from F1) | -707.1 for 63.0 | loss |
| F3 | pdf, cdf, sf, isf | a > 1.34e154 | `(a + b)*(a - b)` overflows, gamma = inf (pattern 3) | pdf inf; cdf 1.0 for 0.159; isf ValueError | loss |
| F4 | isf | a < ~2.2e-162 | gamma underflows to 0, `self.mean` = nan/inf is the bracket start | ValueError / OverflowError | loss |
| F5 | isf | results close to 0 (q near sf(0)), all shapes | brentq `xtol = 1e-14` is absolute | 0.0 for 1.07e-16; 2.3e10 ulp | loss |
| F6 | cdf, sf | tails (value < 1.5e-8) for all shapes; bulk | quad default epsabs = epsrel = 1.49e-8 (pattern 10) | 5.1e10 ulp (tails), 4.8e5 ulp (bulk) | severe |
| F7 | isf | small q, all shapes | inherits F6 tail errors of sf | 7.7e6 ulp (moderate shapes) | severe |
| F8 | pdf | large a near the mode; b close to a, large x | `b*x - a*sq + gamma` cancels (pattern 5) | 1.6e8 ulp | severe |

### F1: cdf and sf miss the probability mass of a narrow or distant peak

- **Code:** `_norminvgauss_quadrature`, lines 5506-5513: `integrate.quad(norminvgauss._pdf, x, y, args=(a, b))` with one infinite limit and default settings; `_cdf` line 5592, `_sf` line 5595.
- **Mechanism:** QUADPACK `qagi` maps the half-infinite interval onto (0, 1] and starts with a 15-point Kronrod rule. If the density is concentrated in a region that is narrow relative to its distance from the finite limit, no node lands in it. The error estimate is then tiny, quad returns about 0, and the result is clipped to [0, 1]. The comment at lines 5501-5505 explains the `1 - quad` branches for |x| > |mean|. Those branches do not help between -|mean| and |mean|, or on the side of the mean where the peak is narrow (pattern 10: quadrature used outside its reliable range). With (1e6, 5e5) the mean is 0.577 and the standard deviation is 1.6e-3, so `sf(x)` for every x in (-0.577, 0.54) integrates over [x, inf) and returns 0.0.
- **Regime:** in the bulk for b = 0 already at a = 1e7 (sd 3e-4) and a = 1e8 (sd 1e-4): `cdf(-sd)` = 0 instead of 0.159. For a = 1e6, b = 5e5, and for b/a = 1 - 1e-6 (mean 707, sd 5946, heavy right tail), all x between -|mean| and the peak are affected. The far right tail of (10, 9.99999) is also affected: at x >~ 1.4733e5, quad returns -5.6e-9, which is clipped to 0, while the true sf is 2.9e-4.
- **Reproducer:** `python accuracy.py refs/norminvgauss_ref.py --at sf -1e-10 1e6 5e5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(-1e-10, 1e+06, 500000)` | 0.0 | 1.0 | inf | 0 | loss |
  | `sf(-707.10625086973585, 10, 9.99999)` | 0.0 | 1.0 | inf | 0 | loss |
  | `sf(179088.24039927218, 10, 9.99999)` | 0.0 | 0.00017082512756882697 | inf | 2.95 | loss |
  | `cdf(0.54012607482554176, 1e+06, 500000)` | 9.579618954078395e-223 | 3.380440201163224e-201 | 4.5e+15 | 1.34e+04 | loss |
  | `cdf(0.57114623679561172, 1e+06, 500000)` | 2.587712866247061e-10 | 2.77216473079964e-07 | 4.5e+15 | 2.4e+03 | severe |
  | `cdf(-0.29999999999999999, 10000, 0)` | 2.690619371530036e-195 | 7.784620646337754e-194 | 4.3e+15 | 863 | severe |
  | `cdf(179088.24039927218, 10, 9.99999)` | 1.000000002748245 | 0.9998291748724312 | 7.7e+11 | 0.000504 | severe |
  | `cdf(-0.0001, 1e8, 0)` | 0.0 | 0.15865525332653024 | inf | 1.53 | loss |
  | `cdf(-0.001, 1e7, 0)` | 2.763467851903938e-21 | 0.0007827018727833291 | 4.5e+15 | 10.9 | loss |
  | `cdf(-1e-77, 1e+154, 0)` | 0.0 | 0.15865525393145707 | inf | - | loss |

  Note that `cdf(179088, 10, 9.99999)` returns a value above 1 (1.000000002748245):
  the `1 - quad` branches (lines 5508, 5510) return before the `np.clip` at line 5513,
  so `1 - (negative quad result)` is not clipped.
- **Fix direction:** give quad the scale of the problem: split at the mode or mean, integrate over finite pieces of a few standard deviations (`points=`, or substitution x = mean + sd*t), and use `epsabs=0`. Better, compute cdf/sf from the normal mixture integral over V or in u = asinh(x), where the integrand is smooth and decays double-exponentially.

### F2: isf returns the location of a jump in the broken sf

- **Code:** `_isf`, lines 5600-5628: `eq = self._sf(x, a, b) - q`, the bracket search from `xm = self.mean(a, b)`, then `optimize.brentq(eq, left, right, xtol=self.xtol)`.
- **Mechanism:** brentq needs only a sign change. Because of F1, `_sf` jumps from 1 to 0 at x = -|mean| (the switch between `1 - quad(-inf, x)` and `quad(x, inf)` at lines 5509-5512), or drops to 0 in the far tail. brentq converges to that discontinuity, whatever q is.
- **Regime:** shapes where F1 occurs. Where brentq lands on the jump depends on how the bracket search meets it, so the affected q come in bands, not a single threshold. For (10, 9.99999) on the q grid (spacing ~0.07 in the bulk, so the band edges are known only to one grid step): isf returns -mean = -707.106 (loss) at q = 0.22-0.29, q = 0.57-0.71 and every q >= 0.99 (up to 1 - 3e-16). It is not caught by the jump at q = 0.36-0.50 and 0.78-0.92 (340-1.9e3 ulp, minor/moderate) or at q = 0.01-0.15 (1.2e3-2.6e4 ulp, moderate; sf error, F6). Every q < 1e-3 returns ~147334, the point where the tail quad breaks down (severe, F7). For (1e6, 5e5) every q > ~0.5 returns -0.5773 = -mean. For (1e8, 0), `isf(0.3)` returns 0.0.
- **Reproducer:** `python accuracy.py refs/norminvgauss_ref.py --at isf 0.29 10 9.99999`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.29, 10, 9.99999)` | -707.1062508697356 | 67.44590318634648 | inf | 2.01 | loss |
  | `isf(0.99, 10, 9.99999)` | -707.1062508697362 | 1.360732639133925 | inf | 32.4 | loss |
  | `isf(0.56999999999999995, 1e+06, 500000)` | -0.5773502691896271 | 0.5771311001290594 | inf | 0.00312 | loss |
  | `isf(1e-300, 10, 9.99999)` | 147334.10625086943 | 67549018.8430328 | 4.5e+15 | 0.00148 | severe |
  | `isf(1e-10, 10, 9.99999)` | 147334.22332359312 | 1351130.1118913342 | 4.0e+15 | 0.067 | severe |
  | `isf(0.3, 1e8, 0)` | 0.0 | 5.244005109217989e-05 | inf | 1.65 | loss |

  In the grid run, isf has 25 loss points for (10, 9.99999) (8 bulk, 17 high tail; re-measured with the fixed reference) and 22 for (1e6, 5e5).
  Away from the jumps, bulk isf for (10, 9.99999) is only moderate (e.g. `isf(0.01)` 2.6e4 ulp, `isf(0.1)` 2.9e3 ulp, `isf(0.15)` 1.2e3 ulp, `isf(0.36)` 1.9e3 ulp; sf error, F6).
- **Fix direction:** fix sf first (F1). Then a solver that checks the residual (or a Newton step using pdf) would catch remaining discontinuities.

### F3: gamma overflows for a > 1.34e154

- **Code:** `_pdf`, line 5586: `gamma = np.sqrt((a + b) * (a - b))`; the same expression is in `_norminvgauss_quadrature` line 5506 and `_stats` line 5648.
- **Mechanism:** the product (a + b)(a - b) exceeds 1.8e308 although gamma <= a is representable (pattern 3). `exp(b*x - a*sq + gamma)` is then inf and the pdf is inf. cdf/sf integrate an infinite pdf, and isf's brentq meets nan.
- **Regime:** (a + |b|)(a - |b|) > 1.8e308, for b = 0 that is a > 1.3408e154. At a = 1.3e154 the pdf is exact (0.64 ulp).
- **Reproducer:** `python accuracy.py refs/norminvgauss_ref.py --at pdf 0 1.4e154 0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0, 1.4e+154, 0)` | inf | 4.720348719413148e+76 | inf | 0 | loss |
  | `pdf(1e-80, 1e+160, 0)` | inf | 2.4197072451914337e+79 | inf | 1 | loss |
  | `cdf(-1e-80, 1e+160, 0)` | 1.0 | 0.15865525393145705 | 2.4e+16 | 1.53 | loss |
  | `sf(1e-80, 1e+160, 0)` | 1.0 | 0.15865525393145705 | 2.4e+16 | 1.53 | loss |

  `isf(0.3, 1e160, 0)` raises `ValueError: The function value at x=2.557336412418861e+148 is NaN`.
- **Fix direction:** `gamma = a * sqrt((1 + b/a) * (1 - b/a))` (or `np.hypot`-style scaling); the exponent `-a*sq + gamma` should be formed without the large terms (see F8).

### F4: isf raises for tiny a because the start point mean is nan or inf

- **Code:** `_isf`, line 5607: `xm = self.mean(a, b)`, i.e. `_stats` line 5648-5649: `gamma = np.sqrt((a + b) * (a - b))`, `mean = b / gamma`; bracket loops lines 5612-5626.
- **Mechanism:** (a + b)(a - b) underflows to 0 for a < ~2.2e-162 (pattern 4), so gamma = 0. With b = 0 the mean is 0/0 = nan: no branch of the bracket search triggers, and brentq raises on a nan function value. With b != 0 the mean is inf, `eq(inf) < 0`, and the `while eq(left, a, b, q) < 0` loop doubles the integer `delta` until `xm - delta` raises OverflowError. The distribution itself is harmless here: it is Cauchy-like, and `cdf(1, 1e-170, 0)` = 0.7499999999999988 is fine.
- **Regime:** a < ~2.2e-162, any b, any q.
- **Reproducer:** `python scratch/norminvgauss/tiny_a.py` (`--at` aborts on the exception)

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.3, 1e-170, 0)` | ValueError (nan in brentq) | 0.72654252800536094 | - | - | loss |
  | `isf(0.3, 1e-170, 5e-171)` | OverflowError (int too large to convert to float) | 0.72654252800536094 | - | - | loss |

- **Fix direction:** start the bracket from 0 or the median of a simple approximation instead of the mean, and compute gamma as in F3.

### F5: isf loses relative accuracy near zero (absolute xtol)

- **Code:** `_isf`, lines 5627-5628: `optimize.brentq(eq, left, right, args=(a, b, q), xtol=self.xtol)`, where `self.xtol` = 1e-14 (rv_continuous default).
- **Mechanism:** brentq stops when the bracket is below `xtol + 4*eps*|x|`. For roots |x| << 1 the absolute 1e-14 dominates, and the relative error is about 1e-14/|x|. For q within 1e-16 of sf(0), the root is returned as exactly 0.0.
- **Regime:** isf results with |x| < ~1e-6 (q near sf(0)), all shapes; worst for large a, where the scale is ~a^-1/2.
- **Reproducer:** `python accuracy.py refs/norminvgauss_ref.py --at isf 0.49999999999999994 1 0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.49999999999999994, 1, 0)` | 0.0 | 1.0658744816094324e-16 | inf | 9.01e+15 | loss |
  | `isf(0.49999999999999994, 10000, 0)` | 0.0 | 1.3914060362399358e-18 | inf | 9.01e+15 | loss |
  | `isf(0.49999999949999741, 1, 0)` | 9.600544086918054e-10 | 9.600593660978633e-10 | 2.3e+10 | 1e+09 | severe |
  | `isf(0.50000000050000404, 10000, 0)` | -1.2537663327983836e-11 | -1.2532772630854958e-11 | 1.8e+12 | 1e+09 | severe |
  | `isf(0.5000001, 1, 0)` | -1.9201086948534174e-07 | -1.9201087662693555e-07 | 1.7e+08 | 5.0e+06 | severe |

- **Fix direction:** `xtol` relative to the scale of the result (e.g. `xtol=1e-14*scale` with a tiny absolute floor, or `rtol` only), plus an accurate sf near the median.

### F6: cdf and sf limited by the default quad tolerances

- **Code:** `_norminvgauss_quadrature`, lines 5508, 5510, 5512: `integrate.quad(...)` with defaults `epsabs=1.49e-8`, `epsrel=1.49e-8`.
- **Mechanism:** QUADPACK stops as soon as the error estimate is below max(epsabs, epsrel*|result|). For tail values below 1.5e-8 the absolute tolerance is met immediately, so only the first Kronrod estimates are returned. That gives relative errors around 1e-5 to 1e-6 even for benign shapes (pattern 10). In the bulk, epsrel limits the result to ~1e-10 relative. The complements `1 - quad(...)` at lines 5508/5510 carry the tail's absolute error into the values near 1.
- **Regime:** both tails (cdf for x < -|mean|, sf for x > |mean|, and their direct quad branches) for every parameter set; bulk errors up to ~5e5 ulp.
- **Reproducer:** `python accuracy.py refs/norminvgauss_ref.py --at cdf -30 1 0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-600, 1, 0)` | 1.9503365952741109e-265 | 1.9503586841127083e-265 | 5.1e+10 | 601 | severe |
  | `cdf(-30, 1, 0)` | 5.860504081189471e-16 | 5.860566997067416e-16 | 4.8e+10 | 31.4 | severe |
  | `sf(10, 1, 0)` | 1.3409748707993554e-06 | 1.3409847489184868e-06 | 3.3e+10 | 11.4 | severe |
  | `sf(800, 1.25, 0.5)` | 2.184075839238858e-265 | 2.1840732813829774e-265 | 5.3e+09 | 601 | severe |
  | `cdf(-6, 1.25, 0.5)` | 1.2357878404604198e-06 | 1.2357882365041296e-06 | 1.4e+09 | 11.8 | severe |
  | `sf(9, 1.25, 0.5)` | 6.488622741867742e-05 | 6.4886227458238e-05 | 2.7e+06 | 8.05 | severe |
  | `cdf(-4, 1, 0)` | 0.0017556126536584164 | 0.0017556126534699257 | 4.8e+05 | 5.19 | moderate |
  | `cdf(10, 1, 0)` | 0.9999986590251292 | 0.9999986590152511 | 4.4e+04 | 1.52e-05 | moderate |

  Grid counts (all parameter sets, including F1 points): cdf 1 loss / 50 severe / 35 moderate / 404 minor; sf 74 / 39 / 51 / 398.
- **Fix direction:** `epsabs=0` and a tight `epsrel` (or a scale-aware quadrature as in F1); for the tails a dedicated asymptotic or mixture formula.

### F7: isf in the tails inherits the sf tail error

- **Code:** `_isf`, line 5602: `self._sf(x, a, b) - q`.
- **Mechanism:** the root of an sf with relative error delta is off by delta*sf/(x*pdf) relative. In the exponential tails sf/(x pdf) ~ 1/((a - |b|) x), so the 1e-6 relative errors of F6 become relative errors ~1e-9 in x, while the condition number is ~1e-3.
- **Regime:** q <~ 1e-3 for all shapes (173 of 192 low-tail points severe, the other 19 moderate, all for (2, -1.5); re-measured with the fixed reference); also q near 1 where sf = 1 - quad, but there the conditioning (~1/(1 - q)) covers most of it.
- **Reproducer:** `python accuracy.py refs/norminvgauss_ref.py --at isf 1e-300 1.25 0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1e-300, 1.25, 0.5)` | 908.2422224785639 | 908.2422209188078 | 7.7e+06 | 0.00146 | severe |
  | `isf(1e-10, 1.25, 0.5)` | 24.982928010737744 | 24.982927090751097 | 1.7e+08 | 0.0496 | severe |
  | `isf(1e-300, 1e+06, 500000)` | 0.6214306568613908 | 0.6237847906292711 | 1.7e+13 | 5.48e-05 | severe |

- **Fix direction:** an accurate tail sf (F6); then the brentq root is good to about xtol.

### F8: pdf cancels in the exponent b*x - a*sq + gamma

- **Code:** `_pdf`, line 5589: `fac1 * sc.k1e(a * sq) * np.exp(b*x - a*sq + gamma) / sq`.
- **Mechanism:** the exponent is a sum of terms of size ~a*max(1, |x|) whose result is O(1) to O(100) (pattern 5). Each term has a rounding error of ~a*|x|*eps, which `exp` turns into a relative error of the same size, i.e. ~a*max(1, |x|) ulp. Two regimes: (a) large a near the mode, where `-a*sq + gamma` ~ -a*x**2/2 (and with b != 0, also b*x); (b) b close to a and large x > 0, where `b*x - a*sq` ~ (b - a)*x.
- **Regime:** error ~ a*max(1,|x|) ulp versus cond ~ |(b - a)x| + a x^2 (bulk); visible for a >= 1e4, or for (a - b)/a <= 1e-3 with x >~ 1e4.
- **Reproducer:** `python accuracy.py refs/norminvgauss_ref.py --at pdf 6e7 10 9.99999`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(60000000.002271466, 10, 9.99999)` | 7.2968571921800745e-273 | 7.296857453961407e-273 | 1.6e+08 | 601 | severe |
  | `pdf(10000000.000378577, 10, 9.99999)` | 1.5052322460026693e-54 | 1.5052322335669274e-54 | 3.7e+07 | 101 | severe |
  | `pdf(179088.24039927218, 10, 9.99999)` | 2.8162352606138153e-09 | 2.8162352600363275e-09 | 9.2e+05 | 3.29 | moderate |
  | `pdf(0.5401260748255418, 1e+06, 500000)` | 8.380794622365078e-197 | 8.380794623933074e-197 | 8.4e+05 | 1.34e+04 | moderate |
  | `pdf(0.5810726886260341, 1e+06, 500000)` | 3.5890381929791744 | 3.5890381925419423 | 5.5e+05 | 1.4e+03 | moderate |
  | `pdf(0.001, 1e+06, 0)` | 241.9706640055733 | 241.97066402645464 | 3.9e+05 | 1 | moderate |
  | `pdf(599999.99999999942, 1, 0.999)` | 2.3790925736351155e-270 | 2.3790925735579303e-270 | 1.5e+05 | 601 | moderate |
  | `pdf(0.01, 10000, 0)` | 24.196467517588285 | 24.19646751754857 | 7.39e+03 | 1 | moderate |

  cond (shapes) is large in regime (b) (1.2e9 at the first point): the pdf there is very sensitive to b, but the shapes are exact inputs.
- **Fix direction:** write the exponent without the large terms: `b*x - a*x**2/(sq + 1) - b**2/(a + gamma)` (exact rewrite of `b*x - a*(sq - 1) - (a - gamma)`); for b close to a and x > 0 additionally `(b - a)*x + a*(x - sq)` with `x - sq = -1/(x + sq)`.

## RuntimeWarnings

- `underflow encountered in exp` (and `in scalar divide/multiply`) in pdf, cdf, sf
  and isf for all 8 parameter sets. quad samples `_pdf` far in the tails, so the
  warning also appears on accurate results (e.g. `cdf(0.3, 1.25, 0.5)`, 6.1 ulp, ok):
  spurious there.
- `overflow encountered in multiply` at line 5586 for a > 1.34e154 (F3, result inf).
- `invalid value` / `divide by zero` in `_stats` (lines 5649-5652) and in line 5506 for
  a < ~2.2e-162 (F4), followed by the exceptions.
- `IntegrationWarning: roundoff error is detected` from line 5512 for a = 1e-170
  (the cdf value there is nevertheless accurate, 0.7499999999999988 vs 0.75).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid for (1.25, 0.5), (1, 0), (2, -1.5), (0.001, 0) | 7.3 ulp in the bulk; tails 753 ulp at x = -1200 for (2, -1.5) (cond 601, minor) and 128 ulp at x = 133 for (1.25, 0.5) (cond 101, minor), the small form of F8 |
| pdf | at the mode for a = 2e9 (where `kve` fails, see geninvgauss) and a = 1.3e154 | 0 ulp (`pdf(0, 2e9, 0)`), 0.64 ulp (`pdf(0, 1.3e154, 0)`); `sc.k1e` also agrees with the asymptotic form at z = 1e20, 1e300 and with 1/z at 1e-300 (hand check) |
| cdf, sf | bulk of (1.25, 0.5) | 410 ulp (minor, epsrel of quad, F6) |
| isf | moderate q (0.01-0.99) for (1.25, 0.5), (2, -1.5) | 550 ulp (minor; sf error, F6/F7) |

## Suspected, not confirmed

- `a * sq` overflows for |x| > 1.8e308/a, so `k1e(inf) = 0` and the exponent is
  -inf. Wherever that happens the true pdf underflows anyway
  ((a - |b|)|x| >= eps*a*|x| > 700 for any representable a - |b|), so it was not
  measurable.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `log(a/pi) + log(k1e(a*sq)) + (b*x - a*sq + gamma) - log(sq)`, with the exponent rewritten as in F8 | pdf underflows for (a - \|b\|)\|x\| >~ 745: `logpdf(1300, 1.25, 0.5)` = -inf, true -985.4171526637348 |
| logcdf, logsf | log(cdf), log(sf) | log of the tail integral computed with a scaled integrand (e.g. the normal-mixture integral with `log_ndtr`, or the leading asymptotic term K_1 tail / (a -+ b)) | cdf/sf underflow: `logcdf(-600, 1.25, 0.5)` = -inf, true -1059.8185364122198; `logsf(1000, 1.25, 0.5)` = -inf, true -759.737994469763 |
| ppf | numerical inversion of cdf | `ppf(q, a, b) = -isf(q, a, -b)` (X(a, -b) has the law of -X(a, b)), once isf is fixed | wherever cdf fails (F1, F6): `ppf(0.3, 1e8, 0)` = -2.031447610522748e-15, true -5.2440051092179889e-05 |

## Shared code

none. `_norminvgauss_quadrature` (F1, F6) is used only by norminvgauss.
`sc.k1e` is accurate over the whole range tested (1e-300 to 1e300), so it does not
have the `kve`/`kv` nan limit at z >= 2**30 noted in geninvgauss.md. The isf
problem F5 comes from the generic `rv_continuous.xtol = 1e-14`, which is
absolute; other distributions that pass `self.xtol` to `brentq` for roots near 0
would be affected the same way.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable). Excluded as unrelated: the
closed norminvgauss parameterisation report (apparently #14568; not about
accuracy).

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `sf(-1e-10, 1e6, 5e5)` | 0.0 | 1.0 | verified | the direct `quad(x, inf)` branch (line 5512) returns 0 |
| F1 | `sf(179088.24039927218, 10, 9.99999)` | 0.0 | 0.00017082512756882696 | verified | |
| F1 | `cdf(0.54012607482554176, 1e6, 5e5)` | 9.579618954078395e-223 | 3.380440201163224e-201 | verified | |
| F1 | `cdf(179088.24039927218, 10, 9.99999)` | 1.000000002748245 | 0.99982917487243117 (as 1 - sf) | verified | 7.7e11 ulp; value > 1 because line 5508 returns before the clip. The verifier's direct mixture cdf was unstable at this one point (100 vs 200 digits differ by 1e-6), so 1 - sf was used |
| F1 | `cdf(-0.0001, 1e8, 0)` | 0.0 | 0.15865525332653023 | verified | |
| F2 | `isf(0.29, 10, 9.99999)` | -707.1062508697356 | 67.445903186346489 | verified (reference disagreement on the value) | loss either way. The analyst's value 67.44589577432565 is off by 1.1e-7 relative: the analyst's mixture trapezoid gives cdf(67.445903186346489) = 0.71000001589, but mp.quad of the mixture (100/200 digits) and a tight (epsrel 1e-13) quad of the pdf in asinh(x) both give 0.7100000000000. So SciPy's generic `ppf(0.71) = 67.44590318634658` in the Opportunities table is accurate, not wrong by 1.1e-7. Analyst bulk values for (10, 9.99999) may carry ~1e-8 absolute error, which is far below every error reported for that set |
| F2 | `isf(0.57, 1e6, 5e5)` | -0.5773502691896271 | 0.57713110012905941 | verified | sf jumps from 1.0 to 0.0 at x = -mean, as stated (checked at -mean*(1 -+ 1e-9)) |
| F2 | `isf(0.3, 1e8, 0)` | 0.0 | 5.2440051092179889e-05 | verified | |
| F3 | `pdf(0, 1.4e154, 0)` | inf | 4.720348719413148e+76 | verified | `mp.besselk` route; `pdf(0, 1.3e154, 0)` is finite, so the threshold is as stated (line 5586) |
| F3 | `cdf(-1e-80, 1e160, 0)` | 1.0 | 0.15865525393145706 | verified | `sf(1e-80, 1e160, 0)` = 1.0 also; `isf(0.3, 1e160, 0)` raises ValueError (NaN), reproduced |
| F4 | `isf(0.3, 1e-170, 0)` | ValueError (NaN in brentq) | 0.72654252800536094 | verified | `mean(1e-170, 0)` = nan, `mean(1e-170, 5e-171)` = inf, as stated; agrees with the Cauchy limit tan(0.2*pi) |
| F4 | `isf(0.3, 1e-170, 5e-171)` | OverflowError | - | verified | exception reproduced |
| F5 | `isf(0.49999999999999994, 1, 0)` | 0.0 | 1.0658744816094324e-16 | verified | also equals (1/2 - q)/pdf(0) |
| F5 | `isf(0.49999999949999741, 1, 0)` | 9.600544086918054e-10 | 9.6005936609786335e-10 | verified | 2.3e10 ulp |
| F6 | `cdf(-30, 1, 0)` | 5.860504081189471e-16 | 5.8605669970674156e-16 | verified | 4.8e10 ulp |
| F6 | `sf(10, 1, 0)` | 1.3409748707993554e-06 | 1.3409847489184869e-06 | verified | 3.3e10 ulp |
| F6 | `cdf(-600, 1, 0)` | 1.9503365952741109e-265 | 1.9503586841127082e-265 | verified | 5.1e10 ulp |
| F7 | `isf(1e-10, 1.25, 0.5)` | 24.982928010737744 | 24.982927090751096 | verified | 1.7e8 ulp; the 1e-300 point was not recomputed (reference too slow) |
| F8 | `pdf(60000000.002271466, 10, 9.99999)` | 7.2968571921800745e-273 | 7.2968574539614068e-273 | verified | 1.6e8 ulp |
| F8 | `pdf(0.001, 1e6, 0)` | 241.9706640055733 | 241.97066402645465 | verified | 3.9e5 ulp |

Verifier references: cdf/sf from the normal variance-mean mixture over V, integrated with `mp.quad` in log V with breakpoints around the integrand's maximum; pdf from `mp.besselk`; isf by secant on log sf. All at 100 and 200 digits, agreeing to more than 30 digits except where noted. Scripts are in `scratch/norminvgauss/verify/`. Mechanisms and cited lines (5506-5513, 5586, 5589, 5607, 5628, 5648) match the source.

Verified by dist-accuracy-verifier: 8 of 8 verified.

Corrected after verification: the verifier was right about F2. The reference's fixed-step (w/20) trapezoid was not converged for (10, 9.99999), so the step is now halved until it converges (agrees with mp.quad to 25 digits). This changed the reference values `isf(0.29)` -> 67.44590318634648, `isf(0.1)`, `isf(0.15)`, `isf(0.36)`, `isf(0.43)`, `cdf/sf(707.106)` (only for this parameter set, rechecked over all grid points). It removed the rows `cdf(707.106)` (F1, now 10.9 ulp, minor) and `isf(0.15)` (F2, now 1.2e3 ulp, moderate), updated the F6 grid counts, and replaced the ppf example in Opportunities (`ppf(0.71, 10, 9.99999)` is accurate).
Re-measured afterwards with the fixed reference: the full isf grid for (10, 9.99999) (`accuracy.py --methods isf --params "10,9.99999"`, 60 points, self-checks PASS; `scratch/norminvgauss/remeasure/isf_skew.txt`). It still gives 25 loss points (8 bulk, 17 high tail) and 24 of 24 low-tail points severe. Also the isf low-tail references (q < 1e-3) for the other 7 sets (168 points, `scratch/norminvgauss/remeasure/lowtail_cmp.py`): all are identical to the old grid values, so their severities stand. The F2 count (25) and the F7 count (173 of 192) are therefore unchanged. The F2 regime text now lists the measured q bands instead of "every q > ~0.2", and no severity changed.
