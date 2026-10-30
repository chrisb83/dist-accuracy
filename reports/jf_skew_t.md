# jf_skew_t: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `jf_skew_t_gen`, lines 5729-5826
- **Implemented:** pdf, cdf, sf, ppf (ppf calls `beta.ppf`)
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf), isf = ppf(1 - q)
- **Related:** none (ppf uses `beta.ppf`, i.e. Boost `ibeta_inv`); helpers: none; special functions: sc.beta, sc.betainc, sc.betaincc, beta.ppf (sc.comb, special.beta only in `_munp`)
- **Reference:** `refs/jf_skew_t_ref.py`, 400 digits; self-checks: PASS (NOTE/WARN lines for (600, 600) and (0.001, 0.001) are SciPy failures, F1/F2/F3/F4 below)
- **Parameter sets:** (8, 4), (0.5, 0.5), (2, 0.5), (0.5, 2), (0.001, 0.001), (1000, 5), (600, 600)
- **Worst finding:** loss in pdf, cdf, sf, ppf

## Summary

With moderate shapes all four methods are accurate in the bulk: pdf within 10 ulp, cdf within 3, sf within 22, ppf within 13. Every method fails completely in at least one tail:
- **pdf, cdf, sf:** they form `1 +- x/sqrt(a+b+x**2)` by subtraction, so the tail factor cancels to 0 once |x| exceeds about `sqrt((a+b)/eps)` (about 3e8 for (8, 4), 3e6 for (0.001, 0.001)). `x**2` overflows for |x| > 1.34e154, which sends cdf/sf to the median value `I_{1/2}(a, b)`.
- **pdf normalisation:** the constant `2**(a+b-1) * beta(a, b)` overflows for a + b > 1025 (pdf 0 or nan everywhere). With integer shapes it wraps around in int64 for a + b >= 64 (negative or infinite pdf).
- **ppf:** it goes through the beta variable `u = beta.ppf(q, a, b)`. It cancels in `1 - u` for q near 1 (inf for 1 - q < ~1e-8 when b = 0.5) and in `2u - 1` near the median for large shapes. It saturates when u falls below DBL_MIN (small a, tiny q), and it inherits the Boost `ibeta_inv` failures documented in reports/beta.md.
- **Shared `scipy.special` problems:** `sc.beta(1000, 5)` is 2.6e3 ulp off, and `sc.betainc` loses up to 8.7e7 ulp for results near DBL_MIN.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf, cdf, sf | abs(x) > 1.34e154, any shapes | `x ** 2` overflows, t = x/inf = 0 (pattern 3) | cdf(1e200, 8, 4) = 0.113 for 1.0; cdf(-1e300, 0.5, 0.5) = 0.5 for 3.2e-301 | loss |
| F2 | pdf | abs(x) > ~sqrt((a+b)/eps); error ~ (b+1/2) x^2/(a+b) ulp before that | `1 - x/sqrt(...)` (x > 0) and `1 + x/sqrt(...)` (x < 0) cancel (pattern 1) | 0.0 for 2.1e-76 | loss |
| F3 | cdf (left tail), sf (right tail) | same as F2 | `y = (1 + x/sqrt(...))/2` cancels for x << 0; `betaincc(a, b, y)` with y rounded near 1 for x >> 0 (patterns 1, 10) | 0.0 for 0.48 ((0.001, 0.001), x = -4.5e6) | loss |
| F4 | pdf | a + b > 1025 (float shapes) | `2 ** (a + b - 1)` overflows, `sc.beta` underflows: inf -> pdf 0, inf*0 -> nan (pattern 3) | nan for 0.3989; 0.0 for 0.3988 | loss |
| F5 | pdf | integer shapes with a + b >= 64 | `2 ** (a + b - 1)` evaluated in int64 wraps (2**63 < 0, 2**64 = 0) | -0.350 for 0.350; inf | loss |
| F6 | ppf | q near 1, worst for b <= 1 | `1 - d1` with d1 = beta.ppf(q) near 1 cancels; d1 rounds to 1 -> division by 0 (patterns 1, 8) | inf for 1.19e10 ((2, 0.5), q = 1 - 1e-10) | loss |
| F7 | ppf | small a, tiny q: beta variable u < DBL_MIN | Boost `ibeta_inv` returns DBL_MIN or 0 for u below the normal range; x = (2u-1)sqrt(a+b)/(2sqrt(u(1-u))) saturates (pattern 3/4) | -5.3e153 for -1.19e200; -inf for -3.2e199 | loss |
| F8 | ppf | (0.5, 2), q in about [1e-14, 1e-9] | `beta.ppf` (Boost `ibeta_inv`) does not converge (reports/beta.md F1; pattern 10) | -1.6e12 for -1.19e10 | loss |
| F9 | ppf | q near 1/2, large shapes (600, 600) | `2 * d1 - 1` with d1 near 1/2 cancels (pattern 1) | 1.4e10 ulp (cond (arg) 5e9); 0.0 for -1.4e-16 | loss |
| F10 | cdf, sf | result near DBL_MIN, e.g. (1000, 5), x = -1 | `sc.betainc` (Boost `ibeta`) loses accuracy for results near the underflow limit (pattern 10, shared) | 8.7e7 ulp | severe |
| F11 | pdf | large a, small b, every x, e.g. (1000, 5) | `sc.beta(1000, 5)` is 2.56e3 ulp off (shared) | 3.04e3 ulp | moderate |
| F12 | ppf | (600, 600), q <= ~1e-120 | `beta.ppf` 1.6e3 ulp off in the far tail (reports/beta.md F7, shared) | 1.54e3 ulp | moderate |

### F1: x**2 overflows for abs(x) > 1.34e154

- **Code:** `_pdf` lines 5773-5774, `_cdf` line 5784, `_sf` line 5788: `x / np.sqrt(a + b + x ** 2)`.
- **Mechanism:** `x ** 2` is inf for abs(x) > 1.34e154, so t = x/inf = 0 instead of +-1. Then cdf and sf return the median value `I_{1/2}(a, b)` far out in either tail, and pdf returns its value at 0, `1/C`. The pdf case cannot be measured with the shapes used here because the true pdf is subnormal there. Examples: `pdf(1e200, 8, 4)` = 0.186 and `pdf(1e200, 0.5, 0.5)` = 0.318, where the true values are about 1e-1800 and 3e-401. For tiny shapes the true value is normal and it is measured: `pdf(-1e300, 0.001, 0.001)` = 0.0223 against a true 2.49e-304.
- **Regime:** abs(x) > 1.34e154, all shapes.
- **Reproducer:** `python accuracy.py refs/jf_skew_t_ref.py --at cdf 1e200 8 4`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e200, 8, 4)` | 0.11328125 | 1.0 | 4.0e+15 | 0 | severe |
  | `sf(-1e200, 8, 4)` | 0.88671875 | 1.0 | 5.1e+14 | 0 | severe |
  | `cdf(-1e300, 0.5, 0.5)` | 0.5000000000000001 | 3.1830988618379065e-301 | inf | 1 | loss |
  | `cdf(-1e300, 0.5, 2)` | 0.8838834764831843 | 1.1858541225631422e-300 | inf | 1 | loss |
  | `cdf(1e200, 1000, 5)` | 2.4646016013438567e-292 | 1.0 | 4.5e+15 | 0 | loss |
  | `pdf(-1e300, 0.001, 0.001)` | 0.022329739444621757 | 2.4928702993908764e-304 | inf | 1 | loss |

- **Fix direction:** `t = x / np.hypot(x, np.sqrt(a + b))` (or `sign(x) / sqrt(1 + (a+b)/x**2)` for large abs(x)), together with the F2/F3 rewrite.

### F2: pdf tail factor 1 -+ t cancels

- **Code:** `_pdf` lines 5773-5774: `(1 + x / np.sqrt(a + b + x ** 2)) ** (a + 0.5)` and `(1 - x / np.sqrt(a + b + x ** 2)) ** (b + 0.5)`.
- **Mechanism:** for x >> sqrt(a+b), `1 - t = (a+b)/(r (r + x))` with `r = sqrt(a+b+x^2)`, which is about (a+b)/(2x^2), but it is computed as a difference of two numbers near 1 (pattern 1). The relative error is about x^2/(a+b) ulp, multiplied by the exponent b + 1/2. Once (a+b)/(2x^2) < eps/2 the factor is exactly 0. The left tail has the same problem with `1 + t` and exponent a + 1/2.
- **Regime:** abs(x) > ~sqrt(a+b) * 1e2 for moderate (1e4 ulp) errors; total loss for abs(x) > ~sqrt((a+b)/eps), e.g. 3.5e8 for (8, 4) and 1e8 for (0.5, 0.5).
- **Reproducer:** `python accuracy.py refs/jf_skew_t_ref.py --at pdf 1e9 8 4`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e9, 8, 4)` | 0.0 | 2.1384e-76 | inf | 9 | loss |
  | `pdf(-1e10, 8, 4)` | 0.0 | 1.732104e-163 | inf | 17 | loss |
  | `pdf(1e20, 8, 4)` | 0.0 | 2.1384e-175 | inf | 9 | loss |
  | `pdf(1e8, 0.5, 0.5)` | 0.0 | 3.183098861837906e-17 | inf | 2 | loss |
  | `pdf(34641.016151377546, 8, 4)` | 2.9769621808327534e-36 | 2.9769621617760854e-36 | 2.9e+07 | 9 | severe |
  | `pdf(-346.41016151377545, 8, 4)` | 1.1620041205600151e-36 | 1.1620041205676442e-36 | 2.96e+04 | 17 | moderate |
  | `pdf(3170.1734968294713, 1000, 5)` | 2.5277316199190057e-13 | 2.5277316199311236e-13 | 2.16e+04 | 10.9 | moderate |
  | `pdf(951052049.0488415, 1000, 5)` | 3.9721471475039575e-73 | 1.4636539508024533e-73 | 7.7e+15 | 11 | loss |

  (cond (shapes) is 80-360 at these points, far below the errors.) The minor 20-40 ulp errors in the moderate tails of (8, 4), e.g. `pdf(-8, 8, 4)` with 39.7 ulp, come from the same rounding of `1 + t` raised to the power a + 1/2 (pattern 6).
- **Fix direction:** form `1 + t = (r + x)/r` and `1 - t = (a+b)/(r (r + x))` for x >= 0 (mirrored for x < 0), with `r = hypot(x, sqrt(a+b))`. Evaluate the powers in log space, `exp((a+1/2) log(1+t) + (b+1/2) log(1-t) - logC)`.

### F3: cdf left tail and sf right tail cancel in y = (1 + t)/2

- **Code:** `_cdf` lines 5784-5785: `y = (1 + x / np.sqrt(a + b + x ** 2)) * 0.5; sc.betainc(a, b, y)`; `_sf` lines 5788-5789: same y, `sc.betaincc(a, b, y)`.
- **Mechanism:** for x << 0, y = (a+b)/(2 r (r - x)) is computed as `1 + t` with t near -1, so it cancels (pattern 1) and becomes 0 for abs(x) > ~sqrt((a+b)/eps). `betainc(a, b, 0)` then returns 0. For x >> 0, y rounds toward 1 and only `1 - y` matters to `betaincc`, so the argument has already lost the information (pattern 10). Because sf(x; a, b) = cdf(-x; b, a), the two failures mirror each other. With tiny shapes the cdf is still about 0.25-0.48 deep in the tail, so the loss happens at bulk probabilities.
- **Regime:** cdf for x < -sqrt(a+b)*1e2 (moderate) up to total loss beyond ~ -sqrt((a+b)/eps); sf mirrored for x > 0.
- **Reproducer:** `python accuracy.py refs/jf_skew_t_ref.py --at cdf -4472135.954999579 0.001 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-4472135.954999579, 0.001, 0.001)` | 0.0 | 0.48124768976728216 | inf | 0.002 | loss |
  | `cdf(-346410161.51377547, 8, 4)` | 0.0 | 2.517700195312497e-131 | inf | 16 | loss |
  | `cdf(-1e8, 0.5, 0.5)` | 0.0 | 3.1830988618379066e-09 | inf | 1 | loss |
  | `sf(346410161.51377547, 8, 4)` | 0.0 | 1.2890624999999992e-64 | inf | 8 | loss |
  | `sf(103923048.45413263, 8, 4)` | 8.02185563161787e-61 | 1.9647347965249112e-60 | 2.7e+15 | 8 | severe |
  | `sf(34641.016151377546, 8, 4)` | 1.289062450616166e-32 | 1.2890624432812517e-32 | 2.6e+07 | 8 | severe |
  | `cdf(-346.41016151377545, 8, 4)` | 2.5160223678130953e-35 | 2.516022367828643e-35 | 2.78e+04 | 16 | moderate |

  The self-check NOTE/WARN lines for (0.001, 0.001) (cdf 0.0 and sf 1.0 at x = -1.5e152, where the true values are 0.246 and 0.754) are this finding.
- **Fix direction:** for x < 0 compute `y = (a+b)/(2 r (r - x))` and call `betainc(a, b, y)`. For x > 0 compute `1 - y = (a+b)/(2 r (r + x))` and use `betainc(b, a, 1 - y)` for the sf (and `betaincc(b, a, 1 - y)` for the cdf).

### F4: normalising constant overflows for a + b > 1025

- **Code:** `_pdf` line 5772: `c = 2 ** (a + b - 1) * sc.beta(a, b) * np.sqrt(a + b)`.
- **Mechanism:** `2 ** (a + b - 1)` is inf for a + b > 1025, although the product `2**(a+b-1) B(a, b)` stays representable (for a = b it is about sqrt(pi/a)). If `sc.beta` underflows to 0, c = nan and the pdf is nan everywhere. Otherwise c = inf and the pdf is 0 everywhere (pattern 3).
- **Regime:** a + b > 1025, all x.
- **Reproducer:** `python accuracy.py refs/jf_skew_t_ref.py --at pdf 3 600 600`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(3, 600, 600)` | nan | 0.0044891505726911834 | inf | 8.94 | loss |
  | `pdf(-1e-10, 600, 600)` | nan | 0.39885917609296695 | inf | 1e-20 | loss |
  | `pdf(0, 513, 513)` | 0.0 | 0.3988450841009643 | inf | 0 | loss |
  | `pdf(0, 1000, 30)` | 0.0 | 9.424630925600231e-253 | inf | 0 | loss |

- **Fix direction:** `logC = (a+b-1)*log(2) + betaln(a, b) + 0.5*log(a+b)` and work in log space (F2 fix).

### F5: integer shapes overflow in int64

- **Code:** `_pdf` line 5772: `2 ** (a + b - 1)`.
- **Mechanism:** `rv_continuous.pdf` passes integer shape arrays to `_pdf` unchanged, so `2 ** (a + b - 1)` is an int64 power. It wraps around: 2**63 is -9.2e18 (pdf negative) and 2**64 and above are 0 (division by 0, pdf inf). The same shapes given as floats give the right answer.
- **Regime:** integer a and b with a + b = 64 (negative pdf) or a + b >= 65 (inf); all x.
- **Reproducer:** the harness converts all inputs to float, so it cannot show this. Use `python scratch/jf_skew_t/int_shapes.py`; the reference values are the float-shape values from `--at pdf 0.5 32 32` etc.

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.5, 32, 32)` (int shapes) | -0.35009510325807486 | 0.35009510325807486 | inf (wrong sign) | - | loss |
  | `pdf(0.5, 31, 33)` (int shapes) | -0.29932722720755334 | 0.29932722720755334 | inf (wrong sign) | - | loss |
  | `pdf(0.5, 40, 40)` (int shapes) | inf | 0.35048807042726604 | inf | - | loss |
  | `pdf(0.5, 100, 3)` (int shapes) | inf | 1.0477415107157944e-24 | inf | - | loss |

  ("reference" here is SciPy with the same shapes as floats; the float path for these bulk points is not affected by F1-F4 and F11.)
- **Fix direction:** `2.0 ** (a + b - 1)` or `np.exp2`, or better the log-space constant from F4.

### F6: ppf cancels in 1 - d1 for q near 1

- **Code:** `_ppf` lines 5792-5795: `d1 = beta.ppf(q, a, b)`; `d3 = 2 * np.sqrt(d1 * (1 - d1))`.
- **Mechanism:** for q -> 1, d1 -> 1 and x is about sqrt(a+b)/(2 sqrt(1 - d1)). The value of `1 - d1` has only the digits of d1 beyond its leading 1s (pattern 1/8), even though `beta.ppf` returns d1 accurately (0.02 ulp at q = 0.99999, (2, 0.5)). The relative error of x is then about eps/(4(1 - d1)). Since 1 - d1 ~ (1 - q)^(1/b), this is far beyond the conditioning (~1/(2b(1-q))) whenever b <= 1. Once 1 - d1 < eps/2, d1 is 1.0 and the ppf is inf (or 0.0 at q = 1 - 3e-16, (2, 0.5)).
- **Regime:** q > ~0.99 for b = 0.5; inf for 1 - q < ~1e-8 with b = 0.5. For b >= 2 (e.g. (8, 4)) errors stay within conditioning (max 3.0e3 ulp at cond 1.1e15).
- **Reproducer:** `python accuracy.py refs/jf_skew_t_ref.py --at ppf 0.99999 2 0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999, 2, 0.5)` | inf | 11858540244.451403 | inf | 1.0e+10 | loss |
  | `ppf(0.9999999943765867, 0.5, 0.5)` | inf | 56604391.27194941 | inf | 1.8e+08 | loss |
  | `ppf(0.9999999999999997, 2, 0.5)` | 0.0 | 3560408122994089.5 | inf | 3.0e+15 | loss |
  | `ppf(0.99999, 2, 0.5)` | 118585.40734250835 | 118585.4122471914 | 1.9e+08 | 1e+05 | severe |
  | `ppf(0.999999, 0.5, 0.5)` | 318312.5598907831 | 318309.88617359026 | 3.8e+10 | 1.0e+06 | severe |
  | `ppf(0.99, 2, 0.5)` | 118.57574942561588 | 118.57574942564533 | 1.12e+03 | 99 | moderate |
  | `ppf(0.57, 0.001, 0.001)` | inf | 1.2607124801315919e+31 | inf | 663 | loss |

- **Fix direction:** for q > 1/2 compute `v = 1 - d1` directly as `betaincinv(b, a, 1 - q)` (or `beta.isf`), then `x = (1 - 2v) sqrt(a+b) / (2 sqrt(v (1 - v)))`. Equivalently `ppf(q, a, b) = -ppf(1 - q, b, a)` with the left-tail branch.

### F7: ppf saturates when the beta variable underflows

- **Code:** `_ppf` lines 5792-5795.
- **Mechanism:** in the left tail, x is about -sqrt(a+b)/(2 sqrt(u)) with u = beta.ppf(q, a, b) ~ q^(1/a). For small a and tiny q, u is below DBL_MIN even though x is a normal double. Boost `ibeta_inv` then returns DBL_MIN for (0.5, 2) and (0.001, 0.001), so the ppf saturates at -sqrt(a+b)/(2*1.49e-154). For (0.5, 0.5) it returns 0, so `d3 = 0` and the ppf is -inf (patterns 3/4).
- **Regime:** u < 2.2e-308, i.e. abs(ppf) > ~sqrt(a+b) * 3.4e153. That is q < ~1e-154 for a = 0.5, and q <= 0.22 for a = 0.001.
- **Reproducer:** `python accuracy.py refs/jf_skew_t_ref.py --at ppf 1e-200 0.5 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-200, 0.5, 2)` | -5.299901436085741e+153 | -1.1858541225631423e+200 | 4.5e+15 | 1 | loss |
  | `ppf(1e-200, 0.5, 0.5)` | -inf | -3.183098861837907e+199 | inf | 1 | loss |
  | `ppf(1e-300, 0.5, 0.5)` | -inf | -3.1830988618379066e+299 | inf | 1 | loss |
  | `ppf(0.15, 0.001, 0.001)` | -1.4990384980306195e+152 | -6.15480522792334e+259 | 4.5e+15 | 500 | loss |

- **Fix direction:** in the far left tail use the asymptotic inversion of cdf ~ K abs(x)^(-2a) (from `I_y(a,b) ~ y^a/(a B(a,b))`, y ~ (a+b)/(4x^2)), or a log-space beta inverse.

### F8: ppf inherits beta.ppf non-convergence

- **Code:** `_ppf` line 5792: `beta.ppf(q, a, b)` (scipy.special `_beta_ppf`, Boost `ibeta_inv`).
- **Mechanism:** Boost `ibeta_inv` hits its iteration cap (RuntimeWarning "Root finding evaluation exceeded ... iterations") and returns a wrong u. `beta.ppf(1e-10, 0.5, 2)` = 2.45e-25 against the true 4.44e-21. `sc.betaincinv(0.5, 2, 1e-10)` gives the correct 4.444444444444446e-21, so the problem is specific to the `beta.ppf` path (see reports/beta.md F1).
- **Regime:** (0.5, 2), q in about [1e-14, 1e-9]; reports/beta.md lists further bands.
- **Reproducer:** `python accuracy.py refs/jf_skew_t_ref.py --at ppf 1e-10 0.5 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-10, 0.5, 2)` | -1596661004737.4812 | -11858541225.631422 | 6.0e+17 | 1 | loss |
  | `ppf(3.4807005884285624e-14, 0.5, 2)` | -107786815814006.95 | -34069409086936.77 | 9.7e+15 | 1 | loss |

- **Fix direction:** fix Boost `ibeta_inv` convergence (shared), or use `sc.betaincinv` here.

### F9: ppf cancels in 2 d1 - 1 near the median

- **Code:** `_ppf` line 5793: `d2 = (2 * d1 - 1) * np.sqrt(a + b)`.
- **Mechanism:** near q = 1/2, d1 is close to 1/2, and storing it as a double loses everything in d1 - 1/2 below 5.5e-17 in absolute terms. At q = 0.4999999999 with (600, 600), d1 - 1/2 = -3.6e-12, and d1 is accurate to 0.1 ulp, yet `2*d1 - 1` keeps only ~5 significant digits (pattern 1). That is 3x the conditioning (cond (arg) 5e9). At q = 0.49999999999999994, d1 rounds to exactly 0.5 and the ppf is 0. The effect grows with the beta pdf at 1/2, i.e. with large a = b.
- **Regime:** abs(q - 1/2) small, large shapes; seen for (600, 600) only.
- **Reproducer:** `python accuracy.py refs/jf_skew_t_ref.py --at ppf 0.4999999999 600 600`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.4999999999, 600, 600)` | -2.507158750506891e-10 | -2.507150750638087e-10 | 1.4e+10 | 5.0e+09 | severe |
  | `ppf(0.5000000001, 600, 600)` | 2.507158750506891e-10 | 2.507150750638087e-10 | 1.4e+10 | 5.0e+09 | severe |
  | `ppf(0.49999999999999994, 600, 600)` | 0.0 | -1.3917481296285676e-16 | inf | 9.0e+15 | loss |

  (cond (arg) is 9e15 at the last point: a 1-ulp change in q moves the result by 100 %, so practical impact is small; the severity is the tool's.)
- **Fix direction:** near the median, refine x with a Newton step on the (fixed) cdf. Or invert a centred variable, e.g. via the t distribution for a = b.

### F10: cdf/sf near the underflow limit (sc.betainc)

- **Code:** `_cdf` line 5785: `sc.betainc(a, b, y)`.
- **Mechanism:** `y` is accurate here (2e-18 relative at x = -1). `sc.betainc(1000, 5, y)` itself is 8.7e7 ulp off for this result of 3.4e-306. Away from DBL_MIN the same function is accurate: 559 ulp at 4.5e-301 (y = 0.49) and 0 ulp at 2.5e-292 (y = 0.5). Boost `ibeta` loses accuracy when its result approaches the underflow threshold (pattern 10, shared; compare reports/beta.md F5 for the inverse).
- **Regime:** results within a few decades of DBL_MIN; seen for (1000, 5).
- **Reproducer:** `python accuracy.py refs/jf_skew_t_ref.py --at cdf -1.0 1000 5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-1.0, 1000, 5)` | 3.407884291429799e-306 | 3.407884356854066e-306 | 8.7e+07 | 32.4 | severe |

- **Fix direction:** in `scipy.special` / Boost (shared), e.g. scale the power terms in log space.

### F11: pdf off by ~2.6e3 ulp for large a, small b (sc.beta)

- **Code:** `_pdf` line 5772: `sc.beta(a, b)`.
- **Mechanism:** `sc.beta(1000, 5)` = 2.376155164065174e-14 is 2.56e3 ulp from the mpmath value. `sc.beta(600, 300)` is 1.04e3 ulp off; (8, 4) and (100, 5) are within 2 ulp. This error goes straight into the pdf at every x: at x = -1e-20 the power factors are exactly 1 and the pdf is still 2.54e3 ulp off.
- **Regime:** large a (or b) with the other shape small or moderate; every x.
- **Reproducer:** `python accuracy.py refs/jf_skew_t_ref.py --at pdf 1e-10 1000 5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-10, 1000, 5)` | 7.743307870718751e-291 | 7.743307870723397e-291 | 2.7e+03 | 3.14e-09 | moderate |
  | `pdf(1.0, 1000, 5)` | 1.9966706441933302e-277 | 1.996670644194676e-277 | 3.04e+03 | 30.4 | moderate |

  (cond (shapes) is about 700 here; the error is above it. The shapes are exact inputs either way.)
- **Fix direction:** `betaln` in the log-space constant (F4 fix) avoids `sc.beta`. The `sc.beta` inaccuracy itself is shared.

### F12: ppf far left tail for (600, 600) (beta.ppf)

- **Code:** `_ppf` line 5792.
- **Mechanism:** `beta.ppf(9.085175756517067e-139, 600, 600)` is 1.63e3 ulp off (checked separately against mpmath). The transformation to x adds nothing. Compare reports/beta.md F7 (Boost `ibeta_inv` far tail with large symmetric shapes).
- **Regime:** (600, 600), q <= ~1e-120 (minor, 154-772 ulp, for q in 1e-101 ... 1e-126).
- **Reproducer:** `python accuracy.py refs/jf_skew_t_ref.py --at ppf 9.085175756517067e-139 600 600`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(9.085175756517067e-139, 600, 600)` | -28.716204314381375 | -28.716204314391202 | 1.54e+03 | 0.00204 | moderate |

- **Fix direction:** shared, in Boost `ibeta_inv`.

## RuntimeWarnings

- `overflow encountered in square` (pdf, cdf, sf) for abs(x) > 1.34e154: comes with the wrong results of F1.
- `underflow encountered in square` for tiny abs(x): spurious. `x**2` underflows harmlessly and the results are accurate (e.g. pdf at x = -1e-20 is fine apart from F11).
- `overflow encountered in power`, `invalid value encountered in multiply` (pdf, a + b > 1025): come with F4.
- `underflow encountered in power/divide` (pdf, (1000, 5)): `(1 - t)**(b+1/2)` and the result underflow in the far tail. The results there are subnormal or 0 where the truth is below DBL_MIN, so they are not measured.
- `divide by zero encountered in divide` (ppf): comes with the inf/-inf results of F6 and F7.
- Boost `ibeta<double>: Root finding evaluation exceeded ... iterations` (ppf, (0.5, 2) and (2, 0.5)): comes with F8.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | bulk, (8, 4), (2, 0.5), (0.5, 2), (0.5, 0.5) | 10.4 ulp for (8, 4); minor 20-40 ulp in moderate tails (abs(x) ~ 6-9, (8, 4)), part of F2's mechanism |
| cdf | bulk, all sets except F1/F3/F10 regimes | 2.76 ulp for (8, 4); (1000, 5) near x = 0...9 up to 492 ulp minor (Boost `ibeta` for results ~1e-190...1e-280, same family as F10); (600, 600) left tail up to 572 ulp minor at x = -10 (cond (arg) 93) |
| sf | bulk, all sets except F1/F3 regimes | 21.5 ulp for (8, 4) |
| ppf | (8, 4) everywhere; (1000, 5) everywhere; left tail for a >= 2 down to q = 1e-300; (600, 600) right tail | 13.2 ulp (8, 4); 81.5 ulp (1000, 5), minor; within conditioning in the right tail of (8, 4) (3.0e3 ulp at cond 1.1e15) |

## Suspected, not confirmed

- Pattern 6 in pdf for large exponents (`(1 + t)**(a + 0.5)` with a = 1000): the expected error is up to ~a ulp. For (1000, 5) it is hidden by the 2.6e3 ulp `sc.beta` error (F11) and was not measured separately.
- The int64 overflow of F5 also affects any other code path that reaches `_pdf` with integer shapes (e.g. `expect`, `fit` with fixed integer shapes). Only `pdf` was checked.

Measurement notes: the run was split by parameter set (`--params`) because condition numbers for ppf with tiny shapes are slow. ppf was rerun after the reference was changed to invert in the beta variable: inverting the cdf in x is limited by the bracket widening in `invert` (abs(x) <= 2**81). The error plot was not produced.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `(a+1/2) log(1+t) + (b+1/2) log(1-t) - logC`, with 1 +- t from the F2 rewrite and `logC = (a+b-1) log 2 + betaln(a, b) + log(a+b)/2` | wherever pdf underflows or fails (F2, F4). `logpdf(-1e10, 8, 4)` = -inf, true -374.77203330350255 |
| logcdf | log(cdf) | log of the fixed cdf; in the far left tail asymptotically `a log(y) - log(a B(a, b))` + log1p corrections, with y = (a+b)/(2r(r-x)) | cdf underflows for x << 0 (for (8, 4) below x ~ -1e19 even with a fixed cdf); currently -inf from F3 much earlier |
| logsf | log(sf) | logcdf(-x; b, a) by reflection | `logsf(1e9, 8, 4)` = -inf, true -155.59258488643832 |
| isf | ppf(1 - q) | `isf(q, a, b) = -ppf(q, b, a)` (reflection X(a, b) = -X(b, a)), using the accurate left-tail branch | `1 - q` rounds away small q: `isf(1e-20, 8, 4)` = inf, true 1130.7657045575322 |

## Shared code

- `scipy.special.beta` (Boost): 2.56e3 ulp off at (1000, 5) and 1.04e3 ulp at (600, 300) (F11). Every distribution that calls `sc.beta` with one large and one small shape is affected.
- `scipy.special.betainc` (Boost `ibeta`): 8.7e7 ulp off for a result near DBL_MIN at (1000, 5), y = 0.4842 (x = -1) (F10). It affects `beta.cdf` and every distribution built on `betainc` (f, t-like, binom-like).
- `beta.ppf` / Boost `ibeta_inv`: non-convergence (F8 = reports/beta.md F1), far-tail inaccuracy (F12 = reports/beta.md F7), and clamping of roots below DBL_MIN to DBL_MIN or 0 (F7). jf_skew_t inherits all three through `_ppf`.

## Existing GitHub issues

(web search; `gh` CLI unavailable)

- none found for `jf_skew_t` itself.
- Shared `beta.ppf` / Boost `ibeta_inv` issues listed in reports/beta.md
  (#12796, closed; boostorg/math #1511, closed upstream) concern the same
  routine as F8/F12; not checked whether they cover these regimes.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e200, 8, 4)` | 0.11328125 | 1.0 | verified | ref: 1 - I_{1-y}(4, 8) via 2F1 series, 1-y formed without cancellation; also `pdf(-1e300, 0.001, 0.001)` = 0.0223 vs 2.4928702993908765e-304 |
| F2 | `pdf(1e9, 8, 4)` | 0.0 | 2.1384e-76 | verified | ref: log-space pdf with 1-t = (a+b)/(r(r+x)); also `pdf(34641.016151377546, 8, 4)` 2.88e7 ulp |
| F3 | `cdf(-4472135.954999579, 0.001, 0.001)` | 0.0 | 0.48124768976728216 | verified | ref: I_y(a,b) = y^a(1-y)^b/(aB) 2F1(a+b,1;a+1;y); also `sf(34641.016151377546, 8, 4)` 2.56e7 ulp |
| F4 | `pdf(3.0, 600.0, 600.0)` | nan | 0.0044891505726911835 | verified | with Python-int shapes (600, 600) SciPy returns inf instead (int64 path of F5) |
| F5 | `pdf(0.5, int64 32, int64 32)` | -0.35009510325807486 | 0.35009510325807482 | verified | mpmath ref, not float-shape SciPy; `pdf(0.5, 40, 40)` int = inf vs 0.35048807042726446 (float-shape SciPy is 20 ulp off, harmless) |
| F6 | `ppf(0.99999, 2, 0.5)` | 118585.40734250835 | 118585.4122471914 | verified | ref: Newton on I_v(b, a) = 1-q in log v; cdf(x_ref) = q checked; 1.86e8 ulp; `ppf(0.9999999999, 2, 0.5)` = inf vs 11858540244.451403 |
| F7 | `ppf(1e-200, 0.5, 2)` | -5.299901436085741e+153 | -1.1858541225631423e+200 | verified | `ppf(1e-200, 0.5, 0.5)` = -inf vs -3.1830988618379068e+199 |
| F8 | `ppf(1e-10, 0.5, 2)` | -1596661004737.4812 | -11858541225.631422 | verified | 6.0e17 ulp |
| F9 | `ppf(0.4999999999, 600, 600)` | -2.507158750506891e-10 | -2.5071507506380868e-10 | verified | ref: x-space Newton on mp.quad of the pdf from 0; 1.44e10 ulp |
| F10 | `cdf(-1.0, 1000, 5)` | 3.407884291429799e-306 | 3.4078843568540658e-306 | verified | 8.65e7 ulp |
| F11 | `pdf(1e-10, 1000.0, 5.0)` | 7.743307870718751e-291 | 7.7433078707233962e-291 | verified | 2.7e3 ulp; sc.beta(1000, 5) 2.54e3 ulp off. Python-int shapes give inf (F5) |
| F12 | `ppf(9.085175756517067e-139, 600, 600)` | -28.716204314381375 | -28.716204314391202 | verified | 1.54e3 ulp; cdf(x_ref) = q checked |

Verifier references: `scratch/jf_skew_t/verify/verify.py` and `float_shapes.py`, at 100 and 200 digits (agree to 30 digits at every point). Mechanisms match the cited lines 5772-5773, 5784, 5788, 5792-5793.

Verified by dist-accuracy-verifier: 12 of 12 verified.
