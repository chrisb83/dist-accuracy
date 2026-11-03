# crystalball: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `crystalball_gen`, lines 11703-11868
- **Implemented:** pdf, logpdf, cdf, sf, ppf
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf), isf = ppf(1 - q)
- **Related:** none; helpers: `_norm_cdf`, `_norm_sf`, `_norm_ppf`, `_norm_pdf_C`; special functions: sc.ndtr, sc.ndtri (through the helpers); sc.binom, sc.gamma, sc.gammainc only in `_munp`
- **Reference:** `refs/crystalball_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (2, 3), (0.001, 3), (1e-10, 2), (1e-150, 3), (10, 3), (2, 50), (1, 140), (1, 150), (1, 1000), (1, 1e6), (0.5, 1.001)
- **Worst finding:** loss in pdf, logpdf, cdf, sf, ppf

## Summary

The Gaussian part (x > -beta) is accurate in every implemented method: pdf,
logpdf, cdf and sf there are within conditioning at all 804-987 measured points
per method (max 262 ulp, cond 1051). The power-law part (x <= -beta, and
ppf for q < cdf(-beta)) is where things fail. pdf, cdf and sf form
`(m/beta)**m` explicitly. It overflows once m*log(m/beta) > 709.8 (for
example beta = 1, m >= 144, or beta = 1e-150, m = 3), and then the whole power-law
side returns nan or inf (F1). ppf forms the reciprocal `(m/beta)**(-m)`, which underflows, so ppf returns
-inf (F3). For large m, `(B - x)**(-m)` also underflows while the true pdf is
normal (F2). Other findings: sf left of -beta is `1 - cdf` and loses up to
3.9e9 ulp for small beta (F4). logpdf cancels two large `m*log` terms for
large m (F5, 4.7e6 ulp at m = 1e6). logpdf repeats the `-x**2/2` overflow
of `norm` (F6). ppf for q near 1 goes through `_norm_ppf` of a number near
1 instead of `_norm_isf` (F7).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf, cdf, sf | x <= -beta, m*log(m/beta) > 709.8 | `(m/beta)**m` overflows (pattern 3) | nan / inf / -inf on the whole power-law side | loss |
| F2 | pdf, cdf (sf via cdf) | x <= -beta, large m, (B - x)**m > ~1e308 | `(B - x)**(-m)` subnormal or 0 although result normal (patterns 3, 4) | 0.0 for 2.6e-43; 5.0e10 ulp | loss |
| F3 | ppf | q < cdf(-beta), m*log(m/beta) > ~708, or small q with large m | `(m/beta)**(-m)` (or the product) underflows, `0**(1/(1-m)) = inf` (pattern 3/4) | -inf for -1.816; 2.5e7 ulp | loss |
| F4 | sf | x <= -beta, small beta (cdf(-beta) near 1) | `1 - self._cdf` cancels (pattern 1) | 3.9e9 ulp | severe |
| F5 | logpdf | x <= -beta near -beta, large m | `m*log(m/beta) - m*log(m/beta - beta - x)` cancels (pattern 5) | 4.7e6 ulp | severe |
| F6 | logpdf | x in (1.34e154, 1.9e154), all shapes | `-x**2/2` overflows (pattern 3; same as norm F1) | -inf for -1.125e308 | loss |
| F7 | ppf | q near 1 (Gaussian branch) | `_norm_ppf` of a sum that is near 1 (pattern 8) | 5.0e12 ulp (cond 4.7e12) | severe |
| F8 | ppf | q = 0.5 (and q near 0.5), large beta | ndtri argument `0.5 - 2e-24` rounds to 0.5 (pattern 8) | 0.0 for -4.9e-24 | loss |
| F9 | ppf | q just below cdf(-beta), m close to 1 | rounding of the base amplified by the exponent 1/(1-m) (pattern 6) | 1.13e3 ulp (cond 1e3) | moderate |
| F10 | ppf, pdf, cdf | power-law side, moderate m | `B - Y` cancellation in ppf (pattern 5); `(...)**(-m)` rounding (pattern 6) | 196 ulp | minor |

### F1: `(m/beta)**m` overflows on the power-law side

- **Code:** `_pdf` `lhs`, line 11771: `(m/beta)**m * np.exp(-beta**2 / 2.0) * (m/beta - beta - x)**(-m)`;
  `_cdf` `lhs`, line 11803: same with `**(-m+1) / (m-1)`; `_sf` `lhs`, line 11820:
  `1 - self._cdf(x, beta, m)`.
- **Mechanism:** pattern 3. The prefactor `(m/beta)**m` is inf once
  m*log(m/beta) > 709.78. The second power `(m/beta - beta - x)**(-m)` is at most
  `(m/beta)**(-m)`, so it underflows to 0 (or is tiny), which gives `inf * 0 = nan`. The result is
  `inf` in cdf when the second factor is still nonzero (small beta). sf becomes `1 - nan = nan` or
  `1 - inf = -inf`. The true values are ordinary numbers. For example, the pdf at x = -beta is
  N*exp(-beta**2/2).
- **Regime:** every x <= -beta when m*log(m/beta) > 709.78: beta = 1 for m >= ~144
  (m = 140 is fine, m = 150 fails); m = 3 for beta < ~5e-103. Measured: all power-law points of (1, 150),
  (1, 1000), (1, 1e6) and (1e-150, 3), i.e. 17 + 14 + 13 + 53 pdf points, plus the
  same numbers for cdf/sf (sf: 48, 46, 46 and 63 points including x down to -1e300).
- **Reproducer:** `python accuracy.py refs/crystalball_ref.py --at pdf -2 1 150`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(-2, 1, 150)` | nan | 0.08231975971863677 | inf | 1.99 | loss |
  | `cdf(-2, 1, 150)` | nan | 0.08342472293633658 | inf | 1.97 | loss |
  | `sf(-2, 1, 150)` | nan | 0.9165752770636634 | inf | 0.18 | loss |
  | `pdf(-1, 1, 1000)` | nan | 0.22331134576235845 | inf | 1 | loss |
  | `pdf(-1e140, 1e-150, 3)` | nan | 6.666666666e-151 | inf | 1e-10 | loss |
  | `cdf(-1e140, 1e-150, 3)` | inf | 0.9999999999333333 | inf | 6.67e-11 | loss |
  | `sf(-1e140, 1e-150, 3)` | -inf | 6.666666666333334e-11 | inf | 1 | loss |
  | `cdf(-1e200, 1e-150, 3)` | nan | 9e-100 | inf | 2 | loss |

- **Fix direction:** never form the two powers separately. Use
  `exp(-beta**2/2) * (1 + (-beta - x)/(m/beta))**(-m)`, i.e.
  `exp(-beta**2/2 - m*log1p((-beta - x)*beta/m))`. Use the same with `(1 - m)` for the cdf.

### F2: `(B - x)**(-m)` underflows although the pdf/cdf is normal

- **Code:** line 11772 `(m/beta - beta - x)**(-m)`, line 11804 `(m/beta - beta - x)**(-m+1)`.
- **Mechanism:** patterns 3/4. When `(m/beta)**m` is still finite but large (up to
  ~1e308), the true result `(m/beta)**m * (B - x)**(-m)` can be normal while the
  second factor is already subnormal (~1e-310 to 5e-324, fewer significant bits)
  or 0. Subnormal gives severe/moderate errors, and 0 gives a 0 result.
- **Regime:** x <= -beta with (B - x)**m > ~1e308, i.e. B - x > 10**(308/m), as long as
  (m/beta)**m (B - x)**(-m) > 2.2e-308. For (1, 140): x <= ~-54. For (2, 50): x <= ~-2.5e6.
  Only matters for large m; for m = 3 both factors underflow together with the result.
- **Reproducer:** `python accuracy.py refs/crystalball_ref.py --at pdf -140 1 140`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(-140, 1, 140)` | 0.0 | 2.6402624657544074e-43 | inf | 70.3 | loss |
  | `pdf(-54.06977288832496, 1, 140)` | 6.400747940312173e-21 | 6.4008191994632504e-21 | 5.0e10 | 39.2 | severe |
  | `cdf(-54.06977288832496, 1, 140)` | 8.890691125982136e-21 | 8.890681360716545e-21 | 4.9e9 | 38.9 | severe |
  | `pdf(-2511863.4315095823, 2, 50)` | 4.238758584483167e-252 | 4.238805774333309e-252 | 5.0e10 | 50 | severe |
  | `cdf(-2511863.4315095823, 2, 50)` | 2.1729385123613467e-247 | 2.1729385123167972e-247 | 9.23e4 | 49 | moderate |
  | `pdf(-15848908.924611142, 2, 50)` | 0.0 | 4.238805774333402e-292 | inf | 50 | loss |

  (cdf (1, 140) also returns 0.0 at x = -129.27, -140, -233.76, -1391.)
- **Fix direction:** same as F1: one `exp` of the combined logarithm
  `-beta**2/2 - m*log1p((-beta - x)*beta/m)` (+ `log(N)` for logpdf-style evaluation).

### F3: ppf power-law branch returns -inf when `(m/beta)**(-m)` underflows

- **Code:** `_ppf` `ppf_less`, line 11833-11834:
  `m/beta - beta - ((m - 1)*(m/beta)**(-m)/eb2*p/N)**(1/(1-m))`.
- **Mechanism:** pattern 3/4. `(m/beta)**(-m)` underflows to 0 when m*log(m/beta) > 745
  (subnormal from 708). The base is then 0 and `0**(1/(1-m)) = inf`, so the result is `-inf`.
  For smaller m/beta the factor is finite, but the product with a small p underflows
  (subnormal: severe; 0: -inf).
- **Regime:** all q < cdf(-beta) when m*log(m/beta) > ~745 ((1, 150), (1, 1000),
  (1, 1e6), (1e-150, 3): 35, 35, 35, 58 points). For (2, 50), q < ~1e-250. For (1, 140),
  q < ~1e-76.
- **Reproducer:** `python accuracy.py refs/crystalball_ref.py --at ppf 0.1 1 150`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.1, 1, 150)` | -inf | -1.8164536051069315 | inf | 0.557 | loss |
  | `ppf(0.001, 1e-150, 3)` | -inf | -9.186832980505138e+151 | inf | 0.516 | loss |
  | `ppf(1e-300, 1, 140)` | -inf | -19801.47048966306 | inf | 0.00724 | loss |
  | `ppf(1e-300, 2, 50)` | -inf | -30797172.66009173 | inf | 0.0204 | loss |
  | `ppf(6.8129206905797616e-251, 2, 50)` | -2961426.776870402 | -2961426.760601957 | 2.5e7 | 0.0204 | severe |

- **Fix direction:** log form:
  `x = B - (m/beta) * exp((log(p) + log(m-1) - log(N) + beta**2/2 - log(m/beta))/(1 - m))`.
  This is `B - (m/beta)*(p/pbeta)**(1/(1-m))` with `pbeta = N*(m/beta)*exp(-beta**2/2)/(m-1)`,
  so only `log(p/pbeta)` is needed.

### F4: sf left of -beta is `1 - cdf` and cancels for small beta

- **Code:** `_sf` `lhs`, line 11820: `1 - self._cdf(x, beta, m)` (comment: "Default behavior is OK in the left tail of the SF").
- **Mechanism:** pattern 1. For x <= -beta, sf >= sf(-beta) = N*sqrt(2pi)*Phi(beta).
  This is small when the power-law mass dominates, i.e. small beta (sf(-beta) ~ beta*(m-1)/m*1.25)
  or m close to 1. Then the cdf is close to 1, and `1 - cdf` loses about cdf/sf ulp.
  The comment holds only for beta of order 1.
- **Regime:** x <= -beta with sf(x) small: beta = 1e-10: 3.9e9 ulp (all 16 points from
  -beta to -10); beta = 1e-3: up to 506 ulp; (0.5, 1.001): up to 781 ulp.
  The finding's severity comes from the beta = 1e-10 set; the other two sets are minor.
- **Reproducer:** `python accuracy.py refs/crystalball_ref.py --at sf -1 1e-10 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(-1, 1e-10, 2)` | 1.1266576560586827e-10 | 1.1266570685621474e-10 | 2.3e9 | 0.444 | severe |
  | `sf(-1e-09, 1e-10, 2)` | 6.266576146884972e-11 | 6.266570691184802e-11 | 3.9e9 | 7.98e-10 | severe |
  | `sf(-2, 0.001, 3)` | 0.0021657346829683233 | 0.0021657346829684976 | 363 | 0.614 | minor |
  | `sf(-2, 0.5, 1.001)` | 0.0015385495969050256 | 0.0015385495969047586 | 781 | 0.371 | minor |

- **Fix direction:** closed form without subtraction:
  sf(x) = N*[sqrt(2pi)*Phi(beta) + (m/beta)*exp(-beta**2/2)/(m-1)*(1 - ((B - x)*beta/m)**(1-m))],
  with `-expm1((1-m)*log1p((-beta - x)*beta/m))` for the last bracket.

### F5: logpdf cancels two large `m*log` terms for large m

- **Code:** `_logpdf` `lhs`, line 11787: `m*np.log(m/beta) - beta**2/2 - m*np.log(m/beta - beta - x)`.
- **Mechanism:** pattern 5. Near x = -beta both `m*log(...)` terms are about
  m*log(m/beta) (1.4e7 for m = 1e6, beta = 1), but their difference is O(1). The
  rounding of `m/beta - beta - x` and of each log term (absolute error ~ m*log(m/beta)*eps)
  dominates the result.
- **Regime:** x <= -beta with |x + beta| small compared with m/beta, large m: 4.7e6 ulp
  at m = 1e6, 1.35e3 ulp at m = 1000, 300 ulp (minor) at m = 140. The error grows
  like m*log(m/beta)/|logpdf|.
- **Reproducer:** `python accuracy.py refs/crystalball_ref.py --at logpdf -1.5 1 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(-1.5, 1, 1e6)` | -1.998964852480873 | -1.9989648504154947 | 4.7e6 | 0.75 | severe |
  | `logpdf(-1.000000001, 1, 1e6)` | -1.498964977278098 | -1.498964976415453 | 2.6e6 | 0.667 | severe |
  | `logpdf(-1.5, 1, 1000)` | -1.9990633535739164 | -1.9990633535733158 | 1.35e3 | 0.75 | moderate |
  | `logpdf(-1.5, 1, 140)` | -1.9996796410221078 | -1.9996796410219748 | 300 | 0.747 | minor |

- **Fix direction:** `-beta**2/2 - m*np.log1p((-beta - x)*beta/m)`. The argument
  `-beta - x` is exact near -beta (Sterbenz), so this is accurate.

### F6: logpdf overflows in `x**2` before halving

- **Code:** `_logpdf` `rhs`, line 11784: `-x**2/2`.
- **Mechanism:** pattern 3; identical to `norm` F1 / INDEX item 24 but a separate copy
  (crystalball does not call `_norm_logpdf`). `x**2` overflows for x > 1.34e154
  although `-x**2/2 + log(N)` is finite up to x = 1.9e154.
- **Regime:** 1.3407807929942597e154 <= x < ~1.896e154, every parameter set (3 grid points each, 33 total).
- **Reproducer:** `python accuracy.py refs/crystalball_ref.py --at logpdf 1.5e154 2 3`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.5e154, 2, 3)` | -inf | -1.1250000000000002e+308 | inf | 2 | loss |

- **Fix direction:** `-(0.5*x)*x`.

### F7: ppf for q near 1 uses `_norm_ppf` of a number near 1

- **Code:** `_ppf` `ppf_greater`, line 11840:
  `_norm_ppf(_norm_cdf(-beta) + (1/_norm_pdf_C)*(p/N - C))`.
- **Mechanism:** pattern 8. The ndtri argument `a` is a sum of rounded terms near 1.
  Its absolute error (a few eps) is then divided by 1 - a ~ 1 - p. The result is roughly as
  bad as the argument conditioning (cond ~ 1/((1-p)*x**2)). It exceeds that only slightly
  for (1, 150) (5.0e12 ulp vs cond 4.7e12), so the tool marks it severe. In all other sets the
  same error is just inside the conditioning (e.g. 1.2e12 ulp at cond 4.5e13 for (2, 3)).
  Checked: `_norm_isf((1-p)/(N*_norm_pdf_C))` (1 - p exact) gives
  7.148377040565187 and 7.776293155470989 for the two points below, i.e. the reference.
- **Regime:** q > ~1 - 1e-8 in the Gaussian branch; flagged for (1, 150) (3 points).
- **Reproducer:** `python accuracy.py refs/crystalball_ref.py --at ppf 0.9999999999999966 1 150`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999995953, 1, 150)` | 7.148293886221173 | 7.148377040565187 | 5.2e10 | 4.7e10 | severe |
  | `ppf(0.9999999999999966, 1, 150)` | 7.767681015116431 | 7.776293155470988 | 5.0e12 | 4.7e12 | severe |

- **Fix direction:** for p > 1/2 use `_norm_isf((1 - p)/(N*_norm_pdf_C))`. This is also the
  natural custom `_isf` (see Opportunities).

### F8: ppf returns 0 at q = 0.5 for large beta

- **Code:** line 11840 (as F7).
- **Mechanism:** pattern 8. For beta = 10 the tail mass shifts the median by ~5e-24:
  the ndtri argument is 0.5 - 2e-24, which rounds to exactly 0.5, so `_norm_ppf` returns 0.
  The argument condition number is huge (2.5e23), so this is a low-priority edge case. The tool
  reports a loss because 0 is returned for a nonzero true value.
- **Regime:** q within ~1e-16 of 0.5 with tail mass (m/beta)exp(-beta**2/2) below ~eps.
- **Reproducer:** `python accuracy.py refs/crystalball_ref.py --at ppf 0.5 10 3`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.5, 10, 3)` | 0.0 | -4.915554340282728e-24 | inf | 2.5e23 | loss |
  | `ppf(0.49999999999999994, 10, 3)` | 0.0 | -1.391458261491427e-16 | inf | 9.0e15 | loss |

- **Fix direction:** would need a correction term `(0.5 - a)` computed without forming `a`
  (e.g. `(p - 0.5)/(N*sqrt(2pi)) + (Phi(beta) - 1/(2N sqrt(2pi)))...`); probably not worth it.

### F9: ppf power-law branch amplifies rounding for m close to 1

- **Code:** line 11834: `(...)**(1/(1-m))`.
- **Mechanism:** pattern 6. The base is a product of ~6 rounded factors. The exponent
  1/(1 - m) = -1000 for m = 1.001 multiplies their relative rounding error by 1000.
  The argument conditioning is also ~1000 here, and the measured error is just above it.
- **Regime:** q < cdf(-beta), m - 1 small.
- **Reproducer:** `python accuracy.py refs/crystalball_ref.py --at ppf 0.49999999999999994 0.5 1.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.49999999999999994, 0.5, 1.001)` | -8.046623125847387e+300 | -8.046623125845372e+300 | 1.13e3 | 1e3 | moderate |

- **Fix direction:** the log form of F3 (one rounding in `log(p/pbeta)`, then `exp`).

### F10: minor errors on the power-law side for moderate m

- **Code:** line 11833 `m/beta - beta - (...)` (ppf); lines 11771-11772 (pdf), 11803-11804 (cdf).
- **Mechanism:** ppf: the subtraction `B - Y` with Y ~ m/beta cancels to x ~ -beta
  (pattern 5), losing about (m/beta)/|x| ulp. pdf/cdf: the rounding of `m/beta - beta - x`
  raised to the power m (pattern 6).
- **Regime:** power-law side, m/beta large compared with |x|.
- **Reproducer:** `python accuracy.py refs/crystalball_ref.py --at ppf 0.21999999999999997 1 140`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.21999999999999997, 1, 140)` | -1.0208765072839583 | -1.0208765072839139 | 196 | 0.987 | minor |
  | `ppf(0.01, 2, 50)` | -2.519873607926659 | -2.519873607926667 | 14.3 | 0.207 | minor |
  | `pdf(-1.000000001, 1, 140)` | 0.22300292768130012 | 0.22300292768129773 | 48.2 | 1 | minor |

- **Fix direction:** ppf: `-beta - (m/beta)*expm1(log(p/pbeta)/(1-m))`; pdf/cdf: `log1p` form of F1.

## RuntimeWarnings

- pdf, cdf, sf: `overflow encountered in power`, `invalid value encountered in multiply`
  and `underflow encountered in power` together with the wrong results of F1/F2.
  `underflow encountered in exp/square/multiply` in the far tails where the true result also
  underflows (correct 0, spurious warning).
- logpdf: `overflow encountered in square` for x > 1.34e154. The result is wrong in the F6 window
  and correct (-inf) beyond 1.9e154.
- ppf: `divide by zero encountered in power` (0**negative) and `underflow encountered in power/multiply/divide`
  together with the -inf results of F3.
- No warnings on accurate results in the bulk. `xpx.apply_where` evaluates each branch only on its
  own elements, so there are no cross-branch warnings.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | Gaussian side x > -beta, all 11 sets (804 points), incl. x = 0, -beta(1 - 1e-9), large x | 249 ulp at x = 32.4, (1, 150), cond 1050 (within conditioning) |
| logpdf | Gaussian side (987 points, except F6 window); power-law side for m <= 50 | 1.33 ulp (Gaussian side); 21.9 ulp minor at (2, 50), x = -2.000000002 (F5 mechanism, small) |
| cdf | Gaussian side, all sets, both sides of -beta (`_norm_cdf(x) - _norm_cdf(-beta)` added to a positive constant does not cancel badly) | 32.7 ulp at x = -9, (10, 3), cond 82 |
| sf | Gaussian side (`_norm_sf(x)/M`), all sets, to x = 1e150 | 262 ulp at x = 32.4, (1, 150), cond 1051 |
| ppf | Gaussian branch q from 1e-300 to 1 - 3e-16, both sides of q = cdf(-beta) (pbeta(1 +- 1e-9)), except F7/F8 points | within conditioning (e.g. 1.2e12 ulp at cond 4.5e13, (2, 3), q -> 1) |
| pdf, cdf | power-law side for m*log(m/beta) < 709 and (B - x)**m < 1e308: (2, 3), (0.001, 3), (1e-10, 2), (10, 3), (0.5, 1.001) | 61.6 ulp minor at x = -2.15e103, (0.001, 3), result 1.8e-303 near the underflow edge; otherwise <= 2.4 ulp |

Not measured: cdf/sf at x >= 1e160 (165 points per method). mpmath's `ncdf` raises
OverflowError there in the reference, but the true values are 1 and 0 (underflow), which SciPy returns.
`_munp` (moments) is out of scope.

## Suspected, not confirmed

- N (lines 11764, 11780, 11795, 11825) overflows in `m/beta` for beta < ~m/1.8e308, giving
  N = 0 and pdf = 0. The true pdf there (~beta*(m-1)/m) is subnormal, so this is not measurable and
  not a finding.
- `ppf_less` divides by `eb2 = exp(-beta**2/2)`, which underflows for beta > 38.6 (division by 0). Then
  pbeta = 0, so `ppf_less` is never selected for p > 0. Not reachable.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | x <= -beta: `log(N) - beta**2/2 - log(m-1) + log(m/beta) + (1-m)*log1p((-beta - x)*beta/m)`; x > -beta: `log1p(-sf)` or log(cdf) | power-law tail where cdf underflows: `logcdf(-1e300, 2, 3)` = -inf, true -1383.9643337076443; also inherits F1 (nan) for large m |
| logsf | log(sf) | x > -beta: `log(_norm_pdf_C) + _norm_logsf(x) + log(N)`; x <= -beta: `log1p(-cdf)` (or the F4 closed form) | Gaussian tail where sf underflows (x > ~37.5): `logsf(40, 2, 3)` = -inf, true -804.62602953553059 |
| isf | ppf(1 - q) | q < 1 - pbeta: `_norm_isf(q/(N*_norm_pdf_C))`; else power-law closed form with p = 1 - q | q < eps/2: `isf(1e-20, 2, 3)` = inf, true 9.2604624840457566; digits lost for all q < ~1e-3 |

## Shared code

- No defect in the shared `_norm_*` helpers or in scipy.special here. The `-x**2/2` overflow (F6) is
  a local copy of the pattern in INDEX item 24 (`_norm_logpdf`, norm F1), not a call to the helper.
- F7: ppf uses `_norm_ppf` of a value near 1 instead of `_norm_isf` of the complement. This is a usage
  issue in crystalball, not a helper defect.

## Existing GitHub issues

none found (web search; gh unavailable). Related: PR #6798 (added crystalball, 2017; follow-ups in gh-7687, not found).

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(-2, 1, 150)` | nan | 0.08231975971863677 | verified | also cdf/sf(-2, 1, 150), pdf(-1, 1, 1000), pdf/cdf/sf(-1e140, 1e-150, 3), cdf(-1e200, 1e-150, 3): all nan/inf/-inf, refs equal the report's; lines 11771/11803/11820 match |
| F2 | `pdf(-54.06977288832496, 1, 140)` | 6.400747940312173e-21 | 6.4008191994632504e-21 | verified | 5.01e10 ulp; pdf(-140, 1, 140) = 0 vs 2.64e-43, cdf(-140, 1, 140) = 0 vs 5.30e-43, (2, 50) points identical to report; `(B - x)**(-m)` = 1e-320 (subnormal) confirmed |
| F3 | `ppf(0.1, 1, 150)` | -inf | -1.8164536051069315 | verified | other four points match (incl. 2.47e7 ulp at q = 6.8e-251, (2, 50)); verifier ppf checked by cdf(x_ref) = q to 1e-199 |
| F4 | `sf(-1e-09, 1e-10, 2)` | 6.266576146884972e-11 | 6.266570691184802e-11 | verified | 3.92e9 ulp; sf(-1, 1e-10, 2) 2.35e9 ulp; minor points 363 and 781 ulp match |
| F5 | `logpdf(-1.5, 1, 1e6)` | -1.998964852480873 | -1.9989648504154947 | verified | 4.65e6 ulp; 2.59e6, 1.35e3, 299 ulp at the other three points |
| F6 | `logpdf(1.5e154, 2, 3)` | -inf | -1.1250000000000002e+308 | verified | line 11784 `-x**2/2` |
| F7 | `ppf(0.9999999999999966, 1, 150)` | 7.767681015116431 | 7.776293155470988 | verified | 4.99e12 ulp; 5.24e10 ulp at q = 0.9999999999995953; ref checked by sf(x_ref) = 1 - q |
| F8 | `ppf(0.5, 10, 3)` | 0.0 | -4.915554340282728e-24 | verified | ndtri argument evaluates to exactly 0.5 (checked); also q = 0.49999999999999994 -> 0 vs -1.39e-16 |
| F9 | `ppf(0.49999999999999994, 0.5, 1.001)` | -8.046623125847387e+300 | -8.046623125845372e+300 | verified | 1.13e3 ulp (moderate) |

Verifier reference: `scratch/crystalball/verify/verify.py`, independent route (normalization and cdf/sf by
`mp.quad` of the pdf; power-law part in u = log t; sf left of -beta integrated directly; ppf by closed
form in log space / `erfinv`, each checked against the quadrature cdf/sf). 100 vs 200 digits agree to
30 digits at every point.

Verified by dist-accuracy-verifier: 9 of 9 verified.
