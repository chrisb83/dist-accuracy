# powerlognorm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `powerlognorm_gen`, lines 8922-8974
- **Implemented:** pdf, logpdf, cdf, sf, logsf, ppf, isf (pdf = `exp(logpdf)`, cdf = `-expm1(logsf)`, sf = `exp(logsf)`, ppf = `isf(1 - q)`)
- **Inherited (not measured):** logcdf = log(cdf)
- **Related:** powernorm (same construction without the log, separate class, reviewed separately); helpers: `_norm_logpdf`, `_norm_logcdf`, `_norm_ppf`; special functions: `sc.log_ndtr`, `sc.ndtri` (via the helpers), `sc.expm1`
- **Reference:** `refs/powerlognorm_ref.py`, 120 digits (all formulas avoid `1 - (number near 1)`; checked against 400 digits, and the closed-form ppf/isf against `invert()` of the cdf); self-checks: PASS
- **Parameter sets:** (2.1413923530064087, 0.44639540782048337), (0.001, 1), (1000, 0.5), (1e6, 1), (2, 1000), (0.5, 1e-152), (5, 0.001), (1e20, 1)
- **Worst finding:** loss in ppf, isf, logpdf, logsf, cdf

## Summary

pdf, logpdf, cdf, sf and logsf are built from `log_ndtr`. For ordinary shapes
they are accurate: the error stays at a few hundred ulp or less, and most of it
comes from the far-tail error of `sc.ndtr`/`sc.log_ndtr` and from `exp` of a
large log value. The quantile functions are the weak part. `ppf` computes
`isf(1 - q)`, which returns 0.0 for q < 1.1e-16 and is severely wrong up to
q ~ 1e-3. In `isf`, `q**(1/c)` underflows when log(q)/c < -745 (small c, or
small q), which returns inf. When c is large, `q**(1/c)` is close to 1:
`ndtri` then loses up to 2.6e14 ulp, and for c > ~6e15, or q close to 1, it
returns 0.0 for every q. With extreme shapes, intermediates in the log-space
methods overflow or underflow before they are multiplied by c:
- tiny s: `z**2` overflows (logpdf -inf/nan), and `log_ndtr(-z)` overflows
  before it is multiplied by c < 1 (logsf -inf);
- huge c: `log_ndtr(-z)` underflows before it is multiplied by c (cdf 0.0).

For small c, pdf loses up to 1.8e5 ulp in the right tail, because two
z**2-sized terms of logpdf cancel.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf | q < ~1e-3 (any shapes); total loss for q < 1.1e-16 | `1 - q` rounds before `isf` (pattern 8) | 0.0 for 0.0217; 3.7e8 ulp at q = 1e-10 | loss |
| F2 | isf, ppf | log(q)/c < -708 (isf) or log(1-q)/c < -708 (ppf): small c, or c < 1 with tiny q | `q**(1/c)` subnormal / 0, `ndtri(0) = -inf` (pattern 4/8) | inf for 2.7e29; 3.2e13 ulp | loss |
| F3 | isf, ppf | large c (q**(1/c) close to 1), and q -> 1 for any c | rounding of `q**(1/c)` near 1 amplified by `ndtri`; `q**(1/c) == 1.0` gives 0.0 (pattern 8) | 0.0 for 9.1e-5 (c = 1e20); 2.6e14 ulp (c = 1e16) | loss |
| F4 | logpdf | tiny s: \|log(x)/s\| > 1.34e154 with finite true logpdf | `z**2` overflows in `_norm_logpdf` (pattern 3, shared helper) | -inf for -1.7e308 | loss |
| F5 | logsf, logpdf | tiny s, c < 1: log(x)/s > ~1.9e154 with finite c*z**2/2 | `log_ndtr(-z)` overflows to -inf before `* c` (pattern 3) | logsf -inf, logpdf nan for -1.3e308 | loss |
| F6 | cdf, logsf | huge c (> ~1e8), log(x)/s < -37.5 where c*Phi(z) is normal | `log_ndtr(-z)` underflows to 0 before `* c` (pattern 4) | 0.0 for 6.6e-303 | loss |
| F7 | pdf, logpdf | small c (measured 1e-3, 1e-2), right tail x >> 1 | `-z**2/2` and `(c-1)*log_ndtr(-z)` cancel (pattern 5), exp turns it into relative error | pdf 1.8e5 ulp, logpdf 365 ulp | moderate |
| F8 | cdf, logsf | left tail, log(x)/s in about [-38, -8] | far-tail error of `sc.log_ndtr(t > 0)` = `ndtr(-t)` (shared) | 347 ulp | minor |
| F9 | pdf, sf | \|logpdf\| or \|logsf\| large (up to ~708) | `exp` of a log value turns its absolute rounding error into relative error | pdf 461 ulp, sf 296 ulp | minor |

### F1: ppf rounds 1 - q before calling isf

- **Code:** `_ppf`, line 8965: `return self._isf(1 - q, c, s)`
- **Mechanism:** `1 - q` keeps only the bits of q beyond the leading 1. The
  sf-level target `1 - q` then has an absolute error of ~eps/2, which is a
  relative error of up to eps/(2q) in the lower-tail probability. For q < 1.1e-16,
  `1 - q == 1.0`, so `q**(1/c) = 1`, `ndtri(1) = inf`, and `exp(-inf*s) = 0.0`
  (pattern 8).
- **Regime:** every shape set. Loss for q < 1.1e-16. Severe down to q ~ 1e-10, and
  moderate/minor up to q ~ 1e-2 (for large c, F3 makes it worse).
- **Reproducer:** `python accuracy.py refs/powerlognorm_ref.py --at ppf 1e-17 2.1413923530064087 0.44639540782048337`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-17, 2.1413923530064087, 0.44639540782048337)` | 0.0 | 0.02169057110102796 | inf | 0.0513 | loss |
  | `ppf(1e-17, 1, 1)` | 0.0 | 0.00020473517891357395 | inf | - | loss |
  | `ppf(1e-300, 2.1413923530064087, 0.44639540782048337)` | 0.0 | 6.51339741312985e-08 | inf | 0.012 | loss |
  | `ppf(3.4807005884285624e-14, 2.1413923530064087, 0.44639540782048337)` | 0.0338107168331514 | 0.03380280279533151 | 1.05e+12 | 0.0579 | severe |
  | `ppf(1e-10, 2.1413923530064087, 0.44639540782048337)` | 0.05549716387054267 | 0.05549715933872704 | 3.7e+08 | 0.0674 | severe |
  | `ppf(3.4807005884285624e-14, 5, 0.001)` | 0.9923330823971203 | 0.9923324824052361 | 2.7e+09 | 0.000128 | severe |
  | `ppf(0.001, 1000, 0.5)` | 0.09286005897174059 | 0.09286005897165295 | 4.25e+03 | 0.101 | moderate |

- **Fix direction:** work from the log of the sf target: `exp(-s * ndtri_exp(log1p(-q)/c))`
  (no `1 - q`, no `q**(1/c)`; this also fixes F2 and F3 for ppf).

### F2: q**(1/c) underflows in isf (and in ppf via 1 - q)

- **Code:** `_isf`, line 8974: `return np.exp(-_norm_ppf(q**(1/c)) * s)`
- **Mechanism:** `q**(1/c) = exp(log(q)/c)`. Once log(q)/c < -708 the
  power is subnormal and keeps only a few bits. Below -745 it is 0, so
  `ndtri(0) = -inf` and `exp(+inf) = inf`. The true quantile is finite, for
  example exp(67.9*s) for log(q)/c = -2303 (patterns 4 and 8).
- **Regime:** isf: q < exp(-708*c), i.e. q < ~0.49 for c = 1e-3, q < ~1e-154 for c = 0.5,
  every q < 1e-300 once c < ~0.97. ppf: the same with 1 - q, i.e. q > ~0.51 for
  c = 1e-3. With tiny s the true result is 1.0 but scipy returns inf.
- **Reproducer:** `python accuracy.py refs/powerlognorm_ref.py --at isf 0.1 0.001 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.1, 0.001, 1)` | inf | 2.7475727890520268e+29 | inf | 14.7 | loss |
  | `ppf(0.9, 0.001, 1)` | inf | 2.747572789052038e+29 | inf | 133 | loss |
  | `isf(1e-170, 0.5, 1)` | inf | 1.3626402522782349e+17 | inf | 0.0507 | loss |
  | `isf(1e-200, 0.5, 1e-152)` | inf | 1.0 | inf | 0 | loss |
  | `isf(0.4752, 0.001, 1)` | 4.992980334872479e+16 | 5.028476396326024e+16 | 3.2e+13 | - | severe |
  | `isf(0.4771139155210344, 0.001, 1)` | 4.528801257576691e+16 | 4.5291056434669064e+16 | 3.0e+11 | - | severe |

- **Fix direction:** `-sc.ndtri_exp(np.log(q)/c)` instead of `-ndtri(q**(1/c))`.

### F3: q**(1/c) close to 1 for large c (and for q near 1)

- **Code:** `_isf`, line 8974: `_norm_ppf(q**(1/c))`; ppf via line 8965.
- **Mechanism:** when log(q)/c is small, `u = q**(1/c)` is close to 1 and only
  `1 - u` matters to `ndtri`. Rounding `u` to a double leaves `1 - u` with a
  relative error of up to eps/(1 - u) ≈ eps*c/|log q|, which `ndtri` and
  `exp(-s*z)` pass through, scaled by about s*z. Once |log q|/c < 1.1e-16,
  `u == 1.0`, so `ndtri(1) = inf` and the result is 0.0. That happens for every q
  when c > ~6e15, and for q close to 1 when c > (1 - q)/1.1e-16 (pattern 8).
  The error exceeds the conditioning with respect to q by about a factor c.
- **Regime:** large c everywhere: c = 1e3 gives ~10-800 ulp in the bulk; c = 1e6 gives
  up to 1.7e5 ulp in the bulk; c = 1e10 gives 5.2e8 ulp; c >= 6e15 gives severe or 0.0.
  For isf with q -> 1 (ppf with q -> 0), any c > 1, growing with c.
- **Reproducer:** `python accuracy.py refs/powerlognorm_ref.py --at isf 0.5 1e20 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.5, 1e20, 1)` | 0.0 | 9.129780070956563e-05 | inf | 0.153 | loss |
  | `ppf(0.5, 1e20, 1)` | 0.0 | 9.129780070956563e-05 | inf | 0.153 | loss |
  | `isf(0.9999999999999627, 1000, 0.5)` | 0.0 | 0.015456144143349443 | inf | 1.6e+12 | loss |
  | `isf(0.5, 1e16, 1)` | 0.0002720468805377754 | 0.0002571340108560717 | 2.6e+14 | - | severe |
  | `isf(0.5, 1e10, 1)` | 0.0016329105558084646 | 0.001632910745719139 | 5.2e+08 | - | severe |
  | `isf(0.99, 1e6, 1)` | 0.0036569253406651585 | 0.003656925341746108 | 1.3e+06 | 17.2 | severe |
  | `isf(0.9999999999956061, 5, 0.001)` | 0.9929723237942852 | 0.9929723307743933 | 3.2e+07 | 3.17e+07 | severe |
  | `isf(0.5, 1e6, 1)` | 0.008010796196284992 | 0.008010796196386694 | 5.72e+04 | 0.287 | moderate |
  | `isf(1e-300, 1e6, 1)` | 0.040820228872752544 | 0.04082022887275282 | 30.6 | 0.000417 | minor |

  (The c = 5 row is right at the edge: its ulp is about the size of cond (arg),
  but the tool rates it severe. The two loss rows at q close to 1 are badly
  conditioned in q, but scipy returns 0.0.)
- **Fix direction:** same as F2: `ndtri_exp(log(q)/c)` (for log(q)/c close to 0 it
  works with `-expm1(log(q)/c)`). For ppf use `log1p(-q)/c`.

### F4: z**2 overflows in _norm_logpdf for tiny s

- **Code:** `_logpdf`, line 8958: `_norm_logpdf(np.log(x) / s)`; helper `_norm_logpdf`, line 367: `-x**2 / 2.0 - _norm_pdf_logC`
- **Mechanism:** `z**2` overflows for |z| > 1.34e154 even though `-z**2/2`
  is finite up to |z| = 1.9e154 (pattern 3; INDEX item 2 for the shared helper).
  The total logpdf can stay finite beyond that: for c < 1 it is about -c*z**2/2.
- **Regime:** s < ~1e-152 and |log x| > 1.34e154*s. For x < 1, only the band
  1.34e154 < |z| < 1.9e154 is affected (the true logpdf is ≈ -z**2/2 there). For x > 1 and
  c < 1 the band extends to |z| < 1.9e154/sqrt(c); above 1.9e154 F5 applies too.
- **Reproducer:** `python accuracy.py refs/powerlognorm_ref.py --at logpdf 1e-80 0.5 1e-152`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-80, 0.5, 1e-152)` | -inf | -1.6966073953530872e+308 | inf | 0.0109 | loss |
  | `logpdf(1e+70, 0.5, 1e-152)` | -inf | -6.494825185336037e+307 | inf | 0.0124 | loss |
  | `logpdf(1.4e+65, 1, 1e-152)` | -inf | -1.1250675502324898e+308 | inf | - | loss |

- **Fix direction:** in `_norm_logpdf`, `-(x/2)*x` or `-0.5*x*x` with the
  halving first (e.g. `-(x*np.sqrt(0.5))**2`); for c < 1 also combine with F5.

### F5: log_ndtr(-z) overflows before it is multiplied by c < 1

- **Code:** `_logsf`, line 8971: `_norm_logcdf(-np.log(x) / s) * c`; `_logpdf`, line 8959: `_norm_logcdf(-np.log(x) / s) * (c - 1.)`
- **Mechanism:** `log_ndtr(-z) ≈ -z**2/2` is -inf (correctly rounded) for
  z > 1.9e154, but `c * log_ndtr(-z)` is finite for c < 1 (pattern 3). logsf
  is then -inf. In logpdf the term `(c-1)*(-inf) = +inf` meets `-inf` from F4,
  which gives nan.
- **Regime:** s < ~1e-152, c < 1, 1.9e154 < log(x)/s < 1.9e154/sqrt(c).
- **Reproducer:** `python accuracy.py refs/powerlognorm_ref.py --at logsf 1e100 0.5 1e-152`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logsf(1e+100, 0.5, 1e-152)` | -inf | -1.3254745276195994e+308 | inf | - | loss |
  | `logpdf(1e+100, 0.5, 1e-152)` | nan | -1.3254745276195994e+308 | inf | - | loss |
  | `logpdf(7.225973768125954e+86, 0.5, 1e-152)` | nan | -1.0000000000000002e+308 | inf | - | loss |

- **Fix direction:** for very large z use the asymptotic form with c folded into
  the square, `-(z*sqrt(c/2))**2 - c*log(z*sqrt(2 pi))`, or reduce z before squaring.

### F6: log_ndtr(-z) underflows before it is multiplied by a huge c

- **Code:** `_logsf`, line 8971: `_norm_logcdf(-np.log(x) / s) * c`; `_cdf`, line 8962: `-sc.expm1(self._logsf(...))`
- **Mechanism:** for t = -z > 37.5, `log_ndtr(t) = -Phi(-t)` is subnormal, and
  for t > ~37.7 it is 0 (`ndtr(-37.8)` already returns 0). Multiplying by
  c afterwards cannot recover the bits, although `c*Phi(z)` is a normal double
  once c > ~1e8 (pattern 4). cdf and logsf return 0.0 / -0.0. In the
  subnormal band they lose bits.
- **Regime:** c > ~1e8 and log(x)/s < -37.5 with c*Phi(log(x)/s) > 2.2e-308.
- **Reproducer:** `python accuracy.py refs/powerlognorm_ref.py --at cdf 2.104223637677623e-17 1e20 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(2.104223637677623e-17, 1e20, 1)` | 0.0 | 6.601599854326758e-303 | inf | 38.4 | loss |
  | `logsf(2.104223637677623e-17, 1e20, 1)` | -0.0 | -6.601599854326758e-303 | inf | - | loss |
  | `cdf(3.1391327920480296e-17, 1e10, 1)` | 0.0 | 2.885428360068783e-306 | inf | - | loss |
  | `cdf(4.8e-17, 1e10, 1)` | 2.718453729790811e-299 | 2.7184537297911605e-299 | 579 | - | minor |

- **Fix direction:** in the far left tail use `logsf = -exp(log(c) + log_ndtr(z))`
  (first order in c*Phi(z); the next term is O((c Phi(z))**2)), and
  `cdf = -expm1(logsf)` from that.

### F7: logpdf terms cancel for small c in the right tail

- **Code:** `_logpdf`, lines 8957-8959: `... + _norm_logpdf(np.log(x) / s) + _norm_logcdf(-np.log(x) / s) * (c - 1.)`; `_pdf`, line 8954: `np.exp(self._logpdf(...))`
- **Mechanism:** for large z, `_norm_logpdf(z) ≈ -z**2/2` and
  `(c-1)*log_ndtr(-z) ≈ +(1-c) z**2/2`. Their sum is about -c*z**2/2, which loses
  about 1/c of relative accuracy (pattern 5). logpdf loses up to 365 ulp
  (about z**2/2 divided by |logpdf|). `exp` turns the absolute error of the
  z**2/2-sized terms (~1e5 at x = 1e180) into a relative error of the same
  number of ulp in pdf.
- **Regime:** c << 1, x >> 1 (z = log(x)/s large). Measured for c = 1e-3 (s = 1):
  moderate for x in [1e40, 1e240], up to 1.8e5 ulp. For c = 1e-2: 355 ulp at x = 1e20.
- **Reproducer:** `python accuracy.py refs/powerlognorm_ref.py --at pdf 1e180 0.001 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e+180, 0.001, 1)` | 2.0539772773959644e-218 | 2.0539772773125646e-218 | 1.8e+05 | 1.41 | moderate |
  | `pdf(1e+40, 0.001, 1)` | 1.3179391812523632e-43 | 1.3179391812506349e-43 | 5.91e+03 | 1.08 | moderate |
  | `pdf(1e+20, 0.01, 1)` | 1.0905727415923019e-25 | 1.090572741592216e-25 | 355 | - | minor |
  | `logpdf(1e+180, 0.001, 1)` | -501.2437722238683 | -501.2437722239089 | 365 | 0.00282 | minor |

- **Fix direction:** regroup as `c*log_ndtr(-z) + (_norm_logpdf(z) - log_ndtr(-z))`.
  The bracket is log(phi(z)/Phi(-z)) = `-log(sqrt(pi/2) * sc.erfcx(z/sqrt(2)))`,
  which is O(log z) and accurate.

### F8: far-tail error of log_ndtr in the cdf/logsf left tail

- **Code:** `_logsf`, line 8971: `_norm_logcdf(-np.log(x) / s) * c` (= `sc.log_ndtr(t)` with t = -z > 0)
- **Mechanism:** `log_ndtr(t)` for t > 0 returns `log1p(-ndtr(-t))`, so it
  carries the relative error of `ndtr(-t)` in the far tail. At the
  double argument t = 23.03 that error alone is 305 ulp, and at t = 30 it is 259 ulp
  (measured separately against mpmath at the same float argument). The rest
  comes from rounding z = log(x)/s, covered by cond (arg). This is the same
  shared `scipy.special` issue as lognorm F6.
- **Regime:** log(x)/s in about [-38, -8], any c (independent of c).
- **Reproducer:** `python accuracy.py refs/powerlognorm_ref.py --at cdf 1e-10 2 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-10, 2, 1)` | 2.5683512612868638e-117 | 2.5683512612870617e-117 | 347 | - | minor |
  | `logsf(1e-10, 2, 1)` | -2.5683512612868638e-117 | -2.5683512612870617e-117 | 347 | - | minor |
  | `cdf(3.059023205018258e-07, 1000, 0.5)` | 4.906713927147908e-195 | 4.9067139271481845e-195 | 254 | 60.1 | minor |

- **Fix direction:** in `scipy.special`, a more accurate far-tail `ndtr`
  (shared). Nothing to change in powerlognorm itself.

### F9: exp of a large log value in pdf and sf

- **Code:** `_pdf`, line 8954: `np.exp(self._logpdf(x, c, s))`; `_sf`, line 8968: `np.exp(self._logsf(x, c, s))`
- **Mechanism:** a correctly rounded logpdf of size L still has an absolute error of
  up to L*eps/2, and `exp` turns it into a relative error of about L/2 ulp
  (up to ~350 ulp for L ~ 700). Errors in `log_ndtr` add to that.
- **Regime:** |logpdf| or |logsf| in the hundreds: large |log x| with large s,
  and the far tails generally. Also pdf at x = 1 for tiny s, where logpdf = 348.7
  (75 ulp).
- **Reproducer:** `python accuracy.py refs/powerlognorm_ref.py --at pdf 1e280 2 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e+280, 2, 1000)` | 1.6823057284131905e-284 | 1.6823057284133628e-284 | 461 | 1 | minor |
  | `sf(1e+280, 0.001, 1)` | 5.438208945608438e-91 | 5.438208945608795e-91 | 296 | 0.645 | minor |

- **Fix direction:** inherent in the exp(log) construction. A direct product
  form would avoid it only where nothing over- or underflows; it is probably
  not worth changing.

## RuntimeWarnings

- pdf, sf: `underflow encountered in exp` wherever the result underflows (true
  value below the normal range): spurious, the results are correct.
- cdf, sf, logsf: `underflow encountered in multiply` for c = 1e-3 in the left tail
  (`log_ndtr(t)*c` with subnormal `log_ndtr`); the true values are not normal doubles, so this is spurious.
- logpdf, pdf: `overflow encountered in square` and `invalid value encountered in add`
  for s = 1e-152. These come with the F4/F5 losses, and also at points where
  the true value also overflows (correct -inf/0).
- ppf, isf: `underflow encountered in power` comes with F2 (`q**(1/c)` underflow). The values
  are wrong (inf/0) unless the true quantile also overflows. `overflow/underflow encountered in exp`
  for s = 1000 at extreme q, where the true quantile is also 0/inf (spurious).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf, logsf | bulk and right tail (cdf -> 1, logsf large negative), all parameter sets | 1.96 ulp (cdf bulk), 33 ulp (logsf right tail, within cond) |
| sf | left tail (sf -> 1), bulk | 8.9 ulp |
| logpdf | all parameter sets except F4/F5/F7 regimes, including c = 1e6, 1e20, s = 1e3, s = 1e-3 | 59.6 ulp (bulk, minor, c = 1e-3 at x = 1e40; F7 mechanism) / ~35 ulp elsewhere (within cond) |
| pdf | moderate c, all x (beyond F7/F9) | 583 ulp at (5, 0.001), within cond (arg) 3.75e4 |
| ppf, isf | moderate c (2.14) and s = 1000, bulk | 701 ulp (s = 1000, cond 4.8e3) |

## Suspected, not confirmed

- In the subnormal band of `log_ndtr(-z)` (t between 37.5 and 37.7), cdf/logsf
  lose bits even for moderate c whenever c*Phi(z) is normal. This needs
  c > ~1e8, so it is the same finding as F6. With c = 1e10 at x = 4.8e-17 the error is only 579 ulp,
  part of which is F8. No moderate-c point exists where the result is normal.
- logpdf for 1.9e154 < z < 2.68e154 with c = 0.5 is covered by F5 (nan measured). With c = 1
  exactly, `(c-1)*log_ndtr(-z)` is `0*(-inf) = nan` for z > 1.9e154, but
  there the true logpdf (-z**2/2) is also below -1.8e308, so this cannot be measured.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | `log1mexp(logsf)` = `sc._ufuncs._log1mexp(c*log_ndtr(-z))` for moderate z; for very negative z (where `log_ndtr(-z)` underflows) `log(c) + log_ndtr(z)` (+ O(c Phi(z))) | left tail: cdf underflows for z < ~-38 (c = 2). `logcdf(4.248354255291589e-18, 2, 1)` = -inf vs reference -803.9152948331938 (harness `--at logcdf`, loss) |

## Shared code

- `_norm_logpdf` (line 367): `-x**2/2` overflows for |x| > 1.34e154 (F4). Used by norm,
  lognorm-related code, powernorm, exponnorm and others (INDEX item 2).
- `sc.log_ndtr(t)` for t > 0 / `sc.ndtr` far tail: 259-305 ulp at the exact double
  argument (F8). This is the same issue as lognorm F6 / INDEX item 6.
- `sc.ndtri(q**(1/c))` (F2, F3) is not a special-function bug. The fix uses the existing
  `sc.ndtri_exp`. powernorm uses the same `q**(1/c)` construction and should be
  checked for the same F1-F3 behaviour.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(1e-17, 2.1413923530064087, 0.44639540782048337)` | 0.0 | 0.021690571101027959 | verified (loss) | Reference by root-finding c*logPhi(-z) = log1p(-q) (erfc/log1p route); cdf(x_ref) = q to 1e-200. Also checked: q = 1e-17 (1, 1) and 1e-300 give 0.0; q = 1e-10 gives 3.68e8 ulp, q = 3.48e-14 gives 1.05e12 ulp (c = 2.14) and 2.72e9 ulp (5, 0.001). Line 8965 `self._isf(1 - q, c, s)` confirmed; `1 - 1e-17 == 1.0` |
| F2 | `isf(0.1, 0.001, 1)` | inf | 2.7475727890520266e+29 | verified (loss) | `0.1**(1/0.001)` = 0.0 (log q / c = -2302.6), line 8974. Also checked: ppf(0.9, 0.001, 1), isf(1e-170, 0.5, 1), isf(1e-200, 0.5, 1e-152) give inf (refs 2.7475727890520379e+29, 1.3626402522782349e+17, 1.0); isf(0.4752, 0.001, 1) has 3.18e13 ulp |
| F3 | `isf(0.5, 1e20, 1)` | 0.0 | 9.1297800709565627e-05 | verified (loss) | `0.5**(1/1e20) == 1.0`, `ndtri(1.0) = inf`, line 8974. Also checked: ppf(0.5, 1e20, 1) and isf(0.9999999999999627, 1000, 0.5) give 0.0; c = 1e16 has 2.61e14 ulp, c = 1e10 has 5.24e8 ulp, isf(0.99, 1e6, 1) has 1.33e6 ulp |
| F4 | `logpdf(1e-80, 0.5, 1e-152)` | -inf | -1.6966073953530871e+308 | verified (loss) | Reference uses an erfc-based logPhi, with an asymptotic series for \|w\| > 1e10. Also checked: logpdf(1e70, 0.5, 1e-152) and logpdf(1.4e65, 1, 1e-152) give -inf. `x**2` in `_norm_logpdf` (line 367) overflows (1.4e154**2 = inf) |
| F5 | `logsf(1e100, 0.5, 1e-152)` | -inf | -1.3254745276195993e+308 | verified (loss) | `log_ndtr(-1.95e154) = -inf` before `* c` (line 8971). logpdf at x = 1e100 and x = 7.225973768125954e+86 returns nan (refs -1.3254745276195993e+308, -1.0000000000000002e+308) |
| F6 | `cdf(2.104223637677623e-17, 1e20, 1)` | 0.0 | 6.6015998543267587e-303 | verified (loss) | Confirmed independently by mp.quad of the docstring pdf in log x (agrees to 2e-13, limited by quadrature). logsf gives -0.0. cdf(3.1391327920480296e-17, 1e10, 1) = 0.0 vs 2.8854283600687831e-306. `log_ndtr(37.8) = -0.0`, `log_ndtr(37.6)` is subnormal (line 8971) |
| F7 | `pdf(1e180, 0.001, 1)` | 2.0539772773959644e-218 | 2.0539772773125645e-218 | verified (moderate) | 1.83e5 ulp. Also checked: pdf(1e40, 0.001, 1) has 5.91e3 ulp, logpdf(1e180, 0.001, 1) has 365 ulp. Mechanism matches lines 8957-8959 (cancellation of -z**2/2 against (c-1)*log_ndtr(-z)) |

All references were computed at 100 and 200 digits and agree to at least 1e-96 relative, except isf(1e-200, 0.5, 1e-152): there the value is 1 + O(1e-150) and the two precisions agree to 4e-151. Scripts: `scratch/powerlognorm/verify/verify.py`, `scratch/powerlognorm/verify/quadcheck.py`. The analyst's reference module was not needed: there were no disagreements.

Verified by dist-accuracy-verifier: 7 of 7 verified.
