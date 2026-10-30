# johnsonsu: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `johnsonsu_gen`, lines 5891-5981
- **Implemented:** pdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf)
- **Related:** johnsonsb (sibling with the same structure, see Shared code); helpers: `_norm_pdf`, `_norm_cdf`, `_norm_sf`, `_norm_ppf`, `_norm_isf`; special functions: `sc.ndtr`, `sc.ndtri` (via the helpers), `np.arcsinh`, `np.sinh` (`sc.expm1` is used only in `_stats`)
- **Reference:** `refs/johnsonsu_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (2.554395574161155, 2.2482281679651965), (0, 1), (30, 0.5), (-30, 0.5), (0.5, 0.001), (0, 0.01), (0, 1e6), (-50, 50), (0, 1e100)
- **Worst finding:** loss in pdf

## Summary

cdf and sf are accurate everywhere. They use `ndtr(z)` and `ndtr(-z)` with
z = a + b*arcsinh(x), and `np.arcsinh` is accurate for all x (max 9.6 ulp in
the bulk). The pdf has two total failures. First, `x*x` overflows for
|x| > 1.34e154, so the pdf returns 0 for small b, where the true density
there (~b/|x| * phi(z)) is still a normal double. Second, the johnsonsb F1
pattern: `_norm_pdf(z)` goes subnormal or underflows for |z| > ~37.5 before
it is multiplied by b, so for large b the result is 0 or has 3 to 4 correct
digits (3.9e12 ulp). ppf and isf lose up to 2.8e11 ulp (6e-5 relative) where
ndtri(q) is close to a (x near 0). There the rounding of the double ndtri
result is divided by the small difference ndtri(q) - a. For small b, the
rounding of the sinh argument costs up to 1.1e3 ulp. Errors of a few hundred
ulp (minor, z**2 amplification) appear in pdf/cdf/sf in the normal tails, as
in johnsonsb.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf | \|x\| > 1.34e154 with small b (true pdf normal) | `x*x` overflows, `b/sqrt(inf) = 0` (pattern 3) | 0.0 for 3.9e-304 | loss |
| F2 | pdf | 37.5 < \|z\| with large b (b*phi(z) normal) | `_norm_pdf(z)` subnormal/0 before multiplying by b (pattern 4) | 0.0 for 1.2e-302; 3.9e12 ulp | loss |
| F3 | ppf, isf | ndtri(q) close to a (x near 0), any \|a\| > 0 | `_norm_ppf(q) - a` cancels; rounding of ndtri result amplified (pattern 5) | 2.8e11 ulp | severe |
| F4 | isf, ppf | small b (large \|t\| = \|ndtri(q) - a\|/b) | absolute rounding error of t amplified by \|t\| in sinh (pattern 8) | 1.08e3 ulp | moderate |
| F5 | pdf, cdf, sf | \|z\| ~ 25-38 (large \|a\| in bulk, or tails) | rounding of z amplified by ~z**2 in the normal tail (pattern 6-like) | 804 ulp | minor |

### F1: pdf returns 0 for |x| > 1.34e154 because x*x overflows

- **Code:** `_pdf`, lines 5938-5940: `x2 = x*x`; `return b*1.0/np.sqrt(x2+1.0)*trm`.
- **Mechanism:** pattern 3. For |x| > 1.34e154, `x*x` is inf, `np.sqrt(inf)`
  is inf, and `b/inf` is 0, so the pdf is 0 whatever `trm` is. The true pdf
  is about b/|x| * phi(a + b*log(2|x|)). For small b the normal factor stays
  moderate (z = 0.5 - 0.001*690 = -0.19 at x = -1e300 for (0.5, 0.001)), so
  the true pdf is ~b/|x|, normal down to |x| ~ b*1e308. Every point with
  |x| > 1.34e154 in sets (0.5, 0.001) and (0, 0.01) is a loss. Emits
  `overflow encountered in multiply`.
- **Regime:** |x| > 1.34e154 with b*|log(2|x|)| small enough that the true
  pdf is above 2.2e-308. In practice this means b below ~0.05 (heavy-tailed
  members); for b ~ 1 the true pdf is already subnormal there.
- **Reproducer:** `python accuracy.py refs/johnsonsu_ref.py --at pdf 1e200 0.5 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(-1e300, 0.5, 0.001)` | 0.0 | 3.916962312129173e-304 | inf | 1 | loss |
  | `pdf(1.35e154, 0.5, 0.001)` | 0.0 | 2.0493520766613274e-158 | inf | 1 | loss |
  | `pdf(1e200, 0.5, 0.001)` | 0.0 | 2.5135197668451642e-204 | inf | 1 | loss |
  | `pdf(1e200, 0, 0.01)` | 0.0 | 9.59105769101805e-208 | inf | ~1 | loss |

- **Fix direction:** use `np.hypot(x, 1.0)` (or `1/|x| / sqrt(1 + 1/x**2)`
  for large |x|) instead of `np.sqrt(x*x + 1.0)`.

### F2: pdf underflows in the normal factor before the factor b is applied

- **Code:** `_pdf`, lines 5939-5940: `trm = _norm_pdf(a + b * np.arcsinh(x))`;
  `return b*1.0/np.sqrt(x2+1.0)*trm`, with
  `_norm_pdf(x) = np.exp(-x**2/2.0) / _norm_pdf_C` (line 363).
- **Mechanism:** pattern 4, the same as johnsonsb F1. phi(z) drops below
  2.2e-308 for |z| > ~37.5 and to 0 for |z| > ~38.6. The prefactor
  b/sqrt(1 + x**2) is at most b, so the effect needs large b. With b = 1e100,
  x ~ 4e-99 gives |z| ~ 38 to 43, where the true pdf (1e100 * phi(z)) is
  1e-214 to 1e-302, but phi(z) is subnormal (few significant bits) or 0.
  At z = 38.4, phi = 2.5e-321 has about 10 significant bits, so the relative
  error is 8.6e-4. At z = 38.8 and beyond the pdf is 0.0.
- **Regime:** 37.5 < |z| and b*phi(z)/sqrt(1+x**2) > 2.2e-308, which needs
  b > ~1. The band is |z| in (37.5, sqrt(2*(708 + ln b))): empty for b = 50,
  just reachable at b = 1e6 (true pdf at most ~1e-308), and |z| up to 43 for
  b = 1e100.
- **Reproducer:** `python accuracy.py refs/johnsonsu_ref.py --at pdf 4.3e-99 0 1e100`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(4.3e-99, 0, 1e100)` | 0.0 | 1.2464117838545492e-302 | inf | 1849 | loss |
  | `pdf(3.88e-99, 0, 1e100)` | 0.0 | 4.997670824716177e-228 | inf | 1505 | loss |
  | `pdf(3.84e-99, 0, 1e100)` | 2.534556763165595e-221 | 2.536731186788674e-221 | 3.9e12 | 1475 | severe |
  | `pdf(3.8e-99, 0, 1e100)` | 1.0972210519949712e-214 | 1.0972210520077055e-214 | 5.2e4 | 1444 | moderate |

  (`x` values are the doubles nearest 3.84e-99 etc.; the harness prints them
  as 3.8399999999999996e-99.)
- **Fix direction:** log space for large |z|:
  `exp(log(b) - 0.5*log1p(x**2) + _norm_logpdf(z))` (with the F1 fix for
  the log1p term).

### F3: ppf/isf lose digits where ndtri(q) is close to a (x near 0)

- **Code:** `_ppf`, line 5946: `np.sinh((_norm_ppf(q) - a) / b)`;
  `_isf`, line 5952: `np.sinh((_norm_isf(x) - a) / b)`.
- **Mechanism:** pattern 5. Near q0 = Phi(a) (ppf) or 1 - Phi(a) (isf), the
  true x ~ (z - a)/b is small, and the difference `z - a` loses digits. The
  double z = ndtri(q) has absolute error ~0.3 ulp * |a|. Dividing by z - a
  turns this into a relative error of eps*|a|/|z - a| in x. The conditioning
  w.r.t. q near q0 is only Phi(a)/(phi(a)*|z - a|) ~ 1/(|a|*|z - a|) for
  a << 0, so the error exceeds it by a factor ~a**2 (here 900 for |a| = 30,
  ~7 at the example parameters). Checked in scratch at
  `ppf(4.906713932054622e-198, -30, 0.5)`: scipy's ndtri is 0.31 ulp off;
  even a correctly rounded z gives a relative error of 4.5e-5 (scipy:
  6.2e-5). So the loss comes from rounding z to a double, not from
  arithmetic after it.
- **Regime:** q within a relative 1e-6 of Phi(a) (ppf) or of 1 - Phi(a)
  (isf). The error grows like eps*|a|/|z - a|. It appears for every a != 0
  (a = 0 is exact because ndtri(0.5) = 0). With b = 0.5, a = -30 it reaches
  6e-5 relative at x = 6.7e-11.
- **Reproducer:** `python accuracy.py refs/johnsonsu_ref.py --at ppf 4.906713932054622e-198 -30 0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(4.906713932054622e-198, -30, 0.5)` | 6.658495976807899e-11 | 6.658905606357334e-11 | 2.8e11 | 1.0e9 | severe |
  | `ppf(4.906718833861835e-198, -30, 0.5)` | 6.659279705445448e-08 | 6.65928010952469e-08 | 2.7e8 | 1.0e6 | severe |
  | `isf(4.906713932054622e-198, 30, 0.5)` | -6.658495976807899e-11 | -6.658905606357334e-11 | 2.8e11 | 1.0e9 | severe |
  | `isf(0.005318616581433098, 2.554395574161155, 2.2482281679651965)` | 1.5485348038455645e-10 | 1.5485324351775996e-10 | 6.9e9 | 1.0e9 | severe |
  | `isf(0.005318611268135127, 2.554395574161155, 2.2482281679651965)` | 1.5485329075716623e-07 | 1.5485329070934285e-07 | 1.4e6 | 1.0e6 | severe |

  (cond (shapes) is 9.0e11 and 7.3e9 at the two worst points, above the
  error. a and b are exact user inputs, so this does not excuse it.)
- **Fix direction:** near q0 refine with one Newton step on the cdf,
  x1 = x0 - (ndtr(a + b*asinh(x0)) - q)/pdf(x0). The residual q - Phi(...)
  carries only conditioning-sized error. Alternatively, use the expansion
  x ~ (q - ndtr(a))/(b*phi(a)).

### F4: rounding of the sinh argument amplified for small b

- **Code:** `_ppf`, line 5946 and `_isf`, line 5952:
  `np.sinh((_norm_ppf(q) - a) / b)`.
- **Mechanism:** pattern 8. For large |t|, t = (ndtri(q) - a)/b, sinh(t)
  has relative sensitivity |t| to t. t carries absolute error from the ndtri
  rounding (scaled by 1/b), the subtraction and the division (~eps*|t| each),
  which becomes up to ~|t| ulp in x. The harness condition number covers only
  the ndtri input part (q*phi'/...), not the roundings of the subtraction and
  division. At `isf(0.15, 0.5, 0.001)`, t = 536: 1.08e3 ulp vs cond 643.
- **Regime:** small b (b <= 0.01 here) with |t| in the hundreds; the error is
  bounded by roughly |t| <= 710 ulp plus the ndtri part before sinh
  overflows.
- **Reproducer:** `python accuracy.py refs/johnsonsu_ref.py --at isf 0.15 0.5 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.15, 0.5, 0.001)` | 4.666926596337987e+232 | 4.6669265963368677e+232 | 1.08e3 | 643 | moderate |
  | `ppf(1e-08, 0, 0.01)` | -2.6612598523392374e+243 | -2.6612598523390107e+243 | 384 | 17.3 | minor |
  | `ppf(1e-12, 0, 0.01)` | -1.594853665674414e+305 | -1.5948536656745045e+305 | 255 | 13.9 | minor |
  | `ppf(2.6101572156826508e-126, 30, 0.5)` | -3.137330318703137e+46 | -3.137330318703182e+46 | 64.8 | 0.0836 | minor |

  (cond (shapes) 1.04e3 at the moderate point, comparable to the error.)
- **Fix direction:** largely inherent. `(z - a) * (1/b)` vs `/ b` makes no
  difference; only extended-precision t (two-sum of z - a) would help, so
  this is low priority.

### F5: rounding of z amplified by ~z**2 in the normal tails (pdf, cdf, sf)

- **Code:** lines 5939 (`_norm_pdf(a + b * np.arcsinh(x))`), 5943
  (`_norm_cdf(a + b * np.arcsinh(x))`), 5949 (`_norm_sf(a + b * np.arcsinh(x))`).
- **Mechanism:** as johnsonsb F2. z is rounded (absolute error ~eps*|z|),
  and phi(z) and Phi(-|z|) have relative sensitivity ~z**2, giving ~z**2/4
  ulp (~225 ulp at |z| = 30). The harness conditioning w.r.t. x is only
  ~b*|z|*|x|/sqrt(1+x**2), which is tiny near x = 0 (e.g. 1.5e-9 at x = -1e-10)
  and does not include the rounding of z for a large |a|.
- **Regime:** |z| between ~25 and the underflow at ~38: the whole bulk for
  |a| = 30, b = 0.5, and the far tails for the other sets. Bounded by
  ~900 ulp before the result underflows.
- **Reproducer:** `python accuracy.py refs/johnsonsu_ref.py --at cdf 2.0 -30 0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(2.0, -30, 0.5)` | 9.832172839852132e-189 | 9.832172839853888e-189 | 804 | 13.1 | minor |
  | `sf(-2.0, 30, 0.5)` | 9.832172839852132e-189 | 9.832172839853888e-189 | 804 | 13.1 | minor |
  | `cdf(-1e-10, -30, 0.5)` | 4.906713919779477e-198 | 4.9067139197799565e-198 | 440 | 1.5e-09 | minor |
  | `cdf(-10000000000.0, 0, 1)` | 1.1481263119700735e-124 | 1.148126311970144e-124 | 277 | 23.8 | minor |

  pdf is minor in the same regime (max 389 ulp for (30, 0.5)/(-30, 0.5),
  251 ulp for the example set). cond (shapes) is ~900 at these points,
  comparable to the error.
- **Fix direction:** only via extra precision for z (double-double
  `a + b*arcsinh(x)` and a split `exp(-z**2/2)`); low priority.

## RuntimeWarnings

- pdf: `overflow encountered in multiply` from `x*x` for |x| > 1.34e154.
  These accompany the F1 losses for small b; for b ~ 1 the true value there
  is subnormal and the 0 is acceptable. `underflow encountered in exp /
  multiply / divide / square` where phi(z) underflows: spurious where the
  true pdf is subnormal, real (F2) for large b.
- cdf, sf, pdf: `underflow encountered in arcsinh` for tiny |x| (1e-300 and
  similar). Spurious: arcsinh(x) = x is returned accurately, and cdf/sf are
  correct there.
- ppf, isf: `overflow encountered in sinh` for (0.5, 0.001) and (0, 0.01)
  where the true quantile exceeds 1.8e308. The returned +-inf is correct.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | bulk, all parameter sets; upper tail (0 ulp); |x| up to 1e300 both signs, tiny b and huge b | 9.55 ulp bulk; low tail only F5 (804 ulp minor) |
| sf | bulk; lower-tail region (0 ulp); |x| up to 1e300 both signs | 9 ulp bulk; high tail only F5 (1.34e3 ulp at (-50, 50), within cond 1.38e3) |
| pdf | bulk for moderate b; |x| < 1.34e154; (0, 1e6) where phi(z) is near underflow; (-50, 50) | 213 ulp (0, 1e6, cond 900), 1.09e3 ulp (-50, 50, cond 1.38e3), within conditioning |
| ppf, isf | q in [1e-300, 1 - 2**-53], all sets; high tail | 329 ulp high tail, within conditioning; q = 1/2 with a = 0 exact |

Hypotheses checked: `np.arcsinh` is accurate for huge |x| and tiny x (no
`log(x + sqrt(x**2+1))` cancellation in SciPy); sf/cdf use the complementary
`ndtr(-z)`, and isf uses `-ndtri(q)` rather than `ndtri(1 - q)`, so there
are no pattern 1 losses.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `log(b) - 0.5*log1p(x**2) + _norm_logpdf(z)` (large \|x\|: `-log\|x\| - 0.5*log1p(x**-2)`) | \|z\| > ~38.6 or \|x\| > 1.34e154: `logpdf(sinh(40), 0, 1)` = -inf, true -840.2257913526447 |
| logcdf | log(cdf) | `sc.log_ndtr(a + b*arcsinh(x))` | z < ~-38.4: `logcdf(-1.1769263341851e17, 0, 1)` = -inf, true -804.6084420137538 (log_ndtr gives -804.6084420137539) |
| logsf | log(sf) | `sc.log_ndtr(-(a + b*arcsinh(x)))` | z > ~38.4: `logsf(1.1769263341851e17, 0, 1)` = -inf, true -804.6084420137538 |

## Shared code

- Same pattern as johnsonsb F1: both distributions compute `_norm_pdf(z)`
  (line 363, `np.exp(-x**2/2.0) / _norm_pdf_C`) and then multiply by a
  Jacobian factor (here b/sqrt(1+x**2), at most b). The underflow is a
  property of how the distributions use the helper, not a bug in
  `_norm_pdf`. The shared fix is `exp(log(jacobian) + _norm_logpdf(z))` when
  |z| is large. F5 is the same z**2 amplification as johnsonsb F2.
- `sc.ndtri`, `sc.ndtr`: accurate. The F3 error is caused by rounding the
  ndtri result to a double, not by ndtri itself (0.31 ulp error measured).

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(-1e300, 0.5, 0.001)` | 0.0 | 3.916962312129173e-304 | verified | docstring pdf with log1p form of log(x+sqrt(x^2+1)) and odd symmetry; 100/200 digits agree |
| F1 | `pdf(1e200, 0, 0.01)` | 0.0 | 9.5910576910180499e-208 | verified | `x*x` overflow at line 5938 confirmed |
| F2 | `pdf(4.3e-99, 0, 1e100)` | 0.0 | 1.2464117838545493e-302 | verified | |
| F2 | `pdf(3.84e-99, 0, 1e100)` | 2.534556763165595e-221 | 2.536731186787883e-221 | verified (3.86e12 ulp) | report's reference 2.536731186788674e-221 is for x = 3.8399999999999996e-99 (the next double below 3.84e-99), not the double nearest 3.84e-99; at that x I get the same value. SciPy returns the same value at both x, so severity does not change. 3.88e-99 row: same one-double offset |
| F2 | `pdf(3.8e-99, 0, 1e100)` | 1.0972210519949712e-214 | 1.0972210520077054e-214 | verified (5.23e4 ulp) | |
| F3 | `ppf(4.906713932054622e-198, -30, 0.5)` | 6.658495976807899e-11 | 6.6589056063573339e-11 | verified (2.77e11 ulp) | ref from findroot on log Phi(z) = log q, checked by Phi(a) + quad(pdf, 0, x) = q (rel 4e-23). Mechanism checked: correctly rounded z gives 4.5e-5 rel, scipy ndtri (-0.31 ulp) gives 6.2e-5 |
| F3 | `isf(4.906713932054622e-198, 30, 0.5)` | -6.658495976807899e-11 | -6.6589056063573339e-11 | verified (2.77e11 ulp) | checked by Phi(-a) - quad(pdf, 0, x) = q |
| F3 | `isf(0.005318616581433098, 2.554395574161155, 2.2482281679651965)` | 1.5485348038455645e-10 | 1.5485324351775995e-10 | verified (6.89e9 ulp) | |
| F4 | `isf(0.15, 0.5, 0.001)` | 4.666926596337987e+232 | 4.6669265963368678e+232 | verified (1.08e3 ulp) | |
| F4 | `ppf(1e-08, 0, 0.01)` | -2.6612598523392374e+243 | -2.6612598523390107e+243 | verified (384 ulp) | |

Verified by dist-accuracy-verifier: 4 of 4 verified.
