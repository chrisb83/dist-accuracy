# genexpon: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `genexpon_gen`, lines 3288-3349
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf = log(cdf) / log1p(-sf) (median split), logsf = log(sf) / log1p(-cdf) (median split)
- **Related:** none; helpers: none; special functions: sc.expm1, sc.lambertw
- **Reference:** `refs/genexpon_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (9.1326, 16.232, 3.28196), (1, 1, 1), (0.001, 1, 1), (1e-06, 10, 2), (1, 1e-06, 1), (0.5, 2, 1000), (1, 2, 0.001), (1000, 10000, 0.1)
- **Worst finding:** loss in ppf, isf, logpdf

## Summary

All methods are built on the exponent E(x) = -(a+b)x + (b/c)(1 - exp(-cx))
(sf = exp(E), cdf = -expm1(E)). pdf, sf and cdf are accurate for moderate
shapes. The quantile functions fail badly: `(t + lambertw(...))/c` cancels
whenever c*x is small. As a result, ppf returns negative or O(1e-16) garbage
for every parameter set once p is below about 1e-16, and loses up to 1e11 ulp
in the bulk for small c or large a, b. isf does the same for q -> 1. logpdf
returns the wrong sign or loses all digits for small x when a = 1, because
`np.log(a + ...)` is used instead of log1p. cdf loses up to 9e6 ulp for small
x when a << b, because the two terms of E cancel. pdf/sf lose up to 2.6e3 ulp
when the two terms of E are large and cancel.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf | c*x << 1: small p (all shapes), or small c / large a, b in the bulk | `t + lambertw(...)` cancels (patterns 5, 8) | -1.1e-16 for 1e-20 | loss |
| F2 | isf | c*x << 1: q -> 1 (all shapes), or small c / large a, b in the bulk | same cancellation as F1 (patterns 5, 8) | 0.0 / negative for 3.3e-10 | loss |
| F3 | logpdf | a = 1, small x | `np.log(1 + b*u)` (pattern 2); for b*c = 1 also first-order cancellation (pattern 5) | 8.3e-18 for -1.5e-20 | loss |
| F4 | cdf | small x (c*x << 1), a << b | -(a+b)x and (b/c)(-expm1(-cx)) cancel (pattern 5) | 9.1e6 ulp | severe |
| F5 | pdf, sf | large (a+b)x and b/c with small result exponent ratio | same cancellation inside E, amplified by exp (pattern 5) | 2.57e3 ulp | moderate |

### F1: ppf cancels in `t + lambertw(...)` when c*x is small

- **Code:** `_ppf`, lines 3339-3341:
  `t = (b - c*np.log1p(-p))/s`; `return (t + sc.lambertw(-b/s * np.exp(-t)).real)/c`
- **Mechanism:** as p -> 0, t -> b/s and the Lambert W value -> -b/s, so
  `t + W` is a difference of two O(b/s) numbers that equals c*x. The rounding
  errors of t, exp(-t) and lambertw (each ~eps*b/s) leave an absolute error
  of ~eps*(b/s)/c in x. The relative error is about eps*(b/s)/(c*x). For p
  below ~eps*a*b/(s*c), the result is pure rounding noise: negative values
  (-1.1e-16) or about 1e-16 where the true quantile is ~p/a. The same holds in
  the bulk whenever c*x is small, i.e. small c or large a+b (x ~ 1/(a+b)).
  This is pattern 5 inside a pattern-8 quantile formula. sc.lambertw itself
  is accurate (spot check: <= 3 ulp away from the branch point; within its
  conditioning near -1/e).
- **Regime:** all shapes for small p (loss for p below ~1e-16 * a*b/(s*c));
  in the bulk, relative error ~ 2e-16*(b/s)/(c*x): severe for
  (1000, 10000, 0.1), moderate for (1, 2, 0.001).
- **Reproducer:** `python accuracy.py refs/genexpon_ref.py --at ppf 1e-20 1 1 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-20, 1, 1, 1)` | -1.1102230246251565e-16 | 1e-20 | inf | 1 | loss |
  | `ppf(1e-300, 9.13259764654189, 16.23195660059063, 3.2819552690843983)` | 3.382809738704596e-17 | 1.094978711099416e-301 | 1.4e+300 | 1 | loss |
  | `ppf(1e-08, 1, 2, 0.001)` | 1.0000222872008635e-08 | 1.00000000499e-08 | 1.0e+11 | 1 | severe |
  | `ppf(0.01, 1000, 10000, 0.1)` | 1.0050285357277033e-05 | 1.0050285349400559e-05 | 3.5e+06 | 1.01 | severe |
  | `ppf(0.01, 1, 2, 0.001)` | 0.010050234846570305 | 0.010050234846619351 | 2.2e+04 | 1.01 | moderate |
  | `ppf(0.9991516571017559, 1000, 10000, 0.1)` | 0.0070473985620644886 | 0.007047398562074629 | 6.48e+03 | 166 | moderate |

  Summary: ppf low tail 186 loss + 6 severe out of 192 points (every
  parameter set); bulk 1 severe, 25 moderate, 17 minor.
- **Fix direction:** with y = c*x, solve a*y + b*(y + expm1(-y)) = -c*log1p(-p)
  directly. The left side is well conditioned and ~ a*y + b*y**2/2 for small
  y; compute `y + expm1(-y)` by a series for small y. Use the Lambert-W
  value as a starting point with one or two Newton steps, or use p/a-type
  series for tiny p.

### F2: isf cancels in the same way for q near 1

- **Code:** `_isf`, lines 3347-3349:
  `t = (b - c*np.log(p))/s`; `return (t + sc.lambertw(-b/s * np.exp(-t)).real)/c`
- **Mechanism:** identical to F1. For q -> 1, log(q) -> 0, so t -> b/s and
  `t + W` cancels to c*x. The result is 0.0, negative, or off by orders of
  magnitude. The argument conditioning here is large (cond ~ 1/(1-q), up to
  3e15), but negative and zero results are losses regardless. The same
  cancellation gives severe/moderate errors in the bulk for small c or large
  a+b, and minor errors (<= 933 ulp) in the low tail of (1000, 10000, 0.1),
  where c*x stays < 0.05.
- **Regime:** q > 1 - ~1e-12 for all shapes (loss/severe); bulk for small c
  or large a+b.
- **Reproducer:** `python accuracy.py refs/genexpon_ref.py --at isf 0.9999999999999627 1 2 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.9999999999999997, 1e-06, 10, 2)` | -1.9982571153320805e-11 | 3.319649004365723e-10 | inf | 3.0e+15 | loss |
  | `isf(0.9999999999999627, 1, 2, 0.001)` | 0.0 | 3.7303493627405954e-14 | inf | 2.7e+13 | loss |
  | `isf(0.9999999999999997, 1000, 10000, 0.1)` | 1.1102230246251565e-14 | 3.33066907387547e-19 | 1.5e+20 | 3.0e+15 | loss |
  | `isf(0.9999999999999966, 9.13259764654189, 16.23195660059063, 3.2819552690843983)` | 3.382809738704596e-16 | 3.768578787264545e-16 | 4.6e+14 | 2.9e+14 | severe |
  | `isf(0.9907894468231052, 1000, 10000, 0.1)` | 9.253189795499495e-06 | 9.253189780546343e-06 | 7.28e+06 | 108 | severe |
  | `isf(0.99, 1000, 10000, 0.1)` | 1.0050285357277033e-05 | 1.0050285349400567e-05 | 3.5e+06 | 99.5 | severe |
  | `isf(3.4807005884285624e-14, 1000, 10000, 0.1)` | 0.03052358665235122 | 0.030523586652357545 | 933 | 0.0318 | minor |

  Summary: isf high tail 9 loss, 33 severe, 6 moderate; bulk 2 severe,
  25 moderate, 16 minor; low tail 27 minor (all (1000, 10000, 0.1)).
- **Fix direction:** same as F1 with `-c*log(q)` on the right side.

### F3: logpdf loses all digits for small x when a = 1

- **Code:** `_logpdf`, line 3333:
  `np.log(a+b*(-sc.expm1(-c*x))) + (-a-b)*x+b*(-sc.expm1(-c*x))/c`
- **Mechanism:** for small x, logpdf ~ log(a) + (b*c/a - a)*x. When a = 1,
  log(a) = 0 and the result is O(x), but `np.log(1 + b*u)` (u = 1 - exp(-cx))
  drops b*u once it falls below eps (pattern 2). The computed value then is
  only the exponent E ~ -x instead of (b*c - 1)*x. The relative error is
  b*c/|b*c - 1|, i.e. 2e-3 for (1, 2, 0.001) and 1e-6 for (1, 1e-6, 1). If
  also b*c = 1 (= a**2), the first-order terms cancel exactly. The true value
  is then -1.5*x**2, and the result is -x or even has the wrong sign
  (pattern 5, a special shape relation like burr's c*d = 1).
- **Regime:** a = 1 (exactly, or so close that |log a| << b*c*x), x small:
  relative error ~ eps/(b*c*x) for b*c*x > eps, total once b*c*x < eps;
  total loss for any x below ~1e-8 when b*c = 1.
- **Reproducer:** `python accuracy.py refs/genexpon_ref.py --at logpdf 1e-10 1 1 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-10, 1, 1, 1)` | 8.264037103045511e-18 | -1.4999999998833333e-20 | inf | 2 | loss |
  | `logpdf(1e-150, 1, 1, 1)` | -1e-150 | -1.5e-300 | 3.0e+165 | 2 | loss |
  | `logpdf(1e-08, 1, 1, 1)` | -1.60774708295963e-16 | -1.4999999883333334e-16 | 3.2e+14 | 2 | severe |
  | `logpdf(1e-70, 1, 2, 0.001)` | -1.0000000000000002e-70 | -9.98e-71 | 9.0e+12 | 1 | severe |
  | `logpdf(1e-10, 1, 1e-06, 1)` | -1e-10 | -9.999990000000001e-11 | 4.5e+09 | 1 | severe |

  Summary: logpdf low tail 15 loss, 63 severe, 4 moderate, all with a = 1.
  `cond (shapes)` is huge here (up to 6.7e+299) because the result is near 0.
- **Fix direction:** `log(a) + log1p(b*u/a)` together with the accurate E of F4.
  For a = 1, b*c = 1, the series -x*(...) of the combined expression is needed.

### F4: cdf cancels in the exponent for small x when a << b

- **Code:** `_cdf`, line 3336: `-sc.expm1((-a-b)*x + b*(-sc.expm1(-c*x))/c)`
- **Mechanism:** for c*x << 1, `-(a+b)*x` and `b*(-expm1(-cx))/c ~ b*x` are
  both ~b*x but their sum is ~ -a*x - b*c*x**2/2 (pattern 5). Each term
  carries a rounding error of ~eps*b*x, so the relative error of E, and of
  cdf ~ -E, is ~eps*(a+b)/(a + b*c*x/2).
- **Regime:** x << 1/c and a << b: ~1e3 ulp for a/b = 1e-3, ~1e7 ulp for
  a/b = 1e-7; persists down to the smallest x (x = 1e-300).
- **Reproducer:** `python accuracy.py refs/genexpon_ref.py --at cdf 1e-190 1e-6 10 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-190, 1e-06, 10, 2)` | 9.999999979840115e-197 | 9.999999999999999e-197 | 9.1e+06 | 1 | severe |
  | `cdf(1e-08, 1e-06, 10, 2)` | 1.099999997372792e-14 | 1.0999999993333272e-14 | 8.0e+06 | 1.09 | severe |
  | `cdf(1e-08, 0.001, 1, 1)` | 1.0000049999947292e-11 | 1.0000049999949834e-11 | 1.14e+03 | 1 | moderate |

  Summary: cdf low tail 25 severe, 9 moderate, 34 minor.
- **Fix direction:** E = -a*x - (b/c)*(c*x + expm1(-c*x)), with
  `y + expm1(-y)` from a series (y**2/2 - y**3/6 + ...) for small y.

### F5: pdf and sf lose accuracy when the two terms of E are large

- **Code:** `_pdf`, lines 3329-3330, and `_sf`, line 3344: `np.exp((-a-b)*x + b*(-sc.expm1(-c*x))/c)`
- **Mechanism:** the same cancellation as F4, now at moderate c*x. At
  (1000, 10000, 0.1), x = 0.5: -(a+b)*x = -5500 and b/c*(1 - exp(-0.05)) = 4877
  sum to -623. The absolute error ~eps*5500 becomes a relative error of
  several thousand ulp after `exp` (pattern 5). The argument conditioning
  (|x*E'|, 744 here) covers part of it.
- **Regime:** (a+b)*x large, c*x not large (b/c comparable to (a+b)*x);
  tail values close to underflow.
- **Reproducer:** `python accuracy.py refs/genexpon_ref.py --at pdf 0.5 1000 10000 0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.5, 1000, 10000, 0.1)` | 4.2859631598062153e-268 | 4.2859631598037653e-268 | 2.57e+03 | 744 | moderate |
  | `sf(0.5, 1000, 10000, 0.1)` | 2.8809212745344157e-271 | 2.880921274532769e-271 | 2.57e+03 | 744 | moderate |
  | `pdf(0.06772727272727273, 1000, 10000, 0.1)` | 4.178115892719847e-28 | 4.178115892720531e-28 | 737 | 72.2 | minor |
  | `pdf(253.33333333333334, 1, 2, 0.001)` | 3.0139828995550343e-136 | 3.0139828995547708e-136 | 394 | 366 | minor |

  Summary: pdf and sf high tail 1 moderate, 5 minor each; all other errors
  are within conditioning.
- **Fix direction:** the same E = -a*x - (b/c)*(c*x + expm1(-c*x)) form as F4.
  It has no large cancelling terms because both parts are negative.

## RuntimeWarnings

- pdf, sf (all 8 parameter sets): "underflow encountered in exp" (pdf also
  "in multiply") at large x where the true value is below the normal range.
  This is expected underflow and not an accuracy issue.
- ppf, isf for (0.5, 2, 1000): "underflow encountered in exp" from
  `np.exp(-t)` when t > 745 (isf with tiny q, ppf with q near 1, c/s large).
  The result is then W(0) = 0, x = t/c, which is the correct limit. These
  points measure as accurate, so the warning is spurious. (The harness
  attributes the warning to the whole ppf call for that set, including the
  small-p points that fail for the F1 reason.)

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | low tail, bulk (all sets) | 12.7 ulp (bulk) |
| sf | low tail, bulk (all sets) | 12.8 ulp |
| cdf | bulk, high tail (all sets) | 18.2 ulp (bulk, 1 minor); high tail 0 ulp |
| logpdf | bulk, high tail; low tail for a != 1 | bulk 457 ulp (1 minor, (1e-06, 10, 2), cond 571); high tail 15.2 ulp (2 minor) |
| ppf | high tail except (1000, 10000, 0.1) and (1, 2, 0.001) (F1) | within conditioning |
| isf | low tail except (1000, 10000, 0.1) (F2) | within conditioning, including exp(-t) underflow for (0.5, 2, 1000) |
| sc.lambertw | spot check vs mpmath at z = -0.3, -0.303, -0.366, -1e-3 | <= 3 ulp; 1.4e4 ulp at z = -1/e*(1 - 1e-10), within its conditioning ~1/(1+W) |

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | log(sf) for x > median, log1p(-cdf) below | closed form `logsf = E = -a*x - (b/c)*(c*x + expm1(-c*x))` | sf underflows for (a+b)x > ~745: `logsf(1000, 1, 1, 1)` = -inf, true -1999.0 |
| logcdf | log(cdf) below median, log1p(-sf) above | `log(-expm1(E))` (= `_log1mexp(E)`) with the accurate E | default is accurate (`logcdf(40, 1, 1, 1)` = -4.906094730649281e-35, true -4.90609473064928e-35) but inherits F4 for small x, a << b |

## Shared code

none (sc.lambertw and sc.expm1 are accurate; all problems are in the
distribution's own formulas)

## Existing GitHub issues

none found (searched "genexpon"; the current Lambert-W `_ppf`/`_isf` of F1/F2 came from PR #17708, closed/merged, which does not discuss small-c*x cancellation)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(1e-20, 1, 1, 1)` | -1.1102230246251565e-16 | 9.9999999999999995e-21 | verified | Newton on cancellation-free E = -a*x - (b/c)*(cx + expm1(-cx)) (series for small cx), checked by x*quad(pdf(x*u)) = p; other 5 rows also reproduced (1.4e300, 1.0e11, 3.5e6, 2.2e4, 6.48e3 ulp). Mechanism matches line 3341 |
| F2 | `isf(0.9999999999999627, 1, 2, 0.001)` | 0.0 | 3.7303493627405954e-14 | verified | same root solve on -E = -log(q), checked by quad of pdf over [x, inf); all 7 rows reproduced (incl. negative result for (1e-06, 10, 2), 933 ulp minor row). Mechanism matches line 3349 |
| F3 | `logpdf(1e-10, 1, 1, 1)` | 8.264037103045511e-18 | -1.4999999998833334e-20 | verified | log(a) + log1p(b*u/a) + E (1e-150 point at 300/400 digits); wrong sign confirmed; all 5 rows reproduced. Mechanism matches line 3333 |
| F4 | `cdf(1e-190, 1e-06, 10, 2)` | 9.999999979840115e-197 | 9.9999999999999997e-197 | verified | -expm1(E) with series E, checked by scaled quad of pdf; 9.08e6 ulp; other rows 8.03e6 and 1.14e3 ulp. Mechanism matches line 3336 |
| F5 | `pdf(0.5, 1000, 10000, 0.1)` | 4.2859631598062153e-268 | 4.2859631598037656e-268 | verified | 2.57e3 ulp; sf(0.5, ...) also 2.57e3 ulp (quad of pdf agrees with exp(E) to 1.7e-11); minor rows 737 and 393 ulp reproduced. Mechanism matches lines 3329-3330, 3344 |

Verified by dist-accuracy-verifier: 5 of 5 verified.
