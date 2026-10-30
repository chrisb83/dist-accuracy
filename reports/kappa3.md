# kappa3: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `kappa3_gen`, lines 7532-7606
- **Implemented:** pdf (7568-7570), cdf (7572-7573), sf (7575-7589, `1 - cdf` with a log1p/expm1 branch for sf < 0.01), ppf (7591-7592), isf (7594-7597)
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf)
- **Related:** none; helpers: none; special functions: sc.expm1, sc.xlog1py
- **Reference:** `refs/kappa3_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (1), (0.001), (1e-06), (1e-12), (0.1), (2), (10), (20), (1000)
- **Worst finding:** loss in cdf, ppf, isf (including the bulk for a = 1000: `cdf(3, 1000) = 0.0`, true 1.0; `ppf(0.49, 1000) = 0.0`, `isf(0.51, 1000) = 0.0`, true 0.4934)

## Summary

For a of order 1 to 10 all five methods are accurate in the bulk, and the
sf branch for small sf is accurate all the way into the right tail. The
methods fail in four ways. (1) `x**a` (cdf), `q**-a` (ppf) and
`expm1(-a*log1p(-q))` (isf) overflow once their exponent passes 709.8, so
cdf returns 0 instead of 1, and ppf and isf return 0. For large a this
happens in the bulk (a = 1000: cdf for x > 2.03, ppf for q < 0.49, isf for
q > 0.51). (2) pdf, cdf and therefore sf raise the rounded `a + x**a` to the
power -1/a, which multiplies the rounding error by about 1/a: 8.6e5 ulp at
a = 1e-6, 9e11 ulp at a = 1e-12, and 36 % relative error at a = 1e-300.
(3) ppf forms `q**-a - 1`, which cancels for q -> 1 and for small a; the
error is then raised to the power 1/a again. It returns inf for q = 1 - 3e-16
and is up to about 3x beyond the conditioning for a = 0.1 and up to 500x for
a = 1e-3. (4) The rounding of the exponent 1/a gives up to about 240 ulp
(minor) when the result has a large logarithm.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | a*log(x) > 709.78 (bulk for a = 1000: x > 2.03) | `x**a` overflows (pattern 3) | 0.0 for 1.0 | loss |
| F2 | ppf | a*log(1/q) > 709.78 (small q; bulk for a = 1000: q < 0.4917) | `q**-a` overflows (patterns 3, 8) | 0.0 for 0.4934 | loss |
| F3 | isf | -a*log1p(-q) > 709.78 (a >= ~19.4 as q -> 1; bulk for a = 1000: q > 0.508) | `expm1` overflows (patterns 3, 8) | 0.0 for 0.4934 | loss |
| F4 | ppf | q -> 1; small a | `q**-a - 1` cancels, amplified by power 1/a (patterns 8, 6) | inf for 5.95e154; 4.2e15 ulp (cond 2.9e15); 9.7e5 ulp (cond 2e3) | loss |
| F5 | pdf, cdf, sf | small a (error ~ 1/a ulp), any x | `(a + x**a)**(-1/a)` amplifies rounding by 1/a (patterns 6, 9) | 9.0e11 ulp at a = 1e-12; 1.6e15 ulp at a = 1e-300 | severe |
| F6 | pdf, cdf, isf | moderate a, results with large log magnitude | rounding of the exponent `1.0/a` (pattern 6) | 242 ulp | minor |

### F1: cdf returns 0 when x**a overflows

- **Code:** `_cdf`, line 7573: `x*(a + x**a)**(-1.0/a)`
- **Mechanism:** for a*log(x) > 709.78, `x**a` is inf, `inf**(-1/a)` is 0 and
  the cdf is `x*0 = 0`. The true cdf is 1 - O(x**-a), i.e. 1.0 in double
  (pattern 3). For large a this is in the bulk: with a = 1000 every x > 2.03
  gives cdf = 0.
- **Regime:** x > exp(709.78/a): x > 4.7e155 for a = 2, x > 1.4e31 for a = 10,
  x > 2.03 for a = 1000.
- **Reproducer:** `python accuracy.py refs/kappa3_ref.py --at cdf 3 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(3, 1000)` | 0.0 | 1.0 | inf | 0 | loss |
  | `cdf(2.1, 1000)` | 0.0 | 1.0 | inf | 6.04e-320 | loss |
  | `cdf(1e160, 2)` | 0.0 | 1.0 | inf | 2e-320 | loss |
  | `cdf(1e40, 10)` | 0.0 | 1.0 | inf | 0 | loss |

  Consequence in sf (not measurable, see "Suspected, not confirmed"):
  `super()._sf` gives `1 - 0 = 1`. That is not below the 0.01 cutoff, so the
  log1p branch is never tried and `sf(3, 1000)` returns 1.0, where the true
  value is 1.07e-401 (subnormal, so the harness does not measure it). The
  inherited `logsf(1e200, 2)` therefore returns 0.0 instead of -920.97 (see
  Opportunities).
- **Fix direction:** for x >= 1 use `cdf = exp(-xlog1py(1/a, a*x**-a))`, i.e.
  `(1 + a*x**-a)**(-1/a)`, which never forms `x**a`.

### F2: ppf returns 0 when q**-a overflows

- **Code:** `_ppf`, line 7592: `(a/(q**-a - 1.0))**(1.0/a)`
- **Mechanism:** for a*log(1/q) > 709.78, `q**-a` is inf, `a/inf = 0` and
  `0**(1/a) = 0`. The true value is about `q*a**(1/a)`, a normal double
  (patterns 3 and 8).
- **Regime:** q < exp(-709.78/a): q < 2e-155 for a = 2, q < 3e-31 for a = 10,
  q < 1.5e-16 for a = 20, q < 0.4917 for a = 1000 (bulk).
- **Reproducer:** `python accuracy.py refs/kappa3_ref.py --at ppf 0.49 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.49, 1000)` | 0.0 | 0.493396517737384 | inf | 1 | loss |
  | `ppf(1e-300, 2)` | 0.0 | 1.414213562373095e-300 | inf | 1 | loss |
  | `ppf(1e-40, 10)` | 0.0 | 1.2589254117941672e-40 | inf | 1 | loss |
  | `ppf(2.7086801531713753e-16, 20)` | 0.0 | 3.1463658914688316e-16 | inf | 1 | loss |

- **Fix direction:** `ppf = q * (a / -expm1(a*log(q)))**(1/a)` (multiply
  through by q**a; `-expm1(a*log(q))` lies in (0, 1]).

### F3: isf returns 0 when expm1 overflows

- **Code:** `_isf`, lines 7595-7597: `lg = sc.xlog1py(-a, -q)`;
  `denom = sc.expm1(lg)`; `(a / denom)**(1.0 / a)`
- **Mechanism:** `lg = -a*log1p(-q)` exceeds 709.78, `expm1(lg)` is inf and
  the result is `0**(1/a) = 0`. The true value is about `(1 - q)*a**(1/a)`
  (patterns 3 and 8). This is the same overflow as F2 in the other tail.
- **Regime:** 1 - q < exp(-709.78/a). Since 1 - q >= 1.1e-16 in double, this
  requires a > ~19.4 (a = 20: q >= 1 - 3e-16); for a = 1000, every q > 0.508.
- **Reproducer:** `python accuracy.py refs/kappa3_ref.py --at isf 0.51 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.51, 1000)` | 0.0 | 0.493396517737384 | inf | 1.04 | loss |
  | `isf(0.9, 1000)` | 0.0 | 0.10069316688518039 | inf | 9 | loss |
  | `isf(0.9999999999999997, 20)` | 0.0 | 3.868859731386983e-16 | inf | 3.0e+15 | loss |

- **Fix direction:** `isf = (1 - q) * (a / -expm1(a*log1p(-q)))**(1/a)`
  (`1 - q` can be formed as `-expm1(log1p(-q))`; it is exact for q >= 0.5).

### F4: ppf cancels in q**-a - 1 (q near 1, small a)

- **Code:** `_ppf`, line 7592: `(a/(q**-a - 1.0))**(1.0/a)`
- **Mechanism:** `q**-a = 1 + a*|log q| + ...`, so subtracting 1 keeps only
  the digits beyond the leading 1 (pattern 8). That relative error is then
  raised to the power 1/a (pattern 6), which multiplies it by 1/a. With
  respect to q, the relative error is about 1/a times the conditioning of the
  ppf near q = 1, and it is large in the bulk for small a. When `q**-a` rounds
  to 1 the result is `a/0 = inf`.
- **Regime:** q -> 1 for a < 1 (a = 0.1: severe for 1 - q < ~1e-5, inf at
  q = 1 - 3e-16). Small a anywhere (a = 1e-3: 2e3 to 9.7e5 ulp across the bulk,
  up to ~500 times the conditioning). For a = 1 the error stays within the
  conditioning (max 1.1e15 ulp, cond 3.0e15).
- **Reproducer:** `python accuracy.py refs/kappa3_ref.py --at ppf 0.9999999 0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999999997, 0.1)` | inf | 5.952304699464506e+154 | inf | 3.0e+16 | loss |
  | `ppf(0.9999999999999966, 0.1)` | 3.351951982485651e+143 | 4.288256588768844e+144 | 4.2e+15 | 2.9e+15 | severe |
  | `ppf(0.99999280314327, 0.1)` | 2.6825896109213652e+51 | 2.6825896086969218e+51 | 3.7e+06 | 1.4e+06 | severe |
  | `ppf(0.9999999, 0.1)` | 9.999993946410968e+69 | 9.999994552636748e+69 | 2.7e+08 | 1.0e+08 | severe |
  | `ppf(0.6, 0.001)` | 4.134251707531436e+291 | 4.13425170664398e+291 | 9.7e+05 | 1.96e+03 | moderate |
  | `ppf(0.5, 0.001)` | 1.0568546507316403e+159 | 1.0568546508972939e+159 | 7.1e+05 | 1.44e+03 | moderate |
  | `ppf(0.36, 0.001)` | 2.988565026553367e-10 | 2.988565026282682e-10 | 4.1e+05 | 979 | moderate |

  (cond (shapes) is 333 and 367 at the worst points, small compared with the
  error.)
- **Fix direction:** `denom = expm1(-a*log(q))` (accurate for q near 1 and for
  small a), and evaluate the power in log form,
  `exp((log(a) - log(denom))/a)`. Combined with F2:
  `q * (a / -expm1(a*log(q)))**(1/a)`.

### F5: pdf, cdf and sf amplify rounding by 1/a for small a

- **Code:** `_pdf`, line 7570: `a*(a + x**a)**(-1.0/a-1)`; `_cdf`, line 7573:
  `x*(a + x**a)**(-1.0/a)`; `_sf`, line 7577: `super()._sf(x, a)` = `1 - cdf`.
- **Mechanism:** for small a, `x**a = 1 + a*log(x) + ...` and
  `a + x**a = 1 + a*(1 + log x) + ...`. The rounding error of this sum
  (about eps, relative) is raised to the power -1/a, which multiplies it by
  1/a (patterns 6 and 9). The information carried by `a*(1 + log x)` is lost
  once it falls below eps. At a = 1e-300 the sum is exactly 1.0, so
  `cdf = x`, `pdf = a` and `sf = 1 - x`, while the true cdf tends to e**-1.
  The conditioning with respect to x is O(1) for pdf and below 1 for cdf and
  sf, so the error is all computational. sf (sf > 0.01 here, so the
  `1 - cdf` path is used) inherits the cdf error.
- **Regime:** all x for small a; error about 0.9/a ulp (a = 1e-3: 881 ulp,
  minor; a = 1e-6: 8.6e5 ulp, moderate; a = 1e-12: 9.0e11 ulp, severe).
- **Reproducer:** `python accuracy.py refs/kappa3_ref.py --at cdf 8.5 1e-12`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(5.5, 1e-12)` | 6.687384284314276e-14 | 6.688717112204697e-14 | 9.0e+11 | 1 | severe |
  | `pdf(8.5, 1e-12)` | 4.327609812016719e-14 | 4.327993425544216e-14 | 4.0e+11 | 1 | severe |
  | `cdf(8.5, 1e-12)` | 0.3678468340225762 | 0.36787944117241356 | 4.0e+11 | 1e-12 | severe |
  | `sf(8.5, 1e-12)` | 0.6321531659774238 | 0.6321205588275864 | 2.3e+11 | 5.82e-13 | severe |
  | `cdf(0.5, 1e-300)` | 0.5 | 0.36787944117144233 | 1.6e+15 | 3.07e-301 | severe |
  | `pdf(0.5, 1e-300)` | 1e-300 | 7.3575888234288465e-301 | 1.6e+15 | 1 | severe |
  | `sf(0.5, 1e-300)` | 0.5 | 0.6321205588285577 | 9.4e+14 | 1.79e-301 | severe |
  | `pdf(8.5, 1e-06)` | 4.3279912623366975e-08 | 4.3279912615410186e-08 | 8.3e+05 | 1 | moderate |
  | `cdf(8.5, 1e-06)` | 0.3678804124646698 | 0.3678804123970369 | 8.3e+05 | 1e-06 | moderate |
  | `sf(8.5, 1e-06)` | 0.6321195875353303 | 0.632119587602963 | 4.8e+05 | 5.82e-07 | moderate |
  | `pdf(8.5, 0.001)` | 4.325820801404127e-05 | 4.325820801403281e-05 | 881 | 1 | minor |

  (The relative error of cdf(0.5, 1e-300) is 0.36, below 1, so the tool
  rates it severe, not loss.)
- **Fix direction:** `log(a + x**a) = a*log(x) + log1p(a*x**-a)`, giving
  `cdf = exp(-log1p(a*x**-a)/a)` and
  `pdf = a * x**(-a-1) * exp(-(1/a + 1)*log1p(a*x**-a))`. Where `x**-a`
  overflows (x < 1, large a), use `log1p(x**a/a) + log(a)` instead. Then
  `sf = -expm1(-log1p(a*x**-a)/a)` (already used at line 7584) is accurate
  everywhere and the cutoff logic can go.

### F6: rounding of the exponent 1/a (minor)

- **Code:** line 7570 `(-1.0/a-1)`, line 7573 `x * (...)**(-1.0/a)`,
  line 7597 `(a / denom)**(1.0 / a)`.
- **Mechanism:** `1.0/a` is rounded (for example `1/0.1` rounds to exactly
  10), and the result `b**(1/a)` then has a relative error of about
  |log(result)| * eps/2 (pattern 6, applied to the exponent). In cdf there
  is a second cause: `x` and `(x**a)**(-1/a)` cancel in magnitude
  (x = 1e300 gives 1e300 * 1e-300), and the rounding of `x**a` is amplified
  by 1/a = 10.
- **Regime:** moderate a with results or intermediates of large log
  magnitude: pdf for a = 10, x ~ 1e20-1e26; pdf and cdf for a = 0.1,
  x ~ 1e230-1e300; isf for a = 0.1, small q. The errors are of the same order
  as cond (shapes) (460-605).
- **Reproducer:** `python accuracy.py refs/kappa3_ref.py --at pdf 1e20 10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(2.080873135168054e+26, 10)` | 3.1571728779406935e-289 | 3.1571728779408635e-289 | 242 | 11 | minor |
  | `pdf(1e+20, 10)` | 9.999999999999591e-220 | 1e-219 | 184 | 11 | minor |
  | `cdf(1e+300, 0.1)` | 0.9999999999999618 | 1.0 | 172 | 1e-31 | minor |
  | `isf(1.2115276586286942e-26, 0.1)` | 1.4677992676207905e+259 | 1.4677992676207401e+259 | 155 | 10 | minor |

- **Fix direction:** the log1p form of F5 removes it for cdf (no `x * x**-1`
  cancellation) and reduces it for pdf. For isf the remaining error equals
  the sensitivity to the last bit of a (cond (shapes) 597), so it is
  essentially unavoidable.

## RuntimeWarnings

- `overflow encountered in power`: in cdf (F1), ppf (F2) and pdf (`x**a`
  overflows where the true pdf is subnormal, so the result 0 is acceptable
  there). These warnings coincide with the losses in cdf and ppf.
- `overflow encountered in divide` and overflow in isf (F3): the warning
  matches the loss.
- `divide by zero encountered in divide` in ppf at q -> 1 (F4, `q**-a - 1 = 0`).
- `underflow encountered in power/multiply` in pdf, cdf, sf, ppf and isf
  where results are genuinely subnormal or 0. These warnings are spurious:
  the results are acceptable.
- sf: the `np.broadcast_arrays` + `super()._sf` path issues the same overflow
  warning as cdf. The returned sf = 1.0 there is wrong, but the true value
  is subnormal (see below).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | low and high tails, all parameter sets; bulk for a = 1, 2 | 349 ulp at a = 1000 (within cond 1e3); 1.01 ulp for a = 1, 2 |
| cdf | low tail (x down to 1e-300), all parameter sets | 7.97 ulp |
| cdf | high tail below the F1 overflow point | 12 ulp (a = 10, x = 1e20); F6 otherwise |
| sf | right tail (log1p/expm1 branch at line 7584), all parameter sets including a = 1000 near x = 1 | 2.44 ulp |
| sf | bulk, moderate a | 60 ulp at sf(1e10, 0.1) = 0.0947 and 29.5 ulp at sf(1.5, 10) = 0.0159 (minor: `1 - cdf` just above the 0.01 cutoff cancels by up to 100x, pattern 1) |
| ppf | a = 1, 2, 10, 20 outside F2; a = 1 near q = 1 | 1.1e15 ulp at a = 1, within cond 3.0e15 |
| isf | a <= 10 everywhere; a = 1000 for q <= 0.5 | 17 ulp except F6 (155 ulp); a = 1e-3: 1.45e3 ulp, within cond 1.44e3 |

The results of ppf and isf for a = 1e-6 and a = 1e-12 lie outside the double
range for every grid q (e.g. `ppf(0.5, 1e-6)` ~ 1e159000), so no points
could be measured there.

## Suspected, not confirmed

- **sf returns 1.0 where x**a overflows.** `_sf` (line 7577) starts from
  `1 - cdf`; after F1 this is `1 - 0 = 1`, which fails the `sf < 0.01` test,
  so the accurate branch at line 7584 is never evaluated. `sf(3, 1000)`
  returns 1.0 where the true value is 1.07e-401. The true sf is about
  x**-a < 5.6e-309 throughout this regime, i.e. always subnormal or 0, so the
  harness does not measure it. It becomes visible through the inherited
  logsf (see below). Fix: compute sf from the line-7584 formula
  unconditionally, using the F5 form for x < 1.
- The "blows up for large a" comment at line 7580 is not confirmed for the
  line-7584 formula itself: with a = 1000 near x = 1 the sf branch is
  accurate (0.94 ulp).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `log(a) - (1/a + 1)*L` with `L = log(a + x**a) = a*log(x) + log1p(a*x**-a)` (or `log(a) + log1p(x**a/a)` for x < 1) | pdf underflows for large x: `logpdf(1e200, 2)` = -inf, true -1380.8579086158675 |
| logcdf | log(cdf) | `-xlog1py(1/a, a*x**-a)` for x >= 1; `log(x) - (log(a) + log1p(x**a/a))/a` for x < 1 | wherever cdf fails (F1: cdf = 0, and F5). Near cdf -> 1 the default is fine (`logcdf(1e20, 1)` = -1e-20, exact) |
| logsf | log(sf) | `log(-expm1(-log1p(a*x**-a)/a))`, i.e. `log1mexp(xlog1py(-1/a, a*x**-a))`; for tiny `t = a*x**-a` use `log(t/a) = -a*log(x)` | sf underflows or is wrong (F1) for a*log(x) > ~708: `logsf(1e200, 2)` = 0.0, true -920.96945941285311 |

## Shared code

none. sc.expm1 and sc.xlog1py are called correctly; the losses come from the
expressions in kappa3_gen.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable). Related: PR #18822 "ENH:stats: Add _sf and _isf methods to kappa3" introduced the sf/isf code reviewed here - relates to F3.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(3, 1000)` | 0.0 | 1.0 | verified | also `cdf(2.1, 1000)`, `cdf(1e160, 2)`, `cdf(1e40, 10)` = 0.0, true 1.0; reference `exp(-log1p(a*x**-a)/a)`, cross-checked by `mp.quad` of the pdf; line 7573 matches |
| F2 | `ppf(0.49, 1000)` | 0.0 | 0.49339651773738403 | verified | also `ppf(1e-300, 2)`, `ppf(1e-40, 10)`, `ppf(2.7086801531713753e-16, 20)` = 0.0 (refs 1.4142135623730951e-300, 1.2589254117941671e-40, 3.1463658914688315e-16); reference by bisection on log cdf, cdf(x_ref)/q - 1 < 1e-198; line 7592 matches |
| F3 | `isf(0.51, 1000)` | 0.0 | 0.49339651773738403 | verified | also `isf(0.9, 1000)` (ref 0.10069316688518039) and `isf(0.9999999999999997, 20)` (ref 3.8688597313869831e-16) = 0.0; reference by bisection on log sf; lines 7595-7597 match |
| F4 | `ppf(0.9999999999999997, 0.1)` | inf | 5.9523046994645061e+154 | verified | `ppf(0.9999999999999966, 0.1)` 4.15e15 ulp, `ppf(0.9999999, 0.1)` 2.73e8 ulp, `ppf(0.6, 0.001)` 9.67e5 ulp, `ppf(0.36, 0.001)` 4.08e5 ulp: all as reported; `q**-a` rounds to 1 at line 7592, mechanism matches |
| F5 | `pdf(5.5, 1e-12)` | 6.687384284314276e-14 | 6.6887171122046977e-14 | verified | 8.97e11 ulp; `cdf(8.5, 1e-12)` 3.99e11 ulp, `sf(8.5, 1e-12)` 2.32e11 ulp, `cdf(0.5, 1e-300)` = 0.5 vs 0.36787944117144232 (rel. 0.36), `pdf(8.5, 1e-6)` 8.28e5 ulp; reference uses the `log1p(a*x**-a)` form (a different route from `(a + x**a)**(-1/a)`); lines 7570/7573 match |

All references were evaluated at 100 and 200 digits and agree to more than 30 digits. Scripts: `scratch/kappa3/verify/verify.py`. `refs/kappa3_ref.py` was not needed (no disagreements).

Verified by dist-accuracy-verifier: 5 of 5 verified.
