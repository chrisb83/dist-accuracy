# truncexpon: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `truncexpon_gen`, lines 10142-10202
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf, logsf (the generic `rv_continuous` versions: `log(cdf)` / `log1p(-sf)` and `log(sf)` / `log1p(-cdf)`, split at the median)
- **Related:** none; helpers: none; special functions: sc.expm1, sc.log1p (sc.gammainc/sc.gamma appear only in a commented-out `_munp` line)
- **Reference:** `refs/truncexpon_ref.py`, 400 digits; self-checks: PASS for the reference. The one FAIL (`sf(1e-12)`, "disagrees at every bulk point") and the NOTEs for `sf`, `logcdf`, `logsf` at b = 1e-300 are SciPy errors (F1 and the Opportunities section). The reference value is easy to check by hand: `sf(b/2, b)` is 0.5 to first order for tiny b. The three WARNs at b = 1e-300 ("consistency check failed to run: ZeroDivisionError") came from a first version of the reference that used `1 - exp(-b)`. Switching to `expm1`/`log1p` removed them.
- **Parameter sets:** (4.690772545681048), (1e-300), (1e-12), (1e-06), (0.5), (20), (50), (800)
- **Worst finding:** loss in sf, isf, logpdf

## Summary

pdf, cdf and ppf are accurate over the whole support for every b tested
(at most 1.5 ulp, or within conditioning). Two methods break down for small b:
sf and isf compute a difference of two exponentials that are both close to 1.
The relative error grows like eps/b. Both methods return 0 (sf can also return
-0.0 or a value > 1) once b < ~1.1e-16, and they are severe for b = 1e-12 and
moderate for b = 1e-6. logpdf rounds `-expm1(-b)` to 1 before taking the log.
For b >~ 20 this makes logpdf severe near x = 0, and for b >~ 37 logpdf
returns -0.0 or -x where the true value is +e^-b. There is also a moderate
error in cdf for subnormal x. It comes from `sc.expm1` with subnormal
arguments (shared code).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | sf | small b (b <~ 1e-3), all x in [0, b] | exp(-b) - exp(-x) cancels, both ~1 (patterns 1, 5) | -0.0 for 0.5; 1.11 for 0.5 (b = 1e-16) | loss |
| F2 | isf | small b, all q | log(exp(-b) - q*expm1(-b)) of a value near 1 (patterns 2, 8) | 0.0 for 5e-301 | loss |
| F3 | logpdf | large b (b >~ 20), x <~ e^-b | log(-expm1(-b)) with -expm1(-b) rounded to ~1 (pattern 2) | -0.0 for 1.9e-22 | loss |
| F4 | cdf | subnormal x, small b | sc.expm1 is not exact for subnormal arguments (pattern 4, 10) | 2.23e4 ulp | moderate |

### F1: sf cancels for small b

- **Code:** `_sf`, line 10184: `(np.exp(-b) - np.exp(-x))/sc.expm1(-b)`
- **Mechanism:** for b << 1 both `exp(-b)` and `exp(-x)` (with 0 <= x <= b)
  are 1 - O(b). Each is rounded with absolute error ~1.1e-16, and their
  difference is O(b), so the relative error is ~eps/b at every x, including
  the bulk (patterns 1 and 5). Once b < ~1.1e-16, both exponentials round
  to 1.0 or to the same neighbour. The result is then 0, -0.0 (0 divided by a
  negative `expm1(-b)`), or a value above 1 (one ulp below 1 divided by
  -b, e.g. 1.11). Near x = b the same cancellation also makes the result 0
  for b = 1e-12. In that regime cond (arg) is large, but the error is far
  beyond it.
- **Regime:** relative error ~1.1e-16/b for all x in [0, b]: loss for
  b < ~1.1e-16, severe for b <~ 1e-10, moderate for b <~ 1e-3. For b >= 0.5 the
  measured errors stay within conditioning (max 2.4e9 ulp at cond 1e10, near
  x = b).
- **Reproducer:** `python accuracy.py refs/truncexpon_ref.py --at sf 5e-301 1e-300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(5e-301, 1e-300)` | -0.0 | 0.5 | inf | 1 | loss |
  | `sf(5e-17, 1e-16)` | 1.1102230246251565 | 0.4999999999999999875 | 5.5e15 | 1 | loss |
  | `sf(9.99999e-13, 1e-12)` | -0.0 | 9.9999999997092e-07 | inf | 1.0e6 | loss |
  | `sf(5e-13, 1e-12)` | 0.49993342798895796 | 0.499999999999875 | 6.0e11 | 1 | severe |
  | `sf(1e-300, 1e-12)` | 0.9999778782803785 | 1.0 | 9.96e10 | 1.0e-288 | severe |
  | `sf(9.5e-07, 1e-06)` | 0.04999997617936004 | 0.04999997625000351 | 6.36e6 | 19 | severe |
  | `sf(1e-300, 1e-06)` | 0.9999999999841389 | 1.0 | 7.14e4 | 1.0e-294 | moderate |

- **Fix direction:** `exp(-x) * (-expm1(x - b)) / (-expm1(-b))`. This is
  algebraically `expm1(b - x)/expm1(b)`, but written this way it does not
  overflow for large b.

### F2: isf loses accuracy for small b

- **Code:** `_isf`, line 10187: `-np.log(np.exp(-b) - q * sc.expm1(-b))`
- **Mechanism:** the argument of the log is `e^-b + q(1 - e^-b) = 1 - (1-q)(1-e^-b)`,
  which is 1 - O(b) for small b. Rounding `exp(-b)` and the sum to doubles
  leaves an absolute error ~eps, and `log` turns this into an absolute error
  ~eps in a result of size ~b(1-q) (pattern 2/8). For b < ~1.1e-16 the
  argument is exactly 1.0 and the result is 0.0. For b = 1e-6 and q -> 1,
  the term (1-q)*b drops below eps and the result is also 0.0.
- **Regime:** relative error ~eps/(b(1-q)): loss for b < ~1e-16 (every q) and
  for b(1-q) < ~1e-16, severe for b = 1e-12, moderate for b = 1e-6 at
  small q. For b >= 0.5 the errors stay within conditioning (max 6.9e14 ulp at
  cond 3.0e15, q -> 1).
- **Reproducer:** `python accuracy.py refs/truncexpon_ref.py --at isf 0.5 1e-300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(0.5, 1e-300)` | 0.0 | 5e-301 | inf | 1 | loss |
  | `isf(1e-300, 1e-300)` | 0.0 | 1e-300 | inf | 1e-300 | loss |
  | `isf(0.9999999999999627, 1e-06)` | 0.0 | 3.7303474975664663e-20 | inf | 2.68e13 | loss |
  | `isf(0.5, 1e-12)` | 4.99933427988833e-13 | 4.99999999999875e-13 | 6.0e11 | 1 | severe |
  | `isf(1e-10, 1e-12)` | 9.999778782803785e-13 | 9.999999999e-13 | 1.0e11 | 1e-10 | severe |
  | `isf(0.9999999, 1e-6)` | 1.000310945187316e-13 | 9.999994994738611e-14 | 1.4e12 | 1.0e7 | severe |
  | `isf(0.5, 1e-6)` | 4.999998749365581e-07 | 4.99999875e-07 | 5.7e5 | 1 | moderate |

- **Fix direction:** for small b use `b - log1p(q * expm1(b))`. This is the same
  quantity, `-log(e^-b (1 + q(e^b - 1)))`, without the near-1 argument. Keep
  the current form for large b, where `expm1(b)` would overflow.

### F3: logpdf drops the normalisation term for large b

- **Code:** `_logpdf`, line 10175: `-x - np.log(-sc.expm1(-b))`
- **Mechanism:** `-expm1(-b) = 1 - e^-b` is accurate as a number, but for
  large b it is a double near 1 whose rounding error (~eps absolute) is large
  compared with `log(1 - e^-b) ~ -e^-b`. The log then has relative error
  ~eps/e^-b. Once e^-b < eps/2 (b > ~36.7), the value rounds to exactly 1 and
  the log gives 0 (pattern 2). The term matters wherever x is comparable to
  or smaller than e^-b, in particular at x = 0, where logpdf is +e^-b and
  cond (arg) is 0. Near the zero crossing x = -log(1 - e^-b) the error also
  exceeds the conditioning (3.0e9 ulp at cond 32.7 for b = 20). The same
  mechanism gives 26 ulp (minor) at the example b = 4.69, x = 0.
- **Regime:** x <~ e^-b * (some factor): loss for b > ~37 (result -0.0 or -x
  instead of e^-b - x); severe for b ~ 20 to 37; moderate where x is a little
  larger than e^-b (e.g. x = 1e-12 at b = 50); minor (~26 ulp) at moderate b.
- **Reproducer:** `python accuracy.py refs/truncexpon_ref.py --at logpdf 0 50`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0, 50)` | -0.0 | 1.9287498479639178e-22 | inf | 0 | loss |
  | `logpdf(0, 38)` | -0.0 | 3.1391327920480296e-17 | inf | 0 | loss |
  | `logpdf(1e-20, 50)` | -1e-20 | -9.807125015203608e-21 | 8.9e13 | 1.02 | severe |
  | `logpdf(0, 36)` | 2.2204460492503136e-16 | 2.3195228302435696e-16 | 1.9e14 | 0 | severe |
  | `logpdf(2e-09, 20)` | 6.115358326962427e-11 | 6.115362456273484e-11 | 3.04e9 | 32.7 | severe |
  | `logpdf(0, 20)` | 2.0611535832696244e-09 | 2.061153624552735e-09 | 9.0e7 | 0 | severe |
  | `logpdf(1e-12, 50)` | -1e-12 | -9.99999999807125e-13 | 8.7e5 | 1 | moderate |
  | `logpdf(0, 4.690772545681048)` | 0.009221983842139467 | 0.009221983842139413 | 26.3 | 0 | minor |

- **Fix direction:** `-x - sc._ufuncs._log1mexp(-b)`, i.e. `log(1 - e^-b)`
  computed as `log1p(-exp(-b))` for large b. `_log1mexp(-50.0)` returns
  -1.9287498479639178e-22 exactly.

### F4: cdf for subnormal x (sc.expm1 with subnormal arguments)

- **Code:** `_cdf`, line 10178: `sc.expm1(-x)/sc.expm1(-b)`
- **Mechanism:** for subnormal x, `sc.expm1(-x)` does not return `-x`:
  `sc.expm1(-1e-312) = -1.000000000003e-312` and
  `sc.expm1(1e-315) = 9.99999994e-316`, a relative error of up to ~5e-9.
  `np.expm1` returns `-x` exactly. Cephes' rational form computes `x*P(x^2)`
  with a subnormal product, so it loses the low bits (pattern 4). Dividing by
  a small `expm1(-b)` lifts the result back to a normal double while keeping
  the relative error (pattern 10).
- **Regime:** x subnormal (x < 2.2e-308) with b small enough that cdf
  ~ x/b is normal. This has little practical relevance, but the cause sits in
  shared code.
- **Reproducer:** `python accuracy.py refs/truncexpon_ref.py --at cdf 1e-312 1e-12`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-312, 1e-12)` | 1.000000000003906e-300 | 9.999999999989654e-301 | 2.23e4 | 1 | moderate |
  | `cdf(1e-310, 1e-300)` | 1.0000000000000463e-10 | 9.999999999999969e-11 | 222 | 1 | minor |

- **Fix direction:** in `sc.expm1`, return x for |x| below ~1e-300 (or use
  `np.expm1`). In truncexpon nothing beyond that is needed.

## RuntimeWarnings

- `underflow encountered in multiply` in ppf and isf for b = 1e-300 and
  b = 1e-12 (`q * sc.expm1(-b)` is subnormal for tiny q). The ppf values are
  accurate, so for ppf the warning is spurious. In isf it appears together
  with the F2 errors, but it is not their cause.
- `underflow encountered in exp` in pdf, sf and isf for b = 800 (`exp(-b)`
  and `exp(-x)` for x > 745 underflow). The results are accurate (or the true
  value is subnormal), so these warnings are spurious.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full support, all parameter sets (b = 1e-300 ... 800) | 1.05 ulp |
| cdf | full support, all parameter sets, apart from subnormal x (F4) | 1.54 ulp |
| ppf | default q grid (1e-300 ... 1 - 3e-16), all parameter sets | 11.1 ulp at cond 23; 7.3e7 ulp at cond 8.4e13 (q -> 1, b = 50); always within conditioning |
| logpdf | x >> e^-b, all parameter sets; all x for b <= 1e-6 and b = 800 | 1.11 ulp (bulk), 0.579 ulp (right tail) |
| sf | b >= 0.5, full support | within conditioning (max 2.4e9 ulp at cond 1e10, x -> b) |
| isf | b >= 0.5, all q | within conditioning (max 6.9e14 ulp at cond 3.0e15, q -> 1) |

## Suspected, not confirmed

none. H4 (cancellation in `log1p(q*expm1(-b))` for q -> 1 and large b) was
tested with b = 20, 50 and q up to 1 - 3e-16 and stayed within conditioning.
For b > 745 (`exp(-b)` = 0 in sf/isf), b = 800 was tested and is fine.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | `rv_continuous._logcdf`: `log(_cdf)` below the median, `log1p(-_sf)` above it | `log(expm1(-x)/expm1(-b))`, i.e. `log(_cdf)` everywhere: cdf is accurate and never underflows for normal x. Near x = b, `log1p(-sf)` with a fixed sf (F1) would be slightly better. | Small b, x above the median: the bad `_sf` (F1) is fed into `log1p`. `logcdf(5e-301, 1e-300)` returns 0.0 (true -0.6931471805599453, loss); `logcdf(5e-13, 1e-12)` returns -0.6930140454007399 (true -0.6931471805599453). |
| logsf | `rv_continuous._logsf`: `log(_sf)` above the median, `log1p(-_cdf)` below it | `-x + log(-expm1(x - b)) - log1mexp(-b)`, i.e. `-x + _log1mexp(x - b) - _log1mexp(-b)` | Small b, x above the median (F1 inherited): `logsf(9e-301, 1e-300)` returns -inf (true -2.302585092994046); `logsf(9e-13, 1e-12)` returns -2.3022741961397153 (true -2.3025850929940455). |

## Shared code

`sc.expm1` is not exact for subnormal arguments: `sc.expm1(-1e-312)` returns
-1.000000000003e-312 and `sc.expm1(1e-315)` returns 9.99999994e-316, while
`np.expm1` returns x exactly. Any distribution that divides `expm1` of a tiny
argument by another small quantity can show this error (F4). The relative
error is at most ~5e-9 and occurs only for subnormal inputs.

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `sf(5e-301, 1e-300)` | -0.0 | 0.5 | verified | Reference = quadrature of exp(-t) over [x, b] / over [0, b] (rescaled to [0, 1]), 100 vs 200 digits agree. All 7 table rows reproduced with the same ulp (e.g. `sf(5e-17, 1e-16)` = 1.1102230246251565, `sf(5e-13, 1e-12)` 6.0e11 ulp, `sf(1e-300, 1e-06)` 7.14e4 ulp). Mechanism matches line 10184: `np.exp(-1e-16)` = 0.9999999999999999, `np.exp(-5e-17)` = 1.0. |
| F2 | `isf(0.5, 1e-300)` | 0.0 | 5.0e-301 | verified | Reference `b - log1p(q*expm1(b))`, checked by quadrature sf(x_ref) = q (rel. err. <= 1e-192; the check cannot resolve `isf(1e-300, 1e-300)` at 200 digits, where x_ref = b(1 - 1e-300)). All 7 table rows reproduced (6.0e11, 9.96e10, 1.4e12, 5.71e5 ulp; three 0.0 results). Mechanism matches line 10187. |
| F3 | `logpdf(0, 50)` | -0.0 | 1.9287498479639178e-22 | verified | Reference log(exp(-x) / integral of exp(-t) over [0, b]). All 7 measured rows reproduced (8.86e13, 1.92e14, 3.04e9, 9.02e7, 8.69e5 ulp). `-sc.expm1(-50.)` is exactly 1.0, as stated (line 10175). Typo in the table: the reference for `logpdf(0, 20)` is 2.061153624562735e-09, not ...552735e-09; the stated 9.0e7 ulp is correct. |
| F4 | `cdf(1e-312, 1e-12)` | 1.000000000003906e-300 | 9.9999999999896536e-301 | verified | 2.23e4 ulp, moderate. Quadrature over the subnormal-length interval must be rescaled to [0, 1]; unscaled `mp.quad` on [0, 1e-312] is off by 4e-14 relative. `sc.expm1(-1e-312)` = -1.000000000003e-312 vs `np.expm1` = -1e-312 confirms the mechanism (line 10178). `cdf(1e-310, 1e-300)`: 222 ulp, reproduced. |

Verified by dist-accuracy-verifier: 4 of 4 verified.
