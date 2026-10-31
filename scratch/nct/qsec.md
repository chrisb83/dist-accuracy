### F11: ppf and isf saturate at ±1.3407807929942603e154 for small df

- **Code:** `_ppf` line 8190 (`sc.nctdtrit`), `_isf` line 8198 (`scu._nct_isf`). Both are the Boost `non_central_t` quantile.
- **Mechanism:** the root finder inverts the Boost cdf/sf. Beyond |x| = sqrt(DBL_MAX) = 1.34e154 that cdf/sf is exactly 0 or 1 (F2: `t*t` overflows). So the search stops at the last x where `t*t` is finite and returns ±1.3407807929942603e154 for every smaller q. This is the same failure as t.md F4. Just above the saturation the result is already off by 1e8 ulp (the F1 error of the cdf, at cond 2).
- **Regime:** small df (true quantile beyond 1.34e154). Saturation for q <~ 1e-154 at df = 1, and for q <~ 1e-89 at (0.5, 5).
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at ppf 1e-300 1 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 1, 1)` | -1.3407807929942603e154 | -6.647612765794013e298 | 4.5e15 | 1 | loss |
  | `isf(1e-300, 1, 1)` | 1.3407807929942603e154 | 8.643606884608054e299 | 4.5e15 | 1 | loss |

  Run counts: ppf low tail 17 loss and isf low tail 17 loss, all at (1, 1) and (0.5, 5).
- **Fix direction:** fix F2 first, so that the cdf has a usable tail beyond 1.34e154. Then invert the tail asymptote q = C |x|^-df analytically for tiny q.

### F12: ppf and isf inherit the cdf/sf failures of F1, F9 and F10

- **Code:** `_ppf` line 8190, `_isf` line 8198.
- **Mechanism:** the Boost quantile inverts the cdf (ppf) or the complement (isf). Where those carry the F1 error (left tail for nc > 0, right tail for nc < 0), the F9-type zero (sf far right tail at large df) or the F10 bulk error (df = 1e6), the quantile carries it too, divided by its condition number. For df = 1e6, nc = 2 and small q, the cdf is wrong by orders of magnitude or nan (F1). The root finder then returns essentially arbitrary values between -7.6 and -37 instead of the smooth true quantile from -5.5 down to -35.
  - isf(1e-300, 1e6, 2) = 38.94 instead of 39.06, because sf is 0 already from x ~ 38.9 although the true sf there is 8e-299. sf(38, 1e6, 2) still returns a value but is off by 2.6e6 ulp: the sf part of F9 (subnormal/underflowing intermediate, pattern 4, inferred).
  - isf with q -> 1 at df = 1e6 is the left tail through the complement (F1). It is beyond conditioning even though cond is large there (4.2e9 ulp at cond 7.5e7).
- **Regime:** ppf for q below cdf(0) = Phi(-nc) with nc > 0. isf for small q with nc < 0, and for q -> 1 with nc > 0. Bulk ppf/isf at df ~ 1e6 (moderate, F10).
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at ppf 1e-300 1e6 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 1e6, 2)` | -8.761810522124264 | -35.058484473649436 | 3.4e15 | 0.00077 | severe |
  | `ppf(3.481e-14, 1e6, 2)` | -5.4886136305315425 | -5.48864294645784 | 2.4e10 | 0.0239 | severe |
  | `isf(1e-300, 1e6, 2)` | 38.94287354630109 | 39.06123139155966 | 1.4e13 | 0.000691 | severe |
  | `isf(0.9999999994820525, 1e6, 2)` | -4.103801921317024 | -4.103805730707064 | 4.2e9 | 7.5e7 | severe |
  | `ppf(3.481e-14, 0.5, 5)` | -6093324983307.059 | -6093325092628.5 | 8.1e7 | 2 | severe |
  | `isf(1.212e-26, 5, -3)` | 33467.14352571071 | 33467.14352194843 | 5.1e5 | 0.2 | moderate |
  | `ppf(0.78, 1e6, 2)` | 2.772195390817912 | 2.77219539082521 | 1.19e4 | 0.95 | moderate |
  | `sf(38, 1e6, 2)` | 6.679566177308012e-284 | 6.679566181144653e-284 | 2.6e6 | 1.37e3 | severe |
  | `sf(38.94287354630109, 1e6, 2)` | 0.0 | 7.953452836473855e-299 | inf | 1.44e3 | loss |

  Run counts: ppf low tail 32 severe; isf high tail 7 severe; isf low tail 24 moderate (all of (5, -3)) and 3 severe; ppf/isf bulk 5 moderate each (df = 1e6).
- **Fix direction:** fix F1, F9 and F10 in the cdf/sf. Then let ppf invert the cdf only where cdf <= 1/2 and isf invert the sf only where sf <= 1/2, using the directly computed small tail.

### F13: ppf returns nan for q = 1e-300 at tiny nc

- **Code:** `_ppf`, line 8190 (`sc.nctdtrit`).
- **Mechanism:** unclear. The cdf at the true quantile -2.56e30 is accurate: (10, 1e-8) has cdf <= 4.7 ulp on the whole grid. ppf(q, 10, 1e-8) is correct for q = 1e-200, 1e-250 and 1e-280, but nan for q = 1e-300. At nc = 0 and nc = 1e-6 the same q works. So the Boost quantile iteration fails for this combination of tiny q and tiny nonzero nc. No warning is emitted.
- **Regime:** observed at (10, 1e-8), q = 1e-300; not mapped further.
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at ppf 1e-300 10 1e-8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 10, 1e-8)` | nan | -2.5645257106334987e30 | inf | 0.1 | loss |

- **Fix direction:** for |nc| below ~1e-6, start the iteration from the central-t quantile `stdtrit(df, q)`, or fall back to it with a first-order nc correction.
