# moyal: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `moyal_gen`, lines 7612-7701
- **Implemented:** pdf, cdf, sf, ppf
- **Inherited (not measured):** logpdf = log(pdf); logcdf, logsf = `rv_continuous` defaults (log(cdf) / log(sf) on the tail side of the median, log1p(-sf) / log1p(-cdf) on the other side); isf = ppf(1 - q)
- **Related:** none (no helpers, no other distribution calls into moyal; moyal approximates `landau`, see `reports/landau.md`); special functions: sc.erf, sc.erfc, sc.erfcinv (sc.zeta in `_stats`/`_munp` only)
- **Reference:** `refs/moyal_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in ppf (wrong sign next to its zero at q0 = erfc(1/sqrt(2)) = 0.3173105...; the absolute error stays below 3 eps)

## Summary

pdf, cdf and sf are accurate on the whole real line. The error is at most 2.6 ulp in the bulk
and always within conditioning in the tails. pdf and cdf reach 222 and 1.1e3 ulp near
x = -7.2, where the condition number is 4.8e3. ppf is accurate in both tails (<= 0.67 ulp)
but not near its zero crossing at q0 = erfc(1/sqrt(2)) = 0.31731 (x = 0). It computes
`-log(2*erfcinv(q)**2)`, so the 1-3 ulp relative error of `y = 2*erfcinv(q)**2` ~ 1
turns into a 1.6-2.9 eps *absolute* error in x. That is about twice what the
conditioning allows. It is measured as minor at q = 0.3 and 0.315, severe within 1e-6
relative of q0, and wrong sign (loss) at the double next to q0. All inherited log
methods and isf have simple better closed forms (see Opportunities).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf | q near q0 = 0.3173105078629141 (ppf near 0); minor already at q = 0.3 / 0.315, severe for \|q/q0 - 1\| <= 1e-6 | -log(y) of y = 2*erfcinv(q)**2 ~ 1: relative error of y becomes absolute error of x (patterns 2/8) | wrong sign: -2.2e-16 for 1.885e-16 (abs error 1.85 eps) | loss |

### F1: ppf loses its relative accuracy near its zero at q0 = erfc(1/sqrt(2))

- **Code:** `_ppf`, line 7673: `-np.log(2 * sc.erfcinv(x)**2)`
- **Mechanism:** near q0 the intermediate `y = 2*erfcinv(q)**2` is close to 1 and
  `ppf = -log(y)` is close to 0. `sc.erfcinv` is accurate (0.7-1.5 ulp at the
  flagged points), but squaring doubles its relative error, and the rounding of
  the square and of `2*` adds to it. The log of y ~ 1 then gives an *absolute*
  error in x equal to the relative error of y: 1.6 to 2.9 eps, as measured.
  The conditioning of ppf with respect to q allows only about
  `q0/f(0) * eps = 1.31 eps` of absolute error, where f(0) = exp(-1/2)/sqrt(2*pi).
  So the error stays a factor of about 1.2-2.2 beyond conditioning while |x| goes to 0. Relative
  to |x| it grows without bound and the sign can flip (pattern 8, quantile
  formed through an intermediate near 1; pattern 2, `log` of a value near 1).
  A Newton step in double precision does not help: it gave 0.5-3.9 eps
  absolute at the same points, because `cdf` itself carries about 2 ulp near x = 0.
- **Regime:** q in roughly [0.29, 0.32] (|x| <~ 0.1). The ulp count scales like
  1/|x|, from minor (33 ulp at q = 0.3) to severe (1e6-2e10 ulp for |q/q0 - 1| = 1e-6 ... 1e-10)
  to loss (wrong sign at the doubles next to q0). Points at q = 0.28, 0.31, 0.32, 0.33,
  0.35 and further out are within conditioning. In every flagged case the absolute error
  is <= 2.9 eps.
- **Reproducer:** `python accuracy.py refs/moyal_ref.py --at ppf 0.31731050786291415`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.31731050786291415)` | -2.2204460492503128e-16 | 1.8852462727279873e-16 | inf | 7.0e+15 | loss |
  | `ppf(0.31731050789464516)` | 1.311365460751554e-10 | 1.3113591485727735e-10 | 2.2e+10 | 1.0e+10 | severe |
  | `ppf(0.3173108251734219)` | 1.3113590849510228e-06 | 1.311359084593029e-06 | 1.2e+06 | 1.0e+06 | severe |
  | `ppf(0.3173101905524062)` | -1.3113590845797497e-06 | -1.3113590849042177e-06 | 1.1e+06 | 1.0e+06 | severe |
  | `ppf(0.315)` | -0.009548781016245383 | -0.009548781016244741 | 303 | 136 | minor |
  | `ppf(0.3)` | -0.07157077197811597 | -0.07157077197811544 | 33.2 | 17.3 | minor |

  Error decomposition (scratch computation, eps = 2**-52):

  | q | erfcinv rel. err | y rel. err | abs. err of x |
  |---|---|---|---|
  | 0.31731050786291415 | 0.73 eps | 1.85 eps | 1.85 eps |
  | 0.31731050789464516 | 1.31 eps | 2.84 eps | 2.84 eps |
  | 0.3173108251734219 | 0.80 eps | 1.61 eps | 1.61 eps |
  | 0.315 | 1.49 eps | 2.89 eps | 2.89 eps |
  | 0.3 | 1.26 eps | 2.37 eps | 2.38 eps |

- **Fix direction:** for |q - q0| small, use a Taylor series of ppf in
  `d = (q - q0_hi) - q0_lo` (q0 split into two doubles; `q - q0_hi` is exact by
  Sterbenz), with ppf'(q0) = 1/f(0) = sqrt(2*pi*e). Elsewhere keep the current
  formula, preferably as `-2*log(erfcinv(q)) - log(2)` (no squaring; see also isf).

## RuntimeWarnings

- `pdf` (line 7664): "overflow encountered in exp" for x < -709.8 (`np.exp(-x)`).
  The result 0 is correct: the true pdf is about exp(-e^709/2). Also
  "underflow encountered in exp" for x > ~1416, where the true pdf is subnormal. Spurious.
- `cdf` / `sf` (lines 7667, 7670): "overflow encountered in exp" for x < -1419
  (`np.exp(-0.5*x)`); `erfc(inf) = 0` and `erf(inf) = 1` are correct. "underflow
  encountered in exp / divide" for x > ~1416, where `z = exp(-x/2)/sqrt(2)` is
  subnormal. The returned sf is then itself subnormal and still accurate (e.g.
  `sf(1418.3)` = 8.35618833651148e-309, equal to the reference), and cdf = 1.0 is correct. Spurious.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | x from -7.4 (underflow edge) to 1418.6, plus the default grid to +-1e300; left tail where `exp(-x)` is large (rounding amplified by \|x + e^-x\|/2 ~ 700, within cond 4.8e3); right edge where `exp(-x/2)` approaches the subnormal range | 222 ulp at x = -7.2 (cond 4.8e3, ok); bulk 2.64 ulp |
| cdf | same grid; left tail where `erfc(z)` with large z has cond 2z^2 (matches cond(x)); right tail (cdf ~ 1) | 1.13e3 ulp at x = -7.2 (cond 4.83e3, ok); bulk 2.38 ulp |
| sf | same grid; hypothesis "z subnormal while sf still normal" (x ~ 1416-1418.6) tested: z becomes subnormal only where sf is subnormal too, so no loss | 1 ulp |
| ppf | q from 2.3e-308 to 1 - 2.2e-16, dense near q0 | low tail 0.67 ulp, high tail 0.59 ulp (F1 only near q0) |
| sc.erfcinv | q = 2.3e-308, 1e-300, 1e-200, 1e-100, 1e-50, 1e-20, 1e-10 (direct check, 400 digits) | 1.1 ulp |

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `-0.5*(x + exp(-x)) - 0.5*log(2*pi)` (closed form) | pdf underflows: x < ~-7.3 and x > ~1490. `logpdf(-8)` = -inf, true -1487.3979320540688 |
| logcdf | log(cdf) for x < median, log1p(-sf) above | for x < median: `log(sc.erfcx(z)) - z**2` with `z**2 = exp(-x)/2` (no erfc underflow) | cdf underflows for x < ~-7.3 (z > ~26.5). `logcdf(-8)` = -inf, true -1494.7051200552633. The right side via log1p(-sf) is fine (`logcdf(100)` = -1.5389197253412838e-22, correct) |
| logsf | log(sf) for x > median, log1p(-cdf) below | for x > median: `log(erf(z))`, and for small z `log(2/sqrt(pi)) + log(z) + log1p(-z**2/3 + ...)` with `log(z) = -x/2 - log(2)/2`, i.e. `-x/2 + 0.5*log(2/pi)` once z**2 < eps | sf underflows for x > ~1416. `logsf(1500)` = -inf, true -750.22579135264473. The left side is fine (`logsf(-5)` = -3.853162382724658e-34, correct) |
| isf | ppf(1 - q) | `-log(2*sc.erfinv(q)**2)`, better `-2*log(sc.erfinv(q)) - log(2)` (avoids the underflow of `erfinv(q)**2` for q < ~1.7e-154) | `1 - q` rounds away small q (INDEX cross-cutting item 3): relative error ~eps/q, total loss (inf) for q < 1.1e-16. `isf(1e-12)` = 54.81050377049726, true 54.810459526567642 (8e-7 relative); `isf(1e-17)` = inf, true 77.836310456508098 |

## Shared code

none (`sc.erfcinv`, `sc.erfc` and `sc.erf` were accurate wherever measured; F1 comes
from how the distribution combines the erfcinv result, not from erfcinv itself).

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(0.31731050786291415)` | -2.2204460492503128e-16 | 1.8852462727279873e-16 | verified | wrong sign (rel. err 9.8e15 ulp), abs. err 1.85 eps; loss |
| F1 | `ppf(0.31731050789464516)` | 1.311365460751554e-10 | 1.3113591485727734e-10 | verified | 2.2e10 ulp, abs. 2.84 eps; severe |
| F1 | `ppf(0.3173108251734219)` | 1.3113590849510228e-06 | 1.311359084593029e-06 | verified | 1.2e6 ulp, abs. 1.61 eps; severe |
| F1 | `ppf(0.3173101905524062)` | -1.3113590845797497e-06 | -1.3113590849042177e-06 | verified | 1.1e6 ulp, abs. 1.46 eps; severe |
| F1 | `ppf(0.315)` | -0.009548781016245383 | -0.0095487810162447413 | verified | 303 ulp; minor |
| F1 | `ppf(0.3)` | -0.07157077197811597 | -0.071570771978115441 | verified | 33.2 ulp; minor |

Method: the reference is the analytic inverse -log(2*erfinv(1 - q)**2) at 100 and 200 digits (they agree to more than 30 digits). It is cross-checked by mp.findroot on the closed-form cdf erfc(exp(-x/2)/sqrt(2)), with differences of 1e-201 or less. The closed-form cdf is checked once against mp.quad of the docstring pdf over [-1, 0] (they agree to 40 digits). The mechanism matches line 7673 (`-np.log(2 * sc.erfcinv(x)**2)`). Script: `scratch/moyal/verify/verify_f1_light.py`.

Verified by dist-accuracy-verifier: 1 of 1 verified.
