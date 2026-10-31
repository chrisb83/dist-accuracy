# powerlaw: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `powerlaw_gen`, lines 8631-8916
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, ppf
- **Inherited (not measured):** logsf = log(sf), isf = ppf(1 - q)
- **Related:** none sharing code (mathematically beta with b = 1); helpers: `_check_fit_input_parameters` (fit only); special functions: sc.powm1, sc.xlogy
- **Reference:** `refs/powerlaw_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (1.659113328990585), (0.6591133289905851), (1), (0.001), (0.1), (3), (1e6), (1e-12)
- **Worst finding:** loss in logpdf

## Summary

cdf, logcdf and sf (via `-powm1(x, a)`) are accurate to <= 1.7 ulp over
the whole grid, including x within 1e-16 of 1 and a from 1e-12 to 1e6. The
pdf is accurate except for a minor (<= 86 ulp) error from the inexact
`a - 1.0` for small a and tiny x. logpdf loses all its digits next to the
point where pdf = 1 (x = a**(-1/(a-1))), because `log(a)` and
`(a-1)*log(x)` cancel; for a << 1 the error goes beyond the (already large)
conditioning, up to 1.1e7 ulp. ppf has a minor error (<= 211 ulp) from
rounding `1.0/a`. The inherited isf = ppf(1 - q) is severely inaccurate for
small a (1.4e11 ulp at one demonstration point).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf | x near a**(-1/(a-1)) (pdf = 1), any a != 1; beyond cond for a << 1 | log(a) + (a-1)*log(x) cancels (pattern 5) | 0.0 for 6.1e-17; 1.1e7 ulp | loss |
| F2 | ppf | small q, a not a power of 2 | rounded `1.0/a` times log(q) (pattern 8, INDEX item 4) | 211 ulp | minor |
| F3 | pdf | tiny x, a < 0.5 (a - 1.0 inexact) | rounded `a - 1.0` times log(x) (INDEX item 4) | 85.8 ulp | minor |

### F1: logpdf cancels where the pdf is close to 1

- **Code:** `_logpdf`, line 8678: `np.log(a) + sc.xlogy(a - 1, x)`
- **Mechanism:** logpdf is zero at x0 = a**(-1/(a-1)). Near x0 the two terms
  are equal and opposite, each of size |log a| and each rounded to 1/2 ulp,
  so the absolute error is about eps*|log a| while the result goes to 0
  (pattern 5). At `a = 1.659113328990585`, `x = 0.4638806987506132` both
  terms are exactly +-0.5062833205030135 in double and the sum is 0.0. The
  relative error is ~ eps*|log a|/|logpdf|, while cond (arg) is
  |a-1|/|logpdf|; so the error is beyond conditioning by about
  |log a|/|a-1|, which is > 1 for a < 1 and large for a << 1 (27.6 for
  a = 1e-12, where the measured error is 1.1e7 ulp at cond 1.0e6).
- **Regime:** any a != 1, x within a relative ~1e-6 of x0 for the severe
  cases (x0 = 0.0774 for a = 0.1, 9.93e-4 for a = 1e-3, ~1e-12 for a = 1e-12);
  total loss (0.0) at the doubles adjacent to x0. For a > 1 the errors are
  within the (huge) conditioning except where the result rounds to 0.0,
  which the tool counts as loss. Context: cond (arg) and cond (shapes) are
  1e14 to 1e16 at the loss points, so the logpdf value there is
  intrinsically hypersensitive; the loss is still reported as the tool rates it.
- **Reproducer:** `python accuracy.py refs/powerlaw_ref.py --at logpdf 1.0000009999723714e-12 1e-12`

  | shapes | arg | reference | scipy 1.18.1 value | scipy 1.18.1 ulp | scipy 1.18.1 severity | cond (arg) | cond (shapes) |
  |---|---|---|---|---|---|---|---|
  | (1.65911) | 0.4638806987506132 | 6.131460095413845e-17 | 0.0 | inf | loss | 1.1e+16 | 4.5e+15 |
  | (0.001) | 0.00099310918137498 | -2.927604664722567e-16 | 0.0 | inf | loss | 3.4e+15 | 3.4e+15 |
  | (0.1) | 0.07742636826811275 | -4.662505997819814e-16 | 0.0 | inf | loss | 1.9e+15 | 1.6e+15 |
  | (3) | 0.5773502691896257 | -1.1587517153601563e-16 | 0.0 | inf | loss | 1.7e+16 | 5.6e+15 |
  | (1e-12) | 9.999999999723714e-13 | -2.461642093131002e-15 | 0.0 | inf | loss | 4.1e+14 | 4.1e+14 |
  | (1e-12) | 1.0000009999723714e-12 | -9.99999502460024e-07 | -9.999995000953277e-07 | 1.1e+07 | severe | 1.0e+06 | 1.0e+06 |
  | (1e-12) | 9.999989999723714e-13 | 1.00000049753674e-06 | 1.0000004984078714e-06 | 3.9e+06 | severe | 1.0e+06 | 1.0e+06 |
  | (0.001) | 0.0009931081882657985 | 9.990004993125212e-07 | 9.990005001014879e-07 | 3.6e+06 | severe | 1.0e+06 | 9.9e+05 |
  | (0.1) | 0.07742629084174447 | 9.000004496155499e-07 | 9.000004501658054e-07 | 2.8e+06 | severe | 1.0e+06 | 8.3e+05 |

- **Fix direction:** near x0 evaluate `(a - 1)*log1p((x - x0)/x0)` with
  x0 = `exp(-log(a)/(a - 1))` (error then ~ cond (arg)), or accept as
  inherent; the gain is limited to a < 1.

### F2: ppf amplifies the rounding of 1.0/a by log(q)

- **Code:** `_ppf`, line 8687: `pow(q, 1.0/a)`
- **Mechanism:** `1.0/a` carries a relative error up to eps/2; the result
  exp(log(q)/a) then has a relative error up to (eps/2)*|log q|/a, i.e.
  ~0.5*|log q|/a ulp, while cond (arg) is only 1/a. Beyond conditioning by
  ~|log q|/2 (INDEX item 4, rounded exponent times a large logarithm).
- **Regime:** small q (|log q| in the hundreds), any a whose reciprocal is
  not exact; up to ~350 ulp at q = 1e-300.
- **Reproducer:** `python accuracy.py refs/powerlaw_ref.py --at ppf 1e-200 0.6591133289905851`

  | shapes | arg | reference | scipy 1.18.1 value | scipy 1.18.1 ulp | scipy 1.18.1 severity | cond (arg) | cond (shapes) |
  |---|---|---|---|---|---|---|---|
  | (0.659113) | 4.641588833612982e-201 | 1.1384555569206165e-304 | 1.138455556920563e-304 | 211 | minor | 1.52 | 700 |
  | (0.659113) | 1e-200 | 3.6479258717773335e-304 | 3.6479258717771635e-304 | 210 | minor | 1.52 | 699 |
  | (1.65911) | 1e-300 | 1.5153943761627304e-181 | 1.515394376162679e-181 | 153 | minor | 0.603 | 416 |
  | (0.1) | 1.2115276586286942e-26 | 6.812920690585783e-260 | 6.812920690585558e-260 | 149 | minor | 10 | 597 |

- **Fix direction:** `exp(log(q)/a)` (division instead of a rounded reciprocal), which bounds the error by a few ulp times cond.

### F3: pdf amplifies the rounding of a - 1.0 by log(x)

- **Code:** `_pdf`, line 8675: `a*x**(a-1.0)`
- **Mechanism:** for a < 0.5, `a - 1.0` is not exact (absolute error up to
  2**-54); `x**(a-1)` then has a relative error up to 2**-54*|log x|,
  ~170 ulp at x = 1e-300, while cond (arg) is |a-1| <= 1 (INDEX item 4).
- **Regime:** a < 0.5 with a - 1.0 inexact (e.g. 0.1, 1e-12), tiny x.
- **Reproducer:** `python accuracy.py refs/powerlaw_ref.py --at pdf 1e-300 0.1`

  | shapes | arg | reference | scipy 1.18.1 value | scipy 1.18.1 ulp | scipy 1.18.1 severity | cond (arg) | cond (shapes) |
  |---|---|---|---|---|---|---|---|
  | (0.1) | 1e-300 | 9.999999999999962e+268 | 1.0000000000000153e+269 | 85.8 | minor | 0.9 | 68.1 |
  | (0.1) | 1e-290 | 9.999999999999962e+259 | 1.0000000000000148e+260 | 83.6 | minor | 0.9 | 65.8 |
  | (1e-12) | 1e-300 | 9.999999993092245e+287 | 9.999999993092396e+287 | 68.2 | minor | 1 | 1 |
  | (1e-12) | 1e-290 | 9.999999993322502e+277 | 9.99999999332265e+277 | 66.7 | minor | 1 | 1 |

- **Fix direction:** `a*exp(xlogy(a, x))/x` (exact exponent `a`; the
  division by x adds only 1/2 ulp). Note logpdf is unaffected (the same
  absolute error is negligible relative to |logpdf| ~ 690).

## RuntimeWarnings

"underflow encountered in power" from pdf (2 parameter sets), cdf (3) and
ppf (4), for large a or tiny a where `x**(a-1)`, `x**a` or `q**(1/a)`
genuinely underflows to a subnormal or 0. The true values there are below
the normal range too, so these warnings accompany correct results and are
not accuracy findings.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid incl. x within 1e-16 of 1, a from 1e-12 to 1e6 | 0.81 ulp |
| logcdf | full grid, all parameter sets | 0.961 ulp |
| sf | full grid incl. 1 - x = 1e-16 ... 1e-3 with a = 1e-12 and 1e6 (`-powm1` accurate) | 1.7 ulp |
| pdf | large a = 1e6 with x**(a-1) subnormal (log x = -700 ... -740 /(a-1)) | 348 ulp, within cond (1e6) |
| pdf | all other sets, bulk and tails | F3 aside, <= 5.76 ulp |
| logpdf | away from the pdf = 1 point, all sets (a = 1e6 left tail 5.4e10 ulp is within cond 2.1e16) | within conditioning |
| ppf | a = 1e-12 (1.37e4 ulp, within cond 1e12), q near 1 | within conditioning |

The subnormal intermediate in pdf (pattern 4) can never exceed
conditioning: the pdf is normal only if a*x**(a-1) >= 2.2e-308, so the
relative error of the subnormal factor is at most a*eps, while cond (arg)
is |a-1| ~ a.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| isf | ppf(1 - q) = (1 - q)**(1/a) | `exp(log1p(-q)/a)` | small a: the rounding of 1 - q (up to eps/2) is amplified by 1/a. `isf(1e-13, 1e-12)` = 0.904809282989345 vs 0.9048374180359551, 1.4e11 ulp, severe (cond 0.1); `isf(1e-17, 1e-6)` = 1.0 vs 0.99999999999, 4.5e4 ulp, moderate |
| logsf | log(sf) = log(-powm1(x, a)) | `log(-expm1(a*log(x)))` / `log1mexp(a*log x)`; essentially the same | only when sf itself is subnormal (a < ~1e-292 near x = 1); `logsf(0.5, 1e-310)` is exact (0 ulp). No practical gain |

## Shared code

none. `sc.powm1` (INDEX item 10: returns 0.0 instead of +inf on overflow)
is used in `_sf` as `-powm1(x, a)` with 0 <= x <= 1 and a > 0, so its
result lies in [-1, 0] and cannot overflow; it was accurate to 1.7 ulp on
the whole grid.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `powerlaw.logpdf(0.4638806987506132, 1.659113328990585)` | 0.0 | 6.1314600954138443e-17 | verified | loss (0.0 returned); same at the (0.001), (0.1), (3) and (1e-12) points next to x0, all reference values match the report |
| F1 | `powerlaw.logpdf(1.0000009999723714e-12, 1e-12)` | -9.999995000953277e-07 | -9.9999950246002405e-07 | verified | 1.06e7 ulp, severe; other severe rows: 3.92e6, 3.56e6, 2.75e6 ulp, as reported. Reference: (a-1)*log(x/x0), x0 = exp(-log(a)/(a-1)), cross-checked with log(a*x**(a-1)), 100 vs 200 digits agree. Mechanism matches line 8678; for a << 1 the rounding of `a - 1` times log x (about eps*|log x|) adds an error of the same order as the rounding of log(a) |

Verified by dist-accuracy-verifier: 1 of 1 verified.
