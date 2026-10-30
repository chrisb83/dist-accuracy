# levy_l: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `levy_l_gen`, lines 6410-6508
- **Implemented:** pdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logpdf = log(pdf); logcdf and logsf use the rv_continuous median split (log(cdf) / log1p(-sf) for logcdf, log(sf) / log1p(-cdf) for logsf, `_distn_infrastructure.py` lines 2035-2049)
- **Related:** levy (mirror image, levy_l(x) = levy(-x), separate code using erf/erfc/erfinv; see `reports/levy.md`); levy_stable (alpha = 1/2, beta = -1, no shared code); helpers: `_norm_cdf` (sc.ndtr), `_norm_sf` (sc.ndtr(-x)), `_norm_ppf` (sc.ndtri), `_norm_isf` (-sc.ndtri); special functions: sc.ndtr, sc.ndtri (through the helpers)
- **Reference:** `refs/levy_l_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in cdf and ppf

## Summary

pdf and isf are accurate everywhere tested (pdf 244 ulp max, within
conditioning in the low-|x| tail; isf 3.01 ulp). cdf fails in the far left tail
(x -> -inf, where the cdf is small): `2*ndtr(1/sqrt(|x|)) - 1` cancels. It loses
about 1/cdf ulp, and it returns 0.0 for |x| > ~1e32 although the true cdf is
normal down to x = -1.7e308. ppf fails for small q:
`(q + 1.0)/2` rounds away q. The error is 992 ulp already at q = 1e-3, and
the result is -inf for q <= ~1.1e-16. sf loses up to 1.75e3 ulp (about 2.5x the
conditioning) near x = -7e-4. The cause is rounding of the intermediate `1/sqrt(|x|)`
amplified by ndtr. levy (the mirror image, which uses erf/erfc/erfinv) has none of
these problems.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | x < ~-1e6 (cdf < ~1e-3); total for x < ~-1e32 | 2*ndtr(z) - 1 cancels, z = 1/sqrt(\|x\|) -> 0 (pattern 1) | 0.0 for 6.1e-155 at x = -1.7e308 | loss |
| F2 | ppf | q < ~1e-3; total for q <= ~1.1e-16 | (q + 1.0)/2 rounds away q (pattern 8/1) | -inf for -6.4e31 at q = 1e-16 | loss |
| F3 | sf | x ~ -7e-4 ... -1e-3 (sf tiny) | rounding of 1/sqrt(\|x\|) amplified by ndtr's cond z^2 (pattern 10) | 1.75e3 ulp, cond 705 | moderate |

### F1: cdf cancels in 2*ndtr(z) - 1 for large |x|

- **Code:** `_cdf`, line 6494: `2 * _norm_cdf(1 / np.sqrt(ax)) - 1` (`_norm_cdf` = `sc.ndtr`).
- **Mechanism:** for large |x|, z = 1/sqrt(|x|) is small and ndtr(z) = 0.5 +
  z/sqrt(2 pi) + ..., so `2*ndtr(z) - 1` subtracts two numbers close to 1
  (pattern 1). The absolute error is ~eps, so the relative error is
  ~eps/cdf. Once z/sqrt(2 pi) < eps/4 (|x| > ~1e32), ndtr(z) rounds to 0.5
  and the result is exactly 0. The true cdf ~ sqrt(2/pi)/sqrt(|x|) is still
  6.1e-155 at x = -1.7e308.
- **Regime:** x < ~-1e6 (557 ulp, minor), -1e8 (3.09e3 ulp, moderate),
  -1e16 (1.3e7 ulp, severe), total loss for x <= -1e32. Measured on the grid:
  36 loss, 5 severe, 3 moderate, 1 minor points.
- **Reproducer:** `python accuracy.py refs/levy_l_ref.py --at cdf -1e32`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-1.7e308)` | 0.0 | 6.119495232776587e-155 | inf | 0.5 | loss |
  | `cdf(-1e32)` | 0.0 | 7.978845608028653e-17 | inf | 0.5 | loss |
  | `cdf(-1e31)` | 2.220446049250313e-16 | 2.52313252202016e-16 | 5.4e+14 | 0.5 | severe |
  | `cdf(-1e20)` | 7.978839811073613e-11 | 7.978845608028654e-11 | 3.3e+09 | 0.5 | severe |
  | `cdf(-1e16)` | 7.97884558423334e-09 | 7.978845608028653e-09 | 1.3e+07 | 0.5 | severe |
  | `cdf(-1e10)` | 7.978845607992113e-06 | 7.978845607895674e-06 | 5.44e+04 | 0.5 | moderate |
  | `cdf(-1e8)` | 7.97884559473605e-05 | 7.978845594730578e-05 | 3.09e+03 | 0.5 | moderate |
  | `cdf(-1e6)` | 0.0007978844278220265 | 0.0007978844278221252 | 557 | 0.5 | minor |

- **Fix direction:** `sc.erf(1/np.sqrt(2*ax))` (as in levy's sf). Spot checks
  of this form: 0.56 ulp at x = -1e10, 0.34 ulp at x = -1e20.

### F2: ppf rounds away small q in (q + 1)/2

- **Code:** `_ppf`, line 6501: `val = _norm_ppf((q + 1.0) / 2)`, then line 6502 `-1.0 / (val * val)`.
- **Mechanism:** `q + 1.0` keeps only the digits of q above 2^-52 (absolute
  error up to 1.1e-16), so for small q the relative error of q is ~1.1e-16/q.
  ppf ~ -2/(pi q^2) has cond 2 in q, so the error in ppf is about 2.2e-16/q
  relative. For q <= ~1.1e-16, `q + 1.0 == 1.0`, ndtri(0.5) = 0 and the result is
  `-1/0 = -inf` (pattern 8, a variant of pattern 1).
- **Regime:** q < ~1e-3 (992 ulp), q = 1e-6 (7.4e5 ulp), q = 1e-8 (5.5e7 ulp),
  -inf for q <= 1e-16. Measured on the grid: 16 loss, 7 severe, 1 moderate,
  2 minor points.
- **Reproducer:** `python accuracy.py refs/levy_l_ref.py --at ppf 1e-16`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-100)` | -inf | -6.366197723675813e+199 | inf | 2 | loss |
  | `ppf(1e-16)` | -inf | -6.366197723675813e+31 | inf | 2 | loss |
  | `ppf(1e-15)` | -5.164873193977107e+29 | -6.366197723675812e+29 | 8.5e+14 | 2 | severe |
  | `ppf(1e-12)` | -6.365065957230227e+23 | -6.366197723675814e+23 | 8.0e+11 | 2 | severe |
  | `ppf(1e-08)` | -6366197801056577.0 | -6366197723675813.0 | 5.5e+07 | 2 | severe |
  | `ppf(1e-06)` | -636619772471.9932 | -636619772367.248 | 7.4e+05 | 2 | moderate |
  | `ppf(0.001)` | -636619.4390343358 | -636619.4390341956 | 992 | 2 | minor |

- **Fix direction:** `-0.5/sc.erfinv(q)**2` (levy's isf formula with the sign
  flipped; spot checks 0.12 ulp at q = 1e-8, 0.27 ulp at q = 1e-16). Keep the
  current form or use `-1/_norm_isf((1-q)/2)**2` for q near 1.

### F3: sf near x = -7e-4 loses about twice the conditioning

- **Code:** `_sf`, line 6498: `2 * _norm_sf(1 / np.sqrt(ax))` (`_norm_sf(z)` = `sc.ndtr(-z)`).
- **Mechanism:** sf = erfc(1/sqrt(2|x|)) has cond ~1/(2|x|) with respect to x
  (705 at x = -7.1e-4). The code forms z = 1/sqrt(|x|) in two rounded
  steps (0.3-0.45 ulp measured). ndtr then multiplies by sqrt(1/2) internally,
  another rounding. The cond of ndtr(-z) with respect to z is z^2 = 1/|x|,
  twice the cond with respect to x. So each half-ulp of intermediate
  rounding costs ~z^2/2 ulp (pattern 10, special function argument rounded
  before the call). With z correctly rounded, `2*ndtr(-z)` still gives
  1.27e3 ulp at x = -7.1e-4. The levy-style form `sc.erfc(1/np.sqrt(2*ax))`
  gives 781, 302 and 135 ulp at x = -7.1e-4, -1e-3 and -8e-4.
- **Regime:** sf tiny, |x| < ~1e-3 down to sf underflow at |x| ~ 7.1e-4.
  2 grid points, both moderate. The error is 1.5-2.5x cond (arg).
- **Reproducer:** `python accuracy.py refs/levy_l_ref.py --at sf -0.00071`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(-0.00071)` | 3.062506317539051e-308 | 3.0625063175399134e-308 | 1.75e+03 | 705 | moderate |
  | `sf(-0.001)` | 1.795832784800317e-219 | 1.7958327848007448e-219 | 1.07e+03 | 500 | moderate |

- **Fix direction:** `sc.erfc(np.sqrt(0.5/ax))` or `sc.erfc(1/np.sqrt(2*ax))`
  (fewer roundings in the erfc argument). The errors stay at about cond level.

## RuntimeWarnings

- pdf: `underflow encountered in exp` (plus divide/multiply under- and overflow,
  and `invalid value encountered in multiply`). These occur for small |x|
  (exp(-1/(2|x|)) subnormal or zero, |x| < ~7.06e-4) and for huge |x|
  (`2*np.pi*ax` overflows for |x| > 2.86e307). The results are accurate
  (`pdf(-0.0007)` = 1.3271462650773607e-306, 157 ulp, cond 713, ok) or the true
  value is subnormal. The warnings are spurious.
- ppf: `divide by zero encountered in divide` for q <= ~1.1e-16. The result
  is -inf and wrong (F2).
- isf: `underflow encountered in divide` from `p/2` for p near the smallest
  normal. The results are accurate (`isf(3e-308)` 1.38 ulp). The warning is spurious.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid plus \|x\| in [6e-4, 7.5e-4] (exp(-1/(2\|x\|)) subnormal) and \|x\| up to 1.7e308 | 244 ulp (near-zero tail, cond 703; within conditioning); bulk 0.968 ulp |
| cdf | bulk and x -> 0- (cdf -> 1) | 6.85 ulp (bulk); 0 ulp near 0 |
| sf | bulk and x -> -inf (sf -> 1) | 6.24 ulp |
| ppf | bulk q >= 0.01; q -> 1 down to 1 - 2.2e-16 | 9.01 ulp at q = 0.01 (ok); high tail within conditioning (cond up to 1.3e14, e.g. `ppf(1 - 2.2e-16)` 0.527 ulp) |
| isf | default grid plus p in [2.3e-308, 1e-307] (p/2 subnormal), p up to 1 - 2.2e-16 (ndtri near 0.5) | 3.01 ulp |

Hypotheses from code reading tested on purpose:
- pdf, line 6490: subnormal `exp(-1/(2|x|))` (pattern 4). The window where it
  matters is narrow (|x| ~ 7.0e-4 ... 7.06e-4). The error there is within
  conditioning, as for levy.
- ppf for q -> 1: `q + 1.0` rounds off up to half an ulp of 1 - q. The
  resulting error stays inside the (very large) conditioning. Max 3.9e13 ulp,
  0 points flagged.
- isf, line 6505: `p/2` is exact apart from the subnormal range, and
  `ndtri(p/2)` near 0.5 is accurate. Max 3.01 ulp.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `-0.5/ax - 1.5*log(ax) - 0.5*log(2*pi)` with ax = \|x\| | pdf underflows for \|x\| < ~6.7e-4 and \|x\| > ~1e205; default returns -inf. `logpdf(-5e-4)`: -inf vs -989.51758484389153 (closed form gives -989.5175848438915) |
| logcdf | log(cdf) for x < median, log1p(-sf) above | `log(sc.erf(1/sqrt(2*ax)))` for x < median | inherits F1: -inf for x < ~-1e32 and large errors before that. `logcdf(-1e40)`: -inf vs -46.277493212525641 (erf form gives -46.27749321252564) |
| logsf | log(sf) for x > median, log1p(-cdf) below | `log(2) + sc.log_ndtr(-1/sqrt(ax))` for x > median; `log1p(-sc.erf(1/sqrt(2*ax)))` below | sf underflows for \|x\| < ~7.1e-4, so -inf: `logsf(-5e-4)` -inf vs -1004.0267419589519 (log_ndtr form -1004.0267419589522). Below the median it inherits F1: `logsf(-1e20)` = -7.978839811391922e-11 vs -7.9788456083469634e-11 (relative error 7e-7; erf form gives -7.978845608346965e-11) |

## Shared code

none. The helpers `_norm_cdf`, `_norm_sf`, `_norm_ppf` and `_norm_isf` (`sc.ndtr`,
`sc.ndtri`) are called correctly and are accurate for their own results. The
problems come from how levy_l combines them (`2*ndtr - 1`, `(q + 1)/2`), not
from the helpers.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable; only unrelated `levy_stable` issues such as #18432 and #14994 came up)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-1.7e308)` | 0.0 | 6.119495232776587e-155 | verified | reference from the power series of 2*int_0^z phi, z = \|x\|^-1/2; mechanism matches line 6494 (`2*_norm_cdf(1/np.sqrt(ax)) - 1`) |
| F1 | `cdf(-1e32)` | 0.0 | 7.9788456080286533e-17 | verified | reference = mp.quad of 2*phi on [0, z] (substitution t = \|y\|^-1/2) |
| F1 | `cdf(-1e16)` | 7.97884558423334e-09 | 7.9788456080286534e-09 | verified | 1.34e7 ulp; direct mp.quad of the docstring pdf in x agrees to 7e-20 |
| F1 | `cdf(-1e6)` | 0.0007978844278220265 | 0.00079788442782212517 | verified | 557 ulp, same as report |
| F2 | `ppf(1e-16)` | -inf | -6.3661977236758137e+31 | verified | x_ref from findroot on quad cdf; cdf(x_ref)/q - 1 = 1.6e-201; mechanism matches lines 6501-6502 (1e-16 < 2^-53, so q + 1.0 == 1.0) |
| F2 | `ppf(1e-100)` | -inf | -6.3661977236758132e+199 | verified | the 100- and 200-digit values agree to only 2e-30 (findroot tolerance at 100 digits); the 200-digit value inverts the cdf to 2e-132 |
| F2 | `ppf(1e-8)` | -6366197801056577.0 | -6366197723675812.8 | verified | 5.47e7 ulp, same as report |
| F3 | `sf(-0.00071)` | 3.062506317539051e-308 | 3.0625063175399133e-308 | verified | reference = (2/z) phi(z) * quad of exp(-v - v^2/(2z^2)); 1.27e3 ulp at 2.2e-16 relative per ulp (report's 1.75e3 uses the actual double spacing near the smallest normal); moderate either way; mechanism matches line 6498 |
| F3 | `sf(-0.001)` | 1.795832784800317e-219 | 1.7958327848007449e-219 | verified | 1.07e3 ulp, same as report |

Verified by dist-accuracy-verifier: 3 of 3 verified.
