# laplace_asymmetric: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `laplace_asymmetric_gen`, lines 6158-6251
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** laplace (kappa = 1 case, separate class, no shared code); helpers: none; special functions: none (np.exp, np.log only)
- **Reference:** `refs/laplace_asymmetric_ref.py`, 400 digits; self-checks: PASS except two "integral of pdf != cdf" FAILs at (2) and (0.5). These come from the harness: `mp.quad` integrates over an interval that contains the kink of the pdf at x = 0. With a breakpoint at 0 the integral matches the cdf difference to 1e-31 (relative). Without it the error is 8.8e-6, the size the check reported. The reference is correct, and the cdf agrees with SciPy in the bulk to < 3 ulp for those shapes.
- **Parameter sets:** (2), (1), (0.5), (0.001), (1000), (1e-08), (1e+08), (1e-150), (1e+150)
- **Worst finding:** loss in cdf, sf, ppf, isf

## Summary

logpdf is correctly rounded everywhere (< 1 ulp). In the tails, the branches of cdf and sf that are not written as `1 - ...` are also accurate (0.5 ulp). The `1 - ...` branches lose accuracy when kappa is far from 1. For small kappa, cdf(0) = kappa**2/(1+kappa**2) is tiny, and the x >= 0 branch `1 - exp(-x*kappa)/(1+kappa**2)` cancels: moderate at kappa = 1e-3, severe at 1e-8, and it returns 0 at 1e-150. sf mirrors this for large kappa and x < 0. ppf and isf use `log((1 - q)*...)`, which has the same problem in the matching quantile range. The rounding of `1 - q` for small q, and `log` instead of `log1p` near the branch point, give results of 0.0 (loss) for extreme kappa and errors up to 5e17 ulp for moderate kappa. pdf = exp(logpdf) loses up to ~300 ulp (minor) because the large constant log(kappa + 1/kappa) passes through exp.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | kappa << 1, x >= 0 with cdf(x) small (x*kappa small) | `1 - exp(-x*kappa)*kapinv/kappkapinv` cancels (pattern 1) | 0.0 for 1e-300; 6.6e14 ulp | loss |
| F2 | sf | kappa >> 1, x < 0 with sf(x) small (|x|/kappa small) | `1 - exp(x/kappa)*kappa/kappkapinv` cancels (pattern 1) | 0.0 for 1e-300; 5.0e14 ulp | loss |
| F3 | ppf | q >= q0 = kappa**2/(1+kappa**2) with q < 1/2 (small kappa), or q just above q0 (any kappa) | `log((1 - q)*kappkapinv*kappa)`: `1 - q` rounds, and the log of a value near 1 is formed without log1p (patterns 1, 2, 8) | 0.0 for 2.9e-138; 4.0e11 ulp | loss |
| F4 | isf | q <= 1/(1+kappa**2) mirror: q > 1/(1+kappa**2) with q < 1/2 (large kappa), or q just above 1/(1+kappa**2) | `log((1 - q)*kappkapinv/kappa)`, same as F3 (patterns 1, 2, 8) | 0.0 for -1.2e124; 7.7e7 ulp | loss |
| F5 | pdf | all kappa far from 1, small |x| | exp(logpdf) amplifies the rounding of the large term `log(kappa+kapinv)` | 167 ulp at cond 1e-5 | minor |

### F1: cdf cancels for x >= 0 when kappa is small

- **Code:** `_cdf`, line 6218: `1 - np.exp(-x*kappa)*(kapinv/kappkapinv)`
- **Mechanism:** `kapinv/kappkapinv = 1/(1+kappa**2)` is close to 1 when kappa is small, so for x >= 0 with x*kappa not large the subtraction cancels (pattern 1). Only the digits of the subtrahend beyond its leading 1s survive. The relative error is about eps/cdf(x). At x = 0 the true value is kappa**2/(1+kappa**2). Once kappa**2 < eps/2 the result is exactly 0 (kappa = 1e-150), or a multiple of eps (kappa = 1e-8 gives 1.11e-16 instead of 1e-16).
- **Regime:** x >= 0 and cdf(x) << 1, i.e. kappa << 1 and x*kappa << 1. The error is roughly 1/cdf(x) ulp: about 2.8e5 ulp at kappa = 1e-3 near x = 0, and 5e14 ulp at kappa = 1e-8. At kappa = 1e-150, every x in [0, ~1e130] returns 0.0 although the true cdf is 1e-300 ... 1e-20.
- **Reproducer:** `python accuracy.py refs/laplace_asymmetric_ref.py --at cdf 0 1e-8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-300, 1e-150)` | 0.0 | 1e-300 | inf | 0 | loss |
  | `cdf(1e+130, 1e-150)` | 0.0 | 1.0000000000000001e-20 | inf | 1 | loss |
  | `cdf(1e+140, 1e-150)` | 1.000000082740371e-10 | 9.999999999500001e-11 | 3.7e8 | 1 | severe |
  | `cdf(3e-09, 1e-08)` | 1.1102230246251565e-16 | 1.3e-16 | 6.6e14 | 0.231 | severe |
  | `cdf(0, 1e-08)` | 1.1102230246251565e-16 | 1e-16 | 5.0e14 | 0 | severe |
  | `cdf(10, 1e-08)` | 9.999999506238311e-08 | 9.999999510000016e-08 | 1.7e6 | 1 | severe |
  | `cdf(0, 0.001)` | 9.99998999939855e-07 | 9.99999000001e-07 | 2.8e5 | 0 | moderate |
  | `cdf(0.3, 0.001)` | 0.00030095470354485343 | 0.000300954703544959 | 1.58e3 | 0.997 | moderate |

  Counts: 65 loss (kappa = 1e-150), 53 severe, 45 moderate.
- **Fix direction:** for x >= 0 use `(kappa**2 - expm1(-x*kappa)) / (1 + kappa**2)`. Both terms are non-negative, so nothing cancels.

### F2: sf cancels for x < 0 when kappa is large

- **Code:** `_sf`, line 6226: `1 - np.exp(x*kapinv)*(kappa/kappkapinv)`
- **Mechanism:** this is the mirror of F1. `kappa/kappkapinv = kappa**2/(1+kappa**2)` is close to 1 for large kappa, so `1 - ...` cancels for x < 0 with |x|/kappa small (pattern 1).
- **Regime:** x < 0, kappa >> 1, |x|/kappa << 1. The error is about 1/sf(x) ulp. At kappa = 1e150 the result is 0.0 for x in [-1e130, 0).
- **Reproducer:** `python accuracy.py refs/laplace_asymmetric_ref.py --at sf -1e-300 1e8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(-1e-300, 1e+150)` | 0.0 | 1e-300 | inf | 0 | loss |
  | `sf(-1e+130, 1e+150)` | 0.0 | 1.0000000000000001e-20 | inf | 1 | loss |
  | `sf(-1e+140, 1e+150)` | 1.000000082740371e-10 | 9.999999999500001e-11 | 3.7e8 | 1 | severe |
  | `sf(-1e-300, 1e+08)` | 1.1102230246251565e-16 | 9.999999999999999e-17 | 5.0e14 | 1e-292 | severe |
  | `sf(-10, 1e+08)` | 9.999999506238311e-08 | 9.999999510000016e-08 | 1.7e6 | 1 | severe |
  | `sf(-1e-300, 1000)` | 9.99998999939855e-07 | 9.99999000001e-07 | 2.8e5 | 1e-297 | moderate |
  | `sf(-0.3, 1000)` | 0.00030095470354485343 | 0.000300954703544959 | 1.58e3 | 0.997 | moderate |

  Counts: 64 loss (kappa = 1e150), 52 severe, 44 moderate.
- **Fix direction:** for x < 0 use `(1 - kappa**2 * expm1(x/kappa)) / (1 + kappa**2)` (expm1 < 0, so there is no cancellation). Written as `(kapinv**2 - expm1(x*kapinv)) / (1 + kapinv**2)`, it avoids overflow of kappa**2.

### F3: ppf upper branch rounds 1 - q and takes log of a value near 1

- **Code:** `_ppf`, line 6232: `-np.log((1 - q)*kappkapinv*kappa)*kapinv` (used for q >= kappa/kappkapinv = q0)
- **Mechanism:** the argument of the log is (1 - q)(1 + kappa**2), which is close to 1 whenever q is close to q0. Two errors occur:
  1. `1 - q` is not exact for q < 1/2. It carries an absolute error of up to eps/2, which is large compared with q - q0 when q0 is small (patterns 1, 8). For small kappa, q0 ~ kappa**2 is tiny. Once q < eps/2 (at kappa = 1e-150), `1 - q` is exactly 1, so the log is 0 and the result is 0.0 even though the true quantile is about q/kappa, with condition number 1.
  2. Even with an exact `1 - q`, the product rounds by about 1 ulp. The log of a value near 1 turns this into an absolute error eps in a small result: pattern 2, log instead of log1p. This also affects moderate kappa just above q0, e.g. kappa = 0.5, q0 = 0.2.
- **Regime:** q0 <= q < ~1/2 with q0 small (kappa << 1): error about eps/(kappa*x) ulp-relative, loss when q < eps/2. For any kappa, q slightly above q0: errors beyond conditioning.
- **Reproducer:** `python accuracy.py refs/laplace_asymmetric_ref.py --at ppf 2.872984833353539e-288 1e-150`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(2.872984833353539e-288, 1e-150)` | 0.0 | 2.872984833352539e-138 | inf | 1 | loss |
  | `ppf(1.2115276586286942e-26, 1e-150)` | 0.0 | 1.2115276586286942e+124 | inf | 1 | loss |
  | `ppf(1.3000000000000002e-16, 1e-08)` | 0.0 | 3.0000000000000025e-09 | inf | 4.33 | loss |
  | `ppf(9.999990000020002e-07, 0.001)` | 1.1102230246251565e-13 | 1.0001558162268073e-15 | 5.0e17 | 1.0e12 | loss |
  | `ppf(1.0000999999999997e-12, 1e-08)` | 9.998668559779158e-05 | 0.00010000000000004999 | 6.0e11 | 1 | severe |
  | `ppf(1.0000000000000002e-06, 0.001)` | 1.000088900582841e-09 | 1.0000000001253731e-09 | 4.0e11 | 1.0e6 | severe |
  | `ppf(1.0000000099999999e-08, 1e-08)` | 0.9999999878202988 | 1.000000005 | 7.7e7 | 1 | severe |
  | `ppf(1e-08, 1e-150)` | 1.0000000100247594e+142 | 1.000000005e+142 | 2.3e7 | 1 | severe |
  | `ppf(0.200000008, 0.5)` | 1.999999997845058e-08 | 2.000000011722846e-08 | 3.1e7 | 2.5e7 | severe |
  | `ppf(1.2999987000013002e-06, 0.001)` | 0.00030000004506414683 | 0.0003000000450000091 | 9.6e5 | 4.33 | moderate |
  | `ppf(0.20002, 0.5)` | 5.000062501030033e-05 | 5.000062501042523e-05 | 1.12e4 | 1e4 | moderate |

  Counts: 34 loss, 11 severe, 2 moderate. Some reproducers have large cond (arg) near q0, e.g. 1e12 for `ppf(9.99999000002e-07, 0.001)`. The tool still classifies them beyond conditioning: returning 1.1e-13 for 1.0e-15 is off by a factor of 100.
- **Fix direction:** `-log1p(kappa**2 - q*(1 + kappa**2)) / kappa`, i.e. -log1p(-(q - q0)/(1 - q0))/kappa, which never forms `1 - q`. For q >= 1/2 the current form is fine, since `1 - q` is exact there.

### F4: isf lower branch, mirror of F3

- **Code:** `_isf`, line 6240: `np.log((1 - q)*kappkapinv/kappa)*kappa` (used for q > kapinv/kappkapinv = 1/(1+kappa**2))
- **Mechanism:** same as F3. For large kappa, p0 = 1/(1+kappa**2) is tiny. For p0 < q < 1/2, `1 - q` rounds, and `log` of a value near 1 is used instead of `log1p`. At kappa = 1e150 and q < eps/2 the result is exactly 0.0 instead of about -q*kappa.
- **Regime:** p0 < q < ~1/2 with kappa >> 1; for any kappa, q slightly above p0. The first branch (line 6239) also hits this at q just below p0 = 1/(1+kappa**2) close to 1 (small kappa). `isf(0.9999990000010001, 0.001)` returns 0.0 for -6.1e-14, a loss, though cond (arg) there is 1.6e16.
- **Reproducer:** `python accuracy.py refs/laplace_asymmetric_ref.py --at isf 1.2115276586286942e-26 1e150`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1.2115276586286942e-26, 1e+150)` | 0.0 | -1.2115276586286942e+124 | inf | 1 | loss |
  | `isf(1.3e-16, 1e+08)` | 0.0 | -3.0000000000000004e-09 | inf | 4.33 | loss |
  | `isf(9.999990000020002e-07, 1000)` | -1.1102230246251565e-13 | -1.0001974495485974e-15 | 5.0e17 | 1.0e12 | loss |
  | `isf(0.9999990000010001, 0.001)` | 0.0 | -6.114501960490932e-14 | inf | 1.6e16 | loss |
  | `isf(3.4807005884285624e-14, 1e+150)` | -3.486100297323052e+136 | -3.480700588428623e+136 | 7.0e12 | 1 | severe |
  | `isf(1.0000000099999999e-08, 1e+08)` | -0.9999999878202988 | -1.000000005 | 7.7e7 | 1 | severe |
  | `isf(0.200000008, 2)` | -1.999999997845058e-08 | -2.000000011722846e-08 | 3.1e7 | 2.5e7 | severe |
  | `isf(1.2999987000013002e-06, 1000)` | -0.00030000004506414683 | -0.00030000004500000917 | 9.6e5 | 4.33 | moderate |
  | `isf(0.20002, 2)` | -5.000062501030033e-05 | -5.000062501042523e-05 | 1.12e4 | 1e4 | moderate |

  Counts: 35 loss, 11 severe, 2 moderate.
- **Fix direction:** `kappa * log1p(kapinv**2 - q*(1 + kapinv**2))` for the lower branch (q > p0); for the upper branch near q -> p0 analogously `-log1p(q*(1 + kappa**2) - 1)/kappa`.

### F5: pdf = exp(logpdf) amplifies the rounding of log(kappa + 1/kappa)

- **Code:** `_pdf`, line 6206: `np.exp(self._logpdf(x, kappa))`; `_logpdf`, line 6211: `lPx -= np.log(kappa+kapinv)`
- **Mechanism:** logpdf is correctly rounded, but its absolute error (0.5 ulp of |logpdf|) becomes a relative error of |logpdf|*eps/2 after exp. For kappa far from 1 the constant term log(kappa + 1/kappa) is large (345 for kappa = 1e150), even where x*kappa or x/kappa is tiny and the pdf itself is well conditioned.
- **Regime:** |log(kappa + 1/kappa)| large, i.e. kappa far from 1; 210 points minor, up to ~300 ulp.
- **Reproducer:** `python accuracy.py refs/laplace_asymmetric_ref.py --at pdf 1e-155 1e150`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-155, 1e+150)` | 9.999900000500369e-151 | 9.999900000499998e-151 | 167 | 1e-05 | minor |
  | `pdf(1e-160, 1e+150)` | 9.999999999000243e-151 | 9.999999999e-151 | 109 | 1e-10 | minor |
  | `pdf(1e-152, 1e+150)` | 9.900498337491888e-151 | 9.90049833749168e-151 | 94.4 | 0.01 | minor |

- **Fix direction:** `exp(x*rate) / (kappa + kapinv)` (rate = -kappa or kapinv) instead of exp of the combined log; or `exp(x*rate) * kappa/(1 + kappa**2)` to avoid forming the large log.

## RuntimeWarnings

- cdf, sf: `np.where` evaluates both branches. For x < 0 the x >= 0 branch computes `exp(-x*kappa)`, which overflows for large |x|, e.g. `cdf(-1000, 2)` warns "overflow encountered in exp" but returns the accurate 5.699661125393029e-218 (0 ulp). These warnings are spurious. Underflow warnings in exp and in multiply also appear, both on discarded branches and where results are truly subnormal or 0.
- cdf/sf F1/F2 losses at kappa = 1e-150 / 1e150 come with "underflow encountered in multiply" (`kapinv/kappkapinv` or `kappa/kappkapinv` underflows). That warning goes with an inaccurate result.
- pdf/logpdf: "overflow encountered in multiply" for x*kappa beyond 1.8e308 (e.g. x = 1e300, kappa = 1e150). The returned logpdf = -inf / pdf = 0 is correct in the sense that the true value is not a normal double.
- ppf, isf: no warnings on the measured grid.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | full grid, both sides of x = 0, all parameter sets including kappa = 1e-150 and 1e150 | 0.994 ulp |
| cdf | x < 0 branch (left tail), all parameter sets; x >= 0 for kappa >= 0.5 | 0.5 ulp in the left tail; 2.5 ulp at kappa = 0.5; 108 ulp at kappa = 1e150 within cond 700 |
| sf | x >= 0 branch (right tail), all parameter sets; x < 0 for kappa <= 2 | 0.5 ulp in the right tail; 2.5 ulp at kappa = 2; 149 ulp at kappa = 1e-150 within cond 700 |
| ppf | lower branch q < q0 (`log(q*kappkapinv/kappa)*kappa`), q -> 1 (upper branch with exact 1 - q), kappa = 1 | within conditioning (e.g. 1.25e3 ulp at cond 1e4 for kappa = 2); kappa = 1: 0 ulp |
| isf | upper branch q < p0 away from p0, all parameter sets | within conditioning except F4 points |
| pdf | full grid | max 364 ulp; beyond conditioning only as F5 (minor) |

## Suspected, not confirmed

- `_ppf` line 6233 forms `q*kappkapinv/kappa`, which is about q/kappa**2 for tiny kappa. It overflows for kappa < ~1e-154 and q not tiny; only the discarded branch is affected (q >= q0), so at worst a spurious warning. No warning was seen at kappa = 1e-150, and smaller kappa was not tested. In `_ppf`/`_isf`, `kappkapinv*kappa` (= 1 + kappa**2) overflows for kappa > ~1.3e154. This was not tested; kappa = 1e150 was the largest value. In that regime, q0 rounds to 1 so the branch is never taken.
- `kapinv = 1/kappa` overflows for kappa < 5.6e-309 (subnormal kappa, allowed by the domain). Not tested.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | x < 0: `x/kappa + log(kappa**2/(1+kappa**2))` (= `x/kappa - log1p(kapinv**2)`); x >= 0: `log1p(-exp(-x*kappa)/(1+kappa**2))` | the left-tail cdf underflows once x/kappa < ~-745: `logcdf(-2000, 2)` = -inf, true -1000.2231435513143. Also inherits F1 for small kappa |
| logsf | log(sf) | x >= 0: `-x*kappa - log1p(kappa**2)`; x < 0: `log1p(-exp(x/kappa)*kappa**2/(1+kappa**2))` | the right-tail sf underflows once x*kappa > ~745: `logsf(400, 2)` = -inf, true -801.6094379124341. Also inherits F2 for large kappa |

## Shared code

none

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(0, 1e-08)` | 1.1102230246251565e-16 | 1e-16 | verified | 4.96e14 ulp. Reference from `mp.quad` of the docstring pdf in scaled variables (u = t*kappa, u = t/kappa), 100 vs 200 digits agree. All 8 table rows reproduced to the stated digits, including 0.0 for 1e-300 at kappa = 1e-150. Line 6218 matches the stated mechanism |
| F1 | `cdf(1e+130, 1e-150)` | 0.0 | 1.0000000000000001e-20 | verified | loss |
| F2 | `sf(-1e-300, 1e+08)` | 1.1102230246251565e-16 | 9.999999999999999e-17 | verified | 4.96e14 ulp. Same quad route; all 7 table rows reproduced. Line 6226 matches |
| F2 | `sf(-1e+130, 1e+150)` | 0.0 | 1.0000000000000001e-20 | verified | loss |
| F3 | `ppf(2.872984833353539e-288, 1e-150)` | 0.0 | 2.872984833352539e-138 | verified | Reference from the inverse `-log1p(kappa**2 - q*(1+kappa**2))/kappa`, checked by cdf(x_ref) = q (closed-form cdf with expm1, rel. 9e-202; the quad cdf gives 3.7e-14 here, a quad limitation on the tiny interval). All 9 loss/severe rows reproduced. Line 6232 matches |
| F3 | `ppf(9.999990000020002e-07, 0.001)` | 1.1102230246251565e-13 | 1.0001558162268073e-15 | verified | 4.95e17 ulp, cdf(x_ref) = q checked with quad |
| F4 | `isf(1.2115276586286942e-26, 1e+150)` | 0.0 | -1.2115276586286942e+124 | verified | Reference from `kappa*log1p(kappa**-2 - q*(1+kappa**-2))`, checked by quad sf(x_ref) = q. All 7 loss/severe rows reproduced. Line 6240 matches |
| F4 | `isf(0.9999990000010001, 0.001)` | 0.0 | -6.114501960490932e-14 | verified | loss. This point goes through the first branch (line 6239): the true isf lies in the x < 0 branch, but the rounded threshold `kapinv/kappkapinv` sends it to the other branch, and `-log(q*...)` of a value near 1 gives 0. cond 1.6e16 as stated |

Verified by dist-accuracy-verifier: 4 of 4 verified.
