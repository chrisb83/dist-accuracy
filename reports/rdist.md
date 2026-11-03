# rdist: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `rdist_gen`, lines 9037-9095
- **Implemented:** pdf (9075-9076, `exp(_logpdf)`), logpdf (9078-9079, via `beta._logpdf`, lines 764-767), cdf (9081-9082, via `beta._cdf` = `sc.betainc`, 769-770), sf (9084-9085, via `beta._sf` = `sc.betaincc`, 772-773), ppf (9087-9088, `2*beta._ppf(q, c/2, c/2) - 1`, Boost `ibeta_inv` through `scu._beta_ppf`, 778-779). Every method calls beta with a = b = c/2 and y = (x + 1)/2.
- **Inherited (not measured):** logcdf, logsf (rv_continuous default in 1.18.1: `log(cdf)` / `log1p(-sf)` split at the median, resp. `log(sf)` / `log1p(-cdf)`), isf = ppf(1 - q)
- **Related:** `beta` (all methods delegated), `semicircular` (its ppf is `rdist._ppf(q, 3)`); helpers: none; special functions: `sc.betainc`, `sc.betaincc`, `sc.betaln`, `sc.xlogy`, `sc.xlog1py`, `scu._beta_ppf` (Boost `ibeta_inv`); `sc.beta` only in `_munp` (moments, not measured)
- **Reference:** `refs/rdist_ref.py`, 120 digits (continued fraction for I_y(a, a) on y <= 1/2 with the symmetry F(x) = 1 - F(-x) and sf(x) = F(-x); quantiles by Newton in log y from a double seed, verified to |log F - log q| < 1e-100, with `invert` as fallback); self-checks: PASS
- **Parameter sets:** (1.6), (3), (10), (0.5), (1000), (1e6), (0.001); single points at c = 1e8
- **Worst finding:** loss in ppf (near q = 1/2, every c), and in pdf, logpdf, sf at x = 1 - 2**-53 (every c)

## Summary

cdf and sf are accurate for moderate c, apart from the last double below
x = 1. The ppf returns 0 or a value of the wrong sign for |q - 1/2| <~ 1e-14
for every c. This is because `2*p - 1` cancels near the median (F1). Even a
correctly rounded p leaves an absolute error of about 1e-16 in x. For large c
the same cancellation costs about sqrt(c) ulp in the whole bulk (3.95e3 ulp at
c = 1e6, q = 0.6). At x = 1 - 2**-53, `(x + 1)/2` rounds to 1.0, so pdf/logpdf
return inf/-inf, sf returns 0 and cdf returns 1.0 (true 0.509 for c = 0.001) (F2).
For large c, logpdf cancels `betaln(c/2, c/2)` against terms of size ~0.69*c.
That costs 6e5 ulp in logpdf and 4.1e6 ulp in pdf at c = 1e6 (F3). cdf/sf lose
1e3 to 1e4 ulp for large c (1.35e4 at c = 1e6 in the far tail, 1.07e3 near x = 0 at
c = 1e8). Half of that comes from rounding `(x + 1)/2` and half from Boost `ibeta`
for large shapes (F4).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf | \|q - 1/2\| <= 1e-16 ... 1e-14 every c (loss); severe up to \|q - 1/2\| = 1e-14 ... 1e-4 depending on c; whole bulk for c >~ 1e3 | `2*p - 1` cancels; p = `ibeta_inv(c/2, c/2, q)` carries >= 0.5 ulp(0.5) absolute error (patterns 5, 10) | 0.0 for 2.5e-16; 2.84e-14 for -1.74e-16 (wrong sign) | loss |
| F2 | pdf, logpdf, sf, cdf | x = 1 - 2**-53 = 0.9999999999999999, every c | `(x + 1)/2` rounds to 1.0 (tie to even) before the beta call (pattern 10) | pdf inf / 0.0, logpdf +-inf, sf 0.0 for 0.49; cdf 1.0 for 0.509 | loss |
| F3 | pdf, logpdf | large c, bulk and tails: c = 1e6 severe (pdf), c = 1e3 moderate; c = 10 minor | `xlog1py + xlogy - betaln(c/2, c/2)` cancel terms of size ~0.69*c to an O(log c) result (pattern 5) | pdf 4.1e6 ulp, logpdf 8.8e5 ulp | severe |
| F4 | cdf, sf | large c (>~ 1e3), bulk and tails: c = 1e6 up to 1.35e4 ulp at 30 sd, 1.82e3 at 2 sd, 379 near x = 0; c = 1e8 1.07e3 near x = 0 | `(x + 1)/2` rounds (pattern 10), amplified by the beta cdf's sensitivity to y ~ sqrt(c); plus Boost `ibeta` large-shape prefactor error (beta F8) of the same order | 1.35e4 ulp (cond (arg) 902) | moderate |

### F1: ppf cancels in 2*p - 1 near the median

- **Code:** `_ppf`, line 9088: `2*beta._ppf(q, c/2, c/2) - 1` -> line 779 `scu._beta_ppf(q, a, b)` (Boost `ibeta_inv`).
- **Mechanism:** the result x = 2p - 1 is small when p is close to 1/2. Any
  absolute error in p carries over to x unchanged (pattern 5). A correctly
  rounded p already has an absolute error of up to 2**-54 = 5.6e-17, so the
  relative error in x is about 5.6e-17/|x|, while the condition number of ppf
  stays O(1) for large c and ~|q - 1/2|**-1 near the median. Boost
  `ibeta_inv(c/2, c/2, q)` is not correctly rounded there either
  (`scratch/rdist/probe_median.txt`). The error in p is 0.06 to 4 ulp(0.5) for
  c = 0.5 and 1.6, 0.17 to 0.93 for c = 10, and 129 ulp(0.5) at c = 3,
  q = 0.4999999999999999 (the semicircular F2 point). For c >= 1e3 it is
  <= 0.12 ulp(0.5) for |q - 1/2| <= 1e-8, so the loss there is structural and not caused by Boost
  (pattern 10).
- **Regime:** loss (0.0, or 2.84e-14 with the wrong sign at c = 3) for
  |q - 1/2| <= 1e-16 at every c, up to 1e-15 at c = 1e3 and 1e-14 at c = 1e6.
  Severe for |q - 1/2| up to 1e-14 (c = 0.5), 1e-10 (c = 3), 1e-8
  (c = 0.001, 1.6, 10), 1e-6 (c = 1e3) and 1e-4 (c = 1e6); moderate up to
  1e-4 (c = 10, 1e3) and 1e-2 (c = 1e6). For large c the
  whole bulk is affected because |x| ~ 1/sqrt(c): 1.97e3 to 3.95e3 ulp at
  q = 0.4/0.6 for c = 1e6 (moderate), 94.6 to 157 ulp at c = 1e3 (minor).
- **Reproducer:** `python accuracy.py refs/rdist_ref.py --at ppf 0.4999999999999999 3`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.4999999999999999, 3)` | 2.842170943040401e-14 | -1.743934249004316e-16 | inf | 4.5e+15 | loss |
  | `ppf(0.5000000000000001, 1.6)` | 0.0 | 2.5527222762810825e-16 | inf | 4.5e+15 | loss |
  | `ppf(0.4999999999999999, 0.001)` | 0.0 | -2.2219847656146154e-13 | inf | 4.5e+15 | loss |
  | `ppf(0.49999999999999, 1e6)` | 0.0 | -2.504625408360864e-17 | inf | 5.0e+13 | loss |
  | `ppf(0.500000000000001, 1000)` | 0.0 | 7.922299322819136e-17 | inf | 5.0e+14 | loss |
  | `ppf(0.499999999999999, 1000)` | -1.1102230246251565e-16 | -7.922299322819136e-17 | 1.8e+15 | 5.0e+14 | severe |
  | `ppf(0.49999999999999, 3)` | -1.532107773982716e-14 | -1.5695408241038843e-14 | 1.1e+14 | 5.0e+13 | severe |
  | `ppf(0.500001, 1e6)` | 2.506628901362852e-09 | 2.5066289044417545e-09 | 5.53e+06 | 5e+05 | severe |
  | `ppf(0.4999, 1e6)` | -2.506628926646215e-07 | -2.5066289275371893e-07 | 1.6e+06 | 5e+03 | severe |
  | `ppf(0.4999, 10)` | -8.126984198575382e-05 | -8.126984198552566e-05 | 1.26e+04 | 5e+03 | moderate |
  | `ppf(0.6, 1e6)` | 0.00025334716240710264 | 0.0002533471624073247 | 3.95e+03 | 6.13 | moderate |
  | `ppf(0.4, 1e6)` | -0.00025334716240721367 | -0.0002533471624073247 | 1.97e+03 | 4.09 | moderate |
  | `ppf(0.6, 1000)` | 0.008013413319785112 | 0.008013413319785391 | 157 | 6.13 | minor |

  (26 loss, 43 severe, 9 moderate points in the bulk region over all seven
  parameter sets; every parameter set has a loss.)

- **Fix direction:** use X**2 ~ Beta(1/2, c/2):
  `x = sign(q - 1/2)*sqrt(sc.betaincinv(0.5, c/2, abs(2*q - 1)))`. `2*q - 1` is
  exact for q in [0.25, 1]. A probe (`scratch/rdist/probe_ppf_fix.txt`) measured 0.004 to 3.1 ulp at
  all the points above for c = 0.001 ... 1e6, including both q = 1/2 +- 1 ulp.
  Keep the current form in the tails, where p is near 0 or 1.

### F2: x = 1 - 2**-53 is rounded to the endpoint

- **Code:** lines 9079, 9082, 9085: `beta._logpdf((x + 1)/2, ...)`,
  `beta._cdf((x + 1)/2, ...)`, `beta._sf((x + 1)/2, ...)`; pdf = `exp(_logpdf)` (9076).
- **Mechanism:** for x in [0.5, 1) the sum `x + 1` lies in [1.5, 2), where the
  double spacing is twice that of x. For x = 1 - 2**-53, `x + 1 = 2 - 2**-53` is
  a tie and rounds to even, i.e. 2.0. So y = 1.0 exactly (pattern 10, argument
  rounded before the call). Then `xlog1py(a-1, -1)` is -inf for c > 2 and +inf
  for c < 2, `betaincc(a, a, 1) = 0` and `betainc(a, a, 1) = 1`. For other x in
  (0.5, 1) with the last bit set, 1 - y gets a relative error 2**-53/(1 - x). That
  is within the conditioning (the harness rates these points ok, up to 1.45e16
  ulp at 1 - x = 3.3e-16 with cond 1.5e16) but avoidable. The mirror point
  x = -1 + 2**-53 is exact (`x + 1` is exact by Sterbenz) and all methods are
  accurate there.
- **Regime:** x = 0.9999999999999999, every c. For small c the true sf is
  large (0.49 at c = 0.001), so the 0.0 / 1.0 results are badly wrong, not just
  tail effects.
- **Reproducer:** `python accuracy.py refs/rdist_ref.py --at sf 0.9999999999999999 3`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(0.9999999999999999, 0.001)` | 0.0 | 0.49072973358385713 | inf | 4.5e+12 | loss |
  | `sf(0.9999999999999999, 0.5)` | 0.0 | 4.6555161346546215e-05 | inf | 2.3e+15 | loss |
  | `sf(0.9999999999999999, 3)` | 0.0 | 7.021327110271799e-25 | inf | 1.4e+16 | loss |
  | `pdf(0.9999999999999999, 1.6)` | inf | 587.6498321743347 | inf | 1.8e+15 | loss |
  | `pdf(0.9999999999999999, 3)` | 0.0 | 9.48637384723993e-09 | inf | 4.5e+15 | loss |
  | `logpdf(0.9999999999999999, 1000)` | -inf | -17983.248352063132 | inf | 2.5e+14 | loss |
  | `logpdf(0.9999999999999999, 0.001)` | inf | 28.424036366633196 | inf | 3.2e+14 | loss |
  | `cdf(0.9999999999999999, 0.001)` | 1.0 | 0.5092702664161429 | 4.3e+15 | 4.3e+12 | severe |
  | `cdf(0.9999999999999999, 0.5)` | 1.0 | 0.9999534448386534 | 2.1e+11 | 1.0e+11 | severe |

  (pdf, logpdf and sf are a loss at this x for all seven parameter sets.)

- **Fix direction:** use the symmetry for x > 0: `sf(x) = beta._cdf((1 - x)/2, a, a)`,
  `cdf(x) = beta._sf((1 - x)/2, a, a)` (`(1 - x)/2` is exact for x in [0, 1]), and
  `logpdf = xlogy(c/2 - 1, (1 - x)*(1 + x)) - log B(1/2, c/2)` (see F3).

### F3: logpdf and pdf lose about c ulp for large c

- **Code:** `_logpdf`, line 9079: `-np.log(2) + beta._logpdf((x + 1)/2, c/2, c/2)`;
  lines 765-767: `xlog1py(b - 1.0, -x) + xlogy(a - 1.0, x) - betaln(a, b)`;
  `_pdf`, line 9076: `np.exp(self._logpdf(x, c))`.
- **Mechanism:** with a = c/2 and y near 1/2, the two log terms are each about
  -a*log(2) and `betaln(a, a)` is about -2a*log(2). They cancel to
  logpdf = O(log c) (pattern 5). `sc.betaln(a, a)` itself is only 3 to 5 ulp off,
  but that is an absolute error of 4.6e-13 at a = 500 and 7.8e-10 at
  a = 5e5. That absolute error passes unchanged into logpdf, and through
  `exp` it becomes a relative error of the pdf (`scratch/rdist/probe_logpdf.py`).
  At c = 1e6, x = 0, the terms are -346573 each and betaln is -693152; the
  result is 5.99. For c = 10 the cancellation is mild and costs up to 63 ulp
  where logpdf is close to 0.
- **Regime:** error about c * 1e-16 absolute in logpdf, at every x where the
  pdf is not negligible. pdf: 1.4e3 to 2.5e3 ulp at c = 1e3 (moderate), 3.4e6
  to 4.1e6 ulp at c = 1e6 (severe), 1.14e9 ulp at c = 1e8 (probe). logpdf:
  up to 4.69e3 ulp at c = 1e3 and 8.8e5 ulp at c = 1e6 (moderate, the result
  is O(1)). The condition number with respect to x is <= 25 at all these points.
- **Reproducer:** `python accuracy.py refs/rdist_ref.py --at pdf 0.0 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.004999997500001875, 1e6)` | 0.0014865426012902333 | 0.001486542602640429 | 4.1e+06 | 25 | severe |
  | `pdf(0.0, 1e6)` | 398.94218034735945 | 398.94218066587507 | 3.6e+06 | 0 | severe |
  | `pdf(0.029999985000011248, 1e6)` | 1.20498686924025e-193 | 1.204986870235738e-193 | 3.7e+06 | 901 | severe |
  | `pdf(0.0, 1000)` | 12.612509089175546 | 12.612509089180401 | 1.73e+03 | 0 | moderate |
  | `pdf(-0.8, 1000)` | 4.963058280590591e-221 | 4.963058280593386e-221 | 2.54e+03 | 1.77e+03 | moderate |
  | `logpdf(0.00199999900000075, 1e6)` | 3.988818494994293 | 3.9888184957767976 | 8.8e+05 | 1 | moderate |
  | `logpdf(0.0, 1e6)` | 5.988816494979064 | 5.988816495777464 | 6.0e+05 | 0 | moderate |
  | `logpdf(0.0632139541241014, 1000)` | 0.5366884389897583 | 0.536688438990317 | 4.69e+03 | 7.46 | moderate |
  | `logpdf(1e-08, 1000)` | 2.53468910632745 | 2.5346891063280124 | 999 | 3.94e-14 | minor |
  | `logpdf(-0.19999999999999996, 10)` | 0.044107216265050764 | 0.044107216265050146 | 63.1 | 7.56 | minor |

- **Fix direction:** use the rdist form directly,
  `logpdf = xlog1py(c/2 - 1, -x*x) - log B(1/2, c/2)`, with
  `log B(1/2, c/2) = 0.5*log(pi) - log(Gamma(a + 1/2)/Gamma(a))` and an accurate
  gamma ratio. The x-dependent term is then small near x = 0, and the constant is
  O(log c) instead of O(c). Probe (`scratch/rdist/probe_fix.py`): with
  `log(sc.poch(a, 0.5))` for the gamma ratio the error drops to 0.01 to 0.3 ulp at
  c = 1e6 and 1e8. But `sc.poch(a, 0.5)` is itself 138 to 9.35e3 ulp off for
  a = 50 to 5000 (`scratch/rdist/poch.txt`), so the result still has 674 ulp at c = 1e3. An
  asymptotic series for the ratio (or `betaln(0.5, a)` where accurate) is needed for moderate a.

### F4: cdf and sf lose 1e3 to 1e4 ulp for large c

- **Code:** lines 9082, 9085: `beta._cdf((x + 1)/2, c/2, c/2)`, `beta._sf((x + 1)/2, c/2, c/2)`
  -> `sc.betainc(a, a, y)` / `sc.betaincc(a, a, y)` (Boost `ibeta`).
- **Mechanism:** two parts of similar size, separated by probes
  (`scratch/rdist/probe_cdf0.py`, `scratch/rdist/probe_tail.txt`) that evaluate
  I_y(a, a) exactly at the rounded y.
  1. `x + 1` rounds to the double grid near 1, which gives an absolute error of up to
     5.6e-17 in y (pattern 10). With respect to y the beta cdf has condition
     y f/F, about 0.8*sqrt(c)/F near the median and growing in the tails. With
     respect to x, rdist's condition x f/F is small because |x| ~ 1/sqrt(c). So
     this error is beyond the conditioning of rdist. Its size is 181 to 219 ulp near
     x = 0 and 575 to 3.76e3 ulp at 2 to 30 sd for c = 1e6, and 941 to 2.18e3 ulp
     near x = 0 for c = 1e8.
  2. Boost `ibeta` for large symmetric shapes is itself off by 53 to 2.07e3 ulp
     near the median and 0.7e3 to 9.8e3 ulp in the tails, measured against the exact
     value at the same rounded y. This is the large-shape prefactor
     problem of beta F8 (pattern 5/6) and lives in `scipy.special`. The two parts
     sometimes cancel (4.4 ulp at c = 1e8, x = 1e-8).
- **Regime:** c >~ 1e3, everywhere the cdf is not 0 or 1. At c = 1e6: 1.35e4 ulp at
  30 sd, 1.82e3 ulp at 2 sd, up to 379 ulp near x = 0 (cdf ~ 1/2). At c = 1e8
  (single points, not in PARAMS): 1.07e3 ulp near x = 0. At c = 1e3: up to
  356 ulp (minor).
- **Reproducer:** `python accuracy.py refs/rdist_ref.py --at cdf -0.029999985000011248 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-0.029999985000011248, 1e6)` | 4.0085698223056e-198 | 4.008569822293553e-198 | 1.35e+04 | 902 | moderate |
  | `sf(0.029999985000011248, 1e6)` | 4.0085698223056e-198 | 4.008569822293553e-198 | 1.35e+04 | 902 | moderate |
  | `cdf(-0.0199999900000075, 1e6)` | 2.6461574146971615e-89 | 2.646157414702139e-89 | 8.47e+03 | 401 | moderate |
  | `sf(0.00999999500000375, 1e6)` | 7.601215220966033e-24 | 7.601215220975822e-24 | 5.8e+03 | 101 | moderate |
  | `cdf(-0.00199999900000075, 1e6)` | 0.02275010495260729 | 0.022750104952598094 | 1.82e+03 | 4.75 | moderate |
  | `cdf(3e-09, 1e8)` | 0.5000119682682618 | 0.5000119682683803 | 1.07e+03 | 2.39e-05 | moderate |
  | `sf(3e-09, 1e8)` | 0.4999880317317383 | 0.49998803173161965 | 1.07e+03 | 2.39e-05 | moderate |
  | `cdf(1e-08, 1e6)` | 0.5000039894217645 | 0.5000039894218066 | 379 | 7.98e-06 | minor |

  (cond (shapes) is about half of cond (arg), e.g. 451 at the first row.)

- **Fix direction:** avoid rounding x into y. Use X**2 ~ Beta(1/2, c/2):
  `cdf(x) = 0.5*sc.betaincc(0.5, c/2, x*x)` for x < 0 and
  `0.5 + 0.5*sc.betainc(0.5, c/2, x*x)` for x >= 0 (sf by symmetry). Probe:
  0.08 to 2.6 ulp near the median (c = 1e3, 1e6, 1e8), 1.1 to 26 ulp at
  c = 1e3, x = -0.3, and 67 to 340 ulp at 20 to 30 sd for c = 1e6. The remaining
  tail error is Boost's (part 2; see beta F8).

## RuntimeWarnings

- pdf, c = 1000 and 1e6: `underflow encountered in exp` from `np.exp(_logpdf)`
  (line 9076). The harness records it for every point of the batch. It comes
  from the points near x = +-1 where the true pdf is below the normal range
  (e.g. exp(-17983)), so 0 is the correct result there. `--at pdf 0.0 1000`
  emits no warning. Spurious; the accuracy problems of F3 are unrelated.
- No other warnings in the measured methods. In particular the F2 results
  (inf/-inf/0) are returned without a warning (`xlog1py` returns +-inf silently).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | c = 0.001, 0.5, 1.6, 3 over the full grid: both endpoints from 2 ulp to 0.1 inside, x = 0, +-1e-300, +-1e-20, bulk | within conditioning (near x = 1 up to 1.1e15 ulp with cond 3e15, see F2 for the avoidable part); bulk median 2.01 ulp |
| logpdf | same | within conditioning except F2; 0.74 ulp median in the low tail |
| cdf | c <= 10, full grid; left tail (x -> -1) for every c (`x + 1` exact) | 368 ulp at c = 1.6 near x = 1 (cond 588, within); 3.65 ulp at c = 3, 3 ulp at c = 10 |
| sf | c <= 10, full grid | within conditioning except F2; low tail 0.5 ulp |
| ppf | tails q in [1e-300, 1e-2] and 1 - q in [1e-16, 1e-2], all parameter sets | 436 ulp at c = 1e3, q = 2.2e-101 (cond 0.0017) and up to 66 ulp at c = 1e6 (32 minor points in the low tail). This is the Boost `ibeta_inv` large-shape tail error of beta F7, and too small for a finding without an rdist-specific mechanism. The iteration-cap failures of beta F1 were not hit for a = b in [5e-4, 5e5]. Max 91 ulp for c <= 10 away from the median (0.49 at c = 10, part of F1). |

## Suspected, not confirmed

- ppf for tiny c (c = 0.001) in the bulk returns -1.0 or 1.0, and that is the correct
  double (1 - |x| ~ 1e-600). Near the median the result is also covered by F1. No
  separate problem was found.
- Boost `ibeta_inv` iteration-cap failures (beta F1) for symmetric shapes:
  none were seen at the 7 measured c values with q down to 1e-300. Other c in
  between were not measured.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| isf | ppf(1 - q) | `-ppf(q)` by symmetry (or `2*beta._isf(q, a, a) - 1`) | small q: `1 - q` rounds. `isf(1e-20, 1e6)` = 1.0 vs 0.0092621437525438904 (4.8e17 ulp; -ppf gives 39.7 ulp); `isf(1e-10, 3)` 75 ulp vs 0.23 ulp (`scratch/rdist/probe_inherited.txt`) |
| logcdf | log(cdf) below the median, log1p(-sf) above | a log-space incomplete beta for the left tail: `a*log(y) + a*log1p(-y) - log(a) - betaln(a, a) + log(CF)` (the Boost/continued-fraction prefix in log form), or an asymptotic series `logpdf(x) + log((1 - x*x)/((c - 2)*abs(x))) + ...` for large c | cdf underflows for large c: `logcdf(-0.05, 1e6)` = -inf vs -1256.3964694498814; `logcdf(-0.9, 1000)` = -inf vs -834.63354303100137 |
| logsf | log(sf) above the median, log1p(-cdf) below | `logcdf(-x)` by symmetry, with the log-space tail above | `logsf(0.05, 1e6)` = -inf vs -1256.3964694498814; also inherits F2 at x = 1 - 2**-53 |

## Shared code

- **`scu._beta_ppf` / Boost `ibeta_inv` near p = 1/2 (special-functions.md row 39):**
  now measured for c = 0.5 ... 1e6 (`scratch/rdist/probe_median.txt`). The
  absolute error is 0.06 to 4 ulp(0.5) for a = 0.25 and 0.8, 129 ulp(0.5) at
  a = 1.5 (one point; 0.5 to 1.7 elsewhere), <= 0.93 for a = 5 and <= 0.12 for
  a >= 500 (|q - 1/2| <= 1e-8). For beta.ppf itself this is only a few ulp. The rdist/semicircular
  loss (F1) is caused mainly by the `2p - 1` cancellation and would occur with a
  correctly rounded p as well.
- **`sc.betaln(a, a)` for large a** is accurate relative to itself (3 to 5 ulp)
  but is used in a cancelling sum in `beta._logpdf`. That is the same mechanism
  as beta F4 and is inherited by rdist (F3).
- **Boost `ibeta` large-shape prefactor** (beta F8) shows up in rdist as part 2 of F4
  (0.7e3 to 9.8e3 ulp at c = 1e6 against the exact value at the same argument).
- **`sc.poch(a, 0.5)`** (not used by rdist; found while testing the F3 fix):
  138 ulp at a = 50, 1.71e3 at a = 500 and 9.35e3 ulp at a = 5000. It is accurate
  (< 1 ulp) for a = 5 and a >= 5e4 (`scratch/rdist/poch.txt`).
- `rdist._ppf` is used by `semicircular.ppf` (c = 3), which inherits F1
  (semicircular F2). semicircular's own pdf/cdf do not call rdist.

## Existing GitHub issues

none found (web search; gh unavailable). Related: PR #18586 (ENH: override sf for rdist; the current `_sf`).

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(0.4999999999999999, 3)` | 2.842170943040401e-14 | -1.7439342490043159e-16 | verified | Reference by root solve of F(x) = 1/2 + sign(x)/2 I_{x^2}(1/2, c/2); \|cdf(x_ref) - q\| < 1e-180. Wrong sign, loss. Mechanism matches line 9088 (`2*p - 1`). |
| F1 | `ppf(0.5000000000000001, 1.6)` | 0.0 | 2.5527222762810827e-16 | verified | loss |
| F1 | `ppf(0.4999999999999999, 0.001)` | 0.0 | -2.2219847656146154e-13 | verified | loss |
| F1 | `ppf(0.49999999999999, 1e6)` | 0.0 | -2.5046254083608641e-17 | verified | loss |
| F1 | `ppf(0.499999999999999, 1000)` | -1.1102230246251565e-16 | -7.922299322819136e-17 | verified | 1.81e15 ulp, severe |
| F1 | `ppf(0.500001, 1e6)` | 2.5066289044417545e-09 | 2.5066289013628522e-09 | verified | 5.53e6 ulp, severe. The report table swaps the scipy and reference columns in this row (the harness `--at` output confirms SciPy returns ...44417545e-09); correct the row. |
| F1 | `ppf(0.6, 1e6)` | 0.00025334716240710264 | 0.00025334716240732469 | verified | 3.95e3 ulp, moderate |
| F2 | `sf(0.9999999999999999, 0.001)` | 0.0 | 0.49072973358385713 | verified | Reference via sf(x) = I_{1-x^2}(c/2, 1/2)/2. Mechanism (`(x + 1)/2` rounds to 1.0, lines 9079/9082/9085) confirmed. |
| F2 | `pdf(0.9999999999999999, 1.6)` | inf | 587.64983217433472 | verified | direct (1 - x^2)^(c/2-1)/B(1/2, c/2) |
| F2 | `logpdf(0.9999999999999999, 1000)` | -inf | -17983.248352063133 | verified | loss |
| F2 | `cdf(0.9999999999999999, 0.001)` | 1.0 | 0.50927026641614287 | verified | 4.34e15 ulp, severe |
| F3 | `pdf(0.004999997500001875, 1e6)` | 0.0014865426012902333 | 0.0014865426026404292 | verified | 4.09e6 ulp, severe. Mechanism (betaln cancellation, lines 765-767) matches. |
| F3 | `pdf(0.0, 1e6)` | 398.94218034735945 | 398.94218066587504 | verified | 3.6e6 ulp |
| F3 | `logpdf(0.00199999900000075, 1e6)` | 3.988818494994293 | 3.9888184957767977 | verified | 8.83e5 ulp, moderate |
| F4 (moderate) | `cdf(-0.029999985000011248, 1e6)` | 4.0085698223056e-198 | 4.0085698222935532e-198 | verified | 1.35e4 ulp; reference via I_{1-x^2}(c/2, 1/2)/2 |
| F4 (moderate) | `cdf(3e-09, 1e8)` | 0.5000119682682618 | 0.50001196826838033 | verified | 1.07e3 ulp |
| Shared code | Boost `ibeta_inv(a, a, q)` near q = 1/2, error of p in units of 2**-53 | see note | see note | partly verified | Measured independently in two ways (`scratch/rdist/verify/shared_median.txt`, `shared_median3.txt`; Newton residual agrees at 100 and 200 digits). Confirmed: 0.07 to 2.6 at a = 0.25 and 0.15 to 1.9 at a = 0.8 (within the stated 0.06 to 4); outlier +128.8 at a = 1.5, q = 0.4999999999999999 (also +128.4 at q = 0.49999999999999994), other a = 1.5 points 0.07 to 1.7. Not confirmed: "<= 0.12 for a >= 500 (\|q - 1/2\| <= 1e-8)". The analyst probed only q <= 1/2 (plus 1/2 + 1 ulp). Above the median: a = 500 gives -0.46 (q = 0.50000001), +0.59 (q = 0.5000000001), -0.36 (q = 0.500000000000001); a = 5e5 gives +0.71 (q = 0.500000000001), -0.53 (q = 0.50000001). Correct the bound to <= ~0.7 ulp(0.5). The conclusion still holds: below 1 ulp(0.5), so the F1 loss is structural. |

Verified by dist-accuracy-verifier: 4 of 4 verified.
