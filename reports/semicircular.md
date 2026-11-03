# semicircular: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `semicircular_gen`, lines 9591-9648
- **Implemented:** pdf (9625-9626), logpdf (9628-9629), cdf (9631-9632), ppf (9634-9635, via `rdist._ppf(q, 3)` = `2*beta._ppf(q, 1.5, 1.5) - 1`, lines 9087-9088 and 778-779, Boost `ibeta_inv` through `scu._beta_ppf`)
- **Inherited (not measured):** logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** `rdist` (c = 3; ppf delegated), `beta` (through rdist); helpers: none; special functions: `sc.log1p`, `np.arcsin`, `scu._beta_ppf` (Boost `ibeta_inv`)
- **Reference:** `refs/semicircular_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in cdf (left tail) and ppf (q near 1/2)

## Summary

pdf and logpdf are within conditioning everywhere, and ppf is accurate in both
tails. The cdf formula `0.5 + (x*sqrt(1-x*x) + arcsin(x))/pi` cancels in the
left tail. The relative error grows like (1+x)**-1.5. It is severe from
1 + x ~ 1e-4 on, and for 1 + x <= ~1e-11 the cdf returns 0 or a negative value
(-1.1e-16). The same `1 - x*x` rounding costs up to 59 ulp near x = 1 (minor).
ppf near q = 1/2 subtracts `2*p - 1` with a Boost `ibeta_inv` result `p` that
is off by 1.7 to 129 ulp of 0.5. This gives wrong-sign results for
|q - 1/2| ~ 1e-16 and severe errors for |q - 1/2| up to at least 1e-10.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | x -> -1: 1 + x <= ~1e-4 severe, 1 + x <= ~1e-11 loss | `0.5 + (...)/pi` cancels to O(eps) absolute error; `1 - x*x` rounding adds eps/sqrt(1+x) (patterns 5, 1) | -1.1e-16 for 6.0e-19; 0.0 for 7.0e-25 | loss |
| F2 | ppf | \|q - 1/2\| <= ~1e-10 | `2*p - 1` cancels; Boost `ibeta_inv(1.5, 1.5, q)` has an absolute error of 1.7 to 129 ulp(0.5) near p = 1/2 (patterns 5, 10) | 2.84e-14 for -1.74e-16 (wrong sign) | loss |
| F3 | cdf | x -> 1: 1 - x in [1e-9, 1e-6] | `1 - x*x` rounding inside sqrt (pattern 1) | 59 ulp | minor |

### F1: cdf cancels to zero or a negative value in the left tail

- **Code:** `_cdf`, line 9632: `0.5+1.0/np.pi*(x*np.sqrt(1-x*x) + np.arcsin(x))`
- **Mechanism:** with d = 1 + x, the true cdf is about (4*sqrt(2)/(3*pi)) d**1.5. It comes out of three
  canceling terms. `arcsin(x) ~ -pi/2 + sqrt(2d)` and `x*sqrt(1-x*x) ~ -sqrt(2d)`
  cancel to O(d**1.5). The `0.5 + (-0.5 + small)` step then cancels again
  (pattern 5), which leaves an absolute error of about eps. Also,
  `1 - x*x` takes its rounding error from `x*x` (about 1e-16 absolute,
  pattern 1). In the sqrt this becomes an absolute error of about eps/sqrt(2d)
  in the `x*sqrt(1-x*x)` term. A probe measured 3.9e-14 at d = 1e-8, which
  dominates the total error there. With `(1-x)*(1+x)` that term is
  exact to 1e-20, but the cancellation against 0.5 still leaves 3.65e5 ulp at d = 1e-4 and
  8.4e17 ulp at d = 1e-12 (`scratch/semicircular/probe.py`). So the
  formula needs restructuring, not only `(1-x)*(1+x)`.
- **Regime:** relative error about eps/d**1.5, against a condition number of
  about 1.5/d. Measured: moderate at d = 1e-3 (2.44e3 ulp), severe from
  d = 1e-4 (3.8e6 ulp) to d = 1e-10. For d <= 1e-11 (cdf < 1.9e-17) the
  result is 0.0 or -1.1102230246251565e-16 (wrong sign). This affects
  `semicircular.cdf` and also the inherited `logcdf`, which returns nan
  there.
- **Reproducer:** `python accuracy.py refs/semicircular_ref.py --at cdf -0.999999999999`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-0.999999999999)` | -1.1102230246251565e-16 | 6.001909610025613e-19 | inf | 1.5e+12 | loss |
  | `cdf(-0.9999999999999999)` | 0.0 | 7.021327110271799e-25 | inf | 1.4e+16 | loss |
  | `cdf(-0.99999999999)` | 0.0 | 1.8980336846756176e-17 | inf | 1.5e+11 | loss |
  | `cdf(-0.9999999999)` | 3.885780586188048e-16 | 6.002109519215751e-16 | 1.6e+15 | 1.5e+10 | severe |
  | `cdf(-0.99999999)` | 6.125655538369301e-13 | 6.002108810616271e-13 | 9.3e+13 | 1.5e+08 | severe |
  | `cdf(-0.999999)` | 6.002083274836423e-10 | 6.002107874323203e-10 | 1.8e+10 | 1.5e+06 | severe |
  | `cdf(-0.9999)` | 6.002018747008897e-07 | 6.002018741944231e-07 | 3.8e+06 | 1.5e+04 | severe |
  | `cdf(-0.999)` | 1.8977487186710995e-05 | 1.897748718670073e-05 | 2.44e+03 | 1.5e+03 | moderate |
  | `cdf(-0.99)` | 0.0005993097557009386 | 0.0005993097557010032 | 486 | 148 | minor |

- **Fix direction:** for x < 0 use `sc.betainc(1.5, 1.5, (1+x)/2)` (that is
  `rdist._cdf(x, 3)`; `(1+x)/2` is exact for x in [-1, 0]). The probe measured
  0.3 to 1.9 ulp at the points above. For x >= 0 use `1 - cdf(-x)` or `betaincc`.
  Another option is a series in d for small 1 + x.

### F2: ppf wrong sign / severe relative error near q = 1/2

- **Code:** `_ppf`, line 9635: `rdist._ppf(q, 3)` -> line 9088
  `2*beta._ppf(q, c/2, c/2) - 1` -> line 779 `scu._beta_ppf(q, a, b)` (Boost `ibeta_inv`).
- **Mechanism:** the result x = 2p - 1 is small when p ~ 1/2, so any absolute
  error in p carries over to x unchanged (pattern 5). Even a correctly rounded p (0.5 ulp
  = 5.6e-17 absolute) would be borderline against the conditioning of ppf
  (cond ~ 0.785/|x|). Boost `ibeta_inv(1.5, 1.5, q)` is not correctly rounded
  near 1/2. At q = 0.49999999999999 it returns p with an error of 1.69 ulp(0.5). At
  q = 0.4999999999999999 it returns 0.5000000000000142 (true
  0.49999999999999991), off by 129 ulp(0.5) and on the wrong side of 1/2. At
  q = 0.5000000000000001 it returns exactly 0.5. `sc.betaincinv(1.5, 1.5, q)`
  returns the same values (pattern 10; see Shared code).
- **Regime:** |q - 1/2| <= ~1e-10 measured beyond conditioning. The absolute error in x stays at
  about 2e-16 to 4e-16, so the relative error is about 3e-16/|x|. At
  |q - 1/2| = 1e-8 the error is within conditioning. Wrong sign or 0 for
  |q - 1/2| ~ 1e-16.
- **Reproducer:** `python accuracy.py refs/semicircular_ref.py --at ppf 0.4999999999999999`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.4999999999999999)` | 2.842170943040401e-14 | -1.743934249004316e-16 | inf | 4.5e+15 | loss |
  | `ppf(0.49999999999999994)` | 2.842170943040401e-14 | -8.71967124502158e-17 | inf | 9.0e+15 | loss |
  | `ppf(0.5000000000000001)` | 0.0 | 1.743934249004316e-16 | inf | 4.5e+15 | loss |
  | `ppf(0.49999999999999)` | -1.532107773982716e-14 | -1.5695408241038843e-14 | 1.1e+14 | 5.0e+13 | severe |
  | `ppf(0.499999999999)` | -1.5705214906347464e-12 | -1.5707615780781874e-12 | 6.9e+11 | 5.0e+11 | severe |
  | `ppf(0.4999999999)` | -1.5707946054988042e-10 | -1.5707964567631673e-10 | 5.3e+09 | 5.0e+09 | severe |

  (and the mirror points q = 0.50000000000001, 0.500000000001, 0.5000000001
  with the same errors.)

- **Fix direction:** for q near 1/2, solve `x*sqrt((1-x)*(1+x)) + arcsin(x) = pi*(q - 0.5)`
  directly. `q - 0.5` is exact for q in [0.25, 1]. Start from
  x0 = pi*(q - 0.5)/2 and take one or two Newton steps, or use the series x = t + t**3/6 + ...
  with t = pi*(q-0.5)/2. Then use symmetry `ppf(q) = -ppf(1-q)` for q > 1/2.

### F3: cdf loses up to 59 ulp near x = 1

- **Code:** `_cdf`, line 9632: `np.sqrt(1-x*x)`
- **Mechanism:** the rounding of `x*x` (about 1e-16 absolute) becomes a
  relative error eps/(2d) in `1 - x*x` (d = 1 - x). In the sqrt this is an
  absolute error of about eps/sqrt(2d) in `x*sqrt(1-x*x)`: 3.9e-14 at
  d = 1e-8, and 1.24e-14 after the division by pi (pattern 1). The cdf is
  close to 1 and its condition number is close to 0, so this shows directly.
- **Regime:** 1 - x between ~1e-9 and ~1e-6. Measured: 59 ulp at
  x = 0.9999999888977698, 56 at 0.99999999, 15.5 at 0.999999999, 13 at
  0.9999999, 11 at 0.999999.
- **Reproducer:** `python accuracy.py refs/semicircular_ref.py --at cdf 0.99999999`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(0.9999999888977698)` | 0.999999999999311 | 0.9999999999992979 | 59 | 9.49e-05 | minor |
  | `cdf(0.99999999)` | 0.9999999999993874 | 0.9999999999993998 | 56 | 9e-05 | minor |

- **Fix direction:** `np.sqrt((1-x)*(1+x))`. The probe measured 0.11 ulp at
  x = 0.99999999.

## RuntimeWarnings

- pdf, logpdf, cdf: `underflow encountered in multiply` from `x*x` at
  x = +-1e-300. The results are accurate, so the warning is spurious.
- Inherited logcdf: `invalid value encountered in log` at
  x = -0.999999999999 because `_cdf` returns -1.1e-16 (F1). The result is nan.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid: x = 0, +-1e-300, +-1e-20, bulk, both endpoints from 1 ulp to 0.1 inside | 1.2e6 ulp at 1 - x = 1e-8 (cond 4.5e7, within). Without the `1 - x*x` rounding, `(1-x)*(1+x)` would give 0.45 ulp there (probe), but the error is within conditioning, so it is not a finding. |
| logpdf | same grid | 1.3e5 ulp near the endpoints (cond 4.9e6, within); 0.84 ulp in the bulk |
| cdf | bulk and x >= 0 apart from F3; x near 0 (cdf near 1/2) | 1.67 ulp in the bulk |
| ppf | q from 1e-300 to 1e-2 (left tail, result -1 + 2p rounds to about -1), q from 1 - 1e-16 to 1 - 1e-2, q = 0.25, 0.75, 0.5 +- 1e-8 and further from 1/2 | 0 ulp in the left tail, 0.506 ulp in the right tail, 2.9 ulp median in the bulk. The beta-report failures of Boost `ibeta_inv` at small q do not show for a = b = 1.5 (or they are absorbed by the -1). |

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | `cdf(-x)` by symmetry, with a cdf that is accurate in the left tail (e.g. `sc.betainc(1.5, 1.5, (1-x)/2)`, 1.23 ulp / 0.3 ulp at the demo points) | x -> 1: `sf(0.99999999)` = 6.126e-13 vs 6.0021e-13 (9.3e13 ulp); `sf(0.999999999999)` = 0.0 vs 6.0e-19 |
| logsf | log(sf) | `log(cdf(-x))` with the accurate left-tail cdf | x -> 1: `logsf(0.999999999999)` = -inf vs -41.957 |
| logcdf | log(cdf) | `log` of an accurate left-tail cdf (betainc form or series) | x -> -1, inherits F1: `logcdf(-0.999999999999)` = nan vs -41.957 |
| isf | ppf(1 - q) | `-ppf(q)` by symmetry | small q: `1 - q` rounds. `isf(1e-20)` = 1.0 vs 0.99999999999993477 (294 ulp; -ppf gives 0.22 ulp); `isf(1e-10)` 75 ulp vs 0.23 ulp |

## Shared code

- `scu._beta_ppf` (Boost `ibeta_inv`) and `sc.betaincinv` with a = b = 1.5
  are not correctly rounded near p = 1/2. For q = 0.4999999999999999,
  both return p = 0.5000000000000142 (true 0.49999999999999991), an
  absolute error of 129 ulp(0.5) on the wrong side of 1/2. For
  q = 0.5000000000000001 they return exactly 0.5. For beta itself this is
  only about 129 ulp. It becomes a loss after the `2p - 1` in `rdist._ppf`
  (line 9088). semicircular inherits it (F2). `rdist.ppf` for other c
  presumably has the same near-median cancellation, but only c = 3 was
  measured. The `reports/beta.md`
  findings on Boost `ibeta_inv` at small q are a separate regime.
- `rdist._cdf` (line 9082) is *not* used by semicircular. Its form
  `beta._cdf((x+1)/2, c/2, c/2)` is the accurate fix for F1.

## Existing GitHub issues

- #12796 (closed): nonmonotonicity in betaincinv, btdtri, stats.beta.ppf - related to F2 (inverse incomplete beta) but a different defect (fixed); the near-median inaccuracy of F2 is not reported
- F1, F3: none found

(web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-0.999999999999)` | -1.1102230246251565e-16 | 6.001909610025613e-19 | verified | Reference is `mp.quad` of the docstring pdf 2/pi*sqrt(1-t^2) over [-1, x]. Values at 100 and 200 dps agree to 30 digits and match the analyst's at all 8 points: 0.0 at -0.9999999999999999 and -0.99999999999, 1.59e15 ulp at -0.9999999999, 9.27e13 at -0.99999999, 1.85e10 at -0.999999, 3.8e6 at -0.9999, 2.44e3 at -0.999. The mechanism matches line 9632. |
| F2 | `ppf(0.4999999999999999)` | 2.842170943040401e-14 | -1.743934249004316e-16 | verified | Reference comes from `mp.findroot` on x*sqrt(1-x^2)+asin(x) = pi*(q-1/2), and quad-cdf(x_ref) - q = 0. The 100/200-dps values agree. Also confirmed: 0.0 vs 1.7439e-16 at q = 0.5000000000000001, and 1.07e14 / 6.88e11 / 5.31e9 ulp at 0.5 -+ 1e-14, 1e-12, 1e-10 (mirror points identical). `sc.betaincinv(1.5, 1.5, q)` is off by 128.8, -0.79 and 1.69 ulp(0.5) vs mpmath betainc inversion. The mechanism matches lines 9635 -> 9088 -> 779. |

Verified by dist-accuracy-verifier: 2 of 2 verified.
