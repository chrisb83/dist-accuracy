# truncpareto: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `truncpareto_gen`, lines 10501-10849 (methods 10569-10609)
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf (logpdf, logcdf, logsf have a closed form only for b > 0; for b < 0 they dispatch via `xpx.apply_where` to the generic `rv_continuous` log(pdf), log(cdf), log(sf))
- **Inherited (not measured):** none
- **Related:** none (no other distribution calls truncpareto code); helpers: `_check_fit_input_parameters` (fit only); special functions: none (numpy `expm1`, `log1p`, `pow` only)
- **Reference:** `refs/truncpareto_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (1.8, 5.3), (2, 5), (-2, 5), (1e-8, 5), (-1e-8, 5), (2, 1.00000001), (1000, 5), (-1000, 5), (-2, 1e200), (-400, 5)
- **Worst finding:** loss in pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf

## Summary

For moderate shapes (the example sets, b = -400 ... 1000 with c = 5) every method
is accurate to a few ulp or within conditioning. Four regimes fail. (1) For b < 0
with |b| log c above about 709, `c**b` underflows, so `1/c**b` is inf and every
method returns nan, inf, 0 or -inf. (2) When |b| log c is small (tiny |b| or c
close to 1), the normaliser `1 - 1/c**b` cancels. This costs about 1/(|b| log c)
ulp in pdf, cdf, sf, ppf, isf and the b < 0 logpdf (2e7 to 5e7 ulp at
|b| = 1e-8 or c = 1 + 1e-8). (3) For tiny |b|, `1 - x**-b` and `x**-b - c**-b`
round to 0 near the support endpoints, so cdf, sf, logcdf and logsf return
0 or -inf. (4) For large b > 0, logsf returns -inf once `x**-b` underflows.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | all eight | b < 0, \|b\| log c > ~709 (pdf from ~703) | `c**b` underflows, `1/c**b` = inf; `x**-(b+1)` overflows (pattern 3) | nan / inf / 0 / -inf | loss |
| F2 | cdf, logcdf, sf, logsf | tiny \|b\|, x within ~eps/\|b\| of 1 or of c | `1 - x**-b`, `x**-b - c**-b` round to 0 (patterns 1, 2) | 0.0 for 6.2e-13; -inf for -28.1 | loss |
| F3 | logsf | b > 0, b log x > ~708 | `log(x**-b - c**-b)`: `x**-b` subnormal/underflows (patterns 3, 4) | -inf for -788.5; 3.4e12 ulp | loss |
| F4 | pdf, cdf, sf, logpdf (b < 0) | \|b\| log c small (tiny \|b\| or c near 1), any x | `1 - 1/c**b` cancels (pattern 2) | 5.4e7 ulp | severe |
| F5 | ppf, isf | tiny \|b\|, any q | cancelling `1 - 1/c**b` plus base rounding raised to the power `-1/b` (patterns 2, 6, 8) | 4.1e7 ulp | severe |
| F6 | logpdf (b > 0) | small b | `log(b) - log(-expm1(-b log c))` large terms cancel (pattern 5) | 13.1 ulp | minor |

### F1: b < 0 with large |b| log c: c**b underflows, every method fails

- **Code:** every method forms `1/c**b`: `_pdf` line 10572 `b * x**-(b+1) / (1 - 1/c**b)`,
  `_cdf` line 10583, `_sf` line 10598, `_ppf` line 10594, `_isf` line 10609; for b < 0,
  `_logpdf`/`_logcdf`/`_logsf` (lines 10576, 10587, 10602) go to `super()`, i.e. log of these.
- **Mechanism:** for b < 0, `c**b = c**-|b|` underflows to 0 once |b| log c > 745 (subnormal
  from 708). Then `1/c**b` is inf, so the normaliser `1 - inf` is -inf. pdf becomes
  `-inf/-inf` = nan, or 0 where `x**-(b+1)` is still finite. cdf and sf become nan
  (`inf/inf`, `inf - inf`) or 0. ppf gives `pow(1 + inf*q, -1/b)` = inf and isf gives
  `inf + (-inf)` = nan. The log methods take log of these. In pdf, `b * x**-(b+1)` also
  overflows a little earlier: `pdf(4.99, -440.5, 5)` returns inf (|b| log c = 709.0,
  `c**b` subnormal but `1/c**b` finite), while `pdf(3, -440, 5)` is fine (0.52 ulp).
  The true values are ordinary: the pdf at x = c is |b|/c = 200.
- **Regime:** b < 0 and |b| log c > ~709.8 (cdf/sf/ppf/isf and logs). pdf and logpdf fail from
  about |b| log c > 709 because of the `b * x**-(b+1)` product. Covers large |b| (c = 5, b = -1000)
  and huge c (b = -2, c = 1e200, where `c**-b` = 1e400).
- **Reproducer:** `python accuracy.py refs/truncpareto_ref.py --at pdf 5 -1000 5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(5, -1000, 5)` | nan | 200.0 | inf | 999 | loss |
  | `pdf(4.99, -440.5, 5)` | inf | 36.546753516811606 | inf | 440 | loss |
  | `logpdf(1, -1000, 5)` | -inf | -1602.5301571551183 | inf | 0.623 | loss |
  | `cdf(3, -1000, 5)` | nan | 1.4166102623834862e-222 | inf | 1e3 | loss |
  | `logcdf(1.000000000001, -1000, 5)` | -inf | -1630.1610893739164 | inf | 6.1e8 | loss |
  | `sf(4.9999999996, -1000, 5)` | nan | 8.000000342242924e-08 | inf | 1.25e10 | loss |
  | `ppf(0.5, -1000, 5)` | inf | 4.996535464952262 | inf | 0.001 | loss |
  | `isf(0.5, -1000, 5)` | nan | 4.996535464952262 | inf | 0.001 | loss |
  | `pdf(1e100, -2, 1e200)` | 0.0 | 2e-300 | inf | 1 | loss |
  | `logpdf(1, -2, 1e200)` | -inf | -920.3408900170583 | inf | 0.00109 | loss |
  | `cdf(1e100, -2, 1e200)` | 0.0 | 1.0000000000000001e-200 | inf | 2 | loss |
  | `logsf(1e100, -2, 1e200)` | -0.0 | -1.0000000000000001e-200 | inf | 2 | loss |
  | `ppf(0.5, -2, 1e200)` | inf | 7.071067811865475e+199 | inf | 0.5 | loss |

  Every point of (-1000, 5) and (-2, 1e200) for every method is a loss (17 to 56 points per method).
- **Fix direction:** for b < 0 write a = -b and normalise by c**a in log space:
  pdf = (a/x) exp(a (log x - log c)) / -expm1(-a log c),
  cdf = exp(a (log x - log c)) * expm1(a log x) / expm1(a log c)
  (with -expm1 forms), log x = log c + log(q + (1 - q) c**-a)/a for ppf. Give the b < 0
  log methods closed forms as well.

### F2: tiny |b|: endpoint differences round to zero in cdf, logcdf, sf, logsf

- **Code:** `_cdf` line 10583 `(1 - x**-b) / (1 - 1/c**b)`; `_logcdf_pos_b` line 10590
  `np.log1p(-x**-b) - np.log1p(-1/c**b)`; `_sf` line 10598 `(x**-b - 1/c**b) / (1 - 1/c**b)`;
  `_logsf_pos_b` line 10605 `np.log(x**-b - 1/c**b) - np.log1p(-1/c**b)`.
- **Mechanism:** `x**-b = exp(-b log x)`. When |b| log x < eps/2 (x close to 1), it rounds to 1.
  Then `1 - x**-b` = 0, so cdf = 0, logcdf = -inf and logsf = log(sf) = 0 (patterns 1, 2). At the
  other end, for x close to c, `x**-b` and `c**-b` round to the same double, so sf = 0,
  logsf = -inf and logcdf = 0. Before total loss, the relative error grows like eps/(|b|(x - 1)) or
  eps/(|b|(c - x)/c). For moderate |b| this is of the same order as the conditioning
  (x/(x - 1) or c/(c - x)) and therefore not flagged. For tiny |b| it exceeds the conditioning by
  1/|b|, and the result becomes exactly 0 while the true value is still far from eps.
- **Regime:** |b| (x - 1) < ~1e-16 or |b| (c - x)/c < ~1e-16; for b = ±1e-8 that is
  x - 1 < ~1e-8, c - x < ~5e-8. Both signs of b: for b < 0 the generic log(cdf), log(sf) fail the same way.
- **Reproducer:** `python accuracy.py refs/truncpareto_ref.py --at cdf 1.000000000001 1e-8 5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1.000000000001, 1e-8, 5)` | 0.0 | 6.213901765972567e-13 | inf | 1.0e12 | loss |
  | `logcdf(1.000000000001, 1e-8, 5)` | -inf | -28.10681720657805 | inf | 3.6e10 | loss |
  | `logsf(1.000000000001, 1e-8, 5)` | 0.0 | -6.213901765974498e-13 | inf | 1.0e12 | loss |
  | `sf(4.9999999996, 1e-8, 5)` | 0.0 | 4.970679847951583e-11 | inf | 1.2e10 | loss |
  | `logsf(4.9999999996, 1e-8, 5)` | -inf | -23.7248794018486 | inf | 5.3e8 | loss |
  | `logcdf(4.9999999996, 1e-8, 5)` | 0.0 | -4.970679848075121e-11 | inf | 1.2e10 | loss |

  (cond (arg) is large because x near 1 or c is itself sensitive to relative changes of x, but a
  result of exactly 0 or -inf is a loss irrespective of conditioning.)
- **Fix direction:** cdf numerator `-expm1(-b*log(x))`; sf numerator
  `c**-b * expm1(b*(log(c) - log(x)))`; logcdf/logsf as logs of these ratios with
  `log(-expm1(...))` instead of `log1p(-x**-b)`.

### F3: logsf for large b > 0: x**-b underflows

- **Code:** `_logsf_pos_b`, line 10605: `np.log(x**-b - 1/c**b) - np.log1p(-1/c**b)`.
- **Mechanism:** for b log x > 708, `x**-b` is subnormal (loses significant bits); for
  b log x > 745 it is 0 and `log(0 - 0)` = -inf (patterns 3, 4). The true logsf
  ≈ -b log x + log1p(-(x/c)**b) is an ordinary number.
- **Regime:** b > 0 and x > exp(708/b), for example x > 2.03 for b = 1000, c = 5.
- **Reproducer:** `python accuracy.py refs/truncpareto_ref.py --at logsf 2.2 1000 5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logsf(2.1064414349807272, 1000, 5)` | -744.4400719213812 | -745.0000000000001 | 3.4e12 | 1.34 | severe |
  | `logsf(2.2, 1000, 5)` | -inf | -788.4573603642702 | inf | 1.27 | loss |
  | `logsf(2.4000000000000004, 1000, 5)` | -inf | -875.4687373539001 | inf | 1.14 | loss |

- **Fix direction:** `-b*log(x) + log(-expm1(-b*(log(c) - log(x)))) - log1p(-c**-b)`.

### F4: normaliser 1 - 1/c**b cancels when |b| log c is small

- **Code:** `_pdf` line 10572, `_cdf` line 10583, `_sf` line 10598: denominator
  `1 - 1/c**b`; for b < 0 `_logpdf` (line 10576) is log of `_pdf`. (`_logpdf_pos_b` line 10579
  uses `-expm1(-b*log(c))` and is not affected.)
- **Mechanism:** `1/c**b = exp(-b log c)` is close to 1 when |b| log c is small. The 1/2-ulp
  rounding of `c**b` and of `1/c**b` becomes a relative error of about eps/(|b| log c) in the
  difference (pattern 2). It affects every x equally.
- **Regime:** all x, error ≈ 1/(|b| log c) ulp: about 1e7 to 5e7 ulp at |b| = 1e-8, c = 5, and at
  b = 2, c = 1 + 1e-8. Measured moderate already at b = 1e-5 (pdf 1.7e4 ulp) and at
  c = 1.0001 (pdf 1.6e3 ulp); minor at b = 1e-3 (pdf 176 ulp).
  For c near 1, the same error is present in cdf/sf but is within their conditioning
  (cond ≥ 1/(c - 1) because the whole support is [1, c]):
  `cdf(1.000000005, 2, 1.00000001)` 1.6e7 ulp, cond 2.0e8, ok.
- **Reproducer:** `python accuracy.py refs/truncpareto_ref.py --at pdf 1 1e-8 5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1, 1e-8, 5)` | 0.6213349416500269 | 0.6213349395596118 | 1.5e7 | 1 | severe |
  | `pdf(1, -1e-8, 5)` | 0.6213349330778372 | 0.6213349295596118 | 2.6e7 | 1 | severe |
  | `pdf(1, 2, 1.00000001)` | 100000002.82819323 | 100000002.10774711 | 3.2e7 | 3 | severe |
  | `pdf(3, 1e-5, 5)` | 0.20711103616377133 | 0.2071110361645481 | 1.7e4 | 1 | moderate |
  | `pdf(1.00005, 2, 1.0001)` | 9999.999950009716 | 9999.999950006099 | 1.6e3 | 3 | moderate |
  | `logpdf(1, -1e-8, 5)` | -0.47588499771193493 | -0.4758850033743002 | 5.4e7 | 2.1 | severe |
  | `cdf(4.999999949999999, 1e-8, 5)` | 1.0 | 0.9999999937866506 | 2.8e7 | 0.621 | severe |
  | `sf(1.00000001, 1e-8, 5)` | 0.9999999931017964 | 0.9999999937866507 | 3.1e6 | 0.621 | severe |

- **Fix direction:** denominator `-expm1(-b*log(c))` (already used in `_logpdf_pos_b`).

### F5: ppf and isf for tiny |b|

- **Code:** `_ppf` line 10594 `pow(1 - (1 - 1/c**b)*q, -1/b)`; `_isf` line 10609
  `pow(1/c**b + (1 - 1/c**b)*q, -1/b)`.
- **Mechanism:** two contributions of about 1/|b| ulp each. First, the cancelling normaliser
  `1 - 1/c**b` (F4) is inaccurate. Second, the base is close to 1 and its rounding error is
  multiplied by the exponent -1/b in `pow` (patterns 6, 8). Already `1/c**b` alone is off by
  ~eps relative, and isf at q → 0 is `pow(1/c**b, -1/b)` = c * (1 + O(eps/|b|)).
- **Regime:** all q when |b| is small; ~1e7 to 4e7 ulp at |b| = 1e-8, 141 ulp (minor) at b = 1e-3.
- **Reproducer:** `python accuracy.py refs/truncpareto_ref.py --at ppf 0.5 -1e-8 5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.01, 1e-8, 5)` | 1.0162245935641185 | 1.016224591137026 | 1.1e7 | 0.0161 | severe |
  | `ppf(0.5, -1e-8, 5)` | 2.2360679745509757 | 2.2360679847398717 | 2.1e7 | 0.805 | severe |
  | `isf(1e-300, 1e-8, 5)` | 4.9999999729260916 | 5.0 | 2.4e7 | 1.6e-300 | severe |
  | `ppf(0.5, 0.001, 5)` | 2.2353440866019896 | 2.2353440866019194 | 141 | 0.805 | minor |

  Across the run: ppf (1e-8, 5) 31 severe points, (-1e-8, 5) 29 severe (max 8.3e7 ulp);
  isf (1e-8, 5) 49 severe, (-1e-8, 5) 48 severe (max 6.9e7 ulp).
- **Fix direction:** `ppf = exp(-log1p(q*expm1(-b*log(c)))/b)`;
  `isf = exp(-log1p(-(1 - q)*(-expm1(-b*log(c))))/b)` written with `log1p` of the small
  quantity (or `log(c) - log1p(q*expm1(b*log(c)))/b` for isf).

### F6: logpdf (b > 0) loses a few digits for small b

- **Code:** `_logpdf_pos_b`, line 10579: `np.log(b) - np.log(-np.expm1(-b*np.log(c))) - (b+1)*np.log(x)`.
- **Mechanism:** for b = 1e-8 the first two terms are about -18.4 and -17.9. Their difference,
  -0.48, keeps their absolute rounding error, about 40 times its own ulp (pattern 5).
- **Regime:** small b > 0; loss ≈ |log b| / |logpdf| ulp, at most tens of ulp.
- **Reproducer:** `python accuracy.py refs/truncpareto_ref.py --at logpdf 1 1e-8 5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1, 1e-8, 5)` | -0.47588498727992246 | -0.47588498727992107 | 13.1 | 2.1 | minor |

- **Fix direction:** `np.log(b / -np.expm1(-b*np.log(c)))` (the ratio is accurate) before subtracting `(b+1)*log(x)`.

## RuntimeWarnings

- Spurious warnings on accurate results: for b = 1000, c = 5, `c**b` overflows to inf
  ("overflow encountered in power"), and `1/c**b` = 0 is then exactly what is needed. This appears in
  pdf, cdf, sf, logcdf, ppf, isf. Similarly "underflow encountered in power"/"in expm1" (logpdf
  b = 1000) and "underflow encountered in multiply" in ppf/isf (`D*q` for q = 1e-300) on accurate
  results.
- Warnings coinciding with wrong results: "divide by zero", "invalid value in divide/subtract/add",
  "overflow encountered in multiply" for b < 0 with large |b| log c (F1); "divide by zero encountered
  in log1p" (logcdf) and "in log" (logsf) for F2 and F3.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, (1.8, 5.3), (2, 5), (-2, 5), (1000, 5), (-400, 5); x where `x**-(b+1)` is subnormal for b = 1000 | 2.35 ulp; `pdf(2.0329, 1000, 5)` 1.51 ulp |
| logpdf | same sets, b > 0 closed form and b < 0 generic | 2.67 ulp |
| cdf, sf, logcdf, logsf | same sets, both endpoints to 1e-15 relative | within conditioning (cond up to 1e15 near x = 1 and x = c) |
| logcdf | b = 1000, x up to c | within conditioning |
| ppf | same sets, q in [1e-300, 1 - 3e-16]; q -> 1 with large b log c (b = 1000) | 6.04 ulp |
| isf | same sets | 4 ulp; (-400, 5) within conditioning |
| ppf, isf | c = 1 + 1e-8 (b = 2) | 1 ulp |

## Suspected, not confirmed

- pdf for b > 0 large: `b * x**-(b+1)` with `x**-(b+1)` subnormal (pattern 4) can lose up to
  log2(b) bits only where the result is within a factor b of the subnormal range. Tested
  `pdf(2.0329, 1000, 5)`: 1.51 ulp. The affected window is too narrow to show for b ≤ 1000.
- For b < 0, near the F1 threshold (708 < |b| log c < 745), `c**b` is subnormal and
  `1/c**b` loses bits. `cdf(4.99, -440.5, 5)` was still 0.6 ulp. Larger |b| fails completely (F1)
  first, so a partial-precision regime was not observed.

## Opportunities: inherited methods

All eight methods are implemented. For b < 0, however, `_logpdf`, `_logcdf` and `_logsf`
dispatch to the generic `rv_continuous` log(pdf), log(cdf), log(sf). Closed forms with
a = -b > 0:
logpdf = log(a) - log(-expm1(-a log c)) + (a - 1) log x - a log c,
logcdf = log(expm1(a log x)) - log(expm1(a log c)) (as `a log x + log1mexp(-a log x) - ...`),
logsf = log(-expm1(a (log x - log c))) + a log c - log(expm1(a log c)) (equivalently with
`log1mexp`). The generic forms fail wherever F1 or F2 apply: `logpdf(1, -1000, 5)` is -inf for
-1602.53, and `logcdf(1.000000000001, -1e-8, 5)` is -inf.

## Shared code

none

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(5, -1000, 5)` | nan | 200.0 | verified | All 13 table points reproduced (nan/inf/0/-inf vs finite refs; e.g. `cdf(3, -1000, 5)` ref 1.4166102623834862e-222, `isf(0.5, -1000, 5)` ref 4.9965354649522626, `logsf(1e100, -2, 1e200)` ref -1.0e-200). Mechanism checked: `pdf(4.99, -440.5, 5)` overflows in `b * x**-(b+1)` (log ≈ 712.6) with `1/c**b` ≈ 8e307 finite, as stated |
| F2 | `cdf(1.000000000001, 1e-8, 5)` | 0.0 | 6.2139017659725677e-13 | verified | All 6 table points plus `logcdf(1.000000000001, -1e-8, 5)` = -inf (ref -28.10681722267243) reproduced |
| F3 | `logsf(2.2, 1000, 5)` | -inf | -788.45736036427025 | verified | `logsf(2.1064414349807272, 1000, 5)` = -744.44 (log of the smallest subnormal) vs -745.00000000000007, 3.4e12 ulp; `logsf(2.4000000000000004, 1000, 5)` -inf vs -875.46873735390008 |
| F4 | `logpdf(1, -1e-8, 5)` | -0.47588499771193493 | -0.47588500337430019 | verified | 5.4e7 ulp. All 8 table points match the report's ulp values (pdf 1.5e7/2.6e7/3.2e7, cdf 2.8e7, sf 3.1e6, moderate pdf 1.7e4 and 1.6e3) |
| F5 | `ppf(0.5, -1e-8, 5)` | 2.2360679745509757 | 2.2360679847398715 | verified | 2.1e7 ulp; `ppf(0.01, 1e-8, 5)` 1.1e7 ulp, `isf(1e-300, 1e-8, 5)` 2.4e7 ulp, `ppf(0.5, 0.001, 5)` 141 ulp. Quantile refs checked by cdf(x_ref) - q < 1e-93 |

Reference route: pdf from the docstring formula; cdf = expm1(-b log x)/expm1(-b log c), sf = c**-b expm1(b(log c - log x))/(-expm1(-b log c)) (logsf as log1p(-cdf) where cdf < sf), checked by d/dx cdf = pdf and cdf + sf = 1; quantiles in closed form checked by cdf(x_ref) = q; 100 vs 200 digits agree to 30 digits at every point. (A first attempt with `mp.quad` of the pdf was unreliable for |b| ≈ 1000 and was discarded.) Mechanisms and line numbers 10572-10609 match the source. Scripts: `scratch/truncpareto/verify/verify.py`, output in `scratch/truncpareto/verify/output.txt`.

Verified by dist-accuracy-verifier: 5 of 5 verified.
