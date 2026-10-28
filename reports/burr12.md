# burr12: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `burr12_gen`, lines 1270-1354 (methods 1320-1346)
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf (all in `burr12_gen`; pdf = exp(logpdf), cdf = -expm1(logsf), sf = exp(logsf))
- **Inherited (not measured):** none
- **Related:** none sharing code (fisk is documented as a special case with d = 1 but `fisk_gen` forwards to `burr_gen`, not to burr12); helpers: none; special functions: sc.xlogy, sc.xlog1py, sc.log1p, sc.expm1 (sc.beta only in `_munp`)
- **Reference:** `refs/burr12_ref.py`, 400 digits; self-checks: PASS (the NOTE/WARN entries for (100, 0.001) are SciPy losses, confirmed with `--at`, see F1)
- **Parameter sets:** (10, 4), (2, 1), (100, 0.001), (3, 1e6), (0.001, 2), (0.5, 0.001), (1e4, 1e-6), (10, 1e12)
- **Worst finding:** loss in logpdf, pdf, logsf, sf, cdf, logcdf, ppf, isf

## Summary

burr12 is written mostly in log space (`xlog1py`, `expm1`, `log1p`), so cdf,
sf, pdf and the quantiles are accurate in the bulk for moderate shapes and
the left tail of cdf/sf/logsf/logpdf is fine. The one systematic gap is that
`x**c` is formed explicitly everywhere: once `c*log(x) > 709.8` it overflows,
and logpdf and logsf return `-inf` even for the inventory example (10, 4)
(x > 6.7e30). For small `c*d`, pdf, sf, cdf and logcdf also return 0, 1 or
-0.0 in the bulk (for example `cdf(2, 1e4, 1e-6) = 1.0` where the true value is 0.0069).
logcdf is the weakest method. It computes `log1p(-(1 + x**c)**(-d))`, which
returns `-inf` once `d*x**c < eps` (left tail, every parameter set) and loses
up to 1.5e12 ulp in the bulk for large d (pattern 6). ppf and isf overflow in
`expm1` when `-log(1-q)/d` or `-log(q)/d` exceeds 709.8 (small d). Compared
with burr (Burr III, also used by fisk), burr12 does better: there is no
`1 - cdf` cancellation in sf/logsf, no `inf/inf` in pdf and no c*d = 1
cancellation in logpdf. It shares burr's power overflow, only in the
opposite tail (`x**c` for large x instead of `x**-c` for small x), and it
has an extra logcdf weakness that burr does not have.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf, logsf, pdf, sf, cdf, logcdf | right tail, c*log(x) > 709.8 (any shapes; pdf/sf/cdf/logcdf only when c*d is small) | `x**c` overflows (pattern 3) | logpdf -inf for -3772.6; cdf 1.0 for 0.0069 | loss |
| F2 | logcdf | left tail, d*x**c < ~eps; partial loss for d*x**c < ~1e-3 | `log1p(-(1 + x**c)**(-d))`: 1 + x**c rounds to 1 (patterns 1, 2) | -inf for -228.87; 4.7e10 ulp | loss |
| F3 | ppf, isf | -log1p(-q)/d or -log(q)/d > 709.8 (small d; d = 1 only for subnormal q) | `expm1(z)` overflows before `**(1/c)` (pattern 8) | inf for 2.732 | loss |
| F4 | logcdf | bulk and right tail, large d (1e6, 1e12) | rounding of `1 + x**c` raised to the power -d (pattern 6) | 1.5e12 ulp | severe |
| F5 | cdf, logsf | left tail, `x**c` subnormal while d*x**c is normal (large d) | subnormal intermediate (pattern 4) | 1.5e9 ulp | severe |
| F6 | ppf | tiny q with huge d, or subnormal q | subnormal `-log1p(-q)/d`; `sc.expm1` off by one subnormal unit (pattern 4) | 5.5e4 ulp | moderate |
| F7 | pdf, sf, logpdf, ppf, isf | large \|logpdf\|, \|logsf\| or large z; large c with small d (logpdf) | `exp` of a log-space result amplifies its absolute rounding error; partial cancellation of log terms (patterns 5, 6) | 636 ulp | minor |

### F1: x**c overflows in the right tail

- **Code:** `_logpdf`, line 1325: `... + sc.xlog1py(-d-1, x**c)`; `_logsf`, line 1337: `sc.xlog1py(-d, x**c)`; `_logcdf`, line 1331: `sc.log1p(-(1 + x**c)**(-d))`. `_pdf` (1322), `_cdf` (1328) and `_sf` (1334) are built on `_logpdf`/`_logsf`.
- **Mechanism:** for c*log(x) > 709.78, `x**c` is inf. Then `xlog1py(-d, inf) = -inf`, so logsf and logpdf are `-inf`, sf and pdf are `exp(-inf) = 0`, cdf is `-expm1(-inf) = 1.0`, and logcdf is `log1p(-inf**(-d)) = log1p(-0) = -0.0`. The true values are finite: logsf = -d*log1p(x**c) is about -c*d*log(x), and sf is about x**(-c*d), which stays a normal double for small c*d (pattern 3).
- **Regime:** x > exp(709.78/c), e.g. x > 6.7e30 for c = 10, x > 1.35e154 for c = 2, x > 1209 for c = 100, x > 1.074 for c = 1e4. logpdf and logsf fail there for all shapes, including the inventory example. pdf, sf, cdf and logcdf fail where their true value is still normal and away from 0/1. That needs small c*d, e.g. (100, 0.001) and (1e4, 1e-6), where the whole bulk past the threshold is lost (55 bulk points per method).
- **Reproducer:** `python accuracy.py refs/burr12_ref.py --at logpdf 1e40 10 4`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e40, 10, 4)` | -inf | -3772.550673056121 | inf | 0.0109 | loss |
  | `logsf(1e40, 10, 4)` | -inf | -3684.1361487904733 | inf | 0.0109 | loss |
  | `logpdf(1e155, 2, 1)` | -inf | -1070.0089210616713 | inf | 0.0028 | loss |
  | `logpdf(1.5, 10000, 1e-6)` | -inf | -5.014689945177337 | inf | 0.201 | loss |
  | `pdf(1e10, 100, 0.001)` | 0.0 | 1e-12 | inf | 1.1 | loss |
  | `sf(1e10, 100, 0.001)` | 0.0 | 0.09999999999999999 | inf | 0.1 | loss |
  | `cdf(2, 10000, 1e-6)` | 1.0 | 0.006907504562964098 | 6.5e+17 | 1.44 | loss |
  | `logcdf(1e10, 100, 0.001)` | -0.0 | -0.1053605156578263 | inf | 0.105 | loss |

- **Fix direction:** for x > 1 use `log1p(x**c) = c*log(x) + log1p(x**-c)`, so that logsf = `-d*(c*log(x) + log1p(x**-c))` and logpdf = `log(c*d) - (c*d + 1)*log(x) - (d + 1)*log1p(x**-c)`.

### F2: logcdf collapses in the left tail

- **Code:** `_logcdf`, line 1331: `sc.log1p(-(1 + x**c)**(-d))`.
- **Mechanism:** the code first forms sf = `(1 + x**c)**(-d)` and then `log1p(-sf)`. For small x the cdf is about d*x**c, and sf is 1 minus that. Once `1 + x**c` rounds to 1 (x**c < eps/2), or `(1 + x**c)**(-d)` rounds to 1 (d*x**c < ~eps/2), the result is `log1p(-1) = -inf`. Before that point only the digits of the cdf beyond the leading 1s of sf survive (patterns 1 and 2). The relative error of the cdf is about eps/cdf, and logcdf divides it by |logcdf|.
- **Regime:** left tail of every parameter set: total loss for d*x**c < ~1.1e-16 (282 grid points), severe for d*x**c up to ~1e-8, moderate up to ~1e-4. This includes the bulk where the cdf is small, e.g. (1e4, 1e-6) at x = 1, cdf 6.9e-7: 2.0e4 ulp.
- **Reproducer:** `python accuracy.py refs/burr12_ref.py --at logcdf 1e-10 10 4`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logcdf(1e-10, 10, 4)` | -inf | -228.87221493828469 | inf | 0.0437 | loss |
  | `logcdf(1e-20, 0.5, 0.001)` | -29.933295312068765 | -29.933606208972645 | 4.7e+10 | 0.0167 | severe |
  | `logcdf(1e-5, 2, 1)` | -23.02585084720009 | -23.025850930040455 | 1.6e+07 | 0.0869 | severe |
  | `logcdf(1.0, 10000, 1e-6)` | -14.182023825183233 | -14.182023825119508 | 2.0e+04 | 509 | moderate |

- **Fix direction:** `logcdf = log(-expm1(logsf))` with `logsf = -d*log1p(x**c)` (i.e. `log1mexp(logsf)`). This also fixes F4.

### F3: expm1 overflows in ppf and isf for small d

- **Code:** `_ppf`, line 1343: `sc.expm1(-1/d * sc.log1p(-q))**(1/c)`; `_isf`, line 1346: `sc.expm1(-1/d * np.log(p))**(1/c)`.
- **Mechanism:** with z = -log1p(-q)/d (ppf) or z = -log(p)/d (isf), the quantile is expm1(z)**(1/c), which is about exp(z/c) for large z. `expm1(z)` overflows for z > 709.78, but after the 1/c root the result can be moderate (pattern 8, pattern 3).
- **Regime:** z > 709.78: q > 1 - exp(-709.78*d) for ppf, p < exp(-709.78*d) for isf. For d = 1e-3 this is q > 0.508 / p < 0.492; for d = 1e-6 nearly all q (ppf(0.01) is lost); for d = 1 only subnormal p in isf.
- **Reproducer:** `python accuracy.py refs/burr12_ref.py --at ppf 0.01 10000 1e-6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.01, 10000, 1e-6)` | inf | 2.731999026429026 | inf | 1.01 | loss |
  | `ppf(0.9, 100, 0.001)` | inf | 10000000000.000017 | inf | 90 | loss |
  | `isf(0.1, 100, 0.001)` | inf | 9999999999.99999 | inf | 10 | loss |
  | `isf(2.0322308024e-313, 2, 1)` | inf | 2.2182652975417935e+156 | inf | 0.5 | loss |

- **Fix direction:** for large z use `exp((z + log1p(-exp(-z)))/c)` (log of expm1 in closed form).

### F4: logcdf amplifies rounding by d for large d

- **Code:** `_logcdf`, line 1331: `(1 + x**c)**(-d)`.
- **Mechanism:** `1 + x**c` carries a rounding error of up to eps/2 relative. Raising it to the power -d multiplies that error by d (pattern 6), so sf = `(1 + x**c)**(-d)` has a relative error of up to about d*eps/2, capped by the range where sf is not 0. `log1p(-sf)` then passes this on to logcdf. In the right tail logcdf is about -sf, so the error appears directly; in the bulk the cancellation in 1 - sf amplifies it further. logsf/sf/cdf are unaffected because they use `xlog1py`.
- **Regime:** large d, bulk and right tail (d*x**c >~ 1e-3): (3, 1e6) up to 1.1e6 ulp, (10, 1e12) up to 1.5e12 ulp. The error grows with d.
- **Reproducer:** `python accuracy.py refs/burr12_ref.py --at logcdf 0.05011872336272723 10 1e12`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logcdf(0.05011872336272723, 10, 1e12)` | -2.3529287602702946 | -2.3521684610441387 | 1.5e+12 | 4.04 | severe |
  | `logcdf(0.004641588833612776, 3, 1e6)` | -2.3521685080305885 | -2.35216850858575 | 1.1e+06 | 1.21 | severe |
  | `logcdf(0.1, 10, 1e12)` | -3.7200452147017624e-44 | -3.7200759946204933e-44 | 3.7e+10 | 1000 | severe |

- **Fix direction:** same as F2, `log(-expm1(-d*log1p(x**c)))`.

### F5: subnormal x**c in cdf and logsf for large d

- **Code:** `_logsf`, line 1337: `sc.xlog1py(-d, x**c)`; `_cdf`, line 1328: `-sc.expm1(self._logsf(x, c, d))`.
- **Mechanism:** in the left tail, logsf is about -d*x**c and cdf about d*x**c. When `x**c` is subnormal (below 2.2e-308, fewer than 53 significant bits) but d is large enough that d*x**c is normal again, the lost bits of `x**c` show up as a large relative error in the result (pattern 4).
- **Regime:** c*log(x) < -708.4 and d*x**c > 2.2e-308. This needs large d: (3, 1e6) gives moderate errors, (10, 1e12) severe ones. 12 cdf points and 12 logsf points are flagged.
- **Reproducer:** `python accuracy.py refs/burr12_ref.py --at cdf 1.839921625376204e-32 10 1e12`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1.839921625376204e-32, 10, 1e12)` | 4.44624496661913e-306 | 4.446243454351469e-306 | 1.5e+09 | 10 | severe |
  | `logsf(1.839921625376204e-32, 10, 1e12)` | -4.44624496661913e-306 | -4.446243454351469e-306 | 1.5e+09 | 10 | severe |
  | `cdf(4.6248075735263605e-105, 3, 1e6)` | 9.891929276894487e-308 | 9.891929277114256e-308 | 1.1e+05 | 3 | moderate |

- **Fix direction:** when `x**c` is tiny, use `d*x**c = exp(log(d) + c*log(x))` (e.g. logsf = `-exp(log(d) + c*log(x))` for x**c < eps).

### F6: subnormal intermediate in ppf for tiny q

- **Code:** `_ppf`, line 1343: `sc.expm1(-1/d * sc.log1p(-q))**(1/c)`.
- **Mechanism:** for tiny q, z = -log1p(-q)/d is about q/d. If it falls below 2.2e-308 it is subnormal and has lost bits (pattern 4). The root `**(1/c)` then brings the result back to a normal double, where the lost bits show (divided by c). Separately, `sc.expm1` is not correctly rounded for subnormal and near-underflow arguments: it returns x minus one unit of the double grid for about a quarter of the inputs below 4.4e-308 (see Shared code). With a subnormal q this costs 5.5e4 ulp after the square root.
- **Regime:** q/d < 2.2e-308: q = 1e-300 with d = 1e12, or subnormal q with any d.
- **Reproducer:** `python accuracy.py refs/burr12_ref.py --at ppf 1e-300 10 1e12`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 10, 1e12)` | 6.309573444804056e-32 | 6.309573444801932e-32 | 1.52e+03 | 0.1 | moderate |
  | `ppf(2.0322308024e-313, 2, 1)` | 4.5080270655453625e-157 | 4.508027065600161e-157 | 5.47e+04 | 0.5 | moderate |

- **Fix direction:** for small z evaluate in log space: `exp((log(-log1p(-q)) - log(d) + log1p(z/2 + ...))/c)`, or `exp(log(expm1(z))/c)` with `log(z)` computed from `log(q) - log(d)`.

### F7: exp of log-space results and partial log cancellation (minor)

- **Code:** `_pdf`, line 1322: `np.exp(self._logpdf(x, c, d))`; `_sf`, line 1334: `np.exp(self._logsf(x, c, d))`; `_logpdf`, line 1325; `_ppf`/`_isf`, lines 1343/1346.
- **Mechanism:** logpdf and logsf have an absolute rounding error of about |value|*eps. `exp` turns this into a relative error of |value| ulp, while the condition number of pdf/sf with respect to x is only about c*d + 1 (pdf at x = 1e290, c = 0.5, logpdf = -675: 636 ulp). In ppf/isf the rounding of a large z is amplified the same way by `expm1(z)` (up to 87 ulp in isf). In logpdf, `xlogy(c - 1, x)` and `xlog1py(-d - 1, x**c)` partly cancel when c is large and d small, because the terms are about c*log(x) and the result about (c*d + 1)*log(x) (pattern 5, up to 121 ulp).
- **Regime:** pdf/sf with |logpdf| or |logsf| in the hundreds; logpdf with c >> 1 and c*d << c.
- **Reproducer:** `python accuracy.py refs/burr12_ref.py --at pdf 1e290 0.5 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e290, 0.5, 0.001)` | 3.5807170510650155e-294 | 3.58071705106451e-294 | 636 | 1 | minor |
  | `pdf(1e-280, 2, 1)` | 1.9999999999997827e-280 | 2e-280 | 489 | 1 | minor |
  | `sf(1e140, 2, 1)` | 9.999999999999464e-281 | 9.999999999999998e-281 | 241 | 2 | minor |
  | `logpdf(1.0734835558915026, 10000, 1e-6)` | -4.676788296187965 | -4.676788296188091 | 121 | 0.216 | minor |
  | `logpdf(9.5, 100, 0.001)` | -4.77900607146114 | -4.779006071461191 | 47.7 | 0.23 | minor |

- **Fix direction:** intrinsic to the exp(log) construction. Direct power forms such as `c*d*x**(c-1)*exp(-(d+1)*log1p(x**c))` would reduce it in the bulk. For logpdf with x > 1, combine the terms into `-(c*d + 1)*log(x) - (d + 1)*log1p(x**-c)` (same rewrite as F1).

## RuntimeWarnings

- `overflow encountered in power` from `x**c` in every x-method in the F1 regime. The results there are wrong (loss), so these warnings are not spurious. The same warning appears in ppf/isf in the F3 regime.
- `underflow encountered in power` from `x**c` in the left tail. In most cases the result is still accurate (cdf, logsf and logpdf are accurate when d*x**c is subnormal or negligible), so these warnings are spurious. Where d is large they coincide with F5.
- `underflow encountered in exp` in pdf/sf where the true result is subnormal (not a finding).
- `underflow encountered in multiply` in ppf (F6, `-1/d * log1p(-q)` subnormal).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid for (10, 4), (2, 1), (0.001, 2), (0.5, 0.001); left tail of all sets outside F5 | 1.6 ulp |
| sf | left tail and bulk for all sets; right tail outside F1 | 1.6 ulp in the left tail; up to 241 ulp in the far right tail (F7) |
| logsf | left tail (outside F5) and bulk outside F1 | 1.1 ulp |
| logpdf | left tail, all sets (x**c underflow is harmless here); bulk for moderate shapes | 1.5 ulp outside F1/F7 |
| ppf | bulk and q -> 1 for all sets except small d (F3); small shape c = 0.001 | 17 ulp for (10, 4) at q ~ 1e-288 (1/c = 0.1 is rounded and amplified by log q, minor); 1.0e3 ulp for (0.001, 2) but within conditioning (cond 2.2e3) |
| isf | all sets except small d (F3) | 87 ulp for (2, 1) at p ~ 1e-251 (F7 mechanism, minor); within conditioning for c = 0.001 |
| logcdf | right tail for d <= 4 outside F1 | 0 ulp |

## Suspected, not confirmed

- The logpdf cancellation (F7) should grow to moderate for c/(c*d + 1) >~ 1e4, e.g. c = 1e6, d = 1e-9. This was not measured: with such shapes the F1 overflow threshold is x > 1.0007, so the regime is almost entirely covered by the loss in F1.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| none | - | - | all eight methods are implemented |

## Shared code

- No burr12-specific helpers; no other distribution calls `burr12_gen`.
- `scipy.special.expm1` is not correctly rounded for tiny arguments. For x below about 4.4e-308 it returns the double one unit below x for about 27% of the inputs (5436 of 20001 log-spaced points in [1e-323, 1e-295]). The relative error is at most 1 ulp for normal arguments but large for subnormal ones (the F6 point `ppf(2.03e-313, 2, 1)`). `np.expm1` returns x exactly at all those points. This affects every distribution that passes tiny arguments to `sc.expm1`; it only matters for subnormal outputs that are later scaled back up.
- Comparison with burr (Burr III; fisk forwards to it; see `reports/fisk.md` and the bundled burr review): burr12 does better. burr's sf/logsf cancel in `1 - cdf` in the right tail, its pdf returns nan from `inf/inf`, and its logpdf loses everything for c*d = 1. burr12 avoids all three by using `xlog1py` for logsf/logpdf. Both share the overflow of a power of x (burr: `x**-c` in the left tail; burr12: `x**c` in the right tail, F1), the expm1/power overflow in the quantiles (burr F5/F6, burr12 F3) and the large-d power amplification (burr F8, burr12 F4, here only in logcdf). burr12 adds a logcdf left-tail collapse (F2) that burr does not have.

## Existing GitHub issues

none found. Excluded: #7491 (closed, PR #10381): pdf at the left endpoint x = 0 for fisk/burr/burr12/f - endpoint value, not the regimes here; #18838 (closed): invalid moments (not a measured method); #17832 (open): tracking issue for custom methods (generic). (Searched via the GitHub search API for "burr12"; `gh` was not available.)

## Verification

Verifier reference (`scratch/burr12/verify/verify.py`): cdf/logcdf via `mp.betainc(1, d, 0, u/(1+u))`, sf/logsf via the complementary `mp.betainc(d, 1, 0, 1/(1+u))` (u = x**c), logpdf/pdf from the docstring pdf, ppf/isf from the closed form checked by forward betainc evaluation; 100 and 200 digits plus guard digits for |log10 u| and 1 - q; the two precisions agree to > 30 digits at every point. One F4 point also cross-checked with `mp.quad` of the pdf.

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `logpdf(1e40, 10, 4)` | -inf | -3772.550673056121 | verified | mechanism matches line 1325 (`x**c` = inf) |
| F1 | `logsf(1e40, 10, 4)` | -inf | -3684.1361487904733 | verified | line 1337 |
| F1 | `logpdf(1e155, 2, 1)` | -inf | -1070.0089210616713 | verified | |
| F1 | `logpdf(1.5, 10000, 1e-6)` | -inf | -5.014689945177337 | verified | |
| F1 | `pdf(1e10, 100, 0.001)` | 0.0 | 1e-12 | verified | |
| F1 | `sf(1e10, 100, 0.001)` | 0.0 | 0.09999999999999999 | verified | |
| F1 | `cdf(2, 10000, 1e-6)` | 1.0 | 0.006907504562964098 | verified | 6.5e17 ulp |
| F1 | `logcdf(1e10, 100, 0.001)` | -0.0 | -0.1053605156578263 | verified | line 1331 |
| F2 | `logcdf(1e-10, 10, 4)` | -inf | -228.87221493828469 | verified | mechanism matches line 1331 (`log1p(-sf)`) |
| F2 | `logcdf(1e-20, 0.5, 0.001)` | -29.933295312068765 | -29.933606208972645 | verified | 4.7e10 ulp |
| F2 | `logcdf(1e-5, 2, 1)` | -23.02585084720009 | -23.025850930040455 | verified | 1.6e7 ulp |
| F2 | `logcdf(1.0, 10000, 1e-6)` | -14.182023825183233 | -14.182023825119508 | verified | 2.0e4 ulp (moderate row) |
| F3 | `ppf(0.01, 10000, 1e-6)` | inf | 2.731999026429026 | verified | mechanism matches line 1343 (`expm1` overflow) |
| F3 | `ppf(0.9, 100, 0.001)` | inf | 10000000000.000017 | verified | |
| F3 | `isf(0.1, 100, 0.001)` | inf | 9999999999.99999 | verified | line 1346 |
| F3 | `isf(2.0322308024e-313, 2, 1)` | inf | 2.2182652975417935e+156 | verified | |
| F4 | `logcdf(0.05011872336272723, 10, 1e12)` | -2.3529287602702946 | -2.3521684610441387 | verified | 1.5e12 ulp; mechanism matches line 1331 |
| F4 | `logcdf(0.004641588833612776, 3, 1e6)` | -2.3521685080305885 | -2.35216850858575 | verified | 1.1e6 ulp |
| F4 | `logcdf(0.1, 10, 1e12)` | -3.7200452147017624e-44 | -3.7200759946210096e-44 | verified | 3.7e10 ulp. The report's reference -3.7200759946204933e-44 is a transcription slip: `refs/burr12_ref.py` (via `--at`), `mp.quad` and betainc all give -3.7200759946210096e-44. The ulp figure is unaffected |
| F5 | `cdf(1.839921625376204e-32, 10, 1e12)` | 4.44624496661913e-306 | 4.446243454351469e-306 | verified | 1.5e9 ulp; mechanism matches lines 1328/1337 |
| F5 | `logsf(1.839921625376204e-32, 10, 1e12)` | -4.44624496661913e-306 | -4.446243454351469e-306 | verified | 1.5e9 ulp |
| F5 | `cdf(4.6248075735263605e-105, 3, 1e6)` | 9.891929276894487e-308 | 9.891929277114256e-308 | verified | 1.0e5 ulp (moderate row) |
| F6 | `ppf(1e-300, 10, 1e12)` | 6.309573444804056e-32 | 6.309573444801932e-32 | verified | 1.5e3 ulp; line 1343 |
| F6 | `ppf(2.0322308024e-313, 2, 1)` | 4.5080270655453625e-157 | 4.508027065600161e-157 | verified | 5.5e4 ulp |

Verified by dist-accuracy-verifier: 6 of 6 verified.
