# fisk: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `fisk_gen`, lines 1360-1439
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf — every one is a
  one-line delegation to `burr_gen` with `d = 1.0` (`burr._pdf(x, c, 1.0)` etc.,
  lines 1406-1430), so all measured code lives in `burr_gen`, lines 1199-1235
- **Inherited (not measured):** none (all eight functions are implemented;
  `_stats`, `_munp` also delegate to burr, `_entropy = 2 - log(c)`)
- **Related:** burr (same code with d = 1), burr12, mielke; helpers: none;
  special functions: `sc.log1p`, `sc.xlogy`, `sc.xlog1py`, `sc.expm1`
- **Reference:** `refs/fisk_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** c = 3.085754862225318 (inventory example), 1.0, 1.0000000001,
  1.2, 2.0, 0.5, 1e-3, 100.0, 1e6
- **Worst finding:** loss in pdf, logpdf, logcdf, sf, logsf

## Summary

`cdf` is excellent everywhere (max 1.0 ulp over the whole grid) and `ppf`/`isf`
are good (max 64 ulp beyond conditioning). The other five methods each fail
completely in one tail: `sf`/`logsf` return `0.0`/`-inf` as soon as `x**-c` falls
below 2^-53, because they are built on `log1p(-cdf)` with `cdf -> 1` (419 and 532
grid points lost, with severe errors up to 5.0e14 ulp just before the cliff);
`pdf` returns `nan` or `inf` and `logpdf`/`logcdf` return `-inf` in the left tail,
where `x**(-c-1)` resp. `x**(-c)` overflows although the true result is a normal
double (144, 178 and 133 points lost). `logpdf` additionally loses all digits for
c close to 1 — the fisk instance of burr's `c*d = 1` cancellation:
`logpdf(1e-300, 1.0)` returns `0.0` where the true value is -2e-300. Two minor
mechanisms (rounded exponents `-c-1` and `-1/c` amplified by large logarithms)
cost up to 484 ulp in `pdf` and 64 ulp in `ppf`/`isf`. Since `fisk_gen` only
forwards to `burr_gen`, every loss-level finding is the `d = 1` instance of a
finding in the bundled burr review.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | sf, logsf | right tail, `x**-c < 2**-53`, i.e. x > exp(36.7/c); any c | `log1p(-cdf)` with cdf rounded to 1 (pattern 1) | sf 0.0, logsf -inf; 5.0e14 ulp before the cliff | loss |
| F2 | pdf | left tail, x < exp(-709.8/(c+1)) | `x**(-c-1)` (and `(1+x**-c)**2`) overflow -> inf or inf/inf (pattern 3) | nan for 2e-300 | loss |
| F3 | logpdf, logcdf | left tail, x < exp(-709.8/c) | `x**(-c)` overflows inside `xlog1py`/`log1p` (pattern 3) | -inf for -1439.66 | loss |
| F4 | logpdf | \|c-1\| <~ 1e-4, x far from 1 | the two large log terms cancel (pattern 5) | 0.0 for -2e-300; 8.0e10 ulp at c = 1 | loss |
| F5 | pdf | any c, \|log x\| >~ 100; worst relative damage for c << 1 | the exponent `-c-1.0` is a rounded sum, error amplified by \|log x\| (patterns 11/6) | 484 ulp | minor |
| F6 | ppf, isf | q <~ 1e-10, c with inexact 1/c | the exponent `-1.0/c` is rounded, amplified by log(1/q) (patterns 11/6) | 64 ulp | minor |

### F1: sf and logsf cancel in 1 - cdf and collapse in the right tail

- **Code:** `burr_gen._logsf`, line 1228: `np.log1p(- (1 + x**(-c))**(-d))`;
  `burr_gen._sf`, lines 1224-1225: `np.exp(self._logsf(x, c, d))`.
- **Mechanism:** with d = 1, cdf = 1/(1 + u) and sf = u/(1 + u), u = x**-c. The
  code forms the cdf first; it rounds to exactly 1.0 once u < 2^-53, and then
  `log1p(-1.0) = -inf`, `sf = exp(-inf) = 0.0`. Before that cliff only the digits
  of u beyond the leading 1 of the cdf survive, so the relative error grows like
  eps/sf (pattern 1): 5.0e14 ulp at u = 1e-15, 4.0e11 ulp at u = 1e-12,
  3.4e2 ulp at u = 1.1e-3.
- **Regime:** total loss for x > exp(36.7/c) — x > 9.3e7 for c = 2, x > 1.45 for
  c = 100, x > 1.0000367 for c = 1e6; severe (>1e6 ulp) already where
  x**-c <~ 1e-10. 419 of 427 right-tail `sf` points and 532 of 540 `logsf`
  points are lost. For c <= 1e-3 the threshold lies outside the x grid.
- **Reproducer:** `python accuracy.py refs/fisk_ref.py --at sf 1e10 3.085754862225318`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `sf(1e10, 3.085754862225318)` | 0.0 | 1.3881978829312955e-31 | inf | 3.09 | loss |
  | `logsf(1e10, 3.085754862225318)` | -inf | -71.05213146393912 | inf | 0.0434 | loss |
  | `sf(1.5, 100.0)` | 0.0 | 2.4596544265798292e-18 | inf | 100 | loss |
  | `logsf(1.5, 100.0)` | -inf | -40.546510810816436 | inf | 2.47 | loss |
  | `sf(1e20, 1.0)` | 0.0 | 1e-20 | inf | 1 | loss |
  | `sf(1e30, 0.5)` | 1.1102230246251546e-15 | 9.999999999999989e-16 | 5.0e14 | 0.5 | severe |
  | `logsf(1e30, 0.5)` | -34.43421547668306 | -34.538776394910684 | 1.4e13 | 0.0145 | severe |
  | `sf(1e10, 1.2)` | 1.0000889005823404e-12 | 9.99999999999001e-13 | 4.0e11 | 1.2 | severe |
  | `sf(9.0, 3.085754862225318)` | 0.0011348759339604437 | 0.0011348759339603577 | 3.4e2 | 3.08 | minor |

- **Fix direction:** closed form for d = 1: `sf = 1/(1 + x**c)`,
  `logsf = -log1p(x**c)`; generically `sf = -expm1(logcdf)`,
  `logsf = log1mexp(logcdf)`.

### F2: pdf overflows in the left tail (inf or nan)

- **Code:** `burr_gen._pdf`, lines 1204-1205 (the branch used for every x != 0):
  `c*d * x**(-c-1.0) / ((1 + x**(-c))**(d+1.0))`.
- **Mechanism:** pattern 3. `x**(-c-1)` overflows to inf once
  (c+1)|log x| > 709.8. If the denominator is still finite the quotient is
  `inf`; if `(1 + x**-c)**2` has also overflowed (2c|log x| > 709.8) it is
  `inf/inf = nan`. The true pdf is `c*x**(c-1)/(1+x**c)**2 ~ c*x**(c-1)`, a
  normal double over a wide window below the overflow threshold (non-empty for
  c <~ 2400; tens of decades wide for c near 3). The `x == 0` branch of
  `apply_where` (line 1203) does contain the safe form
  `c*d*x**(c*d-1)/(1 + x**c)`, but it is selected only for x exactly 0.
- **Regime:** x < exp(-709.8/(c+1)) = 10**(-308.3/(c+1)) while the true pdf is
  still normal (x > exp(-709/|c-1|)). Re-measured thresholds on a decade grid:
  c = 0.5 gives `inf` for x <= 1e-206 (down to 1e-300, true pdf 5e102 ... 5e149);
  c = 1 `nan` for x <= 1e-155 (true pdf = 1.0); c = 2 `nan` for x <= 1e-103
  (true pdf 2e-103 ... 2e-300); c = 3.0857 `nan` for x <= 1e-76, with the true
  pdf still normal down to about 1e-147; c = 100 only in the sliver near
  1e-3.1 (6 points); c = 1e-3 has its threshold at 1e-308, so only subnormal
  inputs are affected (the 2 "bulk" loss points). 144 points lost in total
  (142 "low tail", 2 "bulk").
- **Reproducer:** `python accuracy.py refs/fisk_ref.py --at pdf 1e-300 2.0`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-300, 2.0)` | nan | 2e-300 | inf | 1 | loss |
  | `pdf(1e-140, 3.085754862225318)` | nan | 3.0456550269891725e-292 | inf | 2.09 | loss |
  | `pdf(1e-300, 0.5)` | inf | 5e+149 | inf | 0.5 | loss |

- **Fix direction:** use the x < 1 form `c*d*x**(c*d-1)/(1+x**c)**(d+1)` (already
  written for x == 0) on all of x < 1, or evaluate `exp(logpdf)` with a repaired
  `logpdf` (F3/F4).

### F3: logpdf and logcdf return -inf in the left tail

- **Code:** `burr_gen._logpdf`, lines 1208-1216 (expression 1213-1215):
  `... - sc.xlog1py(d+1, x**(-c))`; `burr_gen._logcdf`, line 1222:
  `sc.log1p(x**(-c)) * (-d)`.
- **Mechanism:** pattern 3. `x**(-c)` overflows for c|log x| > 709.8 and
  `log1p(inf) = inf` turns both results into `-inf`, although the true values are
  ordinary doubles (-1439.7 and -2131.6 at x = 1e-300, c = 3.0857). These are
  log-scale quantities, so nothing is out of range in the result at all; the
  overflow is purely an artefact of evaluating `x**(-c)` explicitly.
- **Regime:** x < exp(-709.8/c), every c (for c = 1 reachable only at subnormal
  x; from c = 1.2 on hit at x = 1e-300; 38 of 124 points at c = 1e6).
  178 `logpdf` points (133 from this mechanism, the rest from F4) and 133
  `logcdf` points lost.
- **Reproducer:** `python accuracy.py refs/fisk_ref.py --at logcdf 1e-300 3.085754862225318`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logcdf(1e-300, 3.085754862225318)` | -inf | -2131.5639439181737 | inf | 0.00145 | loss |
  | `logpdf(1e-300, 3.085754862225318)` | -inf | -1439.6616197046164 | inf | 0.00145 | loss |
  | `logcdf(1e-300, 1e6)` | -inf | -690775527.8982137 | inf | 0.00145 | loss |

- **Fix direction:** for x < 1 use `log1p(x**-c) = log1p(x**c) - c*log(x)`, hence
  `logcdf = d*(c*log(x) - log1p(x**c))` and
  `logpdf = log(c) + log(d) + (c*d-1)*log(x) - (d+1)*log1p(x**c)`.

### F4: logpdf loses every digit for c close to 1

- **Code:** `burr_gen._logpdf`, lines 1213-1215:
  `np.log(c) + np.log(d) + sc.xlogy(-c-1, x) - sc.xlog1py(d+1, x**(-c))`.
- **Mechanism:** pattern 5. For x < 1 both terms are about (c+1)|log x| and their
  analytic difference is `(c*d-1)*log(x) - (d+1)*log1p(x**c)`. fisk fixes d = 1,
  so `c*d - 1 = c - 1`, and at c = 1 the `log(x)` parts cancel exactly: at
  x = 1e-300 both terms are (d+1)*log1p(x**-c) = 2*690.7755278982137 =
  1381.5510557964274, round to the same double, and the result is `0.0` instead
  of -2e-300. Near c = 1 the loss is roughly (c+1)|log x| / |logpdf| ulp.
- **Regime:** |c - 1| small together with |log x| large (x away from 1). The
  sets c = 1 and c = 1 + 1e-10 give 47 loss + 1 severe and 2 loss + 49 severe
  points; a re-measured scan in c shows the loss decays slowly, staying
  **severe** out to |c - 1| ~ 1e-6 (2.5e6 ulp at c = 1.000001, x = 1e-10) and
  **moderate** out to |c - 1| ~ 1e-4 (1.85e3 ulp at c = 1.0001, x = 1e-10).
- **Reproducer:** `python accuracy.py refs/fisk_ref.py --at logpdf 1e-300 1.0`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-300, 1.0)` | 0.0 | -2e-300 | inf | 1 | loss |
  | `logpdf(1e-10, 1.0)` | -1.999964638343954e-10 | -1.9999999999000001e-10 | 8.0e10 | 1 | severe |
  | `logpdf(1e-130, 1.0000000001)` | -2.9833472581231035e-08 | -2.983360867737124e-08 | 2.1e10 | 0.00335 | severe |
  | `logpdf(1e-10, 1.000001)` | -2.202605141121694e-05 | -2.2026051423523004e-05 | 2.5e6 | 0.0454 | severe |
  | `logpdf(1e-10, 1.0001)` | -0.002202590292199602 | -0.0022025902922005076 | 1.85e3 | 0.0454 | moderate |

  (cond (shapes) is 3.4e302, 1.1e11 and 1.0e10 at these points: logpdf is
  extremely sensitive to c here, but c is an exact user input, so the error
  remains an implementation loss.)
- **Fix direction:** combine the large terms analytically first:
  `log(c) + (c-1)*log(x) - 2*log1p(x**c)` for x < 1 (burr:
  `log(c) + log(d) + (c*d-1)*log(x) - (d+1)*log1p(x**c)`).

### F5: pdf loses ~|log x| * ulp(c+1) because the exponent `-c-1.0` is a rounded sum

- **Code:** `burr_gen._pdf`, lines 1204-1205: `x ** (-c - 1.0)` and `x ** (-c)`.
- **Mechanism:** patterns 11/6. `pow` is accurate for the arguments it gets, but
  `-c - 1.0` is a rounded double with absolute error up to ulp(c+1)/2, and in the
  exponent that error is multiplied by |log x|: the result loses about
  |log x| * ulp(c+1)/2 relative. For c = 1e-3, x = 1e-300 that is
  690.8 * 1.1e-16 = 7.6e-14 = 345 ulp (measured 343), while a 1-ulp change of c
  itself moves the result by only 0.76 ulp — i.e. the loss is created by forming
  `c + 1`, not by the conditioning, and is avoidable. For c of order 1 to 10,
  ulp(c+1) ~ ulp(c), so the same error (484 ulp at c = 3.0857, x = 2.4e75) is
  comparable to cond (shapes) = 535 and largely unavoidable.
- **Regime:** |log x| >~ 100, all parameter sets; 61 "bulk", 46 "low tail" and
  26 "high tail" minor points, max 484 ulp.
- **Reproducer:** `python accuracy.py refs/fisk_ref.py --at pdf 1e-300 0.001`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-300, 0.001)` | 2.2239769159917216e+296 | 2.2239769159918908e+296 | 3.4e2 | 1 | minor |
  | `pdf(2.41667e+75, 3.085754862225318)` | 3.104663961353436e-308 | 3.104663961353675e-308 | 484 | 4.09 | minor |

- **Fix direction:** form the power once and divide by x:
  `u = x**(-c); pdf = c*d*u/(x*(1+u)**(d+1))`, or
  `exp(log(c)+log(d)+xlogy(-c,x)-log(x)-(d+1)*log1p(u))`, so that only the
  exactly given `c` multiplies `log x`.

### F6: ppf/isf lose ~log(1/q) * ulp(1/c) because the exponent `-1.0/c` is rounded

- **Code:** `burr_gen._ppf`, line 1231: `(q**(-1.0/d) - 1)**(-1.0/c)`; `_isf`,
  lines 1234-1235: `sc.expm1(sc.xlog1py(-1.0/d, -q)) ** (-1.0/c)`.
- **Mechanism:** same as F5 for the outer power: `-1.0/c` is rounded (relative
  error up to eps/2) and multiplied by log(1/q - 1) ~ 690 for q = 1e-300, which
  permits ~3.4e2 ulp; measured 64 ulp. This stays below cond (shapes) = 224 (the
  result genuinely depends that strongly on the last bit of c), so the gain from
  a rewrite is small; it is listed because the mechanism is explicit in the code.
- **Regime:** q <~ 1e-10 with c whose reciprocal is inexact: 34 `ppf` and 35
  `isf` minor points (c = 3.0857 and c = 1.2); exact-reciprocal shapes
  (c = 0.5, 1, 2) show nothing.
- **Reproducer:** `python accuracy.py refs/fisk_ref.py --at ppf 1e-300 3.085754862225318`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 3.085754862225318)` | 6.012515559619745e-98 | 6.012515559619659e-98 | 64 | 0.324 | minor |
  | `isf(1e-300, 3.085754862225318)` | 1.6631973590489036e+97 | 1.6631973590489274e+97 | 64 | 0.324 | minor |

- **Fix direction:** replace the rounded reciprocal by a division, e.g.
  `exp(-log1p(expm1(-log(q)/d))/c)`; the residual error (~|log q| * eps) is
  intrinsic.

## RuntimeWarnings

- `pdf`: `overflow encountered in power`, `invalid value encountered in divide`,
  `overflow/underflow encountered in multiply`, `underflow encountered in
  divide/power` — in the F2 regime and in the far tails. Separately,
  `divide by zero encountered in power` is raised by the `x == 0` branch
  (line 1203, `x**(c*d-1)`) when x = 0 and c < 1, where `inf` is the correct
  answer: a spurious warning on a correct value.
- `logpdf`, `cdf`, `logcdf`: `overflow encountered in power` in the F3 regime and
  `underflow encountered in power` in the far right tail. For `cdf` the value is
  correct anyway (`1/(1+inf) = 0` matches the subnormal true cdf), so those are
  spurious; for `logpdf`/`logcdf` they accompany the F3 loss.
- `sf`, `logsf`: `divide by zero encountered in log1p` exactly in the F1 loss
  regime, plus `overflow/underflow encountered in power`.
- `ppf`, `isf`: `overflow/underflow encountered in power` for extreme q with
  c = 1e-3 and c = 0.5, where the true quantile is outside the double range
  (no accuracy issue).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid, all 9 parameter sets (1154 points) | 1.0 ulp |
| logcdf | bulk and right tail (654 points) | 1.29 ulp |
| logpdf | bulk and right tail (873 points) | 2.77 ulp |
| sf, logsf | left tail and bulk | 0 / 1.35 ulp in the left tail; up to 3.4e2 (sf) and 50 (logsf) ulp in the bulk, continuous with F1 |
| ppf | q in [1e-10, 1 - 3e-16], all parameter sets | 3.4e15 ulp at q = 1 - 3e-16, within cond (arg) = 1.8e16; <= 1e3 ulp for q <= 0.999. burr's F4/F5 (`q**(-1/d)` cancelling for large d, overflowing for small d) cannot occur at d = 1: `q**(-1)` is a single rounding and never overflows for q >= 1e-308 |
| isf | full q grid | 31 ulp near q = 1 (median 1.9); burr's F6 (`expm1` overflow) would need 1 - q < 1e-308 at d = 1 and is unreachable |
| pdf | right tail at c = 1e6 where `x**(-c-1)` is subnormal (x = 1.00071 ... 1.00072; pattern 4) | 3.4e5 ulp, classified **ok** by the tool because cond (arg) = c = 1e6 there. The mechanism (subnormal intermediate lifted back to normal by the factor c*d = c) is real, but for fisk the lift factor equals the conditioning, so it never exceeds it |
| pdf, logpdf at x = 0 | not measured — the mpmath reference raises ZeroDivisionError at x = 0 (2 points skipped per method) | code reading plus direct check: the `x == 0` branch returns the correct limits inf / 1.0 / 0.0 for c < 1 / c = 1 / c > 1 |

## Suspected, not confirmed

- F1 for very small c: the total-loss threshold x > exp(36.7/c) is outside the x
  grid for c <= 1e-3 (it would need x > e^36700), so that corner could not be
  measured; the mechanism does not depend on c, so the behaviour is expected to
  be identical.
- Pattern 4 in `pdf` for c > 1e6: the error grows with the lift factor c, but so
  does cond (arg) = c, so it should remain "ok"; not tested beyond c = 1e6.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| (none) | — | — | all eight distribution functions are implemented via `burr_gen`; nothing is left to the generic `rv_continuous` defaults |

## Shared code

Every measured line is `burr_gen` code — `fisk_gen` only forwards
`(x, c) -> (x, c, 1.0)` (lines 1406-1430). Hence

- F1 = burr F1 (`log1p(-cdf)` in `_sf`/`_logsf`),
- F2 = burr F2 for the pdf, F3 = burr F2 for logpdf/logcdf,
- F4 = burr F3 specialized to `c*d = 1`, which for fisk means c = 1,

exactly as documented in the bundled burr review for the same SciPy version
(`.claude/skills/dist-accuracy-review/references/burr-example.md`). F5 and F6
(rounded exponents `-c-1.0` and `-1.0/c` amplified by large logarithms) are minor
and were not called out there. A fix inside
`burr_gen._pdf/_logpdf/_logcdf/_sf/_logsf` repairs burr and fisk at once;
`burr12` and `mielke` contain the same `1 - (...)` / `x**-c` idioms and should be
reviewed separately. No `scipy.special` inaccuracy was observed: `log1p`,
`xlogy`, `xlog1py` and `expm1` all return correct values for the arguments they
receive — the arguments, and the absence of complementary forms, are the problem.

## Existing GitHub issues
- #17832 (open): Tracking issue: custom methods for continuous distributions -
  relates to F1; explicitly names the `sf = 1 - _cdf` pattern as a
  "risk of catastrophic cancellations in the right tail", and lists both burr
  and fisk as having `_sf` but no `_logsf`/`_logcdf` override
- #7491 (closed): Several scipy.stats distributions (fisk, burr, burr12, f)
  return incorrect pdf at left endpoint - relates to F2/F3; fixed the exact
  endpoint `pdf(0)` (PR #10381, SciPy 1.4.0, the `x == 0` branch now at
  burr_gen line 1203), but not the overflow for small x > 0 reported here
- No issue found for F4 (`c*d = 1` logpdf cancellation), F5 or F6

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `sf(1e10, 3.085754862225318)` | 0.0 | 1.3881978829312955e-31 | verified (loss) | own ref `1/(1+x**c)`; cliff at `x**-c < eps/2` reproduced, `sf(1e20, 1.0) = 0.0` |
| F1 | `logsf(1e10, 3.085754862225318)` | -inf | -71.05213146393912 | verified (loss) | own ref `-log1p(x**c)` |
| F1 | `sf(1e30, 0.5)` | 1.1102230246251546e-15 | 9.999999999999989e-16 | verified (severe) | 5.0e14 ulp, matches report |
| F1 | `sf(1e10, 1.2)` | 1.0000889005823404e-12 | 9.99999999999001e-13 | verified (severe) | 4.0e11 ulp |
| F2 | `pdf(1e-300, 2.0)` | nan | 2e-300 | verified (loss) | own ref `c*x**(c-1)/(1+x**c)**2` |
| F2 | `pdf(1e-140, 3.085754862225318)` | nan | 3.0456550269891725e-292 | verified (loss) | |
| F2 | `pdf(1e-300, 0.5)` | inf | 5e+149 | verified (loss) | |
| F2 | regime scan over x = 1e-308..1e-1 | nan from x < 1e-103 (c = 2), x < 1e-76 (c = 3.0857), inf from x < 1e-206 (c = 0.5) | — | regime understated | the stated windows ("1e-300 ... 1e-237" for c = 2, "1e-148 ... 1e-100" for c = 3.0857) are too narrow; onset matches `x < exp(-709.8/(c+1))` as stated in the table row, so only the per-c numbers in the F2 "Regime" line need correcting |
| F3 | `logcdf(1e-300, 3.085754862225318)` | -inf | -2131.5639439181737 | verified (loss) | own ref `c*log(x) - log1p(x**c)` |
| F3 | `logpdf(1e-300, 3.085754862225318)` | -inf | -1439.6616197046164 | verified (loss) | own ref `log(c) + (c-1)*log(x) - 2*log1p(x**c)` |
| F3 | `logcdf(1e-300, 1e6)` | -inf | -690775527.8982137 | verified (loss) | -inf onset measured at 1e-257 (c = 1.2), 1e-100 (c = 3.0857), 1e-1 (c = 1e6) = `10**(-308.25/c)`, exactly as claimed |
| F4 | `logpdf(1e-300, 1.0)` | 0.0 | -2e-300 | verified (loss) | |
| F4 | `logpdf(1e-10, 1.0)` | -1.999964638343954e-10 | -1.9999999999000001e-10 | verified (severe) | 8.0e10 ulp |
| F4 | `logpdf(1e-130, 1.0000000001)` | -2.9833472581231035e-08 | -2.983360867737124e-08 | verified (severe) | 2.1e10 ulp; scan shows severe loss persists to c = 1.000001 (2.1e6 ulp), i.e. wider than the stated \|c-1\| <~ 1e-8 |

Mechanisms: all four cited expressions are present in the installed 1.18.1 source
and the attributed overflow/cancellation is confirmed by the emitted
RuntimeWarnings. Two wording issues: the `burr_gen` line numbers are 3-4 too high
(installed file has `_pdf` 1199-1205, `_logpdf` 1208-1215, `_logcdf` 1222, `_sf`
1224, `_logsf` 1228, `_ppf` 1231, `_isf` 1233-1235; the `fisk_gen` numbers are
correct), and in F4 the two cancelling terms are 1381.5510557964274
(= `(d+1)*690.7755278982137`), not 690.77552789821368.

References were built independently from the docstring pdf
(`scratch/fisk/verify/ref.py`), cross-checked against `mp.quad` of the pdf and by
`cdf(ppf(q)) == q`, and evaluated at 100 and 200 digits (agreement ~1e-102).

Verified by dist-accuracy-verifier: 4 of 4 verified.
