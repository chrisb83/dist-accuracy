# rel_breitwigner: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `rel_breitwigner_gen`, lines 12473-12618
- **Implemented:** pdf, cdf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), ppf = numerical inversion of cdf, isf = ppf(1 - q)
- **Related:** none; helpers: `_check_fit_input_parameters` (fit only); special functions: none (NumPy complex `np.sqrt`, `np.arctan`)
- **Reference:** `refs/rel_breitwigner_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (36.545206797050334), (1), (0.001), (1e-08), (1000), (1e+06), (1e+20), (1e-160), (1e-154), (1e-150), (1e+150), (1e+152), (1e+154), (1e+160)
- **Worst finding:** loss in pdf and cdf

## Summary

For 1e-150 <= rho <= 1e150, pdf (max 1.99 ulp) and cdf are accurate. The cdf's
large ulp counts around the peak at x ~ rho for large rho stay within the
argument conditioning, which is about rho. Both methods fail completely at
extreme rho because `rho**2` or `1/rho**2` overflow:
- For rho < ~1e-154, pdf returns inf/nan and cdf returns 0.
- For rho > 1.34e154, cdf returns 0 everywhere.
- For 8.6e153 <= rho < 1.34e154, the cdf is wrong (factor 3 in the left tail,
  0.5 or 1.0 where the true value is 4e-155). The cause is a wrong result from
  NumPy's complex `sqrt` near DBL_MAX.

The pdf also loses its far right tail to an overflowing square. This matters
mostly for small rho: at rho = 1e-150, pdf(127.6) = 0 although the true value
is 3.4e-234.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | rho > 1.34e154, all x | `-rho*(rho + 1j)` overflows, b = inf (pattern 3) | 0.0 for 0.5 and 1.0 | loss |
| F2 | cdf | 8.63e153 <= rho < 1.34e154, all x | NumPy complex sqrt halves the imaginary part of its result once \|Re\| or \|Im\| of the argument >= 7.446e307; here Re = -rho**2 and the halved part is the large one (shared code) | 0.5 for 3.9e-155 (5.8e169 ulp) | loss |
| F3 | cdf | rho < 7.46e-155, all x | `1/rho**2` overflows, C = 0 (pattern 3) | 0.0 for 1.0 | loss |
| F4 | pdf | rho < 1.055e-154, all x | `2*(1 + 1/rho**2)` overflows, C = inf or nan (pattern 3) | nan / inf | loss |
| F5 | pdf | right tail, x > sqrt(1.34e154*rho); a wide range for small rho | `(... / rho)**2` overflows (pattern 3) | 0.0 for 3.4e-234 | loss |
| F6 | pdf | rho > ~3.36e154, near the peak (\|x - rho\| > ~9e307/rho) | `(x - rho)*(x + rho)` overflows before `/rho` (pattern 3) | 0.0 for 1.6e-305 | loss |

### F1: cdf is 0 for rho > 1.34e154

- **Code:** `_cdf`, line 12554: `np.arctan(x/np.sqrt(-rho*(rho + 1j)))`
- **Mechanism:** `-rho*(rho + 1j)` has real part `-rho**2`, which overflows to
  -inf for rho > 1.34e154. The sqrt is then `-inf j`, `x/b` is 0, and the
  arctan is 0, so the cdf is 0 for every x (pattern 3). The final result
  does not need `rho**2`: b = sqrt(rho) * sqrt(-(rho + 1j)) is finite.
- **Regime:** rho > 1.34e154 (checked: rho = 1.34e154 does not hit this path,
  rho = 1.35e154 gives 0), all x.
- **Reproducer:** `python accuracy.py refs/rel_breitwigner_ref.py --at cdf 1e+20 1e+160`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e+20, 1e+160)` | 0.0 | 6.366197723675813e-301 | inf | 1 | loss |
  | `cdf(1e+300, 1e+160)` | 0.0 | 1.0 | inf | 0 | loss |
  | `cdf(1e+160, 1e+160)` | 0.0 | 0.5 | inf | 1.3e+160 | loss |

- **Fix direction:** form `b = np.sqrt(rho) * np.sqrt(-rho - 1j)` (or scale by
  rho before the complex sqrt) so that `rho**2` never forms.

### F2: cdf is wrong for 8.63e153 <= rho < 1.34e154 (NumPy complex sqrt)

- **Code:** `_cdf`, line 12554: `np.sqrt(-rho*(rho + 1j))`
- **Mechanism:** with NumPy 2.5.4 on this platform, `np.sqrt` of a complex128
  returns the imaginary part of the result halved (the real part is correct)
  once |Re| or |Im| of the argument reaches DBL_MAX/(1+sqrt 2) =
  7.446288774449766e307. Bisected: 7.446288774449765e307 is correct, the next
  double is not. The threshold is the same for Re < 0, Re > 0 and a large Im.
  Here the argument is `-rho*(rho + 1j)` with Re = -rho**2 < 0, so the
  imaginary part of the result is the large component (about -rho), and that
  is the part that is halved. Examples (mpmath and `cmath.sqrt` agree):
  - `np.sqrt(complex(-1e308, -1e154))` gives `0.5-5e+153j` instead of
    `0.5-1e+154j`.
  - `np.sqrt(complex(-7.5e307, 0))` gives `4.33e153j` instead of `8.66e153j`.

  b is therefore wrong by a factor of 2 in its imaginary part. The cdf comes
  out 3 times too large in the left tail and jumps to 0.5 and then 1.0 far
  below the peak. SciPy calls the function correctly; the error is in NumPy
  (see "Shared code").
- **Regime:** rho**2 >= 7.446e307, i.e. 8.63e153 <= rho < 1.34e154, all x.
  Checked: rho = 8.6e153 is correct in the left tail, 8.65e153 is not.
- **Reproducer:** `python accuracy.py refs/rel_breitwigner_ref.py --at cdf 5e+153 1e+154`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(5e+153, 1e+154)` | 0.5 | 3.870561670722237e-155 | 5.8e+169 | 1.46 | loss |
  | `cdf(3.5, 1e+154)` | 6.684507609859605e-308 | 2.228169203286535e-308 | 9.0e+15 | 1 | loss |
  | `cdf(9.9999999e+153, 1e+154)` | 1.0 | 1.5915497289599643e-147 | 2.83e+162 | 1e+08 | loss |

- **Fix direction:** the F1 fix (`sqrt(rho)*sqrt(-rho - 1j)`) also keeps the
  complex sqrt argument far from DBL_MAX.

### F3: cdf is 0 for rho < 7.46e-155

- **Code:** `_cdf`, line 12551: `C = np.sqrt(2/(1 + np.sqrt(1 + 1/rho**2)))/np.pi`
- **Mechanism:** `1/rho**2` overflows to inf for rho < 7.46e-155, so C =
  sqrt(2/inf) = 0 and the cdf is 0 for every x, including where the true
  value is 1 (pattern 3). The true C is about sqrt(2*rho)/pi, which is a
  normal number.
- **Regime:** rho < 7.46e-155, all x. For 7.46e-155 <= rho, C is finite and the
  cdf is accurate: rho = 1e-154 gives max 1.57 ulp.
- **Reproducer:** `python accuracy.py refs/rel_breitwigner_ref.py --at cdf 1 1e-160`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1, 1e-160)` | 0.0 | 1.0 | inf | 9e-241 | loss |
  | `cdf(1e-300, 1e-160)` | 0.0 | 9.00316316157106e-221 | inf | 1 | loss |

- **Fix direction:** with `h = np.hypot(1, 1/rho)`, or better
  `rho*sqrt(1 + 1/rho**2) = hypot(rho, 1)`, write
  `C = sqrt(2*rho/(rho + hypot(rho, 1)))/pi`. This form never overflows.

### F4: pdf is inf or nan for rho < 1.055e-154

- **Code:** `_pdf`, lines 12543-12545:
  `C = np.sqrt(2 * (1 + 1/rho**2) / (1 + np.sqrt(1 + 1/rho**2))) * 2 / np.pi`
- **Mechanism:**
  - For rho < 1.055e-154, `2*(1 + 1/rho**2)` overflows while the denominator
    is still finite, so C = inf. The pdf is inf near the peak and
    `inf/inf = nan` in the tail, where the denominator also overflows.
  - For rho < 7.46e-155, both overflow and C = sqrt(inf/inf) = nan
    everywhere (pattern 3).

  The true C is about (2/pi)*sqrt(2/rho), at most ~1e162 for rho = 5e-324.
- **Regime:** rho < 1.055e-154, all x. Checked: 1.05e-154 gives inf, 1.06e-154 is
  finite.
- **Reproducer:** `python accuracy.py refs/rel_breitwigner_ref.py --at pdf 1e-154 1e-154`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-154, 1e-154)` | inf | 9.00316316157106e+76 | inf | 0 | loss |
  | `pdf(10, 1e-154)` | nan | 9.00316316157106e-236 | inf | 4 | loss |
  | `pdf(1, 1e-160)` | nan | 9.003163161571061e-241 | inf | 4 | loss |
  | `pdf(1e-300, 1e-160)` | nan | 9.00316316157106e+79 | inf | 0 | loss |

- **Fix direction:** `h = hypot(rho, 1)`;
  `C = (2/pi)*sqrt(2*h/(rho + h)) * sqrt(h)/rho`. Equivalently, split the
  sqrt so that no factor exceeds ~1/rho.

### F5: pdf right tail is 0 where the true value is normal

- **Code:** `_pdf`, line 12547: `C / (((x - rho)*(x + rho)/rho)**2 + 1)`
- **Mechanism:** u = (x**2 - rho**2)/rho overflows when squared once
  u > 1.34e154, so the pdf returns C/inf = 0 (pattern 3). The true value
  there is about C/u**2 ~ C/1.8e308. This is normal whenever C > ~1, i.e.
  for rho < ~1. As rho shrinks, the threshold x = sqrt(1.34e154*rho) drops
  and C grows like rho**-1/2, so the lost range widens. For rho = 1e-150 the
  pdf is 0 from x ~ 116 although the true value stays normal up to x ~ 1e20.
  For rho >= 1 the pdf at the threshold is already subnormal and nothing
  measurable is lost.
- **Regime:** x > sqrt(1.34e154*rho) and pdf > 2.2e-308, which needs rho < ~1.
- **Reproducer:** `python accuracy.py refs/rel_breitwigner_ref.py --at pdf 127.6 1e-150`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(127.6, 1e-150)` | 0.0 | 3.3961934751049296e-234 | inf | 4 | loss |
  | `pdf(1e+20, 1e-150)` | 0.0 | 9.003163161571061e-306 | inf | 4 | loss |
  | `pdf(4.035066294374852e+75, 0.001)` | 0.0 | 1.0734343612857905e-307 | inf | 4 | loss |
  | `pdf(1.2760000000000002e+73, 1e-08)` | 0.0 | 3.396193458123959e-305 | inf | 4 | loss |

- **Fix direction:** for large u use `(C/u)/u`. This is the same overflow as
  INDEX item 2 (x**2 / x**4).

### F6: pdf is 0 near the peak for rho > ~3.36e154

- **Code:** `_pdf`, line 12547: `(x - rho)*(x + rho)/rho`
- **Mechanism:** the product `(x - rho)*(x + rho)` is about 2*rho*|x - rho|
  and overflows before the division by rho. For rho = 1e160 and
  x = rho*(1 - 1e-8), the product is 2e312 although the quotient is 2e152 and
  the pdf is 1.6e-305 (pattern 3).
- **Regime:** |x - rho| > ~9e307/rho (the product overflows) while the true pdf
  is still normal. Just past the overflow, u = (x**2 - rho**2)/rho is about
  DBL_MAX/rho, so the true pdf is about (2/pi)*(rho/DBL_MAX)**2. That is a
  normal double only for rho > DBL_MAX*sqrt(DBL_MIN*pi/2) = 3.36e154. For
  smaller rho the overflow only zeroes subnormal values, which is not
  measured. Checked at the first overflowing double above rho: for rho =
  1.6e154, 3.3e154 and 3.35e154 the true values are 5.04e-309, 2.15e-308 and
  2.21e-308 (all subnormal). For rho = 3.37e154 and 3.4e154 they are 2.24e-308
  and 2.28e-308 (normal, SciPy returns 0). The previous double gives the
  correct value in each case. Script: `scratch/rel_breitwigner/recheck/f6_recheck.py`.
- **Reproducer:** `python accuracy.py refs/rel_breitwigner_ref.py --at pdf 9.9999999e+159 1e+160`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(9.9999999e+159, 1e+160)` | 0.0 | 1.5915494491129255e-305 | inf | 2.0e+08 | loss |
  | `pdf(1.00000001e+160, 1e+160)` | 0.0 | 1.5915494172819369e-305 | inf | 2.0e+08 | loss |

  (cond (arg) is 2e8, but returning 0 is a loss regardless of conditioning.)
- **Fix direction:** `(x - rho)*((x + rho)/rho)`.

## RuntimeWarnings

- pdf (12 parameter sets):
  - `overflow encountered in square`/`divide` accompanies the F5/F6 zeros.
  - `invalid value encountered in divide` and `overflow encountered in
    multiply` accompany F4.
  - `underflow encountered in square`/`divide` appears for huge x, where the
    result is subnormal (not measured), and for tiny rho, where
    `(x - rho)*(x + rho)/rho` underflows harmlessly. The result is accurate
    there (e.g. rho = 1e-154 at large x is the F4 nan, but rho = 1e-150 at
    small x is accurate), so these warnings are spurious.
- cdf (9 parameter sets):
  - `overflow encountered in multiply/square/divide` accompanies F1/F3.
  - `underflow encountered in arctan`/`divide`/`multiply` appears on accurate
    results in the far left tail (e.g. `cdf(0.5, 1e-154)` = 1.0, 0 ulp) and
    alongside F2.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | x = 1e-300 ... 1e300, both sides of x = rho (rho*(1 +- 1e-8), rho +- 0.25 ... 2, rho/2, 2*rho), both sides of the square overflow x = 1.16e77*sqrt(rho); rho = 1e-150 ... 1e154 | 1.99 ulp (outside F4-F6) |
| cdf | same x grid, left tail down to x = 1e-300 (cancellation hypothesis H3), right tail where cdf -> 1 and the `np.clip(result, None, 1)` at line 12558; rho = 1e-154 ... 1e150 | 1.65 ulp for rho <= 1; 37.7 ulp at cond 45 (rho = 36.5452); 318 ulp at cond 1.44e3 (rho = 1000); 5.9e5 ulp at cond 1.3e6 (1e6); 1.7e7 at cond 1.0e8 (1e20); 2.9e7 at cond 1.0e8 (1e150) - all ok |

Around the peak for large rho the cdf is the imaginary part of
`arctan(z)` with z = x/b close to the branch point i. Rounding z to a
double shifts the result by about eps*rho. For example, `cdf(1e152, 1e152)`
returns 1.4335402848056703e-137 instead of 0.5 (4.5e15 ulp). This is still
within the argument conditioning (1.3e152), because one ulp of x near rho is
far larger than the peak width (~1). The tool rates it ok and it is not a
finding.

## Suspected, not confirmed

none. H3 (cancellation in the imaginary part of the complex product for
tiny cdf values in the left tail) was tested down to x = 1e-300 for all
parameter sets and did not show.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | `2*C*Im(sqrt(-1 + 1j/rho) * arctan(b/x))` with b = sqrt(-rho*(rho + 1j)), using arctan(z) + arctan(1/z) = pi/2 for Re z > 0; for large x use the series sf ~ k/(3 x**3) | right tail, sf < ~1e-16: `sf(1e6, 36.5452)` = 2.220446049250313e-16 vs 2.8349252324843297e-16 |
| logsf | log(sf) | log of the custom sf above (asymptote log(k/3) - 3 log x) | same as sf; -inf once cdf rounds to 1 |
| logpdf | log(pdf) | `log(C) - log1p(u**2)`, with `log(C) - 2*log(u)` for large u, u = (x - rho)*((x + rho)/rho) | wherever pdf is 0/inf (F4-F6) or subnormal: `logpdf(1e80, 1)` = -inf vs -737.0263520763339 |
| logcdf | log(cdf) | log of cdf; for small x use log(pdf(0)) + log(x) | only where cdf is subnormal (x < ~1e-305/pdf(0)); minor |
| ppf | numerical inversion of cdf | no closed form (the cdf is an arctan of a complex argument); root finding on the custom cdf with a start from the Cauchy-like approximation near rho is adequate | inherits F1-F3 at extreme rho; slow |
| isf | ppf(1 - q) | invert a custom sf; for small q use x ~ (k/(3q))**(1/3) as start | q < ~1e-16: `isf(1e-20, 36.5452)` = 10000000.0 vs 30491622.631388664 |

## Shared code

- NumPy complex `np.sqrt` (NumPy 2.5.4, Windows build): the imaginary part of
  the result is halved, and the real part is correct, whenever |Re| or |Im| of
  the complex128 argument is >= DBL_MAX/(1+sqrt 2) = 7.446288774449766e307. The
  threshold was bisected to adjacent doubles. It is the same for negative
  real, positive real and imaginary arguments, and it matches the overflow
  threshold of the msun-style csqrt scaling. The halved part is the large
  component when Re < 0 (the F2 case) and the small component when Re > 0.
  Examples (mpmath reference; `cmath.sqrt` agrees with it):
  - `np.sqrt(complex(-1e308, 0))` gives `5e+153j` instead of `1e+154j`.
  - `np.sqrt(complex(1e308, 1))` gives `1e+154+2.5e-155j` instead of
    `1e+154+5e-155j`.
  - `np.sqrt(1.5e308j)` gives `8.66e153+4.33e153j` instead of
    `8.66e153+8.66e153j`.

  Script: `scratch/rel_breitwigner/recheck/npsqrt_recheck.py`. This causes F2 here and can affect any SciPy code
  that takes complex square roots of values near DBL_MAX. It may be specific
  to NumPy's fallback csqrt used on this platform; this was not checked on
  other platforms.
- Inherited `sf = 1 - cdf` (INDEX item 1) and x**2/x**4 overflow (INDEX item 2,
  F5) are the shared patterns.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable; no NumPy issue found for the complex `np.sqrt` behaviour behind F2 either)

## Verification

Verifier reference: the docstring pdf in mpmath, rewritten in u = x - rho (exact double difference), with
x**2 - rho**2 = u*(u + 2*rho) so the peak can be resolved at any precision; cdf by `mp.quad` of the pdf over
log-spaced segments, each segment rescaled to O(1) (x-route left of rho/2, `1 - int_x^inf` beyond the peak).
Normalization checked to 1e-100 for rho = 1e-160 ... 1e160. All values agree between 100 and 200 digits
to 30+ digits. Scripts: `scratch/rel_breitwigner/verify/` (vref.py, run.py, npsqrt.py, npsqrt2.py, mech.py, f6thr.py).

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e+20, 1e+160)` | 0.0 | 6.3661977236758133e-301 | verified | `-rho*(rho+1j)` = `-inf-1e160j` and sqrt = `-inf j` confirmed; rho = 1.34e154 finite, 1.35e154 gives 0 |
| F1 | `cdf(1e+300, 1e+160)` | 0.0 | 1.0 | verified | |
| F1 | `cdf(1e+160, 1e+160)` | 0.0 | 0.5 (to 17 digits) | verified | |
| F2 | `cdf(5e+153, 1e+154)` | 0.5 | 3.8705616707222366e-155 | verified | NumPy claim confirmed directly: `np.sqrt(complex(-1e308,-1e154))` = `0.5-5e+153j`, `cmath`/mpmath give `0.5-1e+154j`. Refinement: NumPy 2.5.4 halves the **imaginary part** whenever \|Re\| or \|Im\| >= DBL_MAX/(1+sqrt 2) = 7.4463e307 (the msun csqrt THRESH; bisected: 7.446288774449759e307 correct, ...774e307 halved). That is the large component only for Re < 0, which is the case here, so the F2 mechanism holds. For Re > 0 it is the small component (`1e308+1j` -> imag 2.5e-155), and a pure large imaginary argument also triggers it (`1.5e308j`), so "large component" and "\|Re\| > 7.45e307" in the F2 and Shared-code text are imprecise |
| F2 | `cdf(3.5, 1e+154)` | 6.684507609859605e-308 | 2.2281692032865345e-308 | verified | factor 3. Boundary confirmed: rho = 8.62e153 correct, 8.63e153 wrong (sqrt(THRESH) = 8.629e153) |
| F2 | `cdf(9.9999999e+153, 1e+154)` | 1.0 | 1.5915497289599643e-147 | verified | |
| F3 | `cdf(1, 1e-160)` | 0.0 | 1.0 | verified | `1/rho**2` = inf, C = 0 at line 12551; 7.45e-155 gives 0, 7.47e-155 gives 1.0 |
| F3 | `cdf(1e-300, 1e-160)` | 0.0 | 9.003163161571061e-221 | verified | |
| F4 | `pdf(1e-154, 1e-154)` | inf | 9.0031631615710608e+76 | verified | numerator overflows below 1.055e-154 (1.05e-154 inf, 1.06e-154 finite); both overflow below 7.46e-155 -> nan, lines 12543-12545 |
| F4 | `pdf(10, 1e-154)` | nan | 9.0031631615710603e-236 | verified | |
| F4 | `pdf(1, 1e-160)` | nan | 9.0031631615710605e-241 | verified | |
| F4 | `pdf(1e-300, 1e-160)` | nan | 9.0031631615710607e+79 | verified | |
| F5 | `pdf(127.6, 1e-150)` | 0.0 | 3.3961934751049295e-234 | verified | u = 1.63e154, u**2 = inf (line 12547); x = 115 still finite, threshold ~115.8 as stated |
| F5 | `pdf(1e+20, 1e-150)` | 0.0 | 9.0031631615710608e-306 | verified | |
| F5 | `pdf(4.035066294374852e+75, 0.001)` | 0.0 | 1.0734343612857906e-307 | verified | |
| F5 | `pdf(1.2760000000000002e+73, 1e-08)` | 0.0 | 3.3961934581239592e-305 | verified | |
| F6 | `pdf(9.9999999e+159, 1e+160)` | 0.0 | 1.5915494491129256e-305 | verified | `(x-rho)*(x+rho)` = -inf, while `(x-rho)*((x+rho)/rho)` = -2.0e152. Regime bound is off: just past the product overflow the true pdf is ~(2/pi)*(rho/DBL_MAX)**2, which is normal only for rho > ~3.4e154 (checked: 3.3e154 -> 1.98e-308, 3.4e154 -> 2.11e-308), not ~1.6e154 |
| F6 | `pdf(1.00000001e+160, 1e+160)` | 0.0 | 1.591549417281937e-305 | verified | |

Verified by dist-accuracy-verifier: 6 of 6 verified.
