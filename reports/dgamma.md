# dgamma: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `dgamma_gen`, lines 1757-1831
- **Implemented:** pdf (1797-1800), logpdf (1802-1804), cdf (1806-1809), sf (1811-1814), ppf (1819-1822), isf (1824-1827)
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf) (the `rv_continuous` defaults that switch to `log1p(-sf)` / `log1p(-cdf)` at the median)
- **Related:** `gamma` (dgamma is the gamma distribution reflected about 0; the methods call the same `scipy.special` functions as `gamma_gen`, see reports/gamma.md); helpers: none; special functions: `sc.gamma`, `sc.gammaln`, `sc.xlogy`, `sc.gammainc`, `sc.gammaincc`, `sc.gammaincinv`, `sc.gammainccinv`
- **Reference:** `refs/dgamma_ref.py`, 120 digits (mpmath's upper `gammainc` takes seconds at 400 digits); self-checks: PASS except 4 FAIL lines of the pdf-integral check for a < 1, shown to be a quadrature artifact (see below)
- **Parameter sets:** (1.1023326088288166), (0.001), (0.1), (1), (10.5), (150), (200), (1000), (1e6); single points at a = 1e-100, 1e-300, 1e-310
- **Worst finding:** loss in pdf (inf/nan/0 at the mode for a >~ 144, everywhere for a > 171.6) and logpdf (subnormal a); severe in pdf (exp underflow) and ppf/isf (a = 1e6)

## Summary

cdf and sf are accurate (<= 15 ulp) except where they inherit known
`sc.gammainc`/`sc.gammaincc` errors from the gamma review, diluted by the
`0.5 + 0.5*P` form (5.4e3 ulp at a = 1e6, 2e3 ulp at a = 1e3). The pdf is the
weak method: `_pdf` multiplies `1/(2*gamma(a))`, `|x|**(a-1)` and
`exp(-|x|)` as separate doubles. It returns inf or nan at the mode once
a >~ 144, and 0 or nan everywhere once `gamma(a)` overflows (a > 171.6). In
the far tail, where `exp(-|x|)` is subnormal, it loses up to 1.3e15 ulp. logpdf
has the same large-a cancellation as gamma's (4.3e5 ulp at a = 1e6). It returns
-inf for subnormal a because `sc.gammaln` returns inf there. ppf/isf
near q = 1/2 use `gammaincinv`/`gammainccinv` at full strength, so they
inherit gamma's F3 (1.3e6 ulp at a = 1e6) and F5 (2.9e3 ulp at a = 1e-3).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf | (a-1)*log\|x\| > 709.78 (a >~ 100 in the tail, a >~ 144 at the mode); a > 171.62 everywhere | `\|x\|**(a-1)` and `sc.gamma(a)` overflow (pattern 3) | inf / nan / 0.0 for normal values | loss |
| F2 | pdf | \|x\| > ~708 with a normal pdf (a = 10.5: 708 < \|x\| < ~755; a = 100: up to \|x\| ~ 1300) | `np.exp(-ax)` subnormal or 0 before multiplying by a large prefactor (pattern 4) | 1.3e15 ulp; 0.0 for 2.4e-303 | loss |
| F3 | ppf, isf | a = 1e6, q within ~1e-4 of 1/2 | `gammaincinv`/`gammainccinv` invert the unconverged `gammainc` (gamma F3; pattern 10) | 1.3e6 ulp | severe |
| F4 | logpdf | a >~ 1e3, \|x\| within a few sqrt(a) of a | cancellation of `xlogy(a-1,\|x\|)`, `\|x\|`, `gammaln(a)` (pattern 5; gamma F4) | 4.3e5 ulp | moderate |
| F5 | logpdf, pdf | a < ~5.6e-309 (subnormal shape) | `sc.gammaln(a)` returns inf; `sc.gamma(a)` overflows (patterns 3, 10) | -inf for -715.49; 0.0 for 5e-11 | loss |
| F6 | ppf, isf | a = 1e-3, q in the bulk (result 1e-300 ... 1e-70) | `gammaincinv` exponent amplification for a << 1 (gamma F5) | 2.91e3 ulp | moderate |
| F7 | cdf, sf | a = 1e6, x = +-(a - 5 sqrt(a)); a >= 200 with \|x\| far from a | `gammainc` unconverged series (gamma F1) and exp-prefactor rounding (gamma F6) | 5.37e3 ulp | moderate |
| F8 | pdf | a not exactly representable near 1 (a = 0.1), \|x\| -> 0 | rounding of `a - 1.0` amplified by \|log x\| in `ax**(a-1.0)` (pattern 6) | 86 ulp | minor |

### F1: pdf overflows for moderate and large a

- **Code:** `_pdf`, line 1800: `1.0/(2*sc.gamma(a))*ax**(a-1.0) * np.exp(-ax)`
- **Mechanism:** pattern 3. The three factors are formed separately.
  `ax**(a-1.0)` overflows to inf once `(a-1)*log|x| > 709.78`, although
  `/Gamma(a)` and `exp(-|x|)` would bring the product back into range.
  This gives inf, or nan (`inf*0`) where `exp(-ax)` underflows. `sc.gamma(a)`
  overflows for a > 171.62, so `1/(2*gamma(a)) = 0`. The pdf is then 0.0
  where the power is finite and nan (`0*inf`) where it overflows.
- **Regime:** in the tail for a >~ 100, where the power overflows while the
  pdf is still normal (e.g. a = 110, |x| = 1000); a >~ 144 at the mode (`(a-1) log(a-1) > 709.78` for
  a - 1 > 143.0), i.e. the whole bulk for a in [144, 171.6]; every x for
  a > 171.62. Both signs of x.
- **Reproducer:** `python accuracy.py refs/dgamma_ref.py --at pdf -150 150`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(-150, 150)` | inf | 0.016277704728418274 | inf | 1 | loss |
  | `pdf(120, 150)` | inf | 0.0006321726243224847 | inf | 29 | loss |
  | `pdf(1000, 110)` | nan | 1.7577744250901928e-284 | inf | 891 | loss |
  | `pdf(171, 171)` | inf | 0.015246507464251494 | inf | 1 | loss |
  | `pdf(172, 172)` | nan | 0.015202164685508059 | inf | 1 | loss |
  | `pdf(200, 200)` | nan | 0.014098863842960411 | inf | 1 | loss |
  | `pdf(10, 200)` | 0.0 | 5.756606462848521e-179 | inf | 189 | loss |
  | `pdf(1e6, 1e6)` | nan | (normal; all 22 bulk points of a = 1e6 are nan) | inf | 1 | loss |

  The harness flags 198 pdf points, all of them loss except the F8 ones, at
  a = 150, 200, 1e3 and 1e6.
- **Fix direction:** evaluate in log space,
  `0.5*np.exp(sc.xlogy(a-1, ax) - ax - sc.gammaln(a))` (or
  `0.5*stats.gamma._pdf(ax, a)`). For large a this leaves gamma's F2
  (~max(|(a-1) log x|, x, gammaln a) ulp), so a Lanczos/`log1pmx` form is
  better (reports/gamma.md F2).

### F2: pdf loses everything where exp(-|x|) is subnormal

- **Code:** `_pdf`, line 1800: `... * np.exp(-ax)`
- **Mechanism:** pattern 4. For |x| > 708.4, `np.exp(-ax)` is subnormal
  and has fewer than 53 significant bits. At 745 it is 0.0 (numpy gives
  `exp(-745.0) = 0.0`). The prefactor `|x|**(a-1)/(2 Gamma(a))` is large
  (1e21 for a = 10.5, |x| = 745), so the product is a normal double that carries
  the relative error of the subnormal factor.
- **Regime:** |x| > 708.4 wherever the true pdf is still normal. For
  a = 10.5 that is 708.4 < |x| <~ 755: the error grows from 1e4 ulp at 720 to
  1e15 ulp at 744, and the result is 0.0 from 745 on. The window widens with a
  because `|x|**(a-1)/Gamma(a)` is larger. At a = 100 the pdf is 0.0 at
  |x| = 800 (true 5e-217), up to where F1 takes over (`|x|**99` overflows at
  |x| ~ 1300).
- **Reproducer:** `python accuracy.py refs/dgamma_ref.py --at pdf 744 10.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(720, 10.5)` | 1.2510225643727718e-292 | 1.2510225643764243e-292 | 1.31e+04 | 710 | moderate |
  | `pdf(735, 10.5)` | 4.655131697676255e-299 | 4.654991793469717e-299 | 1.4e+11 | 726 | severe |
  | `pdf(744, 10.5)` | 8.305981079922254e-303 | 6.448842182765746e-303 | 1.3e+15 | 734 | severe |
  | `pdf(745, 10.5)` | 0.0 | 2.402862780803914e-303 | inf | 736 | loss |
  | `pdf(-745, 10.5)` | 0.0 | 2.402862780803914e-303 | inf | 736 | loss |
  | `pdf(750, 10.5)` | 0.0 | 1.7252576907331347e-305 | inf | 740 | loss |
  | `pdf(720, 103)` | 2.9643704112492105e-184 | 2.964370411257865e-184 | 1.31e+04 | 618 | moderate |
  | `pdf(750, 105)` | 0.0 | 9.36880512132602e-194 | inf | 646 | loss |
  | `pdf(800, 100)` | 0.0 | 5.003680134991525e-217 | inf | 701 | loss |
- **Fix direction:** same as F1: a single `exp` of the combined exponent.

### F3: ppf/isf near q = 1/2 for large a inherit gammaincinv's error

- **Code:** `_ppf`, lines 1820-1822 (`sc.gammaincinv(a, 2*q - 1)` /
  `-sc.gammainccinv(a, 2*q)`); `_isf`, lines 1825-1827.
- **Mechanism:** pattern 10. `2q - 1` and `2q` are exact, so the
  distribution code is correct. Near q = 1/2 the target of `gammaincinv`
  is small (here 2e-8), and its root lies in the window
  `a - 15 sqrt(a) < y < a - 4.5 sqrt(a)`. There cephes `igam` returns its
  2000-term partial sum, and the inverse converges to the wrong root
  (reports/gamma.md F1/F3). dgamma does not dilute this error, because the
  quantile is the gamma quantile itself.
- **Regime:** a >~ 2e5 and |q - 1/2| roughly in [1e-51, 1e-6] at a = 1e6
  (the same band as gamma's ppf(2|q - 1/2|)); symmetric in q; isf = mirror.
- **Reproducer:** `python accuracy.py refs/dgamma_ref.py --at ppf 0.50000001 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.50000001, 1e6)` | 994518.8614947301 | 994518.8611986521 | 1.3e+06 | 8.85e+03 | severe |
  | `ppf(0.49999999, 1e6)` | -994518.8614937534 | -994518.8611976753 | 1.3e+06 | 8.85e+03 | severe |
  | `isf(0.49999999, 1e6)` | 994518.8614937534 | 994518.8611976753 | 1.3e+06 | 8.85e+03 | severe |
  | `ppf(0.500000005, 1e6)` | 994398.1600137133 | 994398.1597850738 | 1.0e+06 | 1.73e+04 | severe |

- **Fix direction:** none inside `dgamma_gen`; fix `scipy.special.gammainc`
  (gamma report F1).

### F4: logpdf cancels terms of size a*log(a)

- **Code:** `_logpdf`, line 1804:
  `sc.xlogy(a - 1.0, ax) - ax - np.log(2) - sc.gammaln(a)`
- **Mechanism:** pattern 5, identical to reports/gamma.md F4. Near |x| ~ a
  the terms are ~1.4e7 at a = 1e6 and each rounds to ~3.7e-9 absolute, while
  the result is ~ -8.5. The extra `- np.log(2)` does not change this.
- **Regime:** a >~ 1e3 and |x| within a few sqrt(a) of a. The error is
  925 ulp (minor) at a = 1e3 and up to 4.3e5 ulp at a = 1e6. It shrinks
  (1.5e3 ulp) at |x| = 0.9a and 1.1a, where |logpdf| is large.
- **Reproducer:** `python accuracy.py refs/dgamma_ref.py --at logpdf 1000000 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(995000, 1e6)` | -21.056652076542377 | -21.056652078538587 | 4.3e+05 | 237 | moderate |
  | `logpdf(-1003000, 1e6)` | -13.0138567853719 | -13.013856786581075 | 4.2e+05 | 231 | moderate |
  | `logpdf(1000000, 1e6)` | -8.519841076806188 | -8.519841076080088 | 3.8e+05 | 0.117 | moderate |
  | `logpdf(1100000, 1e6)` | -4698.4353469293565 | -4698.4353469310245 | 1.6e+03 | 21.3 | moderate |
  | `logpdf(999, 1000)` | -5.065546519835152 | -5.065546519836192 | 925 | 0 | minor |

- **Fix direction:** Lanczos-scaled / `log1pmx` form of the gamma logpdf
  (reports/gamma.md F2/F4), plus `- log(2)`.

### F5: logpdf and pdf fail for subnormal shapes

- **Code:** `_logpdf`, line 1804 (`sc.gammaln(a)`); `_pdf`, line 1800
  (`sc.gamma(a)`).
- **Mechanism:** patterns 3 and 10. For a < ~5.6e-309, `Gamma(a) ~ 1/a`
  overflows, so `sc.gamma(a) = inf` and the pdf is 0.0. `sc.gammaln(a)` also
  returns inf (measured: `gammaln(1e-309) = inf`, true 711.50; correct for
  a >= 2e-308). So logpdf = -inf. The shape domain is (0, inf), so
  these a are valid inputs.
- **Regime:** 0 < a < ~5.6e-309, every x (pdf only where the true value is
  normal, e.g. |x| < ~1e-2).
- **Reproducer:** `python accuracy.py refs/dgamma_ref.py --at logpdf 1.0 1e-310`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.0, 1e-310)` | -inf | -715.4945260087142 | inf | 0.0028 | loss |
  | `logpdf(-1e-300, 1e-310)` | -inf | -23.718998110500404 | inf | 0.0422 | loss |
  | `pdf(1e-300, 1e-310)` | 0.0 | 4.9999999999999847e-11 | inf | 1 | loss |
  | `pdf(1e-100, 1e-310)` | 0.0 | 4.999999999999985e-211 | inf | 1 | loss |

- **Fix direction:** in `sc.gammaln`, return `-log(a)` (or `-log(a) -
  euler*a`) for tiny a instead of `log(gamma(a))`. In dgamma, use log space
  (F1 fix) with `gammaln`.

### F6: ppf/isf for a << 1 amplify gammaincinv's exponent rounding

- **Code:** `_ppf`/`_isf`, lines 1821/1826 (`sc.gammaincinv(a, 2*q - 1)`).
- **Mechanism:** reports/gamma.md F5. For a << 1 the quantile is
  `exp((log p + gammaln(a+1))/a)` with an exponent of size |log x| ~ 225-580,
  and each half-ulp of that exponent costs ~|log x| ulp. dgamma maps
  every q in (0.5, ~0.95) to such tiny gamma quantiles.
- **Regime:** a <~ 1e-2, |q - 1/2| such that the result is a tiny normal
  double (here 1e-253 ... 1e-98).
- **Reproducer:** `python accuracy.py refs/dgamma_ref.py --at ppf 0.9 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9, 0.001)` | 6.912934354184663e-98 | 6.912934354189132e-98 | 2.91e+03 | 2.25e+03 | moderate |
  | `ppf(0.21999999999999997, 0.001)` | -8.663680881031975e-253 | -8.66368088103506e-253 | 1.6e+03 | 786 | moderate |
  | `isf(0.9, 0.001)` | -6.912934354184663e-98 | -6.912934354189132e-98 | 2.91e+03 | 2.25e+03 | moderate |

  (The errors are only ~1.3-2x the conditioning; cond (shapes) 223-580.)
- **Fix direction:** in `scipy.special` `igami`, final Newton step on log P
  (gamma report F5).

### F7: cdf/sf inherit gammainc/gammaincc errors, diluted by 0.5 + 0.5*P

- **Code:** `_cdf`, lines 1807-1809: `0.5 + 0.5*sc.gammainc(a, x)` /
  `0.5*sc.gammaincc(a, -x)`; `_sf`, lines 1812-1814 (mirror).
- **Mechanism:** there is no cancellation in the dgamma code: the x > 0 branch is >= 0.5,
  and the x <= 0 branch uses the complementary function. The errors come from
  the special functions. (1) At a = 1e6, x = 995000 = a - 5 sqrt(a),
  `gammainc` returns the unconverged series (gamma F1, 4.3e-6 relative on
  P = 2.7e-7). Added to 0.5 this is 1.2e-12 absolute, i.e. 5.4e3 ulp. (2) At
  a = 1e3, |x| = 2a, `gammaincc` rounds its `exp(a log x - x - lgam a)`
  prefactor (gamma F6); here it enters undiluted (sf(2000, 1000) is
  gammaincc/2).
- **Regime:** (1) a >~ 2e5, a - 15 sqrt(a) < |x| < a - 4.5 sqrt(a);
  (2) a >~ 1e2 with |x - a| > 0.4a (cdf for x < 0, sf for x > 0).
- **Reproducer:** `python accuracy.py refs/dgamma_ref.py --at cdf 995000 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(995000, 1e6)` | 0.5000001374784223 | 0.500000137479018 | 5.36e+03 | 0.00143 | moderate |
  | `sf(995000, 1e6)` | 0.4999998625215777 | 0.49999986252098205 | 5.37e+03 | 0.00143 | moderate |
  | `sf(2000, 1000)` | 3.423674729808879e-136 | 3.4236747298073765e-136 | 1.98e+03 | 1e+03 | moderate |
  | `cdf(-400, 200)` | 3.104755921497587e-29 | 3.1047559214980446e-29 | 663 | 202 | minor |

- **Fix direction:** none inside `dgamma_gen`; fix in `scipy.special`
  (gamma report F1, F6).

### F8: pdf rounding of a - 1 amplified by |log x| near 0

- **Code:** `_pdf`, line 1800: `ax**(a-1.0)`
- **Mechanism:** pattern 6. `a - 1.0` is rounded when a has bits below
  2^-53 (a = 0.1: 1 - 0.1 loses 3 bits, error ~0.14 ulp of 0.9). The power
  multiplies that relative error by |(a-1) log x| (~620 at x = 1e-300).
  logpdf is not affected in relative terms.
- **Regime:** a < 1 (or generally a with low-order bits), |x| << 1; error
  ~ 0.14*|log x| ulp for a = 0.1. a = 0.001 happens to round favourably
  (3.3 ulp at x = 1e-300).
- **Reproducer:** `python accuracy.py refs/dgamma_ref.py --at pdf 1e-300 0.1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-300, 0.1)` | 5.2556850305589695e+268 | 5.255685030558869e+268 | 86.1 | 0.9 | minor |
  | `pdf(1e-150, 0.1)` | 5.255685030558929e+133 | 5.255685030558879e+133 | 43.2 | 0.9 | minor |
  | `pdf(1e-40, 0.1)` | 5.2556850305589e+34 | 5.255685030558887e+34 | 11.1 | 0.9 | minor |

- **Fix direction:** `exp(a*log(x)) / x` (or `x**a / x`) avoids rounding
  `a - 1`. The `xlogy(a-1, x)` form of the F1 fix has the same weakness.

## RuntimeWarnings

- pdf: `overflow encountered in power`, `invalid value encountered in
  multiply`, `underflow encountered in exp/multiply/power` in all 9
  parameter sets. Overflow and invalid warnings go with the wrong results
  of F1. The underflow warnings go with F2 or with points whose true pdf
  underflows. `divide by zero encountered in power` comes from x = 0 with
  a < 1 (true value inf, so correct).
- cdf, sf: `underflow encountered in multiply` at a = 1.10233 and a = 1.
  It comes from `0.5*sc.gammaincc(a, |x|)` on grid points with subnormal
  results. The harness evaluates a whole grid per call, so the warning is attributed to every
  point of the call, including accurate ones (spurious on those points).
- logpdf, ppf, isf: no warnings.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf, sf | full grid (+-1e-300 ... +-1e300, x = 0, both sides of x = +-a, +-1) for a = 0.001, 0.1, 1, 1.10233, 10.5, 150 | 14.6 ulp (cdf(-2.15, 1.10233), minor, no clear mechanism) |
| cdf, sf | tiny a (single points) | 392 ulp at a = 1e-300 (`cdf(-1, 1e-300)`), 177 ulp at 1e-100, 8.5 ulp at 1e-20: minor, inside `sc.gammaincc` for tiny a, mechanism unclear |
| logpdf | a <= 200, full grid | <= 8 ulp beyond conditioning except F4 (a >= 1e3) |
| pdf | a = 0.001, 1, 1.10233, 10.5 for \|x\| < 708 | within conditioning except F2/F8 |
| ppf, isf | q in [1e-300, 1 - 3e-16] and q = 0.5 +- (1.1e-16 ... 1e-4) for a = 0.1, 1, 1.10233, 10.5, 150, 200, 1e3; tails of a = 1e6 | 18.8 ulp (ppf(0.01, 0.1)), 13.5 ulp (a = 1.10233); `2q - 1`/`2q` are exact, no cancellation in the branch logic |
| ppf, isf | low and high tail (q -> 0, q -> 1), all sets | 8.8 ulp |

## Suspected, not confirmed

- `sc.gammaincc(1e-310, 1.0)` returns -5.77e-311 (a negative probability);
  `dgamma.cdf(-1, 1e-310)` = -2.886e-311 against the true 1.097e-311. The
  true value is subnormal, so the harness does not measure it. It looks like a
  sign/cancellation problem in `igamc` for subnormal a (the -5.77e-311 equals
  -euler*a). Not reported as a finding.
- a in [1e7, 1e10]: gamma's F1 grows to 0.375 relative at a = 1e8, so
  dgamma's ppf/isf near q = 1/2 should be worse there. That range was not
  measured (mpmath's `gammainc` does not converge there; see reports/gamma.md).

The reference self-check reported `FAIL: integral of pdf != cdf difference`
on [-ppf(0.9), 0] and [0, ppf(0.9)] for a = 0.001 and 0.1 only. These
intervals end at the integrable singularity `|x|**(a-1)` at x = 0, and the
harness's 30-digit `mp.quad` misses it (relative error -0.903 at a = 0.001).
`scratch/dgamma/check_integral.py` repeats the integral after the
substitution x = t**(1/a), which removes the singularity, and matches
the reference cdf difference to 1e-41 on both sides. The reference is
correct; all other self-checks (cdf + sf = 1, log consistency, quantile
round trips, precision stability) pass.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | `log(cdf)` below the median (0), `log1p(-sf)` above | x <= 0: `log(0.5) + log Q(a, -x)` with a log-space `igamc` (continued fraction: `-y + a log y - gammaln(a) + log(CF)`); x > 0: `log1p(-0.5*gammaincc(a, x))` (already what the default does) | cdf underflows for -x - (a-1)log(-x) + gammaln(a) >~ 745: `logcdf(-1000, 1.10233)` returns -inf, true -999.93529932564399 (loss) |
| logsf | `log(sf)` above the median, `log1p(-cdf)` below | mirror of logcdf: `log(0.5) + log Q(a, x)` for x >= 0 | `logsf(1000, 1.10233)` returns -inf, true -999.93529932564399 (loss) |

Near the median the defaults are fine: `logcdf(40, 1.10233)` =
-3.268183968862209e-18 vs -3.2681839688622186e-18.

## Shared code

- **`sc.gammainc` / `sc.gammaincc` / `sc.gammaincinv` / `sc.gammainccinv`**:
  F3, F6 and F7 come from the special functions already documented in
  reports/gamma.md (F1 unconverged `igam_series` for a >~ 2e5; F3 inverse
  inherits it; F5 `igami` for a << 1; F6 `igam_fac` prefactor). dgamma adds
  nothing new there. ppf/isf hit these errors undiluted; cdf/sf see only
  `0.5*error`, relative to 0.5.
- **`sc.gammaln` for subnormal a** (new, F5): `gammaln(a)` returns inf for
  a <~ 5.6e-309 (true ~ -log a ~ 709-744), apparently computed as
  `log(gamma(a))` with an overflowing gamma. This affects every
  distribution with `gammaln(shape)` in its logpdf when the shape is subnormal
  (gamma, invgamma, gengamma, loggamma, ...).
- **`sc.gammaincc` for subnormal a** (suspected): negative value
  `gammaincc(1e-310, 1) = -5.77e-311`; see "Suspected, not confirmed".
- F1, F2, F4 and F8 are in `dgamma_gen._pdf`/`_logpdf` themselves. `gamma._pdf`
  uses `exp(logpdf)` and does not have F1/F2.

## Existing GitHub issues

- none found specific to dgamma (web search of scipy/scipy issues for "dgamma", dgamma pdf nan/inf at large shape, and `gammaln` with subnormal arguments; `gh` was not available)
- #26471 (open): BUG: inaccuracy in special.gammainc for certain large parameters - relates to F3 and F7 (inherited from gamma F1/F3, see reports/gamma.md)
- #17832 (open): Tracking issue: custom methods for continuous distributions - relates to the inherited `logcdf`/`logsf` (generic `log(cdf)`/`log(sf)`)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(-150, 150)` | inf | 0.016277704728418274 | verified | all 8 table calls reproduced (inf/nan/0.0 vs normal values; `pdf(1e6, 1e6)` true 1.9947112357812202e-4); ref = exp((a-1)log\|x\| - \|x\| - log 2 - loggamma(a)); mechanism matches line 1800 (`150.**149` overflows, `gamma(172) = inf`) |
| F2 | `pdf(744, 10.5)` | 8.305981079922254e-303 | 6.448842182765746e-303 | verified | 1.3e15 ulp; all 9 table calls reproduced (720: 1.31e4, 735: 1.35e11, 745/750/-745 and a = 100/105: 0.0); `np.exp(-744.) = 1e-323` (subnormal), `np.exp(-745.) = 0.0` |
| F3 | `ppf(0.50000001, 1e6)` | 994518.8614947301 | 994518.8611986521 | verified | 1.34e6 ulp; ref = root of own cdf (Kummer series P = x^a e^-x/Gamma(a+1) 1F1(1; a+1; x), not `mp.gammainc`), residual 0; the other 3 calls reproduce (1.34e6, 1.34e6, 1.04e6 ulp); mechanism confirmed: `sc.gammainc(1e6, scipy root) = 2.00000001005e-8` = target, true P there is 2.0000034e-8 |
| F4 | `logpdf(995000, 1e6)` | -21.056652076542377 | -21.056652078538587 | verified | 4.27e5 ulp; other calls 4.18e5, 3.84e5, 1.6e3 ulp as reported; mechanism matches line 1804 |
| F5 | `logpdf(1.0, 1e-310)` | -inf | -715.4945260087142 | verified | all 4 calls reproduced; `sc.gammaln(1e-309) = inf`, `sc.gammaln(2e-308) = 708.503` (correct), `sc.gamma(1e-310) = inf` |
| F7 | `cdf(995000, 1e6)` | 0.5000001374784223 | 0.500000137479018 | verified | 5.37e3 ulp (Kummer-series cdf); `sc.gammainc(1e6, 995000)` is -4.33e-6 relative off; `sf(2000, 1000)` 1.98e3 ulp and `cdf(-400, 200)` 663 ulp also reproduced. For `sf(2000, 1000)` the 1 - P route cancels ~136 digits, so it was checked at 200/300 digits and with upper `mp.gammainc` at 100/200 (all agree to 40 digits) |

Verifier references: `scratch/dgamma/verify/verify.py` and `scratch/dgamma/verify/sf_tail.py`, evaluated at 100 and 200 digits (agree to 30 digits for every call except `sf(2000, 1000)` at 100 digits, handled as noted). `refs/dgamma_ref.py` was not needed: there were no disagreements.

Verified by dist-accuracy-verifier: 6 of 6 verified.
