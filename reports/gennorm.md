# gennorm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `gennorm_gen`, lines 11520-11617
- **Implemented:** pdf (11570-11571), logpdf (11573-11574), cdf (11576-11579), sf (11586-11587, = cdf(-x)), ppf (11581-11584), isf (11589-11590, = -ppf(q))
- **Inherited (not measured):** logcdf, logsf (the `rv_continuous` defaults: `log(cdf)` below the median, `log1p(-sf)` above it, and mirrored for logsf)
- **Related:** `laplace` (beta = 1), `norm` (beta = 2, scale 1/sqrt(2)), `gamma` (cdf/ppf are the gamma P/Q of y = |x|**beta with a = 1/beta, see reports/gamma.md); helpers: none; special functions: `sc.gammaln`, `sc.gammaincc`, `sc.gammainccinv` (`special.gamma` is not used by the measured methods)
- **Reference:** `refs/gennorm_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (1.2988442399460265), (0.006), (0.05), (0.5), (1), (2), (20), (1000), (1e6), (1e100), (1e300); single points at beta = 100
- **Worst finding:** loss in ppf/isf (0.0 for normal quantiles, beta >~ 22); severe in cdf/sf (0.5 returned for 0.7)

## Summary

pdf, logpdf, cdf and sf are accurate for moderate beta (the Laplace/normal
range and the inventory example), including both tails. Everything that
forms `abs(x)**beta` for large beta breaks: inside |x| < exp(-745/beta)
the power underflows to 0 although the distribution there is close to
uniform on (-1, 1), so cdf/sf return exactly 0.5 (1.3e15 ulp at
`cdf(0.4, 1000)`, true 0.70), and ppf/isf return 0.0 for normal quantiles
because `gammainccinv` returns 0 or a subnormal before `**(1/beta)`
(`ppf(0.6, 1000)` = 0 vs 0.1999; 2.7e7 ulp in the subnormal band). The
affected interval grows with beta: |x| < 5.8e-4 at beta = 100, < 0.475 at
beta = 1e3, < 0.99926 at beta = 1e6 and all of (-1, 1) at beta = 1e300.
Smaller, minor issues: logpdf/pdf lose up to ~1e3 ulp from cancellation of
`log(0.5*beta) - gammaln(1/beta)` at huge beta and from `exp` of a large
logpdf at small beta.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf, isf | beta >~ 22, result \|x\| < ~exp(-745/beta) (q within ~f(0)exp(-745/beta) of 1/2: q in (0.26, 0.74) at beta = 1e3, every q at beta = 1e300) | `gammainccinv` result underflows to 0 or a subnormal, then `**(1/beta)` (pattern 4) | 0.0 for 0.1999 (inf ulp); 2.7e7 ulp | loss |
| F2 | cdf, sf | beta >~ 22, \|x\| < exp(-745/beta) | `abs(x)**beta` underflows to 0, `gammaincc(a, 0) = 1`, result exactly 0.5 (pattern 4) | 0.5 for 0.2999 (3.0e15 ulp) | severe |
| F3 | logpdf, pdf | beta >~ 1e100 | `log(0.5*beta) - gammaln(1/beta)` cancels (pattern 5) | 357 ulp (logpdf), 248 ulp (pdf) | minor |
| F4 | pdf | small beta (logpdf ~ -gammaln(1/beta) ~ -700, beta = 0.006) | `exp` of a large logpdf amplifies its absolute rounding error | 975 ulp | minor |

### F1: ppf and isf return 0 for large beta

- **Code:** `_ppf`, line 11584: `c * sc.gammainccinv(1.0/beta, (1.0 + c) - 2.0*c*x)**(1.0/beta)`; `_isf` = `-_ppf`.
- **Mechanism:** pattern 4. The quantile is computed as `z**(1/beta)` with
  `z = P^-1(1/beta, |2q - 1|) = |x|**beta`. For large beta, `z` is
  `|x|**beta`, which underflows: for |x| < exp(-745/beta) `gammainccinv`
  returns 0 and the result is 0.0; for exp(-745/beta) < |x| <
  exp(-708/beta) it returns a subnormal with few significant bits, and
  `**(1/beta)` keeps that relative error (2.7e7 ulp at beta = 1e3,
  q = 0.74). `gammainccinv` itself is correct here (the true z is not a
  normal double); the formula needs z, which is not representable.
- **Regime:** beta >~ 22 (below that the affected q are within an ulp of
  1/2, where ppf is ill-conditioned anyway: `ppf(0.5+1.1e-16, 20)` is 6.3e3
  ulp off but cond is 4.5e15). The 0.0 band is q in
  (0.5 - d, 0.5 + d) with d ~ 0.5*P(1/beta, exp(-745)) ~ exp(-745/beta)/2:
  d = 2.9e-4 at beta = 100, 0.24 at beta = 1e3, 0.4996 at beta = 1e6, and
  the whole of (0, 1) at beta = 1e100 and 1e300.
- **Reproducer:** `python accuracy.py refs/gennorm_ref.py --at ppf 0.6 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.6, 1000)` | 0.0 | 0.19988475449691906 | inf | 6 | loss |
  | `isf(0.6, 1000)` | 0.0 | -0.19988475449691906 | inf | 6 | loss |
  | `ppf(0.5001, 100)` | 0.0 | 0.0001988651702382793 | inf | 5e+03 | loss |
  | `ppf(0.7, 1e6)` | 0.0 | 0.3999997691141296 | inf | 3.5 | loss |
  | `ppf(0.99, 1e300)` | 0.0 | 0.98 | inf | 2.02 | loss |
  | `ppf(0.74, 1000)` | 0.4797234079228047 | 0.47972341079260583 | 2.7e+07 | 3.08 | severe |

- **Fix direction:** for small a = 1/beta work with log z: e.g. for
  |2q - 1| small, x ~ (|2q - 1| Gamma(1 + a))**a (z**a never forms z), or
  invert in log space (`exp(a * log(z))` with log z from a log-space inverse);
  as beta -> inf the quantile tends to the uniform one, 2q - 1.

### F2: cdf and sf return 0.5 inside (-exp(-745/beta), exp(-745/beta))

- **Code:** `_cdf`, line 11579: `(0.5 + c) - c * sc.gammaincc(1.0/beta, abs(x)**beta)`; `_sf` = `_cdf(-x)` (line 11587).
- **Mechanism:** pattern 4. `abs(x)**beta` underflows to 0 for
  beta*log|x| < -745, so `gammaincc(a, 0) = 1` and the result is exactly
  0.5. The true value is 0.5 + sign(x)*P(1/beta, |x|**beta)/2, and
  `P(a, y) ~ y**a / Gamma(1 + a) = |x| / Gamma(1 + 1/beta)` does not
  underflow: for large beta the distribution is close to uniform on (-1, 1)
  and cdf(x) ~ (1 + x)/2. In the subnormal band (-745 < beta log|x| < -708)
  the precision loss of y is divided by beta in `y**a`, so that band is
  accurate.
- **Regime:** |x| < exp(-745/beta), visible once f(0)*exp(-745/beta)
  exceeds half an ulp of 0.5, i.e. beta >~ 22. Interval: |x| < 5.8e-4 at
  beta = 100, < 0.475 at beta = 1e3, < 0.99926 at beta = 1e6, all |x| < 1
  for beta = 1e100, 1e300. Not loss by the tool's definition (relative
  error < 1), but up to 3e15 ulp.
- **Reproducer:** `python accuracy.py refs/gennorm_ref.py --at cdf -0.4 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-0.4, 1000)` | 0.5 | 0.2998846880510012 | 3.0e+15 | 0.667 | severe |
  | `cdf(0.4, 1000)` | 0.5 | 0.7001153119489988 | 1.3e+15 | 0.286 | severe |
  | `sf(-0.4, 1000)` | 0.5 | 0.7001153119489988 | 1.3e+15 | 0.286 | severe |
  | `cdf(0.99, 1e6)` | 0.5 | 0.9950002857214295 | 2.2e+15 | 0.497 | severe |
  | `cdf(0.5, 1e300)` | 0.5 | 0.75 | 1.5e+15 | 0.333 | severe |
  | `cdf(5e-4, 100)` | 0.5 | 0.5002514266321251 | 2.3e+12 | 0.000503 | severe |
  | `cdf(1e-4, 100)` | 0.5 | 0.500050285326425 | 4.5e+11 | 0.000101 | severe |

  Outside the band the values are fine: `cdf(-0.9, 1000)` = 0.04974054811475274, 0.63 ulp.
- **Fix direction:** for small a use P(a, y) = exp(a*log y) * (series) with
  log y = beta*log|x| (no underflow), e.g. `sc.gammainc` on y only when y is
  normal and the leading term `|x|/Gamma(1 + 1/beta)` times
  `(1 - a*y/(a+1) + ...)` (or a log-space incomplete gamma) otherwise.

### F3: logpdf cancels for huge beta

- **Code:** `_logpdf`, line 11574: `np.log(0.5*beta) - sc.gammaln(1.0/beta) - abs(x)**beta`; `_pdf` = `exp(_logpdf)` (line 11571).
- **Mechanism:** pattern 5. For small a = 1/beta, `gammaln(a) ~ -log(a) = log(beta)`,
  so both terms are ~log(beta) (690.1 and 690.8 at beta = 1e300) while the
  result is ~-log 2. Their rounding errors (~log(beta)*eps each) are
  ~log(beta)/log(2) ulp of the result; pdf inherits the absolute error.
- **Regime:** beta >~ 1e6 (3 ulp at 1e6, 12 ulp at 1e100, 357 ulp at 1e300), |x| < 1.
- **Reproducer:** `python accuracy.py refs/gennorm_ref.py --at logpdf 0.5 1e300`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.5, 1e300)` | -0.6931471805598903 | -0.6931471805599453 | 357 | 0 | minor |
  | `pdf(0.5, 1e300)` | 0.5000000000000275 | 0.5 | 248 | 0 | minor |

- **Fix direction:** `log(0.5) + log(beta) - gammaln(1/beta)` combined as
  `log(0.5) - log(Gamma(1 + 1/beta))`, i.e. `-log(2) - sc.gammaln(1 + 1/beta)` (no large terms; gammaln(1+a) ~ -0.577a).

### F4: pdf amplifies the large normalising constant for small beta

- **Code:** `_pdf`, line 11571: `np.exp(self._logpdf(x, beta))`.
- **Mechanism:** for small beta, `-gammaln(1/beta)` is large (-683 at
  beta = 0.006); its absolute rounding error (~|logpdf|*eps/2) becomes the
  relative error of `exp`. cond (arg) is only beta*|x|**beta.
- **Regime:** small beta, error ~|logpdf| ulp; up to ~1e3 ulp
  at the smallest beta for which the pdf is still normal (~0.0057).
- **Reproducer:** `python accuracy.py refs/gennorm_ref.py --at pdf 1 0.006`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1, 0.006)` | 6.7411312189827385e-301 | 6.741131218984198e-301 | 975 | 0.006 | minor |

  (cond (shapes) is 853 here.)
- **Fix direction:** inherent to `exp` of a large argument unless the
  constant is carried in extended precision; low priority.

## RuntimeWarnings

`underflow encountered in power` from `abs(x)**beta` in cdf/sf (F2) and
in logpdf/pdf for large beta. In cdf/sf it coincides with the wrong 0.5;
in logpdf/pdf it is spurious (`exp(-0)` is the right limit, e.g.
`logpdf(0.5, 1e300)` warns and is only F3's 357 ulp off).
`underflow encountered in exp` in pdf where the true pdf is subnormal or 0
(`pdf(1e300, 0.006)`, true 7e-328), harmless.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf, sf | beta = 1.2988, 0.05, 0.5, 2: bulk and both tails (`sf(3e56, 0.05)` 119 ulp minor, cond 32; `cdf(-8e56, 0.05)` 33.5 ulp, within cond 34) | 119 ulp |
| cdf | beta = 1e3 outside F2's band (`cdf(-0.9, 1000)`) | 0.63 ulp |
| ppf, isf | beta = 1.2988, 0.05: tails and near 1/2 (`isf(1e-200, 1.2988)` 1.1 ulp, `ppf(1e-300, 0.05)` 28 ulp minor at cond 0.026, `ppf(0.9, 0.05)` 25 ulp within cond 27) | 28 ulp |
| ppf | beta = 1e3, q outside F1's band (`ppf(0.9, 1000)`) | 0 ulp |
| logpdf | all beta below ~1e6 | see F3 |

The full grid run (`scratch/gennorm/run1.txt`) did not complete its
condition-number phase within the time limit (1472 flagged points; ppf/isf
condition numbers at beta >= 1e3 need slow quantile references), so the
table above is from single-point `--at` evaluations; grid maxima are not
available (time limit). The run was stopped at the time limit after 50 minutes
in that phase. Self-checks had already passed. Reference values are cached
in `.cache/dist-accuracy/`, so a rerun only repeats the condition numbers.
Restricting it with `--params "1.2988442399460265;0.05;2;1000"` should finish.

## Suspected, not confirmed

- Small beta, ppf: `z**(1/beta)` multiplies the rounding of
  `gammainccinv` by 1/beta (pattern 6). At beta = 0.05 (factor 20) the
  measured errors stay within conditioning (28 ulp at cond 0.026 is the
  worst seen); at beta < ~0.006 no quantile is a finite double except
  q = 1/2, so the regime cannot be reached. Not confirmed beyond minor.
- Small beta, cdf/sf tail: rounding of y = |x|**beta amplified by ~y in
  Q(a, y), excess ~1/(2 beta) ulp over cond = beta*y. Seen as 119 ulp in
  `sf(3e56, 0.05)` (minor); not mapped on a grid (time limit).
- The `scipy.special.gammainc` unconverged-series defect (INDEX item 7) needs
  a = 1/beta >~ 2e5 and y near a, but y = |x|**beta <= exp(709.78/a) ~ 1,
  so it cannot be reached through gennorm; the small-a `gammaincinv`
  amplification (gamma F5) is attenuated here by `**(1/beta) = **a`.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | `log(cdf)` below the median | `log(0.5) + log Q(1/beta, |x|**beta)` with a log-space upper incomplete gamma (leading term `(a-1)*log y - y - gammaln(a)` plus log of the continued fraction); for beta = 2 `log_ndtr` | cdf underflows for |x|**beta >~ 745: `logcdf(-30, 2)` returns -inf, true -904.66726429120382; `logcdf(-1000, 1.2988)` -inf vs -7883.0576541283654 |
| logsf | `log(sf)` above the median | same as logcdf by symmetry, `logsf(x) = logcdf(-x)` | `logsf(30, 2)` -inf (same values as above) |

## Shared code

- `scipy.special.gammaincc` / `gammainccinv`: no defect of the special
  functions found. F1/F2 are gennorm's own formulation (forming the
  underflowing y = |x|**beta and z = |x|**beta), not errors of the
  special functions. The `sc.gammainc` unconverged-series issue
  (INDEX item 7, reports/gamma.md F1/F3) is unreachable through gennorm
  (see "Suspected, not confirmed").
- The underflow pattern (forming `|x|**beta` and then taking `**(1/beta)`)
  may exist in other power-transformed gamma distributions (`gengamma`,
  `weibull`-type with c large); not checked here.

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

Independent reference (`scratch/gennorm/verify/verify.py`): cdf = 1/2 + sign(x) * `mp.quad` of the docstring pdf over [0, |x|]; ppf by `mp.findroot` on that cdf (residual |cdf(x_ref) - q| = 0 at 200 digits), isf = -ppf by symmetry. Values at 100 and 200 digits agree to > 30 digits at every point. F3 and F4 are minor and not selected.

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(0.6, 1000)` | 0.0 | 0.19988475449691905 | verified | loss |
| F1 | `isf(0.6, 1000)` | 0.0 | -0.19988475449691905 | verified | loss |
| F1 | `ppf(0.5001, 100)` | 0.0 | 0.00019886517023827931 | verified | loss |
| F1 | `ppf(0.7, 1e6)` | 0.0 | 0.39999976911412957 | verified | loss |
| F1 | `ppf(0.99, 1e300)` | 0.0 | 0.98 (0.97999999999999998) | verified | loss |
| F1 | `ppf(0.74, 1000)` | 0.4797234079228047 | 0.47972341079260581 | verified | 2.69e7 ulp, severe; `gammainccinv(1e-3, 0.48)` = 9.793e-320 (subnormal), `gammainccinv(1e-3, 0.8)` = 0.0: mechanism at line 11584 confirmed |
| F2 | `cdf(-0.4, 1000)` | 0.5 | 0.29988468805100121 | verified | 3.01e15 ulp |
| F2 | `cdf(0.4, 1000)` | 0.5 | 0.70011531194899879 | verified | 1.29e15 ulp |
| F2 | `sf(-0.4, 1000)` | 0.5 | 0.70011531194899879 | verified | 1.29e15 ulp |
| F2 | `cdf(0.99, 1e6)` | 0.5 | 0.99500028572142946 | verified | 2.24e15 ulp |
| F2 | `cdf(0.5, 1e300)` | 0.5 | 0.75 | verified | 1.5e15 ulp |
| F2 | `cdf(5e-4, 100)` | 0.5 | 0.5002514266321251 | verified | 2.26e12 ulp |
| F2 | `cdf(1e-4, 100)` | 0.5 | 0.50005028532642502 | verified | 4.53e11 ulp; `0.4**1000` = 0.0 in double, so `gammaincc(a, 0) = 1` at line 11579: mechanism confirmed |

Verified by dist-accuracy-verifier: 2 of 2 verified.
