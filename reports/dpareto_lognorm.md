# dpareto_lognorm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `dpareto_lognorm_gen`, lines 1837-1969
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf (pdf/cdf/sf are `exp` of the log versions; everything goes through `_logpdf` and `_logcdf`)
- **Inherited (not measured):** ppf (numerical inversion of cdf), isf = ppf(1 - q)
- **Related:** none; helpers: `_R`, `_logR` (class methods built on `norm._logsf`/`norm._logpdf`, i.e. `sc.log_ndtr` and `-x**2/2 - log(sqrt(2 pi))`); special functions: `sc.log_ndtr`, `sc.logsumexp`, `scu._log1mexp`, `np.logaddexp`
- **Reference:** `refs/dpareto_lognorm_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (3, 1.2, 1.5, 2), (0, 0.01, 1, 1), (0, 10, 2, 3), (0, 1, 100, 100), (0, 1, 0.001, 0.001), (500, 0.1, 2, 3), (-2, 0.5, 5, 0.5)  (u, s, a, b)
- **Worst finding:** loss in logsf; severe in pdf, logpdf, cdf, logcdf, sf, logsf

## Summary

For the example parameters the bulk is accurate (pdf at most 7.6 ulp, logpdf 1.9 ulp
in the bulk). Away from the bulk, every method loses accuracy because of one design
choice. `logphi(z)` (about -z**2/2) is added to `logR(y)`, which contains +y**2/2,
and the two cancel to a term that is only linear in z. The relative error of
pdf/cdf/sf (the absolute error of the log versions) is therefore about
max(z**2, y1**2, y2**2)/2 ulp. That is 1e4-1e5 ulp in the far tails for moderate
shapes, up to 4.6e9 ulp for small s (s = 0.01), and about 1e4 ulp in the bulk for
large a, b (= 100). In addition, logsf = log1mexp(logcdf) returns -inf in the right
tail once sf drops below the smallest subnormal (about 4.9e-324), although logsf
itself is around -750. Just before that point, where logcdf is subnormal, it loses
most of its digits. Large |u| adds a minor error from `log(x) - u`.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf, logpdf, cdf, logcdf, sf, logsf | large \|z\| = \|log x - u\|/s (tails, small s) or large a*s, b*s (also in the bulk) | logphi(z) + logR(y) cancel, terms ~ y**2/2 (pattern 5) | 6.6e9 ulp (logpdf), 4.6e9 ulp (pdf) | severe |
| F2 | logsf | right tail, sf < ~2.2e-308 (loss for sf < ~4.9e-324) | log1mexp(logcdf) with logcdf = -sf underflowing (patterns 1, 4) | -inf for -754.3; 1.3e9 ulp | loss |
| F3 | pdf (all methods) | large \|u\| relative to s, bulk | z = (log x - u)/s: rounding of log x (~eps*\|u\|) amplified by 1/s (pattern 5) | 224 ulp | minor |

### F1: cancellation between logphi(z) and logR(y)

- **Code:** `_logR`, line 1898: `self._logPhic(z) - self._logphi(z)`, i.e.
  `log_ndtr(-y) + y**2/2 + log(sqrt(2 pi))`. It is combined with
  `out += self._logphi(z)` (line 1925, i.e. `-z**2/2 - ...`) and
  `np.logaddexp(self._logR(x1), self._logR(x2))` (line 1926) in `_logpdf`. In `_logcdf` the same happens through
  `t2 = self._logphi(z)` (line 1937), `t3`/`t4` (lines 1938-1939) and `t2 + t5` (line 1945).
  pdf, cdf, sf and logsf inherit the error through `exp(...)` / `log1mexp(...)`.
- **Mechanism:** pattern 5. With y1 = a*s - z and y2 = b*s + z, the exact sum
  `logphi(z) + logR(y)` equals `log_ndtr(-y) + (y**2 - z**2)/2`, which is linear in z
  (`a*s*(a*s - 2z)/2` resp. `b*s*(b*s + 2z)/2`). The code forms `-z**2/2` and `+y**2/2`
  separately. Each has a rounding error of about eps*z**2/2, and the error survives the
  cancellation as an absolute error in the log-quantity, i.e. a relative error
  of about max(z**2, y**2)/2 ulp in pdf/cdf/sf. Example: logpdf(1e-100; 0, 0.01, 1, 1)
  has z = -23026 and z**2/2 = 2.65e8. The two terms cancel to -0.693, with an absolute error of 6.6e-8.
  The intermediate values themselves (`log_ndtr`, `logsumexp`) were checked against mpmath and are
  correctly rounded. Only the cancellation loses digits.
- **Regime:** any point with large |z| = |log x - u|/s or large a*s, b*s. The error is
  about z**2/2 ulp: above 1e3 ulp (moderate) for |z| >~ 45, and above 1e6 ulp (severe) for
  |z| >~ 1400, which needs s <~ 0.5 since |log x| <= 745. Large a or b put y1/y2 far
  from 0 even in the bulk: (0, 1, 100, 100) gives 9.8e3 ulp at x = 9.5. Small s
  (0.01) gives 2.3e4 ulp already at x = 10. Count over the grid: 217 severe
  and 869 moderate points across all six methods.
- **Reproducer:** `python accuracy.py refs/dpareto_lognorm_ref.py --at logpdf 1e-100 0 0.01 1 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-290, 0, 0.01, 1, 1)` | -0.6930961608886719 | -0.6930971805599453 | 6.6e+09 | 0 | severe |
  | `logpdf(1e-100, 0, 0.01, 1, 1)` | -0.6930971145629883 | -0.6930971805599453 | 4.3e+08 | 0 | severe |
  | `pdf(1e-290, 0, 0.01, 1, 1)` | 0.5000255104863995 | 0.5000250006250104 | 4.6e+09 | 0 | severe |
  | `cdf(1e-290, 0, 0.01, 1, 1)` | 5.000254049480568e-291 | 5.000250006250105e-291 | 3.6e+09 | 1 | severe |
  | `logcdf(1e+290, 0, 0.01, 1, 1)` | -5.000254049480568e-291 | -5.000250006250104e-291 | 3.6e+09 | 1 | severe |
  | `sf(1e+290, 0, 0.01, 1, 1)` | 5.000254049480568e-291 | 5.000250006250104e-291 | 3.6e+09 | 1 | severe |
  | `logsf(1e+290, 0, 0.01, 1, 1)` | -668.4427733402279 | -668.4427741488332 | 5.4e+06 | 0.0015 | severe |
  | `cdf(1e+120, 500, 0.1, 2, 3)` | 1.5131444453630854e-292 | 1.5131444461014997e-292 | 2.2e+06 | 3 | severe |
  | `pdf(10, 0, 0.01, 1, 1)` | 0.005000250006275993 | 0.005000250006250104 | 2.33e+04 | 2 | moderate |
  | `pdf(9.5, 0, 1, 100, 100)` | 0.003332671709315517 | 0.0033326717093227353 | 9.75e+03 | 3.25 | moderate |
  | `pdf(1e-270, 3, 1.2, 1.5, 2)` | 3.784900150370755e-272 | 3.784900150259388e-272 | 1.3e+05 | 1 | moderate |
  | `sf(1e+200, 3, 1.2, 1.5, 2)` | 2.5992268256729944e-298 | 2.599226825711571e-298 | 6.68e+04 | 1.5 | moderate |
  | `logpdf(1e-170, 500, 0.1, 2, 3)` | -2282.651610046625 | -2282.6516100611816 | 2.87e+04 | 0.000876 | moderate |

  cond (shapes) is large in the small-s cases (332 to 692). In the far tails the
  results are sensitive to s, but s is an exact user input, so this does not excuse
  the error. At the bulk points (x = 10 and x = 9.5) cond (shapes) is only 2.3 and 4.1.
- **Fix direction:** combine analytically,
  `logphi(z) + logR(y1) = log_ndtr(-y1) + a*s*(a*s - 2*z)/2 - ...` (and
  `b*s*(b*s + 2*z)/2` for y2), so `-z**2/2` is never formed. Alternatively use
  `R(t) = sqrt(pi/2)*erfcx(t/sqrt(2))` for t >= 0 together with that exponent.

### F2: logsf = log1mexp(logcdf) fails when sf underflows

- **Code:** `_logsf`, line 1951: `scu._log1mexp(self._logcdf(x, u, s, a, b))`.
- **Mechanism:** patterns 1 and 4. In the right tail logcdf = log(1 - sf) is
  about -sf. `_logcdf` computes it accurately as a tiny number via `logsumexp`
  (line 1946), but once sf < 2.2e-308 that number is subnormal and has few
  significant bits. Once sf < ~4.9e-324 it is -0.0, and `log1mexp(-0.0) = -inf`.
  logsf itself is about -708 ... -64000 there, a perfectly normal double. (sf = exp(logsf)
  is subnormal or 0 there, which is not a finding for sf.)
- **Regime:** right tail where sf < 2.2e-308. For a power-law tail sf ~ C*x**-a, this is
  x >~ 1e205 for a = 1.5, x >~ 1e64 for a = 5, and x >~ 1e20 already for a = 100.
  Severe for sf between 4.9e-324 and 2.2e-308, loss (-inf) below that. 75 loss
  points and 9 severe points (the latter are F1, small s) in the high-tail region
  of the grid.
- **Reproducer:** `python accuracy.py refs/dpareto_lognorm_ref.py --at logsf 1e220 3 1.2 1.5 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logsf(1e+220, 3, 1.2, 1.5, 2)` | -inf | -754.2926964759705 | inf | 0.00199 | loss |
  | `logsf(1e+20, 0, 1, 100, 100)` | -inf | -1064.890452531217 | inf | 0.0433 | loss |
  | `logsf(1.886180808490652e+64, -2, 0.5, 5, 0.5)` | -inf | -749.2728952727983 | inf | 0.00667 | loss |
  | `logsf(1e+300, 0, 10, 2, 3)` | -inf | -1182.0618814201935 | inf | 0.00169 | loss |
  | `logsf(1e+215, 3, 1.2, 1.5, 2)` | -737.0230923000001 | -737.0233082785152 | 1.3e+09 | 0.00204 | severe |
  | `logsf(1e+210, 3, 1.2, 1.5, 2)` | -719.7539200810804 | -719.7539200810598 | 129 | 0.00208 | minor |

- **Fix direction:** compute logsf directly, mirroring `_logcdf`:
  S = Phic(z) + phi(z)*(b R(y1) - a R(y2))/(a + b), so
  `logsf = logsumexp([log_ndtr(-z), t2 + t5 - log(a+b)], b=[1, sign])`. Alternatively, use
  log1mexp(logcdf) only where logcdf < -log(2), and the direct form otherwise.

### F3: rounding of log(x) - u for large |u|

- **Code:** `_logpdf` line 1921 and `_logcdf` line 1933: `z = (log_y - m) / s`.
- **Mechanism:** pattern 5 (mild). `np.log(x)` has an absolute rounding error of up to
  0.5 ulp of |log x| ~ |u|, while the condition with respect to x only
  accounts for an absolute error of eps in log x. The extra error eps*|u|/s in z
  enters every method through d(log f)/dz.
- **Regime:** |u| large compared with s; bulk of (500, 0.1, 2, 3): pdf up to 224 ulp,
  sf up to 163 ulp, logsf 25 ulp (all minor).
- **Reproducer:** `python accuracy.py refs/dpareto_lognorm_ref.py --at pdf 3.8153592203558975e+217 500 0.1 2 3`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(3.8153592203558975e+217, 500, 0.1, 2, 3)` | 4.342529109424843e-219 | 4.342529109424627e-219 | 224 | 3 | minor |

  (cond (shapes) 1e3: the result is far more sensitive to u than this error.)
- **Fix direction:** none cheap inside the distribution. Users get full accuracy
  by putting u into `scale = exp(u)`. Document this rather than change the code.

## RuntimeWarnings

`underflow encountered in exp` (cdf, logcdf, sf, logsf, pdf) and
`underflow encountered in logaddexp` (pdf, logpdf) are emitted in all 7 parameter sets,
in the tails and in parts of the bulk. They come from `np.logaddexp` / `sc.logsumexp`
exponentiating the smaller of two very different log terms, and from `exp` of the final
log value. Most are spurious: they also occur on accurate results, e.g.
`pdf(1.234e-4, 3, 1.2, 1.5, 2)` (8.2 ulp, ok) and `logcdf(1e-190, 3, 1.2, 1.5, 2)`
(1.2 ulp, ok). They are not the cause of F1-F3.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf, logpdf | bulk, example parameters (3, 1.2, 1.5, 2) | 7.6 ulp (pdf), 1.9 ulp (logpdf) |
| cdf | high tail (cdf -> 1), all parameter sets | 0.5 ulp |
| sf | low tail (sf -> 1), all parameter sets | 0 ulp |
| logcdf | far left tail where cdf underflows (computed in log space), e.g. (3, 1.2, 1.5, 2) at 1e-300 | 60 ulp (minor, F1) |
| logcdf, sf | sign handling in `t5` (b R(y1) - a R(y2) changes sign in the bulk); no extra loss beyond F1 | covered by grid |
| logcdf | large a, b = 100, left tail where Phi(z) and the correction term are of similar size | 632 ulp (F1 level) |
| all | x = 0 (pdf 0, logpdf/logcdf -inf) and x = inf | exact |
| all | minor points not explained by F1/F3 | none found |

## Suspected, not confirmed

none. Hypothesis H3 (cancellation inside `logsumexp([t3, t4], b=[1, -1])` where
b R(y1) ~ a R(y2)) was covered by the bulk grid for all parameter sets. The
cancelled term is then small compared with Phi(z), so it causes no measurable
loss beyond F1.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| ppf | `brentq` on `cdf(x) - q` with absolute `xtol = 1e-14`, starting at the support edge 0 | no closed form. Solve `logcdf(exp(t)) = log(q)` in t = log x (bracket from the Pareto-tail asymptotics x ~ (q/C)**(1/b)), or Newton on log x | small q: x is below xtol. `ppf(1e-50, 3, 1.2, 1.5, 2)` returns 0.0, true 7.269219524869406e-25 (loss) |
| isf | `ppf(1 - q)` | solve `logsf(exp(t)) = log(q)` in t = log x (needs F2 fixed). Right-tail asymptotics x ~ (C/q)**(1/a) give a bracket | q < ~1e-16: 1 - q rounds to 1. `isf(1e-20, 3, 1.2, 1.5, 2)` returns 1e13, true 8.774642952763599e14 (4.5e15 ulp, severe) |

## Shared code

none. The helpers `_R`/`_logR` are private to this class. `sc.log_ndtr`,
`sc.logsumexp` and `scu._log1mexp` give correctly rounded results for their own inputs (checked
at the F1 point x = 1e-20, shapes (0, 1, 100, 100) against mpmath). The losses
come from how the class combines them.

## Existing GitHub issues

- #21900 (closed): BUG: stats: New XSLOW test failure in test_sampling.py - private `_cdf` at the support edge x=0 (UNU.RAN "CDF not increasing"); a cdf edge case, not the cancellation in F1-F3

## Verification

Verifier reference: `scratch/dpareto_lognorm/verify/verify.py`, independent route via the representation log X - u = s*N + T, T = E1/a - E2/b (asymmetric Laplace), with pdf, cdf and sf as `mp.quad` convolutions E[h(d - s N)], E[H(d - s N)] and E[1 - H(d - s N)] (sf integrated directly, not as 1 - cdf). Evaluated at 100 and 200 digits; all values agree to at least 1e-98 relative.

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `logpdf(1e-290, 0, 0.01, 1, 1)` | -0.6930961608886719 | -0.69309718055994531 | verified (6.6e9 ulp) | |
| F1 | `logpdf(1e-100, 0, 0.01, 1, 1)` | -0.6930971145629883 | -0.69309718055994531 | verified (4.3e8 ulp) | |
| F1 | `pdf(1e-290, 0, 0.01, 1, 1)` | 0.5000255104863995 | 0.50002500062501042 | verified (4.6e9 ulp) | |
| F1 | `cdf(1e-290, 0, 0.01, 1, 1)` | 5.000254049480568e-291 | 5.0002500062501045e-291 | verified (3.6e9 ulp) | |
| F1 | `logcdf(1e290, 0, 0.01, 1, 1)` | -5.000254049480568e-291 | -5.0002500062501039e-291 | verified (3.6e9 ulp) | |
| F1 | `sf(1e290, 0, 0.01, 1, 1)` | 5.000254049480568e-291 | 5.0002500062501039e-291 | verified (3.6e9 ulp) | |
| F1 | `logsf(1e290, 0, 0.01, 1, 1)` | -668.4427733402279 | -668.44277414883319 | verified (5.4e6 ulp) | |
| F1 | `cdf(1e120, 500, 0.1, 2, 3)` | 1.5131444453630854e-292 | 1.5131444461014998e-292 | verified (2.2e6 ulp) | |
| F1 | `pdf(10, 0, 0.01, 1, 1)` | 0.005000250006275993 | 0.0050002500062501042 | verified (2.33e4 ulp) | moderate, bulk |
| F1 | `pdf(9.5, 0, 1, 100, 100)` | 0.003332671709315517 | 0.0033326717093227354 | verified (9.75e3 ulp) | moderate, bulk |
| F1 | `pdf(1e-270, 3, 1.2, 1.5, 2)` | 3.784900150370755e-272 | 3.7849001502593879e-272 | verified (1.3e5 ulp) | |
| F1 | `sf(1e200, 3, 1.2, 1.5, 2)` | 2.5992268256729944e-298 | 2.5992268257115711e-298 | verified (6.7e4 ulp) | |
| F1 | `logpdf(1e-170, 500, 0.1, 2, 3)` | -2282.651610046625 | -2282.6516100611816 | verified (2.9e4 ulp) | |
| F2 | `logsf(1e220, 3, 1.2, 1.5, 2)` | -inf | -754.2926964759705 | verified (loss) | SciPy's logcdf there is 0.0 (+0.0, not -0.0 as the report says); log1mexp(0) = -inf either way |
| F2 | `logsf(1e20, 0, 1, 100, 100)` | -inf | -1064.890452531217 | verified (loss) | |
| F2 | `logsf(1.886180808490652e64, -2, 0.5, 5, 0.5)` | -inf | -749.27289527279837 | verified (loss) | |
| F2 | `logsf(1e300, 0, 10, 2, 3)` | -inf | -1182.0618814201934 | verified (loss) | |
| F2 | `logsf(1e215, 3, 1.2, 1.5, 2)` | -737.0230923000001 | -737.02330827851516 | verified (1.3e9 ulp) | logcdf = -8.22e-321 (subnormal), as described |

The mechanisms match the cited code. F1: lines 1898, 1925-1926, 1937 and 1945 form `-z**2/2` and `+y**2/2` separately. F2: line 1951 is `_log1mexp(_logcdf)`, and `_logcdf` goes subnormal (-2.6e-313 at x = 1e210, -8.2e-321 at 1e215) and then reaches 0 (1e220). F3 (minor) was not checked.

Verified by dist-accuracy-verifier: 2 of 2 verified.
