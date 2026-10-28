# betaprime: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `betaprime_gen`, lines 973-1081
- **Implemented:** pdf (1033-1035, `exp(_logpdf)`), logpdf (1037-1038), cdf (1040-1051, via `beta._cdf`/`beta._sf`, i.e. `sc.betainc`/`sc.betaincc`), sf (1053-1057, same), ppf (1059-1075, via `beta._ppf` = Boost `ibeta_inv` through `scu._beta_ppf`, and `beta._isf` = `sc.betainccinv`)
- **Inherited (not measured):** logcdf, logsf (rv_continuous default in 1.18.1: `log(cdf)` / `log1p(-sf)` split at the median, resp. `log(sf)` / `log1p(-cdf)`), isf = `ppf(1 - q)`
- **Related:** `beta` (calls `beta._cdf`, `beta._sf`, `beta._ppf`, `beta._isf`); helpers: none; special functions: `sc.betaln`, `sc.xlogy`, `sc.xlog1py`, `sc.betainc`, `sc.betaincc`, `sc.betainccinv`, `scu._beta_ppf`
- **Reference:** `refs/betaprime_ref.py`, 120 digits (incomplete beta by continued fraction with y = x/(1+x) and 1 - y = 1/(1+x) passed separately; quantiles by inversion of whichever of F, S is <= 1/2); self-checks: PASS
- **Parameter sets:** (5, 6), (0.5, 2), (2, 10), (2, 0.001), (0.001, 2), (1000, 0.5), (300, 1000); separate run for pdf, logpdf, cdf, sf only: (1e5, 3e5)
- **Worst finding:** loss in ppf

## Summary

cdf and sf map onto `sc.betainc`/`sc.betaincc` with arguments that are
always correctly formed (the split at x = 1 is right). They are accurate
except for a few hundred ulp at large shapes, which is inherited from Boost.
ppf inherits the Boost `ibeta_inv` non-convergence documented for `beta`.
It returns values that are wrong by up to 1e114 ulp, or nan, for small q,
e.g. (2, 10) with q <= 1e-113 and (0.5, 2) with q in [1e-14, 1e-9]. ppf also
loses up to ~2e3 ulp in the bulk when `r = beta.ppf` is just below its 0.9999
switch, because `r/(1 - r)` amplifies the rounding of r. logpdf cancels large
`xlogy`/`xlog1py`/`betaln` terms for large or unbalanced shapes. pdf =
`exp(logpdf)` turns that absolute error into relative error: up to 3.5e4 ulp
at (1000, 0.5), and 1.5e6 ulp near the mode at (1e5, 3e5), where `sc.betaln`
itself is 3.2e-10 off.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf | small q, shape-dependent bands: (2, 10) q <= 1e-101, (0.5, 2) q in [1e-14, 1e-9], (5, 2) q = 1e-200 | Boost `ibeta_inv` (`beta._ppf`) hits its iteration cap and returns the last iterate or nan (pattern 10; beta F1) | 8.2e-53 for 1.3e-151 (2.7e114 ulp); nan | loss |
| F2 | pdf | large shapes: (1e5, 3e5) everywhere measured; (1000, 0.5) everywhere, worst at large x | pdf = exp(logpdf): absolute error of logpdf (F3 + `sc.betaln` error) becomes relative error (pattern 5) | 1.5e6 ulp | severe |
| F3 | logpdf | large a+b near the mode ((1e5, 3e5)); a >> b at x >~ a/b ((1000, 0.5)); a small near x ~ 1e-3 ((0.001, 2)) | `xlogy(a-1, x) - xlog1py(a+b, x) - betaln(a, b)` cancel large terms (pattern 5); `sc.betaln` is 1.2e3 ulp off at (1000, 0.5) and 3.2e-10 absolute at (1e5, 3e5) | 3.9e5 ulp | moderate |
| F4 | ppf | r = beta.ppf(q) just below 0.9999 (x up to ~1e4), e.g. (1000, 0.5) bulk, (2, 0.001) | `r / (1 - r)` amplifies the rounding of r by 1/(1 - r) ~ x (pattern 1/8) | 1.91e3 ulp | moderate |

### F1: ppf returns garbage or nan for small q (Boost ibeta_inv)

- **Code:** `_ppf`, line 1066: `r = stats.beta._ppf(p, a, b)`, i.e.
  `scu._beta_ppf` (Boost `ibeta_inv`), followed by `r / (1 - r)` at line 1068.
- **Mechanism:** identical to `beta` F1 (reports/beta.md). Boost's Halley
  iteration exceeds its iteration limit and SciPy returns the last iterate
  (with `RuntimeWarning: ... Root finding evaluation exceeded %1% iterations`)
  or nan (pattern 10). For small q, r is tiny, so `r/(1-r)` = r and the
  garbage passes straight through. The `rnear1` fallback (line 1072) is never
  taken, because r is tiny. For (2, 10) the result is the constant
  8.196050044267321e-53 for every q <= 7.5e-114.
- **Regime:** small q, in bands that depend irregularly on the shapes:
  (2, 10) q <= 1e-101 (18 loss, 1 severe, 1 moderate point); (0.5, 2)
  q in [1e-14, 1e-9] (4 severe); (5, 2) at q = 1e-200 (nan). (5, 6),
  (2, 0.001), (0.001, 2), (1000, 0.5) showed no such failures.
- **Reproducer:** `python accuracy.py refs/betaprime_ref.py --at ppf 1e-120 2 10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 2, 10)` | 8.196050044267321e-53 | 1.3483997249264841e-151 | 2.7e+114 | 0.5 | loss |
  | `ppf(1e-120, 2, 10)` | 8.196050044267321e-53 | 1.3483997249264841e-61 | 2.7e+24 | 0.5 | loss |
  | `ppf(1e-200, 5, 2)` | nan | 6.988271187715792e-41 | inf | 0.2 | loss |
  | `ppf(1e-09, 0.5, 2)` | 2.4678339236759816e-24 | 4.444444444444445e-19 | 4.5e+15 | 2 | severe |
  | `ppf(1e-101, 2, 10)` | 4.26401908944277e-52 | 4.264014327112209e-52 | 5.0e+09 | 0.5 | severe |

  (`ppf(1e-09, 0.5, 2)` has relative error ~1 but the tool classifies it as
  severe; reported as given.)
- **Fix direction:** fix in Boost / `scu._beta_ppf` (see beta F1). Locally:
  for small q use the leading-order start `x0 = (q*a*B(a,b))**(1/a)` (log
  space) and refine with Newton on log(cdf).

### F2: pdf = exp(logpdf) amplifies the absolute error of logpdf

- **Code:** `_pdf`, line 1035: `np.exp(self._logpdf(x, a, b))`.
- **Mechanism:** an absolute error e in logpdf becomes a relative error e
  in pdf. logpdf has absolute errors well above eps for large shapes (F3,
  and the error of `sc.betaln`, see Shared code), so the pdf error does not
  depend on how small or large the pdf is. At (1e5, 3e5) the `sc.betaln`
  error alone is 3.25e-10 absolute, which is 1.46e6 ulp of the pdf at every
  x. That matches the observed 1.4e6-1.5e6 ulp at all 14 points, including
  the mode where cond (arg) = 1. At (1000, 0.5) `sc.betaln` contributes
  7.7e-13 (3.5e3 ulp of the pdf) everywhere. For large x the cancellation
  of F3 adds more: logpdf(1e200) = -722 with a 50-ulp error, which is
  8e-12 absolute, giving 3.48e4 ulp in pdf. Even with an exact logpdf, a
  0.5-ulp rounding of logpdf gives about 0.5*|logpdf| ulp in pdf. That is
  why the moderate shapes show 199-746 ulp minor errors in the far tails
  (e.g. (2, 0.001): 746 ulp; (0.001, 2): 548 ulp).
- **Regime:** severe for a + b ~ 4e5 at every x; moderate (1e3-3.5e4 ulp)
  for (1000, 0.5) at every x (49 moderate points), largest for x >~ 1e50;
  minor (<= ~750 ulp, about 0.5*|logpdf|) in the far tails for all other
  sets.
- **Reproducer:** `python accuracy.py refs/betaprime_ref.py --at pdf 1e200 1000 0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.3345527195017296, 1e5, 3e5)` | 198.01421973424152 | 198.0142196671459 | 1.5e+06 | 275 | severe |
  | `pdf(0.3333333333333333, 1e5, 3e5)` | 327.7642333828316 | 327.764233279224 | 1.4e+06 | 1 | severe |
  | `pdf(1e+200, 1000, 0.5)` | 1.7839011145716372e-299 | 1.783901114585432e-299 | 3.48e+04 | 1.5 | moderate |
  | `pdf(6000.0, 1000, 0.5)` | 3.24886249492298e-05 | 3.2488624949275873e-05 | 6.39e+03 | 1.33 | moderate |

  (cond (shapes) is 548 and 226 at the first and third rows, context only.)
- **Fix direction:** compute the pdf without going through a large
  logarithm, e.g. `scu._beta_pdf(x/(1+x), a, b) / (1+x)**2` in the bulk
  (Boost `ibeta_power_terms` is ~2 ulp at large shapes, see beta report).
  Fix `sc.betaln` (Shared code).

### F3: logpdf cancels large terms

- **Code:** `_logpdf`, line 1038:
  `sc.xlogy(a - 1.0, x) - sc.xlog1py(a + b, x) - sc.betaln(a, b)`.
- **Mechanism:** pattern 5, in three regimes.
  (1) a >> b, x large: the first two terms are both about a*log(x) and
  cancel to `-(a-1)*log1p(1/x) - (b+1)*log1p(x)`. At (1000, 0.5) and
  x = 2e5 the terms are 1.2e4 against a result of -15.4. A 0.5-ulp error
  in each term gives ~400 ulp; observed 679.
  (2) large a + b near the mode: the terms are ~2e5 at (1e5, 3e5) against
  a result of O(1-10). In addition `sc.betaln(1e5, 3e5)` is 3.25e-10 off
  (6.5 ulp of -2.25e5), which alone is ~2.5e5 ulp of a logpdf of 5.8
  (observed 2.5e5 at x = 1/3).
  (3) a small: at (0.001, 2), x = 0.001, `xlogy(-0.999, 0.001)` = 6.90 and
  `betaln(0.001, 2)` = 6.91 cancel to -0.0079 (777 ulp, minor, cond (arg) 127).
  `sc.betaln(1000, 0.5)` is also 7.7e-13 off (1.2e3 ulp), which adds ~350
  ulp to every logpdf value near -10 at (1000, 0.5).
- **Regime:** moderate (14 points) for (1e5, 3e5) within ~10 relative sd of
  the mode; minor (<= 777 ulp) for (1000, 0.5) at x >~ 600, for (300, 1000)
  near the mode, and for (0.001, 2) near x = 1e-3.
- **Reproducer:** `python accuracy.py refs/betaprime_ref.py --at logpdf 0.330907878145448 1e5 3e5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(0.330907878145448, 1e5, 3e5)` | 3.8020329421851784 | 3.802032941859189 | 3.9e+05 | 144 | moderate |
  | `logpdf(0.3333333333333333, 1e5, 3e5)` | 5.792294549260987 | 5.792294548944883 | 2.5e+05 | 0.173 | moderate |
  | `logpdf(0.001, 0.001, 2)` | -0.00790825511240012 | -0.007908255112398755 | 777 | 127 | minor |
  | `logpdf(200000.0, 1000, 0.5)` | -15.43272375921515 | -15.432723759217476 | 679 | 0.0969 | minor |

- **Fix direction:** for x > 1 use
  `-(a-1)*log1p(1/x) - (b+1)*log1p(x) - betaln(a, b)`. For large shapes use
  Stirling-corrected forms in which the large terms cancel analytically
  (as Boost `ibeta_power_terms` does), and fix `sc.betaln` for large/unbalanced
  shapes.

### F4: ppf loses digits in r/(1 - r) just below the 0.9999 switch

- **Code:** `_ppf`, lines 1068-1069: `out = r / (1 - r)`, with the
  alternative `1/beta._isf(p, b, a) - 1` used only when `r > 0.9999`.
- **Mechanism:** r is rounded to a double (relative error up to 1.1e-16,
  plus Boost's own error). `x = r/(1-r)` amplifies an error in r by
  `1/(1 - r) = 1 + x` (patterns 1 and 8). With the switch at r = 0.9999
  the loss reaches ~1e4 * 0.5 ulp just below x = 9999. The error grows
  linearly in x: 101 ulp at x = 652, 419 ulp at x = 4395, 1.48e3 at
  x = 9140, 1.91e3 at x = 9990. The ppf is well conditioned there
  (cond 4-8), so the error is not inherent.
- **Regime:** any shapes where the result x is in ~[500, 9999], with q
  in the bulk or tail. Seen at (1000, 0.5) (median x = 4395) and at
  (2, 0.001) near q = 0.008. Above x = 9999 the `1/betainccinv - 1`
  branch is accurate (high-tail max 2.2e3 ulp, all within conditioning).
- **Reproducer:** `python accuracy.py refs/betaprime_ref.py --at ppf 0.6546077528522537 1000 0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.6546077528522537, 1000, 0.5)` | 9990.000000004231 | 9989.999999999996 | 1.91e+03 | 4.05 | moderate |
  | `ppf(0.6399999999999999, 1000, 0.5)` | 9140.399074450379 | 9140.399074447378 | 1.48e+03 | 3.83 | moderate |
  | `ppf(0.008176429292048043, 2, 0.001)` | 9990.000000004231 | 9989.999999999995 | 1.91e+03 | 8.25 | moderate |
  | `ppf(0.49999999999999994, 1000, 0.5)` | 4394.619799181028 | 4394.619799181437 | 419 | 2.33 | minor |

- **Fix direction:** for x > 1 (r > 0.5) take the complementary branch
  `x = 1/y - 1` with `y = sc.betainccinv(b, a, p)` (y = 1/(1+x)), so the
  small quantity 1 - r is computed directly instead of by subtraction.

## RuntimeWarnings

- ppf: `Error in function boost::math::ibeta<double>(...): Root finding
  evaluation exceeded %1% iterations, giving up now` in (0.5, 2) and
  (2, 10). It coincides with the wrong results of F1. The nan at (5, 2)
  comes without this warning.
- pdf: `underflow encountered in exp` in all 7 parameter sets, where the
  true pdf is below the normal range. Expected, not an accuracy problem.
- No spurious warnings on accurate results were seen.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid incl. both sides of x = 1 and x = 1 +- 1e-12, x up to 1e300, all parameter sets | 6.6 ulp for moderate shapes; 624 ulp minor at (300, 1000) (cond 262) and 3.0e3 moderate at (1e5, 3e5) (cond 2.7e3), slightly beyond conditioning from Boost `ibeta` at large shapes (see beta F8) |
| sf | same, incl. far right tail (x = 1e300 with b = 0.001) | 5.6 ulp for moderate shapes; 463 ulp minor at (300, 1000) (cond 351); 1.65e3 moderate at (1e5, 3e5) (cond 1.43e3), Boost as above |
| ppf | q near 1 (high tail, the `1/betainccinv - 1` branch) | 2.21e3 ulp, within conditioning (cond up to 1e11) |
| ppf | (300, 1000) small q | 760 ulp minor (cond 0.004); (0.001, 2) at q = 0.5: 1.09e3 ulp moderate with cond 1e3, result 3.4e-302. Boost `ibeta_inv` near-tiny results (cf. beta F7), mechanism inside Boost |
| logpdf | moderate shapes (5, 6), (0.5, 2), (2, 10), (2, 0.001) full grid | 35 ulp |

## Suspected, not confirmed

- ppf `rnear1` branch (line 1072) with `beta._isf(p, b, a)` = `sc.betainccinv(b, a, p)`.
  beta F3 shows nan from `betainccinv` when its second shape is 2 and the
  target is tiny. Here that would need a = 2 and r > 0.9999 at tiny p, which
  does not happen, because r > 0.9999 implies p is not tiny. Not reachable
  in the measured sets.
- ppf for very large shapes (e.g. (1e5, 3e5)) was not measured: each
  quantile reference takes 10-25 s at 120 digits. The same Boost issues as
  beta F6/F7 are expected.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| isf | `ppf(1 - q)` | for x > 1: `1/sc.betaincinv(b, a, q) - 1` (y = 1/(1+x) solves I_y(b, a) = q); for x <= 1: r = `sc.betainccinv(a, b, q)`, x = r/(1-r) | q < ~1e-16: `1 - q` rounds to 1 and the result is inf. `isf(1e-20, 5, 6)` gives inf, reference 5250.974713815796 (loss) |
| logsf | `log(sf)` below the median | with yc = 1/(1+x), y = x/(1+x): `b*log(yc) + a*log(y) - log(b) - betaln(a, b) + log(hyp2f1(a+b, 1, b+1, yc))` | sf underflows for x**-b/(b*B) < 1e-308. `logsf(1e70, 5, 6)` gives -inf, reference -961.7386315267818 (loss) |
| logcdf | `log(cdf)` below the median | `a*log(y) + b*log(yc) - log(a) - betaln(a, b) + log(hyp2f1(a+b, 1, a+1, y))` | cdf underflows for x**a/(a*B) < 1e-308. `logcdf(1e-70, 5, 6)` gives -inf, reference -800.3753534604045 (loss) |

## Shared code

- `beta._ppf` = `scu._beta_ppf` (Boost `ibeta_inv`): non-convergence for small
  q (F1) is the same defect as beta F1. It affects `beta`, `betaprime`, and
  every other caller of `scu._beta_ppf`.
- `sc.betaln`: 7.7e-13 absolute (1.2e3 ulp) at (1000, 0.5) and 3.25e-10
  absolute (6.5 ulp of -2.25e5) at (1e5, 3e5). This drives most of F2 and
  part of F3. It is the same cephes `lbeta` weakness as beta F4: the
  `lgam(a+b) - lgam(a) - lgam(b)` cancellation for large a, where the
  asymptotic branch is taken only for a > 1e6*b. Measured with
  `sc.betaln` vs `mp.loggamma` at 50 digits.
- Boost `ibeta`/`ibetac` slightly beyond conditioning for large shapes
  (cdf/sf, see beta F8). Minor here.

## Existing GitHub issues

- #17631 (closed): BUG: numerical issues for cdf/ppf of the betaprime distribution - fixed in PR #17909 (SciPy 1.11), which introduced the current r = 0.9999 switch in ppf; relates to F4 (background)
- boostorg/math #1511 (closed, Boost PR #1512): `ibeta_inv` iteration-limit failures, the same defect cited for beta F1 - relates to F1 (whether SciPy 1.18.1's bundled Boost includes the fix is not established; F1 reproduces in 1.18.1)

Excluded: #18274 (spurious warnings from `betaprime.fit`), #4030 (simplify cdf, closed), #20208 (`_beta_ppf` warnings on M2 mac, test-only). Searched via the GitHub search API for "betaprime"; `gh` was not available.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(1e-300, 2, 10)` | 8.196050044267321e-53 | 1.3483997249264841e-151 | verified | 2.7e114 ulp. Ref: findroot on log cdf, cdf via 2F1 series of I_y; cdf(x_ref)/q - 1 ~ 1e-200; 100 vs 200 digits agree to 1e-102. `beta._ppf(1e-120, 2, 10)` itself returns 8.196e-53, so the mechanism (Boost ibeta_inv, line 1066) is confirmed |
| F1 | `ppf(1e-120, 2, 10)` | 8.196050044267321e-53 | 1.3483997249264841e-61 | verified | 2.7e24 ulp |
| F1 | `ppf(1e-200, 5, 2)` | nan | 6.988271187715792e-41 | verified | `beta._ppf(1e-200, 5, 2)` is nan |
| F1 | `ppf(1e-09, 0.5, 2)` | 2.4678339236759816e-24 | 4.444444444444445e-19 | verified | 4.5e15 ulp (relative error ~1) |
| F1 | `ppf(1e-101, 2, 10)` | 4.26401908944277e-52 | 4.264014327112209e-52 | verified | 5.0e9 ulp |
| F2 | `pdf(0.3345527195017296, 1e5, 3e5)` | 198.01421973424152 | 198.0142196671459 | verified | 1.53e6 ulp. Ref: docstring formula x^(a-1)(1+x)^(-a-b)/B(a,b) with mp.beta. `sc.betaln(1e5, 3e5)` error measured as -3.25e-10, consistent with the mechanism |
| F2 | `pdf(0.3333333333333333, 1e5, 3e5)` | 327.7642333828316 | 327.764233279224 | verified | 1.42e6 ulp |
| F2 | `pdf(1e200, 1000, 0.5)` | 1.7839011145716372e-299 | 1.783901114585432e-299 | verified | 3.48e4 ulp; `sc.betaln(1000, 0.5)` error -7.7e-13 confirmed |
| F2 | `pdf(6000.0, 1000, 0.5)` | 3.24886249492298e-05 | 3.2488624949275873e-05 | verified | 6.39e3 ulp |
| F3 | `logpdf(0.330907878145448, 1e5, 3e5)` | 3.8020329421851784 | 3.802032941859189 | verified | 3.86e5 ulp. Ref: (a-1)log x - (a+b)log1p x - log mp.beta(a, b) |
| F3 | `logpdf(0.3333333333333333, 1e5, 3e5)` | 5.792294549260987 | 5.792294548944883 | verified | 2.46e5 ulp, = betaln error 3.25e-10 / 5.79 |
| F3 | `logpdf(0.001, 0.001, 2)` | -0.00790825511240012 | -0.007908255112398755 | verified | 778 ulp (minor, as reported) |
| F3 | `logpdf(200000.0, 1000, 0.5)` | -15.43272375921515 | -15.432723759217476 | verified | 678 ulp (minor, as reported) |
| F4 | `ppf(0.6546077528522537, 1000, 0.5)` | 9990.000000004231 | 9989.999999999996 | verified | 1.91e3 ulp. Ref: findroot on log sf (2F1 series in 1/(1+x)); sf(x_ref)/(1-q) - 1 ~ 3e-199. `beta._ppf` r is only 0.19 ulp off; r/(1-r) amplifies it by ~1e4, matching the mechanism (line 1068) |
| F4 | `ppf(0.6399999999999999, 1000, 0.5)` | 9140.399074450379 | 9140.399074447378 | verified | 1.48e3 ulp |
| F4 | `ppf(0.008176429292048043, 2, 0.001)` | 9990.000000004231 | 9989.999999999995 | verified | 1.91e3 ulp; same r (0.19 ulp off) |
| F4 | `ppf(0.49999999999999994, 1000, 0.5)` | 4394.619799181028 | 4394.619799181437 | verified | 419 ulp (minor) |

Verifier scripts: `scratch/betaprime/verify/verify.py` (references at 100 and 200 digits, all agree to better than 1e-96 relative) and `scratch/betaprime/verify/mech.py` (mechanism checks). `refs/betaprime_ref.py` was not consulted.

Verified by dist-accuracy-verifier: 4 of 4 verified.
