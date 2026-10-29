# genhalflogistic: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `genhalflogistic_gen`, lines 3957-4004
- **Implemented:** pdf, cdf, ppf (also `_entropy`, not reviewed)
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** none; helpers: none; special functions: none (plain numpy `**`)
- **Reference:** `refs/genhalflogistic_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (0.7727472780992932), (1), (0.5), (3), (0.001), (1e-08), (1e-14), (1e+06)
- **Worst finding:** loss in cdf, ppf and pdf

## Summary

pdf, cdf and ppf are accurate in the bulk for moderate c (0.5 to 3). The cdf
loses all digits in the left tail for every c: `1 - (1 - c x)**(1/c)` cancels,
so it returns 0 for x below about 1e-16 and has relative error eps/x above
that. For large c this covers the whole representable support. ppf has the
same cancellation in `1 - r**c` for small q. For small c (1e-8, 1e-14) all
three methods lose 7 to 13 digits everywhere, because the rounding error of
`1 - c*x` or `r**c` gets amplified by 1/c. pdf returns nan for c > 1 where
`1 - c*x` rounds to 0 just inside the support.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | small x, any c (whole support for large c) | `1.0 - tmp2` cancels (patterns 1, 2) | 0.0 for 5e-301 | loss |
| F2 | ppf | small q, any c | `1 - ((1-q)/(1+q))**c` cancels (pattern 8) | 0.0 for 2e-300 | loss |
| F3 | pdf | c > 1, x where `1 - c*x` rounds to 0 | `0**(1/c-1) * 0 = inf*0 = nan` (pattern 3) | nan for 1.37e11 | loss |
| F4 | cdf | small c, all x | rounding of `1 - c*x` raised to 1/c (patterns 6, 9) | 5e14 ulp | severe |
| F5 | ppf | small c, all q | `1 - r**c` cancels, r**c ~ 1 (patterns 8, 9) | 5e14 ulp | severe |
| F6 | pdf | small c, all x | rounding of `1 - c*x` raised to 1/c - 1 (patterns 6, 9) | 1.4e13 ulp | severe |

### F1: cdf cancels in 1 - (1 - c x)**(1/c) for small x

- **Code:** `_cdf`, lines 3996-3998: `tmp2 = tmp**(limit)`; `return (1.0-tmp2) / (1+tmp2)`.
- **Mechanism:** tmp2 = (1 - c x)**(1/c) = 1 - x + O(x^2), so `1.0 - tmp2`
  keeps only the digits of tmp2 beyond its leading 1s (patterns 1 and 2).
  The relative error is about eps/x; for x below ~1.1e-16 tmp2 rounds to 1
  and the cdf is 0. For large c the true cdf is about -log1p(-c x)/(2c),
  i.e. below ~2e-5 everywhere in the representable support, so tmp2 is
  within c*1e-5 of 1 and the cancellation covers the whole support.
- **Regime:** loss for x < ~1e-16 (all c); severe up to x ~ 1e-8; moderate
  up to x ~ 1e-4. For c = 1e6, moderate to severe for all x in (0, 1/c).
- **Reproducer:** `python accuracy.py refs/genhalflogistic_ref.py --at cdf 1e-300 0.7727472780992932`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-300, 0.7727472780992932)` | 0.0 | 5e-301 | inf | 1 | loss |
  | `cdf(1e-16, 0.7727472780992932)` | 5.551115123125783e-17 | 5e-17 | 5e+14 | 1 | severe |
  | `cdf(1e-12, 1.0)` | 4.999889391401893e-13 | 5.0000000000025e-13 | 1.0e+11 | 1 | severe |
  | `cdf(1e-08, 3.0)` | 5.000000105634949e-09 | 5.000000075000001e-09 | 2.8e+07 | 1 | severe |
  | `cdf(0.0001, 3.0)` | 5.0007501458671676e-05 | 5.000750145865216e-05 | 1.8e+03 | 1 | moderate |
  | `cdf(5e-08, 1e6)` | 2.5646647186460236e-08 | 2.564664719377526e-08 | 1.3e+06 | 1 | severe |
  | `cdf(5e-07, 1e6)` | 3.4657359025787395e-07 | 3.4657359027995875e-07 | 2.9e+05 | 1.44 | moderate |

- **Fix direction:** with `L = log1p(-c*x)/c`, `cdf = tanh(-L/2)` (equivalently `-expm1(L)/(1 + exp(L))`); this also fixes F4.

### F2: ppf cancels in 1 - ((1-q)/(1+q))**c for small q

- **Code:** `_ppf`, line 4001: `1.0/c*(1-((1.0-q)/(1.0+q))**c)`.
- **Mechanism:** r = (1-q)/(1+q) = 1 - 2q + ..., so r**c = 1 - 2cq + ... and
  `1 - r**c` cancels (pattern 8). `1.0-q` and `1.0+q` already round away q
  below eps; the relative error is about eps/q and the result is 0 once
  q < ~1.1e-16.
- **Regime:** loss for q < ~1e-16 (all c); severe for q up to ~1e-8,
  moderate up to ~1e-4.
- **Reproducer:** `python accuracy.py refs/genhalflogistic_ref.py --at ppf 1e-300 0.7727472780992932`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 0.7727472780992932)` | 0.0 | 2e-300 | inf | 1 | loss |
  | `ppf(1e-16, 1.0)` | 1.1102230246251565e-16 | 1.9999999999999997e-16 | 2e+15 | 1 | severe |
  | `ppf(1e-12, 1.0)` | 2.0000667788622195e-12 | 1.999999999998e-12 | 1.5e+11 | 1 | severe |
  | `ppf(1e-08, 3.0)` | 1.99999993603465e-08 | 1.9999999400000013e-08 | 8.9e+06 | 1 | severe |
  | `ppf(0.0001, 0.7727472780992932)` | 0.00019998454651720088 | 0.00019998454651715537 | 1e+03 | 1 | moderate |

- **Fix direction:** log r = log1p(-q) - log1p(q) = -2 atanh(q), so
  `ppf = -expm1(-2*c*atanh(q))/c`; this also fixes F5.

### F3: pdf returns nan for c > 1 where 1 - c*x rounds to 0

- **Code:** `_pdf`, lines 3989-3992: `tmp = 1-c*x`; `tmp0 = tmp**(limit-1)`; `tmp2 = tmp0*tmp`; `return 2*tmp0 / (1+tmp2)**2`.
- **Mechanism:** for x = fl(1/c) the product c*x rounds to 1, so tmp = 0.
  For c > 1 the exponent 1/c - 1 is negative and tmp0 = inf; `tmp2 = inf*0`
  is nan and the pdf is nan (pattern 3: an intermediate overflows that the
  final formula would cancel; tmp2 should be `tmp**limit` = 0). The true
  value is finite (1.37e11 for c = 3). The same happens at the exact
  endpoint x = 1/c (e.g. `pdf(0.5, 2)` gives nan where the limit is +inf;
  not measured because the true value is not finite).
- **Regime:** c > 1 and x within ~eps/c of 1/c.
- **Reproducer:** `python accuracy.py refs/genhalflogistic_ref.py --at pdf 0.3333333333333333 3.0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.3333333333333333, 3.0)` | nan | 137437904901.99997 | inf | 1.2e+16 | loss |
  | `pdf(1e-06, 1e6)` | nan | 1.1049262693727366e+16 | inf | 2.2e+16 | loss |

  (cond (arg) is 1.2e16 and 2.2e16: here the pdf is extremely sensitive to
  x, so no double-precision evaluation of `1 - c*x` gives many correct
  digits; the problem is the nan instead of a large finite value or inf.)
- **Fix direction:** compute `tmp2 = tmp**limit` directly instead of `tmp0*tmp`
  (gives inf/1 = inf instead of nan), or evaluate in log space.

### F4: cdf loses ~1/c ulp for small c

- **Code:** `_cdf`, lines 3996-3997: `tmp = np.asarray(1-c*x)`; `tmp2 = tmp**(limit)`.
- **Mechanism:** `1-c*x` carries a rounding error of up to eps/2 relative,
  and raising it to the power 1/c multiplies that error by 1/c (patterns 6
  and 9). Once c*x < eps/2, tmp is exactly 1 and the cdf is 0. The c -> 0
  limit (half-logistic, cdf = tanh(x/2)) is thus not approached.
- **Regime:** small c, all x: ~1e7 ulp at c = 1e-8, ~1e12 ulp at c = 1e-14;
  2.1e4 ulp at c = 1e-3, x = 0.01 (combined with F1).
- **Reproducer:** `python accuracy.py refs/genhalflogistic_ref.py --at cdf 1.0 1e-8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(0.01, 1e-14)` | 0.005551058104848119 | 0.004999958333749996 | 5e+14 | 1 | severe |
  | `cdf(1.0, 1e-14)` | 0.4618028041004026 | 0.46211715726001173 | 3.1e+12 | 0.851 | severe |
  | `cdf(10.0, 1e-14)` | 0.9999094861363302 | 0.9999092042625952 | 1.3e+09 | 0.00091 | severe |
  | `cdf(0.5, 1e-08)` | 0.24491866156299036 | 0.24491866299121842 | 2.6e+07 | 0.96 | severe |
  | `cdf(1.0, 1e-08)` | 0.4621171612019843 | 0.4621171592261291 | 1.9e+07 | 0.851 | severe |
  | `cdf(0.01, 0.001)` | 0.004999983333268897 | 0.004999983333291667 | 2.1e+04 | 1 | moderate |

- **Fix direction:** `L = log1p(-c*x)/c`, `cdf = tanh(-L/2)` (as F1).

### F5: ppf cancels in 1 - r**c for small c

- **Code:** `_ppf`, line 4001: `1.0/c*(1-((1.0-q)/(1.0+q))**c)`.
- **Mechanism:** for small c, r**c = exp(c log r) = 1 + c log r + ..., so
  `1 - r**c` cancels and loses about 1/(c |log r|) ulp, which the division by
  c does not recover (patterns 8 and 9).
- **Regime:** small c, all q: ~1e7 ulp at c = 1e-8, ~1e12 ulp at c = 1e-14,
  up to 7.7e3 ulp at c = 1e-3. In the right tail the errors exceed the
  ill-conditioning of ppf near q = 1 by orders of magnitude.
- **Reproducer:** `python accuracy.py refs/genhalflogistic_ref.py --at ppf 0.5 1e-8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.01, 1e-14)` | 0.02220446049250313 | 0.020000666706669522 | 5e+14 | 1 | severe |
  | `ppf(0.5, 1e-14)` | 1.099120794378905 | 1.0986122886681036 | 2.1e+12 | 1.21 | severe |
  | `ppf(0.99999999, 1e-14)` | 19.118040484045196 | 19.113827914485725 | 9.9e+11 | 5.2e+06 | severe |
  | `ppf(0.01, 1e-08)` | 0.020000667788622195 | 0.020000666704669393 | 2.4e+08 | 1 | severe |
  | `ppf(0.5, 1e-08)` | 1.098612278926936 | 1.0986122826333649 | 1.5e+07 | 1.21 | severe |
  | `ppf(0.01, 0.001)` | 0.0200004666947029 | 0.02000046669466863 | 7.7e+03 | 1 | moderate |

- **Fix direction:** `ppf = -expm1(-2*c*atanh(q))/c` (as F2).

### F6: pdf loses ~1/c ulp for small c

- **Code:** `_pdf`, lines 3989-3991: `tmp = np.asarray(1-c*x)`; `tmp0 = tmp**(limit-1)`; `tmp2 = tmp0*tmp`.
- **Mechanism:** same as F4: the rounding error of `1-c*x` is amplified by
  the exponent 1/c - 1 (patterns 6 and 9). For c*x < eps/2, tmp = 1 exactly
  and the pdf is exactly 0.5.
- **Regime:** small c, all x: ~1e7 ulp at c = 1e-8, ~1e12 ulp at c = 1e-14.
  At c = 1e-3 the errors are minor (up to 605 ulp, mostly within the
  conditioning).
- **Reproducer:** `python accuracy.py refs/genhalflogistic_ref.py --at pdf 2.0 1e-14`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(10.0, 1e-14)` | 9.050976729009773e-05 | 9.079161547186703e-05 | 1.4e+13 | 10 | severe |
  | `pdf(2.0, 1e-14)` | 0.21024291907910922 | 0.20998717080701404 | 5.5e+12 | 1.52 | severe |
  | `pdf(0.0001, 1e-14)` | 0.5 | 0.49999999875 | 1.1e+07 | 5e-09 | severe |
  | `pdf(10.0, 1e-08)` | 9.079157963722699e-05 | 9.079157915938358e-05 | 2.4e+07 | 10 | severe |
  | `pdf(1.0, 1e-08)` | 0.39322386859354846 | 0.3932238695066249 | 1.0e+07 | 0.462 | severe |
  | `pdf(0.01, 1e-08)` | 0.49998750025625943 | 0.4999875002583279 | 1.86e+04 | 5e-05 | moderate |

- **Fix direction:** `L = log1p(-c*x)`; `tmp0 = exp((1/c - 1)*L)`, `tmp2 = exp(L/c)`.

## RuntimeWarnings

- pdf, c > 1, x at or within ~eps/c of 1/c: "divide by zero encountered in
  power" and "invalid value encountered in multiply"; the result is nan (F3).
- pdf and cdf, small c (1e-3) far in the right tail and near the endpoint:
  "underflow encountered in power/multiply" from `tmp**(limit-1)` and
  `tmp**limit`. Where the true pdf/sf parts are subnormal or zero, the
  returned values are right (pdf subnormal is not measured; cdf rounds to 1
  correctly): spurious warnings.
- ppf, c = 1e6: "underflow encountered in power" from `r**c` with r <= 1/3;
  the result 1/c is correct (spurious).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid for c = 0.7727, 1, 0.5, 3, 1e6 (except F3); near the upper endpoint (1/c)(1 - r), r = 1e-15 ... 0.5 | within conditioning (7.6e12 ulp at cond 2.9e14 near 1/c for c = 0.7727); 1.5 ulp for c = 1 |
| pdf | c = 1e-3, full grid | 605 ulp (minor, cond 538); 60 ulp at x = 0.5 (minor, tail of F6) |
| cdf | bulk and right tail for c = 0.5 ... 3, 1e6 (except F1 region) | 61 ulp (minor, residual of F1 at x = 0.002, c = 0.5) |
| ppf | bulk and q -> 1 (1 - q down to 3e-16) for c = 0.5 ... 3 and 1e6 | within conditioning |

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | `2*t/(1 + t)` with `t = exp(log1p(-c*x)/c)` | near the upper end 1/c: `sf(2*(1-1e-10), 0.5)` gives 0.0, true 2.0000003309614977e-20 |
| logsf | log(sf) | `log(2) + L - log1p(exp(L))`, `L = log1p(-c*x)/c` | same: `logsf(2*(1-1e-10), 0.5)` gives -inf, true -45.35855451384023 |
| isf | ppf(1 - q) | `t = q/(2 - q)`, `isf = -expm1(c*log(t))/c` | small q: `1 - q` rounds; `isf(1e-20, 0.5)` gives 2.0, true 1.9999999998585787 |
| logpdf | log(pdf) | `log(2) + (1/c - 1)*log1p(-c*x) - 2*log1p(exp(log1p(-c*x)/c))` | pdf underflows near 1/c for c < 1: `logpdf(1000*(1-1e-4), 1e-3)` gives -inf, true -9200.436884423627 |
| logcdf | log(cdf) | `log(tanh(-L/2))`, `L = log1p(-c*x)/c` (for small x: `log(-expm1(L)) - log1p(exp(L))`) | small x, through F1: `logcdf(1e-300, 0.7727472780992932)` gives -inf, true -691.4686750787737 |

## Shared code

none

## Existing GitHub issues

none found (GitHub search API: no issue or PR with genhalflogistic in the title; full-text hits are only generic tracking issues such as #17832 and #15195; gh not installed).

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e-300, 0.7727472780992932)` | 0.0 | 5.000000000000000e-301 | verified | reference from `mp.quad` of the pdf (rescaled t = x u) and from `-expm1(L)/(1+exp(L))`; 100 and 200 digits agree |
| F1 | `cdf(1e-12, 1.0)` | 4.999889391401893e-13 | 5.0000000000024999e-13 | verified | 1e11 ulp, `mp.quad` of the pdf |
| F1 | `cdf(5e-08, 1e6)` | 2.5646647186460236e-08 | 2.564664719377526e-08 | verified | 1.3e6 ulp |
| F2 | `ppf(1e-300, 0.7727472780992932)` | 0.0 | 2.0000000000000000501e-300 | verified | x = -expm1(c (log1p(-q) - log1p(q)))/c; cdf(x_ref) = q checked |
| F2 | `ppf(1e-16, 1.0)` | 1.1102230246251565e-16 | 1.9999999999999998e-16 | verified | 2e15 ulp; `(1.0-q)/(1.0+q)` rounds to 1 - 2^-53 |
| F3 | `pdf(0.3333333333333333, 3.0)` | nan | 137437904901.99997 | verified | `3*x` rounds to 1, so tmp = 0, `0**(-2/3)*0 = inf*0 = nan` (lines 3989-3991) |
| F3 | `pdf(1e-06, 1e6)` | nan | 1.1049262693727366e+16 | verified | same mechanism |
| F4 | `cdf(0.01, 1e-14)` | 0.005551058104848119 | 0.0049999583337499961 | verified | 5e14 ulp, `mp.quad` of the pdf |
| F4 | `cdf(1.0, 1e-08)` | 0.4621171612019843 | 0.4621171592261291 | verified | 1.9e7 ulp |
| F5 | `ppf(0.5, 1e-14)` | 1.099120794378905 | 1.0986122886681037 | verified | 2.1e12 ulp; cdf(x_ref) = q to 1e-200 |
| F5 | `ppf(0.5, 1e-08)` | 1.098612278926936 | 1.0986122826333649 | verified | 1.5e7 ulp |
| F6 | `pdf(10.0, 1e-14)` | 9.050976729009773e-05 | 9.079161547186703e-05 | verified | 1.4e13 ulp; pdf evaluated as exp((1/c-1) log1p(-cx)) / (1+exp(log1p(-cx)/c))^2 |
| F6 | `pdf(1.0, 1e-08)` | 0.39322386859354846 | 0.39322386950662492 | verified | 1e7 ulp |

Verifier note (not a finding against the methods): the docstring pdf writes the exponent as `1/(c-1)`; the correct exponent, obtained by differentiating the cdf (1-t)/(1+t) with t = (1-cx)^(1/c), is `1/c - 1`, as in the code comment and the code. The verifier reference uses `1/c - 1`. The docstring has a typo.

Verified by dist-accuracy-verifier: 6 of 6 verified.
