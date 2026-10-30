# invgauss: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `invgauss_gen`, lines 5023-5174
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf (ppf/isf call Boost `scu._invgauss_ppf` / `scu._invgauss_isf`, with the generic `rv_continuous._ppf` / `_isf` as nan fallback)
- **Inherited (not measured):** none
- **Related:** wald (subclass of `invgauss_gen`, calls `invgauss._pdf(x, 1.0)` etc. with mu = 1, so it inherits all mu = 1 findings, in particular F1); helpers: `_norm_logcdf`, `_norm_logsf` (= `sc.log_ndtr`); special functions: `sc.log_ndtr`, `scu._invgauss_ppf`, `scu._invgauss_isf` (Boost `inverse_gaussian` quantile)
- **Reference:** `refs/invgauss_ref.py`, 400 digits (ppf grid run at 100 digits to keep condition numbers affordable, see below); self-checks: PASS (400 digits). At 100 digits only the isf reference fails the precision check (`1 - q` is not exact for q = 1e-200), so isf was measured at 400 digits. Runs were split by method (`--methods pdf,logpdf,cdf,logcdf,sf,logsf`; `--methods ppf --dps 100`; `--methods isf --params "0.14546264555347513;1;1000"`) because a full run crashes on isf (F4) and ppf condition numbers at 400 digits took ~2.7 h. Runs with `--plot` exited with code 127 without output, so no error plot was produced. For mu = 1e-310 the ppf/isf reference inversion fails its self-check; ppf/isf were not measured at that mu.
- **Parameter sets:** (0.14546264555347513), (1), (1e-3), (1e-8), (1e-20), (1e3), (1e8), (1e20); supplementary run `--params "1e60;1e-310"` for pdf ... logsf; isf grid only on (0.14546264555347513), (1), (1e3) because isf raises for the others (F4)
- **Worst finding:** loss in pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf

## Summary

All methods are accurate in the bulk for moderate mu (about 0.01 to 1e3).
The cdf family is built from `a = log_ndtr(z1)` and `b = 2/mu + log_ndtr(-z2)`;
`log1p(-exp(b - a))` (logsf, sf) and `a + log1p(exp(b - a))` (logcdf) cancel
whenever sf is much smaller than Phi(-z1), i.e. in the right tail and in the
whole range x < mu**2 for large mu: errors grow like x/mu ulp, reach 5e7 ulp
for mu = 1e3 and give sf = 0, logsf = -inf and a positive logcdf for mu >= 1e20.
The Boost quantiles are unreliable: isf returns values like 4.4e34 (true 443.8)
in the far right tail for every mu, ppf returns 1.1e248 in the far left tail
for mu around 0.01-0.4, both return nan for mu < ~3e-3 so that the generic
root finder (absolute `xtol = 1e-14`) takes over (ppf 0.0 for mu = 1e-20), and
isf then raises `OverflowError` for small q. pdf and logpdf lose everything
where `x**3` or `(x/mu)**2` overflow, and the cdf family returns nan for
mu < 1.1e-308 where `2/mu` overflows.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | sf, logsf, logcdf | right tail x >> mu (any mu), and x < ~mu**2 for large mu | `log1p(-exp(b - a))` / `a + log1p(exp(b - a))` cancel (patterns 1, 5) | sf 0.0, logsf -inf, logcdf > 0 | loss |
| F2 | isf (and ppf for q > 0.5) | far right tail, small q, all mu | Boost `_invgauss_isf` does not converge, returns its search bound | 4.4e34 for 443.76 | loss |
| F3 | ppf | far left tail, mu in ~[0.01, 0.4] | Boost `_invgauss_ppf` does not converge | 1.1e248 for 7.2e-4 | loss |
| F4 | isf | small q with Boost nan (mu <= ~2e-3, mu >= ~1e8) | fallback `_ppf(1 - q)` hits `_invgauss_ppf(1.0)`, Boost raises | OverflowError | loss |
| F5 | ppf, isf | mu < ~3e-3 (Boost returns nan everywhere) | generic brentq with absolute `xtol = 1e-14`; isf as `ppf(1 - q)` (pattern 8) | ppf 0.0 for 1e-20 | loss |
| F6 | logpdf | x/mu > 1.34e154 | `(x/mu - 1)**2` overflows (pattern 3) | -inf for -5e99 | loss |
| F7 | pdf | x > 5.6e102 (pdf normal only for mu > ~1e52) | `x**3.0` overflows (pattern 3) | 0.0 for 1.3e-155 | loss |
| F8 | cdf, logcdf, sf, logsf | mu < 1.1e-308 | `2 / mu` overflows, `inf + (-inf)` (pattern 3) | nan for 0.5 | loss |
| F9 | logcdf | tiny mu, x a fraction of mu, &#124;logcdf&#124; > ~1e16 | absolute error eps*&#124;a&#124; of `b - a` exponentiated (pattern 5) | inf for -4.99e22 | loss |
| F10 | cdf, logcdf, sf, logsf | small mu, x near mu | `2/mu` cancels inside b (pattern 5) | 1.07e4 ulp | moderate |

### F1: sf, logsf and logcdf cancel when sf << Phi(-z1)

- **Code:** `_logsf`, lines 5092-5095:
  `a = _norm_logsf(fac*(x/mu - 1)); b = 2/mu + _norm_logcdf(-fac*(x/mu + 1)); return a + np.log1p(-np.exp(b - a))`;
  `_sf` = `exp(_logsf)` (line 5098); `_logcdf`, lines 5086-5089: `a + np.log1p(np.exp(b - a))`.
- **Mechanism:** sf = Phi(-z1) - exp(2/mu) Phi(-z2) with z1 = (x/mu - 1)/sqrt(x),
  z2 = (x/mu + 1)/sqrt(x). In the right tail (x >> mu) the ratio
  exp(b - a) is about (x/mu - 1)/(x/mu + 1) = 1 - 2mu/x, and `log1p(-exp(b - a))`
  cancels (pattern 1). Both a and b are of size x/(2 mu**2) with relative
  error eps, so the absolute error of b - a, divided by 1 - exp(b - a) ~ 2mu/x,
  gives an error of about x/mu ulp in sf (cond of sf is only ~x/(2 mu**2)).
  For large mu the same happens already at moderate x: for x << mu**2,
  z1 ~ -1/sqrt(x), z2 ~ 1/sqrt(x), so Phi(-z1) ~ 0.5 ~ exp(2/mu) Phi(-z2) and
  the difference (~0.8/sqrt(x)) is rounded away; once exp(b - a) rounds to 1,
  sf = 0 and logsf = -inf. logcdf is about -sf there and is computed as
  `a + log1p(exp(b - a))` with a ~ -Phi(-z1): the same cancellation, and when
  it fails completely the result has the wrong sign (logcdf > 0, cdf > 1).
- **Regime:** relative error about x/mu ulp for x >> mu (e.g. 5e7 ulp at
  mu = 1e3, x = 1e9; 3.3e5 ulp at mu = 1, x = 1e3); for mu >= ~1e13 total loss
  in the right tail; for mu = 1e60 sf = 0 already at x = 1e40.
  wald (mu = 1) gets up to 3.3e5 ulp in sf at x = 1e3.
- **Reproducer:** `python accuracy.py refs/invgauss_ref.py --at logsf 1e40 1e20`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1e40, 1e20)` | 0.0 | 1.666309411753726e-21 | inf | 1.45 | loss |
  | `logsf(1e40, 1e20)` | -inf | -47.84367570503361 | inf | 0.0304 | loss |
  | `logcdf(1e40, 1e20)` | 2.7755575615628914e-17 | -1.666309411753726e-21 | inf | 1.45 | loss |
  | `sf(1e40, 1e60)` | 0.0 | 7.978845608028654e-21 | inf | 0.5 | loss |
  | `logsf(1e30, 1e20)` | -34.79089839949923 | -34.764580280725326 | 3.4e12 | 0.0144 | severe |
  | `sf(1e19, 1e8)` | 1.7864223337137606e-230 | 1.7922589638433318e-230 | 1.5e13 | 501 | severe |
  | `sf(1e9, 1e3)` | 1.7940521203370445e-225 | 1.7940521003996762e-225 | 5.0e7 | 501 | severe |
  | `logcdf(1e9, 1e3)` | -1.794052128286454e-225 | -1.7940521003996762e-225 | 7.0e7 | 501 | severe |
  | `logsf(1e9, 1e3)` | -517.4971691078609 | -517.4971691189739 | 9.67e4 | 0.969 | moderate |
  | `sf(1e3, 1)` | 4.869434437047494e-222 | 4.8694344366891734e-222 | 3.3e5 | 501 | moderate |
  | `sf(21.159381251415937, 0.14546264555347513)` | 1.1641675805255685e-218 | 1.1641675805402133e-218 | 5.7e4 | 501 | moderate |

- **Fix direction:** evaluate the difference without cancellation, e.g. via the
  scaled complementary error function: sf = phi-scaled form
  `exp(-z1**2/2) * (erfcx(z1/sqrt2) - erfcx(z2/sqrt2)) / 2` (the exponents
  of both terms are equal), with `erfcx(u) - erfcx(v)` expanded/series-summed
  when u ~ v; logcdf = `log1p(-sf)` where sf is small.

### F2: Boost isf returns its search bound in the far right tail

- **Code:** `_isf`, line 5116: `isf = scu._invgauss_isf(x, mu, 1)`; `_ppf`,
  line 5108 for q > 0.5 (`_invgauss_isf(1 - x)`).
- **Mechanism:** Boost's `quantile(complement(inverse_gaussian))` emits
  "Unable to locate solution in a reasonable time ... Current best guess is"
  and returns the current bracket value (e.g. 4.4e34, 1.1e248) instead of
  nan; the value is finite, so the nan fallback (line 5119) is not triggered.
  The results are not even monotone in q (`isf(1e-20, 1e20)` = 1.16e33 <
  `isf(1e-17, 1e20)` = 1.5e33).
- **Regime:** total loss for q <= ~1e-76 (mu = 1, 1e3), q <= ~1e-51 (mu = 0.145), q <= 1e-300 ... 1e-20 (mu = 0.005 ... 1e20, spot checks); moderate errors for mu = 1e3 already from q = 3.5e-14.
- **Reproducer:** `python accuracy.py refs/invgauss_ref.py --at isf 1e-100 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `isf(1e-100, 1)` | 4.43905398835726e+34 | 443.76389884560626 | 4.5e47 | 0.00448 | loss |
  | `isf(1e-300, 1)` | 4.849712021959747e+233 | 1361.4454371385305 | 1.6e246 | 0.00147 | loss |
  | `isf(1.4677992676221335e-51, 0.14546264555347513)` | 1.118707184315428e+248 | 4.965054058170915 | 1.0e263 | 0.00842 | loss |
  | `isf(1e-300, 0.01)` | 1.118707184315428e+248 | 0.15618506183069128 | 3.2e264 | 0.00128 | loss |
  | `isf(1.778279410038923e-76, 1000)` | 118829797424.55945 | 317278502.071925 | 1.68e18 | 0.00624 | loss |
  | `isf(1e-20, 1e20)` | 1.1608049681869191e+33 | 1.903808702719755e+39 | 4.5e15 | 1.2 | severe |
  | `isf(5.1089697745070394e-64, 1000)` | 260493339.92046005 | 260493339.91770887 | 4.76e4 | 0.00759 | moderate |
  | `isf(3.4807005884285624e-14, 1000)` | 36750412.440663435 | 36750412.44070894 | 5.58e3 | 0.0505 | moderate |

  Where Boost does converge (q >= ~1e-64 for mu = 1 and 1e3), the result
  still carries up to 154 ulp (mu = 1, minor) and 4.8e4 ulp (mu = 1e3,
  moderate), growing with x/mu like F1; Boost's complemented cdf probably
  suffers the same Phi(-z1) - exp(2/mu) Phi(-z2) cancellation (mechanism
  inside Boost not inspected).

- **Fix direction:** detect non-convergence (treat the Boost warning/bound as
  nan) and invert `logsf` (after F1 is fixed) with a tail-aware starting
  value, e.g. x ~ 2 mu**2 (-log q) for the right tail.

### F3: Boost ppf returns 1.1e248 in the far left tail

- **Code:** `_ppf`, line 5106: `ppf = np.asarray(scu._invgauss_ppf(x, mu, 1))`.
- **Mechanism:** same non-convergence as F2 in Boost's lower quantile; the
  bound 1.118707184315428e+248 is returned as a finite value, so no fallback.
- **Regime:** mu = 0.01: q <= 1e-123; mu = 0.05: q <= 1e-50; mu = 0.1: q <= 1e-34;
  mu = 0.2: q <= 1e-22; mu = 0.4: q <= 1e-14; none for mu >= 0.5.
- **Reproducer:** `python accuracy.py refs/invgauss_ref.py --dps 100 --at ppf 1e-300 0.14546264555347513`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 0.14546264555347513)` | 1.118707184315428e+248 | 0.0007206802364432764 | 7.0e266 | 0.00144 | loss |
  | `ppf(2.15e-101, 0.14546264555347513)` | 1.118707184315428e+248 | 0.0021248272528774453 | 2.4e266 | 0.00424 | loss |
  | `ppf(1.21e-26, 0.14546264555347513)` | 0.6889953483046007 | 0.00784805325842498 | 3.9e17 | 0.0156 | loss |

- **Fix direction:** as F2; in the left tail log(F) ~ -(x - mu)**2/(2 mu**2 x)
  gives a starting value x ~ 1/(2 (-log q)).

### F4: isf raises OverflowError for small q when Boost returns nan

- **Code:** `_isf`, lines 5119-5120:
  `isf[i_nan] = super()._isf(x[i_nan], mu[i_nan])`; the generic `_isf` is
  `self._ppf(1.0 - q)`, which is invgauss's `_ppf`; for q < eps/2, `1.0 - q`
  is exactly 1 and line 5106 calls `scu._invgauss_ppf(1.0, mu, 1)`, which
  raises `OverflowError: ... probability parameter is 1, but must be < 1!`
  (the `errstate` context does not catch Boost policy errors).
- **Regime:** isf with q < 1.1e-16 for every mu where Boost's isf returns nan:
  mu <= ~2e-3 (all q) and mu >= ~1e8 (q around 1e-300). A whole array
  call fails if any element triggers it.
- **Reproducer:**
  `python -c "from scipy.stats import invgauss; invgauss.isf(1e-17, 1e-3)"`
  raises `OverflowError`; so do `invgauss.isf(1e-300, 1e8)` and
  `invgauss.isf(1e-300, 1e20)`. (The harness cannot evaluate these points;
  the isf grid was therefore run only for mu = 0.145, 1, 1e3.)
- **Fix direction:** do not route isf through `ppf(1 - q)`; use a dedicated
  root find on sf/logsf (and catch the Boost error).

### F5: Generic root-finding fallback for small mu (ppf, isf)

- **Code:** `_ppf`, lines 5109-5110 / `_isf`, lines 5119-5120: nan from Boost
  -> `rv_continuous._ppf` -> `brentq(..., xtol=self.xtol)` with the class
  default `xtol = 1e-14` (absolute, `_distn_infrastructure.py` line 1995);
  generic `_isf` = `_ppf(1 - q)` (pattern 8).
- **Mechanism:** Boost's quantiles return nan for every q when mu < ~3e-3
  (mu = 2e-3: nan, 3e-3: fine). The fallback stops at an absolute
  tolerance of 1e-14 on x, so the relative error is ~1e-14/mu: about 1e4 ulp
  for mu = 1e-3, 1e8-1e9 ulp for mu = 1e-8, and for mu = 1e-20 the result is
  0.0 or 8.9e-15 (pure tolerance noise). For isf the `1 - q` rounding adds a
  pattern-8 loss for small q.
- **Regime:** all q for mu < ~3e-3.
- **Reproducer:** `python accuracy.py refs/invgauss_ref.py --dps 100 --at ppf 0.57 1e-8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.01, 1e-20)` | 0.0 | 9.999999997673652e-21 | inf | 0 | loss |
  | `ppf(0.500000000001, 1e-20)` | 8.881784197019016e-15 | 1e-20 | 4.0e21 | 0 | loss |
  | `ppf(0.57, 1e-8)` | 1.0000176481018166e-08 | 1.0000176325719079e-08 | 7.0e7 | 0 | severe |
  | `ppf(0.9999999999999997, 1e-8)` | 1.000808628305774e-08 | 1.0008079783162146e-08 | 2.9e9 | 0 | severe |
  | `ppf(1.6155980984399093e-213, 0.001)` | 0.00038682816053386135 | 0.00038682816053153157 | 2.7e4 | 0.000909 | moderate |
  | `isf(1e-16, 0.001)` | 0.0012969969996452888 | 0.0012953124867375499 | 5.9e12 | 0.00376 | severe |

  (cond (arg) is the harness's numerical derivative of the quantile with
  respect to q; it is tiny here because the distribution is extremely
  concentrated around mu.)
- **Fix direction:** scale the problem (solve for x/mu, or use a relative
  tolerance), and fix Boost's failure for small mu (e.g. compute the quantile
  for the equivalent wald/standardized form, x = mu * y with shape 1/mu).

### F6: logpdf overflows in (x/mu - 1)**2

- **Code:** `_logpdf`, line 5079: `- (x/mu - 1)**2/(2*x)`.
- **Mechanism:** `(x/mu)**2` overflows for x/mu > 1.34e154 although the
  result, about -x/(2 mu**2), is finite (pattern 3).
- **Regime:** x/mu > 1.34e154 (all mu; e.g. x > 1.95e153 for mu = 0.145).
- **Reproducer:** `python accuracy.py refs/invgauss_ref.py --at logpdf 1e300 1e100`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e300, 1e100)` | -inf | -5e+99 | inf | 1 | loss |
  | `logpdf(2.0364770377486517e+153, 0.14546264555347513)` | -inf | -4.812232015556635e+154 | inf | 1 | loss |
  | `logpdf(1.4e+154, 1)` | -inf | -7e+153 | inf | 1 | loss |

- **Fix direction:** `-(x - mu)**2/(2*x*mu**2)` written as
  `-((x/mu - 1)/sqrt(x))**2/2` or `-(x/(2*mu**2) - 1/mu + 1/(2*x))`.

### F7: pdf overflows in x**3

- **Code:** `_pdf`, line 5076: `1.0/np.sqrt(2*np.pi*x**3.0)*np.exp(...)`.
- **Mechanism:** `x**3.0` overflows for x > 5.6e102, giving pdf = 0 although
  pdf ~ x**-1.5 is normal when x/(2 mu**2) is small (pattern 3).
- **Regime:** x > 5.6e102 and mu > ~1e52 (for smaller mu the true pdf underflows there).
- **Reproducer:** `python accuracy.py refs/invgauss_ref.py --at pdf 1e103 1e60`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e+103, 1e+60)` | 0.0 | 1.26156626101008e-155 | inf | 1.5 | loss |
  | `pdf(5e+102, 1e+60)` | 0.0 | 3.5682482323055423e-155 | inf | 1.5 | loss |
  | `pdf(1e+120, 1e+60)` | 0.0 | 2.4197072451914333e-181 | inf | 2 | loss |

- **Fix direction:** `exp(logpdf)` or `x**-1.5 / sqrt(2*pi)`.

### F8: cdf family returns nan for mu < 1.1e-308

- **Code:** `_logcdf` line 5088 and `_logsf` line 5094: `b = 2 / mu + _norm_logcdf(...)`.
- **Mechanism:** `2/mu` overflows to inf, and `log_ndtr` of the huge
  negative argument is -inf, so b = nan; every method built on it (cdf,
  logcdf, sf, logsf) returns nan (pattern 3). mu down to 5e-324 is in the
  documented domain.
- **Regime:** mu < 1.1e-308, all x.
- **Reproducer:** `python accuracy.py refs/invgauss_ref.py --at cdf 1e-308 1e-308`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-308, 1e-308)` | nan | 0.5 | inf | 8.0e153 | loss |
  | `cdf(1e-310, 1e-310)` | nan | 0.5 | inf | 8.0e154 | loss |
  | `cdf(2e-310, 1e-310)` | nan | 1.0 | inf | 0 | loss |

- **Fix direction:** combine `2/mu` with the leading `-z2**2/2` analytically
  (see F1 fix: the scaled-erfc form has no `2/mu` term).

### F9: logcdf returns inf in the far left tail for tiny mu

- **Code:** `_logcdf`, lines 5087-5089.
- **Mechanism:** a and b are both about -(x/mu - 1)**2/(2x) with the
  `2/mu` term cancelling in b; their difference is O(1) but carries an
  absolute error eps*|a|. When |a| > ~1e16 this error exceeds 700 and
  `exp(b - a)` overflows (or underflows), so logcdf = inf (pattern 5).
- **Regime:** x/mu not negligible (else z1 == z2 and b - a = 2/mu exactly)
  and |logcdf| > ~1e16, i.e. mu below ~1e-16; e.g. mu = 1e-20, x = 1e-23.
- **Reproducer:** `python accuracy.py refs/invgauss_ref.py --at logcdf 1e-23 1e-20`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logcdf(1e-23, 1e-20)` | inf | -4.990005e+22 | inf | 1 | loss |

- **Fix direction:** compute b - a analytically: b - a =
  log(Phi(-z2)/phi(z2)) - log(Phi(z1)/phi(z1)) (Mills ratios, via `erfcx`),
  since 2/mu - z2**2/2 + z1**2/2 = 0 exactly.

### F10: cdf family loses ~|2/mu| ulp near the mean for small mu

- **Code:** `_logcdf` / `_logsf`, lines 5088 / 5094: `b = 2/mu + log_ndtr(-z2)`.
- **Mechanism:** the two terms of b are each about 2/mu and cancel to
  O(log(1/mu)); b carries an absolute error of eps*2/mu, which enters the
  result through `exp(b - a)` (pattern 5). The error is only somewhat larger
  than the (already large) conditioning of the cdf with respect to x.
- **Regime:** mu <= ~1e-3, x near mu or in the left tail.
- **Reproducer:** `python accuracy.py refs/invgauss_ref.py --at cdf 1.00000001e-8 1e-8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1.00000001e-08, 1e-08)` | 0.5000598413428412 | 0.5000598413416574 | 1.07e4 | 7.98e3 | moderate |
  | `cdf(0.0005, 0.001)` | 6.339735243147693e-111 | 6.339735243149632e-111 | 1.38e3 | 751 | moderate |
  | `sf(0.002, 0.001)` | 3.1656625234058417e-111 | 3.1656625234045085e-111 | 1.9e3 | 752 | moderate |
  | `logcdf(0.002, 0.001)` | -3.1656625234058332e-111 | -3.1656625234045085e-111 | 1.88e3 | 752 | moderate |

  (cond (shapes) equals cond (arg) here: cdf(x; mu) is very sensitive to mu.)
- **Fix direction:** same as F9 (no explicit `2/mu` term).

## RuntimeWarnings

- pdf: "overflow encountered in power", "divide by zero encountered in divide",
  "invalid value encountered in multiply", underflow warnings. For x < ~1e-103
  `x**3` underflows to 0 and `1/sqrt(0) = inf` is multiplied by `exp(...) = 0`;
  the true pdf is far below the double range there, so the warnings are
  spurious for practical purposes but the value is nan rather than 0. For
  large x the overflow is F7.
- logpdf: "overflow encountered in square" (F6).
- cdf/logcdf/sf/logsf: "overflow encountered in divide/exp",
  "underflow encountered in exp" on accurate results in the tails (spurious);
  "divide by zero encountered in log1p" and "invalid value encountered in
  log1p/subtract/add" accompany the F1/F8 losses.
- ppf/isf: Boost "Unable to locate solution in a reasonable time ... Current
  best guess is %1%" (unformatted message) for F2/F3, and for all q when
  mu < ~3e-3 (F5, where the fallback result is then used).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, 8 parameter sets + mu = 1e60, 1e-310 (below x**3 overflow) | 480 ulp, within conditioning (cond 502) |
| logpdf | full grid below x/mu = 1.34e154 | 1.95 ulp in the bulk; tail errors within conditioning |
| cdf | left tail and bulk, mu >= 0.145 | 660 ulp at `cdf(0.001, 1)` (cond 500; minor, from exp(logcdf) with logcdf = -502) |
| logsf | left tail and bulk, mu >= 0.145 (right tail is F1) | 5.6 ulp |
| ppf | mu = 1, 1e3, 1e8, 1e20, all q on the grid (q > 0.5 goes through Boost isf of 1 - q >= 1.1e-16) | no point beyond conditioning (21 ulp for mu = 1; up to 8.8e14 ulp for mu = 1e20 at q -> 1, where cond is 6.0e15) |
| isf | bulk and q -> 1 for mu = 0.145, 1, 1e3 | 5.41 ulp (bulk), 1.94 ulp (q near 1) |

The small-mu cancellation of `2/mu` (hypothesis H3) is mostly within
conditioning with respect to x (e.g. `cdf(1e-20, 1e-20)`: 1.8e5 ulp, cond
8.0e9), because the distribution is very narrow; only F10 points exceed it.

## Suspected, not confirmed

- ppf for q > 0.5 calls `_invgauss_isf(1 - q)` (line 5108); F2 should
  therefore also appear in ppf for 1 - q below ~1e-20, which cannot be
  represented (1 - q >= 1.1e-16), so it is not reachable through ppf.
- Boost isf for mu >= 1e8 at q = 1e-300 returns nan and then raises (F4);
  how accurate Boost isf is for large mu and 1e-300 < q < 1e-20 could only
  be spot-checked (`isf(1e-20, 1e20)`: 4.5e15 ulp, severe; listed with F2).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| none | - | - | all eight methods are implemented |

## Shared code

- `scu._invgauss_ppf` / `scu._invgauss_isf` (Boost `inverse_gaussian`
  quantile, `scipy/special` boost ufuncs): non-convergence returns the search
  bound as a finite value (F2, F3), and nan for every q when mu < ~3e-3 (F5);
  `_invgauss_ppf(1.0)` raises OverflowError instead of returning inf (F4).
  These are Boost/special-function problems; wald uses the same code with
  mu = 1 (F2 affects wald.isf and wald.ppf for q > 0.5).
- `sc.log_ndtr` itself is accurate here; the problems are in how its results
  are combined.
- wald (`wald_gen(invgauss_gen)`) inherits F1, F2, F6, F7 at mu = 1.

## Existing GitHub issues

- PR #16790 (merged): ENH: stats.invgauss: use Boost implementation of ppf/isf - introduced the code behind F2, F3, F4 and F5. In the review thread of PR #13665, Boost's ppf round trip gave 26 nan results and was noted to struggle at very small mu (F5).
- PR #13665 (closed, superseded by #16790): ENH: Implement ppf and isf for invgauss - reported far-tail isf errors (`isf(1e-16, mu=1.05)`), the same tail as F2.
- PR #13616 (merged): earlier fix of cdf for very small mu - relates to F8 and F10. Those regimes remain.
- #13666 (closed): pdf with mu = inf returns nan - different regime from F6 and F7.
- F1 and F9: none found

## Verification

Verifier reference: `scratch/invgauss/verify/verify.py` (route independent of the analyst's: second cdf term written as exp(-z1**2/2) * U(1/2, 1/2, z2**2/2)/(2 sqrt(pi)) via the exact identity 2/mu - z2**2/2 = -z1**2/2, so no exp(2/mu); right-tail sf = exp(-z1**2/2) (M(z1) - M(z2)); logcdf = log1p(-sf) in the right tail; quantiles by geometric bisection on log cdf / log sf with cdf(x_ref) == q checked). Every value agrees to 30 digits at 100 and 200 dps; sf at moderate points corroborated by `mp.quad` of the pdf (`quadcheck2.py`, agreement 4e-13).

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `sf(1e40, 1e20)`; `logsf(1e40, 1e20)`; `logcdf(1e40, 1e20)` | 0.0; -inf; 2.7755575615628914e-17 | 1.6663094117537259e-21; -47.84367570503361; -1.6663094117537259e-21 | verified | also `sf(1e9, 1e3)` 5.0e7 ulp, `logcdf(1e9, 1e3)` 7.0e7 ulp, `sf(1e19, 1e8)` 1.47e13 ulp, `logsf(1e30, 1e20)` 3.41e12 ulp, `sf(1e3, 1)` 3.31e5 ulp, `sf(1e40, 1e60)` = 0; mechanism matches lines 5089/5095 |
| F2 | `isf(1e-100, 1)` | 4.43905398835726e+34 | 443.76389884560625 | verified | also `isf(1e-300, 0.01)`, `isf(1.778e-76, 1000)` (1.68e18 ulp), `isf(1e-20, 1e20)` (4.5e15 ulp) reproduce exactly; Boost returns a finite wrong value so line 5119 fallback is not triggered |
| F3 | `ppf(1e-300, 0.14546264555347513)` | 1.118707184315428e+248 | 7.2068023644327642e-4 | verified | `ppf(1.21e-26, 0.145...)` with q exactly 1.21e-26 gives 0.6946501493720032 vs 7.8478985951593352e-3 (3.9e17 ulp, loss); the report's q in that row is a rounded grid value, hence slightly different numbers |
| F4 | `isf(1e-17, 1e-3)` | OverflowError | 1.3063924839828634e-3 | verified | `_invgauss_isf(1e-17, 1e-3)` = nan, `_invgauss_ppf(1.0, 1e-3)` raises the quoted Boost OverflowError; mechanism matches lines 5119-5120 |
| F5 | `ppf(0.57, 1e-8)` | 1.0000176481018166e-08 | 1.0000176325719078e-08 | verified | 6.99e7 ulp; `ppf(0.01, 1e-20)` = 0.0 (ref 9.9999999976736516e-21), `isf(1e-16, 0.001)` 5.86e12 ulp; Boost nan at mu = 2e-3, finite at 3e-3; xtol = 1e-14 confirmed |
| F6 | `logpdf(1e300, 1e100)` | -inf | -5.0000000000000001e+99 | verified | also `logpdf(1.4e154, 1)` = -inf (ref -7.0e153); line 5079 |
| F7 | `pdf(1e103, 1e60)` | 0.0 | 1.26156626101008e-155 | verified | line 5076 `x**3.0` overflow |
| F8 | `cdf(1e-308, 1e-308)` | nan | 0.5 | verified | also `cdf(2e-310, 1e-310)` = nan (ref 1.0); `2 / mu` overflow, lines 5088/5094 |
| F9 | `logcdf(1e-23, 1e-20)` | inf | -4.9900050000000002e+22 | verified | |
| F10 | `cdf(1.00000001e-8, 1e-8)` | 0.5000598413428412 | 0.50005984134165737 | verified | 1.07e4 ulp; also `sf(0.002, 0.001)` 1.9e3 ulp |

Verified by dist-accuracy-verifier: 10 of 10 verified.
