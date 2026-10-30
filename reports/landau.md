# landau: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `landau_gen`, lines 5987-6077
- **Implemented:** pdf, cdf, sf, ppf, isf (each a one-line call into `scipy.special._ufuncs._landau_*(x, 0, 1)`, which wrap Boost.Math `landau_distribution`)
- **Inherited (not measured):** logpdf = log(pdf); logcdf, logsf = generic median-split forms (log(cdf) / log1p(-sf) and log(sf) / log1p(-cdf))
- **Related:** none; helpers: none; special functions: scu._landau_pdf, scu._landau_cdf, scu._landau_sf, scu._landau_ppf, scu._landau_isf
- **Reference:** `refs/landau_ref.py`, 100 digits (integrals use log10|y| + 10 extra digits internally); self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** none beyond conditioning

## Summary

All five implemented methods (pdf, cdf, sf, ppf, isf) are accurate to the
limit set by conditioning everywhere they were measured: x from -5.1 (where
pdf/cdf reach the subnormal range) to 1e300, and q from 1e-308 to 1 - 2.2e-16.
In the bulk and the whole right tail, errors are at most 1.4 ulp for pdf, cdf
and sf and 3.1 ulp for ppf and isf. The only errors above 8 ulp are in the
double-exponential left tail of pdf and cdf (up to 914 ulp at x = -5.1), and
all of them are within cond (arg) of about 1.2e3 to 5.7e3 there. So there are
no findings.

There is one documentation issue, not an accuracy problem. The docstring pdf
`f(x) = 1/pi int_0^inf exp(-t log t - x t) sin(pi t) dt` is Landau's original
standardization, but SciPy/Boost compute the Wikipedia/Boost form
`f(x) = 1/pi int_0^inf exp(-t) cos(x t + (2/pi) t log t) dt`. The two are
related by `f_scipy(x) = (pi/2) f_doc(pi x/2 + log(pi/2))`. For example, the
docstring integral at x = 0 is 0.178854, while `landau.pdf(0)` is 0.262240.
The tail is `sf ~ 2/(pi x)` in SciPy and `1/x` for the docstring formula. The
"loc = mu + 2c/pi log c" remark in the docstring refers to the Boost form.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| - | - | none beyond conditioning | - | - | - |

No point was flagged beyond conditioning (the tool's "Points beyond
conditioning" list is empty).

## RuntimeWarnings

None recorded for any measured point (pdf, cdf, sf over 160 x values; ppf and
isf over 67 q values).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | left tail x in [-5.1, -0.5] (double-exponential decay, pdf down to 4.5e-306), bulk, right tail to x = 1e154 (pdf ~ 2/(pi x^2) is subnormal beyond) | 914 ulp at x = -5.1 (cond 5.65e3); bulk 1.09 ulp; right tail 0.94 ulp |
| cdf | left tail to x = -5.05 (cdf 6.0e-286; at -5.1 it is subnormal), bulk, right tail to 1e300 | 886 ulp at x = -4.75 (cond 3.04e3); bulk 1.37 ulp; right tail 0 ulp |
| sf | sf -> 1 (left tail), bulk, right tail sf ~ 2/(pi x) to x = 1e300 | 0.94 ulp |
| ppf | q from 1e-308 (x = -5.10) to 1 - 2.2e-16 (x ~ 2.9e15), including q = 1 - 1e-6 ... 1 - 1e-14 | 3.14 ulp (q = 0.36, root near -0.0199); low tail 1.58 ulp |
| isf | q from 1e-308 (x ~ 6.4e307) to 1 - 2.2e-16 (x = -3.16) | 3.14 ulp (q = 0.64, root near -0.0199); tails 1.27 ulp |

Worst left-tail points (all `ok` within conditioning):

| call | scipy | reference | ulp | cond | severity |
|---|---|---|---|---|---|
| `pdf(-5.1)` | 4.481544073170816e-306 | 4.4815440731699085e-306 | 914 | 5.65e3 | ok |
| `pdf(-4.75)` | 1.5396287883657162e-176 | 1.5396287883654146e-176 | 882 | 3.04e3 | ok |
| `cdf(-4.75)` | 2.403173486399823e-179 | 2.4031734863993503e-179 | 886 | 3.04e3 | ok |
| `cdf(-4.25)` | 6.3572493529771024e-83 | 6.357249352977611e-83 | 360 | 1.24e3 | ok |

In the left tail, f ~ exp(-exp(-pi x/2 - 1 - log(pi/2))), so
cond (arg) = |x f'/f| ~ (pi/2)|x| e^{-y-1} grows double-exponentially. A
half-ulp change in x already moves the result by thousands of ulp, and the
errors observed are about 1/6 of that. Reproducer:
`python accuracy.py refs/landau_ref.py --at pdf -5.1`.

Hypotheses that were tested deliberately and did not show:
- Boost uses piecewise rational approximations on x intervals. The grid covers
  every integer in [-10, 10], the extra points -0.5 ... -5.1 in steps of 0.25
  to 0.5, and powers of ten up to 1e300. No jump in accuracy was seen at any of
  them.
- Neither ppf near q -> 1 nor isf near q -> 1 rounds the complement
  (1 - q down to 2.2e-16 measured: at most 1.27 ulp).
- The far right tail of pdf (~ 2/(pi x^2)) and sf (~ 2/(pi x)) holds up to
  the subnormal limit or 1e300. isf(1e-300) = 6.366197723675814e+299 is exact
  to 1 ulp.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | Left tail: log of Boost's tail form, i.e. -s + log(rational) with s ~ exp(-pi x/2 - 1 - log(pi/2)) computed without exponentiating, using the steepest-descent saddle s0 = exp(-y-1). Right tail: log(2/pi) - 2 log x + log1p(correction). | pdf underflows. x < ~-5.13: `logpdf(-5.2)` = -inf, true -823.0788122202633. x > ~1.3e154: `logpdf(1e200)` = -inf, true -921.4856199029077. |
| logcdf | log(cdf) for x < median | Left-tail log form as for logpdf (log F_L(y) ~ -s0 + O(log s0)) | cdf underflows for x < ~-5.13: `logcdf(-5.2)` = -inf, true -830.2475576160388 |
| logsf | log(sf) above the median, log1p(-cdf) below | none needed: sf ~ 2/(pi x) never underflows for finite x, and log1p(-cdf) keeps the left tail (`logsf(-4)` = -1.2148504477782128e-56, ref -1.2148504477782299e-56) | does not fail |

Boost.Math has no log-pdf/log-cdf for landau. A fix would need either
a SciPy-side tail asymptotic or new Boost functionality.

## Shared code

none. The scipy.special `_landau_*` ufuncs (Boost.Math) are accurate within
conditioning over the full range, and no other scipy.stats distribution
calls them.

## Existing GitHub issues

not checked

## Verification

pending
