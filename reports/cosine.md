# cosine: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `cosine_gen`, lines 1699-1751
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf, logsf (rv_continuous defaults: `log(_cdf)` / `log1p(-_sf)` split at the median, and the mirror for logsf)
- **Related:** none; helpers: none; special functions: scu._cosine_cdf, scu._cosine_invcdf
- **Reference:** `refs/cosine_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in pdf and logpdf (near both ends of the support); loss in cdf/sf only at the single double endpoint x = -/+np.pi

## Summary

cdf, sf, ppf and isf (all done by `scu._cosine_cdf` / `scu._cosine_invcdf`)
are accurate to within 1.7 ulp everywhere, including the cubic tails near
+-pi and q down to 1e-305. pdf and logpdf use `1 + cos(x)`, which cancels as
|x| -> pi. The error grows like 1/(pi - |x|)**2 and is well beyond
conditioning once pi - |x| < ~0.01. The result is exactly 0 (pdf) or -inf
(logpdf) once pi - |x| < ~1.5e-8, where `cos(x)` rounds to -1. The public
cdf and sf also return 0 at the double endpoint -np.pi (resp. +np.pi). That
point lies strictly inside the true support, and the 0 comes from the generic
open-support mask, not from the cosine code.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf, logpdf | pi - \|x\| < ~0.01; exactly 0 / -inf for pi - \|x\| < ~1.5e-8 | `1 + cos(x)` cancels near x = +-pi (patterns 1, 5) | pdf 0.0 for 7.96e-18, logpdf -inf for -39.37 | loss |
| F2 | cdf, sf | x = -np.pi (cdf), x = +np.pi (sf) only | generic `_open_support_mask` treats fl(pi) as the endpoint (pattern 11 / 7) | 0.0 for 4.87e-50 | loss |

### F1: pdf and logpdf cancel in 1 + cos(x) near the ends of the support

- **Code:** `_pdf`, line 1725: `1.0/2/np.pi*(1+np.cos(x))`; `_logpdf`,
  lines 1728-1731: `c = np.cos(x)`, `np.log1p(c) - np.log(2*np.pi)` with
  `fill_value=-np.inf` where `c == -1`.
- **Mechanism:** near |x| = pi, cos(x) = -1 + d**2/2 with d = pi - |x|. The
  rounding error of `np.cos(x)` is about eps/2 absolute, so `1 + cos(x)`
  (and `log1p(c)`) has relative error ~eps/d**2 (patterns 1 and 5). The
  conditioning with respect to x is only ~2|x|/d. Once d**2/2 < eps/2
  (d < ~1.5e-8), `cos(x)` rounds to exactly -1. Then the pdf is 0 and
  logpdf takes the `fill_value=-inf` branch, although the true pdf is
  d**2/(4 pi) (about 1.2e-33 even at x = fl(pi)).
- **Regime:** both tails symmetric in x. Moderate at d = 1e-2 (1.4e3 ulp),
  severe from d = 1e-4 (2.4e7 ulp), loss for d < ~1.5e-8. logpdf is
  minor at d = 1e-2 (117 ulp), moderate at d = 1e-3, severe from d = 1e-4,
  loss for d < ~1.5e-8. Bulk (|x| <= pi - 0.1) is within 1.03 ulp.
- **Reproducer:** `python accuracy.py refs/cosine_ref.py --at pdf 3.141592653589793`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(3.141592653589793)` | 0.0 | 1.1934709143053064e-33 | inf | 5.1e+16 | loss |
  | `pdf(3.141592643589793)` | 0.0 | 7.957747252777404e-18 | inf | 6.3e+08 | loss |
  | `pdf(3.141592609180872)` | 1.5902773407317583e-16 | 1.5693889146290697e-16 | 6.0e+13 | 1.4e+08 | severe |
  | `pdf(3.141582653589793)` | 7.957747813021719e-12 | 7.957747154827627e-12 | 3.7e+08 | 6.3e+05 | severe |
  | `pdf(3.141492653589793)` | 7.95774710623179e-10 | 7.957747148016388e-10 | 2.4e+07 | 6.28e+04 | severe |
  | `pdf(3.1405926535897932)` | 7.957746491324551e-08 | 7.95774649144939e-08 | 7.07e+04 | 6.28e+03 | moderate |
  | `pdf(3.1315926535897933)` | 7.95768084025849e-06 | 7.957680840256049e-06 | 1.38e+03 | 626 | moderate |
  | `logpdf(3.141592653589793)` | -inf | -75.80844227238724 | inf | 6.8e+14 | loss |
  | `logpdf(3.141592643589793)` | -inf | -39.37238572253603 | inf | 1.6e+07 | loss |
  | `logpdf(3.141592609180872)` | -36.37745305875023 | -36.39067517066366 | 1.6e+12 | 3.9e+06 | severe |
  | `logpdf(3.141492653589793)` | -20.951704996999126 | -20.95170499174832 | 1.1e+06 | 3e+03 | severe |
  | `logpdf(3.1405926535897932)` | -16.346534888282562 | -16.346534888266874 | 4.32e+03 | 384 | moderate |
  | `logpdf(3.1315926535897933)` | -11.741372952285463 | -11.741372952285769 | 117 | 53.3 | minor |

  The same values occur at the mirrored negative arguments. In total 24 loss
  points in pdf and 24 in logpdf (12 per tail).
- **Fix direction:** use the half-angle form `1 + cos(x) = 2 cos(x/2)**2`:
  `pdf = cos(x/2)**2/pi`, `logpdf = 2*log(cos(x/2)) - log(pi)`. `x/2` is
  exact and libm's `cos` near pi/2 is accurate. A candidate with exactly
  these two lines (`scratch/cosine/cand.py`) gives max 1.35 ulp (pdf) and
  0.86 ulp (logpdf) over the full grid, including `pdf(fl(pi))`
  = 1.1934709143053064e-33 (0 ulp).

### F2: cdf and sf are 0 at the double support endpoint

- **Code:** `_cdf`, line 1734 (`scu._cosine_cdf(x)`) is correct here:
  `scu._cosine_cdf(-np.pi)` = 4.871934450264861e-50. The 0 comes from
  the generic `rv_continuous.cdf` / `.sf` (`_distn_infrastructure.py`
  lines 2164 and 2247): `cond1 = self._open_support_mask(x)` excludes
  `x == a`, so x = a = -np.pi gets the default output 0. The same happens for
  sf at x = b = +np.pi.
- **Mechanism:** the support is declared as `a=-np.pi, b=np.pi`, and
  fl(pi) = pi - 1.22e-16. So x = -np.pi is strictly inside the true support,
  where cdf = (x + pi + sin x)/(2 pi) ~ (1.22e-16)**3/(12 pi) = 4.87e-50
  (patterns 7/11: the threshold is a rounded constant).
- **Regime:** exactly one point per method. The tool rates it loss, but
  cond (arg) is 7.7e16, so a 1-ulp change of x changes the result
  completely. The neighbouring double `-np.pi + spacing` is accurate.
- **Reproducer:** `python accuracy.py refs/cosine_ref.py --at cdf -3.141592653589793`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-3.141592653589793)` | 0.0 | 4.871934450264861e-50 | inf | 7.7e+16 | loss |
  | `sf(3.141592653589793)` | 0.0 | 4.871934450264861e-50 | inf | 7.7e+16 | loss |

- **Fix direction:** low priority. Accept 0 as the defined value at the
  double endpoint, or (generic) evaluate `_cdf` on the closed support mask.
  Same issue as anglit F1.

## RuntimeWarnings

None recorded by the harness for any measured method. `_logpdf` uses
`xpx.apply_where`, so `log1p(-1)` is never evaluated at `cos(x) == -1`.

## Checked and fine

| Method | Regime checked | Max error |
|---|---|---|
| pdf | bulk \|x\| <= pi - 0.1, small \|x\| down to 1e-300 | 1.03 ulp |
| logpdf | bulk \|x\| <= pi - 0.1, small \|x\| | 0.874 ulp |
| cdf | full grid incl. x = -pi + k ulp (k = 1 ... 1e8) and pi - d (d = 1e-14 ... 0.5) | 1.69 ulp (except F2 point) |
| sf | full grid, mirror of cdf | 1.69 ulp (except F2 point) |
| ppf | q in [1e-305, 1 - 3e-16], q = 0.5 +- 1e-16 ... 0.1 | 1.68 ulp |
| isf | same q grid | 1.68 ulp |

The hypothesis that the C helpers lose accuracy in the cubic tails (cdf near
-pi, ppf for tiny q) was tested with points down to cdf ~ 4.8e-48 and
q = 1e-305 and did not show.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | rv_continuous `_logcdf`: `log(_cdf)` for x < median (0), `log1p(-_sf)` otherwise | not needed | none found: `logcdf(3.14)` = -1.0716002362506411e-10 vs -1.0716002362506409e-10, 1.09 ulp (cond 5.91e3), ok |
| logsf | rv_continuous `_logsf`: `log(_sf)` for x > 0, `log1p(-_cdf)` otherwise | not needed | none found: `logsf(-3.14)` 1.09 ulp, ok. The extra `_ppf(0.5)` call per evaluation is the only cost. |

## Shared code

`scu._cosine_cdf` and `scu._cosine_invcdf` are accurate in every measured
regime. F2 is in the generic `rv_continuous.cdf`/`sf` support mask, which
affects every distribution whose support bound is a rounded constant
(anglit, cosine, ...). It is not specific to cosine.

## Existing GitHub issues

- #18382 (merged PR): TST: stats.cosine: modify test to silence failure - relates to F1; only loosened `test_cosine_logpdf_endpoints`, no library change. In the thread, `1 + cos(x)` near +-pi is named as catastrophic cancellation, with a ufunc rewrite of pdf/logpdf suggested; a follow-up issue was proposed but none is linked.
- #15195 (closed): Tracking issue for distributions that warn and/or return nan at support endpoints - background for F1/F2 (cosine endpoint handling via #15196); does not cover the near-endpoint accuracy loss.

(`gh` unavailable; web search of scipy/scipy for cosine accuracy, then the two threads read directly.)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(3.141592653589793)` | 0.0 | 1.1934709143053063e-33 | verified | ref `cos(x/2)**2/pi` at 100/200 digits; loss; same at -x |
| F1 | `pdf(3.141592643589793)` | 0.0 | 7.957747252777404e-18 | verified | loss |
| F1 | `pdf(3.141592609180872)` | 1.5902773407317583e-16 | 1.5693889146290697e-16 | verified | 5.99e13 ulp, severe |
| F1 | `pdf(3.141492653589793)` | 7.95774710623179e-10 | 7.9577471480163883e-10 | verified | 2.36e7 ulp, severe |
| F1 | `logpdf(3.141592653589793)` | -inf | -75.808442272387246 | verified | ref `log(cos(x/2)**2/pi)`; loss |
| F1 | `logpdf(3.141592643589793)` | -inf | -39.372385722536028 | verified | loss |
| F1 | `logpdf(3.141592609180872)` | -36.37745305875023 | -36.390675170663663 | verified | 1.64e12 ulp, severe; mechanism matches lines 1725 / 1728-1731 |
| F2 | `cdf(-3.141592653589793)` | 0.0 | 4.8719344502648614e-50 | verified | ref `mp.quad` of pdf on [-pi, x]; `_cdf(-np.pi)` itself returns 4.871934450264861e-50, so the 0 comes from `_open_support_mask` (infrastructure line 2164), as stated |
| F2 | `sf(3.141592653589793)` | 0.0 | 4.8719344502648614e-50 | verified | ref `1 - quad`; mask at line 2247; cond 7.7e16, so the low-priority rating is fair |

Verified by dist-accuracy-verifier: 2 of 2 verified.
