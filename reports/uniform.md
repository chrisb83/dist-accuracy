# uniform: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `uniform_gen`, lines 10931-11127 (methods at 10965-10972)
- **Implemented:** pdf, cdf, ppf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf) / log1p(-sf) (rv_continuous switches at the median), sf = 1 - cdf, logsf = log(sf) / log1p(-cdf) (switch at the median), isf = ppf(1 - q)
- **Related:** none; helpers: `_remove_optimizer_parameters` (fit only); special functions: none
- **Reference:** `refs/uniform_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes); loc/scale probed separately with explicit points (`scratch/uniform/locscale.py`, 800 digits)
- **Worst finding:** none beyond conditioning

## Summary

The standardized methods are identities (`_pdf = 1.0*(x == x)`, `_cdf = x`,
`_ppf = q`) and the harness measures 0 ulp at every point of the default grid plus
the added edge points (x down to 2.2e-308, x = 1 - 2**-53, q up to 1 - 2**-53).
The inherited generic methods are accurate as well. `1 - x` is exact for
x >= 0.5 (Sterbenz) and correctly rounded otherwise. The rv_continuous
logcdf/logsf defaults use `log1p` on the far side of the median.
The generic loc/scale wrappers (`(x - loc)/scale` and `q*scale + loc`) add at
most about one rounding each. The only large relative error found is
`ppf` when `loc + q*scale` cancels to near 0, which lies within the
conditioning with respect to q. There are no findings.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|

None. Every measured point is 0 ulp (severity ok).

## RuntimeWarnings

- `cdf`/`sf` with loc/scale near the overflow limit, for x outside the
  support: `uniform.cdf(1.5e308, loc=-1e308, scale=1.7e308)` warns
  "overflow encountered in subtract" because `x - loc` overflows. The results
  (cdf 1.0, sf 0.0) are correct, so the warning is spurious. Inside the support,
  `x - loc <= scale` cannot overflow.
- `pdf` with scale < ~5.6e-309 (subnormal): `uniform.pdf(5e-310, loc=0, scale=1e-309)`
  warns "overflow encountered in divide" and returns inf. The true density 1/scale
  (about 1.0e309) is also beyond the double range, so inf is the correct result. `logpdf` at the
  same point returns 711.4987937351601 (0.04 ulp).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full default grid, support edges 0 and 1, x = 2.2e-308 | 0 ulp |
| cdf | full default grid, x = 2.2e-308, 1e-300, 1 - 2**-52, 1 - 2**-53 | 0 ulp |
| ppf | full default q grid, q = 2.2e-308, 1 - 2**-53, 1 - 2**-52 | 0 ulp |
| sf, logsf, isf (inherited, spot checks) | x = 1e-300, 1e-17, 0.5, 1 - 2**-52; q = 1e-300, 1e-17, 1 - 2**-52 | 0.32 ulp (`logsf(1 - 2**-52)`) |
| loc/scale, cdf/sf | loc = 1e16 (scale 1 and 4), loc = 1e-300 with scale = 1e-299, scale = 1e-309 (subnormal), scale = 1.7e308 | 0.35 ulp |
| loc/scale, pdf/logpdf | scale = 1.7e308, scale = 1e-309 | 1.21 ulp (`pdf(5e307, scale=1.7e308)`, result 5.9e-309 is near the normal limit) |
| loc/scale, ppf/isf | loc = 1e16; loc = -1e308 with scale = 1.7e308; `isf(1e-17, loc=2, scale=3)`; `ppf(0.3, loc=-1, scale=3)` | 2.5 ulp (`ppf(0.3, loc=-1, scale=3)` = -0.10000000000000009 vs -0.10000000000000003) |

Context for loc/scale: `ppf(q, loc, scale)` is computed as `q*scale + loc`.
When the result is near 0, the single rounding of `q*scale` is amplified:
`uniform.ppf(0.3333333333333333, loc=-1, scale=3)` returns 0.0, but the true value is
-5.551115123125783e-17. This is ill-conditioned with respect to q:
cond = |q*scale/ppf| = 1.8e16, so under the harness convention the error is
within conditioning. An fma would make it exact, but this generic wrapper is
shared by all distributions and is not specific to uniform.
With loc = 1e16 and scale = 1, the only double inside the support is 1e16 itself,
so cdf can only return 0 or 1. This follows from the representation of x and is not
an accuracy problem.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | `1 - x` (same expression) | none: exact for x >= 0.5, correctly rounded below |
| isf | ppf(1 - q) | `1 - q` (same) | none: exact for q >= 0.5, correctly rounded below |
| logsf | log1p(-cdf) for x <= 0.5, log(sf) above | `log1p(-x)` | none measured (`logsf(1e-17)` = -1e-17, 0.02 ulp). A direct form would only skip the median `apply_where` |
| logcdf | log(cdf) for x < 0.5, log1p(-sf) above | `log(x)` | none |
| logpdf | log(pdf) | `0.0*x` (zero inside the support) | none |

Custom methods would only save a little overhead. They would not improve accuracy.

## Shared code

none (`rv_continuous` loc/scale wrappers behave as expected; see "Checked and fine")

## Existing GitHub issues

not searched (no findings)

## Verification

pending
