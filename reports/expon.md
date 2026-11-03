# expon: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `expon_gen`, lines 2050-2156
- **Implemented:** pdf, logpdf, cdf, sf, logsf, ppf, isf
- **Inherited (not measured):** logcdf (`rv_continuous._logcdf`, `_distn_infrastructure.py` lines 2035-2041: `log(cdf)` below the median, `log1p(-sf)` above it)
- **Related:** none in the inventory (special case of gamma with a = 1); helpers: `_remove_optimizer_parameters` (fit only); special functions: sc.expm1, sc.log1p
- **Reference:** `refs/expon_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** none beyond conditioning

## Summary

expon has closed forms for every method, and each one uses the right
primitive: `exp(-x)` for pdf and sf, `-x` for logpdf and logsf,
`-expm1(-x)` for cdf, `-log1p(-q)` for ppf and `-log(q)` for isf. On the
full default grid plus the extra points (smallest normal x, the region
where the cdf rounds to 1, the edge of the subnormal range of `exp(-x)`,
and q next to 1) every measured method is within 1 ulp of the reference.
I found no accuracy problem. The `sc.expm1` subnormal inexactness
(truncexpon F4) also reaches `expon.cdf`, but only where the true cdf is
subnormal, so it cannot be measured here. The one RuntimeWarning is
underflow in pdf/sf where the true value is subnormal or below.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|

None. No point on the grid was beyond conditioning.

## RuntimeWarnings

- `underflow encountered in exp` in pdf and sf (`_pdf` line 2085, `_sf`
  line 2097: `np.exp(-x)`) for x > ~708.4. There the true value is subnormal
  (x up to ~745.1) or below the smallest subnormal. The value returned is the
  correctly rounded subnormal or 0, so the warning is expected and not an
  accuracy problem. With `np.errstate(all='raise')`, `expon.pdf(708.3)`
  returns 2.450295530965988e-308 silently and `expon.pdf(708.5)` raises.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, x from 2.2e-308 to 708.39 (result normal), plus 1e-17, 36-38, 700 | 0.5 ulp |
| logpdf | full grid incl. x up to 1e300 (exact: `-x`) | 0 ulp |
| cdf | full grid; smallest normal x (2.2250738585072014e-308, 3e-308, 1e-307 ... 1e-300); x where cdf rounds to 1 (36-38) | 1 ulp |
| sf | full grid, same points as pdf | 0.5 ulp |
| logsf | full grid incl. x up to 1e300 (exact: `-x`) | 0 ulp |
| ppf | q in [2.2e-308, 1 - 2**-53], incl. 1e-17, 1 - 2**-52, 1 - 1e-10 | 0.721 ulp |
| isf | q in [2.2e-308, 1 - 2**-53], same points | 0 ulp |

In a separate scan of 20001 log-spaced x in [2.2e-308, 1e-290],
`-sc.expm1(-x)` was within 0.996 ulp of x everywhere. So the `sc.expm1`
inexactness is limited to subnormal arguments.

## Suspected, not confirmed

- **cdf for subnormal x (sc.expm1, shared code).** `_cdf` line 2091:
  `-sc.expm1(-x)`. `expon.cdf(1e-312)` returns 1.000000000003e-312 and
  `expon.cdf(1e-310)` returns 1.00000000000005e-310, while the true values are
  1e-312 and 1e-310 (x - x**2/2). This is the inexactness reported in
  truncexpon F4. For expon the true cdf is itself subnormal there, so by
  convention it is not measured (subnormals hold fewer than 53 bits; the
  error is about 3e-12 relative, i.e. a few ulp of the subnormal spacing).
  Once x is normal, the error is <= 1 ulp (see above). No accuracy finding
  for expon.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | `rv_continuous._logcdf`: `log(_cdf)` for x < median (ln 2), `log1p(-_sf)` above | `sc._ufuncs._log1mexp(-x)` (= `log(-expm1(-x))`), one call and no `ppf(0.5)` evaluation | None found. Below the median `log(-expm1(-x))` is accurate. Above it, `log1p(-exp(-x))` is accurate. Demonstration: `logcdf(40)` = -4.248354255291589e-18, 0 ulp (cond 40, ok). Only a speed and simplicity gain. |

## Shared code

- `scipy.special.expm1` is inexact for subnormal arguments (truncexpon F4:
  `sc.expm1(-1e-312) = -1.000000000003e-312`). `expon._cdf` calls it
  directly and returns the same inexact subnormal values (e.g.
  `expon.cdf(1e-312) = 1.000000000003e-312`). For expon this stays inside
  the subnormal output range, so it is not a measured finding. The fix belongs
  in `sc.expm1` (return x for tiny |x|), not in expon.

## Existing GitHub issues

not searched (no findings)

## Verification

pending
