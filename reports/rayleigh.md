# rayleigh: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `rayleigh_gen`, lines 9101-9219
- **Implemented:** pdf (9131-9133), logpdf (9135-9136), cdf (9138-9139), ppf (9141-9142), sf (9144-9145), logsf (9147-9148), isf (9150-9151)
- **Inherited (not measured):** logcdf (generic: `log(cdf)` below the median, `log1p(-sf)` above)
- **Related:** `chi` with df = 2 (but rayleigh has its own closed-form code, it does not call `chi`; only `_rvs` uses `chi.rvs`); helpers: `_check_fit_input_parameters`, `_get_left_bracket` (fit only, not measured); special functions: `sc.expm1`, `sc.log1p`
- **Reference:** `refs/rayleigh_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** minor in pdf (241 ulp at cond 1)

## Summary

rayleigh uses closed forms throughout and is accurate: logpdf, cdf, logsf,
ppf and isf are within 1 ulp on the whole grid (x from 1e-307 to 1.9e154,
q from 1e-300 to 1 - 1e-16), and sf is within its conditioning. The
x**2 overflow/underflow problem of chi (reports/chi.md F1-F3) does not occur:
`-0.5 * r * r` is evaluated as `(-0.5*r)*r`, which stays finite up to the
end of the double range, and `cdf = -expm1(-0.5*r**2)` only loses `r**2` where
the cdf itself is subnormal. The only finding is minor: `_pdf = exp(_logpdf)`
turns the absolute rounding error of `log(r)` into a relative error of up to
241 ulp for tiny r, where the pdf is simply r.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf | x < ~1e-30 (left tail, pdf ~ x) | `exp(log(r) - 0.5*r*r)` amplifies the absolute rounding of `log(r)` (patterns 3, 5) | 241 ulp at cond 1 | minor |

### F1: pdf loses up to ~|log x| / 2 ulp in the left tail via exp(logpdf)

- **Code:** `_pdf`, line 9133: `np.exp(self._logpdf(r))`, with `_logpdf`, line 9136: `np.log(r) - 0.5 * r * r`.
- **Mechanism:** patterns 3/5. `log(r)` is correctly rounded, but its
  absolute error is up to |log r| * 1.1e-16 (about 7.7e-14 for r = 1e-300);
  `exp` converts this absolute error into a relative error of the same size,
  i.e. up to ~|log r|/2 ulp, although the pdf (~r) has condition number 1
  there. Logpdf itself is accurate (0.58 ulp); the loss is only in the
  round trip.
- **Regime:** left tail, all x < ~1e-30 measured are minor (10-241 ulp,
  growing roughly with |log x|); in the right tail the same amplification
  (up to 292 ulp at x ~ 37.6) is within the conditioning (cond ~ x**2 = 1.4e3).
- **Reproducer:** `python accuracy.py refs/rayleigh_ref.py --at pdf 1e-280`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-280)` | 9.999999999999464e-281 | 1e-280 | 241 | 1 | minor |
  | `pdf(1e-305)` | 9.999999999999578e-306 | 1e-305 | 190 | 1 | minor |
  | `pdf(1e-290)` | 1.0000000000000418e-290 | 1e-290 | 188 | 1 | minor |
  | `pdf(2.2e-154)` | 2.2000000000000613e-154 | 2.2e-154 | 125 | 1 | minor |
  | `pdf(1e-100)` | 9.99999999999989e-101 | 1e-100 | 49.7 | 1 | minor |
  | `pdf(1e-30)` | 1.0000000000000024e-30 | 1e-30 | 10.3 | 1 | minor |

- **Fix direction:** `r * np.exp(-0.5 * r * r)` (the docstring formula); this
  is exact to a few ulp for small r and no worse in the right tail. Keep
  `(-0.5*r)*r` ordering (or `-0.5*r*r`) to avoid the `r**2` overflow.

## RuntimeWarnings

- logpdf, logsf, pdf, sf: "underflow encountered in multiply" from
  `0.5 * r * r` for tiny r; the results are accurate (spurious warning).
  "overflow encountered in multiply" occurs only for r beyond ~1.9e154,
  where the true logpdf/logsf is below -1.8e308 (legitimate). pdf/sf also
  warn "underflow encountered in exp" where the true value underflows
  (legitimate).
- cdf: "overflow encountered in square" for r > 1.34e154 (`r**2` = inf,
  `-expm1(-inf) = 1.0` is the correct result: spurious warning) and
  "underflow encountered in square" for tiny r, where the true cdf is
  subnormal (not measured).
- ppf, isf: none.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| logpdf | x from 1e-307 to 1.896e154, incl. the 1.34e154-1.897e154 window where `r**2` would overflow (`(-0.5*r)*r` does not) | 0.73 ulp |
| cdf | full grid, incl. x**2 underflow edge 1.5e-154 ... 3e-154 | 1 ulp |
| sf | full grid, incl. underflow edge 37-38.6 | 229 ulp at cond 1.4e3 (within conditioning) |
| logsf | full grid, incl. 1.34e154 ... 1.896e154 (`logsf(1.8e154)` = -1.62e308 exact) | 0 ulp |
| ppf | q in [1e-300, 1 - 1e-16] | 0.89 ulp |
| isf | q in [1e-300, 1 - 1e-16] | 0.94 ulp |
| pdf | bulk, right tail (within conditioning, max 292 ulp at cond 1.4e3) | 1.9 ulp in the bulk; left tail see F1 |

## Suspected, not confirmed

none. The chi x**2 overflow in logpdf (chi F3) was tested deliberately at
1.3e154-1.896e154 and does not occur in rayleigh because of the evaluation
order `(-0.5*r)*r`; the x**2 underflow in cdf (chi F1) does not matter for
df = 2 because the cdf r**2/2 is itself subnormal wherever `r**2` is.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | `log(cdf)` below the median, `log1p(-sf)` above | `log(-expm1(-0.5*r*r))` for r <= sqrt(2 log 2), `log1p(-exp(-0.5*r*r))` above (i.e. `log1mexp(-0.5*r*r)`); for r < ~1e-150 simply `2*log(r) - log(2)` | cdf underflows for x < ~2.1e-154: `logcdf(1e-200)` returns -inf (true -921.72718437817822); `logcdf(1e-160)` = -737.5203880715338 vs -737.52037693865456 (cdf subnormal). Right tail is fine (log1p(-sf)). |

## Shared code

none (only `sc.expm1` and `sc.log1p`, used correctly and accurate here).

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

pending
