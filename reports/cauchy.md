# cauchy: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `cauchy_gen`, lines 1445-1518
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf (ppf/isf via Boost `scu._cauchy_ppf` / `scu._cauchy_isf`)
- **Inherited (not measured):** logcdf, logsf (generic; in 1.18.1 these switch at the median to log1p(-sf) / log1p(-cdf))
- **Related:** none; helpers: none; special functions: scu._cauchy_ppf, scu._cauchy_isf (Boost), np.arctan2
- **Reference:** `refs/cauchy_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** none beyond conditioning

## Summary

All six implemented methods are accurate to about 1 ulp (or within
conditioning) on the full grid, x from -1.7e308 to 1.7e308 and q from 1e-308
to 1 - 2.2e-16. cdf/sf use `arctan2(1, -x)` / `arctan2(1, x)`, which avoids
the `1/2 + atan(x)/pi` cancellation in both tails; logpdf switches to a
log-of-|x| form at |x| = 1 that avoids the overflow of x**2. The only large
ulp counts are ppf/isf near q = 0.5 (up to 2.8e15 ulp), which are at most
0.47 of the condition number there and are inherent to the problem.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|

No findings: no point is beyond conditioning (tool output: "Points beyond
conditioning: none").

## RuntimeWarnings

- pdf: `underflow encountered in divide/multiply` for |x| beyond ~1.34e154,
  where the true pdf is subnormal (e.g. `pdf(1.4e154)` = 0.0 vs
  1.6e-309); not measured and not a finding.
- logpdf: `underflow encountered in divide; underflow encountered in square`
  from `(1/absx)**2` for huge |x|; the result is accurate (max 0.966 ulp), so
  the warning is spurious.
- cdf/sf: `underflow encountered in divide` when `arctan2(1, |x|)/pi` is
  subnormal (|x| > ~1.4e307); the true value is subnormal there, so not measured.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | x = 0, +-1e-300 ... +-1.7e308, both sides of x*x overflow (1.3e154, 1.4e154) | 0.908 ulp |
| logpdf | both sides of the branch at abs(x) = 1 (1 - 1e-16, 1 + 2.2e-16), tails to +-1.7e308 | 0.966 ulp |
| cdf | both tails to +-1.7e308, near 0 | 1 ulp |
| sf | both tails to +-1.7e308, near 0 | 1 ulp |
| ppf | q from 1e-308 to 1 - 2.2e-16, q = 0.5 +- 1e-16 ... 0.25 | 0.962 ulp in tails; near q = 0.5 up to 2.8e15 ulp with cond 9.0e15 (max ulp/cond 0.47, e.g. `ppf(0.50000001)` 1.7e7 ulp, cond 5.0e7) - ok |
| isf | same as ppf | same as ppf (0.962 ulp in tails, max ulp/cond 0.47 near 0.5) |

Near q = 0.5 the quantile tan(pi*(q - 1/2)) is ill-conditioned (cond about
0.5/|q - 0.5|): a double q near 0.5 carries an absolute uncertainty of
~1.1e-16 relative to the tiny result. The Boost result stays below half the
condition number, so nothing is lost beyond what the input precision allows.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf), log1p(-sf) above the median | `log(arctan2(1, -x)) - log(pi)` | none found: even where cdf is subnormal (x = -1.7e308) the log is accurate, `logcdf(-1.7e308)` = -710.8715667790776, 0.10 ulp. A closed form would only save the generic overhead. |
| logsf | log(sf), log1p(-cdf) below the median | `log(arctan2(1, x)) - log(pi)` | none found (mirror image of logcdf; `logsf(1.7e308)` = -710.8715667790776). |

## Shared code

none (Boost `_cauchy_ppf` / `_cauchy_isf` are accurate in all measured regimes)

## Existing GitHub issues

none found open. Background: #20761 (closed): ENH: stats.cauchy: improve cdf and quantile accuracy in tail - the fix that produced the current accurate cdf/sf/ppf/isf implementations. (Searched via the GitHub search API for "cauchy" in issue titles; `gh` was not available.)

## Verification

pending
