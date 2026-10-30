# loglaplace: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `loglaplace_gen`, lines 6753-6847
- **Implemented:** pdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf)
- **Related:** none sharing code (mathematically exp of `laplace`, see reports/laplace.md); helpers: `_check_fit_input_parameters` (fit only); special functions: none (only `np.power`)
- **Reference:** `refs/loglaplace_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (3.2505926592051435), (0.3), (0.7), (1), (3), (0.001), (50), (1e6)
- **Worst finding:** minor in pdf (343 ulp), ppf and isf (130 ulp)

## Summary

loglaplace is a clean implementation: cdf and sf are within 1 ulp everywhere
(the subtracted side `1 - 0.5*x**(+-c)` is always >= 0.5, so no cancellation),
and ppf/isf only form `1 - q` for q >= 0.5, where it is exact. The only
errors beyond conditioning are minor and come from one mechanism: the
exponent passed to `np.power` (`c - 1`, `-c - 1`, `1/c`, `-1/c`) is rounded,
and that rounding is multiplied by `|log x|` (pdf) or `|log 2q|` (ppf/isf),
giving up to 343 ulp in pdf and 130 ulp in ppf/isf in the far tails. No
loss, severe or moderate case was found in the implemented methods. The
inherited logpdf/logcdf/logsf return `-inf` in the tails although trivial
closed forms exist.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf | c with inexact `c - 1` or `-c - 1` (e.g. 0.3, 0.001, 3.2506), |log x| large | rounded exponent amplified by log x (INDEX pattern 4) | 343 ulp | minor |
| F2 | ppf, isf | c with inexact `1/c` (e.g. 0.7, 3), q (ppf) or q (isf) tiny | rounded `1/c` amplified by log(2q) | 130 ulp | minor |

### F1: pdf exponent `c - 1` / `-c - 1` is rounded

- **Code:** `_pdf`, lines 6793-6794: `c = np.where(x < 1, c, -c)`; `cd2*x**(c-1)`.
- **Mechanism:** `c - 1` (x < 1) and `-c - 1` (x >= 1) are not exact when c
  and c +- 1 have different binades (Sterbenz does not apply). An absolute
  exponent error d gives a relative pdf error `d*|log x|`. Measured exponent
  errors: c = 0.001: `-c-1` off by 1.1e-16; c = 3.2505926592051435: `-c-1`
  off by 4.4e-16; c = 0.3: `c-1` off by 5.6e-17. Times |log x| = 690, 161,
  690 this predicts 343, 322, 173 ulp, matching the measurements exactly.
  Exponents that are exact (c = 0.7, 1, 3, 50) give <= 1 ulp.
- **Regime:** both tails, error ~ `|log x| * err(c -+ 1) / eps` ulp; bounded
  by about 350 ulp since the pdf leaves the normal range once
  `(c+-1)|log x| > ~708`. cond (arg) there is |c -+ 1| (<= 4.25).
- **Reproducer:** `python accuracy.py refs/loglaplace_ref.py --at pdf 1e300 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e300, 0.001)` | 2.505936168136552e-304 | 2.5059361681363614e-304 | 343 | 1 | minor |
  | `pdf(1e70, 3.2505926592051435)` | 4.671393049624671e-298 | 4.671393049624336e-298 | 322 | 4.25 | minor |
  | `pdf(1e-300, 0.3)` | 1.4999999999999538e+209 | 1.5000000000000114e+209 | 173 | 0.7 | minor |
  | `pdf(1e200, 0.3)` | 1.4999999999999693e-261 | 1.5000000000000077e-261 | 115 | 1.3 | minor |

  (cond (shapes) is 523 at the c = 3.25 point and 206 at the c = 0.3 point,
  i.e. a 1-ulp change of c moves the pdf by that much; at c = 0.001 it is only
  0.3, because the rounding of `-c - 1` is far larger than an ulp of c.)

- **Fix direction:** keep c exact in the exponent: `cd2 * x**c / x` for
  x < 1 and `cd2 * x**(-c) / x` for x >= 1 (or
  `exp(log(c/2) - c*|log x| - log x)`), guarding against intermediate
  under/overflow of `x**+-c`.

### F2: ppf/isf exponent `1/c` is rounded

- **Code:** `_ppf`, line 6803: `(2.0*q)**(1.0/c)`, `(2*(1.0-q))**(-1.0/c)`;
  `_isf`, line 6806: `(2.0*(1.0 - q))**(1.0/c)`, `(2*q)**(-1.0/c)`.
- **Mechanism:** `1.0/c` carries a relative rounding of up to eps/2; the
  result's relative error is that times `|log(2q)/c|`, i.e. times |log x|.
  For c = 0.7, `1/0.7` is off by 5.9e-17, and |log result| = 698 at
  q = 1.6e-213, predicting 130 ulp. cond (arg) is only 1/c.
- **Regime:** ppf for small q, isf for small q (both via the `(2q)**(+-1/c)`
  branch), whenever 1/c is inexact and the result is far from 1; up to
  ~|log x|/2 ulp, so at most a few hundred ulp.
- **Reproducer:** `python accuracy.py refs/loglaplace_ref.py --at ppf 1.6155980984399093e-213 0.7`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1.6155980984399093e-213, 0.7)` | 2.7666079008171126e-304 | 2.766607900817033e-304 | 130 | 1.43 | minor |
  | `isf(1.6155980984399093e-213, 0.7)` | 3.6145346064567076e+303 | 3.614534606456812e+303 | 130 | 1.43 | minor |
  | `ppf(1e-300, 3)` | 1.2599210498948891e-100 | 1.2599210498948732e-100 | 57.2 | 0.333 | minor |
  | `isf(1e-300, 3)` | 7.937005259840896e+99 | 7.937005259840997e+99 | 57.3 | 0.333 | minor |

- **Fix direction:** none cheap in double precision (`exp(log(2q)/c)` rounds
  the same quantity); would need a correctly rounded `pow(y, 1/c)`
  (e.g. `cbrt` for c = 3) or a double-double exponent. Low priority.

## RuntimeWarnings

- cdf, sf, ppf, isf: `np.where` evaluates both branches, so e.g.
  `cdf(1e-200, 3)` raises "overflow encountered in power" from `x**(-c)` of
  the unused x >= 1 branch, and ppf/isf raise overflow/underflow from the
  unused `(...)**(+-1/c)` branch. Results are accurate (or correctly 0 for
  subnormal true values): spurious warnings.
- pdf: "underflow encountered in power/multiply" where the true pdf is
  subnormal or zero (legitimate); "divide by zero encountered in power" at
  x = 0 for c < 1 (`pdf(0, 0.3)` = inf, the correct limit of `x**(c-1)`).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid incl. both sides of x = 1, all parameter sets | 1 ulp |
| sf | full grid incl. both sides of x = 1, all parameter sets | 1 ulp |
| ppf | q >= 0.5 (`1 - q` exact by Sterbenz), q near 0.5, all sets | 8.9 ulp (high tail) |
| isf | q <= 0.5 side via `2*(1-q)`, all sets | 9.1 ulp (high tail) |
| ppf, isf | c = 0.001 (1/c = 1000) | 51.4 ulp, within cond 1e3 |
| pdf | c = 1e6, x**(c-1) subnormal (pattern 4): `pdf(0.999282..., 1e6)` | 6.9e3 ulp, within cond (arg) 1e6 |
| pdf | c with exact c +- 1 (0.7, 1, 3, 50, 1e6) | 1 ulp |

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `log(c/2) + (c-1)*log(x)` (x < 1), `log(c/2) - (c+1)*log(x)` (x >= 1) | pdf under/overflows: `(c+-1)|log x| > ~708`; `logpdf(1e300, 3)` = -inf vs -2762.6966464847465 (loss) |
| logcdf | log(cdf) | `c*log(x) - log(2)` (x < 1), `log1p(-0.5*x**(-c))` (x >= 1) | cdf underflows for `c*log x < -708`; `logcdf(1e-300, 3)` = -inf vs -2073.0197308752013 (loss) |
| logsf | log(sf) | `log1p(-0.5*x**c)` (x < 1), `-c*log(x) - log(2)` (x >= 1) | sf underflows for `c*log x > 708`; `logsf(1e300, 3)` = -inf vs -2073.0197308752013 (loss) |

## Shared code

none (no helpers or scipy.special functions in the measured methods; the
rounded-exponent mechanism is the cross-cutting INDEX pattern 4 also seen in
fisk, pareto, the Weibull family).

## Existing GitHub issues

none found (web search; `gh` CLI unavailable; only #19996 / PR #20018 came up, a documentation issue about the relation to `laplace`)

## Verification

pending
