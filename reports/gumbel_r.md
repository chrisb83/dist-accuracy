# gumbel_r: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `gumbel_r_gen`, lines 4301-4427
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, ppf, isf
- **Inherited (not measured):** logsf (generic `rv_continuous._logsf`: `log1p(-cdf)` for x <= median, `log(sf)` for x > median)
- **Related:** genextreme with c = 0 (separate class, same formulas; see `reports/genextreme.md`); gumbel_l is the mirror distribution (reviewed separately in `reports/gumbel_l.md`); helpers: `_average_with_log_weights`, `_check_fit_input_parameters` (fit only, not reviewed); special functions: sc.expm1 (sf), sc.logsumexp (fit only)
- **Reference:** `refs/gumbel_r_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in ppf, isf (a single point each, ill-conditioned there: cond 3e16 / 5.1e16)

## Summary

All x-methods (pdf, logpdf, cdf, logcdf, sf) are accurate on the whole real
line, from x = -709.7 to 1.7e308. Errors are within conditioning everywhere:
at most 1.6 ulp in the bulk and 135 ulp in the left tail, where the cdf/pdf
have cond up to 4.3e3. sf uses `-expm1(-exp(-x))` and is accurate in both
tails. ppf and isf are accurate except at the single double nearest
exp(-1) (ppf) and nearest 1 - exp(-1) (isf). There `-log(q)` rounds to exactly
1.0 and the result comes out as 0.0 where the true value is 3.4e-17. That is a
loss by the scale, but the problem is extremely ill-conditioned there. The
neighbouring doubles are also off by up to 4e15 ulp but count as within
conditioning. The finding is the same as genextreme F4.
The inherited logsf returns -inf for x > ~745 (true value -x).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf, isf | q = fl(exp(-1)) (ppf), q = fl(1 - exp(-1)) (isf) | `-log(q)` rounds to 1.0, outer `log` gives 0 (pattern 2) | 0.0 for 3.38e-17 | loss |

### F1: ppf and isf return 0 at the q nearest exp(-1) / 1 - exp(-1)

- **Code:** `_ppf`, line 4346: `-np.log(-np.log(q))`; `_isf`, line 4352: `-np.log(-np.log1p(-p))`.
- **Mechanism:** pattern 2. At q = fl(exp(-1)) = 0.36787944117144233 the
  exact -log(q) is 1 - 3.4e-17, which rounds to 1.0, so `log(1.0)` = 0.
  The outer log takes an argument close to 1 and has no log1p form. isf
  has the same problem at p = fl(1 - exp(-1)), where `-log1p(-p)` rounds to 1.0.
- **Regime:** q within about one ulp of exp(-1) (ppf) or 1 - exp(-1) (isf).
  For the neighbouring doubles the relative error is ~eps/|ppf| (up to
  4.0e15 ulp at q = 0.3678794411714423). The tool rates these as within
  conditioning, because cond (arg) is 3e16 to 6e14 there. The absolute
  error, about 1e-16, is what one ulp of q causes.
- **Reproducer:** `python accuracy.py refs/gumbel_r_ref.py --at ppf 0.36787944117144233`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.36787944117144233)` | 0.0 | 3.3784855259134224e-17 | inf | 3.0e+16 | loss |
  | `isf(0.6321205588285577)` | 0.0 | 3.3784855259134224e-17 | inf | 5.1e+16 | loss |
  | `ppf(0.3678794411714423)` (context) | -2.2204460492503128e-16 | -1.1711009840963588e-16 | 4.0e+15 | 8.5e+15 | ok |

- **Fix direction:** same as genextreme F4: near q0 = exp(-1), use
  `log(-log q) = log1p(-(log(q) + 1))`, with `log(q) + 1` computed as
  `log1p((q - q0_hi - q0_lo)/q0)` (q0 split into two doubles); for isf the
  analogous form in p - (1 - q0). Low practical value given the conditioning.

## RuntimeWarnings

- `overflow encountered in exp` (pdf, logpdf, cdf, logcdf, sf) for
  x < -709.78, where `exp(-x)` overflows. The returned values are correct:
  logpdf/logcdf = -inf where the true value is below -1.8e308, pdf/cdf = 0
  where the true value is ~exp(-1e308), sf = 1.0. The warning is therefore
  spurious. Example: `logpdf(-710)` = -inf with the warning.
- `underflow encountered in exp` (reported by the harness, which enables all
  floating-point warnings) for large |x|, where `exp(-x)` or the outer `exp`
  underflows. The results are 0/-0.0 where the true values are subnormal or
  below the double range, so they are correct. With NumPy's default errstate
  these warnings are not shown.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | x in [-709.7, 1.7e308], incl. mode, x ~ 36.7 (exp(-x) < eps), underflow at 708..746 | 1.58 ulp bulk; 135 ulp left tail (within cond 4.3e3); 5.5 ulp right tail |
| logpdf | same grid; `-x - exp(-x)` cannot cancel since x + exp(-x) >= 1 | 1 ulp |
| cdf | same grid | 0.76 ulp bulk; 135 ulp left tail (within cond 4.3e3); 0 right tail |
| logcdf | same grid | 0 ulp |
| sf | same grid, incl. right tail up to sf underflow and left tail sf -> 1 | 1.58 ulp |
| ppf | q in [1e-308, 1 - 2.2e-16], around exp(-1) +- 1e-15 ... 1e-2 | 0.74 ulp except F1 (neighbours of exp(-1) within cond) |
| isf | q in [1e-308, 1 - 2.2e-16], around 1 - exp(-1) +- 1e-15 ... 1e-2 | 0.71 ulp except F1 (neighbours within cond) |

## Suspected, not confirmed

none. Hypotheses checked: pdf = exp(logpdf) error amplification
(found within conditioning), ppf/isf near q = 1 (accurate, since `log(q)`
of an exact double is accurate), and logpdf cancellation (impossible).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | `log1p(-cdf)` for x <= median (accurate), `log(sf)` for x > median | `log(-expm1(-t))` with t = exp(-x), as `-x + log(-expm1(-t)/t)` (the ratio is 1 once t < eps), or `sc._ufuncs._log1mexp(-t)` for moderate t | right tail once sf underflows: x > ~745 returns -inf, and for 708 < x < 745 sf is subnormal. Demonstration: `gumbel_r.logsf(800)` = -inf, true -800.0 (mpmath, 50 digits) |

## Shared code

none. The F1 expression is duplicated in genextreme (F4 there, lines 3446/3453)
and probably also appears mirrored in gumbel_l, but it is not a shared helper.

## Existing GitHub issues

none found (searched the web for gumbel_r accuracy/ppf/logsf issues; `gh` unavailable). The same ppf expression is in genextreme F4; see reports/genextreme.md for related issues.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(0.36787944117144233)` | 0.0 | 3.3784855259134225844e-17 | verified | Reference from `-log1p(-(log q + 1))` and separately from `mp.findroot` on cdf(x) = q; 100 and 200 digits agree to 85 digits; cdf(x_ref) = q exactly. Relative error is 1 (4.5e15 ulp by the stated formula, not "inf"); still a loss. Mechanism matches line 4346 |
| F1 | `isf(0.6321205588285577)` | 0.0 | 3.3784855259134225844e-17 | verified | Same check with target 1 - p; mechanism matches line 4352 (`-log1p(-p)` rounds to 1.0) |
| F1 (context) | `ppf(0.3678794411714423)` | -2.2204460492503128e-16 | -1.1711009840963588041e-16 | verified | 4.0e15 ulp, within cond as stated |

Verified by dist-accuracy-verifier: 1 of 1 verified.
