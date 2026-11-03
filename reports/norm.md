# norm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `norm_gen`, lines 394-503 (helpers lines 358-391)
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf (all via module-level helpers `_norm_*`)
- **Inherited (not measured):** none
- **Related:** gengamma, loggamma, nakagami, t (listed by inventory); the `_norm_*` helpers are also called by alpha, foldnorm, johnsonsb, johnsonsu, powerlognorm, powernorm, skewnorm, truncnorm, halfnorm and others; helpers: `_norm_pdf`, `_norm_logpdf`, `_norm_cdf`, `_norm_logcdf`, `_norm_sf`, `_norm_logsf`, `_norm_ppf`, `_norm_isf`; special functions: sc.ndtr, sc.log_ndtr, sc.ndtri (sc.factorial2 only in `_munp`)
- **Reference:** `refs/norm_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in logpdf

## Summary

pdf, cdf, logcdf, sf, logsf, ppf and isf are accurate everywhere tested (full
default grids plus both sides of every threshold), all within conditioning. The
largest errors are 513 ulp (cdf/sf/logcdf/logsf at |x| = 26, cond 677) and 124 ulp
(pdf at |x| = 36.99, cond 1368). They come from rounding `x/sqrt(2)` (inside
ndtr) and `x**2` (pdf) before an exponential. ppf/isf are within 1.93 ulp. The
one finding is logpdf: it returns -inf for |x| in (1.3407807929942597e154,
1.8961503816218355e154), because `x**2` overflows although `-x**2/2` is still
a finite double.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | logpdf | \|x\| in (1.3407807929942597e154, ~1.896e154) | `x**2` overflows before the division by 2 (pattern 3) | -inf for -9.1125e+307 | loss |

### F1: logpdf overflows in x**2 before halving

- **Code:** `_norm_logpdf`, line 367: `return -x**2 / 2.0 - _norm_pdf_logC` (called by `_logpdf`, line 428).
- **Mechanism:** pattern 3. `x**2` overflows to inf once |x| > sqrt(1.797e308) =
  1.3407807929942596e154, so the result is -inf. The true value -x**2/2 - 0.919
  stays a normal double until x**2/2 reaches 1.797e308, i.e. |x| < 1.8961503816218355e154.
  Above that, -inf is correct. The function is well conditioned here (cond (arg) = 2).
- **Regime:** 1.3407807929942597e154 <= |x| < 1.896e154 (both signs). The
  boundary is sharp: at 1.3407807929942596e154 the result is exact (0 ulp), and at the next
  double the result is -inf.
- **Reproducer:** `python accuracy.py refs/norm_ref.py --at logpdf 1.35e154`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.3407807929942597e154)` | -inf | -8.98846567431158e+307 | inf | 2 | loss |
  | `logpdf(1.35e154)` | -inf | -9.1125e+307 | inf | 2 | loss |
  | `logpdf(-1.5e154)` | -inf | -1.1250000000000002e+308 | inf | 2 | loss |
  | `logpdf(1.8e154)` | -inf | -1.62e+308 | inf | 2 | loss |
  | `logpdf(1.89e154)` | -inf | -1.78605e+308 | inf | 2 | loss |
  | `logpdf(1.896e154)` | -inf | -1.797408e+308 | inf | 2 | loss |

  (8 grid points flagged, all loss, ±1.35e154, ±1.5e154, ±1.8e154, ±1.89e154.)
- **Fix direction:** halve before squaring: `-(0.5*x)*x - _norm_pdf_logC`
  (checked: gives -1.78605e+308 at x = 1.89e154 and -9.1125e+307 at 1.35e154).
  Alternatively, use `-(x*sqrt(0.5))**2`, but that rounds one more time.

## RuntimeWarnings

- logpdf: `overflow encountered in square` (line 367) in the F1 regime, together with the
  wrong -inf result. For |x| >= 1.896e154 the same warning appears, but the -inf result is
  correct there (true value below -1.8e308).
- pdf: `overflow encountered in square` (line 363) for |x| > 1.34e154. The
  result 0.0 is correct (true pdf underflows from |x| ~ 38.6), so this warning is
  spurious. The harness also records underflow warnings in exp/divide/square for large |x|
  (numpy ignores underflow by default, so the public API does not show them); all those results are correct.
- No warnings from the ndtr / log_ndtr / ndtri paths.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid to ±1e300; non-integer x up to 38.55 (x**2 rounded before exp, line 363); underflow edge 37.5-38.6; overflow of x**2 (returns 0, correct) | 124 ulp at ±36.987654321, cond (arg) 1368 (rounding of x**2 as expected: about x**2/4 ulp, within conditioning) |
| logpdf | bulk, tails up to 1.34e154, tiny x | 0.74 ulp (outside F1) |
| cdf | full grid; left tail to -38.6 (subnormal edge); ±1e-300; non-integer x | 513 ulp at x = -26 (cond 677); 512 ulp at -37 (cond 1370); within conditioning, from rounding x/sqrt(2) inside ndtr |
| sf | mirror of cdf (`ndtr(-x)`, line 383) | 513 ulp at x = 26 (cond 677) |
| logcdf | both sides of log_ndtr branch at x = -1 (1 ± 1e-9); left tail to -1.89e154 (`t*t` path); right tail where logcdf ~ -sf | 1.97 ulp in left tail and bulk; 513 ulp at x = 26 (= sf error, cond 677) |
| logsf | mirror of logcdf (`log_ndtr(-x)`, line 387) | same as logcdf |
| ppf | q from 1e-300 to 1 - 2**-53; both sides of ndtri's switches near q = exp(-2) ≈ 0.135 and 1 - exp(-2) ≈ 0.865; q = 0.5 ± 1e-9 | 1.93 ulp (q = 0.85) |
| isf | same q grid (`-ndtri(q)`, line 391) | 1.93 ulp |

Constants: `_norm_pdf_C` = sqrt(2*pi) is 0.47 ulp off and `_norm_pdf_logC` = log(sqrt(2*pi))
0.35 ulp off (both correctly rounded, pattern 11 not present).

Not measured: grid points with |x| >~ 1.9e154 for logcdf/logsf (15 points per
method: the reference is "unavailable" because the true value is below -1.8e308).
SciPy's -inf is correct there. The same points for cdf/sf have true values of
0 or 1.

## Suspected, not confirmed

none. The note in the task about `exp(-x**2/2)` rounding x**2 before exp was
tested with non-integer x up to 38.55. The error grows like x**2/4 ulp (124 ulp
at 36.99), but this stays inside the argument conditioning (x**2 = 1368), so it is
not a finding. An exact-split evaluation of x**2 (as in R's dnorm) would bring it to
a few ulp if desired. The same applies to ndtr's x/sqrt(2) rounding (513 ulp,
cond 677).

## Opportunities: inherited methods

none (all methods are implemented)

## Shared code

- **F1 is in the shared helper `_norm_logpdf` (line 366-367).** Callers in
  `_continuous_distns.py` that pass an argument which can reach |z| in
  (1.34e154, 1.9e154) inherit the -inf:
  `alpha._logpdf` (line 551, `_norm_logpdf(a - 1.0/x)`, x ~ 1e-154),
  `powerlognorm._logpdf` (line 8958, `_norm_logpdf(log(x)/s)`, tiny s),
  `powernorm._logpdf` (line 9016), `skewnorm._logpdf` (line 9769-9770),
  `truncnorm._logpdf` (line 10345). halfnorm has its own copy of the same bug
  (halfnorm report F3). These callers were not measured here.
- `sc.ndtr`, `sc.log_ndtr`, `sc.ndtri`: accurate for their inputs in all
  regimes tested (no defect found).

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `norm.logpdf(1.3407807929942597e154)` | -inf | -8.9884656743115795e+307 | verified | Reference by a different route, `mp.log(mp.npdf(x))`, at 100 and 200 digits (agree). Boundary confirmed: 1.3407807929942596e154 gives -8.988465674311578e+307 (0 ulp). |
| F1 | `norm.logpdf(1.35e154)` | -inf | -9.1125000000000004e+307 | verified | |
| F1 | `norm.logpdf(-1.5e154)` | -inf | -1.1250000000000002e+308 | verified | |
| F1 | `norm.logpdf(1.896e154)` | -inf | -1.7974079999999999e+308 | verified | Mechanism matches line 367 (`-x**2 / 2.0`, square overflows before halving). |
| F1 callers | `alpha.logpdf(6e-155, 1.0)` (z = a - 1/x = -1.67e154) | -inf | -1.388888888888889e+308 | verified | Caller line 551. Returns -inf although the true value is finite. |
| F1 callers | `powerlognorm.logpdf(0.5, 1.0, 4.5e-155)` (z = log(x)/s = -1.54e154) | -inf | -1.1863037380696329e+308 | verified | Line 8958. -inf, true value finite. |
| F1 callers | `powernorm.logpdf(1.5e154, 1.0)` | -inf | -1.1250000000000002e+308 | verified | Line 9016. -inf, true value finite. |
| F1 callers | `skewnorm.logpdf(1.5e154, 1.0)` and `(1.5e154, 0.0)` | -inf, -inf | -1.1250000000000002e+308 (both) | verified | Lines 9769-9770, both branches. -inf, true value finite. |
| F1 callers | `truncnorm.logpdf(1.5e154, -2e154, 2e154)` | -inf | -1.1250000000000002e+308 | verified | Line 10345. The mass is 1 to any precision, so the reference is logphi(x). -inf, true value finite. |

Scripts: `scratch/norm/verify/verify_f1.py`. All 5 listed callers inherit F1 (each returns -inf at a point where the true logpdf is finite).

Verified by dist-accuracy-verifier: 1 of 1 verified.
