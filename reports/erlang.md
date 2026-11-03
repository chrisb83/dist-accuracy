# erlang: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `erlang_gen`, lines 3794-3846 (subclass of `gamma_gen`, lines 3581-3788)
- **Implemented:** none of its own. pdf, logpdf, cdf, sf, ppf, isf all come from `gamma_gen` (lines 3628-3645). `erlang_gen` overrides only `_argcheck`, `_shape_info`, `_fitstart` and `fit`.
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf) (as in gamma)
- **Related:** special case of `gamma` with integer shape a >= 1. This report refers to `reports/gamma.md` and does not repeat its analysis.
- **Reference:** `refs/gamma_ref.py` (the gamma reference, 400 digits), applied at integer a. erlang returns the same bits as gamma at all 174 points compared (`scratch/erlang/bits.py`, `bits.txt`)
- **Parameter sets:** gamma's integer-shape sets: a = 1, 1000, 1e6, 1e8
- **Worst finding:** severe in cdf (relative error 0.375 at a = 1e8), as gamma F1

## Summary

`erlang` is `gamma` restricted to integer shape. Every numerical method is
`gamma_gen`'s: erlang returns the same bits as gamma at all 174 points
compared (`scratch/erlang/bits.txt`). So gamma's findings apply wherever
their regime contains an integer a. That is F1-F4 and F6, since every
worst-case point in `reports/gamma.md` for those findings is at an integer
shape (a = 1000, 1e6, 1e8). F5 (a << 1) is outside erlang's intended domain.
It is still reachable, because `_argcheck` accepts any a > 0 and only warns
for non-integer a. Nothing here is specific to `erlang_gen`. All fixes belong
in `gamma_gen` or in `scipy.special` (see Shared code).

## Findings

All entries are gamma findings at integer shape. The numbers are gamma's
verified values at those (integer) points, reproduced bit for bit by
`erlang` (e.g. `erlang.cdf(99954000, 1e8)` = 1.3154316493365769e-06, the
same as `gamma.cdf`; true value 2.1056e-06).

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | a >~ 2e5 and 4.5*sqrt(a) < a - x <~ 15*sqrt(a) (gamma F1) | `sc.gammainc` returns the 2000-term partial sum (unconverged) | cdf 1.7e15 ulp (rel. 0.375) at a = 1e8; sf 3.6e9 ulp | severe |
| F2 | pdf | all a; worst for large a near the mode (gamma F2) | `exp(_logpdf)` amplifies logpdf's absolute error | 6.3e6 ulp at a = 1e6, x = 1.01e6 | severe |
| F3 | isf, ppf | a >~ 2e5, q in the left tail (gamma F3) | `gammaincinv`/`gammainccinv` invert the inaccurate `gammainc` of F1 | isf 5.4e6 ulp, ppf 1.0e6 ulp at a = 1e6 | severe |
| F4 | logpdf | a >~ 1e3, x within a few sqrt(a) of a (gamma F4) | `xlogy(a-1,x) - x - gammaln(a)` cancels | 4.6e5 ulp at a = 1e6 | moderate |
| F5 | cdf, sf | \|a log x - x - gammaln(a)\| large, \|x - a\| > 0.4a (gamma F6) | `igam`/`igamc` prefactor exponent rounding | sf 2.0e3 ulp, cdf 1.6e3 ulp at a = 1000 | moderate |

Details, reproducers and mechanisms: `reports/gamma.md`, F1-F4 and F6.

## RuntimeWarnings

Non-integer a raises a `RuntimeWarning` ("The shape parameter of the erlang
distribution has been given a non-integer value") and then evaluates the
gamma formulas anyway (`erlang.pdf(1, 0.5)` = 0.2075537487102974 = gamma).
Through this path gamma F5 (`ppf`/`isf` for a << 1, 3.6e3 ulp at a = 1e-3)
is reachable (`erlang.ppf(0.5, 1e-3)` returns the same value as gamma).

## Checked and fine

- Bit identity with gamma: 174 points over pdf, logpdf, cdf, sf, ppf and isf
  at integer a, 0 differences (`scratch/erlang/bits.txt`).
- a = 1 (exponential): `xlogy(0, x)` is exact, so logpdf = -x exactly. The
  other gamma results for small and moderate a are within a few ulp
  (see gamma "Checked and fine").

## Suspected, not confirmed

- `_fitstart` uses `int(4/(1e-8 + skew**2))`, which gives a = 0 (an invalid
  start) when |skew| > 2. This affects fitting, not accuracy, and was not
  measured.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | `log(sf)` | log-space `igamc` (see gamma) | as gamma: `erlang.logsf(2000, 2)` returns `-inf` |
| logcdf | `log(cdf)` | log-space `igam` (see gamma) | as gamma: `erlang.logcdf(1000, 1e6)` returns `-inf`, true -5908763.104675533 |

## Shared code

All findings are in `gamma_gen` and `scipy.special`. See `reports/gamma.md`
"Shared code" and INDEX.md items 6-7: `sc.gammainc`/`sc.gammaincc` (cephes
`igam`/`igamc`, the 2000-iteration cap and prefactor rounding),
`sc.gammaincinv`/`sc.gammainccinv`, and the `_logpdf` cancellation in
`gamma_gen`. Fixing gamma fixes erlang.

## Existing GitHub issues

Same as `reports/gamma.md`:
- #26471 (open): BUG: inaccuracy in special.gammainc for certain large parameters - relates to F1 and F3
- #17832 (open): Tracking issue: custom methods for continuous distributions - relates to the inherited logcdf/logsf
- F2, F4, F5: none found

(web search; gh unavailable)

## Verification

Not run separately. The findings are gamma's and were verified there
(`reports/gamma.md`, Verification). The only erlang-specific claim, bit
identity with gamma at integer a, was checked directly
(`scratch/erlang/bits.py`).
