# burr: numerical accuracy review

*(Worked example. Produced with `accuracy.py` and `burr_ref_example.py`
against SciPy 1.18.1. Tables are shortened; a real report lists every
finding in full.)*

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `burr_gen`, lines 1145-1264
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf
- **Inherited (not measured):** none
- **Related:** fisk (calls `burr._pdf(x, c, 1.0)` etc. for every method, so it inherits all findings with d = 1); special functions: sc.expm1, sc.log1p, sc.xlog1py, sc.xlogy
- **Reference:** `refs/burr_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (10.5, 4.3), (2, 3), (0.5, 0.001), (100, 0.001), (2, 0.5), (3, 1e4), (100, 1e6)
- **Worst finding:** loss in pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf

## Summary

All methods are accurate in the bulk for moderate shapes. Every method
fails completely in at least one tail: `x**-c` overflows for small x
(pdf, logpdf, cdf, logcdf, and through cdf also sf/logsf), `1 - cdf` cancels
in the right tail (sf, logsf), `q**(-1/d) - 1` cancels for q near 1 and
overflows for small q (ppf), and `expm1` overflows for small d (isf).
logpdf also loses every digit when c*d = 1. Large d costs up to ~d ulp in
pdf and cdf.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | sf, logsf | right tail, x**-c < ~eps/d | 1 - cdf cancels (pattern 1) | sf 0.0, logsf -inf | loss |
| F2 | logpdf, logcdf, pdf, cdf, sf, logsf | left tail, c*log(x) < -709.8 | x**-c overflows (pattern 3) | -inf, nan, 0.0 | loss |
| F3 | logpdf | x < 1, c*d close to 1 | large log terms cancel (pattern 5) | 0.0 for -1.5e-20 | loss |
| F4 | ppf | q near 1, large d | q**(-1/d) - 1 cancels (pattern 8) | 1.2e12 ulp | severe |
| F5 | ppf | small q or small d | q**(-1/d) overflows (pattern 8) | 0.0 for 1e-300 | loss |
| F6 | isf | small d | expm1 overflows (pattern 8) | 0.0 for 2.2e-4 | loss |
| F7 | pdf | large c*d, x**(-c-1) subnormal | subnormal intermediate (pattern 4) | 2.6e6 ulp | severe |
| F8 | pdf, cdf | large d | (1 + u)**d amplifies rounding (pattern 6) | 9.2e5 ulp | moderate |

### F1: sf and logsf cancel in 1 - cdf

- **Code:** `_logsf`, line 1228: `np.log1p(- (1 + x**(-c))**(-d))`; `_sf` = `exp(_logsf)`.
- **Mechanism:** for large x, cdf = 1 - d*x**-c + ...; once d*x**-c < eps/2
  the cdf rounds to 1, so `log1p(-cdf)` is `-inf`. Before that, digits are
  lost in proportion to 1/sf.
- **Regime:** right tail; total loss once d*x**-c < 1.1e-16.
- **Reproducer:** `python accuracy.py refs/burr_ref.py --at sf 1e8 2 3`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `sf(1e8, 2, 3)` | 0.0 | 2.9999999999999994e-16 | inf | 2 | loss |
  | `logsf(1e8, 2, 3)` | -inf | -35.74274919923662 | inf | 0.056 | loss |

- **Fix direction:** `sf = -expm1(logcdf)`, `logsf = log1mexp(logcdf)`.

### F2: x**-c overflows in the left tail

- **Code:** lines 1203-1215 and 1219-1228: every method forms `x**(-c)`.
- **Mechanism:** for c*log(x) < -709.8, `x**-c` is inf; pdf becomes
  inf/inf = nan, logpdf and logcdf become -inf, cdf becomes 0 (wrong where the
  true cdf is still normal, e.g. small d), and sf = exp(log1p(-0)) = 1.0
  exactly although the true sf is 1 - 1e-8.
- **Regime:** x < exp(-709.8/c); for c = 2 that is x < 1e-154.
- **Reproducer:**

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-200, 2, 3)` | -inf | -2300.7933335248176 | inf | 0.0022 | loss |
  | `pdf(1e-160, 2, 0.6)` | nan | 1.2000000000000196e-32 | inf | 0.2 | loss |
  | `logcdf(1e-200, 2, 3)` | -inf | -2763.1021115928547 | inf | 0.0022 | loss |
  | `cdf(1e-300, 100, 0.001)` | 0.0 | 9.999999999999985e-31 | inf | 0.1 | loss |
  | `sf(1e-70, 100, 0.001)` | 1.0 | 0.9999999 | 4.5e8 | 1e-8 | severe |

- **Fix direction:** for x < 1 use `log1p(x**c) - c*log(x)` for
  `log1p(x**-c)`, and `c*d*x**(c*d-1) * (1 + x**c)**(-d-1)` for the pdf.

### F3: logpdf cancels when c*d is close to 1

- **Code:** `_logpdf`, lines 1213-1215:
  `log(c) + log(d) + xlogy(-c-1, x) - xlog1py(d+1, x**(-c))`.
- **Mechanism:** for small x the last two terms are both about
  (c+1)*|log x| and their difference is
  `(c*d - 1)*log(x) - (d+1)*log1p(x**c)`. For c*d = 1 the log(x) parts cancel
  exactly. With c = 2, d = 0.5, x = 1e-10 both terms are
  69.07755278982137..., differing in the 22nd digit; both round to the same
  double and the result is 0.0.
- **Regime:** x < 1 with |c*d - 1| small; the loss is about
  (c+1)/|c*d - 1| ulp, total when c*d = 1 exactly.
- **Reproducer:**

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-10, 2, 0.5)` | 0.0 | -1.5e-20 | inf | 2 | loss |
  | `logpdf(1e-10, 1, 1)` | -1.999964638343954e-10 | -1.9999999999000001e-10 | 8.0e10 | 1 | severe |

  (cond (shapes) is 2.9e21 and 2.2e11: logpdf is extremely sensitive to c and
  d here, but c and d are exact user inputs, so this does not excuse the error.)
- **Fix direction:** for x < 1 compute `(c*d - 1)*log(x)` directly.

### F4 ... F8

*(In a real report each finding gets a subsection like the ones above.
Abbreviated here:)*

- **F4** `ppf(0.99999999, 3, 1e6)`: 46428.26132663437 vs 46415.88818102501,
  1.2e12 ulp, cond (arg) 3.3e7, severe. `q**(-1/d) = 1 + 1e-14`.
  Fix: `expm1(-log(q)/d)`.
- **F5** `ppf(1e-300, 2, 0.5)`: 0.0 vs 1e-300, loss. `q**(-1/d) = 1e600`
  overflows. Fix: log form `exp(-(log(expm1(z)))/c)` with `z = -log(q)/d`.
- **F6** `isf(0.57, 100, 0.001)`: 0.0 vs 2.1611482313284278e-4, loss.
  `expm1(-log1p(-q)/d)` overflows for `-log1p(-q)/d > 709.8`. Same fix as F5.
- **F7** `pdf(1300, 100, 1e6)`: 3.102568774501476e-307 vs
  3.102568772686277e-307, 2.6e6 ulp, severe. `1300**-101 = 3e-315` is
  subnormal (~30 significant bits); `c*d = 1e8` lifts the product back.
  Fix: evaluate in log space where `(c+1)*log(x) > 708`.
- **F8** `pdf(100, 3, 1e6)`: 3.7e5 ulp; `cdf(1e3, 3, 1e4)`: 3.7e3 ulp,
  moderate. Rounding of `1 + x**-c` raised to the power d. Fix:
  `exp(-d*log1p(x**-c))`.

## RuntimeWarnings

Overflow/underflow/invalid warnings in every method in the regimes of F2,
F5, F6; `divide by zero in log1p` in sf/logsf (F1). The warnings coincide
with inaccurate results; no spurious warnings on accurate results were seen.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | bulk, all parameter sets except (100, 0.001) (F2) | 3.1 ulp |
| logcdf | right tail | 1.7e3 ulp where x**-c is subnormal (c*log(x) > 708); moderate, same mechanism as F7 |
| isf | q < 1e-3, all parameter sets | 59 ulp (35 points minor, slightly beyond conditioning; no clear mechanism) |
| logpdf | right tail | 2 ulp |

## Suspected, not confirmed

none

## Opportunities: inherited methods

none (all methods are implemented)

## Shared code

`fisk` delegates every method to `burr` with d = 1, so F1, F2, F4 and F5
apply to fisk as well (F3 needs c = 1 there).

## Existing GitHub issues

not checked

## Verification

pending
