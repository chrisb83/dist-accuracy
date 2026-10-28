# anglit: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `anglit_gen`, lines 566-606
- **Implemented:** pdf, cdf, sf, ppf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = median split log(cdf) / log1p(-sf), logsf = median split log(sf) / log1p(-cdf), isf = ppf(1 - q)
- **Related:** none; helpers: none; special functions: none (numpy `cos`, `sin`, `arcsin`, `sqrt`)
- **Reference:** `refs/anglit_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in cdf and sf, but only at the single support endpoint (artifact of `np.pi/4` as the support bound; cond (arg) 5.1e16)

## Summary

pdf, cdf, sf and ppf are accurate everywhere apart from one point each: the
default grid plus points a few ulp to 1e-3 inside both support ends, small |x|,
and q within 1e-16 to 1e-2 of 0.5. pdf has 0 ulp everywhere, and cdf/sf have at
most 3.2 ulp in the bulk. Large ulp values near the support ends and near q = 0.5
are all within conditioning (cond up to 5e16). The only flagged points are cdf at
x = -np.pi/4 and sf at x = +np.pi/4. The double `np.pi/4` lies 3.06e-17 inside the
true support, so the true values there are 9.4e-34, but SciPy returns 0. This is
"loss" by the tool's scale but has no practical weight. Among the inherited
methods, logpdf (log(pdf) returns 0 for |x| < ~1e-8) and isf (ppf(1 - q) loses
up to all digits for small q) can easily be improved.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | x = -np.pi/4 (cdf), x = +np.pi/4 (sf) only | `x + np.pi/4` is exact but `np.pi/4` is short of pi/4 by 3.06e-17 (pattern 11) | 0.0 for 9.37e-34 | loss |

### F1: cdf and sf are 0 at the double support endpoints

- **Code:** `_cdf`, line 594: `np.sin(x+np.pi/4)**2.0`; `_sf`, line 597:
  `np.cos(x + np.pi / 4) ** 2.0`.
- **Mechanism:** `fl(pi/4) = pi/4 - 3.0616e-17`. At x = -fl(pi/4), x + np.pi/4
  is exactly 0, so cdf = 0. The true value is sin(3.06e-17)**2 = 9.37e-34. For sf at
  x = +fl(pi/4), `_sf` itself gives cos(fl(pi/2))**2 = 3.749e-33 (4x too
  large, because `x + np.pi/4` rounds). The public `sf` replaces that with 0
  in its support check. SciPy's support `[-np.pi/4, np.pi/4]` treats both
  points as boundaries, so the public cdf and sf return 0 there. Near the ends the same constant error
  shifts the argument by about 0.3 ulp of x. That stays within conditioning:
  `cdf(-0.7853981633974482)` = 1.23e-32 vs 2.01e-32, 1.7e15 ulp, cond 1.1e16, ok.
- **Regime:** exactly one point per method. The tool rates it loss (relative
  error 1), but cond (arg) is 5.1e16, so a 1-ulp change of x changes the
  result completely.
- **Reproducer:** `python accuracy.py refs/anglit_ref.py --at cdf -0.7853981633974483`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-0.7853981633974483)` | 0.0 | 9.37349864163661e-34 | inf | 5.1e+16 | loss |
  | `sf(0.7853981633974483)` | 0.0 | 9.37349864163661e-34 | inf | 5.1e+16 | loss |

- **Fix direction:** low priority. One option is a two-part constant,
  `(x + PI4_HI) + PI4_LO`, together with a support bound that is consistent
  with it. Another is to accept 0 as the defined value at the double endpoint.
  Also, `sf = sin(np.pi/4 - x)**2` would avoid the rounding of `x + np.pi/4`
  near pi/2.

## RuntimeWarnings

None were recorded for pdf, cdf, sf or ppf on the measured grid, and none for
the inherited methods at the demonstration points.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, a few ulp to 1e-3 inside both ends, \|x\| from 1e-300 to 1e-5 | 0 ulp |
| cdf | bulk; left end (a + k ulp, a + 1e-14 ... 1e-3), within conditioning; right end | bulk 3.17 ulp; left end up to 1.7e15 ulp with cond 1.1e16 (constant error of np.pi/4, see F1) |
| sf | bulk; right end, within conditioning; left end | bulk 3.17 ulp; right end median 2.5e11 ulp, all within conditioning (rounding of `x + np.pi/4` near pi/2) |
| ppf | q from 1e-300 to 1 - 3e-16; q = 0.5 ± 1e-16 ... 1e-2 (arcsin(sqrt q) - pi/4 cancels) | 4.5e15 ulp at q = 0.5 + 1e-16 (cond 9.0e15), 1.6e7 ulp at q = 1 - 3e-16 (cond 3.5e7); no minor points |

## Suspected, not confirmed

none. The cancellation in `arcsin(sqrt(q)) - np.pi/4` near q = 0.5 (pattern 5/8)
was tested with q = 0.5 ± 1e-16 ... 1e-2. The absolute error stays at about
1e-16, which matches the conditioning (cond ~ 0.5/|ppf|).

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(cos(2x)) | `np.log1p(-2*np.sin(x)**2)` (cos 2x = 1 - 2 sin^2 x) | \|x\| < ~1e-4 loses digits; returns 0 for \|x\| < ~1e-8. `logpdf(1e-9)` = 0.0 vs -2.0000000000000003e-18 (4.5e15 ulp, loss); the candidate formula gives -2e-18 |
| isf | ppf(1 - q) | `np.pi/4 - np.arcsin(np.sqrt(q))` (= -ppf(q) by symmetry), or `np.arccos(np.sqrt(q)) - np.pi/4` | small q: 1 - q rounds. `isf(1e-18)` = 0.7853981633974483 vs 0.78539816239744831 (5.7e6 ulp, severe); `isf(1e-10)`: 2.4e3 ulp. The candidate gives 0.7853981623974483 |
| logcdf | log(cdf) below the median, log1p(-sf) above | none needed. The rv_continuous default already splits at the median, and the remaining error comes from sf's `x + np.pi/4` rounding (within conditioning) | `logcdf(pi/4 - 1e-9)` = -1.0000002879e-18 vs -1.0000000047e-18 (1.3e9 ulp, cond ~1.6e9: within conditioning) |
| logsf | log(sf) above the median, log1p(-cdf) below | none needed (same as logcdf) | `logsf(-pi/4 + 1e-9)`: 2.8e8 ulp, within conditioning |

## Shared code

none

## Existing GitHub issues

none found (web search of scipy/scipy issues for "anglit"; `gh` was not available)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-0.7853981633974483)` | 0.0 | 9.3734986416366099665e-34 | verified | Reference from mp.quad of cos(2t) over [-pi/4, x] and from (1 + sin 2x)/2; both agree to 40 digits at 100 and 200 dps. Line 594 matches; fl(pi/4) - pi/4 = -3.0616e-17 confirmed |
| F1 | `sf(0.7853981633974483)` | 0.0 | 9.3734986416366099665e-34 | verified | Reference from mp.quad over [x, pi/4] and from (1 - sin 2x)/2. `_sf` (line 597) gives 3.749e-33; the public sf returns 0 at the support bound, as the report says |

Verified by dist-accuracy-verifier: 1 of 1 verified.
