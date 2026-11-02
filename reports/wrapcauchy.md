# wrapcauchy: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `wrapcauchy_gen`, lines 11450-11517
- **Implemented:** pdf, cdf, ppf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** none; helpers: none; special functions: none (np.cos, np.tan, np.arctan only)
- **Reference:** `refs/wrapcauchy_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (0.031071279018614728), (1e-8), (0.5), (0.9), (0.999), (1 - 1e-6), (0.9999999870854271), (1 - 1e-10), (1 - 2**-52)
- **Worst finding:** loss in pdf

## Summary

cdf and ppf are accurate everywhere measured: all points are within the
conditioning of the problem (the large ulp counts for c close to 1 sit where
cdf and ppf are extremely ill-conditioned, see "Checked and fine"). pdf is
accurate for c up to about 0.5 but gets steadily worse as c approaches 1, for
two reasons. Near the peak (x near 0 or 2*pi) the denominator
`1 + c*c - 2*c*cos(x)` cancels down to about (1-c)**2. This costs
~eps/((1-c)**2 + x**2) relative error, and the result is `inf` once
(1-c)**2 < eps. Everywhere else, the numerator `1.0 - c*c` loses up to ~0.1/(1-c) ulp
(measured up to 9.6e6 ulp, severe).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf | c near 1, x near 0 or 2*pi ((1-c)**2 + x**2 small) | 1 + c*c - 2*c*cos(x) cancels (patterns 1, 5) | inf for 3.18e9; 1.12e15 ulp (25 % relative) | loss |
| F2 | pdf | c near 1, all x | 1.0 - c*c cancels (pattern 1) | 9.6e6 ulp | severe |

### F1: pdf denominator cancels at the peak for c near 1

- **Code:** `_pdf`, line 11480: `(1.0-c*c)/(2*np.pi*(1+c*c-2*c*np.cos(x)))`
- **Mechanism:** the denominator equals (1-c)**2 + 4*c*sin(x/2)**2, but it is
  formed as the difference of the O(1) terms `1 + c*c` and `2*c*cos(x)`.
  Rounding `1 + c*c`, `c*c` and `cos(x)` (which is 1.0 for |x| < 1e-8) leaves
  an absolute error of about 1e-16. The relative error of the result is
  therefore about eps/((1-c)**2 + c*x**2) (patterns 1/5). Once (1-c)**2 is
  below about eps, the denominator rounds to exactly 0 and the pdf is `inf`
  (with "divide by zero" warnings). Near x = 2*pi the same happens because
  `cos(x)` rounds to 1. The pdf is flat at the peak (cond (arg) ~ 0), so none of
  this error comes from conditioning.
- **Regime:** x within about sqrt(1-c) of 0 or 2*pi. Errors reach ~100 ulp at
  c = 0.9, 8e5 ulp at c = 0.999, 1e12 ulp at c = 1 - 1e-6. For 1 - c below
  ~1e-8 the pdf at the peak is `inf`. The error decays to the F2 level once
  c*x**2 >> (1-c)**2.
- **Reproducer:** `python accuracy.py refs/wrapcauchy_ref.py --at pdf 1e-08 0.9999999999`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.0, 0.9999999999)` | inf | 3183098598.307993 | inf | 0 | loss |
  | `pdf(1e-08, 0.9999999999)` | inf | 318278.08472304186 | inf | 2 | loss |
  | `pdf(6.283185307179586, 0.9999999999)` | inf | 3183098598.2888975 | inf | 3.1e+05 | loss |
  | `pdf(1e-08, 0.9999999999999998)` | inf | 0.7067899292141145 | inf | 2 | loss |
  | `pdf(0.0, 0.9999999870854271)` | 18513560.290364236 | 24647341.23909139 | 1.12e+15 | 0 | severe |
  | `pdf(1e-06, 0.9999999999)` | 31.828161709976015 | 31.83099093537112 | 4.0e+11 | 2 | severe |
  | `pdf(0.0, 0.999999)` | 318352.11372254073 | 318309.72701969434 | 6.0e+11 | 0 | severe |
  | `pdf(0.0001, 0.9999999999)` | 0.00318309914455388 | 0.0031830991280172418 | 2.34e+07 | 2 | severe |
  | `pdf(0.0, 0.999)` | 318.1507312668677 | 318.1507312406985 | 3.7e+05 | 0 | moderate |
  | `pdf(1e-06, 0.9)` | 3.023943918473802 | 3.023943918473857 | 82 | 1.8e-10 | minor |

  (`pdf(0.0, 0.9999999870854271)` is off by 25 % but just below the tool's
  "relative error >= 1" loss threshold. cond (shapes) is large in this regime,
  e.g. 1e10 at c = 1 - 1e-10, but c is an exact user input.)

- **Fix direction:** denominator `(1-c)**2 + 4*c*np.sin(x/2)**2` (no
  cancellation; `sin(x/2)` is accurate near x = 2*pi because x/2 is an exact
  double near pi).

### F2: pdf numerator 1 - c*c cancels for c near 1

- **Code:** `_pdf`, line 11480: `1.0-c*c`
- **Mechanism:** `c*c` is rounded (absolute error up to 2**-54 ~ 5.6e-17)
  before the subtraction. The result is about 2*(1-c), so the relative error
  is up to ~2.8e-17/(1-c), i.e. up to ~0.125/(1-c) ulp (pattern 1). The actual
  error depends on how c*c rounds. It is largest for 1 - c around 1e-8 and
  shrinks for c within ~1e-12 of 1, where c has few significant bits below 1
  and c*c is nearly exact (scan in `scratch/wrapcauchy/scan_num.py`).
- **Regime:** c near 1, every x (the whole support). Measured bulk errors:
  64 ulp (c = 0.999), 5.0e4 ulp (c = 1 - 1e-6), 2.25e5 ulp (c = 1 - 1e-10),
  9.6e6 ulp (c = 0.9999999870854271).
- **Reproducer:** `python accuracy.py refs/wrapcauchy_ref.py --at pdf 3.141592653589793 0.9999999870854271`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(3.141592653589793, 0.9999999870854271)` | 1.027709058379842e-09 | 1.0277090605785046e-09 | 9.63e+06 | 1.9e-16 | severe |
  | `pdf(2.0, 0.9999999870854271)` | 1.4514159575226087e-09 | 1.4514159606277423e-09 | 9.63e+06 | 1.3 | severe |
  | `pdf(3.141592653589793, 0.9999999999)` | 7.957747813817494e-12 | 7.957747813419606e-12 | 2.25e+05 | 1.9e-16 | moderate |
  | `pdf(1.0, 0.999999)` | 3.462167087354547e-07 | 3.462167087316251e-07 | 4.98e+04 | 1.8 | moderate |
  | `pdf(1.0, 0.999)` | 0.0003463894400279605 | 0.0003463894400279654 | 64.1 | 1.8 | minor |

  (cond (shapes) 7.7e7 at c = 0.9999999870854271, context only.)

- **Fix direction:** `(1-c)*(1+c)` (1 - c is exact for c >= 0.5 by Sterbenz).

## RuntimeWarnings

- pdf: "divide by zero encountered in divide" for c = 1 - 1e-10 and
  c = 1 - 2**-52 at x near 0 and 2*pi. Each time the result is `inf` (F1),
  so these warnings flag real failures.
- cdf: "underflow encountered in arctan" in all parameter sets at tiny x
  (x ~ 1e-300). The results are accurate (max 1.53 ulp in the low tail), so
  the warning is spurious.
- ppf: "underflow encountered in arctan; underflow encountered in multiply"
  at tiny q. The results are accurate (max 1.01 ulp in the low tail), so the
  warnings are spurious. `np.where` on lines 11497-11499 evaluates both
  branches.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid, c in {1e-8, 0.031, 0.5} | 1.57 ulp |
| cdf | full grid incl. both sides of x = pi and x within 1e-300 ... 0.1 of 0 and 2*pi, all parameter sets | 3.5 ulp for c <= 0.9; larger values (up to 1.6e15 ulp at c = 1 - 2**-52) are all within conditioning |
| ppf | full grid incl. q = 1/2 and its neighbours, q in [1e-300, 1 - 1e-16], all parameter sets | 8.28 ulp for c <= 0.9; larger values (up to 1.5e15 ulp) all within conditioning |

Context on the large within-conditioning values for c close to 1:

- **cdf near x = 2*pi:** the cdf branch `f2` (line 11490) uses
  `2*np.pi - x`, and the support end `b = 2*np.pi` is 2.45e-16 below the true
  2*pi. With a sharp peak, the mass in that last 2.45e-16 is visible. For
  example, `cdf(6.283185307179586, 0.9999999999)` returns 1.0 against the
  reference 0.999999220365698 (3.5e9 ulp), and with c = 1 - 2**-52 it returns
  1.0 against 0.734 (1.6e15 ulp). cond (arg) there is 2e10 and 5.5e15: a
  1-ulp change of x moves the cdf by that much, so the tool rates these ok.
- **ppf at q = 1/2:** at q = 0.5 the code uses `tan(np.pi*0.5)` = 1.6e16, which is
  finite, rather than infinity. `ppf(0.5, 0.9999999999)` returns
  3.1415951028831883 instead of pi (9.0e9 ulp, cond 2e10), and
  `ppf(0.5, 0.9999999999999998)` returns 4.1496 (1.45e15 ulp, cond 9e15).
  Both are within conditioning: the ppf is extremely steep at q = 1/2 when
  c is near 1.

## Suspected, not confirmed

none. Hypotheses H3-H6 in `scratch/wrapcauchy/notes.md` (the `2*np.pi - x`
constant in the cdf, tan near q = 1/2 in the ppf, cdf saturation near x = pi,
tiny c) were all covered by the grid and gave no error beyond conditioning.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `log((1-c)*(1+c)) - log(2*pi) - log((1-c)**2 + 4*c*sin(x/2)**2)` | inherits F1/F2: c near 1, e.g. `logpdf(1e-08, 0.999999)` = 12.670913325041601, 8.3e10 ulp (severe); pdf never underflows otherwise (pdf >= (1-c)/(2*pi*(1+c))) |
| sf | 1 - cdf | sf(x) = arctan(r*tan((2*pi - x)/2))/pi for x >= pi, and 1 - arctan(r*tan(x/2))/pi for x < pi (r = (1+c)/(1-c)); equivalently cdf of the reflected argument | 1 - cdf cancels near x = 2*pi, but the problem is just as ill-conditioned there (cond (arg) ~ x*f/sf). `sf(6.283185307079586, 0.5)`: 4.774647344163441e-11 vs 4.774660382328529e-11, 1.2e10 ulp, cond 6.3e10, so it rates ok. Only a small gain is possible |
| logsf | log(sf) | log of the direct sf above | same as sf: `logsf(6.283185307079586, 0.5)` 5.2e8 ulp, within conditioning |
| logcdf | log(cdf) | none needed | cdf is accurate down to x = 1e-300 and does not underflow early |
| isf | ppf(1 - q) | 2*arctan(tan(pi*q)/r) for q > 1/2 is what the default already computes (1 - q is exact for q >= 1/2) | no failure: `isf(1e-10, 0.5)` 0 ulp, `isf(0.9999999999, 0.5)` 0.556 ulp |

## Shared code

none

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(0.0, 0.9999999999)` | inf | 3183098598.307993 | verified | reference via c = exp(-a): sinh(a)/(2*pi*(cosh(a) - cos(x))), 100 vs 200 digits agree to 30 digits |
| F1 | `pdf(1e-08, 0.9999999999)` | inf | 318278.08472304186 | verified | loss |
| F1 | `pdf(6.283185307179586, 0.9999999999)` | inf | 3183098598.2888975 | verified | loss |
| F1 | `pdf(1e-08, 0.9999999999999998)` | inf | 0.7067899292141145 | verified | loss |
| F1 | `pdf(0.0, 0.9999999870854271)` | 18513560.290364236 | 24647341.23909139 | verified | 1.12e15 ulp (25 % relative) |
| F1 | `pdf(1e-06, 0.9999999999)` | 31.828161709976015 | 31.83099093537112 | verified | 4.0e11 ulp; only denominator `(1-c)**2 + 4*c*sin(x/2)**2` removes it (0.19 ulp), so mechanism (line 11480 denominator) confirmed |
| F1 | `pdf(0.0, 0.999)` | 318.1507312668677 | 318.1507312406985 | verified | 3.7e5 ulp; also matches wrapped-Cauchy sum (mp.nsum) 318.150731240698 |
| F2 | `pdf(3.141592653589793, 0.9999999870854271)` | 1.027709058379842e-09 | 1.0277090605785046e-09 | verified | 9.63e6 ulp; numerator `(1-c)*(1+c)` alone brings it to 0.5 ulp, so mechanism (line 11480 `1.0-c*c`) confirmed |
| F2 | `pdf(2.0, 0.9999999870854271)` | 1.4514159575226087e-09 | 1.4514159606277423e-09 | verified | 9.63e6 ulp; wrapped-Cauchy sum gives 1.45141596062774e-9 |
| F2 | `pdf(3.141592653589793, 0.9999999999)` | 7.957747813817494e-12 | 7.957747813419606e-12 | verified | 2.25e5 ulp |
| F2 | `pdf(1.0, 0.999999)` | 3.462167087354547e-07 | 3.462167087316251e-07 | verified | 4.98e4 ulp |

Verified by dist-accuracy-verifier: 2 of 2 verified.
