# truncweibull_min: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `truncweibull_min_gen`, lines 2895-2999
- **Implemented:** pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf
- **Inherited (not measured):** none
- **Related:** none share code (docstring points to weibull_min and truncexpon; c = 1, a = 0 is truncexpon, but the code is independent); helpers: none; special functions: sc.xlogy (logpdf); sc.gamma, sc.gammainc only in `_munp` (not measured)
- **Reference:** `refs/truncweibull_min_ref.py`, 400 digits; self-checks: PASS for the reference. The six FAILs at (2, 0, 1e-06) ("disagrees at every bulk point") come from SciPy (F2), not from the parametrization. Hand check: `cdf(sqrt(1e-13), 2, 0, 1e-6)` must be 0.1 to first order (x**c / b**c). SciPy returns 0.10003330742755634 because `1 - np.exp(-1e-12)` = 9.999778782798785e-13 (true value 9.999999999995e-13). All other sets pass every check.
- **Parameter sets (c, a, b):** (2.5, 0.25, 1.75), (1, 0, 4.69), (2, 0, 1e-06), (0.5, 1e-10, 1e-08), (2, 30, 31), (0.001, 0.5, 2), (2.3, 15.3, 16), (5, 0, 3), (2, 1, 1.00000001), (2, 26.8, 27.5), (3, 0, 0.01)
- **Worst finding:** loss in pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf

## Summary

All eight methods are written as differences of `exp(-pow(., c))` terms:
the normaliser `exp(-a**c) - exp(-b**c)`, the numerators
`exp(-a**c) - exp(-x**c)` and `exp(-x**c) - exp(-b**c)`, and the ppf/isf
argument `(1-q) exp(-a**c) + q exp(-b**c)`. None of them uses `expm1`,
`log1p` or factoring out `exp(-a**c)`. For the example parameters the bulk
is accurate (pdf 1.8 ulp, ppf 3 ulp, isf 8 ulp). The code breaks down in
five regimes:
(1) `a**c > ~745`: everything underflows, and every method returns nan or inf;
`a**c` in about 708...745 gives moderate errors and spurious zeros;
(2) small `b**c - a**c` (small b with a = 0, a narrow interval, or tiny c):
the normaliser cancels, giving relative errors ~eps/(b**c - a**c) in every
method, and nan/inf once `b**c < ~eps/2`;
(3) x near a: `1 - exp(-x**c)` gives cdf = 0 and logcdf = -inf for
x**c < eps/2 (a = 0), and logsf returns 0;
(4) x near b: sf = 0 and logsf = -inf, and logcdf = 0 because two log terms
cancel;
(5) ppf for q < ~eps (a = 0) returns 0, with severe errors up to q ~ 1e-10.
Small c also costs ~1/c ulp in ppf/isf.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | all eight | a**c > ~745 (exp(-a**c) underflows); a**c in ~708...745 (subnormal) | exp(-a**c), exp(-x**c), exp(-b**c) all underflow, ratio 0/0 (patterns 3, 4) | nan / inf for every x, q; pdf 0.0 for 1.7e-10 | loss |
| F2 | all eight | b**c - a**c small: a = 0 and b**c << 1; narrow [a, b]; tiny c | normaliser exp(-a**c) - exp(-b**c) cancels (patterns 1, 5) | pdf inf, cdf/sf nan, ppf/isf 0.0 when b**c < ~5.5e-17; 1.0e11 ulp at b**c = 1e-12 | loss |
| F3 | cdf, logcdf, logsf | x near a; total for a = 0, x**c < eps/2 | exp(-a**c) - exp(-x**c) cancels (pattern 1); logsf = log(num) - log(denum) with num == denum (pattern 5) | cdf 0.0, logcdf -inf, logsf 0.0 | loss |
| F4 | sf, logsf, logcdf | x near b with b**c not large | exp(-x**c) - exp(-b**c) cancels (pattern 1); logcdf = lognum - logdenum cancels (pattern 5) | sf 0.0, logsf -inf, logcdf 0.0 | loss |
| F5 | ppf, isf | ppf: q small with a**c small; isf: q -> 1 with a**c small | -log of a value 1 - O(q b**c) (patterns 2, 8) | ppf 0.0 for q < ~1e-16 (a = 0); 1.5e9 ulp at q = 1e-10 | loss |
| F6 | ppf, isf | tiny c (1/c large) | pow(X, 1/c) amplifies the rounding of X (pattern 6) | 1.34e3 ulp | moderate |

### F1: every method returns nan or inf once exp(-a**c) underflows

- **Code:** lines 2957, 2961, 2965-2966, 2970-2971, 2975-2976, 2980-2981, 2986, 2991: `np.exp(-pow(a, c)) - np.exp(-pow(b, c))` and the analogous numerators.
- **Mechanism:** the distribution depends only on differences such as
  a**c - x**c, but the code forms each `exp(-.)` separately (patterns 3, 4).
  For a**c > ~745 both the normaliser and the numerators are 0. pdf, cdf and
  sf return 0/0 = nan, logpdf returns `... - log(0)` = +inf, logcdf/logsf
  return -inf - (-inf) = nan, and ppf/isf return `(-log(0))**(1/c)` = inf.
  For a**c between ~708 and 745, `exp(-a**c)` is subnormal (fewer than 53
  significant bits): cdf has 3.5e5 ulp errors. Where `exp(-x**c)` crosses
  into underflow while the ratio `exp(a**c - x**c)` is still normal, pdf and
  sf return 0 and logsf returns -inf.
- **Regime:** a**c > ~745 (e.g. c = 2, a > 27.3): total loss everywhere.
  a**c in ~708...745: moderate errors, and zeros for x**c > 745.
- **Reproducer:** `python accuracy.py refs/truncweibull_min_ref.py --at pdf 30.5 2 30 31`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(30.5, 2, 30, 31)` | nan | 4.4455116984500124e-12 | inf | 1.86e3 | loss |
  | `logpdf(30, 2, 30, 31)` | inf | 4.0943445622221 | inf | 439 | loss |
  | `cdf(30.5, 2, 30, 31)` | nan | 0.9999999999999272 | inf | 1.36e-10 | loss |
  | `logsf(30.5, 2, 30, 31)` | nan | -30.250000000000043 | inf | 61.5 | loss |
  | `ppf(0.5, 2, 30, 31)` | inf | 30.011550229545957 | inf | 5.55e-4 | loss |
  | `isf(0.5, 2, 30, 31)` | inf | 30.011550229545957 | inf | 5.55e-4 | loss |
  | `pdf(27.29, 2, 26.8, 27.5)` | 0.0 | 1.6844152066888336e-10 | inf | 1.49e3 | loss |
  | `ppf(0.9999999999995953, 2, 26.8, 27.5)` | inf | 27.32719546767614 | inf | 1.7e9 | loss |
  | `cdf(26.8007, 2, 26.8, 27.5)` | 0.03682531793337559 | 0.03682531793052067 | 3.5e5 | 3.76e4 | moderate |

- **Fix direction:** factor out `exp(-a**c)`: normaliser
  `-expm1(a**c - b**c)`, pdf `c x**(c-1) exp(a**c - x**c) / -expm1(a**c - b**c)`,
  cdf `-expm1(a**c - x**c) / -expm1(a**c - b**c)`, and the log forms via
  `log1mexp`. For ppf, `x**c = a**c - log1p(q*expm1(a**c - b**c))`.

### F2: the normaliser cancels when b**c - a**c is small

- **Code:** `denum`/`logdenum`, lines 2957, 2961, 2966, 2971, 2976, 2981: `np.exp(-pow(a, c)) - np.exp(-pow(b, c))`; the same cancellation sits inside the log argument of `_ppf`/`_isf`, lines 2986, 2991.
- **Mechanism:** each exponential has an absolute rounding error of
  ~eps*exp(-a**c), and their difference is ~exp(-a**c)*(b**c - a**c). The
  normaliser therefore has relative error ~eps/(b**c - a**c), and that error
  goes straight into every method, including in the bulk where cond = 1
  (patterns 1, 5). For a = 0 the term is `1 - exp(-b**c)`. Once b**c < ~5.5e-17,
  `exp(-b**c)` rounds to 1.0 and the normaliser is exactly 0. pdf and logpdf
  then return inf, cdf and sf return nan, and ppf and isf return 0. The same
  happens for a narrow interval [a, a(1 + d)] (relative error ~eps/(c d a**c)
  when a**c <~ 1) and for tiny c, where a**c and b**c are both close to 1.
  This is the same mechanism as truncexpon F1 (sf), but here it affects
  every method, because truncexpon uses `expm1(-b)` for the normaliser and
  truncweibull_min does not.
- **Regime:** relative error ~1.1e-16/(b**c - a**c) for all x and q:
  loss for b**c - a**c < ~5.5e-17 (a = 0); severe at b**c = 1e-12 and for
  a = 1, b - a = 1e-8; moderate at b**c = 1e-6 and for c = 1e-3 with
  a = 0.5, b = 2.
- **Reproducer:** `python accuracy.py refs/truncweibull_min_ref.py --at pdf 5e-9 2 0 7e-9`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(5e-9, 2, 0, 7e-9)` | inf | 204081632.65306124 | inf | 1 | loss |
  | `logpdf(5e-9, 2, 0, 7e-9)` | inf | 19.13403063182983 | inf | 0.0523 | loss |
  | `cdf(5e-9, 2, 0, 7e-9)` | nan | 0.5102040816326531 | inf | 2 | loss |
  | `sf(5e-9, 2, 0, 7e-9)` | nan | 0.48979591836734687 | inf | 2.08 | loss |
  | `ppf(0.5, 2, 0, 7e-9)` | 0.0 | 4.949747468305832e-09 | inf | 0.5 | loss |
  | `pdf(5e-9, 2, 0, 1e-8)` | 90071992.54740992 | 100000000.0 | 4.5e14 | 1 | severe |
  | `pdf(5e-7, 2, 0, 1e-6)` | 1000022.1222092527 | 1000000.0000002501 | 1.0e11 | 1 | severe |
  | `logpdf(1e-12, 2, 0, 1e-6)` | 0.6931693025247547 | 0.6931471805604453 | 1.4e11 | 1.44 | severe |
  | `ppf(0.5, 2, 0, 1e-6)` | 7.071382115904185e-07 | 7.071067811864591e-07 | 2.0e11 | 0.5 | severe |
  | `isf(0.5, 2, 0, 1e-6)` | 7.071382115904185e-07 | 7.071067811864591e-07 | 2.0e11 | 0.5 | severe |
  | `pdf(1.000000005, 2, 1, 1.00000001)` | 100000001.11016832 | 100000000.60774711 | 2.3e7 | 1 | severe |
  | `logpdf(1.000000005, 2, 1, 1.00000001)` | 18.42068075505405 | 18.420680750029838 | 1.2e6 | 0.0543 | severe |
  | `pdf(0.005, 3, 0, 0.01)` | 75.00002812619176 | 75.00002812500215 | 7.14e4 | 2 | moderate |
  | `logpdf(0.725, 0.001, 0.5, 2)` | -0.005050607477913971 | -0.00505060747778828 | 1.1e5 | 198 | moderate |

  (cond (shapes) is 2.0e8 for pdf and 1.1e7 for logpdf in the narrow-interval case. The result is very sensitive to a and b there, but a and b are exact user inputs.)
- **Fix direction:** `-expm1(a**c - b**c)` (times `exp(-a**c)` where it
  is needed). For narrow intervals, also compute `b**c - a**c` without
  cancellation, e.g. `a**c * expm1(c*log1p((b - a)/a))`.

### F3: cdf, logcdf and logsf cancel near the left end x = a

- **Code:** `_cdf` line 2965 and `_logcdf` line 2970: `np.exp(-pow(a, c)) - np.exp(-pow(x, c))`; `_logsf` lines 2980-2982: `log(exp(-x**c) - exp(-b**c)) - logdenum`.
- **Mechanism:** for a = 0 the numerator is `1 - exp(-x**c)` (pattern 1).
  It is 0 for x**c < eps/2, so cdf returns 0 and logcdf returns -inf, and it
  is inaccurate (~eps/x**c) above that. For a > 0 the relative error is
  ~eps/(x**c - a**c). This exceeds cond (arg) ~ x/(x - a) by a factor
  ~1/(c a**c), which is large when a**c is small. logsf near a computes the
  difference of two nearly equal logs, and the result ~ -cdf is lost: it
  returns exactly 0.0 (pattern 5). Compare truncexpon, where cdf uses
  `expm1` and is accurate in this regime.
- **Regime:** a = 0: loss for x**c < ~1.1e-16, severe for x**c <~ 1e-8,
  moderate up to x**c ~ 1e-5. a > 0: severe within ~1e-12 relative of a for
  (2.5, 0.25, 1.75) and (0.5, 1e-10, 1e-8).
- **Reproducer:** `python accuracy.py refs/truncweibull_min_ref.py --at cdf 4.69e-20 1 0 4.69`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(4.69e-20, 1, 0, 4.69)` | 0.0 | 4.733485041501553e-20 | inf | 1 | loss |
  | `logcdf(4.69e-20, 1, 0, 4.69)` | -inf | -44.49704013343398 | inf | 0.0225 | loss |
  | `logsf(4.69e-20, 1, 0, 4.69)` | 0.0 | -4.733485041501553e-20 | inf | 1 | loss |
  | `cdf(3e-60, 5, 0, 3)` | 0.0 | 2.4300000000000007e-298 | inf | 5 | loss |
  | `cdf(1e-5, 1, 0, 4.69)` | 1.0092668175176866e-05 | 1.0092668175171082e-05 | 2.58e3 | 1 | moderate |
  | `cdf(1.000000000099e-10, 0.5, 1e-10, 1e-8)` | 4.9345959475381656e-12 | 5.500244636262493e-12 | 4.6e14 | 1.0e10 | severe |
  | `cdf(0.2500000000015, 2.5, 0.25, 1.75)` | 4.774093695269896e-13 | 4.773259071259952e-13 | 7.9e11 | 1.7e11 | severe |
  | `logsf(0.2500000000015, 2.5, 0.25, 1.75)` | -4.774097783766251e-13 | -4.773259071261091e-13 | 7.9e11 | 1.7e11 | severe |

- **Fix direction:** numerator `exp(-a**c) * -expm1(a**c - x**c)` (with
  `a**c - x**c` computed as in F2 for x close to a); logsf
  `log1p(-cdf)` when cdf is small.

### F4: sf, logsf and logcdf cancel near the right end x = b

- **Code:** `_sf` line 2975 and `_logsf` line 2980: `np.exp(-pow(x, c)) - np.exp(-pow(b, c))`; `_logcdf` lines 2970-2972: `lognum - logdenum`.
- **Mechanism:** mirror image of F3. The sf numerator cancels (pattern 1)
  and becomes exactly 0, so sf returns 0 and logsf returns -inf. The relative
  error ~eps/(b**c - x**c) exceeds cond (arg) ~ b/(b - x) by ~1/(c b**c).
  logcdf near b is the difference of two nearly equal logs, each of size
  ~a**c or ~|log(b**c - a**c)|, so its absolute error is ~eps times that
  size, and a result ~ -sf is lost (pattern 5). This also happens when
  a**c is large and sf itself is accurate: for (2.3, 15.3, 16) the logs are
  ~530 and logcdf returns 0.0.
- **Regime:** sf/logsf: b**c small (a = 0, b**c <= 1e-6; (0.5, 1e-10, 1e-8);
  tiny c), within ~1e-6 relative of b, where 1 - x/b < ~eps/(c b**c) gives
  loss. logcdf: any shapes, x close enough to b that sf < ~eps*max(a**c, 1).
- **Reproducer:** `python accuracy.py refs/truncweibull_min_ref.py --at sf 9.99999e-07 2 0 1e-06`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(9.99999e-07, 2, 0, 1e-06)` | 0.0 | 1.9999990000542072e-06 | inf | 1.0e6 | loss |
  | `logsf(9.99999e-07, 2, 0, 1e-06)` | -inf | -13.12236387737735 | inf | 7.62e4 | loss |
  | `logcdf(9.99999e-07, 2, 0, 1e-06)` | 0.0 | -2.0000010000548743e-06 | inf | 1.0e6 | loss |
  | `sf(0.00999999999999, 3, 0, 0.01)` | 0.0 | 3.0002027515970116e-12 | inf | 1.0e12 | loss |
  | `logsf(9.9999999999901e-09, 0.5, 1e-10, 1e-08)` | -inf | -28.22893366586837 | inf | 3.6e10 | loss |
  | `logcdf(15.685, 2.3, 15.3, 16)` | 0.0 | -2.7789878924530306e-14 | inf | 1.29e3 | loss |
  | `sf(1.9999999999985, 0.001, 0.5, 2)` | 6.530862563554385e-13 | 5.409785641636475e-13 | 9.3e14 | 1.3e12 | severe |
  | `logcdf(1.7499999999985, 2.5, 0.25, 1.75)` | -1.5863005353722315e-13 | -1.5868704830916863e-13 | 1.6e12 | 1.2e12 | severe |

- **Fix direction:** sf numerator `exp(-x**c) * -expm1(x**c - b**c)`;
  logcdf `log1p(-sf)` when sf < 0.5.

### F5: ppf (small q) and isf (q -> 1) take the log of a value near 1

- **Code:** `_ppf` line 2991: `-np.log((1 - q) * np.exp(-pow(a, c)) + q * np.exp(-pow(b, c)))`; `_isf` line 2986 (a and b swapped).
- **Mechanism:** for small a**c the log argument is
  `1 - q(1 - exp(-b**c))` approximately, i.e. 1 - O(q). Forming it in double
  leaves an absolute error ~eps, which becomes a relative error
  ~eps/(q(1 - e^{-b**c})) in x**c (patterns 2, 8). For q < ~eps the argument
  is exactly 1.0 and ppf returns 0. isf mirrors this as q -> 1. When
  b**c is also small, the error is present at every q (see F2).
- **Regime:** ppf with a = 0: loss for q <~ 1e-16, severe up to q ~ 1e-10,
  moderate beyond. Moderate errors for a = 1e-10, b = 1e-8, c = 0.5 at small
  q. isf: same for 1 - q small and b**c small.
- **Reproducer:** `python accuracy.py refs/truncweibull_min_ref.py --at ppf 1e-17 1 0 4.69`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 1, 0, 4.69)` | 0.0 | 9.908133138437553e-301 | inf | 1 | loss |
  | `ppf(1e-17, 1, 0, 4.69)` | 0.0 | 9.908133138437554e-18 | inf | 1 | loss |
  | `ppf(1e-300, 5, 0, 3)` | 0.0 | 1e-60 | inf | 0.2 | loss |
  | `ppf(1e-10, 1, 0, 4.69)` | 9.908129872606834e-11 | 9.908133138928409e-11 | 1.5e9 | 1 | severe |
  | `isf(0.99999280314327, 2, 0, 1e-06)` | 0.0 | 2.68269579527828e-09 | inf | 6.95e4 | loss |
  | `ppf(3.4807005884285624e-14, 0.5, 1e-10, 1e-8)` | 1.0000000000233508e-10 | 1.0000000000006265e-10 | 1.0e5 | 6.26e-13 | moderate |
  | `isf(0.9999218629262349, 0.5, 1e-10, 1e-08)` | 1.0014068985231808e-10 | 1.0014068985366698e-10 | 6.07e4 | 18 | moderate |

- **Fix direction:** `x**c = a**c - log1p(q * expm1(a**c - b**c))` for ppf;
  for isf `b**c - log1p(q * expm1(b**c - a**c))`, or the ppf form with
  `-expm1`/`log1p` of `(1 - q)` and logaddexp to avoid overflow for large
  b**c - a**c.

### F6: ppf and isf amplify rounding by 1/c for tiny c

- **Code:** `_ppf` line 2990 / `_isf` line 2985: `pow(-np.log(...), 1/c)`.
- **Mechanism:** with c = 1e-3, `-log(...)` = x**c is close to 1 with a
  relative rounding error of ~eps. Raising it to 1/c = 1000 multiplies
  that error by 1000 (pattern 6). The result is beyond cond (arg) ~ 1.
- **Regime:** tiny c, all q; ~1/c ulp (moderate for c = 1e-3, 1.21e3 to
  1.34e3 ulp; 52 points of isf and 48 of ppf are minor).
- **Reproducer:** `python accuracy.py refs/truncweibull_min_ref.py --at ppf 0.999999938945977 0.001 0.5 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.999999938945977, 0.001, 0.5, 2)` | 1.9999998307228788 | 1.9999998307222846 | 1.34e3 | 1.39 | moderate |
  | `isf(0.08, 0.001, 0.5, 2)` | 1.7900501172776497 | 1.7900501172771697 | 1.21e3 | 0.111 | moderate |

- **Fix direction:** compute `log(x**c)` accurately (`log1p(x**c - 1)`, with
  `x**c - 1` from `expm1(c*log(a))` and the log1p term of F5), then
  `x = exp(log(x**c)/c)`.

## RuntimeWarnings

- `underflow encountered in exp` whenever b**c > ~745, even when the result
  is accurate (spurious): e.g. `cdf(1.0, 2, 0, 30)` returns
  0.6321205588285577 exactly (0 ulp) and still warns.
- `invalid value encountered in divide`, `divide by zero encountered in log`,
  `invalid value encountered in subtract`: in the F1 and F2 regimes (nan/inf
  results) and in logcdf/logsf when a numerator is 0 (F3, F4). In these
  cases the warnings coincide with wrong results.
- `underflow encountered in power`/`multiply`: for very small x or q
  (x**c or q*exp(-b**c) underflow). The results there are either accurate or
  covered by F3/F5.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | (2.5, 0.25, 1.75), (1, 0, 4.69), (5, 0, 3), (2.3, 15.3, 16): full support | 505 ulp, within cond 1.25e3 (a**c ~ 530: exp of a rounded large power, within conditioning) |
| logpdf | same sets | 246 ulp within cond 1.0e3; minor 37 ulp at (1, 0, 4.69), x = 0.00469 (result ~0.0045; several O(1) terms partly cancel) |
| cdf, sf | same sets away from the end points (F3/F4) | 39 ulp (minor, (5, 0, 3), x = 0.3) |
| logcdf, logsf | same sets away from the end points | 400 ulp minor at (5, 0, 3), x = 1.5 (cond 38): `log(1 - exp(-x**c))` with exp(-x**c) ~ 5e-4 instead of `log1p(-exp(-x**c))` (pattern 2, same mechanism as F4, milder) |
| ppf | (2.5, 0.25, 1.75), (2.3, 15.3, 16), q in [1e-300, 1 - 3e-16]; (1, 0, 4.69) for q >~ 1e-3 | 3 ulp; 13 ulp minor at (1, 0, 4.69), q = 0.01 (F5 tail) |
| isf | (2.5, 0.25, 1.75), (2.3, 15.3, 16), (5, 0, 3), (2, 1, 1.00000001) | 8 ulp |
| cdf, sf, ppf, isf | narrow interval (2, 1, 1.00000001) | errors up to 1.9e13 ulp, all within cond (arg) ~ x/(x - a) |

## Suspected, not confirmed

- `_munp` (lines 2994-2999, not a measured method): `sc.gamma(n/c + 1)`
  overflows for n/c > ~170 (pattern 3), and `gammainc(., b**c) - gammainc(., a**c)`
  cancels for narrow intervals or when both are near 1 (pattern 5). It shares
  the normaliser of F1/F2 (nan for a**c > 745). Moments are outside the
  scope of this harness, so this is not measured.

## Opportunities: inherited methods

none (all methods are implemented)

## Shared code

none (no helpers; the problems are local expressions. sc.xlogy is used
correctly).

## Existing GitHub issues

none found (web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(30.5, 2, 30, 31)` | nan | 4.4455116984500124e-12 | verified | also cdf/logsf nan, logpdf(30) inf, ppf/isf(0.5) inf as reported; cdf ref cross-checked by mp.quad of pdf |
| F1 | `pdf(27.29, 2, 26.8, 27.5)` | 0.0 | 1.6844152066888337e-10 | verified | ppf(0.9999999999995953) inf vs 27.32719546767614; cdf(26.8007) 3.5e5 ulp (moderate) also reproduced |
| F2 | `pdf(5e-9, 2, 0, 7e-9)` | inf | 204081632.65306124 | verified | cdf/sf nan, ppf(0.5) 0.0 reproduced |
| F2 | `pdf(5e-7, 2, 0, 1e-6)` | 1000022.1222092527 | 1000000.00000025 | verified | 1.0e11 ulp; ppf/isf(0.5) 2.0e11 ulp; narrow interval (2, 1, 1.00000001) pdf 2.3e7 ulp also reproduced |
| F3 | `cdf(4.69e-20, 1, 0, 4.69)` | 0.0 | 4.7334850415015533e-20 | verified | logcdf -inf, logsf 0.0 reproduced; cdf(0.2500000000015, 2.5, 0.25, 1.75) 7.9e11 ulp reproduced |
| F4 | `sf(9.99999e-07, 2, 0, 1e-6)` | 0.0 | 1.9999990000542074e-06 | verified | logsf -inf, logcdf 0.0 reproduced; logcdf(15.685, 2.3, 15.3, 16) 0.0 vs -2.7789878924530306e-14 reproduced |
| F5 | `ppf(1e-10, 1, 0, 4.69)` | 9.908129872606834e-11 | 9.9081331389284092e-11 | verified | 1.5e9 ulp; ppf(1e-17) and ppf(1e-300) return 0.0 (loss); isf(0.99999280314327, 2, 0, 1e-6) 0.0 also reproduced |
| F6 | `ppf(0.999999938945977, 0.001, 0.5, 2)` | 1.9999998307228788 | 1.9999998307222846 | verified | 1.3e3 ulp; isf(0.08) 1.2e3 ulp |

All reference values agree to 30 digits at 100 vs 200 digits (script: `scratch/truncweibull_min/verify/verify.py`). Mechanisms match the cited lines 2957-2991. Every row in the report's tables for F1-F6 (48 calls) was rechecked, and all match the report's SciPy values and references to the quoted precision.

Verified by dist-accuracy-verifier: 6 of 6 verified.
