# tukeylambda: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `tukeylambda_gen`, lines 10856-10915
- **Implemented:** pdf, cdf, ppf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** none; helpers: `_tlvar`, `_tlkurt` (stats only); special functions: `sc.tklmbda` (cdf, pdf), `sc.boxcox`, `sc.boxcox1p` (ppf)
- **Reference:** `refs/tukeylambda_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (3.1321477856738267,), (-1,), (-3,), (1e-05,), (0.001,), (0,), (0.14,), (0.5,), (2,), (50,)
- **Worst finding:** loss in cdf, pdf and ppf

## Summary

The distribution is defined through its quantile function
Q(p) = (p**lam - (1-p)**lam)/lam, and the cdf is `sc.tklmbda`, which inverts Q
by plain bisection with an *absolute* stopping tolerance (~1e-14) and at most
~60 steps. As a result the cdf is only accurate to ~10-100 ulp in the bulk,
loses all relative accuracy once F(x) is below ~1e-13 (the result is pinned at
7.1e-15 = 2**-47 for every lam != 0 with unbounded support, e.g. cdf(-1e300, -1)),
and loses up to ~5e7 ulp near the finite support ends for lam > 0. For
0 < |lam| < 1e-4 `tklmbda` substitutes the logistic cdf, giving relative errors
of 5e-4 to 6e-2. The pdf is built from that cdf and inherits every cdf error;
in addition it forms `1 - F` and returns 0 in the right tail. The ppf is
accurate in both tails (up to ~400 ulp from `sc.boxcox` for lam < 0 and
tiny q) but `boxcox(q) - boxcox1p(-q)` cancels near q = 1/2, severely for
large lam (3.2e13 ulp at q = 0.49, lam = 50) and to 0 next to q = 1/2.
Only lam = 0 exactly (logistic closed form) is accurate everywhere in cdf.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, pdf | lam != 0, F(x) < ~1e-13 (left tail, both tails for pdf) | bisection in `sc.tklmbda` stops at absolute tolerance (pattern 10) | cdf 7.1e-15 for 1e-300; pdf 5.0e-29 for 1e-300 | loss |
| F2 | cdf, pdf | 0 < \|lam\| < 1e-4, any x away from 0 | `sc.tklmbda` replaces the cdf by the logistic cdf (pattern 9) | 2.0e13 ulp (relative 4.5e-3) | severe |
| F3 | cdf, pdf | lam != 0, bulk and near the support ends for lam > 0 | absolute bisection tolerance gives ~1e-14 absolute error (pattern 10) | cdf 5.2e7 ulp near x = -1/lam; pdf 1.2e6 ulp | severe |
| F4 | pdf | lam < 1, right tail x > 0 where F rounds to 1 | `(1-Fx)**(lam-1)` cancels (pattern 1) | 0.0 for 3.7e-44 | loss |
| F5 | ppf | q near 1/2, all lam; large lam over a wide range | `boxcox(q) - boxcox1p(-q)` cancels (pattern 5) | 0.0 for 3.6e-25; 3.2e13 ulp at q = 0.49 | loss |
| F6 | cdf, pdf | lam > 0, x = -1/lam or 1/lam rounded to double | support test on the rounded endpoint | 0.0 for 1.5e-17 (cdf), 0.0 for 1.0 (pdf) | loss |
| F7 | ppf | lam < 0, tiny q (\|lam log q\| large) | `sc.boxcox` uses `expm1(lam*log(q))`, the rounding of the product is amplified (pattern 6) | 406 ulp | minor |

### F1: cdf (and pdf) pinned at ~7e-15 in the tails

- **Code:** `_cdf`, line 10901: `sc.tklmbda(x, lam)`; `_pdf`, line 10894 uses the same value.
- **Mechanism:** `tklmbda` (cephes `tukey.c`, now in xsf) bisects
  `p` in [0, 1] on `(p**lam - (1-p)**lam)/lam = x` and stops when
  `|pmid - plow| <= 1e-14` (absolute) or after a fixed count. In the left
  tail `pmid` halves from 0.5 until the stop rule triggers at
  2**-47 = 7.105e-15, so every F(x) below that is returned as 7.1e-15
  (relative accuracy is lost already around F ~ 1e-10). The pdf then uses this
  F in `Fx**(lam-1) + (1-Fx)**(lam-1)`, so pdf saturates too (at 5.05e-29 for
  lam = -1, 2.5e-57 for lam = -3); by symmetry this affects both tails of the
  pdf. Not a bug in how stats calls the function: `tklmbda` itself is inaccurate.
- **Regime:** lam != 0 (lam = 0 has a closed-form branch) with F(x) below
  ~1e-10; for lam = -1 that is x < -1e10, for lam = 0.14 x < ~-6.5,
  for lam = 0.001 x < ~-30.
- **Reproducer:** `python accuracy.py refs/tukeylambda_ref.py --at cdf -1e300 -1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-1e+300, -1)` | 7.105427357601002e-15 | 1e-300 | 3.2e+301 | 1 | loss |
  | `cdf(-500, 0.001)` | 7.105427357601002e-15 | 9.332636185032129e-302 | 3.4e+302 | 1e+03 | loss |
  | `cdf(-7.142857142857071, 0.14)` | 7.105427357601002e-15 | 9.466179456031815e-101 | 3.4e+101 | 7.2e+14 | loss |
  | `cdf(-1e5, -1)` | 9.999899994284078e-06 | 9.99990000000001e-06 | 2.6e+06 | 1 | severe |
  | `pdf(-1e+150, -1)` | 5.048709793414476e-29 | 1e-300 | 2.3e+287 | 2 | loss |
  | `pdf(1e+150, -1)` | 5.048709793414476e-29 | 1e-300 | 2.3e+287 | 2 | loss |
  | `pdf(-1e+230, -3)` | 2.5489470578119236e-57 | 4.979338607285738e-308 | 2.6e+266 | 1.33 | loss |
  | `pdf(-500, 0.001)` | 7.340719233781989e-15 | 1.866527237006426e-301 | 1.8e+302 | 999 | loss |

- **Fix direction:** in `tklmbda`, solve for log p (or for p with a relative
  tolerance) with Newton/Halley on Q, starting from the tail asymptote
  p ~ (lam*x)**(1/lam) (lam < 0) / p ~ (lam*(x + 1/lam))**(1/lam); compute
  only x <= 0 and use F(x) = 1 - F(-x).

### F2: logistic substitution for 0 < |lam| < 1e-4

- **Code:** `sc.tklmbda` (xsf/cephes `tukey`): for `-1e-4 < lam < 1e-4` it
  returns `1/(1 + exp(-x))`; used by `_cdf` (line 10901) and `_pdf` (line 10894).
- **Mechanism:** the lam -> 0 limit is used for nonzero lam (pattern 9).
  Q(p) = log(p/(1-p)) + (lam/2)(log(p)**2 - log(1-p)**2) + O(lam**2), so the
  x error is ~lam*log(p)**2/2, i.e. a relative cdf error of ~lam*x**2/2:
  4.4e-4 at x = -3 and 4.6e-2 at x = -30 for lam = 9.99e-5. Just above the
  threshold (lam = 1.0001e-4) cdf(-3) is accurate to 7e-13, so the
  approximation also creates a jump in lam.
- **Regime:** 0 < |lam| < 1e-4, |x| beyond ~0.01 (bulk and tails).
- **Reproducer:** `python accuracy.py refs/tukeylambda_ref.py --at cdf -10 1e-5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-10, 1e-05)` | 4.5397868702434395e-05 | 4.537517475312456e-05 | 2.3e+12 | 10 | severe |
  | `cdf(-30, 1e-05)` | 9.3576229688393e-14 | 9.315599883522394e-14 | 2.0e+13 | 30 | severe |
  | `cdf(10, 1e-05)` | 0.9999546021312976 | 0.9999546248252469 | 1.0e+08 | 0.000454 | severe |
  | `pdf(-10, 1e-05)` | 4.54003473582049e-05 | 4.5377653426672114e-05 | 2.3e+12 | 10 | severe |
  | `pdf(100, 1e-05)` | 0.0 | 3.542069758101544e-44 | inf | 100 | loss (F4) |

- **Fix direction:** drop the logistic shortcut for lam != 0; evaluate Q with
  `(expm1(lam*log p) - expm1(lam*log1p(-p)))/lam` (or `sc.boxcox`), which is
  accurate for any small lam, inside an accurate root finder (F1).

### F3: absolute bisection tolerance in the bulk and near the support ends

- **Code:** `sc.tklmbda` bisection (see F1), via `_cdf` line 10901 and `_pdf` line 10894.
- **Mechanism:** the bisection result carries an absolute error of order
  1e-14 (e.g. `cdf(-1e-15, -1)` = 0.4999999999999929, true 0.4999999999999997).
  In the bulk this is 10-100 ulp (median 51 ulp over the bulk grid, 96 ulp at
  `cdf(-0.3, 2)`). The relative error grows as 1e-14/F, so it is moderate
  to severe for F in 1e-12 ... 1e-4. For lam > 0 the support end -1/lam is
  approached with F ~ (x + 1/lam) small: the error exceeds the (already large)
  conditioning there. The pdf for lam != 1, 2 inherits the error amplified by
  |lam - 1|.
- **Regime:** all lam != 0; severe once F < ~1e-6 (lam > 0 near -1/lam;
  lam <= 0 for |x| moderately large), minor (10-1000 ulp) in the bulk.
- **Reproducer:** `python accuracy.py refs/tukeylambda_ref.py --at cdf -0.4999999 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-0.4999999, 2)` | 1.0000000116860974e-07 | 1.0000000000287557e-07 | 5.2e+07 | 5.0e+06 | severe |
  | `cdf(-0.019999, 50)` | 1.000024504094199e-06 | 1.000024500809114e-06 | 1.5e+07 | 2e+04 | severe |
  | `cdf(-6, 0.14)` | 2.066032386949246e-06 | 2.066032387570324e-06 | 1.4e+06 | 37.5 | severe |
  | `pdf(-6, 0.14)` | 1.2912559025713061e-05 | 1.2912559029051275e-05 | 1.2e+06 | 32.2 | severe |
  | `pdf(-0.0199, 50)` | 1.0049243760684752 | 1.0049243760688205 | 1.55e+03 | 0.98 | moderate |
  | `cdf(-0.3, 2)` | 0.19999999999999574 | 0.2 | 96.2 | 1.5 | minor |

- **Fix direction:** same as F1 (relative-accuracy root finding, e.g. Newton
  on Q in log p, terminating on a relative tolerance).

### F4: pdf right tail: `1 - Fx` cancels

- **Code:** `_pdf`, line 10895: `Px = Fx**(lam-1.0) + (np.asarray(1-Fx))**(lam-1.0)`
- **Mechanism:** for x > 0, Fx is close to 1 and `1 - Fx` keeps only the
  digits beyond the leading ones (pattern 1); once Fx rounds to 1, `0**(lam-1)`
  is inf for lam < 1 (divide-by-zero warning) and the pdf becomes 0. The pdf is
  symmetric, so the left-tail value is the exact answer.
- **Regime:** lam < 1, x > 0 with sf(x) below ~1e-16 (lam = 0: x > ~37);
  partial loss proportional to 1/sf before that. Present even for lam = 0,
  where the cdf itself is exact.
- **Reproducer:** `python accuracy.py refs/tukeylambda_ref.py --at pdf 100 0`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(100, 0)` | 0.0 | 3.720075976020836e-44 | inf | 100 | loss |
  | `pdf(700, 0)` | 0.0 | 9.85967654375977e-305 | inf | 700 | loss |
  | `pdf(100, 1e-05)` | 0.0 | 3.542069758101544e-44 | inf | 100 | loss |

- **Fix direction:** evaluate at `-abs(x)`: `Fx = tklmbda(-abs(x), lam)`, then
  `1 - Fx` is accurate because Fx <= 1/2.

### F5: ppf cancels near q = 1/2 (catastrophically for large lam)

- **Code:** `_ppf`, line 10904: `sc.boxcox(q, lam) - sc.boxcox1p(-q, lam)`
- **Mechanism:** boxcox(q, lam) = (q**lam - 1)/lam and
  boxcox1p(-q, lam) = ((1-q)**lam - 1)/lam are both close to (2**-lam - 1)/lam;
  their difference Q(q) is much smaller (pattern 5). The absolute error is
  ~eps*|(2**-lam - 1)/lam|, i.e. relative error ~eps/(|Q(q)| * lam). Next to
  q = 1/2 the result is 0 for every lam. For large lam, Q(q) ~ q**lam/lam is
  tiny over a wide range of q (lam = 50: Q(0.49) = -4.1e-17 while
  each term is ~-0.02), so the error far exceeds the conditioning (63).
- **Regime:** |q - 1/2| within a few ulp for all lam (loss; ill-conditioned
  there anyway, cond 4.5e15); for large lam (50) severe for q in
  ~0.25 ... 0.75, moderate out to ~0.2 / 0.8.
- **Reproducer:** `python accuracy.py refs/tukeylambda_ref.py --at ppf 0.49 50`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.49, 50)` | -4.163336342344337e-17 | -4.134325502367792e-17 | 3.2e+13 | 63.4 | severe |
  | `ppf(0.42999999999999994, 50)` | -1.2437967322753707e-14 | -1.24386271535106e-14 | 2.4e+11 | 37.7 | severe |
  | `ppf(0.29, 50)` | -7.310505881208407e-10 | -7.31050589444507e-10 | 8.2e+06 | 20.4 | severe |
  | `ppf(0.21999999999999997, 50)` | -8.049337368518095e-08 | -8.049337368732137e-08 | 1.2e+05 | 14.1 | moderate |
  | `ppf(0.5000000001, 50)` | 0.0 | 3.5527139727533488e-25 | inf | 5.0e+09 | loss |
  | `ppf(0.5000000000000001, 3.1321477856738267)` | 0.0 | 5.065237188887399e-17 | inf | 4.5e+15 | loss |

- **Fix direction:** avoid the common `-1/lam`: for q <= 1/2,
  `Q = -(1-q)**lam * expm1(lam*log1p((2q-1)/(1-q))) / lam`... more simply
  `Q = (q**lam - (1-q)**lam)/lam` with a series in (2q - 1) near 1/2, and
  `isf = -ppf` by symmetry.

### F6: support endpoint rounded to double

- **Code:** `_pdf`, line 10898: `abs(x) < 1.0/np.asarray(lam)`; `sc.tklmbda`
  returns 0 for `x <= -1/lam` (computed in double).
- **Mechanism:** the support end 1/lam is rounded; when `fl(1/lam)` is below
  the true 1/lam (lam = 3.1321477856738267), the point x = fl(1/lam) lies inside
  the support but is treated as outside, so cdf(-fl(1/lam)) = 0 and
  pdf = 0 (true pdf 1.0 for lam > 1, as f -> 1 at the endpoints).
- **Regime:** lam > 0, |x| = fl(1/lam) only (one double per side).
  Practically harmless; reported because the tool flags it as loss.
- **Reproducer:** `python accuracy.py refs/tukeylambda_ref.py --at pdf 0.3192697370711285 3.1321477856738267`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.3192697370711285, 3.1321477856738267)` | 0.0 | 1.0 | inf | 0.681 | loss |
  | `cdf(-0.3192697370711285, 3.1321477856738267)` | 0.0 | 1.4820741312818743e-17 | inf | 2.2e+16 | loss |

- **Fix direction:** none needed in practice; could compare `lam*|x| < 1`
  instead of `|x| < 1/lam`.

### F7: ppf for lam < 0 and tiny q: rounding amplified in `sc.boxcox`

- **Code:** `_ppf`, line 10904; `sc.boxcox(q, lam)` computes
  `expm1(lam*log(q))/lam`.
- **Mechanism:** the rounding of `lam*log(q)` (~690 for q = 1e-100, lam = -3)
  is amplified by exp into a relative error of ~|lam*log q|*eps/2 (pattern 6);
  `q**lam` directly would be accurate to ~1 ulp.
- **Regime:** lam < 0, q small (and by the same formula q near 1 via
  `boxcox1p`); error ~|lam*log(q)|/2 ulp, at most ~400 ulp.
- **Reproducer:** `python accuracy.py refs/tukeylambda_ref.py --at ppf 1e-100 -3`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-100, -3)` | -3.3333333333336335e+299 | -3.333333333333333e+299 | 406 | 3 | minor |
  | `ppf(1e-300, -1)` | -9.999999999999763e+299 | -9.999999999999999e+299 | 106 | 1 | minor |

- **Fix direction:** in the tails use `(q**lam - (1-q)**lam)/lam` (`pow`
  instead of `expm1(lam*log q)`) when |lam*log q| is large.

## RuntimeWarnings

`pdf` emits "divide by zero encountered in power" when `Fx` rounds to 1 and
lam < 1 (`0**(lam-1)`, F4); the returned 0 is wrong (loss). "underflow
encountered in power" appears for large lam (50), where `Fx**(lam-1)` or
`(1-Fx)**(lam-1)` underflows; the other term dominates and the result is as
accurate as the cdf allows (spurious warning). Three parameter sets affected.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | lam = 0 (exact logistic branch), full grid | 0.869 ulp |
| pdf | lam = 2 (uniform), full grid | 0 ulp |
| ppf | low tail and high tail, all parameter sets, apart from F7 | 411 ulp (F7), others within conditioning |
| ppf | tiny lam (1e-5, 0.001, 0), full q grid | within conditioning except next to q = 1/2 (F5) |
| ppf | lam < 0 with q**lam overflow (true value overflows too) | not a finding |

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | `cdf(-x)` (distribution is symmetric) | right tail; total loss once cdf rounds to 1 (lam = 0: x > ~37) |
| logcdf | log(cdf) | log(tklmbda) once tklmbda is accurate; lam <= 0 tail: `log p` from the root finder in log p | inherits F1/F3 (cdf pinned at 7.1e-15 gives logcdf = log(7.1e-15) = -32.6 for every x beyond the F1 threshold) |
| logsf | log(sf) | `logcdf(-x)` | right tail, as sf |
| logpdf | log(pdf) | `-log(F**(lam-1) + (1-F)**(lam-1))` with F = cdf(-abs(x)), e.g. `-(lam-1)*log(F) - log1p(((1-F)/F)**(lam-1))` | pdf underflow (lam = 0: x > ~745) and all pdf findings |
| isf | ppf(1 - q) | `-ppf(q)` (symmetry) | small q: `1 - q` rounds, e.g. `isf(1e-10, -1)` = 9999999171.6 vs 9999999999.0 (8e-8 relative); q < eps/2 gives inf (lam <= 0) or 1/lam |

Demonstration: by symmetry `sf(x) = cdf(-x)`; with lam = 0 the default
`sf(40, 0)` returns 0.0 while `cdf(-40, 0)` = 4.248354255291589e-18 (exact branch); `isf(1e-20, -1)` returns inf while `-ppf(1e-20, -1)` = 1.0000000000000008e+20.

## Shared code

- `scipy.special.tklmbda` (the public Tukey-lambda cdf) is the source of F1, F2
  and F3: bisection with an absolute tolerance (~1e-14, cdf pinned at 2**-47 in
  the tails) and a logistic substitution for 0 < |lam| < 1e-4. Any user of
  `sc.tklmbda` is affected; in `scipy.stats` only tukeylambda uses it.
- `scipy.special.boxcox` / `boxcox1p` lose up to ~|lam*log(x)|/2 ulp for large
  |lam*log(x)| (`boxcox(1e-100, -3)`: 410 ulp, `boxcox(1e-300, -1)`: 108 ulp)
  because of `expm1(lam*log(x))` (F7). Minor; affects every user of boxcox.

## Existing GitHub issues

- #21370 (open): BUG: stats.tukeylambda: Bad behavior of the cdf() and sf() methods in the tails - relates to F1 (tklmbda bisection; pdf levels off) and F5 (ppf near 1/2, from a comment)

(web search; gh unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-1e300, -1)`; `pdf(-1e150, -1)` | 7.105427357601002e-15; 5.048709793414476e-29 | 1e-300; 1e-300 | verified | Reference by full-precision bisection in t = log p on Q(e^t) = x (100 vs 200 digits agree, Q(F) - x ~ 1e-200). All 8 table rows reproduce (also `cdf(-500, 0.001)`, `cdf(-7.142857142857071, 0.14)`, `cdf(-1e5, -1)` 2.6e6 ulp, `pdf(-1e230, -3)`, `pdf(-500, 0.001)`). Returned value is exactly 2**-47, consistent with the stated absolute stop rule. |
| F2 | `cdf(-10, 1e-5)`; `cdf(-30, 1e-5)` | 4.5397868702434395e-05; 9.3576229688393e-14 | 4.537517475312456e-05; 9.315599883522394e-14 | verified | 2.3e12 / 2.0e13 ulp. `sc.tklmbda(-10, 1e-5)` equals the logistic 1/(1+e^10) bit for bit; lam = 1e-4 itself already uses bisection, so the regime is \|lam\| < 1e-4 as stated. Jump check: `cdf(-3, 1.0001e-4)` is 3.2e3 ulp (7e-13 relative), as reported. |
| F3 | `cdf(-0.4999999, 2)`; `pdf(-6, 0.14)` | 1.0000000116860974e-07; 1.2912559025713061e-05 | 1.0000000000287557e-07; 1.2912559029051275e-05 | verified | 5.2e7 / 1.2e6 ulp; other rows also reproduce (`cdf(-0.019999, 50)` 1.5e7, `cdf(-6, 0.14)` 1.4e6, `cdf(-0.3, 2)` 96 ulp). Small text error in the mechanism: the true `cdf(-1e-15, -1)` is 0.4999999999999999 (F = 1/2 + x/8), not 0.4999999999999997; SciPy's 0.4999999999999929 is 63 ulp off. |
| F4 | `pdf(100, 0)`; `pdf(700, 0)` | 0.0; 0.0 | 3.720075976020836e-44; 9.85967654375977e-305 | verified | Reference from logistic pdf e^-\|x\|/(1+e^-\|x\|)^2. Mechanism confirmed: `tklmbda(100, 0)` = 1.0 so `(1-Fx)**(lam-1)` = inf (line 10895). `pdf(100, 1e-5)` also 0.0 vs 3.542069758101544e-44. |
| F5 | `ppf(0.49, 50)`; `ppf(0.5000000001, 50)` | -4.163336342344337e-17; 0.0 | -4.134325502367792e-17; 3.5527139727533488e-25 | verified | 3.2e13 ulp / total loss. `boxcox(0.49, 50)` = -0.019999999999999993 and `boxcox1p(-0.49, 50)` = -0.019999999999999952, confirming the cancellation at line 10904. Rows at q = 0.43, 0.29, 0.22 and `ppf(0.5000000000000001, 3.132...)` also reproduce. |
| F6 | `pdf(0.3192697370711285, 3.1321477856738267)`; `cdf(-0.3192697370711285, 3.1321477856738267)` | 0.0; 0.0 | 1.0; 1.4820741312818743e-17 | verified | x == fl(1/lam) and x < 1/lam exactly, so the point is inside the support; the pdf mask (line 10898) and `tklmbda` both treat it as outside. |

Verified by dist-accuracy-verifier: 6 of 6 verified.
