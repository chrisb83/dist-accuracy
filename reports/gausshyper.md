# gausshyper: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `gausshyper_gen`, lines 4871-4924
- **Implemented:** pdf (lines 4915-4918)
- **Inherited (not measured):** logpdf = log(pdf), cdf = numerical integration of pdf, sf = 1 - cdf, logcdf / logsf = generic median switch (log(cdf) / log1p(-sf) and log(sf) / log1p(-cdf)), ppf = numerical inversion of cdf, isf = ppf(1 - q)
- **Related:** beta (z = 0 or c = 0 case; separate code, `reports/beta.md`); helpers: none; special functions: sc.beta, sc.hyp2f1
- **Reference:** `refs/gausshyper_ref.py`, 400 digits; self-checks: PASS (the module has pdf and logpdf only, no cdf: see Summary; normalization checked separately in `scratch/gausshyper/check_norm.py`, |integral - 1| < 2e-34 for all sets)
- **Parameter sets:** (13.7637716041307, 3.118963664868143, 2.514598035018302, 5.1811649903971615), (2, 3, 1, 0.5), (1.5, 2.5, 3, -0.999999), (300, 300, 5, 2), (2, 1e4, 3, 0.5), (3, 2, 1e4, 1e-4), (2.5, 1.5, 2.5, 1e6), (2, 3, -60, 1e3), (0.01, 0.01, 2, 10), (2.5, 1.5, 2.500000000001, 1e6), (3, 2, 1e4, 1), (3, 2, 1e3, -0.5)
- **Worst finding:** loss in pdf

## Summary

`_pdf` is the only implemented method: it multiplies `x**(a-1)`, `(1-x)**(b-1)`,
`(1+z*x)**-c` and `1/(sc.beta(a, b) * sc.hyp2f1(c, a, a+b, -z))` in linear
space. For moderate shapes it is accurate to a few ulp (2.2 ulp max for
(2, 3, 1, 0.5); z close to -1 stays within conditioning). It fails completely
(0, inf, nan) when any factor over- or underflows although the pdf is a
normal number: large a, b (powers underflow, beta underflows), large |c| with
large z (hyp2f1 overflows or returns nan). Separately, the normalization
constant inherits large constant relative errors from `scipy.special`:
`sc.hyp2f1` loses up to 1.6e11 ulp (c within 1e-12 of a, z = 1e6) and 2e6 ulp
(large c, small z), and `sc.beta` up to ~1e6 ulp for very unbalanced shapes;
these errors apply to every x. A subnormal `x**(a-1)` costs up to 6.8e6 ulp
in the far left tail. No cdf reference was built (the cdf is an Appell F1
function; `mp.appellf1` and `mp.quad` take minutes per point at 400 digits),
so the harness reports region "all" rather than tails; the plot step of the
harness exited with code 127 and no `errors.png` was produced.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf | large a or b, x**(a-1) or (1-x)**(b-1) < 2.2e-308 while 1/B(a, b) is huge | power underflows before the large normalization multiplies it (pattern 3/4) | 0.0 for 8.6e-214 | loss |
| F2 | pdf | B(a, b) < 2.2e-308 (e.g. a = b >~ 540), all x | sc.beta underflows to 0, 1/0 = inf (pattern 3) | inf for 27.6 | loss |
| F3 | pdf | H = 2F1(c, a; a+b; -z) overflows (c << 0, large z) or sc.hyp2f1 returns nan (large c > 0 with z >~ 1, or c ~ 1e5) | normalization not in log space; sc.hyp2f1 returns nan (pattern 3 / 10) | nan for 0.0273 | loss |
| F4 | pdf | c equal or close to a, large z (all x) | sc.hyp2f1 inaccurate in the (nearly) degenerate 1/w transformation (pattern 10, shared) | 1.6e11 ulp | severe |
| F5 | pdf | large c, small z (all x) | sc.hyp2f1 inaccurate for large first parameter (pattern 10, shared) | 2.0e6 ulp | severe |
| F6 | pdf | x**(a-1) subnormal, pdf normal (small x, large 1/(B H)) | subnormal intermediate (pattern 4) | 6.8e6 ulp | severe |
| F7 | pdf | max(a, b) large, especially min(a, b) small (all x) | sc.beta inaccurate (pattern 10, shared) | 9.6e5 ulp | moderate |
| F8 | pdf | large c with small z*x; large b with x < 0.5 | rounding of 1 + z*x and 1 - x raised to power c, b - 1 (pattern 6) | 501 ulp | minor |
| F9 | pdf | a < 1 (or b < 1) not exactly representable as a - 1, tiny x | exponent a - 1 rounded before the power (pattern 10) | 27 ulp | minor |

### F1: power factors underflow before the normalization lifts them

- **Code:** `_pdf`, lines 4917-4918: `1./normalization_constant * x**(a - 1.) * (1. - x)**(b - 1.0) / (1.0 + z*x)**c`
- **Mechanism:** for large a and b, `1/B(a, b)` is huge (2e181 for a = b = 300)
  and the pdf is a ratio of two extreme numbers. `x**(a-1)` underflows to 0
  for (a-1)*log(x) < -745 even though the product with `1/B` is a normal number
  (pattern 3/4). The same happens to `(1-x)**(b-1)` near x = 1.
- **Regime:** x < exp(-745/(a-1)) (x < 0.083 for a = 300) with pdf above
  2.2e-308; symmetric near x = 1 with b. Needs log B(a, b) <~ -300, i.e.
  roughly a, b >~ 100 (both) or one shape large enough for 1/B to be large.
- **Reproducer:** `python accuracy.py refs/gausshyper_ref.py --at pdf 0.05 300 300 5 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.04683895000436239, 300, 300, 5, 2)` | 0.0 | 7.888886841497217e-222 | inf | 284 | loss |
  | `pdf(0.05, 300, 300, 5, 2)` | 0.0 | 8.582836406612371e-214 | inf | 283 | loss |
  | `pdf(0.07494232000697983, 300, 300, 5, 2)` | 0.0 | 8.586608358514109e-165 | inf | 274 | loss |
  | `pdf(0.9250576799930201, 300, 300, 5, 2)` | 0.0 | 9.178692601023885e-167 | inf | 3.39e+03 | loss |
  | `pdf(0.9500000000000001, 300, 300, 5, 2)` | 0.0 | 6.739136779323213e-216 | inf | 5.39e+03 | loss |
  | `pdf(0.9531610499956376, 300, 300, 5, 2)` | 0.0 | 5.953113830742001e-224 | inf | 5.79e+03 | loss |

- **Fix direction:** compute `exp(_logpdf)` with
  `_logpdf = xlogy(a-1, x) + xlog1py(b-1, -x) - c*log1p(z*x) - betaln(a, b) - log(hyp2f1(...))`.

### F2: sc.beta(a, b) underflows, pdf = inf in the bulk

- **Code:** `_pdf`, line 4916: `normalization_constant = sc.beta(a, b) * sc.hyp2f1(c, a, a + b, -z)`
- **Mechanism:** B(a, b) < 2.2e-308 underflows to 0 (pattern 3); `1./0` gives
  inf (with a divide-by-zero warning), and `inf * 0` would give nan where the
  power factors underflow too. The pdf itself is moderate.
- **Regime:** log B(a, b) < -708, e.g. a = b >~ 540 (B(600, 600) ~ 1e-363),
  every x in (0, 1).
- **Reproducer:** `python accuracy.py refs/gausshyper_ref.py --at pdf 0.5 600 600 1 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.5, 600, 600, 1, 1)` | inf | 27.63121729789657 | inf | 0.333 | loss |
  | `pdf(0.5, 1000, 1000, 0, 0.5)` | inf | 35.67802229170864 | inf | 0 | loss |

- **Fix direction:** `betaln` in the log-space form of F1.

### F3: hyp2f1 overflows or returns nan, pdf = nan for all x

- **Code:** `_pdf`, line 4916 (`sc.hyp2f1(c, a, a + b, -z)`) and line 4918 (`(1.0 + z*x)**c`).
- **Mechanism:** two cases.
  (a) c << 0 with large z: H = 2F1(c, a; a+b; -z) grows like z**(-c) and
  overflows (H = 1.46e355 for (2, 3, -120, 1e3)); `(1 + z*x)**c` underflows at
  the same time. The two cancel in the pdf, but the code forms them
  separately (pattern 3). `sc.hyp2f1` returns nan rather than inf here.
  (b) large c > 0: `sc.hyp2f1` returns nan although the true value is a
  normal number, e.g. `hyp2f1(1e4, 3, 5, -1)`, `hyp2f1(1000, 2, 5, -10)`,
  `hyp2f1(1e5, 3, 5, -1e-5)` (pattern 10; a scipy.special problem, see
  Shared code).
- **Regime:** (a) roughly -c*log10(z) > 308; (b) c >~ 1e3 with z >~ 1, or
  c >~ 1e5 already for small z (c*z = 1). Every x is affected.
- **Reproducer:** `python accuracy.py refs/gausshyper_ref.py --at pdf 0.9 2 3 -120 1000`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.9, 2, 3, -120, 1000)` | nan | 0.02732079271919567 | inf | 103 | loss |
  | `pdf(0.5, 2, 3, -120, 1000)` | nan | 9.833610142185653e-32 | inf | 119 | loss |
  | `pdf(1e-100, 3, 2, 1e4, 1)` | nan | 4.9985007002401806e-189 | inf | 2 | loss |
  | `pdf(1e-150, 3, 2, 1e4, 1)` | nan | 4.99850070024018e-289 | inf | 2 | loss |
  | `pdf(0.0001, 2, 3, 1000, 10)` | nan | 3670.336952688618 | inf | 0.000799 | loss |
  | `pdf(0.5, 3, 2, 1e5, 1e-5)` | nan | 1.6243847091650265 | inf | 0.5 | loss |

  The parameter set (3, 2, 1e4, 1) is a loss at all 23 grid points where the
  pdf is a normal number.
- **Fix direction:** for (a), Pfaff: H = (1+z)**-c * 2F1(c, b; a+b; z/(1+z)),
  and combine `(1+z)**-c` with `(1+z*x)**-c` into
  `exp(-c*(log1p(z*x) - log1p(z)))` (scipy's Pfaff-form hyp2f1 is 18.7 ulp
  accurate for (2, 3, -120, 1e3)); for (b) the special function needs fixing
  (or a log-space / asymptotic evaluation of H for large c).

### F4: hyp2f1 inaccurate when c equals or nearly equals a, z large

- **Code:** `_pdf`, line 4916: `sc.hyp2f1(c, a, a + b, -z)`
- **Mechanism:** for argument w = -z << -1, hyp2f1 uses the 1/w (or
  1/(1-w)) transformation, whose two terms have Gamma(a - c) / Gamma(c - a)
  factors; these are singular for c = a (degenerate case, handled by a
  limit formula) and cancel catastrophically for c - a tiny but non-zero
  (pattern 10). The relative error of H is copied into the pdf at every x.
- **Regime:** |c - a| small and z large. Measured hyp2f1 error for
  (a, b) = (2.5, 1.5): c = a: 29 ulp at z = 1e3, 1.9e4 at 1e6, 1.1e6 at 1e8,
  1.8e7 at 1e10, 3.9e9 at 1e12; z = 1e6: c - a = 1e-12: 1.6e11, 1e-8:
  1.6e7, 1e-4: 604, 0.1: 0.85 ulp (`scratch/gausshyper/factors.py`).
  The Pfaff form `2F1(c, b; a+b; z/(1+z))` in scipy is just as bad.
- **Reproducer:** `python accuracy.py refs/gausshyper_ref.py --at pdf 0.5 2.5 1.5 2.500000000001 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.4, 2.5, 1.5, 2.500000000001, 1e6)` | 0.16244243469837005 | 0.16243655282797095 | 1.6e+11 | 1.33 | severe |
  | `pdf(0.5, 2.5, 1.5, 2.500000000001, 1e6)` | 0.1186313293291851 | 0.11862703381301852 | 1.6e+11 | 1.5 | severe |
  | `pdf(0.5, 2.5, 1.5, 2.5, 1e10)` | 0.06692356424482472 | 0.06692356398247225 | 1.8e+07 | 1.5 | severe |
  | `pdf(0.5, 2.5, 1.5, 2.5, 1e6)` | 0.11862703381421483 | 0.11862703381371847 | 1.9e+04 | 1.5 | moderate |
  | `pdf(0.5, 2, 2, 2, 1e6)` | 0.08463397091080739 | 0.08463397091045013 | 1.9e+04 | 2 | moderate |

  The set (2.5, 1.5, 2.500000000001, 1e6) is severe at all 53 measured grid
  points (cond (shapes) about 37 to 40 in the bulk, far below the error).
- **Fix direction:** in `scipy.special.hyp2f1` (shared); nothing in
  `gausshyper` short of its own evaluation of H.

### F5: hyp2f1 inaccurate for large c

- **Code:** `_pdf`, line 4916: `sc.hyp2f1(c, a, a + b, -z)`
- **Mechanism:** `sc.hyp2f1(c, a; a+b; w)` with a large first parameter and
  small |w| loses accuracy roughly in proportion to c (pattern 10, a
  scipy.special problem). The error is a constant factor on the pdf.
- **Regime:** c >~ 100 with z of order 1/c or larger. Measured hyp2f1 error
  for (a, b) = (3, 2): c = 100, z = 1e-3: 617 ulp; c = 1e3, z = 1e-3: 1.6e4;
  c = 1e3, z = 1e-2: 1.3e4; c = 1e4, z = 1e-4: 2.0e6; c = 1e4, z = 1e-3:
  5.4e6; c = 1e3, z = -0.1: 3.6e3; c = 1e3, z = -0.5: 0.4 ulp. Beyond this
  regime hyp2f1 returns nan (F3b). The inventory example
  (13.76, 3.12, 2.51, 5.18) has a 36 ulp hyp2f1 error, which is why its bulk
  pdf points are 32 ulp (minor).
- **Reproducer:** `python accuracy.py refs/gausshyper_ref.py --at pdf 0.5 3 2 1e4 1e-4`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.5, 3, 2, 1e4, 1e-4)` | 1.6243770886862354 | 1.6243770894238292 | 2.0e+06 | 0.5 | severe |
  | `pdf(0.15000000000000002, 3, 2, 1e4, 1e-4)` | 0.3526764128506433 | 0.35267641301076025 | 2.0e+06 | 1.67 | severe |
  | `pdf(1e-100, 3, 2, 1e4, 1e-4)` | 2.1424892658156925e-199 | 2.1424892667862893e-199 | 2.0e+06 | 2 | severe |
  | `pdf(0.5, 3, 2, 1000, 0.01)` | 0.6062313889442201 | 0.6062313889424548 | 1.31e+04 | 3.98 | moderate |
  | `pdf(0.5, 3, 2, 100, 0.001)` | 1.514760688827982 | 1.5147606888281815 | 593 | 0.95 | minor |

  The set (3, 2, 1e4, 1e-4) is severe at all 45 measured points (cond
  (shapes) 4.5 in the bulk).
- **Fix direction:** in `scipy.special.hyp2f1` (shared).

### F6: subnormal x**(a-1) in the left tail

- **Code:** `_pdf`, line 4917: `1./normalization_constant * x**(a - 1.)`
- **Mechanism:** `x**(a-1)` is subnormal (fewer than 53 significant bits)
  while the pdf is normal because `1/(B H)` is large (1.6e13 for
  (2.5, 1.5, 2.5, 1e6), 1.2e5 for the inventory example) (pattern 4).
- **Regime:** 4.9e-324 < x**(a-1) < 2.2e-308 and pdf > 2.2e-308, i.e. a band
  of x just above the point where the pdf itself becomes subnormal.
- **Reproducer:** `python accuracy.py refs/gausshyper_ref.py --at pdf 1e-210 2.5 1.5 2.5 1e6`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-210, 2.5, 1.5, 2.5, 1e6)` | 8.388239932524648e-302 | 8.388239945225553e-302 | 6.8e+06 | 1.5 | severe |
  | `pdf(4.063247033746088e-25, 13.7637716041307, 3.118963664868143, 2.514598035018302, 5.1811649903971615)` | 5.811511216503015e-307 | 5.811511216501189e-307 | 1.42e+03 | 12.8 | moderate |

  (The 1e-210 point includes the constant 1.9e4 ulp of F4.)
- **Fix direction:** log-space evaluation as in F1.

### F7: sc.beta inaccurate for large, unbalanced shapes

- **Code:** `_pdf`, line 4916: `sc.beta(a, b)`
- **Mechanism:** `sc.beta` (cephes) loses accuracy when max(a, b) is large;
  the error is a constant factor on the pdf (pattern 10, shared).
  Measured `sc.beta` errors: (2, 1e3) 4.1e3 ulp, (2, 1e4) 3.8e4, (2, 1e6)
  1.1e6, (10, 1e4) 8.3e4, (0.5, 1e4) 3.6e3, (1, 1e4) 2.2e4, (3, 3000) 1.4e4,
  (100, 100) 135, (300, 300) 714; balanced moderate shapes (30, 30),
  (50, 60) are below 2 ulp. `sc.betaln` is much more accurate (2.1e3 ulp of
  a value of size 18 at (2, 1e4), i.e. ~3.8e4 ulp on exp, so betaln is not a
  drop-in cure either; a direct formula such as `gamma` ratios or
  `exp(betaln)` with care is needed).
- **Regime:** max(a, b) >~ 1e3, or both >~ 100; every x.
- **Reproducer:** `python accuracy.py refs/gausshyper_ref.py --at pdf 1e-6 2 1e6 3 0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-06, 2, 1e6, 3, 0.5)` | 367880.54488575936 | 367880.5448075739 | 9.6e+05 | 1.5e-06 | moderate |
  | `pdf(0.0016136787423080556, 2, 1e4, 3, 0.5)` | 0.0156265491704714 | 0.01562654917033011 | 4.07e+04 | 15.2 | moderate |
  | `pdf(1e-300, 2, 1e4, 3, 0.5)` | 1.0003999700264762e-292 | 1.000399970017987e-292 | 3.8e+04 | 1 | moderate |
  | `pdf(0.45, 300, 300, 5, 2)` | 1.2428615583585216 | 1.2428615583582872 | 850 | 52 | minor |
  | `pdf(0.5, 2, 1e3, 3, 0.5)` | 4.797408716109793e-296 | 4.7974087161054493e-296 | 4.08e+03 | 999 | moderate |

  The set (2, 1e4, 3, 0.5) is moderate at all 41 measured points; the extra
  ~2.5e3 ulp at x = 1.6e-3 over the 3.8e4 baseline is F8.
- **Fix direction:** shared: improve `sc.beta` for large arguments; in
  gausshyper, `betaln` in a log-space pdf at least removes the underflow (F2).

### F8: rounding of 1 + z*x and 1 - x amplified by large exponents

- **Code:** `_pdf`, line 4917-4918: `(1. - x)**(b - 1.0)` and `(1.0 + z*x)**c`
- **Mechanism:** the half-ulp rounding of `1 + z*x` (and of `1 - x` for
  x < 0.5) is multiplied by c (resp. b - 1) when raised to the power
  (pattern 6). The condition number with respect to x is about
  c*z*x/(1+z*x) and (b-1)*x/(1-x), which is small when z*x (resp. x) is small,
  so the error exceeds the conditioning there.
- **Regime:** large |c| with small |z*x|; large b with small x. Measured on
  the set (3, 2, 1e3, -0.5) (where `sc.hyp2f1` and `sc.beta` are exact to
  < 1 ulp): 8 minor points, worst 501 ulp. In isolation the factor errors are
  4.8e3 ulp for `(1 + 1e-4*0.5)**1e4` (cond 0.5) and 2.5e3 ulp for
  `(1 - 0.0016)**9999` (cond 16) (`scratch/gausshyper/factors.py`), but in
  the measured pdf those are hidden behind F5 and F7.
- **Reproducer:** `python accuracy.py refs/gausshyper_ref.py --at pdf 0.0031 3 2 1000 -0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(0.999, 3, 2, 1000, -0.5)` | 367.6992989044348 | 367.6992989043939 | 501 | 1 | minor |
  | `pdf(0.4, 3, 2, 1000, -0.5)` | 7.290005933125092e-200 | 7.290005933125594e-200 | 310 | 251 | minor |
  | `pdf(0.15000000000000002, 3, 2, 1000, -0.5)` | 1.2891763966459591e-263 | 1.2891763966460363e-263 | 270 | 82.9 | minor |
  | `pdf(0.0031, 3, 2, 1000, -0.5)` | 4.2217966141696236e-300 | 4.221796614169406e-300 | 232 | 3.55 | minor |

- **Fix direction:** `exp(-c*log1p(z*x))` and `exp((b-1)*log1p(-x))`
  (part of the log-space form of F1).

### F9: exponent a - 1 rounded before the power

- **Code:** `_pdf`, line 4917: `x**(a - 1.)`
- **Mechanism:** for a < 1 (here a = 0.01), `a - 1.` is rounded (error
  8.7e-18 for a = 0.01); the power multiplies this exponent error by
  |log x| = 690 at x = 1e-300 (pattern 10, argument rounded before the call).
  The error grows linearly in |log x|.
- **Regime:** small a not of the form 1 - 2**-k, tiny x; analogous for
  `(1-x)**(b-1)` with x close to 1.
- **Reproducer:** `python accuracy.py refs/gausshyper_ref.py --at pdf 1e-300 0.01 0.01 2 10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-300, 0.01, 0.01, 2, 10)` | 1.0237200666023874e+295 | 1.0237200666023935e+295 | 27 | 0.99 | minor |
  | `pdf(1e-230, 0.01, 0.01, 2, 10)` | 5.130754281891783e+225 | 5.130754281891805e+225 | 19.6 | 0.99 | minor |

  20 minor points on this set, all in the left tail.
- **Fix direction:** `exp((a - 1)*log(x))` does not help (same exponent);
  `x**a / x` (a exact) avoids rounding a - 1, or `xlogy(a, x) - log(x)` in log space.

## RuntimeWarnings

"underflow encountered in power", "underflow encountered in multiply",
"underflow encountered in divide", "overflow encountered in power" and
"divide by zero" (in power, and in divide for F2) occur in 10 of 12 parameter
sets. They coincide with the failures of F1, F2, F3 and F6, and also appear
at points where the true pdf is subnormal or 0 (e.g. `pdf(0.5, 2, 1e4, 3,
0.5)` = 0.0 vs a true 2.6e-3003), where the result is correct. No warning is
emitted when sc.hyp2f1 silently returns nan (F3b).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | (2, 3, 1, 0.5), full grid (0 + 1e-300 ... 1 - 1e-300) | 2.19 ulp |
| pdf | z = -0.999999 (1 + z*x cancels near x = 1, H = 5984) | within conditioning (1.5e5 ulp at x = 1 - 1e-10 with cond 1.5e10) |
| pdf | c = -60, z = 1e3 (H = 1e176, no overflow yet) | 23.6 ulp, one point minor, otherwise within conditioning |
| pdf | tiny shapes a = b = 0.01 (pdf up to 1e295) | 27 ulp (F9) |
| pdf | inventory example, bulk | 32 ulp, constant (36 ulp hyp2f1 error, F5) |

## Suspected, not confirmed

- `x**(a-1)` overflow for a < 1 at subnormal x: not reachable on the
  harness grid (x must be a normal double); at x = 1e-300 with a = 0.01 the
  power is 1e297 and finite.
- `_munp` (moments) uses the same `sc.beta` / `sc.hyp2f1` ratio and inherits
  F2, F3, F4, F5 and F7 (`sc.beta(n+a, b)/sc.beta(a, b)` underflows to 0/0 for
  large shapes); moments are outside the scope of the harness and were not
  measured.

## Opportunities: inherited methods

Demonstration points computed with `scratch/gausshyper/opps.py` (mp.quad of
the reference pdf at 40 digits) for (a, b, c, z) = (2, 3, 1, 0.5), except logpdf
(harness `--at`).

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `xlogy(a-1, x) + xlog1py(b-1, -x) - c*log1p(z*x) - betaln(a, b) - log(hyp2f1(c, a, a+b, -z))` (with the Pfaff form of F3 for large z) | wherever the pdf underflows: `logpdf(0.05, 300, 300, 5, 2)` gives -inf, reference -490.60344545833533 (loss); also every F1-F3 regime |
| cdf | quad of pdf | no scipy closed form (Appell F1: `x**a/(a B H) * F1(a; 1-b, c; a+1; x, -z x)`); for b = 1 it reduces to `x**a * hyp2f1(c, a, a+1, -z x)/(a B H)`. Quadrature is adequate in the bulk | `cdf(1e-5)` = 7.150626792885499e-10 vs 7.1506267928854971e-10 (1 ulp); fine, but slow and absolute-tolerance limited for tiny values |
| sf | 1 - cdf | reflection: 1 - X ~ gausshyper(b, a, c, -z/(1+z)) (exact identity, z' > -1 always), so `sf(x; a, b, c, z) = cdf(1 - x; b, a, c, -z/(1+z))` | right tail: `sf(1 - 1e-6)` gives 0.0, reference 3.1781077317437751e-18 (loss) |
| logcdf | log(cdf) below the median, log1p(-sf) above | as cdf, in log space for small x: `a*log(x) - log(a) - betaln(a, b) - log(H)` + log of the F1 factor (-> 0 as x -> 0) | cdf underflows: `logcdf(1e-200)` gives -inf, reference -919.06682051472084 (loss) |
| logsf | log(sf) above the median (sf = 1 - cdf) | log of the reflected cdf | right tail as sf: `logsf(1 - 1e-6)` gives -inf, reference -40.290245707125748 (loss) |
| ppf | brentq on cdf, xtol absolute | solve in log space / relative tolerance | small q: `ppf(1e-20)` = 3.7396206736713306e-11 vs 3.7395928863208347e-11 (3.3e10 ulp, relative 7e-6: brentq stops at the absolute xtol) |
| isf | ppf(1 - q) | ppf of the reflected distribution: `isf(q; a, b, c, z) = 1 - ppf(q; b, a, c, -z/(1+z))` | small q: `isf(1e-20)` gives 1.0, true 1 - x = 1.4653579784744646e-7 (loss) |

## Shared code

- **`scipy.special.hyp2f1`** (used here and in other distributions'
  normalizations and moments):
  - nearly degenerate parameters: `hyp2f1(c, a; a+b; w)` with c = a, w << -1
    loses 1.9e4 ulp at w = -1e6 and 3.9e9 ulp at w = -1e12; with c - a =
    1e-12 it loses 1.6e11 ulp at w = -1e6 (F4).
  - large first parameter: 2e6 to 5e6 ulp for c = 1e4, |w| = 1e-4 ... 1e-3
    (F5).
  - returns nan where the true value is a normal number
    (`hyp2f1(1e4, 3, 5, -1)`, `hyp2f1(1000, 2, 5, -10)`,
    `hyp2f1(1e5, 3, 5, -1e-5)`) and nan instead of inf on overflow
    (`hyp2f1(-120, 2, 5, -1000)`, true 1.46e355) (F3).
- **`scipy.special.beta`**: up to 1.1e6 ulp for (2, 1e6), 3.8e4 for (2, 1e4),
  714 for (300, 300) (F7); underflows to 0 below 2.2e-308 (expected for the
  function, but gausshyper should use `betaln`). Other users in
  `_continuous_distns.py`: burr `_munp`/moments (lines 1240, 1261, 1351),
  jf_skew_t (lines 5772-5815), t (line 8121), rdist moments (lines 9094-9095).

## Existing GitHub issues

- #26089 (closed): `scipy.special.hyp2f1` returns wrong values for large positive parameters at negative real z - relates to F5 (and possibly F3)
- #17832 (open): Tracking issue: custom methods for continuous distributions - relates to the inherited methods under Opportunities
- PR #18799 (closed): ENH: simplify `gausshyper.pdf` - origin of the current `_pdf`, context only

(GitHub REST search API; `gh` unavailable.)

## Verification

Verifier reference: `scratch/gausshyper/verify/verify.py`, a different route that uses no hyp2f1 and no beta. The normalization is `mp.quad` of the unnormalized docstring density x^(a-1) (1-x)^(b-1) (1+zx)^(-c) over [0, 1] (log/linear breakpoints, scaled by its maximum), and the pdf is evaluated in log space. Values at 100 and 200 digits agree to at least 98 digits at every point. Factor-level mechanism check in `scratch/gausshyper/verify/factors.py`: the sc.hyp2f1 error against the Euler transform matches F4 and F5, and the sc.beta error matches F7. All mechanisms match the cited lines 4916-4918.

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(0.05, 300, 300, 5, 2)` | 0.0 | 8.5828364066123716e-214 | verified (loss) | x**299 underflows |
| F1 | `pdf(0.9500000000000001, 300, 300, 5, 2)` | 0.0 | 6.7391367793232129e-216 | verified (loss) | (1-x)**299 underflows |
| F2 | `pdf(0.5, 600, 600, 1, 1)` | inf | 27.63121729789657 | verified (loss) | |
| F2 | `pdf(0.5, 1000, 1000, 0, 0.5)` | inf | 35.678022291708641 | verified (loss) | |
| F3 | `pdf(0.9, 2, 3, -120, 1000)` | nan | 0.02732079271919567 | verified (loss) | case (a) |
| F3 | `pdf(1e-100, 3, 2, 1e4, 1)` | nan | 4.9985007002401803e-189 | verified (loss) | case (b) |
| F3 | `pdf(0.0001, 2, 3, 1000, 10)` | nan | 3670.3369526886182 | verified (loss) | case (b) |
| F3 | `pdf(0.5, 3, 2, 1e5, 1e-5)` | nan | 1.6243847091650265 | verified (loss) | case (b) |
| F4 | `pdf(0.5, 2.5, 1.5, 2.500000000001, 1e6)` | 0.1186313293291851 | 0.11862703381301852 | verified (severe) | 1.63e11 ulp; the hyp2f1 error alone is 1.63e11 ulp |
| F4 | `pdf(0.5, 2.5, 1.5, 2.5, 1e10)` | 0.06692356424482472 | 0.066923563982472242 | verified (severe) | 1.77e7 ulp |
| F5 | `pdf(0.5, 3, 2, 1e4, 1e-4)` | 1.6243770886862354 | 1.6243770894238292 | verified (severe) | 2.04e6 ulp; the hyp2f1 error alone is 2.04e6 ulp |
| F6 | `pdf(1e-210, 2.5, 1.5, 2.5, 1e6)` | 8.388239932524648e-302 | 8.3882399452255533e-302 | verified (severe) | 6.82e6 ulp |
| F7 | `pdf(1e-06, 2, 1e6, 3, 0.5)` | 367880.54488575936 | 367880.5448075739 | verified (moderate) | 9.57e5 ulp; the sc.beta error alone is 1.09e6 ulp |
| F7 | `pdf(1e-300, 2, 1e4, 3, 0.5)` | 1.0003999700264762e-292 | 1.000399970017987e-292 | verified (moderate) | 3.82e4 ulp |

Verified by dist-accuracy-verifier: 7 of 7 verified.
