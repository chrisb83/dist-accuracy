# nct: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `nct_gen`, lines 8137-8205
- **Implemented:** pdf (8183-8184, `scu._nct_pdf`), cdf (8186-8187, `sc.nctdtr`), ppf (8189-8190, `sc.nctdtrit`), sf (8192-8194, `np.clip(scu._nct_sf, 0, 1)`), isf (8196-8198, `scu._nct_isf`). All five are thin wrappers around Boost `non_central_t_distribution`; the only SciPy-side arithmetic is the `np.clip` in `_sf`.
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf)
- **Related:** none in scipy.stats. `sc.nctdtr` and `sc.nctdtrit` are public `scipy.special` functions, so the cdf and ppf findings apply to them directly. Helpers: none. Special functions: `sc.nctdtr`, `sc.nctdtrit`, `scu._nct_pdf`, `scu._nct_sf`, `scu._nct_isf` (moments `scu._nct_mean/_variance/_skewness/_kurtosis_excess` not reviewed)
- **Reference:** `refs/nct_ref.py`, 100 digits. cdf/sf use the Guenther/Lenth Poisson-weighted incomplete-beta series with the reflection X(k, c) = -X(k, -c), so every tail value comes from a direct sum with no `1 - F`. Extra digits cover the p/q cancellation for c < 0. The pdf uses the 1F1 closed form; for df > 1e5 it uses the exact identity f = (k/x)[F_{k+2}(x sqrt(1+2/k)) - F_k(x)].
  - Self-checks: PASS in the main x-method run (12 parameter sets).
  - The large-df pdf rerun FAILs only where SciPy itself is wrong. "pdf(1e12, 2) disagrees with scipy at every bulk point" is finding F5: at x = 2 the reference gives 0.398942280400934, i.e. 1/sqrt(2 pi) to 12 digits as expected for df -> inf, while SciPy gives 0.39895993.
  - The rerun also reports precision instability of the log references at x = ±1e290 for df >= 1e6. Those values are about exp(-6.6e8), outside the double range for pdf/cdf/sf, and the log methods are inherited and not measured.
- **Parameter sets:** (14, 0.24045031331198066), (1, 1), (5, -3), (0.5, 5), (30, 20), (10, 30), (1e6, 2), (10, 1e-8), (300, 2), (1e12, 2), (10, 8), (10, -8). Quantiles: (14, 0.24045031331198066), (1, 1), (0.5, 5), (1e6, 2), (10, 1e-8), (5, -3).
- **Worst finding:** loss in cdf, sf, pdf, ppf, isf

## Summary

For moderate df and small |nc|, every method is accurate in the bulk and in most of the tails: <= 70 ulp for cdf/sf at (14, 0.24); <= 5 ulp at nc = 1e-8. The failures come from Boost `non_central_t`:

- **cdf left tail (x < 0, nc > 0) and sf right tail (x > 0, nc < 0) (F1).** Boost reflects to the other sign of nc and sums a series whose terms alternate in sign, or forms `1 - result`. The absolute error is about eps, whatever the size of the result. So the relative error grows like 1/cdf:
  - 3e7 ulp at (0.5, 5);
  - values 1e51 to 1e113 times too large at (30, 20);
  - nan at (10, 8), (30, 20), (10, 30), (300, 2) and (1e6, 2);
  - negative raw sf values, which SciPy's `np.clip` turns into 0.
- **Overflow of t*t.** For |x| > 1.34e154, cdf/sf return 0 or 1 (F2) and the pdf raises `OverflowError` or returns nan (F3). For small df the true tail values there are normal doubles.
- **df == 1 exactly.** cdf and sf return nan for 1e-160 <= |x| <= 1e-110 (F4).
- **pdf, large df.**
  - The pdf is wrong in the bulk by about 0.2*df ulp: 4.5% at df = 1e15 (F5).
  - It returns nan or raises `OverflowError` in the left tail for df >= 200 (F6).
- **pdf, other regimes.**
  - It returns pdf(0) for |x| < 1.5e-8, measured up to 1.4e7 ulp (F7).
  - It loses up to 1e7 ulp in the sign-alternating regime of F1 (F8).
  - It returns 0 in the far right tail for large nc (F9).
- **cdf/sf at df = 1e6.** cdf/sf lose up to 1.9e4 ulp in the bulk (F10).
- **ppf/isf.**
  - They saturate at ±1.34e154 for small df (F11).
  - They inherit F1/F10 errors: ppf(1e-300, 1e6, 2) = -8.76 instead of -35.06. isf(1e-300, 1e6, 2) is wrong because sf is flushed to 0 near 1e-299 (F12).
  - ppf returns nan at (10, 1e-8), q = 1e-300 (F13).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf (and ppf/isf through them) | cdf: x < 0 with nc > 0; sf: x > 0 with nc < 0; grows with \|nc\|, df and \|x\| | Boost reflects (t, delta) -> (-t, -delta) and sums a sign-alternating Poisson series, then `1 - result` in the p-branch: absolute error ~eps for a result << 1 (patterns 1, 5) | cdf 8.2e-24 for 5.4e-121; nan; sf 0.0 for 2.1e-22 | loss |
| F2 | cdf, sf | \|x\| > 1.34e154, small df (df = 0.5, 1) where the tail is a normal double | `t*t` overflows inside Boost (pattern 3), as t.md F1 | 0.0 for 6.6e-302 | loss |
| F3 | pdf | \|x\| >= ~1.3e154 (all df tested); for df < 1 the true pdf is normal there | Boost `ibeta_derivative` overflow from `t*t` (pattern 3); raises `OverflowError`, aborting the whole array call | OverflowError for 5.9e-232 | loss |
| F4 | cdf, sf | df == 1 exactly, any nc, 1e-160 <= \|x\| <= 1e-110 | Boost non-central beta series "did not converge" for a = b = 1/2 and tiny y = t^2/(1+t^2); mechanism unclear | nan for 0.1587 | loss |
| F5 | pdf | large df, bulk: ~0.2*df ulp up to df ~ 4e15 | inside Boost `non_central_t_pdf`; error proportional to df suggests rounding amplified by b = df/2 (pattern 6, inferred) | 0.00423 for 0.00443 (df = 1e15) | severe |
| F6 | pdf | df >= ~200, left tail (x <= -5 at nc = 2) | Boost pdf quadrature path: nan ("exp_sinh ... singular point") for 200 <= df < ~350, `OverflowError` from `tgamma` for df >= ~350 (pattern 3) | nan for 2.3e-11 | loss |
| F7 | pdf | 0 < \|x\| < ~1.5e-8 (sqrt(eps)), nc != 0 | Boost returns the t = 0 value for small \|t\| (zeroth-order shortcut, pattern 7) | 1.1e7 ulp at x = 1e-8 | severe |
| F8 | pdf | x < 0 with nc > 0 (x > 0 with nc < 0), moderate \|x\| | same sign-alternating series as F1 (pattern 5) | 9.5e6 ulp | severe |
| F9 | pdf | large nc (30, 20), x >~ 3e9 | far right tail flushed to 0: intermediate underflow in the Poisson-weighted series (pattern 4, inferred; ncx2 F1 analogue) | 0.0 for 4.8e-248 | loss |
| F10 | sf, cdf | df = 1e6, bulk | mechanism unclear (large b = df/2 in the Boost series) | 1.9e4 ulp | moderate |
| F11 | ppf, isf | small df, true quantile beyond 1.34e154 (q <~ 1e-154 at df = 1) | root finder on a cdf that saturates beyond sqrt(DBL_MAX) (inherits F2), as t.md F4 | -1.34e154 for -6.6e298 | loss |
| F12 | ppf, isf (and sf far tail) | ppf small q with nc > 0; isf small q with nc < 0 or q -> 1; sf/isf far right tail at df = 1e6; bulk at df = 1e6 | inherits F1, F10 and an sf flush to 0 near 1e-299 at df = 1e6 (pattern 4, inferred) | ppf -8.76 for -35.06; sf 0.0 for 8.0e-299 | loss |
| F13 | ppf | q = 1e-300, (10, 1e-8) | Boost quantile iteration; mechanism unclear | nan for -2.56e30 | loss |

### F1: cdf left tail for nc > 0 (sf right tail for nc < 0) is garbage

- **Code:** `_cdf`, line 8187: `sc.nctdtr(df, nc, x)`. `_sf`, line 8194: `np.clip(scu._nct_sf(x, df, nc), 0, 1)`. Both are Boost `non_central_t_cdf`.
- **Mechanism:** inferred from the outputs together with the structure of Boost's `non_central_t_cdf` (the Boost source is not shipped with the binary install).
  - For t < 0, Boost sets t -> -t, delta -> -delta and toggles `invert`. It then evaluates P = Phi(-delta) + 1/2 sum [p_j I_x(j+1/2, k/2) + q_j I_x(j+1, k/2)] or the complementary Q-series, depending on a crossover x < cross.
  - With delta < 0 the q_j terms are negative, so the series alternates in sign. Its terms are O(Phi(|delta|)) ~ 1, while the requested tail is tiny.
  - In the P-branch the result is also formed as `1 - result` (pattern 1).
  - Either way the absolute error is a few eps, independent of the true value. The relative error is therefore about eps/cdf (patterns 1, 5).
  - Once the terms cancel completely, Boost returns nan, values of the wrong size, or negative values. In `_sf`, SciPy's `np.clip(..., 0, 1)` turns a negative raw sf into exactly 0. For example, `scu._nct_sf(3, 10, -8)` = -1.16e-20, which `nct.sf` reports as 0.0.
  - Errors appear on both sides of Boost's crossover. With (10, 5) and cross = 0.712: x = -3 (x_beta = 0.47) gives 7.8e10 ulp, and x = -10 (x_beta = 0.91) gives 4.8e14 ulp. So the series itself fails, not only the final subtraction.
- **Regime:** cdf for x < 0 and nc > 0, and by symmetry sf for x > 0 and nc < 0.
  - Loss in ulp is roughly 1/cdf times the size of the cancelling terms.
  - At (14, 0.24) only <= 70 ulp (minor), at (0.5, 5) 3e7 ulp throughout the left tail, at (10, 3) 6e8 ulp at x = -100.
  - Total loss (nan, wrong magnitude, 0) for nc >~ 6 at df = 10, and already for nc = 2 at df >= 300.
  - Large df also hits it near the bulk: (1e6, 2) at x = -3 (cdf 2.9e-7) has 6.9e9 ulp.
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at cdf -10 30 20` (`scratch/nct/run_harness.py` runs accuracy.py unchanged but patches `nct_gen` at runtime so that a Boost exception on one element returns nan for that element instead of aborting the array call; without it the harness stops at the first OverflowError, see F3)

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-3, 30, 20)` | 2.4504221499602274e-70 | 2.101564366544592e-106 | 5.3e51 | 25.2 | loss |
  | `cdf(-10, 30, 20)` | 8.175079901756256e-24 | 5.372466788681364e-121 | 6.9e112 | 29.4 | loss |
  | `cdf(-20, 30, 20)` | nan | 6.240680565301596e-130 | inf | 29.9 | loss |
  | `cdf(-10, 1e6, 2)` | 7.487052905137681e-31 | 1.7829863478601284e-33 | 1.9e18 | 121 | loss |
  | `cdf(-9, 1e6, 2)` | nan | 1.915432598954983e-28 | inf | 99.8 | loss |
  | `cdf(-4, 1e6, 2)` | 9.86834430011953e-10 | 9.867395519980425e-10 | 4.3e11 | 24.6 | severe |
  | `cdf(-3, 10, 5)` | 2.4581686655110952e-12 | 2.4581259660912683e-12 | 7.8e10 | 8.13 | severe |
  | `cdf(-5, 0.5, 5)` | 3.84184335914149e-08 | 3.8418433837097875e-08 | 2.9e7 | 0.5 | severe |
  | `sf(3, 10, -8)` | 0.0 | 2.0891237822317642e-22 | inf | 8.87 | loss |
  | `sf(100, 10, -3)` | 5.261522796057957e-22 | 5.261523544397649e-22 | 6.4e8 | 10 | severe |
  | `sf(7, 5, -3)` | 2.8193128333256213e-08 | 2.8193128313498637e-08 | 3.2e6 | 4.86 | severe |

  Counts in the x-run: cdf low tail 78 loss, 47 severe, 6 moderate; sf high tail 44 loss, 9 severe, 12 moderate (most in F1; the rest in F2 and F4).
- **Fix direction:** for the reflected tail compute the small quantity directly. Use the Q-series S+(|t|; k, delta) = 1/2 sum [p_j I_y'(k/2, j+1/2) + q_j I_y'(k/2, j+1)] (y' = k/(k+t^2)), written for delta < 0 as a positive-term series. Alternatively integrate Phi over the chi distribution in log space. Never return `1 - result` for a small tail. In `_sf`, do not hide negative values with `np.clip`.

### F2: cdf and sf return 0/1 for |x| > 1.34e154

- **Code:** `_cdf` line 8187 and `_sf` line 8194 (Boost `non_central_t_cdf`).
- **Mechanism:** Boost forms x = t*t/(v + t*t). For |t| > 1.34e154, `t*t` = inf, so x is 1 and xc = 0, and the tail is returned as exactly 0 (pattern 3). This is the same failure as t.md F1 (`stdtr`). In the non-central code it applies for every nc.
- **Regime:** |x| > 1.34e154 where the tail C*|x|^-df is still a normal double, i.e. df <~ 2. At df = 0.5 the tail is about 1e-77 there.
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at cdf -1e300 1 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-1e300, 1, 1)` | 0.0 | 6.647612765794012e-302 | inf | 1 | loss |
  | `sf(1e200, 1, 1)` | 0.0 | 8.643606884608055e-201 | inf | 1 | loss |
  | `cdf(-1.4e154, 0.5, 5)` | 0.0 | 7.262182859450632e-85 | inf | 0.5 | loss |
  | `sf(1.4e154, 0.5, 5)` | 0.0 | 1.4666124279063214e-77 | inf | 0.5 | loss |

- **Fix direction:** compute x and xc as `1/(1 + v/t^2)` and `(v/t^2)/(1 + v/t^2)` (or with t scaled by sqrt(v)) so that xc stays representable. Use the tail asymptote C*|t|^-v for huge |t|.

### F3: pdf raises OverflowError (or returns nan) for |x| >~ 1.3e154

- **Code:** `_pdf`, line 8184: `scu._nct_pdf(x, df, nc)`.
- **Mechanism:**
  - Boost evaluates the non-central beta density at x = t^2/(v+t^2), xc = v/(v+t^2).
  - Once `t*t` overflows (or xc is at the bottom of the double range), `ibeta_derivative` hits (1-x)^(b-1) with xc = 0 and b = v/2 < 1. It then reports "Overflow Error". SciPy translates that into a Python `OverflowError` (pattern 3).
  - For df = 0.5 the error appears from |x| = 1.3e154 up. There the true pdf is normal (5.9e-232 ... 4.3e-308).
  - For df = 1 it starts at 1.4e154, where the true value is subnormal.
  - For df = 5, 10, 30 it appears at exactly |x| = 1.3e154 (the true value underflows).
  - In all of these cases the exception aborts the whole array call. For example, `stats.nct.pdf([0.0, 1.3e154], 10, 8)` raises instead of returning [value, 0].
- **Regime:** |x| >= ~1.3e154. Loss of a normal value for df < 1. For larger df it is a robustness problem: an exception where 0 is correct.
- **Reproducer:** direct call `stats.nct.pdf(1.3e154, 0.5, 5)` raises `OverflowError: Error in function ibeta_derivative<double>(%1%,%1%,%1%): Overflow Error`. Under the launcher (exception -> nan):

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1.3e154, 0.5, 5)` | OverflowError (nan) | 5.853752476881288e-232 | inf | 1.5 | loss |
  | `pdf(-1e200, 0.5, 5)` | OverflowError (nan) | 4.296365319596844e-308 | inf | 1.5 | loss |

  Logged exceptions in the x-run: 34 points (0.5, 5), 32 (1, 1), 2 each for (5, -3), (30, 20), (10, ±8), (10, 30) (`scratch/nct/exceptions.txt`).
- **Fix direction:** compute the density from t directly for huge |t|: f ~ v*C*|t|^-(v+1), or evaluate in log space with log xc = log v - 2 log|t|. Return 0 instead of raising when the result underflows.

### F4: cdf and sf return nan for df == 1 and tiny |x|

- **Code:** `_cdf` line 8187, `_sf` line 8194.
- **Mechanism:** unclear. Boost emits "Error in function cdf(non_central_beta_distribution<double>, %1%): Series did not converge" (seen in the sf run). With df = 1 the non-central beta has a = b = 1/2. For y = t^2/(1+t^2) between ~1e-320 and 1e-220, the series used for the P/Q split apparently fails to converge and returns nan. The failure is specific to df == 1: df = 1 + 1e-15, 0.999, 0.5, 2 and 3 are all fine at the same x. This parallels t.md F6 (a df == 1 special path in Boost `students_t`).
- **Regime:** df == 1 exactly, any nc (checked 1, 0.1, -1), 1e-160 <= |x| <= 1e-110. |x| = 1e-170 and 1e-100 are fine.
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at cdf 1e-130 1 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-130, 1, 1)` | nan | 0.15865525393145705 | inf | 0 | loss |
  | `sf(-1e-130, 1, 1)` | nan | 0.8413447460685429 | inf | 0 | loss |

- **Fix direction:** for |t| below ~1e-100 return Phi(-delta) + t*pdf(0) (cdf) and its complement (sf). The first-order term is below eps there.

### F5: pdf loses ~0.2*df ulp in the bulk for large df

- **Code:** `_pdf`, line 8184.
- **Mechanism:** inside Boost `non_central_t_pdf`. The error scales almost exactly like df: 60 ulp at df = 300, 1e3 at 1e4, 2e5 at 1e6, 5e6 at 1e8, 2e11 at 1e12, 2e14 at 1e15. That suggests a rounding error in a quantity like xc = v/(v+t^2), raised to the power b = v/2 (pattern 6, inferred). At df >~ 4e15 Boost switches to the Student-t shortcut (|delta/(4v)| < eps) and the error drops to <= 40 ulp (measured with a direct script: `scratch/nct/bigdf_ref.py`).
- **Regime:** the whole support, df >~ 1e4 (moderate) and df >~ 1e7 (severe), up to ~4e15.
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at pdf 2 1e12 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(5, 1e15, 2)` | 0.004234632584836879 | 0.004431848411938211 | 2.0e14 | 15 | severe |
  | `pdf(2, 1e12, 2)` | 0.3989599313703442 | 0.398942280400934 | 2.0e11 | 3e-12 | severe |
  | `pdf(1, 1e6, 2)` | 0.241970724529031 | 0.2419707245190778 | 1.9e5 | 1 | moderate |

  The pdf rerun counts: (1e12, 2) 16 severe in [-4, 10]; (1e6, 2) 17 moderate in the bulk. The reference at df = 1e12, x = 2 equals the normal limit 1/sqrt(2 pi) = 0.3989422804014327 to 12 digits, so this is not a reference problem.
- **Fix direction:** for large df evaluate in log space, using `log1p(t^2/v)` for log xc. Alternatively switch to the normal/Edgeworth limit much earlier, e.g. when the O(1/df) correction is below eps.

### F6: pdf returns nan or raises for large df in the left tail

- **Code:** `_pdf`, line 8184.
- **Mechanism:** in this regime Boost emits "boost::math::quadrature::exp_sinh<double>::integrate: The exp_sinh quadrature evaluated your function at a singular point", which shows that the pdf is computed by numerical quadrature. For 200 <= df < ~350 the integrand produces a non-finite value and the pdf is nan. For df >= ~350, `tgamma` of an argument of order df/2 overflows and SciPy raises `OverflowError: Error in function boost::math::tgamma<double>(%1%): Result of tgamma is too large to represent.` (pattern 3). Mapped with nc = 2: x <= -5 fails for every df from 200 to 1e15, while x = -4 still returns a value (6e6 ulp at df = 300). At df = 1e16 the Student-t shortcut takes over.
- **Regime:** df >~ 200, left tail for nc > 0 (x <= -5 at nc = 2). The right tail also raises at df = 1e6 for x >= 1e3, where the true value underflows.
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at pdf -5 300 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(-5, 300, 2)` | nan | 2.277290956000706e-11 | inf | 31.9 | loss |
  | `pdf(-6, 1e6, 2)` | OverflowError (nan) | 5.055074534637103e-15 | inf | 48 | loss |
  | `pdf(-4, 300, 2)` | 9.323974210072416e-09 | 9.323974222700357e-09 | 6.1e6 | 22.6 | severe |

  Logged exceptions: 146 points at (1e6, 2) and 142 at (1e12, 2) (`tgamma`), most of them where the true value underflows. x in [-10, -5] are normal values: loss.
- **Fix direction:** use `lgamma`/`tgamma_delta_ratio` in the integrand, or the series form for this regime.

### F7: pdf returns pdf(0) for |x| < sqrt(eps)

- **Code:** `_pdf`, line 8184.
- **Mechanism:** for small |t| Boost returns the closed-form value at t = 0. The measured switch is between 1e-8 and 1e-7, consistent with sqrt(eps) = 1.49e-8. Values at ±1e-12, ±1e-10 and ±1e-8 are bit-identical to `pdf(0)`. The true pdf has slope f'(0) proportional to nc, so the relative error is |f'(0)/f(0)|*|x|. By extrapolation that is ~1.6e7 ulp just below the threshold at (14, 0.24) (pattern 7: the shortcut is zeroth order). For nc = 0 the slope is zero, and (10, 1e-8) shows <= 2.8 ulp.
- **Regime:** 0 < |x| < 1.5e-8, nc != 0. The error is |f'(0)/f(0)|*|x|/eps ulp, with f'(0)/f(0) = 0.245 at (14, 0.24): 1.1e3 ulp at x = 1e-12, 1.1e5 at 1e-10 and 1.1e7 at 1e-8.
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at pdf 1e-8 14 0.24045031331198066`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-8, 14, 0.24045031331198066)` | 0.3807208842611732 | 0.3807208851930977 | 1.1e7 | 2.45e-9 | severe |
  | `pdf(1e-10, 30, 20)` | 5.475140065402087e-88 | 5.475140076443983e-88 | 9.1e6 | 2.02e-9 | severe |

  Also in the x-run: (10, 30) 1.4e7, (10, ±8) 3.7e6, (5, -3) 1.4e6, (1, 1) 5.6e5, (14, 0.24) 1.1e5 ulp at x = ±1e-10.
- **Fix direction:** add the first-order term f(0)*(1 + x*f'(0)/f(0)), with f'(0)/f(0) = sqrt(2) nc Gamma((k+2)/2)/(sqrt(k) Gamma((k+1)/2)). Alternatively lower the threshold to ~eps; the regular path is accurate at 1e-7 (3 ulp).

### F8: pdf loses accuracy in the sign-alternating regime

- **Code:** `_pdf`, line 8184.
- **Mechanism:** Boost's pdf uses the same reflection (t < 0 -> delta -> -delta) and a Poisson-weighted series whose q-terms are negative for the reflected delta (pattern 5). The cancellation is milder than in F1 because the density is not formed as `1 - result`.
- **Regime:** x < 0 with nc > 0, and x > 0 with nc < 0. Severe at (10, ±8) near |x| = 1 and at (5, -3) far right (x = 1e50).
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at pdf -1 10 8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(-1, 10, 8)` | 4.07145699218661e-18 | 4.071456983629179e-18 | 9.5e6 | 5.99 | severe |
  | `pdf(1e50, 5, -3)` | 2.5442855808253672e-303 | 2.5442855790124722e-303 | 3.2e6 | 6 | severe |

- **Fix direction:** as F1. Sum the reflected series in a positive-term form, or evaluate the density from the 1F1 representation in log space.

### F9: pdf returns 0 in the far right tail for large nc

- **Code:** `_pdf`, line 8184.
- **Mechanism:** inferred. For (30, 20) the pdf is accurate up to x = 1e9 (2.985e-233) and exactly 0 from about 3e9 on, where the true value is 4.8e-248. The Poisson weights exp(-nc^2/2) ~ 1e-87 multiplied by the beta-density term (~xc^(b-1), xc = v/(v+t^2)) underflow before the final scaling (pattern 4). This is the analogue of ncx2 F1 / ncf, where Boost starts the series at the Poisson mode and flushes an underflowing start term to 0.
- **Regime:** large nc and large x, roughly where exp(-nc^2/2)*xc^(b-1) < 1e-308.
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at pdf 3e9 30 20`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(3e9, 30, 20)` | 0.0 | 4.833061941834159e-248 | inf | 31 | loss |
  | `pdf(1e10, 30, 20)` | 0.0 | 2.985253784063395e-264 | inf | 31 | loss |

- **Fix direction:** evaluate the weighted terms in log space before exponentiating, or use the tail asymptote f ~ v C |t|^-(v+1).

### F10: cdf and sf lose ~1e4 ulp in the bulk at df = 1e6

- **Code:** `_cdf` line 8187, `_sf` line 8194.
- **Mechanism:** unclear. At df = 1e6, nc = 2 the sf is off by 1e3-2e4 ulp over the whole bulk and the right tail (x = 3 ... 7), and the cdf by 2.4e3 ulp at x = 3. At df = 1e12 the same points are fine (<= 36 ulp), and at df = 300 sf is <= 58 ulp. This points to the Boost non-central beta series with b = df/2 = 5e5 rather than to a simple df*eps law.
- **Regime:** df ~ 1e6 (not mapped finer), bulk.
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at sf 4 1e6 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(4, 1e6, 2)` | 0.022750617867801966 | 0.022750617867896894 | 1.88e4 | 9.49 | moderate |
  | `cdf(3, 1e6, 2)` | 0.841344020157246 | 0.8413440201576851 | 2.35e3 | 0.863 | moderate |

- **Fix direction:** to be determined in the fixing step (compare with a positive-term series at large b).

### F11: ppf and isf saturate at ±1.3407807929942603e154 for small df

- **Code:** `_ppf` line 8190 (`sc.nctdtrit`), `_isf` line 8198 (`scu._nct_isf`). Both are the Boost `non_central_t` quantile.
- **Mechanism:** the root finder inverts the Boost cdf/sf. Beyond |x| = sqrt(DBL_MAX) = 1.34e154 that cdf/sf is exactly 0 or 1 (F2: `t*t` overflows). So the search stops at the last x where `t*t` is finite and returns ±1.3407807929942603e154 for every smaller q. This is the same failure as t.md F4. Just above the saturation the result is already off by 1e8 ulp (the F1 error of the cdf, at cond 2).
- **Regime:** small df (true quantile beyond 1.34e154). Saturation for q <~ 1e-154 at df = 1, and for q <~ 1e-89 at (0.5, 5).
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at ppf 1e-300 1 1`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 1, 1)` | -1.3407807929942603e154 | -6.647612765794013e298 | 4.5e15 | 1 | loss |
  | `isf(1e-300, 1, 1)` | 1.3407807929942603e154 | 8.643606884608054e299 | 4.5e15 | 1 | loss |

  Run counts: ppf low tail 17 loss and isf low tail 17 loss, all at (1, 1) and (0.5, 5).
- **Fix direction:** fix F2 first, so that the cdf has a usable tail beyond 1.34e154. Then invert the tail asymptote q = C |x|^-df analytically for tiny q.

### F12: ppf and isf inherit the cdf/sf failures of F1, F9 and F10

- **Code:** `_ppf` line 8190, `_isf` line 8198.
- **Mechanism:** the Boost quantile inverts the cdf (ppf) or the complement (isf). Where those carry the F1 error (left tail for nc > 0, right tail for nc < 0), the F9-type zero (sf far right tail at large df) or the F10 bulk error (df = 1e6), the quantile carries it too, divided by its condition number. For df = 1e6, nc = 2 and small q, the cdf is wrong by orders of magnitude or nan (F1). The root finder then returns essentially arbitrary values between -7.6 and -37 instead of the smooth true quantile from -5.5 down to -35.
  - isf(1e-300, 1e6, 2) = 38.94 instead of 39.06, because sf is 0 already from x ~ 38.9 although the true sf there is 8e-299. sf(38, 1e6, 2) still returns a value but is off by 2.6e6 ulp: the sf part of F9 (subnormal/underflowing intermediate, pattern 4, inferred).
  - isf with q -> 1 at df = 1e6 is the left tail through the complement (F1). It is beyond conditioning even though cond is large there (4.2e9 ulp at cond 7.5e7).
- **Regime:** ppf for q below cdf(0) = Phi(-nc) with nc > 0. isf for small q with nc < 0, and for q -> 1 with nc > 0. Bulk ppf/isf at df ~ 1e6 (moderate, F10).
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at ppf 1e-300 1e6 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 1e6, 2)` | -8.761810522124264 | -35.058484473649436 | 3.4e15 | 0.00077 | severe |
  | `ppf(3.481e-14, 1e6, 2)` | -5.4886136305315425 | -5.48864294645784 | 2.4e10 | 0.0239 | severe |
  | `isf(1e-300, 1e6, 2)` | 38.94287354630109 | 39.06123139155966 | 1.4e13 | 0.000691 | severe |
  | `isf(0.9999999994820525, 1e6, 2)` | -4.103801921317024 | -4.103805730707064 | 4.2e9 | 7.5e7 | severe |
  | `ppf(3.481e-14, 0.5, 5)` | -6093324983307.059 | -6093325092628.5 | 8.1e7 | 2 | severe |
  | `isf(1.212e-26, 5, -3)` | 33467.14352571071 | 33467.14352194843 | 5.1e5 | 0.2 | moderate |
  | `ppf(0.78, 1e6, 2)` | 2.772195390817912 | 2.77219539082521 | 1.19e4 | 0.95 | moderate |
  | `sf(38, 1e6, 2)` | 6.679566177308012e-284 | 6.679566181144653e-284 | 2.6e6 | 1.37e3 | severe |
  | `sf(38.94287354630109, 1e6, 2)` | 0.0 | 7.953452836473855e-299 | inf | 1.44e3 | loss |

  Run counts: ppf low tail 32 severe; isf high tail 7 severe; isf low tail 24 moderate (all of (5, -3)) and 3 severe; ppf/isf bulk 5 moderate each (df = 1e6).
- **Fix direction:** fix F1, F9 and F10 in the cdf/sf. Then let ppf invert the cdf only where cdf <= 1/2 and isf invert the sf only where sf <= 1/2, using the directly computed small tail.

### F13: ppf returns nan for q = 1e-300 at tiny nc

- **Code:** `_ppf`, line 8190 (`sc.nctdtrit`).
- **Mechanism:** unclear. The cdf at the true quantile -2.56e30 is accurate: (10, 1e-8) has cdf <= 4.7 ulp on the whole grid. ppf(q, 10, 1e-8) is correct for q = 1e-200, 1e-250 and 1e-280, but nan for q = 1e-300. At nc = 0 and nc = 1e-6 the same q works. So the Boost quantile iteration fails for this combination of tiny q and tiny nonzero nc. No warning is emitted.
- **Regime:** observed at (10, 1e-8), q = 1e-300; not mapped further.
- **Reproducer:** `python scratch/nct/run_harness.py refs/nct_ref.py --at ppf 1e-300 10 1e-8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 10, 1e-8)` | nan | -2.5645257106334987e30 | inf | 0.1 | loss |

- **Fix direction:** for |nc| below ~1e-6, start the iteration from the central-t quantile `stdtrit(df, q)`, or fall back to it with a first-order nc correction.

## RuntimeWarnings

- pdf, df >= 200 (3 parameter sets): "boost::math::quadrature::exp_sinh<double>::integrate: The exp_sinh quadrature evaluated your function at a singular point ...". It coincides with nan results (F6), so it is not spurious.
- sf, df == 1, tiny |x|: "cdf(non_central_beta_distribution<double>, %1%): Series did not converge". It coincides with nan (F4).
- isf (3 parameter sets, the F11/F12 regimes): "quantile(const complement(non_central_t_distribution<double>, %1%)): Unable to locate solution in a reasonable time: either there is no answer to quantile or the answer is infinite. Current best guess is %1%", together with the non-central beta "Series did not converge" message. These accompany the saturated or wrong results; the returned values are inaccurate.
- `_sf` and `_isf` suppress overflow warnings (`np.errstate(over='ignore')`, gh-17432). Boost exceptions (`OverflowError` from `ibeta_derivative` and `tgamma`) are raised rather than warned (F3, F6).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | (14, 0.24): full grid | 70 ulp at x = -100 (cond 14; minor, the small-nc end of F1) |
| cdf, sf, pdf | (10, 1e-8): full grid (continuity to the central t) | cdf 4.7, sf 4.6, pdf 2.8 ulp |
| cdf | (5, -3), right side and left tail (nc < 0, no reflection problem) | 3.6 ulp |
| sf | (14, 0.24), (10, 8), (300, 2), (1e12, 2), (30, 20), (10, 30) outside F1/F2 | <= 250 ulp (minor; (30, 20) at x = 1e10 cond 30, (10, 30) at 1e30: 217 ulp) |
| pdf | (30, 20), (10, 30) left tail, x in [-1000, -0.5] | <= 420 ulp (minor, cond ~11-31) |
| pdf | (1, 1) tails, \|x\| up to 1e150 | 24 ulp |
| ppf, isf | (14, 0.24): full q grid | ppf 4.4 ulp, isf 5.8 ulp |
| ppf, isf | (10, 1e-8) except F13; (5, -3) ppf; (1, 1) and (0.5, 5) for q above the F11 saturation | within conditioning (e.g. isf (10, 1e-8) 7.5e6 ulp at cond 1.3e8; ppf (5, -3) 4.5e5 at cond 6e14) |
| pdf | gh-19348 (pdf = inf for large t and nc): (10, 40), (30, 20), (3, 38), (100, 30), (5, 50), (1000, 30) at x up to 1e150 | no inf observed (only the zeros and exceptions of F3, F6, F9) |

## Suspected, not confirmed

- **Large-df Student-t shortcut, left tail.** For |nc/(4 df)| < eps (df >~ 4e15 at nc = 2), Boost returns `students_t_cdf(t - delta)` and applies `invert ? 1 - result : result` after the reflection. That would give an absolute-eps error for small left-tail cdf values. A direct script (`scratch/nct/bigdf_ref.py`) shows `cdf(-4, 4e15, 2)` with 2.5e8 ulp, consistent with this, but df = 4e15 was not in the harness PARAMS.
- **Inherited moments** (`_nct_mean` etc.): not reviewed.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | log of the 1F1 representation, `xlogy`-based prefactor (k/2) log k + lgamma(k+1) - nc^2/2 - k log 2 - (k/2) log(k+x^2) - lgamma(k/2), plus the log of the bracket; tail asymptote log f ~ log(k C) - (k+1) log\|x\| | pdf underflows: logpdf(1e30, 14, 0.24) = -inf, true -1016.4058143026502 |
| logcdf | log(cdf) | log of the direct left-tail series (positive-term Q-series of F1), tail asymptote log C - k log\|x\| with C = k^(k/2) E[(Z+nc)_-^k] / (2^(k/2) Gamma(k/2+1)) (E[...] via a parabolic cylinder function D_{-k-1}(nc)) | cdf underflows or is wrong (F1): logcdf(-1e30, 14, 0.24) = -inf, true -951.79938018146884 |
| logsf | log(sf) | as logcdf with nc -> -nc | sf underflows: logsf(1e30, 14, 0.24) = -inf, true -949.9673188424441 |

## Shared code

- `scipy.special.nctdtr` (public) is the cdf. F1, F2 and F4 are bugs of that public function, e.g. `nctdtr(30, 20, -10)` = 8.2e-24 (true 5.4e-121) and `nctdtr(10, 8, -3)` = nan.
- `scipy.special.nctdtrit` (public) is the ppf; see the quantile findings.
- `scu._nct_pdf`, `_nct_sf` and `_nct_isf` are private and used only by nct.
- Boost `non_central_t` depends on Boost `non_central_beta`, which is also behind `ncf` (reports/ncf.md). F9 is the nct analogue of the Poisson-mode underflow in ncx2 F1 and ncf. F2 matches t.md F1 (`stdtr`), and F4 parallels t.md F6 (df == 1 path).
- None of these change SciPy code outside nct; the fixes belong in Boost or in an nct-specific wrapper.

## Existing GitHub issues

- #20693 (closed, fixed for 1.15 via PR #20785 / Boost update): BUG: stats.noncentral_t: incorrect pdf values - far-left-tail pdf garbage at (df=8, nc=16); relates to the pdf left-tail findings (F6), which persist at df >= 200
- #19348: `stats.nct.pdf` returns inf for large t and nc (regression in 1.10) - did not reproduce in 1.18.1 (see the report)
- PR #21728 (merged): moved `nctdtr` to Boost; its thread reports `_nct_cdf` nan / failure at high noncentrality (nc = 38, cdf ~ 1e-310) - relates to F1
No issue found for the other findings (web search; `gh` CLI unavailable).

## Verification

Independent reference: `scratch/nct/verify/vref.py`. It uses the mixture definition T = (Z + nc)/S with S = sqrt(V/df): cdf = E[Phi(t S - nc)], sf = E[Phi(nc - t S)], pdf = E[S phi(t S - nc)]. These are integrated with `mp.quad` in u = log S, between peak-located bounds. No Poisson series, incomplete beta or 1F1 is used. Every value was computed at 100 and 200 digits, and the two agree to at least 91 digits. Quantiles were checked by evaluating the verifier cdf/sf and pdf at the analyst's x_ref and taking one log-space Newton step (`scratch/nct/verify/quant.py`). Outputs are in `scratch/nct/verify/out1.txt` to `out4.txt`.

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(-10, 30, 20)`; `cdf(-3, 30, 20)`; `cdf(-9, 1e6, 2)`; `cdf(-4, 1e6, 2)`; `sf(3, 10, -8)`; `sf(100, 10, -3)`; `cdf(-5, 0.5, 5)` | 8.175e-24; 2.450e-70; nan; 9.86834430011953e-10; 0.0; 5.261522796057957e-22; 3.84184335914149e-08 | 5.372466788681364e-121; 2.1015643665445919e-106; 1.915432598954983e-28; 9.8673955199804242e-10; 2.0891237822317641e-22; 5.2615235443976489e-22; 3.8418433837097877e-8 | verified | Same ulp values as the report: 6.9e112, 5.3e51, 4.3e11, 6.4e8, 2.9e7. The raw `scu._nct_sf(3, 10, -8)` = -1.16e-20 is confirmed; the clip at line 8194 turns it into 0. The Boost-internal mechanism (reflection, alternating series) is inferred because the Boost source is not shipped. It is consistent with the outputs. |
| F2 | `cdf(-1e300, 1, 1)`; `cdf(-1.4e154, 0.5, 5)`; `sf(1.4e154, 0.5, 5)` | 0.0; 0.0; 0.0 | 6.6476127657940129e-302; 7.2621828594506314e-85; 1.4666124279063213e-77 | verified | `sc.nctdtr` gives the same 0.0. |
| F3 | `pdf(1.3e154, 0.5, 5)` | OverflowError (`ibeta_derivative`) | 5.8537524768812879e-232 | verified | The exception text matches the stated mechanism. `pdf(-1e200, 0.5, 5)` also raises. |
| F4 | `cdf(1e-130, 1, 1)` | nan | 0.15865525393145705 | verified | `sf(-1e-130, 1, 1)` is also nan, with the "Series did not converge" warning. Mechanism unclear, as the report says. |
| F5 | `pdf(2, 1e12, 2)`; `pdf(5, 1e15, 2)` | 0.3989599313703442; 0.004234632584836879 | 0.398942280400934; 0.004431848411938211 | verified | 2.0e11 and 2.0e14 ulp, severe. The mechanism (b = df/2 amplification) is inferred and could not be checked against the Boost source. |
| F6 | `pdf(-5, 300, 2)`; `pdf(-6, 1e6, 2)`; `pdf(-4, 300, 2)` | nan; OverflowError (`tgamma`); 9.323974210072416e-09 | 2.2772909560007059e-11; 5.0550745346371027e-15; 9.3239742227003576e-9 | verified | The exp_sinh warning and the `tgamma` exception are observed, as described. -4 gives 6.1e6 ulp. |
| F7 | `pdf(1e-8, 14, 0.24045031331198066)` | 0.3807208842611732 | 0.38072088519309769 | verified | 1.1e7 ulp. The values at 1e-12, 1e-10 and 1e-8 are bit-identical to `pdf(0)`, while 1e-7 differs. This is consistent with the zeroth-order shortcut. |
| F8 | `pdf(-1, 10, 8)` | 4.07145699218661e-18 | 4.0714569836291795e-18 | verified | 9.5e6 ulp. The mechanism is inferred (Boost internal). |
| F9 | `pdf(3e9, 30, 20)` | 0.0 | 4.833061941834159e-248 | verified | The mechanism (underflow of a Poisson-weighted term) is inferred and was not checkable. |
| F10 | `sf(4, 1e6, 2)`; `cdf(3, 1e6, 2)` | 0.022750617867801966; 0.841344020157246 | 0.022750617867896895; 0.8413440201576851 | verified | 1.88e4 and 2.35e3 ulp, moderate. Mechanism unclear, as the report says. |
| F11 | `ppf(1e-300, 1, 1)` | -1.3407807929942603e154 | -6.6476127657940129e298 (verifier cdf(x_ref)/q = 1.0) | verified | `isf(1e-300, 1, 1)` = +1.34e154 is reproduced as well. |
| F12 | `ppf(1e-300, 1e6, 2)`; `isf(1e-300, 1e6, 2)`; `sf(38.94287354630109, 1e6, 2)` | -8.761810522124264; 38.94287354630109; 0.0 | -35.058484473649435; 39.061231391559657; 7.9534528364728753e-299 | verified | 3.4e15 and 1.4e13 ulp; cdf(x_ref)/q = 1 - 3.9e-14 and sf(x_ref)/q = 1 - 1.1e-13. The relative errors are 0.75 (ppf) and 0.003 (isf), both below 1, so "severe" is correct for these rows. The sf row is a loss. |
| F13 | `ppf(1e-300, 10, 1e-8)` | nan | -2.5645257106334988e30 (verifier cdf(x_ref)/q = 1 + 2e-16) | verified | Mechanism unclear, as the report says. |

Verified by dist-accuracy-verifier: 13 of 13 verified.
