# studentized_range: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `studentized_range_gen`, lines 12274-12470 (`_pdf` 12414-12437, `_cdf` 12439-12466). The integrands are in `scipy/stats/_stats.pyx` (`_studentized_range_cdf`, `_studentized_range_pdf`, their `_asymptotic` variants, `_phi`, `_logphi`, `_Phi`). The `.pyx` is not in the installed wheel, so the integrand expressions quoted below come from the upstream source. The two `cpdef` log-constants are present locally and were evaluated directly.
- **Implemented:** pdf, cdf (both via `integrate.nquad` over a 2-D integral; 1-D infinite-df integrand for df >= 100000)
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), ppf = numerical inversion of cdf, isf = ppf(1 - q)
- **Related:** none; helpers: Cython `_phi`, `_logphi`, `_Phi` (= `0.5*erfc(-z/sqrt 2)`, libm `erfc`), `_studentized_range_{cdf,pdf}_logconst`; also `integrate.nquad`, `LowLevelCallable`
- **Reference:** `refs/studentized_range_ref.py`. The module has two evaluation paths:
  - **k = 2 (exact):** `cdf = I_{t^2/(df+t^2)}(1/2, df/2)` with `t = x/sqrt 2`, `pdf = sqrt 2 * f_t(x/sqrt 2; df)`. Used for the full grids at 400 digits.
  - **Other k:** nested trapezoid quadrature in z and u = log s, with a step/half-step consistency check. Used only for single `--at` points at 30 digits. It matches the exact k = 2 values to 2e-30 and SciPy's k = 3 and k = 1.5 bulk cdf to about 1e-15. `cond` is not available on this path (printed as "-", so the tool uses an 8 ulp tolerance).
  - **Self-checks:** no FAIL. The NOTE/WARN at (2, 0.1), x = 2.3e9, is a SciPy error (F2). It was confirmed independently with `scipy.stats.t`: `2*t.cdf(x/sqrt 2, 0.1) - 1 = 0.9` and `sqrt 2 * t.pdf(...) = 4.3e-12`.
- **Parameter sets:** full grids (k = 2): (2, 10), (2, 1), (2, 0.1), (2, 1000), (2, 99999), (2, 100000). Single points for general k: (1.5, 10), (1.9, 10), (1.5, 1e6), (3, 10), (100, 10); (1000, 10) attempted, reference not converged.
- **Worst finding:** loss in pdf and cdf. Also loss in every inherited method at its demonstration point.

## Summary

For k = 2 with df <= 1000, both methods are accurate in the bulk to about 10 ulp (pdf, cdf) or about 1e3-1e4 ulp (pdf at small df). Large k (k = 100) is also fine. Outside that, the two quadrature-based methods fail in several independent ways:
- pdf returns nan for every x when 1 < k < 2 (F1).
- For large x the quadrature misses the region s ~ 1/x that carries the mass. pdf drops to ~0 and cdf jumps to ~1 even where the true values are large, e.g. cdf(1e6, 2, 0.1) = 0.9999999993 instead of 0.783 (F2).
- The cdf left tail (and the pdf left tail for k > 2) loses all accuracy, through the epsabs = 1e-11 floor and the cancellation in `Phi(z + qs) - Phi(z)`. It reaches 0 for x <~ 1e-17 (F3).
- The pdf has only absolute accuracy (~1e-14) in the right tail (F4).
- df >= 100000 switches to the infinite-df integrand with a 3e-6 relative jump (F5).
- Just below the switch the narrow chi peak is missed in the tails (F6), and the bulk carries a df*eps bias from the log constants (F7).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf | 1 < k < 2, every x, every df | `pow(D, k-2)` with D == 0 gives inf, then 0*inf = nan (pattern 3) | nan for 0.296 | loss |
| F2 | pdf, cdf | large x: x >~ 2e3-5e3 for df <= 10; worst for small df | outer `qagi` over s in (0, inf) never samples s ~ 1/x, where all of the pdf mass and all of 1 - cdf lie (pattern 15) | pdf 0.0 for 2.7e-7; cdf 0.99999999928 for 0.783 | loss |
| F3 | cdf, pdf (k > 2) | small x: x <~ 1e-4 (k = 2), x <~ 1e-2 (k = 3) | epsabs = 1e-11 floor plus cancellation in D = `_Phi(z+qs) - _Phi(z)` (patterns 15, 1); D == 0 for qs < ~eps | cdf 0.0 for 5.5e-19; 5.7e10 ulp at k = 3 | loss |
| F4 | pdf | right tail (any df), plus the bulk at small df | absolute error control: errors ~1e-14 absolute (epsabs = 1e-11, epsrel = 1e-12) (pattern 15) | 0.0 for 7.9e-104; 9.1e14 ulp at 7.8e-16 | loss |
| F5 | pdf, cdf | df >= 100000, all x | switch to the infinite-df integrand, O(1/df) model error (pattern 7) | 2.0e10 ulp bulk; 5.3e15 ulp tail | loss |
| F6 | pdf, cdf | df just below 1e5 (2-D branch), tails | chi density in s has width ~1/sqrt(2 df); quadrature misses the peak once values are small (pattern 15) | pdf 5.4e-13 for 9.0e-11 | severe |
| F7 | pdf, cdf | large df < 1e5, all x | log-constant and `(df-1) log s - df s^2/2` are ~df/2-sized terms that cancel (pattern 5) | 4.4e4 ulp at df = 99999 | moderate |

### F1: pdf is nan for every x when 1 < k < 2

- **Code:** `_pdf`, line 12414 onwards, via the Cython integrand (upstream `_stats.pyx`) `exp(log_pdf_const + df*log(s) - df*s*s/2 + _logphi(z) + _logphi(s*q + z)) * pow(_Phi(s*q + z) - _Phi(z), k - 2)`. The infinite-df variant has the same `pow(..., k - 2)` factor.
- **Mechanism:** for k < 2 the exponent k - 2 is negative. D = `_Phi(z + qs) - _Phi(z)` is exactly 0 wherever both `_Phi` values round to the same double: z >~ 8.3, where both are 1, and z <~ -38, where both underflow. QUADPACK's infinite-range transform samples such z. There `pow(0, negative) = inf`, and with `exp(...)` underflowed to 0 the product is nan, which poisons the whole integral (pattern 3). The cdf integrand uses `pow(D, k - 1)` with a positive exponent and is fine (cdf(1, 1.5, 10) = 0.72330942954920760, reference 0.72330942954920668).
- **Regime:** every x and every df (both the 2-D and the infinite-df branch) for 1 < k < 2. Checked: k in {1.01, 1.5, 1.9, 1.999}, x in {0.1, 1, 5}, df in {3, 1e6}: all nan. The domain explicitly allows these k (`_argcheck: k > 1`).
- **Reproducer:** `python accuracy.py refs/studentized_range_ref.py --dps 30 --at pdf 1.0 1.5 10`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1.0, 1.5, 10)` | nan | 0.29594882740355294 | inf | - | loss |
  | `pdf(3.0, 1.9, 10)` | nan | 0.06347525568314379 | inf | - | loss |
  | `pdf(1.0, 1.5, 1e6)` | nan | 0.3034741138986432 | inf | - | loss |

- **Fix direction:** evaluate the integrand as `exp(log_terms + (k-2)*log(D))` with D computed without subtraction (from the tail on the far side of the centre, e.g. `ndtr(-z) - ndtr(-z-qs)` for z > -qs/2), so that D = 0 only where the whole integrand underflows anyway.

### F2: pdf collapses to 0 and cdf to 1 at large x

- **Code:** `_pdf`/`_cdf`, `ranges = [(-np.inf, np.inf), (0, np.inf)]` (lines 12424, 12451) with `integrate.nquad`. The outer range is over s (upstream integrand order).
- **Mechanism:** with S = chi_df/sqrt(df), the cdf is E[R_k(qS)] and the pdf is E[S R_k'(qS)]. For large q, the pdf integrand (and 1 - cdf) is nonzero only for s <~ few/q, so it is confined to s ~ 1/x near the lower end of (0, inf). `qagi` maps (0, inf) to (0, 1] by s = (1-t)/t, so its first Kronrod pass places its smallest s node at ~4e-3. The adaptive error estimate sees nothing there and never subdivides towards s -> 0. The pdf then returns the tiny contribution it does see (1e-17 ... 1e-78, or 0), and the cdf returns 1 - 1e-10 instead of 1 - P(S <~ 1/x). For small df the missed mass is large, because P(S < 1/x) ~ x^-df. This explains the bulk NOTE at df = 0.1, where x = 2.3e9 is the 0.9 quantile. Pattern 15 (quadrature misses a feature); this confirms the earlier tiny-df suspicion S8.
- **Regime:** pdf from x ~ 2e3 (df = 1: 2.7e6 ulp at 2e3) and fully wrong from 5e3. cdf for df = 1 from x ~ 1e4, and for df = 0.1 from x ~ 1e6, where the true cdf is still 0.78. For df = 10 the true pdf there is already below 1e-33, so F4 masks it.
- **Reproducer:** `python accuracy.py refs/studentized_range_ref.py --at pdf 1e5 2 0.1`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1e5, 2, 0.1)` | 0.0 | 2.73283528597987e-07 | inf | 1.1 | loss |
  | `pdf(1e4, 2, 1)` | 3.377186561941024e-57 | 9.0031629815078e-09 | 4.5e15 | 2 | loss |
  | `pdf(5000, 2, 0.1)` | 6.725032544568351e-17 | 7.374735521411716e-06 | 4.5e15 | 1.1 | severe |
  | `pdf(2000, 2, 1)` | 2.250789663664412e-07 | 2.250789664997933e-07 | 2.67e6 | 2 | severe |
  | `pdf(2323456607.000256, 2, 0.1)` (bulk, q = 0.9) | 0.0 | 4.303932326461949e-12 | inf | - | loss |
  | `cdf(1e6, 2, 0.1)` | 0.9999999992757298 | 0.7829231771471516 | 1.25e15 | 0.028 | severe |
  | `cdf(1e5, 2, 1)` | 1.0 | 0.999990996836839 | 4.05e10 | 9.0e-5 | severe |
  | `cdf(1e10, 2, 0.1)` | 0.9999999998968147 | 0.9135801602613093 | 4.26e14 | 0.0095 | severe |

- **Fix direction:** integrate over u = log s (or split the s range at ~1/x and pass `points`), and compute the upper tail as its own integral of `1 - D^(k-1)` rather than as the remainder of the cdf.

### F3: cdf (and pdf for k > 2) lose all accuracy for small x

- **Code:** `opts = dict(epsabs=1e-11, epsrel=1e-12)` (lines 12433, 12460), and in the integrands `pow(_Phi(z + q*s) - _Phi(z), k - 1)` (cdf) and `pow(..., k - 2)` (pdf), with `_Phi(z) = 0.5*erfc(-z/sqrt 2)`.
- **Mechanism:** two effects add up.
  - **Tolerance floor:** QUADPACK stops at `max(epsabs, epsrel*|I|)`. With cdf ~ x^(k-1), the result has no guaranteed relative accuracy once it is below ~1e-11, and the inner integrals (one per outer s node) each get the same absolute allowance.
  - **Cancellation in D:** D = `_Phi(z+qs) - _Phi(z)` cancels for small qs, with a relative error of about eps*Phi(z)/(qs*phi(z)) (pattern 1). It becomes exactly 0 once qs < ~eps*|z|, so the cdf integrand vanishes for x <~ 1e-17.

  The two effects were separated in the infinite-df 1-D integrand at k = 3 (`scratch/studentized_range/isolate_s3.py`). With SciPy's float64 D but a tight tolerance (epsabs = 0, epsrel = 2e-14), the relative error is 1.5e-11 at w = 1e-6 and 2e-9 at w = 1e-8 (about 1e-17/w); this is the cancellation alone. With SciPy's tolerances, the cdf error at w = 1e-6 is 2.3e-8 (tolerance floor). In the 2-D finite-df case, the tolerance dominates for k = 3. For k = 2 below x ~ 1e-8, the cancellation dominates.
- **Regime:** cdf for x <~ 1e-4 at k = 2 (moderate), x <~ 1e-8 (severe), x <= 1e-17 to 1e-18 (0, loss), at every df. cdf at k = 3 is severe already for x = 1e-4 (value 2.8e-9). pdf at k = 3: 1.2e5 ulp at x = 1e-6, 3.1e8 ulp at 1e-9. The pdf at k = 2 is unaffected, because `pow(D, 0) = 1`.
- **Reproducer:** `python accuracy.py refs/studentized_range_ref.py --at cdf 1e-8 2 10`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-18, 2, 10)` | 0.0 | 5.502823538378389e-19 | inf | 1 | loss |
  | `cdf(1e-16, 2, 1000)` | 4.075931537321811e-17 | 5.640485538048267e-17 | 1.25e15 | 1 | severe |
  | `cdf(1e-14, 2, 10)` | 5.478784657013699e-15 | 5.502823538378389e-15 | 1.97e13 | 1 | severe |
  | `cdf(1e-8, 2, 10)` | 5.5028235130033e-09 | 5.502823538378388e-09 | 2.08e7 | 1 | severe |
  | `cdf(1e-6, 2, 10)` | 5.50282353834932e-07 | 5.502823538377884e-07 | 2.34e4 | 1 | moderate |
  | `cdf(1e-6, 2, 0.1)` | 2.094338320256284e-07 | 2.094338898302982e-07 | 1.24e9 | 1 | severe |
  | `cdf(1e-4, 3, 10)` (`--dps 30`) | 2.7566444648004866e-09 | 2.756644472514553e-09 | 1.3e7 | - | severe |
  | `cdf(1e-8, 3, 10)` (`--dps 30`) | 2.7566793994803397e-17 | 2.7566444771089602e-17 | 5.7e10 | - | severe |
  | `pdf(1e-6, 3, 10)` (`--dps 30`) | 5.51328895407424e-07 | 5.513288954216082e-07 | 1.2e5 | - | moderate |
  | `pdf(1e-9, 3, 10)` (`--dps 30`) | 5.513288569888619e-10 | 5.51328895421792e-10 | 3.1e8 | - | severe |

- **Fix direction:** `epsabs=0`; compute D as `0.5*(erfc(a) - erfc(b))` on the far side of the centre and, for small qs, as `qs*phi(z + qs/2)*(1 + O((qs)^2))` (or an `erf`-difference with expm1-style evaluation); add the small-x asymptotic `cdf ~ c_k(df) x^(k-1)` if needed.

### F4: pdf has only absolute accuracy (~1e-14); right tail lost

- **Code:** `opts = dict(epsabs=1e-11, epsrel=1e-12)`, line 12433.
- **Mechanism:** QUADPACK's error target is `max(1e-11, 1e-12*|I|)`. In practice the absolute errors are about 1e-15 to 1e-14 (pdf(4, 2, 10): 8e-15; pdf(10, 2, 10): 9.4e-15), so the relative error grows as 1/pdf in the right tail (pdf ~ x^(-df-1)). Once the pdf is far below the floor, the result is 0 or unrelated to the true value (pattern 15). In the bulk the relative error sits at the requested epsrel = 1e-12 (4.5e3 ulp) or slightly above. Bulk examples: 1.16e4 ulp at pdf(2, 2, 0.1) and 2.5e3 ulp at pdf(3, 2, 1). The cdf right tail is accurate (about 9 ulp at k = 2): the cdf is near 1 there, so the same absolute error is small in relative terms. The sf is not implemented (see Opportunities).
- **Regime:** pdf at df = 10: moderate from x ~ 4, severe from x = 10, 0 from x ~ 2e4. At df = 1000: severe from x ~ 8 (2e7 ulp). The pdf at small df in the bulk: up to ~1e4 ulp.
- **Reproducer:** `python accuracy.py refs/studentized_range_ref.py --at pdf 100 2 10`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1e10, 2, 10)` | 0.0 | 7.875e-104 | inf | 11 | loss |
  | `pdf(100, 2, 10)` | 6.223183851362113e-16 | 7.78893525910727e-16 | 9.05e14 | 11 | severe |
  | `pdf(30, 2, 10)` | 3.939248089680154e-10 | 3.939285995137895e-10 | 4.33e10 | 10.8 | severe |
  | `pdf(10, 2, 10)` | 2.889041027514437e-05 | 2.889041026578296e-05 | 1.46e6 | 9.2 | severe |
  | `pdf(10, 2, 1000)` | 1.407305148930835e-11 | 1.399820358807682e-11 | 2.41e13 | 47.7 | severe |
  | `pdf(4, 2, 10)` | 0.02170635034038713 | 0.02170635034039526 | 1.69e3 | 4.9 | moderate |
  | `pdf(2, 2, 0.1)` | 0.03924879672785628 | 0.03924879672795746 | 1.16e4 | 1.05 | moderate |

- **Fix direction:** `epsabs=0` (relative control), plus the s-range handling of F2 so that the tail mass is found; for k = 2 the exact t form is available.

### F5: df >= 100000 switches to the infinite-df integrand

- **Code:** `_pdf` line 12419 / `_cdf` line 12446: `if df < 100000: ... else: cython_symbol = '_studentized_range_{pdf,cdf}_asymptotic'`.
- **Mechanism:** at df >= 1e5 the integration over s is dropped (S = 1 exactly). The model error of that approximation is O(1/df) in the bulk and grows like x^4/df in the pdf tail. That is ~1e-6 relative at df = 1e5, about 1e6 times the requested tolerance (pattern 7). cdf(1, 2, df) jumps from 0.5204982300875673 (df = 99999) to 0.5204998778130465 (df = 100000), while the true values differ by 1.6e-11.
- **Regime:** every x for df >= 1e5. The relative error is 2e-6 to 5e-6 in the bulk, ~1e-2 in the pdf tail at x ~ 10, and a factor of 2 at x = 30. It shrinks like 1/df but stays above 1e-12 until df ~ 1e11.
- **Reproducer:** `python accuracy.py refs/studentized_range_ref.py --at cdf 1 2 100000`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(30, 2, 100000)` | 3.914053259901054e-98 | 1.792234338495166e-98 | 5.3e15 | 448 | loss |
  | `pdf(1, 2, 100000)` | 0.4393912894677224 | 0.4393893671368669 | 1.97e10 | 0.5 | severe |
  | `pdf(10, 2, 100000)` | 7.83543330278589e-12 | 7.882551555608993e-12 | 2.69e13 | 50 | severe |
  | `cdf(1, 2, 100000)` | 0.5204998778130465 | 0.5204982300988578 | 1.43e10 | 0.84 | severe |
  | `cdf(2, 2, 100000)` | 0.8427007929497148 | 0.8426976796508351 | 1.66e10 | 0.49 | severe |

- **Fix direction:** keep the 2-D integral (with the s-substitution of F6) for all finite df, or add the 1/df (and 1/df^2) correction terms to the asymptotic integrand; use the infinite form only for df = inf.

### F6: narrow chi peak missed just below df = 1e5

- **Code:** `ranges = [(-np.inf, np.inf), (0, np.inf)]` (lines 12424, 12451), outer s over (0, inf) by `qagi`.
- **Mechanism:** for large df the s-weight `s^(df-1) exp(-df s^2/2)` is a spike of width ~1/sqrt(2 df) (0.0022 at df = 99999) at s = 1, on an infinite interval mapped to (0, 1]. When the total value is small (pdf tail, cdf left tail), the error test `<= 1e-11` passes before the spike is resolved, and the result is wrong by ~100% (pattern 15). At df = 1000 (width 0.022) the same points are only moderately wrong (2.4e13 ulp at pdf(10, 2, 1000), mostly F4).
- **Regime:** 2-D branch with large df (seen at df = 99999) whenever the value is <~ 1e-10: pdf x >~ 9, cdf x <~ 1e-10.
- **Reproducer:** `python accuracy.py refs/studentized_range_ref.py --at pdf 9.5 2 99999`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(9.5, 2, 99999)` | 5.378034985538992e-13 | 9.010875057797595e-11 | 4.48e15 | 45.1 | severe |
  | `pdf(10, 2, 99999)` | 4.828030559017686e-14 | 7.882552028051351e-12 | 4.48e15 | 50 | severe |
  | `cdf(1e-10, 2, 99999)` | 3.116934162438113e-13 | 5.641881730614557e-11 | 4.48e15 | 1 | severe |
  | `cdf(1e-14, 2, 99999)` | 3.120758491978146e-17 | 5.641881730614557e-15 | 4.48e15 | 1 | severe |

- **Fix direction:** substitute s = 1 + u/sqrt(2 df) (or u = log s) so that the peak has unit width, or pass the peak location as `points` on a finite s interval.

### F7: df*eps bias from the log-constant at large df

- **Code:** `_studentized_range_{cdf,pdf}_logconst` (`(df/2)*log(df) - lgamma(df/2) - (df/2 - 1)*log 2` plus `log k` terms), and in the integrand `(df-1)*log(s) - df*s*s/2`.
- **Mechanism:** the constant and the integrand terms are each ~df/2 in size and cancel to O(log df) (pattern 5). Each carries an absolute rounding error of ~df*eps/2 that becomes a relative error after `exp`. At df = 99999 the double constant `_studentized_range_cdf_logconst(3, 99999) = 50005.78270341153` differs from the exact 50005.782703411520 by 9.8e-12 (doubles near 5e4 are spaced 7.3e-12). This matches the observed uniform bias of 4.4e4 ulp (9.8e-12) across the whole bulk. At df = 1000 the bias is 722 ulp (1.6e-13); it shows as `cdf(1e8, 2, 1000) = 0.9999999999998385` instead of 1.
- **Regime:** all x, growing ~linearly with df up to the 1e5 switch: ~0.7 ulp per unit df (722 ulp at df = 1000, 4.4e4 ulp at df = 99999).
- **Reproducer:** `python accuracy.py refs/studentized_range_ref.py --at cdf 1 2 99999`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(1, 2, 99999)` | 0.5204982300875673 | 0.5204982300823805 | 4.49e4 | 0.84 | moderate |
  | `pdf(1, 2, 99999)` | 0.4393893671219536 | 0.4393893671176434 | 4.42e4 | 0.5 | moderate |
  | `pdf(1, 2, 1000)` | 0.4391991161360436 | 0.439199116136114 | 722 | 0.5 | minor |
  | `cdf(1e8, 2, 1000)` | 0.9999999999998385 | 1.0 | 728 | 0 | minor |

- **Fix direction:** substitute s = 1 + u/sqrt(2 df) (or s = exp(u)) and combine the constant with the integrand analytically. The constant then becomes a Stirling-difference term (`log(df/2) * df/2 - lgamma(df/2) - df/2` via a `gammaln` correction helper) and the integrand becomes `df*(log1p(v) - v - v^2/2 ...)`, with no df-sized cancellation.

## RuntimeWarnings

The Cython integrands set floating-point flags, and `np.frompyfunc` reports them as numpy `RuntimeWarning`s. The light review predicted no numpy warnings; that prediction was wrong.
- **k in (1, 2), pdf:** "divide by zero" and "invalid value encountered in _single_pdf" with default numpy settings. These flag the real nan of F1.
- **Large x, pdf:** "overflow encountered in _single_pdf" with default settings, e.g. pdf(1e300, 2, 10) and pdf(1e200, 2, 1). The result is 0 there; it is wrong only where the true value is normal (F2, F4).
- **Every call, with `np.errstate(all='warn')`:** "underflow encountered in _single_cdf/_single_pdf", including accurate bulk results (e.g. pdf(1, 2, 10), 10.7 ulp). These are spurious: the exponentials of the Gaussian factors underflow in the tails of the z integration.

nquad also emits `IntegrationWarning` ("roundoff error", "maximum number of subdivisions") for the F1 calls.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | k = 2, df in {10, 1, 0.1}, x from 1e-300 to ~1 | 14.6 ulp (cond <= 5.5) |
| cdf | k = 2, df in {10, 1}, x from 0.01 to the right tail (cdf near 1) | 10.2 ulp (cond <= 1) |
| cdf | k = 2, df = 1000, bulk and right tail | 727 ulp (F7 bias only) |
| cdf, pdf | k = 100, df = 10, x = 5 (`--dps 30`) | cdf 20.6 ulp, pdf 113 ulp (minor; cond not available) |
| cdf | k = 1.5, df = 10, x = 1 (cdf integrand has a positive exponent) | agrees with the reference to 1.3e-15 |
| cdf, pdf | k = 3, df = 10, x = 1 | 1.2e-15, 1.6e-15 relative (validation run) |

The minor 9-15 ulp in the k = 2 bulk is at the level of the requested `epsrel` and has no separate mechanism. The log-space evaluation of `s**(df-1) * exp(-df s^2/2) * phi(z)` causes no overflow or underflow for any df tested (0.1 to 99999). Its precision cost at large df is F7.

## Suspected, not confirmed

- **S7 (pattern 6), large k:** `pow(D, k-1)` multiplies D's relative error by about k. At k = 100 the measured errors are 20.6 ulp (cdf) and 113 ulp (pdf) at x = 5, df = 10, so nothing beyond minor shows. At k = 1000 (x = 7, df = 10) the reference was unavailable: the outer u step does not resolve the step-like R_1000(qs), so the step/half-step check rejected the value. SciPy returns cdf 0.5716708517365563, pdf 0.22596539819688524, both unverified. Large k (>~ 1000) is therefore not measured.
- **S9, cdf clipping (line 12466):** `np.clip(..., 0, 1)` maps overshoots to 1. cdf(1e5, 2, 1) returns exactly 1.0. It cannot be told from the output whether that was an overshoot clipped to 1 or an integral that rounded to 1; the error itself is F2. This is not an accuracy problem in itself.
- **Moments (`_munp`, line 12407):** same `nquad` tolerances, so the F3/F4/F6 mechanisms likely apply. Not measured: `_munp` is not one of the reviewed methods.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | 1 - cdf | Same double integral with `1 - D^(k-1)` computed as `-expm1((k-1)*log1p(-(1-D)))`, where `1 - D = _Phi(z) + ndtr(-(z+qs))` (sum of two tails, no cancellation); `epsabs=0`, s handled as in F2. Exact for k = 2: `betainc(df/2, 1/2, df/(df+x^2/2))`. | right tail. `sf(30, 2, 10)` = 1.2058192222497155e-09 vs 1.2058211594276538e-09 (7.2e9 ulp, severe). `sf(1000, 2, 10)` = 0.0 vs 7.874278165216991e-25 (loss). |
| logsf | log(sf) | log of the sf integral above, or log-space integration | same as sf: `logsf(1000, 2, 10)` = -inf vs -55.50102580590037 (loss) |
| logcdf | log(cdf) | log-space integration of `(k-1)*log(D) + log_terms` (logsumexp-style), `epsabs=0`, with D from F3's fix | left tail: `logcdf(1e-20, 2, 10)` = -inf vs -46.649025621751896 (loss) |
| logpdf | log(pdf) | log-space integration of the pdf integrand | right tail and F1-F2: `logpdf(1e5, 2, 10)` = -inf vs -110.76297638299654 (loss) |
| ppf | numerical inversion of cdf | No closed form. A dedicated solver on log(cdf) with an accurate lower tail; it is very slow now (one 2-D `nquad` per iteration). | small q: `ppf(1e-12, 2, 10)` = 1.8172364301147584e-12 vs 1.817248895999829e-12 (3.1e10 ulp, severe), inherited from F3 |
| isf | ppf(1 - q) | Invert an accurate sf (row 1) | small q: `isf(1e-12, 2, 10)` = 61.44520617839392 vs 61.457343742292394 (8.9e11 ulp, severe); all q < 1.1e-16 give inf |

## Shared code

None confirmed beyond this class. The `nquad` tolerance choice (`epsabs=1e-11, epsrel=1e-12`) and the s-range handling are local to `studentized_range` (`_pdf`, `_cdf`, `_munp`). They are the same pattern as INDEX item 15 (`geninvgauss`, `norminvgauss`). The Cython `_Phi` uses libm `erfc` directly, not `scipy.special.ndtr`. The cancellation in F3 comes from how the difference is formed, not from an inaccurate special function.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(1.0, 1.5, 10)` | nan | 0.2959488274035536 | verified (loss) | The 2-D mp.quad was too slow, so this reference is in double precision: a log-space integrand with D taken from log_ndtr on the far tail, and an adaptive quad and a trapezoid rule that agree to 1.6e-13. Matches the analyst to 2e-15. When the Cython integrand is called directly at k = 1.5, it gives inf at z = 9 and at s = 1e-20, and nan at z = -40. So the D == 0 mechanism holds, but the nodes give inf as well as nan (inf where exp has not underflowed, and also at small s from cancellation). |
| F1 | `pdf(3.0, 1.9, 10)` | nan | 0.0634752556831439 | verified (loss) | Same method; agrees with the analyst to 2e-15. |
| F1 | `pdf(1.0, 1.5, 1e6)` | nan | 0.303474114 (double; ~1e-9 accuracy) | verified (loss) | At df = 1e6 the double log-constant limits my reference to ~1e-9 (my code is off by 1.2e-9 for k = 2 there). The analyst's 0.3034741138986432 is consistent with this. |
| F2 | `pdf(1e5, 2, 0.1)` | 0.0 | 2.7328352859798695e-07 | verified (loss) | k = 2 references: route A integrates erf / the normal density over u = log s with mp.quad; route B is the Student-t form via hyp2f1 for the cdf and the closed form for the pdf. Routes A and B agree; A at 100 and 200 digits agrees to < 1e-93 unless noted. Outer s range confirmed: nquad's ranges[1] = (0, inf), lines 12424/12451. |
| F2 | `pdf(1e4, 2, 1)` | 3.3771865619410243e-57 | 9.0031629815078011e-09 | verified (loss) | 4.5e15 ulp |
| F2 | `cdf(1e6, 2, 0.1)` | 0.9999999992757298 | 0.78292317714715163 | verified (severe) | 1.25e15 ulp |
| F2 | `cdf(1e5, 2, 1)` | 1.0 | 0.99999099683683903 | verified (severe) | 4.05e10 ulp |
| F3 | `cdf(1e-18, 2, 10)` | 0.0 | 5.5028235383783891e-19 | verified (loss) | Route B (hyp2f1) has no cancellation at small x. |
| F3 | `cdf(1e-8, 2, 10)` | 5.5028235130033e-09 | 5.5028235383783888e-09 | verified (severe) | 2.08e7 ulp |
| F3 | `cdf(1e-6, 2, 10)` | 5.50282353834932e-07 | 5.502823538377884e-07 | verified (moderate) | 2.34e4 ulp |
| F3 | `cdf(1e-4, 3, 10)` | 2.7566444648004866e-09 | 2.7566444725145531e-09 | verified (severe) | 1.26e7 ulp. k = 3 reference uses Owen's T: R_3(w) = 1 - 12 T(w/sqrt2, 1/sqrt3). Averaging over S then gives 1-D integrals for the pdf and cdf; the cdf is taken as int_0^x pdf. Cross-checked with the closed form r_3(w) = (3/sqrt pi) e^{-w^2/4} erf(w/(2 sqrt3)) to 6e-60. |
| F3 | `cdf(1e-8, 3, 10)` | 2.7566793994803397e-17 | 2.7566444771089603e-17 | verified (severe) | 5.71e10 ulp |
| F3 | `pdf(1e-9, 3, 10)` | 5.513288569888619e-10 | 5.5132889542179208e-10 | verified (severe) | 3.14e8 ulp; `pdf(1e-6, 3, 10)` 1.16e5 ulp also reproduced |
| F4 | `pdf(1e10, 2, 10)` | 0.0 | 7.875e-104 | verified (loss) | At 100 digits route A was not converged (1e-5); at 200 digits it agrees with the closed form to 6e-104. |
| F4 | `pdf(100, 2, 10)` | 6.223183851362113e-16 | 7.7889352591072694e-16 | verified (severe) | 9.05e14 ulp |
| F4 | `pdf(10, 2, 10)` | 2.8890410275144366e-05 | 2.8890410265782959e-05 | verified (severe) | 1.46e6 ulp |
| F5 | `pdf(30, 2, 100000)` | 3.9140532599010535e-98 | 1.7922343384951665e-98 | verified (loss) | 5.33e15 ulp. Route A at 100 digits agrees only to 4e-14; at 200 digits it agrees with the closed form to 3e-128. |
| F5 | `pdf(1, 2, 100000)` | 0.4393912894677224 | 0.43938936713686688 | verified (severe) | 1.97e10 ulp |
| F5 | `cdf(1, 2, 100000)` | 0.5204998778130465 | 0.52049823009885773 | verified (severe) | 1.43e10 ulp. SciPy's value equals erf(1/2) = R_2(1) at S = 1, which confirms the asymptotic switch (lines 12419/12446). |
| F6 | `pdf(9.5, 2, 99999)` | 5.378034985538992e-13 | 9.010875057797595e-11 | verified (severe) | 4.48e15 ulp |
| F6 | `cdf(1e-10, 2, 99999)` | 3.1169341624381133e-13 | 5.6418817306145571e-11 | verified (severe) | 4.48e15 ulp |
| F7 | `cdf(1, 2, 99999)` | 0.5204982300875673 | 0.52049823008238046 | verified (moderate) | 4.49e4 ulp. The mechanism is consistent, with one correction: `_studentized_range_cdf_logconst(3, 99999)` is too large by 1.24e-11 against the exact 50005.78270341152019, not 9.8e-12. The sign and size match the +1.0e-11 relative bias. |
| F7 | `pdf(1, 2, 99999)` | 0.4393893671219536 | 0.43938936711764344 | verified (moderate) | 4.42e4 ulp |

Verified by dist-accuracy-verifier: 7 of 7 verified.
