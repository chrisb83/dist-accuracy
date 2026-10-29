# gengamma: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `gengamma_gen`, lines 3852-3951
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** gamma (c = 1), weibull_min (a = 1), halfgennorm (a = 1/beta, c = beta; has its own implementation, so it does not share this code); norm (only in `_entropy`); helpers: none; special functions: sc.gammainc, sc.gammaincc, sc.gammaincinv, sc.gammainccinv, sc.gammaln, sc.xlogy (sc.poch, sc.psi in moments/entropy, not reviewed)
- **Reference:** `refs/gengamma_ref.py`, 400 digits; self-checks: PASS, except two "integral of pdf != cdf difference" FAILs for (0.001, 10). These come from the harness itself: it integrates x**(-0.99) over 70 decades in linear x. Integrating the same reference pdf in log x reproduces both cdf differences to 40 digits (0.39999999957169453342..., 0.39999999912946255080...). The NOTE/WARN lines for (0.001, 10) are finding F1. `--plot` crashed matplotlib (Windows fatal exception 0xc06d007f in `bezier.py`), so no plot was produced. The tables come from the text/JSON output.
- **Parameter sets:** (4.4162385429431925, 3.119309167924276), (4.4162385429431925, -3.119309167924276), (0.001, 10), (0.001, -10), (1/3, 3), (1000, 2), (2, 0.001), (0.5, -0.5)
- **Worst finding:** loss in cdf, sf, ppf, isf

## Summary

For moderate shapes (both signs of c) all six methods are accurate to a few hundred ulp or better. Two mechanisms lose every digit when a is small (a < ~1) and |c| is large or moderate.
(1) cdf/sf form `x**c` explicitly. Once it underflows (c > 0, small x; c < 0, large x), `gammainc(a, 0)` returns 0 (and `gammaincc` returns 1), although the true value `y**a/Gamma(a+1)` is a normal double. Examples: `cdf(1e-40, 0.001, 10)` = 0 instead of 0.398, and `cdf(1e-200, 1/3, 3)` = 0.
(2) ppf/isf compute `gammaincinv(a, p)**(1/c)`. The inner value underflows to 0 for small a and p, which gives 0 for c > 0 and inf for c < 0. Examples: `ppf(0.1, 0.001, 10)` = 0 instead of 9.4e-101, and `ppf(1e-150, 1/3, 3)` = 0.
Smaller problems, all moderate:
- logpdf cancels terms of size ~a*log(a) for large a (2.4e3 ulp at a = 1e3), and pdf = exp(logpdf) multiplies that (7.2e3 ulp).
- `c*a - 1` rounds when c*a is close to 1 (logpdf 1.5e3 ulp at a = 1/3, c = 3).
- for small |c|, the power `**(1/c)` multiplies the last-bit error of `gammaincinv` by 1/c (2.0e3 ulp at c = 0.001).
- the `sc.gammainc`/`sc.gammaincc` prefactor error documented in reports/gamma.md (F6).

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf, sf | c > 0: c*log(x) < -708 (x**c subnormal or 0) while a*c*log(x) > -708; c < 0: mirror image at large x. Needs a <~ 1 | `x**c` underflows before `gammainc`/`gammaincc` (patterns 3/4, 10) | cdf 0.0 for 0.398; sf 1.0 for 0.602 (3.0e15 ulp) | loss |
| F2 | ppf, isf | small a, small p = q (ppf, c > 0) or 1 - q (ppf, c < 0), where (p*Gamma(a+1))**(1/a) < ~1e-308 but the result is a normal double | `gammaincinv`/`gammainccinv` underflows to 0, then `0**(1/c)` (patterns 3, 8) | 0.0 for 9.4e-101; inf for 1.06e100 | loss |
| F3 | logpdf, pdf | large a (1e3), x**c within a few sqrt(a) of a | `xlogy(c*a-1, x) - x**c - gammaln(a)` cancel (pattern 5); pdf exponentiates it | logpdf 2.42e3 ulp, pdf 7.24e3 ulp | moderate |
| F4 | logpdf | c*a close to 1 (e.g. a = 1/3, c = 3), x << 1 | `c*a - 1` rounds and the error is multiplied by \|log x\| (pattern 5) | 1.52e3 ulp at x = 1e-300 | moderate |
| F5 | ppf, isf | small \|c\| (0.001), bulk q | the 0.5-2 ulp error of the double `gammaincinv` result is raised to the power 1/c (pattern 6) | 2.04e3 ulp | moderate |
| F6 | cdf, sf | large a (1e3), x**c far from a (y = 500, 2000) | `sc.gammainc`/`sc.gammaincc` prefactor error (reports/gamma.md F6; pattern 10) | 2.42e3 ulp | moderate |
| F7 | pdf | \|logpdf\| large (far tails), any shapes | pdf = exp(logpdf) turns the absolute rounding error of logpdf into a relative error of about \|logpdf\| ulp (pattern 3/5) | 807 ulp | minor |

### F1: cdf and sf lose everything once x**c underflows

- **Code:** `_cdf`, lines 3908-3911, and `_sf`, lines 3918-3921: `xc = x**c; val1 = sc.gammainc(a, xc); val2 = sc.gammaincc(a, xc)`.
- **Mechanism:** for y = x**c -> 0 the lower function is P(a, y) ~ y**a / Gamma(a+1). If a < 1 this is much larger than y, so it can stay a normal double long after y itself has underflowed (pattern 3/4). Once `x**c` is 0, `gammainc` returns 0 and `gammaincc` returns 1. While `x**c` is subnormal (c*log(x) between -745 and -708) it keeps only a few significant bits, and P inherits a*(relative error of y) (pattern 4). For c < 0 the same thing happens at large x, where y = x**c -> 0 and the roles of cdf and sf swap.
- **Regime:** c > 0: c*log(x) < -708 (degraded) or < -745 (zero), together with a*c*log(x) - gammaln(a+1) > -708 (result still normal). That is only possible for a <~ 0.95. c < 0: the same with x -> 1/x. The complementary method (sf for c > 0, cdf for c < 0) returns 1.0 there, which is severe while P is still visible relative to 1: 3.0e15 ulp at P = 0.398.
- **Reproducer:** `python accuracy.py refs/gengamma_ref.py --at cdf 1e-40 0.001 10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(1e-40, 0.001, 10)` | 0.0 | 0.3983367031222318 | inf | 0.01 | loss |
  | `cdf(1e-300, 0.001, 10)` | 0.0 | 0.0010005765597449938 | inf | 0.01 | loss |
  | `cdf(1e-200, 1/3, 3)` | 0.0 | 1.1198465217222143e-200 | inf | 1 | loss |
  | `sf(1e+40, 0.001, -10)` | 0.0 | 0.3983367031222318 | inf | 0.01 | loss |
  | `sf(1e-40, 0.001, 10)` | 1.0 | 0.6016632968777682 | 3.0e15 | 0.00662 | severe |
  | `cdf(1e+40, 0.001, -10)` | 1.0 | 0.6016632968777682 | 3.0e15 | 0.00662 | severe |
  | `cdf(4.416325735416545e-33, 0.001, 10)` (x**c subnormal) | 0.4752740574266901 | 0.4750080126268256 | 2.5e12 | 0.01 | severe |
  | `cdf(1e-105, 1/3, 3)` (x**c = 1e-315, subnormal) | 1.119846521155438e-105 | 1.1198465217222007e-105 | 2.3e6 | 1 | severe |
  | `cdf(1.4132001038995107e-108, 1/3, 3)` | 1.9073046065863866e-108 | 1.5825672208493203e-108 | 9.2e14 | 1 | severe |

  Counts on the grid: cdf 27 loss + 1 severe for (0.001, 10), 28 severe for (0.001, -10), 20 loss + 1 severe for (1/3, 3); sf is the mirror image.
- **Fix direction:** work with log y = c*log(x). Where y < ~1e-300 (or a*log y is very negative), use P = exp(a*log y - gammaln(a+1)) * (1 - a*y/(a+1) + ...) (the leading term is exact to double precision once y < eps), and Q = -expm1(log P).

### F2: ppf and isf return 0 or inf when gammaincinv underflows

- **Code:** `_ppf`, lines 3924-3926, and `_isf`, lines 3929-3931: `np.where(c > 0, val1, val2)**(1.0/c)` with `val1 = sc.gammaincinv(a, q)`, `val2 = sc.gammainccinv(a, q)`.
- **Mechanism:** for small p, y = P^{-1}(a, p) ~ (p*Gamma(a+1))**(1/a), so log y = log(p*Gamma(a+1))/a. For a = 0.001 and p = 0.1 that is 1e-1000. `gammaincinv` returns 0, and the final power then gives 0 (c > 0) or inf with a divide-by-zero warning (c < 0). The true quantile y**(1/c) is often an ordinary number: 1e-100 for c = 10 (pattern 3, 8). For c < 0 the small-y branch is `gammainccinv(a, q)` with q close to 1 (ppf) or `gammaincinv(a, q)` with small q (isf).
- **Regime:** log(p*Gamma(a+1))/a < -745 with |log(p*Gamma(a+1))/(a*c)| < 708, where p = q for ppf with c > 0 and isf with c < 0, and p = 1 - q otherwise. For a = 0.001, c = ±10 this covers every q < 0.5 (ppf, c > 0) and the mirror cases. For a = 1/3, c = 3 it covers q < ~1e-113.
- **Reproducer:** `python accuracy.py refs/gengamma_ref.py --at ppf 0.1 0.001 10`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.1, 0.001, 10)` | 0.0 | 9.439903268894046e-101 | inf | 100 | loss |
  | `isf(0.9, 0.001, 10)` | 0.0 | 9.439903268893785e-101 | inf | 900 | loss |
  | `ppf(0.9, 0.001, -10)` | inf | 1.0593328888180283e+100 | inf | 900 | loss |
  | `isf(0.1, 0.001, -10)` | inf | 1.0593328888179987e+100 | inf | 100 | loss |
  | `ppf(1e-150, 1/3, 3)` | 0.0 | 8.929795115692321e-151 | inf | 1 | loss |
  | `isf(1e-150, 1/3, -3)` | inf | 1.1198465217222071e+150 | inf | 1 | loss |

  Counts on the grid: ppf 9 loss for (0.001, 10), 11 for (0.001, -10), 16 for (1/3, 3); isf 11 and 9 for (0.001, ±10).
- **Fix direction:** in the small-p regime compute log y = (log p + gammaln(a+1))/a (plus series corrections, or a Newton step on log P), then x = exp(log y / c). Use `log1p(-q)` for the complementary target.

### F3: logpdf cancels terms of size a*log(a) for large a; pdf amplifies it

- **Code:** `_logpdf`, lines 3903-3904: `np.log(abs(c)) + sc.xlogy(c*a - 1, x) - x**c - sc.gammaln(a)`; `_pdf`, line 3898: `np.exp(self._logpdf(x, a, c))`.
- **Mechanism:** at a = 1000, c = 2, x = 31.1 (mode region) the three terms are ~6.9e3, ~1e3 and ~5.9e3, and they cancel to -0.72. Their rounding errors (a few ulp of 6.9e3) remain as an absolute error (pattern 5). The pdf turns that into the same relative error. This is the same mechanism as gamma F2/F4 (reports/gamma.md) with y = x**c in place of x.
- **Regime:** large a, x**c within a few sqrt(a) of a. The error is about max(c*a*|log x|, x**c, gammaln(a)) ulp in pdf and that divided by |logpdf| in logpdf.
- **Reproducer:** `python accuracy.py refs/gengamma_ref.py --at logpdf 31.118759991335068 1000 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(31.118759991335068, 1000, 2)` | -0.7206053203653937 | -0.7206053203657804 | 2.42e3 | 86.4 | moderate |
  | `logpdf(33.166247903554, 1000, 2)` | -4.963349971550997 | -4.963349971552605 | 1.46e3 | 40.5 | moderate |
  | `pdf(33.166247903554, 1000, 2)` | 0.0069894740305756296 | 0.006989474030564394 | 7.24e3 | 201 | moderate |
  | `pdf(31.118759991335068, 1000, 2)` | 0.48645770406466216 | 0.4864577040644741 | 1.74e3 | 62.2 | moderate |

- **Fix direction:** write logpdf = log|c| - log x + [a*log y - y - gammaln(a)] with y = x**c, and evaluate the bracket in the cancellation-free form used for the gamma pdf (Lanczos / `log1pmx(y/a - 1)`, as in cephes `igam_fac`). Also form y once in log space.

### F4: logpdf rounds c*a - 1 when c*a is close to 1

- **Code:** `_logpdf`, line 3903: `sc.xlogy(c*a - 1, x)`.
- **Mechanism:** with a = float(1/3) and c = 3 the exact c*a - 1 is -5.55e-17, but `c*a` rounds to 1.0, so the coefficient becomes 0. The lost term -5.55e-17*log(x) is 3.8e-14 at x = 1e-300, against logpdf = 0.113 (pattern 5). In general the coefficient carries an absolute error of eps/2, multiplied by |log x|.
- **Regime:** c*a within a few eps of 1 (halfgennorm-type parameters a = 1/c) and x far from 1. The error is about |log x| / |logpdf| ulp. cond (shapes) is 1.2e4 here, so logpdf is very sensitive to the shapes, but the shapes are exact inputs.
- **Reproducer:** `python accuracy.py refs/gengamma_ref.py --at logpdf 1e-300 0.3333333333333333 3`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e-300, 1/3, 3)` | 0.11319164174034269 | 0.1131916417403809 | 1.52e3 | 4.9e-16 | moderate |
  | `logpdf(1e-290, 1/3, 3)` | 0.11319164174034269 | 0.11319164174037963 | 1.47e3 | 4.9e-16 | moderate |

  Grid counts: 11 moderate points for x <= 1e-200, and 22 minor points up to x = 1e-10.

  The pdf at the same points is only 172 ulp off, because the absolute error 3.8e-14 is small relative to 1.
- **Fix direction:** compute `c*a - 1` with a fused multiply-add (`fma(c, a, -1)`, exact up to one rounding) or an error-free product.

### F5: ppf/isf for small |c| raise the inverse's last-bit error to the power 1/c

- **Code:** `_ppf`, line 3926, and `_isf`, line 3931: `(...)**(1.0/c)`.
- **Mechanism:** x = y**(1/c) multiplies the relative error of y by 1/|c| (pattern 6). Even a correctly rounded y (0.5 ulp) gives 500 ulp at c = 0.001. `sc.gammaincinv(2, 0.57)` is 1.99 ulp off (measured against mpmath), which gives the observed 2.0e3 ulp. The harness's cond (arg) (1.05e3) only covers the propagation of the rounding of q.
- **Regime:** |c| << 1, all q; the error is about (error of y in ulp)/|c|.
- **Reproducer:** `python accuracy.py refs/gengamma_ref.py --at ppf 0.57 2 0.001`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.57, 2, 0.001)` | 5.862012699995163e+281 | 5.862012699992511e+281 | 2.04e3 | 1.05e3 | moderate |
  | `isf(0.42999999999999994, 2, 0.001)` | 5.862012699995844e+281 | 5.862012699993715e+281 | 1.63e3 | 796 | moderate |
  | `ppf(0.42999999999999994, 2, 0.001)` | 3.1161873432101454e+165 | 3.1161873432094965e+165 | 938 | 867 | minor |

  The grid point is q = 0.42999999999999994 (one ulp below the double 0.43). `--at isf 0.43 2 0.001` gives reference 5.8620126999931134e+281 and 2.1e3 ulp, and prints cond (arg) 4.9e+235, which is a harness glitch: the references at the two neighbouring q differ by 1.03e-13 relative for a 1.3e-16 relative change of q, so cond is ~800. The real maximum of F5 is the ppf row above (2.04e3 ulp).

- **Fix direction:** obtain log y with more than double precision (e.g. one Newton step on log P(a, y) = log p carried out in log y), then x = exp(log y / c). Alternatively, do a final Newton correction on the gengamma cdf in x.

### F6: cdf/sf for large a inherit the sc.gammainc/gammaincc prefactor error

- **Code:** `_cdf` line 3909 / `_sf` line 3920 call `sc.gammainc(a, xc)` / `sc.gammaincc(a, xc)` correctly.
- **Mechanism:** this is the defect in cephes `igam`/`igamc` described in reports/gamma.md F6. The prefactor `exp(a*log(y) - y - lgam(a))` rounds its exponent of size ~1e3 (pattern 10). The points y = 500 and y = 2000 at a = 1000 are the same points measured there (gamma: cdf 1.6e3, sf 2.0e3 ulp).
- **Regime:** large a, |y - a| > 0.4a (y = x**c). For a >~ 2e5 gamma F1 (the unconverged series) applies as well; that regime was not re-measured here.
- **Reproducer:** `python accuracy.py refs/gengamma_ref.py --at sf 44.721359549995796 1000 2`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(44.721359549995796, 1000, 2)` | 6.84734945961776e-136 | 6.847349459614086e-136 | 2.42e3 | 2e3 | moderate |
  | `cdf(22.360679774997898, 1000, 2)` | 3.298272797068251e-86 | 3.29827279706726e-86 | 1.35e3 | 1e3 | moderate |

- **Fix direction:** none inside `gengamma_gen`; fix in `scipy.special` (see reports/gamma.md).

### F7: pdf = exp(logpdf) in the far tails

- **Code:** `_pdf`, line 3898: `np.exp(self._logpdf(x, a, c))`.
- **Mechanism:** logpdf is accurate to a few ulp of its largest term. Exponentiating it turns that absolute error into a relative error of about max term * eps (pattern 3/5). At x = 1e230, a = 0.5, c = -0.5, logpdf is about -663 and the pdf is 807 ulp off, while cond (arg) is 1.25. F3 is the large-a version of the same issue.
- **Regime:** every parameter set, wherever |logpdf| (or one of its terms) is hundreds: 497 ulp for (0.001, 10), 478 for (2, 0.001), 397 for the example parameters.
- **Reproducer:** `python accuracy.py refs/gengamma_ref.py --at pdf 1e230 0.5 -0.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(1e+230, 0.5, -0.5)` | 8.920620580765454e-289 | 8.920620580763855e-289 | 807 | 1.25 | minor |
  | `pdf(1e+230, 2, 0.001)` | 5.27791209684815e-234 | 5.277912096847658e-234 | 420 | 1 | minor |

- **Fix direction:** compute the pdf as a product, `|c| * exp((c*a - 1)*log x - gammaln(a)) * exp(-x**c)` with the factors split so that no factor carries a large exponent error; or accept it (bounded by ~745 ulp outside F3).

## RuntimeWarnings

- `underflow encountered in power` (x**c) in cdf, sf, logpdf and pdf throughout the F1 regime. There it coincides with wrong cdf/sf results. In logpdf/pdf it is spurious: the term `-x**c` is negligible and the result is accurate.
- `overflow encountered in power` in logpdf/pdf/cdf/sf where x**c overflows (c > 0 and large x, c < 0 and small x). The results (pdf 0, logpdf -inf, cdf 1 or 0) are correct there, because the true logpdf is below -1.8e308 (e.g. `logpdf(1e-300, 1/3, -3)` = -1e900).
- `divide by zero encountered in power` (lines 3926/3931) in ppf/isf for c < 0 in the F2 regime (`0.0**(1/c)`). The returned inf is wrong.
- `underflow encountered in exp` in pdf in the far tails where the true pdf is subnormal or 0 (not measured).

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf, sf | example parameters (c = ±3.12), (2, 0.001), (0.5, -0.5): full grid incl. x**c over/underflow edges | 169 ulp (cdf, x = 1e-20, example c > 0: small-y prefactor of `igam`, gamma F6, minor); 2.95 ulp for c = 0.001 (the hypothesised x**c rounding amplification for \|c\| < 1 does not show) |
| cdf | (0.5, -0.5), x in [0.5, 4.9] | 117 ulp, minor: `sc.gammaincc(0.5, 1.0)` = 0.15729920705028105 vs erfc(1) = 0.15729920705028513 (shared, see below) |
| logpdf | all sets except F3/F4 | 26.3 ulp (example parameters, c < 0) |
| ppf, isf | example parameters, (1000, 2), (0.5, -0.5), q in [1e-300, 1 - 1e-16] | 32 ulp (example), 4.07 (a = 1e3), 241 (0.5, -0.5), minor. The rounding of 1.0/c is amplified by \|log x\| (weibull_min F3 mechanism) |
| ppf, isf | (0.001, ±10) outside F2 | 298 ulp, minor: the a << 1 error of `gammaincinv` (gamma F5, 3.6e3 ulp in y) divided by \|c\| = 10 |
| ppf | (1/3, 3) outside F2 | 133 ulp, minor |

## Suspected, not confirmed

- For a >~ 2e5, gamma F1 (2000-term cap in the cephes `igam` series) and F3 carry over to gengamma's cdf/sf/ppf/isf with y = x**c. These were not re-measured here (a = 1e6 references are slow); the code path is identical.
- `_logpdf` at x = 0 with c > 0 and c*a = 1 in floating point but not exactly (F4 parameters): `xlogy(0, 0)` = 0 gives log|c| - gammaln(a), while the true limit is +inf or 0. This is a measure-zero point and was not measured.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | c > 0: log P(a, y) with y = x**c, via `a*c*log(x) - gammaln(a+1) + log1p(series)` for small y (log-space `igam`); for P near 1 use `log1p(-Q)`. c < 0: same with Q, via `log(gammaincc)` or an asymptotic form `(a-1)*log y - y - gammaln(a)` for large y | cdf underflows in the left tail: `logcdf(1e-120, 4.4162385429431925, 3.119309167924276)` returns -inf, true -3810.166210085287 (loss). Also inherits F1 |
| logsf | log(sf) | c > 0: log Q(a, y) ~ (a-1)*log y - y - gammaln(a) + log(1 + (a-1)/y + ...) for large y (log-space `igamc` continued fraction); c < 0: log P as above | sf underflows in the right tail: `logsf(1000, 4.4162385429431925, 3.119309167924276)` returns -inf, true -2279961377.886318 (loss) |

## Shared code

- `sc.gammainc`/`sc.gammaincc` (cephes `igam`/`igamc`): F6 is reports/gamma.md F6 (exponent of the prefactor), and gamma F1 (series cap for a >~ 2e5) applies unchanged. `sc.gammaincinv`/`sc.gammainccinv`: gamma F5 (a << 1) appears here divided by |c|, and the ~2 ulp last-bit error behind F5 is magnified by 1/|c|. F1 and F2 of this report are not special-function defects: the functions return the correct (underflowed) values, and gengamma passes them y = x**c instead of working with log y.
- New observation (minor, mechanism inside `igamc` not identified): `sc.gammaincc(0.5, x)` is 42-118 ulp off for x in [0.5, 1.0] (0.5: 42, 0.8: 80, 1.0: 118 ulp; Q(0.5, x) = erfc(sqrt x), cond ~0.66). Neighbouring a (0.25, 0.75, 1.5, 2.5) stay at or below 14 ulp. This affects every distribution that calls `gammaincc(0.5, .)`, e.g. gamma and gengamma with a = 0.5, and probably chi2 with df = 1 (`chdtrc(1, x)` = `igamc(0.5, x/2)`; not measured).

## Existing GitHub issues

- #5090 (closed, PR): MAINT: stats: add explicit _sf, _isf to gengamma distribution - background for F1/F2 (introduced the current sf/isf forms)
- #3985 (closed): scipy.stats.gengamma.pdf(0,1,1) returns nan. The correct value is 1.0. - endpoint handling of pdf, no overlap with current findings
- No open issue found for any finding (searched via GitHub search API, gh not installed).

## Verification

Verifier references (`scratch/gengamma/verify/verify.py`, 100 and 200 digits, agreement to 30 digits at every point): cdf/sf from the substitution u = t^a, P(a, y) = Y/Gamma(a+1) * int_0^1 exp(-(Y v)^(1/a)) dv with Y = x^(c a) formed in log space (mp.quad); quantiles from findroot in log y on the series P = y^a e^-y sum y^n/Gamma(a+n+1), checked with cdf/sf(x_ref) = q via the quad route (relative mismatch <= 2e-199); F5 also from the closed form Q(2, y) = e^-y (1 + y) (`f5_isf.py`); logpdf/pdf from the docstring pdf directly.

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e-40, 0.001, 10)` | 0.0 | 0.3983367031222318 | verified | loss; `1e-40**10` = 0.0, so `gammainc(a, 0)` = 0 (lines 3908-3909) |
| F1 | `cdf(1e-300, 0.001, 10)` | 0.0 | 0.0010005765597449937 | verified | loss |
| F1 | `cdf(1e-200, 1/3, 3)` | 0.0 | 1.1198465217222143e-200 | verified | loss |
| F1 | `sf(1e+40, 0.001, -10)` | 0.0 | 0.3983367031222318 | verified | loss |
| F1 | `sf(1e-40, 0.001, 10)` | 1.0 | 0.6016632968777682 | verified | 2.98e15 ulp, severe |
| F1 | `cdf(1e+40, 0.001, -10)` | 1.0 | 0.6016632968777682 | verified | 2.98e15 ulp, severe |
| F1 | `cdf(4.416325735416545e-33, 0.001, 10)` | 0.4752740574266901 | 0.47500801262682561 | verified | 2.52e12 ulp, severe (subnormal x**c) |
| F1 | `cdf(1e-105, 1/3, 3)` | 1.119846521155438e-105 | 1.1198465217222007e-105 | verified | 2.28e6 ulp, severe |
| F1 | `cdf(1.4132001038995107e-108, 1/3, 3)` | 1.9073046065863866e-108 | 1.5825672208493202e-108 | verified | 9.24e14 ulp, severe |
| F2 | `ppf(0.1, 0.001, 10)` | 0.0 | 9.4399032688940463e-101 | verified | loss; `gammaincinv(0.001, 0.1)` = 0.0, then `0**(1/c)` (line 3926) |
| F2 | `isf(0.9, 0.001, 10)` | 0.0 | 9.4399032688937843e-101 | verified | loss |
| F2 | `ppf(0.9, 0.001, -10)` | inf | 1.0593328888180282e+100 | verified | loss; `gammainccinv(0.001, 0.9)` = 0.0 |
| F2 | `isf(0.1, 0.001, -10)` | inf | 1.0593328888179988e+100 | verified | loss |
| F2 | `ppf(1e-150, 1/3, 3)` | 0.0 | 8.9297951156923209e-151 | verified | loss |
| F2 | `isf(1e-150, 1/3, -3)` | inf | 1.1198465217222072e+150 | verified | loss |
| F3 | `logpdf(31.118759991335068, 1000, 2)` | -0.7206053203653937 | -0.72060532036578034 | verified | 2.42e3 ulp, moderate |
| F3 | `pdf(33.166247903554, 1000, 2)` | 0.0069894740305756296 | 0.0069894740305643941 | verified | 7.24e3 ulp, moderate |
| F5 | `ppf(0.57, 2, 0.001)` | 5.862012699995163e+281 | 5.8620126999925111e+281 | verified | 2.04e3 ulp; `gammaincinv(2, 0.57)` is 1.98 ulp off, times 1/c = 1000 |
| F5 | `isf(0.43, 2, 0.001)` | 5.862012699995844e+281 | 5.8620126999931134e+281 | verified | 2.10e3 ulp, not 1.63e3. The report's reference 5.862012699993715e+281 is wrong: sf of it minus 0.43 is -5.5e-17. Both refs/gengamma_ref.py and `accuracy.py --at isf 0.43 2 0.001` give 5.862012699993113e+281 (2.1e3 ulp), so this is a transcription error. The current `--at` run prints cond (arg) 4.9e+235 rather than 796. Severity is still moderate. `gammainccinv(2, 0.43)` is 2.04 ulp off |

The mechanisms match the cited lines: 3908-3910 / 3918-3920 (`xc = x**c` passed to `gammainc`/`gammaincc`), 3926 / 3931 (`(...)**(1.0/c)`), 3903 (`xlogy(c*a - 1, x) - x**c - gammaln(a)`) and 3898 (`exp(logpdf)`).

Verified by dist-accuracy-verifier: 4 of 4 verified.
