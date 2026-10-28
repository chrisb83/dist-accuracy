# Failure patterns: what to look for in the code

Each pattern: what goes wrong, how to spot it in the source, where to test it,
and the usual fix direction (one line in the report; fixing is a later step).
All examples are from `burr` in SciPy 1.18.1 unless noted.

## 1. Subtracting from 1: `1 - F`, `1 - x**a`, `1 - exp(-x)`

* **Mechanism:** when `F` is close to 1, `1 - F` keeps only the digits of `F`
  beyond its leading 1s; when `1 - F < eps/2` the result is exactly 0.
* **Spot:** `1 - `, `1.0 - `, `-(...) + 1`, `np.log1p(-cdf)` with a cdf near 1,
  `sf = 1 - cdf` written out explicitly, `betainc(a, b, 1 - x)`.
* **Test:** the tail where `F -> 1` (right tail for sf/logsf, q -> 1 for ppf).
  Burr `logsf(1e8, 2, 3)` returned `-inf` (true -35.74).
* **Fix direction:** compute the small quantity directly: `-expm1(log F)`,
  complementary special functions (`gammaincc`, `betaincc`, `ndtr(-x)`,
  `erfc`), or a closed form for sf.

## 2. `log(1 + x)`, `log(1 - x)`, `exp(x) - 1` with small `x`

* **Mechanism:** `1 + x` rounds away `x` below eps.
* **Spot:** `np.log(1 + ...)`, `np.log(1 - ...)`, `np.exp(...) - 1`, `1 - np.exp(...)`.
* **Fix direction:** `log1p`, `expm1`, `-expm1`; `log(1 - exp(x))` ->
  `scipy.special._ufuncs._log1mexp(x)`; `log(exp(y) - 1)` -> `y + log1mexp(-y)`.

## 3. Overflow of an intermediate that cancels later

* **Mechanism:** `x**-c`, `exp(x)`, `gamma(a)`, `x**(a-1)` overflow to inf even
  though the final result is finite; then `inf/inf = nan`, `inf - inf = nan`,
  `log(inf) = inf`, `inf**-d = 0`.
* **Spot:** negative powers of `x` with `x` near 0, `exp` of an argument that
  grows with `x` or a shape, `sc.gamma`/`sc.beta` (instead of `gammaln`/`betaln`)
  in ratios.
* **Test:** the argument or shape where the intermediate passes 1.8e308:
  `c*log(x) < -709.8` for `x**-c`. Burr `pdf(1e-160, 2, 0.6)` gave nan
  (true 1.2e-32); `logpdf(1e-200, 2, 3)` gave `-inf`.
* **Fix direction:** rewrite so the large factor never forms (multiply through
  by `x**c`), or work in log space (`gammaln`, `betaln`, `xlogy`, `xlog1py`).

## 4. Underflow or subnormal intermediates

* **Mechanism:** a factor underflows to 0 (or becomes subnormal, with fewer
  than 53 significant bits) while the final result is normal after
  multiplying by something large.
* **Spot:** products like `c * d * x**(-c-1)` with large `c*d`;
  `x**a * exp(-x)` with large `a`; `exp(a) * exp(b)` instead of `exp(a + b)`.
* **Test:** where the intermediate drops below 2.2e-308 but the result does not.
  Burr `pdf(1300, 100, 1e6)`: `1300**-101` is subnormal, error 2.6e6 ulp.
* **Fix direction:** combine exponents in log space before `exp`.

## 5. Cancellation of large terms (often in log space)

* **Mechanism:** two terms of size `L` that cancel to a much smaller result
  lose `L/|result|` ulp; if the result is below `L*eps` it can come out 0.
* **Spot:** sums of terms with opposite signs that both grow, e.g.
  `-(c+1)*log(x) - (d+1)*log1p(x**-c)`; `log(a) - log(b)` with `a ~ b`;
  differences of `gammaln`/`betaln` with large arguments.
* **Test:** the regime where the terms are large but the result is small,
  including special parameter relations. Burr `logpdf(1e-10, 2, 0.5)`
  (c*d = 1) returned 0.0, true -1.5e-20.
* **Fix direction:** combine the large terms analytically first (here into
  `(c*d - 1)*log(x)`), or use dedicated functions (`log1p`, `logbeta` helpers).

## 6. A rounding error raised to a large power

* **Mechanism:** `(1 + u)**d` multiplies the rounding error of `1 + u`
  (up to 1/2 ulp) by `d`.
* **Spot:** `(...)**d`, `(...)**(-d)`, `(...)**(d + 1)` with a shape that can
  be large.
* **Test:** large shapes (1e3 to 1e6) in the bulk and near 1.
  Burr `cdf(1e3, 3, 1e4)`: 3.7e3 ulp; `pdf(100, 3, 1e6)`: 3.7e5 ulp.
* **Fix direction:** `exp(d * log1p(u))`; the error is then bounded by
  about `min(d, 745)` ulp.

## 7. Branches and thresholds

* **Mechanism:** each branch is accurate only on its side; a wrong or missing
  threshold leaves a gap, or the switch point is placed where both forms are
  poor.
* **Spot:** `np.where`, `xpx.apply_where`, `_lazywhere`, `_lazyselect`,
  `if`/`else` on arguments or shapes, `np.select`.
* **Test:** both sides of every threshold (pass them via `x_points`), and the
  regime each branch was *not* written for.
* **Note:** `np.where` evaluates both branches on all elements, which causes
  RuntimeWarnings (and possibly nan propagation) even when the result is
  right. Record such warnings separately from accuracy findings.

## 8. Quantile functions

* **Mechanism:** formulas like `(q**(-1/d) - 1)**(-1/c)` cancel when
  `q**(-1/d)` is close to 1 (q -> 1, or large d), and overflow for small q
  or small d; `isf` implemented as `ppf(1 - q)` rounds away small q.
* **Spot:** `q**(...) - 1`, `1 - q`, `log(q)` where `log1p(q - 1)` is
  needed, inverse special functions applied to `1 - q`.
* **Test:** q in [1e-300, 1e-3] and 1 - q in [1e-16, 1e-3]; small and large shapes.
  Burr `ppf(1 - 1e-8, 3, 1e6)`: relative error 2.7e-4; `isf(0.9, 100, 1e-3)` gave 0.
* **Fix direction:** `expm1(-log(q)/d)`, `log1p(-q)`, complementary inverse
  functions (`gammainccinv`, `betainccinv`), log-space forms for overflow.
* **Note:** ppf near q = 1 is ill-conditioned with respect to q itself
  (cond ~ 1/(1 - q)); the harness accounts for that. Errors beyond the
  conditioning are still findings.

## 9. Limits of a shape parameter

* **Mechanism:** formulas with `1/c`, `x**(1/c)`, `(1 + c*x)**(-1/c)` lose
  accuracy as `c -> 0` (the limit is often an exponential), or overflow as
  `c -> inf`; a special branch for `c == 0` exactly does not help `c = 1e-12`.
* **Spot:** division by a shape, `== 0` tests on shapes, `boxcox`-like
  expressions. Example (not burr): genpareto near c = 0.
* **Test:** shapes 1e-300, 1e-12, 1e-6 and large values; both sides of 0 if allowed.
* **Fix direction:** `log1p`/`expm1`-based forms, `sc.boxcox`, `sc.inv_boxcox`,
  series for small shapes.

## 10. Special functions used outside their accurate range

* **Mechanism:** the special function is accurate for its own result but the
  distribution needs a different quantity (e.g. `1 - gammainc` instead of
  `gammaincc`, `log(gammainc)` where it underflows, `exp(gammaln(a)) *`
  ratios). Also: an argument rounded before the call (`betainc(b, a, 1 - x)`).
* **Spot:** calls into `sc.*`; whether the complementary/log variant is used.
* **Test:** the regime where the special function's result is near 0 or 1.
* **Note:** if a `scipy.special` function itself is inaccurate, report it
  as a shared issue (it affects all distributions using it) and say so.

## 11. Inaccurate constants and literals

* **Spot:** float literals with few digits (`0.5772`, `1.6449`), `np.pi/2`
  computed at import is fine. Rare but cheap to check.

## 12. Missing methods (opportunities, not measured)

* Inherited generic defaults: `logpdf = log(pdf)`, `logcdf = log(cdf)`,
  `sf = 1 - cdf`, `logsf = log(sf)`, `isf = ppf(1 - q)`, and numerical
  `cdf`/`ppf`. Do not benchmark them. Instead note whether a closed form or
  a better formula exists (e.g. sf via a complementary function, logcdf via
  `log_ndtr`), in which regime the default fails, and what the formula is.
  One cheap demonstration point is fine; no grids.
