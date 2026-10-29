# exponnorm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `exponnorm_gen`, lines 2162-2259
- **Implemented:** pdf, logpdf, cdf, sf
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf), ppf = numerical inversion of cdf, isf = ppf(1 - q)
- **Related:** none; helpers: `_norm_cdf` (sc.ndtr), `_norm_logcdf` (sc.log_ndtr); special functions: sc.erfcx, sc.ndtr, sc.log_ndtr
- **Reference:** `refs/exponnorm_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (1.5), (0.999), (1), (0.1), (1e-05), (1e-200), (10), (1e4), (1e8), (1e17)
- **Worst finding:** loss in cdf

## Summary

sf is accurate everywhere (max 256 ulp, always within conditioning), and so
are logpdf and pdf for all shapes except extreme ones. cdf is the problem.
Both branches compute it as a difference,
`Phi(x) - exp(1/(2K^2) - x/K) Phi(x - 1/K)`, and the two terms cancel in two
regimes. For large K they cancel in the bulk (about K ulp lost: 3.3e3 ulp at
K = 1e4, 5.8e6 ulp at K = 1e8, and 0.0 or negative values at K = 1e17). In
the left tail they cancel for every K, by a factor of about 1 + K|x|
(moderate for K ~ 1, severe for K = 1e4). logpdf loses up to 98.5 ulp for
tiny K because `-log(K)` cancels against `log(erfcx(u))`, and
pdf = exp(logpdf) turns the absolute error of logpdf into a relative error.
Both of these are minor.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | large K (>~ 1e3), bulk and right of the mode while x/K is O(1) or less | Phi(x) - exp(logprod) cancels, cdf ~ Phi(x)*x/K (pattern 5/1) | 0.0 and negative values at K = 1e17; 5.8e6 ulp at K = 1e8 | loss |
| F2 | cdf | left tail x << 0, any K >~ 0.1 (both branches) | both terms ~phi(x)/abs(x) cancel by a factor ~1 + K*abs(x) (pattern 5), which amplifies the up-to-~x^2/2 ulp errors of `ndtr(x)` (both branches, see Shared code) and of `exp(logprod)` (default branch) | 6.3e7 ulp (K = 1e4), 2.2e4 ulp (K = 1.5), 2.4e3 ulp (K = 0.1) | severe |
| F3 | logpdf | tiny K (erfcx branch, K < 1, x <= 1/K), abs(log K) >> abs(logpdf) | -log(K) and log(erfcx(u)) cancel (pattern 5) | 98.5 ulp at K = 1e-200 | minor |
| F4 | pdf | abs(logpdf) large because of the shape (tiny or huge K) | pdf = exp(logpdf) turns the absolute error of logpdf into a relative error | 90.7 ulp (K = 1e-200), 79.3 ulp (K = 1e17) | minor |

### F1: cdf cancels in the bulk for large K

- **Code:** `_cdf`, `cdf_default`, lines 2234-2236:
  `logprod = invK*(0.5*invK - x) + _norm_logcdf(x - invK)`; `return _norm_cdf(x) - np.exp(logprod)`.
- **Mechanism:** for large K, `exp(logprod) = exp(-x/K + ...) Phi(x - 1/K)`
  is Phi(x)(1 - O(1/K)). The difference therefore keeps only about
  `1/(K * Phi(x))` of the relative digits (pattern 5, a 1 - F form of
  pattern 1). For K >~ 1e16 the two terms round to the same double, which
  gives 0.0 or a negative value of the size of an ulp of Phi(x).
- **Regime:** K >= 1 (default branch), x from the left tail up to where x/K
  is O(1). The loss is about K ulp in the bulk and becomes total (loss) once
  K >~ 1/eps. Measured: K = 1e4 gives 3.3e3 ulp at x = 0; K = 1e8 gives
  5.8e6 ulp; K = 1e17 gives 0.0 or negative values for x in [-30, 1] and
  still 2.2e6 ulp at x = 1e10.
- **Reproducer:** `python accuracy.py refs/exponnorm_ref.py --at cdf 1 1e8`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-1.0, 1e17)` | 0.0 | 8.33154705876863e-19 | inf | 1.9 | loss |
  | `cdf(-10.0, 1e17)` | -8.816207631167156e-39 | 7.474560254589328e-42 | inf | 102 | loss |
  | `cdf(1.0, 1e17)` | 0.0 | 1.0833154705876862e-17 | inf | 0.777 | loss |
  | `cdf(1e10, 1e17)` | 9.99999949513608e-08 | 9.999999500000016e-08 | 2.2e+06 | 1 | severe |
  | `cdf(1.0, 1e8)` | 1.0833154595601968e-08 | 1.0833154609643852e-08 | 5.8e+06 | 0.777 | severe |
  | `cdf(0.0, 1e4)` | 3.989172817314701e-05 | 3.989172817311778e-05 | 3.3e+03 | 0 | moderate |
  | `cdf(5.0, 1e4)` | 0.0004998700286763524 | 0.0004998700286763543 | 16.6 | 1 | minor |

  (cond (shapes) is 1 at all of these points.)
- **Fix direction:** avoid the subtraction. Write
  `cdf = Phi(x) * (-expm1(logprod - log_ndtr(x)))`, with the exponent
  difference `-x/K + 1/(2K^2) + [log_ndtr(x - 1/K) - log_ndtr(x)]` computed
  stably for small 1/K (series in 1/K or an integral form).

### F2: cdf cancels in the left tail for every K

- **Code:** `cdf_erfcx`, line 2231: `_norm_cdf(x) - 0.5*np.exp(-0.5*x**2)*sc.erfcx(u)`
  (K < 1, x <= 1/K), and `cdf_default`, line 2236 (K >= 1, or x > 1/K).
- **Mechanism:** for x -> -inf, Phi(x) ~ phi(x)/|x| and the subtracted term
  ~ phi(x)/(|x| + 1/K), so F(x) is smaller than Phi(x) by a factor of about
  1 + K|x| (pattern 5). Both terms carry relative errors of up to ~x^2/2 ulp,
  each conditioning-level for the term on its own, and the cancellation
  multiplies them by Phi(x)/F(x) ~ 1 + K|x|, so the result goes beyond the
  conditioning of the cdf (about x^2) by that factor. Which term dominates
  depends on the branch (per-term split in
  `scratch/exponnorm/f2recheck/decompose.txt`, errors in ulp of F):
  - `_norm_cdf(x)` = `sc.ndtr(x)` (both branches) is itself off by up to
    ~512 ulp in the far left tail (511 ulp at x = -37; see Shared code).
    The size jumps between ~0.2 and 2^k ulp from point to point.
  - Erfcx branch (K < 1): the subtracted term is accurate at integer x
    (`0.5*x**2` is exact; 0.36 ulp at x = -37, K = 0.1). There F2 comes
    almost entirely from ndtr: 511 ulp x Phi/F = 4.7 gives 2.41e3 of the
    measured 2.41e3 ulp, and K = 0.999 gives 1.94e4 of 1.94e4. At non-integer
    x, rounding `0.5*x**2` adds a smaller part (term error 26 ulp at
    x = -30.3, K = 0.1).
  - Default branch (K >= 1): `exp(logprod)` is itself off by 100-700 ulp,
    because `logprod` (size ~x^2/2) is rounded. That is the same order as
    ndtr. At `cdf(-37, 1.5)` the ndtr part is 2.9e4 and the exp part 6.8e3,
    and they partly cancel to the measured 2.21e4. At x = -33 and -28
    (K = 1.5), where ndtr happens to be accurate, the whole error
    (1.6e4 and 1.8e4 ulp) comes from `exp(logprod)`. At `cdf(-30, 10)` the
    exp part dominates (2.1e5 vs 7.7e4).
- **Regime:** x << 0, growing with K. At x = -30 ... -37 it is about 2.4e3 ulp
  for K = 0.1, about 2e4 ulp for K ~ 1, 1.4e5 for K = 10 and 6.3e7 for
  K = 1e4 (where it combines with F1). Below x ~ -38 the cdf is subnormal
  and is not measured. For K <= 1e-5 the extra factor is ~1 and the error
  stays within conditioning (max 512 ulp).
- **Reproducer:** `python accuracy.py refs/exponnorm_ref.py --at cdf -37 1.5`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `cdf(-37.0, 1e4)` | 1.5451950017616929e-305 | 1.5451950234146176e-305 | 6.3e+07 | 1.37e+03 | severe |
  | `cdf(-20.0, 1e8)` | 1.3700727692148748e-98 | 1.3700124940496235e-98 | 2.0e+11 | 402 | severe |
  | `cdf(-30.0, 10)` | 1.626552805595171e-200 | 1.6265528055459852e-200 | 1.4e+05 | 902 | moderate |
  | `cdf(-37.0, 1.0)` | 1.5046216304283307e-301 | 1.504621630436529e-301 | 2.45e+04 | 1.37e+03 | moderate |
  | `cdf(-37.0, 1.5)` | 1.0119391755794792e-301 | 1.0119391755844516e-301 | 2.21e+04 | 1.37e+03 | moderate |
  | `cdf(-37.0, 0.999)` | 1.5060881673755431e-301 | 1.5060881673820442e-301 | 1.94e+04 | 1.37e+03 | moderate |
  | `cdf(-37.0, 0.1)` | 1.2169581614394047e-300 | 1.2169581614400552e-300 | 2.41e+03 | 1.37e+03 | moderate |
  | `cdf(-10.0, 1.5)` | 4.679659012813426e-25 | 4.679659012813828e-25 | 386 | 102 | minor |

- **Fix direction:** in the left tail, factor out the common
  `0.5*exp(-x^2/2)` and evaluate `erfcx(-x/sqrt2) - erfcx(u)` stably. The two
  arguments differ by 1/(K*sqrt2), so a difference formula, series or
  continued fraction works. Both terms then share one exp factor, so the
  ~x^2/2 ulp error of that factor is no longer amplified. Do not subtract
  `sc.ndtr(x)` (whose tail error is not accurately rounded) from a separately
  rounded exponential. Fixing ndtr alone would not be enough for K >= 1,
  where `exp(logprod)` has an error of the same size.

### F3: logpdf cancels -log(K) against log(erfcx(u)) for tiny K

- **Code:** `logpdf_erfcx`, line 2219: `-np.log(2) - 0.5*(x**2) - np.log(K) + erfcx_term`.
- **Mechanism:** for tiny K, u ~ 1/(K*sqrt2) is huge and
  `log(erfcx(u)) ~ log(K) + log(sqrt(2/pi))`. With K = 1e-200 both terms are
  about 460.5 in magnitude and the result is about -0.919 - x^2/2. The
  rounding of the large terms (0.5 ulp of 460) therefore appears as ~100 ulp
  of the result (pattern 5).
- **Regime:** K < 1, x <= 1/K, |log K| >> |logpdf|. The loss is about
  |log K| / |logpdf| ulp: 98.5 ulp at K = 1e-200, and within conditioning at
  K = 1e-5 (2.8 ulp).
- **Reproducer:** `python accuracy.py refs/exponnorm_ref.py --at logpdf -1e-10 1e-200`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(-1e-10, 1e-200)` | -0.9189385332046527 | -0.9189385332046728 | 98.5 | 1.09e-20 | minor |

- **Fix direction:** combine analytically: `log(erfcx(u)) - log(K)` =
  `log(u*erfcx(u)) - log(K*u)`, with `K*u = (1 - K*x)/sqrt2` (use `log1p(-K*x)`)
  and `u*erfcx(u)` ~ 1/sqrt(pi), which is O(1).

### F4: pdf = exp(logpdf) amplifies the absolute error of logpdf

- **Code:** `_pdf`, line 2213: `np.exp(self._logpdf(x, K))`.
- **Mechanism:** an absolute error d in logpdf becomes a relative error d in
  pdf. logpdf is a sum of terms of size |log K|, x^2/2 and x/K, each rounded
  to 0.5 ulp, so the pdf's relative error is of order max(|log K|, ...) ulp.
  When |log K| dominates (tiny or huge K), this goes beyond the pdf's
  conditioning with respect to x. For K = 1e-200 it also includes F3.
- **Regime:** K = 1e-200 near the mode (up to 90.7 ulp); K = 1e8 (up to 14
  ulp); K = 1e17 (up to 79 ulp). For moderate K, the pdf errors in the tails
  (up to 839 ulp) stay within the conditioning (~x^2).
- **Reproducer:** `python accuracy.py refs/exponnorm_ref.py --at pdf -6 1e-200`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `pdf(-6.0, 1e-200)` | 6.075882849823408e-09 | 6.075882849823285e-09 | 90.7 | 36 | minor |
  | `pdf(-8.0, 1e17)` | 6.220960574271674e-33 | 6.220960574271784e-33 | 79.3 | 65 | minor |
  | `pdf(7.0, 1e8)` | 9.999999299987195e-09 | 9.999999299987227e-09 | 14.2 | 6.99e-08 | minor |

- **Fix direction:** in the bulk, compute pdf directly as
  `exp(-x^2/2) * erfcx(u) / (2K)` (erfcx branch) or
  `exp(exparg) * ndtr(x - 1/K) / K`. Keep exp(logpdf) only where an
  intermediate would underflow.

## RuntimeWarnings

Overflow and underflow warnings (`overflow encountered in square` /
`multiply`, `underflow encountered in exp` / `square` / `multiply`) occur in
every method. They come from `x**2` for |x| > 1e154, from `exp` underflow in
the far tails (where the true result is subnormal or 0, or the term is
negligible next to `ndtr`), and for K = 1e-200 from `1/K` products. The
harness records warnings per vectorized call, so it attributes them to whole
parameter sets. Single bulk points (`cdf(-1, 1.5)`, `cdf(-30, 1.5)`,
`logpdf(1, 1e-200)`, `sf(-1, 0.999)`) give no warnings. The flagged
inaccuracies (F1, F2) do not produce their own warnings; the warnings listed
with them come from extreme points in the same call. No warning points to a
wrong result.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| sf | full grid incl. both sides of u = 0 (x = 1/K), K = 1 vs 0.999, right exponential tail up to x/K ~ 1e3, all parameter sets | 256 ulp, all within conditioning |
| logpdf | full grid, all parameter sets except K = 1e-200 (F3); both branches, both sides of x = 1/K and of K = 1 | 2.82 ulp |
| pdf | full grid for K in [1e-5, 1e4], tails | 839 ulp, within conditioning (cond up to 1.37e3) |
| cdf | right tail, all parameter sets; small K (1e-5, 1e-200) everywhere | 0.5 ulp right tail; 512 ulp (within conditioning) for K <= 1e-5 |

Hypothesis H4 (overflow of `0.5*invK*invK` for K < ~1e-154 in the default
branch) turned out harmless. For tiny K that branch is used only when
x > 1/K, where every result underflows. The branch switch at u = 0 / K = 1
causes no change in accuracy.

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | log(cdf) | left tail: `-x^2/2 + log(0.5*(erfcx(-x/sqrt2) - erfcx(u)))` with a stable difference; elsewhere `log_ndtr(x) + log1p(-exp(logprod - log_ndtr(x)))` | cdf underflows for x < ~-38: `logcdf(-40, 1.5)` = -inf, true -808.72053232827719 |
| logsf | log(sf) | `logaddexp(log_ndtr(-x), logprod)` with `logprod = (0.5/K - x)/K + log_ndtr(x - 1/K)` (default branch), and the analogous erfcx form for K < 1 | exponential tail, x/K > ~745: `logsf(2000, 1.5)` = -inf, true -1333.1111111111111 |
| isf | ppf(1 - q) | numerical inversion of the implemented (accurate) `_sf`; asymptotically x ~ 1/(2K) - K*log(q) | small q: `isf(1e-20, 1.5)` = 100.0, true 69.410886123154704 |
| ppf | numerical inversion of cdf | invert in log space using a fixed logcdf | left tail is limited by F2, and large K by F1; one point at K = 1.5 is fine: `ppf(1e-300, 1.5)` = -36.938126479864096 vs -36.938126479864169 |

## Shared code

- **sc.ndtr, far left tail (used via `_norm_cdf` by many distributions):**
  measured alone against mpmath for x in [-38, -5]
  (`scratch/exponnorm/f2recheck/ndtr_tail.txt`). The relative error reaches
  511-514 ulp at x = -37, -36, -34, -32, -31, -29 and -26, 256 ulp at -30,
  -27 and -24, 128 ulp around -16 ... -22, 64 ulp at -12 and -15, and ~30 ulp
  at -8 ... -11. At other points it is below 1 ulp (e.g. -35, -33, -28). The
  power-of-two sizes match one ulp of the internal exponent z^2 ~ x^2/2 in
  erfc's exp(-z^2), not just the rounding of z = x/sqrt2
  (`ndtr_mech.txt`). `0.5*sc.erfc(-x/sqrt2)` behaves the same way.
  `sc.erfcx(-x/sqrt2)` and `sc.log_ndtr(x)` are accurate (<= 1.8 ulp).
  This stays within ndtr's own conditioning (~x^2, 1.37e3 at x = -37), so
  `norm.cdf` itself is not a finding. It matters to every distribution that
  subtracts `_norm_cdf(x)` from a nearly equal quantity in the tail, where the
  cancellation amplifies it (exponnorm F2). An accurately rounded ndtr in this
  range could use `0.5*erfcx(-x/sqrt2)*exp(-x^2/2)`, with x^2/2 split exactly
  (Dekker product) and the low part applied as a correction factor.
- sc.erfcx and sc.log_ndtr: no problem found.

## Existing GitHub issues

- #24551 (closed, fixed by PR #24597): BUG: Instability in exponnorm for very small K - small-K logpdf/cdf/sf; relates to F3 (residual tiny-K logpdf cancellation remains in 1.18.1)
- #13428 (closed): `stats.exponnorm.cdf` returns `nan` for small values of `K` - earlier small-K cdf issue; does not cover F1 (large K) or F2 (left tail)
- #12326 (closed): BUG: stats.exponnorm.pdf returns 0 for small K - historical, relates to F3/F4 regime
- #10003 (closed): exponnorm.pdf returns NAN for small K - historical, relates to F3/F4 regime
- No issue found for F1 (large-K cdf cancellation) or F2 (left-tail cdf), nor for the sc.ndtr tail error.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1.0, 1e17)` | 0.0 | 1.0833154705876863e-17 | verified (loss) | reference = mp.quad of convolution int_0^inf (1/K) e^(-t/K) Phi(x - t) dt; 100 vs 200 digits agree to 1e-101 |
| F1 | `cdf(-10.0, 1e17)` | -8.816207631167156e-39 | 7.474560254589328e-42 | verified (loss) | negative result; same reference also by layer integral (below) |
| F1 | `cdf(-1.0, 1e17)` | 0.0 | 8.3315470587686298e-19 | verified (loss) | |
| F1 | `cdf(1e10, 1e17)` | 9.99999949513608e-08 | 9.9999995000000167e-08 | verified (2.19e6 ulp, severe) | |
| F1 | `cdf(1.0, 1e8)` | 1.0833154595601968e-08 | 1.0833154609643853e-08 | verified (5.84e6 ulp, severe) | |
| F1 | `cdf(0.0, 1e4)` | 3.989172817314701e-05 | 3.9891728173117778e-05 | verified (3.3e3 ulp, moderate) | pdf integral agrees to 7e-102. Mechanism matches lines 2234-2236 |
| F2 | `cdf(-37.0, 1e4)` | 1.5451950017616929e-305 | 1.5451950234146175e-305 | verified (6.31e7 ulp, severe) | reference = phi(x) * int_0^inf exp(x d - d^2/2) (-expm1(-d/K)) dd (mp.quad, split on the 1/abs(x) scale); 100 vs 200 digits agree to 1e-101 |
| F2 | `cdf(-20.0, 1e8)` | 1.3700727692148748e-98 | 1.3700124940496235e-98 | verified (1.98e11 ulp, severe) | |
| F2 | `cdf(-30.0, 10)` | 1.626552805595171e-200 | 1.6265528055459853e-200 | verified (1.36e5 ulp, moderate) | |
| F2 | `cdf(-37.0, 1.0)` | 1.5046216304283307e-301 | 1.5046216304365289e-301 | verified (2.45e4 ulp, moderate) | |
| F2 | `cdf(-37.0, 1.5)` | 1.0119391755794792e-301 | 1.0119391755844516e-301 | verified (2.21e4 ulp, moderate) | |
| F2 | `cdf(-37.0, 0.1)` | 1.2169581614394047e-300 | 1.2169581614400553e-300 | verified (2.41e3 ulp, moderate) | Mechanism needs a correction. The cancellation by about 1 + K*abs(x) is right, but the amplified error mostly comes from `ndtr(x)` (511 ulp at x = -37, its own conditioning-level error), not from the subtracted exp term. At integer x, `0.5*x**2` is exact and the erfcx-branch term has 0.36 ulp error (K = 0.1); in the default branch, exp(logprod) has 123 ulp (K = 1.5). Check: 511 * 4.7 = 2.4e3 (K = 0.1) |

Verified by dist-accuracy-verifier: 2 of 2 verified.
