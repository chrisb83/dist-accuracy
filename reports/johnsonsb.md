# johnsonsb: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `johnsonsb_gen`, lines 5832-5885
- **Implemented:** pdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logpdf = log(pdf), logcdf = log(cdf), logsf = log(sf)
- **Related:** none; helpers: `_norm_pdf`, `_norm_cdf`, `_norm_sf`, `_norm_ppf`, `_norm_isf`; special functions: `sc.logit`, `sc.expit` (plus `sc.ndtr`, `sc.ndtri` via the helpers)
- **Reference:** `refs/johnsonsb_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (4.317267509914106, 3.1837781130785063), (0, 1), (30, 0.5), (-30, 0.5), (0.5, 0.001), (2, 100), (-50, 50)
- **Worst finding:** loss in pdf

## Summary

Every method has the form g(z) with z = a + b*logit(x) (or x = expit((Phi^-1(q) - a)/b)),
and the complementary forms are used correctly (`ndtr(-z)` for sf, `-ndtri(q)`
for isf), so cdf, sf, ppf and isf have no cancellation and never go to 0/1
early. The one real defect is in the pdf: `_norm_pdf(z)` goes subnormal or
underflows for |z| > ~37.5 before it is multiplied by the large factor
b/(x(1-x)), so in the far left tail the pdf comes out 0 (loss) or with only
3 to 4 correct digits (3.9e12 ulp, severe) although the true value is a normal
double. Apart from that, errors of a few hundred ulp (max 891 ulp, minor)
appear in pdf/cdf/sf where |z| is 30 to 38, because the rounding of z is
amplified by about z**2 in the normal tail; ppf/isf have up to 65 ulp
(minor) from the rounding of the expit argument amplified by its size.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | pdf | 37.5 < \|z\| < ~39 with large b/(x(1-x)), i.e. small x (left tail) | `_norm_pdf(z)` subnormal/underflows before multiplication by b/(x(1-x)) (pattern 4) | 0.0 for 5.06e-273; 3.9e12 ulp | loss |
| F2 | pdf, cdf, sf | \|z\| ~ 30-38 (large \|a\| in the bulk, or tails) | rounding of z = a + b*logit(x) amplified by ~z**2 in exp(-z**2/2) / ndtr (pattern 6-like) | 891 ulp (cdf), 824 ulp (sf), 422 ulp (pdf) | minor |
| F3 | ppf, isf | x = expit(t) with t << 0 (x tiny) | rounding of t = (ndtri(q) - a)/b amplified by \|t\| in expit | 64.9 ulp | minor |

### F1: pdf underflows in the normal factor before the large prefactor is applied

- **Code:** `_pdf`, lines 5872-5873:
  `trm = _norm_pdf(a + b*sc.logit(x))`; `return b*1.0/(x*(1-x))*trm`,
  with `_norm_pdf(x) = np.exp(-x**2/2.0) / _norm_pdf_C` (line 363).
- **Mechanism:** pattern 4. phi(z) drops below 2.2e-308 for |z| > ~37.5 and
  to 0 for |z| > ~38.6. In the left tail the prefactor b/(x(1-x)) ~ b/x is
  huge (1e17 to 1e60 in the examples), so the true pdf is a normal double,
  but it is formed from a subnormal phi (few significant bits) or from 0.
  For (0, 1) at x = 2.1e-17, z = -38.4, phi(z) = 2.5e-321 has about 10
  significant bits, giving a relative error of 8.6e-4. For (30, 0.5) at
  x = 1e-60, z = -39.08, phi(z) underflows to 0 and the pdf is 0.0 instead
  of 5.06e-273.
- **Regime:** 37.5 < |z| and b/(x(1-x)) * phi(z) > 2.2e-308, i.e. the left
  tail (x small) whenever b*|log x| ~ 38 is reached with |log x| large enough
  that 1/x compensates; the band widens as b decreases (for b = 0.5 it spans
  x from ~1e-60 down to where the true pdf itself becomes subnormal). The
  mirrored right tail is only marginally reachable because 1 - x >= 1.1e-16.
- **Reproducer:** `python accuracy.py refs/johnsonsb_ref.py --at pdf 1e-60 30 0.5`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1e-60, 30, 0.5)` | 0.0 | 5.059077895564703e-273 | inf | 18.5 | loss |
  | `pdf(3.877170285277934e-60, 30, 0.5)` | 3.268565186303012e-262 | 3.2713693236792274e-262 | 3.9e12 | 18.2 | severe |
  | `pdf(2.104223637677623e-17, 0, 1)` | 1.204509215552259e-304 | 1.2055425770181894e-304 | 3.9e12 | 37.4 | severe |

- **Fix direction:** evaluate in log space when |z| is large:
  `exp(log(b) - log(x) - log1p(-x) + _norm_logpdf(z))`.

### F2: rounding of z amplified by ~z**2 in the normal tails (pdf, cdf, sf)

- **Code:** lines 5872 (`_norm_pdf(a + b*sc.logit(x))`), 5876
  (`_norm_cdf(a + b*sc.logit(x))`), 5882 (`_norm_sf(a + b*sc.logit(x))`).
- **Mechanism:** z = a + b*logit(x) is rounded (relative error ~0.25 ulp in
  the examples, i.e. absolute error ~eps*|z|); phi(z) and Phi(-|z|) have
  relative sensitivity ~z**2 to z, so the result error is ~z**2 * 0.25 ulp
  (~225 ulp at z = 30). `_norm_pdf` adds the rounding of `x**2/2` (another
  up to ~z**2/2 ulp). Measured: for `pdf(0.2, 30, 0.5)`, evaluating phi
  exactly at the rounded double z already gives 220 of the 333 ulp; for
  `cdf(0.4, -30, 0.5)` 205 of 680 ulp; for `sf(0.8, 30, 0.5)` 241 of 824 ulp.
  The harness conditioning w.r.t. x is only ~b*|z|/(1-x) (15 to 300 here),
  so these errors are beyond conditioning. All are below 1e3 ulp (minor).
- **Regime:** |z| between ~25 and the underflow at ~38, which covers the whole
  bulk of x when |a| is large (e.g. a = +-30, b = 0.5) and the tails
  otherwise. Error grows like z**2/2 ulp, so it is bounded by ~900 ulp
  before the result underflows.
- **Reproducer:** `python accuracy.py refs/johnsonsb_ref.py --at cdf 0.4 -30 0.5`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(0.4, -30, 0.5)` | 1.0903813975483827e-200 | 1.0903813975485473e-200 | 680 | 25.2 | minor |
  | `sf(0.8, 30, 0.5)` | 3.512878164054336e-207 | 3.5128781640549784e-207 | 824 | 76.8 | minor |
  | `pdf(1e-50, 30, 0.5)` | 2.039490341983427e-116 | 2.039490341983618e-116 | 422 | 12.8 | minor |
  | `pdf(0.2, 30, 0.5)` | 3.888786089632932e-187 | 3.8887860896332196e-187 | 333 | 19.1 | minor |

  (cdf (-50, 50) reached 891 ulp in the low tail, cond 4.29e3 there; cond
  (shapes) is ~900 at all these points, comparable to the error.)
- **Fix direction:** only with extra precision for z (error-free
  `a + b*logit(x)` via two-sum/two-product and a split `exp(-z**2/2)`);
  low priority.

### F3: ppf/isf rounding of the expit argument amplified by its magnitude

- **Code:** `_ppf`, line 5879: `sc.expit(1.0 / b * (_norm_ppf(q) - a))`;
  `_isf`, line 5885: `sc.expit(1.0 / b * (_norm_isf(q) - a))`.
- **Mechanism:** for t = (ndtri(q) - a)/b << 0, expit(t) ~ exp(t), whose
  relative sensitivity to t is |t|. The absolute error of t (rounding of
  ndtri, of the subtraction, of `1.0 / b` and the product, ~eps*|t|) becomes
  ~|t| ulp in x. With a = 30, b = 0.5, |t| is 60 to 130, matching the
  10 to 65 ulp observed. The harness cond w.r.t. q is below 1 there.
- **Regime:** x = ppf(q) or isf(q) very small (t < ~-20), e.g. large
  positive a with small b, or q -> 0 for ppf / q -> 1 for isf.
- **Reproducer:** `python accuracy.py refs/johnsonsb_ref.py --at ppf 2.872984833353539e-288 -30 0.5`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(2.872984833353539e-288, -30, 0.5)` | 3.6158049254220767e-06 | 3.6158049254220246e-06 | 64.9 | 0.0551 | minor |
  | `ppf(2.6101572156826508e-126, 30, 0.5)` | 1.593711688626662e-47 | 1.5937116886266391e-47 | 64.6 | 0.0836 | minor |
  | `isf(2.6101572156826508e-126, 30, 0.5)` | 4.811165807538987e-06 | 4.811165807539023e-06 | 33.3 | 0.0836 | minor |

- **Fix direction:** `(z - a) / b` instead of `1.0 / b * (z - a)` removes one
  rounding; otherwise inherent (cond (shapes) 70 to 190 at these points).

## RuntimeWarnings

pdf emits `underflow encountered in exp` / `divide` / `multiply` (6 of 7
parameter sets) wherever phi(z) underflows. In most of these points the true
pdf is itself subnormal or zero, so the warning is legitimate; in the F1
regime it accompanies the inaccurate results. No warnings from cdf, sf, ppf,
isf.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | bulk, all parameter sets; x near 0.5 and 1 - 2**-53 | 21 ulp (bulk); tails see F1, F2 |
| cdf | full grid incl. high tail (cdf -> 1), all sets | 29.9 ulp bulk, 0 ulp high tail; low tail see F2 |
| sf | full grid incl. low tail (sf -> 1), all sets | 30.7 ulp bulk, 0 ulp low tail; high tail see F2 |
| ppf | q in [1e-300, 1 - 2**-53], all sets | 269 ulp in bulk for (0.5, 0.001), within cond 1.09e3; q -> 1 23.1 ulp |
| isf | q in [1e-300, 1 - 2**-53], all sets | 107 ulp for (0.5, 0.001), within cond; others see F3 |
| sc.logit / sc.expit | spot checks at x = 0.5 + 1e-9, 0.3 ... 0.7, 1e-10, 1 - 1e-10; t = -700 ... 40 | < 0.5 ulp |

## Suspected, not confirmed

none

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logpdf | log(pdf) | `log(b) - log(x) - log1p(-x) + _norm_logpdf(z)` | pdf underflows for \|z\| > ~38: `logpdf(1e-20, 4.317267509914106, 3.1837781130785063)` = -inf, true -10078.515386326121 |
| logcdf | log(cdf) | `sc.log_ndtr(z)` | z < ~-38: `logcdf(1e-20, 4.317267509914106, 3.1837781130785063)` = -inf, true -10130.683151607724 |
| logsf | log(sf) | `sc.log_ndtr(-z)` | z > ~38: `logsf(1 - 1e-12, 4.317267509914106, 3.1837781130785063)` = -inf, true -4264.016350652269 |

(z = a + b*logit(x); demonstration values from `scipy.stats.johnsonsb` vs
mpmath at 50 digits.)

## Shared code

none in the helpers themselves: `_norm_pdf` (`exp(-x**2/2)`) has errors of
up to ~z**2/2 ulp, within the conditioning of norm.pdf with respect to its
own argument, but it contributes to F2 here. The F1 pattern (subnormal
`_norm_pdf` times a large Jacobian factor) may recur in other distributions
that transform a normal variable the same way (e.g. johnsonsu, lognorm);
not checked here.

## Existing GitHub issues

none found (web search; `gh` CLI unavailable)

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(1e-60, 30, 0.5)` | 0.0 | 5.0590778955647029e-273 | verified | loss; z = -39.08, phi(z) underflows to 0 in `_norm_pdf` (line 362-363) before the b/(x(1-x)) factor (line 5873) |
| F1 | `pdf(3.877170285277934e-60, 30, 0.5)` | 3.268565186303012e-262 | 3.2713693236792277e-262 | verified | 3.9e12 ulp (severe); z = -38.40, phi(z) = 2.535e-321 subnormal |
| F1 | `pdf(2.104223637677623e-17, 0, 1)` | 1.204509215552259e-304 | 1.2055425770181894e-304 | verified | 3.9e12 ulp (severe); same subnormal phi(z) |

Reference: pdf as d/dx of Phi(a + b*logit(x)) via `mp.diff`, cross-checked against a log-space closed form (agree to > 130 digits); 100 vs 200 digits agree to > 65 digits. Script: `scratch/johnsonsb/verify/verify_f1.py`. Mechanism matches the cited code. F2 and F3 are minor and were not selected.

Verified by dist-accuracy-verifier: 1 of 1 verified.
