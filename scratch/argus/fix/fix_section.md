
## Fix

- **Branch:** `argus` in scipy, based on main da43ab1957 (not committed)
- **Check script:** `dist-accuracy/fixes/argus.py`
- **Fixed:** F1 (where b = chi**2 (1 - x**2)/2 >= 1), F2, F3, F4, F5, F6 (chi <= sqrt(2)), F7 (tiny x)
- **Not fixed:**
  - F1 for small x (x below ~0.1-0.3, depending on chi): this needs a series (e.g. a mixture of
    Beta(k+1, 3/2) cdfs); decided with the user to keep the change to closed forms.
    `cdf(1e-10, 1)` is still 0.
  - F7 for large chi: the remaining error comes from rounding chi**2 (1 - x**2)/2 inside exp,
    i.e. conditioning with respect to chi; fixing it would need extended precision.
  - F6 for chi > sqrt(2) (sc.gammainc(1.5, z) for small z, shared code): unchanged, errors there
    are at most ~28 ulp.
  - Opportunities: ppf/isf (closed form via gammaincinv/gammainccinv): left for a follow-up PR.
    logcdf: skipped, `log(cdf)` is accurate where the cdf does not underflow. logsf: needs no
    custom method any more, since the new sf does not underflow (sf >= ~3e-24 for x < 1).
  - `argus._stats` still uses `_argus_phi(chi)` and fails for chi < ~1e-102 (outside this review).

### Pull request summary

**ENH: stats.argus: improve accuracy of pdf, logpdf, cdf and sf for small and large chi**

`argus` lost accuracy in several regimes: for chi < ~1e-102 the normalizing constant
`_argus_phi(chi) ~ chi**3` underflowed, so pdf/logpdf returned inf and cdf/sf nan; for small chi,
logpdf lost up to 1.6e4 ulp through the cancellation of `3*log(chi)` and `log(_argus_phi(chi))`, and
sf returned 0 close to x = 1; the cdf was computed as `1 - sf` and returned 0 (or lost all digits)
for large chi, e.g. `cdf(0.95, 100)` = 0 instead of 4.8e-211; and logpdf returned -inf for
chi > 1.3e154 because `chi**2` overflowed.

Changes in `argus_gen`:

- new `_log_norm(chi)`: log of the normalizing constant. For chi <= sqrt(2) it uses
  `log(3) + a - log(hyp1f1(1, 2.5, a))`, a = chi**2/2, which follows from the Kummer
  representation of the incomplete gamma function and does not form chi**3. For larger chi the
  previous expression is kept.
- `_logpdf`: uses `_log_norm`; `chi**2*y/2` is evaluated as `(0.5*chi)*(chi*y)` to avoid overflow.
- `_pdf`: `x*sqrt(1 - x**2)*exp(_log_norm(chi) - a*(1 - x**2))` instead of `exp(_logpdf)`, so
  log(x) no longer passes through exp (relative error ~|logpdf| ulp for tiny x before).
- `_sf`: for chi <= sqrt(2), `y**1.5 * exp(a*x**2) * hyp1f1(1, 2.5, a*y) / hyp1f1(1, 2.5, a)`
  with y = (1 - x)(1 + x); unchanged for larger chi.
- `_cdf`: for b = a*(1 - x**2) >= 1, `(gammaincc(1.5, b) - gammaincc(1.5, a)) / gammainc(1.5, a)`;
  otherwise `1 - sf` as before.
- spurious overflow warnings for chi > 1.3e154 are suppressed (the results were and are correct).

Tests: `TestArgus::test_accuracy` with 29 cases for pdf, logpdf, cdf and sf against mpmath
references; 21 of them fail on main.

Not addressed: the cdf for small x (still `1 - sf`), and the generic ppf/isf (follow-up).

### Derivations

Notation as in the code: `a = chi**2/2`, `y = 1 - x**2 = (1 - x)(1 + x)`, `b = a*y`,
`P(t) = gammainc(1.5, t)`, `Q(t) = gammaincc(1.5, t) = 1 - P(t)`, `M(z) = hyp1f1(1, 2.5, z)`.
The docstring pdf is `K x sqrt(y) exp(-b)` with `K = chi**3/(sqrt(2 pi) Psi(chi))`.

**Pass 1: algebra**

1. **Psi via P** (existing, docstring of `_argus_phi`): `Psi(chi) = P(a)/2`.
2. **Kummer form of P** (DLMF 8.5.1 with `M(1, 1 + s, z)`):
   `P(s, z) = z**s exp(-z) M(1, s + 1, z) / Gamma(s + 1)`; for s = 3/2:
   `P(z) = z**1.5 exp(-z) M(z) / Gamma(2.5)`, with `Gamma(2.5) = 3 sqrt(pi)/4`.
   Numerical check: `mp.gammainc(1.5, 0, z, regularized=True) - z**1.5*mp.exp(-z)*mp.hyp1f1(1, 2.5, z)/mp.gamma(2.5)`
   is ~1e-32 at 30 digits for z = 0.7.
3. **Normalizing constant** (`_log_norm`, small branch). Insert 1 and 2 with
   `a**1.5 = chi**3 / 2**1.5`:
   `K = 2 chi**3 / (sqrt(2 pi) P(a)) = 2 chi**3 * 2**1.5 * (3 sqrt(pi)/4) / (sqrt(2 pi) chi**3 exp(-a) M(a)) = 3 exp(a) / M(a)`.
   So `log K = log(3) + a - log(M(a))`. The factor chi**3 cancels exactly. Limit chi -> 0:
   K -> 3, pdf -> 3 x sqrt(1 - x**2), the chi = 0 limit already used in `test_rvs_small_chi`.
   The large branch `3 log(chi) - log(sqrt(2 pi)) - log(P(a)/2)` is `log K` written directly
   (`_norm_pdf_logC = log(sqrt(2 pi))`).
4. **logpdf / pdf.** `logpdf = log K + log(x) + 0.5 log1p(-x**2) - b` and
   `pdf = x sqrt(y) exp(log K - b)`; `b = a*y = (0.5*chi)*(chi*y)` is only a reordering.
5. **sf, small branch.** `sf = P(b)/P(a)` (existing). With 2:
   `P(b)/P(a) = (b/a)**1.5 exp(a - b) M(b)/M(a) = y**1.5 exp(a x**2) M(b)/M(a)`,
   since `b/a = y` and `a - b = a x**2`.
6. **cdf.** `cdf = 1 - P(b)/P(a) = (P(a) - P(b))/P(a) = (Q(b) - Q(a))/P(a)` (using `P = 1 - Q`).
7. **Switch-overs.** Both branches of `_log_norm` and `_sf` are the same function (3 and 5), so
   the switch at chi = sqrt(2) only changes rounding. Same for the cdf at b = 1 (6).

**Pass 2: numerics**

Error of the cdf (item 6). Let `D = P(a) - P(b) = Q(b) - Q(a)` (> 0). `1 - sf` rounds sf with
relative error ~eps, i.e. absolute error ~eps P(b)/P(a), so the relative error of the cdf is
~eps P(b)/D. The gamma difference has absolute error ~eps Q(b) (Q(a) <= Q(b)), i.e. relative
error ~eps Q(b)/D. The second form is better iff Q(b) < P(b), i.e. b > median of Gamma(3/2) =
1.18299. Switching at b = 1 costs at most a factor Q(1)/P(1) = 0.5724/0.4276 = 1.34 against
the optimal choice. For small x, b -> a and D -> 0 in both forms: this is the unfixed part of F1.

| Where | Trick | Avoids | Switch-over |
|---|---|---|---|
| `_log_norm` | `log(3) + a - log(M(a))` | underflow of `_argus_phi(chi) ~ chi**3` (F2), cancellation of `3 log(chi)` and `log(_argus_phi)` (F5) | chi <= sqrt(2) (a <= 1); M(a) grows like exp(a), and for a > 1 the old form has no cancellation |
| `_logpdf`, `_pdf` | `(0.5*chi)*(chi*y)` | overflow of `chi**2` for chi > 1.3e154 (F4) | - |
| `_pdf` | `x*sqrt(y)*exp(log K - b)` | relative error ~\|logpdf\| eps from `exp(logpdf)` for tiny x (F7) | - |
| `_sf` | `y**1.5 exp(a x**2) M(b)/M(a)` | underflow of `P(b) ~ b**1.5` near x = 1 (F3), of `P(a)` (F2), and the error of `gammainc(1.5, z)` for tiny z (F6) | chi <= sqrt(2) |
| `_sf` | `y = (1 - x)(1 + x)` | rounding of `1 - x*x` near x = 1 | - |
| `_cdf` | `(Q(b) - Q(a))/P(a)` | cancellation in `1 - sf` when the cdf is small (F1) | b >= 1 (see above) |
| all | `np.errstate(over='ignore')` | spurious warnings when chi**2 overflows; `gammainc(1.5, inf) = 1` and `gammaincc(1.5, inf) = 0` give the correct limits | - |

`hyp1f1` (Boost-based in scipy) was checked against mpmath for z in [0, 1]: `_log_norm` is within
1.2 ulp and the new sf within 2.9 ulp for chi in [1e-300, sqrt(2)], x up to 1 - 2**-53.

### Verification

- `python fixes/argus.py <site-packages>`: 0 of 29 cases outside the test tolerance, 0 stale
  values. The references use mpmath's incomplete gamma integrals over [0, b], [b, a] and [0, a]
  directly (no cancellation), with Psi via P(a) checked against Phi - chi phi - 1/2.
- New tests on main (source stashed, tests kept): 21 of 29 fail, at least one per fixed finding;
  the 8 that pass are the cases on unchanged code paths and at the switch-over points.
- `pixi run test`: `TestArgus` 89 passed; `test_distributions.py` 2577 passed;
  `test_continuous_basic.py -k argus` 13 passed. `pixi run lint`: all checks passed.
- Sweep against `refs/argus_ref.py` (chi from 1e-300 to 1e155, 21 x values up to 1 - 2**-53):
  pdf, logpdf and sf within 8 + cond ulp everywhere (largest ratio 3.3 for pdf at chi = 10,
  the unfixed F7 part); cdf 0.1-40 ulp where b >= 1 or x >= ~0.3 (`cdf(0.5, 10)`: 16 ulp, was
  3.6e14; `cdf(0.95, 100)`: 94 ulp at cond 9e3, was 0); for x <= 0.1 and moderate chi the cdf
  still has up to ~5e4 ulp (unfixed part of F1).
- Switch-overs: cdf monotone on 20001 points across b = 1 for chi in {1.4143, 1.5, 2, 3, 10,
  100}, |cdf + sf - 1| <= 1.3e-15; relative steps across chi = sqrt(2) at x = 0.5: logpdf
  1.7e-14 (absolute 1.3e-15, i.e. a few ulp of log K ~ 1.1), sf 1.3e-15, cdf 3.9e-15.
- No RuntimeWarnings for chi = 1e155 or 1e-300 (checked with `-W error`).
- Scratch scripts: `scratch/argus/fix/` (sweep.py, focus.py, switch.py, prototypes).
