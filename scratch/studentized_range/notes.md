# studentized_range: hypotheses (light review, code reading only)

Sources: _continuous_distns.py 12274-12466 (installed); integrands in
scipy/stats/_stats.pyx (NOT shipped in the wheel; only _stats.pxd signatures
and the cpdef _studentized_range_{cdf,pdf}_logconst are present; the logconst
formulas were confirmed by calling them at (3, 10) against a hand formula).
Integrand bodies below are from the upstream _stats.pyx as read, not from the install.

H1 (pattern 15) cdf/pdf: nquad epsabs=1e-11, epsrel=1e-12 -> no relative accuracy
   once the value is < ~1e-11: cdf left tail (small x), pdf both tails.
H2 (pattern 1 + 6) cdf/pdf integrand: D = Phi(z+q s) - Phi(z), both via
   0.5*erfc; for small q s, D ~ q s phi(z) cancels (rel err ~ eps*Phi(z)/(q s phi(z)))
   and is raised to k-1 (cdf) / k-2 (pdf). D == 0 when q s < ~eps|z|.
H3 (pattern 3) pdf for 1 < k < 2: pow(D, k-2) with D == 0 (z > ~8.3 where both
   Phi round to 1, or z < -38 where both underflow) -> inf; times exp(...) -> inf,
   or 0*inf = nan when exp(log_terms) underflows. Expect pdf inf/nan for all x.
H4 (pattern 7) df >= 100000 switches to infinite-df integrand: approximation
   error O(1/df) ~ 1e-5..1e-6 at df = 1e5 -> jump across the threshold.
H5 (pattern 5) large df < 1e5: logconst (df/2)log df - lgamma(df/2) and integrand
   (df-1)log s - df s^2/2 are ~df-sized terms that cancel -> ~df*eps relative error
   (1e5*eps = 2e-11 > epsrel).
H6 (pattern 15) large df: chi density in s has width ~1/sqrt(2 df); qagi over
   (0, inf) may miss/under-resolve the peak.
H7 (pattern 1) inherited sf = 1 - cdf: cdf clipped to [0,1] and only ~1e-11
   absolute -> sf useless below ~1e-11, exactly 0 below eps.
H8 cdf clip to [0,1] hides overshoot (no accuracy issue per se).
H9 (pattern 6) large k: pow(D, k-1) amplifies D's rounding by k.
H10 tiny df: s^(df-1) singular at s=0 and very heavy right tail in s: quad difficulty.

## Measurement pass (2026-10-31)

Reference: refs/studentized_range_ref.py now has (a) exact k = 2 path (t / betainc),
used for full grids at --dps 100; (b) general-k nested trapezoid quadrature
(z and u = log s, h and h/2 consistency check), used via --at --dps 30.
Validation: general path vs exact k=2: 2e-30 (cdf, pdf); vs SciPy bulk k=3, k=1.5 cdf: ~1e-15.
cond via mp.diff fails on the general path (reported "-"), tolerance then 8 ulp.

S1 confirmed: pdf(1, 1.5, 10) = nan (ref 0.29595); nan for k in {1.01,1.5,1.9,1.999},
x in {0.1,1,5}, df in {3, 1e6} (both integrands). RuntimeWarnings divide/invalid/underflow.
Runs: run_a (df 10, 1, 0.1), run_b (df 1e3, 99999, 1e5, 1e7), at_batch1 (general k).
Outcome: F1 (S1) loss; F2 large-x s~1/x region missed (incl. S8) loss; F3 (S2+S3) small-x cdf loss;
F4 (S2) pdf absolute-only accuracy loss; F5 (S4) df>=1e5 switch loss; F6 (S6) severe; F7 (S5) moderate.
S7: k=100 minor only; k=1000 reference not converged (outer u step). S9 unresolved. isolate_s3.py separates S3 from S2.
