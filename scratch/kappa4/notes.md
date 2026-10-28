# kappa4 hypotheses (from code reading, lines 7397-7501)

Notation: y = (1 - k x)^(1/k) for k != 0, y = exp(-x) for k = 0.
cdf = (1 - h y)^(1/h) for h != 0, exp(-y) for h = 0.

H1 (pattern 6, amplified rounding). `(1.0 - k*x)**(1.0/k)` appears literally in
   `_logpdf.f0/f1` (7420, 7427) and `_logcdf.f0` (7453). fl(1 - k x) carries a
   relative error ~eps/2 which the exponent 1/k multiplies: relerr(y) ~ eps/(2|k|).
   Regime: |k| <= 1e-4, any h. Expected ~0.5/|k| ulp.
   Fix: exp(log1p(-k*x)/k).
   PROBE: cdf(1, 0.5, 1e-8) -> 0.6659543826 vs 0.6659543811 (1e7 ulp). CONFIRMED.

H2 (pattern 1/9, cancellation in the ppf). `_ppf.f0` (7487) `(1.0 - q**h)/h`
   and `(1.0 - y**k)/k`: q**h -> 1 as h -> 0 and y**k -> 1 as k -> 0, so the
   numerator cancels; relerr ~ eps/(2|h log q|) resp. eps/(2|k log y|).
   Regime: |h| <= 1e-4 or |k| <= 1e-4.
   Fix: -expm1(h*log q)/h, -expm1(k*log y)/k.
   PROBE: ppf(0.3, 1e-8, 0.5) -> -0.19451388625 vs -0.19451388428 (4.5e7 ulp);
   ppf(0.3, 1e-14, 0.5) 2% wrong. CONFIRMED.

H3 (pattern 3, overflow). `_logcdf.f2` (7465) `-h*np.exp(-x)` and `_logpdf.f2`
   (7433): for h < 0 the support is all of R, exp(-x) overflows for x < -709.8,
   log1p(inf) = inf, logcdf = -inf, cdf = 0 while the true cdf
   ~ (|h| e^-x)^(1/h) = exp((log|h| - x)/h) is a normal double whenever
   |x/h| <~ 700.  Regime: h <~ -1e3, x < -709.
   PROBE: cdf(-1e8, -1e6, 0) -> 0.0, true 3.72e-44. CONFIRMED (loss).

H4 (pattern 3, overflow). Same in `_logcdf.f0` (7453)/`_logpdf.f0` (7420):
   (1 - k x)^(1/k) overflows for k > 0, x -> -inf; with h < 0 large the true
   cdf is still O(1).  PROBE: cdf(-1e35, -1000, 0.1) -> 0.0, true 0.4539.
   CONFIRMED (loss, worst case: no digits where the answer is ~1/2).

H5 (pattern 7 + domain error). `_ppf.f2` (7493)
   `-sc.log1p(-(q**h)) + np.log(h)` is used for k == 0, h != 0, i.e. also for
   h < 0.  Then q**h > 1 so log1p(-(q**h)) is log of a negative number and
   np.log(h) is log of a negative number: the branch returns nan for every q.
   Includes the logistic case (h = -1, k = 0). PROBE: ppf(0.5, -1, 0) = nan
   (true 0), ppf(0.5, -0.5, 0) = nan, isf likewise. CONFIRMED (loss).

H6 (pattern 1). `_logcdf.f0` (7453) `log1p(-h*(1-k x)^(1/k))` with h > 0 near
   the lower support endpoint a: h y -> 1, so 1 - h y cancels. Left tail for
   h > 0. Expect large errors but also large cond(x) (cdf ~ C (x-a)^(1/h)).

H7 (pattern 1, inherited). sf = 1 - cdf, logsf = log(sf), isf = ppf(1-q):
   right-tail collapse for 1 - F < eps, and isf loses small q.

H8 (pattern 8). `_ppf.f1` (7490) `(-np.log(q))**k` uses log(q) rather than
   log1p(q-1) for q -> 1; and `1 - (-log q)**k` cancels near q = exp(-1).

H9 pdf = exp(logpdf), cdf = exp(logcdf) (7397, 7440): the absolute error of the
   log (~eps*|log F|) becomes the relative error of the result, so up to
   ~700 eps = 3e3 ulp in the far tails. Generic amplification.
