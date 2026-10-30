# johnsonsb hypotheses

z = a + b*logit(x) everywhere.

1. pdf (5870-5873): _norm_pdf = exp(-z**2/2)/C. Rounding of z (and of z**2) is
   amplified by z**2/2 (up to ~740 ulp at |z| = 38.5). Regime: |z| large, i.e.
   large |a| in the bulk or tails. Cond wrt x ~ |z|*b/(1-x); excess when b small
   relative to |z|. Pattern 5/6-like (rounding in exp argument). Expect minor.
2. pdf: b*1.0/(x*(1-x)): x*(1-x) fine; large prefactor b/x with tiny b,x - fine.
3. cdf/sf: ndtr(z)/ndtr(-z): complementary form used; fine.
4. logit near x = 0.5: absolute error ~eps, conditioned (logit -> 0). For
   cdf, z = a + b*logit with a = -b*logit(x0) cancellation is conditioned too.
5. ppf/isf: expit((ndtri(q) - a)/b); isf uses -ndtri(q) (complement correct).
   expit near 1: result 1 - tiny rounds - only relative error of x, fine.
   Cancellation ndtri(q) - a: absolute error eps*|a|/b in t; relative error in
   x ~ eps*|a|/b*(1-x). Large |a|/b with t near 0? t = (z-a)/b, |a|/b large
   and x not near 0/1 requires z ~ a -> q extreme, conditioned. Check with
   (-50, 50) and (30, 0.5).
6. ndtri accuracy for q near 1 (1 - 2**-53): ppf ill-conditioned there.
7. Inherited logpdf/logcdf/logsf: log of underflowing pdf/cdf/sf for |z| > 38.
   Closed forms: logpdf = log b - log x - log1p(-x) + _norm_logpdf(z);
   logcdf = log_ndtr(z); logsf = log_ndtr(-z).
