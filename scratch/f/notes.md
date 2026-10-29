# f hypotheses

1. logpdf/pdf (line 2599-2600), pattern 5: m/2*log(m) - (n+m)/2*log(m+n*x) - betaln(n/2, m/2)
   cancel terms of size ~m*log(m) (and n*log n) to an O(1) result. Regime: large dfd and/or dfn,
   x in the bulk. Expected loss ~ m*log(m)/|logpdf| ulp in logpdf, ~ m*log(m) ulp in pdf.
2. logpdf, pattern 3: m + n*x overflows for n*x > 1.8e308 (x ~ 1e300, n >= 1e9; or x near 1e308).
   Also log(m + n*x) when n*x << m: log(m+nx) rounds nx/m away -> loses the x-dependence
   (pattern 2, small n*x/m): combined with cancellation vs m/2 log m.
3. pdf = exp(logpdf): error amplified by |logpdf| (pdf near underflow, |logpdf| ~ 700 -> ~700 ulp,
   inherent conditioning though).
4. cdf/sf/ppf: sc.fdtr/fdtrc/fdtri (xsf/cephes or Boost). fdtr computes w = n x/(m + n x)
   then incbet(n/2, m/2, w): for large x, w near 1 rounds (but cdf near 1, fine). fdtrc
   uses m/(m + n x). Possible loss if implementation uses 1 - w. Test far tails, large shapes.
5. ppf: fdtri historically y = 1 - q (pattern 8) -> loss for small q; quick probe shows
   fdtri(29,18,1e-20) is finite, so test q grid and large/small shapes.
6. isf inherited = ppf(1 - q): fails for q < eps (opportunity, pattern 12).
7. logcdf/logsf inherited.
