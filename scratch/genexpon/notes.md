# genexpon hypotheses
E(x) = -(a+b)x + b(1-exp(-cx))/c  (<= 0); sf = exp(E), cdf = -expm1(E)

H1 (pattern 5): cdf line 3336, small x: -(a+b)x and b*(-expm1(-cx))/c cancel to -ax - bc x^2/2.
   relative error of E ~ eps*(a+b)/(a + bcx/2). cdf ~ -E, so cdf loses (a+b)/a ulp when a << b, cx small.
   Also logpdf near 0 sees E but absolute error only -> fine. pdf/sf: exp(E) ~1 -> fine.
H2 (pattern 5/8): ppf line 3340-3341, small p: t -> b/s, W -> -b/s, t + W cancels to ~ c p / a.
   Catastrophic for any shapes when p << 1. Also isf with q -> 1 (log(q) ~ 0).
H3 (pattern 10): lambertw near branch point -1/e when b/s -> 1 (a << b) and t near 1:
   argument -b/s exp(-t) near -1/e; W ill-conditioned, plus t+W cancellation.
H4 (pattern 4): ppf/isf: exp(-t) underflows when t > 745 (isf tiny q, c/s large). Then W(0)=0, x=t/c;
   correct limit (W ~ -b/s e^{-t} negligible) -> expect fine.
H5: logpdf for large x: -(a+b)x huge, fine. logpdf cancellation log(a+..) + E near zero -> conditioning.
H6 inherited logsf = log(sf): sf underflows for x > ~745/(a+b); logsf closed form = E.
H7 inherited logcdf = log(cdf): fine except near cdf->1; better log(-expm1(E)) = log1mexp(E).
