# genhalflogistic hypotheses
- H1 (p1/2) cdf: small x, tmp2=(1-cx)^(1/c) ~ 1 - x, `1.0 - tmp2` cancels; 0 for x < ~eps. Fix -expm1(log1p(-cx)/c).
- H2 (p6/9) pdf, cdf: small c, rounding of 1-c*x raised to 1/c -> ~1/c ulp (beyond cond for x ~ 1). cx < eps/2 -> 1-cx == 1, cdf 0, pdf 2/4.
- H3 (p8/9) ppf: small c, 1 - r**c cancels (rel err eps/(c|log r|)); small q: (1-q)/(1+q) ~ 1-2q, loses for q<eps -> 0.
- H4 (p3) pdf at/near upper end for c>1: negative exponent, fine (true value huge).  c<1 tiny tmp underflow of tmp0 -> pdf 0 where true normal? only if true also tiny.
- Opportunities: sf = 2t/(1+t) closed form; isf closed form with t=q/(2-q); logpdf, logcdf, logsf in log space.
