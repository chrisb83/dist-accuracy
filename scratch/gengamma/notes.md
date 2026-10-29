# gengamma hypotheses (SciPy 1.18.1, lines 3852-3951)

H1 (p3/4/10) cdf, c>0: gammainc(a, x**c) with x**c underflowing to 0 or subnormal
   while P(a, y) ~ y**a/Gamma(a+1) is normal (small a). cdf -> 0. Same for sf, c<0.
H2 (p3/8) ppf/isf: gammaincinv(a,q) (or gammainccinv) underflows to 0 for small a
   while result**(1/c) is normal (large |c|). c>0 -> 0, c<0 -> 0**(neg)=inf.
H3 (p5) logpdf/pdf: xlogy(c*a-1, x) - x**c - gammaln(a) cancel for large a (gamma F2/F4).
H4 (p5/11) logpdf: c*a - 1 rounds; with c*a ~ 1 (halfgennorm a=1/beta) the
   rounding error is amplified by |log x|.
H5 (p10) cdf/sf: rounding of x**c amplified relative to cond(arg) by 1/|c| (|c|<1).
H6 ppf/isf: rounding of 1.0/c amplified by |log result| (weibull_min F3), minor.
H7 shared: gamma report F1/F5/F6 (igam series cap, igami small a, prefactor).
H8 inherited logcdf/logsf: underflow in tails; log-space forms exist.
