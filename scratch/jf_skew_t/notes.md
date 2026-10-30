# jf_skew_t hypotheses
t = x/sqrt(a+b+x^2), y = (1+t)/2

H1 pdf, pattern 3: c = 2**(a+b-1)*beta(a,b)*sqrt(a+b): 2**(a+b-1) overflows for a+b > 1025, beta underflows -> inf*0 = nan (large a+b, e.g. (600,600)).
H2 pdf, pattern 1: 1 - t (x >> 0) and 1 + t (x << 0) cancel; error ~ (b+1/2)*x^2/(a+b) ulp, 0 once (a+b)/(2x^2) < eps/2 (|x| > ~1e8 sqrt(a+b)).
H3 pdf/cdf/sf, pattern 3: x**2 overflows for |x| > 1.34e154 -> t = 0 -> pdf = 1/C, cdf = sf = I_{1/2}.
H4 cdf left tail (x << 0): y = (1+t)/2 cancels (pattern 1/10), cdf -> betainc(a,b,0)=0. 
H5 sf right tail (x >> 0): betaincc(a,b,y) with y near 1 rounded (pattern 10).
H6 ppf q -> 1: 1 - d1 with d1 = beta.ppf(q) near 1 (pattern 1/8); beyond cond when b < 1. Also beta.ppf bugs (reports/beta.md F1, F6, F7) for small q.
H7 ppf near median: 2*d1 - 1 cancels; probably within conditioning.
H8 pdf large shapes, pattern 6: (1+t)**(a+1/2) amplifies rounding by ~a.
