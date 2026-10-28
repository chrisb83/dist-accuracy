# burr12 hypotheses (code lines 1320-1346)

1. logpdf/pdf (1325, 1322): x**c overflows for c*log(x) > 709.8 -> xlog1py(-d-1, inf) = -inf, pdf 0. Pattern 3. Loss where true pdf normal (small d / large c).
2. logsf/sf (1337, 1334): same overflow -> logsf -inf, sf 0 where true sf = x**(-c*d) normal (small c*d). Pattern 3.
3. logcdf (1331): log1p(-(1 + x**c)**(-d)): left tail, 1 + x**c rounds to 1 -> -inf (pattern 1/2); cancellation before. Right tail: (1+x^c)^(-d) amplifies rounding by d (pattern 6); x**c overflow -> 0 where true -x^(-cd) normal (pattern 3).
4. cdf (1328): -expm1(xlog1py(-d, x**c)): x**c subnormal with large d -> cdf normal but inaccurate (pattern 4).
5. logpdf: (c-1)log x and -(d+1)c log x cancel partially for large c, small d -> ~c/(cd+1) ulp (pattern 5).
6. pdf = exp(logpdf), sf = exp(logsf): error ~|log| ulp vs cond ~ c*d -> minor.
7. ppf/isf (1343, 1346): expm1(z) overflows for z = -log(1-q)/d or -log(p)/d > 709.8 (small d) -> inf where true quantile exp(z/c) finite (pattern 8). expm1(z) subnormal for z tiny (huge d, tiny q) then **(1/c) (pattern 4).
