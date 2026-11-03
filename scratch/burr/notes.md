# burr hypotheses
- P1 sf/logsf (l.1225,1228): log1p(-(1+x**-c)**-d) cancels, right tail d*x**-c < eps.
- P3 all x>0 branches: x**-c overflows for c*log(x) < -709.8 -> pdf nan, logpdf/logcdf -inf, cdf 0, sf 1.
- P5 logpdf (l.1213-1215): xlogy(-c-1,x) - xlog1py(d+1,x**-c) cancel for x<1, c*d ~ 1.
- P8 ppf (l.1231): q**(-1/d)-1 cancels q->1 / large d; overflows small q / small d.
- P8 isf (l.1234): expm1(-log1p(-q)/d) overflows small d.
- P4 pdf: x**(-c-1) subnormal for large c, compensated by large c*d.
- P6 pdf/cdf: (1+u)**d with large d.
- P9 small c (0.01): x**-c ~ 1 everywhere, check bulk.
