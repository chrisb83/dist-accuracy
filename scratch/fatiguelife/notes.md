# fatiguelife hypotheses

1. logpdf/pdf, pattern 3: `(x-1)**2` overflows for x > 1.34e154 -> term inf -> logpdf -inf (true finite, ~ -x/(2c^2)). Large x, any c. pdf true value underflows there, so only logpdf.
2. logpdf, pattern 4: `c**2` subnormal for c < 1.5e-154, zero for c < ~1e-162 -> nan at x == 1 (0/0), large error near x = 1 with c ~ 1e-155.
3. pdf = exp(logpdf), pattern 5: sum of large logs (log(2c), 1.5 log x, log(x+1)) with small result -> error ~ max|term| ulp in pdf; large c with tiny/huge x.
4. ppf (tmp < 0, i.e. q < 0.5) and isf (q > 0.5), pattern 8: `tmp + sqrt(tmp**2 + 4)` cancels for large |tmp| = c*|ndtri(q)|; total loss (0.0) for |tmp| > ~1e8; tmp**2 overflow for |tmp| > 1.3e154 -> inf.
5. cdf/sf near x = 1: sqrt(x) - 1/sqrt(x) cancels, but cond wrt x ~ 1/|x-1| covers it; check small c.
6. inherited logcdf/logsf: ndtr underflows at z < -38 -> log_ndtr opportunity.
