# foldcauchy hypotheses
- H1 (pattern 5) cdf line 2529: arctan(x-c)+arctan(x+c) cancels for x << max(1,c), c > 0:
  terms ~ +-atan(c), result ~ 2x/(pi(1+c^2)). Loss ~ atan(c)(1+c^2)/x ulp; total loss (0.0) once 2x/(1+c^2) < eps*atan(c).
  Fix: atan2(2x, 1 + c^2 - x^2)/pi (exact identity: arg((1+i(x-c))(1+i(x+c)))).
- H2 cdf near 1 (large x): cdf = 1 - O(1/x), fine (value near 1, abs error ok).
- H3 sf line 2536: atan2(1, x-c) + atan2(1, x+c): no cancellation (both positive). Expect fine.
- H4 pdf: (x+-c)**2 overflow for |x+-c| > 1.3e154; true value then subnormal -> not measurable. 1+(x-c)^2 fine.
- H5 c=0 and tiny c: no branch, expect fine.
- Inherited: logpdf (pdf underflow x>~1e154), logcdf (cdf underflow / H1), logsf, ppf (closed form via tan), isf.
