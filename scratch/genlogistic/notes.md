# genlogistic hypotheses

1. logpdf/pdf, x<0, small c: `mult = -(c-1)*(x<0) - 1` rounds (c-1); mult = -c with abs error ~eps,
   relative error eps/c (pattern 5/9). c < 1.1e-16 -> mult = 0 exactly. Fails where c*|x| dominates logpdf.
2. logpdf, x ~ log(c) with large c: log(c) - x cancels (pattern 5), mild (~log c ulp).
3. cdf: (1+exp(-x))**(-c): exp(-x) overflows for x < -709.78 -> 0 (pattern 3); true ~exp(c*x) normal for small c.
4. cdf: large c, rounding of 1+exp(-x) raised to power c (pattern 6).
5. logcdf: log1p(exp(-x)) overflows x < -709.78 -> -inf; true ~ c*x (pattern 3).
6. sf: -expm1(logcdf): inherits 5 -> sf = 1 where true sf = 1 - exp(c*x) < 1 (small c).
7. ppf: powm1(q, -1/c) overflows for small c / small q (log(q)/c < -709) -> -inf (pattern 8).
8. isf = ppf(1-q): 1-q rounds for q < 1e-16 -> inf; digits lost for small q (pattern 8).
9. pdf = exp(logpdf): absolute error in large logpdf (e.g. log(c) for c=1e-300) amplified.
