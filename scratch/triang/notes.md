# triang hypotheses
1. cdf, x >= c branch, line 10121: (x*x - 2*x + c)/(c-1). Numerator = (c-1) + (1-x)^2 formed
   from terms of size ~1-2: absolute error ~eps, relative eps/(1-c). c near 1 -> big error
   for x in [c,1] (pattern 5). Also c small, x just above c? numerator ~ c - 2x, mild.
2. cdf, x < c branch, line 10120: x*x/c. x*x underflows/subnormal for x < 1.5e-154 while
   x^2/c is normal if c tiny (pattern 4). Same in c==1 branch but there result underflows too.
3. ppf, line 10127: sqrt(c*q) - c*q underflow for tiny c,q (pattern 4).
4. ppf else-branch 1 - sqrt((1-c)(1-q)) for small c and small q >= c: cancellation (pattern 1/8);
   c = 0 gives 0 for q < eps. Rel. error ~ eps/ppf.
5. pdf: 2*(1-x)/(1-c) - 1-x exact for x>=0.5; fine expected. 2x/c fine.
Inherited: sf = 1-cdf (right tail, pattern 1/12), logsf, logcdf, logpdf, isf = ppf(1-q).
