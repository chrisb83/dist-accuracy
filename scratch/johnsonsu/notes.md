# johnsonsu hypotheses

z = a + b*arcsinh(x).

1. pdf, pattern 3: `x2 = x*x` overflows for |x| > 1.34e154 -> sqrt(inf) = inf -> pdf 0.
   True pdf ~ b/|x| * phi(a + b*log(2|x|)) is normal when b*log(2x) is small (small b).
   Test: (0, 0.01), (0.5, 1e-3), x = 1e200, 1e300, -1e300.
2. pdf, pattern 4 (as johnsonsb F1): phi(z) subnormal/0 for |z| > 37.5, multiplied by
   b/sqrt(1+x^2). Prefactor <= b, so only matters for large b (b > 1). Test (0, 1e6), (-50, 50).
3. pdf/cdf/sf: rounding of z amplified ~z^2 in normal tails (johnsonsb F2). Test |a| = 30.
4. ppf/isf: (ndtri(q) - a) cancels when ndtri(q) ~ a (x ~ 0); errors of ndtri ~ eps|a|.
   Test q = ndtr(-a) neighbourhood with |a| large.
5. ppf/isf: sinh amplifies absolute error of its argument t by |t| (relative) for large |t|
   (small b). Test (0.5, 1e-3), (0, 0.01).
