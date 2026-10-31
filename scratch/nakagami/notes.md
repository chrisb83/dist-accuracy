# nakagami hypotheses (a = nu, y = nu*x**2; cdf = P(a, y))

1. logpdf (l.7751-7752) pattern 5: xlogy(nu,nu) - gammaln(nu) + xlogy(2nu-1,x) - nu*x**2,
   terms ~ nu*log(nu) cancel to O(1) near x = 1 for large nu (1e3, 1e6). pdf = exp(logpdf)
   turns it into relative error (pattern 3/5), as chi F5/F6, gamma F2/F4.
2. logpdf pattern 3: x**2 overflows for x > 1.34e154 while -nu*x**2 is finite if
   nu*x**2 < 1.8e308 (small nu: up to x ~ sqrt(1.8e308/nu)), as chi F3.
3. cdf/sf (l.7755, 7761) patterns 4/10: nu*x*x underflows/subnormal for
   x < sqrt(2.2e-308/nu); for small nu cdf ~ (nu x^2)^nu/Gamma(nu+1) still normal. As chi F1.
4. ppf/isf (l.7758, 7764) patterns 4/8: gammaincinv result y underflows/subnormal for
   small nu while x = sqrt(y/nu) is normal. As chi F2.
5. cdf/sf/ppf/isf for nu >~ 2e5: unconverged igam series (gamma F1/F3) in
   y in a - (4.5..15) sqrt(a), i.e. x = sqrt(1 - k/sqrt(nu)).
6. ppf/isf tiny nu: gammaincinv exponent rounding (gamma F5).
7. cdf small x: igam prefactor rounding (gamma F6), minor.
8. ppf/isf: 1.0/nu overflows for nu < 5.6e-309 (subnormal nu) - quantiles are 0 there anyway
   (cdf ~ 1 for any normal x), so likely not measurable.
