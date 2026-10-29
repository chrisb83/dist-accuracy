# exponweib hypotheses (t = x**c)
H1 sf (l.2349) -_pow1pm1(-exp(-t), a) = -expm1(a*log1p(-exp(-t))): 1 - exp(-t) cancels for small t
   (pattern 1/2); t < eps/2 -> log1p(-1) = -inf -> sf = 1. Visible when cdf = t**a not << eps, i.e. small a, small x.
H2 sf: exp(-t) subnormal for t in [708, 745] -> large a lifts sf = a*exp(-t) back into normal (pattern 4).
H3 isf (l.2352) -log(-_pow1pm1(-p,1/a)): -log(1 - (1-p)**(1/a)) cancels when (1-p)**(1/a) small
   (small a, or p near 1), i.e. x**c < eps -> log(1)=0 -> isf 0 (pattern 2).
H4 ppf (l.2346) -log1p(-q**(1/a)): q**(1/a) near 1 for large a -> 1 - q**(1/a) cancels (pattern 1/8);
   q**(1/a) underflows for small a/small q -> 0 although (q**(1/a))**(1/c) normal (large c) (pattern 3/8).
H5 logpdf/pdf (l.2335-2338): x**c underflows (c*log x < -745) -> exm1c = 0 -> xlogy(a-1, 0) = +-inf (pattern 4);
   subnormal x**c loses digits.
H6 cdf (l.2343) (-expm1(-t))**a: t underflow -> 0 where t**a normal (a<1) (pattern 4); large a -> a ulp (pattern 6).
H7 logpdf large a: xlogy(a-1, exm1c) error a*eps abs -> pdf ~a ulp (pattern 6).
