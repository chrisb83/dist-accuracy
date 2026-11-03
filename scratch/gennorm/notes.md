# gennorm hypotheses (a = 1/beta, y = |x|**beta)

1. cdf/sf, pattern 4: `abs(x)**beta` underflows to 0 for |x| < exp(-745/beta)
   while P(a, y) ~ |x| * (beta/(2 Gamma(1/beta))) is normal; cdf returns 0.5
   exactly. Visible for beta >~ 20. Probe: cdf(0.4, 1000) = 0.5 (true ~0.7);
   cdf(-0.9, 1000) = 0.0497 (?), check.
2. ppf/isf, pattern 4/8: gammainccinv(a, 2-2q) = P^-1(a, 2q-1) ~ (2d Gamma(a+1))**beta
   underflows for large beta, then 0**(1/beta) = 0. Probe ppf(0.6, 1000) = 0.
3. logpdf, pattern 5: log(0.5*beta) - gammaln(1/beta) cancels for huge beta
   (both ~ log beta), up to ~log(beta)/log(2) ulp (1e3 at beta = 1e300).
   pdf = exp(logpdf) inherits it.
4. pdf, small beta: exp(logpdf) with |logpdf| ~ gammaln(1/beta) up to ~700
   amplifies the absolute error of the large constant (~|logpdf| ulp).
5. ppf, small beta, pattern 6: z**(1/beta) amplifies z's rounding by 1/beta
   (q near 1/2 where cond is ~1). beta >= ~0.006 needed for representable x.
6. cdf, small beta, left tail: rounding of y = |x|**beta amplified by y in Q(a,y),
   cond_x = beta*y, so excess ~ a/2 ulp.
7. gammaincc/gammainccinv shared issues (INDEX 7): large-a regime unreachable
   because y <= exp(709.78*beta) << a - 5 sqrt(a) for a >= ~100. Small a
   gammaincinv amplification (gamma F5) attenuated by **(1/beta)=**a.
