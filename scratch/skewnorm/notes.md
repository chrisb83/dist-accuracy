# skewnorm hypotheses

1. cdf (pattern 10/5): Boost cdf = Phi(x) - 2 T(x, a); for a > 0, x < 0 it cancels. Code replaces only
   cdf < 1e-6 by generic quad (super()._cdf). Between 1e-6 and ~0.01 with large a, Phi(x)/cdf can be large
   -> error ~ Phi(x)/cdf ulp. Test a = 10, 100, 1e3 near cdf ~ 1e-5..1e-2.
2. cdf fallback (pattern 7/12): super()._cdf = quad of pdf from -inf; accuracy in the far left tail and for
   large a (narrow integrand) unknown. Also: if quad underflows/returns 0.
3. sf = cdf(-x, -a): same as 1/2 mirrored (a < 0, right tail).
4. ppf (Boost quantile, Newton on Boost cdf): small q with a > 0 relies on inaccurate Boost cdf, no fallback.
   isf small q with a < 0 likewise. Test q 1e-300..1e-6.
5. pdf/logpdf: _norm_pdf exp(-x**2/2) rounding (within cond). ndtr / log_ndtr tail error (INDEX item 6),
   within cond ~ (a x)^2? check. _norm_logpdf x**2 overflow for |x| in (1.34e154, 1.9e154) (INDEX item 2).
6. tiny a (1e-10) and huge a (1e6): Owen's T regimes in Boost.
7. a == 0 branch in pdf/logpdf: exact equality only; a=1e-300 should be fine.
