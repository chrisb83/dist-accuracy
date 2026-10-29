# dweibull hypotheses (SciPy 1.18.1, lines 1975-2044)

- H1 (p6, as weibull_min F1) cdf x<0, sf x>0, pdf |x|>1: exp(-|x|**c) amplifies rounding of |x|**c by t/2 ulp; beyond cond (c*t) for c<1.
- H2 (rounding of c-1, weibull_min F2) pdf, c<0.5, small |x|: ~|log x|*2^-54/eps ulp.
- H3 (rounding of 1/c, weibull_min F3) ppf/isf: power(-log(2q), 1/c), small q, inexact 1/c -> up to ~0.5|log result| ulp.
- H4 (p3/p7) pdf large c, |x|>1: ax**(c-1)=inf * exp(-ax**c)=0 -> nan (true 0, not measurable).
- H5 (p1) cdf x>0: 1 - Cx1 with Cx1 <= 0.5: no cancellation (result >= 0.5). sf x<0 same. Expect fine.
- H6 (p8) ppf/isf near q=0.5: 1-q exact (Sterbenz), 2q exact, log(1-d) accurate -> expect fine. Test q = 0.5 +- 1e-k.
- H7 logpdf: log(c) - log(2) + xlogy(c-1,|x|) - |x|**c; small results only near pdf=1 where cond is large; expect fine.
- H8 inherited logcdf/logsf: log(cdf) for x<0 underflows when |x|**c > ~745 (true -log2-|x|**c). Opportunity.
