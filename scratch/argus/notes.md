# argus hypotheses (SciPy 1.18.1, lines 11874-11946)

- H1 cdf (line 11943) `1.0 - self._sf`: pattern 1. cdf small for x -> 0 (cdf ~ x^2 for small chi)
  and for large chi everywhere except near x=1 (cdf ~ exp(-chi^2/2 * ...)). Expect loss for x < ~1e-8 and for large chi.
- H2 logpdf/pdf (line 11936) `3*log(chi) - log(_argus_phi(chi))`: pattern 5 for small chi
  (both ~ 3|log chi|, A -> const); pattern 3/4: _argus_phi(chi) = gammainc(1.5, chi^2/2)/2 ~ chi^3
  underflows for chi < ~1e-103 -> log(0) -> logpdf = +inf, pdf = inf. Also chi**2 underflows chi<1e-162.
- H3 sf (line 11946) same underflow of _argus_phi for tiny chi: 0/0 = nan (or ratio of subnormals).
- H4 logpdf (line 11935/11937) `y = 1 - x*x`, `log1p(-x*x)`: x*x rounded before subtraction (pattern 1),
  near x = 1. Probably within conditioning (cond ~ 1/(1-x)), check. (1-x)*(1+x) would be exact-ish.
- H5 logpdf `-chi**2*y/2`: chi**2 overflows for chi > 1.34e154 -> -inf (pattern 3).
- H6 pdf = exp(logpdf): error ~ |logpdf| ulp, cond also ~ |logpdf|; likely within conditioning.
- H7 _argus_phi(chi*sqrt(...)): argument squared again inside -> chi*sqrt(y) then **2, rounding fine.
