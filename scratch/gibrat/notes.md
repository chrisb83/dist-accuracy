# gibrat hypotheses (SciPy 1.18.1, lines 7061-7119)

gibrat = lognorm(s=1). Methods: pdf=exp(logpdf), logpdf=_lognorm_logpdf(x,1.0),
cdf=ndtr(log x), sf=ndtr(-log x), ppf=exp(ndtri(q)), isf=exp(-ndtri(q)).
Inherited: logcdf=log(cdf), logsf=log(sf).

1. logpdf, x > 7.2e307: log(1.0*x*sqrt(2pi)) overflows -> -inf (pattern 3; lognorm F1, shared).
2. pdf: exp(logpdf) with large |log x| terms -> ~|L| ulp (lognorm F5, shared).
3. cdf/sf tails: rounding of log(x) magnified ~z^2 in ndtr (lognorm F6, s=1 -> z=log x up to 38).
4. ppf/isf: exp(ndtri(q)) magnifies ndtri rounding by |log result| (lognorm F7; s=1 no product rounding).
5. logpdf near x=1: -log(x)**2/2 - log(x*sqrt(2pi)): no zeros for s=1 (logpdf <= -0.919+...); no cancellation expected.
   Near x = exp(-1) (mode) fine.
6. Opportunities: logcdf = log(ndtr(log x)) -> -inf for log x < -38.4 (x < ~2e-17), use log_ndtr(log x).
   logsf same for x > ~5e16. Also logcdf right tail: log(cdf near 1) loses relative accuracy -> log_ndtr.
