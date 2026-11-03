# loguniform hypotheses (reciprocal_gen, _continuous_distns.py 9225-9328)

Code: _pdf = exp(_logpdf) (9294-9296); _logpdf = -log(x) - log(log(b) - log(a)) (9299);
_cdf = (log(x) - log(a)) / (log(b) - log(a)) (9302); _ppf = exp(log(a) + q*(log(b) - log(a))) (9305).
Inherited: sf = 1 - cdf, logcdf = log(cdf), logsf = log(1 - cdf), isf = ppf(1 - q).
_munp (9307-9312, not measured) uses _log_diff(n log b, n log a).

H1 pattern 5, pdf/logpdf/cdf/ppf: L = log(b) - log(a) with b/a near 1 and |log a| large:
   rel error of L ~ eps*|log a|/log(b/a) ulp. log(b) == log(a) in double -> L = 0: logpdf inf, cdf nan/inf.
   Test: (1e10, 1.000001e10), (1e300, 1.0000000001e300), (1e10, nextafter(1e10)).
H2 pattern 5, cdf near x = a: log(x) - log(a); excess over cond ~ |log a|. Test (1e-300, 2e-300).
H3 pdf = exp(logpdf): abs error of logpdf (eps*|log x|) -> ~|log x| ulp in pdf, cond 1. Test x ~ 1e+-300.
H4 ppf: exp(log a + ...) -> ~|log x| ulp; cond(q) = q*L small for q -> 0. Test (1e-300, ...) small q.
H5 _munp: _log_diff(p, p) when n*log b == n*log a (shared with truncnorm, INDEX 20). Not a measured method.
Inherited: sf near x = b (1 - cdf) cancels; isf = ppf(1 - q) rounds small q. Opportunities.
