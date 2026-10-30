# geninvgauss hypotheses
Code: _logpdf -> _stats.geninvgauss_logpdf (Cython, source not shipped; verified numerically equal to
  -log(2) - log(kve(p,b)) + b + (p-1)*log(x) - b*(x+1/x)/2 ; NaN if kve is inf).
_pdf = exp(_logpdf). _cdf = integrate.quad(pdf, 0, x) (default epsabs=epsrel=1.49e-8).

H1 (pattern 3/10) logpdf/pdf: kve(p,b) overflows for large |p|, small b (e.g. p=100, b=0.01) -> NaN. loss.
H2 (pattern 5) logpdf/pdf: large b, x near 1: b - b(x+1/x)/2 cancels (terms ~b, result ~ -b(x-1)^2/(2x)). ~b ulp abs err in logpdf.
H3 (pattern 5) large p: (p-1)log x vs -log kve cancel near the mode; ~p ulp.
H4 cdf quad: tolerance 1.49e-8 -> up to ~1e8 ulp in bulk; tiny left-tail cdf relative error unbounded.
H5 cdf quad: peaked pdf (large b) or very wide pdf (tiny b) over [0, x] with x >> mode -> quad misses mass (cdf(1e10,0.5,1e4) = 0.0). loss.
H6 cdf near 1: fine (no 1-...), but sf = 1 - cdf inherited (opportunity).
