# invgauss hypotheses
H1 (p3) pdf: x**3.0 overflows for x > 5.6e102 -> pdf 0; true pdf normal if mu large (x/(2mu^2) small).
H2 (p3) logpdf: (x/mu - 1)**2 overflows for x/mu > 1.34e154 while result x/(2mu^2) finite.
H3 (p5) logcdf/cdf/logsf/sf: b = 2/mu + log_ndtr(-(x/mu+1)/sqrt x): two terms ~2/mu cancel; abs error eps*2/mu.
    Small mu (1e-8 .. 1e-20), x near mu -> moderate..loss. mu < 1.1e-308: 2/mu = inf -> nan.
H4 (p1/p5) logsf/sf: log1p(-exp(b-a)) with exp(b-a) -> 1 when x >> mu (right tail) or mu >> x >> 1 (Levy-like);
    error ~ eps * |b| / (1 - exp(b-a)) ~ x/mu ulp. Total loss when x/mu > 1/eps with sf normal (mu > 1e13).
H5 ppf/isf: Boost inverse_gaussian quantile (Newton); generic fallback for nan. Check tails, small/large mu.
