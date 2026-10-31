# powerlognorm hypotheses (z = log(x)/s, lines 8953-8974)
H1 (p.8) isf: q**(1/c) underflows to 0 for small c (log(q)/c < -745) -> ndtri(0) = -inf -> isf = inf. Also q**(1/c) near 1 for large c -> ndtri of value near 1 (rounded) loses relative accuracy of 1 - q**(1/c).
H2 (p.8) ppf = isf(1 - q): 1 - q rounds for q < eps -> 0.0; relative error ~eps/q for small q.
H3 (p.5) logpdf right tail with c < 1: -z**2/2 and (c-1)*log_ndtr(-z) ~ (1-c) z**2/2 cancel; loss ~1/c.
H4 (p.3) logpdf: _norm_logpdf squares z; z**2 overflows for |z| > 1.34e154 (tiny s) -> -inf; with c = 1, (c-1)*log_ndtr = 0*(-inf) = nan.
H5 sf/pdf = exp(large): absolute error of c*log_ndtr(-z) (size |logsf|) -> relative error; rounding of z magnified by ~c z**2.
H6 (p.4) logsf/cdf left tail, huge c: log_ndtr(-z) subnormal/0 for -z > ~37.5 while c*Phi(z) normal.
H7 log_ndtr(z>0) far-tail error (shared, see lognorm F6) in logsf/cdf left tail.
H8 (p.6) isf/ppf large s: exp(-s*ndtri) magnifies rounding by |log result|.
