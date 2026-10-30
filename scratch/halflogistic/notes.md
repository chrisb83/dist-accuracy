# halflogistic hypotheses
- logpdf (l.4642) log(2) - x - 2*log1p(exp(-x)): no cancellation (all terms same sign beyond log2; result <= -log2). Expect ok.
- pdf = exp(logpdf): error ~ |logpdf| * eps, matches cond ~ x. Expect ok within cond.
- cdf tanh(x/2): small x fine; no 1-F. Expect ok.
- sf 2*expit(-x): ok; underflows only where true value subnormal (x > ~745).
- ppf 2*arctanh(q): q->1 ill-conditioned only. Expect ok.
- isf branch q<0.5 (pattern 7): -logit(q/2) fine; q>=0.5: 1-q exact (Sterbenz). Test both sides of 0.5.
- inherited logsf = log(sf): sf underflows for x > ~745 -> -inf (pattern 12); closed form log(2) + log_expit(-x).
- inherited logcdf = log(tanh(x/2)): fine except maybe; closed form log(tanh(x/2)) / log1p(-2*expit(-x)) for large x - log(cdf) loses relative digits? logcdf ~ -2e^-x for large x; log(tanh) where tanh rounds to 1 -> 0 (loss, pattern 1/12).
