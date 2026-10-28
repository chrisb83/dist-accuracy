# cosine hypotheses
- H1 pdf (l.1725) 1+cos(x) near |x| -> pi: cancellation (pattern 1/5); also pi not representable, cos(x) rounds to -1 -> 0. Severe/loss expected, beyond cond ~ 2pi/(pi-|x|).
- H2 logpdf (l.1728-1731) same: log1p(cos x) with cos x ~ -1 -> -inf at c == -1 (fill). Loss.
- H3 cdf/sf via scu._cosine_cdf: left tail x -> -pi (cdf ~ (x+pi)^3/(12pi)) -- does C code use series? Check tail.
- H4 ppf/isf via _cosine_invcdf: small q (x ~ -pi + (12 pi q)^(1/3)), q near 0.5, q near 1 (isf = -invcdf(p) fine; ppf near 1 conditioning).
- H5 logcdf/logsf inherited: log(cdf) fine as long as cdf normal; cdf down to ~1e-49 at x=-pi double -> ok.
