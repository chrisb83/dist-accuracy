# gompertz hypotheses
- H1 logpdf (P5): log(c) + x - c*expm1(x) cancels for small x when c ~ 1 (c=1: x - expm1(x) ~ -x^2/2): ~ 2/x ulp.
- H2 logpdf (P3): c*expm1(x) = c*inf for x > 709.78 -> -inf, true finite for tiny c (c*e^x moderate).
- H3 ppf/isf (P3): -log1p(-q)/c, -log(p)/c overflow for c < ~1e-307 -> inf, true ~ 700.
- H4 sf: exp(-c*expm1(x)) error ~ |c e^x| ulp, but cond(x) ~ x c e^x -> expected ok.
- H5 isf p near 1: log(p) of exact double accurate -> expected ok.
- Opportunities: logsf = -c*expm1(x) (log(sf) -inf once c e^x > 745); logcdf = log(-expm1(-c*expm1(x))) underflow for c*x < 1e-308.
