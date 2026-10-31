# rayleigh hypotheses
- logpdf (l.9136) `log(r) - 0.5*r*r`: r*r overflows for r > 1.34e154, true logpdf finite until ~1.897e154 (pattern 3). Loss expected.
- logsf (l.9148) `-0.5*r*r`: same overflow window (pattern 3). Loss expected.
- pdf (l.9133) `exp(logpdf)`: abs error of logpdf ~ eps*|logpdf| becomes relative; for tiny r, log(r) ~ -690 -> up to ~350 ulp with cond 1 (pattern 3/5, minor). Right tail covered by cond ~ r^2.
- sf (l.9145) `exp(-0.5 r r)`: cond r^2, expect within conditioning.
- cdf (l.9139) `-expm1(-0.5*r**2)`: r**2 subnormal only where cdf itself subnormal -> expect fine.
- ppf (l.9142) `sqrt(-2*log1p(-q))`: fine. isf (l.9151) `sqrt(-2*log(q))`: q near 1 -> log(q) of exact double, fine.
- logcdf inherited: log(cdf) / log1p(-sf) generic. Opportunity: log(-expm1(-x^2/2)) / log1mexp.
