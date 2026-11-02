# trapezoid hypotheses
- H1 cdf left branch `x**2 / c / (d-c+1)` (l.10008): x**2 underflows (subnormal/0) for x < 1.5e-154 while cdf is normal when c is tiny (pattern 4). Regime c < ~1e-150, x < c.
- H2 ppf left branch `sqrt(q * c * (1+d-c))` (l.10017): q*c underflows for q*c < 2.2e-308 while result sqrt is normal (pattern 4). Regime small c and small q.
- H3 ppf right branch `1 - sqrt((1-q)...)`: q->1, ill-conditioned wrt q; should be within cond (pattern 8).
- H4 cdf right branch 1 - (...): result near 1 -> fine for cdf; sf inherited = 1 - cdf loses everything (pattern 1/12).
- H5 branch thresholds x==c, x==d, q==qc, q==qd: check both sides (pattern 7).
- H6 d close to 1 / c close to 0: divisions by (1-d), c; check d=1-1e-12, c=0 edge.
- Inherited: logcdf = log(cdf) inherits H1 and underflows; isf = ppf(1-q) loses small-q info; logsf=log(1-cdf).
