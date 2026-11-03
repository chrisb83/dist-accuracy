# crystalball hypotheses (SciPy 1.18.1, lines 11703-11868)

Notation: B = m/beta - beta, A = (m/beta)**m exp(-beta^2/2), T = (m/beta) exp(-beta^2/2)/(m-1),
1/N = T + sqrt(2pi) Phi(beta), pbeta = N*T = cdf(-beta).

1. pdf/cdf lhs (lines 11771, 11803): (m/beta)**m overflows (pattern 3) for m*log(m/beta) > 709.8
   (large m, e.g. m=1000 beta=1; or tiny beta, e.g. beta=1e-150 m=3) -> inf*0 = nan / inf / 0.
   Also (m/beta - beta - x)**(-m) underflows separately for large |x| and m. Pattern 3/4.
2. pdf/cdf lhs: rounding of m/beta and of (B - x) raised to power m (pattern 6), ~m ulp. Test m=50.
3. logpdf lhs (line 11787): m*log(m/beta) - m*log(m/beta - beta - x) cancels near x = -beta
   for large m (pattern 5). m=1e6 beta=1 x=-1: terms 1.4e7, result ~ -1.4.
4. logpdf rhs (line 11784): -x**2/2 overflows for x in (1.34e154, 1.9e154) (same as norm F1).
5. sf lhs (line 11820): 1 - cdf for x <= -beta. cdf(-beta) = pbeta close to 1 for small beta
   (sf(-beta) ~ sqrt(2pi)Phi(beta)*N ~ beta) -> 1/sf ulp loss (pattern 1). beta=1e-10: ~1e10 ulp.
6. ppf_less (line 11834): (m/beta)**(-m) underflows for large m / overflow for small beta (pattern 3)
   -> inf/-inf/nan. /eb2 overflows for beta > 38.6 but then pbeta = 0 so branch not taken.
7. ppf_less: m/beta - beta - Y cancels near p = pbeta; cond(arg) probably covers it.
8. N for tiny beta (< ~m/1.8e308): m/beta overflows -> N = 0 (pattern 3). Edge only.
9. inherited isf = ppf(1-q): loses q < eps. logcdf = log(cdf): power-law tail underflows
   (x=-1e300). logsf = log(sf): Gaussian tail underflows for x > 38.6.
