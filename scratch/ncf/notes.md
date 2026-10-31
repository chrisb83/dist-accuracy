# ncf hypotheses (SciPy 1.18.1, all methods Boost non_central_f / non_central_beta)

Boost non_central_f maps x -> x1 = x*alpha/beta (alpha = n/2, beta = m/2), y = x1/(1+x1).

1. pdf, right tail (pattern 1/10): Boost pdf passes only y = x1/(1+x1) to the
   non-central beta pdf, which forms 1 - y internally. For x1 >~ 1e8 the relative
   error of 1 - y grows like x1*eps; total loss at x1 > 2/eps. Also
   `(1+x1)*(1+x1)` overflows for x1 > 1.3e154 (pattern 3). Regime: x > ~1e8 m/n.
2. cdf/sf: `x*alpha` overflows for x > 1.8e308/alpha (pattern 3, as f F2).
3. pdf/cdf/sf left tail with large nc (pattern 4/7): Boost sums the Poisson series
   starting at the Poisson mode k ~ nc/2 and breaks on `term == 0`; at tiny x the
   mode term (ibeta(a+k,b,y) ~ y^(a+k)) underflows -> result 0 although the j=0
   term is a normal double.
4. ppf/isf (pattern 8/1): Boost quantile returns (r/(1-r))*(m/n) with r the
   non-central beta quantile; for r near 1 (large quantiles, isf small q,
   heavy right tail small dfd) 1 - r loses ~x1 ulp; r == 1 -> overflow (gh-17432).
5. ppf left tail: beta root tiny (small q, large n) -> ibeta_inv-like failures
   (f F3/F4, beta F1)?
6. Large nc (1e3, 1e4): series iteration limits / convergence -> nan or error?
7. Tiny nc (1e-8): Boost special-cases only l == 0; check continuity.
8. Inherited logcdf/logsf/logpdf: log of underflowing values (opportunities).
