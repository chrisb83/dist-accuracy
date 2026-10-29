# exponnorm hypotheses (SciPy 1.18.1, lines 2212-2252)

Branch everywhere: erfcx branch iff u = (1/K - x)/sqrt2 >= 0 and K < 1; else default.

H1 (pattern 5/1) cdf default branch (K >= 1): ndtr(x) - exp(logprod), logprod = (0.5/K - x)/K + log_ndtr(x - 1/K).
    For large K the two terms are nearly equal in the bulk (cdf ~ Phi(x) * x/K): loss ~K ulp; total for K >~ 1e16.
    Also left tail x -> -inf: both ~ phi(x)/|x|, cancellation factor ~ 1 + K|x|.
H2 (pattern 5) cdf erfcx branch (K < 1): ndtr(x) - 0.5 exp(-x^2/2) erfcx(u), left tail: factor 1 + K|x| (<~ 38).
H3 (pattern 5) logpdf erfcx branch, tiny K: -log(K) + log(erfcx(u)) cancel (~ log(1/K)) -> ~|log K|/|logpdf| ulp.
H4 (pattern 3/4) logpdf/cdf/sf with K tiny: 0.5*invK*invK overflows for K < ~1e-154 (default branch, x > 1/K only).
H5 sf: sum of positives, expect fine. Underflow of exp(logprod) where sf is normal? check right tail K<1.
H6 (pattern 7) both sides of u = 0 (x = 1/K) and K = 1.
H7 pdf = exp(logpdf): fine if logpdf accurate in absolute terms.
Inherited: logcdf = log(cdf) (cdf underflows for x < ~-38 (K small) ...), logsf = log(sf) (exp tail x/K > 745),
ppf/isf numeric.
