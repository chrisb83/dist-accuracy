# halfgennorm hypotheses (a = 1/beta, y = x**beta)

H1 cdf/sf (lines 11682, 11688), pattern 4/10: y = x**beta underflows (beta*log x < -745)
   while P(a, y) ~ x/Gamma(1+a) is normal. Large beta: whole (0, exp(-745/beta)) -> cdf 0, sf 1.
   Moderate beta too: x = 1e-300, beta = 2 -> cdf 0 vs 8.9e-301 (loss). Subnormal band -> severe.
H2 ppf (11685), pattern 4/8: gammaincinv(a, q) = z ~ (q Gamma(1+a))**(1/a) underflows for
   small q (any beta: ppf(1e-300, 2) ~ 1e-300 needs z = 1e-600) and for large beta at any q.
H3 isf (11691): gammainccinv(a, q) underflows for q near 1 / large beta (mirror of H2).
H4 logpdf (11679), pattern 5: log(beta) - gammaln(1/beta) = -gammaln(1+1/beta) ~ 0.577/beta
   for large beta; terms ~log(beta) -> huge relative error when x**beta small. Also for beta near 1
   the result is near... (log(1) - gammaln(1) = 0, fine). x**beta ~ log(beta)-gammaln: zero of logpdf
   (pdf=1 crossing) -> cancellation near logpdf=0 for any beta>1? pdf max = beta/Gamma(1/beta)
   = 1/Gamma(1+1/beta) > 1 for beta > 1, so logpdf crosses 0 at x**beta = -gammaln(1+a): cancellation,
   but cond(arg) is large there too.
H5 pdf (11676): exp(logpdf) amplifies absolute error: small beta (gammaln(1/beta) large).
H6 ppf/isf small beta: z**(1/beta) amplifies rounding of gammaincinv by 1/beta (pattern 6, gamma F5).
H7 gammainc unconverged series (a > 2e5, y near a) unreachable: y = x**beta <= 1.8e308 << a**... needs
   x = a**a; for beta = 5e-6, y = 2e5 requires x = 2e5**2e5 = inf. Unreachable.
