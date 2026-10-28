# beta hypotheses
- H1 logpdf (l.764-767): xlog1py + xlogy - betaln; large a,b near mode: terms ~(a+b)*H cancel to O(log(a+b)) -> pattern 5, loss ~ (a+b)*H/|result| ulp. Also betaln itself for large args (relative error of betaln is fine, absolute error ~ betaln*eps).
- H2 logpdf near x=1 with b != 1: log1p(-x) fine; x small with a != 1: xlogy fine. logpdf at a=1/b=1 (xlogy(0,..)=0) fine.
- H3 pdf (boost ibeta_derivative via _beta_pdf): tiny x with a<1 -> huge pdf, overflow suppressed (over='ignore'); a large, b large: prefactor computed by Boost with Lanczos; check. Subnormal/huge intermediates for x near 0, a ~ 1e-3.
- H4 cdf/sf (betainc/betaincc, Boost ibeta): large a,b (1e5..1e6) near mode - analogous to gamma F1? Boost uses asymptotic expansions, likely fine. Tiny a (1e-3) and x tiny.
- H5 ppf (_beta_ppf, Boost ibeta_inv) / isf (betainccinv): tiny shapes (q -> x**a regime), q tiny (1e-300), q near 1 for ppf (cond large), large shapes.
- H6 isf: betainccinv - complementary variant used, so small q fine expected.
- Inherited logcdf/logsf: rv_continuous switches at median to log1p(-sf)/log1p(-cdf); fails when cdf underflows (x tiny, a large) -> opportunity.
