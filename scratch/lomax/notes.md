# lomax hypotheses
- H1 (p6/p3) pdf line 8401 `c/(1.0+x)**(c+1)`: 1+x rounding raised to c+1 -> ~(c+1)*x/(1+x) ulp for large c; denominator overflows when (c+1)log1p(x)>709.8 though pdf=c*exp(-...) normal for large c (c=1e3..1e6).
- H2 (p2) pdf small x: 1.0+x rounds away x; error (c+1)*eps relative; large c only.
- H3 (p9/p6) pdf tiny c: c+1.0 rounds to 1 for c<1.1e-16; error c*log1p(x) relative -> <~1e-14 rel.
- H4 (exp amplification) sf line 8410 exp(-c*log1p(x)): rounding of product amplified by |c log1p x| up to 745 -> up to ~700 ulp minor (cond(x) = c x/(1+x)).
- H5 (p5) logpdf line 8404: log(c) - (c+1)log1p(x) cancels near zero crossing (x = c**(1/(c+1)) - 1); conditioning large there too.
- H6 (p8) isf line 8419 `q**(-1/c) - 1`: cancels when q**(-1/c) ~ 1: large c (bulk), q -> 1. Should use expm1(-log(q)/c).
- H7 isf small c: rounding of -1/c, amplified by log(q) -> minor.
- H8 ppf/cdf/logsf: expm1/log1p forms, expect fine.
- logcdf inherited: log(cdf) -> 0 when cdf rounds to 1 (sf < eps) -> loss; better log1mexp(-c*log1p(x)).
