# weibull_min hypotheses

- H1 (p4) pdf, large c, x<1: pow(x, c-1) subnormal ((c-1)log x in [-745,-708]) while c*x**(c-1) normal. c=1e6, 1e10.
- H2 (p4) pdf, large c, x>1: exp(-x**c) subnormal (x**c in [708,745]) while c*x**(c-1)*exp(-x**c) normal.
- H3 (rounding of c-1) pdf, c<0.5 (c-1 inexact), small x: error ~ |log x| * 2^-54 / eps ~ up to ~180 ulp. minor.
- H4 (rounding of 1/c) ppf/isf, c with inexact 1/c, result far from 1: error ~ |log result| * 0.5 ulp, up to ~370 ulp. minor.
- H5 isf small c, small q: rel. error of -log(q) amplified by 1/c, cond 1/(c|log q|). minor.
- H6 pdf: pow(x,c-1) = inf times exp(-x**c) = 0 -> nan where true pdf is 0 (not measurable; note only).
- logcdf inherited: log(-expm1(-x**c)) -> 0 for x**c > ~37. Opportunity: log1mexp.
--plot exits with code 127 after the condition-number step (no PNG written); run2 without --plot is complete.
