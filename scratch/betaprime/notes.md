# betaprime hypotheses
- H1 (p5) logpdf L1038: xlogy(a-1,x) - xlog1py(a+b,x) cancel for large x and large a (terms ~a log x, result ~(b+1) log x).
- H2 (p5/p10) logpdf: betaln error for large shapes (beta F4), e.g. (1e5,3e5) near mode, (1e3,0.5).
- H3 pdf = exp(logpdf): absolute error of logpdf -> relative error of pdf ~ |logpdf| ulp (minor) plus H1/H2.
- H4 (p10) ppf L1066: beta._ppf (Boost ibeta_inv) non-convergence for small q (beta F1): (0.5,2) q 1e-14..1e-9, (2,10) q<1e-101.
- H5 (p1/p8) ppf L1068: r/(1-r) for r just below 0.9999 amplifies rounding of r by 1/(1-r) ~ 1e4: a large, b moderate (1e3,0.5) bulk.
- H6 ppf L1072: beta._isf(p,b,a) = betainccinv(b,a,p) for r>0.9999; beta F3 (b=2 nan) analog: here first shape is b... betainccinv(b, a, p) nan when its *second* shape = 2 i.e. a = 2 and p tiny? but branch is only used for r>0.9999 (p not tiny unless a huge).
- H7 cdf/sf threshold x=1: both sides fine (args rounded relative 1 ulp).
- isf inherited = ppf(1-q): loses small q (opportunity).
