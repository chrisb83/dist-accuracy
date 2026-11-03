# semicircular hypotheses
- H1 (p1/p5) cdf line 9632, x -> -1: 0.5 + (x*sqrt(1-x*x) + arcsin(x))/pi cancels; cdf ~ (4*sqrt2/(3pi)) d^1.5 with d=1+x.
  Error ~ eps absolute -> d^-1.5 ulp vs cond ~1.5/d; loss (0) for cdf < ~1e-16 (d < ~1e-11).
- H2 (p1) pdf line 9626 / logpdf 9629, |x| -> 1: 1-x*x rounding; expect within conditioning (cond ~ 1/(2d)).
- H3 (p10) ppf line 9635 -> rdist._ppf(q,3) = 2*scu._beta_ppf(q,1.5,1.5)-1 (line 9088): beta report found Boost ibeta_inv non-convergence for small q; also 2p-1 cancels near q=0.5 (cond also large), and near q->0 result is -1+2p (fine unless p wrong).
- H4 inherited sf = 1-cdf near x -> 1 (p1, p12): sf(x) = cdf(-x) exactly by symmetry; isf = ppf(1-q) rounds q < eps/2, better -ppf(q).
- H5 logcdf = log(cdf) inherits H1.
