# laplace_asymmetric hypotheses
- H1 (p1) cdf x>=0 branch `1 - exp(-x k)*kapinv/kk`: cdf(0)=k^2/(1+k^2) small for small kappa -> cancellation for small x>=0 (cdf small). Regime kappa<<1, x >= 0 with x*kappa not large.
- H2 (p1) sf x<0 branch `1 - exp(x/k)*k/kk`: mirror, kappa>>1, x<0 small.
- H3 (p8/p2) ppf upper branch `-log((1-q)*kk*k)/k`: 1-q rounds for q<0.5 (small kappa, q just above q0=k^2/(1+k^2)); log of value near 1 (log1p needed) at q slightly above q0 for any kappa.
- H4 (p8/p2) isf lower branch `log((1-q)*kk/k)*k`: mirror, kappa large, q in (1/(1+k^2), 0.5).
- H5 (p3) ppf lower branch q*kk/kappa overflows for tiny kappa (only in discarded branch -> warning). isf/ppf overflow kk*k for kappa>1e154.
- H6 (p7) np.where evaluates both branches: exp overflow warnings in cdf/sf for large |x|.
- logpdf: same-sign terms, expected fine.
