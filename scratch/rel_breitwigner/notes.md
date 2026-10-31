# rel_breitwigner hypotheses
- H1 (p3): pdf C uses 1/rho**2 -> inf for rho < ~1e-154 -> C = sqrt(inf/inf) = nan. Also cdf C and 1j/rho, -rho*(rho+1j).
- H2 (p3): cdf large rho: -rho*(rho+1j) -> rho**2 overflows for rho > 1.3e154; 1/rho**2 underflow harmless.
- H3 (p5): cdf = 2C*Im(sqrt(-1+i/rho)*arctan(x/b)): left tail (x << rho, cdf tiny) is Im of a product of O(1) complex numbers -> absolute error ~eps, relative error large when cdf << 1, esp. large rho (cdf(0+) region tiny ~ x/rho^2...).
- H4 (p1): cdf near 1 clipped to 1; inherited sf = 1 - cdf loses right tail (opportunity).
- H5: pdf squares (x^2-rho^2)/rho -> overflow at x ~ 1.16e77*sqrt(rho); true pdf ~ C rho^2/x^4 ~ 1e-308 there: borderline subnormal.
- H6 (p5): pdf near x = rho: (x-rho)*(x+rho) factored, good.
- H7: complex arctan accuracy in numpy (bulk), and arctan near branch for x/b near +-i? b = sqrt(-rho^2 - i rho) is nearly imaginary for large rho: b ~ -i*rho... x/b ~ i*x/rho -> near branch point i when x ~ rho (the peak!). Expect cancellation around peak for large rho.
