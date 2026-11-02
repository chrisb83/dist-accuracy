# tukeylambda hypotheses
Q(p) = (p^lam - (1-p)^lam)/lam ; F = Q^{-1}; f = 1/(F^(lam-1) + (1-F)^(lam-1))

H1 (p10, shared): sc.tklmbda is bisection with absolute stopping tol ~1e-14 / 60 iters
   -> cdf has abs error ~1e-14: bulk ~10-100 ulp, left tail (F < 1e-10) severe/loss
   (quantized to multiples of 2^-k, returns 0 beyond). Probe: cdf(-1e-20,-1)=0.49999999999999290 (true 0.5-...).
H2 (p9/p10): |lam| < 1e-4 uses logistic cdf exactly -> error ~ lam*log^2(p) in x; severe for lam=1e-5, |x|~10.
H3 (p1): pdf line 10895 (1-Fx)**(lam-1): right tail cancels, x>0 large -> pdf wrong (pdf even; should use F(-|x|)).
H4: pdf inherits all cdf errors (Fx**(lam-1) amplifies relative error of F by |lam-1|).
H5 (p5/p8): ppf boxcox(q)-boxcox1p(-q) cancel near q=1/2; probably within cond. Large lam: both ~ -1/lam.
H6 (p3): ppf for lam<0 and tiny q: q^lam overflows -> true value overflows too (fine).
H7: cdf for lam>0 near support ends |x| -> 1/lam: bisection absolute tol -> loss.
